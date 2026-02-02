"""OpenAI API wrapper with concurrent execution, caching, and retry logic.

Uses the ``instructor`` library (v2) to guarantee Pydantic-validated JSON
responses from every model, including reasoning models (GPT-5, o-series).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
    CancelledError,
)
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type

import httpx
import instructor
from openai import BadRequestError, OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from src.cache import ResponseCache
from src.schemas import LLMConfig, LLMRequest, LLMResponse, RepairStats

logger = logging.getLogger(__name__)

# Models whose API rejects temperature / max_tokens and requires
# max_completion_tokens instead.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class LLMClient:
    """Thread-safe OpenAI client with caching, validation, and retry."""

    def __init__(self, config: LLMConfig, cache: ResponseCache, max_workers: int = 100) -> None:
        self.config = config
        self.cache = cache
        self.max_workers = max_workers

        # Raw OpenAI client with explicit httpx transport.
        # IMPORTANT: retries=0 — transport-level retries silently multiply
        # the timeout (e.g. retries=2 means 3×60s=180s per request) and are
        # redundant because _call_once already handles retries with backoff.
        transport = httpx.HTTPTransport(
            retries=0,
        )
        http_client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(
                config.timeout,
                connect=10,
                pool=5,  # fail fast if no connection available in pool
            ),
            limits=httpx.Limits(
                max_connections=max_workers + 10,
                max_keepalive_connections=20,  # keep few idle connections to avoid stale sockets
            ),
        )
        raw_client = OpenAI(
            api_key=config.api_key,
            http_client=http_client,
            max_retries=0,
        )

        # Instructor-patched client — handles JSON extraction, repair,
        # and Pydantic validation automatically.
        self._client = instructor.from_openai(raw_client)

        # Repair stats (thread-safe)
        self._stats_lock = Lock()
        self._repair_stats = RepairStats()
        self._repair_strategies: Dict[str, int] = {}

        # Per-task stats
        self._task_stats: Dict[str, RepairStats] = {}

    # ------------------------------------------------------------------
    # Single request (with retry)
    # ------------------------------------------------------------------

    def _call_once(
        self,
        request: LLMRequest,
        timeout_override: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> LLMResponse:
        """Execute a single LLM request with instructor-managed validation.

        Parameters
        ----------
        timeout_override : int, optional
            Per-request timeout in seconds, overriding ``config.timeout``.
        max_attempts : int, optional
            Override the number of retry attempts (default: ``config.max_retries``).
            Use ``max_attempts=1`` for escalated-timeout retries to avoid
            compounding the timeout (e.g. 3 attempts × 240s = 720s).
        """
        MAX_ATTEMPTS = max_attempts if max_attempts is not None else self.config.max_retries
        effective_timeout = timeout_override or self.config.timeout

        for attempt in range(1, MAX_ATTEMPTS + 1):
            t0 = time.perf_counter()
            try:
                # Build model-appropriate kwargs
                is_reasoning = request.model.startswith(_REASONING_PREFIXES)
                kwargs: Dict[str, Any] = {}
                if is_reasoning:
                    # Reasoning models use internal reasoning tokens from the
                    # same budget, so we need a much larger ceiling to leave
                    # room for the actual JSON output.
                    kwargs["max_completion_tokens"] = max(self.config.max_tokens, 32_000)
                else:
                    kwargs["max_tokens"] = self.config.max_tokens
                    kwargs["temperature"] = self.config.temperature

                # instructor handles JSON extraction + Pydantic validation
                validated_obj = self._client.chat.completions.create(
                    model=request.model,
                    response_model=request.expected_schema,
                    max_retries=0,  # we handle retries ourselves
                    timeout=effective_timeout,
                    messages=[
                        {"role": "system", "content": request.prompt_system},
                        {"role": "user", "content": request.prompt_user},
                    ],
                    **kwargs,
                )

                latency_ms = int((time.perf_counter() - t0) * 1000)

                # Track stats
                self._record_success(request.task, attempt)

                raw_json = (
                    validated_obj.model_dump_json()
                    if isinstance(validated_obj, BaseModel)
                    else str(validated_obj)
                )

                return LLMResponse(
                    request_id=request.request_id,
                    case_id=request.case_id,
                    model=request.model,
                    task=request.task,
                    success=True,
                    response_raw=raw_json,
                    response_parsed=validated_obj,
                    error_message=None,
                    latency_ms=latency_ms,
                    tokens_prompt=0,
                    tokens_completion=0,
                    timestamp=datetime.now(),
                    attempt_number=attempt,
                    repair_applied=False,
                )

            except BadRequestError as e:
                # 400 errors (unsupported params, invalid model, etc.)
                # are non-retryable — raise immediately to stop the run.
                raise RuntimeError(
                    f"Fatal API error for model {request.model!r}: {e}"
                ) from e

            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                err_str = str(e).lower()

                # Quota exhaustion is non-retryable — retrying will
                # never succeed until billing is resolved.
                if "insufficient_quota" in err_str:
                    raise RuntimeError(
                        f"Fatal: OpenAI API quota exceeded. Check your plan "
                        f"and billing details at "
                        f"https://platform.openai.com/account/billing . "
                        f"Original error: {e}"
                    ) from e

                # Token-limit truncation is non-retryable — the same
                # prompt will hit the same limit every time.
                is_token_limit = (
                    "max_tokens" in err_str
                    or "length limit" in err_str
                    or "incomplete" in err_str
                )
                if is_token_limit:
                    is_reasoning = request.model.startswith(_REASONING_PREFIXES)
                    # Tasks whose input length varies (neutralize, rewrite,
                    # remedy) can hit the token ceiling on outlier-long
                    # narratives even on reasoning models.  Only abort the
                    # batch for fixed-length tasks (e.g. commentary) where
                    # truncation signals a systematic budget problem.
                    _variable_input_tasks = {
                        "neutralize", "rewrite", "remedy", "persona_estimate",
                    }
                    if is_reasoning and request.task not in _variable_input_tasks:
                        # Systematic budget problem — abort the batch.
                        raise RuntimeError(
                            f"Fatal: output truncated by token limit for model "
                            f"{request.model!r} on {request.task}: {e}"
                        ) from e
                    # Outlier input — log a warning and fail just this request.
                    prompt_chars = len(request.prompt_system) + len(request.prompt_user)
                    est_prompt_tokens = prompt_chars // 4  # rough estimate
                    effective_limit = (
                        max(self.config.max_tokens, 32_000)
                        if is_reasoning
                        else self.config.max_tokens
                    )
                    # Extract just the first line of the error to avoid
                    # dumping the verbose <failed_attempts> block.
                    err_first_line = str(e).split("\n", 1)[0][:200]
                    logger.warning(
                        "Token limit hit for %s (%s/%s): skipping. "
                        "prompt_chars=%d (~%d tokens), "
                        "user_text_chars=%d, "
                        "max_tokens=%d. %s",
                        request.case_id, request.model, request.task,
                        prompt_chars, est_prompt_tokens,
                        len(request.prompt_user),
                        effective_limit,
                        err_first_line,
                    )
                    self._record_failure(request.task)
                    return LLMResponse(
                        request_id=request.request_id,
                        case_id=request.case_id,
                        model=request.model,
                        task=request.task,
                        success=False,
                        error_message=(
                            f"Token limit exceeded: "
                            f"prompt_chars={prompt_chars} (~{est_prompt_tokens} tokens), "
                            f"max_tokens={effective_limit}"
                        ),
                        latency_ms=latency_ms,
                        timestamp=datetime.now(),
                        attempt_number=attempt,
                    )
                if attempt < MAX_ATTEMPTS:
                    logger.debug(
                        "Attempt %d failed for %s (%s): %s. Retrying...",
                        attempt, request.case_id, request.task, e,
                    )
                    time.sleep(self.config.retry_delay * (2 ** (attempt - 1)))
                    continue
                else:
                    self._record_failure(request.task)
                    logger.debug(
                        "All %d attempts failed for %s (%s): %s",
                        MAX_ATTEMPTS, request.case_id, request.task, e,
                    )
                    return LLMResponse(
                        request_id=request.request_id,
                        case_id=request.case_id,
                        model=request.model,
                        task=request.task,
                        success=False,
                        response_raw=None,
                        response_parsed=None,
                        error_message=(
                            f"Failed after {MAX_ATTEMPTS} attempts "
                            f"(prompt_len={len(request.prompt_user)} chars, "
                            f"timeout={effective_timeout}s): {e}"
                        ),
                        latency_ms=latency_ms,
                        tokens_prompt=0,
                        tokens_completion=0,
                        timestamp=datetime.now(),
                        attempt_number=attempt,
                        repair_applied=False,
                    )

        # Should not reach here, but just in case
        self._record_failure(request.task)
        return LLMResponse(
            request_id=request.request_id,
            case_id=request.case_id,
            model=request.model,
            task=request.task,
            success=False,
            error_message="Exhausted all attempts",
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # Cached single call
    # ------------------------------------------------------------------

    def call_with_cache(
        self,
        request: LLMRequest,
        narrative: str,
        timeout_override: Optional[int] = None,
        max_attempts: Optional[int] = None,
    ) -> LLMResponse:
        """Check cache first; on miss call LLM and cache the result."""
        cached = self.cache.get(request.model, request.task, narrative)
        if cached is not None:
            # Rebuild validated object from cached response
            try:
                validated = request.expected_schema(**cached)
                return LLMResponse(
                    request_id=request.request_id,
                    case_id=request.case_id,
                    model=request.model,
                    task=request.task,
                    success=True,
                    response_raw=str(cached),
                    response_parsed=validated,
                    timestamp=datetime.now(),
                    attempt_number=0,  # 0 = cache hit
                )
            except Exception:
                pass  # Cache entry invalid; fall through to API call

        response = self._call_once(request, timeout_override=timeout_override, max_attempts=max_attempts)

        if response.success and response.response_parsed is not None:
            self.cache.put(
                request.model,
                request.task,
                narrative,
                response.response_parsed.model_dump()
                if isinstance(response.response_parsed, BaseModel)
                else response.response_parsed,
            )

        return response

    # ------------------------------------------------------------------
    # Batch execution (with circuit breaker)
    # ------------------------------------------------------------------

    # If more than this fraction of the first CIRCUIT_BREAKER_WINDOW
    # requests fail, abort the entire batch.
    CIRCUIT_BREAKER_THRESHOLD = 0.5
    CIRCUIT_BREAKER_WINDOW = 10

    # Per-future timeout: must exceed worst-case _call_once duration.
    # With timeout=60s, max_retries=3, backoff 1/2/4s → worst case ~195s.
    # 300s gives ample headroom.
    FUTURE_TIMEOUT = 300  # seconds

    def execute_batch(
        self,
        requests: List[LLMRequest],
        narratives: List[str],
        desc: str = "LLM calls",
    ) -> List[LLMResponse]:
        """Execute a batch of requests concurrently using a thread pool."""
        if len(requests) != len(narratives):
            raise ValueError(
                "execute_batch requires matching requests/narratives lengths: "
                f"{len(requests)} requests vs {len(narratives)} narratives."
            )
        if not requests:
            logger.info("No requests submitted for %s; skipping batch.", desc)
            return []

        responses: List[Optional[LLMResponse]] = [None] * len(requests)
        completed = 0
        failed = 0

        logger.info("Submitting %d requests (%s) with %d workers",
                     len(requests), desc, self.max_workers)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self.call_with_cache, req, narr): i
                for i, (req, narr) in enumerate(zip(requests, narratives))
            }
            pending = set(future_to_idx.keys())
            with tqdm(total=len(requests), desc=desc, unit="req") as pbar:
                while pending:
                    done, pending = wait(
                        pending,
                        timeout=self.FUTURE_TIMEOUT,
                        return_when=FIRST_COMPLETED,
                    )

                    # If nothing finished within the timeout window, cancel the batch
                    if not done:
                        logger.error(
                            "No futures completed within %ds for %s. Cancelling %d pending requests",
                            self.FUTURE_TIMEOUT,
                            desc,
                            len(pending),
                        )
                        for fut in list(pending):
                            idx = future_to_idx[fut]
                            fut.cancel()
                            responses[idx] = LLMResponse(
                                request_id=requests[idx].request_id,
                                case_id=requests[idx].case_id,
                                model=requests[idx].model,
                                task=requests[idx].task,
                                success=False,
                                error_message=f"Timed out after {self.FUTURE_TIMEOUT}s (batch wait)",
                                timestamp=datetime.now(),
                            )
                        cancelled = len(pending)
                        completed += cancelled
                        failed += cancelled
                        pbar.update(cancelled)
                        raise RuntimeError(
                            f"Batch stalled: no futures completed within {self.FUTURE_TIMEOUT}s for '{desc}'."
                        )

                    for future in done:
                        idx = future_to_idx[future]
                        try:
                            resp = future.result()
                            responses[idx] = resp
                            completed += 1
                            if resp is not None and not resp.success:
                                failed += 1
                        except CancelledError:
                            responses[idx] = LLMResponse(
                                request_id=requests[idx].request_id,
                                case_id=requests[idx].case_id,
                                model=requests[idx].model,
                                task=requests[idx].task,
                                success=False,
                                error_message="Cancelled due to batch timeout",
                                timestamp=datetime.now(),
                            )
                            completed += 1
                            failed += 1
                        except RuntimeError:
                            # Fatal errors (e.g. bad model params) — cancel remaining work and propagate immediately.
                            for f in pending:
                                f.cancel()
                            raise
                        except Exception as e:
                            logger.error("Unexpected worker error: %s", e)
                            responses[idx] = LLMResponse(
                                request_id=requests[idx].request_id,
                                case_id=requests[idx].case_id,
                                model=requests[idx].model,
                                task=requests[idx].task,
                                success=False,
                                error_message=f"Worker error: {e}",
                                timestamp=datetime.now(),
                            )
                            completed += 1
                            failed += 1
                        pbar.update(1)

                        # Circuit breaker: abort if early results are mostly failures
                        window = min(self.CIRCUIT_BREAKER_WINDOW, len(requests))
                        if completed == window and failed > window * self.CIRCUIT_BREAKER_THRESHOLD:
                            for f in pending:
                                f.cancel()
                            raise RuntimeError(
                                f"Circuit breaker tripped: {failed}/{window} "
                                f"early requests failed for '{desc}'. Aborting batch."
                            )

        return [r for r in responses if r is not None]

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------

    def _record_success(self, task: str, attempt: int) -> None:
        with self._stats_lock:
            self._repair_stats.total_requests += 1
            task_stats = self._task_stats.setdefault(task, RepairStats())
            task_stats.total_requests += 1

            if attempt == 1:
                self._repair_stats.successful_first_attempt += 1
                task_stats.successful_first_attempt += 1
            else:
                self._repair_stats.successful_after_retry += 1
                task_stats.successful_after_retry += 1

    def _record_failure(self, task: str) -> None:
        with self._stats_lock:
            self._repair_stats.total_requests += 1
            self._repair_stats.failed_after_all_attempts += 1
            task_stats = self._task_stats.setdefault(task, RepairStats())
            task_stats.total_requests += 1
            task_stats.failed_after_all_attempts += 1

    def get_repair_stats(self) -> Dict[str, Any]:
        """Return repair statistics as a serialisable dict."""
        with self._stats_lock:
            return {
                "total_requests": self._repair_stats.total_requests,
                "successful_first_attempt": self._repair_stats.successful_first_attempt,
                "successful_after_repair": self._repair_stats.successful_after_repair,
                "successful_after_retry": self._repair_stats.successful_after_retry,
                "failed_after_all_attempts": self._repair_stats.failed_after_all_attempts,
                "repair_strategies_used": dict(self._repair_strategies),
                "by_task": {
                    task: {
                        "total_requests": s.total_requests,
                        "successful_first_attempt": s.successful_first_attempt,
                        "successful_after_repair": s.successful_after_repair,
                        "successful_after_retry": s.successful_after_retry,
                        "failed_after_all_attempts": s.failed_after_all_attempts,
                        "repair_strategies_used": dict(s.repair_strategies_used),
                    }
                    for task, s in self._task_stats.items()
                },
            }
