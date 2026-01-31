"""OpenAI API wrapper with concurrent execution, caching, and retry logic.

Uses the ``instructor`` library (v2) to guarantee Pydantic-validated JSON
responses from every model, including reasoning models (GPT-5, o-series).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        # Raw OpenAI client with explicit httpx transport to avoid
        # connection-pool exhaustion under high concurrency.
        transport = httpx.HTTPTransport(
            retries=2,
        )
        http_client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(config.timeout, connect=10),
            limits=httpx.Limits(
                max_connections=max_workers + 10,
                max_keepalive_connections=max_workers,
            ),
        )
        raw_client = OpenAI(
            api_key=config.api_key,
            http_client=http_client,
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

    def _call_once(self, request: LLMRequest) -> LLMResponse:
        """Execute a single LLM request with instructor-managed validation."""
        MAX_ATTEMPTS = self.config.max_retries  # 3

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
                    kwargs["max_completion_tokens"] = max(self.config.max_tokens, 16_000)
                else:
                    kwargs["max_tokens"] = self.config.max_tokens
                    kwargs["temperature"] = self.config.temperature

                # instructor handles JSON extraction + Pydantic validation
                validated_obj = self._client.chat.completions.create(
                    model=request.model,
                    response_model=request.expected_schema,
                    max_retries=0,  # we handle retries ourselves
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
                # Token-limit truncation is non-retryable — the same
                # prompt will hit the same limit every time.
                is_token_limit = (
                    "max_tokens" in err_str
                    or "length limit" in err_str
                    or "incomplete" in err_str
                )
                if is_token_limit:
                    is_reasoning = request.model.startswith(_REASONING_PREFIXES)
                    if is_reasoning:
                        # For reasoning models this is systematic (budget
                        # consumed by reasoning tokens) — abort the batch.
                        raise RuntimeError(
                            f"Fatal: output truncated by token limit for model "
                            f"{request.model!r} on {request.task}: {e}"
                        ) from e
                    # For standard models it's likely an outlier input —
                    # log a warning and fail just this request.
                    logger.warning(
                        "Token limit hit for %s (%s/%s): skipping. %s",
                        request.case_id, request.model, request.task, e,
                    )
                    self._record_failure(request.task)
                    return LLMResponse(
                        request_id=request.request_id,
                        case_id=request.case_id,
                        model=request.model,
                        task=request.task,
                        success=False,
                        error_message=f"Token limit exceeded: {e}",
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
                    logger.error(
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
                        error_message=f"Failed after {MAX_ATTEMPTS} attempts: {e}",
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

    def call_with_cache(self, request: LLMRequest, narrative: str) -> LLMResponse:
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

        response = self._call_once(request)

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

    # Per-future timeout: generous enough for slow API calls + retries,
    # short enough to unblock a stalled batch.
    FUTURE_TIMEOUT = 180  # seconds

    def execute_batch(
        self,
        requests: List[LLMRequest],
        narratives: List[str],
        desc: str = "LLM calls",
    ) -> List[LLMResponse]:
        """Execute a batch of requests concurrently using a thread pool."""
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
            with tqdm(total=len(requests), desc=desc, unit="req") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        resp = future.result(timeout=self.FUTURE_TIMEOUT)
                        responses[idx] = resp
                        completed += 1
                        if resp is not None and not resp.success:
                            failed += 1
                    except TimeoutError:
                        logger.error(
                            "Future timed out after %ds for %s (%s)",
                            self.FUTURE_TIMEOUT,
                            requests[idx].case_id,
                            requests[idx].task,
                        )
                        responses[idx] = LLMResponse(
                            request_id=requests[idx].request_id,
                            case_id=requests[idx].case_id,
                            model=requests[idx].model,
                            task=requests[idx].task,
                            success=False,
                            error_message=f"Timed out after {self.FUTURE_TIMEOUT}s",
                            timestamp=datetime.now(),
                        )
                        completed += 1
                        failed += 1
                    except RuntimeError:
                        # Fatal errors (e.g. bad model params) — cancel
                        # remaining work and propagate immediately.
                        for f in future_to_idx:
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
                        for f in future_to_idx:
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
