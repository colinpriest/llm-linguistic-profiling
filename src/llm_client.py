"""OpenAI API wrapper with concurrent execution, caching, and retry logic.

Uses the ``instructor`` library (v2) to guarantee Pydantic-validated JSON
responses from every model, including reasoning models (GPT-5, o-series).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
    CancelledError,
)
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx
import instructor
from openai import (
    AuthenticationError, BadRequestError, NotFoundError,
    OpenAI, PermissionDeniedError,
)
from pydantic import BaseModel
from tqdm import tqdm

from src.cache import ResponseCache
from src.schemas import LLMConfig, LLMRequest, LLMResponse, RepairStats
from src.token_usage import estimate_cost, PRICING_PER_1M_TOKENS

logger = logging.getLogger(__name__)

# Models whose API rejects temperature / max_tokens and requires
# max_completion_tokens instead.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Tasks whose input length varies — truncation on these may be transient
# and should not abort the entire batch on reasoning models.
_VARIABLE_INPUT_TASKS = frozenset({
    "neutralize", "rewrite", "remedy", "persona_estimate",
})


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
        self._http_client = http_client
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

        # Deferred warnings (logged after tqdm completes to avoid breaking progress bar)
        self._deferred_warnings: List[str] = []

        # Token usage tracking (thread-safe)
        self._usage_lock = Lock()
        self._usage_by_model: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying httpx client and release connections."""
        self._http_client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Single request (with retry)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(validated_obj: Any) -> Optional[Dict[str, int]]:
        """Extract usage info from an instructor response, if available."""
        raw_response = getattr(validated_obj, "_raw_response", None)
        if raw_response is None:
            raw_response = getattr(validated_obj, "raw_response", None)
        if raw_response is None and hasattr(validated_obj, "__dict__"):
            raw_response = validated_obj.__dict__.get("_raw_response")

        usage = None
        if raw_response is not None:
            usage = getattr(raw_response, "usage", None)
            if usage is None and isinstance(raw_response, dict):
                usage = raw_response.get("usage")

        if usage is None:
            return None

        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        else:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _record_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._usage_lock:
            stats = self._usage_by_model.setdefault(
                model,
                {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            stats["requests"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion_tokens

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
        max_attempts_eff = max_attempts if max_attempts is not None else self.config.max_retries
        effective_timeout = timeout_override if timeout_override is not None else self.config.timeout

        # Track truncation failures for retry prompt modification (escalating hints)
        truncation_failure_count = 0

        for attempt in range(1, max_attempts_eff + 1):
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

                # Build messages, potentially with conciseness hint on retry
                prompt_user = request.prompt_user
                if truncation_failure_count > 0 and request.task == "neutralize":
                    # Escalating conciseness hints for neutralization retries
                    if truncation_failure_count == 1:
                        concise_hint = (
                            "\n\nIMPORTANT: Keep your response concise. "
                            "The neutralized text should be approximately the same length as "
                            "the original. Focus on the key content and omit minor markers_removed items."
                        )
                    else:
                        # More aggressive on final attempt
                        concise_hint = (
                            "\n\nCRITICAL: Your previous response was too long. "
                            "The neutralized text MUST be SHORTER than the original. "
                            "Summarize any legal citations or statute references rather than quoting verbatim. "
                            "Use an empty array [] for markers_removed to save space."
                        )
                    prompt_user = request.prompt_user + concise_hint

                # instructor handles JSON extraction + Pydantic validation
                validated_obj = self._client.chat.completions.create(
                    model=request.model,
                    response_model=request.expected_schema,
                    max_retries=0,  # we handle retries ourselves
                    timeout=effective_timeout,
                    messages=[
                        {"role": "system", "content": request.prompt_system},
                        {"role": "user", "content": prompt_user},
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

                usage = self._extract_usage(validated_obj)
                prompt_tokens = usage["prompt_tokens"] if usage else 0
                completion_tokens = usage["completion_tokens"] if usage else 0
                if usage:
                    self._record_usage(request.model, prompt_tokens, completion_tokens)

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
                    tokens_prompt=prompt_tokens,
                    tokens_completion=completion_tokens,
                    timestamp=datetime.now(),
                    attempt_number=attempt,
                    repair_applied=False,
                )

            except BadRequestError as e:
                # 400 errors (unsupported params, invalid model, etc.)
                # are non-retryable — raise immediately to stop the run.
                # Extract first line only — Instructor wraps errors verbosely.
                short_err = str(e).split("\n", 1)[0][:200]
                raise RuntimeError(
                    f"Fatal API error for model {request.model!r}: {short_err}"
                ) from e

            except (AuthenticationError, NotFoundError, PermissionDeniedError) as e:
                # Non-retryable auth/permission/resource errors — retrying will
                # never succeed without configuration changes.
                short_err = str(e).split("\n", 1)[0][:200]
                self._record_failure(request.task)
                raise RuntimeError(
                    f"Non-retryable API error for model {request.model!r}: {short_err}"
                ) from e

            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                err_str = str(e).lower()

                # Quota exhaustion is non-retryable — retrying will
                # never succeed until billing is resolved.
                if "insufficient_quota" in err_str:
                    raise RuntimeError(
                        "OpenAI API quota exceeded. "
                        "Check your plan and billing at "
                        "https://platform.openai.com/account/billing"
                    ) from e

                # Response truncation or validation failure — may be retryable
                # for transient issues, but persistent for true output overflow.
                is_truncation = (
                    "max_tokens" in err_str
                    or "length limit" in err_str
                    or "incomplete" in err_str
                )
                if is_truncation:
                    # Tasks whose input length varies (neutralize, rewrite,
                    # remedy) can hit the token ceiling on outlier-long
                    # narratives even on reasoning models.  Only abort the
                    # batch for fixed-length tasks (e.g. commentary) where
                    # truncation signals a systematic budget problem.
                    if is_reasoning and request.task not in _VARIABLE_INPUT_TASKS:
                        # Systematic budget problem — abort the batch.
                        short_err = str(e).split("\n", 1)[0][:200]
                        raise RuntimeError(
                            f"Output truncated by token limit for model "
                            f"{request.model!r} on {request.task}: {short_err}"
                        ) from e
                    # Allow retries for incomplete responses (may be transient).
                    # Only fail definitively after all attempts exhausted.
                    if attempt < max_attempts_eff:
                        # Log full error on retry to help diagnose persistent failures
                        full_err = str(e)[:500]
                        logger.debug(
                            "Response truncated/incomplete for %s (%s/%s), attempt %d. Error: %s. Retrying with conciseness hint...",
                            request.case_id, request.task, request.model, attempt, full_err,
                        )
                        truncation_failure_count += 1  # Escalate conciseness hint on retry
                        time.sleep(self.config.retry_delay * (2 ** (attempt - 1)))
                        continue
                    # All retries exhausted — record the failure with clear messaging.
                    narrative_chars = len(request.prompt_user)
                    # Capture more error context (not just first line)
                    err_full = str(e)[:500]
                    # Store warning for deferred logging (avoid breaking tqdm)
                    with self._stats_lock:
                        self._deferred_warnings.append(
                            f"Response truncated for {request.case_id} ({request.model}/{request.task}) "
                            f"after {max_attempts_eff} attempts: narrative={narrative_chars} chars, "
                            f"API error: {err_full}"
                        )
                    self._record_failure(request.task)
                    return LLMResponse(
                        request_id=request.request_id,
                        case_id=request.case_id,
                        model=request.model,
                        task=request.task,
                        success=False,
                        error_message=(
                            f"Response truncated/incomplete after {max_attempts_eff} attempts: "
                            f"narrative={narrative_chars} chars. "
                            f"API error: {err_full}"
                        ),
                        latency_ms=latency_ms,
                        timestamp=datetime.now(),
                        attempt_number=attempt,
                    )
                if attempt < max_attempts_eff:
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
                        max_attempts_eff, request.case_id, request.task, e,
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
                            f"Failed after {max_attempts_eff} attempts "
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
            except Exception as e:
                logger.warning(
                    "Invalid cache entry for %s/%s (%s); deleting and re-fetching",
                    request.model,
                    request.task,
                    request.case_id,
                    exc_info=True,
                )
                self.cache.delete(request.model, request.task, narrative)

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
        cache_hits = 0
        breaker_checked = False

        logger.info("Submitting %d requests (%s) with %d workers",
                     len(requests), desc, self.max_workers)

        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        force_shutdown = False
        try:
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
                        force_shutdown = True
                        raise RuntimeError(
                            f"Batch stalled: no futures completed within {self.FUTURE_TIMEOUT}s for '{desc}'."
                        )

                    for future in done:
                        idx = future_to_idx[future]
                        try:
                            resp = future.result()
                            responses[idx] = resp
                            completed += 1
                            if resp is not None and resp.success and resp.attempt_number == 0:
                                cache_hits += 1
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
                            force_shutdown = True
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
                        pbar.set_postfix_str(
                            f"cache {cache_hits / completed:.0%}" if completed else ""
                        )

                        # Circuit breaker: abort if early results are mostly failures
                        window = min(self.CIRCUIT_BREAKER_WINDOW, len(requests))
                        if not breaker_checked and completed >= window and failed > window * self.CIRCUIT_BREAKER_THRESHOLD:
                            breaker_checked = True
                            for f in pending:
                                f.cancel()
                            force_shutdown = True
                            raise RuntimeError(
                                f"Circuit breaker tripped: {failed}/{window} "
                                f"early requests failed for '{desc}'. Aborting batch."
                            )
        finally:
            pool.shutdown(wait=not force_shutdown, cancel_futures=force_shutdown)
            # Flush deferred warnings after tqdm completes (avoids breaking progress bar)
            self._flush_deferred_warnings()

        # Log warning if any slots were not filled
        none_count = sum(1 for r in responses if r is None)
        if none_count > 0:
            logger.warning(
                "%d of %d responses were None and will be dropped from results",
                none_count, len(responses),
            )

        return [r for r in responses if r is not None]

    # ------------------------------------------------------------------
    # Token usage reporting
    # ------------------------------------------------------------------

    def usage_summary(self) -> Dict[str, Any]:
        """Return aggregated token usage and estimated cost totals."""
        with self._usage_lock:
            per_model = {
                model: stats.copy() for model, stats in self._usage_by_model.items()
            }

        totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        total_cost = 0.0
        cost_known = False

        for model, stats in per_model.items():
            prompt_tokens = stats["prompt_tokens"]
            completion_tokens = stats["completion_tokens"]
            totals["requests"] += stats["requests"]
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens

            estimated_cost, pricing = estimate_cost(model, prompt_tokens, completion_tokens)
            if estimated_cost is not None and pricing is not None:
                stats["estimated_cost_usd"] = round(estimated_cost, 6)
                stats["pricing_per_1m"] = {
                    "prompt": pricing.prompt_per_1m,
                    "completion": pricing.completion_per_1m,
                }
                total_cost += estimated_cost
                cost_known = True
            else:
                stats["estimated_cost_usd"] = None
                stats["pricing_per_1m"] = None

        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
        totals["estimated_cost_usd"] = round(total_cost, 6) if cost_known else None

        return {
            "totals": totals,
            "per_model": per_model,
            "pricing_reference": {
                model: {"prompt": pricing.prompt_per_1m, "completion": pricing.completion_per_1m}
                for model, pricing in PRICING_PER_1M_TOKENS.items()
            },
        }

    def write_usage_log(self, output_dir: Path, run_metadata: Dict[str, Any]) -> Path:
        """Persist token usage and cost estimates to a JSON log."""
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.usage_summary()
        payload = {
            "metadata": run_metadata,
            "usage": summary,
        }
        output_path = output_dir / f"token_usage_{run_metadata.get('timestamp', 'unknown')}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    def _flush_deferred_warnings(self) -> None:
        """Log any warnings that were deferred during batch execution."""
        with self._stats_lock:
            warnings = self._deferred_warnings
            self._deferred_warnings = []
        for msg in warnings:
            logger.warning(msg)

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
