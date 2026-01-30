"""OpenAI API wrapper with concurrent execution, caching, and retry logic."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type

from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

from src.cache import ResponseCache
from src.json_utils import (
    JSONExtractionError,
    JSONParseError,
    SchemaValidationError,
    parse_llm_response,
)
from src.schemas import LLMConfig, LLMRequest, LLMResponse, RepairStats

logger = logging.getLogger(__name__)


class LLMClient:
    """Thread-safe OpenAI client with caching, validation, repair, and retry."""

    def __init__(self, config: LLMConfig, cache: ResponseCache, max_workers: int = 100) -> None:
        self.config = config
        self.cache = cache
        self.max_workers = max_workers
        self._client = OpenAI(api_key=config.api_key, timeout=config.timeout)

        # Repair stats (thread-safe)
        self._stats_lock = Lock()
        self._repair_stats = RepairStats()
        self._repair_strategies: Dict[str, int] = {}

        # Per-task stats
        self._task_stats: Dict[str, RepairStats] = {}

    # ------------------------------------------------------------------
    # Single request (with retry + repair)
    # ------------------------------------------------------------------

    def _call_once(self, request: LLMRequest) -> LLMResponse:
        """Execute a single LLM request with validation, repair, and retry."""
        MAX_ATTEMPTS = self.config.max_retries  # 3 = 1 initial + 2 retries

        response_text: Optional[str] = None
        tokens_prompt = 0
        tokens_completion = 0

        for attempt in range(1, MAX_ATTEMPTS + 1):
            t0 = time.perf_counter()
            try:
                raw = self._client.chat.completions.create(
                    model=request.model,
                    messages=[
                        {"role": "system", "content": request.prompt_system},
                        {"role": "user", "content": request.prompt_user},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                response_text = raw.choices[0].message.content or ""
                tokens_prompt = raw.usage.prompt_tokens if raw.usage else 0
                tokens_completion = raw.usage.completion_tokens if raw.usage else 0

                # Parse, repair, validate
                validated_obj, raw_json, repair_applied, strategies = parse_llm_response(
                    response_text, request.expected_schema
                )

                # Track repair stats
                self._record_success(request.task, attempt, repair_applied, strategies)

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
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    timestamp=datetime.now(),
                    attempt_number=attempt,
                    repair_applied=repair_applied,
                )

            except (JSONExtractionError, JSONParseError, SchemaValidationError) as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
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
                        response_raw=response_text,
                        response_parsed=None,
                        error_message=str(e),
                        latency_ms=latency_ms,
                        tokens_prompt=tokens_prompt,
                        tokens_completion=tokens_completion,
                        timestamp=datetime.now(),
                        attempt_number=attempt,
                        repair_applied=False,
                    )

            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if attempt < MAX_ATTEMPTS:
                    logger.debug(
                        "API error on attempt %d for %s (%s): %s. Retrying...",
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
                        response_raw=response_text,
                        response_parsed=None,
                        error_message=f"API error after {MAX_ATTEMPTS} attempts: {e}",
                        latency_ms=latency_ms,
                        tokens_prompt=tokens_prompt,
                        tokens_completion=tokens_completion,
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
    # Batch execution
    # ------------------------------------------------------------------

    def execute_batch(
        self,
        requests: List[LLMRequest],
        narratives: List[str],
        desc: str = "LLM calls",
    ) -> List[LLMResponse]:
        """Execute a batch of requests concurrently using a thread pool."""
        responses: List[Optional[LLMResponse]] = [None] * len(requests)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self.call_with_cache, req, narr): i
                for i, (req, narr) in enumerate(zip(requests, narratives))
            }
            with tqdm(total=len(requests), desc=desc, unit="req") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        responses[idx] = future.result()
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
                    pbar.update(1)

        return [r for r in responses if r is not None]

    # ------------------------------------------------------------------
    # Repair stats tracking
    # ------------------------------------------------------------------

    def _record_success(
        self, task: str, attempt: int, repair_applied: bool, strategies: List[str]
    ) -> None:
        with self._stats_lock:
            self._repair_stats.total_requests += 1
            task_stats = self._task_stats.setdefault(task, RepairStats())
            task_stats.total_requests += 1

            if attempt == 1 and not repair_applied:
                self._repair_stats.successful_first_attempt += 1
                task_stats.successful_first_attempt += 1
            elif repair_applied and attempt == 1:
                self._repair_stats.successful_after_repair += 1
                task_stats.successful_after_repair += 1
            else:
                self._repair_stats.successful_after_retry += 1
                task_stats.successful_after_retry += 1

            for s in strategies:
                self._repair_strategies[s] = self._repair_strategies.get(s, 0) + 1
                task_stats.repair_strategies_used[s] = task_stats.repair_strategies_used.get(s, 0) + 1

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
