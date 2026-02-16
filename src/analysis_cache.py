"""Disk-based caching for expensive analysis results.

Provides caching for severity amplification and Bayesian robustness analyses
which involve many bootstrap iterations or MCMC sampling.

**Two-tier caching strategy:**

1. **High-level cache** - Caches entire analysis results (e.g., severity_analysis).
   Keys are based on DataFrame CONTENT (case_ids + relevant column values), not
   shape. This means adding/removing columns that aren't used in the analysis
   won't invalidate the cache.

2. **Operation-level cache** - Caches individual expensive operations (bootstrap,
   ANOVA). Even if the high-level cache misses, inner operations can still hit
   if the specific input values match. This handles minor data instabilities
   (e.g., 39999 vs 40000 rows) by caching at the granularity of individual
   computations rather than the entire analysis.

Results are stored as JSON files in cache/analysis/{analysis_type}/.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

from src.estimable import NEValue

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Increment when analysis logic changes to invalidate cached results
_ANALYSIS_VERSIONS: Dict[str, str] = {
    "severity_analysis": "1.1",  # v1.1: content-based hashing (not shape-based)
    "bayes_robustness": "3.1",  # v3.1: always-generate postmortem, tier_probs for VI/pathfinder/MAP
    "repeated_measures": "1.0",  # v1.0: includes model diagnostics
    "correlation_analysis": "1.0",  # v1.0: includes model diagnostics
}

# Columns that are relevant for each analysis type (used for content-based hashing)
_ANALYSIS_RELEVANT_COLUMNS: Dict[str, List[str]] = {
    "severity_analysis": [
        "case_id", "model", "tier", "confidence", "monetary_severity",
        "issue_severity", "persona_label", "age", "geographic", "education",
        "socioeconomic", "product_category", "issue_category",
    ],
    "bayes_robustness": [
        "case_id", "model", "tier", "persona_label", "age", "geographic",
        "education", "socioeconomic",
    ],
    "repeated_measures": [
        "case_id", "model", "tier", "variant", "persona_label",
    ],
    "correlation_analysis": [
        "case_id", "model", "tier", "persona_label", "age", "geographic",
        "education", "socioeconomic", "gender", "esl",
    ],
}


def _hash_dataframe(
    df: Optional[pd.DataFrame],
    name: str = "",
    relevant_columns: Optional[List[str]] = None,
) -> str:
    """Compute a stable content-based hash of a DataFrame.

    **Content-based hashing**: Hashes the actual VALUES in relevant columns,
    not the shape. This means:
    - Adding/removing columns not in `relevant_columns` won't change the hash
    - Row count differences only matter if they affect the content

    Parameters
    ----------
    df : DataFrame to hash
    name : Name for the hash component (for debugging)
    relevant_columns : Columns to include in hash. If None, uses all columns.
        Columns not present in the DataFrame are silently ignored.

    Returns
    -------
    Hash string in format "{name}:{hash}"
    """
    if df is None or len(df) == 0:
        return f"{name}:empty"

    # Determine which columns to hash
    if relevant_columns is not None:
        # Only hash columns that exist in both the DataFrame and the relevant list
        cols_to_hash = [c for c in relevant_columns if c in df.columns]
    else:
        cols_to_hash = list(df.columns)

    if not cols_to_hash:
        return f"{name}:no_relevant_cols"

    parts = []

    # Hash case_ids (if present) - the foundation of content identity
    if "case_id" in cols_to_hash:
        case_ids = sorted(df["case_id"].dropna().unique().tolist())
        # Include count AND hash of case_ids for both debugging and collision resistance
        parts.append(f"n_cases:{len(case_ids)}")
        parts.append(f"case_hash:{hashlib.blake2b(str(case_ids).encode(), digest_size=8).hexdigest()}")

    # Hash value distributions for categorical columns
    categorical_cols = ["model", "tier", "confidence", "monetary_severity", "issue_severity",
                        "age", "geographic", "education", "socioeconomic", "gender", "esl",
                        "persona_label", "variant", "product_category", "issue_category"]
    for col in categorical_cols:
        if col in cols_to_hash and col in df.columns:
            value_counts = df[col].value_counts().sort_index().to_dict()
            # Convert any non-string keys to strings for consistent hashing
            value_counts_str = {str(k): v for k, v in value_counts.items()}
            parts.append(f"{col}:{value_counts_str}")

    # For truly content-based hashing, include a sample of row hashes
    # This catches cases where values change even if distributions look the same
    if "case_id" in df.columns and len(df) > 0:
        # Sort by case_id for determinism, take every Nth row for efficiency
        sample_size = min(100, len(df))
        sample_indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
        df_sorted = df.sort_values("case_id").reset_index(drop=True)
        sample_df = df_sorted.iloc[sample_indices][cols_to_hash]
        sample_str = sample_df.to_json(orient="records")
        parts.append(f"sample:{hashlib.blake2b(sample_str.encode(), digest_size=8).hexdigest()}")

    content = "|".join(parts)
    return f"{name}:{hashlib.blake2b(content.encode(), digest_size=16).hexdigest()}"


def _compute_cache_key(
    analysis_type: str,
    dataframes: Dict[str, Optional[pd.DataFrame]],
    params: Dict[str, Any],
    debug: bool = False,
) -> str:
    """Compute a content-based cache key from DataFrames and parameters.

    Uses content-based hashing: only columns relevant to the analysis type
    are included in the hash. This prevents cache invalidation when
    unrelated columns change.
    """
    version = _ANALYSIS_VERSIONS.get(analysis_type, "1.0")
    relevant_cols = _ANALYSIS_RELEVANT_COLUMNS.get(analysis_type)

    parts = [f"v:{version}"]

    # Hash each DataFrame using content-based hashing
    for name, df in sorted(dataframes.items()):
        df_hash = _hash_dataframe(df, name, relevant_columns=relevant_cols)
        parts.append(df_hash)
        if debug:
            logger.info("  Cache key component [%s]: %s", name, df_hash[:80])

    # Include parameters
    for key, value in sorted(params.items()):
        if isinstance(value, list):
            param_str = f"{key}:{','.join(map(str, value))}"
        else:
            param_str = f"{key}:{value}"
        parts.append(param_str)
        if debug:
            logger.info("  Cache key component [param:%s]: %s", key, param_str)

    combined = "|".join(parts)
    result = hashlib.blake2b(combined.encode(), digest_size=24).hexdigest()
    if debug:
        logger.info("  Final cache key: %s", result)
    return result


class AnalysisCache:
    """Cache for expensive analysis results (severity, Bayesian, etc.)."""

    ANALYSIS_TYPES = ["severity_analysis", "bayes_robustness", "repeated_measures", "correlation_analysis"]

    def __init__(self, cache_dir: str = "./cache") -> None:
        self.cache_dir = Path(cache_dir) / "analysis"
        self._ensure_dirs()
        self._hits = 0
        self._misses = 0

    def get(
        self,
        analysis_type: str,
        dataframes: Dict[str, Optional[pd.DataFrame]],
        params: Dict[str, Any],
        debug: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Look up cached analysis result."""
        # Check if any cache file exists for this analysis type (for miss diagnosis)
        type_dir = self.cache_dir / analysis_type
        existing_files = list(type_dir.glob("*.json")) if type_dir.exists() else []

        # Enable debug logging if cache exists but we might miss
        should_debug = debug or (len(existing_files) > 0)
        if should_debug:
            logger.info("Computing cache key for %s...", analysis_type)

        key = _compute_cache_key(analysis_type, dataframes, params, debug=should_debug)
        path = self._path(analysis_type, key)

        # Log cache lookup attempt
        logger.debug("Cache lookup for %s: key=%s", analysis_type, key[:16])

        if path.exists():
            try:
                data = _json_loads_with_ne(path.read_text(encoding="utf-8"))
                # Verify version matches
                stored_version = data.get("analysis_version", "")
                current_version = _ANALYSIS_VERSIONS.get(analysis_type, "1.0")
                if stored_version != current_version:
                    logger.info(
                        "Cache version mismatch for %s: stored=%s, current=%s",
                        analysis_type, stored_version, current_version,
                    )
                    self._delete_path(path)
                    self._misses += 1
                    return None

                self._hits += 1
                logger.info(
                    "Cache HIT for %s (key=%s, created=%s)",
                    analysis_type, key[:12], data.get("created_at", "unknown"),
                )
                return data.get("result")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Corrupt cache entry %s: %s", path, e)
                self._delete_path(path)

        self._misses += 1

        # Log cache miss with diagnosis
        if existing_files:
            # Cache files exist but key doesn't match - log details for debugging
            logger.info(
                "Cache MISS for %s: computed key=%s, found %d existing file(s) with different keys",
                analysis_type, key[:16], len(existing_files),
            )
            # Compare with existing cache to show what changed
            for cache_file in existing_files[:1]:
                try:
                    cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                    cached_shapes = cached_data.get("dataframe_shapes", {})
                    logger.info("  Existing cache shapes: %s", cached_shapes)
                    # Compare shapes
                    for name, df in dataframes.items():
                        cached_shape = cached_shapes.get(name)
                        current_shape = list(df.shape) if df is not None else None
                        if cached_shape != current_shape:
                            logger.info("  MISMATCH in %s: cached=%s, current=%s", name, cached_shape, current_shape)
                except Exception as e:
                    logger.debug("Could not compare with cached file: %s", e)
        else:
            logger.info("Cache MISS for %s: no cached results found", analysis_type)

        return None

    def put(
        self,
        analysis_type: str,
        dataframes: Dict[str, Optional[pd.DataFrame]],
        params: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Store analysis result in cache."""
        key = _compute_cache_key(analysis_type, dataframes, params)
        path = self._path(analysis_type, key)

        entry = {
            "cache_key": key,
            "analysis_type": analysis_type,
            "analysis_version": _ANALYSIS_VERSIONS.get(analysis_type, "1.0"),
            "params": {k: str(v) if isinstance(v, list) else v for k, v in params.items()},
            "dataframe_shapes": {
                name: list(df.shape) if df is not None else None
                for name, df in dataframes.items()
            },
            "result": result,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Convert numpy types to native Python for JSON serialization
        payload = json.dumps(entry, ensure_ascii=False, indent=2, default=_json_default)

        # Atomic write
        tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)

        logger.info("Cached %s result (key=%s)", analysis_type, key[:12])

    def clear(self, analysis_type: Optional[str] = None) -> int:
        """Delete cached analysis results. Returns count deleted."""
        count = 0
        types_to_clear = [analysis_type] if analysis_type else self.ANALYSIS_TYPES

        for atype in types_to_clear:
            d = self.cache_dir / atype
            if d.exists():
                for f in d.iterdir():
                    if f.is_file() and f.suffix == ".json":
                        f.unlink()
                        count += 1

        logger.info("Cleared %d cached analysis result(s)", count)
        return count

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def stats_summary(self) -> str:
        total = self._hits + self._misses
        if total == 0:
            return "Analysis cache: no lookups yet"
        return f"Analysis cache: {self._hits} hits, {self._misses} misses ({self._hits / total:.0%} hit rate)"

    def _path(self, analysis_type: str, key: str) -> Path:
        type_dir = self.cache_dir / analysis_type
        type_dir.mkdir(parents=True, exist_ok=True)
        return type_dir / f"{key}.json"

    def _delete_path(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _ensure_dirs(self) -> None:
        for atype in self.ANALYSIS_TYPES:
            (self.cache_dir / atype).mkdir(parents=True, exist_ok=True)


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types and NEValue."""
    if isinstance(obj, NEValue):
        # Serialize NEValue as a dict with special marker
        return {
            "__ne__": True,
            "code": obj.code.value,
            "detail": obj.detail,
            "context": obj.context,
            "description": obj.description,
        }
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_object_hook(obj: Dict[str, Any]) -> Any:
    """JSON deserializer hook to reconstruct NEValue objects."""
    if obj.get("__ne__") is True:
        from src.estimable import NECode, NEValue as NEV
        code_str = obj.get("code", "missing_data")
        try:
            code = NECode(code_str)
        except ValueError:
            code = NECode.MISSING_DATA
        return NEV(
            code=code,
            detail=obj.get("detail"),
            context=obj.get("context"),
        )
    return obj


def _json_loads_with_ne(text: str) -> Any:
    """Load JSON with NEValue reconstruction."""
    return json.loads(text, object_hook=_json_object_hook)


# Global instance (created lazily)
_analysis_cache: Optional[AnalysisCache] = None
_analysis_cache_lock = threading.Lock()


def get_analysis_cache(cache_dir: str = "./cache") -> AnalysisCache:
    """Get or create the global analysis cache instance."""
    global _analysis_cache
    if _analysis_cache is None:
        with _analysis_cache_lock:
            if _analysis_cache is None:
                _analysis_cache = AnalysisCache(cache_dir)
    else:
        # Warn if caller passes a different cache_dir than the existing singleton
        existing_dir = str(_analysis_cache.cache_dir.parent)
        if str(Path(cache_dir) / "analysis") != str(_analysis_cache.cache_dir):
            logger.warning(
                "get_analysis_cache() called with cache_dir=%r but singleton "
                "already exists with cache_dir=%r; ignoring new value",
                cache_dir, existing_dir,
            )
    return _analysis_cache


# ---------------------------------------------------------------------------
# Operation-Level Cache (Tier 2)
# ---------------------------------------------------------------------------


class OperationCache:
    """Granular cache for individual expensive operations.

    Unlike AnalysisCache which caches entire analysis results, OperationCache
    caches individual operations (bootstrap, ANOVA, etc.) based on their
    specific input values. This provides cache hits even when:
    - The high-level cache misses due to minor data changes
    - Different analyses share common computations

    Cache keys are computed from the actual input VALUES, not DataFrame shapes.
    This means a bootstrap computation for the same set of case_ids and tier
    values will hit the cache even if the parent DataFrame has extra rows.

    Example usage:
        op_cache = get_operation_cache()

        # Wrap an expensive function
        @op_cache.cached("amplification_bootstrap")
        def compute_amplification(df, metric, model, n_boot, seed):
            # expensive bootstrap computation
            return result

        # Or use explicit get/put
        key = op_cache.compute_key("my_op", df, {"n_boot": 1000})
        result = op_cache.get("my_op", key)
        if result is None:
            result = expensive_computation()
            op_cache.put("my_op", key, result)
    """

    VERSION = "1.0"

    def __init__(self, cache_dir: str = "./cache") -> None:
        self.cache_dir = Path(cache_dir) / "operations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def compute_key(
        self,
        operation: str,
        df: Optional[pd.DataFrame],
        params: Dict[str, Any],
        columns: Optional[List[str]] = None,
    ) -> str:
        """Compute a cache key from DataFrame content and parameters.

        Parameters
        ----------
        operation : Name of the operation (e.g., "amplification_bootstrap")
        df : Input DataFrame
        params : Operation parameters
        columns : Columns to include in hash. If None, uses common columns.

        Returns
        -------
        Cache key string
        """
        parts = [f"v:{self.VERSION}", f"op:{operation}"]

        # Hash DataFrame content
        if df is not None and len(df) > 0:
            default_cols = ["case_id", "model", "tier", "persona_label",
                            "monetary_severity", "issue_severity"]
            cols = columns if columns is not None else default_cols
            available_cols = [c for c in cols if c in df.columns]

            if available_cols:
                # Hash based on sorted values in relevant columns
                subset = df[available_cols]
                if "case_id" in subset.columns:
                    subset = subset.sort_values("case_id")

                # Create a content hash
                content = subset.to_json(orient="records")
                content_hash = hashlib.blake2b(content.encode(), digest_size=16).hexdigest()
                parts.append(f"data:{len(df)}:{content_hash}")
            else:
                parts.append(f"data:{len(df)}:no_cols")
        else:
            parts.append("data:empty")

        # Hash parameters
        for key, value in sorted(params.items()):
            if isinstance(value, (list, tuple)):
                parts.append(f"p:{key}:{','.join(map(str, value))}")
            else:
                parts.append(f"p:{key}:{value}")

        combined = "|".join(parts)
        return hashlib.blake2b(combined.encode(), digest_size=20).hexdigest()

    def get(self, operation: str, key: str) -> Optional[Dict[str, Any]]:
        """Look up a cached operation result."""
        path = self._path(operation, key)
        if path.exists():
            try:
                data = _json_loads_with_ne(path.read_text(encoding="utf-8"))
                self._hits += 1
                return data.get("result")
            except (json.JSONDecodeError, KeyError):
                self._delete_path(path)
        self._misses += 1
        return None

    def put(
        self,
        operation: str,
        key: str,
        result: Any,
    ) -> None:
        """Store an operation result in cache."""
        path = self._path(operation, key)
        entry = {
            "operation": operation,
            "cache_key": key,
            "version": self.VERSION,
            "result": result,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = json.dumps(entry, ensure_ascii=False, default=_json_default)

        # Atomic write
        tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)

    def cached(
        self,
        operation: str,
        columns: Optional[List[str]] = None,
        param_keys: Optional[List[str]] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to cache function results based on input DataFrame and params.

        Parameters
        ----------
        operation : Name for this operation in the cache
        columns : DataFrame columns to include in cache key
        param_keys : Function argument names to include in cache key params

        Example
        -------
        @op_cache.cached("bootstrap_ci", columns=["case_id", "tier"], param_keys=["n_boot", "seed"])
        def bootstrap_ci(df, n_boot=1000, seed=42):
            # expensive computation
            return result
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                # Get function signature to map args to names
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Extract DataFrame (assumed to be first positional arg)
                df = None
                for name, value in bound.arguments.items():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break

                # Build params dict from specified keys
                params = {}
                if param_keys:
                    for key in param_keys:
                        if key in bound.arguments:
                            params[key] = bound.arguments[key]

                # Compute cache key and check cache
                cache_key = self.compute_key(operation, df, params, columns)
                cached_result = self.get(operation, cache_key)
                if cached_result is not None:
                    return cached_result

                # Compute and cache
                result = func(*args, **kwargs)
                self.put(operation, cache_key, result)
                return result

            return wrapper
        return decorator

    def clear(self, operation: Optional[str] = None) -> int:
        """Delete cached operation results. Returns count deleted."""
        count = 0
        if operation:
            op_dir = self.cache_dir / operation
            if op_dir.exists():
                for f in op_dir.iterdir():
                    if f.is_file() and f.suffix == ".json":
                        f.unlink()
                        count += 1
        else:
            for op_dir in self.cache_dir.iterdir():
                if op_dir.is_dir():
                    for f in op_dir.iterdir():
                        if f.is_file() and f.suffix == ".json":
                            f.unlink()
                            count += 1
        return count

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def stats_summary(self) -> str:
        total = self._hits + self._misses
        if total == 0:
            return "Operation cache: no lookups yet"
        return f"Operation cache: {self._hits} hits, {self._misses} misses ({self._hits / total:.0%} hit rate)"

    def _path(self, operation: str, key: str) -> Path:
        op_dir = self.cache_dir / operation
        op_dir.mkdir(parents=True, exist_ok=True)
        return op_dir / f"{key}.json"

    def _delete_path(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# Global operation cache instance
_operation_cache: Optional[OperationCache] = None
_operation_cache_lock = threading.Lock()


def get_operation_cache(cache_dir: str = "./cache") -> OperationCache:
    """Get or create the global operation cache instance."""
    global _operation_cache
    if _operation_cache is None:
        with _operation_cache_lock:
            if _operation_cache is None:
                _operation_cache = OperationCache(cache_dir)
    else:
        # Warn if caller passes a different cache_dir than the existing singleton
        existing_dir = str(_operation_cache.cache_dir.parent)
        if str(Path(cache_dir) / "operations") != str(_operation_cache.cache_dir):
            logger.warning(
                "get_operation_cache() called with cache_dir=%r but singleton "
                "already exists with cache_dir=%r; ignoring new value",
                cache_dir, existing_dir,
            )
    return _operation_cache
