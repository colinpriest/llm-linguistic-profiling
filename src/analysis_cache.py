"""Disk-based caching for expensive analysis results.

Provides caching for severity amplification and Bayesian robustness analyses
which involve many bootstrap iterations or MCMC sampling.

Cache keys are computed from:
- DataFrame content hashes (case_ids, key columns, shape)
- Analysis parameters (n_boot, seed, models)
- Code version (via manual versioning)

Results are stored as JSON files in cache/analysis/{analysis_type}/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Increment when analysis logic changes to invalidate cached results
_ANALYSIS_VERSIONS: Dict[str, str] = {
    "severity_analysis": "1.0",
    "bayes_robustness": "1.1",  # v1.1: ordinal models use softplus cutpoint reparameterization
    "repeated_measures": "1.0",  # v1.0: includes model diagnostics
    "correlation_analysis": "1.0",  # v1.0: includes model diagnostics
}


def _hash_dataframe(df: Optional[pd.DataFrame], name: str = "") -> str:
    """Compute a stable hash of a DataFrame's content.

    Hashes: shape, column names, sorted case_ids (if present), and
    a sample of numeric columns for collision resistance.
    """
    if df is None or len(df) == 0:
        return f"{name}:empty"

    parts = [
        f"shape:{df.shape[0]}x{df.shape[1]}",
        f"cols:{','.join(sorted(df.columns))}",
    ]

    # Include sorted case_ids for determinism
    if "case_id" in df.columns:
        case_ids = sorted(df["case_id"].dropna().unique().tolist())
        parts.append(f"cases:{len(case_ids)}:{hashlib.md5(str(case_ids).encode()).hexdigest()[:16]}")

    # Include tier distribution if present (captures outcome changes)
    if "tier" in df.columns:
        tier_counts = df["tier"].value_counts().sort_index().to_dict()
        parts.append(f"tiers:{tier_counts}")

    # Include model list if present
    if "model" in df.columns:
        models = sorted(df["model"].dropna().unique().tolist())
        parts.append(f"models:{models}")

    content = "|".join(parts)
    return f"{name}:{hashlib.blake2b(content.encode(), digest_size=16).hexdigest()}"


def _compute_cache_key(
    analysis_type: str,
    dataframes: Dict[str, Optional[pd.DataFrame]],
    params: Dict[str, Any],
) -> str:
    """Compute a cache key from DataFrames and parameters."""
    version = _ANALYSIS_VERSIONS.get(analysis_type, "1.0")

    parts = [f"v:{version}"]

    # Hash each DataFrame
    for name, df in sorted(dataframes.items()):
        parts.append(_hash_dataframe(df, name))

    # Include parameters
    for key, value in sorted(params.items()):
        if isinstance(value, list):
            parts.append(f"{key}:{','.join(map(str, value))}")
        else:
            parts.append(f"{key}:{value}")

    combined = "|".join(parts)
    return hashlib.blake2b(combined.encode(), digest_size=24).hexdigest()


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
    ) -> Optional[Dict[str, Any]]:
        """Look up cached analysis result."""
        key = _compute_cache_key(analysis_type, dataframes, params)
        path = self._path(analysis_type, key)

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
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
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Global instance (created lazily)
_analysis_cache: Optional[AnalysisCache] = None


def get_analysis_cache(cache_dir: str = "./cache") -> AnalysisCache:
    """Get or create the global analysis cache instance."""
    global _analysis_cache
    if _analysis_cache is None:
        _analysis_cache = AnalysisCache(cache_dir)
    return _analysis_cache
