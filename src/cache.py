"""Disk-based caching system for LLM responses."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.prompts import PROMPT_VERSIONS

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple file-based cache keyed on (model, task, narrative, prompt_version)."""

    TASK_DIRS = ["persona_estimate", "remedy_decision", "persona_rewrite", "neutralize", "dollar_extract"]

    def __init__(self, cache_dir: str = "./cache") -> None:
        self.cache_dir = Path(cache_dir)
        self._ensure_dirs()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, model: str, task: str, narrative: str) -> Optional[Dict[str, Any]]:
        """Look up a cached response. Returns the response dict or None."""
        key = self._key(model, task, narrative)
        path = self._path(task, key)
        if path.exists():
            try:
                with self._lock:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    self._hits += 1
                return data.get("response")
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt cache entry %s – ignoring", path)
                self._delete_path(path)
        with self._lock:
            self._misses += 1
        return None

    def put(
        self,
        model: str,
        task: str,
        narrative: str,
        response: Dict[str, Any],
    ) -> None:
        """Store a response in the cache."""
        key = self._key(model, task, narrative)
        path = self._path(task, key)
        entry = {
            "cache_key": key,
            "model": model,
            "task": task,
            "request_hash": hashlib.blake2b(narrative.encode(), digest_size=16).hexdigest(),
            "response": response,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_version": PROMPT_VERSIONS.get(task, "1.0"),
        }
        payload = json.dumps(entry, ensure_ascii=False, indent=2)
        with self._lock:
            tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)

    def delete(self, model: str, task: str, narrative: str) -> None:
        """Remove a cached response if it exists."""
        path = self._path(task, self._key(model, task, narrative))
        self._delete_path(path)

    def clear(self) -> int:
        """Delete all cached files. Returns number of files removed."""
        count = 0
        for task_dir in self.TASK_DIRS:
            d = self.cache_dir / task_dir
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
                        count += 1
        logger.info("Cache cleared: %d files removed", count)
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
            return "Cache: no lookups yet"
        return f"Cache: {self._hits} hits, {self._misses} misses ({self._hits / total:.0%} hit rate)"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _key(self, model: str, task: str, narrative: str) -> str:
        prompt_version = PROMPT_VERSIONS.get(task, "1.0")
        content = f"{model}|{task}|{prompt_version}|{narrative}"
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

    def _path(self, task: str, key: str) -> Path:
        task_dir = self.cache_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir / f"{key}.json"

    def _delete_path(self, path: Path) -> None:
        with self._lock:
            if path.exists():
                path.unlink()

    def _ensure_dirs(self) -> None:
        for task_dir in self.TASK_DIRS:
            (self.cache_dir / task_dir).mkdir(parents=True, exist_ok=True)
