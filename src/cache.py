"""Disk-based caching system for LLM responses.

Lock-free design: each cache entry is an independent file keyed by a
content hash, so concurrent reads never conflict.  Writes use a unique
temp file + atomic ``os.replace``, which is safe without a lock.
Hit/miss counters are advisory stats — a minor race on the increment
is acceptable and avoids serializing 100+ threads through a single lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.prompts import PROMPT_VERSIONS

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple file-based cache keyed on (model, task, narrative, prompt_version)."""

    TASK_DIRS = ("persona_estimate", "remedy_decision", "persona_rewrite", "neutralize", "dollar_extract", "fact_extraction", "debiased_triage")

    def __init__(self, cache_dir: str = "./cache") -> None:
        self.cache_dir = Path(cache_dir)
        self._ensure_dirs()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, model: str, task: str, narrative: str) -> Optional[Dict[str, Any]]:
        """Look up a cached response. Returns the response dict or None.

        Lock-free: each key maps to its own file, so concurrent reads
        on different keys touch different files — no shared state.
        Two threads reading the *same* key both get a valid snapshot
        because writes are atomic (see ``put``).
        """
        key = self._key(model, task, narrative)
        path = self._path(task, key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Counter race is benign — stats only, not correctness.
            self._hits += 1
            return data["response"]
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, KeyError, PermissionError, OSError):
            logger.warning("Corrupt cache entry %s – ignoring", path)
            self._delete_path(path)
        self._misses += 1
        return None

    def put(
        self,
        model: str,
        task: str,
        narrative: str,
        response: Dict[str, Any],
    ) -> None:
        """Store a response in the cache.

        Lock-free write strategy:
        1. Serialize to a temp file with a uuid4 suffix — unique per
           call, so concurrent writes never collide on the temp name.
        2. ``os.replace`` atomically swaps the temp file into the final
           path (atomic on both NTFS/Windows and POSIX).  Readers will
           see either the old file or the new file, never a partial write.
        3. If two threads write the *same* key concurrently, both produce
           identical content (same input → same hash → same response),
           so last-writer-wins is correct.

        On Windows, ``os.replace`` can raise ``PermissionError`` when
        another thread holds the target file open for reading.  We retry
        with exponential backoff (up to ~1 s total) to handle this.
        """
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
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            for attempt in range(5):
                try:
                    os.replace(tmp_path, path)
                    return
                except PermissionError:
                    # Windows: target file held open by a concurrent reader
                    time.sleep(0.05 * (2 ** attempt))  # 50ms, 100ms, 200ms, 400ms, 800ms
            # Final attempt — let it raise if still blocked
            try:
                os.replace(tmp_path, path)
            except PermissionError:
                logger.warning("Cache put failed (file locked): %s", path.name)
        finally:
            # Clean up orphaned temp file if it still exists
            self._delete_path(tmp_path)

    def delete(self, model: str, task: str, narrative: str) -> None:
        """Remove a cached response if it exists."""
        path = self._path(task, self._key(model, task, narrative))
        self._delete_path(path)

    def clear(self) -> int:
        """Delete all cached files. Returns number of files removed."""
        count = 0
        if not self.cache_dir.exists():
            return count
        for d in self.cache_dir.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()
                            count += 1
                        except (FileNotFoundError, PermissionError):
                            pass
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
        return self.cache_dir / task / f"{key}.json"

    def _delete_path(self, path: Path) -> None:
        # EAFP instead of lock-protected exists()+unlink() — avoids
        # TOCTOU race and doesn't need a lock.
        try:
            path.unlink()
        except (FileNotFoundError, PermissionError):
            pass

    def _ensure_dirs(self) -> None:
        for task_dir in self.TASK_DIRS:
            (self.cache_dir / task_dir).mkdir(parents=True, exist_ok=True)
