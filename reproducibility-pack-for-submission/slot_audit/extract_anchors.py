"""Issue-anchor slot extraction using a configurable YAML dictionary."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "issue_anchors.yaml"


def load_anchor_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load anchor config from YAML, falling back to the bundled default."""
    p = Path(path) if path else _DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_anchors(
    text: str,
    config: Dict[str, Any],
) -> Tuple[Set[str], List[Tuple[int, int]]]:
    """Extract issue-anchor slots from *text*.

    Returns
    -------
    found : set[str]
        Canonical anchor names present in *text*.
    spans : list[tuple[int, int]]
        Character-level ``(start, end)`` spans of every matched phrase.
        Useful for masking before polarity extraction.
    """
    matching_cfg = config.get("matching", {})
    allow_plural = matching_cfg.get("allow_plural_s", True)

    # Normalise text
    text_norm = text.lower()
    if config.get("normalization", {}).get("collapse_whitespace", True):
        text_norm = re.sub(r"\s+", " ", text_norm)

    # Pre-compute excluded-phrase spans
    exclude_spans: List[Tuple[int, int]] = []
    for phrase in config.get("exclude_phrases", []):
        pat = r"\b" + re.escape(phrase.lower()) + r"\b"
        for m in re.finditer(pat, text_norm):
            exclude_spans.append((m.start(), m.end()))

    def _is_excluded(start: int, end: int) -> bool:
        for es, ee in exclude_spans:
            if start >= es and end <= ee:
                return True
        return False

    found: Set[str] = set()
    spans: List[Tuple[int, int]] = []

    for anchor_name, anchor_info in config.get("anchors", {}).items():
        for synonym in anchor_info.get("synonyms", []):
            base = re.escape(synonym.lower())
            pat = r"\b" + base + (r"s?\b" if allow_plural else r"\b")
            for m in re.finditer(pat, text_norm):
                if not _is_excluded(m.start(), m.end()):
                    found.add(anchor_name)
                    spans.append((m.start(), m.end()))

    return found, spans
