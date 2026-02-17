"""Polarity slot extraction: event-verb detection with local negation scope."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    def _sent_tokenize(text: str) -> List[str]:
        return _nltk_sent_tokenize(text)
except Exception:
    # Fallback: split on sentence-ending punctuation
    def _sent_tokenize(text: str) -> List[str]:  # type: ignore[misc]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "event_lexicon.yaml"


def load_event_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load event-lexicon config from YAML."""
    p = Path(path) if path else _DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_lemma_index(config: Dict[str, Any]) -> Dict[str, str]:
    """Map each lowered lemma → its canonical event key."""
    index: Dict[str, str] = {}
    for event_key, event_info in config.get("events", {}).items():
        for lemma in event_info.get("lemmas", []):
            index[lemma.lower()] = event_key
    return index


def _token_positions(text: str) -> List[Tuple[str, int, int]]:
    """Return ``[(word_lower, char_start, char_end), ...]``.

    Contraction-aware: keeps "didn't" as one token so negation cues match.
    """
    return [
        (m.group().lower(), m.start(), m.end())
        for m in re.finditer(r"\w+(?:['\u2019]\w+)*", text)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_polarity(
    text: str,
    config: Dict[str, Any],
    anchor_spans: Optional[List[Tuple[int, int]]] = None,
    negation_window: Optional[int] = None,
) -> Dict[str, str]:
    """Extract event→polarity map from *text*.

    Parameters
    ----------
    text : str
        The narrative to analyse.
    config : dict
        Event-lexicon configuration (from YAML or default).
    anchor_spans : list[(int, int)] | None
        Character spans of issue-anchor matches to mask out (avoids
        false-positive event detection on words like "credit" inside
        "credit report").
    negation_window : int | None
        Token lookback window for negation cues.  Falls back to the
        config value, then to 4.

    Returns
    -------
    dict[str, str]
        Mapping of ``event_key → "POS" | "NEG" | "AMBIG"``.
    """
    neg_cfg = config.get("negation", {})
    window = negation_window or neg_cfg.get("window_tokens", 4)
    neg_cues: Set[str] = set()
    for cue in neg_cfg.get("cues", []):
        # Store both the raw cue and a version with punctuation stripped
        neg_cues.add(cue.lower())
        neg_cues.add(re.sub(r"['\u2019]", "", cue.lower()))  # didn't → didnt
    ignore_pats = [p.lower() for p in neg_cfg.get("ignore_patterns", [])]

    lemma_index = _build_lemma_index(config)

    # Tokenise whole document with character offsets
    tokens = _token_positions(text)
    if not tokens:
        return {}

    # Mark tokens that fall inside an anchor span
    in_anchor = [False] * len(tokens)
    if anchor_spans:
        for i, (_w, ts, te) in enumerate(tokens):
            for a_start, a_end in anchor_spans:
                if ts >= a_start and te <= a_end:
                    in_anchor[i] = True
                    break

    # Compute sentence indices for scoping negation
    sentences = _sent_tokenize(text)
    sent_boundaries: List[Tuple[int, int]] = []
    offset = 0
    for sent in sentences:
        idx = text.find(sent, offset)
        if idx == -1:
            idx = offset
        sent_boundaries.append((idx, idx + len(sent)))
        offset = idx + len(sent)

    token_sent = [0] * len(tokens)
    si = 0
    for i, (_w, ts, _te) in enumerate(tokens):
        while si < len(sent_boundaries) - 1 and ts >= sent_boundaries[si][1]:
            si += 1
        token_sent[i] = si

    # Detect events
    events: Dict[str, str] = {}

    for i, (word, _ts, _te) in enumerate(tokens):
        if in_anchor[i]:
            continue
        event_key = lemma_index.get(word)
        if event_key is None:
            continue

        # Determine polarity via negation lookback (same sentence only)
        polarity = "POS"
        lookback_start = max(0, i - window)
        for j in range(lookback_start, i):
            if token_sent[j] != token_sent[i]:
                continue
            prev_word = tokens[j][0]
            if prev_word in neg_cues:
                # Check ignore patterns (e.g. "not only")
                ignored = False
                if j + 1 < len(tokens):
                    bigram = f"{prev_word} {tokens[j + 1][0]}"
                    if bigram in ignore_pats:
                        ignored = True
                if not ignored:
                    polarity = "NEG"
                    break

        # Merge into event map (flag AMBIG on conflicting polarity)
        if event_key in events:
            if events[event_key] != polarity:
                events[event_key] = "AMBIG"
        else:
            events[event_key] = polarity

    return events
