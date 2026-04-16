"""Temporal slot extraction: durations, frequencies, year mentions."""
from __future__ import annotations

import re
from typing import List, Tuple

from . import TemporalSlot

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Duration with a leading relation word (more specific — extract first)
_DURATION_REL = re.compile(
    r'\b(for|within|after|over|in)\s+(\d+)\s+(days?|weeks?|months?|years?)\b',
    re.IGNORECASE,
)

# Bare duration (no relation)
_DURATION_BARE = re.compile(
    r'\b(\d+)\s+(days?|weeks?|months?|years?)\b',
    re.IGNORECASE,
)

# Frequency — word forms
_FREQ_WORD = re.compile(r'\b(once|twice|thrice)\b', re.IGNORECASE)

# Frequency — N times / multiple times
_FREQ_NTIMES = re.compile(
    r'\b(two|three|four|five|six|seven|eight|nine|ten|\d+'
    r'|multiple|several)\s+times?\b',
    re.IGNORECASE,
)

# Standalone year
_YEAR = re.compile(r'\b((?:19|20)\d{2})\b')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNIT_SINGULAR = {
    "day": "day", "days": "day",
    "week": "week", "weeks": "week",
    "month": "month", "months": "month",
    "year": "year", "years": "year",
}

_FREQ_WORD_MAP = {
    "once": "1",
    "twice": "2",
    "thrice": "3",
}

_FREQ_NUM_MAP = {
    "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "multiple": "multiple", "several": "several",
}


def _overlaps(start: int, end: int, used: List[Tuple[int, int]]) -> bool:
    for us, ue in used:
        if start < ue and end > us:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_temporal(text: str) -> List[TemporalSlot]:
    """Extract temporal slots from *text*.

    Extraction order: duration-with-relation → bare-duration → frequency → year.
    Overlapping spans suppressed.
    """
    slots: List[TemporalSlot] = []
    used: List[Tuple[int, int]] = []

    # --- Durations with relation ---
    for m in _DURATION_REL.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        rel = m.group(1).lower()
        val = m.group(2)
        unit = _UNIT_SINGULAR.get(m.group(3).lower(), m.group(3).lower())
        slots.append(TemporalSlot("duration", val, unit=unit, relation=rel, raw=m.group()))
        used.append((m.start(), m.end()))

    # --- Bare durations ---
    for m in _DURATION_BARE.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        val = m.group(1)
        unit = _UNIT_SINGULAR.get(m.group(2).lower(), m.group(2).lower())
        slots.append(TemporalSlot("duration", val, unit=unit, raw=m.group()))
        used.append((m.start(), m.end()))

    # --- Frequency (word) ---
    for m in _FREQ_WORD.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        val = _FREQ_WORD_MAP.get(m.group(1).lower(), m.group(1).lower())
        slots.append(TemporalSlot("frequency", val, raw=m.group()))
        used.append((m.start(), m.end()))

    # --- Frequency (N times) ---
    for m in _FREQ_NTIMES.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        raw_val = m.group(1).lower()
        val = _FREQ_NUM_MAP.get(raw_val, raw_val)
        slots.append(TemporalSlot("frequency", val, raw=m.group()))
        used.append((m.start(), m.end()))

    # --- Year ---
    for m in _YEAR.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        slots.append(TemporalSlot("year", m.group(1), raw=m.group()))
        used.append((m.start(), m.end()))

    return slots
