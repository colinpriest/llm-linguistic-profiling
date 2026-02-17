"""Numeric slot extraction: money, percent/APR, counts, bounds."""
from __future__ import annotations

import re
from typing import List, Tuple

from . import NumericSlot

# ---------------------------------------------------------------------------
# Regex patterns (ordered by specificity for span-priority extraction)
# ---------------------------------------------------------------------------

# Bounds (most specific — may include trailing money keyword)
_BOUND_BETWEEN = re.compile(
    r"""\bbetween\s+\$?\s*(\d[\d,]*(?:\.\d{1,2})?)
        \s*(?:dollars?\s+)?and\s+\$?\s*(\d[\d,]*(?:\.\d{1,2})?)
        (?:\s*(?:dollars?|usd))?\b""",
    re.IGNORECASE | re.VERBOSE,
)

_BOUND_SINGLE = re.compile(
    r"""\b(over|under|less\s+than|more\s+than|at\s+least|at\s+most
        |above|below|exceeding)
        \s+\$?\s*(\d[\d,]*(?:\.\d{1,2})?)
        (?:\s*(?:dollars?|usd))?\b""",
    re.IGNORECASE | re.VERBOSE,
)

# APR (before generic percent to avoid double-match)
_APR_PREFIX = re.compile(r'\bapr\s*(\d+(?:\.\d+)?)\s*%?\b', re.IGNORECASE)
_APR_SUFFIX = re.compile(r'\b(\d+(?:\.\d+)?)\s*%?\s*apr\b', re.IGNORECASE)

# Percent — no trailing \b after % (non-word char has no boundary with punctuation)
_PERCENT = re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:%|percent\b)', re.IGNORECASE)

# Money
_MONEY_DOLLAR = re.compile(r'\$\s*(\d[\d,]*(?:\.\d{1,2})?)', re.IGNORECASE)
_MONEY_WORD = re.compile(
    r'\b(\d[\d,]*(?:\.\d{1,2})?)\s*(?:usd|dollars?)\b', re.IGNORECASE,
)
_MONEY_USD_PREFIX = re.compile(
    r'\busd\s+(\d[\d,]*(?:\.\d{1,2})?)\b', re.IGNORECASE,
)

# Counts
_COUNT_UNITS = r'(?:calls?|disputes?|payments?|letters?|times?)'
_COUNT = re.compile(rf'\b(\d+)\s+({_COUNT_UNITS})\b', re.IGNORECASE)

# Unit normalisation map
_UNIT_SINGULAR = {
    "call": "call", "calls": "call",
    "dispute": "dispute", "disputes": "dispute",
    "payment": "payment", "payments": "payment",
    "letter": "letter", "letters": "letter",
    "time": "time", "times": "time",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_number(s: str) -> float:
    """Parse a numeric string with optional commas."""
    return float(s.replace(",", ""))


def _overlaps(start: int, end: int, used: List[Tuple[int, int]]) -> bool:
    for us, ue in used:
        if start < ue and end > us:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_numeric(text: str) -> List[NumericSlot]:
    """Extract all numeric slots from *text*, returning canonical-able objects.

    Extraction order: bounds → APR → percent → money → counts.
    Overlapping spans are suppressed (first-match wins).
    """
    slots: List[NumericSlot] = []
    used: List[Tuple[int, int]] = []

    # --- Bounds (between) ---
    for m in _BOUND_BETWEEN.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        v1 = _parse_number(m.group(1))
        v2 = _parse_number(m.group(2))
        slots.append(NumericSlot("bound", v1, operator="between_low", raw=m.group()))
        slots.append(NumericSlot("bound", v2, operator="between_high", raw=m.group()))
        used.append((m.start(), m.end()))

    # --- Bounds (single) ---
    for m in _BOUND_SINGLE.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        op = re.sub(r'\s+', '_', m.group(1).lower())
        val = _parse_number(m.group(2))
        slots.append(NumericSlot("bound", val, operator=op, raw=m.group()))
        used.append((m.start(), m.end()))

    # --- APR ---
    for pat in (_APR_PREFIX, _APR_SUFFIX):
        for m in pat.finditer(text):
            if _overlaps(m.start(), m.end(), used):
                continue
            slots.append(NumericSlot("apr", float(m.group(1)), raw=m.group()))
            used.append((m.start(), m.end()))

    # --- Percent ---
    for m in _PERCENT.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        slots.append(NumericSlot("percent", float(m.group(1)), raw=m.group()))
        used.append((m.start(), m.end()))

    # --- Money ---
    for pat in (_MONEY_DOLLAR, _MONEY_WORD, _MONEY_USD_PREFIX):
        for m in pat.finditer(text):
            if _overlaps(m.start(), m.end(), used):
                continue
            slots.append(NumericSlot("money", _parse_number(m.group(1)), raw=m.group()))
            used.append((m.start(), m.end()))

    # --- Counts ---
    for m in _COUNT.finditer(text):
        if _overlaps(m.start(), m.end(), used):
            continue
        unit = _UNIT_SINGULAR.get(m.group(2).lower(), m.group(2).lower())
        slots.append(NumericSlot("count", float(m.group(1)), unit=unit, raw=m.group()))
        used.append((m.start(), m.end()))

    return slots
