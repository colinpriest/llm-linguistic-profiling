"""Reason code to human-readable footnote mapping.

Provides consistent NE (not estimable) reason messaging across all panels
in both HTML and LaTeX output formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Load reason map from JSON (single source of truth)
_REASON_MAP_PATH = Path(__file__).parent.parent / "reason_code_map.json"
REASON_MAP: Dict[str, str] = json.loads(_REASON_MAP_PATH.read_text(encoding="utf-8"))


def get_reason_text(reason_code: str) -> str:
    """Get human-readable text for a reason code.

    Parameters
    ----------
    reason_code : str
        The reason code (e.g., "ALL_ZERO_DENOMINATOR").

    Returns
    -------
    str
        Human-readable explanation, or a fallback message if unknown.
    """
    return REASON_MAP.get(reason_code, f"Not estimable: {reason_code}")


def ordered_unique(seq: List[Any]) -> List[Any]:
    """Return unique items preserving first occurrence order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def footnote_markers(reason_codes: List[str]) -> Dict[str, str]:
    """Assign a, b, c, ... markers to unique reason codes.

    Parameters
    ----------
    reason_codes : List[str]
        List of reason codes (may contain duplicates).

    Returns
    -------
    Dict[str, str]
        Mapping from reason_code to its footnote marker (a, b, c, ...).
    """
    unique_codes = ordered_unique([c for c in reason_codes if c in REASON_MAP])
    return {code: chr(ord('a') + i) for i, code in enumerate(unique_codes)}


def panel_footnotes(reason_codes: List[str]) -> List[Dict[str, str]]:
    """Generate footnote list for a panel.

    Parameters
    ----------
    reason_codes : List[str]
        All reason codes used in the panel rows.

    Returns
    -------
    List[Dict[str, str]]
        List of {"code": ..., "marker": ..., "text": ...} for rendering.
    """
    markers = footnote_markers(reason_codes)
    return [
        {"code": c, "marker": markers[c], "text": REASON_MAP[c]}
        for c in markers
    ]


def format_ne_cell(reason_code: str, markers: Dict[str, str]) -> str:
    """Format an NE cell with footnote marker for HTML.

    Parameters
    ----------
    reason_code : str
        The reason code for this NE cell.
    markers : Dict[str, str]
        Mapping from reason codes to markers (from footnote_markers).

    Returns
    -------
    str
        HTML string like "NE<sup>a</sup>" or just "NE" if no marker.
    """
    marker = markers.get(reason_code, "")
    if marker:
        return f'NE<sup>{marker}</sup>'
    return "NE"


def format_footnotes_html(footnotes: List[Dict[str, str]]) -> str:
    """Render footnotes as HTML list.

    Parameters
    ----------
    footnotes : List[Dict[str, str]]
        Output from panel_footnotes().

    Returns
    -------
    str
        HTML <ul> list of footnotes.
    """
    if not footnotes:
        return ""
    items = [
        f'<li><sup>{fn["marker"]}</sup> {fn["text"]}</li>'
        for fn in footnotes
    ]
    return '<ul class="footnotes" style="font-size:0.85em;margin-top:10px;">\n' + \
           '\n'.join(items) + '\n</ul>'


def format_footnotes_latex(footnotes: List[Dict[str, str]]) -> str:
    """Render footnotes as LaTeX tablenotes.

    Parameters
    ----------
    footnotes : List[Dict[str, str]]
        Output from panel_footnotes().

    Returns
    -------
    str
        LaTeX tablenotes block.
    """
    if not footnotes:
        return ""
    items = [
        f'\\item[\\textsuperscript{{{fn["marker"]}}}] {fn["text"]}'
        for fn in footnotes
    ]
    return '\\begin{tablenotes}\n\\footnotesize\n' + \
           '\n'.join(items) + '\n\\end{tablenotes}'
