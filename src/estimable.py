"""Non-estimable value handling with reason codes.

This module provides a systematic way to handle non-estimable (NE) values
in statistical analyses, replacing raw NaN/inf with structured objects
that carry explanation codes.

Usage
-----
Instead of:
    return float("nan")

Use:
    return NE.zero_denominator("ratio requires non-zero baseline")

In display code:
    formatted = format_value(val, ".3f")  # Returns "—" with footnote for NE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class NECode(Enum):
    """Standard reason codes for non-estimable values."""

    # Data insufficiency
    ZERO_DENOMINATOR = "zero_denom"
    INSUFFICIENT_N = "insuff_n"
    EMPTY_GROUP = "empty_group"
    NO_VARIANCE = "no_variance"
    ALL_IDENTICAL = "all_identical"
    MISSING_DATA = "missing_data"

    # Model failures
    NO_CONVERGENCE = "no_convergence"
    SINGULAR_MATRIX = "singular"
    RANK_DEFICIENT = "rank_deficient"

    # Structural issues
    NOT_APPLICABLE = "not_applicable"
    SKIPPED = "skipped"
    DEGENERATE = "degenerate"

    # Numerical issues
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    NUMERICAL_INSTABILITY = "numerical"


# Human-readable descriptions for each code
NE_DESCRIPTIONS: Dict[NECode, str] = {
    NECode.ZERO_DENOMINATOR: "denominator is zero",
    NECode.INSUFFICIENT_N: "insufficient sample size",
    NECode.EMPTY_GROUP: "one or more groups empty",
    NECode.NO_VARIANCE: "zero variance in data",
    NECode.ALL_IDENTICAL: "all values identical",
    NECode.MISSING_DATA: "required data missing",
    NECode.NO_CONVERGENCE: "model failed to converge",
    NECode.SINGULAR_MATRIX: "singular covariance matrix",
    NECode.RANK_DEFICIENT: "design matrix rank-deficient",
    NECode.NOT_APPLICABLE: "metric not applicable",
    NECode.SKIPPED: "analysis skipped",
    NECode.DEGENERATE: "degenerate case",
    NECode.OVERFLOW: "numerical overflow",
    NECode.UNDERFLOW: "numerical underflow",
    NECode.NUMERICAL_INSTABILITY: "numerical instability",
}

# Short codes for footnote references
NE_SHORT_CODES: Dict[NECode, str] = {
    NECode.ZERO_DENOMINATOR: "a",
    NECode.INSUFFICIENT_N: "b",
    NECode.EMPTY_GROUP: "c",
    NECode.NO_VARIANCE: "d",
    NECode.ALL_IDENTICAL: "e",
    NECode.MISSING_DATA: "f",
    NECode.NO_CONVERGENCE: "g",
    NECode.SINGULAR_MATRIX: "h",
    NECode.RANK_DEFICIENT: "i",
    NECode.NOT_APPLICABLE: "j",
    NECode.SKIPPED: "k",
    NECode.DEGENERATE: "l",
    NECode.OVERFLOW: "m",
    NECode.UNDERFLOW: "n",
    NECode.NUMERICAL_INSTABILITY: "o",
}


@dataclass(frozen=True)
class NEValue:
    """A non-estimable value with reason code and optional detail.

    Note: float(NEValue) returns NaN. Direct comparison of NEValue instances
    uses dataclass field equality, not float comparison.

    Attributes
    ----------
    code : NECode
        The reason code explaining why the value is non-estimable.
    detail : str, optional
        Additional context specific to this instance.
    context : str, optional
        The metric/field name this NE value belongs to.
    """

    code: NECode
    detail: Optional[str] = None
    context: Optional[str] = None

    @property
    def description(self) -> str:
        """Human-readable description of why value is non-estimable."""
        base = NE_DESCRIPTIONS.get(self.code, "unknown reason")
        if self.detail:
            return f"{base}: {self.detail}"
        return base

    @property
    def short_code(self) -> str:
        """Short code for footnote reference (a, b, c, ...)."""
        return NE_SHORT_CODES.get(self.code, "?")

    def __float__(self) -> float:
        """Allow NEValue to be used where float is expected (returns NaN)."""
        return float("nan")

    def __format__(self, format_spec: str) -> str:
        """Allow NEValue in f-strings with numeric format specs (e.g. :.2f)."""
        if not format_spec:
            return repr(self)
        return "N/E"

    def __repr__(self) -> str:
        return f"NE({self.code.value})"


# Type alias for values that might be estimable or not
Estimable = Union[float, int, NEValue]


class NE:
    """Factory class for creating non-estimable values with standard reasons.

    Usage
    -----
    >>> val = NE.zero_denominator("baseline rate is 0.000")
    >>> val = NE.insufficient_n(n=5, min_required=10)
    >>> val = NE.no_convergence("MCMC sampler diverged")
    """

    @staticmethod
    def zero_denominator(detail: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable due to zero denominator (e.g., ratio undefined)."""
        return NEValue(NECode.ZERO_DENOMINATOR, detail, context)

    @staticmethod
    def insufficient_n(
        n: Optional[int] = None,
        min_required: Optional[int] = None,
        detail: Optional[str] = None,
        context: Optional[str] = None,
    ) -> NEValue:
        """Value non-estimable due to insufficient sample size."""
        if n is not None and min_required is not None:
            detail = f"n={n}, need >={min_required}"
        elif n is not None:
            detail = f"n={n}"
        return NEValue(NECode.INSUFFICIENT_N, detail, context)

    @staticmethod
    def empty_group(group_name: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable because one or more groups are empty."""
        detail = f"group '{group_name}' is empty" if group_name else None
        return NEValue(NECode.EMPTY_GROUP, detail, context)

    @staticmethod
    def no_variance(detail: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable due to zero variance."""
        return NEValue(NECode.NO_VARIANCE, detail, context)

    @staticmethod
    def all_identical(value: Optional[Any] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable because all observations are identical."""
        detail = f"all values = {value}" if value is not None else None
        return NEValue(NECode.ALL_IDENTICAL, detail, context)

    @staticmethod
    def missing_data(field: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable due to missing required data."""
        detail = f"'{field}' is missing" if field else None
        return NEValue(NECode.MISSING_DATA, detail, context)

    @staticmethod
    def no_convergence(detail: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable because model failed to converge."""
        return NEValue(NECode.NO_CONVERGENCE, detail, context)

    @staticmethod
    def singular_matrix(detail: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable due to singular covariance matrix."""
        return NEValue(NECode.SINGULAR_MATRIX, detail, context)

    @staticmethod
    def rank_deficient(detail: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Value non-estimable due to rank-deficient design matrix."""
        return NEValue(NECode.RANK_DEFICIENT, detail, context)

    @staticmethod
    def not_applicable(reason: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Metric is not applicable in this context."""
        return NEValue(NECode.NOT_APPLICABLE, reason, context)

    @staticmethod
    def skipped(reason: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Analysis was explicitly skipped."""
        return NEValue(NECode.SKIPPED, reason, context)

    @staticmethod
    def degenerate(detail: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Degenerate case that cannot be estimated."""
        return NEValue(NECode.DEGENERATE, detail, context)

    @staticmethod
    def numerical(detail: Optional[str] = None, context: Optional[str] = None) -> NEValue:
        """Numerical instability prevented estimation."""
        return NEValue(NECode.NUMERICAL_INSTABILITY, detail, context)


def is_estimable(val: Any) -> bool:
    """Check if a value is estimable (not NaN, inf, or NEValue).

    Parameters
    ----------
    val : Any
        Value to check.

    Returns
    -------
    bool
        True if the value is a finite number, False otherwise.
    """
    if isinstance(val, NEValue):
        return False
    if val is None:
        return False
    try:
        return np.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def get_ne_reason(val: Any) -> Optional[NEValue]:
    """Extract NEValue from a value if it's non-estimable.

    Parameters
    ----------
    val : Any
        Value to check.

    Returns
    -------
    Optional[NEValue]
        The NEValue if val is non-estimable, None otherwise.
    """
    if isinstance(val, NEValue):
        return val
    if val is None:
        return NEValue(NECode.MISSING_DATA)
    try:
        f = float(val)
        if np.isnan(f):
            return NEValue(NECode.MISSING_DATA, "value is NaN")
        if np.isinf(f):
            return NEValue(NECode.OVERFLOW if f > 0 else NECode.UNDERFLOW)
        return None
    except (TypeError, ValueError):
        return NEValue(NECode.MISSING_DATA, f"cannot convert {type(val).__name__}")


def format_value(
    val: Any,
    spec: str = ".4f",
    ne_symbol: str = "—",
    include_footnote: bool = True,
) -> Tuple[str, Optional[NEValue]]:
    """Format a value, handling non-estimable cases.

    Parameters
    ----------
    val : Any
        Value to format. Can be a number, NEValue, None, or NaN.
    spec : str
        Format specification for numeric values (e.g., ".3f", ".2e").
    ne_symbol : str
        Symbol to display for non-estimable values.
    include_footnote : bool
        If True, append footnote superscript for NE values.

    Returns
    -------
    Tuple[str, Optional[NEValue]]
        Formatted string and the NEValue if non-estimable.

    Examples
    --------
    >>> format_value(0.123, ".2f")
    ('0.12', None)

    >>> format_value(NE.zero_denominator("rate is 0"), ".2f")
    ('—ᵃ', NEValue(zero_denom))

    >>> format_value(float("nan"), ".2f")
    ('—', NEValue(missing_data))
    """
    ne_val = get_ne_reason(val)
    if ne_val is not None:
        if include_footnote:
            return f"{ne_symbol}<sup>{ne_val.short_code}</sup>", ne_val
        return ne_symbol, ne_val

    try:
        fval = float(val)
        return f"{fval:{spec}}", None
    except (TypeError, ValueError):
        return str(val), None


def format_value_plain(
    val: Any,
    spec: str = ".4f",
    ne_symbol: str = "—",
) -> str:
    """Format a value for plain text (no HTML footnotes).

    Parameters
    ----------
    val : Any
        Value to format.
    spec : str
        Format specification.
    ne_symbol : str
        Symbol for non-estimable values.

    Returns
    -------
    str
        Formatted string.
    """
    formatted, _ = format_value(val, spec, ne_symbol, include_footnote=False)
    return formatted


def collect_footnotes(ne_values: List[NEValue]) -> str:
    """Generate footnote text for a collection of NE values.

    Parameters
    ----------
    ne_values : List[NEValue]
        List of non-estimable values encountered.

    Returns
    -------
    str
        HTML string with footnote explanations, e.g.:
        "<sup>a</sup> denominator is zero; <sup>b</sup> insufficient sample size"
    """
    if not ne_values:
        return ""

    # Deduplicate by code, keeping first detail for each
    seen: Dict[NECode, NEValue] = {}
    for nev in ne_values:
        if nev.code not in seen:
            seen[nev.code] = nev

    parts = []
    for code, nev in sorted(seen.items(), key=lambda x: NE_SHORT_CODES.get(x[0], "z")):
        sup = NE_SHORT_CODES.get(code, "?")
        desc = nev.description
        parts.append(f"<sup>{sup}</sup>&nbsp;{desc}")

    return "; ".join(parts)


def make_footnote_row(ne_values: List[NEValue], colspan: int = 1) -> str:
    """Generate an HTML table row with footnote explanations.

    Parameters
    ----------
    ne_values : List[NEValue]
        List of non-estimable values.
    colspan : int
        Number of columns to span.

    Returns
    -------
    str
        HTML table row with footnotes, or empty string if no NE values.
    """
    footnotes = collect_footnotes(ne_values)
    if not footnotes:
        return ""
    return (
        f'<tr><td colspan="{colspan}" style="font-size:0.85em;color:#666;'
        f'border-top:1px solid #ddd;padding-top:8px;">{footnotes}</td></tr>'
    )


def sanitize_for_json(obj: Any) -> Any:
    """Convert NEValue objects to None for JSON serialization.

    Recursively walks dicts and lists, converting any NEValue to None,
    which is valid JSON (serialised as ``null``).

    Parameters
    ----------
    obj : Any
        Object to sanitize.

    Returns
    -------
    Any
        Object with NEValue replaced by None.
    """
    if isinstance(obj, NEValue):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    return obj
