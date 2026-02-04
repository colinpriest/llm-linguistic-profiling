"""Pre-regression diagnostics and targeted fallback helpers.

Every regression model in the project should call ``validate_regression_inputs``
*before* fitting.  If problems are detected, the function returns a structured
diagnostic dict that the caller can use to choose a fallback or raise a fatal
error.

Failure-mode taxonomy
---------------------
* **rank_deficient** – design matrix has fewer linearly independent columns
  than columns (aliased dummies, perfect collinearity).
* **separation_detected** – one or more predictor levels perfectly predict the
  binary outcome (Perfect / quasi-complete separation).
* **sparse_cells** – factor-level combinations have too few observations for
  reliable estimation.
* **convergence_failure** – optimiser did not converge within iteration limit
  (diagnosed *after* a fit attempt; see ``diagnose_fit_failure``).
* **no_variance_y** – response variable is constant.
* **no_variance_x** – one or more predictor columns have zero variance.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public: pre-fit validation
# ---------------------------------------------------------------------------

def validate_regression_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_label: str = "regression",
    is_binary: bool = False,
    min_obs: int = 30,
    vif_threshold: float = 10.0,
    condition_number_threshold: float = 1e10,
) -> Dict[str, Any]:
    """Run all pre-flight checks and return a diagnostic report.

    Parameters
    ----------
    X : predictor matrix (may include a constant column).
    y : response vector.
    model_label : human-readable label for error messages.
    is_binary : set True for logistic / binary models to enable separation
        checks.
    min_obs : minimum sample size.
    vif_threshold : flag collinearity when any VIF exceeds this.
    condition_number_threshold : flag rank deficiency when condition number
        exceeds this.

    Returns
    -------
    Dict with key ``"ok"`` (bool).  If ``ok`` is False, ``"issues"`` contains
    a list of issue dicts, each with ``"type"`` (one of the taxonomy labels)
    and ``"detail"``.
    """
    issues: List[Dict[str, Any]] = []

    # ---- sample size ---------------------------------------------------------
    if len(y) < min_obs:
        issues.append({
            "type": "insufficient_data",
            "detail": f"{model_label}: n={len(y)} < minimum {min_obs}",
        })

    # ---- response variance ---------------------------------------------------
    if y.nunique() < 2:
        val = y.iloc[0] if len(y) > 0 else "N/A"
        issues.append({
            "type": "no_variance_y",
            "detail": f"{model_label}: response variable is constant (value={val})",
        })

    # ---- predictor variance --------------------------------------------------
    constant_cols = [c for c in X.columns if c != "const" and X[c].nunique() < 2]
    if constant_cols:
        issues.append({
            "type": "no_variance_x",
            "detail": (
                f"{model_label}: {len(constant_cols)} predictor(s) have no "
                f"variance: {constant_cols[:10]}"
            ),
            "columns": constant_cols,
        })

    # ---- rank deficiency / collinearity (on non-constant X) ------------------
    X_numeric = X.select_dtypes(include=[np.number])
    non_const = [c for c in X_numeric.columns if c != "const" and X_numeric[c].nunique() > 1]
    if len(non_const) >= 2:
        rank_issues = _check_rank(X_numeric[non_const], model_label, condition_number_threshold)
        issues.extend(rank_issues)

        vif_issues = _check_vif(X_numeric[non_const], model_label, vif_threshold)
        issues.extend(vif_issues)

    # ---- separation (binary models only) ------------------------------------
    if is_binary and len(issues) == 0:
        sep_issues = _check_separation(X, y, model_label)
        issues.extend(sep_issues)

    # ---- sparse cells (categorical predictors) ------------------------------
    sparse_issues = _check_sparse_cells(X, y, model_label)
    issues.extend(sparse_issues)

    if issues:
        primary_type = issues[0]["type"]
        logger.warning(
            "%s: pre-regression validation found %d issue(s), primary: %s",
            model_label, len(issues), primary_type,
        )
        return {"ok": False, "issues": issues, "primary_type": primary_type}

    return {"ok": True, "issues": []}


# ---------------------------------------------------------------------------
# Public: post-fit failure diagnosis
# ---------------------------------------------------------------------------

def diagnose_fit_failure(
    exc: Exception,
    *,
    model_label: str = "regression",
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Classify an exception raised during model fitting.

    Returns a dict with ``"type"`` and ``"detail"`` matching the taxonomy.
    """
    msg = str(exc).lower()

    if "singular" in msg or "not invertible" in msg or "rank" in msg:
        detail = f"{model_label}: singular matrix – design matrix is rank-deficient"
        diag_type = "rank_deficient"
        # Try to identify the aliased columns
        if X is not None:
            aliased = _find_aliased_columns(X)
            if aliased:
                detail += f"; aliased column pairs: {aliased[:5]}"
        return {"type": diag_type, "detail": detail, "aliased": aliased if X is not None else []}

    if "separation" in msg or "perfect" in msg:
        return {"type": "separation_detected", "detail": f"{model_label}: {exc}"}

    if "converge" in msg or "iteration" in msg or "maximum" in msg:
        return {"type": "convergence_failure", "detail": f"{model_label}: {exc}"}

    return {"type": "unknown", "detail": f"{model_label}: {exc}"}


# ---------------------------------------------------------------------------
# Public: fatal error helper
# ---------------------------------------------------------------------------

def fatal_regression_error(
    model_label: str,
    diagnosis: Dict[str, Any],
    *,
    response_var: str = "",
    predictor_vars: Optional[List[str]] = None,
) -> None:
    """Print a detailed fatal error and exit.

    Called when no fallback is available or appropriate.
    """
    diag_type = diagnosis.get("type") or diagnosis.get("primary_type", "unknown")
    detail = diagnosis.get("detail", "")
    if isinstance(diagnosis.get("issues"), list):
        detail = "; ".join(i["detail"] for i in diagnosis["issues"])

    msg_parts = [
        f"FATAL: {model_label} cannot be estimated.",
        f"  Failure mode: {diag_type}",
        f"  Detail: {detail}",
    ]
    if response_var:
        msg_parts.append(f"  Response variable: {response_var}")
    if predictor_vars:
        msg_parts.append(f"  Predictor variables: {predictor_vars}")
    msg_parts.append(
        "  Action: inspect input data for collinearity, constant columns, "
        "or sparse factor-level combinations."
    )

    full_msg = "\n".join(msg_parts)
    logger.critical(full_msg)
    print(full_msg, file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Public: drop aliased / constant columns (replacement for _drop_constant_columns)
# ---------------------------------------------------------------------------

def sanitise_design_matrix(
    X: pd.DataFrame,
    *,
    model_label: str = "regression",
    drop_constant: bool = True,
    drop_aliased: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a cleaned design matrix and list of dropped columns.

    1. Drops columns with zero variance (constant).
    2. Drops linearly dependent columns (keeps the first of each alias set).
    """
    dropped: List[str] = []

    if drop_constant:
        constant = [c for c in X.columns if X[c].nunique() < 2 and c != "const"]
        if constant:
            logger.info("%s: dropping %d constant column(s): %s", model_label, len(constant), constant)
            dropped.extend(constant)
            X = X.drop(columns=constant)

    if drop_aliased and X.shape[1] >= 2:
        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 2:
            # Pivoted QR to detect rank deficiency
            try:
                _, r, piv = np.linalg.qr(numeric.values.astype(float), mode="economic")  # type: ignore[call-overload]
            except Exception:
                # QR doesn't have pivoting in all numpy versions; use SVD
                r = None

            if r is None:
                # SVD approach
                u, s, vt = np.linalg.svd(numeric.values.astype(float), full_matrices=False)
                tol = max(numeric.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 1e-12
                rank = int(np.sum(s > tol))
                if rank < numeric.shape[1]:
                    # Find which columns to drop via column pivoted QR of transpose
                    _, _, perm = np.linalg.svd(numeric.values.astype(float).T, full_matrices=False)
                    # Keep only the first `rank` columns
                    keep_idx = set(range(rank))
                    to_drop = [numeric.columns[i] for i in range(numeric.shape[1]) if i not in keep_idx and numeric.columns[i] != "const"]
                    if to_drop:
                        logger.info(
                            "%s: dropping %d aliased column(s) to restore full rank: %s",
                            model_label, len(to_drop), to_drop,
                        )
                        dropped.extend(to_drop)
                        X = X.drop(columns=to_drop)
            else:
                diag = np.abs(np.diag(r))
                tol = max(numeric.shape) * np.finfo(float).eps * diag[0] if len(diag) > 0 else 1e-12
                rank = int(np.sum(diag > tol))
                if rank < numeric.shape[1]:
                    # Drop columns beyond the rank from the pivoted order
                    to_drop_idx = list(range(rank, numeric.shape[1]))
                    to_drop = [numeric.columns[piv[i]] for i in to_drop_idx if numeric.columns[piv[i]] != "const"]
                    if to_drop:
                        logger.info(
                            "%s: dropping %d aliased column(s) to restore full rank: %s",
                            model_label, len(to_drop), to_drop,
                        )
                        dropped.extend(to_drop)
                        X = X.drop(columns=to_drop)

    return X, dropped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_rank(
    X: pd.DataFrame, model_label: str, threshold: float,
) -> List[Dict[str, Any]]:
    """Check design matrix rank via condition number and SVD."""
    issues: List[Dict[str, Any]] = []
    vals = X.values.astype(float)
    try:
        cond = np.linalg.cond(vals)
    except np.linalg.LinAlgError:
        cond = float("inf")

    if cond > threshold:
        aliased = _find_aliased_columns(X)
        issues.append({
            "type": "rank_deficient",
            "detail": (
                f"{model_label}: design matrix condition number = {cond:.1e} "
                f"(threshold {threshold:.1e}); aliased pairs: {aliased[:5]}"
            ),
            "condition_number": float(cond),
            "aliased_columns": aliased,
        })

    return issues


def _check_vif(
    X: pd.DataFrame, model_label: str, threshold: float,
) -> List[Dict[str, Any]]:
    """Variance inflation factor check for multicollinearity."""
    issues: List[Dict[str, Any]] = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vals = X.values.astype(float)
        high_vif: List[Dict[str, Any]] = []
        for i, col in enumerate(X.columns):
            try:
                vif = variance_inflation_factor(vals, i)
                if np.isfinite(vif) and vif > threshold:
                    high_vif.append({"column": col, "vif": float(vif)})
            except (np.linalg.LinAlgError, ZeroDivisionError):
                high_vif.append({"column": col, "vif": float("inf")})

        if high_vif:
            issues.append({
                "type": "rank_deficient",
                "detail": (
                    f"{model_label}: {len(high_vif)} predictor(s) with VIF > "
                    f"{threshold}: {[(h['column'], round(h['vif'], 1)) for h in high_vif[:5]]}"
                ),
                "high_vif_columns": high_vif,
            })
    except ImportError:
        pass  # statsmodels not available

    return issues


def _check_separation(
    X: pd.DataFrame, y: pd.Series, model_label: str,
) -> List[Dict[str, Any]]:
    """Detect quasi-complete separation in binary outcomes."""
    issues: List[Dict[str, Any]] = []
    sep_cols: List[str] = []
    for col in X.columns:
        if col == "const":
            continue
        for level in X[col].unique():
            subset_y = y[X[col] == level]
            if len(subset_y) > 0 and subset_y.nunique() < 2:
                sep_cols.append(col)
                break

    if sep_cols:
        issues.append({
            "type": "separation_detected",
            "detail": (
                f"{model_label}: quasi-complete separation in {len(sep_cols)} "
                f"predictor(s): {sep_cols[:10]}"
            ),
            "columns": sep_cols,
        })

    return issues


def _check_sparse_cells(
    X: pd.DataFrame, y: pd.Series, model_label: str,
    min_cell_n: int = 5,
) -> List[Dict[str, Any]]:
    """Check for factor-level combinations with very few observations."""
    issues: List[Dict[str, Any]] = []
    # Only check columns that look categorical (few unique values)
    cat_cols = [c for c in X.columns if c != "const" and X[c].nunique() <= 20]
    if not cat_cols:
        return issues

    sparse_combos: List[str] = []
    for col in cat_cols:
        counts = X[col].value_counts()
        sparse_levels = counts[counts < min_cell_n]
        if len(sparse_levels) > 0:
            for level, cnt in sparse_levels.items():
                sparse_combos.append(f"{col}={level} (n={cnt})")

    if sparse_combos:
        issues.append({
            "type": "sparse_cells",
            "detail": (
                f"{model_label}: {len(sparse_combos)} sparse cell(s) with <{min_cell_n} obs: "
                f"{sparse_combos[:10]}"
            ),
            "sparse_cells": sparse_combos,
        })

    return issues


def _find_aliased_columns(X: pd.DataFrame) -> List[Tuple[str, str]]:
    """Find pairs of perfectly correlated columns."""
    aliased: List[Tuple[str, str]] = []
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return aliased
    try:
        corr = numeric.corr().abs()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if j > i and corr.iloc[i, j] > 0.999:
                    aliased.append((c1, c2))
    except Exception:
        pass
    return aliased


def standardise_predictors(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Standardise numeric predictors (mean=0, sd=1) for convergence.

    Returns the standardised DataFrame and a dict mapping column names to
    (mean, std) for back-transformation.  The ``const`` column is left
    untouched.
    """
    stats: Dict[str, Tuple[float, float]] = {}
    X = X.copy()
    for col in X.columns:
        if col == "const":
            continue
        sd = X[col].std()
        if sd > 0:
            mean = X[col].mean()
            X[col] = (X[col] - mean) / sd
            stats[col] = (float(mean), float(sd))
    return X, stats
