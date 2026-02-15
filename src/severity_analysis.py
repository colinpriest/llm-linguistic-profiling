"""Severity amplification analysis for the LLM fairness experiment.

Tests whether LLM bias in complaint resolution is amplified in
high-severity cases.  Two severity metrics are analysed independently:

* ``monetary_severity`` – derived from dollar amounts in complaint text
* ``issue_severity`` – derived from product × issue category mapping

The single public entry point is :func:`run_severity_analysis`.
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

from src.statistics import (
    bootstrap_ci, chi_squared_test, fdr_correction, one_way_anova,
    interpret_cohens_d, interpret_eta_sq,
)
from src.estimable import NE, NEValue, is_estimable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ["low", "medium", "high"]
SEVERITY_METRICS = ["monetary_severity", "issue_severity"]
SEVERITY_ORDINAL: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}

_MIN_CELL_SIZE = 30  # flag cells with fewer observations
_MONETARY_LOW_MAX = 100
_MONETARY_MEDIUM_MAX = 1000


# ---------------------------------------------------------------------------
# Pre-flight Diagnostics for ANOVA
# ---------------------------------------------------------------------------

@dataclass
class ANOVAPreflightResult:
    """Result of ANOVA pre-flight checks."""
    ok: bool
    error: Optional[str] = None
    diagnosis: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    n_obs: int = 0
    n_factors: int = 0
    cell_counts: Optional[Dict[str, int]] = None


def anova_preflight(
    df: pd.DataFrame,
    outcome_col: str,
    factor_cols: List[str],
    min_n: int = 30,
    min_cell_n: int = 2,
    context: str = "ANOVA",
) -> ANOVAPreflightResult:
    """Run pre-flight diagnostics before ANOVA.

    Checks performed:
    1. Required columns exist
    2. Minimum sample size
    3. Factor levels (at least 2 per factor)
    4. Outcome variance (not constant)
    5. NaN/Inf values in outcome
    6. Outcome dtype is numeric or can be coerced
    7. Cell counts (minimum observations per cell for interactions)

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    outcome_col : str
        Name of the outcome/response variable.
    factor_cols : List[str]
        Names of factor (categorical) columns.
    min_n : int
        Minimum total observations required.
    min_cell_n : int
        Minimum observations per cell for interaction terms.
    context : str
        Label for error messages.

    Returns
    -------
    ANOVAPreflightResult
        Dataclass with ok=True if all checks pass, else error details.
    """
    warnings_list: List[str] = []

    # 1. Check required columns exist
    missing_cols = []
    if outcome_col not in df.columns:
        missing_cols.append(outcome_col)
    for fc in factor_cols:
        if fc not in df.columns:
            missing_cols.append(fc)
    if missing_cols:
        return ANOVAPreflightResult(
            ok=False,
            error=f"Missing columns: {missing_cols}",
            diagnosis="missing_columns",
        )

    # 2. Check minimum sample size
    n_obs = len(df)
    if n_obs < min_n:
        return ANOVAPreflightResult(
            ok=False,
            error=f"Insufficient observations: {n_obs} < {min_n}",
            diagnosis="insufficient_n",
            n_obs=n_obs,
        )

    # 3. Check outcome dtype and coercibility
    outcome = df[outcome_col]
    if outcome.dtype == object:
        try:
            outcome = pd.to_numeric(outcome, errors="raise")
        except (ValueError, TypeError) as e:
            return ANOVAPreflightResult(
                ok=False,
                error=f"Outcome '{outcome_col}' cannot be coerced to numeric: {e}",
                diagnosis="outcome_not_numeric",
                n_obs=n_obs,
            )
    elif not np.issubdtype(outcome.dtype, np.number):
        return ANOVAPreflightResult(
            ok=False,
            error=f"Outcome '{outcome_col}' has non-numeric dtype: {outcome.dtype}",
            diagnosis="outcome_not_numeric",
            n_obs=n_obs,
        )

    # 4. Check for NaN/Inf in outcome
    outcome_arr = outcome.to_numpy(dtype=float, na_value=np.nan)
    n_nan = np.isnan(outcome_arr).sum()
    n_inf = np.isinf(outcome_arr).sum()
    if n_nan > 0:
        warnings_list.append(f"Outcome has {n_nan} NaN values")
    if n_inf > 0:
        return ANOVAPreflightResult(
            ok=False,
            error=f"Outcome '{outcome_col}' has {n_inf} Inf values",
            diagnosis="outcome_has_inf",
            n_obs=n_obs,
        )

    # Filter to valid outcome rows for remaining checks
    valid_mask = np.isfinite(outcome_arr)
    n_valid = valid_mask.sum()
    if n_valid < min_n:
        return ANOVAPreflightResult(
            ok=False,
            error=f"Insufficient valid observations after removing NaN: {n_valid} < {min_n}",
            diagnosis="insufficient_valid_n",
            n_obs=n_valid,
        )

    # 5. Check outcome variance
    valid_outcome = outcome_arr[valid_mask]
    if np.unique(valid_outcome).size < 2:
        return ANOVAPreflightResult(
            ok=False,
            error=f"Outcome '{outcome_col}' is constant (value={valid_outcome[0]:.4g})",
            diagnosis="no_variance_y",
            n_obs=n_valid,
        )

    # 6. Check factor levels
    for fc in factor_cols:
        n_levels = df.loc[valid_mask, fc].nunique()
        if n_levels < 2:
            return ANOVAPreflightResult(
                ok=False,
                error=f"Factor '{fc}' has fewer than 2 levels ({n_levels})",
                diagnosis="insufficient_factor_levels",
                n_obs=n_valid,
                n_factors=len(factor_cols),
            )

    # 7. Check cell counts for interactions (only if >1 factor)
    cell_counts: Dict[str, int] = {}
    if len(factor_cols) > 1:
        df_valid = df.loc[valid_mask]
        cell_sizes = df_valid.groupby(factor_cols).size()
        min_cell = cell_sizes.min() if len(cell_sizes) > 0 else 0
        empty_cells = (cell_sizes == 0).sum() if len(cell_sizes) > 0 else 0
        small_cells = (cell_sizes < min_cell_n).sum() if len(cell_sizes) > 0 else 0

        cell_counts = {
            "min_cell_size": int(min_cell),
            "n_empty_cells": int(empty_cells),
            "n_small_cells": int(small_cells),
            "total_cells": len(cell_sizes),
        }

        if min_cell < min_cell_n:
            warnings_list.append(
                f"Some cells have < {min_cell_n} observations (min={min_cell})"
            )
        if empty_cells > 0:
            # Empty cells cause rank deficiency
            return ANOVAPreflightResult(
                ok=False,
                error=f"Design has {empty_cells} empty cells (incomplete crossing)",
                diagnosis="empty_cells",
                n_obs=n_valid,
                n_factors=len(factor_cols),
                cell_counts=cell_counts,
            )

    return ANOVAPreflightResult(
        ok=True,
        warnings=warnings_list,
        n_obs=n_valid,
        n_factors=len(factor_cols),
        cell_counts=cell_counts if cell_counts else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordinal(series: pd.Series) -> pd.Series:
    """Map severity labels to ordinal integers (low=1, medium=2, high=3)."""
    return series.map(SEVERITY_ORDINAL)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-15:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _safe_div(a: float, b: float) -> Any:
    """Division that returns inf when denominator is zero, NE for 0/0."""
    if b == 0:
        if a > 0:
            return float("inf")
        elif a < 0:
            return float("-inf")
        else:
            return NE.zero_denominator("0/0 is undefined")
    return a / b


def _cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> Any:
    """Cohen's kappa for two categorical arrays (manual implementation)."""
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return NE.insufficient_n(n=0, min_required=1, context="kappa")
    # Build confusion matrix
    k = len(labels)
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    mat = np.zeros((k, k), dtype=int)
    for a, b in zip(y1, y2):
        mat[label_idx[a], label_idx[b]] += 1
    po = np.trace(mat) / n  # observed agreement
    row_sums = mat.sum(axis=1) / n
    col_sums = mat.sum(axis=0) / n
    pe = (row_sums * col_sums).sum()  # expected agreement
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


# ---------------------------------------------------------------------------
# Section 1: Descriptive Analysis
# ---------------------------------------------------------------------------

def severity_descriptives(
    complaints_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Distribution, coverage, thresholds, and agreement between metrics.

    Parameters
    ----------
    complaints_df : DataFrame with columns
        [case_id, monetary_severity, issue_severity, product_category, issue_category]
    """
    result: Dict[str, Any] = {}
    n = len(complaints_df)

    # --- monetary_severity ---
    m_sev = complaints_df["monetary_severity"]
    m_valid = m_sev[m_sev != "unknown"]
    m_dist = m_sev.value_counts()
    result["monetary_severity"] = {
        "coverage": float(len(m_valid) / n) if n else 0.0,
        "distribution": {l: int(m_dist.get(l, 0)) for l in SEVERITY_LEVELS + ["unknown"]},
        "thresholds": {"low_max": _MONETARY_LOW_MAX, "medium_max": _MONETARY_MEDIUM_MAX},
    }

    # --- issue_severity ---
    i_sev = complaints_df["issue_severity"]
    i_dist = i_sev.value_counts()
    result["issue_severity"] = {
        "coverage": float(i_sev.isin(SEVERITY_LEVELS).mean()) if n else 0.0,
        "distribution": {l: int(i_dist.get(l, 0)) for l in SEVERITY_LEVELS},
    }

    # Breakdown by product/issue within each severity level
    col = "issue_severity"
    by_product: Dict[str, Dict[str, int]] = {}
    by_issue: Dict[str, Dict[str, int]] = {}
    top_combos: Dict[str, List[Dict[str, Any]]] = {}
    for level in SEVERITY_LEVELS:
        sub = complaints_df[complaints_df[col] == level]
        by_product[level] = sub["product_category"].value_counts().head(10).to_dict()
        by_issue[level] = sub["issue_category"].value_counts().head(10).to_dict()
        combo = (
            sub.groupby(["product_category", "issue_category"])
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        top_combos[level] = [
            {"product": p, "issue": i, "count": int(c)}
            for (p, i), c in combo.items()
        ]
    result["issue_severity"]["by_product"] = by_product
    result["issue_severity"]["by_issue"] = by_issue
    result["issue_severity"]["top_combos"] = top_combos

    # --- Agreement between metrics (exclude unknown monetary) ---
    both = complaints_df[complaints_df["monetary_severity"].isin(SEVERITY_LEVELS)].copy()
    if len(both) >= 10:
        m_ord = _ordinal(both["monetary_severity"]).values
        i_ord = _ordinal(both["issue_severity"]).values

        spearman_r, spearman_p = sp_stats.spearmanr(m_ord, i_ord)
        kappa = _cohens_kappa(both["monetary_severity"].values, both["issue_severity"].values)
        agreement_rate = float((both["monetary_severity"] == both["issue_severity"]).mean())

        ct = pd.crosstab(
            both["monetary_severity"],
            both["issue_severity"],
            dropna=False,
        )
        # Ensure consistent ordering
        ct = ct.reindex(index=SEVERITY_LEVELS, columns=SEVERITY_LEVELS, fill_value=0)

        # Full cross-tab including "unknown" monetary severity
        ct_full = pd.crosstab(
            complaints_df["monetary_severity"],
            complaints_df["issue_severity"],
            dropna=False,
        )
        full_rows = SEVERITY_LEVELS + ["unknown"]
        ct_full = ct_full.reindex(
            index=[r for r in full_rows if r in ct_full.index],
            columns=SEVERITY_LEVELS,
            fill_value=0,
        )

        result["agreement"] = {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "kappa": float(kappa),
            "agreement_rate": agreement_rate,
            "cross_tab": ct.to_dict(),
            "cross_tab_pct": (ct / ct.sum().sum() * 100).round(1).to_dict(),
            "cross_tab_full": ct_full.to_dict(),
        }
    else:
        ne = NE.insufficient_n(context="severity agreement")
        result["agreement"] = {
            "spearman_r": ne,
            "spearman_p": ne,
            "kappa": ne,
            "agreement_rate": ne,
            "cross_tab": {},
            "cross_tab_pct": {},
            "cross_tab_full": {},
        }

    return result


def severity_coverage(
    severity_df: pd.DataFrame,
    personas_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Coverage analysis per severity metric per persona.

    Flags persona × severity cells with < 30 observations.
    """
    result: Dict[str, Any] = {}
    merged = severity_df.merge(
        personas_df[["case_id", "model", "persona_label"]].drop_duplicates(),
        on="case_id",
        how="left",
    )

    for metric in SEVERITY_METRICS:
        valid = merged[metric].isin(SEVERITY_LEVELS) if metric == "monetary_severity" else pd.Series(True, index=merged.index)
        pct_overall = float(valid.mean()) if len(merged) else 0.0

        per_persona: Dict[str, float] = {}
        if "persona_label" in merged.columns:
            for persona, grp in merged.groupby("persona_label"):
                v = grp[metric].isin(SEVERITY_LEVELS) if metric == "monetary_severity" else pd.Series(True, index=grp.index)
                per_persona[str(persona)] = float(v.mean())

        small_cells: list = []
        valid_rows = merged[valid] if metric == "monetary_severity" else merged
        if "persona_label" in valid_rows.columns:
            for (persona, level), grp in valid_rows.groupby(["persona_label", metric]):
                if level in SEVERITY_LEVELS and len(grp) < _MIN_CELL_SIZE:
                    small_cells.append({
                        "severity_metric": metric,
                        "severity_level": str(level),
                        "persona": str(persona),
                        "n": len(grp),
                    })

        result[metric] = {
            "overall_pct": pct_overall,
            "per_persona": per_persona,
            "small_cells": small_cells,
            "n_small_cells": len(small_cells),
        }

    return result


# ---------------------------------------------------------------------------
# Section 2: Severity Main Effects
# ---------------------------------------------------------------------------

def severity_main_effects(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
) -> Dict[str, Any]:
    """Main effects of severity on outcomes for a single model.

    Parameters
    ----------
    decisions_df : DataFrame with columns
        [case_id, model, tier, confidence, {severity_metric}]
    """
    df = decisions_df[decisions_df["model"] == model].copy()
    df = df[df[severity_metric].isin(SEVERITY_LEVELS)]
    if len(df) < 10:
        return {}

    result: Dict[str, Any] = {}

    # Means, rates by level
    means: Dict[str, float] = {}
    favorable: Dict[str, float] = {}
    unfavorable: Dict[str, float] = {}
    info_request: Dict[str, float] = {}
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]
        if len(sub) == 0:
            continue
        means[level] = float(sub["tier"].mean())
        favorable[level] = float((sub["tier"] >= 1).mean())
        unfavorable[level] = float((sub["tier"] == 0).mean())
        if "confidence" in sub.columns:
            info_request[level] = float((sub["confidence"] == "need_more_info").mean())

    result["means_by_level"] = means
    result["favorable_rates"] = favorable
    result["unfavorable_rates"] = unfavorable
    result["info_request_rates"] = info_request

    # One-way ANOVA: tier ~ severity
    groups = [df.loc[df[severity_metric] == l, "tier"].values for l in SEVERITY_LEVELS]
    groups = [g for g in groups if len(g) > 0]
    labels = [l for l in SEVERITY_LEVELS if len(df[df[severity_metric] == l]) > 0]
    result["anova"] = one_way_anova(groups, labels)

    # Chi-square: favorable × severity
    df["favorable"] = (df["tier"] >= 1).astype(int)
    ct = pd.crosstab(df[severity_metric], df["favorable"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        result["chi2_favorable"] = chi_squared_test(ct)
    else:
        ne = NE.insufficient_n(context="chi2 favorable", detail="crosstab requires 2x2 minimum")
        result["chi2_favorable"] = {"chi2": ne, "p_value": ne, "cramers_v": ne}

    # Spearman trend
    ord_sev = _ordinal(df[severity_metric]).values
    tiers = df["tier"].values
    if len(set(ord_sev)) >= 2:
        r, p = sp_stats.spearmanr(ord_sev, tiers)
        result["trend"] = {"spearman_r": float(r), "p_value": float(p)}
    else:
        ne = NE.no_variance(detail="severity has <2 unique levels", context="spearman trend")
        result["trend"] = {"spearman_r": ne, "p_value": ne}

    # Group Ns and CIs for chart
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]["tier"].values
        if len(sub) >= 2:
            mean = float(sub.mean())
            se = float(sub.std(ddof=1) / np.sqrt(len(sub)))
            result.setdefault("ci_by_level", {})[level] = {
                "mean": mean,
                "ci_lower": mean - 1.96 * se,
                "ci_upper": mean + 1.96 * se,
                "n": len(sub),
            }

    return result


def severity_main_effects_all(
    decisions_df: pd.DataFrame,
    models: List[str],
) -> Dict[str, Any]:
    """Run main effects for both metrics × all models."""
    result: Dict[str, Any] = {}
    for metric in SEVERITY_METRICS:
        result[metric] = {}
        for model in models:
            result[metric][model] = severity_main_effects(decisions_df, metric, model)
    return result


# ---------------------------------------------------------------------------
# Section 3: Severity × Persona Interaction
# ---------------------------------------------------------------------------

def bias_by_severity(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
    outcome_col: str = "tier",
) -> Dict[str, Any]:
    """Compute bias metrics at each severity level.

    Bias = spread in mean outcome across personas.

    Parameters
    ----------
    outcome_col : Column to aggregate (default ``"tier"``).  Use ``"tier2"``
        for Tier-2 rate analysis after adding a binary indicator column.
    """
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ]
    if persona_col not in df.columns or outcome_col not in df.columns or len(df) < 10:
        return {}

    result: Dict[str, Any] = {}
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]
        if len(sub) < 5:
            ne = NE.insufficient_n(n=len(sub), min_required=5, context=f"bias@{level}")
            result[level] = {"n": len(sub), "mean_bias": ne,
                             "sd_across_personas": ne, "range": ne,
                             "cv": ne, "ne_reason": ne}
            continue

        persona_means = sub.groupby(persona_col)[outcome_col].mean()
        if len(persona_means) < 2:
            result[level] = {"n": len(sub), "mean_bias": 0.0,
                             "sd_across_personas": 0.0, "range": 0.0, "cv": 0.0}
            continue

        sd = float(persona_means.std(ddof=1))
        rng = float(persona_means.max() - persona_means.min())
        mean_val = float(persona_means.mean())
        if mean_val == 0:
            cv = NE.zero_denominator("mean tier is 0", context=f"cv@{level}")
        else:
            cv = sd / mean_val

        result[level] = {
            "n": len(sub),
            "mean_bias": rng,
            "sd_across_personas": sd,
            "range": rng,
            "cv": cv,  # May be NEValue if mean is 0
            "persona_means": persona_means.to_dict(),
        }

    return result


def two_way_anova_persona_severity(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
    outcome_col: str = "tier",
) -> Dict[str, Any]:
    """Two-way ANOVA: outcome ~ persona + severity + persona×severity.

    Uses statsmodels Type II SS.

    **Status: exploratory / sanity-check only.**  Not used for confirmatory
    claims (see GEE and conditional logistic models instead).
    """
    from src.regression_diagnostics import diagnose_fit_failure

    model_label = f"TwoWayANOVA({severity_metric},{model})"
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ].copy()

    # Rename columns for formula
    col_map = {severity_metric: "severity", persona_col: "persona", outcome_col: "outcome"}
    for old, new in col_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # Run comprehensive pre-flight checks
    preflight = anova_preflight(
        df,
        outcome_col="outcome",
        factor_cols=["persona", "severity"],
        min_n=30,
        min_cell_n=1,
        context=model_label,
    )
    if not preflight.ok:
        return {
            "error": preflight.error,
            "diagnosis": preflight.diagnosis,
            "status": "exploratory",
            "preflight": {
                "n_obs": preflight.n_obs,
                "cell_counts": preflight.cell_counts,
            },
        }

    # Log any preflight warnings
    for warn_msg in preflight.warnings:
        logger.debug("%s preflight warning: %s", model_label, warn_msg)

    # Drop personas that don't appear at every severity level so the
    # interaction design matrix has full column rank.
    cell_counts = df.groupby(["persona", "severity"]).size().unstack(fill_value=0)
    complete_personas = cell_counts.index[cell_counts.min(axis=1) > 0]
    if len(complete_personas) < 2:
        return {"error": "insufficient complete persona×severity cells", "status": "exploratory"}
    df = df[df["persona"].isin(complete_personas)]

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("outcome ~ C(persona) + C(severity) + C(persona):C(severity)", data=df).fit()
            table = anova_lm(fit, typ=2)

        result: Dict[str, Any] = {"status": "exploratory"}
        for source, row_label in [
            ("persona_main", "C(persona)"),
            ("severity_main", "C(severity)"),
            ("interaction", "C(persona):C(severity)"),
            ("residual", "Residual"),
        ]:
            if row_label in table.index:
                ss = max(0.0, float(table.loc[row_label, "sum_sq"]))
                df_val = float(table.loc[row_label, "df"])
                f_val_raw = table.loc[row_label, "F"] if "F" in table.columns else None
                p_val_raw = table.loc[row_label, "PR(>F)"] if "PR(>F)" in table.columns else None
                f_val = float(f_val_raw) if f_val_raw is not None and np.isfinite(float(f_val_raw)) else NE.missing_data(field="F", context=source)
                p_val = float(p_val_raw) if p_val_raw is not None and np.isfinite(float(p_val_raw)) else NE.missing_data(field="p", context=source)
                if is_estimable(f_val) and f_val < 0:
                    f_val = 0.0
                    p_val = 1.0
                ss_resid = max(0.0, float(table.loc["Residual", "sum_sq"])) if "Residual" in table.index else 0
                eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                result[source] = {
                    "SS": ss, "df": df_val, "F": f_val,
                    "p_value": p_val if (is_estimable(p_val) and np.isfinite(p_val)) else (p_val if isinstance(p_val, NEValue) else NE.numerical(context=f"p-value {source}")),
                    "eta_sq_partial": eta_sq,
                }
            else:
                ne = NE.missing_data(field=row_label, context=f"ANOVA {source}")
                result[source] = {"SS": 0, "df": 0, "F": ne, "p_value": ne, "eta_sq_partial": 0.0}

        return result

    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label)
        logger.warning("Two-way ANOVA failed (%s): %s", fit_diag["type"], e)
        return {"error": str(e), "diagnosis": fit_diag["type"], "status": "exploratory"}


def simple_effects_by_severity(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
    outcome_col: str = "tier",
) -> Dict[str, Any]:
    """One-way ANOVA of persona effect at each severity level."""
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ]
    if persona_col not in df.columns or outcome_col not in df.columns:
        return {}

    result: Dict[str, Any] = {}
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]
        personas = sub[persona_col].unique()
        if len(personas) < 2 or len(sub) < 10:
            ne = NE.insufficient_n(n=len(sub), min_required=10, context=f"simple_effects@{level}")
            result[level] = {"eta_sq_partial": ne,
                             "largest_pairwise_d": ne,
                             "n_significant": 0, "n": len(sub)}
            continue

        # One-way ANOVA
        groups = [sub.loc[sub[persona_col] == p, outcome_col].values for p in personas]
        groups = [g for g in groups if len(g) > 0]
        labels = [str(p) for p in personas if len(sub[sub[persona_col] == p]) > 0]
        anova_res = one_way_anova(groups, labels)

        # Pairwise Cohen's d (find largest)
        max_d = 0.0
        n_sig = 0
        p_values = []
        if len(groups) >= 2:
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    d = abs(_cohens_d(groups[i], groups[j]))
                    max_d = max(max_d, d)
                    # T-test for significance counting
                    if len(groups[i]) >= 2 and len(groups[j]) >= 2:
                        # Skip pairs where both groups have zero variance —
                        # ttest_ind returns NaN (0/0) when std=0 in both groups
                        if np.std(groups[i]) == 0 and np.std(groups[j]) == 0:
                            continue
                        # ttest_ind emits "Precision loss … catastrophic cancellation"
                        # on low-variance ordinal data.  Results remain valid.
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", category=RuntimeWarning,
                                message=".*Precision loss.*catastrophic cancellation.*",
                            )
                            _, p = sp_stats.ttest_ind(groups[i], groups[j])
                        if not np.isfinite(p) or not (0 <= p <= 1):
                            raise ValueError(
                                f"ttest_ind returned invalid p-value {p!r} for "
                                f"severity={level}, pair ({labels[i]}, {labels[j]})"
                            )
                        p_values.append(p)

            # FDR correction
            if p_values:
                adj_p = fdr_correction(p_values)
                n_sig = int((adj_p < 0.05).sum())

        ne_missing = NE.missing_data(context=f"simple effects {level}")
        result[level] = {
            "eta_sq_partial": anova_res.get("eta_sq_partial", ne_missing),
            "eta_sq_interpretation": anova_res.get("eta_sq_interpretation", interpret_eta_sq(anova_res.get("eta_sq_partial", 0.0))),
            "F": anova_res.get("F", ne_missing),
            "p_value": anova_res.get("p_value", ne_missing),
            "largest_pairwise_d": max_d,
            "largest_pairwise_d_interpretation": interpret_cohens_d(max_d),
            "n_significant": n_sig,
            "n": len(sub),
        }

    return result


def amplification_ratio(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
    n_boot: int = 1000,
    seed: int = 42,
    outcome_col: str = "tier",
) -> Dict[str, Any]:
    """Amplification ratio = bias_high / bias_low with bootstrap CI.

    Bias is measured as SD of mean outcome across personas.

    Uses operation-level caching to avoid recomputing expensive bootstrap
    when the same data is analyzed in different runs.
    """
    from src.analysis_cache import get_operation_cache

    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ].copy()
    if persona_col not in df.columns or outcome_col not in df.columns or len(df) < 30:
        ne = NE.insufficient_n(n=len(df), min_required=30, context="amplification")
        return {"point": ne, "ci_lower": ne, "ci_upper": ne, "ne_reason": ne}

    def _bias_at_level(data: pd.DataFrame, level: str):
        """Calculate bias (SD across personas) at a severity level."""
        sub = data[data[severity_metric] == level]
        if len(sub) < 5:
            return NE.insufficient_n(n=len(sub), min_required=5, context=f"bias@{level}")
        pm = sub.groupby(persona_col)[outcome_col].mean()
        if len(pm) < 2:
            return NE.insufficient_n(n=len(pm), min_required=2, context=f"personas@{level}")
        return float(pm.std(ddof=1))

    bias_low = _bias_at_level(df, "low")
    bias_high = _bias_at_level(df, "high")

    # Check if either bias is non-estimable
    if isinstance(bias_low, NEValue) or isinstance(bias_high, NEValue):
        ne = bias_low if isinstance(bias_low, NEValue) else bias_high
        return {"point": ne, "ci_lower": ne, "ci_upper": ne,
                "bias_low": bias_low, "bias_high": bias_high, "ne_reason": ne}

    # Calculate ratio (amplification)
    if bias_low == 0:
        point = NE.zero_denominator("low-severity bias SD is 0.000", context="amplification")
    else:
        point = bias_high / bias_low

    # Check operation cache for bootstrap results
    op_cache = get_operation_cache()
    cache_params = {
        "severity_metric": severity_metric,
        "model": model,
        "persona_col": persona_col,
        "outcome_col": outcome_col,
        "n_boot": n_boot,
        "seed": seed,
    }
    cache_cols = ["case_id", severity_metric, persona_col, outcome_col]
    cache_key = op_cache.compute_key("amplification_bootstrap", df, cache_params, cache_cols)
    cached = op_cache.get("amplification_bootstrap", cache_key)
    if cached is not None:
        # Use cached bootstrap CIs, update point estimate with current data
        # Handle NEValue point estimates
        if isinstance(point, NEValue):
            return {
                "point": point, "ci_lower": point, "ci_upper": point,
                "bias_low": bias_low, "bias_high": bias_high, "ne_reason": point,
            }
        ne_ci = NE.missing_data(context="bootstrap CI")
        return {
            "point": float(point) if np.isfinite(point) else NE.numerical(context="amp ratio"),
            "ci_lower": cached.get("ci_lower", ne_ci),
            "ci_upper": cached.get("ci_upper", ne_ci),
            "bias_low": bias_low,
            "bias_high": bias_high,
        }

    # If point is already NEValue, skip bootstrap
    if isinstance(point, NEValue):
        return {
            "point": point, "ci_lower": point, "ci_upper": point,
            "bias_low": bias_low, "bias_high": bias_high, "ne_reason": point,
        }

    # Bootstrap CI: resample at complaint (case_id) level
    rng = np.random.default_rng(seed)
    case_ids = df["case_id"].unique()
    boot_ratios = []
    grouped = {cid: sub_df for cid, sub_df in df.groupby("case_id")}
    for _ in tqdm(range(n_boot), desc="  bootstrap(amplification)", leave=False, delay=1):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        boot_df = pd.concat([grouped[sid] for sid in sample_ids], ignore_index=True)
        b_low = _bias_at_level(boot_df, "low")
        b_high = _bias_at_level(boot_df, "high")
        # Skip non-estimable bootstrap samples
        if is_estimable(b_low) and is_estimable(b_high) and b_low > 0:
            boot_ratios.append(b_high / b_low)

    if len(boot_ratios) >= 10:
        ci_lower = float(np.percentile(boot_ratios, 2.5))
        ci_upper = float(np.percentile(boot_ratios, 97.5))
    else:
        ci_ne = NE.insufficient_n(n=len(boot_ratios), min_required=10, context="bootstrap CI")
        ci_lower = ci_ne
        ci_upper = ci_ne

    # Cache the bootstrap results
    op_cache.put("amplification_bootstrap", cache_key, {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_boot": n_boot,
        "n_valid_boots": len(boot_ratios),
    })

    return {
        "point": float(point) if is_estimable(point) else point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bias_low": bias_low,
        "bias_high": bias_high,
    }


# ---------------------------------------------------------------------------
# Section 3b: Dimension-Specific Severity Interaction (One-Way)
# ---------------------------------------------------------------------------

def dimension_severity_interaction(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    dimension: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-way ANOVA: tier ~ dimension + severity + dimension:severity.

    Tests severity amplification for a SINGLE demographic dimension,
    yielding much better-powered tests (~200+ obs per cell vs ~15-20 for
    the full 36-persona grid).

    Uses operation-level caching to avoid recomputing expensive bootstrap
    when the same data is analyzed in different runs.

    Parameters
    ----------
    decisions_df : DataFrame with [case_id, model, tier, {severity_metric}, {dimension}]
    severity_metric : ``"monetary_severity"`` or ``"issue_severity"``
    model : Model identifier.
    dimension : One of the causal dimensions (age, geographic, education, socioeconomic).
    n_boot : Number of bootstrap iterations.
    seed : Random seed.

    Returns
    -------
    Dict with keys: anova, amplification, dimension_means_by_severity,
    cell_sizes, min_cell_size, dimension, severity_metric.
    """
    from src.analysis_cache import get_operation_cache

    model_label = f"DimSevANOVA({severity_metric},{model},{dimension})"
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ].copy()

    if dimension not in df.columns:
        return {"error": f"missing dimension column: {dimension}", "dimension": dimension}

    df = df.dropna(subset=[dimension])

    # Rename columns for consistent handling
    df_anova = df.rename(columns={severity_metric: "severity", dimension: "dim_level"})

    # Run comprehensive pre-flight checks
    preflight = anova_preflight(
        df_anova,
        outcome_col="tier",
        factor_cols=["dim_level", "severity"],
        min_n=30,
        min_cell_n=1,
        context=model_label,
    )
    if not preflight.ok:
        return {
            "error": preflight.error,
            "diagnosis": preflight.diagnosis,
            "dimension": dimension,
            "preflight": {
                "n_obs": preflight.n_obs,
                "cell_counts": preflight.cell_counts,
            },
        }

    # Log any preflight warnings
    for warn_msg in preflight.warnings:
        logger.debug("%s preflight warning: %s", model_label, warn_msg)

    result: Dict[str, Any] = {"dimension": dimension, "severity_metric": severity_metric}

    # Cell sizes
    cell_sizes = df.groupby([dimension, severity_metric]).size().to_dict()
    result["cell_sizes"] = {f"{d}_{s}": int(n) for (d, s), n in cell_sizes.items()}
    result["min_cell_size"] = min(cell_sizes.values()) if cell_sizes else 0

    # Two-way ANOVA: tier ~ dimension + severity + dimension:severity
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        # Ensure every dim_level appears at every severity level for full-rank interaction
        cc = df_anova.groupby(["dim_level", "severity"]).size().unstack(fill_value=0)
        complete_levels = cc.index[cc.min(axis=1) > 0]
        if len(complete_levels) < 2:
            raise ValueError("insufficient complete dimension×severity cells")
        df_anova = df_anova[df_anova["dim_level"].isin(complete_levels)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("tier ~ C(dim_level) + C(severity) + C(dim_level):C(severity)",
                          data=df_anova).fit()
            table = anova_lm(fit, typ=2)

        ss_resid = max(0.0, float(table.loc["Residual", "sum_sq"])) if "Residual" in table.index else 0
        anova_result: Dict[str, Any] = {}
        for source, row_label in [
            ("dimension_main", "C(dim_level)"),
            ("severity_main", "C(severity)"),
            ("interaction", "C(dim_level):C(severity)"),
        ]:
            if row_label in table.index:
                ss = max(0.0, float(table.loc[row_label, "sum_sq"]))
                f_val_raw = table.loc[row_label, "F"] if "F" in table.columns else None
                p_val_raw = table.loc[row_label, "PR(>F)"] if "PR(>F)" in table.columns else None
                f_val = float(f_val_raw) if f_val_raw is not None and np.isfinite(float(f_val_raw)) else NE.missing_data(field="F", context=source)
                p_val = float(p_val_raw) if p_val_raw is not None and np.isfinite(float(p_val_raw)) else NE.missing_data(field="p", context=source)
                if is_estimable(f_val) and f_val < 0:
                    f_val, p_val = 0.0, 1.0
                eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                anova_result[source] = {
                    "F": f_val if is_estimable(f_val) else NE.numerical(context=f"ANOVA F for {source}"),
                    "p_value": p_val if is_estimable(p_val) else NE.numerical(context=f"ANOVA p for {source}"),
                    "eta_sq_partial": eta_sq,
                    "eta_sq_interpretation": interpret_eta_sq(eta_sq),
                }
            else:
                ne = NE.missing_data(field=row_label, context=f"dimension ANOVA {source}")
                anova_result[source] = {"F": ne, "p_value": ne,
                                        "eta_sq_partial": 0.0, "eta_sq_interpretation": "N/A"}
        result["anova"] = anova_result

    except Exception as e:
        logger.warning("Dimension severity ANOVA failed for %s: %s", dimension, e)
        result["anova"] = {"error": str(e)}

    # Means by dimension level at each severity
    means_table: Dict[str, Dict[str, float]] = {}
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]
        if len(sub) > 0:
            means_table[level] = {str(k): float(v) for k, v in sub.groupby(dimension)["tier"].mean().items()}
    result["dimension_means_by_severity"] = means_table

    # Amplification ratio: bias_high / bias_low
    # where bias = max(dim_mean) - min(dim_mean) at a severity level
    def _dim_bias(level: str) -> Any:
        means = means_table.get(level, {})
        if len(means) < 2:
            return NE.insufficient_n(n=len(means), min_required=2, context=f"dim bias {level}")
        vals = list(means.values())
        return max(vals) - min(vals)

    bias_low = _dim_bias("low")
    bias_high = _dim_bias("high")
    if not is_estimable(bias_low) or not is_estimable(bias_high):
        amp_point = NE.missing_data(context="dim amplification (insufficient data)")
    else:
        amp_point = _safe_div(bias_high, bias_low)

    # Check operation cache for bootstrap results
    op_cache = get_operation_cache()
    cache_params = {
        "severity_metric": severity_metric,
        "model": model,
        "dimension": dimension,
        "n_boot": n_boot,
        "seed": seed,
    }
    cache_cols = ["case_id", severity_metric, dimension, "tier"]
    cache_key = op_cache.compute_key(f"dim_severity_bootstrap_{dimension}", df, cache_params, cache_cols)
    cached = op_cache.get(f"dim_severity_bootstrap_{dimension}", cache_key)
    if cached is not None:
        ne_ci = NE.missing_data(context="dimension bootstrap CI")
        amp_pt = float(amp_point) if is_estimable(amp_point) and np.isfinite(amp_point) else (amp_point if isinstance(amp_point, NEValue) else NE.numerical(context="dim amp ratio"))
        result["amplification"] = {
            "bias_low": bias_low,
            "bias_high": bias_high,
            "point": amp_pt,
            "ci_lower": cached.get("ci_lower", ne_ci),
            "ci_upper": cached.get("ci_upper", ne_ci),
        }
        return result

    # Bootstrap CI on amplification ratio
    rng = np.random.default_rng(seed)
    case_ids = df["case_id"].unique()
    boot_ratios: list = []
    grouped = {cid: sub_df for cid, sub_df in df.groupby("case_id")}
    for _ in tqdm(range(n_boot), desc=f"  bootstrap(dim={dimension})", leave=False, delay=1):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        boot_df = pd.concat([grouped[sid] for sid in sample_ids], ignore_index=True)
        b_low_means = boot_df[boot_df[severity_metric] == "low"].groupby(dimension)["tier"].mean()
        b_high_means = boot_df[boot_df[severity_metric] == "high"].groupby(dimension)["tier"].mean()
        if len(b_low_means) >= 2 and len(b_high_means) >= 2:
            b_low = max(b_low_means) - min(b_low_means)
            b_high = max(b_high_means) - min(b_high_means)
            if b_low > 0:
                boot_ratios.append(b_high / b_low)

    if len(boot_ratios) >= 10:
        ci_lo: Any = float(np.percentile(boot_ratios, 2.5))
        ci_hi: Any = float(np.percentile(boot_ratios, 97.5))
    else:
        ci_lo = ci_hi = NE.insufficient_n(n=len(boot_ratios), min_required=10, context="dim bootstrap CI")

    # Cache the bootstrap results
    op_cache.put(f"dim_severity_bootstrap_{dimension}", cache_key, {
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_boot": n_boot,
        "n_valid_boots": len(boot_ratios),
    })

    amp_pt = float(amp_point) if is_estimable(amp_point) and np.isfinite(amp_point) else (amp_point if isinstance(amp_point, NEValue) else NE.numerical(context="dim amp ratio"))
    result["amplification"] = {
        "bias_low": bias_low,
        "bias_high": bias_high,
        "point": amp_pt,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
    }

    return result


def dimension_severity_all(
    decisions_df: pd.DataFrame,
    models: List[str],
    dimensions: List[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run dimension-specific severity interaction for all dims x metrics x models.

    Returns
    -------
    Dict[metric][model][dimension] -> dimension_severity_interaction result.
    """
    result: Dict[str, Any] = {m: {mdl: {} for mdl in models} for m in SEVERITY_METRICS}

    tasks = [
        (metric, model, dim)
        for metric in SEVERITY_METRICS
        for model in models
        for dim in dimensions
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                dimension_severity_interaction,
                decisions_df, metric, model, dim,
                n_boot=n_boot, seed=seed,
            ): (metric, model, dim)
            for metric, model, dim in tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Dimension severity", leave=False, delay=1,
        ):
            metric, model, dim = futures[future]
            result[metric][model][dim] = future.result()

    return result


# ---------------------------------------------------------------------------
# Section 3c: Logistic GEE Interaction (Pre-registered Tier2 model)
# ---------------------------------------------------------------------------

def logistic_interaction_tier2(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    dimension: str,
    cluster_col: str = "case_id",
) -> Dict[str, Any]:
    """Pre-registered complaint-clustered logistic model for Y2.

    Fits: logit P(Y2=1) = persona + severity + persona x severity
    with GEE exchangeable correlation clustered by complaint.

    This is the confirmatory interaction test specified in the
    pre-registration plan.

    Parameters
    ----------
    decisions_df : DataFrame with [case_id, model, tier, {severity_metric}, {dimension}].
    severity_metric : ``"monetary_severity"`` or ``"issue_severity"``.
    model : Model identifier.
    dimension : Demographic dimension to test.
    cluster_col : Column for GEE clustering (default ``"case_id"``).

    Returns
    -------
    Dict with keys: interaction_p, coefficients, n_obs, n_clusters,
    method, convergence_warning.
    """
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ].copy()

    if dimension not in df.columns or len(df) < 30:
        return {"skipped": True, "reason": "insufficient data"}

    df = df.dropna(subset=[dimension])
    df["Y2"] = (df["tier"] >= 2).astype(int)

    # Need variation in outcome
    if df["Y2"].nunique() < 2:
        return {"skipped": True, "reason": "degenerate outcome (no Y2 variation)"}

    # Need at least 2 levels per factor
    if df[dimension].nunique() < 2 or df[severity_metric].nunique() < 2:
        return {"skipped": True, "reason": "insufficient factor levels"}

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable

        df_fit = df.rename(columns={severity_metric: "severity", dimension: "dim_level"}).copy()
        # Ensure cluster column is numeric for GEE
        df_fit = df_fit.sort_values(cluster_col)

        # Build design — only keep complete persona x severity cells
        cc = df_fit.groupby(["dim_level", "severity"]).size().unstack(fill_value=0)
        complete = cc.index[cc.min(axis=1) > 0]
        if len(complete) < 2:
            return {"skipped": True, "reason": "insufficient complete cells"}
        df_fit = df_fit[df_fit["dim_level"].isin(complete)]

        # --- Pre-flight: detect cell sparsity / separation ---
        _MIN_CELL = 5
        cell_stats = df_fit.groupby(["dim_level", "severity"])["Y2"].agg(["size", "mean", "std"])
        sparse_cells = cell_stats[cell_stats["size"] < _MIN_CELL]
        zero_var_cells = cell_stats[cell_stats["std"] == 0]

        preflight_issues = []
        if len(sparse_cells) > 0:
            sparse_desc = {
                f"{dl}×{sv}": int(row["size"])
                for (dl, sv), row in sparse_cells.iterrows()
            }
            preflight_issues.append({
                "type": "sparse_cells",
                "detail": f"Cells with n < {_MIN_CELL}: {sparse_desc}",
            })
            logger.info(
                "GEE interaction %s/%s/%s pre-flight: sparse cells %s",
                dimension, severity_metric, model, sparse_desc,
            )
        if len(zero_var_cells) > 0:
            zv_desc = {
                f"{dl}×{sv}": f"n={int(row['size'])}, Y2_mean={row['mean']:.0f}"
                for (dl, sv), row in zero_var_cells.iterrows()
            }
            preflight_issues.append({
                "type": "zero_variance_cells",
                "detail": f"Cells with no Y2 variation (complete separation): {zv_desc}",
            })
            logger.info(
                "GEE interaction %s/%s/%s pre-flight: zero-variance cells %s "
                "(complete separation likely)",
                dimension, severity_metric, model, zv_desc,
            )

        formula = "Y2 ~ C(dim_level) + C(severity) + C(dim_level):C(severity)"

        convergence_warning = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                gee_model = smf.gee(
                    formula,
                    groups=cluster_col,
                    data=df_fit,
                    family=Binomial(),
                    cov_struct=Exchangeable(),
                )
                fit = gee_model.fit(maxiter=100)
            except Exception as e:
                return {"skipped": True, "reason": f"GEE fit failed: {e}"}
            convergence_warning = any("converge" in str(w.message).lower() for w in caught)

        # Extract interaction terms
        params = fit.params
        pvalues = fit.pvalues
        bse = fit.bse
        conf_int = fit.conf_int()

        interaction_terms = [p for p in params.index if ":" in p and "dim_level" in p and "severity" in p]

        # Detect separation: any interaction SE being NaN implies singular
        # covariance — Wald joint test requires full-rank covariance block
        separation_detected = False
        if interaction_terms:
            int_ses = [float(bse[t]) for t in interaction_terms]
            n_nan = sum(np.isnan(s) for s in int_ses)
            if n_nan > 0:
                separation_detected = True
                logger.warning(
                    "GEE interaction %s/%s: separation detected — "
                    "%d of %d interaction SEs are NaN (coefficients: %s)",
                    dimension, model, n_nan, len(interaction_terms),
                    {t: f"{float(params[t]):.2f}" for t in interaction_terms},
                )
                logger.info(
                    "GEE interaction %s/%s: no fallback attempted — "
                    "the interaction term is the quantity of interest so "
                    "main-effects-only GEE would not answer the same question; "
                    "Bayesian robustness models handle separation via "
                    "regularising priors (see bayes_robustness.py)",
                    dimension, model,
                )

        # Joint test for interaction terms
        if interaction_terms and not separation_detected:
            try:
                joint_test = fit.wald_test_terms(scalar=True)
                # Find the interaction row in the joint test
                interaction_p: Any = NE.missing_data(context="GEE interaction p")
                for term_name in joint_test.table.index:
                    if ":" in str(term_name):
                        interaction_p = float(joint_test.table.loc[term_name, "P>chi2"])
                        break
                if isinstance(interaction_p, NEValue):
                    # Fallback: use maximum individual p-value
                    interaction_p = float(max(pvalues[t] for t in interaction_terms))
            except Exception:
                interaction_p = float(max(pvalues[t] for t in interaction_terms)) if interaction_terms else NE.missing_data(context="GEE interaction p")
        elif separation_detected:
            interaction_p = NE.singular_matrix(
                detail="quasi-complete separation in persona x severity cells",
                context=f"GEE interaction {dimension}",
            )
        else:
            interaction_p = NE.missing_data(context="no interaction terms")

        # Build coefficients dict
        coefficients = {}
        ne_ci = NE.missing_data(context="GEE coef CI")
        for term in params.index:
            coefficients[term] = {
                "estimate": float(params[term]),
                "se": float(bse[term]),
                "p_value": float(pvalues[term]),
                "ci_lower": float(conf_int.loc[term, 0]) if term in conf_int.index else ne_ci,
                "ci_upper": float(conf_int.loc[term, 1]) if term in conf_int.index else ne_ci,
                "odds_ratio": float(np.exp(params[term])),
            }

        # Diagnostics summary
        fallback_actions = []
        if separation_detected:
            fallback_actions.append(
                "Wald interaction test not identified due to singular "
                "covariance (quasi-complete separation); p-value omitted"
            )
        diag_status = "WARN" if (preflight_issues or separation_detected) else "OK"

        return {
            "interaction_p": interaction_p,
            "n_interaction_terms": len(interaction_terms),
            "coefficients": coefficients,
            "n_obs": int(fit.nobs),
            "n_clusters": int(df_fit[cluster_col].nunique()),
            "method": "GEE (Binomial, exchangeable, Wald test)",
            "convergence_warning": convergence_warning,
            "separation_detected": separation_detected,
            "diagnostics": {
                "status": diag_status,
                "preflight_issues": preflight_issues,
                "fallback_actions": fallback_actions,
                "cell_stats": {
                    f"{dl}×{sv}": {"n": int(row["size"]), "Y2_mean": round(float(row["mean"]), 4)}
                    for (dl, sv), row in cell_stats.iterrows()
                },
            },
        }

    except Exception as e:
        logger.warning("Logistic GEE interaction failed for %s/%s: %s", dimension, model, e)
        return {"skipped": True, "reason": str(e)}


# ---------------------------------------------------------------------------
# Section 4: Three-Way Analysis
# ---------------------------------------------------------------------------

def three_way_anova(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    persona_col: str = "persona_label",
) -> Dict[str, Any]:
    """Three-way ANOVA: tier ~ model * persona * severity.

    **Status: exploratory / sanity-check only.**
    """
    from src.regression_diagnostics import diagnose_fit_failure

    model_label = f"ThreeWayANOVA({severity_metric})"
    df = decisions_df[decisions_df[severity_metric].isin(SEVERITY_LEVELS)].copy()

    # Rename columns for consistent handling
    df = df.rename(columns={severity_metric: "severity", persona_col: "persona"})

    # Run comprehensive pre-flight checks
    preflight = anova_preflight(
        df,
        outcome_col="tier",
        factor_cols=["model", "persona", "severity"],
        min_n=50,
        min_cell_n=1,
        context=model_label,
    )
    if not preflight.ok:
        return {
            "error": preflight.error,
            "diagnosis": preflight.diagnosis,
            "status": "exploratory",
            "preflight": {
                "n_obs": preflight.n_obs,
                "cell_counts": preflight.cell_counts,
            },
        }

    # Log any preflight warnings
    for warn_msg in preflight.warnings:
        logger.debug("%s preflight warning: %s", model_label, warn_msg)

    # Additional check: need multiple models
    if df["model"].nunique() < 2:
        return {"error": "need multiple models for three-way ANOVA", "status": "exploratory"}

    # Keep only persona×severity×model cells that are fully populated.
    cell_counts = df.groupby(["persona", "severity", "model"]).size().unstack(fill_value=0)
    complete_personas = cell_counts.groupby("persona").apply(
        lambda g: (g.min(axis=1) > 0).all()
    )
    complete_personas = complete_personas[complete_personas].index
    if len(complete_personas) < 2:
        return {"error": "insufficient complete persona×severity×model cells", "status": "exploratory"}
    df = df[df["persona"].isin(complete_personas)]

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(
                "tier ~ C(model) * C(persona) * C(severity)", data=df
            ).fit()
            table = anova_lm(fit, typ=2)

        ss_resid = max(0.0, float(table.loc["Residual", "sum_sq"])) if "Residual" in table.index else 0
        result: Dict[str, Any] = {"status": "exploratory"}
        label_map = {
            "model": "C(model)",
            "persona": "C(persona)",
            "severity": "C(severity)",
            "model_x_persona": "C(model):C(persona)",
            "model_x_severity": "C(model):C(severity)",
            "persona_x_severity": "C(persona):C(severity)",
            "model_x_persona_x_severity": "C(model):C(persona):C(severity)",
            "residual": "Residual",
        }
        for key, row_label in label_map.items():
            if row_label in table.index:
                ss = max(0.0, float(table.loc[row_label, "sum_sq"]))
                df_val = float(table.loc[row_label, "df"])
                f_val_raw = table.loc[row_label, "F"] if "F" in table.columns else None
                p_val_raw = table.loc[row_label, "PR(>F)"] if "PR(>F)" in table.columns else None
                f_val = float(f_val_raw) if f_val_raw is not None and np.isfinite(float(f_val_raw)) else NE.missing_data(field="F", context=key)
                p_val = float(p_val_raw) if p_val_raw is not None and np.isfinite(float(p_val_raw)) else NE.missing_data(field="p", context=key)
                if is_estimable(f_val) and f_val < 0:
                    f_val = 0.0
                    p_val = 1.0
                eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                result[key] = {
                    "F": f_val if is_estimable(f_val) else NE.numerical(context=f"3-way ANOVA F for {key}"),
                    "p_value": p_val if is_estimable(p_val) else NE.numerical(context=f"3-way ANOVA p for {key}"),
                    "eta_sq_partial": eta_sq,
                }
            else:
                ne = NE.missing_data(field=row_label, context=f"3-way ANOVA {key}")
                result[key] = {"F": ne, "p_value": ne, "eta_sq_partial": 0.0}

        return result

    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label)
        logger.warning("Three-way ANOVA failed (%s): %s", fit_diag["type"], e)
        return {"error": str(e), "diagnosis": fit_diag["type"], "status": "exploratory"}


def model_specific_amplification(
    decisions_df: pd.DataFrame,
    models: List[str],
    persona_col: str = "persona_label",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Amplification ratio per model per severity metric."""
    result: Dict[str, Any] = {m: {} for m in models}

    tasks = [(model, metric) for model in models for metric in SEVERITY_METRICS]

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                amplification_ratio,
                decisions_df, metric, model,
                persona_col=persona_col, n_boot=n_boot, seed=seed,
            ): (model, metric)
            for model, metric in tasks
        }
        for future in as_completed(futures):
            model, metric = futures[future]
            result[model][f"{metric}_amplification"] = future.result()

    return result


def worst_case_analysis(
    decisions_df: pd.DataFrame,
    models: List[str],
    persona_col: str = "persona_label",
) -> Dict[str, Any]:
    """Find the most discriminatory model × severity combination."""
    best: Dict[str, Any] = {
        "model": "", "severity_level": "", "severity_metric": "",
        "bias_magnitude": 0.0, "most_disadvantaged": "", "most_advantaged": "",
        "effect_size_d": 0.0,
    }

    for model in models:
        for metric in SEVERITY_METRICS:
            df = decisions_df[
                (decisions_df["model"] == model)
                & (decisions_df[metric].isin(SEVERITY_LEVELS))
            ]
            if persona_col not in df.columns:
                continue

            for level in SEVERITY_LEVELS:
                sub = df[df[metric] == level]
                if len(sub) < 10:
                    continue
                pm = sub.groupby(persona_col)["tier"].mean()
                if len(pm) < 2:
                    continue
                bias = float(pm.max() - pm.min())
                if bias > best["bias_magnitude"]:
                    worst_persona = str(pm.idxmin())
                    best_persona = str(pm.idxmax())
                    # Cohen's d between best and worst
                    g_best = sub.loc[sub[persona_col] == best_persona, "tier"].values
                    g_worst = sub.loc[sub[persona_col] == worst_persona, "tier"].values
                    d = abs(_cohens_d(g_best, g_worst))
                    best = {
                        "model": model,
                        "severity_level": level,
                        "severity_metric": metric,
                        "bias_magnitude": bias,
                        "most_disadvantaged": worst_persona,
                        "most_advantaged": best_persona,
                        "effect_size_d": d,
                        "effect_size_interpretation": interpret_cohens_d(d),
                    }

    return best


# ---------------------------------------------------------------------------
# Section 5: Process Fairness × Severity
# ---------------------------------------------------------------------------

def process_severity_analysis(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Process (info request) discrimination by severity level."""
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ].copy()
    if "confidence" not in df.columns or persona_col not in df.columns or len(df) < 30:
        return {}

    df["need_info"] = (df["confidence"] == "need_more_info").astype(int)

    # Rates by severity
    rates_by_severity: Dict[str, Any] = {}
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]
        if len(sub) == 0:
            continue
        overall_rate = float(sub["need_info"].mean())
        persona_rates = sub.groupby(persona_col)["need_info"].mean()
        disparity = float(persona_rates.max() - persona_rates.min()) if len(persona_rates) >= 2 else 0.0
        rates_by_severity[level] = {
            "overall_rate": overall_rate,
            "persona_range": [float(persona_rates.min()), float(persona_rates.max())] if len(persona_rates) >= 2 else [],
            "disparity": disparity,
            "n": len(sub),
        }

    result: Dict[str, Any] = {"rates_by_severity": rates_by_severity}

    # Logistic regression: need_info ~ persona + severity + persona:severity
    df_reg = df.rename(columns={severity_metric: "severity", persona_col: "persona"})
    if df_reg["severity"].nunique() >= 2 and df_reg["persona"].nunique() >= 2:
        # Keep only personas present at every severity level for full-rank interaction
        _cc = df_reg.groupby(["persona", "severity"]).size().unstack(fill_value=0)
        _complete = _cc.index[_cc.min(axis=1) > 0]
        if len(_complete) >= 2:
            df_reg = df_reg[df_reg["persona"].isin(_complete)]
        try:
            import statsmodels.formula.api as smf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.logit(
                    "need_info ~ C(persona) + C(severity) + C(persona):C(severity)",
                    data=df_reg,
                ).fit(disp=0, maxiter=100)
            result["logistic"] = {
                "pseudo_r2": float(fit.prsquared),
                "converged": fit.mle_retvals.get("converged", False) if hasattr(fit, "mle_retvals") else True,
                "severity_p_values": {
                    name: float(fit.pvalues[name])
                    for name in fit.pvalues.index
                    if "severity" in name.lower()
                },
            }
        except Exception as e:
            result["logistic"] = {"error": str(e)}
    else:
        result["logistic"] = {"error": "insufficient factor levels"}

    # Process amplification ratio (disparity high / disparity low)
    disp_low_raw = rates_by_severity.get("low", {}).get("disparity")
    disp_high_raw = rates_by_severity.get("high", {}).get("disparity")
    ne_disp = NE.missing_data(context="process disparity")
    disp_low = disp_low_raw if is_estimable(disp_low_raw) else ne_disp
    disp_high = disp_high_raw if is_estimable(disp_high_raw) else ne_disp
    if is_estimable(disp_low) and is_estimable(disp_high):
        result["process_amplification"] = {
            "point": _safe_div(disp_high, disp_low),
            "disparity_low": disp_low,
            "disparity_high": disp_high,
        }
    else:
        result["process_amplification"] = {
            "point": NE.missing_data(context="process amp ratio"),
            "disparity_low": disp_low,
            "disparity_high": disp_high,
        }

    return result


# ---------------------------------------------------------------------------
# Section 6: Consistency Check
# ---------------------------------------------------------------------------

def consistency_check(
    main_effects: Dict[str, Any],
    interactions: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare findings across monetary and issue severity metrics."""
    findings: list = []

    def _get_p(data: Dict, *keys: str) -> Any:
        for k in keys:
            data = data.get(k, {})
        return data if isinstance(data, (int, float, NEValue)) else NE.missing_data(context="consistency check p-value")

    def _is_sig(p_val: Any, threshold: float = 0.05) -> bool:
        """Check if p-value is significant, handling NEValue and non-numeric types."""
        if isinstance(p_val, NEValue):
            return False
        if not isinstance(p_val, (int, float)):
            return False
        try:
            return bool(np.isfinite(p_val) and p_val < threshold)
        except (TypeError, ValueError):
            return False

    # For each model present in both metrics
    monetary_me = main_effects.get("monetary_severity", {})
    issue_me = main_effects.get("issue_severity", {})
    all_models = set(monetary_me.keys()) | set(issue_me.keys())

    for model in sorted(all_models):
        m_anova_p = _get_p(monetary_me, model, "anova", "p_value")
        i_anova_p = _get_p(issue_me, model, "anova", "p_value")
        m_sig = _is_sig(m_anova_p)
        i_sig = _is_sig(i_anova_p)
        findings.append({
            "finding": f"Severity main effect ({model})",
            "monetary_p": m_anova_p,
            "issue_p": i_anova_p,
            "consistent": m_sig == i_sig,
            "robustness": "Robust" if (m_sig and i_sig) else ("Null" if (not m_sig and not i_sig) else "Metric-specific"),
        })

    # Interaction terms
    monetary_int = interactions.get("monetary_severity", {})
    issue_int = interactions.get("issue_severity", {})
    for model in sorted(all_models):
        m_p = _get_p(monetary_int, model, "two_way", "interaction", "p_value")
        i_p = _get_p(issue_int, model, "two_way", "interaction", "p_value")
        m_sig = _is_sig(m_p)
        i_sig = _is_sig(i_p)
        findings.append({
            "finding": f"Persona×Severity interaction ({model})",
            "monetary_p": m_p,
            "issue_p": i_p,
            "consistent": m_sig == i_sig,
            "robustness": "Robust" if (m_sig and i_sig) else ("Null" if (not m_sig and not i_sig) else "Metric-specific"),
        })

    # Amplification ratios
    for model in sorted(all_models):
        m_ratio = _get_p(monetary_int, model, "amplification", "point")
        i_ratio = _get_p(issue_int, model, "amplification", "point")
        findings.append({
            "finding": f"Amplification ratio ({model})",
            "monetary_value": m_ratio,
            "issue_value": i_ratio,
            "consistent": True,  # always report
            "robustness": "Reported",
        })

    return {"findings": findings}


# ---------------------------------------------------------------------------
# Section 7: Mitigation × Severity Comparison
# ---------------------------------------------------------------------------

def mitigation_severity_comparison(
    baseline_df: pd.DataFrame,
    neutralized_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
    n_boot: int = 1000,
    seed: int = 42,
    outcome_col: str = "tier",
) -> Dict[str, Any]:
    """Compare severity amplification before and after bias mitigation.

    Reuses existing analysis functions on both baseline and neutralized
    DataFrames to compute deltas and a summary verdict.

    Returns
    -------
    Dict with keys: baseline, neutralized, comparison.
    """
    # --- Baseline metrics (recompute locally for self-contained comparison) ---
    bias_base = bias_by_severity(baseline_df, severity_metric, model, persona_col,
                                 outcome_col=outcome_col)
    amp_base = amplification_ratio(
        baseline_df, severity_metric, model, persona_col, n_boot, seed,
        outcome_col=outcome_col,
    )
    tw_base = two_way_anova_persona_severity(baseline_df, severity_metric, model, persona_col,
                                             outcome_col=outcome_col)

    # --- Neutralized metrics ---
    bias_neut = bias_by_severity(neutralized_df, severity_metric, model, persona_col,
                                 outcome_col=outcome_col)
    amp_neut = amplification_ratio(
        neutralized_df, severity_metric, model, persona_col, n_boot, seed,
        outcome_col=outcome_col,
    )
    tw_neut = two_way_anova_persona_severity(neutralized_df, severity_metric, model, persona_col,
                                             outcome_col=outcome_col)
    se_neut = simple_effects_by_severity(neutralized_df, severity_metric, model, persona_col,
                                         outcome_col=outcome_col)

    # --- Compute deltas ---
    ne_missing = NE.missing_data(context="neutralization delta")
    base_ratio = amp_base.get("point", ne_missing)
    neut_ratio = amp_neut.get("point", ne_missing)

    if is_estimable(base_ratio) and is_estimable(neut_ratio) and base_ratio > 0:
        amp_change: Any = neut_ratio - base_ratio
        amp_pct: Any = amp_change / base_ratio * 100
    else:
        amp_change = NE.missing_data(context="amp change")
        amp_pct = NE.missing_data(context="amp pct change")

    # Per-level bias SD changes
    bias_sd_change: Dict[str, Any] = {}
    for level in SEVERITY_LEVELS:
        b_sd = bias_base.get(level, {}).get("sd_across_personas", ne_missing)
        n_sd = bias_neut.get(level, {}).get("sd_across_personas", ne_missing)
        if is_estimable(b_sd) and is_estimable(n_sd):
            bias_sd_change[level] = n_sd - b_sd
        else:
            bias_sd_change[level] = NE.missing_data(context=f"bias SD change {level}")

    # Interaction p-values
    base_int_p = tw_base.get("interaction", {}).get("p_value", ne_missing)
    neut_int_p = tw_neut.get("interaction", {}).get("p_value", ne_missing)
    interaction_weakened = (
        is_estimable(base_int_p) and is_estimable(neut_int_p)
        and neut_int_p > base_int_p
    )

    # Verdict
    if not is_estimable(base_ratio) or not is_estimable(neut_ratio):
        verdict = "insufficient_data"
    else:
        base_ci_lo = amp_base.get("ci_lower", ne_missing)
        base_ci_hi = amp_base.get("ci_upper", ne_missing)
        neut_ci_lo = amp_neut.get("ci_lower", ne_missing)
        neut_ci_hi = amp_neut.get("ci_upper", ne_missing)
        # "eliminated" if neutralized CI includes 1.0 but baseline CI didn't
        base_includes_1 = (is_estimable(base_ci_lo) and is_estimable(base_ci_hi)
                           and base_ci_lo <= 1.0 <= base_ci_hi)
        neut_includes_1 = (is_estimable(neut_ci_lo) and is_estimable(neut_ci_hi)
                           and neut_ci_lo <= 1.0 <= neut_ci_hi)
        if neut_includes_1 and not base_includes_1:
            verdict = "eliminated"
        elif neut_ratio < base_ratio - 0.01:
            verdict = "reduced"
        elif neut_ratio > base_ratio + 0.01:
            verdict = "increased"
        else:
            verdict = "unchanged"

    return {
        "baseline": {
            "amplification": amp_base,
            "bias": bias_base,
            "interaction": tw_base.get("interaction", {}),
        },
        "neutralized": {
            "amplification": amp_neut,
            "bias": bias_neut,
            "interaction": tw_neut.get("interaction", {}),
            "simple_effects": se_neut,
        },
        "comparison": {
            "amplification_change": amp_change,
            "amplification_pct_change": amp_pct,
            "bias_sd_change_by_level": bias_sd_change,
            "interaction_p_baseline": base_int_p,
            "interaction_p_neutralized": neut_int_p,
            "interaction_weakened": interaction_weakened,
            "verdict": verdict,
        },
    }


# ---------------------------------------------------------------------------
# Monetary missingness diagnostic
# ---------------------------------------------------------------------------


def monetary_missingness_diagnostic(
    baseline_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    models: List[str],
    persona_col: str = "persona_label",
    personas_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compare complaints with vs without extractable dollar amounts.

    Tests whether the ~66% of complaints with ``monetary_severity == "unknown"``
    differ systematically from those with known amounts.  If the missingness is
    non-random (MAR), the monetary severity amplification findings may not
    generalise to the full complaint set.

    Parameters
    ----------
    baseline_df : Baseline decisions with columns [case_id, model, tier,
        confidence, monetary_severity, persona_label].
    complaints_df : Complaint metadata with columns [case_id, monetary_severity,
        issue_category, product_category, word_count].
    models : Model identifiers.
    persona_col : Column name for persona labels.
    personas_df : Persona estimates with demographic columns (gender, esl, etc.).

    Returns
    -------
    Dict with keys: n_present, n_missing, pct_missing,
    complaint_characteristics, fairness_comparison, overall_conclusion,
    distributions.
    """
    present_mask = complaints_df["monetary_severity"].isin(SEVERITY_LEVELS)
    n_present = int(present_mask.sum())
    n_missing = int((~present_mask).sum())
    n_total = n_present + n_missing

    if n_present < 10 or n_missing < 10:
        return {"skipped": True, "reason": "too few complaints in one group"}

    pct_missing = n_missing / n_total if n_total > 0 else 0.0

    df_present = complaints_df[present_mask]
    df_missing = complaints_df[~present_mask]

    present_ids = set(df_present["case_id"])
    missing_ids = set(df_missing["case_id"])

    # ── Complaint characteristic tests ──
    characteristics: Dict[str, Any] = {}
    significant_tests: List[str] = []

    # 1. Issue category distribution
    try:
        ct_issue = pd.crosstab(
            complaints_df["monetary_severity"].apply(lambda x: "present" if x in SEVERITY_LEVELS else "missing"),
            complaints_df["issue_category"],
        )
        if ct_issue.shape[0] >= 2 and ct_issue.shape[1] >= 2:
            result = chi_squared_test(ct_issue)
            characteristics["issue_category"] = result
            if result["p_value"] < 0.05:
                significant_tests.append("issue_category")
    except Exception:
        characteristics["issue_category"] = {"error": True}

    # 2. Product category distribution
    try:
        ct_prod = pd.crosstab(
            complaints_df["monetary_severity"].apply(lambda x: "present" if x in SEVERITY_LEVELS else "missing"),
            complaints_df["product_category"],
        )
        if ct_prod.shape[0] >= 2 and ct_prod.shape[1] >= 2:
            result = chi_squared_test(ct_prod)
            characteristics["product_category"] = result
            if result["p_value"] < 0.05:
                significant_tests.append("product_category")
    except Exception:
        characteristics["product_category"] = {"error": True}

    # 3. Word count
    if "word_count" in complaints_df.columns:
        wc_present = df_present["word_count"].dropna()
        wc_missing = df_missing["word_count"].dropna()
        if len(wc_present) >= 5 and len(wc_missing) >= 5:
            # mannwhitneyu's normal approximation suffers catastrophic
            # cancellation when data has massive ties / near-identical values.
            # Guard against the resulting RuntimeWarning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning,
                    message=".*Precision loss.*catastrophic cancellation.*",
                )
                u_stat, u_p = sp_stats.mannwhitneyu(wc_present, wc_missing, alternative="two-sided")
            n1, n2 = len(wc_present), len(wc_missing)
            rank_biserial = 1 - 2 * u_stat / (n1 * n2) if n1 * n2 > 0 else 0.0
            characteristics["word_count"] = {
                "mean_present": float(wc_present.mean()),
                "mean_missing": float(wc_missing.mean()),
                "U_statistic": float(u_stat),
                "p_value": float(u_p),
                "rank_biserial": float(rank_biserial),
            }
            if u_p < 0.05:
                significant_tests.append("word_count")

    # 4. Mean remedy tier (using first model's baseline decisions)
    # Welch's t-test rather than Mann-Whitney U because tier is a coarse
    # ordinal variable (0/1/2) producing massive ties.  The normal
    # approximation inside mannwhitneyu suffers catastrophic cancellation
    # when computing moments of the U distribution on near-constant data.
    try:
        bl_present = baseline_df[baseline_df["case_id"].isin(present_ids)]
        bl_missing = baseline_df[baseline_df["case_id"].isin(missing_ids)]
        # Use first model to avoid double-counting
        first_model = models[0] if models else None
        if first_model:
            t_pres = bl_present.loc[bl_present["model"] == first_model, "tier"].dropna()
            t_miss = bl_missing.loc[bl_missing["model"] == first_model, "tier"].dropna()
            if len(t_pres) >= 5 and len(t_miss) >= 5:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=RuntimeWarning,
                        message=".*Precision loss.*catastrophic cancellation.*",
                    )
                    t_stat, t_p = sp_stats.ttest_ind(t_pres, t_miss, equal_var=False)
                # Cohen's d as effect size
                n1, n2 = len(t_pres), len(t_miss)
                pooled_std = np.sqrt(
                    ((n1 - 1) * t_pres.var(ddof=1) + (n2 - 1) * t_miss.var(ddof=1))
                    / (n1 + n2 - 2)
                )
                cohens_d = float((t_pres.mean() - t_miss.mean()) / pooled_std) if pooled_std > 0 else 0.0
                characteristics["mean_tier"] = {
                    "mean_present": float(t_pres.mean()),
                    "mean_missing": float(t_miss.mean()),
                    "t_statistic": float(t_stat),
                    "p_value": float(t_p),
                    "cohens_d": float(cohens_d),
                }
                if t_p < 0.05:
                    significant_tests.append("mean_tier")
    except Exception:
        characteristics["mean_tier"] = {"error": True}

    # 5. Tier distribution
    try:
        if first_model:
            bl_fm = baseline_df[baseline_df["model"] == first_model].copy()
            bl_fm["group"] = bl_fm["case_id"].apply(
                lambda x: "present" if x in present_ids else "missing"
            )
            ct_tier = pd.crosstab(bl_fm["group"], bl_fm["tier"])
            if ct_tier.shape[0] >= 2 and ct_tier.shape[1] >= 2:
                result = chi_squared_test(ct_tier)
                characteristics["tier_distribution"] = result
                if result["p_value"] < 0.05:
                    significant_tests.append("tier_distribution")
    except Exception:
        characteristics["tier_distribution"] = {"error": True}

    n_sig = len(significant_tests)
    missingness_type = "MCAR" if n_sig == 0 else "MAR"

    characteristics["missingness_type"] = missingness_type
    characteristics["n_significant_tests"] = n_sig
    characteristics["significant_tests"] = significant_tests

    # ── Fairness comparison per model ──
    fairness: Dict[str, Any] = {}
    for model in models:
        bl_m = baseline_df[baseline_df["model"] == model]
        bl_pres = bl_m[bl_m["case_id"].isin(present_ids)]
        bl_miss = bl_m[bl_m["case_id"].isin(missing_ids)]

        def _bias_stats(df: pd.DataFrame) -> Dict[str, Any]:
            if persona_col not in df.columns or len(df) == 0:
                ne_empty = NE.insufficient_n(n=0, min_required=1, context="bias stats")
                return {"n": 0, "bias_sd": ne_empty, "bias_range": ne_empty, "n_significant_pairs": 0}
            pm = df.groupby(persona_col)["tier"].mean()
            bias_sd = float(pm.std()) if len(pm) > 1 else 0.0
            bias_range = float(pm.max() - pm.min()) if len(pm) > 1 else 0.0

            # Pairwise t-tests with FDR
            personas = sorted(pm.index)
            p_values = []
            for i in range(len(personas)):
                for j in range(i + 1, len(personas)):
                    g1 = df.loc[df[persona_col] == personas[i], "tier"]
                    g2 = df.loc[df[persona_col] == personas[j], "tier"]
                    if len(g1) >= 2 and len(g2) >= 2 and g1.std() + g2.std() > 0:
                        pooled_std = np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
                        # Skip when variance is too low for a meaningful test
                        if pooled_std < 1e-8:
                            continue
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", category=RuntimeWarning,
                                message=".*Precision loss.*catastrophic cancellation.*",
                            )
                            _, p = sp_stats.ttest_ind(g1, g2, equal_var=False)
                        if np.isfinite(p):
                            p_values.append(p)
            n_sig_pairs = 0
            if p_values:
                corrected = fdr_correction(p_values)
                n_sig_pairs = int(sum(1 for p in corrected if p < 0.05))

            return {
                "n": int(len(df)),
                "bias_sd": bias_sd,
                "bias_range": bias_range,
                "n_significant_pairs": n_sig_pairs,
            }

        stats_pres = _bias_stats(bl_pres)
        stats_miss = _bias_stats(bl_miss)

        sd_pres = stats_pres["bias_sd"]
        sd_miss = stats_miss["bias_sd"]
        if is_estimable(sd_pres) and sd_pres > 0 and is_estimable(sd_miss):
            ratio: Any = sd_miss / sd_pres
        else:
            ratio = NE.zero_denominator("baseline SD is zero or non-estimable")

        # Conclusion
        if is_estimable(ratio):
            if ratio > 1.2:
                conclusion = (
                    f"Bias is higher in the missing-amount subset (ratio={ratio:.2f}). "
                    "The analyzed monetary severity subset may understate overall bias."
                )
            elif ratio < 0.8:
                conclusion = (
                    f"Bias is lower in the missing-amount subset (ratio={ratio:.2f}). "
                    "The analyzed monetary severity subset may overstate overall bias."
                )
            else:
                conclusion = (
                    f"Bias is similar in both subsets (ratio={ratio:.2f}). "
                    "Missingness does not substantially affect fairness conclusions."
                )
        else:
            conclusion = "Unable to compare bias — insufficient data in one subset."

        fairness[model] = {
            "present": stats_pres,
            "missing": stats_miss,
            "bias_sd_ratio": ratio,
            "conclusion": conclusion,
        }

    # ── Overall conclusion ──
    if missingness_type == "MCAR":
        overall = (
            f"Missingness appears random (MCAR): none of the 5 characteristic tests "
            f"were significant. Monetary severity findings likely generalise to the full set."
        )
    else:
        sig_str = ", ".join(significant_tests)
        overall = (
            f"Missingness is systematic (MAR): {n_sig}/5 tests significant ({sig_str}). "
            f"Complaints with extractable dollar amounts differ from those without."
        )
        # Append fairness summary
        bias_notes = []
        for model in models:
            r = fairness.get(model, {}).get("bias_sd_ratio", NE.missing_data(context="bias SD ratio"))
            if is_estimable(r):
                if r > 1.2:
                    bias_notes.append(f"{model}: missingness may deflate apparent bias (ratio={r:.2f})")
                elif r < 0.8:
                    bias_notes.append(f"{model}: missingness may inflate apparent bias (ratio={r:.2f})")
                else:
                    bias_notes.append(f"{model}: bias similar in both subsets (ratio={r:.2f})")
        if bias_notes:
            overall += " " + "; ".join(bias_notes) + "."

    # ── Distribution profiles by missingness group ──
    distributions: Dict[str, Dict[str, Dict[str, int]]] = {}

    def _value_counts(series: pd.Series) -> Dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts().items()}

    # Variables from complaints_df (one row per complaint)
    for col in ["product_category", "issue_category", "issue_severity"]:
        if col in complaints_df.columns:
            distributions[col] = {
                "present": _value_counts(df_present[col].dropna()),
                "missing": _value_counts(df_missing[col].dropna()),
            }

    # Tier distribution from baseline_df (use first model to avoid duplication)
    first_model = models[0] if models else None
    if first_model:
        bl_fm = baseline_df[baseline_df["model"] == first_model]
        bl_pres = bl_fm[bl_fm["case_id"].isin(present_ids)]
        bl_miss = bl_fm[bl_fm["case_id"].isin(missing_ids)]
        distributions["tier"] = {
            "present": _value_counts(bl_pres["tier"].dropna()),
            "missing": _value_counts(bl_miss["tier"].dropna()),
        }

        # Demographic dimensions from baseline_df (causal dims)
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            if dim in bl_fm.columns:
                distributions[dim] = {
                    "present": _value_counts(bl_pres[dim].dropna()),
                    "missing": _value_counts(bl_miss[dim].dropna()),
                }

        # Gender and ESL from personas_df (not on baseline_df)
        if personas_df is not None:
            p_fm = personas_df[personas_df["model"] == first_model]
            p_pres = p_fm[p_fm["case_id"].isin(present_ids)]
            p_miss = p_fm[p_fm["case_id"].isin(missing_ids)]
            for dim in ["gender", "esl"]:
                if dim in p_fm.columns:
                    distributions[dim] = {
                        "present": _value_counts(p_pres[dim].dropna()),
                        "missing": _value_counts(p_miss[dim].dropna()),
                    }

    return {
        "n_present": n_present,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
        "complaint_characteristics": characteristics,
        "fairness_comparison": fairness,
        "overall_conclusion": overall,
        "distributions": distributions,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_severity_analysis(
    baseline_df: pd.DataFrame,
    persona_decisions_df: Optional[pd.DataFrame],
    personas_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    models: List[str],
    n_boot: int = 1000,
    seed: int = 42,
    neutralized_df: Optional[pd.DataFrame] = None,
    use_cache: bool = True,
    cache_dir: str = "./cache",
) -> Dict[str, Any]:
    """Run the full severity amplification analysis.

    Parameters
    ----------
    baseline_df : Baseline remedy decisions with severity columns merged in.
        Columns: [case_id, model, tier, confidence, monetary_severity, issue_severity, persona_label]
    persona_decisions_df : Persona-rewritten decisions with severity columns merged.
        Columns: [case_id, model, variant, tier, confidence, monetary_severity, issue_severity]
        May be None if step 4/5 were skipped.
    personas_df : Persona estimates.
        Columns: [case_id, model, persona_label, ...]
    complaints_df : Severity + product/issue metadata.
        Columns: [case_id, monetary_severity, issue_severity, product_category, issue_category]
    models : List of model identifiers.
    n_boot : Number of bootstrap iterations.
    seed : Random seed.
    neutralized_df : Neutralized remedy decisions with severity columns merged.
        Columns: [case_id, model, tier, confidence, monetary_severity, issue_severity, persona_label]
        If provided, a mitigation × severity comparison is computed.
    use_cache : If True (default), check for cached results before computing.
    cache_dir : Directory for analysis cache files.

    Returns
    -------
    Dict with keys: descriptives, coverage, main_effects, interactions,
    three_way, model_comparison, process, consistency, mitigation_severity,
    missingness_diagnostic.
    """
    # Check cache first
    if use_cache:
        from src.analysis_cache import get_analysis_cache
        cache = get_analysis_cache(cache_dir)
        dataframes = {
            "baseline_df": baseline_df,
            "persona_decisions_df": persona_decisions_df,
            "personas_df": personas_df,
            "complaints_df": complaints_df,
            "neutralized_df": neutralized_df,
        }
        params = {"models": models, "n_boot": n_boot, "seed": seed}
        cached_result = cache.get("severity_analysis", dataframes, params)
        if cached_result is not None:
            return cached_result

    logger.info("Running severity amplification analysis...")

    # Total sections for top-level progress bar
    has_neutralized = neutralized_df is not None and len(neutralized_df) > 0
    n_sections = 7 + (2 if has_neutralized else 0)

    # Suppress known warnings for the entire analysis.  warnings.catch_warnings()
    # is NOT thread-safe, so per-function guards inside ThreadPoolExecutor workers
    # are unreliable.  Setting the filters here in the main thread ensures they
    # are in effect for all child threads (they share the global filter list).
    progress = tqdm(total=n_sections, desc="Severity analysis", delay=0.5)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning,
            message=".*Precision loss.*catastrophic cancellation.*",
        )
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
        # statsmodels ValueWarning for rank-deficient covariance of constraints
        try:
            from statsmodels.tools.sm_exceptions import ValueWarning as _SmValueWarning
            warnings.filterwarnings("ignore", category=_SmValueWarning)
        except ImportError:
            pass
        warnings.filterwarnings(
            "ignore",
            message=".*covariance of constraints does not have full rank.*",
        )

        try:
            result = _run_severity_analysis_inner(
                baseline_df, persona_decisions_df, personas_df, complaints_df,
                models, n_boot, seed, neutralized_df, has_neutralized, progress,
            )
            # Cache the result
            if use_cache:
                cache.put("severity_analysis", dataframes, params, result)
            return result
        finally:
            progress.close()


def _run_severity_analysis_inner(
    baseline_df, persona_decisions_df, personas_df, complaints_df,
    models, n_boot, seed, neutralized_df, has_neutralized, progress,
):
    """Inner implementation of run_severity_analysis (called inside warning guard)."""
    # 1. Descriptives
    progress.set_postfix_str("descriptives")
    descriptives = severity_descriptives(complaints_df)
    coverage = severity_coverage(
        complaints_df[["case_id", "monetary_severity", "issue_severity"]],
        personas_df,
    )
    descriptives["coverage"] = coverage

    # 1b. Monetary missingness diagnostic
    progress.set_postfix_str("missingness diagnostic")
    missingness = monetary_missingness_diagnostic(
        baseline_df, complaints_df, models, personas_df=personas_df,
    )
    progress.update(1)

    # 2. Main effects (uses baseline decisions)
    progress.set_postfix_str("main effects")
    main_effects = severity_main_effects_all(baseline_df, models)
    progress.update(1)

    # 3. Interaction analysis (uses baseline merged with personas)
    # Parallelise across metric × model combinations
    progress.set_postfix_str("interactions (tier)")

    def _compute_interaction(metric: str, model: str, outcome: str = "tier") -> Dict[str, Any]:
        model_int: Dict[str, Any] = {}
        model_int["bias"] = bias_by_severity(
            baseline_df, metric, model, persona_col="persona_label",
            outcome_col=outcome,
        )
        model_int["two_way"] = two_way_anova_persona_severity(
            baseline_df, metric, model, persona_col="persona_label",
            outcome_col=outcome,
        )
        model_int["simple_effects"] = simple_effects_by_severity(
            baseline_df, metric, model, persona_col="persona_label",
            outcome_col=outcome,
        )
        model_int["amplification"] = amplification_ratio(
            baseline_df, metric, model,
            persona_col="persona_label", n_boot=n_boot, seed=seed,
            outcome_col=outcome,
        )
        return model_int

    interactions: Dict[str, Any] = {m: {} for m in SEVERITY_METRICS}
    tasks_s3 = [(metric, model) for metric in SEVERITY_METRICS for model in models]

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_compute_interaction, metric, model, "tier"): (metric, model)
            for metric, model in tasks_s3
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="  S3 interactions", leave=False, delay=1,
        ):
            metric, model = futures[future]
            interactions[metric][model] = future.result()
    progress.update(1)

    # 3-t2. Tier 2 rate interaction analysis
    progress.set_postfix_str("interactions (tier2)")
    baseline_df = baseline_df.copy()
    baseline_df["tier2"] = (baseline_df["tier"] >= 2).astype(int)
    interactions_tier2: Dict[str, Any] = {m: {} for m in SEVERITY_METRICS}

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_compute_interaction, metric, model, "tier2"): (metric, model)
            for metric, model in tasks_s3
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="  S3-t2 interactions", leave=False, delay=1,
        ):
            metric, model = futures[future]
            interactions_tier2[metric][model] = future.result()
    progress.update(1)

    # 3b. Dimension-specific severity interaction (one-way)
    progress.set_postfix_str("dimension severity")
    _causal_dims = [d for d in ["age", "geographic", "education", "socioeconomic"]
                    if d in baseline_df.columns]
    dim_severity = dimension_severity_all(
        baseline_df, models, _causal_dims,
        n_boot=n_boot, seed=seed,
    )
    progress.update(1)

    # 4. Three-way analysis
    progress.set_postfix_str("three-way")
    three_way: Dict[str, Any] = {}
    for metric in SEVERITY_METRICS:
        three_way[metric] = three_way_anova(
            baseline_df, metric, persona_col="persona_label",
        )
    model_amp = model_specific_amplification(
        baseline_df, models,
        persona_col="persona_label", n_boot=n_boot, seed=seed,
    )
    worst = worst_case_analysis(baseline_df, models, persona_col="persona_label")
    progress.update(1)

    # 5. Process fairness — parallelise across metric × model
    progress.set_postfix_str("process fairness")
    process: Dict[str, Any] = {m: {} for m in SEVERITY_METRICS}

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_severity_analysis,
                baseline_df, metric, model,
                persona_col="persona_label", n_boot=n_boot, seed=seed,
            ): (metric, model)
            for metric, model in tasks_s3
        }
        for future in as_completed(futures):
            metric, model = futures[future]
            process[metric][model] = future.result()
    progress.update(1)

    # 6. Consistency
    progress.set_postfix_str("consistency")
    consist = consistency_check(main_effects, interactions)

    # 7. Mitigation × severity (optional)
    mitigation_severity: Dict[str, Any] = {}
    if has_neutralized:
        progress.set_postfix_str("mitigation severity")

        def _safe_mitigation(metric: str, model: str, outcome: str = "tier") -> Tuple[str, str, Dict[str, Any]]:
            try:
                return metric, model, mitigation_severity_comparison(
                    baseline_df, neutralized_df, metric, model,
                    persona_col="persona_label", n_boot=n_boot, seed=seed,
                    outcome_col=outcome,
                )
            except Exception as exc:
                logger.warning("  Mitigation severity failed for %s/%s: %s", metric, model, exc)
                return metric, model, {"comparison": {"verdict": "insufficient_data"}}

        mitigation_severity = {m: {} for m in SEVERITY_METRICS}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_safe_mitigation, metric, model, "tier")
                for metric, model in tasks_s3
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="  S7 mitigation", leave=False, delay=1,
            ):
                metric, model, res = future.result()
                mitigation_severity[metric][model] = res
        progress.update(1)

    # 7-t2. Mitigation × severity for Tier 2 rate (optional)
    mitigation_severity_tier2: Dict[str, Any] = {}
    if has_neutralized:
        progress.set_postfix_str("mitigation severity (tier2)")
        neutralized_df = neutralized_df.copy()
        neutralized_df["tier2"] = (neutralized_df["tier"] >= 2).astype(int)
        mitigation_severity_tier2 = {m: {} for m in SEVERITY_METRICS}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(_safe_mitigation, metric, model, "tier2")
                for metric, model in tasks_s3
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="  S7-t2 mitigation", leave=False, delay=1,
            ):
                metric, model, res = future.result()
                mitigation_severity_tier2[metric][model] = res
        progress.update(1)

    progress.set_postfix_str("done")
    logger.info("Severity amplification analysis complete.")

    return {
        "descriptives": descriptives,
        "main_effects": main_effects,
        "interactions": interactions,
        "interactions_tier2": interactions_tier2,
        "dimension_severity": dim_severity,
        "three_way": {
            "anova": three_way,
            "model_amplification": model_amp,
            "worst_case": worst,
        },
        "model_comparison": model_amp,
        "process": process,
        "consistency": consist,
        "mitigation_severity": mitigation_severity,
        "mitigation_severity_tier2": mitigation_severity_tier2,
        "missingness_diagnostic": missingness,
    }
