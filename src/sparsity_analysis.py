"""Sparsity-aware fairness analysis for the LLM fairness experiment.

Provides a second-pass analysis that handles small-cell / sparse subgroups
defensibly.  Two regimes are distinguished:

* **Designed cells** (matched-guise persona rewrites) – balanced by construction,
  sparsity is less of a concern.
* **Observed/inferred cells** (persona estimation from text) – group sizes are
  unbalanced and can be very small; rates and AIR from tiny cells are unstable.

Implements:
  (A) Support-annotated reporting with Wilson/Jeffreys intervals
  (B) Minimum-support thresholds (n<20 insufficient, 20≤n<100 screening, n≥100 full)
  (C) Bayesian interval AIR via Jeffreys prior with P(AIR<0.8) risk probability
  (D) Partial pooling via hierarchical logistic regression for intersectional cells
  (E) Sensitivity analysis: re-run under exclusion / collapse rules

The single public entry point is :func:`run_sparsity_analysis`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.schemas import ESTIMATE_DIMS, CAUSAL_DIMS
from src.statistics import adverse_impact_ratio, fdr_correction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_N_FULL = 100       # full inferential analysis
MIN_N_SCREENING = 20   # screening-only (rates + intervals, no strong claims)
# Below MIN_N_SCREENING: insufficient support

N_POSTERIOR_SAMPLES = 10_000
JEFFREYS_ALPHA = 0.5   # Jeffreys prior: Beta(0.5, 0.5)
JEFFREYS_BETA = 0.5

AIR_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# (A) Support-annotated rate computation with intervals
# ---------------------------------------------------------------------------

def _stability_flag(n: int) -> str:
    """Return a human-readable stability flag based on support size."""
    if n < MIN_N_SCREENING:
        return "insufficient_support"
    if n < MIN_N_FULL:
        return "screening_only"
    return "reliable"


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def jeffreys_interval(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Jeffreys credible interval for a binomial proportion.

    Posterior: Beta(successes + 0.5, n - successes + 0.5).
    """
    if n == 0:
        return (0.0, 1.0)
    a = successes + JEFFREYS_ALPHA
    b = n - successes + JEFFREYS_BETA
    lo = float(sp_stats.beta.ppf(alpha / 2, a, b))
    hi = float(sp_stats.beta.ppf(1 - alpha / 2, a, b))
    return (max(0.0, lo), min(1.0, hi))


def annotated_rate(successes: int, n: int, alpha: float = 0.05) -> Dict[str, Any]:
    """Compute a rate with full support annotation (policy A)."""
    rate = successes / n if n > 0 else float("nan")
    wilson_lo, wilson_hi = wilson_interval(successes, n, alpha)
    jeff_lo, jeff_hi = jeffreys_interval(successes, n, alpha)
    return {
        "n": n,
        "successes": successes,
        "rate": float(rate),
        "wilson_ci": (float(wilson_lo), float(wilson_hi)),
        "jeffreys_ci": (float(jeff_lo), float(jeff_hi)),
        "stability": _stability_flag(n),
    }


def compute_annotated_rates(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
) -> Dict[str, Dict[str, Any]]:
    """Compute annotated rates for every level of *group_col*.

    Parameters
    ----------
    df : DataFrame containing the group and binary outcome columns.
    group_col : Column to group by (e.g. 'persona_label', 'age').
    outcome_col : Binary outcome column (1 = favorable).
    """
    result: Dict[str, Dict[str, Any]] = {}
    for group, sub in df.groupby(group_col):
        n = len(sub)
        successes = int(sub[outcome_col].sum())
        result[str(group)] = annotated_rate(successes, n)
    return result


# ---------------------------------------------------------------------------
# (B) Minimum-support thresholds
# ---------------------------------------------------------------------------

def support_summary(
    df: pd.DataFrame,
    group_col: str,
) -> Dict[str, Any]:
    """Summarise support per group and classify each cell."""
    counts = df[group_col].value_counts()
    cells: Dict[str, Dict[str, Any]] = {}
    for group, n in counts.items():
        cells[str(group)] = {
            "n": int(n),
            "stability": _stability_flag(int(n)),
        }
    n_insufficient = sum(1 for c in cells.values() if c["stability"] == "insufficient_support")
    n_screening = sum(1 for c in cells.values() if c["stability"] == "screening_only")
    n_reliable = sum(1 for c in cells.values() if c["stability"] == "reliable")
    return {
        "cells": cells,
        "n_groups": len(cells),
        "n_insufficient": n_insufficient,
        "n_screening": n_screening,
        "n_reliable": n_reliable,
        "overall_n": len(df),
    }


# ---------------------------------------------------------------------------
# (C) Bayesian interval AIR via Jeffreys posterior
# ---------------------------------------------------------------------------

def bayesian_air(
    successes_g: int,
    n_g: int,
    successes_ref: int,
    n_ref: int,
    n_samples: int = N_POSTERIOR_SAMPLES,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Bayesian AIR from Jeffreys posteriors.

    For each group g and the reference (most-favorable) group, draw from
    Beta(y + 0.5, n - y + 0.5) posteriors, compute AIR = p_g / p_ref,
    and report median, 95% credible interval, and P(AIR < 0.8).
    """
    rng = np.random.RandomState(seed)

    a_g = successes_g + JEFFREYS_ALPHA
    b_g = n_g - successes_g + JEFFREYS_BETA
    a_ref = successes_ref + JEFFREYS_ALPHA
    b_ref = n_ref - successes_ref + JEFFREYS_BETA

    p_g_samples = rng.beta(a_g, b_g, size=n_samples)
    p_ref_samples = rng.beta(a_ref, b_ref, size=n_samples)

    # Avoid division by zero in samples
    valid = p_ref_samples > 0
    if valid.sum() < 100:
        return {
            "median_air": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_below_threshold": float("nan"),
            "stability_g": _stability_flag(n_g),
            "stability_ref": _stability_flag(n_ref),
        }

    air_samples = p_g_samples[valid] / p_ref_samples[valid]
    return {
        "median_air": float(np.median(air_samples)),
        "ci_lower": float(np.percentile(air_samples, 2.5)),
        "ci_upper": float(np.percentile(air_samples, 97.5)),
        "p_below_threshold": float(np.mean(air_samples < AIR_THRESHOLD)),
        "stability_g": _stability_flag(n_g),
        "stability_ref": _stability_flag(n_ref),
    }


def compute_bayesian_air_all(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Compute Bayesian AIR for every group relative to the most-favorable.

    Also includes the annotated rate for each group.
    """
    rates_info = compute_annotated_rates(df, group_col, outcome_col)

    # Identify reference (most-favorable) group
    best_group = max(rates_info, key=lambda g: rates_info[g]["rate"])
    ref = rates_info[best_group]

    result: Dict[str, Dict[str, Any]] = {}
    for group, info in rates_info.items():
        bair = bayesian_air(
            info["successes"], info["n"],
            ref["successes"], ref["n"],
            seed=seed,
        )
        result[group] = {
            **info,
            "bayesian_air": bair,
            "point_air": float(adverse_impact_ratio(info["rate"], ref["rate"])),
            "reference_group": best_group,
        }
    return result


# ---------------------------------------------------------------------------
# (D) Partial pooling via hierarchical logistic regression
# ---------------------------------------------------------------------------

def partial_pooling_model(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str = "persona_label",
    severity_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit a hierarchical (mixed-effects) model with partial pooling.

    Model: outcome ~ 1 + (1 | persona)
    Optionally: outcome ~ severity + (1 | persona)

    Uses MixedLM on the linear probability model.

    Fallback chain:
    1. MixedLM with REML.
    2. rank_deficient / convergence → simpler random-intercept-only model.
    3. Remaining failure → skip with diagnostic.
    """
    from src.regression_diagnostics import diagnose_fit_failure

    model_label = f"PartialPooling({outcome_col})"
    if len(df) < 30:
        return {"error": "insufficient data", "n": len(df)}
    if df[group_col].nunique() < 3:
        return {"error": "too few groups for partial pooling", "n_groups": df[group_col].nunique()}

    # Ensure binary outcome
    df = df.copy()
    df[outcome_col] = df[outcome_col].astype(int)

    # Pre-flight: check response variance
    if df[outcome_col].nunique() < 2:
        return {"error": f"response variable '{outcome_col}' is constant (value={df[outcome_col].iloc[0]})",
                "diagnosis": "no_variance_y", "converged": False}

    # Raw (unpooled) group rates
    raw_rates = df.groupby(group_col)[outcome_col].agg(["mean", "count"]).rename(
        columns={"mean": "rate", "count": "n"}
    )

    result: Dict[str, Any] = {
        "raw_rates": raw_rates.to_dict("index"),
        "n_groups": int(df[group_col].nunique()),
        "n_total": len(df),
    }

    def _fit_pooling(use_severity: bool):
        import statsmodels.formula.api as smf

        fixed = "1"
        if use_severity and severity_col and severity_col in df.columns and df[severity_col].nunique() >= 2:
            fixed = f"C({severity_col})"

        formula = f"{outcome_col} ~ {fixed}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                formula,
                data=df,
                groups=df[group_col],
            ).fit(reml=True)
        return model

    model = None
    # Attempt 1: with severity if available
    try:
        model = _fit_pooling(use_severity=True)
    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label)
        logger.warning("%s attempt 1 failed (%s): %s", model_label, fit_diag["type"], e)
        # Attempt 2: intercept-only (simpler)
        try:
            model = _fit_pooling(use_severity=False)
        except Exception as e2:
            fit_diag2 = diagnose_fit_failure(e2, model_label=model_label)
            logger.warning("%s attempt 2 failed (%s): %s", model_label, fit_diag2["type"], e2)
            result["error"] = str(e2)
            result["diagnosis"] = fit_diag2["type"]
            result["converged"] = False
            return result

    # Extract random effects (shrinkage estimates)
    re = model.random_effects
    grand_mean = float(model.fe_params.get("Intercept", model.fe_params.iloc[0]))
    pooled_rates: Dict[str, Dict[str, Any]] = {}

    for group_name, effects in re.items():
        re_val = float(effects.iloc[0]) if hasattr(effects, "iloc") else float(effects)
        pooled_rate = grand_mean + re_val
        raw = raw_rates.loc[group_name] if group_name in raw_rates.index else None
        pooled_rates[str(group_name)] = {
            "pooled_estimate": float(np.clip(pooled_rate, 0, 1)),
            "raw_rate": float(raw["rate"]) if raw is not None else float("nan"),
            "raw_n": int(raw["n"]) if raw is not None else 0,
            "shrinkage": float(raw["rate"] - np.clip(pooled_rate, 0, 1)) if raw is not None else float("nan"),
            "stability": _stability_flag(int(raw["n"])) if raw is not None else "insufficient_support",
        }

    result["pooled_rates"] = pooled_rates
    result["grand_mean"] = grand_mean
    result["random_effect_variance"] = float(model.cov_re.iloc[0, 0]) if hasattr(model.cov_re, "iloc") else float("nan")
    result["converged"] = True

    # Shrinkage summary: how much did small cells move?
    shrinkage_vals = [v["shrinkage"] for v in pooled_rates.values()
                      if np.isfinite(v["shrinkage"])]
    if shrinkage_vals:
        result["shrinkage_summary"] = {
            "mean_abs_shrinkage": float(np.mean(np.abs(shrinkage_vals))),
            "max_abs_shrinkage": float(np.max(np.abs(shrinkage_vals))),
            "n_cells_shrunk_gt_5pct": int(sum(1 for s in shrinkage_vals if abs(s) > 0.05)),
        }

    return result


# ---------------------------------------------------------------------------
# (E) Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    dims: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Re-run headline disparity results under alternative sparsity rules.

    Rule 1: Exclude cells with n < 20.
    Rule 2: Collapse sparse categories (merge smallest into nearest).

    Returns a comparison of AIR results across the three approaches.
    """
    if dims is None:
        dims = ESTIMATE_DIMS

    results: Dict[str, Any] = {}

    # --- Baseline (all data) ---
    baseline_air = _headline_air(df, group_col, outcome_col)
    results["baseline"] = baseline_air

    # --- Rule 1: Exclude cells with n < 20 ---
    counts = df[group_col].value_counts()
    keep_groups = counts[counts >= MIN_N_SCREENING].index
    df_filtered = df[df[group_col].isin(keep_groups)]
    n_excluded = len(counts) - len(keep_groups)

    if len(df_filtered) > 0 and df_filtered[group_col].nunique() >= 2:
        filtered_air = _headline_air(df_filtered, group_col, outcome_col)
        filtered_air["n_groups_excluded"] = n_excluded
        filtered_air["groups_excluded"] = [str(g) for g in counts.index if g not in keep_groups]
    else:
        filtered_air = {"error": "too few groups after filtering", "n_groups_excluded": n_excluded}
    results["exclude_small_cells"] = filtered_air

    # --- Rule 2: Collapse sparse dimensions ---
    collapse_results: Dict[str, Any] = {}
    for dim in dims:
        if dim not in df.columns:
            continue
        dim_counts = df[dim].value_counts()
        if len(dim_counts) < 2:
            continue
        # Find levels with n < MIN_N_SCREENING
        sparse_levels = dim_counts[dim_counts < MIN_N_SCREENING].index.tolist()
        if not sparse_levels:
            collapse_results[dim] = {"collapsed": False, "reason": "no sparse levels"}
            continue

        # Merge sparse levels into the nearest (most similar) non-sparse level
        df_collapsed = df.copy()
        non_sparse = dim_counts[dim_counts >= MIN_N_SCREENING].index.tolist()
        if not non_sparse:
            # All levels are sparse; merge everything into one
            df_collapsed[dim] = "merged"
            collapse_results[dim] = {"collapsed": True, "merged_into": "merged",
                                      "sparse_levels": sparse_levels, "n_remaining": 1}
        else:
            merge_target = non_sparse[0]  # merge into the largest non-sparse level
            df_collapsed[dim] = df_collapsed[dim].replace({l: merge_target for l in sparse_levels})
            collapse_results[dim] = {
                "collapsed": True,
                "merge_target": merge_target,
                "sparse_levels": sparse_levels,
                "n_remaining": df_collapsed[dim].nunique(),
            }

        # Re-compute AIR on the collapsed dimension
        if df_collapsed[dim].nunique() >= 2:
            df_collapsed["_favorable"] = df_collapsed[outcome_col].astype(int)
            dim_rates = df_collapsed.groupby(dim)["_favorable"].mean()
            best = dim_rates.max()
            collapse_results[dim]["air_after_collapse"] = {
                str(g): float(adverse_impact_ratio(r, best))
                for g, r in dim_rates.items()
            }

    results["collapse_sparse"] = collapse_results

    # --- Robustness verdict ---
    verdict = _robustness_verdict(results)
    results["verdict"] = verdict

    return results


def _headline_air(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
) -> Dict[str, Any]:
    """Compute headline AIR statistics for a given grouping."""
    df = df.copy()
    df["_favorable"] = df[outcome_col].astype(int)
    rates = df.groupby(group_col)["_favorable"].agg(["mean", "sum", "count"]).rename(
        columns={"mean": "rate", "sum": "successes", "count": "n"}
    )
    if rates.empty:
        return {"error": "no data"}
    best_rate = rates["rate"].max()
    best_group = rates["rate"].idxmax()

    air_by_group: Dict[str, Dict[str, Any]] = {}
    for group_name, row in rates.iterrows():
        air = adverse_impact_ratio(row["rate"], best_rate)
        air_by_group[str(group_name)] = {
            "rate": float(row["rate"]),
            "air": float(air),
            "n": int(row["n"]),
            "material": air < AIR_THRESHOLD,
            "stability": _stability_flag(int(row["n"])),
        }

    n_material = sum(1 for v in air_by_group.values() if v["material"])
    n_material_reliable = sum(
        1 for v in air_by_group.values()
        if v["material"] and v["stability"] == "reliable"
    )
    n_material_unstable = sum(
        1 for v in air_by_group.values()
        if v["material"] and v["stability"] != "reliable"
    )

    return {
        "air_by_group": air_by_group,
        "best_group": str(best_group),
        "best_rate": float(best_rate),
        "n_groups": len(air_by_group),
        "n_material": n_material,
        "n_material_reliable": n_material_reliable,
        "n_material_unstable": n_material_unstable,
    }


def _robustness_verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare headline conclusions across baseline, filtered, and collapsed."""
    baseline = results.get("baseline", {})
    filtered = results.get("exclude_small_cells", {})

    baseline_n_mat = baseline.get("n_material", 0)
    filtered_n_mat = filtered.get("n_material", 0) if "error" not in filtered else None

    # Check if conclusions survive filtering
    survives_filtering = (
        filtered_n_mat is not None
        and filtered_n_mat > 0
        and baseline_n_mat > 0
    ) if baseline_n_mat > 0 else True

    # Check how many material flags came from unreliable cells
    baseline_unstable = baseline.get("n_material_unstable", 0)

    if baseline_n_mat == 0:
        summary = "No material disparities detected in any analysis."
    elif baseline_unstable == 0:
        summary = "All material disparities are from groups with reliable support (n≥100)."
    elif survives_filtering:
        summary = (
            f"Results are robust to sparsity handling. "
            f"{baseline_unstable} of {baseline_n_mat} material flags come from "
            f"small cells but conclusions survive exclusion of n<20 groups."
        )
    else:
        summary = (
            f"Caution: {baseline_unstable} of {baseline_n_mat} material flags "
            f"come from groups with insufficient or screening-level support. "
            f"Conclusions may not be stable."
        )

    return {
        "robust": survives_filtering,
        "baseline_material": baseline_n_mat,
        "baseline_material_unstable": baseline_unstable,
        "filtered_material": filtered_n_mat,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Per-dimension sparsity-aware analysis
# ---------------------------------------------------------------------------

def dimension_sparsity_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    dims: Optional[List[str]] = None,
    regime: str = "observed",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run sparsity-aware analysis for each demographic dimension.

    Parameters
    ----------
    df : DataFrame with outcome and dimension columns.
    outcome_col : Binary outcome column.
    dims : Dimensions to analyze.
    regime : 'designed' (matched-guise) or 'observed' (inferred).
    seed : Random seed.
    """
    if dims is None:
        dims = ESTIMATE_DIMS

    result: Dict[str, Any] = {"regime": regime}
    for dim in dims:
        if dim not in df.columns:
            continue
        dim_result: Dict[str, Any] = {}

        # Support summary
        dim_result["support"] = support_summary(df, dim)

        # Annotated rates
        df_dim = df.copy()
        df_dim["_favorable"] = df_dim[outcome_col].astype(int)
        dim_result["annotated_rates"] = compute_annotated_rates(df_dim, dim, "_favorable")

        # Bayesian AIR (only for observed regime or when useful)
        if regime == "observed":
            dim_result["bayesian_air"] = compute_bayesian_air_all(df_dim, dim, "_favorable", seed=seed)

        result[dim] = dim_result

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_sparsity_analysis(
    baseline_df: pd.DataFrame,
    personas_df: pd.DataFrame,
    models: List[str],
    seed: int = 42,
    persona_decisions_df: Optional[pd.DataFrame] = None,
    neutralized_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run the full sparsity-aware analysis.

    Parameters
    ----------
    baseline_df : Baseline remedy decisions merged with persona labels.
        Columns: [case_id, model, tier, confidence, persona_label]
    personas_df : Persona estimates with all dimension columns.
        Columns: [case_id, model, persona_label, age, geographic, education, ...]
    models : List of model identifiers.
    seed : Random seed.
    persona_decisions_df : Optional matched-guise decisions.
        Columns: [case_id, model, variant, tier, confidence]
    neutralized_df : Optional neutralized decisions.
        Columns: [case_id, model, tier, confidence, persona_label]

    Returns
    -------
    Dict with keys: observed, designed, partial_pooling, sensitivity,
    per_dimension, mitigation_sensitivity.
    """
    logger.info("Running sparsity-aware analysis...")
    results: Dict[str, Any] = {}

    # --- Merge persona dimensions onto baseline ---
    dim_cols = [c for c in ESTIMATE_DIMS if c in personas_df.columns]
    merge_cols = ["case_id", "model", "persona_label"] + dim_cols
    merged = baseline_df.merge(
        personas_df[merge_cols].drop_duplicates(),
        on=["case_id", "model"],
        how="left",
        suffixes=("", "_dup"),
    )
    # Clean up any duplicate persona_label columns
    if "persona_label_dup" in merged.columns:
        merged["persona_label"] = merged["persona_label"].fillna(merged["persona_label_dup"])
        merged.drop(columns=["persona_label_dup"], inplace=True)

    merged["favorable"] = (merged["tier"] >= 1).astype(int)

    for model in models:
        logger.info("  Sparsity analysis for model: %s", model)
        model_df = merged[merged["model"] == model].copy()
        model_results: Dict[str, Any] = {}

        # ---- (A+B) Observed / inferred group analysis ----
        logger.info("    (A+B) Support-annotated rates for observed groups")
        if "persona_label" in model_df.columns:
            model_results["observed_persona"] = {
                "support": support_summary(model_df, "persona_label"),
                "annotated_rates": compute_annotated_rates(
                    model_df, "persona_label", "favorable"
                ),
            }

        # ---- Per-dimension (observed) ----
        logger.info("    Per-dimension sparsity analysis (observed)")
        model_results["per_dimension_observed"] = dimension_sparsity_analysis(
            model_df, "favorable", dims=dim_cols, regime="observed", seed=seed,
        )

        # ---- (C) Bayesian interval AIR ----
        logger.info("    (C) Bayesian interval AIR")
        if "persona_label" in model_df.columns:
            model_results["bayesian_air_persona"] = compute_bayesian_air_all(
                model_df, "persona_label", "favorable", seed=seed,
            )

        # ---- (D) Partial pooling ----
        logger.info("    (D) Partial pooling model")
        if "persona_label" in model_df.columns:
            model_results["partial_pooling"] = partial_pooling_model(
                model_df, "favorable", group_col="persona_label",
            )

        # ---- (E) Sensitivity analysis (observed) ----
        logger.info("    (E) Sensitivity analysis")
        if "persona_label" in model_df.columns:
            model_results["sensitivity_persona"] = sensitivity_analysis(
                model_df, "persona_label", "favorable",
                dims=dim_cols, seed=seed,
            )

        # ---- Designed / matched-guise analysis (if available) ----
        if persona_decisions_df is not None and len(persona_decisions_df) > 0:
            logger.info("    Designed-cell (matched-guise) analysis")
            causal_df = persona_decisions_df[persona_decisions_df["model"] == model].copy()
            if len(causal_df) > 0:
                causal_df["favorable"] = (causal_df["tier"] >= 1).astype(int)
                model_results["designed_persona"] = {
                    "support": support_summary(causal_df, "variant"),
                    "annotated_rates": compute_annotated_rates(
                        causal_df, "variant", "favorable"
                    ),
                    "bayesian_air": compute_bayesian_air_all(
                        causal_df, "variant", "favorable", seed=seed,
                    ),
                }

        # ---- Mitigation sensitivity (if neutralized available) ----
        if neutralized_df is not None and len(neutralized_df) > 0:
            logger.info("    Mitigation sparsity-aware comparison")
            neut_model = neutralized_df[neutralized_df["model"] == model].copy()
            if "persona_label" in neut_model.columns and len(neut_model) > 0:
                neut_model["favorable"] = (neut_model["tier"] >= 1).astype(int)
                neut_air = compute_bayesian_air_all(
                    neut_model, "persona_label", "favorable", seed=seed,
                )
                # Compare pre vs post mitigation Bayesian AIR
                pre_air = model_results.get("bayesian_air_persona", {})
                mitigation_comparison: Dict[str, Any] = {}
                for persona in pre_air:
                    pre = pre_air[persona].get("bayesian_air", {})
                    post = neut_air.get(persona, {}).get("bayesian_air", {})
                    if pre and post:
                        mitigation_comparison[persona] = {
                            "pre_median_air": pre.get("median_air", float("nan")),
                            "post_median_air": post.get("median_air", float("nan")),
                            "pre_p_below_08": pre.get("p_below_threshold", float("nan")),
                            "post_p_below_08": post.get("p_below_threshold", float("nan")),
                            "pre_stability": pre.get("stability_g", "unknown"),
                            "post_stability": post.get("stability_g", "unknown"),
                        }
                model_results["mitigation_sensitivity"] = {
                    "post_bayesian_air": neut_air,
                    "comparison": mitigation_comparison,
                }

                # Sensitivity on neutralized
                model_results["sensitivity_neutralized"] = sensitivity_analysis(
                    neut_model, "persona_label", "favorable",
                    dims=dim_cols, seed=seed,
                )

        results[model] = model_results

    # --- Process discrimination sparsity (info request) ---
    logger.info("  Process discrimination sparsity analysis")
    process_sparsity: Dict[str, Any] = {}
    if "confidence" in merged.columns:
        merged["need_info"] = (merged["confidence"] == "need_more_info").astype(int)
        for model in models:
            model_df = merged[merged["model"] == model].copy()
            if "persona_label" in model_df.columns:
                process_sparsity[model] = {
                    "support": support_summary(model_df, "persona_label"),
                    "annotated_rates": compute_annotated_rates(
                        model_df, "persona_label", "need_info"
                    ),
                    "bayesian_air": compute_bayesian_air_all(
                        model_df, "persona_label", "need_info", seed=seed,
                    ),
                }
    results["process_sparsity"] = process_sparsity

    logger.info("Sparsity-aware analysis complete.")
    return results
