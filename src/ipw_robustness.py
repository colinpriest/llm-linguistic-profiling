"""Inverse probability weighting (IPW) robustness for monetary severity.

Implements the pre-registered robustness analysis that addresses selection
bias in the amount-present (M=1) subset.  Dollar amounts are not
extractable for a substantial proportion of complaints; missingness is
expected to be systematic.  IPW re-weights the M=1 subset by
1/P(M=1|Z) so that the monetary amplification analysis better represents
the full complaint population.

Public API
----------
fit_propensity_model       Fit P(M=1|Z) and return weights.
ipw_amplification_analysis Run amplification side-by-side (unweighted vs IPW).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.estimable import NE, NEValue, is_estimable

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ["low", "medium", "high"]

# Weight trimming percentile (stabilises extreme weights)
_WEIGHT_TRIM_PCT = 99


# ------------------------------------------------------------------
# Propensity model
# ------------------------------------------------------------------

def fit_propensity_model(
    complaints_df: pd.DataFrame,
    predictors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit P(M=1|Z) using logistic regression.

    Parameters
    ----------
    complaints_df : DataFrame with at least:
        ``monetary_severity`` (values in low/medium/high/unknown),
        ``issue_category``, ``product_category``, ``word_count``.
    predictors : Predictor column names.  Defaults to
        ``["issue_category", "product_category", "word_count"]``.

    Returns
    -------
    Dict with keys: propensity_scores (ndarray aligned to complaints_df),
    weights, weights_stabilized, diagnostics, case_ids.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    if predictors is None:
        predictors = ["issue_category", "product_category", "word_count"]

    df = complaints_df.copy()
    df["M"] = df["monetary_severity"].isin(SEVERITY_LEVELS).astype(int)

    # Separate categorical and numeric predictors
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in predictors if c not in cat_cols]

    # Build feature matrix
    parts = []
    if cat_cols:
        enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="infrequent_if_exist")
        cat_X = enc.fit_transform(df[cat_cols].fillna("Unknown"))
        parts.append(cat_X)
    if num_cols:
        num_X = df[num_cols].fillna(0).values.astype(float)
        parts.append(num_X)

    if not parts:
        logger.warning("No valid predictors for propensity model")
        n = len(df)
        trivial_p = df["M"].mean()
        return {
            "propensity_scores": np.full(n, trivial_p),
            "weights": np.ones(n),
            "weights_stabilized": np.ones(n),
            "diagnostics": {"coverage_M1": float(trivial_p), "trivial": True},
            "case_ids": df["case_id"].values if "case_id" in df.columns else np.arange(n),
        }

    X = np.hstack(parts)

    # Fit logistic regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
        model.fit(X, df["M"])

    propensity = model.predict_proba(X)[:, 1]

    # Clip propensity to avoid extreme weights
    propensity = np.clip(propensity, 0.01, 0.99)

    # IPW weights for M=1 observations: w = 1/p
    weights = 1.0 / propensity

    # Trim extreme weights
    trim_val = np.percentile(weights, _WEIGHT_TRIM_PCT)
    weights_trimmed = np.minimum(weights, trim_val)

    # Stabilised weights: multiply by marginal P(M=1)
    marginal_p = df["M"].mean()
    weights_stabilized = (marginal_p / propensity)
    weights_stabilized = np.minimum(weights_stabilized, np.percentile(weights_stabilized, _WEIGHT_TRIM_PCT))

    # Diagnostics
    m1_mask = df["M"] == 1
    diagnostics = {
        "coverage_M1": float(marginal_p),
        "n_total": int(len(df)),
        "n_M1": int(m1_mask.sum()),
        "n_M0": int((~m1_mask).sum()),
        "propensity_range": [float(propensity.min()), float(propensity.max())],
        "propensity_mean": float(propensity.mean()),
        "weight_range_M1": [
            float(weights[m1_mask].min()) if m1_mask.any() else 0,
            float(weights[m1_mask].max()) if m1_mask.any() else 0,
        ],
        "effective_sample_size_M1": (
            float((weights_stabilized[m1_mask].sum() ** 2) / (weights_stabilized[m1_mask] ** 2).sum())
            if m1_mask.any() else 0
        ),
    }

    case_ids = df["case_id"].values if "case_id" in df.columns else np.arange(len(df))

    logger.info(
        "IPW propensity model: P(M=1) = %.3f, effective N = %.0f / %d",
        marginal_p, diagnostics["effective_sample_size_M1"], diagnostics["n_M1"],
    )

    return {
        "propensity_scores": propensity,
        "weights": weights,
        "weights_stabilized": weights_stabilized,
        "diagnostics": diagnostics,
        "case_ids": case_ids,
    }


# ------------------------------------------------------------------
# Weighted amplification helpers
# ------------------------------------------------------------------

def _weighted_bias_at_level(
    df: pd.DataFrame,
    severity_metric: str,
    level: str,
    persona_col: str,
    outcome_col: str,
    weight_col: str,
) -> Any:
    """Compute weighted SD of persona-level means at a severity level."""
    sub = df[df[severity_metric] == level]
    if len(sub) < 5:
        return NE.insufficient_n(n=len(sub), min_required=5, context=f"weighted bias {level}")

    # Weighted mean per persona
    grouped = sub.groupby(persona_col).apply(
        lambda g: np.average(g[outcome_col], weights=g[weight_col]) if len(g) > 0 else np.nan,
        include_groups=False,
    )
    grouped = grouped.dropna()
    if len(grouped) < 2:
        return 0.0
    return float(grouped.std(ddof=1))


def _weighted_dim_bias_at_level(
    df: pd.DataFrame,
    severity_metric: str,
    level: str,
    dimension: str,
    weight_col: str,
) -> Any:
    """Compute max-min gap of weighted dimension-level means at a severity level."""
    sub = df[df[severity_metric] == level]
    if len(sub) < 5:
        return NE.insufficient_n(n=len(sub), min_required=5, context=f"weighted dim bias {level}")
    grouped = sub.groupby(dimension).apply(
        lambda g: np.average(g["tier"], weights=g[weight_col]) if len(g) > 0 else np.nan,
        include_groups=False,
    )
    grouped = grouped.dropna()
    if len(grouped) < 2:
        return 0.0
    return float(grouped.max() - grouped.min())


# ------------------------------------------------------------------
# Weighted interaction test
# ------------------------------------------------------------------

def _weighted_interaction_test(
    df: pd.DataFrame,
    severity_metric: str,
    dimension: str,
    weight_col: str,
) -> Dict[str, Any]:
    """Weighted OLS interaction test: tier ~ dim + severity + dim:severity."""
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        df_fit = df.rename(columns={severity_metric: "severity", dimension: "dim_level"}).copy()

        # Ensure complete cells
        cc = df_fit.groupby(["dim_level", "severity"]).size().unstack(fill_value=0)
        complete = cc.index[cc.min(axis=1) > 0]
        if len(complete) < 2:
            return {"error": "insufficient complete cells"}
        df_fit = df_fit[df_fit["dim_level"].isin(complete)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.wls(
                "tier ~ C(dim_level) + C(severity) + C(dim_level):C(severity)",
                data=df_fit,
                weights=df_fit[weight_col],
            ).fit()
            table = anova_lm(fit, typ=2)

        result: Dict[str, Any] = {}
        ss_resid = max(0.0, float(table.loc["Residual", "sum_sq"])) if "Residual" in table.index else 0
        for source, label in [
            ("interaction", "C(dim_level):C(severity)"),
            ("dimension_main", "C(dim_level)"),
            ("severity_main", "C(severity)"),
        ]:
            if label in table.index:
                ss = max(0.0, float(table.loc[label, "sum_sq"]))
                f_val_raw = table.loc[label, "F"] if "F" in table.columns else None
                p_val_raw = table.loc[label, "PR(>F)"] if "PR(>F)" in table.columns else None
                f_val: Any = float(f_val_raw) if f_val_raw is not None and np.isfinite(float(f_val_raw)) else NE.missing_data(field="F", context=source)
                p_val: Any = float(p_val_raw) if p_val_raw is not None and np.isfinite(float(p_val_raw)) else NE.missing_data(field="p", context=source)
                if is_estimable(f_val) and f_val < 0:
                    f_val, p_val = 0.0, 1.0
                eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                result[source] = {"F": f_val, "p_value": p_val, "eta_sq_partial": eta_sq}
        return result

    except Exception as e:
        logger.warning("Weighted interaction test failed: %s", e)
        return {"error": str(e)}


# ------------------------------------------------------------------
# Main public function
# ------------------------------------------------------------------

def ipw_amplification_analysis(
    decisions_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    models: List[str],
    dimensions: Optional[List[str]] = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run monetary amplification analysis with IPW weighting.

    Compares unweighted vs IPW-weighted amplification within the M=1
    (amount-present) subset only.

    Parameters
    ----------
    decisions_df : Persona decisions DataFrame (must have model, case_id,
        tier, monetary_severity columns).
    complaints_df : Complaints DataFrame for propensity model (must have
        case_id, monetary_severity, issue_category, product_category,
        word_count).
    models : List of model identifiers.
    dimensions : Demographic dimensions to test.  Defaults to
        ``["age", "geographic", "education", "socioeconomic"]``.
    n_boot : Bootstrap iterations.
    seed : Random seed.

    Returns
    -------
    Dict[model] -> {unweighted, weighted, diagnostics, dimensions}.
    """
    if dimensions is None:
        dimensions = ["age", "geographic", "education", "socioeconomic"]

    # 1. Fit propensity model on full complaints
    propensity_result = fit_propensity_model(complaints_df)
    propensity_map = dict(zip(propensity_result["case_ids"], propensity_result["weights_stabilized"]))

    result: Dict[str, Any] = {
        "propensity_diagnostics": propensity_result["diagnostics"],
    }

    severity_metric = "monetary_severity"

    for model in models:
        logger.info("IPW amplification analysis for %s", model)
        model_df = decisions_df[
            (decisions_df["model"] == model)
            & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
        ].copy()

        if len(model_df) < 30:
            result[model] = {"skipped": True, "reason": "insufficient M=1 observations"}
            continue

        # Assign IPW weights
        model_df["ipw_weight"] = model_df["case_id"].map(propensity_map).fillna(1.0)

        model_result: Dict[str, Any] = {"dimensions": {}}

        for dim in dimensions:
            if dim not in model_df.columns:
                continue

            dim_result: Dict[str, Any] = {}

            # --- Unweighted ---
            bias_low_uw = _weighted_dim_bias_at_level(
                model_df.assign(w=1.0), severity_metric, "low", dim, "w",
            )
            bias_high_uw = _weighted_dim_bias_at_level(
                model_df.assign(w=1.0), severity_metric, "high", dim, "w",
            )
            ne_missing = NE.missing_data(context=f"IPW {dim}")
            A_uw = bias_high_uw - bias_low_uw if is_estimable(bias_high_uw) and is_estimable(bias_low_uw) else ne_missing
            R_uw = (bias_high_uw / bias_low_uw) if (is_estimable(bias_low_uw) and bias_low_uw > 0) else (NE.zero_denominator(context="unweighted R") if is_estimable(bias_high_uw) else ne_missing)

            uw_interaction = _weighted_interaction_test(
                model_df.assign(w=1.0), severity_metric, dim, "w",
            )

            # Extract the interaction p-value from the nested structure
            uw_int_pval = uw_interaction.get("interaction", {}).get("p_value", ne_missing)
            uw_int_F = uw_interaction.get("interaction", {}).get("F", ne_missing)
            uw_int_eta = uw_interaction.get("interaction", {}).get("eta_sq_partial", ne_missing)

            dim_result["unweighted"] = {
                "bias_low": float(bias_low_uw) if is_estimable(bias_low_uw) else bias_low_uw,
                "bias_high": float(bias_high_uw) if is_estimable(bias_high_uw) else bias_high_uw,
                "A": float(A_uw) if is_estimable(A_uw) else A_uw,
                "R": float(R_uw) if is_estimable(R_uw) else R_uw,
                # Reporter expects amplification.point
                "amplification": {"point": float(R_uw) if is_estimable(R_uw) else R_uw},
                # Reporter expects interaction.p_value (flattened)
                "interaction": {
                    "p_value": uw_int_pval,
                    "F": uw_int_F,
                    "eta_sq_partial": uw_int_eta,
                },
                "interaction_full": uw_interaction,  # Keep full structure for reference
            }

            # --- IPW-weighted ---
            bias_low_w = _weighted_dim_bias_at_level(
                model_df, severity_metric, "low", dim, "ipw_weight",
            )
            bias_high_w = _weighted_dim_bias_at_level(
                model_df, severity_metric, "high", dim, "ipw_weight",
            )
            A_w = bias_high_w - bias_low_w if is_estimable(bias_high_w) and is_estimable(bias_low_w) else ne_missing
            R_w = (bias_high_w / bias_low_w) if (is_estimable(bias_low_w) and bias_low_w > 0) else (NE.zero_denominator(context="weighted R") if is_estimable(bias_high_w) else ne_missing)

            w_interaction = _weighted_interaction_test(
                model_df, severity_metric, dim, "ipw_weight",
            )

            # Extract the interaction p-value from the nested structure
            w_int_pval = w_interaction.get("interaction", {}).get("p_value", ne_missing)
            w_int_F = w_interaction.get("interaction", {}).get("F", ne_missing)
            w_int_eta = w_interaction.get("interaction", {}).get("eta_sq_partial", ne_missing)

            dim_result["weighted"] = {
                "bias_low": float(bias_low_w) if is_estimable(bias_low_w) else bias_low_w,
                "bias_high": float(bias_high_w) if is_estimable(bias_high_w) else bias_high_w,
                "A": float(A_w) if is_estimable(A_w) else A_w,
                "R": float(R_w) if is_estimable(R_w) else R_w,
                # Reporter expects amplification.point
                "amplification": {"point": float(R_w) if is_estimable(R_w) else R_w},
                # Reporter expects interaction.p_value (flattened)
                "interaction": {
                    "p_value": w_int_pval,
                    "F": w_int_F,
                    "eta_sq_partial": w_int_eta,
                },
                "interaction_full": w_interaction,  # Keep full structure for reference
            }

            # --- Bootstrap CIs (clustered by complaint) ---
            rng = np.random.RandomState(seed)
            case_ids = model_df["case_id"].unique()
            boot_A_uw, boot_A_w, boot_R_uw, boot_R_w = [], [], [], []

            for _ in tqdm(range(n_boot), desc=f"  bootstrap(IPW,{dim})", leave=False, delay=1):
                sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
                boot_df = model_df[model_df["case_id"].isin(sample_ids)]

                # Unweighted bootstrap
                bl_uw = _weighted_dim_bias_at_level(boot_df.assign(w=1.0), severity_metric, "low", dim, "w")
                bh_uw = _weighted_dim_bias_at_level(boot_df.assign(w=1.0), severity_metric, "high", dim, "w")
                if is_estimable(bl_uw) and is_estimable(bh_uw):
                    boot_A_uw.append(bh_uw - bl_uw)
                    if bl_uw > 0:
                        boot_R_uw.append(bh_uw / bl_uw)

                # Weighted bootstrap
                bl_w = _weighted_dim_bias_at_level(boot_df, severity_metric, "low", dim, "ipw_weight")
                bh_w = _weighted_dim_bias_at_level(boot_df, severity_metric, "high", dim, "ipw_weight")
                if is_estimable(bl_w) and is_estimable(bh_w):
                    boot_A_w.append(bh_w - bl_w)
                    if bl_w > 0:
                        boot_R_w.append(bh_w / bl_w)

            def _ci(vals: list) -> Dict[str, Any]:
                if len(vals) >= 10:
                    return {"ci_lower": float(np.percentile(vals, 2.5)),
                            "ci_upper": float(np.percentile(vals, 97.5))}
                return {"ci_lower": NE.insufficient_n(n=len(vals), min_required=10, context="bootstrap CI"),
                        "ci_upper": NE.insufficient_n(n=len(vals), min_required=10, context="bootstrap CI")}

            dim_result["unweighted"]["A_ci"] = _ci(boot_A_uw)
            dim_result["unweighted"]["R_ci"] = _ci(boot_R_uw)
            dim_result["weighted"]["A_ci"] = _ci(boot_A_w)
            dim_result["weighted"]["R_ci"] = _ci(boot_R_w)

            # Material change indicator
            if is_estimable(A_uw) and is_estimable(A_w):
                dim_result["material_change"] = abs(A_uw - A_w) > 0.5 * abs(A_uw) if A_uw != 0 else A_w != 0
            else:
                dim_result["material_change"] = None

            model_result["dimensions"][dim] = dim_result

        model_result["n_observations"] = len(model_df)
        model_result["effective_n"] = propensity_result["diagnostics"].get("effective_sample_size_M1", 0)
        result[model] = model_result

    return result
