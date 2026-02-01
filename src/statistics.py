"""Statistical analysis module for disparate impact (outcome) analysis."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.schemas import ESTIMATE_DIMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adverse impact ratio helpers
# ---------------------------------------------------------------------------

def adverse_impact_ratio(rate_a: float, rate_b: float) -> float:
    """Calculate AIR; rate_b is the most favorable group."""
    if rate_b == 0:
        return float("inf") if rate_a > 0 else 1.0
    return rate_a / rate_b


def is_material(air: float, threshold: float = 0.80) -> bool:
    return air < threshold


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------

def fdr_correction(p_values: List[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    from scipy.stats import false_discovery_control
    arr = np.array(p_values, dtype=float)
    if len(arr) == 0:
        return arr
    return false_discovery_control(arr, method="bh")


def interpretation_flag(p_adj: float, air: float, threshold: float = 0.80) -> str:
    sig = p_adj < 0.05
    mat = air < threshold
    if sig and mat:
        return "BOTH"
    if sig:
        return "SIG_ONLY"
    if mat:
        return "MAT_ONLY"
    return "NONE"


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def chi_squared_test(
    contingency: pd.DataFrame,
) -> Dict[str, Any]:
    """Chi-squared test of independence on a contingency table."""
    chi2, p, dof, expected = sp_stats.chi2_contingency(contingency.values)
    n = contingency.values.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v}


def fisher_exact_test(table_2x2: np.ndarray) -> Dict[str, Any]:
    """Fisher's exact test for 2×2 tables."""
    odds_ratio, p = sp_stats.fisher_exact(table_2x2)
    return {"odds_ratio": odds_ratio, "p_value": p}


def mcnemar_test(table_2x2: np.ndarray) -> Dict[str, Any]:
    """McNemar's test for paired categorical data."""
    from statsmodels.stats.contingency_tables import mcnemar
    result = mcnemar(table_2x2, exact=table_2x2.sum() < 25)
    return {"statistic": result.statistic, "p_value": result.pvalue}


def friedman_test(*groups: np.ndarray) -> Dict[str, Any]:
    """Friedman test for repeated measures."""
    stat, p = sp_stats.friedmanchisquare(*groups)
    return {"statistic": stat, "p_value": p}


def one_way_anova(
    groups: List[np.ndarray],
    group_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """One-way ANOVA with partial eta-squared.

    Parameters
    ----------
    groups : list of arrays, one per group
    group_labels : optional names for each group

    Returns
    -------
    dict with F, p_value, eta_sq_partial, group_means, group_ns
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"F": float("nan"), "p_value": float("nan"), "eta_sq_partial": float("nan"),
                "group_means": {}, "group_ns": {}}
    labels = group_labels or [str(i) for i in range(len(groups))]

    # If all groups have zero within-group variance, F-test is undefined (0/0).
    # Return NaN explicitly rather than letting f_oneway produce a RuntimeWarning.
    ss_within = sum(float(np.sum((g - g.mean()) ** 2)) for g in groups)
    if ss_within == 0:
        grand_mean = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        return {
            "F": float("nan"), "p_value": float("nan"),
            "eta_sq_partial": float(ss_between / (ss_between + ss_within)) if ss_between > 0 else 0.0,
            "group_means": {l: float(g.mean()) for l, g in zip(labels, groups)},
            "group_ns": {l: len(g) for l, g in zip(labels, groups)},
        }

    # f_oneway can emit "Precision loss … catastrophic cancellation" RuntimeWarning
    # on low-variance ordinal data (e.g. tier ∈ {0, 1, 2}).  Results remain valid.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message=".*Precision loss.*catastrophic cancellation.*")
        f_stat, p_val = sp_stats.f_oneway(*groups)

    # Partial eta-squared: SS_between / (SS_between + SS_within)
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta_sq = ss_between / (ss_between + ss_within)

    return {
        "F": float(f_stat),
        "p_value": float(p_val),
        "eta_sq_partial": float(eta_sq),
        "group_means": {l: float(g.mean()) for l, g in zip(labels, groups)},
        "group_ns": {l: len(g) for l, g in zip(labels, groups)},
    }


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (estimate, lower, upper)."""
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.array([
        statistic_fn(data[rng.randint(0, n, size=n)]) for _ in range(n_boot)
    ])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(statistic_fn(data)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def logistic_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
) -> Dict[str, Any]:
    """Run logistic regression and return coefficients, p-values, odds ratios."""
    import statsmodels.api as sm
    from statsmodels.tools import add_constant

    y = df[target_col].astype(int)

    # Pre-check: target must have both 0 and 1
    if y.nunique() < 2:
        logger.info(
            "Logistic regression skipped: target '%s' has only value(s) %s",
            target_col, [int(v) for v in sorted(y.unique())],
        )
        return {"skipped": True, "reason": f"target has no variation (all {y.iloc[0]})"}

    X = pd.get_dummies(df[predictor_cols], drop_first=True, dtype=float)
    X = add_constant(X)

    # Pre-check: detect quasi-complete separation
    # If any predictor level perfectly predicts the outcome, flag it
    separation_cols = []
    for col in X.columns:
        if col == "const":
            continue
        for level in X[col].unique():
            subset_y = y[X[col] == level]
            if len(subset_y) > 0 and subset_y.nunique() < 2:
                separation_cols.append(col)
                break
    if separation_cols:
        logger.info(
            "Logistic regression: quasi-complete separation detected in %d predictor(s): %s. "
            "Fitting with increased regularisation.",
            len(separation_cols), separation_cols[:5],
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Perfect separation.*")
            warnings.filterwarnings("ignore", message=".*complete separation.*")
            warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed to converge.*")
            if separation_cols:
                # Use regularised (penalised) fit to handle separation
                model = sm.Logit(y, X).fit_regularized(
                    method="l1", alpha=0.1, disp=0, maxiter=100,
                )
                return {
                    "coefficients": model.params.to_dict(),
                    "odds_ratios": np.exp(model.params).to_dict(),
                    "regularized": True,
                    "separation_columns": separation_cols[:10],
                }
            else:
                model = sm.Logit(y, X).fit(disp=0, maxiter=100)
                return {
                    "coefficients": model.params.to_dict(),
                    "p_values": model.pvalues.to_dict(),
                    "odds_ratios": np.exp(model.params).to_dict(),
                    "pseudo_r2": model.prsquared,
                    "aic": model.aic,
                    "converged": model.mle_retvals["converged"],
                }
    except Exception as e:
        logger.warning("Logistic regression failed: %s", e)
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Correlation analysis (Step 3 - Outcome)
# ---------------------------------------------------------------------------

def correlation_analysis(
    decisions_df: pd.DataFrame,
    personas_df: pd.DataFrame,
    dims: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full correlation analysis between personas and remedy decisions.

    Parameters
    ----------
    decisions_df : DataFrame with columns [case_id, model, tier, confidence]
    personas_df  : DataFrame with columns [case_id, model, age, geographic, education, socioeconomic, gender, esl, persona_label]
    dims : List of persona dimensions to analyze (default: ESTIMATE_DIMS)
    """
    if dims is None:
        dims = ESTIMATE_DIMS
    # Filter to dims that actually exist in the personas_df
    available_dims = [d for d in dims if d in personas_df.columns]

    merged = decisions_df.merge(personas_df, on=["case_id", "model"], how="inner", suffixes=("", "_persona"))
    results: Dict[str, Any] = {}

    # 1. Overall association: persona_label × tier
    ct = pd.crosstab(merged["persona_label"], merged["tier"])
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        results["overall_chi2"] = chi_squared_test(ct)

    # 2. Per-dimension association
    for dim in available_dims:
        ct_dim = pd.crosstab(merged[dim], merged["tier"])
        if ct_dim.shape[0] > 1 and ct_dim.shape[1] > 1:
            results[f"{dim}_chi2"] = chi_squared_test(ct_dim)

    # 3. Logistic regression: favorable (tier >= 1) ~ demographics
    merged["favorable"] = (merged["tier"] >= 1).astype(int)
    results["logistic_favorable"] = logistic_regression_analysis(
        merged, "favorable", available_dims
    )

    # 4. Adverse impact ratios by persona
    results["air_by_persona"] = _compute_air_by_persona(merged)

    # 5. Per-dimension favorable rates
    results["rates_by_dimension"] = _rates_by_dimension(merged, available_dims)

    return results


def _compute_air_by_persona(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute AIR for each persona relative to the most favorable."""
    rates = df.groupby("persona_label")["favorable"].mean()
    if rates.empty:
        return {}
    best_rate = rates.max()
    air_dict = {}
    for persona, rate in rates.items():
        air = adverse_impact_ratio(rate, best_rate)
        air_dict[persona] = {
            "rate": float(rate),
            "air": float(air),
            "material": is_material(air),
        }
    return air_dict


def _rates_by_dimension(df: pd.DataFrame, dims: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Favorable outcome rates grouped by each demographic dimension."""
    if dims is None:
        dims = ESTIMATE_DIMS
    result = {}
    for dim in dims:
        if dim in df.columns:
            result[dim] = df.groupby(dim)["favorable"].mean().to_dict()
    return result


# ---------------------------------------------------------------------------
# Causal analysis (Step 5 - Outcome)
# ---------------------------------------------------------------------------

def causal_analysis(
    baseline_decisions: pd.DataFrame,
    persona_decisions: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze causal effects of persona rewrites on remedy decisions.

    Parameters
    ----------
    baseline_decisions : DataFrame [case_id, model, tier]  (original decisions)
    persona_decisions  : DataFrame [case_id, model, variant, tier]  (persona-rewritten decisions)
    """
    results: Dict[str, Any] = {}

    # 1. Favorable outcome rates by persona variant
    persona_decisions["favorable"] = (persona_decisions["tier"] >= 1).astype(int)
    rates = persona_decisions.groupby("variant")["favorable"].mean()
    results["rates_by_persona"] = rates.to_dict()

    # 2. Within-complaint variance (Friedman test)
    try:
        pivot = persona_decisions.pivot_table(
            index="case_id", columns="variant", values="tier", aggfunc="first"
        ).dropna()
        if pivot.shape[1] >= 3 and pivot.shape[0] >= 5:
            # Skip if no variation across variants (Friedman divides by zero)
            col_means = pivot.mean(axis=0)
            if col_means.nunique() > 1:
                groups = [pivot[col].values for col in pivot.columns]
                results["friedman"] = friedman_test(*groups)
            else:
                logger.info("Friedman test skipped: no variation in tier across persona variants")
    except Exception as e:
        logger.warning("Friedman test failed: %s", e)

    # 3. Pairwise McNemar tests (sample)
    results["pairwise_mcnemar"] = _pairwise_mcnemar(persona_decisions)

    # 4. Effect sizes with CIs
    results["effect_sizes"] = _causal_effect_sizes(persona_decisions)

    return results


def _pairwise_mcnemar(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Pairwise McNemar tests between persona variants."""
    df["favorable"] = (df["tier"] >= 1).astype(int)
    variants = sorted(df["variant"].unique())
    results = []

    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            v1, v2 = variants[i], variants[j]
            merged = df[df["variant"] == v1][["case_id", "favorable"]].merge(
                df[df["variant"] == v2][["case_id", "favorable"]],
                on="case_id", suffixes=("_1", "_2"),
            )
            if len(merged) < 5:
                continue
            a = merged["favorable_1"].values
            b = merged["favorable_2"].values
            table = np.array([
                [(a & b).sum(), (a & ~b).sum()],
                [(~a & b).sum(), (~a & ~b).sum()],
            ])
            try:
                test_result = mcnemar_test(table)
                results.append({
                    "persona_1": v1,
                    "persona_2": v2,
                    "p_value": test_result["p_value"],
                    "rate_1": float(a.mean()),
                    "rate_2": float(b.mean()),
                })
            except Exception:
                pass

    # FDR correction
    if results:
        p_vals = [r["p_value"] for r in results]
        adj = fdr_correction(p_vals)
        for r, p_adj in zip(results, adj):
            r["p_adjusted"] = float(p_adj)
            r["significant"] = float(p_adj) < 0.05

    return results


def _causal_effect_sizes(df: pd.DataFrame) -> Dict[str, Any]:
    """Effect sizes for each persona variant relative to baseline mean."""
    df["favorable"] = (df["tier"] >= 1).astype(int)
    overall_rate = df["favorable"].mean()
    effects = {}
    for variant in df["variant"].unique():
        subset = df[df["variant"] == variant]["favorable"].values
        if len(subset) < 5:
            continue
        est, lo, hi = bootstrap_ci(subset)
        effects[variant] = {
            "rate": est,
            "ci_lower": lo,
            "ci_upper": hi,
            "diff_from_mean": est - overall_rate,
        }
    return effects


# ---------------------------------------------------------------------------
# Mitigation analysis (Step 7 - Outcome)
# ---------------------------------------------------------------------------

def mitigation_analysis(
    original_decisions: pd.DataFrame,
    neutralized_decisions: pd.DataFrame,
    personas_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze effectiveness of neutralization in reducing outcome bias.

    Parameters
    ----------
    original_decisions   : DataFrame [case_id, model, tier, confidence]
    neutralized_decisions: DataFrame [case_id, model, tier, confidence]
    personas_df          : DataFrame [case_id, model, persona_label, age, ...]
    """
    results: Dict[str, Any] = {}

    # 1. Pre/post McNemar
    merged = original_decisions[["case_id", "model", "tier"]].merge(
        neutralized_decisions[["case_id", "model", "tier"]],
        on=["case_id", "model"],
        suffixes=("_orig", "_neut"),
    )
    merged["fav_orig"] = (merged["tier_orig"] >= 1).astype(int)
    merged["fav_neut"] = (merged["tier_neut"] >= 1).astype(int)

    a = merged["fav_orig"].values
    b = merged["fav_neut"].values
    table = np.array([
        [(a & b).sum(), (a & ~b).sum()],
        [(~a & b).sum(), (~a & ~b).sum()],
    ])
    try:
        results["mcnemar_pre_post"] = mcnemar_test(table)
    except Exception as e:
        logger.warning("McNemar pre/post failed: %s", e)

    # 2. Discrimination reduction – compare variance by persona
    orig_merged = original_decisions.merge(personas_df, on=["case_id", "model"], suffixes=("", "_persona"))
    neut_merged = neutralized_decisions.merge(personas_df, on=["case_id", "model"], suffixes=("", "_persona"))

    orig_merged["favorable"] = (orig_merged["tier"] >= 1).astype(int)
    neut_merged["favorable"] = (neut_merged["tier"] >= 1).astype(int)

    orig_rates = orig_merged.groupby("persona_label")["favorable"].mean()
    neut_rates = neut_merged.groupby("persona_label")["favorable"].mean()

    if len(orig_rates) > 1 and len(neut_rates) > 1:
        results["variance_orig"] = float(orig_rates.var())
        results["variance_neut"] = float(neut_rates.var())
        if results["variance_orig"] > 0:
            results["variance_reduction"] = 1 - results["variance_neut"] / results["variance_orig"]
        else:
            results["variance_reduction"] = 0.0

    # 3. Residual bias – repeat correlation on neutralized
    results["residual_correlation"] = correlation_analysis(neutralized_decisions, personas_df)

    # 4. Mitigation effectiveness per persona
    if not orig_rates.empty and not neut_rates.empty:
        best_orig = orig_rates.max()
        best_neut = neut_rates.max()
        orig_air = {p: adverse_impact_ratio(r, best_orig) for p, r in orig_rates.items()}
        neut_air = {p: adverse_impact_ratio(r, best_neut) for p, r in neut_rates.items()}

        mitigation = {}
        for p in orig_air:
            if p in neut_air and orig_air[p] < 1.0:
                orig_bias = 1.0 - orig_air[p]
                neut_bias = 1.0 - neut_air.get(p, 1.0)
                reduction = (orig_bias - neut_bias) / orig_bias if orig_bias > 0 else 0.0
                mitigation[p] = {
                    "orig_air": orig_air[p],
                    "neut_air": neut_air.get(p, 1.0),
                    "bias_reduction_pct": reduction,
                }
        results["mitigation_by_persona"] = mitigation

    # 5. Tier change summary
    results["tier_changes"] = _tier_change_summary(merged)

    return results


def _tier_change_summary(merged: pd.DataFrame) -> Dict[str, int]:
    """Summarize how tiers changed after neutralization."""
    changes = {}
    for _, row in merged.iterrows():
        key = f"{int(row['tier_orig'])}→{int(row['tier_neut'])}"
        changes[key] = changes.get(key, 0) + 1
    return changes
