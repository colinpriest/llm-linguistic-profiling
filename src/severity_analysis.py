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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.statistics import bootstrap_ci, chi_squared_test, fdr_correction, one_way_anova

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ["low", "medium", "high"]
SEVERITY_METRICS = ["monetary_severity", "issue_severity"]
SEVERITY_ORDINAL: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}

_MIN_CELL_SIZE = 30  # flag cells with fewer observations


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
        return 0.0
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def _safe_div(a: float, b: float) -> float:
    """Division that returns inf when denominator is zero."""
    if b == 0:
        return float("inf") if a > 0 else (float("-inf") if a < 0 else float("nan"))
    return a / b


def _cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cohen's kappa for two categorical arrays (manual implementation)."""
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return float("nan")
    # Build confusion matrix
    k = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
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
        "thresholds": {"low_max": 100, "medium_max": 1000},
    }

    # --- issue_severity ---
    i_sev = complaints_df["issue_severity"]
    i_dist = i_sev.value_counts()
    result["issue_severity"] = {
        "coverage": float(i_sev.isin(SEVERITY_LEVELS).mean()) if n else 0.0,
        "distribution": {l: int(i_dist.get(l, 0)) for l in SEVERITY_LEVELS},
    }

    # Breakdown by product/issue within each severity level
    for metric, col in [("issue_severity", "issue_severity")]:
        by_product: Dict[str, Dict[str, int]] = {}
        by_issue: Dict[str, Dict[str, int]] = {}
        top_combos: Dict[str, list] = {}
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

        result["agreement"] = {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "kappa": float(kappa),
            "agreement_rate": agreement_rate,
            "cross_tab": ct.to_dict(),
            "cross_tab_pct": (ct / ct.sum().sum() * 100).round(1).to_dict(),
        }
    else:
        result["agreement"] = {
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "kappa": float("nan"),
            "agreement_rate": float("nan"),
            "cross_tab": {},
            "cross_tab_pct": {},
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
        result["chi2_favorable"] = {"chi2": float("nan"), "p_value": float("nan"), "cramers_v": float("nan")}

    # Spearman trend
    ord_sev = _ordinal(df[severity_metric]).values
    tiers = df["tier"].values
    if len(set(ord_sev)) >= 2:
        r, p = sp_stats.spearmanr(ord_sev, tiers)
        result["trend"] = {"spearman_r": float(r), "p_value": float(p)}
    else:
        result["trend"] = {"spearman_r": float("nan"), "p_value": float("nan")}

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
) -> Dict[str, Any]:
    """Compute bias metrics at each severity level.

    Bias = spread in mean outcome across personas.
    """
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ]
    if persona_col not in df.columns or len(df) < 10:
        return {}

    result: Dict[str, Any] = {}
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]
        if len(sub) < 5:
            result[level] = {"n": len(sub), "mean_bias": float("nan"),
                             "sd_across_personas": float("nan"), "range": float("nan"),
                             "cv": float("nan")}
            continue

        persona_means = sub.groupby(persona_col)["tier"].mean()
        if len(persona_means) < 2:
            result[level] = {"n": len(sub), "mean_bias": 0.0,
                             "sd_across_personas": 0.0, "range": 0.0, "cv": 0.0}
            continue

        sd = float(persona_means.std(ddof=1))
        rng = float(persona_means.max() - persona_means.min())
        mean_val = float(persona_means.mean())
        cv = _safe_div(sd, mean_val) if mean_val != 0 else float("nan")

        result[level] = {
            "n": len(sub),
            "mean_bias": rng,
            "sd_across_personas": sd,
            "range": rng,
            "cv": float(cv),
            "persona_means": persona_means.to_dict(),
        }

    return result


def two_way_anova_persona_severity(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
) -> Dict[str, Any]:
    """Two-way ANOVA: tier ~ persona + severity + persona×severity.

    Uses statsmodels Type II SS.
    """
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ].copy()
    if persona_col not in df.columns or len(df) < 30:
        return {"error": "insufficient data"}

    df = df.rename(columns={severity_metric: "severity", persona_col: "persona"})

    # Need at least 2 levels in each factor
    if df["persona"].nunique() < 2 or df["severity"].nunique() < 2:
        return {"error": "insufficient factor levels"}

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols("tier ~ C(persona) + C(severity) + C(persona):C(severity)", data=df).fit()
            table = anova_lm(fit, typ=2)

        total_ss = table["sum_sq"].sum()
        result: Dict[str, Any] = {}
        for source, row_label in [
            ("persona_main", "C(persona)"),
            ("severity_main", "C(severity)"),
            ("interaction", "C(persona):C(severity)"),
            ("residual", "Residual"),
        ]:
            if row_label in table.index:
                ss = float(table.loc[row_label, "sum_sq"])
                df_val = float(table.loc[row_label, "df"])
                f_val = float(table.loc[row_label, "F"]) if "F" in table.columns else float("nan")
                p_val = float(table.loc[row_label, "PR(>F)"]) if "PR(>F)" in table.columns else float("nan")
                # Partial eta-squared = SS_effect / (SS_effect + SS_residual)
                ss_resid = float(table.loc["Residual", "sum_sq"]) if "Residual" in table.index else 0
                eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                result[source] = {
                    "SS": ss, "df": df_val, "F": f_val,
                    "p_value": p_val if np.isfinite(p_val) else float("nan"),
                    "eta_sq_partial": eta_sq,
                }
            else:
                result[source] = {"SS": 0, "df": 0, "F": float("nan"),
                                  "p_value": float("nan"), "eta_sq_partial": 0.0}

        return result

    except Exception as e:
        logger.warning("Two-way ANOVA failed: %s", e)
        return {"error": str(e)}


def simple_effects_by_severity(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    model: str,
    persona_col: str = "persona_label",
) -> Dict[str, Any]:
    """One-way ANOVA of persona effect at each severity level."""
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ]
    if persona_col not in df.columns:
        return {}

    result: Dict[str, Any] = {}
    for level in SEVERITY_LEVELS:
        sub = df[df[severity_metric] == level]
        personas = sub[persona_col].unique()
        if len(personas) < 2 or len(sub) < 10:
            result[level] = {"eta_sq_partial": float("nan"),
                             "largest_pairwise_d": float("nan"),
                             "n_significant": 0, "n": len(sub)}
            continue

        # One-way ANOVA
        groups = [sub.loc[sub[persona_col] == p, "tier"].values for p in personas]
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

        result[level] = {
            "eta_sq_partial": anova_res.get("eta_sq_partial", float("nan")),
            "F": anova_res.get("F", float("nan")),
            "p_value": anova_res.get("p_value", float("nan")),
            "largest_pairwise_d": max_d,
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
) -> Dict[str, Any]:
    """Amplification ratio = bias_high / bias_low with bootstrap CI.

    Bias is measured as SD of mean tier across personas.
    """
    df = decisions_df[
        (decisions_df["model"] == model)
        & (decisions_df[severity_metric].isin(SEVERITY_LEVELS))
    ].copy()
    if persona_col not in df.columns or len(df) < 30:
        return {"point": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    def _bias_at_level(data: pd.DataFrame, level: str) -> float:
        sub = data[data[severity_metric] == level]
        if len(sub) < 5:
            return float("nan")
        pm = sub.groupby(persona_col)["tier"].mean()
        return float(pm.std(ddof=1)) if len(pm) >= 2 else 0.0

    bias_low = _bias_at_level(df, "low")
    bias_high = _bias_at_level(df, "high")
    point = _safe_div(bias_high, bias_low)

    # Bootstrap CI: resample at complaint (case_id) level
    rng = np.random.RandomState(seed)
    case_ids = df["case_id"].unique()
    boot_ratios = []
    for _ in range(n_boot):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        boot_df = df[df["case_id"].isin(sample_ids)]
        b_low = _bias_at_level(boot_df, "low")
        b_high = _bias_at_level(boot_df, "high")
        if np.isfinite(b_low) and np.isfinite(b_high) and b_low > 0:
            boot_ratios.append(b_high / b_low)

    if len(boot_ratios) >= 10:
        ci_lower = float(np.percentile(boot_ratios, 2.5))
        ci_upper = float(np.percentile(boot_ratios, 97.5))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    return {
        "point": float(point) if np.isfinite(point) else float("nan"),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bias_low": bias_low,
        "bias_high": bias_high,
    }


# ---------------------------------------------------------------------------
# Section 4: Three-Way Analysis
# ---------------------------------------------------------------------------

def three_way_anova(
    decisions_df: pd.DataFrame,
    severity_metric: str,
    persona_col: str = "persona_label",
) -> Dict[str, Any]:
    """Three-way ANOVA: tier ~ model * persona * severity."""
    df = decisions_df[decisions_df[severity_metric].isin(SEVERITY_LEVELS)].copy()
    if persona_col not in df.columns or len(df) < 50:
        return {"error": "insufficient data"}
    if df["model"].nunique() < 2:
        return {"error": "need multiple models for three-way ANOVA"}

    df = df.rename(columns={severity_metric: "severity", persona_col: "persona"})

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(
                "tier ~ C(model) * C(persona) * C(severity)", data=df
            ).fit()
            table = anova_lm(fit, typ=2)

        ss_resid = float(table.loc["Residual", "sum_sq"]) if "Residual" in table.index else 0
        result: Dict[str, Any] = {}
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
                ss = float(table.loc[row_label, "sum_sq"])
                df_val = float(table.loc[row_label, "df"])
                f_val = float(table.loc[row_label, "F"]) if "F" in table.columns else float("nan")
                p_val = float(table.loc[row_label, "PR(>F)"]) if "PR(>F)" in table.columns else float("nan")
                eta_sq = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else 0.0
                result[key] = {
                    "F": f_val if np.isfinite(f_val) else float("nan"),
                    "p_value": p_val if np.isfinite(p_val) else float("nan"),
                    "eta_sq_partial": eta_sq,
                }
            else:
                result[key] = {"F": float("nan"), "p_value": float("nan"), "eta_sq_partial": 0.0}

        return result

    except Exception as e:
        logger.warning("Three-way ANOVA failed: %s", e)
        return {"error": str(e)}


def model_specific_amplification(
    decisions_df: pd.DataFrame,
    models: List[str],
    persona_col: str = "persona_label",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Amplification ratio per model per severity metric."""
    result: Dict[str, Any] = {}
    for model in models:
        model_res: Dict[str, Any] = {}
        for metric in SEVERITY_METRICS:
            model_res[f"{metric}_amplification"] = amplification_ratio(
                decisions_df, metric, model,
                persona_col=persona_col, n_boot=n_boot, seed=seed,
            )
        result[model] = model_res
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
    disp_low = rates_by_severity.get("low", {}).get("disparity", float("nan"))
    disp_high = rates_by_severity.get("high", {}).get("disparity", float("nan"))
    if np.isfinite(disp_low) and np.isfinite(disp_high):
        result["process_amplification"] = {
            "point": _safe_div(disp_high, disp_low),
            "disparity_low": disp_low,
            "disparity_high": disp_high,
        }
    else:
        result["process_amplification"] = {
            "point": float("nan"),
            "disparity_low": disp_low if np.isfinite(disp_low) else float("nan"),
            "disparity_high": disp_high if np.isfinite(disp_high) else float("nan"),
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

    def _get_p(data: Dict, *keys: str) -> float:
        for k in keys:
            data = data.get(k, {})
        return data if isinstance(data, (int, float)) else float("nan")

    # For each model present in both metrics
    monetary_me = main_effects.get("monetary_severity", {})
    issue_me = main_effects.get("issue_severity", {})
    all_models = set(monetary_me.keys()) | set(issue_me.keys())

    for model in sorted(all_models):
        m_anova_p = _get_p(monetary_me, model, "anova", "p_value")
        i_anova_p = _get_p(issue_me, model, "anova", "p_value")
        m_sig = m_anova_p < 0.05 if np.isfinite(m_anova_p) else False
        i_sig = i_anova_p < 0.05 if np.isfinite(i_anova_p) else False
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
        m_sig = m_p < 0.05 if np.isfinite(m_p) else False
        i_sig = i_p < 0.05 if np.isfinite(i_p) else False
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
) -> Dict[str, Any]:
    """Compare severity amplification before and after bias mitigation.

    Reuses existing analysis functions on both baseline and neutralized
    DataFrames to compute deltas and a summary verdict.

    Returns
    -------
    Dict with keys: baseline, neutralized, comparison.
    """
    # --- Baseline metrics (recompute locally for self-contained comparison) ---
    bias_base = bias_by_severity(baseline_df, severity_metric, model, persona_col)
    amp_base = amplification_ratio(
        baseline_df, severity_metric, model, persona_col, n_boot, seed,
    )
    tw_base = two_way_anova_persona_severity(baseline_df, severity_metric, model, persona_col)

    # --- Neutralized metrics ---
    bias_neut = bias_by_severity(neutralized_df, severity_metric, model, persona_col)
    amp_neut = amplification_ratio(
        neutralized_df, severity_metric, model, persona_col, n_boot, seed,
    )
    tw_neut = two_way_anova_persona_severity(neutralized_df, severity_metric, model, persona_col)
    se_neut = simple_effects_by_severity(neutralized_df, severity_metric, model, persona_col)

    # --- Compute deltas ---
    base_ratio = amp_base.get("point", float("nan"))
    neut_ratio = amp_neut.get("point", float("nan"))

    if np.isfinite(base_ratio) and np.isfinite(neut_ratio) and base_ratio > 0:
        amp_change = neut_ratio - base_ratio
        amp_pct = amp_change / base_ratio * 100
    else:
        amp_change = float("nan")
        amp_pct = float("nan")

    # Per-level bias SD changes
    bias_sd_change: Dict[str, float] = {}
    for level in SEVERITY_LEVELS:
        b_sd = bias_base.get(level, {}).get("sd_across_personas", float("nan"))
        n_sd = bias_neut.get(level, {}).get("sd_across_personas", float("nan"))
        if np.isfinite(b_sd) and np.isfinite(n_sd):
            bias_sd_change[level] = n_sd - b_sd
        else:
            bias_sd_change[level] = float("nan")

    # Interaction p-values
    base_int_p = tw_base.get("interaction", {}).get("p_value", float("nan"))
    neut_int_p = tw_neut.get("interaction", {}).get("p_value", float("nan"))
    interaction_weakened = (
        np.isfinite(base_int_p) and np.isfinite(neut_int_p)
        and neut_int_p > base_int_p
    )

    # Verdict
    if not np.isfinite(base_ratio) or not np.isfinite(neut_ratio):
        verdict = "insufficient_data"
    else:
        base_ci_lo = amp_base.get("ci_lower", float("nan"))
        base_ci_hi = amp_base.get("ci_upper", float("nan"))
        neut_ci_lo = amp_neut.get("ci_lower", float("nan"))
        neut_ci_hi = amp_neut.get("ci_upper", float("nan"))
        # "eliminated" if neutralized CI includes 1.0 but baseline CI didn't
        base_includes_1 = (np.isfinite(base_ci_lo) and np.isfinite(base_ci_hi)
                           and base_ci_lo <= 1.0 <= base_ci_hi)
        neut_includes_1 = (np.isfinite(neut_ci_lo) and np.isfinite(neut_ci_hi)
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

    Returns
    -------
    Dict with keys: descriptives, coverage, main_effects, interactions,
    three_way, model_comparison, process, consistency, mitigation_severity.
    """
    logger.info("Running severity amplification analysis...")

    # 1. Descriptives
    logger.info("  Section 1: Severity descriptives")
    descriptives = severity_descriptives(complaints_df)
    coverage = severity_coverage(
        complaints_df[["case_id", "monetary_severity", "issue_severity"]],
        personas_df,
    )
    descriptives["coverage"] = coverage

    # 2. Main effects (uses baseline decisions)
    logger.info("  Section 2: Severity main effects")
    main_effects = severity_main_effects_all(baseline_df, models)

    # 3. Interaction analysis (uses baseline merged with personas)
    logger.info("  Section 3: Severity × persona interaction")
    interactions: Dict[str, Any] = {}
    for metric in SEVERITY_METRICS:
        metric_res: Dict[str, Any] = {}
        for model in models:
            model_int: Dict[str, Any] = {}
            model_int["bias"] = bias_by_severity(
                baseline_df, metric, model, persona_col="persona_label",
            )
            model_int["two_way"] = two_way_anova_persona_severity(
                baseline_df, metric, model, persona_col="persona_label",
            )
            model_int["simple_effects"] = simple_effects_by_severity(
                baseline_df, metric, model, persona_col="persona_label",
            )
            model_int["amplification"] = amplification_ratio(
                baseline_df, metric, model,
                persona_col="persona_label", n_boot=n_boot, seed=seed,
            )
            metric_res[model] = model_int
        interactions[metric] = metric_res

    # 4. Three-way analysis
    logger.info("  Section 4: Three-way analysis")
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

    # 5. Process fairness
    logger.info("  Section 5: Process fairness × severity")
    process: Dict[str, Any] = {}
    for metric in SEVERITY_METRICS:
        process[metric] = {}
        for model in models:
            process[metric][model] = process_severity_analysis(
                baseline_df, metric, model,
                persona_col="persona_label", n_boot=n_boot, seed=seed,
            )

    # 6. Consistency
    logger.info("  Section 6: Consistency check")
    consist = consistency_check(main_effects, interactions)

    # 7. Mitigation × severity (optional)
    mitigation_severity: Dict[str, Any] = {}
    if neutralized_df is not None and len(neutralized_df) > 0:
        logger.info("  Section 7: Mitigation × severity comparison")
        for metric in SEVERITY_METRICS:
            mitigation_severity[metric] = {}
            for model in models:
                try:
                    mitigation_severity[metric][model] = mitigation_severity_comparison(
                        baseline_df, neutralized_df, metric, model,
                        persona_col="persona_label", n_boot=n_boot, seed=seed,
                    )
                except Exception as exc:
                    logger.warning("  Mitigation severity failed for %s/%s: %s", metric, model, exc)
                    mitigation_severity[metric][model] = {"comparison": {"verdict": "insufficient_data"}}

    logger.info("Severity amplification analysis complete.")

    return {
        "descriptives": descriptives,
        "main_effects": main_effects,
        "interactions": interactions,
        "three_way": {
            "anova": three_way,
            "model_amplification": model_amp,
            "worst_case": worst,
        },
        "model_comparison": model_amp,
        "process": process,
        "consistency": consist,
        "mitigation_severity": mitigation_severity,
    }
