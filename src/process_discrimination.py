"""Process discrimination analysis – procedural burden disparities."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.statistics import (
    adverse_impact_ratio,
    bootstrap_ci,
    chi_squared_test,
    fdr_correction,
    logistic_regression_analysis,
    mcnemar_test,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-specific AIR (inverted – higher burden = worse)
# ---------------------------------------------------------------------------

def process_adverse_impact_ratio(rate_a: float, rate_b: float) -> float:
    """Process AIR; rate_b has the LOWEST info request rate (best treatment)."""
    if rate_b == 0:
        return float("inf") if rate_a > 0 else 1.0
    return rate_a / rate_b


def is_process_material(air: float, threshold: float = 1.25) -> bool:
    return air > threshold


def process_interpretation_flag(p_adj: float, air: float, threshold: float = 1.25) -> str:
    sig = p_adj < 0.05
    mat = air > threshold
    if sig and mat:
        return "BOTH"
    if sig:
        return "SIG_ONLY"
    if mat:
        return "MAT_ONLY"
    return "NONE"


# ---------------------------------------------------------------------------
# Information request rate
# ---------------------------------------------------------------------------

def information_request_rate(decisions: pd.DataFrame) -> float:
    need_info = (decisions["confidence"] == "need_more_info").sum()
    return need_info / len(decisions) if len(decisions) > 0 else 0.0


# ---------------------------------------------------------------------------
# Correlation analysis (Step 3 – Process)
# ---------------------------------------------------------------------------

def process_correlation_analysis(
    decisions_df: pd.DataFrame,
    personas_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Assess correlation between persona and information request burden.

    Parameters
    ----------
    decisions_df : DataFrame [case_id, model, tier, confidence, information_needed]
    personas_df  : DataFrame [case_id, model, age, geographic, education, socioeconomic, persona_label]
    """
    merged = decisions_df.merge(personas_df, on=["case_id", "model"], how="inner", suffixes=("", "_persona"))
    merged["need_info"] = (merged["confidence"] == "need_more_info").astype(int)
    results: Dict[str, Any] = {}

    # 1. Overall association: persona_label × need_more_info
    ct = pd.crosstab(merged["persona_label"], merged["need_info"])
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        results["overall_chi2"] = chi_squared_test(ct)

    # 2. Per-dimension
    for dim in ["age", "geographic", "education", "socioeconomic"]:
        ct_dim = pd.crosstab(merged[dim], merged["need_info"])
        if ct_dim.shape[0] > 1 and ct_dim.shape[1] > 1:
            results[f"{dim}_chi2"] = chi_squared_test(ct_dim)

    # 3. Logistic regression
    results["logistic_need_info"] = logistic_regression_analysis(
        merged, "need_info", ["age", "geographic", "education", "socioeconomic"]
    )

    # 4. Process AIR by persona
    rates = merged.groupby("persona_label")["need_info"].mean()
    if not rates.empty:
        min_rate = rates.min()
        air_dict = {}
        for persona, rate in rates.items():
            air = process_adverse_impact_ratio(rate, min_rate)
            air_dict[persona] = {
                "rate": float(rate),
                "air": float(air),
                "material": is_process_material(air),
            }
        results["process_air_by_persona"] = air_dict

    # 5. Per-dimension info request rates
    results["info_rates_by_dimension"] = {
        dim: merged.groupby(dim)["need_info"].mean().to_dict()
        for dim in ["age", "geographic", "education", "socioeconomic"]
    }

    return results


# ---------------------------------------------------------------------------
# Causal analysis (Step 5 – Process)
# ---------------------------------------------------------------------------

def process_causal_analysis(
    persona_decisions: pd.DataFrame,
) -> Dict[str, Any]:
    """Causal process discrimination analysis from matched-guise experiment.

    Parameters
    ----------
    persona_decisions : DataFrame [case_id, model, variant, tier, confidence]
    """
    persona_decisions["need_info"] = (
        persona_decisions["confidence"] == "need_more_info"
    ).astype(int)
    results: Dict[str, Any] = {}

    # 1. Info request rate by persona variant
    rates = persona_decisions.groupby("variant")["need_info"].mean()
    results["info_rates_by_persona"] = rates.to_dict()

    # 2. Cochran's Q test (generalised McNemar for >2 conditions)
    try:
        pivot = persona_decisions.pivot_table(
            index="case_id", columns="variant", values="need_info", aggfunc="first"
        ).dropna()
        if pivot.shape[1] >= 3 and pivot.shape[0] >= 5:
            from statsmodels.stats.contingency_tables import cochrans_q
            result = cochrans_q(pivot.values)
            results["cochrans_q"] = {"statistic": result.statistic, "p_value": result.pvalue}
    except Exception as e:
        logger.warning("Cochran's Q test failed: %s", e)

    # 3. Pairwise McNemar on need_info
    results["pairwise_mcnemar"] = _process_pairwise_mcnemar(persona_decisions)

    # 4. Effect sizes
    results["effect_sizes"] = _process_effect_sizes(persona_decisions)

    return results


def _process_pairwise_mcnemar(df: pd.DataFrame) -> List[Dict[str, Any]]:
    variants = sorted(df["variant"].unique())
    results = []

    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            v1, v2 = variants[i], variants[j]
            m = df[df["variant"] == v1][["case_id", "need_info"]].merge(
                df[df["variant"] == v2][["case_id", "need_info"]],
                on="case_id", suffixes=("_1", "_2"),
            )
            if len(m) < 5:
                continue
            a = m["need_info_1"].values
            b = m["need_info_2"].values
            table = np.array([
                [(a & b).sum(), (a & ~b).sum()],
                [(~a & b).sum(), (~a & ~b).sum()],
            ])
            try:
                test = mcnemar_test(table)
                results.append({
                    "persona_1": v1, "persona_2": v2,
                    "p_value": test["p_value"],
                    "rate_1": float(a.mean()), "rate_2": float(b.mean()),
                })
            except Exception:
                pass

    if results:
        p_vals = [r["p_value"] for r in results]
        adj = fdr_correction(p_vals)
        for r, p_adj in zip(results, adj):
            r["p_adjusted"] = float(p_adj)
            r["significant"] = float(p_adj) < 0.05

    return results


def _process_effect_sizes(df: pd.DataFrame) -> Dict[str, Any]:
    overall_rate = df["need_info"].mean()
    effects = {}
    for variant in df["variant"].unique():
        subset = df[df["variant"] == variant]["need_info"].values
        if len(subset) < 5:
            continue
        est, lo, hi = bootstrap_ci(subset)
        effects[variant] = {
            "rate": est, "ci_lower": lo, "ci_upper": hi,
            "diff_from_mean": est - overall_rate,
        }
    return effects


# ---------------------------------------------------------------------------
# Mitigation analysis (Step 7 – Process)
# ---------------------------------------------------------------------------

def process_mitigation_analysis(
    original_decisions: pd.DataFrame,
    neutralized_decisions: pd.DataFrame,
    personas_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Assess neutralization effectiveness for process discrimination."""
    results: Dict[str, Any] = {}

    # 1. Pre/post McNemar on need_more_info
    merged = original_decisions[["case_id", "model", "confidence"]].merge(
        neutralized_decisions[["case_id", "model", "confidence"]],
        on=["case_id", "model"], suffixes=("_orig", "_neut"),
    )
    merged["info_orig"] = (merged["confidence_orig"] == "need_more_info").astype(int)
    merged["info_neut"] = (merged["confidence_neut"] == "need_more_info").astype(int)

    a = merged["info_orig"].values
    b = merged["info_neut"].values
    table = np.array([
        [(a & b).sum(), (a & ~b).sum()],
        [(~a & b).sum(), (~a & ~b).sum()],
    ])
    try:
        results["mcnemar_pre_post"] = mcnemar_test(table)
    except Exception as e:
        logger.warning("Process McNemar pre/post failed: %s", e)

    # 2. Variance reduction by persona
    orig_m = original_decisions.merge(personas_df, on=["case_id", "model"], suffixes=("", "_persona"))
    neut_m = neutralized_decisions.merge(personas_df, on=["case_id", "model"], suffixes=("", "_persona"))
    orig_m["need_info"] = (orig_m["confidence"] == "need_more_info").astype(int)
    neut_m["need_info"] = (neut_m["confidence"] == "need_more_info").astype(int)

    orig_rates = orig_m.groupby("persona_label")["need_info"].mean()
    neut_rates = neut_m.groupby("persona_label")["need_info"].mean()

    if len(orig_rates) > 1 and len(neut_rates) > 1:
        results["variance_orig"] = float(orig_rates.var())
        results["variance_neut"] = float(neut_rates.var())
        if results["variance_orig"] > 0:
            results["variance_reduction"] = 1 - results["variance_neut"] / results["variance_orig"]

    # 3. Residual process bias
    results["residual_correlation"] = process_correlation_analysis(neutralized_decisions, personas_df)

    # 4. Process mitigation by persona
    if not orig_rates.empty and not neut_rates.empty:
        min_orig = orig_rates.min()
        min_neut = neut_rates.min()
        mitigation = {}
        for p in orig_rates.index:
            orig_air = process_adverse_impact_ratio(orig_rates[p], min_orig)
            neut_air_val = process_adverse_impact_ratio(neut_rates.get(p, 0), min_neut) if p in neut_rates.index else 1.0
            if orig_air > 1.0:
                orig_bias = orig_air - 1.0
                neut_bias = max(neut_air_val - 1.0, 0.0)
                reduction = (orig_bias - neut_bias) / orig_bias if orig_bias > 0 else 0.0
                mitigation[p] = {
                    "orig_air": orig_air,
                    "neut_air": neut_air_val,
                    "bias_reduction_pct": reduction,
                }
        results["mitigation_by_persona"] = mitigation

    return results


# ---------------------------------------------------------------------------
# Information request content analysis
# ---------------------------------------------------------------------------

def info_request_content_analysis(
    decisions_df: pd.DataFrame,
    personas_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze the content of information requests by persona."""
    merged = decisions_df.merge(personas_df, on=["case_id", "model"], how="inner", suffixes=("", "_persona"))
    need_info = merged[merged["confidence"] == "need_more_info"].copy()

    if len(need_info) == 0:
        return {"n_requests": 0}

    results: Dict[str, Any] = {"n_requests": len(need_info)}

    # 1. Request length by persona
    if "information_needed" in need_info.columns:
        need_info["request_length"] = need_info["information_needed"].fillna("").str.len()
        results["mean_length_by_persona"] = (
            need_info.groupby("persona_label")["request_length"].mean().to_dict()
        )

        # ANOVA on request length by persona dimension
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            groups = [
                g["request_length"].values
                for _, g in need_info.groupby(dim)
                if len(g) >= 2
            ]
            if len(groups) >= 2:
                try:
                    f_stat, p_val = sp_stats.f_oneway(*groups)
                    results[f"length_anova_{dim}"] = {"f_statistic": f_stat, "p_value": p_val}
                except Exception:
                    pass

        # Word frequency for word clouds
        all_words: Dict[str, Dict[str, int]] = {}
        for persona, group in need_info.groupby("persona_label"):
            words: Dict[str, int] = {}
            for text in group["information_needed"].fillna(""):
                for w in str(text).lower().split():
                    if len(w) > 3:
                        words[w] = words.get(w, 0) + 1
            all_words[persona] = words
        results["word_frequencies_by_persona"] = all_words

    return results
