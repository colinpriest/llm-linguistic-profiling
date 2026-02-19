"""Statistical analysis module for disparate impact (outcome) analysis."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.schemas import ESTIMATE_DIMS, CAUSAL_DIMS, parse_persona_label
from src.estimable import NE, NEValue, is_estimable

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
    arr = np.array(p_values, dtype=float)
    if len(arr) == 0:
        return arr

    # Filter out NaN p-values, correct the valid ones, then map back
    valid_mask = ~np.isnan(arr)
    valid_p = arr[valid_mask]
    if len(valid_p) == 0:
        return arr  # all NaN — return as-is

    try:
        from scipy.stats import false_discovery_control
    except (ImportError, AttributeError):
        from statsmodels.stats.multitest import multipletests
        corrected_valid = multipletests(valid_p, method="fdr_bh")[1]
    else:
        corrected_valid = false_discovery_control(valid_p, method="bh")

    result = np.full_like(arr, np.nan)
    result[valid_mask] = corrected_valid
    return result


def fdr_correction_with_tracking(
    p_values: List[float],
    family_label: str = "",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """BH FDR correction with metadata for reporting.

    Returns dict with: p_adjusted, n_tests, n_surviving, family_label, method.
    """
    arr = np.array(p_values, dtype=float)
    if len(arr) == 0:
        return {"p_adjusted": arr, "n_tests": 0, "n_surviving": 0,
                "family_label": family_label, "method": "Benjamini-Hochberg"}
    adjusted = fdr_correction(list(arr))
    return {
        "p_adjusted": adjusted,
        "n_tests": len(arr),
        "n_surviving": int((adjusted < alpha).sum()),
        "family_label": family_label,
        "method": "Benjamini-Hochberg",
        "alpha": alpha,
    }


def multiple_comparisons_summary(
    correction_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize all FDR correction families for the dashboard.

    Parameters
    ----------
    correction_records : List of dicts with keys n_tests, n_surviving, family_label.

    Returns
    -------
    Dict with keys: families, total_tests, total_surviving, method, method_description.
    """
    families = []
    total_tests = 0
    total_surviving = 0
    for rec in correction_records:
        n = rec.get("n_tests", 0)
        s = rec.get("n_surviving", 0)
        total_tests += n
        total_surviving += s
        families.append({
            "family": rec.get("family_label", "Unknown"),
            "n_tests": n,
            "n_surviving": s,
            "survival_rate": s / n if n > 0 else 0.0,
        })
    return {
        "families": families,
        "total_tests": total_tests,
        "total_surviving": total_surviving,
        "method": "Benjamini-Hochberg (BH)",
        "method_description": (
            "The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) "
            "at 5%. Unlike Bonferroni, which controls the family-wise error rate "
            "(probability of any false positive), BH controls the expected proportion "
            "of false positives among rejected hypotheses. Each test family is "
            "corrected independently. With 630 pairwise persona comparisons, "
            "uncorrected testing at alpha=0.05 would yield ~31 false positives by "
            "chance; BH correction accounts for this."
        ),
    }


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
# Effect size interpretation (Cohen 1988)
# ---------------------------------------------------------------------------

def interpret_cohens_d(d: float) -> str:
    """Cohen's d interpretation (Cohen 1988): <0.2 negligible, 0.2 small, 0.5 medium, 0.8 large."""
    d = abs(d)
    if not np.isfinite(d):
        return "N/A"
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def interpret_eta_sq(eta_sq: float) -> str:
    """Partial eta-squared interpretation (Cohen 1988): <0.01 negligible, 0.01 small, 0.06 medium, 0.14 large."""
    if not np.isfinite(eta_sq):
        return "N/A"
    if eta_sq < 0.01:
        return "negligible"
    if eta_sq < 0.06:
        return "small"
    if eta_sq < 0.14:
        return "medium"
    return "large"


def interpret_cramers_v(v: float, df_min: int = 1) -> str:
    """Cramer's V interpretation (Cohen 1988, adjusted for df)."""
    if not np.isfinite(v):
        return "N/A"
    if df_min == 1:
        thresholds = (0.1, 0.3, 0.5)
    elif df_min == 2:
        thresholds = (0.07, 0.21, 0.35)
    else:
        thresholds = (0.06, 0.17, 0.29)
    if v < thresholds[0]:
        return "negligible"
    if v < thresholds[1]:
        return "small"
    if v < thresholds[2]:
        return "medium"
    return "large"


def interpret_effect_size(value: float, metric: str, **kwargs: Any) -> str:
    """Unified effect size interpreter.

    Parameters
    ----------
    value : The effect size value.
    metric : One of ``"cohens_d"``, ``"eta_sq"``, ``"cramers_v"``.
    """
    if not np.isfinite(value):
        return "N/A"
    dispatch = {
        "cohens_d": interpret_cohens_d,
        "eta_sq": interpret_eta_sq,
        "cramers_v": lambda v: interpret_cramers_v(v, kwargs.get("df_min", 1)),
    }
    fn = dispatch.get(metric)
    return fn(value) if fn else "unknown"


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def power_analysis(
    n_complaints: int,
    n_personas: int,
    n_per_cell: Dict[str, int],
    causal_sample: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, Any]:
    """Post-hoc power analysis: minimum detectable effect sizes.

    Parameters
    ----------
    n_complaints : Total complaints (e.g. 5000).
    n_personas : Total persona variants (e.g. 36).
    n_per_cell : Dict mapping dimension name -> minimum cell size.
    causal_sample : Number of complaints in the matched-guise sample.
    alpha : Significance level.
    power : Desired power.

    Returns
    -------
    Dict with keys: matched_guise, between_persona, anova, sample_summary.
    """
    from statsmodels.stats.power import TTestPower, FTestAnovaPower

    result: Dict[str, Any] = {"alpha": alpha, "power": power}

    # 1. Matched-guise (within-complaint paired design)
    tt = TTestPower()
    try:
        min_d_paired: Any = tt.solve_power(
            nobs=causal_sample, alpha=alpha, power=power, alternative="two-sided",
        )
    except Exception:
        min_d_paired = NE.numerical(context="paired power solve")
    result["matched_guise"] = {
        "n_pairs": causal_sample,
        "min_detectable_d": float(min_d_paired) if is_estimable(min_d_paired) else min_d_paired,
        "interpretation": interpret_cohens_d(min_d_paired) if is_estimable(min_d_paired) else "N/A",
        "design": "paired within-complaint",
    }

    # 2. Between-persona (independent groups, per dimension)
    between_results: Dict[str, Any] = {}
    for dim, n_cell in n_per_cell.items():
        if n_cell < 2:
            continue
        try:
            min_d: Any = tt.solve_power(
                nobs=n_cell, alpha=alpha, power=power, alternative="two-sided",
            )
        except Exception:
            min_d = NE.numerical(context=f"between power solve {dim}")
        between_results[dim] = {
            "n_per_cell": n_cell,
            "min_detectable_d": float(min_d) if is_estimable(min_d) else min_d,
            "interpretation": interpret_cohens_d(min_d) if is_estimable(min_d) else "N/A",
        }
    result["between_persona"] = between_results

    # 3. ANOVA power (for k-group comparisons)
    k_map = {"age": 2, "geographic": 3, "education": 2, "socioeconomic": 3}
    anova_results: Dict[str, Any] = {}
    ft = FTestAnovaPower()
    for dim, n_cell in n_per_cell.items():
        k = k_map.get(dim, 2)
        if k < 2:
            continue
        try:
            min_f = ft.solve_power(
                nobs=n_cell, k_groups=k, alpha=alpha, power=power,
            )
            eta_sq = float(min_f ** 2 / (1 + min_f ** 2))
            anova_results[dim] = {
                "k_groups": k,
                "n_per_group": n_cell,
                "min_detectable_f": float(min_f),
                "min_detectable_eta_sq": eta_sq,
                "interpretation": interpret_eta_sq(eta_sq),
            }
        except Exception as e:
            logger.debug("ANOVA power solve failed for %s: %s", dim, e)
    result["anova"] = anova_results

    # 4. Sample summary
    result["sample_summary"] = {
        "total_complaints": n_complaints,
        "n_personas": n_personas,
        "total_persona_decisions": causal_sample * n_personas,
        "causal_sample": causal_sample,
    }

    return result


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def chi_squared_test(
    contingency: pd.DataFrame,
) -> Dict[str, Any]:
    """Chi-squared test of independence on a contingency table."""
    chi2, p, dof, _ = sp_stats.chi2_contingency(contingency.values)
    n = contingency.values.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v,
            "cramers_v_interpretation": interpret_cramers_v(cramers_v, k)}


def fisher_exact_test(table_2x2: np.ndarray) -> Dict[str, Any]:
    """Fisher's exact test for 2×2 tables."""
    odds_ratio, p = sp_stats.fisher_exact(table_2x2)
    return {"odds_ratio": odds_ratio, "p_value": p}


def mcnemar_test(table_2x2: np.ndarray) -> Dict[str, Any]:
    """McNemar's test for paired categorical data."""
    from statsmodels.stats.contingency_tables import mcnemar
    result = mcnemar(table_2x2, exact=(table_2x2[0, 1] + table_2x2[1, 0]) < 25)
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
        ne_groups = NE.insufficient_n(n=len(groups), min_required=2, context="ANOVA groups")
        return {"F": ne_groups, "p_value": ne_groups, "eta_sq_partial": ne_groups,
                "group_means": {}, "group_ns": {}}
    labels = group_labels or [str(i) for i in range(len(groups))]

    # If all groups have zero within-group variance, F-test is undefined (0/0).
    # Return NE explicitly rather than letting f_oneway produce a RuntimeWarning.
    ss_within = sum(float(np.sum((g - g.mean()) ** 2)) for g in groups)
    if ss_within == 0:
        grand_mean = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ne_zerovar = NE.no_variance(context="ANOVA within-group")
        return {
            "F": ne_zerovar, "p_value": ne_zerovar,
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
        "eta_sq_interpretation": interpret_eta_sq(eta_sq),
        "group_means": {l: float(g.mean()) for l, g in zip(labels, groups)},
        "group_ns": {l: len(g) for l, g in zip(labels, groups)},
    }


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (estimate, lower, upper)."""
    if len(data) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.array([
        statistic_fn(data[rng.integers(0, n, size=n)]) for _ in range(n_boot)
    ])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(statistic_fn(data)), float(lo), float(hi)


def bca_interval(
    theta_hat: float,
    theta_star: np.ndarray,
    theta_jack: np.ndarray,
    alpha: float = 0.05,
    eps: float = 1e-6,
) -> Tuple[float, float, str]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Uses a two-stage fallback chain:

    1. **BCa** — standard bias-corrected accelerated interval.
    2. **Symmetric percentile** — standard [alpha/2, 1-alpha/2] if BCa is
       degenerate or excludes the point.

    Parameters
    ----------
    theta_hat : float
        Statistic computed on the full sample.
    theta_star : ndarray, shape (B,)
        Bootstrap replicate statistics (same statistic as *theta_hat*).
    theta_jack : ndarray, shape (n,)
        Jackknife leave-one-cluster-out replicate statistics.
    alpha : float
        Significance level (default 0.05 -> 95 % CI).
    eps : float
        Floor / ceiling for adjusted percentiles.

    Returns
    -------
    (ci_lower, ci_upper, method) : tuple of (float, float, str)
        *method* is one of ``"BCa"`` or ``"percentile"``.
    """
    B = len(theta_star)
    if B < 2:
        return (float("nan"), float("nan"), "BCa")

    # --- Bias correction z0 ---
    prop = float(np.mean(theta_star < theta_hat))  # strict <
    prop = np.clip(prop, 1.0 / (2 * B), 1.0 - 1.0 / (2 * B))
    z0 = float(sp_stats.norm.ppf(prop))

    # --- Acceleration a (jackknife) ---
    finite_jack = theta_jack[np.isfinite(theta_jack)]
    if len(finite_jack) < 2:
        a = 0.0
    else:
        theta_dot = float(np.mean(finite_jack))
        u = theta_dot - finite_jack  # influence values
        denom = float(np.sum(u ** 2)) ** 1.5
        a = float(np.sum(u ** 3)) / (6.0 * denom) if abs(denom) > 1e-15 else 0.0

    # --- Adjusted percentiles ---
    z_lo = float(sp_stats.norm.ppf(alpha / 2))
    z_hi = float(sp_stats.norm.ppf(1.0 - alpha / 2))

    def _adj(z: float) -> float:
        numer = z0 + z
        d = 1.0 - a * numer
        if abs(d) < 1e-15:
            return 0.5
        return float(sp_stats.norm.cdf(z0 + numer / d))

    alpha_lo = np.clip(_adj(z_lo), eps, 1.0 - eps)
    alpha_hi = np.clip(_adj(z_hi), eps, 1.0 - eps)

    ci_lo = float(np.percentile(theta_star, 100.0 * alpha_lo))
    ci_hi = float(np.percentile(theta_star, 100.0 * alpha_hi))

    tol = 1e-9
    degenerate = (ci_hi - ci_lo) < tol
    outside = (theta_hat < ci_lo - tol) or (theta_hat > ci_hi + tol)

    if not (degenerate or outside):
        return (ci_lo, ci_hi, "BCa")

    # --- Fallback: symmetric percentile ---
    pct_lo = float(np.percentile(theta_star, 100.0 * alpha / 2))
    pct_hi = float(np.percentile(theta_star, 100.0 * (1.0 - alpha / 2)))
    logger.debug(
        "BCa interval [%.6f, %.6f] %s for theta_hat=%.6f; "
        "falling back to percentile [%.6f, %.6f].",
        ci_lo, ci_hi,
        "degenerate" if degenerate else "excludes point",
        theta_hat, pct_lo, pct_hi,
    )

    return (pct_lo, pct_hi, "percentile")


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def logistic_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
) -> Dict[str, Any]:
    """Run logistic regression with pre-flight diagnostics and targeted fallbacks.

    Fallback chain
    --------------
    1. Standard MLE logit (statsmodels ``Logit.fit``).
    2. **rank_deficient** → drop aliased columns, re-fit.
    3. **separation_detected** → Firth penalised logistic regression
       (``firthlogist``-style bias-reduced logit via iterative adjustment),
       falling back to ridge (L2) regularised logit.
    4. **convergence_failure** → standardise predictors, increase iterations,
       then ridge logit.
    5. **sparse_cells** → ridge logit.
    """
    import statsmodels.api as sm
    from statsmodels.tools import add_constant
    from src.regression_diagnostics import (
        diagnose_fit_failure,
        fatal_regression_error,
        sanitise_design_matrix,
        standardise_predictors,
        validate_regression_inputs,
    )

    model_label = f"Logistic({target_col})"

    if df.empty:
        logger.info("%s skipped: empty dataset", model_label)
        return {"skipped": True, "reason": "empty dataset"}

    y = df[target_col].astype(int)

    # Pre-check: target must have both 0 and 1
    if y.nunique() < 2:
        if y.empty:
            return {"skipped": True, "reason": "empty target"}
        logger.info(
            "%s skipped: target has only value(s) %s",
            model_label, [int(v) for v in sorted(y.unique())],
        )
        return {"skipped": True, "reason": f"target has no variation (all {y.iloc[0]})"}

    X = pd.get_dummies(df[predictor_cols], drop_first=True, dtype=float)
    X = add_constant(X)

    # ---- Pre-flight diagnostics ---------------------------------------------
    diag = validate_regression_inputs(
        X, y, model_label=model_label, is_binary=True,
    )
    # Track diagnostics for reporting
    preflight_issues = list(diag.get("issues", []))
    fallback_actions: List[str] = []

    # Handle no_variance_x up front: drop constant columns
    if not diag["ok"]:
        types = {i["type"] for i in diag["issues"]}

        if "no_variance_y" in types or "insufficient_data" in types:
            fatal_regression_error(
                model_label, diag,
                response_var=target_col, predictor_vars=predictor_cols,
            )

        # rank_deficient or no_variance_x → sanitise design matrix
        if types & {"rank_deficient", "no_variance_x"}:
            X, dropped = sanitise_design_matrix(X, model_label=model_label)
            if X.shape[1] < 2:  # only const remains
                fatal_regression_error(
                    model_label, diag,
                    response_var=target_col, predictor_vars=predictor_cols,
                )
            if dropped:
                fallback_actions.append(f"Removed {len(dropped)} aliased column(s): {dropped}")
                logger.info(
                    "%s: fix → removed %s; re-validating with %d remaining predictor(s)",
                    model_label, dropped, X.shape[1] - 1,  # -1 for const
                )
            else:
                logger.info(
                    "%s: design matrix already full-rank after inspection (no columns removed)",
                    model_label,
                )
            diag = validate_regression_inputs(
                X, y, model_label=model_label, is_binary=True,
            )

    # Classify primary issue for fallback selection
    primary = diag.get("primary_type") if not diag["ok"] else None

    # ---- Attempt 1: standard MLE logit --------------------------------------
    if primary is None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Perfect separation.*")
                warnings.filterwarnings("ignore", message=".*complete separation.*")
                warnings.filterwarnings("ignore", message=".*Maximum Likelihood.*")
                model = sm.Logit(y, X).fit(disp=0, maxiter=100)
                return {
                    "coefficients": model.params.to_dict(),
                    "p_values": model.pvalues.to_dict(),
                    "odds_ratios": np.exp(model.params).to_dict(),
                    "pseudo_r2": model.prsquared,
                    "aic": model.aic,
                    "converged": model.mle_retvals["converged"],
                    "method": "MLE logit",
                    "diagnostics": {
                        "preflight_issues": preflight_issues,
                        "fallback_actions": fallback_actions,
                        "status": "OK" if not preflight_issues else "WARN",
                    },
                }
        except Exception as e:
            diag = diagnose_fit_failure(e, model_label=model_label, X=X, y=y)
            primary = diag["type"]
            logger.warning("%s MLE failed (%s): %s", model_label, primary, e)

    # ---- Attempt 2: separation → Firth penalised logit ----------------------
    if primary == "separation_detected":
        logger.info("%s: separation detected, attempting Firth penalised logit", model_label)
        fallback_actions.append(f"Separation detected → Firth penalised logit")
        firth_result = _firth_logit(X, y, model_label=model_label,
                                     preflight_issues=preflight_issues,
                                     fallback_actions=fallback_actions)
        if firth_result is not None:
            return firth_result
        # Firth failed; fall through to ridge
        logger.info("%s: Firth failed, falling back to ridge logit", model_label)
        fallback_actions.append("Firth failed → ridge (L2) regularised logit")
        primary = "convergence_failure"

    # ---- Attempt 3: convergence_failure → standardise + ridge ---------------
    if primary in ("convergence_failure", "sparse_cells", "rank_deficient", "separation_detected"):
        logger.info(
            "%s: standard MLE not viable (%s) → falling back to ridge (L2) regularised logit",
            model_label, primary,
        )
        if not any("ridge" in a.lower() for a in fallback_actions):
            fallback_actions.append(f"MLE not viable ({primary}) → ridge (L2) regularised logit")
        return _ridge_logit(X, y, model_label=model_label, reason=primary,
                           preflight_issues=preflight_issues,
                           fallback_actions=fallback_actions)

    # ---- Should not reach here; fatal if we do ------------------------------
    fatal_regression_error(
        model_label, diag if isinstance(diag, dict) else {"type": "unknown", "detail": str(diag)},
        response_var=target_col, predictor_vars=predictor_cols,
    )
    return {"error": "unreachable"}  # for type checker


def _firth_logit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_label: str,
    maxiter: int = 200,
    tol: float = 1e-6,
    preflight_issues: Optional[List[Dict[str, Any]]] = None,
    fallback_actions: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Firth bias-reduced logistic regression (Jeffreys-prior penalisation).

    Implements the iteratively re-weighted least squares adjustment from
    Firth (1993) / Heinze & Schemper (2002).  This is the gold-standard
    correction for separation in logistic regression.
    """
    try:
        vals_X = X.values.astype(float)
        vals_y = y.values.astype(float)
        n, p = vals_X.shape
        beta = np.zeros(p)
        converged = False

        for iteration in range(maxiter):
            eta = vals_X @ beta
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            w = mu * (1 - mu)
            # Fisher information (X'WX) using element-wise weight broadcast
            XtWX = (vals_X * w[:, np.newaxis]).T @ vals_X
            try:
                XtWX_inv = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                return None  # truly singular, can't do Firth

            # Hat matrix diagonal only: h_i = w_i * x_i' (X'WX)^{-1} x_i
            # Vectorised as row-wise dot product — O(N*p^2), no N×N matrices
            XW = vals_X * w[:, np.newaxis]          # N x p
            h = np.sum((vals_X @ XtWX_inv) * XW, axis=1)  # N-vector

            # Firth-adjusted working response
            y_adj = vals_y + h * (0.5 - mu)
            score = vals_X.T @ (y_adj - mu)
            delta = XtWX_inv @ score

            beta += delta
            if np.max(np.abs(delta)) < tol:
                converged = True
                break

        # Compute standard errors from the final Fisher information
        eta = vals_X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        w = mu * (1 - mu)
        XtWX = (vals_X * w[:, np.newaxis]).T @ vals_X
        try:
            cov = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            return None
        se = np.sqrt(np.diag(cov))

        # p-values from Wald test
        z = beta / se
        p_vals = 2 * (1 - sp_stats.norm.cdf(np.abs(z)))

        param_names = list(X.columns)
        return {
            "coefficients": dict(zip(param_names, beta.tolist())),
            "std_errors": dict(zip(param_names, se.tolist())),
            "p_values": dict(zip(param_names, p_vals.tolist())),
            "odds_ratios": dict(zip(param_names, np.exp(beta).tolist())),
            "method": "Firth penalised logit (bias-reduced)",
            "converged": converged,
            "diagnostics": {
                "preflight_issues": preflight_issues or [],
                "fallback_actions": fallback_actions or [],
                "status": "WARN",  # Firth is always a fallback
            },
        }
    except Exception as e:
        logger.warning("%s: Firth logit failed: %s", model_label, e)
        return None


def _ridge_logit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_label: str,
    reason: str,
    alpha: float = 1.0,
    preflight_issues: Optional[List[Dict[str, Any]]] = None,
    fallback_actions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Ridge (L2) regularised logistic regression via scikit-learn.

    Ridge is preferred over L1 (lasso) for confirmatory analysis because
    it shrinks but does not zero-out coefficients, preserving interpretability
    of all pre-registered predictors.
    """
    from sklearn.linear_model import LogisticRegression
    from src.regression_diagnostics import standardise_predictors

    X_std, std_stats = standardise_predictors(X)
    param_names = list(X.columns)

    try:
        lr = LogisticRegression(
            C=1.0 / alpha, solver="lbfgs",
            max_iter=500, fit_intercept=False,  # const already in X
        )
        lr.fit(X_std.values, y.values)
        coef = lr.coef_.flatten()

        # Back-transform coefficients from standardised to original scale
        coef_orig = np.copy(coef)
        for i, col in enumerate(param_names):
            if col in std_stats:
                mean, sd = std_stats[col]
                coef_orig[i] = coef[i] / sd

        logger.info(
            "%s: ridge (L2) logit converged successfully (%d predictor(s))",
            model_label, len(param_names) - 1,  # -1 for const
        )
        return {
            "coefficients": dict(zip(param_names, coef_orig.tolist())),
            "odds_ratios": dict(zip(param_names, np.exp(coef_orig).tolist())),
            "method": f"Ridge (L2) logit (fallback for {reason})",
            "regularized": True,
            "alpha": alpha,
            "diagnostics": {
                "preflight_issues": preflight_issues or [],
                "fallback_actions": fallback_actions or [],
                "status": "WARN",  # Ridge is always a fallback
            },
        }
    except Exception as e:
        logger.warning("%s: ridge logit failed: %s", model_label, e)
        return {
            "error": str(e),
            "method": "ridge_logit_failed",
            "diagnostics": {
                "preflight_issues": preflight_issues or [],
                "fallback_actions": fallback_actions or [],
                "status": "FAIL",
            },
        }


# ---------------------------------------------------------------------------
# Correlation analysis (Step 3 - Outcome)
# ---------------------------------------------------------------------------

def correlation_analysis(
    decisions_df: pd.DataFrame,
    personas_df: pd.DataFrame,
    dims: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_dir: str = "./cache",
) -> Dict[str, Any]:
    """Run full correlation analysis between personas and remedy decisions.

    Parameters
    ----------
    decisions_df : DataFrame with columns [case_id, model, tier, confidence]
    personas_df  : DataFrame with columns [case_id, model, age, geographic, education, socioeconomic, gender, esl, persona_label]
    dims : List of persona dimensions to analyze (default: ESTIMATE_DIMS)
    use_cache : If True (default), check for cached results before computing.
    cache_dir : Directory for analysis cache files.
    """
    if dims is None:
        dims = ESTIMATE_DIMS

    # Check cache first
    if use_cache:
        from src.analysis_cache import get_analysis_cache
        cache = get_analysis_cache(cache_dir)
        dataframes = {
            "decisions_df": decisions_df,
            "personas_df": personas_df,
        }
        params = {"dims": dims}
        cached_result = cache.get("correlation_analysis", dataframes, params)
        if cached_result is not None:
            return cached_result
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

    # Cache the result
    if use_cache:
        cache.put("correlation_analysis", dataframes, params, results)

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

    This function provides the marginal (non-parametric) layer of the
    matched-guise causal analysis.  For the dependence-aware parametric
    models (ordinal CLMM, conditional logistic, GEE) that explicitly
    account for repeated measures within ``complaint_id``, see
    :mod:`src.repeated_measures`.

    Parameters
    ----------
    baseline_decisions : DataFrame [case_id, model, tier]  (original decisions)
    persona_decisions  : DataFrame [case_id, model, variant, tier]  (persona-rewritten decisions)

    Notes
    -----
    The Friedman test and pairwise McNemar tests below already use
    complaint-level blocking (each row of the pivot table is one
    complaint), so they are valid repeated-measures tests.  The
    ``repeated_measures`` module extends this with model-based
    approaches that yield fixed-effect estimates and CIs.
    """
    results: Dict[str, Any] = {}

    # Avoid mutating the caller's DataFrame
    persona_decisions = persona_decisions.copy()

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

    # 5. Per-dimension gap analysis with bootstrap CIs
    results["dimension_gaps"] = _dimension_gap_analysis(persona_decisions)

    return results


# ---------------------------------------------------------------------------
# Dimension-level gap analysis (matched-guise)
# ---------------------------------------------------------------------------

def _dimension_gap_analysis(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Per-dimension max-min gap in favorable rates from matched-guise data.

    Computes marginal rates by averaging across persona combinations within
    each dimension level, then computes the max-min gap.  Bootstrap CIs are
    computed via complaint-level resampling to respect within-complaint
    correlation.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: case_id, variant, favorable.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keyed by dimension name (``age``, ``geographic``, …) and
        ``full_grid``.

        Per-dimension dicts have keys:
        ``sd``, ``sd_ci_lower``, ``sd_ci_upper``, ``sd_ci_method``
        (smooth CI-bearing metric), ``gap`` (max-min, descriptive),
        ``p_value``, ``level_rates``.

        The ``full_grid`` dict has keys:
        ``q9010``, ``q9010_ci_lower``, ``q9010_ci_upper``,
        ``q9010_ci_method`` (smooth CI-bearing metric),
        ``gap`` (max-min, descriptive), ``sd``, ``p_value``,
        ``level_rates``.
    """
    df = df.copy()

    # Parse persona labels → dimension columns
    parsed = df["variant"].apply(parse_persona_label)
    unparsed = parsed.isna()
    if unparsed.any():
        logger.warning(
            "Could not parse %d/%d persona variant labels; "
            "dimension_gaps may be incomplete.",
            int(unparsed.sum()), len(df),
        )
    for dim in CAUSAL_DIMS:
        df[dim] = parsed.apply(lambda d, _d=dim: d[_d] if isinstance(d, dict) else None)
    df = df.dropna(subset=CAUSAL_DIMS)

    if df.empty:
        return {}

    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {}

    # --- Per-dimension analysis ---
    for dim in CAUSAL_DIMS:
        # Per-complaint × per-level mean favorable rate
        clm = (
            df.groupby(["case_id", dim])["favorable"]
            .mean()
            .unstack(fill_value=np.nan)
        )
        if clm.shape[1] < 2:
            continue

        level_means = clm.mean(axis=0)
        gap = float(level_means.max() - level_means.min())
        sd = float(level_means.std(ddof=1))

        # Bootstrap (complaint-level cluster resampling)
        # SD is the primary smooth CI-bearing metric for per-dimension rows.
        n_complaints = len(clm)
        boot_sd = np.empty(n_boot)
        clm_values = clm.values  # numpy for speed
        for b in range(n_boot):
            idx = rng.integers(0, n_complaints, size=n_complaints)
            boot_means = clm_values[idx].mean(axis=0)
            boot_sd[b] = float(np.std(boot_means, ddof=1))

        # Jackknife (leave-one-complaint-out) via pre-aggregated sums
        col_sums = clm_values.sum(axis=0)  # (k,)
        if n_complaints > 1:
            jack_means = (col_sums[np.newaxis, :] - clm_values) / (n_complaints - 1)
            jack_sd = np.std(jack_means, ddof=1, axis=1)
        else:
            jack_sd = np.array([sd])

        sd_ci_lo, sd_ci_hi, sd_ci_method = bca_interval(sd, boot_sd, jack_sd)
        sd_ci_lo = max(0.0, sd_ci_lo)

        # Within-complaint paired test
        p_value = _dimension_paired_test(clm)

        results[dim] = {
            "sd": sd,
            "sd_ci_lower": sd_ci_lo,
            "sd_ci_upper": sd_ci_hi,
            "sd_ci_method": sd_ci_method,
            "gap": gap,
            "p_value": p_value,
            "level_rates": {str(k): float(v) for k, v in level_means.items()},
        }

    # --- Full persona grid ---
    cpg = (
        df.groupby(["case_id", "variant"])["favorable"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    if cpg.shape[1] >= 2:
        persona_means = cpg.mean(axis=0)
        full_gap = float(persona_means.max() - persona_means.min())
        full_sd = float(persona_means.std(ddof=1))
        vals = persona_means.values
        full_q9010 = float(np.quantile(vals, 0.9) - np.quantile(vals, 0.1))

        # Bootstrap (complaint-level cluster resampling)
        # Q90-Q10 is the primary smooth CI-bearing metric for full grid.
        n_complaints = len(cpg)
        boot_q9010 = np.empty(n_boot)
        cpg_values = cpg.values
        for b in range(n_boot):
            idx = rng.integers(0, n_complaints, size=n_complaints)
            bm = cpg_values[idx].mean(axis=0)
            boot_q9010[b] = float(np.quantile(bm, 0.9) - np.quantile(bm, 0.1))

        # Jackknife (leave-one-complaint-out) via pre-aggregated sums
        cpg_col_sums = cpg_values.sum(axis=0)  # (36,)
        if n_complaints > 1:
            jack_means_full = (cpg_col_sums[np.newaxis, :] - cpg_values) / (n_complaints - 1)
            jack_q9010 = (
                np.quantile(jack_means_full, 0.9, axis=1)
                - np.quantile(jack_means_full, 0.1, axis=1)
            )
        else:
            jack_q9010 = np.array([full_q9010])

        q9010_ci_lo, q9010_ci_hi, q9010_ci_method = bca_interval(
            full_q9010, boot_q9010, jack_q9010,
        )
        q9010_ci_lo = max(0.0, q9010_ci_lo)
        q9010_ci_hi = min(1.0, q9010_ci_hi)

        results["full_grid"] = {
            "q9010": full_q9010,
            "q9010_ci_lower": q9010_ci_lo,
            "q9010_ci_upper": q9010_ci_hi,
            "q9010_ci_method": q9010_ci_method,
            "gap": full_gap,
            "sd": full_sd,
            "p_value": None,  # Friedman test is in causal["friedman"]
            "level_rates": None,
        }

    return results


def _dimension_paired_test(clm: pd.DataFrame) -> float:
    """Within-complaint paired test for a dimension's effect on favorable rate.

    Parameters
    ----------
    clm : DataFrame
        Rows = complaints, columns = dimension levels.
        Values = mean favorable rate for that complaint at that level.

    Returns
    -------
    float
        p-value.  Returns NaN if the test cannot be computed.
    """
    clm = clm.dropna()
    if len(clm) < 5:
        return float("nan")

    cols = list(clm.columns)
    try:
        if len(cols) == 2:
            # Wilcoxon signed-rank test
            diff = clm.iloc[:, 0].values - clm.iloc[:, 1].values
            if np.all(diff == 0):
                return 1.0
            stat, p = sp_stats.wilcoxon(diff)
            return float(p)
        else:
            # Friedman test for 3+ levels
            groups = [clm[c].values for c in cols]
            stat, p = sp_stats.friedmanchisquare(*groups)
            return float(p)
    except Exception as e:
        logger.warning("Dimension paired test failed: %s", e)
        return float("nan")


def _pairwise_mcnemar(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Pairwise McNemar tests between persona variants."""
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
            a = merged["favorable_1"].values.astype(bool)
            b = merged["favorable_2"].values.astype(bool)
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
            except Exception as e:
                logger.debug("McNemar test failed for pair %s vs %s: %s", v1, v2, e)

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
    overall_rate = df["favorable"].mean()
    overall_sd = df["favorable"].std()
    effects = {}
    for variant in df["variant"].unique():
        subset = df[df["variant"] == variant]["favorable"].values
        if len(subset) < 5:
            continue
        est, lo, hi = bootstrap_ci(subset)
        diff = est - overall_rate
        cohens_d = float(diff / overall_sd) if overall_sd > 0 else 0.0
        effects[variant] = {
            "rate": est,
            "ci_lower": lo,
            "ci_upper": hi,
            "diff_from_mean": diff,
            "cohens_d": cohens_d,
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

    a = merged["fav_orig"].values.astype(bool)
    b = merged["fav_neut"].values.astype(bool)
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
        results["variance_change"] = {
            "baseline_sd": float(orig_rates.std()),
            "neutralized_sd": float(neut_rates.std()),
        }

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
    keys = merged['tier_orig'].astype(int).astype(str) + "→" + merged['tier_neut'].astype(int).astype(str)
    changes = keys.value_counts().to_dict()
    return changes
