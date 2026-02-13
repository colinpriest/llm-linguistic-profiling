"""Repeated-measures causal inference for the matched-guise experiment.

This module implements dependence-aware models that explicitly account for
the fact that the same underlying complaint is observed under multiple
persona framings (repeated measures on ``complaint_id``).

Data layout (long format)
-------------------------
Each row represents one (complaint × model × condition × persona) observation:

* ``complaint_id``  – original CFPB complaint ID (clustering unit)
* ``model``         – LLM identifier (two levels)
* ``condition``     – ``"original"`` or ``"neutralised"``
* ``persona_id``    – one of 36 persona labels
* ``severity_bin``  – ``"low"`` / ``"medium"`` / ``"high"``
* ``outcome_tier``  – ordinal remedy tier (0 / 1 / 2)
* ``favourable``    – binary: ``tier >= 1``
* ``info_request``  – binary: ``confidence == "need_more_info"``

Model classes
-------------
A) **Ordinal mixed model** (cumulative link mixed model approximation)
   – Random intercept for ``complaint_id``
   – Fixed effects: persona attributes, severity_bin, model, condition, interactions
   – Key interaction: persona × severity

B) **Binary GLMM / conditional logistic** for ``favourable``
   – Conditional logistic with ``complaint_id`` as stratum (pure within-complaint)
   – Logistic mixed model with ``complaint_id`` random intercept (flexibility)

C) **Info-request burden** (binary GLMM)
   – Logistic mixed model with ``complaint_id`` random intercept

Inference
---------
* Clustered standard errors by ``complaint_id`` (marginal models)
* Mixed-model fixed-effect estimates + profile CIs
* Cochran's Q / paired permutation tests for multi-persona binary comparisons
* Friedman + post-hoc paired tests with FDR correction
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm

from src.schemas import parse_persona_label
from src.statistics import bootstrap_ci, fdr_correction, friedman_test

logger = logging.getLogger(__name__)

# Ordinal encoding for severity bins used across all model-fitting functions.
_SEVERITY_ORD_MAP: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# Maximum persona disadvantage score: age(1) + geo(2) + edu(1) + ses(2) = 6.
_MAX_PERSONA_SCORE: float = 6.0

# ---------------------------------------------------------------------------
# 1. Long-format data builder
# ---------------------------------------------------------------------------

def build_long_format(
    baseline_df: pd.DataFrame,
    persona_decisions_df: Optional[pd.DataFrame],
    neutralized_df: Optional[pd.DataFrame],
    personas_df: pd.DataFrame,
    severity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the canonical long-format analysis dataset.

    Parameters
    ----------
    baseline_df : DataFrame [case_id, model, tier, confidence]
        Baseline remedy decisions on original complaints.
    persona_decisions_df : DataFrame [case_id, model, variant, tier, confidence]
        Decisions on persona-rewritten variants.  May be None.
    neutralized_df : DataFrame [case_id, model, tier, confidence]
        Decisions on neutralised variants.  May be None.
    personas_df : DataFrame [case_id, model, persona_label, ...]
        Persona estimates for each complaint.
    severity_df : DataFrame [case_id, monetary_severity, issue_severity]
        Severity metadata per complaint.

    Returns
    -------
    Long-format DataFrame with columns:
        complaint_id, model, condition, persona_id, severity_bin,
        outcome_tier, favourable, info_request
    """
    frames: List[pd.DataFrame] = []

    sev = severity_df[["case_id", "monetary_severity", "issue_severity"]].drop_duplicates(subset=["case_id"])

    # --- Persona-rewritten decisions (the core repeated-measures block) ---
    if persona_decisions_df is not None and len(persona_decisions_df) > 0:
        pdf = persona_decisions_df.copy()
        pdf = pdf.rename(columns={"variant": "persona_id"})
        pdf["condition"] = "original"
        pdf = pdf.merge(sev, on="case_id", how="left")
        frames.append(pdf)

    # --- Baseline decisions (original condition, persona from estimation) ---
    if baseline_df is not None and len(baseline_df) > 0:
        bdf = baseline_df.copy()
        bdf["condition"] = "original"
        bdf = bdf.merge(
            personas_df[["case_id", "model", "persona_label"]].drop_duplicates(),
            on=["case_id", "model"], how="left",
        )
        bdf = bdf.rename(columns={"persona_label": "persona_id"})
        bdf = bdf.merge(sev, on="case_id", how="left")
        frames.append(bdf)

    # --- Neutralised decisions ---
    if neutralized_df is not None and len(neutralized_df) > 0:
        ndf = neutralized_df.copy()
        ndf["condition"] = "neutralised"
        ndf = ndf.merge(
            personas_df[["case_id", "model", "persona_label"]].drop_duplicates(),
            on=["case_id", "model"], how="left",
        )
        ndf = ndf.rename(columns={"persona_label": "persona_id"})
        ndf = ndf.merge(sev, on="case_id", how="left")
        frames.append(ndf)

    if not frames:
        return pd.DataFrame()

    long = pd.concat(frames, ignore_index=True)

    # Standardise columns
    long = long.rename(columns={"case_id": "complaint_id", "tier": "outcome_tier"})
    long["favourable"] = (long["outcome_tier"] >= 1).astype(int)
    long["info_request"] = (long["confidence"] == "need_more_info").astype(int)

    # Choose a single severity_bin (prefer issue_severity; fall back to monetary)
    long["severity_bin"] = long["issue_severity"]
    mask = ~long["severity_bin"].isin(["low", "medium", "high"])
    long.loc[mask, "severity_bin"] = long.loc[mask, "monetary_severity"]
    # Impute rows where severity is still unknown to "medium" and log the count
    imputed_mask = ~long["severity_bin"].isin(["low", "medium", "high"])
    n_imputed = imputed_mask.sum()
    if n_imputed > 0:
        logger.warning(
            "Imputed %d rows (%.1f%%) to severity_bin='medium'",
            n_imputed, 100 * n_imputed / len(long),
        )
    long["severity_bin"] = long["severity_bin"].where(~imputed_mask, other="medium")

    # Keep canonical columns
    keep = [
        "complaint_id", "model", "condition", "persona_id",
        "severity_bin", "outcome_tier", "favourable", "info_request",
        "monetary_severity", "issue_severity",
    ]
    existing = [c for c in keep if c in long.columns]
    return long[existing].copy()


# ---------------------------------------------------------------------------
# 2A. Ordinal mixed model (CLMM approximation via OrderedModel + cluster)
# ---------------------------------------------------------------------------

def ordinal_mixed_model(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Cumulative-link mixed model approximation for ``outcome_tier``.

    Uses ``statsmodels.miscmodels.ordinal_model.OrderedModel`` with
    clustered (sandwich) standard errors on ``complaint_id`` to account
    for repeated measures.

    Fixed effects: persona dimension dummies, severity_bin, model, condition,
    persona_id × severity_bin interaction (via pre-computed column).

    Fallback chain (exploratory model):
    1. OrderedModel (logit link) + clustered SEs
    2. rank_deficient → sanitise design matrix, re-fit
    3. convergence_failure → standardise predictors, increase iterations
    4. Any remaining failure → skip with diagnostic (exploratory only)

    Parameters
    ----------
    long_df : Long-format DataFrame from :func:`build_long_format`.
    model_filter : If given, restrict to this LLM model only.

    Returns
    -------
    Dict with keys: coefficients, std_errors, p_values, conf_int,
    thresholds, n_obs, n_clusters, model_info.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    from src.regression_diagnostics import (
        diagnose_fit_failure,
        sanitise_design_matrix,
        standardise_predictors,
        validate_regression_inputs,
    )

    model_label = "OrdinalMixed(outcome_tier)"
    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["outcome_tier"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or tier variation"}

    # Build design: persona dims + severity + condition + interaction + model dummies
    df, exog_cols = _prepare_model_columns(df)

    exog = df[exog_cols].astype(float)
    endog = df["outcome_tier"].astype(float)

    # ---- Pre-flight diagnostics ---------------------------------------------
    diag = validate_regression_inputs(exog, endog, model_label=model_label)
    preflight_issues = list(diag.get("issues", []))
    fallback_actions: List[str] = []

    if not diag["ok"]:
        types = {i["type"] for i in diag["issues"]}
        if types & {"no_variance_y", "insufficient_data"}:
            return {"skipped": True, "reason": diag["issues"][0]["detail"],
                    "estimand_status": "not_estimable"}
        if types & {"rank_deficient", "no_variance_x"}:
            exog, dropped = sanitise_design_matrix(exog, model_label=model_label)
            if dropped:
                fallback_actions.append(f"Removed {len(dropped)} aliased column(s)")
            if exog.shape[1] < 1:
                return {"skipped": True, "reason": "all predictors dropped after sanitisation",
                        "estimand_status": "not_estimable"}

    # ---- Fit ordinal model --------------------------------------------------
    def _fit_ordinal(exog_mat, standardised=False, extra_fallbacks=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            om = OrderedModel(endog, exog_mat, distr="logit")
            fit_plain = om.fit(method="bfgs", disp=False, maxiter=300 if standardised else 200)

        used_cluster = False
        cluster_fallback_msg = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_plain.get_robustcov_results(
                    cov_type="cluster",
                    groups=df["complaint_id"].values,
                )
            used_cluster = True
        except Exception:
            cluster_fallback_msg = "Clustered SEs unavailable, using model-based SEs"
            logger.info("Ordinal model: %s", cluster_fallback_msg)
            fit = fit_plain

        se_base = np.asarray(fit.bse, dtype=float)
        p_base = np.asarray(fit.pvalues, dtype=float)
        try:
            ci_arr = fit.conf_int()
            ci_lo_base = np.asarray(ci_arr)[:, 0]
            ci_hi_base = np.asarray(ci_arr)[:, 1]
        except Exception:
            ci_lo_base = fit.params - 1.96 * se_base
            ci_hi_base = fit.params + 1.96 * se_base

        param_names = list(exog_mat.columns) + [f"threshold_{i}" for i in range(len(fit.params) - len(exog_mat.columns))]

        method_note = "repeated-measures ordinal with clustered SEs"
        if standardised:
            method_note += " (standardised predictors, convergence fallback)"

        # Build diagnostics
        all_fallbacks = list(fallback_actions)
        if extra_fallbacks:
            all_fallbacks.extend(extra_fallbacks)
        if cluster_fallback_msg:
            all_fallbacks.append(cluster_fallback_msg)
        if standardised:
            all_fallbacks.append("Standardised predictors (convergence fallback)")

        diag_status = "OK" if not all_fallbacks and not preflight_issues else "WARN"

        return {
            "coefficients": dict(zip(param_names, fit.params.tolist())),
            "std_errors": dict(zip(param_names, se_base.tolist())),
            "p_values": dict(zip(param_names, p_base.tolist())),
            "conf_int": {
                name: [float(lo), float(hi)]
                for name, lo, hi in zip(param_names, ci_lo_base, ci_hi_base)
            },
            "n_obs": len(df),
            "n_clusters": int(df["complaint_id"].nunique()),
            "clustered_se": used_cluster,
            "log_likelihood": float(fit.llf) if hasattr(fit, "llf") else None,
            "aic": float(fit.aic) if hasattr(fit, "aic") else None,
            "model_info": {
                "type": "OrderedModel (logit link)",
                "clustering": "complaint_id" if used_cluster else "none (model-based SEs)",
                "method": method_note,
                "status": "exploratory",
            },
            "diagnostics": {
                "preflight_issues": preflight_issues,
                "fallback_actions": all_fallbacks,
                "status": diag_status,
            },
        }

    # Attempt 1: standard fit
    try:
        return _fit_ordinal(exog)
    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label, X=exog, y=endog)
        logger.warning("%s attempt 1 failed (%s): %s", model_label, fit_diag["type"], e)

    # Attempt 2: standardise predictors and retry
    try:
        exog_std, _ = standardise_predictors(exog)
        return _fit_ordinal(exog_std, standardised=True)
    except Exception as e2:
        logger.warning("%s attempt 2 (standardised) failed: %s", model_label, e2)
        fallback_actions.append(f"Fit failed after standardisation: {e2}")

    return {
        "skipped": True,
        "reason": f"attempt1: {e}, attempt2: {e2}",
        "estimand_status": "not_estimable",
        "diagnostics": {
            "preflight_issues": preflight_issues,
            "fallback_actions": fallback_actions,
            "status": "FAIL",
        },
    }


# ---------------------------------------------------------------------------
# 2B. Binary GLMM / conditional logistic for favourable outcome
# ---------------------------------------------------------------------------

def conditional_logistic_favourable(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Conditional logistic regression with ``complaint_id`` as stratum.

    Pure within-complaint comparison — the gold standard for matched-guise.
    Uses ``statsmodels.discrete.conditional_models.ConditionalLogit``.

    Fallback chain (confirmatory model):
    1. ConditionalLogit with complaint_id strata.
    2. rank_deficient → sanitise design matrix, re-fit.
    3. convergence_failure → standardise predictors, re-fit.
    4. Remaining failure → fatal error (confirmatory estimand).

    Parameters
    ----------
    long_df : Long-format DataFrame.
    model_filter : Restrict to this LLM model.

    Returns
    -------
    Dict with coefficients, SEs, p-values, CIs, n_obs, n_strata.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit
    from src.regression_diagnostics import (
        diagnose_fit_failure,
        fatal_regression_error,
        sanitise_design_matrix,
        standardise_predictors,
        validate_regression_inputs,
    )

    model_label = "ConditionalLogit(favourable)"
    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    # Need within-complaint variation to estimate conditional logit
    complaint_var = df.groupby("complaint_id")["favourable"].nunique()
    informative = complaint_var[complaint_var > 1].index
    df = df[df["complaint_id"].isin(informative)]

    if len(df) < 20 or len(informative) < 5:
        return {"skipped": True, "reason": "insufficient within-complaint variation",
                "estimand_status": "not_estimable"}

    df, exog_cols = _prepare_model_columns(df)

    exog = df[exog_cols].astype(float)
    endog = df["favourable"].astype(float)

    # ---- Pre-flight diagnostics ---------------------------------------------
    diag = validate_regression_inputs(
        exog, endog, model_label=model_label, is_binary=True,
    )
    if not diag["ok"]:
        types = {i["type"] for i in diag["issues"]}
        if types & {"no_variance_y", "insufficient_data"}:
            return {"skipped": True, "reason": diag["issues"][0]["detail"],
                    "estimand_status": "not_estimable"}
        if types & {"rank_deficient", "no_variance_x"}:
            exog, dropped = sanitise_design_matrix(exog, model_label=model_label)
            if exog.shape[1] < 1:
                fatal_regression_error(
                    model_label, diag,
                    response_var="favourable",
                    predictor_vars=exog_cols,
                )

    # ---- Conditional-logit-specific pre-flight --------------------------------
    # In a conditional logit, only *within-stratum varying* predictors are
    # identifiable.  Predictors constant within each complaint (e.g. sev_ord,
    # model dummies) are absorbed by the stratum fixed effects.  Additionally,
    # persona_x_severity = persona_score * sev_ord is a scalar multiple of
    # persona_score within each stratum (since sev_ord is constant), making
    # them perfectly collinear.  We drop these before fitting.

    preflight_issues: List[Dict[str, str]] = []
    fallback_actions: List[str] = []
    original_cols = list(exog.columns)

    # 1. Drop within-stratum constant predictors
    complaint_groups = df.loc[exog.index].groupby("complaint_id")
    within_constant: List[str] = []
    for col in list(exog.columns):
        within_var = complaint_groups.apply(
            lambda g, c=col: exog.loc[g.index, c].nunique(), include_groups=False
        )
        if (within_var <= 1).all():
            within_constant.append(col)

    if within_constant:
        detail = (
            f"Predictors constant within every complaint stratum "
            f"(absorbed by conditional logit fixed effects): {within_constant}"
        )
        preflight_issues.append({"type": "within_stratum_constant", "detail": detail})
        action = f"Dropped within-stratum constant predictors: {within_constant}"
        fallback_actions.append(action)
        logger.info("%s pre-flight: %s", model_label, detail)
        logger.info("%s action: %s", model_label, action)
        exog = exog.drop(columns=within_constant)

    # 2. Drop within-stratum collinear predictors (e.g. persona_x_severity
    #    is a scalar multiple of persona_score within each stratum)
    if exog.shape[1] >= 2:
        within_collinear: List[str] = []
        cols = list(exog.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                if c2 in within_collinear:
                    continue
                # Check if c2 is a linear function of c1 within every stratum
                is_collinear = True
                for _, grp in complaint_groups:
                    x1 = exog.loc[grp.index, c1]
                    x2 = exog.loc[grp.index, c2]
                    if x1.nunique() <= 1:
                        continue  # constant within this stratum, skip
                    # If perfectly collinear, rank of [x1, x2] == 1
                    mat = np.column_stack([x1.values - x1.mean(), x2.values - x2.mean()])
                    if np.linalg.matrix_rank(mat, tol=1e-8) > 1:
                        is_collinear = False
                        break
                if is_collinear:
                    within_collinear.append(c2)

        if within_collinear:
            detail = (
                f"Predictors perfectly collinear within strata "
                f"(e.g. interaction = scalar × main effect when grouping "
                f"variable is constant per stratum): {within_collinear}"
            )
            preflight_issues.append({"type": "within_stratum_collinear", "detail": detail})
            action = f"Dropped within-stratum collinear predictors: {within_collinear}"
            fallback_actions.append(action)
            logger.info("%s pre-flight: %s", model_label, detail)
            logger.info("%s action: %s", model_label, action)
            exog = exog.drop(columns=within_collinear)

    # 3. Final check: enough predictors and strata?
    if exog.shape[1] < 1:
        return {
            "skipped": True,
            "reason": "no within-stratum varying predictors for conditional logit",
            "estimand_status": "not_estimable",
            "diagnostics": {
                "status": "FAIL",
                "preflight_issues": preflight_issues,
                "fallback_actions": fallback_actions,
            },
        }

    surviving_cols = list(exog.columns)
    if surviving_cols != original_cols:
        action = (
            f"Reduced model from {len(original_cols)} predictors "
            f"({original_cols}) to {len(surviving_cols)} ({surviving_cols})"
        )
        fallback_actions.append(action)
        logger.info("%s action: %s", model_label, action)

    n_params = exog.shape[1]
    if len(informative) < 2 * n_params:
        detail = (
            f"Low strata-to-parameter ratio: {len(informative)} informative "
            f"strata for {n_params} parameters; SEs may be unreliable"
        )
        preflight_issues.append({"type": "low_strata_ratio", "detail": detail})
        logger.info("%s pre-flight: %s", model_label, detail)

    diag_status = "WARN" if preflight_issues else "OK"

    # Sort by complaint_id for grouping
    sort_idx = df["complaint_id"].argsort()
    exog = exog.iloc[sort_idx].reset_index(drop=True)
    endog = endog.iloc[sort_idx].reset_index(drop=True)
    groups = df["complaint_id"].iloc[sort_idx].reset_index(drop=True)

    def _fit_clogit(exog_mat, method_note=""):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clogit = ConditionalLogit(endog, exog_mat, groups=groups)
            fit = clogit.fit(disp=False, maxiter=200)

        # Extract SEs and p-values; covariance matrix may be singular
        n_params = len(exog_mat.columns)
        try:
            bse = fit.bse.tolist()
            pvalues = fit.pvalues.tolist()
        except Exception as cov_err:
            logger.warning(
                "ConditionalLogit covariance unavailable (%s); "
                "reporting coefficients and odds ratios without SEs",
                cov_err,
            )
            bse = [float("nan")] * n_params
            pvalues = [float("nan")] * n_params

        ci_dict = {}
        try:
            ci_df = fit.conf_int()
            for name in exog_mat.columns:
                if name in ci_df.index:
                    ci_dict[name] = [float(ci_df.loc[name].iloc[0]), float(ci_df.loc[name].iloc[1])]
        except Exception:
            for name, se in zip(exog_mat.columns, bse):
                coef = fit.params[exog_mat.columns.get_loc(name)]
                ci_dict[name] = [float(coef - 1.96 * se), float(coef + 1.96 * se)]

        method_str = "within-complaint conditional logistic regression"
        if method_note:
            method_str += f" ({method_note})"
        se_unavailable = any(np.isnan(s) for s in bse)
        if se_unavailable:
            method_str += " (SEs unavailable — singular covariance)"
            cov_action = "Covariance matrix singular; coefficients and odds ratios reported without standard errors or p-values"
            fallback_actions.append(cov_action)
            logger.info("%s: %s", model_label, cov_action)
            if diag_status == "OK":
                diag_status = "WARN"

        return {
            "coefficients": dict(zip(exog_mat.columns, fit.params.tolist())),
            "std_errors": dict(zip(exog_mat.columns, bse)),
            "p_values": dict(zip(exog_mat.columns, pvalues)),
            "conf_int": ci_dict,
            "odds_ratios": dict(zip(exog_mat.columns, np.exp(fit.params).tolist())),
            "n_obs": len(df),
            "n_strata": int(groups.nunique()),
            "n_informative_strata": len(informative),
            "log_likelihood": float(fit.llf) if hasattr(fit, "llf") else None,
            "model_info": {
                "type": "ConditionalLogit",
                "clustering": "complaint_id (stratum)",
                "method": method_str,
            },
            "diagnostics": {
                "status": diag_status,
                "preflight_issues": preflight_issues,
                "fallback_actions": fallback_actions,
                "original_predictors": original_cols,
                "fitted_predictors": list(exog_mat.columns),
            },
        }

    # Attempt 1: standard fit
    try:
        return _fit_clogit(exog)
    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label, X=exog, y=endog)
        fallback_actions.append(f"Attempt 1 failed ({fit_diag['type']}): {e}")
        logger.warning("%s attempt 1 failed (%s): %s", model_label, fit_diag["type"], e)

    # Attempt 2: standardise predictors
    try:
        fallback_actions.append("Retrying with standardised predictors")
        logger.info("%s: retrying with standardised predictors", model_label)
        exog_std, _ = standardise_predictors(exog)
        return _fit_clogit(exog_std, method_note="standardised predictors")
    except Exception as e2:
        fit_diag2 = diagnose_fit_failure(e2, model_label=model_label, X=exog, y=endog)
        fallback_actions.append(f"Attempt 2 failed ({fit_diag2['type']}): {e2}")
        logger.warning("%s attempt 2 failed (%s): %s", model_label, fit_diag2["type"], e2)

    # Confirmatory model — fatal if not estimable
    fatal_regression_error(
        model_label, fit_diag2,
        response_var="favourable", predictor_vars=exog_cols,
    )
    return {"error": "unreachable"}


def logistic_mixed_model_favourable(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Logistic mixed model with ``complaint_id`` random intercept for ``favourable``.

    More flexible than conditional logistic — allows severity and condition
    interactions plus complaint-level covariates.

    Uses GEE (Generalized Estimating Equations) with exchangeable
    correlation as a robust approximation to a full GLMM.

    Fallback chain (confirmatory model):
    1. GEE Binomial + Exchangeable correlation.
    2. rank_deficient → sanitise design matrix, re-fit.
    3. convergence_failure → Independence correlation structure.
    4. Remaining failure → fatal error (confirmatory estimand).
    """
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable, Independence
    from src.regression_diagnostics import (
        diagnose_fit_failure,
        fatal_regression_error,
        sanitise_design_matrix,
        validate_regression_inputs,
    )

    model_label = "GEE_Logistic(favourable)"
    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["favourable"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or outcome variation",
                "estimand_status": "not_estimable"}

    df, exog_cols = _prepare_model_columns(df)

    # Sort by complaint_id for GEE grouping
    df = df.sort_values("complaint_id").reset_index(drop=True)
    exog = df[exog_cols].astype(float)
    endog = df["favourable"].astype(float)
    groups = df["complaint_id"]

    # ---- Pre-flight diagnostics ---------------------------------------------
    diag = validate_regression_inputs(
        exog, endog, model_label=model_label, is_binary=True,
    )
    preflight_issues = list(diag.get("issues", []))
    fallback_actions: List[str] = []

    if not diag["ok"]:
        types = {i["type"] for i in diag["issues"]}
        if types & {"no_variance_y", "insufficient_data"}:
            return {"skipped": True, "reason": diag["issues"][0]["detail"],
                    "estimand_status": "not_estimable"}
        if types & {"rank_deficient", "no_variance_x"}:
            exog, dropped = sanitise_design_matrix(exog, model_label=model_label)
            if dropped:
                fallback_actions.append(f"Removed {len(dropped)} aliased column(s)")
            if exog.shape[1] < 1:
                fatal_regression_error(
                    model_label, diag,
                    response_var="favourable", predictor_vars=exog_cols,
                )

    exog = sm.add_constant(exog)

    def _fit_gee(cov_struct, method_note="", extra_fallbacks=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gee_model = GEE(
                endog, exog, groups=groups,
                family=Binomial(),
                cov_struct=cov_struct,
            )
            fit = gee_model.fit(maxiter=200)

        method_str = "logistic GEE with complaint_id clustering (GLMM approximation)"
        if method_note:
            method_str += f" ({method_note})"

        all_fallbacks = list(fallback_actions)
        if extra_fallbacks:
            all_fallbacks.extend(extra_fallbacks)
        diag_status = "OK" if not all_fallbacks and not preflight_issues else "WARN"

        ci_df = fit.conf_int()
        return {
            "coefficients": fit.params.to_dict(),
            "std_errors": fit.bse.to_dict(),
            "p_values": fit.pvalues.to_dict(),
            "conf_int": {
                name: [float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])]
                for name in fit.params.index
            },
            "odds_ratios": np.exp(fit.params).to_dict(),
            "n_obs": len(df),
            "n_clusters": int(groups.nunique()),
            "scale": float(fit.scale) if hasattr(fit, "scale") else None,
            "model_info": {
                "type": f"GEE (Binomial, {cov_struct.__class__.__name__})",
                "clustering": "complaint_id",
                "method": method_str,
            },
            "diagnostics": {
                "preflight_issues": preflight_issues,
                "fallback_actions": all_fallbacks,
                "status": diag_status,
            },
        }

    # Attempt 1: Exchangeable correlation
    try:
        return _fit_gee(Exchangeable())
    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label, X=exog, y=endog)
        logger.warning("%s Exchangeable failed (%s): %s", model_label, fit_diag["type"], e)
        fallback_actions.append(f"Exchangeable failed ({fit_diag['type']})")

    # Attempt 2: Independence correlation (more robust)
    try:
        return _fit_gee(Independence(), method_note="Independence fallback",
                       extra_fallbacks=["Independence correlation fallback"])
    except Exception as e2:
        fit_diag2 = diagnose_fit_failure(e2, model_label=model_label, X=exog, y=endog)
        logger.warning("%s Independence failed (%s): %s", model_label, fit_diag2["type"], e2)
        fallback_actions.append(f"Independence also failed ({fit_diag2['type']})")

    # Confirmatory model — fatal if not estimable
    fatal_regression_error(
        model_label, fit_diag2,
        response_var="favourable", predictor_vars=exog_cols,
    )
    return {"error": "unreachable"}


# ---------------------------------------------------------------------------
# 2C. Info-request burden (binary GLMM)
# ---------------------------------------------------------------------------

def logistic_mixed_model_info_request(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Logistic mixed model for ``info_request`` with ``complaint_id`` random intercept.

    Same GEE approach and fallback chain as the favourable model.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable, Independence
    from src.regression_diagnostics import (
        diagnose_fit_failure,
        fatal_regression_error,
        sanitise_design_matrix,
        validate_regression_inputs,
    )

    model_label = "GEE_Logistic(info_request)"
    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["info_request"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or outcome variation",
                "estimand_status": "not_estimable"}

    df, exog_cols = _prepare_model_columns(df)

    df = df.sort_values("complaint_id").reset_index(drop=True)
    exog = df[exog_cols].astype(float)
    endog = df["info_request"].astype(float)
    groups = df["complaint_id"]

    # ---- Pre-flight diagnostics ---------------------------------------------
    diag = validate_regression_inputs(
        exog, endog, model_label=model_label, is_binary=True,
    )
    preflight_issues_ir = list(diag.get("issues", []))
    fallback_actions_ir: List[str] = []

    if not diag["ok"]:
        types = {i["type"] for i in diag["issues"]}
        if types & {"no_variance_y", "insufficient_data"}:
            return {"skipped": True, "reason": diag["issues"][0]["detail"],
                    "estimand_status": "not_estimable"}
        if types & {"rank_deficient", "no_variance_x"}:
            exog, dropped = sanitise_design_matrix(exog, model_label=model_label)
            if dropped:
                fallback_actions_ir.append(f"Removed {len(dropped)} aliased column(s)")
            if exog.shape[1] < 1:
                fatal_regression_error(
                    model_label, diag,
                    response_var="info_request", predictor_vars=exog_cols,
                )

    exog = sm.add_constant(exog)

    def _fit_gee_ir(cov_struct, method_note="", extra_fallbacks=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gee_model = GEE(
                endog, exog, groups=groups,
                family=Binomial(),
                cov_struct=cov_struct,
            )
            fit = gee_model.fit(maxiter=200)

        method_str = "logistic GEE with complaint_id clustering (GLMM approximation)"
        if method_note:
            method_str += f" ({method_note})"

        all_fallbacks = list(fallback_actions_ir)
        if extra_fallbacks:
            all_fallbacks.extend(extra_fallbacks)
        diag_status = "OK" if not all_fallbacks and not preflight_issues_ir else "WARN"

        ci_df = fit.conf_int()
        return {
            "coefficients": fit.params.to_dict(),
            "std_errors": fit.bse.to_dict(),
            "p_values": fit.pvalues.to_dict(),
            "conf_int": {
                name: [float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])]
                for name in fit.params.index
            },
            "odds_ratios": np.exp(fit.params).to_dict(),
            "n_obs": len(df),
            "n_clusters": int(groups.nunique()),
            "scale": float(fit.scale) if hasattr(fit, "scale") else None,
            "model_info": {
                "type": f"GEE (Binomial, {cov_struct.__class__.__name__})",
                "clustering": "complaint_id",
                "method": method_str,
            },
            "diagnostics": {
                "preflight_issues": preflight_issues_ir,
                "fallback_actions": all_fallbacks,
                "status": diag_status,
            },
        }

    # Attempt 1: Exchangeable correlation
    try:
        return _fit_gee_ir(Exchangeable())
    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label, X=exog, y=endog)
        logger.warning("%s Exchangeable failed (%s): %s", model_label, fit_diag["type"], e)
        fallback_actions_ir.append(f"Exchangeable failed ({fit_diag['type']})")

    # Attempt 2: Independence correlation (more robust)
    try:
        return _fit_gee_ir(Independence(), method_note="Independence fallback",
                         extra_fallbacks=["Independence correlation fallback"])
    except Exception as e2:
        fit_diag2 = diagnose_fit_failure(e2, model_label=model_label, X=exog, y=endog)
        logger.warning("%s Independence failed (%s): %s", model_label, fit_diag2["type"], e2)
        fallback_actions_ir.append(f"Independence also failed ({fit_diag2['type']})")

    # Confirmatory model — fatal if not estimable
    fatal_regression_error(
        model_label, fit_diag2,
        response_var="info_request", predictor_vars=exog_cols,
    )
    return {"error": "unreachable"}


# ---------------------------------------------------------------------------
# 2D. Linear mixed model (robustness check treating tier as numeric)
# ---------------------------------------------------------------------------

def linear_mixed_model_tier(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Linear mixed model: outcome_tier ~ fixed effects + (1|complaint_id).

    Robustness check: treats ordinal tier as continuous.
    Uses ``statsmodels.regression.mixed_linear_model.MixedLM``.

    Fallback chain:
    1. REML estimation.
    2. ML estimation (if REML singular).
    3. Powell optimiser with zero start params.
    4. Fatal error with diagnostics.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    from src.regression_diagnostics import (
        diagnose_fit_failure,
        fatal_regression_error,
        sanitise_design_matrix,
        validate_regression_inputs,
    )

    model_label = "LMM(outcome_tier)"
    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["outcome_tier"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or tier variation",
                "estimand_status": "not_estimable"}

    df, exog_cols = _prepare_model_columns(df)

    import statsmodels.api as sm
    exog = df[exog_cols].astype(float)
    endog = df["outcome_tier"].astype(float)
    groups = df["complaint_id"]

    # ---- Pre-flight diagnostics ---------------------------------------------
    diag = validate_regression_inputs(exog, endog, model_label=model_label)
    if not diag["ok"]:
        types = {i["type"] for i in diag["issues"]}
        if types & {"no_variance_y", "insufficient_data"}:
            return {"skipped": True, "reason": diag["issues"][0]["detail"],
                    "estimand_status": "not_estimable"}
        if types & {"rank_deficient", "no_variance_x"}:
            exog, dropped = sanitise_design_matrix(exog, model_label=model_label)
            if exog.shape[1] < 1:
                fatal_regression_error(
                    model_label, diag,
                    response_var="outcome_tier", predictor_vars=exog_cols,
                )

    exog = sm.add_constant(exog)

    lmm_preflight: List[Dict[str, str]] = []
    lmm_fallbacks: List[str] = []
    lmm_diag_status = "OK"
    estimation_method = "REML"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lmm = MixedLM(endog, exog, groups=groups)
            try:
                fit = lmm.fit(reml=True, maxiter=200)
                logger.info("%s: REML estimation converged", model_label)
            except (np.linalg.LinAlgError, ValueError) as e_reml:
                lmm_fallbacks.append(f"REML failed ({e_reml}); switching to ML")
                logger.info("%s: REML failed (%s), falling back to ML", model_label, e_reml)
                lmm_diag_status = "WARN"
                estimation_method = "ML"
                try:
                    fit = lmm.fit(reml=False, maxiter=200)
                    logger.info("%s: ML estimation converged", model_label)
                except (np.linalg.LinAlgError, ValueError) as e_ml:
                    lmm_fallbacks.append(f"ML failed ({e_ml}); switching to Powell optimiser with zero start params")
                    logger.info("%s: ML failed (%s), falling back to Powell", model_label, e_ml)
                    estimation_method = "ML + Powell"
                    fit = lmm.fit(
                        reml=False, maxiter=200,
                        start_params=np.zeros(exog.shape[1] + 1),
                        method="powell",
                    )
                    logger.info("%s: Powell estimation completed", model_label)

        re_var = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, "iloc") else float(fit.cov_re)
        resid_var = float(fit.scale)
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0.0

        ci_df = fit.conf_int()
        return {
            "coefficients": fit.fe_params.to_dict(),
            "std_errors": fit.bse_fe.to_dict(),
            "p_values": fit.pvalues.to_dict(),
            "conf_int": {
                name: [float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])]
                for name in fit.fe_params.index
            },
            "random_effects_variance": re_var,
            "residual_variance": resid_var,
            "icc": icc,
            "n_obs": int(fit.nobs),
            "n_clusters": int(groups.nunique()),
            "converged": fit.converged,
            "model_info": {
                "type": f"Linear Mixed Model ({estimation_method})",
                "clustering": "complaint_id (random intercept)",
                "method": f"LMM with complaint_id random intercept (robustness, {estimation_method})",
                "icc_interpretation": (
                    f"ICC = {icc:.3f}: {icc*100:.1f}% of tier variance is "
                    f"between complaints (justifies clustering)"
                ),
            },
            "diagnostics": {
                "status": lmm_diag_status,
                "preflight_issues": lmm_preflight,
                "fallback_actions": lmm_fallbacks,
            },
        }
    except Exception as e:
        fit_diag = diagnose_fit_failure(e, model_label=model_label, X=exog, y=endog)
        lmm_fallbacks.append(f"All estimation methods failed ({fit_diag['type']}): {e}")
        logger.warning("Linear mixed model failed (%s): %s", fit_diag["type"], e)
        fatal_regression_error(
            model_label, fit_diag,
            response_var="outcome_tier", predictor_vars=exog_cols,
        )
        return {"error": "unreachable"}


# ---------------------------------------------------------------------------
# 3. Nonparametric repeated-measures tests
# ---------------------------------------------------------------------------

def repeated_measures_nonparametric(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Nonparametric within-complaint tests for all three outcomes.

    * **Ordinal tier**: Friedman test across persona variants, then
      post-hoc Wilcoxon signed-rank with FDR correction.
    * **Binary favourable**: Cochran's Q across persona variants, then
      pairwise McNemar with FDR correction.
    * **Binary info_request**: same as favourable.

    All tests use ``complaint_id`` as the blocking factor.
    """
    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    # Only persona-rewritten observations (condition == original with persona variants)
    if "persona_id" not in df.columns:
        return {"skipped": True, "reason": "no persona_id column"}

    results: Dict[str, Any] = {}

    # --- Friedman on outcome_tier ---
    results["friedman_tier"] = _friedman_with_posthoc(
        df, outcome_col="outcome_tier", label="outcome_tier"
    )

    # --- Cochran's Q on favourable ---
    results["cochran_favourable"] = _cochran_q_with_posthoc(
        df, outcome_col="favourable", label="favourable"
    )

    # --- Cochran's Q on info_request ---
    results["cochran_info_request"] = _cochran_q_with_posthoc(
        df, outcome_col="info_request", label="info_request"
    )

    return results


def _friedman_with_posthoc(
    df: pd.DataFrame,
    outcome_col: str,
    label: str,
) -> Dict[str, Any]:
    """Friedman test + post-hoc paired Wilcoxon with FDR."""
    try:
        pivot = df.pivot_table(
            index="complaint_id", columns="persona_id",
            values=outcome_col, aggfunc="first",
        ).dropna()

        if pivot.shape[1] < 3 or pivot.shape[0] < 5:
            return {"skipped": True, "reason": "insufficient complete cases"}

        # Check for variation
        col_means = pivot.mean(axis=0)
        if col_means.nunique() <= 1:
            return {"skipped": True, "reason": "no variation across personas"}

        groups = [pivot[col].values for col in pivot.columns]
        ftest = friedman_test(*groups)

        result: Dict[str, Any] = {
            "test": "Friedman",
            "statistic": ftest["statistic"],
            "p_value": ftest["p_value"],
            "n_complaints": pivot.shape[0],
            "n_personas": pivot.shape[1],
        }

        # Post-hoc: pairwise Wilcoxon signed-rank with FDR
        if ftest["p_value"] < 0.05:
            posthoc = _pairwise_wilcoxon(pivot)
            result["posthoc"] = posthoc

        return result

    except Exception as e:
        logger.warning("Friedman (%s) failed: %s", label, e)
        return {"skipped": True, "reason": str(e)}


def _cochran_q_with_posthoc(
    df: pd.DataFrame,
    outcome_col: str,
    label: str,
) -> Dict[str, Any]:
    """Cochran's Q test + post-hoc pairwise McNemar with FDR."""
    try:
        pivot = df.pivot_table(
            index="complaint_id", columns="persona_id",
            values=outcome_col, aggfunc="first",
        ).dropna()

        if pivot.shape[1] < 3 or pivot.shape[0] < 5:
            return {"skipped": True, "reason": "insufficient complete cases"}

        col_sums = pivot.sum(axis=0)
        if col_sums.nunique() <= 1:
            return {"skipped": True, "reason": "no variation across personas"}

        from statsmodels.stats.contingency_tables import cochrans_q
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            q_result = cochrans_q(pivot.values)

        result: Dict[str, Any] = {
            "test": "Cochran's Q",
            "statistic": float(q_result.statistic),
            "p_value": float(q_result.pvalue),
            "n_complaints": pivot.shape[0],
            "n_personas": pivot.shape[1],
        }

        # Post-hoc: pairwise McNemar with FDR
        if q_result.pvalue < 0.05:
            posthoc = _pairwise_mcnemar_binary(pivot)
            result["posthoc"] = posthoc

        return result

    except Exception as e:
        logger.warning("Cochran's Q (%s) failed: %s", label, e)
        return {"skipped": True, "reason": str(e)}


def _pairwise_wilcoxon(pivot: pd.DataFrame, max_pairs: int = 100) -> Dict[str, Any]:
    """Pairwise Wilcoxon signed-rank tests with FDR on pivot table columns."""
    cols = list(pivot.columns)
    pairs = []
    p_values = []

    # Limit to top-N most extreme personas to avoid combinatorial explosion
    means = pivot.mean()
    if len(cols) > 20:
        extremes = list(means.nsmallest(10).index) + list(means.nlargest(10).index)
        cols = sorted(set(extremes))

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if len(pairs) >= max_pairs:
                break
            a = pivot[cols[i]].values
            b = pivot[cols[j]].values
            diff = a - b
            if np.all(diff == 0):
                continue
            try:
                stat, p = sp_stats.wilcoxon(a, b, zero_method="wilcox")
                pairs.append({
                    "persona_1": str(cols[i]),
                    "persona_2": str(cols[j]),
                    "mean_1": float(a.mean()),
                    "mean_2": float(b.mean()),
                    "statistic": float(stat),
                    "p_value": float(p),
                })
                p_values.append(p)
            except Exception as e:
                logger.debug("Wilcoxon test failed: %s", e)

    if p_values:
        adj = fdr_correction(p_values)
        for pair, p_adj in zip(pairs, adj):
            pair["p_adjusted"] = float(p_adj)
            pair["significant"] = float(p_adj) < 0.05

    n_sig = sum(1 for p in pairs if p.get("significant", False))
    return {
        "n_pairs_tested": len(pairs),
        "n_significant": n_sig,
        "pairs": pairs,
    }


def _pairwise_mcnemar_binary(pivot: pd.DataFrame, max_pairs: int = 100) -> Dict[str, Any]:
    """Pairwise McNemar tests with FDR on binary pivot table columns."""
    from src.statistics import mcnemar_test

    cols = list(pivot.columns)
    pairs = []
    p_values = []

    # Limit to extremes
    means = pivot.mean()
    if len(cols) > 20:
        extremes = list(means.nsmallest(10).index) + list(means.nlargest(10).index)
        cols = sorted(set(extremes))

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if len(pairs) >= max_pairs:
                break
            a = pivot[cols[i]].values.astype(int)
            b = pivot[cols[j]].values.astype(int)
            table = np.array([
                [(a & b).sum(), (a & ~b).sum()],
                [(~a & b).sum(), (~a & ~b).sum()],
            ])
            if table[0, 1] + table[1, 0] == 0:
                continue
            try:
                test = mcnemar_test(table)
                pairs.append({
                    "persona_1": str(cols[i]),
                    "persona_2": str(cols[j]),
                    "rate_1": float(a.mean()),
                    "rate_2": float(b.mean()),
                    "statistic": test["statistic"],
                    "p_value": test["p_value"],
                })
                p_values.append(test["p_value"])
            except Exception as e:
                logger.debug("McNemar test failed: %s", e)

    if p_values:
        adj = fdr_correction(p_values)
        for pair, p_adj in zip(pairs, adj):
            pair["p_adjusted"] = float(p_adj)
            pair["significant"] = float(p_adj) < 0.05

    n_sig = sum(1 for p in pairs if p.get("significant", False))
    return {
        "n_pairs_tested": len(pairs),
        "n_significant": n_sig,
        "pairs": pairs,
    }


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------

def run_repeated_measures_analysis(
    baseline_df: pd.DataFrame,
    persona_decisions_df: Optional[pd.DataFrame],
    neutralized_df: Optional[pd.DataFrame],
    personas_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    models: List[str],
    use_cache: bool = True,
    cache_dir: str = "./cache",
) -> Dict[str, Any]:
    """Run the full repeated-measures causal analysis suite.

    Parameters
    ----------
    baseline_df : Baseline remedy decisions.
    persona_decisions_df : Persona-rewritten decisions.
    neutralized_df : Neutralized decisions.
    personas_df : Persona classifications.
    severity_df : Severity metadata.
    models : List of LLM model identifiers.
    use_cache : If True (default), check for cached results before computing.
    cache_dir : Directory for analysis cache files.

    Returns
    -------
    Dict with all model results, ready for reporting.
    """
    # Check cache first
    if use_cache:
        from src.analysis_cache import get_analysis_cache
        cache = get_analysis_cache(cache_dir)
        dataframes = {
            "baseline_df": baseline_df,
            "persona_decisions_df": persona_decisions_df,
            "neutralized_df": neutralized_df,
            "personas_df": personas_df,
            "severity_df": severity_df,
        }
        params = {"models": models}
        cached_result = cache.get("repeated_measures", dataframes, params)
        if cached_result is not None:
            return cached_result

    logger.info("Running repeated-measures causal analysis...")

    # Suppress known warnings for the entire analysis.  warnings.catch_warnings()
    # is NOT thread-safe, so per-function guards inside ThreadPoolExecutor workers
    # are unreliable.  Setting the filters here in the main thread ensures they
    # are in effect for all child threads (they share the global filter list).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning,
            message=".*Precision loss.*catastrophic cancellation.*",
        )
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
        try:
            from statsmodels.tools.sm_exceptions import ValueWarning as _SmValueWarning
            warnings.filterwarnings("ignore", category=_SmValueWarning)
        except ImportError:
            pass
        warnings.filterwarnings(
            "ignore",
            message=".*covariance of constraints does not have full rank.*",
        )

        result = _run_repeated_measures_inner(
            baseline_df, persona_decisions_df, neutralized_df,
            personas_df, severity_df, models,
        )
        # Cache the result
        if use_cache:
            cache.put("repeated_measures", dataframes, params, result)
        return result


def _run_repeated_measures_inner(
    baseline_df: pd.DataFrame,
    persona_decisions_df: Optional[pd.DataFrame],
    neutralized_df: Optional[pd.DataFrame],
    personas_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    models: List[str],
) -> Dict[str, Any]:
    """Inner implementation of run_repeated_measures_analysis (called inside warning guard)."""
    # Build long format
    long = build_long_format(
        baseline_df, persona_decisions_df, neutralized_df,
        personas_df, severity_df,
    )
    if long.empty:
        logger.warning("Long-format data is empty — skipping repeated-measures analysis")
        return {"skipped": True, "reason": "empty long-format data"}

    logger.info("  Long-format dataset: %d rows, %d complaints, %d personas",
                len(long), long["complaint_id"].nunique(),
                long["persona_id"].nunique() if "persona_id" in long.columns else 0)

    results: Dict[str, Any] = {
        "data_summary": {
            "n_rows": len(long),
            "n_complaints": int(long["complaint_id"].nunique()),
            "n_personas": int(long["persona_id"].nunique()) if "persona_id" in long.columns else 0,
            "n_models": int(long["model"].nunique()),
            "conditions": sorted(long["condition"].unique().tolist()),
            "severity_distribution": long["severity_bin"].value_counts().to_dict(),
        },
    }

    # Define the six sub-analyses that run per model
    def _run_model_suite(model_name: str) -> Tuple[str, Dict[str, Any]]:
        model_long = long[long["model"] == model_name]
        if len(model_long) < 20:
            return model_name, {"skipped": True, "reason": "insufficient data"}

        mr: Dict[str, Any] = {}
        mr["ordinal_model"] = ordinal_mixed_model(long, model_filter=model_name)
        mr["conditional_logistic_favourable"] = conditional_logistic_favourable(
            long, model_filter=model_name,
        )
        mr["logistic_gee_favourable"] = logistic_mixed_model_favourable(
            long, model_filter=model_name,
        )
        mr["logistic_gee_info_request"] = logistic_mixed_model_info_request(
            long, model_filter=model_name,
        )
        mr["linear_mixed_model"] = linear_mixed_model_tier(long, model_filter=model_name)
        mr["nonparametric"] = repeated_measures_nonparametric(long, model_filter=model_name)
        return model_name, mr

    # Run per-model analyses in parallel
    analysis_targets = list(models)
    if len(models) >= 2:
        analysis_targets.append("__pooled__")

    with ThreadPoolExecutor() as executor:
        futures = {}
        for target in analysis_targets:
            if target == "__pooled__":
                futures[executor.submit(_run_pooled_suite, long)] = "__pooled__"
            else:
                futures[executor.submit(_run_model_suite, target)] = target

        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Repeated measures", leave=False, delay=0.5,
        ):
            key = futures[future]
            if key == "__pooled__":
                results["pooled"] = future.result()
            else:
                model_name, mr = future.result()
                results[model_name] = mr

    logger.info("Repeated-measures causal analysis complete.")
    return results


def _run_pooled_suite(long: pd.DataFrame) -> Dict[str, Any]:
    """Run the pooled (all-models) repeated-measures suite."""
    pooled: Dict[str, Any] = {}
    pooled["ordinal_model"] = ordinal_mixed_model(long)
    pooled["logistic_gee_favourable"] = logistic_mixed_model_favourable(long)
    pooled["logistic_gee_info_request"] = logistic_mixed_model_info_request(long)
    pooled["linear_mixed_model"] = linear_mixed_model_tier(long)
    pooled["nonparametric"] = repeated_measures_nonparametric(long)
    return pooled


# ---------------------------------------------------------------------------
# Helpers (persona dimension parsing & model column preparation)
# ---------------------------------------------------------------------------


def _prepare_model_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare standard model columns: persona dims, severity, condition, interaction.

    Adds sev_ord, condition_code, persona_score, persona_x_severity columns
    to *df* (in-place) and creates model dummies if multiple LLM models are
    present.

    Returns
    -------
    df : DataFrame with new columns appended (may be a new object if dummies added).
    exog_cols : List of exogenous column names ready for model fitting.
    """
    _add_persona_dims(df)
    df["sev_ord"] = df["severity_bin"].map(_SEVERITY_ORD_MAP).fillna(1).astype(int)
    df["condition_code"] = (df["condition"] == "neutralised").astype(int)
    _add_persona_score(df)
    df["persona_x_severity"] = df["persona_score"] * df["sev_ord"]

    exog_cols = ["sev_ord", "condition_code", "persona_score", "persona_x_severity"]
    if df["model"].nunique() >= 2:
        model_dummies = pd.get_dummies(df["model"], prefix="mdl", drop_first=True, dtype=float)
        df = pd.concat([df, model_dummies], axis=1)
        exog_cols += list(model_dummies.columns)

    return df, exog_cols


def _add_persona_dims(df: pd.DataFrame) -> None:
    """Parse persona_id (e.g. 'young_urban_college_high') into dimension columns.

    Uses :func:`parse_persona_label` to correctly handle multi-word values
    like ``high_school`` instead of naive underscore splitting.
    """
    if "persona_id" not in df.columns:
        return

    def _extract_dims(x):
        d = parse_persona_label(str(x)) if pd.notna(x) else {}
        return d.get("age"), d.get("geographic"), d.get("education"), d.get("socioeconomic")

    extracted = df["persona_id"].apply(_extract_dims)
    df["p_age"] = extracted.str[0]
    df["p_geo"] = extracted.str[1]
    df["p_edu"] = extracted.str[2]
    df["p_ses"] = extracted.str[3]


def _add_persona_score(df: pd.DataFrame) -> None:
    """Create a numeric persona 'disadvantage' score for interaction terms.

    Higher = more traditionally disadvantaged persona:
    old > young, rural > suburban > urban, high_school > college, low > middle > high.
    """
    age_map = {"young": 0, "old": 1}
    geo_map = {"urban": 0, "suburban": 1, "rural": 2}
    edu_map = {"college": 0, "high_school": 1}
    ses_map = {"high": 0, "middle": 1, "low": 2}

    score = pd.Series(0.0, index=df.index)
    if "p_age" in df.columns:
        score += df["p_age"].map(age_map).fillna(0)
    if "p_geo" in df.columns:
        score += df["p_geo"].map(geo_map).fillna(0)
    if "p_edu" in df.columns:
        score += df["p_edu"].map(edu_map).fillna(0)
    if "p_ses" in df.columns:
        score += df["p_ses"].map(ses_map).fillna(0)

    # Normalise to 0-1 range
    df["persona_score"] = score / _MAX_PERSONA_SCORE
