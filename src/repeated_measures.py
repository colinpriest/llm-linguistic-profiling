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

from src.statistics import bootstrap_ci, fdr_correction, friedman_test

logger = logging.getLogger(__name__)

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
    # Drop rows where severity is still unknown
    long["severity_bin"] = long["severity_bin"].where(
        long["severity_bin"].isin(["low", "medium", "high"]), other="medium"
    )

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

    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["outcome_tier"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or tier variation"}

    # Parse persona dimensions from persona_id
    _add_persona_dims(df)

    # Build design: persona dims + severity + condition + interaction
    df["sev_ord"] = df["severity_bin"].map({"low": 0, "medium": 1, "high": 2}).fillna(1).astype(int)
    df["condition_code"] = (df["condition"] == "neutralised").astype(int)

    # Interaction: severity × persona (use a product term on ordinal severity
    # and a summarised persona "disadvantage" score for parsimony)
    _add_persona_score(df)
    df["persona_x_severity"] = df["persona_score"] * df["sev_ord"]

    # If multiple LLM models present, add model dummy
    if df["model"].nunique() >= 2:
        model_dummies = pd.get_dummies(df["model"], prefix="mdl", drop_first=True, dtype=float)
        df = pd.concat([df, model_dummies], axis=1)
        exog_cols = (
            ["sev_ord", "condition_code", "persona_score", "persona_x_severity"]
            + list(model_dummies.columns)
        )
    else:
        exog_cols = ["sev_ord", "condition_code", "persona_score", "persona_x_severity"]

    exog = _drop_constant_columns(df[exog_cols].astype(float))
    endog = df["outcome_tier"].astype(float)

    # Fit ordinal model with clustered sandwich SEs on complaint_id
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            om = OrderedModel(endog, exog, distr="logit")
            # First pass: unclustered fit to get point estimates
            fit_plain = om.fit(method="bfgs", disp=False, maxiter=200)

        # Re-fit with clustered covariance (sandwich estimator on complaint_id)
        used_cluster = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_plain.get_robustcov_results(
                    cov_type="cluster",
                    groups=df["complaint_id"].values,
                )
            used_cluster = True
        except Exception:
            logger.info("Ordinal model: clustered SEs unavailable, using model-based SEs")
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

        param_names = list(exog.columns) + [f"threshold_{i}" for i in range(len(fit.params) - len(exog.columns))]

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
                "method": "repeated-measures ordinal with clustered SEs",
            },
        }

    except Exception as e:
        logger.warning("Ordinal mixed model failed: %s", e)
        return {"skipped": True, "reason": str(e)}


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

    Parameters
    ----------
    long_df : Long-format DataFrame.
    model_filter : Restrict to this LLM model.

    Returns
    -------
    Dict with coefficients, SEs, p-values, CIs, n_obs, n_strata.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit

    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    # Need within-complaint variation to estimate conditional logit
    complaint_var = df.groupby("complaint_id")["favourable"].nunique()
    informative = complaint_var[complaint_var > 1].index
    df = df[df["complaint_id"].isin(informative)]

    if len(df) < 20 or len(informative) < 5:
        return {"skipped": True, "reason": "insufficient within-complaint variation"}

    _add_persona_dims(df)

    df["sev_ord"] = df["severity_bin"].map({"low": 0, "medium": 1, "high": 2}).fillna(1).astype(int)
    df["condition_code"] = (df["condition"] == "neutralised").astype(int)
    _add_persona_score(df)
    df["persona_x_severity"] = df["persona_score"] * df["sev_ord"]

    exog_cols = ["sev_ord", "condition_code", "persona_score", "persona_x_severity"]
    if df["model"].nunique() >= 2:
        model_dummies = pd.get_dummies(df["model"], prefix="mdl", drop_first=True, dtype=float)
        df = pd.concat([df, model_dummies], axis=1)
        exog_cols += list(model_dummies.columns)

    exog = _drop_constant_columns(df[exog_cols].astype(float))
    endog = df["favourable"].astype(float)

    # Sort by complaint_id for grouping
    sort_idx = df["complaint_id"].argsort()
    exog = exog.iloc[sort_idx].reset_index(drop=True)
    endog = endog.iloc[sort_idx].reset_index(drop=True)
    groups = df["complaint_id"].iloc[sort_idx].reset_index(drop=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clogit = ConditionalLogit(endog, exog, groups=groups)
            fit = clogit.fit(disp=False, maxiter=200)

        ci_dict = {}
        try:
            ci_df = fit.conf_int()
            for name in exog.columns:
                if name in ci_df.index:
                    ci_dict[name] = [float(ci_df.loc[name].iloc[0]), float(ci_df.loc[name].iloc[1])]
        except Exception:
            for name, se in zip(exog.columns, fit.bse):
                coef = fit.params[exog.columns.get_loc(name)]
                ci_dict[name] = [float(coef - 1.96 * se), float(coef + 1.96 * se)]

        return {
            "coefficients": dict(zip(exog.columns, fit.params.tolist())),
            "std_errors": dict(zip(exog.columns, fit.bse.tolist())),
            "p_values": dict(zip(exog.columns, fit.pvalues.tolist())),
            "conf_int": ci_dict,
            "odds_ratios": dict(zip(exog.columns, np.exp(fit.params).tolist())),
            "n_obs": len(df),
            "n_strata": int(groups.nunique()),
            "n_informative_strata": len(informative),
            "log_likelihood": float(fit.llf) if hasattr(fit, "llf") else None,
            "model_info": {
                "type": "ConditionalLogit",
                "clustering": "complaint_id (stratum)",
                "method": "within-complaint conditional logistic regression",
            },
        }
    except Exception as e:
        logger.warning("Conditional logistic failed: %s", e)
        return {"skipped": True, "reason": str(e)}


def logistic_mixed_model_favourable(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Logistic mixed model with ``complaint_id`` random intercept for ``favourable``.

    More flexible than conditional logistic — allows severity and condition
    interactions plus complaint-level covariates.

    Uses GEE (Generalized Estimating Equations) with exchangeable
    correlation as a robust approximation to a full GLMM, since
    statsmodels does not expose a direct binomial GLMM.

    Parameters
    ----------
    long_df : Long-format DataFrame.
    model_filter : Restrict to this LLM model.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["favourable"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or outcome variation"}

    _add_persona_dims(df)
    df["sev_ord"] = df["severity_bin"].map({"low": 0, "medium": 1, "high": 2}).fillna(1).astype(int)
    df["condition_code"] = (df["condition"] == "neutralised").astype(int)
    _add_persona_score(df)
    df["persona_x_severity"] = df["persona_score"] * df["sev_ord"]

    exog_cols = ["sev_ord", "condition_code", "persona_score", "persona_x_severity"]
    if df["model"].nunique() >= 2:
        model_dummies = pd.get_dummies(df["model"], prefix="mdl", drop_first=True, dtype=float)
        df = pd.concat([df, model_dummies], axis=1)
        exog_cols += list(model_dummies.columns)

    # Sort by complaint_id for GEE grouping
    df = df.sort_values("complaint_id").reset_index(drop=True)
    exog = sm.add_constant(_drop_constant_columns(df[exog_cols].astype(float)))
    endog = df["favourable"].astype(float)
    groups = df["complaint_id"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gee_model = GEE(
                endog, exog, groups=groups,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            fit = gee_model.fit(maxiter=200)

        return {
            "coefficients": fit.params.to_dict(),
            "std_errors": fit.bse.to_dict(),
            "p_values": fit.pvalues.to_dict(),
            "conf_int": {
                name: [float(fit.conf_int().loc[name, 0]), float(fit.conf_int().loc[name, 1])]
                for name in fit.params.index
            },
            "odds_ratios": np.exp(fit.params).to_dict(),
            "n_obs": len(df),
            "n_clusters": int(groups.nunique()),
            "scale": float(fit.scale) if hasattr(fit, "scale") else None,
            "model_info": {
                "type": "GEE (Binomial, exchangeable correlation)",
                "clustering": "complaint_id",
                "method": "logistic GEE with complaint_id clustering (GLMM approximation)",
            },
        }
    except Exception as e:
        logger.warning("Logistic GEE (favourable) failed: %s", e)
        return {"skipped": True, "reason": str(e)}


# ---------------------------------------------------------------------------
# 2C. Info-request burden (binary GLMM)
# ---------------------------------------------------------------------------

def logistic_mixed_model_info_request(
    long_df: pd.DataFrame,
    model_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Logistic mixed model for ``info_request`` with ``complaint_id`` random intercept.

    Same GEE approach as the favourable model.
    """
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["info_request"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or outcome variation"}

    _add_persona_dims(df)
    df["sev_ord"] = df["severity_bin"].map({"low": 0, "medium": 1, "high": 2}).fillna(1).astype(int)
    df["condition_code"] = (df["condition"] == "neutralised").astype(int)
    _add_persona_score(df)
    df["persona_x_severity"] = df["persona_score"] * df["sev_ord"]

    exog_cols = ["sev_ord", "condition_code", "persona_score", "persona_x_severity"]
    if df["model"].nunique() >= 2:
        model_dummies = pd.get_dummies(df["model"], prefix="mdl", drop_first=True, dtype=float)
        df = pd.concat([df, model_dummies], axis=1)
        exog_cols += list(model_dummies.columns)

    df = df.sort_values("complaint_id").reset_index(drop=True)
    exog = sm.add_constant(_drop_constant_columns(df[exog_cols].astype(float)))
    endog = df["info_request"].astype(float)
    groups = df["complaint_id"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gee_model = GEE(
                endog, exog, groups=groups,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            fit = gee_model.fit(maxiter=200)

        return {
            "coefficients": fit.params.to_dict(),
            "std_errors": fit.bse.to_dict(),
            "p_values": fit.pvalues.to_dict(),
            "conf_int": {
                name: [float(fit.conf_int().loc[name, 0]), float(fit.conf_int().loc[name, 1])]
                for name in fit.params.index
            },
            "odds_ratios": np.exp(fit.params).to_dict(),
            "n_obs": len(df),
            "n_clusters": int(groups.nunique()),
            "scale": float(fit.scale) if hasattr(fit, "scale") else None,
            "model_info": {
                "type": "GEE (Binomial, exchangeable correlation)",
                "clustering": "complaint_id",
                "method": "logistic GEE with complaint_id clustering (GLMM approximation)",
            },
        }
    except Exception as e:
        logger.warning("Logistic GEE (info_request) failed: %s", e)
        return {"skipped": True, "reason": str(e)}


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
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    df = long_df.copy()
    if model_filter:
        df = df[df["model"] == model_filter]

    if len(df) < 30 or df["outcome_tier"].nunique() < 2:
        return {"skipped": True, "reason": "insufficient data or tier variation"}

    _add_persona_dims(df)
    df["sev_ord"] = df["severity_bin"].map({"low": 0, "medium": 1, "high": 2}).fillna(1).astype(int)
    df["condition_code"] = (df["condition"] == "neutralised").astype(int)
    _add_persona_score(df)
    df["persona_x_severity"] = df["persona_score"] * df["sev_ord"]

    exog_cols = ["sev_ord", "condition_code", "persona_score", "persona_x_severity"]
    if df["model"].nunique() >= 2:
        model_dummies = pd.get_dummies(df["model"], prefix="mdl", drop_first=True, dtype=float)
        df = pd.concat([df, model_dummies], axis=1)
        exog_cols += list(model_dummies.columns)

    import statsmodels.api as sm
    exog = sm.add_constant(_drop_constant_columns(df[exog_cols].astype(float)))
    endog = df["outcome_tier"].astype(float)
    groups = df["complaint_id"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lmm = MixedLM(endog, exog, groups=groups)
            try:
                fit = lmm.fit(reml=True, maxiter=200)
            except (np.linalg.LinAlgError, ValueError):
                try:
                    fit = lmm.fit(reml=False, maxiter=200)
                except (np.linalg.LinAlgError, ValueError):
                    # Last resort: use profile method with start_params
                    fit = lmm.fit(
                        reml=False, maxiter=200,
                        start_params=np.zeros(exog.shape[1] + 1),
                        method="powell",
                    )

        re_var = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, "iloc") else float(fit.cov_re)
        resid_var = float(fit.scale)
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0.0

        return {
            "coefficients": fit.fe_params.to_dict(),
            "std_errors": fit.bse_fe.to_dict(),
            "p_values": fit.pvalues.to_dict(),
            "conf_int": {
                name: [float(fit.conf_int().loc[name, 0]), float(fit.conf_int().loc[name, 1])]
                for name in fit.fe_params.index
            },
            "random_effects_variance": re_var,
            "residual_variance": resid_var,
            "icc": icc,
            "n_obs": int(fit.nobs),
            "n_clusters": int(groups.nunique()),
            "converged": fit.converged,
            "model_info": {
                "type": "Linear Mixed Model (REML)",
                "clustering": "complaint_id (random intercept)",
                "method": "LMM with complaint_id random intercept (robustness)",
                "icc_interpretation": (
                    f"ICC = {icc:.3f}: {icc*100:.1f}% of tier variance is "
                    f"between complaints (justifies clustering)"
                ),
            },
        }
    except Exception as e:
        logger.warning("Linear mixed model failed: %s", e)
        return {"skipped": True, "reason": str(e)}


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
            except Exception:
                pass

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
            except Exception:
                pass

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
) -> Dict[str, Any]:
    """Run the full repeated-measures causal analysis suite.

    Returns a structured dict with all model results, ready for reporting.
    """
    logger.info("Running repeated-measures causal analysis...")

    # Suppress known warnings for the entire analysis.  warnings.catch_warnings()
    # is NOT thread-safe, so per-function guards inside ThreadPoolExecutor workers
    # are unreliable.  Setting the filters here in the main thread ensures they
    # are in effect for all child threads (they share the global filter list).
    _warning_ctx = warnings.catch_warnings()
    _warning_ctx.__enter__()
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

    try:
        return _run_repeated_measures_inner(
            baseline_df, persona_decisions_df, neutralized_df,
            personas_df, severity_df, models,
        )
    finally:
        _warning_ctx.__exit__(None, None, None)


def _run_repeated_measures_inner(
    baseline_df, persona_decisions_df, neutralized_df,
    personas_df, severity_df, models,
):
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
# Helpers (persona dimension parsing)
# ---------------------------------------------------------------------------

def _drop_constant_columns(exog: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with zero variance (constant) to avoid rank deficiency."""
    varying = [c for c in exog.columns if exog[c].nunique() > 1]
    return exog[varying]


def _add_persona_dims(df: pd.DataFrame) -> None:
    """Parse persona_id (e.g. 'young_urban_college_high') into dimension columns."""
    if "persona_id" not in df.columns:
        return
    parts = df["persona_id"].str.split("_", expand=True)
    if parts.shape[1] >= 4:
        df["p_age"] = parts[0]
        df["p_geo"] = parts[1]
        df["p_edu"] = parts[2]
        df["p_ses"] = parts[3]


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

    # Normalise to 0-1 range (max possible = 1+2+1+2 = 6)
    df["persona_score"] = score / 6.0
