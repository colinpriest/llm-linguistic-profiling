"""Bayesian hierarchical robustness layer for severity amplification.

Fits two Stan models per (LLM model x severity metric) combination:

* **B1** – hierarchical logistic regression for Tier-2 indicator (Y2)
* **B2** – hierarchical ordinal cumulative-logit for Tier 0/1/2

Both use a non-centred complaint-level random intercept.  Dimension-level
predictors (age, geographic, education, socioeconomic), severity-band
dummies, and their interactions form the shared design matrix.

The module computes posterior amplification estimands:

    D(s) = max − min predicted P(Tier2) across persona dimension levels
    A    = D(high) − D(low)
    P(A > 0)

Results are treated as non-confirmatory robustness checks.

Public API
----------
run_bayes_robustness  Run both Bayesian models and return summary dict.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ["low", "medium", "high"]
DIMENSIONS = ["age", "geographic", "education", "socioeconomic"]

# Reference categories (dropped for dummy coding)
_REFERENCE_LEVELS: Dict[str, str] = {
    "age": "young",
    "geographic": "urban",
    "education": "high_school",
    "socioeconomic": "low",
}
_SEVERITY_REF = "low"

# Sampling defaults
# 1000 warmup needed for mass matrix adaptation: the design matrix has
# ~30-40 columns (persona dummies + severity dummies + interactions)
# plus J complaint-level random effects.  750 warmup was insufficient —
# chains failed to converge (rhat>1.2, ess<15) on issue_severity runs.
_CHAINS = 4
_ITER_SAMPLING = 1000
_ITER_WARMUP = 1000
_ADAPT_DELTA = 0.95
_MAX_TREEDEPTH = 12

# Stan model paths (relative to repo root)
_STAN_DIR = Path(__file__).resolve().parent.parent / "models" / "stan"
_LOGIT_STAN = _STAN_DIR / "tier2_hier_logit.stan"
_ORDINAL_STAN = _STAN_DIR / "tier3_hier_ordinal.stan"


# ------------------------------------------------------------------
# Design matrix
# ------------------------------------------------------------------

def _get_dimension_levels(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return observed levels for each dimension, sorted for determinism."""
    levels: Dict[str, List[str]] = {}
    for dim in DIMENSIONS:
        if dim in df.columns:
            vals = sorted(df[dim].dropna().unique().tolist())
            levels[dim] = vals
    return levels


def _build_design_matrix(
    df: pd.DataFrame,
    severity_metric: str,
) -> Tuple[np.ndarray, List[str]]:
    """Build the shared design matrix X for both Stan models.

    Columns:
        - persona dimension dummies (k-1 per dimension, reference dropped)
        - severity band dummies (k-1, reference = low)
        - interactions: each persona dummy x each severity dummy

    Returns (X array [N, K], column_names list).
    """
    dim_levels = _get_dimension_levels(df)
    col_names: List[str] = []
    parts: List[np.ndarray] = []

    # 1. Persona dimension dummies
    persona_dummy_names: List[str] = []
    for dim in DIMENSIONS:
        if dim not in dim_levels:
            continue
        ref = _REFERENCE_LEVELS.get(dim, dim_levels[dim][0])
        for level in dim_levels[dim]:
            if level == ref:
                continue
            col = f"{dim}__{level}"
            persona_dummy_names.append(col)
            parts.append((df[dim] == level).astype(float).values[:, None])

    # 2. Severity band dummies
    sev_values = sorted(df[severity_metric].dropna().unique().tolist())
    sev_dummy_names: List[str] = []
    for level in sev_values:
        if level == _SEVERITY_REF:
            continue
        col = f"sev__{level}"
        sev_dummy_names.append(col)
        parts.append((df[severity_metric] == level).astype(float).values[:, None])

    # 3. Interactions
    interaction_names: List[str] = []
    for p_name in persona_dummy_names:
        p_idx = persona_dummy_names.index(p_name)
        p_vec = parts[p_idx]
        for s_name in sev_dummy_names:
            s_idx = len(persona_dummy_names) + sev_dummy_names.index(s_name)
            s_vec = parts[s_idx]
            interaction_names.append(f"{p_name}__x__{s_name}")
            parts.append(p_vec * s_vec)

    col_names = persona_dummy_names + sev_dummy_names + interaction_names
    X = np.hstack(parts) if parts else np.empty((len(df), 0))

    logger.info("Design matrix: N=%d, K=%d columns: %s", len(df), X.shape[1], col_names)
    return X, col_names


def _build_prediction_templates(
    dim_levels: Dict[str, List[str]],
    sev_dummy_names: List[str],
    persona_dummy_names: List[str],
    col_names: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build X-row templates for every (dimension, level, severity_band).

    Returns: {severity_band: {dim__level: x_row}} for prediction.
    Each template sets one persona dimension to its level (others at reference)
    and one severity band.
    """
    K = len(col_names)
    col_idx = {name: i for i, name in enumerate(col_names)}

    templates: Dict[str, Dict[str, np.ndarray]] = {}

    for sev_band in ["low"] + [s.replace("sev__", "") for s in sev_dummy_names]:
        band_templates: Dict[str, np.ndarray] = {}
        for dim in DIMENSIONS:
            if dim not in dim_levels:
                continue
            for level in dim_levels[dim]:
                x = np.zeros(K)
                # Set persona dummy
                dummy_name = f"{dim}__{level}"
                if dummy_name in col_idx:
                    x[col_idx[dummy_name]] = 1.0
                # Set severity dummy
                sev_name = f"sev__{sev_band}"
                if sev_name in col_idx:
                    x[col_idx[sev_name]] = 1.0
                # Set interaction
                if dummy_name in col_idx and sev_name in col_idx:
                    int_name = f"{dummy_name}__x__{sev_name}"
                    if int_name in col_idx:
                        x[col_idx[int_name]] = 1.0
                band_templates[f"{dim}__{level}"] = x
        templates[sev_band] = band_templates

    return templates


# ------------------------------------------------------------------
# Stan fitting
# ------------------------------------------------------------------

def _fit_stan_model(
    stan_file: Path,
    stan_data: Dict[str, Any],
    fit_id: str,
    seed: int = 42,
) -> Any:
    """Compile and fit a Stan model. Returns CmdStanMCMC object.

    Raises
    ------
    RuntimeError
        If the Stan model cannot be compiled or sampling fails, with
        an informative message including the fit_id and stan file path.
    """
    from cmdstanpy import CmdStanModel

    if not stan_file.exists():
        raise RuntimeError(
            f"Stan model file not found: {stan_file}\n"
            f"Fit ID: {fit_id}\n"
            f"Ensure the models/stan/ directory contains the required .stan files."
        )

    logger.info(
        "Compiling Stan model: %s (this may take 1-2 minutes)...",
        stan_file.name,
    )
    try:
        model = CmdStanModel(stan_file=str(stan_file))
    except Exception as e:
        raise RuntimeError(
            f"Stan compilation failed for {stan_file.name}\n"
            f"Fit ID: {fit_id}\n"
            f"Ensure CmdStan is installed (run install-stan.py).\n"
            f"Original error: {e}"
        ) from e
    logger.info("Compilation complete: %s", stan_file.name)

    n_params = stan_data.get("K", 0) + stan_data.get("J", 0)
    logger.info(
        "Sampling %s (chains=%d, iter=%d+%d, adapt_delta=%.2f, ~%d parameters) "
        "— this will take a while...",
        fit_id, _CHAINS, _ITER_WARMUP, _ITER_SAMPLING, _ADAPT_DELTA, n_params,
    )
    try:
        fit = model.sample(
            data=stan_data,
            chains=_CHAINS,
            iter_sampling=_ITER_SAMPLING,
            iter_warmup=_ITER_WARMUP,
            seed=seed,
            adapt_delta=_ADAPT_DELTA,
            max_treedepth=_MAX_TREEDEPTH,
            show_progress=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Stan sampling failed for {fit_id}\n"
            f"Model: {stan_file.name}\n"
            f"Data dimensions: N={stan_data.get('N')}, K={stan_data.get('K')}, "
            f"J={stan_data.get('J')}\n"
            f"Original error: {e}"
        ) from e

    return fit


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def _check_diagnostics(fit: Any) -> Dict[str, Any]:
    """Extract MCMC diagnostics from a CmdStanMCMC fit."""
    diag = fit.diagnose()

    summary = fit.summary()
    rhat_vals = summary["R_hat"].dropna()
    ess_vals = summary["N_Eff"].dropna() if "N_Eff" in summary.columns else summary.get("ESS_bulk", pd.Series(dtype=float)).dropna()

    max_rhat = float(rhat_vals.max()) if len(rhat_vals) > 0 else float("nan")
    min_ess = float(ess_vals.min()) if len(ess_vals) > 0 else 0.0

    # Count divergences and treedepth hits from sampler diagnostics
    n_divergent = 0
    n_treedepth = 0
    try:
        sampler_params = fit.method_variables()
        if "divergent__" in sampler_params:
            n_divergent = int(sampler_params["divergent__"].sum())
        if "treedepth__" in sampler_params:
            n_treedepth = int((sampler_params["treedepth__"] >= _MAX_TREEDEPTH).sum())
    except Exception:
        # Parse from diagnose string as fallback
        if "divergent" in diag.lower():
            try:
                for line in diag.split("\n"):
                    if "divergent" in line.lower() and any(c.isdigit() for c in line):
                        nums = [int(s) for s in line.split() if s.isdigit()]
                        if nums:
                            n_divergent = nums[0]
            except Exception:
                pass

    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": n_divergent,
        "treedepth_hits": n_treedepth,
        "diagnose_text": diag,
    }


def _gate_fit(diagnostics: Dict[str, Any]) -> str:
    """Apply acceptance gate. Returns 'OK', 'WARN', or 'FAIL'."""
    rhat = diagnostics.get("max_rhat", float("nan"))
    ess = diagnostics.get("min_ess_bulk", 0)
    div = diagnostics.get("divergences", 0)
    td = diagnostics.get("treedepth_hits", 0)

    if not np.isfinite(rhat) or rhat > 1.05 or ess < 100:
        return "FAIL"
    if rhat > 1.01 or ess < 400:
        return "FAIL"
    if div > 5:
        return "FAIL"

    if div > 0 or td > 0:
        return "WARN"

    return "OK"


# ------------------------------------------------------------------
# Amplification estimands
# ------------------------------------------------------------------

def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _compute_amplification_logistic(
    fit: Any,
    col_names: List[str],
    dim_levels: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute D(s) and A from logistic model posterior draws.

    D(s) = max - min predicted P(Tier2) across persona levels (per dimension)
    A    = D(high) - D(low)
    """
    alpha_draws = fit.stan_variable("alpha")         # (n_draws,)
    beta_draws = fit.stan_variable("beta")           # (n_draws, K)
    n_draws = len(alpha_draws)

    # Build severity dummy names
    sev_dummy_names = [c for c in col_names if c.startswith("sev__")]
    persona_dummy_names = [c for c in col_names if not c.startswith("sev__") and "__x__" not in c]

    templates = _build_prediction_templates(
        dim_levels, sev_dummy_names, persona_dummy_names, col_names,
    )

    # Compute per-dimension amplification
    results_by_dim: Dict[str, Dict[str, Any]] = {}

    for dim in DIMENSIONS:
        if dim not in dim_levels:
            continue

        levels = dim_levels[dim]

        # D(s) for each severity band
        D_draws: Dict[str, np.ndarray] = {}
        for sev_band, band_templates in templates.items():
            # Collect predicted P(Tier2) for each level of this dimension
            probs = []
            for level in levels:
                key = f"{dim}__{level}"
                if key not in band_templates:
                    continue
                x = band_templates[key]
                eta = alpha_draws + beta_draws @ x  # (n_draws,)
                p = _expit(eta)
                probs.append(p)

            if len(probs) < 2:
                D_draws[sev_band] = np.zeros(n_draws)
                continue

            probs_arr = np.stack(probs, axis=0)  # (n_levels, n_draws)
            D_draws[sev_band] = probs_arr.max(axis=0) - probs_arr.min(axis=0)

        # A = D(high) - D(low)
        D_high = D_draws.get("high", np.zeros(n_draws))
        D_low = D_draws.get("low", np.zeros(n_draws))
        A_draws = D_high - D_low

        results_by_dim[dim] = {
            "A_draws": A_draws.tolist(),
            "A_mean": float(np.mean(A_draws)),
            "A_CrI_low": float(np.percentile(A_draws, 2.5)),
            "A_CrI_high": float(np.percentile(A_draws, 97.5)),
            "P_A_gt_0": float(np.mean(A_draws > 0)),
            "D_low_mean": float(np.mean(D_low)),
            "D_high_mean": float(np.mean(D_high)),
        }

    return results_by_dim


def _compute_amplification_ordinal(
    fit: Any,
    col_names: List[str],
    dim_levels: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Compute amplification from ordinal model.

    P(Tier2) = P(y=3) = 1 - logistic(c[2] - eta)

    Note: the ordinal model has no separate alpha; the intercept is
    absorbed into the ordered cutpoints ``c``.
    """
    beta_draws = fit.stan_variable("beta")           # (n_draws, K)
    c_draws = fit.stan_variable("c")                 # (n_draws, 2)
    n_draws = len(beta_draws)

    sev_dummy_names = [c for c in col_names if c.startswith("sev__")]
    persona_dummy_names = [c for c in col_names if not c.startswith("sev__") and "__x__" not in c]

    templates = _build_prediction_templates(
        dim_levels, sev_dummy_names, persona_dummy_names, col_names,
    )

    results_by_dim: Dict[str, Dict[str, Any]] = {}

    for dim in DIMENSIONS:
        if dim not in dim_levels:
            continue

        levels = dim_levels[dim]

        D_draws: Dict[str, np.ndarray] = {}
        for sev_band, band_templates in templates.items():
            probs = []
            for level in levels:
                key = f"{dim}__{level}"
                if key not in band_templates:
                    continue
                x = band_templates[key]
                eta = beta_draws @ x
                # P(y=3) = 1 - logistic(c[2] - eta)
                p_tier2 = 1.0 - _expit(c_draws[:, 1] - eta)
                probs.append(p_tier2)

            if len(probs) < 2:
                D_draws[sev_band] = np.zeros(n_draws)
                continue

            probs_arr = np.stack(probs, axis=0)
            D_draws[sev_band] = probs_arr.max(axis=0) - probs_arr.min(axis=0)

        D_high = D_draws.get("high", np.zeros(n_draws))
        D_low = D_draws.get("low", np.zeros(n_draws))
        A_draws = D_high - D_low

        results_by_dim[dim] = {
            "A_draws": A_draws.tolist(),
            "A_mean": float(np.mean(A_draws)),
            "A_CrI_low": float(np.percentile(A_draws, 2.5)),
            "A_CrI_high": float(np.percentile(A_draws, 97.5)),
            "P_A_gt_0": float(np.mean(A_draws > 0)),
            "D_low_mean": float(np.mean(D_low)),
            "D_high_mean": float(np.mean(D_high)),
        }

    return results_by_dim


def _compute_ordinal_tier_probs(
    fit: Any,
    col_names: List[str],
    dim_levels: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Compute posterior P(tier=0/1/2) by severity band (population-average).

    Returns: {severity_band: {P_tier0: {mean, ci_low, ci_high}, ...}}
    """
    beta_draws = fit.stan_variable("beta")
    c_draws = fit.stan_variable("c")
    n_draws = len(beta_draws)

    sev_dummy_names = [c for c in col_names if c.startswith("sev__")]
    persona_dummy_names = [c for c in col_names if not c.startswith("sev__") and "__x__" not in c]

    templates = _build_prediction_templates(
        dim_levels, sev_dummy_names, persona_dummy_names, col_names,
    )

    tier_probs: Dict[str, Dict[str, Any]] = {}

    for sev_band, band_templates in templates.items():
        if not band_templates:
            continue

        # Average predicted probabilities across all persona templates
        all_p0, all_p1, all_p2 = [], [], []
        for key, x in band_templates.items():
            eta = beta_draws @ x
            p_le1 = _expit(c_draws[:, 0] - eta)  # P(y <= 1)
            p_le2 = _expit(c_draws[:, 1] - eta)  # P(y <= 2)
            p0 = p_le1
            p1 = p_le2 - p_le1
            p2 = 1.0 - p_le2
            all_p0.append(p0)
            all_p1.append(p1)
            all_p2.append(p2)

        # Average across persona templates
        avg_p0 = np.mean(np.stack(all_p0, axis=0), axis=0)
        avg_p1 = np.mean(np.stack(all_p1, axis=0), axis=0)
        avg_p2 = np.mean(np.stack(all_p2, axis=0), axis=0)

        def _summary(draws: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(np.mean(draws)),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
            }

        tier_probs[sev_band] = {
            "P_tier0": _summary(avg_p0),
            "P_tier1": _summary(avg_p1),
            "P_tier2": _summary(avg_p2),
        }

    return tier_probs


# ------------------------------------------------------------------
# Artifact storage
# ------------------------------------------------------------------

def _save_artifacts(
    fit_id: str,
    model_type: str,
    amplification: Dict[str, Any],
    diagnostics: Dict[str, Any],
    status: str,
    stan_file: Path,
    output_dir: Optional[Path],
) -> None:
    """Save posterior draws (Parquet) and model code to artifacts directory."""
    if output_dir is None:
        return

    bayes_dir = output_dir / "analysis" / "stats" / "bayes"
    bayes_dir.mkdir(parents=True, exist_ok=True)

    # Save amplification draws as Parquet
    draws_records = []
    for dim, dim_data in amplification.items():
        a_draws = dim_data.get("A_draws", [])
        for i, val in enumerate(a_draws):
            draws_records.append({"dimension": dim, "draw": i, "A": val})

    if draws_records:
        draws_df = pd.DataFrame(draws_records)
        parquet_path = bayes_dir / f"stats__bayes__amplification_draws__{fit_id}__v1.parquet"
        draws_df.to_parquet(parquet_path, index=False)

    # Copy Stan source
    stan_dest = bayes_dir / f"stats__bayes__model_code__{fit_id}__v1.stan"
    shutil.copy2(stan_file, stan_dest)


# ------------------------------------------------------------------
# Main public API
# ------------------------------------------------------------------

def run_bayes_robustness(
    baseline_sev: pd.DataFrame,
    models: List[str],
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run Bayesian robustness analysis (B1 logistic + B2 ordinal).

    Parameters
    ----------
    baseline_sev : DataFrame with columns: case_id, model, tier,
        monetary_severity, issue_severity, age, geographic, education,
        socioeconomic.
    models : List of LLM model identifiers to iterate over.
    seed : Random seed for Stan sampler.
    output_dir : Paper bundle output root (for artifact storage).

    Returns
    -------
    Nested dict: {model: {metric: {logistic: {...}, ordinal: {...}}}}
    """
    results: Dict[str, Any] = {}

    for model in models:
        logger.info("=" * 50)
        logger.info("Bayesian robustness for model: %s", model)
        model_df = baseline_sev[baseline_sev["model"] == model].copy()

        if len(model_df) < 50:
            logger.warning("Skipping %s: insufficient data (n=%d)", model, len(model_df))
            results[model] = {"skipped": True, "reason": "insufficient data"}
            continue

        model_results: Dict[str, Any] = {}

        for severity_metric in ["monetary_severity", "issue_severity"]:
            logger.info("── Severity metric: %s ──", severity_metric)

            # Filter to valid severity levels
            if severity_metric == "monetary_severity":
                df = model_df[model_df[severity_metric].isin(SEVERITY_LEVELS)].copy()
                subset = "amount_present"
            else:
                df = model_df[model_df[severity_metric].isin(SEVERITY_LEVELS)].copy()
                subset = "all"

            if len(df) < 30:
                logger.warning("Skipping %s/%s: insufficient data (n=%d)",
                               model, severity_metric, len(df))
                model_results[severity_metric] = {
                    "skipped": True, "reason": "insufficient data",
                }
                continue

            # Drop rows with missing dimensions
            for dim in DIMENSIONS:
                if dim in df.columns:
                    df = df[df[dim].notna()]

            # Build design matrix (shared by B1 and B2)
            X, col_names = _build_design_matrix(df, severity_metric)
            dim_levels = _get_dimension_levels(df)

            # Complaint ID integer mapping
            complaint_ids = df["case_id"].values
            unique_complaints = np.unique(complaint_ids)
            complaint_map = {cid: i + 1 for i, cid in enumerate(unique_complaints)}
            jj = np.array([complaint_map[cid] for cid in complaint_ids])
            J = len(unique_complaints)

            metric_result: Dict[str, Any] = {
                "n_obs": len(df),
                "n_complaints": J,
                "K": X.shape[1],
                "col_names": col_names,
                "subset": subset,
            }

            # ── B1: Hierarchical logistic (Tier2) ──
            logger.info("Fitting B1 (hierarchical logistic, Tier2)...")
            fit_id_b1 = f"B1__{model.replace('-', '')}__{severity_metric.replace('_', '')}"
            y_logistic = (df["tier"].values == 2).astype(int)
            stan_data_b1 = {
                "N": len(df),
                "K": X.shape[1],
                "X": X.tolist(),
                "J": J,
                "jj": jj.tolist(),
                "y": y_logistic.tolist(),
            }
            fit_b1 = _fit_stan_model(_LOGIT_STAN, stan_data_b1, fit_id_b1, seed)
            diag_b1 = _check_diagnostics(fit_b1)
            status_b1 = _gate_fit(diag_b1)
            amp_b1 = _compute_amplification_logistic(fit_b1, col_names, dim_levels)

            # Strip draws from JSON-serialisable result
            amp_b1_summary = {}
            for dim, dd in amp_b1.items():
                amp_b1_summary[dim] = {k: v for k, v in dd.items() if k != "A_draws"}

            metric_result["logistic"] = {
                "fit_id": fit_id_b1,
                "diagnostics": {k: v for k, v in diag_b1.items() if k != "diagnose_text"},
                "status": status_b1,
                "amplification": amp_b1_summary,
            }

            _save_artifacts(fit_id_b1, "logistic", amp_b1, diag_b1, status_b1,
                            _LOGIT_STAN, output_dir)
            logger.info("B1 status: %s (rhat=%.4f, ess=%.0f, div=%d)",
                        status_b1, diag_b1["max_rhat"],
                        diag_b1["min_ess_bulk"], diag_b1["divergences"])
            if status_b1 == "FAIL":
                raise RuntimeError(
                    f"B1 hierarchical logistic FAILED for {fit_id_b1}: "
                    f"rhat={diag_b1['max_rhat']:.4f}, "
                    f"ess={diag_b1['min_ess_bulk']:.0f}, "
                    f"div={diag_b1['divergences']}. "
                    f"Posterior draws are unreliable."
                )

            # ── B2: Hierarchical ordinal (Tier 0/1/2) ──
            logger.info("Fitting B2 (hierarchical ordinal, Tier 0/1/2)...")
            fit_id_b2 = f"B2__{model.replace('-', '')}__{severity_metric.replace('_', '')}"
            y_ordinal = (df["tier"].values + 1).astype(int)  # 0->1, 1->2, 2->3
            # Clip to valid range
            y_ordinal = np.clip(y_ordinal, 1, 3)

            stan_data_b2 = {
                "N": len(df),
                "K": X.shape[1],
                "X": X.tolist(),
                "J": J,
                "jj": jj.tolist(),
                "y": y_ordinal.tolist(),
            }
            fit_b2 = _fit_stan_model(_ORDINAL_STAN, stan_data_b2, fit_id_b2, seed)
            diag_b2 = _check_diagnostics(fit_b2)
            status_b2 = _gate_fit(diag_b2)
            amp_b2 = _compute_amplification_ordinal(fit_b2, col_names, dim_levels)
            tier_probs = _compute_ordinal_tier_probs(fit_b2, col_names, dim_levels)

            amp_b2_summary = {}
            for dim, dd in amp_b2.items():
                amp_b2_summary[dim] = {k: v for k, v in dd.items() if k != "A_draws"}

            metric_result["ordinal"] = {
                "fit_id": fit_id_b2,
                "diagnostics": {k: v for k, v in diag_b2.items() if k != "diagnose_text"},
                "status": status_b2,
                "amplification": amp_b2_summary,
                "tier_probs": tier_probs,
            }

            _save_artifacts(fit_id_b2, "ordinal", amp_b2, diag_b2, status_b2,
                            _ORDINAL_STAN, output_dir)
            logger.info("B2 status: %s (rhat=%.4f, ess=%.0f, div=%d)",
                        status_b2, diag_b2["max_rhat"],
                        diag_b2["min_ess_bulk"], diag_b2["divergences"])
            if status_b2 == "FAIL":
                raise RuntimeError(
                    f"B2 hierarchical ordinal FAILED for {fit_id_b2}: "
                    f"rhat={diag_b2['max_rhat']:.4f}, "
                    f"ess={diag_b2['min_ess_bulk']:.0f}, "
                    f"div={diag_b2['divergences']}. "
                    f"Posterior draws are unreliable."
                )

            model_results[severity_metric] = metric_result

        results[model] = model_results

    return results
