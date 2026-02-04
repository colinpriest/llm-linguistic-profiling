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
_LOGIT_FIXED_STAN = _STAN_DIR / "tier2_fixed_logit.stan"
_ORDINAL_FIXED_STAN = _STAN_DIR / "tier3_fixed_ordinal.stan"
_LOGIT_FLEX_STAN = _STAN_DIR / "tier2_hier_logit_flex.stan"
_ORDINAL_FLEX_STAN = _STAN_DIR / "tier3_hier_ordinal_flex.stan"

# If >50% of complaints have exactly 1 observation, the complaint-level
# random intercept is unidentifiable for most groups.  Fall back to
# fixed-effect models for that metric.
_SINGLETON_THRESHOLD = 0.50


# ------------------------------------------------------------------
# Design matrix
# ------------------------------------------------------------------

def _get_dimension_levels(
    df: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Return observed levels for each dimension, sorted for determinism."""
    dims = dimensions if dimensions is not None else DIMENSIONS
    levels: Dict[str, List[str]] = {}
    for dim in dims:
        if dim in df.columns:
            vals = sorted(df[dim].dropna().unique().tolist())
            levels[dim] = vals
    return levels


def _build_design_matrix(
    df: pd.DataFrame,
    severity_metric: str,
    dimensions: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build the shared design matrix X for both Stan models.

    Columns:
        - persona dimension dummies (k-1 per dimension, reference dropped)
        - severity band dummies (k-1, reference = low)
        - interactions: each persona dummy x each severity dummy

    Returns (X array [N, K], column_names list).
    """
    dims = dimensions if dimensions is not None else DIMENSIONS
    dim_levels = _get_dimension_levels(df, dims)
    col_names: List[str] = []
    parts: List[np.ndarray] = []

    # 1. Persona dimension dummies
    persona_dummy_names: List[str] = []
    for dim in dims:
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


def _validate_case_ids(df: pd.DataFrame, context: str) -> None:
    """Ensure case_id exists and is complete before mapping to integers."""
    if "case_id" not in df.columns:
        raise RuntimeError(f"Missing case_id column for Bayesian robustness ({context}).")
    n_missing = int(df["case_id"].isna().sum())
    if n_missing > 0:
        raise RuntimeError(
            f"{n_missing} rows missing case_id after filtering ({context}); "
            "cannot build complaint-level random effects."
        )


# ------------------------------------------------------------------
# Stan fitting
# ------------------------------------------------------------------

def _fit_stan_model(
    stan_file: Path,
    stan_data: Dict[str, Any],
    fit_id: str,
    seed: int = 42,
    adapt_delta: float = _ADAPT_DELTA,
    max_treedepth: int = _MAX_TREEDEPTH,
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
        fit_id, _CHAINS, _ITER_WARMUP, _ITER_SAMPLING, adapt_delta, n_params,
    )
    try:
        fit = model.sample(
            data=stan_data,
            chains=_CHAINS,
            iter_sampling=_ITER_SAMPLING,
            iter_warmup=_ITER_WARMUP,
            seed=seed,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
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
    """Apply acceptance gate. Returns 'OK', 'WARN', or 'FAIL'.

    Logic:
        FAIL — severe non-convergence or many divergences.
        WARN — mild/medium issues (slightly high Rhat, modest ESS, any divergences).
        OK   — clean diagnostics.
    """
    rhat = diagnostics.get("max_rhat", float("nan"))
    ess = diagnostics.get("min_ess_bulk", 0)
    div = diagnostics.get("divergences", 0)
    td = diagnostics.get("treedepth_hits", 0)

    # Hard fail: clearly non-converged or unstable chains
    if (not np.isfinite(rhat)) or rhat > 1.05 or ess < 100 or div > 5:
        return "FAIL"

    # Warning band: marginal convergence or any sampler pathologies
    if rhat > 1.01 or ess < 400 or div > 0 or td > 0:
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

    # Persist diagnostics (including diagnose_text) for post-mortem debugging
    diag_path = bayes_dir / f"stats__bayes__diagnostics__{fit_id}__v1.json"
    diag_payload = {
        "fit_id": fit_id,
        "model_type": model_type,
        "status": status,
        "diagnostics": diagnostics,
    }
    diag_path.write_text(json.dumps(diag_payload, indent=2))


# ------------------------------------------------------------------
# Fit helpers (used by fallback chain)
# ------------------------------------------------------------------

def _try_b1_b2(
    *,
    df: pd.DataFrame,
    X: np.ndarray,
    col_names: List[str],
    dim_levels: Dict[str, List[str]],
    b1_stan_file: Path,
    b2_stan_file: Path,
    stan_data_b1: Dict[str, Any],
    stan_data_b2: Dict[str, Any],
    fit_id_b1: str,
    fit_id_b2: str,
    seed: int,
    output_dir: Optional[Path],
    adapt_delta: float = _ADAPT_DELTA,
    max_treedepth: int = _MAX_TREEDEPTH,
    method_label: str = "hierarchical",
) -> Optional[Dict[str, Any]]:
    """Attempt B1 (logistic) + B2 (ordinal) fits.

    Returns metric_result dict on success, or None if either model FAILs.
    Failed diagnostics are logged at WARNING level but do not raise.
    """
    y_ordinal = (df["tier"].values + 1).astype(int)
    y_ordinal = np.clip(y_ordinal, 1, 3)
    stan_data_b2.setdefault("y", y_ordinal.tolist())

    metric_result: Dict[str, Any] = {}

    # ── B1: Logistic ──
    logger.info("Fitting B1 (%s logistic, Tier2)...", method_label)
    fit_b1 = _fit_stan_model(b1_stan_file, stan_data_b1, fit_id_b1, seed,
                             adapt_delta=adapt_delta, max_treedepth=max_treedepth)
    diag_b1 = _check_diagnostics(fit_b1)
    status_b1 = _gate_fit(diag_b1)
    logger.info("B1 status: %s (rhat=%.4f, ess=%.0f, div=%d) [%s]",
                status_b1, diag_b1["max_rhat"],
                diag_b1["min_ess_bulk"], diag_b1["divergences"], method_label)

    if status_b1 == "FAIL":
        logger.warning("B1 %s logistic FAILED for %s", method_label, fit_id_b1)
        _save_artifacts(fit_id_b1, "logistic", {}, diag_b1, status_b1,
                        b1_stan_file, output_dir)
        return None

    amp_b1 = _compute_amplification_logistic(fit_b1, col_names, dim_levels)
    amp_b1_summary = {dim: {k: v for k, v in dd.items() if k != "A_draws"}
                      for dim, dd in amp_b1.items()}
    diag_b1_payload = {k: v for k, v in diag_b1.items() if k != "diagnose_text"}
    if status_b1 != "OK":
        diag_b1_payload["diagnose_text"] = diag_b1.get("diagnose_text", "")

    metric_result["logistic"] = {
        "fit_id": fit_id_b1, "diagnostics": diag_b1_payload,
        "status": status_b1, "amplification": amp_b1_summary,
        "method": f"{method_label} logistic",
    }
    _save_artifacts(fit_id_b1, "logistic", amp_b1, diag_b1, status_b1,
                    b1_stan_file, output_dir)

    # ── B2: Ordinal ──
    logger.info("Fitting B2 (%s ordinal, Tier 0/1/2)...", method_label)
    fit_b2 = _fit_stan_model(b2_stan_file, stan_data_b2, fit_id_b2, seed,
                             adapt_delta=adapt_delta, max_treedepth=max_treedepth)
    diag_b2 = _check_diagnostics(fit_b2)
    status_b2 = _gate_fit(diag_b2)
    logger.info("B2 status: %s (rhat=%.4f, ess=%.0f, div=%d) [%s]",
                status_b2, diag_b2["max_rhat"],
                diag_b2["min_ess_bulk"], diag_b2["divergences"], method_label)

    if status_b2 == "FAIL":
        logger.warning("B2 %s ordinal FAILED for %s", method_label, fit_id_b2)
        _save_artifacts(fit_id_b2, "ordinal", {}, diag_b2, status_b2,
                        b2_stan_file, output_dir)
        return None

    amp_b2 = _compute_amplification_ordinal(fit_b2, col_names, dim_levels)
    tier_probs = _compute_ordinal_tier_probs(fit_b2, col_names, dim_levels)
    amp_b2_summary = {dim: {k: v for k, v in dd.items() if k != "A_draws"}
                      for dim, dd in amp_b2.items()}
    diag_b2_payload = {k: v for k, v in diag_b2.items() if k != "diagnose_text"}
    if status_b2 != "OK":
        diag_b2_payload["diagnose_text"] = diag_b2.get("diagnose_text", "")

    metric_result["ordinal"] = {
        "fit_id": fit_id_b2, "diagnostics": diag_b2_payload,
        "status": status_b2, "amplification": amp_b2_summary,
        "tier_probs": tier_probs,
        "method": f"{method_label} ordinal",
    }
    _save_artifacts(fit_id_b2, "ordinal", amp_b2, diag_b2, status_b2,
                    b2_stan_file, output_dir)

    return metric_result


def _try_per_dimension(
    *,
    df: pd.DataFrame,
    severity_metric: str,
    jj: np.ndarray,
    J: int,
    use_random_intercept: bool,
    fit_id_prefix: str,
    seed: int,
    output_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Fallback 2: fit separate per-dimension models.

    Each model has one demographic dimension × severity interactions,
    yielding ~4-8 columns instead of ~20+.  Results are merged.
    """
    combined_logistic: Dict[str, Any] = {"amplification": {}, "per_dim_fits": {}}
    combined_ordinal: Dict[str, Any] = {"amplification": {}, "tier_probs": {},
                                        "per_dim_fits": {}}
    all_ok = True

    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue

        X_dim, col_names_dim = _build_design_matrix(df, severity_metric, dimensions=[dim])
        dim_levels = _get_dimension_levels(df, dimensions=[dim])

        if X_dim.shape[1] == 0:
            logger.warning("Per-dimension %s: empty design matrix, skipping", dim)
            continue

        if use_random_intercept:
            b1_stan = _LOGIT_STAN
            b2_stan = _ORDINAL_STAN
            common_data = {"N": len(df), "K": X_dim.shape[1], "X": X_dim.tolist(),
                           "J": J, "jj": jj.tolist()}
        else:
            b1_stan = _LOGIT_FIXED_STAN
            b2_stan = _ORDINAL_FIXED_STAN
            common_data = {"N": len(df), "K": X_dim.shape[1], "X": X_dim.tolist()}

        y_logistic = (df["tier"].values == 2).astype(int).tolist()
        y_ordinal = np.clip((df["tier"].values + 1).astype(int), 1, 3).tolist()

        fit_id_b1 = f"{fit_id_prefix}__dim_{dim}__B1"
        fit_id_b2 = f"{fit_id_prefix}__dim_{dim}__B2"

        logger.info("Per-dimension fallback: fitting %s (K=%d)...", dim, X_dim.shape[1])

        result = _try_b1_b2(
            df=df, X=X_dim, col_names=col_names_dim, dim_levels=dim_levels,
            b1_stan_file=b1_stan, b2_stan_file=b2_stan,
            stan_data_b1={**common_data, "y": y_logistic},
            stan_data_b2={**common_data, "y": y_ordinal},
            fit_id_b1=fit_id_b1, fit_id_b2=fit_id_b2,
            seed=seed, output_dir=output_dir,
            method_label=f"per-dim({dim})",
        )

        if result is None:
            logger.warning("Per-dimension fallback: %s FAILED", dim)
            all_ok = False
            break

        # Merge amplification results
        combined_logistic["amplification"].update(result["logistic"]["amplification"])
        combined_logistic["per_dim_fits"][dim] = result["logistic"]
        combined_ordinal["amplification"].update(result["ordinal"]["amplification"])
        if "tier_probs" in result["ordinal"]:
            combined_ordinal["tier_probs"].update(result["ordinal"].get("tier_probs", {}))
        combined_ordinal["per_dim_fits"][dim] = result["ordinal"]

    if not all_ok:
        return None

    return {
        "logistic": {
            "method": "per-dimension logistic",
            "status": "OK",
            "amplification": combined_logistic["amplification"],
            "per_dim_fits": combined_logistic["per_dim_fits"],
        },
        "ordinal": {
            "method": "per-dimension ordinal",
            "status": "OK",
            "amplification": combined_ordinal["amplification"],
            "tier_probs": combined_ordinal["tier_probs"],
            "per_dim_fits": combined_ordinal["per_dim_fits"],
        },
    }


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

            initial_n = len(df)
            if initial_n == 0:
                logger.warning("Skipping %s/%s: no valid rows after severity filter",
                               model, severity_metric)
                model_results[severity_metric] = {
                    "skipped": True, "reason": "no data",
                }
                continue

            # Drop rows with missing dimensions (track losses per dimension)
            for dim in DIMENSIONS:
                if dim in df.columns:
                    before_dim = len(df)
                    df = df[df[dim].notna()]
                    removed = before_dim - len(df)
                    if before_dim > 0 and removed / before_dim > 0.2:
                        logger.warning(
                            "%s/%s: dropped %.1f%% (%d/%d) rows due to missing %s",
                            model, severity_metric, 100 * removed / before_dim,
                            removed, before_dim, dim,
                        )
                    elif removed > 0:
                        logger.info("%s/%s: dropped %d rows with missing %s",
                                    model, severity_metric, removed, dim)

            post_drop_n = len(df)
            total_removed = initial_n - post_drop_n
            if initial_n > 0 and total_removed / initial_n > 0.2:
                logger.warning(
                    "%s/%s: total row loss %.1f%% (%d→%d) after NA drops",
                    model, severity_metric, 100 * total_removed / initial_n,
                    initial_n, post_drop_n,
                )

            if post_drop_n < 30:
                logger.warning("Skipping %s/%s: insufficient data after NA drop (n=%d from %d)",
                               model, severity_metric, post_drop_n, initial_n)
                model_results[severity_metric] = {
                    "skipped": True, "reason": "insufficient data",
                }
                continue

            _validate_case_ids(df, f"{model}/{severity_metric}")

            # Build design matrix (shared by B1 and B2)
            X, col_names = _build_design_matrix(df, severity_metric)
            dim_levels = _get_dimension_levels(df)

            # Complaint ID integer mapping
            complaint_ids = df["case_id"].values
            unique_complaints = np.unique(complaint_ids)
            complaint_map = {cid: i + 1 for i, cid in enumerate(unique_complaints)}
            jj = np.array([complaint_map[cid] for cid in complaint_ids])
            J = len(unique_complaints)

            if J < 5:
                logger.warning("Skipping %s/%s: insufficient unique complaints (J=%d)",
                               model, severity_metric, J)
                model_results[severity_metric] = {
                    "skipped": True, "reason": "insufficient complaints",
                }
                continue

            # ── Complaint-level observation distribution ──
            obs_per_complaint = pd.Series(complaint_ids).value_counts()
            n_singletons = int((obs_per_complaint == 1).sum())
            pct_singletons = n_singletons / J
            logger.info(
                "%s/%s: obs-per-complaint distribution: "
                "min=%d, median=%.0f, mean=%.1f, max=%d | "
                "singletons: %d/%d (%.1f%%)",
                model, severity_metric,
                int(obs_per_complaint.min()),
                float(obs_per_complaint.median()),
                float(obs_per_complaint.mean()),
                int(obs_per_complaint.max()),
                n_singletons, J, 100 * pct_singletons,
            )

            # Decide whether to use complaint random intercepts.
            # Only drop for monetary_severity — issue_severity uses all
            # complaints so singletons are rare there.
            use_random_intercept = True
            if severity_metric == "monetary_severity" and pct_singletons > _SINGLETON_THRESHOLD:
                use_random_intercept = False
                logger.info(
                    "%s/%s: %.0f%% of complaints are singletons (>%.0f%% threshold) "
                    "→ dropping complaint random intercept (fixed-effect models). "
                    "Random intercepts are unidentifiable when most complaint "
                    "groups contain a single observation.",
                    model, severity_metric,
                    100 * pct_singletons, 100 * _SINGLETON_THRESHOLD,
                )

            metric_result: Dict[str, Any] = {
                "n_obs": len(df),
                "n_complaints": J,
                "K": X.shape[1],
                "col_names": col_names,
                "subset": subset,
                "pct_singletons": round(pct_singletons, 4),
                "use_random_intercept": use_random_intercept,
            }

            # ── Fit B1 + B2 (with fallback chain for monetary_severity) ──
            y_logistic = (df["tier"].values == 2).astype(int).tolist()
            y_ordinal = np.clip((df["tier"].values + 1).astype(int), 1, 3).tolist()

            fit_prefix = f"{model.replace('-', '')}__{severity_metric.replace('_', '')}"

            def _make_stan_data(
                X_mat: np.ndarray,
                use_ri: bool,
                prior_sd: Optional[np.ndarray] = None,
            ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                """Build B1/B2 Stan data dicts from design matrix."""
                common: Dict[str, Any] = {
                    "N": len(df), "K": X_mat.shape[1], "X": X_mat.tolist(),
                }
                if use_ri:
                    common["J"] = J
                    common["jj"] = jj.tolist()
                if prior_sd is not None:
                    common["prior_sd"] = prior_sd.tolist()
                return (
                    {**common, "y": y_logistic},
                    {**common, "y": y_ordinal},
                )

            def _pick_stan_files(
                use_ri: bool, flex: bool,
            ) -> Tuple[Path, Path]:
                """Select B1/B2 Stan model files."""
                if flex:
                    return _LOGIT_FLEX_STAN, _ORDINAL_FLEX_STAN
                if use_ri:
                    return _LOGIT_STAN, _ORDINAL_STAN
                return _LOGIT_FIXED_STAN, _ORDINAL_FIXED_STAN

            # For issue_severity, fit once and fatal on failure (no fallbacks).
            if severity_metric == "issue_severity":
                b1_stan, b2_stan = _pick_stan_files(use_random_intercept, flex=False)
                sd_b1, sd_b2 = _make_stan_data(X, use_random_intercept)
                method_lbl = "hierarchical" if use_random_intercept else "fixed-effect"
                result = _try_b1_b2(
                    df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                    b1_stan_file=b1_stan, b2_stan_file=b2_stan,
                    stan_data_b1=sd_b1, stan_data_b2=sd_b2,
                    fit_id_b1=f"B1__{fit_prefix}", fit_id_b2=f"B2__{fit_prefix}",
                    seed=seed, output_dir=output_dir,
                    method_label=method_lbl,
                )
                if result is None:
                    raise RuntimeError(
                        f"Bayesian models FAILED for {model}/{severity_metric}. "
                        f"Posterior draws are unreliable."
                    )
                metric_result.update(result)

            # For monetary_severity, use 4-step fallback chain.
            else:
                attempted: List[str] = []
                result: Optional[Dict[str, Any]] = None

                # ── Step 0: standard fit ──
                step_label = "standard"
                method_lbl = "hierarchical" if use_random_intercept else "fixed-effect"
                logger.info("monetary_severity fallback chain: "
                            "step 0 — %s fit (K=%d)", method_lbl, X.shape[1])
                b1_stan, b2_stan = _pick_stan_files(use_random_intercept, flex=False)
                sd_b1, sd_b2 = _make_stan_data(X, use_random_intercept)
                result = _try_b1_b2(
                    df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                    b1_stan_file=b1_stan, b2_stan_file=b2_stan,
                    stan_data_b1=sd_b1, stan_data_b2=sd_b2,
                    fit_id_b1=f"B1__{fit_prefix}", fit_id_b2=f"B2__{fit_prefix}",
                    seed=seed, output_dir=output_dir,
                    method_label=method_lbl,
                )
                if result is not None:
                    logger.info("monetary_severity: step 0 (%s) succeeded", method_lbl)
                else:
                    attempted.append(f"step 0: {method_lbl} (K={X.shape[1]})")
                    logger.warning("monetary_severity: step 0 (%s) FAILED", method_lbl)

                # ── Fallback 1: category-level random intercept ──
                if result is None:
                    # Use product_category as grouping variable (coarser
                    # than complaint ID, so random intercepts are
                    # identifiable even with many singletons).
                    cat_col = "product_category"
                    if cat_col in df.columns and df[cat_col].notna().sum() > 0:
                        cat_vals = df[cat_col].fillna("_unknown_").values
                        unique_cats = np.unique(cat_vals)
                        cat_map = {c: i + 1 for i, c in enumerate(unique_cats)}
                        jj_cat = np.array([cat_map[c] for c in cat_vals])
                        J_cat = len(unique_cats)

                        logger.info(
                            "monetary_severity fallback chain: "
                            "step 1 — category-level RE (%s, J=%d categories)",
                            cat_col, J_cat,
                        )
                        b1_stan_cat, b2_stan_cat = _LOGIT_STAN, _ORDINAL_STAN
                        common_cat: Dict[str, Any] = {
                            "N": len(df), "K": X.shape[1], "X": X.tolist(),
                            "J": J_cat, "jj": jj_cat.tolist(),
                        }
                        result = _try_b1_b2(
                            df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                            b1_stan_file=b1_stan_cat, b2_stan_file=b2_stan_cat,
                            stan_data_b1={**common_cat, "y": y_logistic},
                            stan_data_b2={**common_cat, "y": y_ordinal},
                            fit_id_b1=f"B1__{fit_prefix}__catRE",
                            fit_id_b2=f"B2__{fit_prefix}__catRE",
                            seed=seed, output_dir=output_dir,
                            method_label=f"category-RE({cat_col})",
                        )
                        if result is not None:
                            logger.info("monetary_severity: step 1 (category RE) succeeded")
                        else:
                            attempted.append(f"step 1: category-RE({cat_col}, J={J_cat})")
                            logger.warning("monetary_severity: step 1 (category RE) FAILED")
                    else:
                        attempted.append(f"step 1: category-RE skipped ({cat_col} not available)")
                        logger.warning(
                            "monetary_severity fallback chain: "
                            "step 1 skipped — %s column not available", cat_col,
                        )

                # ── Fallback 2: per-dimension models ──
                if result is None:
                    logger.info(
                        "monetary_severity fallback chain: "
                        "step 2 — per-dimension models (one dim at a time)"
                    )
                    result = _try_per_dimension(
                        df=df,
                        severity_metric=severity_metric,
                        jj=jj, J=J,
                        use_random_intercept=use_random_intercept,
                        fit_id_prefix=fit_prefix,
                        seed=seed,
                        output_dir=output_dir,
                    )
                    if result is not None:
                        logger.info("monetary_severity: step 2 (per-dimension) succeeded")
                    else:
                        attempted.append("step 2: per-dimension models")
                        logger.warning("monetary_severity: step 2 (per-dimension) FAILED")

                # ── Fallback 3: stronger priors (main=1.0, interaction=0.5) ──
                if result is None:
                    logger.info(
                        "monetary_severity fallback chain: "
                        "step 3 — stronger priors (main=1.0, interaction=0.5)"
                    )
                    # Build per-column prior_sd: 1.0 for main effects, 0.5
                    # for interaction terms (columns containing ':').
                    prior_sd = np.array([
                        0.5 if ":" in c else 1.0 for c in col_names
                    ])
                    b1_flex, b2_flex = _pick_stan_files(use_random_intercept, flex=True)
                    sd_b1_flex, sd_b2_flex = _make_stan_data(
                        X, use_random_intercept, prior_sd=prior_sd,
                    )
                    result = _try_b1_b2(
                        df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                        b1_stan_file=b1_flex, b2_stan_file=b2_flex,
                        stan_data_b1=sd_b1_flex, stan_data_b2=sd_b2_flex,
                        fit_id_b1=f"B1__{fit_prefix}__flex",
                        fit_id_b2=f"B2__{fit_prefix}__flex",
                        seed=seed, output_dir=output_dir,
                        method_label="flex-prior",
                    )
                    if result is not None:
                        logger.info("monetary_severity: step 3 (stronger priors) succeeded")
                    else:
                        attempted.append("step 3: flex priors (main=1.0, interaction=0.5)")
                        logger.warning("monetary_severity: step 3 (stronger priors) FAILED")

                # ── Fallback 4: aggressive sampler (adapt_delta=0.99, max_treedepth=15) ──
                if result is None:
                    logger.info(
                        "monetary_severity fallback chain: "
                        "step 4 — aggressive sampler (adapt_delta=0.99, max_treedepth=15)"
                    )
                    # Re-use stronger priors from step 3 combined with
                    # aggressive sampler tuning.
                    prior_sd = np.array([
                        0.5 if ":" in c else 1.0 for c in col_names
                    ])
                    b1_flex, b2_flex = _pick_stan_files(use_random_intercept, flex=True)
                    sd_b1_agg, sd_b2_agg = _make_stan_data(
                        X, use_random_intercept, prior_sd=prior_sd,
                    )
                    result = _try_b1_b2(
                        df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                        b1_stan_file=b1_flex, b2_stan_file=b2_flex,
                        stan_data_b1=sd_b1_agg, stan_data_b2=sd_b2_agg,
                        fit_id_b1=f"B1__{fit_prefix}__aggressive",
                        fit_id_b2=f"B2__{fit_prefix}__aggressive",
                        seed=seed, output_dir=output_dir,
                        adapt_delta=0.99,
                        max_treedepth=15,
                        method_label="aggressive-sampler",
                    )
                    if result is not None:
                        logger.info("monetary_severity: step 4 (aggressive sampler) succeeded")
                    else:
                        attempted.append("step 4: aggressive sampler (adapt_delta=0.99, max_treedepth=15)")
                        logger.warning("monetary_severity: step 4 (aggressive sampler) FAILED")

                # ── All fallbacks exhausted → fatal ──
                if result is None:
                    msg_lines = [
                        f"All Bayesian fallbacks FAILED for {model}/monetary_severity.",
                        "Attempted strategies:",
                    ]
                    for a in attempted:
                        msg_lines.append(f"  - {a}")
                    msg_lines.append("Posterior draws are unreliable for all attempted configurations.")
                    full_msg = "\n".join(msg_lines)
                    logger.critical(full_msg)
                    raise RuntimeError(full_msg)

                metric_result.update(result)

            model_results[severity_metric] = metric_result

        results[model] = model_results

    return results
