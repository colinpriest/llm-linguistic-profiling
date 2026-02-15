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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

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
# High adapt_delta (0.99) and max_treedepth (13) reduce divergent
# transitions and treedepth hits for difficult posteriors.
_CHAINS = 4
_ITER_SAMPLING = 1000
_ITER_WARMUP = 1000
_ADAPT_DELTA = 0.99
_MAX_TREEDEPTH = 13

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


def _collapse_rare_severity_bands(
    df: pd.DataFrame,
    severity_metric: str,
    min_per_band: int = 10,
) -> pd.DataFrame:
    """Merge rare severity bands into adjacent bands.

    If any severity band has < min_per_band observations, merge it with
    the adjacent band (prefer merging high->medium, then low->medium).
    Returns a copy of df with the severity column modified.
    """
    df = df.copy()
    counts = df[severity_metric].value_counts()
    merges: List[str] = []

    # Merge high -> medium if too sparse
    if counts.get("high", 0) < min_per_band and "high" in counts.index:
        df[severity_metric] = df[severity_metric].replace({"high": "medium"})
        merges.append(f"high({counts['high']})->medium")

    # Merge low -> medium if too sparse
    if counts.get("low", 0) < min_per_band and "low" in counts.index:
        df[severity_metric] = df[severity_metric].replace({"low": "medium"})
        merges.append(f"low({counts['low']})->medium")

    if merges:
        new_counts = df[severity_metric].value_counts().to_dict()
        logger.warning(
            "Pre-flight: collapsed rare severity bands for %s: %s -> %s",
            severity_metric, ", ".join(merges), new_counts,
        )
    else:
        logger.info(
            "Pre-flight: severity bands OK for %s (counts: %s, min_per_band=%d)",
            severity_metric, counts.to_dict(), min_per_band,
        )

    return df


def _drop_zero_variance_columns(
    X: np.ndarray,
    col_names: List[str],
    threshold: float = 1e-12,
) -> Tuple[np.ndarray, List[str]]:
    """Drop columns with near-zero variance from design matrix.

    Returns filtered (X, col_names).  Logs dropped column names.
    """
    variances = np.var(X, axis=0)
    keep_mask = variances >= threshold
    dropped = [name for name, keep in zip(col_names, keep_mask) if not keep]

    if dropped:
        logger.warning(
            "Pre-flight: dropping %d zero-variance column(s) from K=%d: %s",
            len(dropped), len(keep_mask), dropped,
        )
        X = X[:, keep_mask]
        col_names = [name for name, keep in zip(col_names, keep_mask) if keep]
    else:
        logger.info("Pre-flight: all %d columns have non-zero variance", len(col_names))

    return X, col_names


# ------------------------------------------------------------------
# Pre-flight diagnostics
# ------------------------------------------------------------------

def _cell_count_analysis(
    df: pd.DataFrame,
    severity_metric: str,
    min_cell_count: int = 5,
) -> Tuple[List[str], Dict[str, int]]:
    """Cross-tabulate dimension × severity × tier and flag sparse cells.

    Returns (sparse_cell_descriptions, all_cell_counts).
    """
    sparse: List[str] = []
    counts: Dict[str, int] = {}

    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        for level in sorted(df[dim].dropna().unique()):
            for sev in SEVERITY_LEVELS:
                mask = (df[dim] == level) & (df[severity_metric] == sev)
                for tier in sorted(df["tier"].unique()):
                    n = int(((mask) & (df["tier"] == tier)).sum())
                    key = f"{dim}={level}|sev={sev}|tier={tier}"
                    counts[key] = n
                    if n < min_cell_count:
                        sparse.append(f"{key} (n={n})")

    return sparse, counts


def _preflight_diagnostics(
    *,
    X: np.ndarray,
    col_names: List[str],
    y_logistic: List[int],
    y_ordinal: List[int],
    severity_metric: str,
    model_name: str,
    df: pd.DataFrame,
    J: int,
    pct_singletons: float,
) -> Dict[str, Any]:
    """Run pre-flight statistical checks on the design matrix and outcome.

    Returns a dict with EPP, rank, condition number, sparse cells,
    predicted difficulty, and recommendations.
    """
    N, K = X.shape
    n_pos = sum(y_logistic)
    n_neg = N - n_pos
    n_events = min(n_pos, n_neg)
    epp = n_events / K if K > 0 else float("inf")

    # Design matrix rank and condition
    try:
        matrix_rank = int(np.linalg.matrix_rank(X))
    except Exception:
        matrix_rank = K

    try:
        gram = X.T @ X
        condition_number = float(np.linalg.cond(gram))
        if not np.isfinite(condition_number):
            condition_number = float("inf")
    except Exception:
        condition_number = float("inf")

    rank_deficient = matrix_rank < K
    ill_conditioned = condition_number > 1e10

    # Cell sparsity
    sparse_cells, cell_counts = _cell_count_analysis(df, severity_metric)

    # Tier distribution
    tier_counts = df["tier"].value_counts().sort_index().to_dict()
    tier2_rate = n_pos / N if N > 0 else 0.0

    # Predicted difficulty
    difficulty_score = 0
    if epp < 5:
        difficulty_score += 3
    elif epp < 10:
        difficulty_score += 2
    elif epp < 20:
        difficulty_score += 1
    if rank_deficient:
        difficulty_score += 2
    if ill_conditioned:
        difficulty_score += 1
    if len(sparse_cells) > K:
        difficulty_score += 1
    if pct_singletons > 0.5:
        difficulty_score += 1

    if difficulty_score >= 4:
        predicted_difficulty = "very_hard"
    elif difficulty_score >= 2:
        predicted_difficulty = "hard"
    elif difficulty_score >= 1:
        predicted_difficulty = "moderate"
    else:
        predicted_difficulty = "easy"

    # Recommendations
    recommendations: List[str] = []
    if epp < 10:
        recommendations.append(
            f"Drop interaction terms to reduce K (EPP={epp:.1f}, need >=10)"
        )
    if rank_deficient:
        recommendations.append(
            f"Design matrix is rank-deficient (rank={matrix_rank} < K={K})"
        )
    if ill_conditioned:
        recommendations.append(
            f"Ill-conditioned Gram matrix (cond={condition_number:.2e}); "
            "consider stronger priors or dimensionality reduction"
        )
    if pct_singletons > 0.5:
        recommendations.append(
            "100% singleton observations prevent complaint-level random "
            "intercepts; consider using persona-decisions data (36 obs/complaint)"
        )
    if tier2_rate < 0.05:
        recommendations.append(
            f"Rare outcome (Tier-2 rate={tier2_rate:.1%}); "
            "consider collapsing Tier 0+1 vs Tier 2"
        )
    if len(sparse_cells) > 0:
        recommendations.append(
            f"{len(sparse_cells)} cross-tab cells have <5 observations; "
            "some interaction effects may be unidentifiable"
        )

    return {
        "N": N,
        "K": K,
        "J": J,
        "n_events": n_events,
        "epp": epp,
        "matrix_rank": matrix_rank,
        "rank_deficient": rank_deficient,
        "condition_number": condition_number,
        "ill_conditioned": ill_conditioned,
        "sparse_cells": sparse_cells,
        "n_sparse_cells": len(sparse_cells),
        "tier_counts": tier_counts,
        "tier2_rate": tier2_rate,
        "pct_singletons": pct_singletons,
        "predicted_difficulty": predicted_difficulty,
        "recommendations": recommendations,
    }


def _auto_simplify_design_matrix(
    *,
    df: pd.DataFrame,
    severity_metric: str,
    X_orig: np.ndarray,
    col_names_orig: List[str],
    y_logistic: List[int],
) -> Tuple[np.ndarray, List[str], str]:
    """Drop interaction columns when EPP is dangerously low.

    Returns (X_simplified, col_names_simplified, simplification_label).
    Label is "none", "drop-interactions", or "main-effects-only".
    """
    N = X_orig.shape[0]
    K = X_orig.shape[1]
    n_events = min(sum(y_logistic), N - sum(y_logistic))
    epp = n_events / K if K > 0 else float("inf")

    has_interactions = any("__x__" in c for c in col_names_orig)

    if epp >= 10 or not has_interactions:
        return X_orig, col_names_orig, "none"

    # Rebuild without interactions
    X_main, col_main = _build_design_matrix(
        df, severity_metric, include_interactions=False,
    )
    X_main, col_main = _drop_zero_variance_columns(X_main, col_main)

    epp_main = n_events / len(col_main) if len(col_main) > 0 else float("inf")
    logger.warning(
        "Pre-flight auto-simplify: dropping interactions "
        "(EPP %.1f -> %.1f, K %d -> %d)",
        epp, epp_main, K, len(col_main),
    )

    if epp_main < 5:
        logger.warning(
            "Pre-flight: EPP still very low (%.1f) after dropping interactions; "
            "MCMC convergence unlikely even with main-effects model",
            epp_main,
        )

    return X_main, col_main, "drop-interactions"


_MIN_MARGIN_COUNT = 5  # minimum observations in each margin for interaction terms


def _build_design_matrix(
    df: pd.DataFrame,
    severity_metric: str,
    dimensions: Optional[List[str]] = None,
    min_margin_count: int = _MIN_MARGIN_COUNT,
    include_interactions: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Build the shared design matrix X for both Stan models.

    Columns:
        - persona dimension dummies (k-1 per dimension, reference dropped)
        - severity band dummies (k-1, reference = low)
        - interactions: each persona dummy x each severity dummy
          (only if include_interactions=True and both margins have
          >= min_margin_count observations)

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

    # 3. Interactions (with adaptive pruning)
    interaction_names: List[str] = []
    if include_interactions:
        pruned: List[str] = []
        for p_idx, p_name in enumerate(persona_dummy_names):
            p_vec = parts[p_idx]
            p_count = int(p_vec.sum())
            for s_name in sev_dummy_names:
                s_idx = len(persona_dummy_names) + sev_dummy_names.index(s_name)
                s_vec = parts[s_idx]
                s_count = int(s_vec.sum())
                if p_count < min_margin_count or s_count < min_margin_count:
                    pruned.append(
                        f"{p_name}__x__{s_name}(margins={p_count},{s_count})"
                    )
                    continue
                interaction_names.append(f"{p_name}__x__{s_name}")
                parts.append(p_vec * s_vec)
        if pruned:
            logger.info(
                "Pre-flight: pruned %d sparse interaction(s): %s",
                len(pruned), pruned,
            )
        else:
            logger.info(
                "Pre-flight: all %d interaction(s) have sufficient margins (>=%d)",
                len(interaction_names), min_margin_count,
            )
    elif not include_interactions:
        logger.info("Pre-flight: building main-effects-only matrix (no interactions)")

    col_names = persona_dummy_names + sev_dummy_names + interaction_names
    X = np.hstack(parts) if parts else np.empty((len(df), 0))
    assert X.shape[1] == len(col_names), f"X has {X.shape[1]} columns but col_names has {len(col_names)}"

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
    chains: int = _CHAINS,
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
        fit_id, chains, _ITER_WARMUP, _ITER_SAMPLING, adapt_delta, n_params,
    )
    try:
        fit = model.sample(
            data=stan_data,
            chains=chains,
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
    """Extract MCMC diagnostics from a CmdStanMCMC fit.

    Computes R-hat and ESS only for model parameters (not log_lik
    generated quantities).  Previous implementation called fit.summary()
    which shells out to ``stansummary`` and processes ALL parameters
    including N observation-level log_lik values — extremely slow for
    large datasets.  Now uses ArviZ on filtered draws in-process.
    """
    # ── Divergences & treedepth (from in-memory sampler params) ──────
    n_divergent = 0
    n_treedepth = 0
    try:
        sampler_params = fit.method_variables()
        if "divergent__" in sampler_params:
            n_divergent = int(sampler_params["divergent__"].sum())
        if "treedepth__" in sampler_params:
            n_treedepth = int((sampler_params["treedepth__"] >= _MAX_TREEDEPTH).sum())
    except Exception:
        logger.warning("Could not extract sampler diagnostics from method_variables()")

    # ── R-hat and ESS for model parameters only ──────────────────────
    logger.info("Computing convergence diagnostics for model parameters...")

    all_draws = fit.draws(concat_chains=False)  # (draws, chains, all_params)
    col_names_all = list(fit.column_names)

    # Filter to model parameters (exclude log_lik generated quantities)
    model_idx = [i for i, name in enumerate(col_names_all)
                 if not name.startswith("log_lik[")]
    model_names = [col_names_all[i] for i in model_idx]
    model_draws = all_draws[:, :, model_idx]

    # ── Detect stuck parameters (zero within-chain variance) ─────────
    # A parameter with constant draws within each chain (but different
    # values between chains) produces a huge finite rhat in ArviZ
    # instead of NaN (which stansummary would give).  This poisons
    # max_rhat even when 29/30 parameters are perfectly converged.
    # Common causes: boundary parameters (sigma_u → 0), rare
    # interaction dummies with zero variance, or quasi-separation.
    # Detect these upfront and exclude from the convergence gate
    # (matching the old stansummary behaviour where NaN rhat was
    # dropped by dropna).
    n_model = len(model_names)
    # within-chain variance per parameter: var over draws, mean over chains
    within_var = model_draws.var(axis=0).mean(axis=0)  # shape (n_model,)
    stuck_mask = within_var < 1e-15  # effectively zero variance
    n_stuck = int(stuck_mask.sum())
    stuck_names: List[str] = []
    if n_stuck > 0:
        stuck_names = [model_names[i] for i in range(n_model) if stuck_mask[i]]
        logger.warning(
            "%d/%d parameters have zero within-chain variance (stuck): %s",
            n_stuck, n_model,
            stuck_names[:10] if len(stuck_names) > 10
            else stuck_names,
        )

    # Build xarray Dataset for ArviZ — only non-stuck parameters
    # CmdStanPy draws(concat_chains=False) shape: (draws, chains, params)
    live_idx = [k for k in range(n_model) if not stuck_mask[k]]
    live_names = [model_names[k] for k in live_idx]

    if not live_names:
        # ALL parameters stuck → truly degenerate posterior
        logger.warning("All %d model parameters are stuck — posterior is degenerate", n_model)
        return {
            "max_rhat": float("nan"),
            "min_ess_bulk": 0.0,
            "divergences": n_divergent,
            "treedepth_hits": n_treedepth,
            "n_stuck_params": n_stuck,
            "stuck_params": stuck_names,
            "diagnose_text": f"All {n_model} parameters stuck (zero within-chain variance).",
        }

    ds = xr.Dataset({
        name: xr.DataArray(model_draws[:, :, live_idx[k]], dims=["draw", "chain"])
        for k, name in enumerate(live_names)
    })

    # Suppress ArviZ RuntimeWarning for edge cases where within-chain
    # variance is numerically zero (0/0 in split-rhat folded statistic).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered",
            category=RuntimeWarning,
        )
        rhat_ds = az.rhat(ds)
        ess_ds = az.ess(ds, method="bulk")

    rhat_vals = pd.Series(
        [float(rhat_ds[v].values) for v in rhat_ds.data_vars]
    ).dropna()
    ess_vals = pd.Series(
        [float(ess_ds[v].values) for v in ess_ds.data_vars]
    ).dropna()

    max_rhat = float(rhat_vals.max()) if len(rhat_vals) > 0 else float("nan")
    min_ess = float(ess_vals.min()) if len(ess_vals) > 0 else 0.0

    # Build diagnostic summary text (without shelling out to diagnose)
    diag_parts = []
    if n_divergent > 0:
        diag_parts.append(f"{n_divergent} divergent transitions")
    if n_treedepth > 0:
        diag_parts.append(f"{n_treedepth} transitions hit max treedepth")
    if max_rhat > 1.01:
        diag_parts.append(f"max R-hat {max_rhat:.4f} > 1.01")
    if min_ess < 400:
        diag_parts.append(f"min ESS {min_ess:.0f} < 400")
    if n_stuck > 0:
        diag_parts.append(f"{n_stuck} stuck parameter(s)")
    diag_text = "; ".join(diag_parts) if diag_parts else "No diagnostic issues detected."

    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "divergences": n_divergent,
        "treedepth_hits": n_treedepth,
        "n_stuck_params": n_stuck,
        "stuck_params": stuck_names,
        "diagnose_text": diag_text,
    }


def _gate_fit(diagnostics: Dict[str, Any], *, strict: bool = True) -> str:
    """Apply acceptance gate. Returns 'OK', 'WARN', 'MARGINAL', or 'FAIL'.

    Logic (strict=True, used for steps 1-3.5):
        FAIL — non-convergence (Rhat > 1.01, ESS < 400) or any divergences.
        WARN — minor issues (treedepth hits, domain warnings).
        OK   — clean diagnostics.

    Logic (strict=False, used for steps 4-5):
        FAIL     — Rhat > 1.05, ESS < 200, or > 2 divergences.
        MARGINAL — passes relaxed but would fail strict thresholds.
        WARN     — treedepth hits only.
        OK       — passes strict thresholds too.
    """
    rhat = diagnostics.get("max_rhat", float("nan"))
    ess = diagnostics.get("min_ess_bulk", 0)
    div = diagnostics.get("divergences", 0)
    td = diagnostics.get("treedepth_hits", 0)

    if strict:
        # Hard fail: non-convergent chains or any divergent transitions
        if (not np.isfinite(rhat)) or rhat > 1.01 or ess < 400 or div > 0:
            return "FAIL"
        if td > 0:
            return "WARN"
        return "OK"

    # Relaxed gate (steps 4-5): accept slightly imperfect posteriors
    if (not np.isfinite(rhat)) or rhat > 1.05 or ess < 200 or div > 2:
        return "FAIL"

    # Passes relaxed thresholds — would it also pass strict?
    would_fail_strict = rhat > 1.01 or ess < 400 or div > 0
    if would_fail_strict:
        return "MARGINAL"

    if td > 0:
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
    sev_dummy_names = [col for col in col_names if col.startswith("sev__")]
    persona_dummy_names = [col for col in col_names if not col.startswith("sev__") and "__x__" not in col]

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
        if "high" not in D_draws:
            logger.warning("No 'high' severity band in posterior draws; amplification not estimable")
        if "low" not in D_draws:
            logger.warning("No 'low' severity band in posterior draws; amplification not estimable")
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

    sev_dummy_names = [col for col in col_names if col.startswith("sev__")]
    persona_dummy_names = [col for col in col_names if not col.startswith("sev__") and "__x__" not in col]

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

        if "high" not in D_draws:
            logger.warning("No 'high' severity band in posterior draws; amplification not estimable")
        if "low" not in D_draws:
            logger.warning("No 'low' severity band in posterior draws; amplification not estimable")
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

    sev_dummy_names = [col for col in col_names if col.startswith("sev__")]
    persona_dummy_names = [col for col in col_names if not col.startswith("sev__") and "__x__" not in col]

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
# Post-mortem analysis
# ------------------------------------------------------------------

def _build_postmortem(
    *,
    df: pd.DataFrame,
    X: np.ndarray,
    col_names: List[str],
    y_logistic: List[int],
    y_ordinal: List[int],
    severity_metric: str,
    model_name: str,
    attempted: List[Dict[str, Any]],
    use_random_intercept: bool,
    J: int,
    pct_singletons: float,
    preflight: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured post-mortem report when all fallbacks fail.

    The report contains enough detail for a statistician to evaluate
    whether the attempted strategies were reasonable and suggest
    further approaches.
    """
    N, K = X.shape
    n_pos = sum(y_logistic)
    n_neg = N - n_pos
    epp = min(n_pos, n_neg) / K if K > 0 else float("inf")

    # 1. Data summary
    tier_counts = df["tier"].value_counts().sort_index().to_dict()
    sev_counts = df[severity_metric].value_counts().to_dict()
    data_summary = {
        "N": N,
        "K": K,
        "J": J,
        "n_tier2": n_pos,
        "n_non_tier2": n_neg,
        "tier2_rate": n_pos / N if N > 0 else 0.0,
        "events_per_param": epp,
        "tier_distribution": tier_counts,
        "severity_distribution": sev_counts,
        "pct_singletons": pct_singletons,
        "use_random_intercept": use_random_intercept,
        "col_names": col_names,
    }

    # 2. Outcome distribution (severity × tier cross-tab)
    try:
        xtab = pd.crosstab(
            df[severity_metric], df["tier"], margins=True,
        )
        outcome_distribution = {
            str(sev): {str(t): int(v) for t, v in row.items()}
            for sev, row in xtab.iterrows()
        }
    except Exception:
        outcome_distribution = {}

    # 3. Cross-tab sparsity
    sparse_cells, cell_counts = _cell_count_analysis(df, severity_metric)
    cross_tab_sparsity = {
        "sparse_cells": sparse_cells,
        "n_sparse_cells": len(sparse_cells),
        "total_cells": len(cell_counts),
        "sparsity_ratio": len(sparse_cells) / len(cell_counts) if cell_counts else 0.0,
    }

    # 4. Design matrix diagnostics
    try:
        matrix_rank = int(np.linalg.matrix_rank(X))
    except Exception:
        matrix_rank = K
    try:
        condition_number = float(np.linalg.cond(X.T @ X))
        if not np.isfinite(condition_number):
            condition_number = float("inf")
    except Exception:
        condition_number = float("inf")

    design_matrix_diagnostics = {
        "rank": matrix_rank,
        "K": K,
        "rank_deficient": matrix_rank < K,
        "condition_number": condition_number,
        "ill_conditioned": condition_number > 1e10,
        "n_interaction_cols": sum(1 for c in col_names if "__x__" in c),
        "n_main_effect_cols": sum(1 for c in col_names if "__x__" not in c),
    }

    # 5. Per-step diagnostics (from structured attempted list)
    per_step_diagnostics = []
    for step_info in attempted:
        entry = {
            "step": step_info.get("step", "?"),
            "label": step_info.get("label", "unknown"),
            "description": step_info.get("description", ""),
            "b1_status": step_info.get("b1_status", "N/A"),
            "b2_status": step_info.get("b2_status", "N/A"),
        }
        # Include diagnostic values if present
        for prefix in ("b1", "b2"):
            for key in ("rhat", "ess", "div"):
                val = step_info.get(f"{prefix}_{key}")
                if val is not None:
                    entry[f"{prefix}_{key}"] = val
        per_step_diagnostics.append(entry)

    # 6. Root cause analysis
    root_causes: List[str] = []
    if epp < 10:
        root_causes.append(
            f"Overparameterized: {epp:.1f} events per parameter "
            f"({n_pos} Tier-2 events / {K} parameters; minimum 10 recommended)"
        )
    if pct_singletons > 0.5:
        root_causes.append(
            f"Singleton structure: {pct_singletons:.0%} of complaints have "
            "exactly 1 observation, preventing complaint-level random intercepts"
        )
    if n_pos / N < 0.05 if N > 0 else False:
        root_causes.append(
            f"Rare outcome: Tier-2 rate of {n_pos / N:.1%} causes potential "
            "separation and wide posteriors"
        )
    if condition_number > 1e10:
        root_causes.append(
            f"Ill-conditioned design matrix (condition number = {condition_number:.2e})"
        )
    if matrix_rank < K:
        root_causes.append(
            f"Rank-deficient design matrix (rank={matrix_rank} < K={K})"
        )
    if len(sparse_cells) > K:
        root_causes.append(
            f"Extreme cross-tab sparsity: {len(sparse_cells)} of "
            f"{len(cell_counts)} cells have <5 observations"
        )
    if not root_causes:
        root_causes.append("No clear root cause identified from pre-flight metrics")

    # 7. Recommendations
    recommendations: List[str] = []
    if epp < 10:
        recommendations.append(
            "Consider using main-effects-only model (no interactions) "
            "to improve events-per-parameter ratio"
        )
    if pct_singletons > 0.5:
        recommendations.append(
            "Consider using persona_decisions data (36 obs/complaint) instead "
            "of baseline (1 obs/complaint) to enable random intercepts"
        )
    recommendations.append(
        "Variational inference (ADVI) or Pathfinder may provide approximate "
        "posteriors when MCMC fails to converge"
    )
    recommendations.append(
        "MAP point estimates can provide directional evidence "
        "even without full posterior uncertainty"
    )
    if K > 10:
        recommendations.append(
            "Consider per-dimension models (K=3-5 each) to reduce "
            "effective dimensionality"
        )
    if n_pos / N < 0.10 if N > 0 else False:
        recommendations.append(
            "Consider stronger priors (N(0, 0.3)) to regularize "
            "coefficients for rare outcomes"
        )

    return {
        "model": model_name,
        "severity_metric": severity_metric,
        "data_summary": data_summary,
        "outcome_distribution": outcome_distribution,
        "cross_tab_sparsity": cross_tab_sparsity,
        "design_matrix_diagnostics": design_matrix_diagnostics,
        "per_step_diagnostics": per_step_diagnostics,
        "root_causes": root_causes,
        "recommendations": recommendations,
        "preflight": preflight,
    }


def _save_postmortem(
    postmortem: Dict[str, Any],
    model_name: str,
    severity_metric: str,
    output_dir: Optional[Path],
) -> None:
    """Save post-mortem analysis as a JSON artifact."""
    if output_dir is None:
        return

    bayes_dir = output_dir / "analysis" / "stats" / "bayes"
    bayes_dir.mkdir(parents=True, exist_ok=True)

    # Sanitise model name for file path
    safe_model = model_name.replace("-", "").replace(".", "")
    path = bayes_dir / f"stats__bayes__postmortem__{safe_model}__{severity_metric}__v1.json"

    # Convert any numpy/non-serialisable values
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    path.write_text(json.dumps(postmortem, indent=2, default=_default))
    logger.info("Post-mortem saved to %s", path)


def _log_postmortem(
    postmortem: Dict[str, Any],
    model_name: str,
    severity_metric: str,
) -> None:
    """Log key post-mortem sections to console/log at WARNING level."""
    ds = postmortem["data_summary"]
    dm = postmortem["design_matrix_diagnostics"]

    logger.warning("=" * 60)
    logger.warning("POST-MORTEM: %s / %s", model_name, severity_metric)
    logger.warning("=" * 60)

    # Data summary
    logger.warning(
        "  Data: N=%d, K=%d, J=%d, EPP=%.1f",
        ds["N"], ds["K"], ds["J"], ds["events_per_param"],
    )
    logger.warning(
        "  Outcome: Tier-2 rate=%.1f%% (%d/%d), singletons=%.0f%%",
        ds["tier2_rate"] * 100, ds["n_tier2"], ds["N"],
        ds["pct_singletons"] * 100,
    )
    logger.warning(
        "  Tiers: %s", ds["tier_distribution"],
    )
    logger.warning(
        "  Design matrix: rank=%d/%d, condition=%.2e, "
        "%d interaction cols, %d main-effect cols",
        dm["rank"], dm["K"], dm["condition_number"],
        dm["n_interaction_cols"], dm["n_main_effect_cols"],
    )

    # Sparsity
    sp = postmortem["cross_tab_sparsity"]
    if sp["n_sparse_cells"] > 0:
        logger.warning(
            "  Cell sparsity: %d/%d cells have <5 obs (%.0f%%)",
            sp["n_sparse_cells"], sp["total_cells"],
            sp["sparsity_ratio"] * 100,
        )

    # Per-step diagnostics table
    logger.warning("  Fallback chain summary:")
    for step_info in postmortem["per_step_diagnostics"]:
        b1_detail = ""
        b2_detail = ""
        if "b1_rhat" in step_info:
            b1_detail = f" rhat={step_info['b1_rhat']:.4f} ess={step_info['b1_ess']:.0f} div={step_info['b1_div']}"
        if "b2_rhat" in step_info:
            b2_detail = f" rhat={step_info['b2_rhat']:.4f} ess={step_info['b2_ess']:.0f} div={step_info['b2_div']}"
        logger.warning(
            "    Step %s [%s]: %s | B1=%s%s | B2=%s%s",
            step_info["step"], step_info["label"], step_info["description"],
            step_info["b1_status"], b1_detail,
            step_info["b2_status"], b2_detail,
        )

    # Root causes
    logger.warning("  Root causes:")
    for cause in postmortem["root_causes"]:
        logger.warning("    - %s", cause)

    # Recommendations
    logger.warning("  Recommendations:")
    for rec in postmortem["recommendations"]:
        logger.warning("    - %s", rec)

    logger.warning("=" * 60)


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
    chains: int = _CHAINS,
    method_label: str = "hierarchical",
    attempt: str = "both",
    strict_gate: bool = True,
) -> Dict[str, Any]:
    """Attempt B1 (logistic) and/or B2 (ordinal) fits.

    B1 and B2 are fitted independently — a failure in one does not
    prevent the other from being attempted.

    Parameters
    ----------
    attempt : "both", "b1_only", or "b2_only"
        Which model types to fit.
    strict_gate : bool
        If True, use strict diagnostic thresholds. If False, use
        relaxed thresholds (MARGINAL status for borderline fits).

    Returns
    -------
    Dict with "logistic" and/or "ordinal" keys.  Each sub-dict has
    a "status" field: "OK", "WARN", "MARGINAL", or "FAIL".
    Always returns a dict (never None).
    """
    y_ordinal = (df["tier"].values + 1).astype(int)
    y_ordinal = np.clip(y_ordinal, 1, 3)
    stan_data_b2.setdefault("y", y_ordinal.tolist())

    metric_result: Dict[str, Any] = {}

    # ── B1: Logistic ──
    if attempt in ("both", "b1_only"):
        logger.info("Fitting B1 (%s logistic, Tier2)...", method_label)
        b1_failed = False
        try:
            fit_b1 = _fit_stan_model(b1_stan_file, stan_data_b1, fit_id_b1, seed,
                                     adapt_delta=adapt_delta, max_treedepth=max_treedepth,
                                     chains=chains)
        except Exception as exc:
            logger.warning("B1 %s logistic raised %s for %s: %s",
                           method_label, type(exc).__name__, fit_id_b1, exc)
            metric_result["logistic"] = {
                "fit_id": fit_id_b1, "status": "FAIL",
                "method": f"{method_label} logistic",
            }
            b1_failed = True

        if not b1_failed:
            diag_b1 = _check_diagnostics(fit_b1)
            status_b1 = _gate_fit(diag_b1, strict=strict_gate)
            logger.info("B1 status: %s (rhat=%.4f, ess=%.0f, div=%d) [%s]",
                        status_b1, diag_b1["max_rhat"],
                        diag_b1["min_ess_bulk"], diag_b1["divergences"], method_label)

            if status_b1 == "FAIL":
                logger.warning("B1 %s logistic FAILED for %s", method_label, fit_id_b1)
                diag_b1_payload = {k: v for k, v in diag_b1.items() if k != "diagnose_text"}
                diag_b1_payload["diagnose_text"] = diag_b1.get("diagnose_text", "")
                metric_result["logistic"] = {
                    "fit_id": fit_id_b1, "diagnostics": diag_b1_payload,
                    "status": "FAIL", "method": f"{method_label} logistic",
                }
                _save_artifacts(fit_id_b1, "logistic", {}, diag_b1, status_b1,
                                b1_stan_file, output_dir)
            else:
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
    if attempt in ("both", "b2_only"):
        logger.info("Fitting B2 (%s ordinal, Tier 0/1/2)...", method_label)
        b2_failed = False
        try:
            fit_b2 = _fit_stan_model(b2_stan_file, stan_data_b2, fit_id_b2, seed,
                                     adapt_delta=adapt_delta, max_treedepth=max_treedepth,
                                     chains=chains)
        except Exception as exc:
            logger.warning("B2 %s ordinal raised %s for %s: %s",
                           method_label, type(exc).__name__, fit_id_b2, exc)
            metric_result["ordinal"] = {
                "fit_id": fit_id_b2, "status": "FAIL",
                "method": f"{method_label} ordinal",
            }
            b2_failed = True

        if not b2_failed:
            diag_b2 = _check_diagnostics(fit_b2)
            status_b2 = _gate_fit(diag_b2, strict=strict_gate)
            logger.info("B2 status: %s (rhat=%.4f, ess=%.0f, div=%d) [%s]",
                        status_b2, diag_b2["max_rhat"],
                        diag_b2["min_ess_bulk"], diag_b2["divergences"], method_label)

            if status_b2 == "FAIL":
                logger.warning("B2 %s ordinal FAILED for %s", method_label, fit_id_b2)
                diag_b2_payload = {k: v for k, v in diag_b2.items() if k != "diagnose_text"}
                diag_b2_payload["diagnose_text"] = diag_b2.get("diagnose_text", "")
                metric_result["ordinal"] = {
                    "fit_id": fit_id_b2, "diagnostics": diag_b2_payload,
                    "status": "FAIL", "method": f"{method_label} ordinal",
                }
                _save_artifacts(fit_id_b2, "ordinal", {}, diag_b2, status_b2,
                                b2_stan_file, output_dir)
            else:
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
    attempt: str = "both",
    strict_gate: bool = True,
) -> Optional[Dict[str, Any]]:
    """Fallback: fit separate per-dimension models with partial credit.

    Each model has one demographic dimension x severity interactions,
    yielding ~4-8 columns instead of ~20+.  Results are merged across
    dimensions.

    If some dimensions fail, retries them with 8 chains.  Returns
    partial results (with ``partial=True``) if at least one dimension
    succeeds.  Returns None only if ALL dimensions fail.
    """
    combined_logistic: Dict[str, Any] = {"amplification": {}, "per_dim_fits": {}}
    combined_ordinal: Dict[str, Any] = {"amplification": {}, "tier_probs": {},
                                        "per_dim_fits": {}}
    succeeded_dims: List[str] = []
    failed_dims: List[str] = []

    def _fit_dim(dim: str, chains: int = _CHAINS) -> Optional[Dict[str, Any]]:
        """Fit a single dimension, return result dict or None."""
        if dim not in df.columns:
            return None

        X_dim, col_names_dim = _build_design_matrix(df, severity_metric, dimensions=[dim])
        X_dim, col_names_dim = _drop_zero_variance_columns(X_dim, col_names_dim)
        dim_lvls = _get_dimension_levels(df, dimensions=[dim])

        if X_dim.shape[1] == 0:
            logger.warning("Per-dimension %s: empty design matrix, skipping", dim)
            return None

        if use_random_intercept:
            b1_stan = _LOGIT_STAN
            b2_stan = _ORDINAL_STAN
            common_data: Dict[str, Any] = {
                "N": len(df), "K": X_dim.shape[1], "X": X_dim.tolist(),
                "J": J, "jj": jj.tolist(),
            }
        else:
            b1_stan = _LOGIT_FIXED_STAN
            b2_stan = _ORDINAL_FIXED_STAN
            common_data = {"N": len(df), "K": X_dim.shape[1], "X": X_dim.tolist()}

        y_logistic = (df["tier"].values == 2).astype(int).tolist()
        y_ordinal = np.clip((df["tier"].values + 1).astype(int), 1, 3).tolist()

        suffix = "" if chains == _CHAINS else "__retry8"
        fit_id_b1 = f"{fit_id_prefix}__dim_{dim}__B1{suffix}"
        fit_id_b2 = f"{fit_id_prefix}__dim_{dim}__B2{suffix}"

        logger.info("Per-dimension fallback: fitting %s (K=%d, chains=%d)...",
                     dim, X_dim.shape[1], chains)

        return _try_b1_b2(
            df=df, X=X_dim, col_names=col_names_dim, dim_levels=dim_lvls,
            b1_stan_file=b1_stan, b2_stan_file=b2_stan,
            stan_data_b1={**common_data, "y": y_logistic},
            stan_data_b2={**common_data, "y": y_ordinal},
            fit_id_b1=fit_id_b1, fit_id_b2=fit_id_b2,
            seed=seed, output_dir=output_dir, chains=chains,
            method_label=f"per-dim({dim})",
            attempt=attempt, strict_gate=strict_gate,
        )

    def _dim_succeeded(result: Dict[str, Any]) -> bool:
        """Check if at least one model type succeeded in a dimension fit."""
        b1_ok = result.get("logistic", {}).get("status", "FAIL") != "FAIL"
        b2_ok = result.get("ordinal", {}).get("status", "FAIL") != "FAIL"
        if attempt == "b1_only":
            return b1_ok
        if attempt == "b2_only":
            return b2_ok
        return b1_ok or b2_ok

    def _merge_dim(dim: str, result: Dict[str, Any]) -> None:
        """Merge a successful dimension result into combined dicts."""
        logistic = result.get("logistic", {})
        if logistic.get("status", "FAIL") != "FAIL" and "amplification" in logistic:
            combined_logistic["amplification"].update(logistic["amplification"])
            combined_logistic["per_dim_fits"][dim] = logistic
        ordinal = result.get("ordinal", {})
        if ordinal.get("status", "FAIL") != "FAIL" and "amplification" in ordinal:
            combined_ordinal["amplification"].update(ordinal["amplification"])
            if "tier_probs" in ordinal:
                combined_ordinal["tier_probs"].update(ordinal.get("tier_probs", {}))
            combined_ordinal["per_dim_fits"][dim] = ordinal

    # First pass: try all dimensions with standard chain count
    for dim in DIMENSIONS:
        result = _fit_dim(dim)
        if result is not None and _dim_succeeded(result):
            succeeded_dims.append(dim)
            _merge_dim(dim, result)
        else:
            logger.warning("Per-dimension fallback: %s FAILED (first pass)", dim)
            failed_dims.append(dim)

    # Retry pass: re-attempt failed dimensions with 8 chains
    if failed_dims:
        retry_dims = failed_dims[:]
        for dim in retry_dims:
            logger.info("Per-dimension fallback: retrying %s with 8 chains...", dim)
            result = _fit_dim(dim, chains=8)
            if result is not None and _dim_succeeded(result):
                succeeded_dims.append(dim)
                failed_dims.remove(dim)
                _merge_dim(dim, result)
                logger.info("Per-dimension fallback: %s succeeded on retry", dim)
            else:
                logger.warning("Per-dimension fallback: %s FAILED on retry too", dim)

    # Return partial results if at least one dimension succeeded
    if not succeeded_dims:
        return None

    is_partial = len(failed_dims) > 0
    if is_partial:
        logger.warning(
            "Per-dimension fallback: partial results — succeeded: %s, failed: %s",
            succeeded_dims, failed_dims,
        )

    logistic_result: Dict[str, Any] = {
        "method": "per-dimension logistic",
        "amplification": combined_logistic["amplification"],
        "per_dim_fits": combined_logistic["per_dim_fits"],
        "partial": is_partial,
        "dimensions_succeeded": succeeded_dims,
        "dimensions_failed": failed_dims,
    }
    ordinal_result: Dict[str, Any] = {
        "method": "per-dimension ordinal",
        "amplification": combined_ordinal["amplification"],
        "tier_probs": combined_ordinal["tier_probs"],
        "per_dim_fits": combined_ordinal["per_dim_fits"],
        "partial": is_partial,
        "dimensions_succeeded": succeeded_dims,
        "dimensions_failed": failed_dims,
    }

    # Determine status for each model type based on what we have
    logistic_result["status"] = (
        "OK" if combined_logistic["per_dim_fits"] and not is_partial
        else "WARN" if combined_logistic["per_dim_fits"]
        else "FAIL"
    )
    ordinal_result["status"] = (
        "OK" if combined_ordinal["per_dim_fits"] and not is_partial
        else "WARN" if combined_ordinal["per_dim_fits"]
        else "FAIL"
    )

    return {"logistic": logistic_result, "ordinal": ordinal_result}


# ------------------------------------------------------------------
# Non-MCMC fallback methods
# ------------------------------------------------------------------

def _try_variational(
    *,
    stan_file: Path,
    stan_data: Dict[str, Any],
    fit_id: str,
    col_names: List[str],
    dim_levels: Dict[str, List[str]],
    model_type: str,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    algorithm: str = "meanfield",
) -> Dict[str, Any]:
    """Attempt variational inference (ADVI) as a fallback when MCMC fails.

    CmdStanVB.stan_variable() returns draws compatible with the
    existing _compute_amplification_* functions.

    Returns a dict with status="ADVI" on success, "FAIL" on failure.
    """
    logger.info("Fitting %s via variational inference (ADVI/%s)...", fit_id, algorithm)

    try:
        from cmdstanpy import CmdStanModel
        model = CmdStanModel(stan_file=str(stan_file))
        vb = model.variational(
            data=stan_data,
            seed=seed,
            algorithm=algorithm,
            output_samples=1000,
            require_converged=False,
        )
    except Exception as exc:
        logger.warning(
            "Variational inference failed for %s: %s: %s",
            fit_id, type(exc).__name__, exc,
        )
        return {"status": "FAIL", "fit_id": fit_id, "method": f"ADVI-{algorithm}"}

    # Synthetic diagnostics (no MCMC diagnostics for VI)
    vi_diag = {
        "max_rhat": float("nan"),
        "min_ess_bulk": float("nan"),
        "divergences": 0,
        "treedepth_hits": 0,
        "n_stuck_params": 0,
        "stuck_params": [],
        "diagnose_text": f"Variational inference ({algorithm}); no MCMC diagnostics.",
    }

    try:
        if model_type == "logistic":
            amp = _compute_amplification_logistic(vb, col_names, dim_levels)
        else:
            amp = _compute_amplification_ordinal(vb, col_names, dim_levels)

        amp_summary = {dim: {k: v for k, v in dd.items() if k != "A_draws"}
                       for dim, dd in amp.items()}

        result = {
            "fit_id": fit_id,
            "diagnostics": vi_diag,
            "status": "ADVI",
            "amplification": amp_summary,
            "method": f"ADVI-{algorithm} {model_type}",
        }
        _save_artifacts(fit_id, model_type, amp, vi_diag, "ADVI",
                        stan_file, output_dir)
        logger.info("ADVI succeeded for %s", fit_id)
        return result

    except Exception as exc:
        logger.warning(
            "ADVI amplification computation failed for %s: %s: %s",
            fit_id, type(exc).__name__, exc,
        )
        return {"status": "FAIL", "fit_id": fit_id, "method": f"ADVI-{algorithm}"}


def _try_pathfinder(
    *,
    stan_file: Path,
    stan_data: Dict[str, Any],
    fit_id: str,
    col_names: List[str],
    dim_levels: Dict[str, List[str]],
    model_type: str,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Attempt Pathfinder variational inference as a fallback.

    Pathfinder requires CmdStan >= 2.33. If unavailable, returns FAIL.
    """
    logger.info("Fitting %s via Pathfinder...", fit_id)

    try:
        from cmdstanpy import CmdStanModel
        model = CmdStanModel(stan_file=str(stan_file))
        pf = model.pathfinder(
            data=stan_data,
            seed=seed,
            draws=1000,
            num_paths=4,
        )
    except AttributeError:
        logger.warning(
            "Pathfinder not available (requires CmdStan >= 2.33); skipping %s",
            fit_id,
        )
        return {"status": "FAIL", "fit_id": fit_id, "method": "pathfinder"}
    except Exception as exc:
        logger.warning(
            "Pathfinder failed for %s: %s: %s",
            fit_id, type(exc).__name__, exc,
        )
        return {"status": "FAIL", "fit_id": fit_id, "method": "pathfinder"}

    pf_diag = {
        "max_rhat": float("nan"),
        "min_ess_bulk": float("nan"),
        "divergences": 0,
        "treedepth_hits": 0,
        "n_stuck_params": 0,
        "stuck_params": [],
        "diagnose_text": "Pathfinder approximate inference; no MCMC diagnostics.",
    }

    try:
        if model_type == "logistic":
            amp = _compute_amplification_logistic(pf, col_names, dim_levels)
        else:
            amp = _compute_amplification_ordinal(pf, col_names, dim_levels)

        amp_summary = {dim: {k: v for k, v in dd.items() if k != "A_draws"}
                       for dim, dd in amp.items()}

        result = {
            "fit_id": fit_id,
            "diagnostics": pf_diag,
            "status": "PATHFINDER",
            "amplification": amp_summary,
            "method": f"pathfinder {model_type}",
        }
        _save_artifacts(fit_id, model_type, amp, pf_diag, "PATHFINDER",
                        stan_file, output_dir)
        logger.info("Pathfinder succeeded for %s", fit_id)
        return result

    except Exception as exc:
        logger.warning(
            "Pathfinder amplification computation failed for %s: %s: %s",
            fit_id, type(exc).__name__, exc,
        )
        return {"status": "FAIL", "fit_id": fit_id, "method": "pathfinder"}


def _try_map_estimate(
    *,
    stan_file: Path,
    stan_data: Dict[str, Any],
    fit_id: str,
    col_names: List[str],
    dim_levels: Dict[str, List[str]],
    model_type: str,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Attempt MAP (maximum a posteriori) point estimate as last resort.

    Returns status="MAP" with point-estimate amplification (no
    posterior uncertainty — CrI low/high equal the point estimate).
    """
    logger.info("Fitting %s via MAP optimisation...", fit_id)

    try:
        from cmdstanpy import CmdStanModel
        model = CmdStanModel(stan_file=str(stan_file))
        mle = model.optimize(
            data=stan_data,
            seed=seed,
            algorithm="lbfgs",
            jacobian=True,
        )
    except Exception as exc:
        logger.warning(
            "MAP optimisation failed for %s: %s: %s",
            fit_id, type(exc).__name__, exc,
        )
        return {"status": "FAIL", "fit_id": fit_id, "method": "MAP"}

    map_diag = {
        "max_rhat": float("nan"),
        "min_ess_bulk": float("nan"),
        "divergences": 0,
        "treedepth_hits": 0,
        "n_stuck_params": 0,
        "stuck_params": [],
        "diagnose_text": "MAP point estimate; no posterior uncertainty.",
    }

    try:
        # MAP returns single-point estimates. Wrap in a thin adapter
        # so the amplification functions see (1, K) shaped arrays.
        class _MapAdapter:
            """Wraps CmdStanMLE so stan_variable returns 2-D arrays."""
            def __init__(self, mle_fit: Any) -> None:
                self._mle = mle_fit
            def stan_variable(self, name: str) -> np.ndarray:
                val = self._mle.stan_variable(name)
                val = np.atleast_1d(np.asarray(val, dtype=float))
                if val.ndim == 0:
                    val = val.reshape(1)
                if val.ndim == 1:
                    val = val.reshape(1, -1)  # (1, K)
                return val

        adapter = _MapAdapter(mle)

        # For logistic model, alpha is scalar → needs (1,) shape
        if model_type == "logistic":
            alpha_raw = mle.stan_variable("alpha")
            alpha_arr = np.atleast_1d(np.asarray(alpha_raw, dtype=float))
            if alpha_arr.ndim == 0:
                alpha_arr = alpha_arr.reshape(1)

            class _MapLogisticAdapter(_MapAdapter):
                def stan_variable(self, name: str) -> np.ndarray:
                    if name == "alpha":
                        return alpha_arr
                    return super().stan_variable(name)

            adapter = _MapLogisticAdapter(mle)
            amp = _compute_amplification_logistic(adapter, col_names, dim_levels)
        else:
            # Ordinal: c is (2,) and beta is (K,)
            c_raw = mle.stan_variable("c")
            c_arr = np.atleast_1d(np.asarray(c_raw, dtype=float))
            if c_arr.ndim == 1:
                c_arr = c_arr.reshape(1, -1)  # (1, 2)

            class _MapOrdinalAdapter(_MapAdapter):
                def stan_variable(self, name: str) -> np.ndarray:
                    if name == "c":
                        return c_arr
                    return super().stan_variable(name)

            adapter = _MapOrdinalAdapter(mle)
            amp = _compute_amplification_ordinal(adapter, col_names, dim_levels)

        amp_summary = {dim: {k: v for k, v in dd.items() if k != "A_draws"}
                       for dim, dd in amp.items()}

        result = {
            "fit_id": fit_id,
            "diagnostics": map_diag,
            "status": "MAP",
            "amplification": amp_summary,
            "method": f"MAP {model_type}",
        }
        _save_artifacts(fit_id, model_type, amp, map_diag, "MAP",
                        stan_file, output_dir)
        logger.info("MAP succeeded for %s", fit_id)
        return result

    except Exception as exc:
        logger.warning(
            "MAP amplification computation failed for %s: %s: %s",
            fit_id, type(exc).__name__, exc,
        )
        return {"status": "FAIL", "fit_id": fit_id, "method": "MAP"}


# ------------------------------------------------------------------
# Main public API
# ------------------------------------------------------------------

def run_bayes_robustness(
    baseline_sev: pd.DataFrame,
    models: List[str],
    seed: int = 42,
    output_dir: Optional[Path] = None,
    use_cache: bool = True,
    cache_dir: str = "./cache",
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
    use_cache : If True (default), check for cached results before computing.
    cache_dir : Directory for analysis cache files.

    Returns
    -------
    Nested dict: {model: {metric: {logistic: {...}, ordinal: {...}}}}
    """
    # Check cache first
    cache = None
    if use_cache:
        from src.analysis_cache import get_analysis_cache
        cache = get_analysis_cache(cache_dir)
        dataframes = {"baseline_sev": baseline_sev}
        params = {"models": models, "seed": seed}
        cached_result = cache.get("bayes_robustness", dataframes, params)
        if cached_result is not None:
            return cached_result

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

            # ── Pre-flight: collapse rare severity bands ──
            df = _collapse_rare_severity_bands(df, severity_metric)

            # Build design matrix (shared by B1 and B2)
            # Adaptive interaction pruning is built into _build_design_matrix
            X, col_names = _build_design_matrix(df, severity_metric)

            # ── Pre-flight: drop zero-variance columns ──
            X, col_names = _drop_zero_variance_columns(X, col_names)
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
            # Apply globally for both severity metrics: if too many
            # complaints are singletons, the complaint-level random
            # intercepts are unidentifiable.
            use_random_intercept = True
            if pct_singletons > _SINGLETON_THRESHOLD:
                use_random_intercept = False
                logger.info(
                    "%s/%s: %.0f%% of complaints are singletons (>%.0f%% threshold) "
                    "-> using fixed-effect models. "
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

            # ── Build response variables ──
            y_logistic = (df["tier"].values == 2).astype(int).tolist()
            y_ordinal = np.clip((df["tier"].values + 1).astype(int), 1, 3).tolist()

            # ── Pre-fit separation check ──────────────────────────────
            # Complete or quasi-complete separation makes the logistic
            # likelihood flat, so the MCMC sampler gets stuck at initial
            # values (constant draws → rhat >> 1, ess = num_chains).
            # Detect this BEFORE entering the fallback chain to avoid
            # wasting time on 5 doomed fits.
            _MIN_EVENTS = 10  # need ≥10 in each class for stable logistic
            n_pos = sum(y_logistic)
            n_neg = len(y_logistic) - n_pos
            if n_pos < _MIN_EVENTS or n_neg < _MIN_EVENTS:
                logger.warning(
                    "%s/%s: skipping Bayesian fits — response has %d "
                    "tier-2 vs %d non-tier-2 (need ≥%d in each class to "
                    "avoid complete separation)",
                    model, severity_metric, n_pos, n_neg, _MIN_EVENTS,
                )
                model_results[severity_metric] = {
                    **metric_result,
                    "status": "SKIP",
                    "reason": (
                        f"complete separation: {n_pos} tier-2 vs "
                        f"{n_neg} non-tier-2 (min {_MIN_EVENTS} required)"
                    ),
                }
                continue

            # Similarly check ordinal: need ≥5 observations per tier level
            _MIN_PER_LEVEL = 5
            ordinal_counts = np.bincount(y_ordinal, minlength=4)[1:]  # levels 1,2,3
            sparse_levels = [
                f"tier{i}" for i, c in enumerate(ordinal_counts)
                if c < _MIN_PER_LEVEL
            ]
            if sparse_levels:
                logger.warning(
                    "%s/%s: ordinal response has sparse levels %s "
                    "(counts: tier0=%d, tier1=%d, tier2=%d); "
                    "B2 ordinal may fail",
                    model, severity_metric, sparse_levels,
                    ordinal_counts[0], ordinal_counts[1], ordinal_counts[2],
                )

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
                if flex and use_ri:
                    return _LOGIT_FLEX_STAN, _ORDINAL_FLEX_STAN
                if use_ri:
                    return _LOGIT_STAN, _ORDINAL_STAN
                return _LOGIT_FIXED_STAN, _ORDINAL_FIXED_STAN

            # ── Pre-flight diagnostics ──
            preflight = _preflight_diagnostics(
                X=X, col_names=col_names,
                y_logistic=y_logistic, y_ordinal=y_ordinal,
                severity_metric=severity_metric, model_name=model,
                df=df, J=J, pct_singletons=pct_singletons,
            )
            metric_result["preflight"] = preflight

            if preflight["predicted_difficulty"] == "very_hard":
                logger.warning(
                    "PRE-FLIGHT %s/%s: predicted difficulty VERY HARD — "
                    "EPP=%.1f, rank=%d/%d, condition=%.2e, %d sparse cells. "
                    "Recommendations: %s",
                    model, severity_metric, preflight["epp"],
                    preflight["matrix_rank"], X.shape[1],
                    preflight["condition_number"], preflight["n_sparse_cells"],
                    "; ".join(preflight["recommendations"]),
                )
            elif preflight["predicted_difficulty"] == "hard":
                logger.warning(
                    "PRE-FLIGHT %s/%s: predicted difficulty HARD — "
                    "EPP=%.1f. Some fallback steps may be needed.",
                    model, severity_metric, preflight["epp"],
                )

            # ── Pre-flight auto-simplification ──
            if preflight["predicted_difficulty"] == "very_hard":
                X, col_names, simplification = _auto_simplify_design_matrix(
                    df=df, severity_metric=severity_metric,
                    X_orig=X, col_names_orig=col_names,
                    y_logistic=y_logistic,
                )
                if simplification != "none":
                    metric_result["auto_simplification"] = simplification
                    metric_result["K"] = X.shape[1]
                    metric_result["col_names"] = col_names
                    dim_levels = _get_dimension_levels(df)

            # ── Fallback chain with independent B1/B2 tracking ──
            # The chain progressively tries more robust configurations.
            # B1 (logistic) and B2 (ordinal) are tracked independently:
            # if one succeeds at step N, only the other is retried.
            #
            # Steps 1-3.5 use strict diagnostic gates.
            # Steps 4-5 use relaxed gates (MARGINAL status for borderline fits).
            # Steps 6-8 use non-MCMC methods (ADVI, Pathfinder, MAP).
            attempted: List[Dict[str, Any]] = []
            result_b1: Optional[Dict[str, Any]] = None
            result_b2: Optional[Dict[str, Any]] = None
            fallback_b1: Optional[str] = None
            fallback_b2: Optional[str] = None
            need_b1 = True
            need_b2 = True

            _FALLBACK_ORDER = [
                "standard", "category-RE", "per-dimension",
                "main-effects-only", "flex-prior", "aggressive-sampler",
                "variational-advi", "pathfinder", "map-estimate",
            ]

            def _step_diag(
                step_num: Any,
                label: str,
                description: str,
                step_result: Dict[str, Any],
            ) -> Dict[str, Any]:
                """Build a structured entry for the attempted list."""
                entry: Dict[str, Any] = {
                    "step": step_num,
                    "label": label,
                    "description": description,
                }
                for prefix, mtype in [("b1", "logistic"), ("b2", "ordinal")]:
                    mdata = step_result.get(mtype, {})
                    entry[f"{prefix}_status"] = mdata.get("status", "N/A")
                    diag = mdata.get("diagnostics", {})
                    if diag:
                        entry[f"{prefix}_rhat"] = diag.get("max_rhat", float("nan"))
                        entry[f"{prefix}_ess"] = diag.get("min_ess_bulk", float("nan"))
                        entry[f"{prefix}_div"] = diag.get("divergences", 0)
                return entry

            def _which_attempt() -> str:
                if need_b1 and need_b2:
                    return "both"
                if need_b1:
                    logger.info("  (attempting B1 logistic only — B2 already resolved)")
                    return "b1_only"
                logger.info("  (attempting B2 ordinal only — B1 already resolved)")
                return "b2_only"

            def _harvest(
                step_result: Dict[str, Any],
                label: str,
            ) -> None:
                """Extract successful B1/B2 from step_result into tracking vars."""
                nonlocal result_b1, result_b2, fallback_b1, fallback_b2
                nonlocal need_b1, need_b2
                b1 = step_result.get("logistic", {})
                b2 = step_result.get("ordinal", {})
                if need_b1 and b1.get("status") not in ("FAIL", None):
                    result_b1 = b1
                    fallback_b1 = label
                    need_b1 = False
                    logger.info("  -> B1 logistic resolved at [%s] (status=%s)",
                                label, b1.get("status"))
                if need_b2 and b2.get("status") not in ("FAIL", None):
                    result_b2 = b2
                    fallback_b2 = label
                    need_b2 = False
                    logger.info("  -> B2 ordinal resolved at [%s] (status=%s)",
                                label, b2.get("status"))

            def _latest_fallback() -> Optional[str]:
                """Return the latest (worst) fallback step used."""
                if fallback_b1 is None and fallback_b2 is None:
                    return None
                idx_b1 = _FALLBACK_ORDER.index(fallback_b1) if fallback_b1 in _FALLBACK_ORDER else -1
                idx_b2 = _FALLBACK_ORDER.index(fallback_b2) if fallback_b2 in _FALLBACK_ORDER else -1
                idx = max(idx_b1, idx_b2)
                return _FALLBACK_ORDER[idx] if idx >= 0 else (fallback_b1 or fallback_b2)

            # ── Step 1: standard fit ──
            method_lbl = "hierarchical" if use_random_intercept else "fixed-effect"
            logger.info("%s fallback chain: step 1 -- %s fit (K=%d)",
                        severity_metric, method_lbl, X.shape[1])
            b1_stan, b2_stan = _pick_stan_files(use_random_intercept, flex=False)
            sd_b1, sd_b2 = _make_stan_data(X, use_random_intercept)
            step_result = _try_b1_b2(
                df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                b1_stan_file=b1_stan, b2_stan_file=b2_stan,
                stan_data_b1=sd_b1, stan_data_b2=sd_b2,
                fit_id_b1=f"B1__{fit_prefix}", fit_id_b2=f"B2__{fit_prefix}",
                seed=seed, output_dir=output_dir,
                method_label=method_lbl, attempt=_which_attempt(),
            )
            _harvest(step_result, "standard")
            if not need_b1 and not need_b2:
                logger.info("%s: step 1 (%s) succeeded for both B1+B2", severity_metric, method_lbl)
            else:
                attempted.append(_step_diag(1, "standard", f"{method_lbl} (K={X.shape[1]})", step_result))
                logger.warning("%s: step 1 (%s) — need_b1=%s, need_b2=%s",
                               severity_metric, method_lbl, need_b1, need_b2)

            # ── Fallback 2: category-level random intercept ──
            if need_b1 or need_b2:
                cat_col = "product_category"
                if cat_col in df.columns and df[cat_col].notna().sum() > 0:
                    cat_vals = df[cat_col].fillna("_unknown_").values
                    unique_cats = np.unique(cat_vals)
                    cat_map = {c: i + 1 for i, c in enumerate(unique_cats)}
                    jj_cat = np.array([cat_map[c] for c in cat_vals])
                    J_cat = len(unique_cats)

                    logger.info(
                        "%s fallback chain: step 2 -- category-level RE (%s, J=%d)",
                        severity_metric, cat_col, J_cat,
                    )
                    b1_stan_cat, b2_stan_cat = _LOGIT_STAN, _ORDINAL_STAN
                    common_cat: Dict[str, Any] = {
                        "N": len(df), "K": X.shape[1], "X": X.tolist(),
                        "J": J_cat, "jj": jj_cat.tolist(),
                    }
                    step_result = _try_b1_b2(
                        df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                        b1_stan_file=b1_stan_cat, b2_stan_file=b2_stan_cat,
                        stan_data_b1={**common_cat, "y": y_logistic},
                        stan_data_b2={**common_cat, "y": y_ordinal},
                        fit_id_b1=f"B1__{fit_prefix}__catRE",
                        fit_id_b2=f"B2__{fit_prefix}__catRE",
                        seed=seed, output_dir=output_dir,
                        method_label=f"category-RE({cat_col})",
                        attempt=_which_attempt(),
                    )
                    _harvest(step_result, "category-RE")
                    if not need_b1 and not need_b2:
                        logger.info("%s: step 2 (category RE) succeeded", severity_metric)
                    else:
                        attempted.append(_step_diag(2, "category-RE", f"category-RE({cat_col}, J={J_cat})", step_result))
                        logger.warning("%s: step 2 (category RE) — need_b1=%s, need_b2=%s",
                                       severity_metric, need_b1, need_b2)
                else:
                    attempted.append({"step": 2, "label": "category-RE", "description": f"skipped ({cat_col} not available)", "b1_status": "N/A", "b2_status": "N/A"})
                    logger.warning(
                        "%s fallback chain: step 2 skipped -- %s column not available",
                        severity_metric, cat_col,
                    )

            # ── Fallback 3: per-dimension models (partial credit + retry) ──
            if need_b1 or need_b2:
                logger.info(
                    "%s fallback chain: step 3 -- per-dimension models",
                    severity_metric,
                )
                per_dim_result = _try_per_dimension(
                    df=df,
                    severity_metric=severity_metric,
                    jj=jj, J=J,
                    use_random_intercept=use_random_intercept,
                    fit_id_prefix=fit_prefix,
                    seed=seed,
                    output_dir=output_dir,
                    attempt=_which_attempt(),
                )
                if per_dim_result is not None:
                    is_partial = per_dim_result.get("logistic", {}).get("partial", False)
                    label = "per-dimension (partial)" if is_partial else "per-dimension"
                    _harvest(per_dim_result, label)
                    if not need_b1 and not need_b2:
                        logger.info("%s: step 3 (per-dimension) succeeded", severity_metric)
                    else:
                        attempted.append(_step_diag(3, label, "per-dimension models", per_dim_result))
                        logger.warning("%s: step 3 — need_b1=%s, need_b2=%s",
                                       severity_metric, need_b1, need_b2)
                else:
                    attempted.append({"step": 3, "label": "per-dimension", "description": "per-dimension models", "b1_status": "FAIL", "b2_status": "FAIL"})
                    logger.warning("%s: step 3 (per-dimension) FAILED", severity_metric)

            # ── Fallback 3.5: main-effects-only model ──
            if need_b1 or need_b2:
                logger.info(
                    "%s fallback chain: step 3.5 -- main-effects-only model",
                    severity_metric,
                )
                X_main, col_names_main = _build_design_matrix(
                    df, severity_metric, include_interactions=False,
                )
                X_main, col_names_main = _drop_zero_variance_columns(X_main, col_names_main)
                dim_levels_main = _get_dimension_levels(df)

                b1_stan_main, b2_stan_main = _pick_stan_files(use_random_intercept, flex=False)
                sd_b1_main, sd_b2_main = _make_stan_data(X_main, use_random_intercept)

                step_result = _try_b1_b2(
                    df=df, X=X_main, col_names=col_names_main,
                    dim_levels=dim_levels_main,
                    b1_stan_file=b1_stan_main, b2_stan_file=b2_stan_main,
                    stan_data_b1=sd_b1_main, stan_data_b2=sd_b2_main,
                    fit_id_b1=f"B1__{fit_prefix}__mainonly",
                    fit_id_b2=f"B2__{fit_prefix}__mainonly",
                    seed=seed, output_dir=output_dir,
                    method_label="main-effects-only",
                    attempt=_which_attempt(),
                )
                _harvest(step_result, "main-effects-only")
                if not need_b1 and not need_b2:
                    logger.info("%s: step 3.5 (main-effects-only) succeeded", severity_metric)
                else:
                    attempted.append(_step_diag("3.5", "main-effects-only", f"main-effects-only (K={X_main.shape[1]})", step_result))
                    logger.warning("%s: step 3.5 — need_b1=%s, need_b2=%s",
                                   severity_metric, need_b1, need_b2)

            # ── Fallback 4: stronger priors with relaxed gate ──
            # Build per-column prior_sd: 0.8 for main effects, 0.4 for
            # interaction terms (columns containing '__x__').
            flex_prior_sd = np.array([
                0.4 if "__x__" in col else 0.8 for col in col_names
            ])
            if need_b1 or need_b2:
                logger.info(
                    "%s fallback chain: step 4 -- stronger priors + relaxed gate",
                    severity_metric,
                )
                b1_flex, b2_flex = _pick_stan_files(use_random_intercept, flex=True)
                sd_b1_flex, sd_b2_flex = _make_stan_data(
                    X, use_random_intercept, prior_sd=flex_prior_sd,
                )
                step_result = _try_b1_b2(
                    df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                    b1_stan_file=b1_flex, b2_stan_file=b2_flex,
                    stan_data_b1=sd_b1_flex, stan_data_b2=sd_b2_flex,
                    fit_id_b1=f"B1__{fit_prefix}__flex",
                    fit_id_b2=f"B2__{fit_prefix}__flex",
                    seed=seed, output_dir=output_dir,
                    method_label="flex-prior",
                    attempt=_which_attempt(),
                    strict_gate=False,
                )
                _harvest(step_result, "flex-prior")
                if not need_b1 and not need_b2:
                    logger.info("%s: step 4 (stronger priors) succeeded", severity_metric)
                else:
                    attempted.append(_step_diag(4, "flex-prior", "flex priors + relaxed gate", step_result))
                    logger.warning("%s: step 4 — need_b1=%s, need_b2=%s",
                                   severity_metric, need_b1, need_b2)

            # ── Fallback 5: aggressive sampler with relaxed gate ──
            if need_b1 or need_b2:
                logger.info(
                    "%s fallback chain: step 5 -- aggressive sampler + relaxed gate",
                    severity_metric,
                )
                b1_flex, b2_flex = _pick_stan_files(use_random_intercept, flex=True)
                sd_b1_agg, sd_b2_agg = _make_stan_data(
                    X, use_random_intercept, prior_sd=flex_prior_sd,
                )
                step_result = _try_b1_b2(
                    df=df, X=X, col_names=col_names, dim_levels=dim_levels,
                    b1_stan_file=b1_flex, b2_stan_file=b2_flex,
                    stan_data_b1=sd_b1_agg, stan_data_b2=sd_b2_agg,
                    fit_id_b1=f"B1__{fit_prefix}__aggressive",
                    fit_id_b2=f"B2__{fit_prefix}__aggressive",
                    seed=seed, output_dir=output_dir,
                    adapt_delta=0.99,
                    max_treedepth=15,
                    method_label="aggressive-sampler",
                    attempt=_which_attempt(),
                    strict_gate=False,
                )
                _harvest(step_result, "aggressive-sampler")
                if not need_b1 and not need_b2:
                    logger.info("%s: step 5 (aggressive sampler) succeeded", severity_metric)
                else:
                    attempted.append(_step_diag(5, "aggressive-sampler", "aggressive sampler + relaxed gate", step_result))
                    logger.warning("%s: step 5 — need_b1=%s, need_b2=%s",
                                   severity_metric, need_b1, need_b2)

            # ── Fallback 6: Variational inference (ADVI) ──
            if need_b1 or need_b2:
                logger.info(
                    "%s fallback chain: step 6 -- variational inference (ADVI)",
                    severity_metric,
                )
                b1_stan_vi, b2_stan_vi = _pick_stan_files(use_ri=False, flex=False)
                sd_b1_vi, sd_b2_vi = _make_stan_data(X, False)

                vi_result: Dict[str, Any] = {}
                if need_b1:
                    vi_result["logistic"] = _try_variational(
                        stan_file=b1_stan_vi, stan_data=sd_b1_vi,
                        fit_id=f"B1__{fit_prefix}__advi",
                        col_names=col_names, dim_levels=dim_levels,
                        model_type="logistic", seed=seed, output_dir=output_dir,
                    )
                if need_b2:
                    vi_result["ordinal"] = _try_variational(
                        stan_file=b2_stan_vi, stan_data=sd_b2_vi,
                        fit_id=f"B2__{fit_prefix}__advi",
                        col_names=col_names, dim_levels=dim_levels,
                        model_type="ordinal", seed=seed, output_dir=output_dir,
                    )
                _harvest(vi_result, "variational-advi")
                if not need_b1 and not need_b2:
                    logger.info("%s: step 6 (ADVI) succeeded", severity_metric)
                else:
                    attempted.append(_step_diag(6, "variational-advi", "ADVI meanfield", vi_result))
                    logger.warning("%s: step 6 (ADVI) — need_b1=%s, need_b2=%s",
                                   severity_metric, need_b1, need_b2)

            # ── Fallback 7: Pathfinder ──
            if need_b1 or need_b2:
                logger.info(
                    "%s fallback chain: step 7 -- Pathfinder",
                    severity_metric,
                )
                b1_stan_pf, b2_stan_pf = _pick_stan_files(use_ri=False, flex=False)
                sd_b1_pf, sd_b2_pf = _make_stan_data(X, False)

                pf_result: Dict[str, Any] = {}
                if need_b1:
                    pf_result["logistic"] = _try_pathfinder(
                        stan_file=b1_stan_pf, stan_data=sd_b1_pf,
                        fit_id=f"B1__{fit_prefix}__pathfinder",
                        col_names=col_names, dim_levels=dim_levels,
                        model_type="logistic", seed=seed, output_dir=output_dir,
                    )
                if need_b2:
                    pf_result["ordinal"] = _try_pathfinder(
                        stan_file=b2_stan_pf, stan_data=sd_b2_pf,
                        fit_id=f"B2__{fit_prefix}__pathfinder",
                        col_names=col_names, dim_levels=dim_levels,
                        model_type="ordinal", seed=seed, output_dir=output_dir,
                    )
                _harvest(pf_result, "pathfinder")
                if not need_b1 and not need_b2:
                    logger.info("%s: step 7 (Pathfinder) succeeded", severity_metric)
                else:
                    attempted.append(_step_diag(7, "pathfinder", "Pathfinder (4 paths)", pf_result))
                    logger.warning("%s: step 7 (Pathfinder) — need_b1=%s, need_b2=%s",
                                   severity_metric, need_b1, need_b2)

            # ── Fallback 8: MAP point estimate ──
            if need_b1 or need_b2:
                logger.info(
                    "%s fallback chain: step 8 -- MAP point estimate",
                    severity_metric,
                )
                b1_stan_map, b2_stan_map = _pick_stan_files(use_ri=False, flex=False)
                sd_b1_map, sd_b2_map = _make_stan_data(X, False)

                map_result: Dict[str, Any] = {}
                if need_b1:
                    map_result["logistic"] = _try_map_estimate(
                        stan_file=b1_stan_map, stan_data=sd_b1_map,
                        fit_id=f"B1__{fit_prefix}__map",
                        col_names=col_names, dim_levels=dim_levels,
                        model_type="logistic", seed=seed, output_dir=output_dir,
                    )
                if need_b2:
                    map_result["ordinal"] = _try_map_estimate(
                        stan_file=b2_stan_map, stan_data=sd_b2_map,
                        fit_id=f"B2__{fit_prefix}__map",
                        col_names=col_names, dim_levels=dim_levels,
                        model_type="ordinal", seed=seed, output_dir=output_dir,
                    )
                _harvest(map_result, "map-estimate")
                if not need_b1 and not need_b2:
                    logger.info("%s: step 8 (MAP) succeeded", severity_metric)
                else:
                    attempted.append(_step_diag(8, "map-estimate", "MAP (L-BFGS)", map_result))
                    logger.warning("%s: step 8 (MAP) — need_b1=%s, need_b2=%s",
                                   severity_metric, need_b1, need_b2)

            # ── Assemble final metric_result ──
            logger.info(
                "%s/%s fallback chain summary: B1=%s (step=%s), B2=%s (step=%s)",
                model, severity_metric,
                result_b1.get("status", "FAIL") if result_b1 else "FAIL",
                fallback_b1 or "none",
                result_b2.get("status", "FAIL") if result_b2 else "FAIL",
                fallback_b2 or "none",
            )
            if result_b1 is not None:
                metric_result["logistic"] = result_b1
            else:
                metric_result["logistic"] = {"status": "FAIL"}

            if result_b2 is not None:
                metric_result["ordinal"] = result_b2
            else:
                metric_result["ordinal"] = {"status": "FAIL"}

            # Track which fallback each model type used
            if fallback_b1:
                metric_result["fallback_step_b1"] = fallback_b1
            if fallback_b2:
                metric_result["fallback_step_b2"] = fallback_b2
            # Backward compat: overall fallback_step = latest/worst of the two
            fb = _latest_fallback()
            if fb:
                metric_result["fallback_step"] = fb

            # Record total failure / partial failure and generate post-mortem
            if need_b1 and need_b2:
                msg_lines = [
                    f"All Bayesian fallbacks FAILED for {model}/{severity_metric}.",
                    "Attempted strategies:",
                ]
                for a in attempted:
                    msg_lines.append(
                        f"  - Step {a['step']} [{a['label']}]: {a['description']} "
                        f"| B1={a.get('b1_status', 'N/A')} B2={a.get('b2_status', 'N/A')}"
                    )
                msg_lines.append("Posterior draws are unreliable for all attempted configurations.")
                full_msg = "\n".join(msg_lines)
                logger.error(full_msg)
                metric_result["status"] = "FAIL"
                metric_result["error"] = full_msg

                # Generate post-mortem
                postmortem = _build_postmortem(
                    df=df, X=X, col_names=col_names,
                    y_logistic=y_logistic, y_ordinal=y_ordinal,
                    severity_metric=severity_metric, model_name=model,
                    attempted=attempted,
                    use_random_intercept=use_random_intercept,
                    J=J, pct_singletons=pct_singletons,
                    preflight=preflight,
                )
                _log_postmortem(postmortem, model, severity_metric)
                _save_postmortem(postmortem, model, severity_metric, output_dir)
                metric_result["postmortem"] = postmortem

            elif need_b1 or need_b2:
                failed_type = "B1 logistic" if need_b1 else "B2 ordinal"
                ok_type = "B2 ordinal" if need_b1 else "B1 logistic"
                logger.warning(
                    "%s/%s: %s succeeded but %s failed after all fallbacks",
                    model, severity_metric, ok_type, failed_type,
                )

                # Generate partial post-mortem for the failed model type
                postmortem = _build_postmortem(
                    df=df, X=X, col_names=col_names,
                    y_logistic=y_logistic, y_ordinal=y_ordinal,
                    severity_metric=severity_metric, model_name=model,
                    attempted=attempted,
                    use_random_intercept=use_random_intercept,
                    J=J, pct_singletons=pct_singletons,
                    preflight=preflight,
                )
                _log_postmortem(postmortem, model, severity_metric)
                _save_postmortem(postmortem, model, severity_metric, output_dir)
                metric_result["postmortem"] = postmortem

            model_results[severity_metric] = metric_result

        results[model] = model_results

    # Cache the results
    if use_cache and cache is not None:
        cache.put("bayes_robustness", dataframes, params, results)

    return results
