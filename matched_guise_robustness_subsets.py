#!/usr/bin/env python3
"""Matched-guise robustness subset analysis.

Runs the core outcome analysis on filtered subsets of matched-guise pairs
(FULL, HF, NO_POL, NO_NUM, HF_NO_POL) and generates a robustness report
comparing persona-dispersion effects across subsets.

No API calls — deterministic analysis from cached experiment outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBSET_NAMES = ["FULL", "HF", "NO_POL", "NO_NUM", "HF_NO_POL"]

DEFAULT_THRESHOLDS = {
    "min_pair_retention": 0.40,
    "min_complaint_retention": 0.25,
    "max_attenuation": 0.40,
    "alpha": 0.05,
    "min_complaints_for_tests": 30,
    "min_persona_levels_for_tests": 2,
}

BOOTSTRAP_METRICS = [
    "mean_tier",
    "p_tier2",
    "max_persona_gap_p_tier2",
    "max_persona_gap_mean_tier",
]


# ---------------------------------------------------------------------------
# 1-2. Load, map columns, validate
# ---------------------------------------------------------------------------

def load_and_map(
    decisions_path: Path,
    audit_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSVs and rename columns to canonical names."""
    dec = pd.read_csv(decisions_path)
    aud = pd.read_csv(audit_path)

    # --- Decisions column mapping ---
    dec_rename = {}
    if "case_id" in dec.columns and "complaint_id" not in dec.columns:
        dec_rename["case_id"] = "complaint_id"
    if "model" in dec.columns and "model_name" not in dec.columns:
        dec_rename["model"] = "model_name"
    if "variant" in dec.columns and "persona_label" not in dec.columns:
        dec_rename["variant"] = "persona_label"
    if dec_rename:
        dec = dec.rename(columns=dec_rename)

    # Derived columns
    dec["tier"] = dec["tier"].astype(int)
    dec["tier2"] = (dec["tier"] == 2).astype(int)
    if "confidence" in dec.columns:
        dec["need_more_info"] = (dec["confidence"] == "need_more_info").astype(int)
    else:
        dec["need_more_info"] = 0

    # --- Audit column mapping ---
    aud_rename = {}
    if "high_fidelity_pass" in aud.columns:
        aud_rename["high_fidelity_pass"] = "high_fidelity_pair_pass"
    if "has_contradiction" in aud.columns:
        aud_rename["has_contradiction"] = "polarity_contradiction_flag"
    if aud_rename:
        aud = aud.rename(columns=aud_rename)

    # Convert bool strings to int
    for col in ["high_fidelity_pair_pass", "polarity_contradiction_flag",
                "strict_pair_pass"]:
        if col in aud.columns:
            aud[col] = aud[col].map(
                {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
            ).fillna(0).astype(int)

    # Derive numeric mutation flag
    if "numeric_status" in aud.columns:
        aud["numeric_mutation_flag"] = (
            aud["numeric_status"] == "MUTATION"
        ).astype(int)
    else:
        aud["numeric_mutation_flag"] = 0

    return dec, aud


def validate_inputs(dec: pd.DataFrame, aud: pd.DataFrame) -> List[str]:
    """Run validation checks on mapped DataFrames. Return list of errors."""
    errors = []
    for col in ["complaint_id", "model_name", "persona_label", "tier"]:
        if col not in dec.columns:
            errors.append(f"Decisions missing required column: {col}")
    for col in ["complaint_id", "model_name", "persona_label",
                "high_fidelity_pair_pass", "polarity_contradiction_flag"]:
        if col not in aud.columns:
            errors.append(f"Audit missing required column: {col}")

    if not errors:
        bad_tiers = dec[~dec["tier"].isin([0, 1, 2])]
        if len(bad_tiers) > 0:
            errors.append(
                f"Tier values outside {{0,1,2}}: {bad_tiers['tier'].unique().tolist()}"
            )
        if dec["model_name"].nunique() < 1:
            errors.append("No models found in decisions")
        if dec["persona_label"].nunique() < 2:
            errors.append("Fewer than 2 persona levels in decisions")

    return errors


# ---------------------------------------------------------------------------
# 3. Join
# ---------------------------------------------------------------------------

def merge_decisions_with_audit(
    dec: pd.DataFrame,
    aud: pd.DataFrame,
    output_dir: Path,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Join decisions with audit flags. Write join diagnostics."""
    join_keys = ["complaint_id", "model_name", "persona_label"]
    n_dec = len(dec)
    n_aud = len(aud)

    # Select audit columns needed
    audit_cols = join_keys + [
        "high_fidelity_pair_pass",
        "polarity_contradiction_flag",
        "numeric_mutation_flag",
        "strict_pair_pass",
    ]
    audit_cols = [c for c in audit_cols if c in aud.columns]

    merged = dec.merge(aud[audit_cols], on=join_keys, how="inner")
    n_matched = len(merged)
    coverage = n_matched / n_dec if n_dec > 0 else 0.0

    # Diagnostics
    diag = {
        "decisions_rows": n_dec,
        "audit_rows": n_aud,
        "matched_rows": n_matched,
        "unmatched_decisions": n_dec - n_matched,
        "unmatched_audit": n_aud - n_matched,
        "join_keys": join_keys,
        "coverage": round(coverage, 4),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "join_diagnostics.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8"
    )
    logger.info(
        "Join: %d decisions × %d audit → %d matched (%.1f%% coverage)",
        n_dec, n_aud, n_matched, coverage * 100,
    )

    if coverage < min_coverage:
        raise ValueError(
            f"Join coverage {coverage:.4f} below minimum {min_coverage}. "
            f"Check column alignment between decisions and audit CSVs."
        )

    # Check for duplicates
    dups = merged.duplicated(subset=join_keys, keep=False)
    if dups.any():
        n_dups = dups.sum()
        logger.warning("Found %d duplicate rows after join; dropping duplicates", n_dups)
        merged = merged.drop_duplicates(subset=join_keys, keep="first")

    return merged


# ---------------------------------------------------------------------------
# 4. Subset construction
# ---------------------------------------------------------------------------

def build_subsets(
    df: pd.DataFrame,
    include_hf_no_pol: bool = True,
) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    """Return (masks, aliases) where aliases maps e.g. HF_NO_POL -> HF."""
    masks: Dict[str, pd.Series] = {
        "FULL": pd.Series(True, index=df.index),
        "HF": df["high_fidelity_pair_pass"] == 1,
        "NO_POL": df["polarity_contradiction_flag"] == 0,
        "NO_NUM": df["numeric_mutation_flag"] == 0,
    }
    aliases: Dict[str, str] = {}
    if include_hf_no_pol:
        hf_no_pol = masks["HF"] & masks["NO_POL"]
        masks["HF_NO_POL"] = hf_no_pol
        # Detect if HF_NO_POL is identical to HF (happens when HF already
        # requires no polarity contradiction)
        if (hf_no_pol == masks["HF"]).all():
            aliases["HF_NO_POL"] = "HF"
            logger.info("HF_NO_POL is identical to HF (HF already excludes contradictions); marked as alias")
    return masks, aliases


# ---------------------------------------------------------------------------
# 5. Metric computation
# ---------------------------------------------------------------------------

def _persona_dispersion(
    sub: pd.DataFrame,
) -> Dict[str, float]:
    """Compute persona-level dispersion metrics."""
    persona_stats = sub.groupby("persona_label").agg(
        mean_tier=("tier", "mean"),
        p_tier2=("tier2", "mean"),
    )
    if len(persona_stats) < 2:
        return {
            "max_persona_gap_p_tier2": float("nan"),
            "max_persona_gap_mean_tier": float("nan"),
            "sd_persona_p_tier2": float("nan"),
            "sd_persona_mean_tier": float("nan"),
        }
    return {
        "max_persona_gap_p_tier2": float(
            persona_stats["p_tier2"].max() - persona_stats["p_tier2"].min()
        ),
        "max_persona_gap_mean_tier": float(
            persona_stats["mean_tier"].max() - persona_stats["mean_tier"].min()
        ),
        "sd_persona_p_tier2": float(persona_stats["p_tier2"].std()),
        "sd_persona_mean_tier": float(persona_stats["mean_tier"].std()),
    }


def compute_subset_metrics(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    models: List[str],
) -> pd.DataFrame:
    """Compute metrics per subset × model. Returns a summary DataFrame."""
    # Precompute FULL counts per model for retention calculation
    full_counts: Dict[str, Dict[str, int]] = {}
    for model in models:
        full_m = df[masks["FULL"] & (df["model_name"] == model)]
        full_counts[model] = {
            "n_pairs": len(full_m),
            "n_complaints": full_m["complaint_id"].nunique(),
        }

    rows = []
    for subset_name, mask in masks.items():
        for model in models:
            sub = df[mask & (df["model_name"] == model)]
            n_pairs = len(sub)
            n_complaints = sub["complaint_id"].nunique()

            fc = full_counts[model]
            pair_ret = n_pairs / fc["n_pairs"] if fc["n_pairs"] > 0 else 0.0
            comp_ret = n_complaints / fc["n_complaints"] if fc["n_complaints"] > 0 else 0.0

            rec: Dict[str, Any] = {
                "subset": subset_name,
                "model_name": model,
                "n_pairs": n_pairs,
                "n_complaints": n_complaints,
                "pair_retention": round(pair_ret, 4),
                "complaint_retention": round(comp_ret, 4),
                "mean_tier": round(float(sub["tier"].mean()), 6) if n_pairs else float("nan"),
                "p_tier2": round(float(sub["tier2"].mean()), 6) if n_pairs else float("nan"),
            }

            if "need_more_info" in sub.columns:
                rec["p_need_more_info"] = round(
                    float(sub["need_more_info"].mean()), 6
                ) if n_pairs else float("nan")

            # Persona dispersion
            rec.update(_persona_dispersion(sub))
            rows.append(rec)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Bootstrap CIs (complaint-level resampling)
# ---------------------------------------------------------------------------

def _compute_boot_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    """Compute the four bootstrapped metrics on a (resampled) subset."""
    n = len(sub)
    if n == 0:
        return {m: float("nan") for m in BOOTSTRAP_METRICS}
    rec = {
        "mean_tier": float(sub["tier"].mean()),
        "p_tier2": float(sub["tier2"].mean()),
    }
    rec.update(_persona_dispersion(sub))
    return {m: rec.get(m, float("nan")) for m in BOOTSTRAP_METRICS}


def bootstrap_subset_metrics(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    models: List[str],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap CIs via complaint-level resampling. Returns CI DataFrame."""
    rng = np.random.default_rng(seed)
    ci_rows = []
    total_jobs = sum(1 for _ in masks for _ in models)

    with tqdm(total=total_jobs, desc="Bootstrap CIs") as pbar:
        for subset_name, mask in masks.items():
            for model in models:
                sub = df[mask & (df["model_name"] == model)]
                complaints = sub["complaint_id"].unique()
                n_complaints = len(complaints)

                if n_complaints < 2:
                    for metric in BOOTSTRAP_METRICS:
                        ci_rows.append({
                            "subset": subset_name,
                            "model_name": model,
                            "metric": metric,
                            "estimate": float("nan"),
                            "ci_lower": float("nan"),
                            "ci_upper": float("nan"),
                        })
                    pbar.update(1)
                    continue

                # Point estimates
                point = _compute_boot_metrics(sub)

                # Bootstrap
                boot_results = {m: [] for m in BOOTSTRAP_METRICS}
                complaint_to_rows = sub.groupby("complaint_id")

                for _ in range(n_boot):
                    sampled_ids = rng.choice(complaints, size=n_complaints, replace=True)
                    frames = [complaint_to_rows.get_group(cid) for cid in sampled_ids]
                    boot_sub = pd.concat(frames, ignore_index=True)
                    bm = _compute_boot_metrics(boot_sub)
                    for m in BOOTSTRAP_METRICS:
                        boot_results[m].append(bm[m])

                lo_pct = (alpha / 2) * 100
                hi_pct = (1 - alpha / 2) * 100
                for metric in BOOTSTRAP_METRICS:
                    arr = np.array(boot_results[metric])
                    arr = arr[np.isfinite(arr)]
                    if len(arr) > 0:
                        ci_lo = float(np.percentile(arr, lo_pct))
                        ci_hi = float(np.percentile(arr, hi_pct))
                    else:
                        ci_lo = ci_hi = float("nan")
                    ci_rows.append({
                        "subset": subset_name,
                        "model_name": model,
                        "metric": metric,
                        "estimate": round(point[metric], 6),
                        "ci_lower": round(ci_lo, 6),
                        "ci_upper": round(ci_hi, 6),
                    })
                pbar.update(1)

    return pd.DataFrame(ci_rows)


# ---------------------------------------------------------------------------
# 7. GEE omnibus tests
# ---------------------------------------------------------------------------

def _run_gee(
    sub: pd.DataFrame,
    outcome_col: str,
    family: str,
    cluster_col: str = "complaint_id",
) -> Dict[str, Any]:
    """Fit GEE and return Wald p-value for persona block.

    family: "binomial" or "gaussian"
    """
    import statsmodels.api as sm
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable, Independence
    from statsmodels.genmod.families import Binomial, Gaussian

    fam_obj = Binomial() if family == "binomial" else Gaussian()

    # Design matrix
    y = sub[outcome_col].astype(float)
    X = pd.get_dummies(sub["persona_label"], drop_first=True, dtype=float)
    X.insert(0, "Intercept", 1.0)
    groups = sub[cluster_col]

    for cov_name, cov_struct in [("exchangeable", Exchangeable()),
                                  ("independence", Independence())]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GEE(y, X, groups=groups, family=fam_obj,
                            cov_struct=cov_struct)
                result = model.fit(maxiter=100)

            # Wald test for all persona dummies (exclude intercept)
            persona_cols = [c for c in X.columns if c != "Intercept"]
            if not persona_cols:
                return {"p_value": float("nan"), "converged": False,
                        "cov_struct": cov_name, "status": "no_persona_cols"}

            indices = [list(X.columns).index(c) for c in persona_cols]
            contrast = np.zeros((len(indices), len(X.columns)))
            for i, idx in enumerate(indices):
                contrast[i, idx] = 1.0
            wald = result.wald_test(contrast, scalar=True)
            p_val = float(wald.pvalue)

            return {
                "p_value": round(p_val, 6),
                "converged": True,
                "cov_struct": cov_name,
                "n_obs": len(sub),
                "n_clusters": sub[cluster_col].nunique(),
                "status": "OK",
            }
        except Exception as e:
            logger.debug("GEE %s/%s failed: %s", family, cov_name, e)
            continue

    return {
        "p_value": float("nan"),
        "converged": False,
        "cov_struct": "none",
        "status": "convergence_failed",
    }


def run_omnibus_tests(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    models: List[str],
    min_complaints: int = 30,
    min_persona_levels: int = 2,
) -> pd.DataFrame:
    """Run GEE omnibus tests per subset × model. Returns results DataFrame."""
    rows = []
    for subset_name, mask in masks.items():
        for model in models:
            sub = df[mask & (df["model_name"] == model)].copy()
            n_complaints = sub["complaint_id"].nunique()
            n_personas = sub["persona_label"].nunique()
            skip = n_complaints < min_complaints or n_personas < min_persona_levels

            for endpoint, outcome_col, family in [
                ("tier2", "tier2", "binomial"),
                ("tier_numeric", "tier", "gaussian"),
            ]:
                if skip:
                    rows.append({
                        "subset": subset_name,
                        "model_name": model,
                        "endpoint": endpoint,
                        "family": family,
                        "p_value": float("nan"),
                        "converged": False,
                        "cov_struct": "skipped",
                        "n_obs": len(sub),
                        "n_clusters": n_complaints,
                        "status": f"skipped (complaints={n_complaints}, personas={n_personas})",
                    })
                    continue

                result = _run_gee(sub, outcome_col, family)
                result["subset"] = subset_name
                result["model_name"] = model
                result["endpoint"] = endpoint
                result["family"] = family
                rows.append(result)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Comparison vs FULL + verdicts
# ---------------------------------------------------------------------------

def _ci_overlaps(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    """Check if two confidence intervals overlap."""
    if any(np.isnan(v) for v in [lo1, hi1, lo2, hi2]):
        return True  # treat missing as overlapping (conservative)
    return lo1 <= hi2 and lo2 <= hi1


def compare_to_full(
    metrics_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    models: List[str],
    primary_effects: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare each non-FULL subset to FULL for each model."""
    rows = []
    subsets = [s for s in metrics_df["subset"].unique() if s != "FULL"]

    for subset in subsets:
        for model in models:
            full_row = metrics_df[
                (metrics_df["subset"] == "FULL") & (metrics_df["model_name"] == model)
            ]
            sub_row = metrics_df[
                (metrics_df["subset"] == subset) & (metrics_df["model_name"] == model)
            ]
            if full_row.empty or sub_row.empty:
                continue
            full_row = full_row.iloc[0]
            sub_row = sub_row.iloc[0]

            for effect in primary_effects:
                ef = full_row.get(effect, float("nan"))
                es = sub_row.get(effect, float("nan"))

                if np.isnan(ef) or np.isnan(es):
                    rows.append({
                        "subset": subset, "model_name": model, "effect": effect,
                        "effect_full": ef, "effect_subset": es,
                        "abs_ratio": float("nan"), "attenuation": float("nan"),
                        "effect_change_class": "not_evaluable",
                        "ci_overlap": True, "pass_attenuation": True,
                        "pass_ci_overlap": True, "status": "not_evaluable",
                    })
                    continue

                abs_ef = abs(ef)
                abs_es = abs(es)
                if abs_ef > 0:
                    abs_ratio = abs_es / abs_ef
                    attenuation = 1.0 - abs_ratio
                else:
                    abs_ratio = float("nan")
                    attenuation = float("nan")

                # CI overlap check
                ci_full = ci_df[
                    (ci_df["subset"] == "FULL") & (ci_df["model_name"] == model)
                    & (ci_df["metric"] == effect)
                ]
                ci_sub = ci_df[
                    (ci_df["subset"] == subset) & (ci_df["model_name"] == model)
                    & (ci_df["metric"] == effect)
                ]
                if not ci_full.empty and not ci_sub.empty:
                    overlap = _ci_overlaps(
                        ci_full.iloc[0]["ci_lower"], ci_full.iloc[0]["ci_upper"],
                        ci_sub.iloc[0]["ci_lower"], ci_sub.iloc[0]["ci_upper"],
                    )
                else:
                    overlap = True

                # Classify effect change direction
                if np.isnan(attenuation):
                    change_class = "not_evaluable"
                elif attenuation < -0.05:
                    change_class = "amplified"
                elif attenuation > 0.05:
                    change_class = "attenuated"
                else:
                    change_class = "stable"

                rows.append({
                    "subset": subset, "model_name": model, "effect": effect,
                    "effect_full": round(ef, 6), "effect_subset": round(es, 6),
                    "abs_ratio": round(abs_ratio, 4) if not np.isnan(abs_ratio) else float("nan"),
                    "attenuation": round(attenuation, 4) if not np.isnan(attenuation) else float("nan"),
                    "effect_change_class": change_class,
                    "ci_overlap": overlap,
                    "pass_attenuation": (
                        np.isnan(attenuation) or attenuation <= 0.40
                    ),
                    "pass_ci_overlap": overlap,
                    "status": "OK",
                })

    # Add omnibus significance class comparison
    comp_df = pd.DataFrame(rows)
    if comp_df.empty:
        return comp_df

    # Add sig class columns
    sig_rows = []
    for _, row in comp_df.iterrows():
        subset = row["subset"]
        model = row["model_name"]
        for endpoint in ["tier2", "tier_numeric"]:
            full_test = tests_df[
                (tests_df["subset"] == "FULL") & (tests_df["model_name"] == model)
                & (tests_df["endpoint"] == endpoint)
            ]
            sub_test = tests_df[
                (tests_df["subset"] == subset) & (tests_df["model_name"] == model)
                & (tests_df["endpoint"] == endpoint)
            ]
            sig_full = (
                not full_test.empty
                and full_test.iloc[0]["p_value"] < alpha
                and full_test.iloc[0]["converged"]
            )
            sig_sub = (
                not sub_test.empty
                and sub_test.iloc[0]["p_value"] < alpha
                and sub_test.iloc[0]["converged"]
            )
            sig_rows.append({
                "subset": subset, "model_name": model,
                f"sig_class_full_{endpoint}": sig_full,
                f"sig_class_subset_{endpoint}": sig_sub,
                f"sig_class_changed_{endpoint}": sig_full != sig_sub,
            })

    if sig_rows:
        sig_df = pd.DataFrame(sig_rows).drop_duplicates(
            subset=["subset", "model_name"]
        )
        comp_df = comp_df.merge(sig_df, on=["subset", "model_name"], how="left")

    return comp_df


def assign_verdicts(
    metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    thresholds: Dict[str, float],
    models: List[str],
) -> Dict[str, Any]:
    """Assign per-subset × model verdicts. Returns structured flags dict."""
    verdicts: Dict[str, Any] = {}

    for subset in [s for s in metrics_df["subset"].unique() if s != "FULL"]:
        verdicts[subset] = {}
        for model in models:
            sub_m = metrics_df[
                (metrics_df["subset"] == subset) & (metrics_df["model_name"] == model)
            ]
            if sub_m.empty:
                verdicts[subset][model] = {"verdict": "SKIP", "reasons": ["no data"]}
                continue

            sub_m = sub_m.iloc[0]
            reasons = []

            # Hard gates
            pair_ok = sub_m["pair_retention"] >= thresholds["min_pair_retention"]
            comp_ok = sub_m["complaint_retention"] >= thresholds["min_complaint_retention"]
            if not pair_ok:
                reasons.append(
                    f"pair_retention {sub_m['pair_retention']:.4f} < {thresholds['min_pair_retention']}"
                )
            if not comp_ok:
                reasons.append(
                    f"complaint_retention {sub_m['complaint_retention']:.4f} < {thresholds['min_complaint_retention']}"
                )

            if not (pair_ok and comp_ok):
                verdicts[subset][model] = {"verdict": "FAIL", "reasons": reasons}
                continue

            # Effect-stability gates
            comp_rows = comparison_df[
                (comparison_df["subset"] == subset)
                & (comparison_df["model_name"] == model)
            ]
            n_effect_fails = 0
            for _, cr in comp_rows.iterrows():
                change_class = cr.get("effect_change_class", "")
                if not cr.get("pass_attenuation", True):
                    reasons.append(
                        f"{cr['effect']}: attenuation {cr['attenuation']:.4f} > {thresholds['max_attenuation']}"
                    )
                    n_effect_fails += 1
                elif change_class == "amplified":
                    reasons.append(
                        f"{cr['effect']}: effect amplified in subset (attenuation={cr['attenuation']:.4f})"
                    )
                if not cr.get("pass_ci_overlap", True):
                    reasons.append(f"{cr['effect']}: CI does not overlap with FULL")
                    n_effect_fails += 1

            # Omnibus consistency
            sig_changed = False
            for ep in ["tier2", "tier_numeric"]:
                col = f"sig_class_changed_{ep}"
                if col in comp_rows.columns and comp_rows[col].any():
                    reasons.append(f"omnibus {ep}: significance class changed")
                    sig_changed = True

            # Verdict logic
            if n_effect_fails == 0 and not sig_changed:
                verdict = "PASS"
            elif not pair_ok or not comp_ok:
                verdict = "FAIL"
            elif n_effect_fails >= 2 * len(comp_rows["effect"].unique()):
                verdict = "FAIL"
            else:
                verdict = "PASS_WITH_CAUTION"

            verdicts[subset][model] = {"verdict": verdict, "reasons": reasons}

    return verdicts


# ---------------------------------------------------------------------------
# 8b. Retention-balance analysis
# ---------------------------------------------------------------------------

def retention_balance_analysis(
    df: pd.DataFrame,
    models: List[str],
    severity_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse whether HF filtering disproportionately drops certain personas or severities.

    Parameters
    ----------
    df : merged DataFrame with ``high_fidelity_pair_pass`` and ``persona_label``.
    models : list of model names.
    severity_map : optional ``{complaint_id: {"monetary_severity": ..., "issue_severity": ...}}``.

    Returns
    -------
    persona_balance : DataFrame with HF inclusion rate per persona × model + chi-squared.
    severity_balance : DataFrame with HF inclusion rate per severity band × model (or empty).
    """
    persona_rows = []
    for model in models:
        m = df[df["model_name"] == model]
        by_persona = m.groupby("persona_label")["high_fidelity_pair_pass"].agg(["sum", "count", "mean"])
        by_persona.columns = ["hf_pairs", "total_pairs", "hf_rate"]
        by_persona = by_persona.reset_index()
        by_persona["model_name"] = model

        # Chi-squared test: is HF rate independent of persona?
        try:
            from scipy.stats import chi2_contingency
            ct = pd.crosstab(m["persona_label"], m["high_fidelity_pair_pass"])
            if ct.shape[1] == 2:
                chi2, p, _, _ = chi2_contingency(ct)
                by_persona["chi2_persona_hf"] = round(chi2, 4)
                by_persona["chi2_persona_hf_p"] = round(p, 6)
            else:
                by_persona["chi2_persona_hf"] = float("nan")
                by_persona["chi2_persona_hf_p"] = float("nan")
        except Exception:
            by_persona["chi2_persona_hf"] = float("nan")
            by_persona["chi2_persona_hf_p"] = float("nan")

        # Max/min rate spread
        spread = by_persona["hf_rate"].max() - by_persona["hf_rate"].min()
        by_persona["persona_hf_rate_spread"] = round(spread, 4)

        persona_rows.append(by_persona)

    persona_balance = pd.concat(persona_rows, ignore_index=True) if persona_rows else pd.DataFrame()

    # Severity balance (if severity data provided)
    severity_rows = []
    if severity_map:
        df = df.copy()
        df["monetary_severity"] = df["complaint_id"].map(
            lambda cid: severity_map.get(str(cid), {}).get("monetary_severity", "unknown")
        )
        df["issue_severity"] = df["complaint_id"].map(
            lambda cid: severity_map.get(str(cid), {}).get("issue_severity", "medium")
        )
        for model in models:
            m = df[df["model_name"] == model]
            for sev_col in ["monetary_severity", "issue_severity"]:
                by_sev = m.groupby(sev_col)["high_fidelity_pair_pass"].agg(["sum", "count", "mean"])
                by_sev.columns = ["hf_pairs", "total_pairs", "hf_rate"]
                by_sev = by_sev.reset_index().rename(columns={sev_col: "severity_band"})
                by_sev["severity_type"] = sev_col
                by_sev["model_name"] = model
                severity_rows.append(by_sev)

    severity_balance = pd.concat(severity_rows, ignore_index=True) if severity_rows else pd.DataFrame()

    return persona_balance, severity_balance


# ---------------------------------------------------------------------------
# 8c. Outcome-composition shift (dropped vs kept)
# ---------------------------------------------------------------------------

def outcome_composition_shift(
    df: pd.DataFrame,
    models: List[str],
) -> pd.DataFrame:
    """Compare outcomes between HF=1 (kept) and HF=0 (dropped) pairs.

    For each model, compute mean_tier and p_tier2 for kept vs dropped,
    run a t-test on tier and a proportion z-test on tier2, and state
    the selection bias direction.
    """
    from scipy.stats import ttest_ind

    rows = []
    for model in models:
        m = df[df["model_name"] == model]
        kept = m[m["high_fidelity_pair_pass"] == 1]
        dropped = m[m["high_fidelity_pair_pass"] == 0]

        n_kept, n_dropped = len(kept), len(dropped)
        if n_dropped == 0:
            rows.append({
                "model_name": model,
                "n_kept": n_kept, "n_dropped": n_dropped,
                "mean_tier_kept": float(kept["tier"].mean()) if n_kept else float("nan"),
                "mean_tier_dropped": float("nan"),
                "p_tier2_kept": float(kept["tier2"].mean()) if n_kept else float("nan"),
                "p_tier2_dropped": float("nan"),
                "tier_diff": float("nan"),
                "tier_ttest_p": float("nan"),
                "tier2_diff": float("nan"),
                "tier2_ztest_p": float("nan"),
                "bias_direction": "no_dropped_pairs",
            })
            continue

        mt_kept = float(kept["tier"].mean())
        mt_dropped = float(dropped["tier"].mean())
        pt2_kept = float(kept["tier2"].mean())
        pt2_dropped = float(dropped["tier2"].mean())

        # t-test on tier
        try:
            _, p_tier = ttest_ind(kept["tier"].values, dropped["tier"].values, equal_var=False)
        except Exception:
            p_tier = float("nan")

        # Proportion z-test on tier2
        try:
            from statsmodels.stats.proportion import proportions_ztest
            count = np.array([kept["tier2"].sum(), dropped["tier2"].sum()])
            nobs = np.array([n_kept, n_dropped])
            _, p_tier2 = proportions_ztest(count, nobs)
        except Exception:
            p_tier2 = float("nan")

        tier_diff = mt_kept - mt_dropped
        tier2_diff = pt2_kept - pt2_dropped

        # Classify bias direction
        if abs(tier_diff) < 0.01 and abs(tier2_diff) < 0.005:
            bias_dir = "negligible"
        elif tier_diff > 0:
            bias_dir = "conservative (dropped pairs had lower tiers)"
        else:
            bias_dir = "anti-conservative (dropped pairs had higher tiers)"

        rows.append({
            "model_name": model,
            "n_kept": n_kept, "n_dropped": n_dropped,
            "mean_tier_kept": round(mt_kept, 4),
            "mean_tier_dropped": round(mt_dropped, 4),
            "p_tier2_kept": round(pt2_kept, 4),
            "p_tier2_dropped": round(pt2_dropped, 4),
            "tier_diff": round(tier_diff, 4),
            "tier_ttest_p": round(float(p_tier), 6) if not np.isnan(p_tier) else float("nan"),
            "tier2_diff": round(tier2_diff, 4),
            "tier2_ztest_p": round(float(p_tier2), 6) if not np.isnan(p_tier2) else float("nan"),
            "bias_direction": bias_dir,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Output writers
# ---------------------------------------------------------------------------

def write_subset_manifest(metrics_df: pd.DataFrame, path: Path) -> None:
    cols = ["subset", "model_name", "n_pairs", "n_complaints",
            "pair_retention", "complaint_retention"]
    metrics_df[[c for c in cols if c in metrics_df.columns]].to_csv(path, index=False)


def write_endpoint_summary(
    metrics_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    path: Path,
) -> None:
    """Merge point estimates with CIs and write."""
    # Pivot CIs to wide format
    if ci_df.empty:
        metrics_df.to_csv(path, index=False)
        return

    ci_wide = ci_df.pivot_table(
        index=["subset", "model_name"],
        columns="metric",
        values=["ci_lower", "ci_upper"],
    )
    ci_wide.columns = [f"{val}_{met}" for val, met in ci_wide.columns]
    ci_wide = ci_wide.reset_index()

    merged = metrics_df.merge(ci_wide, on=["subset", "model_name"], how="left")
    merged.to_csv(path, index=False)


def write_robustness_flags(
    verdicts: Dict[str, Any],
    thresholds: Dict[str, float],
    run_id: str,
    path: Path,
    aliases: Optional[Dict[str, str]] = None,
) -> None:
    """Write hierarchical JSON flags."""
    # Overall verdict: PASS if all subsets pass for all models
    all_verdicts = []
    for subset_d in verdicts.values():
        for model_d in subset_d.values():
            all_verdicts.append(model_d["verdict"])

    if all(v == "PASS" for v in all_verdicts):
        overall = "PASS"
    elif any(v == "FAIL" for v in all_verdicts):
        overall = "FAIL"
    else:
        overall = "PASS_WITH_CAUTION"

    flags = {
        "run_id": run_id,
        "thresholds": thresholds,
        "overall_verdict": overall,
        "per_subset": verdicts,
    }
    if aliases:
        flags["subset_aliases"] = aliases
    path.write_text(json.dumps(flags, indent=2, default=str), encoding="utf-8")


def write_robustness_report(
    metrics_df: pd.DataFrame,
    ci_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    verdicts: Dict[str, Any],
    thresholds: Dict[str, float],
    run_id: str,
    path: Path,
    aliases: Optional[Dict[str, str]] = None,
    persona_balance: Optional[pd.DataFrame] = None,
    composition_shift: Optional[pd.DataFrame] = None,
) -> None:
    """Write human-readable Markdown report."""
    lines = [
        "# Matched-Guise Robustness Subset Report\n",
        f"Run ID: `{run_id}`\n",
    ]

    # Overall verdict
    all_verdicts = []
    for subset_d in verdicts.values():
        for model_d in subset_d.values():
            all_verdicts.append(model_d["verdict"])

    if all(v == "PASS" for v in all_verdicts):
        overall = "PASS"
    elif any(v == "FAIL" for v in all_verdicts):
        overall = "FAIL"
    else:
        overall = "PASS_WITH_CAUTION"

    lines.append(f"## Overall Verdict: **{overall}**\n")

    # Alias notes
    aliases = aliases or {}
    if aliases:
        lines.append("### Subset Aliases\n")
        for alias, target in aliases.items():
            lines.append(f"- **{alias}** is identical to **{target}** (HF already excludes polarity contradictions)")
        lines.append("")

    # Thresholds
    lines += [
        "## Thresholds\n",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in thresholds.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Subset manifest
    lines += [
        "## Subset Sizes and Retention\n",
        "| Subset | Model | Pairs | Complaints | Pair Ret. | Complaint Ret. |",
        "|--------|-------|-------|------------|-----------|----------------|",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {r['subset']} | {r['model_name']} | {r['n_pairs']} | {r['n_complaints']} "
            f"| {r['pair_retention']:.4f} | {r['complaint_retention']:.4f} |"
        )
    lines.append("")

    # Endpoint summary
    lines += [
        "## Endpoint Summary\n",
        "| Subset | Model | mean_tier | p_tier2 | max_gap_tier2 | max_gap_mean_tier |",
        "|--------|-------|-----------|---------|---------------|-------------------|",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {r['subset']} | {r['model_name']} "
            f"| {r['mean_tier']:.4f} | {r['p_tier2']:.4f} "
            f"| {r.get('max_persona_gap_p_tier2', float('nan')):.4f} "
            f"| {r.get('max_persona_gap_mean_tier', float('nan')):.4f} |"
        )
    lines.append("")

    # Omnibus tests
    if not tests_df.empty:
        lines += [
            "## Omnibus Tests (GEE Wald p-values)\n",
            "| Subset | Model | Endpoint | p-value | Status |",
            "|--------|-------|----------|---------|--------|",
        ]
        for _, r in tests_df.iterrows():
            p_str = f"{r['p_value']:.4f}" if not np.isnan(r["p_value"]) else "N/A"
            lines.append(
                f"| {r['subset']} | {r['model_name']} | {r['endpoint']} | {p_str} | {r['status']} |"
            )
        lines.append("")

    # Robustness comparison
    if not comparison_df.empty:
        lines += [
            "## Effect Stability vs FULL\n",
            "| Subset | Model | Effect | Full | Subset | Attenuation | Direction | CI Overlap | Pass |",
            "|--------|-------|--------|------|--------|-------------|-----------|------------|------|",
        ]
        for _, r in comparison_df.iterrows():
            att_val = r.get("attenuation", float("nan"))
            att_str = f"{att_val:.4f}" if not np.isnan(att_val) else "N/A"
            change_class = r.get("effect_change_class", "")
            overlap_str = "Yes" if r.get("ci_overlap", True) else "No"
            pass_str = "Y" if r.get("pass_attenuation", True) and r.get("pass_ci_overlap", True) else "N"
            lines.append(
                f"| {r['subset']} | {r['model_name']} | {r['effect']} "
                f"| {r['effect_full']:.4f} | {r['effect_subset']:.4f} "
                f"| {att_str} | {change_class} | {overlap_str} | {pass_str} |"
            )
        lines.append("")

    # Per-subset verdicts
    lines.append("## Verdicts\n")
    for subset, model_verdicts in verdicts.items():
        lines.append(f"### {subset}\n")
        for model, vd in model_verdicts.items():
            v = vd["verdict"]
            lines.append(f"**{model}**: **{v}**")
            if vd["reasons"]:
                for reason in vd["reasons"]:
                    lines.append(f"  - {reason}")
            else:
                lines.append("  - All gates passed")
            lines.append("")

    # Retention balance
    if persona_balance is not None and not persona_balance.empty:
        lines += [
            "## Retention Balance by Persona\n",
            "| Model | Persona | HF Pairs | Total | HF Rate | Rate Spread | Chi2 p |",
            "|-------|---------|----------|-------|---------|-------------|--------|",
        ]
        for _, r in persona_balance.iterrows():
            chi2_p = r.get("chi2_persona_hf_p", float("nan"))
            p_str = f"{chi2_p:.4f}" if not np.isnan(chi2_p) else "N/A"
            lines.append(
                f"| {r['model_name']} | {r['persona_label']} "
                f"| {int(r['hf_pairs'])} | {int(r['total_pairs'])} "
                f"| {r['hf_rate']:.4f} | {r['persona_hf_rate_spread']:.4f} | {p_str} |"
            )
        lines.append("")

    # Outcome composition shift
    if composition_shift is not None and not composition_shift.empty:
        lines += [
            "## Outcome-Composition Shift (Dropped vs Kept)\n",
            "| Model | N Kept | N Dropped | Mean Tier (Kept) | Mean Tier (Dropped) | Diff | t-test p | Tier2 (Kept) | Tier2 (Dropped) | Diff | z-test p | Bias Direction |",
            "|-------|--------|-----------|------------------|---------------------|------|----------|-------------|-----------------|------|----------|----------------|",
        ]
        for _, r in composition_shift.iterrows():
            tp = r.get("tier_ttest_p", float("nan"))
            tp_str = f"{tp:.4f}" if not np.isnan(tp) else "N/A"
            zp = r.get("tier2_ztest_p", float("nan"))
            zp_str = f"{zp:.4f}" if not np.isnan(zp) else "N/A"
            mt_d = r.get("mean_tier_dropped", float("nan"))
            mt_d_str = f"{mt_d:.4f}" if not np.isnan(mt_d) else "N/A"
            pt2_d = r.get("p_tier2_dropped", float("nan"))
            pt2_d_str = f"{pt2_d:.4f}" if not np.isnan(pt2_d) else "N/A"
            lines.append(
                f"| {r['model_name']} | {r['n_kept']} | {r['n_dropped']} "
                f"| {r['mean_tier_kept']:.4f} | {mt_d_str} "
                f"| {r['tier_diff']:.4f} | {tp_str} "
                f"| {r['p_tier2_kept']:.4f} | {pt2_d_str} "
                f"| {r['tier2_diff']:.4f} | {zp_str} "
                f"| {r['bias_direction']} |"
            )
        lines.append("")

    # Interpretation block
    lines += [
        "## Recommended Manuscript Interpretation\n",
    ]
    if overall == "PASS":
        lines.append(
            "> Robustness checks confirm that persona-dispersion effects are stable "
            "across high-fidelity subsets. Filtering for slot-preservation quality "
            "does not attenuate the observed disparity patterns, supporting the "
            "validity of the full-sample findings."
        )
    elif overall == "PASS_WITH_CAUTION":
        lines.append(
            "> Robustness checks indicate that persona-dispersion effects are "
            "partially attenuated in some filtered subsets. The core findings hold, "
            "but readers should note that slot-preservation quality may modulate "
            "the magnitude of observed persona effects."
        )
    else:
        lines.append(
            "> Robustness checks flag potential concerns: one or more filtered "
            "subsets show substantial attenuation or changed significance patterns. "
            "Caution is warranted when interpreting persona-dispersion findings, "
            "as slot-preservation artifacts may inflate apparent effects."
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_fingerprint(
    decisions_path: Path,
    audit_path: Path,
    output_dir: Path,
    run_id: str,
    cli_args: Dict[str, Any],
) -> None:
    """Write analysis provenance fingerprint."""
    def _file_hash(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    fp = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "matched_guise_robustness_subsets.py",
        "script_version": "0.1.0",
        "inputs": {
            "decisions_csv": {
                "path": str(decisions_path),
                "sha256": _file_hash(decisions_path),
            },
            "pair_audit_csv": {
                "path": str(audit_path),
                "sha256": _file_hash(audit_path),
            },
        },
        "cli_args": cli_args,
    }
    (output_dir / "analysis_fingerprint.json").write_text(
        json.dumps(fp, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 10. Programmatic entry point (for experiment integration)
# ---------------------------------------------------------------------------

def run_robustness_analysis(
    decisions_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    severity_map: Optional[Dict[str, Dict[str, str]]] = None,
    bootstrap_reps: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
    include_hf_no_pol: bool = True,
) -> Dict[str, Any]:
    """Run robustness analysis programmatically from DataFrames.

    Returns a results dict suitable for ``analysis_results["robustness_subsets"]``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {**DEFAULT_THRESHOLDS, "alpha": alpha}
    primary_effects = ["max_persona_gap_p_tier2", "max_persona_gap_mean_tier"]

    # Load and map (from already-loaded DataFrames)
    dec, aud = decisions_df.copy(), audit_df.copy()

    # Apply same column mapping as load_and_map
    dec_rename = {}
    if "case_id" in dec.columns and "complaint_id" not in dec.columns:
        dec_rename["case_id"] = "complaint_id"
    if "model" in dec.columns and "model_name" not in dec.columns:
        dec_rename["model"] = "model_name"
    if "variant" in dec.columns and "persona_label" not in dec.columns:
        dec_rename["variant"] = "persona_label"
    if dec_rename:
        dec = dec.rename(columns=dec_rename)
    dec["tier"] = dec["tier"].astype(int)
    dec["tier2"] = (dec["tier"] == 2).astype(int)
    if "confidence" in dec.columns:
        dec["need_more_info"] = (dec["confidence"] == "need_more_info").astype(int)
    else:
        dec["need_more_info"] = 0

    aud_rename = {}
    if "high_fidelity_pass" in aud.columns:
        aud_rename["high_fidelity_pass"] = "high_fidelity_pair_pass"
    if "has_contradiction" in aud.columns:
        aud_rename["has_contradiction"] = "polarity_contradiction_flag"
    if aud_rename:
        aud = aud.rename(columns=aud_rename)
    for col in ["high_fidelity_pair_pass", "polarity_contradiction_flag", "strict_pair_pass"]:
        if col in aud.columns:
            aud[col] = aud[col].map(
                {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
            ).fillna(0).astype(int)
    if "numeric_status" in aud.columns:
        aud["numeric_mutation_flag"] = (aud["numeric_status"] == "MUTATION").astype(int)
    else:
        aud["numeric_mutation_flag"] = 0

    errors = validate_inputs(dec, aud)
    if errors:
        logger.error("Validation errors: %s", errors)
        return {"skipped": True, "errors": errors}

    df = merge_decisions_with_audit(dec, aud, output_dir)
    models = sorted(df["model_name"].unique().tolist())

    masks, aliases = build_subsets(df, include_hf_no_pol=include_hf_no_pol)
    metrics_df = compute_subset_metrics(df, masks, models)
    ci_df = bootstrap_subset_metrics(df, masks, models, n_boot=bootstrap_reps,
                                     alpha=alpha, seed=seed)
    tests_df = run_omnibus_tests(df, masks, models)
    comparison_df = compare_to_full(metrics_df, ci_df, tests_df, models,
                                    primary_effects, alpha)
    verdicts = assign_verdicts(metrics_df, comparison_df, thresholds, models)

    persona_balance, severity_balance = retention_balance_analysis(df, models, severity_map)
    comp_shift = outcome_composition_shift(df, models)

    # Write outputs
    write_subset_manifest(metrics_df, output_dir / "subset_manifest.csv")
    write_endpoint_summary(metrics_df, ci_df, output_dir / "endpoint_summary.csv")
    tests_df.to_csv(output_dir / "omnibus_tests.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(output_dir / "robustness_vs_full.csv", index=False)
    if not persona_balance.empty:
        persona_balance.to_csv(output_dir / "retention_balance_persona.csv", index=False)
    if not severity_balance.empty:
        severity_balance.to_csv(output_dir / "retention_balance_severity.csv", index=False)
    if not comp_shift.empty:
        comp_shift.to_csv(output_dir / "outcome_composition_shift.csv", index=False)
    write_robustness_flags(verdicts, thresholds, run_id,
                           output_dir / "robustness_flags.json", aliases=aliases)
    write_robustness_report(
        metrics_df, ci_df, comparison_df, tests_df,
        verdicts, thresholds, run_id,
        output_dir / "robustness_report.md",
        aliases=aliases,
        persona_balance=persona_balance,
        composition_shift=comp_shift,
    )

    return {
        "output_dir": str(output_dir),
        "verdicts": verdicts,
        "aliases": aliases,
        "overall_verdict": "PASS" if all(
            v["verdict"] == "PASS"
            for sv in verdicts.values() for v in sv.values()
        ) else "FAIL" if any(
            v["verdict"] == "FAIL"
            for sv in verdicts.values() for v in sv.values()
        ) else "PASS_WITH_CAUTION",
    }


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Matched-guise robustness subset analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--decisions-csv", required=True,
                    help="Path to results_persona_decisions_*.csv")
    p.add_argument("--pair-audit-csv", required=True,
                    help="Path to pair_slot_audit.csv from slot audit")
    p.add_argument("--output-dir", default="./robustness_out",
                    help="Output directory (default: ./robustness_out)")
    p.add_argument("--run-id", default=None,
                    help="Run identifier (default: auto from decisions filename)")
    p.add_argument("--min-complaint-retention", type=float, default=0.25,
                    help="Min complaint retention vs FULL (default: 0.25)")
    p.add_argument("--min-pair-retention", type=float, default=0.40,
                    help="Min pair retention vs FULL (default: 0.40)")
    p.add_argument("--max-attenuation", type=float, default=0.40,
                    help="Max effect attenuation vs FULL (default: 0.40)")
    p.add_argument("--alpha", type=float, default=0.05,
                    help="Significance level (default: 0.05)")
    p.add_argument("--bootstrap-reps", type=int, default=1000,
                    help="Bootstrap replications (default: 1000)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    p.add_argument("--include-hf-no-pol", type=int, default=1,
                    help="Include HF_NO_POL subset (default: 1)")
    p.add_argument("--min-complaints-for-tests", type=int, default=30,
                    help="Min complaints to run GEE tests (default: 30)")
    p.add_argument("--pipeline-parquet", default=None,
                    help="Path to data pipeline parquet (for severity balance analysis)")
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Enable debug logging")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    decisions_path = Path(args.decisions_csv)
    audit_path = Path(args.pair_audit_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run ID
    run_id = args.run_id
    if run_id is None:
        import re
        m = re.search(r"(\d{8}_\d{6})", decisions_path.name)
        run_id = m.group(1) if m else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    thresholds = {
        "min_pair_retention": args.min_pair_retention,
        "min_complaint_retention": args.min_complaint_retention,
        "max_attenuation": args.max_attenuation,
        "alpha": args.alpha,
        "min_complaints_for_tests": args.min_complaints_for_tests,
        "min_persona_levels_for_tests": 2,
    }
    primary_effects = [
        "max_persona_gap_p_tier2",
        "max_persona_gap_mean_tier",
    ]

    t0 = time.time()

    # 1-2. Load and validate
    logger.info("Loading inputs...")
    dec, aud = load_and_map(decisions_path, audit_path)
    errors = validate_inputs(dec, aud)
    if errors:
        for e in errors:
            logger.error("Validation: %s", e)
        raise SystemExit(1)

    # 3. Join
    logger.info("Merging decisions with audit flags...")
    df = merge_decisions_with_audit(dec, aud, output_dir)
    models = sorted(df["model_name"].unique().tolist())
    logger.info("Models: %s", models)

    # 4. Build subsets
    masks, aliases = build_subsets(df, include_hf_no_pol=bool(args.include_hf_no_pol))
    for name, mask in masks.items():
        alias_note = f" (= {aliases[name]})" if name in aliases else ""
        logger.info("  %s: %d pairs%s", name, mask.sum(), alias_note)

    # 5. Compute metrics
    logger.info("Computing subset metrics...")
    metrics_df = compute_subset_metrics(df, masks, models)

    # 6. Bootstrap CIs
    logger.info("Running bootstrap CIs (%d reps)...", args.bootstrap_reps)
    ci_df = bootstrap_subset_metrics(
        df, masks, models,
        n_boot=args.bootstrap_reps,
        alpha=args.alpha,
        seed=args.seed,
    )

    # 7. Omnibus tests
    logger.info("Running GEE omnibus tests...")
    tests_df = run_omnibus_tests(
        df, masks, models,
        min_complaints=args.min_complaints_for_tests,
    )

    # 8. Compare to FULL + verdicts
    logger.info("Comparing subsets to FULL...")
    comparison_df = compare_to_full(
        metrics_df, ci_df, tests_df, models, primary_effects, args.alpha,
    )
    verdicts = assign_verdicts(metrics_df, comparison_df, thresholds, models)

    # 8b. Retention balance
    logger.info("Running retention-balance analysis...")
    severity_map: Optional[Dict[str, Dict[str, str]]] = None
    if args.pipeline_parquet:
        try:
            pq = pd.read_parquet(args.pipeline_parquet)
            from src.data_loader import compute_issue_severity
            severity_map = {}
            for _, row in pq.iterrows():
                cid = str(row.get("case_id", row.get("complaint_id", "")))
                # monetary_severity requires dollar amounts (not in parquet); use issue_severity
                issue_sev = compute_issue_severity(
                    str(row.get("product", "")), str(row.get("issue", ""))
                )
                severity_map[cid] = {
                    "monetary_severity": "unknown",
                    "issue_severity": issue_sev,
                }
            logger.info("Loaded severity data for %d complaints from pipeline parquet", len(severity_map))
        except Exception as e:
            logger.warning("Could not load severity from pipeline parquet: %s", e)
            severity_map = None

    persona_balance, severity_balance = retention_balance_analysis(df, models, severity_map)

    # 8c. Outcome-composition shift
    logger.info("Running outcome-composition shift analysis...")
    composition_shift = outcome_composition_shift(df, models)

    # 9. Write outputs
    logger.info("Writing outputs to %s", output_dir)
    write_subset_manifest(metrics_df, output_dir / "subset_manifest.csv")
    write_endpoint_summary(metrics_df, ci_df, output_dir / "endpoint_summary.csv")
    tests_df.to_csv(output_dir / "omnibus_tests.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(output_dir / "robustness_vs_full.csv", index=False)

    # New analysis outputs
    if not persona_balance.empty:
        persona_balance.to_csv(output_dir / "retention_balance_persona.csv", index=False)
    if not severity_balance.empty:
        severity_balance.to_csv(output_dir / "retention_balance_severity.csv", index=False)
    if not composition_shift.empty:
        composition_shift.to_csv(output_dir / "outcome_composition_shift.csv", index=False)
    write_robustness_flags(verdicts, thresholds, run_id, output_dir / "robustness_flags.json",
                           aliases=aliases)
    write_robustness_report(
        metrics_df, ci_df, comparison_df, tests_df,
        verdicts, thresholds, run_id,
        output_dir / "robustness_report.md",
        aliases=aliases,
        persona_balance=persona_balance,
        composition_shift=composition_shift,
    )
    write_fingerprint(decisions_path, audit_path, output_dir, run_id,
                      {**vars(args), "subset_aliases": aliases})

    elapsed = time.time() - t0

    # Console summary
    print(f"\n{'=' * 60}")
    print(f"Robustness Subset Analysis — Run {run_id}")
    print(f"{'=' * 60}")
    for subset, model_verdicts in verdicts.items():
        alias_note = f" (= {aliases[subset]})" if subset in aliases else ""
        for model, vd in model_verdicts.items():
            v = vd["verdict"]
            print(f"  {subset:12s}{alias_note} | {model:15s} | {v}")
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
