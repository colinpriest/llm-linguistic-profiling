"""Extract LLM remedy decisions with original complaint narratives to Excel.

Reads cached results from disk (no API calls). Joins the baseline remedy
decisions CSV with the pipeline parquet to pair each decision with its
complaint narrative, then writes output/remedy-reasons.xlsx.

Also produces output/remedy-changed.xlsx: 100 examples where the
recommended tier changed after persona-style rewriting, with original
and rewritten text, reasoning, and confidence side by side.

Also produces output/instability-analysis-{timestamp}.xlsx: comprehensive
tier instability analysis across persona dimensions, with Tier-2 endpoint
metrics, transition matrices, clustered bootstrap contrasts, severity
heterogeneity, neutralisation comparison, and drift-filter sensitivity.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import compute_issue_severity
from src.prompts import format_persona_rewrite_user, PROMPT_VERSIONS
from src.schemas import parse_persona_label

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = ["socioeconomic", "geographic", "age", "education"]
DIM_LABELS = {
    "socioeconomic": "Income",
    "geographic": "Location",
    "age": "Age",
    "education": "Education",
}
DIM_CONTRASTS = {
    "socioeconomic": [("high", "low"), ("high", "middle"), ("middle", "low")],
    "geographic": [("urban", "rural"), ("urban", "suburban"), ("suburban", "rural")],
    "age": [("old", "young")],
    "education": [("college", "high_school")],
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _latest_file(pattern: str) -> Path:
    """Return the most-recently-modified file matching *pattern*."""
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    if not matches:
        print(f"ERROR: no files matching {pattern}", file=sys.stderr)
        sys.exit(1)
    return Path(matches[-1])


def _largest_parquet() -> Path:
    """Return the largest step10 parquet (highest complaint count)."""
    matches = glob.glob("data/.pipeline/step10_seed42_*.parquet")
    if not matches:
        print("ERROR: no step10 parquet files in data/.pipeline/", file=sys.stderr)
        sys.exit(1)
    return Path(max(matches, key=lambda p: Path(p).stat().st_size))


def _cache_key(model: str, task: str, narrative: str) -> str:
    """Reproduce the Blake2b cache key used by ResponseCache."""
    prompt_version = PROMPT_VERSIONS.get(task, "1.0")
    content = f"{model}|{task}|{prompt_version}|{narrative}"
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()


def _read_rewrite_cache(model: str, original_narrative: str, variant: str,
                        cache_dir: Path) -> str | None:
    """Look up the cached persona rewrite and return the rewritten text."""
    parsed = parse_persona_label(variant)
    if parsed is None:
        return None
    user_prompt = format_persona_rewrite_user(
        original_narrative,
        parsed["age"], parsed["geographic"], parsed["education"], parsed["socioeconomic"],
    )
    key = _cache_key(model, "persona_rewrite", user_prompt)
    path = cache_dir / "persona_rewrite" / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["response"]["rewritten_complaint"]
    except (json.JSONDecodeError, KeyError):
        return None


def _word_jaccard(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    wa = set(str(text_a).lower().split())
    wb = set(str(text_b).lower().split())
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def _round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Round all float columns to *decimals* places."""
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].round(decimals)
    return df


# ===================================================================
# Output 1: remedy-reasons.xlsx
# ===================================================================


def remedy_reasons(complaints: pd.DataFrame) -> None:
    """Output 1: all baseline decisions with narratives."""
    outcome_path = _latest_file("output/results_outcome_*.csv")
    print(f"Outcomes : {outcome_path}")

    outcomes = pd.read_csv(outcome_path)
    merged = outcomes.merge(complaints, on="case_id", how="left")

    cols = [
        "case_id", "complaint_id", "model", "product", "issue",
        "tier", "confidence", "reasoning", "information_needed", "narrative",
    ]
    merged = merged[[c for c in cols if c in merged.columns]]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"output/remedy-reasons-{ts}.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out, index=False, engine="openpyxl")
    print(f"Wrote {len(merged):,} rows to {out}")


# ===================================================================
# Output 2: remedy-changed.xlsx
# ===================================================================


def remedy_changed(complaints: pd.DataFrame) -> None:
    """Output 2: 100 examples where tier changed after persona rewriting."""
    outcome_path = _latest_file("output/results_outcome_*.csv")
    persona_path = _latest_file("output/results_persona_decisions_*.csv")
    cache_dir = Path("cache")
    print(f"Persona decisions: {persona_path}")

    baseline = pd.read_csv(outcome_path)
    persona = pd.read_csv(persona_path)

    # Join baseline tier onto persona decisions by (case_id, model)
    merged = persona.merge(
        baseline[["case_id", "model", "tier", "confidence", "reasoning"]],
        on=["case_id", "model"],
        how="inner",
        suffixes=("_rewritten", "_original"),
    )

    # Keep only rows where the tier actually changed
    changed = merged[merged["tier_original"] != merged["tier_rewritten"]]
    print(f"Found {len(changed):,} tier-changed pairs across all variants")

    if changed.empty:
        print("No tier changes found; skipping remedy-changed.xlsx")
        return

    # Sample 100 (diverse: pick distinct case_ids where possible)
    if len(changed) > 100:
        # Try to get variety across case_ids
        sampled = changed.groupby("case_id", group_keys=False).apply(
            lambda g: g.sample(1, random_state=42), include_groups=False,
        )
        # Restore columns dropped by include_groups=False
        sampled = changed.loc[sampled.index]
        if len(sampled) > 100:
            sampled = sampled.sample(100, random_state=42)
        elif len(sampled) < 100:
            remaining = changed[~changed.index.isin(sampled.index)]
            extra = remaining.sample(min(100 - len(sampled), len(remaining)), random_state=42)
            sampled = pd.concat([sampled, extra])
    else:
        sampled = changed

    # Add original narrative
    sampled = sampled.merge(
        complaints[["case_id", "narrative"]], on="case_id", how="left",
    )

    # Fetch rewritten text from cache
    rewritten_texts = []
    for _, row in sampled.iterrows():
        text = _read_rewrite_cache(
            row["model"], row["narrative"], row["variant"], cache_dir,
        )
        rewritten_texts.append(text or "")
    sampled = sampled.copy()
    sampled["rewritten_narrative"] = rewritten_texts

    found = sum(1 for t in rewritten_texts if t)
    print(f"Retrieved {found}/{len(sampled)} rewritten narratives from cache")

    # Build final output
    out_df = sampled[[
        "case_id", "model", "variant",
        "narrative", "rewritten_narrative",
        "tier_original", "tier_rewritten",
        "reasoning_original", "reasoning_rewritten",
        "confidence_original", "confidence_rewritten",
    ]].rename(columns={
        "narrative": "original_text",
        "rewritten_narrative": "rewritten_text",
        "tier_original": "original_tier",
        "tier_rewritten": "rewritten_tier",
        "reasoning_original": "original_reason",
        "reasoning_rewritten": "rewritten_reason",
        "confidence_original": "original_confidence",
        "confidence_rewritten": "rewritten_confidence",
    })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"output/remedy-changed-{ts}.xlsx")
    out_df.to_excel(out, index=False, engine="openpyxl")
    print(f"Wrote {len(out_df):,} rows to {out}")


# ===================================================================
# Output 3: instability-analysis-{timestamp}.xlsx
# ===================================================================

# --- 3a. Summary sheet (with A2 overall headline) --------------------


def _summary_row(grp: pd.DataFrame, dimension: str, variant: str) -> dict:
    """Compute a single summary row from a group of paired observations."""
    n = len(grp)
    n_c = grp["case_id"].nunique()
    orig_d = grp["tier_orig"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    rw_d = grp["tier_rw"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    return {
        "Dimension": dimension,
        "Variant": variant,
        "N (pairs)": n,
        "Complaints": n_c,
        "Baseline T0%": orig_d[0] * 100,
        "Baseline T1%": orig_d[1] * 100,
        "Baseline T2%": orig_d[2] * 100,
        "Rewritten T0%": rw_d[0] * 100,
        "Rewritten T1%": rw_d[1] * 100,
        "Rewritten T2%": rw_d[2] * 100,
        "Mean Tier Change": grp["tier_change"].mean(),
        "Upgrade%": grp["upgraded"].mean() * 100,
        "Unchanged%": grp["unchanged"].mean() * 100,
        "Downgrade%": grp["downgraded"].mean() * 100,
        "Tier-2 Gain%": grp["t2_gain"].mean() * 100,
        "Tier-2 Loss%": grp["t2_loss"].mean() * 100,
        "Net Tier-2 Delta%": (grp["t2_gain"].mean() - grp["t2_loss"].mean()) * 100,
    }


def _summary_sheet(mdf: pd.DataFrame) -> pd.DataFrame:
    """Enhanced summary: overall headline + per-dimension breakdown."""
    # A2: Overall headline row (all 36 variants, no double-counting)
    rows = [_summary_row(mdf, "OVERALL", "(all variants)")]

    for dim in DIMENSIONS:
        if dim not in mdf.columns:
            continue
        for val in sorted(mdf[dim].unique()):
            rows.append(_summary_row(mdf[mdf[dim] == val], DIM_LABELS.get(dim, dim), val))
    return _round_df(pd.DataFrame(rows))


# --- 3b. Transition matrix sheet --------------------------------------


def _transition_sheet(mdf: pd.DataFrame) -> pd.DataFrame:
    """Full 3x3 transition matrix (row-normalised %) per dimension × variant."""
    rows = []
    tiers = [0, 1, 2]
    for dim in DIMENSIONS:
        if dim not in mdf.columns:
            continue
        for val in sorted(mdf[dim].unique()):
            grp = mdf[mdf[dim] == val]
            n = len(grp)
            row: dict = {"Dimension": DIM_LABELS.get(dim, dim), "Variant": val, "N": n}
            for fr in tiers:
                for to in tiers:
                    cnt = ((grp["tier_orig"] == fr) & (grp["tier_rw"] == to)).sum()
                    row[f"{fr}->{to}"] = cnt / n * 100
            rows.append(row)
    return _round_df(pd.DataFrame(rows))


# --- 3b2. Complaint-level instability (A1) ----------------------------


def _complaint_sheet(mdf: pd.DataFrame) -> pd.DataFrame:
    """Per-complaint instability across the full 36-variant grid.

    Metrics (per complaint):
      any_change  : at least one variant disagrees with baseline
      tier_range  : max(tier_rw) - min(tier_rw) across variants
      t2_disagree : at least one variant yields T=2 AND one yields T!=2
    """
    g = mdf.groupby("case_id")
    comp = pd.DataFrame({
        "baseline_tier": g["tier_orig"].first(),
        "n_variants": g["tier_rw"].count(),
        "min_tier_rw": g["tier_rw"].min(),
        "max_tier_rw": g["tier_rw"].max(),
        "any_change": g.apply(lambda x: int((x["tier_rw"] != x["tier_orig"]).any()), include_groups=False),
        "any_t2_rw": g["tier_rw"].apply(lambda x: int((x == 2).any())),
        "any_not_t2_rw": g["tier_rw"].apply(lambda x: int((x != 2).any())),
    })
    comp["tier_range"] = comp["max_tier_rw"] - comp["min_tier_rw"]
    comp["t2_disagree"] = ((comp["any_t2_rw"] == 1) & (comp["any_not_t2_rw"] == 1)).astype(int)

    n = len(comp)
    # Distribution of tier ranges
    range_dist = comp["tier_range"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)

    rows = [
        {"Metric": "Complaints", "Value": n},
        {"Metric": "Variants per complaint", "Value": int(comp["n_variants"].median())},
        {"Metric": "Any-change rate", "Value": round(comp["any_change"].mean() * 100, 2)},
        {"Metric": "Tier-2 disagreement rate", "Value": round(comp["t2_disagree"].mean() * 100, 2)},
        {"Metric": "Tier range = 0 (fully stable)", "Value": round(range_dist[0] * 100, 2)},
        {"Metric": "Tier range = 1", "Value": round(range_dist[1] * 100, 2)},
        {"Metric": "Tier range = 2 (spans full scale)", "Value": round(range_dist[2] * 100, 2)},
        {"Metric": "Mean any-change rate (all)", "Value": round(comp["any_change"].mean() * 100, 2)},
    ]
    # Break down by baseline tier
    for bt in [0, 1, 2]:
        sub = comp[comp["baseline_tier"] == bt]
        if len(sub) == 0:
            continue
        rows.append({
            "Metric": f"Any-change rate | baseline T={bt}",
            "Value": round(sub["any_change"].mean() * 100, 2),
        })
        rows.append({
            "Metric": f"Tier-2 disagreement | baseline T={bt}",
            "Value": round(sub["t2_disagree"].mean() * 100, 2),
        })

    return pd.DataFrame(rows)


# --- 3c. Contrasts sheet (clustered bootstrap) -----------------------


def _contrasts_sheet(mdf: pd.DataFrame) -> pd.DataFrame:
    """Pairwise dimension-value contrasts with complaint-clustered bootstrap CIs."""
    metrics = ["upgraded", "downgraded", "t2_gain", "t2_loss", "t2_net"]
    metric_labels = {
        "upgraded": "Upgrade Rate",
        "downgraded": "Downgrade Rate",
        "t2_gain": "Tier-2 Gain Rate",
        "t2_loss": "Tier-2 Loss Rate",
        "t2_net": "Net Tier-2 Delta",
    }

    n_boot = 2000
    rng = np.random.RandomState(42)
    rows = []

    for dim in DIMENSIONS:
        if dim not in mdf.columns:
            continue
        for val_a, val_b in DIM_CONTRASTS.get(dim, []):
            # Complaint-level means for each dimension value
            grp_a = mdf[mdf[dim] == val_a].groupby("case_id")[metrics].mean()
            grp_b = mdf[mdf[dim] == val_b].groupby("case_id")[metrics].mean()
            common = grp_a.index.intersection(grp_b.index)
            if len(common) < 10:
                continue

            diff_vals = grp_a.loc[common].values - grp_b.loc[common].values
            point = diff_vals.mean(axis=0)

            # Bootstrap
            n = len(common)
            boot_means = np.empty((n_boot, len(metrics)))
            for b in range(n_boot):
                idx = rng.randint(0, n, size=n)
                boot_means[b] = diff_vals[idx].mean(axis=0)

            ci_lo = np.percentile(boot_means, 2.5, axis=0)
            ci_hi = np.percentile(boot_means, 97.5, axis=0)

            for i, m in enumerate(metrics):
                frac_pos = (boot_means[:, i] > 0).mean()
                p = min(frac_pos, 1 - frac_pos) * 2
                rows.append({
                    "Dimension": DIM_LABELS.get(dim, dim),
                    "Contrast": f"{val_a} - {val_b}",
                    "Metric": metric_labels[m],
                    "Diff (pp)": point[i] * 100,
                    "CI Lo (pp)": ci_lo[i] * 100,
                    "CI Hi (pp)": ci_hi[i] * 100,
                    "p-value": p,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(df["p-value"].clip(lower=1e-10), method="fdr_bh")
        df["p-adj (FDR)"] = np.round(p_adj, 4)
    return _round_df(df, decimals=3)


# --- 3d. Severity heterogeneity sheet (A3) ----------------------------


def _severity_sheet(mdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tier-2 instability stratified by issue-severity bands with bootstrap CIs.

    Returns (main_table, interaction_table):
      main_table    - point estimates + complaint-clustered 95% CIs for key metrics
      interaction_table - severity×dimension interaction tests (bootstrap)
    """
    if "issue_severity" not in mdf.columns:
        return pd.DataFrame(), pd.DataFrame()

    n_boot = 2000
    rng = np.random.RandomState(42)
    metrics = ["upgraded", "downgraded", "t2_gain", "t2_loss", "t2_net"]
    metric_labels = {
        "upgraded": "Upgrade%", "downgraded": "Downgrade%",
        "t2_gain": "Tier-2 Gain%", "t2_loss": "Tier-2 Loss%",
        "t2_net": "Net Tier-2 Delta%",
    }

    def _boot_ci(grp: pd.DataFrame, label_cols: dict) -> list[dict]:
        """Bootstrap CIs for a group, returning one row per metric."""
        comp_means = grp.groupby("case_id")[metrics].mean()
        n_c = len(comp_means)
        if n_c < 10:
            return []
        vals = comp_means.values
        point = vals.mean(axis=0)
        boot = np.empty((n_boot, len(metrics)))
        for b in range(n_boot):
            idx = rng.randint(0, n_c, size=n_c)
            boot[b] = vals[idx].mean(axis=0)
        out_rows = []
        for i, m in enumerate(metrics):
            row = dict(label_cols)
            row["Metric"] = metric_labels[m]
            row["N (pairs)"] = len(grp)
            row["Complaints"] = n_c
            row["Estimate (pp)"] = point[i] * 100
            row["CI Lo (pp)"] = np.percentile(boot[:, i], 2.5) * 100
            row["CI Hi (pp)"] = np.percentile(boot[:, i], 97.5) * 100
            out_rows.append(row)
        return out_rows

    # --- Main table: point estimates + CIs per severity band ---------------
    rows = []
    for sev in ["low", "medium", "high"]:
        sev_df = mdf[mdf["issue_severity"] == sev]
        if len(sev_df) < 50:
            continue
        rows.extend(_boot_ci(sev_df, {"Severity": sev, "Dimension": "OVERALL", "Variant": "(all)"}))
        for dim in DIMENSIONS:
            if dim not in sev_df.columns:
                continue
            for val in sorted(sev_df[dim].unique()):
                grp = sev_df[sev_df[dim] == val]
                if len(grp) < 50:
                    continue
                rows.extend(_boot_ci(grp, {
                    "Severity": sev,
                    "Dimension": DIM_LABELS.get(dim, dim),
                    "Variant": val,
                }))
    main_df = _round_df(pd.DataFrame(rows), decimals=3)

    # --- Interaction test: severity × dimension (high - low net delta) -----
    # For each dimension value, test whether the net Tier-2 delta differs
    # between high-severity and low-severity complaints.
    interaction_rows = []
    hi = mdf[mdf["issue_severity"] == "high"]
    lo = mdf[mdf["issue_severity"] == "low"]
    if len(hi) >= 50 and len(lo) >= 50:
        # OVERALL interaction
        hi_comp = hi.groupby("case_id")["t2_net"].mean()
        lo_comp = lo.groupby("case_id")["t2_net"].mean()
        hi_point = hi_comp.mean()
        lo_point = lo_comp.mean()
        diff_point = hi_point - lo_point
        hi_arr = hi_comp.values
        lo_arr = lo_comp.values
        boot_diff = np.empty(n_boot)
        for b in range(n_boot):
            boot_diff[b] = (
                hi_arr[rng.randint(0, len(hi_arr), size=len(hi_arr))].mean()
                - lo_arr[rng.randint(0, len(lo_arr), size=len(lo_arr))].mean()
            )
        frac_pos = (boot_diff > 0).mean()
        p_val = min(frac_pos, 1 - frac_pos) * 2
        interaction_rows.append({
            "Dimension": "OVERALL",
            "Variant": "(all)",
            "High sev net delta (pp)": round(hi_point * 100, 3),
            "Low sev net delta (pp)": round(lo_point * 100, 3),
            "Diff (high-low, pp)": round(diff_point * 100, 3),
            "CI Lo (pp)": round(np.percentile(boot_diff, 2.5) * 100, 3),
            "CI Hi (pp)": round(np.percentile(boot_diff, 97.5) * 100, 3),
            "p-value": round(p_val, 4),
        })

        # Per dimension value
        for dim in DIMENSIONS:
            if dim not in mdf.columns:
                continue
            for val in sorted(mdf[dim].unique()):
                hi_g = hi[hi[dim] == val].groupby("case_id")["t2_net"].mean()
                lo_g = lo[lo[dim] == val].groupby("case_id")["t2_net"].mean()
                if len(hi_g) < 10 or len(lo_g) < 10:
                    continue
                hi_p = hi_g.mean()
                lo_p = lo_g.mean()
                diff_p = hi_p - lo_p
                hi_a = hi_g.values
                lo_a = lo_g.values
                bd = np.empty(n_boot)
                for b in range(n_boot):
                    bd[b] = (
                        hi_a[rng.randint(0, len(hi_a), size=len(hi_a))].mean()
                        - lo_a[rng.randint(0, len(lo_a), size=len(lo_a))].mean()
                    )
                fp = (bd > 0).mean()
                pv = min(fp, 1 - fp) * 2
                interaction_rows.append({
                    "Dimension": DIM_LABELS.get(dim, dim),
                    "Variant": val,
                    "High sev net delta (pp)": round(hi_p * 100, 3),
                    "Low sev net delta (pp)": round(lo_p * 100, 3),
                    "Diff (high-low, pp)": round(diff_p * 100, 3),
                    "CI Lo (pp)": round(np.percentile(bd, 2.5) * 100, 3),
                    "CI Hi (pp)": round(np.percentile(bd, 97.5) * 100, 3),
                    "p-value": round(pv, 4),
                })

    interaction_df = pd.DataFrame(interaction_rows)
    if not interaction_df.empty:
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(
            interaction_df["p-value"].clip(lower=1e-10), method="fdr_bh",
        )
        interaction_df["p-adj (FDR)"] = np.round(p_adj, 4)

    return main_df, interaction_df


# --- 3d2. Tier-2 mechanism decomposition (A5) -------------------------


def _mechanism_sheet(mdf: pd.DataFrame) -> pd.DataFrame:
    """Decompose Tier-2 gains/losses into transition-type shares."""
    # Classify each Tier-2 transition
    mdf = mdf.copy()
    mdf["t2_gain_from_0"] = ((mdf["tier_orig"] == 0) & (mdf["tier_rw"] == 2)).astype(int)
    mdf["t2_gain_from_1"] = ((mdf["tier_orig"] == 1) & (mdf["tier_rw"] == 2)).astype(int)
    mdf["t2_loss_to_1"] = ((mdf["tier_orig"] == 2) & (mdf["tier_rw"] == 1)).astype(int)
    mdf["t2_loss_to_0"] = ((mdf["tier_orig"] == 2) & (mdf["tier_rw"] == 0)).astype(int)

    rows = []

    def _add_rows(grp: pd.DataFrame, dimension: str, variant: str) -> None:
        total_gain = grp["t2_gain"].sum()
        total_loss = grp["t2_loss"].sum()
        g0 = grp["t2_gain_from_0"].sum()
        g1 = grp["t2_gain_from_1"].sum()
        l1 = grp["t2_loss_to_1"].sum()
        l0 = grp["t2_loss_to_0"].sum()
        rows.append({
            "Dimension": dimension,
            "Variant": variant,
            "N": len(grp),
            "T2 Gains (total)": total_gain,
            "  0->2 count": g0,
            "  1->2 count": g1,
            "  0->2 share%": round(g0 / total_gain * 100, 1) if total_gain > 0 else 0,
            "  1->2 share%": round(g1 / total_gain * 100, 1) if total_gain > 0 else 0,
            "T2 Losses (total)": total_loss,
            "  2->1 count": l1,
            "  2->0 count": l0,
            "  2->1 share%": round(l1 / total_loss * 100, 1) if total_loss > 0 else 0,
            "  2->0 share%": round(l0 / total_loss * 100, 1) if total_loss > 0 else 0,
        })

    _add_rows(mdf, "OVERALL", "(all variants)")
    for dim in DIMENSIONS:
        if dim not in mdf.columns:
            continue
        for val in sorted(mdf[dim].unique()):
            _add_rows(mdf[mdf[dim] == val], DIM_LABELS.get(dim, dim), val)

    return pd.DataFrame(rows)


# --- 3e. Neutralised comparison sheet (A4: same complaint set + CIs) --


def _neutral_sheet(baseline: pd.DataFrame, neutral: pd.DataFrame,
                   persona_merged: pd.DataFrame) -> pd.DataFrame:
    """Apples-to-apples comparison on the SAME 2000 causal-sample complaints.

    Computes neutralised instability, persona instability (overall), and
    the difference with complaint-clustered bootstrap CIs.
    """
    causal_ids = set(persona_merged["case_id"].unique())

    # Build neutral paired set restricted to causal complaints
    neu = neutral[neutral["case_id"].isin(causal_ids)].merge(
        baseline[["case_id", "model", "tier"]],
        on=["case_id", "model"], how="inner", suffixes=("_neutral", "_orig"),
    )
    neu["t2_gain"] = ((neu["tier_neutral"] == 2) & (neu["tier_orig"] != 2)).astype(int)
    neu["t2_loss"] = ((neu["tier_neutral"] != 2) & (neu["tier_orig"] == 2)).astype(int)
    neu["upgraded"] = (neu["tier_neutral"] > neu["tier_orig"]).astype(int)
    neu["downgraded"] = (neu["tier_neutral"] < neu["tier_orig"]).astype(int)
    neu["unchanged"] = (neu["tier_neutral"] == neu["tier_orig"]).astype(int)
    neu["t2_net"] = neu["t2_gain"] - neu["t2_loss"]

    metrics = ["upgraded", "downgraded", "unchanged", "t2_gain", "t2_loss", "t2_net"]
    metric_labels = {
        "upgraded": "Upgrade%", "downgraded": "Downgrade%", "unchanged": "Unchanged%",
        "t2_gain": "Tier-2 Gain%", "t2_loss": "Tier-2 Loss%", "t2_net": "Net Tier-2 Delta%",
    }

    n_boot = 2000
    rng = np.random.RandomState(42)
    rows = []

    for model in sorted(persona_merged["model"].unique()):
        pmdf = persona_merged[persona_merged["model"] == model]
        nmdf = neu[neu["model"] == model]

        # Complaint-level means
        p_comp = pmdf.groupby("case_id")[metrics].mean()
        n_comp = nmdf.groupby("case_id")[metrics].mean()
        common = p_comp.index.intersection(n_comp.index)
        if len(common) < 10:
            continue

        p_vals = p_comp.loc[common].values
        n_vals = n_comp.loc[common].values

        p_point = p_vals.mean(axis=0)
        n_point = n_vals.mean(axis=0)

        # Bootstrap CIs for both and for the difference
        nc = len(common)
        p_boot = np.empty((n_boot, len(metrics)))
        n_boot_arr = np.empty((n_boot, len(metrics)))
        for b in range(n_boot):
            idx = rng.randint(0, nc, size=nc)
            p_boot[b] = p_vals[idx].mean(axis=0)
            n_boot_arr[b] = n_vals[idx].mean(axis=0)

        for i, m in enumerate(metrics):
            lbl = metric_labels[m]
            mult = 100  # convert to percentage points
            rows.append({
                "Model": model,
                "Comparison": "Baseline -> Persona (overall)",
                "Metric": lbl,
                "Estimate (pp)": round(p_point[i] * mult, 3),
                "CI Lo (pp)": round(np.percentile(p_boot[:, i], 2.5) * mult, 3),
                "CI Hi (pp)": round(np.percentile(p_boot[:, i], 97.5) * mult, 3),
            })
            rows.append({
                "Model": model,
                "Comparison": f"Baseline -> Neutralised (same {len(common)} complaints)",
                "Metric": lbl,
                "Estimate (pp)": round(n_point[i] * mult, 3),
                "CI Lo (pp)": round(np.percentile(n_boot_arr[:, i], 2.5) * mult, 3),
                "CI Hi (pp)": round(np.percentile(n_boot_arr[:, i], 97.5) * mult, 3),
            })
            # Difference (persona - neutral)
            diff_boot = p_boot[:, i] - n_boot_arr[:, i]
            diff_point = p_point[i] - n_point[i]
            frac_pos = (diff_boot > 0).mean()
            p_val = min(frac_pos, 1 - frac_pos) * 2
            rows.append({
                "Model": model,
                "Comparison": "Difference (persona - neutral)",
                "Metric": lbl,
                "Estimate (pp)": round(diff_point * mult, 3),
                "CI Lo (pp)": round(np.percentile(diff_boot, 2.5) * mult, 3),
                "CI Hi (pp)": round(np.percentile(diff_boot, 97.5) * mult, 3),
                "p-value": round(p_val, 4),
            })

    return pd.DataFrame(rows)


# --- 3f. Drift-filter sensitivity sheet -------------------------------


def _drift_sheet(merged: pd.DataFrame, narratives: pd.Series,
                 cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jaccard similarity distribution + sensitivity with/without drift filter.

    Returns (comparison_table, distribution_table).
    """
    # Sample one variant per complaint per model for similarity check
    rng = np.random.RandomState(42)
    sample_idx = (
        merged.groupby(["case_id", "model"])
        .apply(lambda g: g.sample(1, random_state=rng), include_groups=False)
        .index
    )
    # include_groups=False drops the grouping cols; index is the original row index
    sample = merged.loc[[idx[-1] for idx in sample_idx]].copy()
    sample["narrative"] = sample["case_id"].map(narratives)

    # Read rewritten text and compute similarity
    sims = []
    for _, row in sample.iterrows():
        rw = _read_rewrite_cache(row["model"], row["narrative"], row["variant"], cache_dir)
        sims.append(_word_jaccard(row["narrative"], rw) if rw else np.nan)
    sample["jaccard"] = sims

    valid = sample["jaccard"].dropna()
    pcts = [5, 10, 25, 50, 75, 90, 95]
    dist_rows = [
        {"Statistic": "N (sampled pairs)", "Value": len(valid)},
        {"Statistic": "Mean", "Value": round(float(valid.mean()), 4)},
        {"Statistic": "Std Dev", "Value": round(float(valid.std()), 4)},
    ]
    for p in pcts:
        dist_rows.append({"Statistic": f"P{p}", "Value": round(float(np.percentile(valid, p)), 4)})
    dist_df = pd.DataFrame(dist_rows)

    # Drift threshold: bottom 10%
    threshold = float(np.percentile(valid, 10))
    drift_cases = set(sample.loc[sample["jaccard"] < threshold, "case_id"].unique())
    n_flagged = len(drift_cases)
    n_total = merged["case_id"].nunique()
    dist_rows.append({"Statistic": "Drift threshold (P10)", "Value": round(threshold, 4)})
    dist_rows.append({"Statistic": "Complaints flagged", "Value": n_flagged})
    dist_rows.append({"Statistic": "% flagged", "Value": round(n_flagged / n_total * 100, 1)})
    dist_df = pd.DataFrame(dist_rows)

    # Sensitivity comparison: key metrics with/without drift filter
    comp_rows = []
    merged_filt = merged[~merged["case_id"].isin(drift_cases)]

    for model in sorted(merged["model"].unique()):
        for dim in DIMENSIONS:
            if dim not in merged.columns:
                continue
            for val in sorted(merged[dim].unique()):
                g_all = merged[(merged["model"] == model) & (merged[dim] == val)]
                g_flt = merged_filt[(merged_filt["model"] == model) & (merged_filt[dim] == val)]
                delta_all = (g_all["t2_gain"].mean() - g_all["t2_loss"].mean()) * 100
                delta_flt = (g_flt["t2_gain"].mean() - g_flt["t2_loss"].mean()) * 100 if len(g_flt) > 0 else np.nan
                comp_rows.append({
                    "Model": model,
                    "Dimension": DIM_LABELS.get(dim, dim),
                    "Variant": val,
                    "N (all)": len(g_all),
                    "Net T2 Delta% (all)": round(delta_all, 3),
                    "N (filtered)": len(g_flt),
                    "Net T2 Delta% (filtered)": round(delta_flt, 3),
                    "Change (pp)": round(delta_flt - delta_all, 3) if np.isfinite(delta_flt) else np.nan,
                })

    comp_df = pd.DataFrame(comp_rows)
    return comp_df, dist_df


# --- 3g. Metadata sheet -----------------------------------------------


def _metadata_sheet(merged: pd.DataFrame) -> pd.DataFrame:
    """Document the unit of analysis and methodology."""
    n_complaints = merged["case_id"].nunique()
    n_models = merged["model"].nunique()
    n_variants = merged["variant"].nunique()
    models = ", ".join(sorted(merged["model"].unique()))
    rows = [
        ("Generated", datetime.now().isoformat()),
        ("Unit of analysis", "Paired comparison: (complaint, model, persona_variant) vs baseline"),
        ("Each complaint contributes", f"{n_variants} paired observations per model ({n_variants} persona variants)"),
        ("Total complaints in causal sample", n_complaints),
        ("Models", models),
        ("Total paired observations", len(merged)),
        ("Variant encoding", "Full persona bundle: age_geo_edu_ses (e.g. young_urban_college_high)"),
        ("Dimension aggregation", "Each row in summary/transition sheets aggregates across all variants sharing that dimension value"),
        ("Tier-2 Gain", "Pr(T_rewrite=2 & T_orig!=2): complaint upgraded TO Tier 2 after rewriting"),
        ("Tier-2 Loss", "Pr(T_rewrite!=2 & T_orig=2): complaint downgraded FROM Tier 2 after rewriting"),
        ("Net Tier-2 Delta", "Gain - Loss: net change in Tier-2 allocation rate (risk difference)"),
        ("Complaint-level instability", "Per-complaint any-change, tier-range, and Tier-2 disagreement across all 36 variants"),
        ("Contrasts", "Pairwise differences (A-B) in complaint-level means, 2000-iteration bootstrap, BH FDR correction"),
        ("Clustering", "Bootstrap resamples at complaint level to respect within-complaint dependence"),
        ("Severity heterogeneity", "Instability stratified by issue-severity bands (low/medium/high) with complaint-clustered bootstrap CIs"),
        ("Severity interaction", "Tests whether net Tier-2 delta differs between high- and low-severity complaints (bootstrap, BH FDR)"),
        ("Mechanism decomposition", "Tier-2 gains split into 0->2 vs 1->2; losses into 2->1 vs 2->0"),
        ("Neutralised comparison", "Same 2000 causal complaints; persona vs neutralised with complaint-clustered bootstrap CIs"),
        ("Drift filter", "Word-level Jaccard similarity; complaints below P10 threshold excluded; sensitivity comparison"),
    ]
    return pd.DataFrame(rows, columns=["Key", "Value"])


# --- Main instability orchestrator ------------------------------------


def instability_analysis(complaints: pd.DataFrame) -> None:
    """Output 3: comprehensive tier instability analysis."""
    outcome_path = _latest_file("output/results_outcome_*.csv")
    persona_path = _latest_file("output/results_persona_decisions_*.csv")
    cache_dir = Path("cache")

    print("Instability analysis")
    print(f"  Baseline : {outcome_path}")
    print(f"  Persona  : {persona_path}")

    baseline = pd.read_csv(outcome_path, usecols=["case_id", "model", "tier"])
    persona = pd.read_csv(persona_path)

    # --- Build merged dataset with derived flags ---
    merged = persona.merge(
        baseline, on=["case_id", "model"], how="inner", suffixes=("_rw", "_orig"),
    )
    merged["tier_change"] = merged["tier_rw"] - merged["tier_orig"]
    merged["upgraded"] = (merged["tier_change"] > 0).astype(int)
    merged["downgraded"] = (merged["tier_change"] < 0).astype(int)
    merged["unchanged"] = (merged["tier_change"] == 0).astype(int)
    merged["t2_gain"] = ((merged["tier_rw"] == 2) & (merged["tier_orig"] != 2)).astype(int)
    merged["t2_loss"] = ((merged["tier_rw"] != 2) & (merged["tier_orig"] == 2)).astype(int)
    merged["t2_net"] = merged["t2_gain"] - merged["t2_loss"]

    # Join product/issue for severity analysis and compute issue_severity
    sev_cols = complaints[["case_id", "product", "issue"]].drop_duplicates().copy()
    sev_cols["issue_severity"] = sev_cols.apply(
        lambda r: compute_issue_severity(str(r["product"]), str(r["issue"])), axis=1,
    )
    merged = merged.merge(sev_cols[["case_id", "issue_severity"]], on="case_id", how="left")

    # Narrative lookup for drift analysis
    narratives = complaints.set_index("case_id")["narrative"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(f"output/instability-analysis-{ts}.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    persona_summaries: dict[str, pd.DataFrame] = {}

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for model in sorted(merged["model"].unique()):
            mdf = merged[merged["model"] == model]
            short = model.replace("-", "")[:10]

            # Sheet: Summary (with A2 overall headline)
            print(f"  [{model}] Summary ...")
            summary = _summary_sheet(mdf)
            persona_summaries[model] = summary
            summary.to_excel(writer, sheet_name=f"{short}_summary", index=False)

            # Sheet: Transitions
            print(f"  [{model}] Transitions ...")
            _transition_sheet(mdf).to_excel(writer, sheet_name=f"{short}_trans", index=False)

            # Sheet: Complaint-level instability (A1)
            print(f"  [{model}] Complaint-level ...")
            _complaint_sheet(mdf).to_excel(writer, sheet_name=f"{short}_complaint", index=False)

            # Sheet: Contrasts
            print(f"  [{model}] Contrasts (bootstrap) ...")
            _contrasts_sheet(mdf).to_excel(writer, sheet_name=f"{short}_contrasts", index=False)

            # Sheet: Severity heterogeneity (A3: issue-severity bands + CIs)
            print(f"  [{model}] Severity heterogeneity (bootstrap) ...")
            sev_main, sev_interaction = _severity_sheet(mdf)
            if not sev_main.empty:
                sev_main.to_excel(writer, sheet_name=f"{short}_severity", index=False)
            if not sev_interaction.empty:
                sev_interaction.to_excel(writer, sheet_name=f"{short}_sev_intxn", index=False)

            # Sheet: Tier-2 mechanism decomposition (A5)
            print(f"  [{model}] Mechanism decomposition ...")
            _mechanism_sheet(mdf).to_excel(writer, sheet_name=f"{short}_mechanism", index=False)

        # Sheet: Neutralised comparison (A4: same complaints + CIs)
        try:
            neutral_path = _latest_file("output/results_neutralized_*.csv")
            print(f"  Neutralised comparison ({neutral_path}) ...")
            neutral = pd.read_csv(neutral_path, usecols=["case_id", "model", "tier"])
            _neutral_sheet(baseline, neutral, merged).to_excel(
                writer, sheet_name="neutral", index=False,
            )
        except SystemExit:
            print("  Neutralised data not found; skipping sheet.")

        # Sheet: Drift-filter sensitivity
        print("  Drift-filter sensitivity (reading cache) ...")
        comp_df, dist_df = _drift_sheet(merged, narratives, cache_dir)
        dist_df.to_excel(writer, sheet_name="drift_dist", index=False)
        comp_df.to_excel(writer, sheet_name="drift_sensitivity", index=False)

        # Sheet: Metadata
        _metadata_sheet(merged).to_excel(writer, sheet_name="metadata", index=False)

    print(f"Wrote {out}")


# ===================================================================
# Entrypoint
# ===================================================================


def main() -> None:
    parquet_path = _largest_parquet()
    print(f"Complaints: {parquet_path}")

    complaints = pd.read_parquet(
        parquet_path,
        columns=["case_id", "narrative", "product", "issue", "complaint_id",
                 "product_category", "issue_category"],
    )

    remedy_reasons(complaints)
    print()
    remedy_changed(complaints)
    print()
    instability_analysis(complaints)


if __name__ == "__main__":
    main()
