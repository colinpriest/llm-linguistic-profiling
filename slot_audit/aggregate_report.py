"""Complaint-level aggregation, reporting, and output generation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import DEFAULT_WEIGHTS, SlotStatus

FAMILIES = ["numeric", "temporal", "polarity", "anchor"]

# Real-data viability thresholds (provisional)
REAL_DATA_THRESHOLDS = {
    "polarity_contradiction_rate": 0.05,
    "numeric_mutation_rate": 0.10,
    "temporal_presence_preservation_rate": 0.70,
    "anchor_presence_preservation_rate": 0.70,
    "high_fidelity_pair_rate": 0.40,
    "high_fidelity_complaint_rate": 0.25,
}


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Complaint-level aggregation
# ---------------------------------------------------------------------------

def aggregate_complaints(
    pair_df: pd.DataFrame,
    strict_threshold: float = 1.0,
) -> pd.DataFrame:
    """Aggregate pair results to one row per complaint."""
    rows = []
    for cid, grp in pair_df.groupby("complaint_id"):
        n = len(grp)
        rec: Dict[str, Any] = {"complaint_id": cid, "n_rewrites": n}

        for fam in FAMILIES:
            col = f"{fam}_pass"
            rec[f"{fam}_pass_rate"] = grp[col].mean()

        rec["strict_pass_count"] = int(grp["strict_pair_pass"].sum())
        rec["strict_pass_rate"] = grp["strict_pair_pass"].mean()

        rec["high_fidelity_count"] = int(grp["high_fidelity_pass"].sum())
        rec["high_fidelity_rate"] = grp["high_fidelity_pass"].mean()

        rec["contradiction_count"] = int(grp["has_contradiction"].sum())
        rec["contradiction_rate"] = grp["has_contradiction"].mean()

        # Strict criteria: pass-rate >= threshold AND no contradictions
        meets_strict = rec["strict_pass_rate"] >= strict_threshold
        meets_practical = (
            rec["strict_pass_rate"] >= strict_threshold
            and rec["contradiction_count"] == 0
        )
        rec["meets_strict"] = meets_strict
        rec["meets_practical"] = meets_practical
        rec["meets_criteria"] = meets_practical

        # High-fidelity criteria: all rewrites are high-fidelity
        rec["meets_high_fidelity"] = rec["high_fidelity_rate"] >= 1.0
        rows.append(rec)

    return pd.DataFrame(rows)


def aggregate_high_fidelity(
    pair_df: pd.DataFrame,
    complaint_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute high-fidelity subset metrics."""
    n_pairs = len(pair_df)
    n_complaints = len(complaint_df)

    hf_pairs = int(pair_df["high_fidelity_pass"].sum())
    hf_pair_rate = hf_pairs / n_pairs if n_pairs else 0.0

    hf_complaints = int(complaint_df["meets_high_fidelity"].sum())
    hf_complaint_rate = hf_complaints / n_complaints if n_complaints else 0.0

    return {
        "n_pairs": hf_pairs,
        "pair_rate": round(hf_pair_rate, 4),
        "pair_rate_ci": list(wilson_ci(hf_pairs, n_pairs)),
        "n_complaints": hf_complaints,
        "complaint_rate": round(hf_complaint_rate, 4),
    }


# ---------------------------------------------------------------------------
# Fixture accuracy (auto-detected when expected_ columns present)
# ---------------------------------------------------------------------------

def compute_fixture_accuracy(
    pair_df: pd.DataFrame,
    input_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compare actual results against expected labels in the fixture.

    Requires ``input_df`` to contain ``expected_*`` columns aligned
    row-by-row with ``pair_df``.
    """
    n = len(pair_df)
    total_correct = 0
    per_family: Dict[str, Dict[str, Any]] = {}

    for fam in FAMILIES:
        actual = pair_df[f"{fam}_status"].values
        expected = input_df[f"expected_{fam}_status"].values
        matches = sum(1 for a, e in zip(actual, expected) if a == e)
        mismatches = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                cid = pair_df.iloc[i]["complaint_id"]
                rid = pair_df.iloc[i]["rewrite_id"]
                mismatches.append({"row": f"{cid}-{rid}", "expected": e, "actual": a})
        per_family[fam] = {
            "accuracy": round(matches / n, 4) if n else 0.0,
            "correct": matches,
            "total": n,
            "mismatches": mismatches,
        }
        total_correct += matches

    # Strict pass accuracy
    actual_strict = pair_df["strict_pair_pass"].astype(int).values
    expected_strict = input_df["expected_strict_pass"].astype(int).values
    strict_matches = int((actual_strict == expected_strict).sum())

    # Contradiction flag accuracy
    actual_contra = pair_df["has_contradiction"].astype(int).values
    expected_contra = input_df["expected_contradiction_flag"].astype(int).values
    contra_matches = int((actual_contra == expected_contra).sum())

    return {
        "overall_accuracy": round(total_correct / (n * len(FAMILIES)), 4) if n else 0.0,
        "per_family": per_family,
        "strict_pass_accuracy": round(strict_matches / n, 4) if n else 0.0,
        "contradiction_accuracy": round(contra_matches / n, 4) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Global / subgroup metric computation
# ---------------------------------------------------------------------------

def _family_metrics(pair_df: pd.DataFrame, family: str) -> Dict[str, Any]:
    """Compute per-family metrics from the pair-level DataFrame."""
    status_col = f"{family}_status"
    n = len(pair_df)
    counts = pair_df[status_col].value_counts()

    pres_abs = int(counts.get(SlotStatus.PRESERVED_ABSENCE.value, 0))
    pres_prs = int(counts.get(SlotStatus.PRESERVED_PRESENT.value, 0))
    omission = int(counts.get(SlotStatus.OMISSION.value, 0))
    addition = int(counts.get(SlotStatus.ADDITION.value, 0))
    mutation = int(counts.get(SlotStatus.MUTATION.value, 0))
    contradiction = int(counts.get(SlotStatus.MUTATION_POLARITY_CONTRADICTION.value, 0))

    src_present = pres_prs + omission + mutation + contradiction
    src_absent = pres_abs + addition

    overall_pres = pres_abs + pres_prs
    overall_rate = overall_pres / n if n else 0.0
    pres_rate = pres_prs / src_present if src_present else float("nan")
    abs_rate = pres_abs / src_absent if src_absent else float("nan")

    metrics: Dict[str, Any] = {
        "total_pairs": n,
        "preserved_absence": pres_abs,
        "preserved_present": pres_prs,
        "omission": omission,
        "addition": addition,
        "mutation": mutation,
        "overall_preservation_rate": round(overall_rate, 4),
        "presence_preservation_rate": round(pres_rate, 4) if not math.isnan(pres_rate) else None,
        "absence_preservation_rate": round(abs_rate, 4) if not math.isnan(abs_rate) else None,
        "omission_rate": round(omission / n, 4) if n else 0.0,
        "addition_rate": round(addition / n, 4) if n else 0.0,
        "mutation_rate": round((mutation + contradiction) / n, 4) if n else 0.0,
    }

    if family == "polarity":
        metrics["contradiction"] = contradiction
        metrics["contradiction_rate"] = round(contradiction / n, 4) if n else 0.0

    # Wilson CIs for key rates
    lo, hi = wilson_ci(overall_pres, n)
    metrics["overall_preservation_ci"] = [round(lo, 4), round(hi, 4)]

    if src_present:
        plo, phi = wilson_ci(pres_prs, src_present)
        metrics["presence_preservation_ci"] = [round(plo, 4), round(phi, 4)]

    return metrics


def compute_metrics(
    pair_df: pd.DataFrame,
    complaint_df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the full metrics dictionary."""
    n_pairs = len(pair_df)
    n_complaints = len(complaint_df)

    strict_pass = int(pair_df["strict_pair_pass"].sum())
    strict_pair_rate = strict_pass / n_pairs if n_pairs else 0.0

    strict_complaint = int(complaint_df["meets_criteria"].sum())
    strict_complaint_rate = strict_complaint / n_complaints if n_complaints else 0.0

    contradict_pairs = int(pair_df["has_contradiction"].sum())
    contradiction_rate = contradict_pairs / n_pairs if n_pairs else 0.0

    # Numeric mutation rate
    num_status = pair_df["numeric_status"].value_counts()
    num_mutation = int(num_status.get(SlotStatus.MUTATION.value, 0))
    numeric_mutation_rate = num_mutation / n_pairs if n_pairs else 0.0

    # High-fidelity metrics
    hf = aggregate_high_fidelity(pair_df, complaint_df)

    # Per-family metrics
    per_family = {fam: _family_metrics(pair_df, fam) for fam in FAMILIES}

    # Real-data viability gate (provisional thresholds)
    t = REAL_DATA_THRESHOLDS
    temp_pres = per_family["temporal"].get("presence_preservation_rate")
    anch_pres = per_family["anchor"].get("presence_preservation_rate")

    viability_checks = {
        "polarity_contradiction_rate": {
            "threshold": f"<= {t['polarity_contradiction_rate']}",
            "actual": round(contradiction_rate, 4),
            "pass": contradiction_rate <= t["polarity_contradiction_rate"],
        },
        "numeric_mutation_rate": {
            "threshold": f"<= {t['numeric_mutation_rate']}",
            "actual": round(numeric_mutation_rate, 4),
            "pass": numeric_mutation_rate <= t["numeric_mutation_rate"],
        },
        "temporal_presence_preservation": {
            "threshold": f">= {t['temporal_presence_preservation_rate']}",
            "actual": temp_pres,
            "pass": temp_pres is not None and temp_pres >= t["temporal_presence_preservation_rate"],
        },
        "anchor_presence_preservation": {
            "threshold": f">= {t['anchor_presence_preservation_rate']}",
            "actual": anch_pres,
            "pass": anch_pres is not None and anch_pres >= t["anchor_presence_preservation_rate"],
        },
        "high_fidelity_pair_rate": {
            "threshold": f">= {t['high_fidelity_pair_rate']}",
            "actual": hf["pair_rate"],
            "pass": hf["pair_rate"] >= t["high_fidelity_pair_rate"],
        },
        "high_fidelity_complaint_rate": {
            "threshold": f">= {t['high_fidelity_complaint_rate']}",
            "actual": hf["complaint_rate"],
            "pass": hf["complaint_rate"] >= t["high_fidelity_complaint_rate"],
        },
    }
    all_pass = all(c["pass"] for c in viability_checks.values())

    metrics: Dict[str, Any] = {
        "global": {
            "n_pairs": n_pairs,
            "n_complaints": n_complaints,
            "strict_pair_pass_rate": round(strict_pair_rate, 4),
            "strict_pair_pass_ci": list(wilson_ci(strict_pass, n_pairs)),
            "strict_complaint_pass_rate": round(strict_complaint_rate, 4),
            "polarity_contradiction_rate": round(contradiction_rate, 4),
            "numeric_mutation_rate": round(numeric_mutation_rate, 4),
            "high_fidelity": hf,
            "viability_gate": {
                "pass": all_pass,
                "checks": viability_checks,
            },
        },
        "per_family": per_family,
    }

    # Subgroup breakdowns
    if group_cols:
        subgroups: Dict[str, Any] = {}
        for col in group_cols:
            if col not in pair_df.columns:
                continue
            grp_metrics: Dict[str, Any] = {}
            for val, gdf in pair_df.groupby(col):
                grp_metrics[str(val)] = {
                    "n_pairs": len(gdf),
                    "strict_pair_pass_rate": round(gdf["strict_pair_pass"].mean(), 4),
                    "high_fidelity_pair_rate": round(gdf["high_fidelity_pass"].mean(), 4),
                    "contradiction_rate": round(gdf["has_contradiction"].mean(), 4),
                }
            subgroups[col] = grp_metrics
        metrics["subgroups"] = subgroups

    return metrics


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_pair_csv(pair_df: pd.DataFrame, path: Path) -> None:
    pair_df.to_csv(path, index=False)


def write_complaint_csv(complaint_df: pd.DataFrame, path: Path) -> None:
    complaint_df.to_csv(path, index=False)


def write_preserved_csv(
    complaint_df: pd.DataFrame,
    path: Path,
    strict_threshold: float,
) -> None:
    preserved = complaint_df[complaint_df["meets_criteria"]].copy()
    if preserved.empty:
        preserved = pd.DataFrame(columns=["complaint_id", "criteria", "strict_pass_rate", "contradiction_rate"])
    else:
        preserved["criteria"] = preserved.apply(
            lambda r: "strict" if r["strict_pass_rate"] >= 1.0 else "practical", axis=1,
        )
    cols = ["complaint_id", "criteria", "strict_pass_rate", "contradiction_rate"]
    preserved[[c for c in cols if c in preserved.columns]].to_csv(path, index=False)


def write_high_fidelity_csvs(
    pair_df: pd.DataFrame,
    complaint_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write high_fidelity_pairs.csv and high_fidelity_complaints.csv."""
    hf_pairs = pair_df[pair_df["high_fidelity_pass"]].copy()
    hf_pairs.to_csv(out_dir / "high_fidelity_pairs.csv", index=False)

    hf_complaints = complaint_df[complaint_df["meets_high_fidelity"]].copy()
    hf_complaints.to_csv(out_dir / "high_fidelity_complaints.csv", index=False)


def write_json_metrics(metrics: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


def write_markdown_report(
    metrics: Dict[str, Any],
    pair_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write a human-readable Markdown summary."""
    g = metrics["global"]
    hf = g["high_fidelity"]
    is_fixture = "fixture_accuracy" in metrics

    lines = ["# Slot Preservation Audit Report\n"]

    # --- Fixture correctness section (if applicable) ---
    if is_fixture:
        fa = metrics["fixture_accuracy"]
        lines += [
            "## Mode: FIXTURE CORRECTNESS\n",
            "Expected-label columns detected. Evaluating implementation accuracy.\n",
            "| Family | Accuracy | Correct | Total |",
            "|--------|----------|---------|-------|",
        ]
        for fam in FAMILIES:
            fm = fa["per_family"][fam]
            lines.append(f"| {fam} | {fm['accuracy']:.4f} | {fm['correct']} | {fm['total']} |")
        lines += [
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall label accuracy | {fa['overall_accuracy']:.4f} |",
            f"| Strict-pass accuracy | {fa['strict_pass_accuracy']:.4f} |",
            f"| Contradiction-flag accuracy | {fa['contradiction_accuracy']:.4f} |",
            "",
        ]
        # Show mismatches
        any_mismatch = False
        for fam in FAMILIES:
            for mm in fa["per_family"][fam]["mismatches"]:
                if not any_mismatch:
                    lines += ["### Mismatches\n",
                              "| Row | Family | Expected | Actual |",
                              "|-----|--------|----------|--------|"]
                    any_mismatch = True
                lines.append(f"| {mm['row']} | {fam} | {mm['expected']} | {mm['actual']} |")
        if any_mismatch:
            lines.append("")
        else:
            lines.append("No mismatches detected.\n")

    # --- Global metrics ---
    lines += [
        "## Global Metrics\n",
        "| Metric | Value | 95% CI |",
        "|--------|-------|--------|",
        f"| Strict pair pass rate | {g['strict_pair_pass_rate']:.4f} "
        f"| [{g['strict_pair_pass_ci'][0]:.4f}, {g['strict_pair_pass_ci'][1]:.4f}] |",
        f"| High-fidelity pair rate | {hf['pair_rate']:.4f} "
        f"| [{hf['pair_rate_ci'][0]:.4f}, {hf['pair_rate_ci'][1]:.4f}] |",
        f"| Strict complaint pass rate | {g['strict_complaint_pass_rate']:.4f} | - |",
        f"| High-fidelity complaint rate | {hf['complaint_rate']:.4f} | - |",
        f"| Polarity contradiction rate | {g['polarity_contradiction_rate']:.4f} | - |",
        f"| Numeric mutation rate | {g['numeric_mutation_rate']:.4f} | - |",
        f"| Total pairs | {g['n_pairs']} | - |",
        f"| Total complaints | {g['n_complaints']} | - |",
        f"| High-fidelity pairs | {hf['n_pairs']} | - |",
        f"| High-fidelity complaints | {hf['n_complaints']} | - |",
        "",
    ]

    # --- Viability gate ---
    vg = g["viability_gate"]
    gate_icon = "PASS" if vg["pass"] else "FAIL"
    lines += [
        f"## Viability Gate: **{gate_icon}**\n",
        "| Criterion | Threshold | Actual | Result |",
        "|-----------|-----------|--------|--------|",
    ]
    for name, check in vg["checks"].items():
        actual_str = f"{check['actual']:.4f}" if check["actual"] is not None else "N/A"
        lines.append(
            f"| {name} | {check['threshold']} | {actual_str} "
            f"| {'PASS' if check['pass'] else 'FAIL'} |"
        )
    lines.append("")

    # --- Per-family (with both overall and presence-only rates) ---
    for fam in FAMILIES:
        fm = metrics["per_family"][fam]
        pres_rate = fm["presence_preservation_rate"]
        pres_ci = fm.get("presence_preservation_ci")
        pres_str = f"{pres_rate:.4f}" if pres_rate is not None else "N/A"
        pres_ci_str = (f"[{pres_ci[0]:.4f}, {pres_ci[1]:.4f}]"
                       if pres_ci else "-")
        abs_rate = fm["absence_preservation_rate"]
        abs_str = f"{abs_rate:.4f}" if abs_rate is not None else "N/A"

        lines += [
            f"## {fam.title()} Slots\n",
            "| Metric | Value | 95% CI |",
            "|--------|-------|--------|",
            f"| Overall preservation rate | {fm['overall_preservation_rate']:.4f} "
            f"| [{fm['overall_preservation_ci'][0]:.4f}, {fm['overall_preservation_ci'][1]:.4f}] |",
            f"| **Presence preservation rate** | **{pres_str}** | {pres_ci_str} |",
            f"| Absence preservation rate | {abs_str} | - |",
            f"| Omission rate | {fm['omission_rate']:.4f} | - |",
            f"| Addition rate | {fm['addition_rate']:.4f} | - |",
            f"| Mutation rate | {fm['mutation_rate']:.4f} | - |",
        ]
        if fam == "polarity":
            lines.append(f"| Contradiction rate | {fm['contradiction_rate']:.4f} | - |")
        lines.append("")

    # --- Top failing pairs (up to 10) ---
    fails = pair_df[~pair_df["strict_pair_pass"]].head(10)
    if not fails.empty:
        lines += [
            "## Top Failing Pairs (up to 10)\n",
            "| complaint_id | rewrite_id | numeric | temporal | polarity | anchor | hi-fi |",
            "|-------------|------------|---------|----------|----------|--------|-------|",
        ]
        for _, r in fails.iterrows():
            hf_icon = "Y" if r["high_fidelity_pass"] else "N"
            lines.append(
                f"| {r['complaint_id']} | {r['rewrite_id']} "
                f"| {r['numeric_status']} | {r['temporal_status']} "
                f"| {r['polarity_status']} | {r['anchor_status']} | {hf_icon} |"
            )
        lines.append("")

    # --- Subgroups ---
    if "subgroups" in metrics:
        lines.append("## Subgroup Analysis\n")
        for col, grp_data in metrics["subgroups"].items():
            lines += [
                f"### By {col}\n",
                "| Group | N pairs | Strict pass | Hi-fi pass | Contradiction |",
                "|-------|---------|-------------|------------|---------------|",
            ]
            for val, gm in grp_data.items():
                lines.append(
                    f"| {val} | {gm['n_pairs']} "
                    f"| {gm['strict_pair_pass_rate']:.4f} "
                    f"| {gm['high_fidelity_pair_rate']:.4f} "
                    f"| {gm['contradiction_rate']:.4f} |"
                )
            lines.append("")

    # --- Limitations ---
    lines += [
        "## Limitations\n",
        "* Regex/rule methods are conservative and may miss paraphrased equivalents.",
        "* Polarity detection is approximate (scope/long-distance negation can be hard).",
        "* Hard-slot preservation does not validate subjective claim equivalence.",
        "* Money sparsity may reduce usefulness of numeric subtype-specific conclusions.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
