"""Main CLI orchestration for the slot-preservation audit."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from . import (
    CONTRADICTION_PENALTY,
    DEFAULT_WEIGHTS,
    SlotStatus,
)
from .aggregate_report import (
    aggregate_complaints,
    aggregate_high_fidelity,
    compute_fixture_accuracy,
    compute_metrics,
    write_complaint_csv,
    write_high_fidelity_csvs,
    write_json_metrics,
    write_markdown_report,
    write_pair_csv,
    write_preserved_csv,
)
from .compare_slots import compare_polarity, compare_set_slots
from .extract_anchors import extract_anchors, load_anchor_config
from .extract_numeric import extract_numeric
from .extract_polarity import extract_polarity, load_event_config
from .extract_temporal import extract_temporal

logger = logging.getLogger(__name__)

# Expected-label columns that trigger fixture-correctness mode
_EXPECTED_COLS = [
    "expected_numeric_status",
    "expected_temporal_status",
    "expected_polarity_status",
    "expected_anchor_status",
    "expected_strict_pass",
    "expected_contradiction_flag",
]


# ---------------------------------------------------------------------------
# Core audit pipeline
# ---------------------------------------------------------------------------

def audit_pair(
    source_text: str,
    rewrite_text: str,
    anchor_config: Dict[str, Any],
    event_config: Dict[str, Any],
    negation_window: int = 4,
) -> Dict[str, Any]:
    """Run the four-family slot audit on a single source->rewrite pair.

    Returns a dict with per-family statuses, pass flags, and score.
    """
    # 1. Extract anchors (with spans for polarity masking)
    src_anch_set, src_anch_spans = extract_anchors(source_text, anchor_config)
    rw_anch_set, rw_anch_spans = extract_anchors(rewrite_text, anchor_config)

    # 2. Extract numeric
    src_num = extract_numeric(source_text)
    rw_num = extract_numeric(rewrite_text)

    # 3. Extract temporal
    src_tmp = extract_temporal(source_text)
    rw_tmp = extract_temporal(rewrite_text)

    # 4. Extract polarity (with anchor masking)
    src_pol = extract_polarity(source_text, event_config, src_anch_spans, negation_window)
    rw_pol = extract_polarity(rewrite_text, event_config, rw_anch_spans, negation_window)

    # 5. Compare each family
    num_cmp = compare_set_slots(
        {s.canonical() for s in src_num},
        {s.canonical() for s in rw_num},
        "numeric",
    )
    tmp_cmp = compare_set_slots(
        {s.canonical() for s in src_tmp},
        {s.canonical() for s in rw_tmp},
        "temporal",
    )
    pol_cmp = compare_polarity(src_pol, rw_pol)
    anch_cmp = compare_set_slots(src_anch_set, rw_anch_set, "anchor")

    # 6. Strict pass / fail (all families must be PRESERVED)
    _preserved = (SlotStatus.PRESERVED_ABSENCE, SlotStatus.PRESERVED_PRESENT)
    passes = {
        "numeric": num_cmp.status in _preserved,
        "temporal": tmp_cmp.status in _preserved,
        "polarity": pol_cmp.status in _preserved,
        "anchor": anch_cmp.status in _preserved,
    }
    strict_pass = all(passes.values())

    # 7. High-fidelity pass (lenient: no contradictions, no numeric mutation,
    #    temporal/anchor tolerate OMISSION/ADDITION but not MUTATION)
    high_fidelity = (
        not pol_cmp.has_contradiction
        and num_cmp.status in _preserved
        and tmp_cmp.status != SlotStatus.MUTATION
        and anch_cmp.status != SlotStatus.MUTATION
    )

    # 8. Weighted score
    score = sum(DEFAULT_WEIGHTS[k] * (1.0 if v else 0.0) for k, v in passes.items())
    if pol_cmp.has_contradiction:
        score = max(0.0, score - CONTRADICTION_PENALTY)

    return {
        "numeric_status": num_cmp.status.value,
        "temporal_status": tmp_cmp.status.value,
        "polarity_status": pol_cmp.status.value,
        "anchor_status": anch_cmp.status.value,
        "numeric_pass": passes["numeric"],
        "temporal_pass": passes["temporal"],
        "polarity_pass": passes["polarity"],
        "anchor_pass": passes["anchor"],
        "strict_pair_pass": strict_pass,
        "high_fidelity_pass": high_fidelity,
        "pair_score": round(score, 4),
        "has_contradiction": pol_cmp.has_contradiction,
    }


def run_audit(
    pairs_df: pd.DataFrame,
    anchor_config: Dict[str, Any],
    event_config: Dict[str, Any],
    negation_window: int = 4,
    strict_threshold: float = 1.0,
    group_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run the full audit pipeline on a pairs DataFrame.

    Returns (pair_df, complaint_df, preserved_df, metrics_dict).
    """
    logger.info("Auditing %d pairs ...", len(pairs_df))
    t0 = time.time()

    result_rows: List[Dict[str, Any]] = []
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Auditing pairs"):
        res = audit_pair(
            source_text=str(row["source_text"]),
            rewrite_text=str(row["rewrite_text"]),
            anchor_config=anchor_config,
            event_config=event_config,
            negation_window=negation_window,
        )
        # Carry through identity columns
        res["complaint_id"] = row["complaint_id"]
        res["rewrite_id"] = row["rewrite_id"]
        for col in ("model_name", "persona_label", "issue_label"):
            if col in row.index:
                res[col] = row[col]
        result_rows.append(res)

    pair_df = pd.DataFrame(result_rows)
    elapsed = time.time() - t0
    logger.info("Audit completed in %.1fs", elapsed)

    # Aggregate: strict + high-fidelity
    complaint_df = aggregate_complaints(pair_df, strict_threshold)
    preserved_df = complaint_df[complaint_df["meets_criteria"]].copy()

    # Auto-detect fixture mode
    has_expected = all(c in pairs_df.columns for c in _EXPECTED_COLS)

    # Metrics (dual-mode)
    metrics = compute_metrics(pair_df, complaint_df, group_cols)

    if has_expected:
        logger.info("Fixture labels detected — running correctness checks")
        fixture_acc = compute_fixture_accuracy(pair_df, pairs_df)
        metrics["fixture_accuracy"] = fixture_acc

    return pair_df, complaint_df, preserved_df, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Matched-guise slot-preservation audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input source (mutually exclusive)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input", type=str, help="Path to pairs CSV/Parquet")
    inp.add_argument(
        "--from-cache", type=str, metavar="RUN_DIR",
        help="Reconstruct pairs from experiment cache (provide run output dir)",
    )

    p.add_argument("--output-dir", type=str, default="./slot_audit_out",
                    help="Output directory (default: ./slot_audit_out)")
    p.add_argument("--cache-dir", type=str, default="./cache",
                    help="LLM response cache directory (for --from-cache)")
    p.add_argument("--data-path", type=str, default="./data/complaints.csv",
                    help="Fallback complaints CSV (for --from-cache)")
    p.add_argument("--models", type=str, nargs="*", default=None,
                    help="Model names to include (for --from-cache)")
    p.add_argument("--issue-anchor-config", type=str, default=None,
                    help="Path to issue_anchors.yaml")
    p.add_argument("--event-lexicon-config", type=str, default=None,
                    help="Path to event_lexicon.yaml")
    p.add_argument("--negation-window", type=int, default=4,
                    help="Token lookback window for negation (default: 4)")
    p.add_argument("--strict-threshold", type=float, default=1.0,
                    help="Strict pass-rate threshold for complaint preservation (default: 1.0)")
    p.add_argument("--group-cols", type=str, nargs="*", default=None,
                    help="Columns for subgroup reporting")
    p.add_argument("--max-pairs", type=int, default=None,
                    help="Limit number of pairs (for quick testing)")
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Enable debug logging")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # --- Load input data ---
    if args.input:
        inp_path = Path(args.input)
        logger.info("Loading pairs from %s", inp_path)
        if inp_path.suffix == ".parquet":
            pairs_df = pd.read_parquet(inp_path)
        else:
            pairs_df = pd.read_csv(inp_path)
    else:
        from .cache_loader import load_pairs_from_cache
        logger.info("Reconstructing pairs from cache (run_dir=%s)", args.from_cache)
        pairs_df = load_pairs_from_cache(
            run_dir=Path(args.from_cache),
            cache_dir=Path(args.cache_dir),
            data_path=Path(args.data_path) if args.data_path else None,
            models=args.models,
            max_pairs=args.max_pairs,
        )

    if args.max_pairs:
        pairs_df = pairs_df.head(args.max_pairs)

    logger.info("Loaded %d pairs across %d complaints",
                len(pairs_df), pairs_df["complaint_id"].nunique())

    # --- Load configs ---
    anchor_config = load_anchor_config(args.issue_anchor_config)
    event_config = load_event_config(args.event_lexicon_config)

    # --- Run audit ---
    pair_df, complaint_df, preserved_df, metrics = run_audit(
        pairs_df,
        anchor_config=anchor_config,
        event_config=event_config,
        negation_window=args.negation_window,
        strict_threshold=args.strict_threshold,
        group_cols=args.group_cols,
    )

    # --- Write outputs ---
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_pair_csv(pair_df, out / "pair_slot_audit.csv")
    write_complaint_csv(complaint_df, out / "complaint_slot_summary.csv")
    write_preserved_csv(complaint_df, out / "preserved_complaints.csv", args.strict_threshold)
    write_high_fidelity_csvs(pair_df, complaint_df, out)
    write_json_metrics(metrics, out / "slot_preservation_metrics.json")
    write_markdown_report(metrics, pair_df, out / "slot_preservation_report.md")

    # --- Console summary ---
    g = metrics["global"]
    hf = g["high_fidelity"]
    vg = g["viability_gate"]
    is_fixture = "fixture_accuracy" in metrics

    # Always print real-data stats
    gate = "N/A (fixture mode)" if is_fixture else ("PASS" if vg["pass"] else "FAIL")
    print(f"\n{'=' * 60}")
    print(f"Slot Preservation Audit — Viability Gate: {gate}")
    print(f"{'=' * 60}")
    print(f"  Pairs analysed:              {g['n_pairs']}")
    print(f"  Complaints:                  {g['n_complaints']}")
    print(f"  Strict pair pass rate:       {g['strict_pair_pass_rate']:.4f}")
    print(f"  High-fidelity pair rate:     {hf['pair_rate']:.4f}")
    print(f"  High-fidelity complaint rate:{hf['complaint_rate']:.4f}")
    print(f"  Polarity contradiction rate: {g['polarity_contradiction_rate']:.4f}")
    print(f"  Numeric mutation rate:       {g['numeric_mutation_rate']:.4f}")
    print(f"  Preserved complaints:        {len(preserved_df)}")
    print(f"  High-fidelity complaints:    {hf['n_complaints']}")

    # Additionally show fixture accuracy when expected columns detected
    if is_fixture:
        fa = metrics["fixture_accuracy"]
        print(f"\n{'=' * 60}")
        print(f"Fixture Correctness Check")
        print(f"{'=' * 60}")
        print(f"  Overall label accuracy:      {fa['overall_accuracy']:.4f}")
        for fam in ("numeric", "temporal", "polarity", "anchor"):
            print(f"    {fam:>10s} accuracy:      {fa['per_family'][fam]['accuracy']:.4f}")
        print(f"  Strict-pass accuracy:        {fa['strict_pass_accuracy']:.4f}")
        print(f"  Contradiction-flag accuracy: {fa['contradiction_accuracy']:.4f}")
        verdict = "PASS" if fa["overall_accuracy"] >= 0.95 else "FAIL"
        print(f"\n  Fixture verdict: {verdict} (>= 0.95 required)")

    print(f"\nOutputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
