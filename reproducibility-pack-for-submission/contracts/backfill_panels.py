#!/usr/bin/env python3
"""Backfill panel payloads from existing IPW robustness JSON files.

Usage:
    python contracts/backfill_panels.py --run-dir "artifacts/runs/20260209_175353__seed42__n20000__modelsgpt4omini-gpt5mini"

Or process all runs:
    python contracts/backfill_panels.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def ipw_to_panel_payload(
    ipw_result: Dict[str, Any],
    run_id: str,
    models: List[str],
) -> Dict[str, Any]:
    """Transform raw IPW results into panel schema format."""
    propensity_diag = ipw_result.get("propensity_diagnostics", {})
    n_m1 = propensity_diag.get("n_M1", 0)
    max_weight = propensity_diag.get("weight_range_M1", [0, 0])[1] if propensity_diag.get("weight_range_M1") else 0

    positivity_pass = max_weight < 20 if max_weight > 0 else False

    rows: List[Dict[str, Any]] = []
    ne_summary: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    dimensions = ["age", "geographic", "education", "socioeconomic"]

    for model in models:
        model_data = ipw_result.get(model, {})

        if model_data.get("skipped"):
            reason = model_data.get("reason", "unknown")
            ne_summary.append({
                "model": model,
                "reason_code": "INSUFFICIENT_CELL_N",
                "message": f"IPW analysis skipped: {reason}",
            })
            diagnostics.append({
                "model": model,
                "model_obs": 0,
                "positivity_pass": False,
                "max_weight": 0.0,
                "rows_emitted": 0,
                "ne_rows": 1,
            })
            continue

        model_dims = model_data.get("dimensions", {})
        n_obs = n_m1
        rows_emitted = 0
        ne_rows = 0

        for dim in dimensions:
            dim_data = model_dims.get(dim, {})
            weighted = dim_data.get("weighted", {})

            R = weighted.get("R")
            R_ci = weighted.get("R_ci", {})
            ci_lower = R_ci.get("ci_lower")
            ci_upper = R_ci.get("ci_upper")

            interaction = weighted.get("interaction", {})
            p_value = interaction.get("p_value")

            # Check if values are estimable (not NE markers)
            def is_estimable(v: Any) -> bool:
                if v is None:
                    return False
                if isinstance(v, dict) and v.get("__ne__"):
                    return False
                if isinstance(v, (int, float)):
                    return True
                return False

            if is_estimable(R) and is_estimable(ci_lower) and is_estimable(ci_upper):
                row = {
                    "model": model,
                    "dimension": dim,
                    "status": "ok",
                    "n_obs": n_obs,
                    "estimate": float(R),
                    "ci_low": float(ci_lower),
                    "ci_high": float(ci_upper),
                    "p_value": float(p_value) if is_estimable(p_value) else None,
                    "q_value": None,
                }
                rows.append(row)
                rows_emitted += 1
            else:
                reason_code = "ESTIMATION_FAILED"
                row = {
                    "model": model,
                    "dimension": dim,
                    "status": "ne",
                    "n_obs": n_obs,
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p_value": None,
                    "q_value": None,
                    "ne_reason_code": reason_code,
                }
                rows.append(row)
                ne_rows += 1

        diagnostics.append({
            "model": model,
            "model_obs": n_obs,
            "positivity_pass": positivity_pass,
            "max_weight": float(max_weight),
            "rows_emitted": rows_emitted,
            "ne_rows": ne_rows,
        })

    payload = {
        "panel_id": "panel12_ipw_robustness_monetary",
        "run_id": run_id,
        "severity_metric": "monetary_severity",
        "subset": "amount_present",
        "models": models,
        "diagnostics": diagnostics,
        "rows": rows,
    }

    if ne_summary:
        payload["ne_summary"] = ne_summary

    return payload


def process_run(run_dir: Path) -> bool:
    """Process a single run directory and generate panel payload."""
    ipw_path = run_dir / "analysis" / "stats" / "ipw_robustness__monetary__all__v1.json"

    if not ipw_path.exists():
        logger.warning("No IPW results found in %s", run_dir.name)
        return False

    # Load IPW results
    ipw_result = load_json(ipw_path)

    # Detect models from the IPW result keys
    models = [k for k in ipw_result.keys() if k.startswith("gpt-")]
    if not models:
        logger.warning("No model results found in %s", ipw_path)
        return False

    # Get run_id from manifest or directory name
    manifest_path = run_dir / "manifest" / "run_config.json"
    if manifest_path.exists():
        config = load_json(manifest_path)
        run_id = config.get("run_id", run_dir.name)
    else:
        run_id = run_dir.name

    # Generate panel payload
    panel_payload = ipw_to_panel_payload(ipw_result, run_id, models)

    # Write to panels directory
    panels_dir = run_dir / "analysis" / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    output_path = panels_dir / "panel12_ipw_robustness_monetary.json"
    output_path.write_text(json.dumps(panel_payload, indent=2), encoding="utf-8")

    logger.info("Generated %s", output_path.relative_to(run_dir.parent.parent))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill panel payloads from existing IPW results")
    parser.add_argument("--run-dir", type=Path, help="Single run directory to process")
    parser.add_argument("--all", action="store_true", help="Process all runs in artifacts/runs/")
    args = parser.parse_args()

    if not args.run_dir and not args.all:
        parser.print_help()
        return 1

    runs_processed = 0
    runs_failed = 0

    if args.all:
        artifacts_dir = Path("artifacts/runs")
        if not artifacts_dir.exists():
            logger.error("artifacts/runs/ directory not found")
            return 1

        for run_dir in sorted(artifacts_dir.iterdir()):
            if run_dir.is_dir():
                if process_run(run_dir):
                    runs_processed += 1
                else:
                    runs_failed += 1
    else:
        if args.run_dir.exists():
            if process_run(args.run_dir):
                runs_processed += 1
            else:
                runs_failed += 1
        else:
            logger.error("Run directory not found: %s", args.run_dir)
            return 1

    logger.info("Processed %d runs, %d failed", runs_processed, runs_failed)
    return 0 if runs_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
