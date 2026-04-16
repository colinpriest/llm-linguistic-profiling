"""Regenerate the robustness ladder figure for an existing run.

Reads endpoint_summary.csv from the robustness subset analysis and
re-runs FigureGenerator._fig_robustness_ladder_visibility, writing the
PDF/PNG/JSON/TXT outputs into the chosen run's analysis/figures/main
directory.

Usage:
    python regen_robustness_ladder_figure.py \
        --endpoint-csv output/robustness_subsets/endpoint_summary.csv \
        --run-dir artifacts/runs/<RUN_ID>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.figure_generator import FigureGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-csv",
        type=Path,
        default=Path("output/robustness_subsets/endpoint_summary.csv"),
        help="Path to endpoint_summary.csv produced by the robustness subset run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory to write the figure into "
             "(e.g. artifacts/runs/20260220_101746__...).",
    )
    args = parser.parse_args()

    if not args.endpoint_csv.exists():
        raise SystemExit(f"endpoint summary not found: {args.endpoint_csv}")
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")

    df = pd.read_csv(args.endpoint_csv)
    models = sorted(df["model_name"].unique().tolist())

    analysis_results = {
        "models": models,
        "robustness_subsets": {"output_dir": str(args.endpoint_csv.parent)},
    }

    fg = FigureGenerator(
        analysis_results=analysis_results,
        complaints=[],
        run_config={"run_id": args.run_dir.name, "seed": 42, "num_complaints": None},
        output_root=args.run_dir,
    )
    result = fg._fig_robustness_ladder_visibility()
    if result is None:
        raise SystemExit("figure generation returned None (check logs).")
    pdf_path, png_path = result
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
