#!/usr/bin/env python3
"""LLM Fairness in Financial Complaint Resolution – Main Entry Point.

Tests whether large language models exhibit discriminatory behavior when
processing financial complaints, following a 7-step methodology.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Fairness Experiment: detect linguistic profiling in complaint resolution",
    )
    parser.add_argument(
        "--num-complaints", type=int, default=5000,
        help="Number of base complaints to process (default: 5000)",
    )
    parser.add_argument(
        "--causal-sample-pct", type=float, default=0.10,
        help="Fraction of complaints for matched-guise causal test (default: 0.10)",
    )
    parser.add_argument(
        "--materiality-threshold", type=float, default=0.80,
        help="Threshold for 80%% rule / adverse impact ratio (default: 0.80)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", default=False,
        help="Clear all cached LLM responses before run",
    )
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o-mini", "gpt-5-mini"],
        help='Models to test (default: ["gpt-4o-mini", "gpt-5-mini"])',
    )
    parser.add_argument(
        "--threads", type=int, default=100,
        help="Number of concurrent LLM request threads (default: 100)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Directory for HTML report and artifacts (default: ./output)",
    )
    parser.add_argument(
        "--data-path", type=str, default="./data/complaints.csv",
        help="Path to CFPB complaints CSV (default: ./data/complaints.csv)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./cache",
        help="Directory for LLM response cache (default: ./cache)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-steps", type=str, default="",
        help='Comma-separated steps to skip, e.g. "4,5" (default: none)',
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable detailed logging",
    )
    return parser.parse_args()


def setup_logging(output_dir: str, verbose: bool) -> None:
    """Configure console and file logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(output_dir) / f"experiment_{timestamp}.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Silence noisy third-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Console handler: INFO+
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(console)

    # File handler: all levels
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or environment.", file=sys.stderr)
        sys.exit(1)

    # Validate data path
    if not Path(args.data_path).exists():
        print(f"ERROR: Data file not found at {args.data_path}", file=sys.stderr)
        sys.exit(1)

    # Parse skip steps
    skip_steps = set()
    if args.skip_steps.strip():
        try:
            skip_steps = {int(s.strip()) for s in args.skip_steps.split(",")}
        except ValueError:
            print(f"ERROR: Invalid --skip-steps value: {args.skip_steps!r}", file=sys.stderr)
            sys.exit(1)

    setup_logging(args.output_dir, args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("LLM Fairness Experiment v1.2")
    logger.info("Configuration:")
    logger.info("  Data path:    %s", args.data_path)
    logger.info("  Complaints:   %d", args.num_complaints)
    logger.info("  Models:       %s", args.models)
    logger.info("  Threads:      %d", args.threads)
    logger.info("  Seed:         %d", args.seed)
    logger.info("  Output:       %s", args.output_dir)
    logger.info("  Cache:        %s", args.cache_dir)
    logger.info("  Skip steps:   %s", skip_steps or "none")

    # Run experiment
    from src.experiment import Experiment

    experiment = Experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        num_complaints=args.num_complaints,
        causal_sample_pct=args.causal_sample_pct,
        materiality_threshold=args.materiality_threshold,
        models=args.models,
        threads=args.threads,
        seed=args.seed,
        skip_steps=skip_steps,
        clear_cache=args.clear_cache,
        verbose=args.verbose,
        api_key=api_key,
    )

    analysis_results = experiment.run()

    # Generate HTML report
    from src.reporter import generate_report

    report_path = generate_report(
        analysis_results=analysis_results,
        text_features_df=experiment.text_features_df,
        complaints=experiment.complaints,
        output_path=str(Path(args.output_dir) / f"fairness_report_{experiment.timestamp}.html"),
        models=args.models,
        num_complaints=args.num_complaints,
        seed=args.seed,
        materiality_threshold=args.materiality_threshold,
        timestamp=experiment.timestamp,
    )
    logger.info("HTML report generated: %s", report_path)

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("Report: %s", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
