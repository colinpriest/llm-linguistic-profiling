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
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def _log_token_usage(llm: object, log: logging.Logger) -> None:
    """Log per-model and total token usage and cost to the console."""
    summary = llm.usage_summary()  # type: ignore[attr-defined]
    totals = summary["totals"]
    log.info("Token usage:")
    for model, stats in summary["per_model"].items():
        cost_str = (
            f"${stats['estimated_cost_usd']:.4f}"
            if stats.get("estimated_cost_usd") is not None
            else "unknown"
        )
        log.info(
            "  %s: %s requests, %s prompt + %s completion = %s tokens, cost: %s",
            model,
            f"{stats['requests']:,}",
            f"{stats['prompt_tokens']:,}",
            f"{stats['completion_tokens']:,}",
            f"{stats['prompt_tokens'] + stats['completion_tokens']:,}",
            cost_str,
        )
    total_cost_str = (
        f"${totals['estimated_cost_usd']:.4f}"
        if totals.get("estimated_cost_usd") is not None
        else "unknown"
    )
    log.info(
        "  TOTAL: %s requests, %s prompt + %s completion = %s tokens, cost: %s",
        f"{totals['requests']:,}",
        f"{totals['prompt_tokens']:,}",
        f"{totals['completion_tokens']:,}",
        f"{totals['total_tokens']:,}",
        total_cost_str,
    )


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
        "--clear-analysis-cache", action="store_true", default=False,
        help="Clear only severity/Bayesian analysis cache (not LLM cache)",
    )
    parser.add_argument(
        "--clear-bayes-cache", action="store_true", default=False,
        help="Clear only Bayesian model cache (retain severity amplification cache)",
    )
    parser.add_argument(
        "--clear-exec-cache", action="store_true", default=False,
        help="Clear only executive summary cache (regenerate summaries without re-running LLM calls)",
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


class _CountingHandler(logging.Handler):
    """Silent handler that tallies WARNING and ERROR records."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.warnings = 0
        self.errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.errors += 1
        elif record.levelno >= logging.WARNING:
            self.warnings += 1


def setup_logging(output_dir: str, verbose: bool, timestamp: str) -> _CountingHandler:
    """Configure console and file logging. Returns the counting handler."""
    log_path = Path(output_dir) / f"experiment_{timestamp}.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
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

    # Counting handler: tallies warnings/errors for end-of-run summary
    counter = _CountingHandler()
    root.addHandler(counter)

    return counter


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

    # Single timestamp (UTC) shared by log file and all output artifacts
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_counter = setup_logging(args.output_dir, args.verbose, run_timestamp)
    logger = logging.getLogger(__name__)

    t_start = time.monotonic()
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
        clear_analysis_cache=args.clear_analysis_cache,
        clear_bayes_cache=args.clear_bayes_cache,
        clear_exec_cache=args.clear_exec_cache,
        verbose=args.verbose,
        api_key=api_key,
        timestamp=run_timestamp,
    )

    try:
        analysis_results = experiment.run()
    except Exception as e:
        logger.error("FATAL: %s", e, exc_info=True)
        # Always report token usage before exiting on failure
        try:
            _log_token_usage(experiment.llm, logger)
            experiment.llm.write_usage_log(
                output_dir=Path("token-usage"),
                run_metadata={
                    "timestamp": experiment.timestamp,
                    "models": args.models,
                    "num_complaints": args.num_complaints,
                    "threads": args.threads,
                    "seed": args.seed,
                    "data_path": args.data_path,
                    "cache_dir": args.cache_dir,
                    "status": "failed",
                    "error": str(e),
                },
            )
        except Exception:
            pass  # Don't mask the original error
        sys.exit(1)

    # Generate HTML report
    logger.info("Generating HTML dashboard report...")
    report_path = None
    try:
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
    except Exception as e:
        logger.warning("Report generation failed (results still saved): %s", e)

    usage_log_path = experiment.llm.write_usage_log(
        output_dir=Path("token-usage"),
        run_metadata={
            "timestamp": experiment.timestamp,
            "models": args.models,
            "num_complaints": args.num_complaints,
            "threads": args.threads,
            "seed": args.seed,
            "data_path": args.data_path,
            "cache_dir": args.cache_dir,
        },
    )
    logger.info("Token usage log written: %s", usage_log_path)

    # Generate worked examples report
    logger.info("Generating worked examples report...")
    from src.worked_examples import generate_worked_examples

    examples_html, examples_json = generate_worked_examples(
        complaints=experiment.complaints,
        persona_estimates=experiment.persona_estimates,
        baseline_decisions=experiment.baseline_decisions,
        persona_rewrites=experiment.persona_rewrites,
        persona_decisions=experiment.persona_decisions,
        neutralized=experiment.neutralized,
        neutralized_decisions=experiment.neutralized_decisions,
        rewrite_persona_estimates=getattr(experiment, "rewrite_persona_estimates", []),
        neutralized_persona_estimates=getattr(experiment, "neutralized_persona_estimates", []),
        models=args.models,
        output_dir=args.output_dir,
        timestamp=experiment.timestamp,
    )
    if examples_html:
        logger.info("Worked examples HTML: %s", examples_html)
        logger.info("Worked examples JSON: %s", examples_json)

    # Generate paper bundle AFTER reporter + worked-examples have written files
    experiment.generate_paper_bundle()

    # Summary
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    if report_path:
        logger.info("Report: %s", report_path)
    if examples_html:
        logger.info("Worked examples: %s", examples_html)
    bundle_info = analysis_results.get("paper_bundle", {})
    if bundle_info and not bundle_info.get("error"):
        logger.info("Paper bundle: %s", bundle_info.get("output_root", ""))
    _log_token_usage(experiment.llm, logger)
    if log_counter.errors or log_counter.warnings:
        logger.info(
            "Issues during run: %d error(s), %d warning(s)",
            log_counter.errors, log_counter.warnings,
        )
    else:
        logger.info("Issues during run: none")
    elapsed = time.monotonic() - t_start
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        logger.info("Total time: %dh %dm %ds", hours, minutes, seconds)
    elif minutes:
        logger.info("Total time: %dm %ds", minutes, seconds)
    else:
        logger.info("Total time: %.1fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
