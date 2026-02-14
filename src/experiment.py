"""Step orchestration for the 7-step LLM fairness experiment."""

from __future__ import annotations

import json
import logging
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cache import ResponseCache
from src.data_loader import load_and_clean
from src.diagnostics import get_diagnostics, DiagnosticCollector, DiagnosticCategory
from src.llm_client import LLMClient
from src.mitigation_analysis import run_mitigation_effectiveness_analysis
from src.persona_analysis import run_persona_analysis
from src.text_rewriting_analysis import run_text_rewriting_analysis
from src.repeated_measures import run_repeated_measures_analysis
from src.severity_analysis import run_severity_analysis
from src.sparsity_analysis import run_sparsity_analysis
from src.process_discrimination import (
    info_request_content_analysis,
    process_causal_analysis,
    process_correlation_analysis,
    process_mitigation_analysis,
)
from src.prompts import (
    DOLLAR_EXTRACT_SYSTEM,
    DOLLAR_EXTRACT_USER,
    NEUTRALIZE_SYSTEM,
    NEUTRALIZE_USER,
    PERSONA_ESTIMATE_SYSTEM,
    PERSONA_ESTIMATE_USER,
    PERSONA_REWRITE_SYSTEM,
    REMEDY_DECISION_SYSTEM,
    REMEDY_DECISION_USER,
    TEXT_QUALITY_RATING_SYSTEM,
    TEXT_QUALITY_RATING_USER,
    format_persona_rewrite_user,
    safe_narrative_format,
)
from src.schemas import (
    CAUSAL_DIMS,
    Complaint,
    CommentaryResponse,
    DollarAmount,
    DollarAmountResponse,
    ESTIMATE_DIMS,
    LLMConfig,
    LLMRequest,
    LLMResponse,
    NeutralizedComplaint,
    NeutralizeResponse,
    PersonaEstimate,
    PersonaEstimateResponse,
    PersonaRewrite,
    PersonaRewriteResponse,
    RemedyDecision,
    RemedyDecisionResponse,
    TextQualityRating,
    TextQualityRatingResponse,
    PERSONA_GRID,
    PERSONA_LABELS,
    parse_persona_label,
    persona_label,
)
from src.statistics import (
    causal_analysis,
    chi_squared_test,
    correlation_analysis,
    mitigation_analysis,
)
from src.text_features import (
    extract_features_batch,
    features_to_dataframe,
    fit_topic_model,
    text_feature_cache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Console summary helpers
# ---------------------------------------------------------------------------

def _log_table(title: str, rows: List[Tuple[str, ...]], col_widths: Optional[List[int]] = None) -> None:
    """Pretty-print a small table to the logger."""
    logger.info("  ── %s ──", title)
    if not rows:
        logger.info("    (no data)")
        return
    if col_widths is None:
        n_cols = len(rows[0])
        col_widths = [max(len(str(r[i])) for r in rows) for i in range(n_cols)]
    for row in rows:
        parts = [str(v).ljust(w) for v, w in zip(row, col_widths)]
        logger.info("    %s", "  ".join(parts))


def _log_value_counts(title: str, series: pd.Series) -> None:
    """Log a value_counts as a simple table."""
    vc = series.value_counts().sort_index()
    rows = [(str(k), str(v)) for k, v in vc.items()]
    _log_table(title, rows)


def _log_mean_by_dimension(df: pd.DataFrame, value_col: str, dims: List[str]) -> None:
    """Log mean of value_col grouped by each dimension."""
    for dim in dims:
        means = df.groupby(dim)[value_col].mean()
        rows = [(str(k), f"{v:.3f}") for k, v in sorted(means.items())]
        _log_table(f"Mean {value_col} by {dim}", rows)


class Experiment:
    """Orchestrates the full 7-step experiment."""

    def __init__(
        self,
        data_path: str = "./data/complaints.csv",
        output_dir: str = "./output",
        cache_dir: str = "./cache",
        num_complaints: int = 5000,
        causal_sample_pct: float = 0.10,
        materiality_threshold: float = 0.80,
        models: Optional[List[str]] = None,
        threads: int = 100,
        seed: int = 42,
        skip_steps: Optional[Set[int]] = None,
        clear_cache: bool = False,
        clear_analysis_cache: bool = False,
        clear_bayes_cache: bool = False,
        clear_exec_cache: bool = False,
        verbose: bool = False,
        api_key: str = "",
    ) -> None:
        if num_complaints <= 0:
            raise ValueError("num_complaints must be a positive integer")
        if threads <= 0:
            raise ValueError("threads must be a positive integer")
        if not (0 < causal_sample_pct <= 1.0):
            raise ValueError("causal_sample_pct must be within (0, 1]")
        if not (0 < materiality_threshold <= 1.0):
            raise ValueError("materiality_threshold must be within (0, 1]")
        if models is not None and not models:
            raise ValueError("models must include at least one model identifier")

        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_complaints = num_complaints
        self.causal_sample_pct = causal_sample_pct
        self.materiality_threshold = materiality_threshold
        self.models = models or ["gpt-4o-mini", "gpt-5-mini"]
        self.threads = threads
        self.seed = seed
        self.skip_steps = skip_steps or set()
        self.verbose = verbose

        # Cache
        self.cache_dir = cache_dir
        self.cache = ResponseCache(cache_dir)
        text_feature_cache.configure(cache_dir)
        if clear_cache:
            self.cache.clear()
            cleared_tf = text_feature_cache.clear()
            if cleared_tf:
                logger.info("Text-feature cache cleared: %d files removed", cleared_tf)
            # Also clear the data pipeline cache
            pipeline_dir = Path(data_path).parent / ".pipeline"
            if pipeline_dir.exists():
                shutil.rmtree(pipeline_dir)
                logger.info("Pipeline cache cleared: %s", pipeline_dir)
            # Clear analysis cache (severity, Bayesian)
            from src.analysis_cache import get_analysis_cache
            analysis_cache = get_analysis_cache(cache_dir)
            cleared_analysis = analysis_cache.clear()
            if cleared_analysis:
                logger.info("Analysis cache cleared: %d files removed", cleared_analysis)
        elif clear_analysis_cache:
            # Clear only analysis cache (severity, Bayesian) - not LLM responses
            from src.analysis_cache import get_analysis_cache
            analysis_cache = get_analysis_cache(cache_dir)
            cleared_analysis = analysis_cache.clear()
            if cleared_analysis:
                logger.info("Analysis cache cleared: %d files removed", cleared_analysis)
            else:
                logger.info("Analysis cache was already empty")
        elif clear_bayes_cache:
            # Clear only Bayesian model cache; retain severity amplification cache
            from src.analysis_cache import get_analysis_cache
            analysis_cache = get_analysis_cache(cache_dir)
            cleared = analysis_cache.clear(analysis_type="bayes_robustness")
            if cleared:
                logger.info("Bayesian model cache cleared: %d files removed", cleared)
            else:
                logger.info("Bayesian model cache was already empty")

        # Clear executive summary cache (separate from main LLM cache)
        if clear_exec_cache:
            exec_cache_dir = Path(cache_dir) / "executive_summary"
            if exec_cache_dir.exists():
                count = 0
                for f in exec_cache_dir.iterdir():
                    if f.is_file():
                        f.unlink()
                        count += 1
                logger.info("Executive summary cache cleared: %d files removed", count)
            else:
                logger.info("Executive summary cache directory does not exist (nothing to clear)")

        # LLM client
        config = LLMConfig(
            api_key=api_key,
            models=self.models,
            temperature=0.0,
            max_tokens=16384,
            timeout=120,
            max_retries=3,
            retry_delay=1.0,
        )
        self.llm = LLMClient(config, self.cache, max_workers=threads)

        # Results storage
        self.complaints: List[Complaint] = []
        self.dollar_amounts: List[DollarAmount] = []
        self.persona_estimates: List[PersonaEstimate] = []
        self.baseline_decisions: List[RemedyDecision] = []
        self.persona_rewrites: List[PersonaRewrite] = []
        self.persona_decisions: List[RemedyDecision] = []
        self.neutralized: List[NeutralizedComplaint] = []
        self.neutralized_decisions: List[RemedyDecision] = []
        self.causal_sample: List[Complaint] = []
        self.rewrite_persona_estimates: List[Tuple[str, str, PersonaEstimate]] = []
        self.neutralized_persona_estimates: List[PersonaEstimate] = []

        # Debiased extraction data: model -> case_id -> FactSet/TriageDecision
        self.debiased_orig_fs: Dict[str, Dict[str, Any]] = {}
        self.debiased_orig_dec: Dict[str, Dict[str, Any]] = {}
        self.debiased_rw_fs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.debiased_rw_dec: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.debiased_rewrite_texts: Dict[str, Dict[str, Optional[str]]] = {}

        # LLM-rated text quality ratings
        self.text_quality_ratings: List[TextQualityRating] = []

        # DataFrames built during analysis
        self.personas_df: Optional[pd.DataFrame] = None
        self.baseline_df: Optional[pd.DataFrame] = None
        self.persona_decisions_df: Optional[pd.DataFrame] = None
        self.neutralized_df: Optional[pd.DataFrame] = None
        self.text_features_df: Optional[pd.DataFrame] = None
        self.text_quality_df: Optional[pd.DataFrame] = None

        # Analysis results
        self.analysis_results: Dict[str, Any] = {}

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment pipeline."""
        # Reset diagnostics collector for this run
        DiagnosticCollector.reset()

        logger.info("=" * 60)
        logger.info("LLM Fairness Experiment – starting")
        logger.info("Models: %s", self.models)
        logger.info("Complaints: %d, Causal sample: %.0f%%", self.num_complaints, self.causal_sample_pct * 100)
        logger.info("Seed: %d, Threads: %d", self.seed, self.threads)
        logger.info("Skip steps: %s", self.skip_steps or "none")
        logger.info("=" * 60)

        # Load data
        self.complaints = load_and_clean(self.data_path, self.num_complaints, self.seed)

        # Fit topic model on original narratives
        logger.info("Fitting topic model on original narratives...")
        fit_topic_model([c.narrative for c in self.complaints], n_topics=10, seed=self.seed, cache_dir=self.cache_dir)

        # Extract dollar amounts (cached, runs once)
        logger.info("-" * 40)
        logger.info("Dollar amount extraction starting")
        self._extract_dollar_amounts()
        logger.info("Dollar amount extraction complete")

        # Run steps
        for step in range(1, 9):
            if step in self.skip_steps:
                logger.info("Skipping Step %d", step)
                continue
            deps = self._STEP_DEPENDENCIES.get(step, set())
            skipped_deps = deps & self.skip_steps
            if skipped_deps:
                logger.warning(
                    "Step %d depends on skipped step(s) %s; results may be incomplete",
                    step, sorted(skipped_deps),
                )
            logger.info("-" * 40)
            logger.info("Step %d starting", step)
            getattr(self, f"_step_{step}")()
            logger.info("Step %d complete", step)

        # Build analysis
        self._run_analyses()

        # Save artifacts
        self._save_artifacts()

        # Generate paper bundle
        self._generate_paper_bundle()

        # Collect all diagnostics for the report
        self.analysis_results["diagnostics"] = get_diagnostics().to_dict()

        logger.info("=" * 60)
        logger.info("Experiment complete. Output: %s", self.output_dir)
        logger.info(self.cache.stats_summary())
        tf_stats = text_feature_cache.stats_summary()
        if tf_stats:
            logger.info(tf_stats)

        # Log diagnostics summary
        diag_summary = self.analysis_results["diagnostics"]
        if diag_summary.get("total_count", 0) > 0:
            logger.info(
                "Diagnostics: %d errors, %d warnings, %d with fallback",
                diag_summary.get("level_counts", {}).get("error", 0),
                diag_summary.get("level_counts", {}).get("warning", 0),
                diag_summary.get("with_fallback", 0),
            )

        return self.analysis_results

    def _generate_paper_bundle(self) -> None:
        """Generate the paper-ready output bundle."""
        logger.info("Generating paper bundle (tables, figures, evidence — may take a few minutes)...")
        try:
            from src.paper_bundle import PaperBundleGenerator

            run_config = {
                "seed": self.seed,
                "num_complaints": self.num_complaints,
                "models": self.models,
                "causal_sample_pct": self.causal_sample_pct,
                "materiality_threshold": self.materiality_threshold,
                "data_path": str(self.data_path),
                "threads": self.threads,
                "skip_steps": list(self.skip_steps),
                "timestamp": self.timestamp,
                "output_dir": str(self.output_dir),
            }

            bundle = PaperBundleGenerator(
                analysis_results=self.analysis_results,
                complaints=self.complaints,
                run_config=run_config,
                output_root=Path("./artifacts/runs"),
            )
            result = bundle.generate()
            self.analysis_results["paper_bundle"] = result

            if result.get("acceptance_missing"):
                logger.warning(
                    "Paper bundle acceptance: %d items missing",
                    len(result["acceptance_missing"]),
                )
        except Exception as e:
            logger.error("Paper bundle generation failed: %s", e, exc_info=True)
            self.analysis_results["paper_bundle"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Batch result validation
    # ------------------------------------------------------------------

    # Step dependency map: step -> set of prerequisite steps
    _STEP_DEPENDENCIES: Dict[int, Set[int]] = {
        3: {1, 2},  # Step 3 needs personas (1) and baseline decisions (2)
        5: {4},     # Step 5 needs persona rewrites (4)
        7: {6},     # Step 7 needs neutralized text (6)
        8: {4},     # Step 8 needs rewrites from cache (4)
    }

    # Fixed model for dollar-amount extraction; gpt-4o-mini is used because
    # extraction is a simple structured task that doesn't need a larger model,
    # and results are cached across all experimental conditions.
    DOLLAR_EXTRACT_MODEL = "gpt-4o-mini"

    # Model used for generating interpretive commentary paragraphs
    COMMENTARY_MODEL = "gpt-5.2"

    # Maximum number of retry rounds for timeout failures
    RETRY_ROUNDS = 3
    # Timeout multiplier per retry round (2×, 4×, 8× base timeout)
    RETRY_TIMEOUT_MULTIPLIER = 2

    def _check_batch_results(
        self,
        responses: List[Any],
        requests: List[Any],
        narratives: List[str],
        step_desc: str,
        max_failure_rate: float = 0.01,
    ) -> List[Any]:
        """Retry timeout failures with escalating timeouts, then raise if
        the final failure rate still exceeds *max_failure_rate*.

        Returns the (possibly updated) responses list.
        """
        # Build index so we can update responses in-place
        resp_by_idx = dict(enumerate(responses))
        req_by_idx = dict(enumerate(requests))
        narr_by_idx = dict(enumerate(narratives))

        base_timeout = self.llm.config.timeout

        for retry_round in range(1, self.RETRY_ROUNDS + 1):
            # Identify timeout failures
            timeout_indices = [
                i for i, resp in resp_by_idx.items()
                if not resp.success and resp.error_message and "timed out" in resp.error_message.lower()
            ]
            if not timeout_indices:
                break

            retry_timeout = base_timeout * (self.RETRY_TIMEOUT_MULTIPLIER ** retry_round)
            logger.info(
                "%s: retrying %d timeout failure(s) (round %d/%d, timeout=%ds)",
                step_desc, len(timeout_indices), retry_round, self.RETRY_ROUNDS, retry_timeout,
            )

            retry_requests = [req_by_idx[i] for i in timeout_indices]
            retry_narratives = [narr_by_idx[i] for i in timeout_indices]

            retry_responses = self._retry_with_timeout(
                retry_requests, retry_narratives, retry_timeout,
                desc=f"{step_desc} retry {retry_round}",
            )

            # Merge successful retries back
            recovered = 0
            for idx, retry_resp in zip(timeout_indices, retry_responses):
                if retry_resp.success:
                    resp_by_idx[idx] = retry_resp
                    recovered += 1
                else:
                    resp_by_idx[idx] = retry_resp  # update with latest error

            logger.info(
                "%s: retry round %d recovered %d/%d",
                step_desc, retry_round, recovered, len(timeout_indices),
            )
            if recovered == len(timeout_indices):
                break  # all recovered

        # Rebuild final response list
        final_responses = [resp_by_idx[i] for i in range(len(responses))]

        # Check final failure rate
        failed_indices = [
            i for i, resp in enumerate(final_responses)
            if not resp.success
        ]
        if not failed_indices:
            return final_responses

        n_total = len(requests)
        n_failed = len(failed_indices)
        failure_rate = n_failed / n_total if n_total > 0 else 0.0

        # Group failures by case_id, include narrative preview for diagnosis
        by_case: Dict[str, List[Dict[str, str]]] = {}
        for i in failed_indices:
            req = req_by_idx[i]
            resp = final_responses[i]
            narr = narr_by_idx[i]
            prompt_chars = len(req.prompt_system) + len(req.prompt_user)
            est_prompt_tokens = prompt_chars // 4  # rough ~4 chars/token
            by_case.setdefault(req.case_id, []).append({
                "error": resp.error_message or "unknown error",
                "variant": getattr(req, "variant", ""),
                "narrative_preview": narr[:200].replace("\n", " "),
                "narrative_len": str(len(narr)),
                "prompt_chars": str(prompt_chars),
                "est_prompt_tokens": str(est_prompt_tokens),
                "max_tokens": str(self.llm.config.max_tokens),
            })

        summary_lines = [
            f"{step_desc}: {n_failed} / {n_total} requests failed ({failure_rate:.2%})"
        ]
        for case_id, details in sorted(by_case.items()):
            # Truncate each error to first line (avoids verbose dumps)
            unique_errors = sorted(set(
                d["error"].split("\n", 1)[0][:200] for d in details
            ))
            variants = sorted(set(d["variant"] for d in details if d["variant"]))
            narr_len = details[0]["narrative_len"]
            prompt_chars = details[0]["prompt_chars"]
            est_tokens = details[0]["est_prompt_tokens"]
            max_tok = details[0]["max_tokens"]
            preview = details[0]["narrative_preview"]
            summary_lines.append(
                f"  case_id={case_id}: {len(details)} failure(s), "
                f"narrative_len={narr_len}, "
                f"prompt_chars={prompt_chars} (~{est_tokens} tokens), "
                f"max_tokens={max_tok}"
            )
            if variants:
                summary_lines.append(f"    variants: {', '.join(variants[:5])}")
            summary_lines.append(f"    errors: {'; '.join(unique_errors)}")
            summary_lines.append(f"    narrative: {preview}...")
        summary = "\n".join(summary_lines)

        if failure_rate > max_failure_rate:
            logger.error(summary)
            raise RuntimeError(summary)
        else:
            logger.warning(summary)

        return final_responses

    def _retry_with_timeout(
        self,
        requests: List[Any],
        narratives: List[str],
        timeout: int,
        desc: str,
    ) -> List[Any]:
        """Retry a small set of requests sequentially with a custom timeout.

        Uses ``max_attempts=1`` so each request gets exactly one shot at the
        escalated timeout — no internal retries that would multiply the wait
        (e.g. 3 attempts × 240s = 720s).
        """
        responses = []
        for req, narr in zip(requests, narratives):
            resp = self.llm.call_with_cache(req, narr, timeout_override=timeout, max_attempts=1)
            responses.append(resp)
        return responses

    # ------------------------------------------------------------------
    # Dollar Amount Extraction (pre-step, cached independently)
    # ------------------------------------------------------------------

    def _extract_dollar_amounts(self) -> None:
        """Extract dollar amounts from complaint narratives using gpt-4o-mini.

        Results are cached under the 'dollar_extract' task so they persist
        across experimental conditions and reruns.  Only ``--clear-cache``
        will invalidate them.
        """
        model = self.DOLLAR_EXTRACT_MODEL
        requests: List[LLMRequest] = []
        narratives: List[str] = []

        for c in self.complaints:
            req = LLMRequest(
                request_id=LLMRequest.new_id(),
                case_id=c.case_id,
                model=model,
                task="dollar_extract",
                prompt_system=DOLLAR_EXTRACT_SYSTEM,
                prompt_user=safe_narrative_format(DOLLAR_EXTRACT_USER, c.narrative),
                expected_schema=DollarAmountResponse,
            )
            requests.append(req)
            narratives.append(c.narrative)

        desc = f"Dollar amount extraction [{model}]"
        responses = self.llm.execute_batch(requests, narratives, desc=desc)
        responses = self._check_batch_results(responses, requests, narratives, desc)

        for resp in responses:
            if resp.success and resp.response_parsed:
                p = resp.response_parsed
                self.dollar_amounts.append(DollarAmount(
                    case_id=resp.case_id,
                    amounts_mentioned=p.amounts_mentioned,
                    primary_amount=p.primary_amount,
                    amount_type=p.amount_type,
                    confidence=p.confidence,
                ))
            else:
                self.dollar_amounts.append(DollarAmount(
                    case_id=resp.case_id,
                    amounts_mentioned=None,
                    primary_amount=None,
                    amount_type=None,
                    confidence=None,
                ))

        logger.info("Dollar extraction: %d complaints processed", len(self.dollar_amounts))

        # Assign monetary_severity to each complaint
        amount_by_case = {da.case_id: da.primary_amount for da in self.dollar_amounts}
        for c in self.complaints:
            amt = amount_by_case.get(c.case_id)
            if amt is None:
                c.monetary_severity = "unknown"
            elif amt <= 100:
                c.monetary_severity = "low"
            elif amt <= 1000:
                c.monetary_severity = "medium"
            else:
                c.monetary_severity = "high"

        self._summarize_dollar_amounts()

    def _summarize_dollar_amounts(self) -> None:
        """Console + dashboard summary of extracted dollar amounts."""
        if not self.dollar_amounts:
            return

        df = pd.DataFrame([
            {
                "case_id": da.case_id,
                "primary_amount": da.primary_amount,
                "amount_type": da.amount_type,
                "confidence": da.confidence,
                "n_amounts": len(da.amounts_mentioned) if da.amounts_mentioned else 0,
            }
            for da in self.dollar_amounts
        ])

        total = len(df)
        has_amount = df["primary_amount"].notna()
        n_with = int(has_amount.sum())
        n_without = total - n_with

        logger.info("── Dollar Amount Extraction Summary ──")
        _log_table("Coverage", [
            ("Total rows", str(total)),
            ("With amount", f"{n_with} ({n_with / total:.1%})"),
            ("Without amount", f"{n_without} ({n_without / total:.1%})"),
        ])

        # Store summary in analysis_results for dashboard
        summary: Dict[str, Any] = {
            "total": total,
            "with_amount": n_with,
            "without_amount": n_without,
            "pct_with_amount": round(n_with / total * 100, 1) if total else 0,
        }

        amounts = df.loc[has_amount, "primary_amount"]
        if len(amounts) > 0:
            stats_rows = [
                ("Count", str(len(amounts))),
                ("Min", f"${amounts.min():,.2f}"),
                ("Max", f"${amounts.max():,.2f}"),
                ("Mean", f"${amounts.mean():,.2f}"),
                ("Median", f"${amounts.median():,.2f}"),
                ("Std", f"${amounts.std():,.2f}"),
                ("10th pctl", f"${amounts.quantile(0.10):,.2f}"),
                ("25th pctl", f"${amounts.quantile(0.25):,.2f}"),
                ("75th pctl", f"${amounts.quantile(0.75):,.2f}"),
                ("90th pctl", f"${amounts.quantile(0.90):,.2f}"),
                ("<= $50", f"{(amounts <= 50).mean():.1%}"),
                ("<= $100", f"{(amounts <= 100).mean():.1%}"),
                ("<= $250", f"{(amounts <= 250).mean():.1%}"),
                ("<= $500", f"{(amounts <= 500).mean():.1%}"),
            ]
            _log_table("Primary Amount Statistics", stats_rows)

            summary["stats"] = {
                "min": round(float(amounts.min()), 2),
                "max": round(float(amounts.max()), 2),
                "mean": round(float(amounts.mean()), 2),
                "median": round(float(amounts.median()), 2),
                "std": round(float(amounts.std()), 2),
                "p10": round(float(amounts.quantile(0.10)), 2),
                "p25": round(float(amounts.quantile(0.25)), 2),
                "p75": round(float(amounts.quantile(0.75)), 2),
                "p90": round(float(amounts.quantile(0.90)), 2),
                "pct_le_50": round(float((amounts <= 50).mean()) * 100, 1),
                "pct_le_100": round(float((amounts <= 100).mean()) * 100, 1),
                "pct_le_250": round(float((amounts <= 250).mean()) * 100, 1),
                "pct_le_500": round(float((amounts <= 500).mean()) * 100, 1),
            }

            # Amount histogram (log-scaled bins)
            bin_edges = [0, 25, 50, 100, 250, 500, 1000, 5000, 10000, float("inf")]
            bin_labels = [
                "$0-25", "$25-50", "$50-100", "$100-250", "$250-500",
                "$500-1K", "$1K-5K", "$5K-10K", "$10K+",
            ]
            hist = pd.cut(amounts, bins=bin_edges, labels=bin_labels, right=False)
            hist_counts = hist.value_counts().sort_index()
            rows = [(str(k), str(v)) for k, v in hist_counts.items()]
            _log_table("Amount Histogram", rows)
            summary["amount_histogram"] = {str(k): int(v) for k, v in hist_counts.items()}

            # Confidence histogram
            conf_series = df.loc[has_amount, "confidence"]
            if conf_series.notna().any():
                conf_counts = conf_series.value_counts().sort_index()
                rows = [(str(k), str(v)) for k, v in conf_counts.items()]
                _log_table("Confidence Distribution", rows)
                summary["confidence_histogram"] = {str(k): int(v) for k, v in conf_counts.items()}

            # Amount type distribution
            type_series = df.loc[has_amount, "amount_type"]
            if type_series.notna().any():
                type_counts = type_series.value_counts().sort_index()
                rows = [(str(k), str(v)) for k, v in type_counts.items()]
                _log_table("Amount Type Distribution", rows)
                summary["amount_type_histogram"] = {str(k): int(v) for k, v in type_counts.items()}

        self.analysis_results["dollar_amounts"] = summary

        # --- Severity distribution summaries ---
        sev_df = pd.DataFrame([
            {
                "case_id": c.case_id,
                "monetary_severity": c.monetary_severity,
                "issue_severity": c.issue_severity,
            }
            for c in self.complaints
        ])

        logger.info("── Severity Distributions ──")
        for col in ("monetary_severity", "issue_severity"):
            vc = sev_df[col].value_counts()
            order = ["low", "medium", "high", "unknown"]
            vc = vc.reindex([o for o in order if o in vc.index])
            rows = [(str(k), f"{v} ({v / len(sev_df):.1%})") for k, v in vc.items()]
            _log_table(col, rows)

            # ASCII bar histogram
            max_bar = 40
            max_count = vc.max() if len(vc) > 0 else 1
            for label, count in vc.items():
                bar_len = int(count / max_count * max_bar) if max_count > 0 else 0
                bar = "\u2588" * bar_len
                logger.info("    %-8s %s %d", label, bar, count)

        self.analysis_results["monetary_severity"] = (
            sev_df["monetary_severity"].value_counts().to_dict()
        )
        self.analysis_results["issue_severity"] = (
            sev_df["issue_severity"].value_counts().to_dict()
        )

    # ------------------------------------------------------------------
    # Step 1: Persona Estimation
    # ------------------------------------------------------------------

    def _step_1(self) -> None:
        """Estimate persona for each complaint × model."""
        for model in self.models:
            requests = []
            narratives = []
            for c in self.complaints:
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=c.case_id,
                    model=model,
                    task="persona_estimate",
                    prompt_system=PERSONA_ESTIMATE_SYSTEM,
                    prompt_user=safe_narrative_format(PERSONA_ESTIMATE_USER, c.narrative),
                    expected_schema=PersonaEstimateResponse,
                )
                requests.append(req)
                narratives.append(c.narrative)

            desc = f"Step 1 [{model}] Persona estimation"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            for resp in responses:
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    self.persona_estimates.append(PersonaEstimate(
                        case_id=resp.case_id,
                        model=resp.model,
                        age=p.age,
                        geographic=p.geographic,
                        education=p.education,
                        socioeconomic=p.socioeconomic,
                        gender=p.gender,
                        esl=p.esl,
                        confidence=p.confidence,
                        linguistic_markers=p.linguistic_markers,
                        persona_label=persona_label(p.age, p.geographic, p.education, p.socioeconomic),
                    ))

        logger.info("Step 1: %d persona estimates collected", len(self.persona_estimates))
        self._summarize_step_1()

    def _summarize_step_1(self) -> None:
        """Console summary: persona dimension distributions + confidence histogram."""
        if not self.persona_estimates:
            return
        df = pd.DataFrame([
            {"age": pe.age, "geographic": pe.geographic, "education": pe.education,
             "socioeconomic": pe.socioeconomic, "gender": pe.gender, "esl": pe.esl,
             "confidence": pe.confidence,
             "model": pe.model, "persona_label": pe.persona_label}
            for pe in self.persona_estimates
        ])
        logger.info("── Step 1 Summary ──")
        for model in self.models:
            m = df[df["model"] == model]
            logger.info("  Model: %s (%d estimates)", model, len(m))
            for dim in ESTIMATE_DIMS:
                _log_value_counts(f"{dim} distribution", m[dim])
            # Full persona label counts (include zeros for missing personas)
            vc = m["persona_label"].value_counts()
            all_personas = vc.reindex(PERSONA_LABELS, fill_value=0).sort_index()
            rows = [(str(k), str(int(v))) for k, v in all_personas.items()]
            _log_table("Full persona type counts (%d/%d present)" % (int((all_personas > 0).sum()), len(PERSONA_LABELS)), rows)
            # Confidence histogram
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
            labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
            hist = pd.cut(m["confidence"], bins=bins, labels=labels, right=False).value_counts().sort_index()
            rows = [(str(k), str(v)) for k, v in hist.items()]
            _log_table("Confidence histogram", rows)

    # ------------------------------------------------------------------
    # Step 2: Remedy Decision (Original)
    # ------------------------------------------------------------------

    def _step_2(self) -> None:
        """Get remedy tier for each complaint × model (baseline)."""
        for model in self.models:
            requests, narratives = [], []
            for c in self.complaints:
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=c.case_id,
                    model=model,
                    task="remedy_decision",
                    prompt_system=REMEDY_DECISION_SYSTEM,
                    prompt_user=safe_narrative_format(REMEDY_DECISION_USER, c.narrative),
                    expected_schema=RemedyDecisionResponse,
                    variant="baseline",
                )
                requests.append(req)
                narratives.append(c.narrative)

            desc = f"Step 2 [{model}] Baseline remedy"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            for resp in responses:
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    self.baseline_decisions.append(RemedyDecision(
                        case_id=resp.case_id,
                        model=resp.model,
                        variant="baseline",
                        tier=p.tier,
                        confidence=p.confidence,
                        reasoning=p.reasoning,
                        information_needed=p.information_needed,
                    ))

        logger.info("Step 2: %d baseline decisions collected", len(self.baseline_decisions))
        self._summarize_step_2()

    def _summarize_step_2(self) -> None:
        """Console summary: tier + confidence category distribution + process discrimination rates."""
        if not self.baseline_decisions:
            return
        df = pd.DataFrame([
            {"tier": d.tier, "confidence": d.confidence, "model": d.model}
            for d in self.baseline_decisions
        ])
        logger.info("── Step 2 Summary ──")
        for model in self.models:
            m = df[df["model"] == model]
            logger.info("  Model: %s (%d decisions)", model, len(m))
            _log_value_counts("Tier distribution", m["tier"])
            _log_value_counts("Confidence categories", m["confidence"])
            # Process discrimination rate
            need_info = (m["confidence"] == "need_more_info").sum()
            info_rate = need_info / len(m) if len(m) > 0 else 0.0
            logger.info("  ── Process Discrimination Rate ──")
            logger.info("    Info request rate: %d/%d (%.3f)", need_info, len(m), info_rate)

    # ------------------------------------------------------------------
    # Text quality rating helper
    # ------------------------------------------------------------------

    def _rate_text_quality(
        self,
        case_ids: List[str],
        narratives: List[str],
        variant: str,
        models: Optional[List[str]] = None,
    ) -> None:
        """Rate text quality for a batch of narratives using each experiment model."""
        if not narratives:
            return

        rate_models = models or self.models

        for model in rate_models:
            requests: List[LLMRequest] = []
            narrs: List[str] = []

            for case_id, narrative in zip(case_ids, narratives):
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=case_id,
                    model=model,
                    task="text_quality_rating",
                    prompt_system=TEXT_QUALITY_RATING_SYSTEM,
                    prompt_user=safe_narrative_format(TEXT_QUALITY_RATING_USER, narrative),
                    expected_schema=TextQualityRatingResponse,
                    variant=variant,
                )
                requests.append(req)
                narrs.append(narrative)

            desc = f"Text quality [{model}] ({variant})"
            responses = self.llm.execute_batch(requests, narrs, desc=desc)
            responses = self._check_batch_results(responses, requests, narrs, desc)

            for resp in responses:
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    self.text_quality_ratings.append(TextQualityRating(
                        case_id=resp.case_id,
                        model=model,
                        variant=variant,
                        convincingness=p.convincingness,
                        persuasiveness=p.persuasiveness,
                        evidence_blame=p.evidence_blame,
                        evidence_harm=p.evidence_harm,
                        confidence=p.confidence,
                        legal_knowledge=p.legal_knowledge,
                        formality=p.formality,
                        coherence=p.coherence,
                        specificity=p.specificity,
                        financial_literacy=p.financial_literacy,
                        emotional_restraint=p.emotional_restraint,
                        grammar_quality=p.grammar_quality,
                    ))

        count = sum(1 for r in self.text_quality_ratings if r.variant == variant)
        logger.info(
            "Text quality rating (%s): %d ratings collected (%d models x %d texts)",
            variant, count, len(rate_models), len(narratives),
        )

    # ------------------------------------------------------------------
    # Step 3: Correlation Analysis (statistics only)
    # ------------------------------------------------------------------

    def _step_3(self) -> None:
        """Statistical correlation between persona estimates and remedy decisions."""
        self._build_dataframes()

        if self.personas_df is not None and self.baseline_df is not None:
            # Text features on original complaints
            logger.info("Extracting text features for original complaints...")
            orig_features = extract_features_batch(
                [c.narrative for c in self.complaints],
                [c.case_id for c in self.complaints],
                "original",
            )
            self.text_features_df = features_to_dataframe(orig_features)

            # LLM-rated text quality on original complaints
            logger.info("Rating text quality for original complaints...")
            self._rate_text_quality(
                case_ids=[c.case_id for c in self.complaints],
                narratives=[c.narrative for c in self.complaints],
                variant="original",
            )

        logger.info("Step 3: Correlation analysis will run after all steps complete")
        self._summarize_step_3()

    def _summarize_step_3(self) -> None:
        """Console summary: persona×tier crosstab + mean tier by dimension + per-model stats + cross-model commentary."""
        if self.personas_df is None or self.baseline_df is None:
            return
        logger.info("── Step 3 Summary ──")

        step3_stats_by_model: Dict[str, Dict[str, Any]] = {}

        for model in self.models:
            p_m = self.personas_df[self.personas_df["model"] == model]
            b_m = self.baseline_df[self.baseline_df["model"] == model]
            merged = b_m.merge(p_m, on=["case_id", "model"], how="inner")
            if merged.empty:
                continue
            logger.info("  Model: %s (%d matched records)", model, len(merged))

            # 2-way crosstab: persona_label × tier (top 10 by count)
            ct = pd.crosstab(merged["persona_label"], merged["tier"])
            top10 = ct.sum(axis=1).nlargest(10).index
            ct_top = ct.loc[top10]
            header = ("persona",) + tuple(f"tier_{c}" for c in ct_top.columns)
            rows = [header]
            for persona in ct_top.index:
                rows.append((persona,) + tuple(str(ct_top.loc[persona, c]) for c in ct_top.columns))
            _log_table("Top 10 persona × tier crosstab", rows)

            # 1-way: mean tier by dimension (all 6 dims)
            available_dims = [d for d in ESTIMATE_DIMS if d in merged.columns]
            _log_mean_by_dimension(merged, "tier", available_dims)

            # Process discrimination rate
            # After merge, baseline's confidence becomes confidence_x (persona's becomes confidence_y)
            conf_col = "confidence_x" if "confidence_x" in merged.columns else "confidence"
            need_info = (merged[conf_col] == "need_more_info").sum() if conf_col in merged.columns else 0
            info_rate = need_info / len(merged) if len(merged) > 0 else 0.0
            logger.info("  ── Process Discrimination Rate ──")
            logger.info("    Info request rate: %d/%d (%.3f)", need_info, len(merged), info_rate)

            # Per-model statistical tests for associations
            model_stats: Dict[str, Any] = {"n_records": len(merged)}

            # Chi-squared: persona_label × tier
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2_result = chi_squared_test(ct)
                model_stats["overall_chi2"] = chi2_result
                logger.info("  ── Association Tests ──")
                logger.info("    Overall persona×tier: chi2=%.2f, p=%.4f, V=%.3f",
                            chi2_result["chi2"], chi2_result["p_value"], chi2_result["cramers_v"])

            # Per-dimension chi2 and mean tier
            dim_stats = {}
            for dim in available_dims:
                ct_dim = pd.crosstab(merged[dim], merged["tier"])
                if ct_dim.shape[0] > 1 and ct_dim.shape[1] > 1:
                    dim_chi2 = chi_squared_test(ct_dim)
                    dim_stats[dim] = dim_chi2
                    logger.info("    %s×tier: chi2=%.2f, p=%.4f, V=%.3f",
                                dim, dim_chi2["chi2"], dim_chi2["p_value"], dim_chi2["cramers_v"])
                # Mean tier by level
                means = merged.groupby(dim)["tier"].mean().to_dict()
                model_stats[f"{dim}_mean_tier"] = means
            model_stats["dim_chi2"] = dim_stats

            # Unique persona count and entropy (for anchoring analysis)
            n_unique_personas = merged["persona_label"].nunique()
            persona_probs = merged["persona_label"].value_counts(normalize=True)
            persona_entropy = -(persona_probs * np.log2(persona_probs.clip(lower=1e-10))).sum()
            model_stats["n_unique_personas"] = n_unique_personas
            model_stats["persona_entropy"] = float(persona_entropy)
            logger.info("    Unique personas: %d, entropy: %.3f", n_unique_personas, persona_entropy)

            step3_stats_by_model[model] = model_stats

        # Cross-model GPT-5.2 commentary
        if step3_stats_by_model:
            self._step3_commentary(step3_stats_by_model)

    def _step3_commentary(self, step3_stats: Dict[str, Dict[str, Any]]) -> None:
        """Generate GPT-5.2 commentary comparing step 3 results across models."""
        lines = ["Step 3 Correlation Analysis Results by Model:\n"]
        for model, stats in step3_stats.items():
            lines.append(f"Model: {model}")
            lines.append(f"  Records: {stats['n_records']}")
            lines.append(f"  Unique personas: {stats.get('n_unique_personas', 'N/A')}")
            lines.append(f"  Persona entropy: {stats.get('persona_entropy', 'N/A'):.3f}")

            chi2 = stats.get("overall_chi2")
            if chi2:
                lines.append(f"  Overall persona×tier: chi2={chi2['chi2']:.2f}, p={chi2['p_value']:.4f}, V={chi2['cramers_v']:.3f}")

            dim_chi2 = stats.get("dim_chi2", {})
            for dim, result in sorted(dim_chi2.items()):
                lines.append(f"  {dim}×tier: chi2={result['chi2']:.2f}, p={result['p_value']:.4f}, V={result['cramers_v']:.3f}")

            for dim in ESTIMATE_DIMS:
                means = stats.get(f"{dim}_mean_tier")
                if means:
                    parts = [f"{k}={v:.3f}" for k, v in sorted(means.items())]
                    lines.append(f"  Mean tier by {dim}: {', '.join(parts)}")
            lines.append("")

        results_text = "\n".join(lines)

        user_prompt = (
            "Here are the numerical results. Write your analysis paragraph now.\n\n"
            + results_text
            + "\n\nNow write exactly one paragraph (4-8 sentences) stating your "
            "findings from the numbers above. Every sentence must contain at least "
            "one specific number from the data. For example: 'The overall persona×tier "
            "association was significant for ModelX (chi2=42.3, p=0.001, V=0.15) "
            "but not for ModelY (p=0.32).' Do NOT write a plan or outline."
        )

        system_prompt = (
            "You are a statistician writing a findings paragraph for a research report. "
            "You will receive chi-squared statistics, Cramer's V effect sizes, p-values, "
            "mean tier values, and persona diversity metrics from an LLM fairness experiment "
            "that tests whether LLMs give different complaint remediation recommendations "
            "based on estimated linguistic persona (age, geographic, education, socioeconomic, "
            "gender, ESL). Your commentary field must contain a single completed paragraph "
            "of 4-8 sentences reporting:\n"
            "(1) Whether the overall persona×tier chi2 is significant for each model (state the p-values).\n"
            "(2) Which dimensions have the largest Cramer's V and whether they reach p<0.05.\n"
            "(3) Which model has more unique personas / higher entropy (less anchoring).\n"
            "(4) Which model shows stronger persona-tier associations (compare V values).\n\n"
            "CRITICAL: Your output must read like a finished paragraph from a statistics "
            "report, e.g. 'The analysis revealed that...' NOT like a to-do list or plan "
            "such as 'Compute per-model significance...' or 'Will summarize...'"
        )

        req = LLMRequest(
            request_id=LLMRequest.new_id(),
            case_id="step3_commentary",
            model=self.COMMENTARY_MODEL,
            task="commentary",
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            expected_schema=CommentaryResponse,
            variant="step3_commentary",
        )

        try:
            resp = self.llm.call_with_cache(req, user_prompt)
            if resp.success and resp.response_parsed:
                commentary = resp.response_parsed.commentary
                logger.info("── Step 3 GPT-5.2 Cross-Model Commentary ──")
                logger.info("  %s", commentary)
                # Store for dashboard
                self.analysis_results["step3_commentary"] = commentary
            else:
                logger.warning("Step 3 commentary request failed: %s", resp.error_message)
        except Exception as e:
            logger.warning("Step 3 commentary request failed: %s", e)

    # ------------------------------------------------------------------
    # Step 4: Persona-Style Rewriting
    # ------------------------------------------------------------------

    def _step_4(self) -> None:
        """Rewrite a sample of complaints into each of 36 persona styles."""
        rng = random.Random(self.seed)
        n_sample = max(1, int(len(self.complaints) * self.causal_sample_pct))
        sample = rng.sample(self.complaints, n_sample)
        self.causal_sample = sample
        logger.info("Step 4: Rewriting %d complaints × 36 personas = %d rewrites", n_sample, n_sample * 36)

        for model in self.models:
            requests, narratives = [], []
            for c in sample:
                for persona in PERSONA_GRID:
                    plabel = persona_label(**persona)
                    user_prompt = format_persona_rewrite_user(
                        c.narrative,
                        persona["age"], persona["geo"], persona["edu"], persona["ses"],
                    )
                    req = LLMRequest(
                        request_id=LLMRequest.new_id(),
                        case_id=c.case_id,
                        model=model,
                        task="persona_rewrite",
                        prompt_system=PERSONA_REWRITE_SYSTEM,
                        prompt_user=user_prompt,
                        expected_schema=PersonaRewriteResponse,
                        variant=plabel,
                    )
                    requests.append(req)
                    narratives.append(user_prompt)

            desc = f"Step 4 [{model}] Persona rewriting"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            sample_lookup = {c.case_id: c for c in sample}
            for resp, req in zip(responses, requests):
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    orig = sample_lookup.get(resp.case_id)
                    if orig:
                        self.persona_rewrites.append(PersonaRewrite(
                            case_id=resp.case_id,
                            model=resp.model,
                            original_narrative=orig.narrative,
                            persona_label=req.variant,
                            rewritten_narrative=p.rewritten_complaint,
                            style_changes_made=p.style_changes_made,
                        ))

        logger.info("Step 4: %d persona rewrites collected", len(self.persona_rewrites))
        self._summarize_step_4()

    def _summarize_step_4(self) -> None:
        """Console summary: re-estimate persona on rewrites + match accuracy."""
        if not self.persona_rewrites:
            return
        logger.info("── Step 4 Summary: Persona re-estimation on rewrites ──")
        self.rewrite_persona_estimates = []  # (intended_label, model, estimate)

        for model in self.models:
            model_rewrites = [r for r in self.persona_rewrites if r.model == model]
            if not model_rewrites:
                continue
            requests, narratives = [], []
            for rw in model_rewrites:
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=rw.case_id,
                    model=model,
                    task="persona_estimate",
                    prompt_system=PERSONA_ESTIMATE_SYSTEM,
                    prompt_user=safe_narrative_format(PERSONA_ESTIMATE_USER, rw.rewritten_narrative),
                    expected_schema=PersonaEstimateResponse,
                    variant=rw.persona_label,
                )
                requests.append(req)
                narratives.append(rw.rewritten_narrative)

            desc = f"Step 4 [{model}] Re-estimate persona on rewrites"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            for resp, rw in zip(responses, model_rewrites):
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    est = PersonaEstimate(
                        case_id=resp.case_id, model=resp.model,
                        age=p.age, geographic=p.geographic, education=p.education,
                        socioeconomic=p.socioeconomic, gender=p.gender, esl=p.esl,
                        confidence=p.confidence,
                        linguistic_markers=p.linguistic_markers,
                        persona_label=persona_label(p.age, p.geographic, p.education, p.socioeconomic),
                    )
                    self.rewrite_persona_estimates.append((rw.persona_label, model, est))

            # Compute match accuracy per dimension
            matched = []
            for (lbl, m, est) in self.rewrite_persona_estimates:
                if m != model:
                    continue
                intended = parse_persona_label(lbl)
                if intended is not None:
                    matched.append((intended, est))

            if not matched:
                continue

            logger.info("  Model: %s (%d re-estimates)", model, len(matched))
            match_rates_model: Dict[str, float] = {}
            for dim in CAUSAL_DIMS:
                correct = sum(1 for intended, est in matched if getattr(est, dim) == intended[dim])
                rate = correct / len(matched) if matched else 0
                match_rates_model[dim] = rate
                logger.info("    %s match: %d/%d (%.1f%%)", dim, correct, len(matched), rate * 100)

            # Store match rates in analysis_results
            mc = self.analysis_results.setdefault("manipulation_checks", {})
            rw_rates = mc.setdefault("rewrite_match_rates", {})
            rw_rates[model] = match_rates_model

            # Estimated persona distribution on rewritten records
            est_df = pd.DataFrame([
                {"age": est.age, "geographic": est.geographic,
                 "education": est.education, "socioeconomic": est.socioeconomic,
                 "gender": est.gender, "esl": est.esl}
                for _, m_name, est in self.rewrite_persona_estimates
                if m_name == model
            ])
            if not est_df.empty:
                logger.info("  ── Estimated persona distribution on rewrites ──")
                for dim in ESTIMATE_DIMS:
                    if dim in est_df.columns:
                        _log_value_counts(f"Estimated {dim}", est_df[dim])

    # ------------------------------------------------------------------
    # Step 5: Remedy Decision (Persona-Rewritten)
    # ------------------------------------------------------------------

    def _step_5(self) -> None:
        """Remedy decisions on persona-rewritten complaints."""
        for model in self.models:
            model_rewrites = [r for r in self.persona_rewrites if r.model == model]
            requests, narratives = [], []
            for rw in model_rewrites:
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=rw.case_id,
                    model=model,
                    task="remedy_decision",
                    prompt_system=REMEDY_DECISION_SYSTEM,
                    prompt_user=safe_narrative_format(REMEDY_DECISION_USER, rw.rewritten_narrative),
                    expected_schema=RemedyDecisionResponse,
                    variant=rw.persona_label,
                )
                requests.append(req)
                narratives.append(rw.rewritten_narrative)

            desc = f"Step 5 [{model}] Persona remedy"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            for resp, req in zip(responses, requests):
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    self.persona_decisions.append(RemedyDecision(
                        case_id=resp.case_id,
                        model=resp.model,
                        variant=req.variant,
                        tier=p.tier,
                        confidence=p.confidence,
                        reasoning=p.reasoning,
                        information_needed=p.information_needed,
                    ))

        logger.info("Step 5: %d persona decisions collected", len(self.persona_decisions))
        self._summarize_step_5()

        # Text features on persona-rewritten complaints
        if self.persona_rewrites:
            logger.info("Extracting text features for persona-rewritten complaints...")
            rewrite_features = extract_features_batch(
                [r.rewritten_narrative for r in self.persona_rewrites],
                [r.case_id for r in self.persona_rewrites],
                # Use persona_label as variant for each rewrite
                "persona_rewrite",  # generic label; per-persona tracking in case_id
            )
            rw_df = features_to_dataframe(rewrite_features)
            # Add persona label column
            rw_df["persona_label"] = [r.persona_label for r in self.persona_rewrites]
            if self.text_features_df is not None:
                self.text_features_df = pd.concat([self.text_features_df, rw_df], ignore_index=True)
            else:
                self.text_features_df = rw_df

            # LLM-rated text quality on persona-rewritten complaints
            logger.info("Rating text quality for persona-rewritten complaints...")
            for model in self.models:
                model_rewrites = [r for r in self.persona_rewrites if r.model == model]
                if model_rewrites:
                    self._rate_text_quality(
                        case_ids=[r.case_id for r in model_rewrites],
                        narratives=[r.rewritten_narrative for r in model_rewrites],
                        variant="persona_rewrite",
                        models=[model],
                    )

    def _summarize_step_5(self) -> None:
        """Console summary: persona×tier crosstab + mean tier by dimension + process discrimination."""
        if not self.persona_decisions:
            return
        df = pd.DataFrame([
            {"variant": d.variant, "tier": d.tier, "confidence": d.confidence, "model": d.model}
            for d in self.persona_decisions
        ])
        logger.info("── Step 5 Summary ──")
        for model in self.models:
            m = df[df["model"] == model]
            if m.empty:
                continue
            logger.info("  Model: %s (%d persona decisions)", model, len(m))

            # Parse variant into dimensions (4 causal dims)
            parsed = m["variant"].apply(parse_persona_label)
            valid = parsed.notna()
            if valid.any():
                m = m.copy()
                parsed_df = pd.DataFrame(parsed[valid].tolist(), index=parsed[valid].index)
                for col in ["age", "geographic", "education", "socioeconomic"]:
                    if col in parsed_df.columns:
                        m.loc[parsed_df.index, col] = parsed_df[col]

                # 2-way crosstab: persona × tier (top 10)
                ct = pd.crosstab(m["variant"], m["tier"])
                top10 = ct.sum(axis=1).nlargest(10).index
                ct_top = ct.loc[top10]
                header = ("persona",) + tuple(f"tier_{c}" for c in ct_top.columns)
                rows = [header]
                for persona in ct_top.index:
                    rows.append((persona,) + tuple(str(ct_top.loc[persona, c]) for c in ct_top.columns))
                _log_table("Top 10 persona × tier crosstab", rows)

                # 1-way: mean tier by dimension (causal dims only for rewritten data)
                _log_mean_by_dimension(m, "tier", CAUSAL_DIMS)

            _log_value_counts("Overall tier distribution", m["tier"])

            # Process discrimination rate
            need_info = (m["confidence"] == "need_more_info").sum()
            info_rate = need_info / len(m) if len(m) > 0 else 0.0
            logger.info("  ── Process Discrimination Rate ──")
            logger.info("    Info request rate: %d/%d (%.3f)", need_info, len(m), info_rate)
            if valid.any():
                for dim in CAUSAL_DIMS:
                    if dim in m.columns:
                        dim_rates = m.groupby(dim)["confidence"].apply(
                            lambda s: (s == "need_more_info").mean()
                        )
                        parts_str = [f"{k}={v:.3f}" for k, v in sorted(dim_rates.items())]
                        logger.info("    Info rate by %s: %s", dim, ", ".join(parts_str))

    # ------------------------------------------------------------------
    # Step 6: Language Neutralization
    # ------------------------------------------------------------------

    def _step_6(self) -> None:
        """Neutralize all original complaints."""
        for model in self.models:
            requests, narratives = [], []
            for c in self.complaints:
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=c.case_id,
                    model=model,
                    task="neutralize",
                    prompt_system=NEUTRALIZE_SYSTEM,
                    prompt_user=safe_narrative_format(NEUTRALIZE_USER, c.narrative),
                    expected_schema=NeutralizeResponse,
                    variant="neutralized",
                )
                requests.append(req)
                narratives.append(c.narrative)

            desc = f"Step 6 [{model}] Neutralization"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            complaints_lookup = {c.case_id: c for c in self.complaints}
            for resp in responses:
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    orig = complaints_lookup.get(resp.case_id)
                    if orig:
                        self.neutralized.append(NeutralizedComplaint(
                            case_id=resp.case_id,
                            model=resp.model,
                            original_narrative=orig.narrative,
                            neutralized_narrative=p.neutralized_complaint,
                            markers_removed=p.markers_removed,
                        ))

        logger.info("Step 6: %d neutralized complaints collected", len(self.neutralized))
        self._summarize_step_6()

    def _summarize_step_6(self) -> None:
        """Console summary: re-estimate persona on neutralized + distribution comparison."""
        if not self.neutralized:
            return
        logger.info("── Step 6 Summary: Persona re-estimation on neutralized ──")
        self.neutralized_persona_estimates = []

        for model in self.models:
            model_neutral = [n for n in self.neutralized if n.model == model]
            if not model_neutral:
                continue
            requests, narratives = [], []
            for nc in model_neutral:
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=nc.case_id,
                    model=model,
                    task="persona_estimate",
                    prompt_system=PERSONA_ESTIMATE_SYSTEM,
                    prompt_user=safe_narrative_format(PERSONA_ESTIMATE_USER, nc.neutralized_narrative),
                    expected_schema=PersonaEstimateResponse,
                    variant="neutralized",
                )
                requests.append(req)
                narratives.append(nc.neutralized_narrative)

            desc = f"Step 6 [{model}] Re-estimate persona on neutralized"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            estimates_this_model = []
            for resp in responses:
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    est = PersonaEstimate(
                        case_id=resp.case_id, model=resp.model,
                        age=p.age, geographic=p.geographic, education=p.education,
                        socioeconomic=p.socioeconomic, gender=p.gender, esl=p.esl,
                        confidence=p.confidence,
                        linguistic_markers=p.linguistic_markers,
                        persona_label=persona_label(p.age, p.geographic, p.education, p.socioeconomic),
                    )
                    self.neutralized_persona_estimates.append(est)
                    estimates_this_model.append(est)

            if not estimates_this_model:
                continue

            # Log estimated persona distribution on neutralized text
            est_df = pd.DataFrame([
                {"age": e.age, "geographic": e.geographic, "education": e.education,
                 "socioeconomic": e.socioeconomic, "gender": e.gender, "esl": e.esl}
                for e in estimates_this_model
            ])
            logger.info("  Model: %s (%d neutralized re-estimates)", model, len(est_df))
            for dim in ESTIMATE_DIMS:
                _log_value_counts(f"Neutralized {dim} distribution", est_df[dim])

            # Compare to original persona estimates
            orig_ests = [pe for pe in self.persona_estimates if pe.model == model]
            if orig_ests:
                orig_df = pd.DataFrame([
                    {"age": e.age, "geographic": e.geographic, "education": e.education,
                     "socioeconomic": e.socioeconomic, "gender": e.gender, "esl": e.esl}
                    for e in orig_ests
                ])
                logger.info("  Comparison: original vs neutralized persona signal")
                signal_model: Dict[str, Dict[str, float]] = {}
                for dim in ESTIMATE_DIMS:
                    orig_entropy = orig_df[dim].value_counts(normalize=True).pipe(
                        lambda s: -(s * np.log2(s.clip(lower=1e-10))).sum()
                    )
                    neut_entropy = est_df[dim].value_counts(normalize=True).pipe(
                        lambda s: -(s * np.log2(s.clip(lower=1e-10))).sum()
                    )
                    reduction_pct = (1 - neut_entropy / orig_entropy) * 100 if orig_entropy > 0 else 0.0
                    signal_model[dim] = {
                        "original_entropy": float(orig_entropy),
                        "neutralized_entropy": float(neut_entropy),
                        "reduction_pct": float(reduction_pct),
                    }
                    logger.info("    %s entropy: original=%.3f, neutralized=%.3f", dim, orig_entropy, neut_entropy)

                # Store signal reduction in analysis_results
                mc = self.analysis_results.setdefault("manipulation_checks", {})
                sr = mc.setdefault("signal_reduction", {})
                sr[model] = signal_model

    # ------------------------------------------------------------------
    # Step 7: Remedy Decision (Neutralized)
    # ------------------------------------------------------------------

    def _step_7(self) -> None:
        """Remedy decisions on neutralized complaints."""
        for model in self.models:
            model_neutral = [n for n in self.neutralized if n.model == model]
            requests, narratives = [], []
            for nc in model_neutral:
                req = LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=nc.case_id,
                    model=model,
                    task="remedy_decision",
                    prompt_system=REMEDY_DECISION_SYSTEM,
                    prompt_user=safe_narrative_format(REMEDY_DECISION_USER, nc.neutralized_narrative),
                    expected_schema=RemedyDecisionResponse,
                    variant="neutralized",
                )
                requests.append(req)
                narratives.append(nc.neutralized_narrative)

            desc = f"Step 7 [{model}] Neutralized remedy"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            for resp in responses:
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    self.neutralized_decisions.append(RemedyDecision(
                        case_id=resp.case_id,
                        model=resp.model,
                        variant="neutralized",
                        tier=p.tier,
                        confidence=p.confidence,
                        reasoning=p.reasoning,
                        information_needed=p.information_needed,
                    ))

        logger.info("Step 7: %d neutralized decisions collected", len(self.neutralized_decisions))
        self._summarize_step_7()

        # Text features on neutralized complaints
        if self.neutralized:
            logger.info("Extracting text features for neutralized complaints...")
            neut_features = extract_features_batch(
                [n.neutralized_narrative for n in self.neutralized],
                [n.case_id for n in self.neutralized],
                "neutralized",
            )
            neut_df = features_to_dataframe(neut_features)
            if self.text_features_df is not None:
                self.text_features_df = pd.concat([self.text_features_df, neut_df], ignore_index=True)
            else:
                self.text_features_df = neut_df

            # LLM-rated text quality on neutralized complaints
            logger.info("Rating text quality for neutralized complaints...")
            for model in self.models:
                model_neutral = [n for n in self.neutralized if n.model == model]
                if model_neutral:
                    self._rate_text_quality(
                        case_ids=[n.case_id for n in model_neutral],
                        narratives=[n.neutralized_narrative for n in model_neutral],
                        variant="neutralized",
                        models=[model],
                    )

    def _summarize_step_7(self) -> None:
        """Console summary: persona×neutralized tier + mean tier comparison + process discrimination."""
        if not self.neutralized_decisions or not self.persona_estimates:
            return
        logger.info("── Step 7 Summary ──")
        neut_df = pd.DataFrame([
            {"case_id": d.case_id, "model": d.model, "neut_tier": d.tier, "confidence": d.confidence}
            for d in self.neutralized_decisions
        ])
        base_df = pd.DataFrame([
            {"case_id": d.case_id, "model": d.model, "orig_tier": d.tier, "orig_confidence": d.confidence}
            for d in self.baseline_decisions
        ]) if self.baseline_decisions else pd.DataFrame()

        persona_df = pd.DataFrame([
            {"case_id": pe.case_id, "model": pe.model, "age": pe.age,
             "geographic": pe.geographic, "education": pe.education,
             "socioeconomic": pe.socioeconomic, "gender": pe.gender,
             "esl": pe.esl, "persona_label": pe.persona_label}
            for pe in self.persona_estimates
        ])

        for model in self.models:
            n_m = neut_df[neut_df["model"] == model]
            p_m = persona_df[persona_df["model"] == model]
            merged = n_m.merge(p_m, on=["case_id", "model"], how="inner")
            if merged.empty:
                continue
            logger.info("  Model: %s (%d matched records)", model, len(merged))

            # 2-way crosstab: persona × neutralized tier (top 10)
            ct = pd.crosstab(merged["persona_label"], merged["neut_tier"])
            top10 = ct.sum(axis=1).nlargest(10).index
            ct_top = ct.loc[top10]
            header = ("persona",) + tuple(f"neut_tier_{c}" for c in ct_top.columns)
            rows = [header]
            for persona in ct_top.index:
                rows.append((persona,) + tuple(str(ct_top.loc[persona, c]) for c in ct_top.columns))
            _log_table("Top 10 persona × neutralized tier", rows)

            # 1-way: mean neutralized tier by dimension (all 6 dims)
            available_dims = [d for d in ESTIMATE_DIMS if d in merged.columns]
            _log_mean_by_dimension(merged, "neut_tier", available_dims)

            # Compare original vs neutralized mean tier
            if not base_df.empty:
                b_m = base_df[base_df["model"] == model]
                comp = b_m.merge(n_m, on=["case_id", "model"], how="inner")
                if not comp.empty:
                    logger.info("    Mean tier: original=%.3f, neutralized=%.3f, delta=%.3f",
                                comp["orig_tier"].mean(), comp["neut_tier"].mean(),
                                comp["neut_tier"].mean() - comp["orig_tier"].mean())

            # Process discrimination rate comparison
            neut_info = (n_m["confidence"] == "need_more_info").sum()
            neut_info_rate = neut_info / len(n_m) if len(n_m) > 0 else 0.0
            logger.info("  ── Process Discrimination Rate ──")
            logger.info("    Neutralized info request rate: %d/%d (%.3f)", neut_info, len(n_m), neut_info_rate)
            if not base_df.empty:
                b_m = base_df[base_df["model"] == model]
                orig_info = (b_m["orig_confidence"] == "need_more_info").sum() if "orig_confidence" in b_m.columns else 0
                orig_info_rate = orig_info / len(b_m) if len(b_m) > 0 else 0.0
                logger.info("    Original info request rate: %d/%d (%.3f)", orig_info, len(b_m), orig_info_rate)
                logger.info("    Delta: %.3f", neut_info_rate - orig_info_rate)

    # ------------------------------------------------------------------
    # Step 8: Debiased Fact Extraction
    # ------------------------------------------------------------------

    def _get_causal_sample(self) -> List[Complaint]:
        """Return the causal sample, re-deriving it deterministically if needed."""
        if self.causal_sample:
            return self.causal_sample
        rng = random.Random(self.seed)
        n_sample = max(1, int(len(self.complaints) * self.causal_sample_pct))
        self.causal_sample = rng.sample(self.complaints, n_sample)
        return self.causal_sample

    def _load_rewrites_from_cache(
        self, sample: List[Complaint], model: str,
    ) -> Optional[Dict[str, Dict[str, Optional[str]]]]:
        """Load persona rewrites from cache for the given sample.

        Returns dict: case_id -> persona_label -> rewritten_text.
        Returns None if any rewrites are missing.
        """
        rewrite_texts: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
        missing: List[str] = []

        for c in sample:
            for persona in PERSONA_GRID:
                plabel = persona_label(**persona)
                narrative_key = format_persona_rewrite_user(
                    c.narrative,
                    persona["age"], persona["geo"], persona["edu"], persona["ses"],
                )
                cached = self.cache.get(model, "persona_rewrite", narrative_key)
                if cached is None:
                    missing.append(f"{model}/{c.case_id}/{plabel}")
                else:
                    try:
                        parsed = PersonaRewriteResponse(**cached)
                        rewrite_texts[c.case_id][plabel] = parsed.rewritten_complaint
                    except Exception as exc:
                        missing.append(f"{model}/{c.case_id}/{plabel} (parse: {exc})")

        if missing:
            logger.error(
                "Step 8: %d persona rewrite(s) missing for %s. "
                "Run Step 4 first.", len(missing), model,
            )
            for m in missing[:10]:
                logger.error("  %s", m)
            if len(missing) > 10:
                logger.error("  ... and %d more", len(missing) - 10)
            return None

        return dict(rewrite_texts)

    def _step_8(self) -> None:
        """Debiased fact extraction on causal sample and persona rewrites."""
        from src.debiased_triage import (
            FactSetResponse,
            build_stage_1_prompt,
            heuristic_triage,
            parse_factset,
            ReasonCategory,
            Tier,
            TriageDecision,
        )

        sample = self._get_causal_sample()
        if not sample:
            logger.warning("Step 8: No causal sample available (did Step 4 run?)")
            return

        n_personas = len(PERSONA_GRID)
        complaint_meta = {c.case_id: (c.product, c.issue) for c in sample}

        fail_decision = TriageDecision(
            tier=Tier.NO_ACTION,
            primary_reason=ReasonCategory.INSUFFICIENT_EVIDENCE,
            explanation="Extraction failed.",
            confidence="low",
        )

        # Initialise per-model dicts
        for model in self.models:
            self.debiased_orig_fs[model] = {}
            self.debiased_orig_dec[model] = {}
            self.debiased_rw_fs[model] = {}
            self.debiased_rw_dec[model] = {}

        # Collect rewrite texts across models (they are model-specific in cache)
        all_rewrite_texts: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}

        for model in self.models:
            # -- 8a: Extract facts from originals --
            requests: List[LLMRequest] = []
            narratives: List[str] = []
            for c in sample:
                sys_msg, usr_msg = build_stage_1_prompt(c.narrative, c.product, c.issue)
                requests.append(LLMRequest(
                    request_id=LLMRequest.new_id(),
                    case_id=c.case_id,
                    model=model,
                    task="fact_extraction",
                    prompt_system=sys_msg,
                    prompt_user=usr_msg,
                    expected_schema=FactSetResponse,
                ))
                narratives.append(c.narrative)

            desc = f"Step 8 [{model}] Extract facts from originals"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            for c, resp in zip(sample, responses):
                if resp.success and resp.response_parsed is not None:
                    fs = parse_factset(resp.response_parsed)
                    self.debiased_orig_fs[model][c.case_id] = fs
                    self.debiased_orig_dec[model][c.case_id] = heuristic_triage(fs)
                else:
                    self.debiased_orig_fs[model][c.case_id] = None
                    self.debiased_orig_dec[model][c.case_id] = fail_decision

            # -- 8b: Load persona rewrites from cache --
            rewrite_texts = self._load_rewrites_from_cache(sample, model)
            if rewrite_texts is None:
                logger.error("Step 8: Skipping rewrite extraction for %s (missing rewrites).", model)
                continue
            all_rewrite_texts[model] = rewrite_texts

            # -- 8c: Extract facts from rewrites --
            requests = []
            narratives = []
            fact_keys: List[Tuple[str, str]] = []
            for case_id, personas in rewrite_texts.items():
                product, issue = complaint_meta[case_id]
                for plabel, text in personas.items():
                    if text is None:
                        continue
                    sys_msg, usr_msg = build_stage_1_prompt(text, product, issue)
                    requests.append(LLMRequest(
                        request_id=LLMRequest.new_id(),
                        case_id=case_id,
                        model=model,
                        task="fact_extraction",
                        prompt_system=sys_msg,
                        prompt_user=usr_msg,
                        expected_schema=FactSetResponse,
                    ))
                    narratives.append(text)
                    fact_keys.append((case_id, plabel))

            desc = f"Step 8 [{model}] Extract facts from rewrites ({len(requests)})"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            rw_ok = rw_fail = 0
            for (case_id, plabel), resp in zip(fact_keys, responses):
                if case_id not in self.debiased_rw_fs[model]:
                    self.debiased_rw_fs[model][case_id] = {}
                    self.debiased_rw_dec[model][case_id] = {}
                if resp.success and resp.response_parsed is not None:
                    fs = parse_factset(resp.response_parsed)
                    self.debiased_rw_fs[model][case_id][plabel] = fs
                    self.debiased_rw_dec[model][case_id][plabel] = heuristic_triage(fs)
                    rw_ok += 1
                else:
                    self.debiased_rw_fs[model][case_id][plabel] = None
                    self.debiased_rw_dec[model][case_id][plabel] = fail_decision
                    rw_fail += 1

            orig_ok = sum(1 for v in self.debiased_orig_fs[model].values() if v is not None)
            logger.info("Step 8 [%s]: originals %d ok / %d fail, rewrites %d ok / %d fail",
                        model, orig_ok, len(sample) - orig_ok, rw_ok, rw_fail)

        # Merge rewrite texts across models (use first model's texts as canonical)
        # All models share the same rewrites, just use whichever was loaded
        for model_rw in all_rewrite_texts.values():
            for case_id, personas in model_rw.items():
                if case_id not in self.debiased_rewrite_texts:
                    self.debiased_rewrite_texts[case_id] = personas

        logger.info("Step 8: Debiased extraction complete for %d complaints × %d model(s)",
                    len(sample), len(self.models))

    # ------------------------------------------------------------------
    # Build DataFrames
    # ------------------------------------------------------------------

    def _build_dataframes(self) -> None:
        """Convert result lists into DataFrames for analysis."""
        if self.persona_estimates:
            self.personas_df = pd.DataFrame([
                {
                    "case_id": pe.case_id, "model": pe.model,
                    "age": pe.age, "geographic": pe.geographic,
                    "education": pe.education, "socioeconomic": pe.socioeconomic,
                    "gender": pe.gender, "esl": pe.esl,
                    "confidence": pe.confidence, "persona_label": pe.persona_label,
                }
                for pe in self.persona_estimates
            ])

        if self.baseline_decisions:
            self.baseline_df = pd.DataFrame([
                {
                    "case_id": d.case_id, "model": d.model,
                    "tier": d.tier, "confidence": d.confidence,
                    "reasoning": d.reasoning, "information_needed": d.information_needed,
                }
                for d in self.baseline_decisions
            ])

        if self.persona_decisions:
            self.persona_decisions_df = pd.DataFrame([
                {
                    "case_id": d.case_id, "model": d.model, "variant": d.variant,
                    "tier": d.tier, "confidence": d.confidence,
                    "reasoning": d.reasoning, "information_needed": d.information_needed,
                }
                for d in self.persona_decisions
            ])
            # Parse variant into dimension columns for downstream analysis (e.g., IPW)
            if "variant" in self.persona_decisions_df.columns:
                parsed = self.persona_decisions_df["variant"].apply(parse_persona_label)
                valid = parsed.notna()
                if valid.any():
                    parsed_df = pd.DataFrame(parsed[valid].tolist(), index=parsed[valid].index)
                    for col in ["age", "geographic", "education", "socioeconomic"]:
                        self.persona_decisions_df[col] = parsed_df[col].reindex(self.persona_decisions_df.index)
                    logger.debug("Parsed dimension columns from variant (%d/%d rows)", valid.sum(), len(parsed))

        if self.neutralized_decisions:
            self.neutralized_df = pd.DataFrame([
                {
                    "case_id": d.case_id, "model": d.model,
                    "tier": d.tier, "confidence": d.confidence,
                    "reasoning": d.reasoning, "information_needed": d.information_needed,
                }
                for d in self.neutralized_decisions
            ])

        if self.text_quality_ratings:
            self.text_quality_df = pd.DataFrame([
                {
                    "case_id": r.case_id,
                    "model": r.model,
                    "variant": r.variant,
                    "convincingness": r.convincingness,
                    "persuasiveness": r.persuasiveness,
                    "evidence_blame": r.evidence_blame,
                    "evidence_harm": r.evidence_harm,
                    "tq_confidence": r.confidence,
                    "legal_knowledge": r.legal_knowledge,
                    "formality": r.formality,
                    "coherence": r.coherence,
                    "specificity": r.specificity,
                    "financial_literacy": r.financial_literacy,
                    "emotional_restraint": r.emotional_restraint,
                    "grammar_quality": r.grammar_quality,
                }
                for r in self.text_quality_ratings
            ])

        # Merge LLM-rated quality into text_features_df for variant="original"
        # (average across models since NLP features are model-independent)
        self._merge_quality_into_text_features()

    def _merge_quality_into_text_features(self) -> None:
        """Merge LLM-rated quality features into text_features_df for original variants.

        Quality ratings are model-specific while NLP text features are
        model-independent, so we average quality across models for the merge.
        Only variant="original" rows are enriched (case_id is unique there).
        Rewrite/neutralized rows get NaN for quality columns.
        """
        if self.text_features_df is None or self.text_quality_df is None:
            return
        if self.text_quality_df.empty:
            return

        quality_cols = [
            "convincingness", "persuasiveness", "evidence_blame",
            "evidence_harm", "tq_confidence", "legal_knowledge",
            "formality", "coherence", "specificity",
            "financial_literacy", "emotional_restraint", "grammar_quality",
        ]

        # Average quality ratings across models for original complaints
        orig_quality = self.text_quality_df[
            self.text_quality_df["variant"] == "original"
        ].groupby("case_id")[quality_cols].mean().reset_index()

        if orig_quality.empty:
            return

        # Split text_features_df into original vs rest
        is_original = self.text_features_df["variant"] == "original"
        orig_tf = self.text_features_df[is_original].copy()
        rest_tf = self.text_features_df[~is_original].copy()

        # Merge quality onto original rows (case_id is unique there)
        orig_tf = orig_tf.merge(orig_quality, on="case_id", how="left")

        # Reassemble (rest rows get NaN for quality columns automatically)
        self.text_features_df = pd.concat([orig_tf, rest_tf], ignore_index=True)

        n_merged = orig_tf[quality_cols[0]].notna().sum()
        logger.info(
            "Merged LLM-rated quality features into text_features_df: "
            "%d/%d original rows enriched",
            n_merged, len(orig_tf),
        )

    # ------------------------------------------------------------------
    # Run all analyses
    # ------------------------------------------------------------------

    def _data_diagnostic(self) -> Dict[str, Any]:
        """Log and capture data suitability diagnostics before running statistical analyses.

        Returns a dictionary with diagnostic results for each model, suitable for
        display in the HTML dashboard.
        """
        logger.info("── Data Diagnostic ──")
        dims = ESTIMATE_DIMS
        diagnostics: Dict[str, Any] = {}

        for model in self.models:
            logger.info("  Model: %s", model)
            model_diag: Dict[str, Any] = {
                "issues": [],
                "warnings": [],
                "tier_distribution": {},
                "dimension_cell_sizes": {},
                "persona_cell_stats": {},
            }

            # Baseline tier variation
            if self.baseline_df is not None:
                b_m = self.baseline_df[self.baseline_df["model"] == model]
                tier_vc = b_m["tier"].value_counts().to_dict()
                model_diag["tier_distribution"] = {str(k): int(v) for k, v in tier_vc.items()}
                model_diag["n_complaints"] = len(b_m)
                logger.info("    Baseline tiers: %s (n=%d)", tier_vc, len(b_m))

                if b_m["tier"].nunique() < 2:
                    model_diag["warnings"].append("No tier variation — logistic regression will be skipped")
                    logger.warning("    ⚠ No tier variation — logistic regression will be skipped")

            # Persona × tier cell sizes
            if self.personas_df is not None and self.baseline_df is not None:
                p_m = self.personas_df[self.personas_df["model"] == model]
                b_m = self.baseline_df[self.baseline_df["model"] == model]
                merged = b_m.merge(p_m, on=["case_id", "model"], how="inner")
                merged["favorable"] = (merged["tier"] >= 1).astype(int)

                if len(merged) > 0:
                    # Check per-dimension cell sizes
                    for dim in dims:
                        grp = merged.groupby(dim)["favorable"]
                        dim_info: Dict[str, Any] = {"min_cell_size": int(grp.size().min()), "constant_levels": []}

                        for level, sub in grp:
                            if sub.nunique() < 2:
                                dim_info["constant_levels"].append(str(level))
                                logger.info(
                                    "    %s=%s: target is constant (%s) — will use regularised logistic regression",
                                    dim, level, sub.iloc[0],
                                )

                        model_diag["dimension_cell_sizes"][dim] = dim_info
                        logger.info("    %s: min cell size = %d", dim, dim_info["min_cell_size"])

                        # Flag small cells
                        if dim_info["min_cell_size"] < 10:
                            model_diag["warnings"].append(f"{dim}: min cell size = {dim_info['min_cell_size']} (< 10)")

                    # Overall persona_label cell stats
                    persona_sizes = merged.groupby("persona_label").size()
                    model_diag["persona_cell_stats"] = {
                        "n_personas": len(persona_sizes),
                        "min_cell": int(persona_sizes.min()),
                        "median_cell": float(persona_sizes.median()),
                        "max_cell": int(persona_sizes.max()),
                    }
                    logger.info(
                        "    Persona cells: %d personas, min=%d, median=%.0f, max=%d",
                        len(persona_sizes), persona_sizes.min(), persona_sizes.median(), persona_sizes.max(),
                    )

                    # Flag sparse personas
                    sparse_count = (persona_sizes < 5).sum()
                    if sparse_count > 0:
                        model_diag["warnings"].append(f"{sparse_count} persona(s) have fewer than 5 observations")

            diagnostics[model] = model_diag

        return diagnostics

    def _run_analyses(self) -> None:
        """Run statistical analyses after all steps complete."""
        self._build_dataframes()
        data_diag = self._data_diagnostic()
        self.analysis_results["data_diagnostic"] = data_diag

        results: Dict[str, Any] = {"models": self.models, "timestamp": self.timestamp}

        for model in self.models:
            model_results: Dict[str, Any] = {}

            personas_m = self.personas_df[self.personas_df["model"] == model] if self.personas_df is not None else None
            baseline_m = self.baseline_df[self.baseline_df["model"] == model] if self.baseline_df is not None else None
            persona_dec_m = self.persona_decisions_df[self.persona_decisions_df["model"] == model] if self.persona_decisions_df is not None else None
            neutralized_m = self.neutralized_df[self.neutralized_df["model"] == model] if self.neutralized_df is not None else None

            # Step 3: Correlation
            if personas_m is not None and baseline_m is not None and len(personas_m) > 0 and len(baseline_m) > 0:
                logger.info("Running correlation analysis for %s...", model)
                model_results["correlation"] = correlation_analysis(
                    baseline_m, personas_m, cache_dir=self.cache_dir
                )
                model_results["process_correlation"] = process_correlation_analysis(baseline_m, personas_m)

            # Step 5: Causal
            if persona_dec_m is not None and len(persona_dec_m) > 0:
                logger.info("Running causal analysis for %s...", model)
                model_results["causal"] = causal_analysis(
                    baseline_m if baseline_m is not None else pd.DataFrame(),
                    persona_dec_m,
                )
                model_results["process_causal"] = process_causal_analysis(persona_dec_m)

            # Step 7: Mitigation
            if baseline_m is not None and neutralized_m is not None and personas_m is not None:
                if len(baseline_m) > 0 and len(neutralized_m) > 0:
                    logger.info("Running mitigation analysis for %s...", model)
                    model_results["mitigation"] = mitigation_analysis(
                        baseline_m, neutralized_m, personas_m
                    )
                    model_results["process_mitigation"] = process_mitigation_analysis(
                        baseline_m, neutralized_m, personas_m
                    )

            # Process: info request content
            if baseline_m is not None and personas_m is not None:
                model_results["info_content"] = info_request_content_analysis(baseline_m, personas_m)

            results[model] = model_results

        self.analysis_results.update(results)

        # Text quality summary statistics
        if self.text_quality_df is not None and not self.text_quality_df.empty:
            tq_summary: Dict[str, Any] = {}
            quality_cols = [
                "convincingness", "persuasiveness", "evidence_blame",
                "evidence_harm", "tq_confidence", "legal_knowledge",
                "formality", "coherence", "specificity",
                "financial_literacy", "emotional_restraint", "grammar_quality",
            ]
            for model in self.models:
                model_tq = self.text_quality_df[self.text_quality_df["model"] == model]
                model_summary: Dict[str, Any] = {}
                for variant in model_tq["variant"].unique():
                    v_data = model_tq[model_tq["variant"] == variant][quality_cols]
                    model_summary[variant] = {
                        col: {
                            "mean": round(float(v_data[col].mean()), 3),
                            "std": round(float(v_data[col].std()), 3),
                            "median": float(v_data[col].median()),
                        }
                        for col in quality_cols
                        if col in v_data.columns
                    }
                tq_summary[model] = model_summary
            self.analysis_results["text_quality"] = tq_summary

        self._run_power_analysis()
        self._run_severity_analysis()
        self._run_bayes_robustness()

        logger.info("=" * 60)
        logger.info("Post-estimation analyses (may take several minutes)...")
        logger.info("=" * 60)

        self._collect_multiple_comparisons()
        self._run_repeated_measures_analysis()
        self._run_sparsity_analysis()
        self._run_mitigation_effectiveness_analysis()
        self._run_persona_analysis()
        self._run_text_rewriting_analysis()
        self._run_debiased_consistency_analysis()
        self._display_analysis_results()

    # ------------------------------------------------------------------
    # Severity amplification analysis
    # ------------------------------------------------------------------

    def _run_severity_analysis(self) -> None:
        """Run severity amplification analysis (spec sections 1-6)."""
        if self.baseline_df is None or self.personas_df is None:
            logger.info("Skipping severity analysis (no baseline decisions or personas)")
            return

        # Build severity DataFrame from complaints
        severity_df = pd.DataFrame([
            {
                "case_id": c.case_id,
                "monetary_severity": c.monetary_severity,
                "issue_severity": c.issue_severity,
                "product_category": c.product_category,
                "issue_category": c.issue_category,
                "word_count": len(c.narrative.split()),
            }
            for c in self.complaints
        ])
        sev_cols = severity_df[["case_id", "monetary_severity", "issue_severity",
                                "product_category", "issue_category"]]

        # Merge severity + persona labels + causal dimensions onto baseline decisions
        persona_merge_cols = ["case_id", "model", "persona_label"]
        for dim in CAUSAL_DIMS:
            if dim in self.personas_df.columns:
                persona_merge_cols.append(dim)
        baseline_sev = self.baseline_df.merge(sev_cols, on="case_id", how="left")
        baseline_sev = baseline_sev.merge(
            self.personas_df[persona_merge_cols].drop_duplicates(),
            on=["case_id", "model"],
            how="left",
        )

        # Merge severity onto persona decisions (if available)
        persona_dec_sev = None
        if self.persona_decisions_df is not None and len(self.persona_decisions_df) > 0:
            persona_dec_sev = self.persona_decisions_df.merge(sev_cols, on="case_id", how="left")

        # Build neutralized severity DataFrame (if neutralized decisions available)
        neutralized_sev = None
        if self.neutralized_df is not None and len(self.neutralized_df) > 0:
            neutralized_sev = self.neutralized_df.merge(sev_cols, on="case_id", how="left")
            neutralized_sev = neutralized_sev.merge(
                self.personas_df[persona_merge_cols].drop_duplicates(),
                on=["case_id", "model"],
                how="left",
            )

        result = run_severity_analysis(
            baseline_df=baseline_sev,
            persona_decisions_df=persona_dec_sev,
            personas_df=self.personas_df,
            complaints_df=severity_df,
            models=self.models,
            n_boot=1000,
            seed=self.seed,
            neutralized_df=neutralized_sev,
            cache_dir=self.cache_dir,
        )
        self.analysis_results["severity"] = result

        # IPW robustness for monetary severity (pre-registered)
        if persona_dec_sev is not None and len(persona_dec_sev) > 0:
            try:
                from src.ipw_robustness import ipw_amplification_analysis
                logger.info("Running IPW robustness analysis for monetary severity...")
                ipw_result = ipw_amplification_analysis(
                    decisions_df=persona_dec_sev,
                    complaints_df=severity_df,
                    models=self.models,
                    n_boot=1000,
                    seed=self.seed,
                )
                self.analysis_results.setdefault("severity", {})["ipw_robustness"] = ipw_result
                logger.info("IPW robustness analysis complete")
            except Exception as e:
                logger.critical("IPW robustness analysis failed: %s", e)
                raise RuntimeError(
                    f"IPW robustness analysis failed (pre-registered analysis "
                    f"cannot be skipped): {e}"
                ) from e

        # Logistic GEE interaction tests (pre-registered confirmatory)
        try:
            from src.severity_analysis import logistic_interaction_tier2
            logger.info("Running logistic GEE interaction tests (confirmatory)...")
            gee_interactions: Dict[str, Any] = {}
            for metric in ["monetary_severity", "issue_severity"]:
                gee_interactions[metric] = {}
                for model in self.models:
                    model_gee: Dict[str, Any] = {}
                    df_for_gee = baseline_sev if metric == "issue_severity" else (
                        baseline_sev[baseline_sev["monetary_severity"].isin(["low", "medium", "high"])]
                    )
                    for dim in CAUSAL_DIMS:
                        if dim in df_for_gee.columns:
                            model_gee[dim] = logistic_interaction_tier2(
                                df_for_gee, metric, model, dim,
                            )
                    gee_interactions[metric][model] = model_gee
            self.analysis_results.setdefault("severity", {})["gee_interactions_tier2"] = gee_interactions
            logger.info("Logistic GEE interaction tests complete")
        except Exception as e:
            logger.error("GEE interaction tests failed: %s", e, exc_info=True)

        # Console summary: missingness diagnostic
        miss_diag = result.get("missingness_diagnostic", {})
        if miss_diag and not miss_diag.get("skipped"):
            logger.info("── Monetary Missingness Diagnostic ──")
            logger.info("  Amount present: %d (%.1f%%), Amount missing: %d (%.1f%%)",
                        miss_diag["n_present"],
                        (1 - miss_diag["pct_missing"]) * 100,
                        miss_diag["n_missing"],
                        miss_diag["pct_missing"] * 100)
            chars = miss_diag.get("complaint_characteristics", {})
            m_type = chars.get("missingness_type", "unknown")
            n_sig = chars.get("n_significant_tests", 0)
            sig_tests = chars.get("significant_tests", [])
            sig_str = ", ".join(sig_tests) if sig_tests else "none"
            logger.info("  Missingness type: %s (%d/5 tests significant: %s)",
                        m_type, n_sig, sig_str)
            for model in self.models:
                fc = miss_diag.get("fairness_comparison", {}).get(model, {})
                pres = fc.get("present", {})
                miss = fc.get("missing", {})
                ratio = fc.get("bias_sd_ratio", float("nan"))
                logger.info("  %s: bias_sd present=%.4f, missing=%.4f, ratio=%.2f",
                            model,
                            pres.get("bias_sd", float("nan")),
                            miss.get("bias_sd", float("nan")),
                            ratio)

        # Console summary of key findings
        logger.info("── Severity Amplification Summary ──")
        for metric in ["monetary_severity", "issue_severity"]:
            interactions = result.get("interactions", {}).get(metric, {})
            main_effects = result.get("main_effects", {}).get(metric, {})
            for model in self.models:
                model_int = interactions.get(model, {})
                amp = model_int.get("amplification", {})
                tw = model_int.get("two_way", {})
                interaction = tw.get("interaction", {})
                ratio = amp.get("point", float("nan"))
                ci_lo = amp.get("ci_lower", float("nan"))
                ci_hi = amp.get("ci_upper", float("nan"))
                int_p = interaction.get("p_value", float("nan"))
                logger.info(
                    "  %s [%s]: amplification=%.2fx [%.2f, %.2f], interaction p=%.4f",
                    metric, model, ratio, ci_lo, ci_hi, int_p,
                )

                # Show bias at each severity level
                bias = model_int.get("bias", {})
                se = model_int.get("simple_effects", {})
                mfx = main_effects.get(model, {})
                means = mfx.get("means_by_level", {})
                for level in ["low", "medium", "high"]:
                    bl = bias.get(level, {})
                    sl = se.get(level, {})
                    n = bl.get("n", 0)
                    sd = bl.get("sd_across_personas", float("nan"))
                    rng = bl.get("range", float("nan"))
                    mean_tier = means.get(level, float("nan"))
                    eta = sl.get("eta_sq_partial", float("nan"))
                    n_sig = sl.get("n_significant", 0)
                    largest_d = sl.get("largest_pairwise_d", float("nan"))
                    logger.info(
                        "    %s: n=%d, mean_tier=%.3f, bias_sd=%.4f, bias_range=%.4f, "
                        "eta²=%.4f, largest_d=%.3f, n_sig_pairs=%d",
                        level, n, mean_tier, sd, rng, eta, largest_d, n_sig,
                    )

                # Show most/least advantaged personas at high severity
                high_bias = bias.get("high", {})
                persona_means = high_bias.get("persona_means", {})
                if persona_means:
                    sorted_pm = sorted(persona_means.items(), key=lambda x: x[1])
                    worst = sorted_pm[0]
                    best = sorted_pm[-1]
                    logger.info(
                        "    high-severity: most disadvantaged=%s (%.3f), "
                        "most advantaged=%s (%.3f), gap=%.3f",
                        worst[0], worst[1], best[0], best[1], best[1] - worst[1],
                    )

        # Console summary: mitigation × severity
        mit_sev = result.get("mitigation_severity", {})
        if mit_sev:
            logger.info("── Severity Amplification: Before vs After Mitigation ──")
            for metric in ["monetary_severity", "issue_severity"]:
                for model in self.models:
                    ms = mit_sev.get(metric, {}).get(model, {})
                    comp = ms.get("comparison", {})
                    verdict = comp.get("verdict", "N/A")
                    base_amp = ms.get("baseline", {}).get("amplification", {})
                    neut_amp = ms.get("neutralized", {}).get("amplification", {})
                    base_ratio = base_amp.get("point", float("nan"))
                    neut_ratio = neut_amp.get("point", float("nan"))
                    delta = comp.get("amplification_change", float("nan"))
                    pct = comp.get("amplification_pct_change", float("nan"))
                    logger.info(
                        "  %s [%s]: verdict=%s, baseline=%.2fx, neutralized=%.2fx, "
                        "delta=%.2f (%.1f%%)",
                        metric, model, verdict, base_ratio, neut_ratio, delta, pct,
                    )
                    # Per-level bias SD changes
                    sd_changes = comp.get("bias_sd_change_by_level", {})
                    base_bias = ms.get("baseline", {}).get("bias", {})
                    neut_bias = ms.get("neutralized", {}).get("bias", {})
                    for level in ["low", "medium", "high"]:
                        b_sd = base_bias.get(level, {}).get("sd_across_personas", float("nan"))
                        n_sd = neut_bias.get(level, {}).get("sd_across_personas", float("nan"))
                        d_sd = sd_changes.get(level, float("nan"))
                        logger.info(
                            "    %s: bias SD %.4f → %.4f (%+.4f)",
                            level, b_sd, n_sd, d_sd,
                        )

    # ------------------------------------------------------------------
    # Bayesian robustness (non-confirmatory)
    # ------------------------------------------------------------------

    def _run_bayes_robustness(self) -> None:
        """Run Bayesian hierarchical robustness analysis (B1 logistic + B2 ordinal).

        Raises on any Stan compilation, sampling, or post-processing failure.
        """
        if self.baseline_df is None or self.personas_df is None:
            logger.info("Skipping Bayesian robustness (no baseline decisions or personas)")
            return

        from src.bayes_robustness import run_bayes_robustness

        logger.info("Running Bayesian robustness analysis...")

        # Build severity-merged DataFrame (same as _run_severity_analysis)
        severity_df = pd.DataFrame([
            {
                "case_id": c.case_id,
                "monetary_severity": c.monetary_severity,
                "issue_severity": c.issue_severity,
                "product_category": c.product_category,
                "issue_category": c.issue_category,
            }
            for c in self.complaints
        ])
        sev_cols = severity_df[["case_id", "monetary_severity", "issue_severity",
                                "product_category", "issue_category"]]

        persona_merge_cols = ["case_id", "model", "persona_label"]
        for dim in CAUSAL_DIMS:
            if dim in self.personas_df.columns:
                persona_merge_cols.append(dim)
        baseline_sev = self.baseline_df.merge(sev_cols, on="case_id", how="left")
        baseline_sev = baseline_sev.merge(
            self.personas_df[persona_merge_cols].drop_duplicates(),
            on=["case_id", "model"],
            how="left",
        )

        # Output dir for artifacts
        output_dir = self.output_dir

        bayes_result = run_bayes_robustness(
            baseline_sev=baseline_sev,
            models=self.models,
            seed=self.seed,
            output_dir=output_dir,
            cache_dir=self.cache_dir,
        )
        self.analysis_results.setdefault("severity", {})["bayesian_robustness"] = bayes_result
        logger.info("Bayesian robustness analysis complete")

    # ------------------------------------------------------------------
    # Power analysis
    # ------------------------------------------------------------------

    def _run_power_analysis(self) -> None:
        """Run post-hoc power analysis: minimum detectable effect sizes."""
        if self.personas_df is None or self.baseline_df is None:
            return

        from src.statistics import power_analysis

        n_complaints = len(self.complaints)
        n_personas = len(PERSONA_LABELS)
        causal_sample = int(n_complaints * self.causal_sample_pct)

        # Compute minimum cell size for each causal dimension
        merged = self.baseline_df.merge(
            self.personas_df[["case_id", "model"] + [d for d in CAUSAL_DIMS if d in self.personas_df.columns]],
            on=["case_id", "model"], how="inner",
        )
        first_model = self.models[0] if self.models else None
        n_per_cell: Dict[str, int] = {}
        if first_model:
            m = merged[merged["model"] == first_model]
            for dim in CAUSAL_DIMS:
                if dim in m.columns:
                    n_per_cell[dim] = int(m.groupby(dim).size().min())

        result = power_analysis(
            n_complaints=n_complaints,
            n_personas=n_personas,
            n_per_cell=n_per_cell,
            causal_sample=causal_sample,
        )
        self.analysis_results["power_analysis"] = result
        logger.info("Power analysis complete: min detectable paired d=%.3f",
                    result.get("matched_guise", {}).get("min_detectable_d", float("nan")))

    # ------------------------------------------------------------------
    # Multiple comparisons summary
    # ------------------------------------------------------------------

    def _collect_multiple_comparisons(self) -> None:
        """Collect metadata about all FDR-corrected test families."""
        logger.info("Collecting multiple comparisons summary...")
        from src.statistics import multiple_comparisons_summary, fdr_correction

        records: List[Dict[str, Any]] = []

        # 1. Pairwise McNemar tests (causal analysis, per model)
        for model in self.models:
            mr = self.analysis_results.get(model, {})
            mcnemar = mr.get("causal", {}).get("pairwise_mcnemar", [])
            if mcnemar:
                n_tests = len(mcnemar)
                n_surv = sum(1 for r in mcnemar if r.get("significant", False))
                records.append({
                    "family_label": f"Pairwise McNemar ({model})",
                    "n_tests": n_tests,
                    "n_surviving": n_surv,
                })

        # 2. Severity simple effects pairwise t-tests
        sev = self.analysis_results.get("severity", {})
        interactions = sev.get("interactions", {})
        for metric in ["monetary_severity", "issue_severity"]:
            for model in self.models:
                se = interactions.get(metric, {}).get(model, {}).get("simple_effects", {})
                for level in ["low", "medium", "high"]:
                    sl = se.get(level, {})
                    n_sig = sl.get("n_significant", 0)
                    n = sl.get("n", 0)
                    if n > 0:
                        # Number of pairwise tests = C(n_personas, 2)
                        n_personas_in_level = n  # approximate
                        records.append({
                            "family_label": f"Severity simple effects ({metric}/{model}/{level})",
                            "n_tests": n_personas_in_level,
                            "n_surviving": n_sig,
                        })

        # 3. Per-dimension chi-squared tests (correlation analysis)
        for model in self.models:
            mr = self.analysis_results.get(model, {})
            corr = mr.get("correlation", {})
            dim_tests = [k for k in corr if k.endswith("_chi2")]
            if dim_tests:
                p_vals = [corr[k].get("p_value", 1.0) for k in dim_tests]
                adj = fdr_correction(p_vals)
                n_surv = int((np.array(adj) < 0.05).sum())
                records.append({
                    "family_label": f"Dimension chi-squared ({model})",
                    "n_tests": len(dim_tests),
                    "n_surviving": n_surv,
                })

        # 4. Process discrimination pairwise McNemar (per model)
        for model in self.models:
            mr = self.analysis_results.get(model, {})
            proc_causal = mr.get("process_causal", {})
            pw = proc_causal.get("pairwise_mcnemar", [])
            if pw:
                n_tests = len(pw)
                n_surv = sum(1 for r in pw if r.get("significant", False))
                records.append({
                    "family_label": f"Process McNemar ({model})",
                    "n_tests": n_tests,
                    "n_surviving": n_surv,
                })

        self.analysis_results["multiple_comparisons"] = multiple_comparisons_summary(records)
        mc = self.analysis_results["multiple_comparisons"]
        logger.info("Multiple comparisons summary: %d tests across %d families, %d surviving FDR",
                    mc["total_tests"], len(mc["families"]), mc["total_surviving"])

    # ------------------------------------------------------------------
    # Repeated-measures causal analysis
    # ------------------------------------------------------------------

    def _run_repeated_measures_analysis(self) -> None:
        """Run dependence-aware repeated-measures models for matched-guise data.

        Builds the canonical long-format dataset and fits:
        A) Ordinal mixed model (outcome tier) with clustered SEs
        B) Conditional logistic / GEE (favourable) with complaint_id strata
        C) GEE (info request) with complaint_id clustering
        D) Linear mixed model (robustness)
        E) Friedman / Cochran's Q nonparametric tests with FDR post-hoc
        """
        if self.baseline_df is None or self.personas_df is None:
            logger.info("Skipping repeated-measures analysis (no baseline decisions or personas)")
            return

        logger.info("Fitting repeated-measures models (ordinal, conditional logit, GEE, LMM, nonparametric)...")

        # Build severity metadata
        severity_df = pd.DataFrame([
            {
                "case_id": c.case_id,
                "monetary_severity": c.monetary_severity,
                "issue_severity": c.issue_severity,
            }
            for c in self.complaints
        ])

        result = run_repeated_measures_analysis(
            baseline_df=self.baseline_df,
            persona_decisions_df=self.persona_decisions_df,
            neutralized_df=self.neutralized_df,
            personas_df=self.personas_df,
            severity_df=severity_df,
            models=self.models,
            cache_dir=self.cache_dir,
        )
        self.analysis_results["repeated_measures"] = result

        # Console summary
        if result.get("skipped"):
            logger.info("Repeated-measures analysis skipped: %s", result.get("reason"))
            return

        ds = result.get("data_summary", {})
        logger.info("── Repeated-Measures Causal Analysis ──")
        logger.info("  Long-format: %d rows, %d complaints, %d personas",
                     ds.get("n_rows", 0), ds.get("n_complaints", 0), ds.get("n_personas", 0))

        for model in self.models:
            mr = result.get(model, {})
            if not mr or mr.get("skipped"):
                continue

            # Ordinal model summary
            om = mr.get("ordinal_model", {})
            if not om.get("skipped"):
                pxs_p = om.get("p_values", {}).get("persona_x_severity", float("nan"))
                logger.info("  %s ordinal model: n=%d, %d clusters, persona×severity p=%.4f%s",
                            model, om.get("n_obs", 0), om.get("n_clusters", 0),
                            pxs_p, " *" if pxs_p < 0.05 else "")

            # Conditional logistic summary
            cl = mr.get("conditional_logistic_favourable", {})
            if not cl.get("skipped"):
                ps_or = cl.get("odds_ratios", {}).get("persona_score", float("nan"))
                logger.info("  %s conditional logistic: %d strata, persona_score OR=%.3f",
                            model, cl.get("n_strata", 0), ps_or)

            # LMM ICC
            lmm = mr.get("linear_mixed_model", {})
            if not lmm.get("skipped"):
                icc = lmm.get("icc", 0)
                logger.info("  %s LMM: ICC=%.3f (%.1f%% variance between complaints)",
                            model, icc, icc * 100)

            # Nonparametric summary
            np_res = mr.get("nonparametric", {})
            if not np_res.get("skipped"):
                fr = np_res.get("friedman_tier", {})
                if not fr.get("skipped"):
                    logger.info("  %s Friedman (tier): stat=%.2f, p=%.4f",
                                model, fr.get("statistic", 0), fr.get("p_value", 1))
                cq = np_res.get("cochran_favourable", {})
                if not cq.get("skipped"):
                    logger.info("  %s Cochran's Q (favourable): stat=%.2f, p=%.4f",
                                model, cq.get("statistic", 0), cq.get("p_value", 1))

    # ------------------------------------------------------------------
    # Sparsity-aware analysis
    # ------------------------------------------------------------------

    def _run_sparsity_analysis(self) -> None:
        """Run sparsity-aware fairness analysis (small-cell handling)."""
        if self.baseline_df is None or self.personas_df is None:
            logger.info("Skipping sparsity analysis (no baseline decisions or personas)")
            return

        logger.info("Running sparsity-aware analysis (small-cell sensitivity)...")
        result = run_sparsity_analysis(
            baseline_df=self.baseline_df,
            personas_df=self.personas_df,
            models=self.models,
            seed=self.seed,
            persona_decisions_df=self.persona_decisions_df,
            neutralized_df=self.neutralized_df,
        )
        self.analysis_results["sparsity"] = result

        # Console summary
        logger.info("── Sparsity-Aware Analysis Summary ──")
        for model in self.models:
            mr = result.get(model, {})
            if not mr:
                continue
            obs = mr.get("observed_persona", {})
            support = obs.get("support", {})
            logger.info(
                "  %s: %d groups (%d insufficient, %d screening, %d reliable)",
                model,
                support.get("n_groups", 0),
                support.get("n_insufficient", 0),
                support.get("n_screening", 0),
                support.get("n_reliable", 0),
            )
            sens = mr.get("sensitivity_persona", {})
            verdict = sens.get("verdict", {})
            if verdict:
                logger.info("    Robustness: %s", verdict.get("summary", "N/A"))

            pooling = mr.get("partial_pooling", {})
            shrinkage = pooling.get("shrinkage_summary", {})
            if shrinkage:
                logger.info(
                    "    Partial pooling: mean |shrinkage|=%.4f, max=%.4f, "
                    "%d cells shrunk >5%%",
                    shrinkage.get("mean_abs_shrinkage", 0),
                    shrinkage.get("max_abs_shrinkage", 0),
                    shrinkage.get("n_cells_shrunk_gt_5pct", 0),
                )

    # ------------------------------------------------------------------
    # Mitigation effectiveness analysis
    # ------------------------------------------------------------------

    def _run_mitigation_effectiveness_analysis(self) -> None:
        """Run mitigation effectiveness analysis with ML models and SHAP."""
        if self.baseline_df is None or self.neutralized_df is None or self.personas_df is None:
            logger.info("Skipping mitigation effectiveness analysis (missing required data)")
            return

        if len(self.baseline_df) < 50 or len(self.neutralized_df) < 50:
            logger.info("Skipping mitigation effectiveness analysis (insufficient data)")
            return

        logger.info("Running mitigation effectiveness analysis (ML models + SHAP)...")

        # Build complaints DataFrame for the analysis
        complaints_df = pd.DataFrame([
            {
                "case_id": c.case_id,
                "narrative": c.narrative,
                "product_category": c.product_category,
                "issue_category": c.issue_category,
                "monetary_severity": c.monetary_severity,
                "issue_severity": c.issue_severity,
            }
            for c in self.complaints
        ])

        try:
            result = run_mitigation_effectiveness_analysis(
                baseline_df=self.baseline_df,
                neutralized_df=self.neutralized_df,
                personas_df=self.personas_df,
                complaints_df=complaints_df,
                models=self.models,
                persona_decisions_df=self.persona_decisions_df,
                text_features_df=self.text_features_df,
                n_examples=7,
            )
            self.analysis_results["mitigation_effectiveness"] = result

            # Console summary
            logger.info("── Mitigation Effectiveness Analysis Summary ──")
            for model in self.models:
                neut_models = result.get("neutralization_models", {}).get(model, [])
                for m in neut_models:
                    if m.get("metrics"):
                        r2 = m["metrics"].get("r2", 0)
                        rmse = m["metrics"].get("rmse", 0)
                        logger.info("  %s %s: R²=%.3f, RMSE=%.3f",
                                    model, m.get("model_type", ""), r2, rmse)

                hte_list = result.get("heterogeneous_effects", {}).get(model, [])
                for hte in hte_list:
                    if hte.get("overall_ate"):
                        logger.info("  %s HTE (%s): ATE=%.3f, CI=[%.3f, %.3f]",
                                    model, hte.get("treatment", ""),
                                    hte["overall_ate"],
                                    hte["ate_ci"][0], hte["ate_ci"][1])

            golden = result.get("golden_example")
            if golden:
                logger.info("  Golden example: %s (%s)",
                            golden.get("complaint_id", ""),
                            golden.get("product", ""))

            n_examples = len(result.get("supporting_examples", []))
            logger.info("  Supporting examples: %d", n_examples)

        except Exception as e:
            logger.error("Mitigation effectiveness analysis failed: %s", e, exc_info=True)
            get_diagnostics().error(
                category=DiagnosticCategory.COMPUTATION,
                module="experiment",
                message=f"Mitigation effectiveness analysis failed: {e}",
                fallback_action=None,
            )
            self.analysis_results["mitigation_effectiveness"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Persona linguistic analysis
    # ------------------------------------------------------------------

    def _run_persona_analysis(self) -> None:
        """Run persona linguistic analysis with LightGBM and PDPs."""
        if self.personas_df is None or self.text_features_df is None:
            logger.info("Skipping persona analysis (missing personas or text features)")
            return

        logger.info("Running persona linguistic analysis...")

        try:
            result = run_persona_analysis(
                personas_df=self.personas_df,
                text_features_df=self.text_features_df,
                models=self.models,
            )
            self.analysis_results["persona_analysis"] = result

            # Console summary
            logger.info("── Persona Analysis Summary ──")
            for model in self.models:
                model_result = result.get(model, {})
                dims = model_result.get("dimensions", {})
                if dims:
                    logger.info("  %s: analyzed %d persona dimensions", model, len(dims))
                    for dim, values in dims.items():
                        logger.info("    %s: %d values analyzed", dim, len(values))

        except Exception as e:
            logger.error("Persona analysis failed: %s", e, exc_info=True)
            get_diagnostics().error(
                category=DiagnosticCategory.COMPUTATION,
                module="experiment",
                message=f"Persona analysis failed: {e}",
                fallback_action=None,
            )
            self.analysis_results["persona_analysis"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Text rewriting analysis
    # ------------------------------------------------------------------

    def _run_text_rewriting_analysis(self) -> None:
        """Run text rewriting analysis to compare text characteristics before/after persona rewriting."""
        if self.text_features_df is None or self.text_features_df.empty:
            logger.info("Skipping text rewriting analysis (no text features data)")
            return

        # Check if we have rewritten variants
        variants = self.text_features_df["variant"].unique()
        has_rewrites = any(
            v not in ["original", "neutralized"] for v in variants
        )

        if not has_rewrites:
            logger.info("Skipping text rewriting analysis (no rewritten variants found)")
            return

        logger.info("Running text rewriting analysis...")

        try:
            result = run_text_rewriting_analysis(self.text_features_df)
            self.analysis_results["text_rewriting"] = result

            # Console summary
            if "error" not in result:
                n_sig = len(result.get("significant_features", []))
                n_total = len(result.get("overall_comparisons", []))
                n_personas = result.get("n_personas", 0)

                logger.info("── Text Rewriting Analysis Summary ──")
                logger.info("  Significant features (FDR-corrected): %d/%d", n_sig, n_total)

                if result.get("significant_features"):
                    logger.info("  Changed features: %s", ", ".join(result["significant_features"][:5]))

                logger.info("  Personas analyzed: %d", n_personas)

                # Top effect sizes
                comparisons = result.get("overall_comparisons", [])
                if comparisons:
                    sorted_comps = sorted(comparisons, key=lambda x: abs(x.get("cohens_d", 0)), reverse=True)
                    logger.info("  Top effect sizes:")
                    for comp in sorted_comps[:3]:
                        d = comp.get("cohens_d", 0)
                        feat = comp.get("feature_label", comp.get("feature", ""))
                        pct = comp.get("pct_change", 0)
                        logger.info("    %s: d=%.3f (%.1f%% change)", feat, d, pct)

        except Exception as e:
            logger.error("Text rewriting analysis failed: %s", e, exc_info=True)
            get_diagnostics().error(
                category=DiagnosticCategory.COMPUTATION,
                module="experiment",
                message=f"Text rewriting analysis failed: {e}",
                fallback_action=None,
            )
            self.analysis_results["text_rewriting"] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Debiased extraction consistency analysis
    # ------------------------------------------------------------------

    def _run_debiased_consistency_analysis(self) -> None:
        """Run debiased extraction consistency analysis (Analyses 1-5)."""
        if not self.debiased_orig_fs:
            logger.info("Skipping debiased consistency analysis (no Step 8 data)")
            return

        from src.debiased_analysis import analyze_extraction_consistency

        sample = self._get_causal_sample()
        results = analyze_extraction_consistency(
            orig_fs=self.debiased_orig_fs,
            orig_dec=self.debiased_orig_dec,
            rw_fs=self.debiased_rw_fs,
            rw_dec=self.debiased_rw_dec,
            rewrite_texts=self.debiased_rewrite_texts,
            complaints=sample,
            models=self.models,
        )
        self.analysis_results["debiased_consistency"] = results.to_dict()

        # Console summary
        avgs = results.model_averages
        for model, avg in avgs.items():
            logger.info("── Debiased Consistency [%s] ──", model)
            logger.info("  Avg tier match:    %.1f%%", avg["tier"] * 100)
            logger.info("  Avg entity match:  %.1f%%", avg["entity"] * 100)
            logger.info("  Avg conduct match: %.1f%%", avg["conduct"] * 100)
            logger.info("  Avg harm match:    %.1f%%", avg["harm"] * 100)

        if results.worked_example_rows:
            logger.info("  Worked example found (%d rows)", len(results.worked_example_rows))

        logger.info("Debiased consistency analysis complete")

    # ------------------------------------------------------------------
    # Display analysis results
    # ------------------------------------------------------------------

    def _display_analysis_results(self) -> None:
        """Print key analysis results to console."""
        logger.info("=" * 60)
        logger.info("ANALYSIS RESULTS")
        logger.info("=" * 60)

        for model in self.models:
            mr = self.analysis_results.get(model, {})
            if not mr:
                continue
            logger.info("━" * 50)
            logger.info("Model: %s", model)
            logger.info("━" * 50)

            # --- Correlation (Step 3) ---
            corr = mr.get("correlation", {})
            if corr:
                logger.info("── Correlation Analysis (Outcome) ──")
                chi2 = corr.get("overall_chi2")
                if chi2:
                    logger.info("  Overall persona×tier: chi2=%.2f, p=%.4f, Cramer's V=%.3f",
                                chi2["chi2"], chi2["p_value"], chi2["cramers_v"])
                for dim in ESTIMATE_DIMS:
                    d_chi2 = corr.get(f"{dim}_chi2")
                    if d_chi2:
                        logger.info("  %s×tier: chi2=%.2f, p=%.4f, V=%.3f",
                                    dim, d_chi2["chi2"], d_chi2["p_value"], d_chi2["cramers_v"])

                # Logistic regression
                logit = corr.get("logistic_favorable", {})
                if logit.get("skipped"):
                    logger.info("  Logistic regression: SKIPPED (%s)", logit.get("reason", ""))
                elif logit.get("regularized"):
                    logger.info("  Logistic regression: regularised (separation in %s)",
                                logit.get("separation_columns", []))
                    odds = logit.get("odds_ratios", {})
                    for k, v in sorted(odds.items()):
                        if k != "const":
                            logger.info("    %s: OR=%.3f", k, v)
                elif logit.get("error"):
                    logger.info("  Logistic regression: FAILED (%s)", logit["error"])
                elif logit.get("coefficients"):
                    logger.info("  Logistic regression: pseudo_R2=%.4f, AIC=%.1f, converged=%s",
                                logit.get("pseudo_r2", 0), logit.get("aic", 0), logit.get("converged"))
                    odds = logit.get("odds_ratios", {})
                    pvals = logit.get("p_values", {})
                    for k in sorted(odds.keys()):
                        if k != "const":
                            sig = "*" if pvals.get(k, 1) < 0.05 else ""
                            logger.info("    %s: OR=%.3f, p=%.4f %s", k, odds[k], pvals.get(k, 1), sig)

                # Rates by dimension
                rates = corr.get("rates_by_dimension", {})
                if rates:
                    logger.info("  Favorable outcome rates by dimension:")
                    for dim, dim_rates in sorted(rates.items()):
                        parts = [f"{k}={v:.3f}" for k, v in sorted(dim_rates.items())]
                        logger.info("    %s: %s", dim, ", ".join(parts))

            # --- Process Correlation ---
            pcorr = mr.get("process_correlation", {})
            if pcorr:
                logger.info("── Correlation Analysis (Process) ──")
                chi2 = pcorr.get("overall_chi2")
                if chi2:
                    logger.info("  Overall persona×need_info: chi2=%.2f, p=%.4f, V=%.3f",
                                chi2["chi2"], chi2["p_value"], chi2["cramers_v"])
                info_rates = pcorr.get("info_rates_by_dimension", {})
                if info_rates:
                    logger.info("  Info request rates by dimension:")
                    for dim, dim_rates in sorted(info_rates.items()):
                        parts = [f"{k}={v:.3f}" for k, v in sorted(dim_rates.items())]
                        logger.info("    %s: %s", dim, ", ".join(parts))

            # --- Causal (Step 5) ---
            causal = mr.get("causal", {})
            if causal:
                logger.info("── Causal Analysis (Outcome) ──")
                rates = causal.get("rates_by_persona", {})
                if rates:
                    sorted_rates = sorted(rates.items(), key=lambda x: x[1])
                    logger.info("  Favorable rates: lowest=%s (%.3f), highest=%s (%.3f)",
                                sorted_rates[0][0], sorted_rates[0][1],
                                sorted_rates[-1][0], sorted_rates[-1][1])
                    logger.info("  Spread: %.3f", sorted_rates[-1][1] - sorted_rates[0][1])
                friedman = causal.get("friedman")
                if friedman:
                    logger.info("  Friedman test: stat=%.2f, p=%.4f", friedman["statistic"], friedman["p_value"])
                # Significant pairwise McNemar
                mcnemar_pairs = causal.get("pairwise_mcnemar", [])
                sig_pairs = [p for p in mcnemar_pairs if p.get("significant")]
                logger.info("  Pairwise McNemar: %d/%d significant (FDR-adjusted p<0.05)",
                            len(sig_pairs), len(mcnemar_pairs))

            # --- Process Causal ---
            pcausal = mr.get("process_causal", {})
            if pcausal:
                logger.info("── Causal Analysis (Process) ──")
                info_rates = pcausal.get("info_rates_by_persona", {})
                if info_rates:
                    sorted_rates = sorted(info_rates.items(), key=lambda x: x[1])
                    logger.info("  Need-info rates: lowest=%s (%.3f), highest=%s (%.3f)",
                                sorted_rates[0][0], sorted_rates[0][1],
                                sorted_rates[-1][0], sorted_rates[-1][1])
                cochran = pcausal.get("cochrans_q")
                if cochran:
                    logger.info("  Cochran's Q: stat=%.2f, p=%.4f", cochran["statistic"], cochran["p_value"])

            # --- Mitigation (Step 7) ---
            mitigation = mr.get("mitigation", {})
            if mitigation:
                logger.info("── Mitigation Analysis (Outcome) ──")
                mcnemar_pp = mitigation.get("mcnemar_pre_post")
                if mcnemar_pp:
                    logger.info("  McNemar pre/post: stat=%.2f, p=%.4f",
                                mcnemar_pp["statistic"], mcnemar_pp["p_value"])
                var_orig = mitigation.get("variance_orig")
                var_neut = mitigation.get("variance_neut")
                var_red = mitigation.get("variance_reduction")
                if var_orig is not None:
                    logger.info("  Variance: original=%.6f, neutralized=%.6f, reduction=%.1f%%",
                                var_orig, var_neut, (var_red or 0) * 100)
                tier_changes = mitigation.get("tier_changes", {})
                if tier_changes:
                    logger.info("  Tier changes (orig->neut): %s", tier_changes)

            # --- Process Mitigation ---
            pmit = mr.get("process_mitigation", {})
            if pmit:
                logger.info("── Mitigation Analysis (Process) ──")
                mcnemar_pp = pmit.get("mcnemar_pre_post")
                if mcnemar_pp:
                    logger.info("  McNemar pre/post: stat=%.2f, p=%.4f",
                                mcnemar_pp["statistic"], mcnemar_pp["p_value"])
                var_red = pmit.get("variance_reduction")
                if var_red is not None:
                    logger.info("  Variance reduction: %.1f%%", var_red * 100)

            # --- Info Content ---
            info = mr.get("info_content", {})
            if info and info.get("n_requests", 0) > 0:
                logger.info("── Information Request Content ──")
                logger.info("  Total info requests: %d", info["n_requests"])
                mean_len = info.get("mean_length_by_persona", {})
                if mean_len:
                    sorted_len = sorted(mean_len.items(), key=lambda x: x[1], reverse=True)
                    for persona, length in sorted_len[:5]:
                        logger.info("    %s: mean length=%.0f chars", persona, length)

            # --- LLM Commentary ---
            logger.info("── GPT-5.2 Commentary ──")
            results_text = self._format_results_for_commentary(model, mr)
            commentary = self._get_llm_commentary(model, results_text)
            if commentary:
                logger.info("  %s", commentary)
                # Store for dashboard
                mr["gpt_commentary"] = commentary
            else:
                logger.info("  (commentary unavailable)")

        # --- Business Executive Summary (6 sections) ---
        exec_sections = self._generate_all_exec_sections()
        if exec_sections:
            self.analysis_results["exec_sections"] = exec_sections
            # Also store the combined text for backwards compatibility
            combined = "\n\n".join(exec_sections.values())
            self.analysis_results["business_executive_summary"] = combined

            # Print all sections concatenated
            logger.info("=" * 60)
            logger.info("BUSINESS EXECUTIVE SUMMARY (7 SECTIONS)")
            logger.info("=" * 60)
            section_titles = {
                "linguistic_differences": "SECTION 1: LINGUISTIC DIFFERENCES",
                "ai_decision_making": "SECTION 2: AI DECISION-MAKING",
                "correlation_vs_cause": "SECTION 3: CORRELATION VS. CAUSE",
                "severity_stakes": "SECTION 4: SEVERITY STAKES",
                "language_neutralisation": "SECTION 5: LANGUAGE NEUTRALISATION",
                "bias_mitigation": "SECTION 6: BIAS MITIGATION",
                "conclusion": "SECTION 7: CONCLUSION",
            }
            for key, title in section_titles.items():
                text = exec_sections.get(key)
                if text:
                    logger.info("─" * 50)
                    logger.info(title)
                    logger.info("─" * 50)
                    logger.info("%s", text)
                    logger.info("")
        else:
            logger.info("  (business executive summary unavailable)")

        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # LLM commentary on analysis results
    # ------------------------------------------------------------------

    def _format_results_for_commentary(self, model: str, mr: Dict[str, Any]) -> str:
        """Format a model's analysis results into a text summary for LLM commentary."""
        lines = [f"Analysis results for model: {model}\n"]

        # Correlation (Outcome)
        corr = mr.get("correlation", {})
        if corr:
            lines.append("CORRELATION ANALYSIS (Outcome Discrimination):")
            chi2 = corr.get("overall_chi2")
            if chi2:
                lines.append(f"  Overall persona x tier: chi2={chi2['chi2']:.2f}, p={chi2['p_value']:.4f}, Cramer's V={chi2['cramers_v']:.3f}")
            for dim in ESTIMATE_DIMS:
                d_chi2 = corr.get(f"{dim}_chi2")
                if d_chi2:
                    lines.append(f"  {dim} x tier: chi2={d_chi2['chi2']:.2f}, p={d_chi2['p_value']:.4f}, V={d_chi2['cramers_v']:.3f}")
            logit = corr.get("logistic_favorable", {})
            if logit.get("skipped"):
                lines.append(f"  Logistic regression: SKIPPED ({logit.get('reason', '')})")
            elif logit.get("regularized"):
                lines.append(f"  Logistic regression: regularised (separation in {logit.get('separation_columns', [])})")
                for k, v in sorted(logit.get("odds_ratios", {}).items()):
                    if k != "const":
                        lines.append(f"    {k}: OR={v:.3f}")
            elif logit.get("coefficients"):
                lines.append(f"  Logistic regression: pseudo_R2={logit.get('pseudo_r2', 0):.4f}, AIC={logit.get('aic', 0):.1f}, converged={logit.get('converged')}")
                odds = logit.get("odds_ratios", {})
                pvals = logit.get("p_values", {})
                for k in sorted(odds.keys()):
                    if k != "const":
                        sig = " *" if pvals.get(k, 1) < 0.05 else ""
                        lines.append(f"    {k}: OR={odds[k]:.3f}, p={pvals.get(k, 1):.4f}{sig}")
            rates = corr.get("rates_by_dimension", {})
            if rates:
                lines.append("  Favorable outcome rates by dimension:")
                for dim, dim_rates in sorted(rates.items()):
                    parts = [f"{k}={v:.3f}" for k, v in sorted(dim_rates.items())]
                    lines.append(f"    {dim}: {', '.join(parts)}")
            # AIR by persona
            air = corr.get("air_by_persona", {})
            if air:
                lines.append("  Adverse Impact Ratios by persona (rate, AIR, material<0.80):")
                for persona, vals in sorted(air.items(), key=lambda x: x[1].get("air", 1)):
                    flag = " MATERIAL" if vals.get("material") else ""
                    lines.append(f"    {persona}: rate={vals['rate']:.3f}, AIR={vals['air']:.3f}{flag}")

        # Correlation (Process)
        pcorr = mr.get("process_correlation", {})
        if pcorr:
            lines.append("\nCORRELATION ANALYSIS (Process Discrimination - info request burden):")
            chi2 = pcorr.get("overall_chi2")
            if chi2:
                lines.append(f"  Overall persona x need_info: chi2={chi2['chi2']:.2f}, p={chi2['p_value']:.4f}, V={chi2['cramers_v']:.3f}")
            info_rates = pcorr.get("info_rates_by_dimension", {})
            if info_rates:
                lines.append("  Info request rates by dimension:")
                for dim, dim_rates in sorted(info_rates.items()):
                    parts = [f"{k}={v:.3f}" for k, v in sorted(dim_rates.items())]
                    lines.append(f"    {dim}: {', '.join(parts)}")

        # Causal (Outcome)
        causal = mr.get("causal", {})
        if causal:
            lines.append("\nCAUSAL ANALYSIS (Outcome - matched-guise experiment):")
            rates = causal.get("rates_by_persona", {})
            if rates:
                sorted_rates = sorted(rates.items(), key=lambda x: x[1])
                lines.append(f"  Favorable rates: lowest={sorted_rates[0][0]} ({sorted_rates[0][1]:.3f}), highest={sorted_rates[-1][0]} ({sorted_rates[-1][1]:.3f})")
                lines.append(f"  Spread: {sorted_rates[-1][1] - sorted_rates[0][1]:.3f}")
            friedman = causal.get("friedman")
            if friedman:
                lines.append(f"  Friedman test: stat={friedman['statistic']:.2f}, p={friedman['p_value']:.4f}")
            mcnemar_pairs = causal.get("pairwise_mcnemar", [])
            sig_pairs = [p for p in mcnemar_pairs if p.get("significant")]
            lines.append(f"  Pairwise McNemar: {len(sig_pairs)}/{len(mcnemar_pairs)} significant (FDR-adjusted p<0.05)")
            if sig_pairs:
                for sp in sig_pairs[:10]:
                    lines.append(f"    {sp['persona_1']} vs {sp['persona_2']}: rate1={sp['rate_1']:.3f}, rate2={sp['rate_2']:.3f}, p_adj={sp['p_adjusted']:.4f}")

        # Causal (Process)
        pcausal = mr.get("process_causal", {})
        if pcausal:
            lines.append("\nCAUSAL ANALYSIS (Process):")
            info_rates = pcausal.get("info_rates_by_persona", {})
            if info_rates:
                sorted_rates = sorted(info_rates.items(), key=lambda x: x[1])
                lines.append(f"  Need-info rates: lowest={sorted_rates[0][0]} ({sorted_rates[0][1]:.3f}), highest={sorted_rates[-1][0]} ({sorted_rates[-1][1]:.3f})")
            cochran = pcausal.get("cochrans_q")
            if cochran:
                lines.append(f"  Cochran's Q: stat={cochran['statistic']:.2f}, p={cochran['p_value']:.4f}")

        # Mitigation (Outcome)
        mitigation = mr.get("mitigation", {})
        if mitigation:
            lines.append("\nMITIGATION ANALYSIS (Outcome - neutralization effectiveness):")
            mcnemar_pp = mitigation.get("mcnemar_pre_post")
            if mcnemar_pp:
                lines.append(f"  McNemar pre/post: stat={mcnemar_pp['statistic']:.2f}, p={mcnemar_pp['p_value']:.4f}")
            var_orig = mitigation.get("variance_orig")
            var_neut = mitigation.get("variance_neut")
            var_red = mitigation.get("variance_reduction")
            if var_orig is not None:
                lines.append(f"  Variance across personas: original={var_orig:.6f}, neutralized={var_neut:.6f}, reduction={((var_red or 0) * 100):.1f}%")
            tier_changes = mitigation.get("tier_changes", {})
            if tier_changes:
                lines.append(f"  Tier changes (orig->neut): {tier_changes}")
            mit_persona = mitigation.get("mitigation_by_persona", {})
            if mit_persona:
                lines.append("  Bias reduction by persona:")
                for persona, vals in sorted(mit_persona.items()):
                    lines.append(f"    {persona}: orig_AIR={vals['orig_air']:.3f}, neut_AIR={vals['neut_air']:.3f}, reduction={vals['bias_reduction_pct']:.1%}")

        # Mitigation (Process)
        pmit = mr.get("process_mitigation", {})
        if pmit:
            lines.append("\nMITIGATION ANALYSIS (Process):")
            mcnemar_pp = pmit.get("mcnemar_pre_post")
            if mcnemar_pp:
                lines.append(f"  McNemar pre/post: stat={mcnemar_pp['statistic']:.2f}, p={mcnemar_pp['p_value']:.4f}")
            var_red = pmit.get("variance_reduction")
            if var_red is not None:
                lines.append(f"  Variance reduction: {var_red * 100:.1f}%")

        # Repeated-measures causal analysis
        rm = self.analysis_results.get("repeated_measures", {})
        rm_model = rm.get(model, {}) if rm and not rm.get("skipped") else {}
        if rm_model and not rm_model.get("skipped"):
            lines.append("\nREPEATED-MEASURES CAUSAL ANALYSIS (dependence-aware models):")
            lines.append("  Design: same complaint observed under 36 persona framings; complaint_id clustering is non-negotiable.")

            lmm = rm_model.get("linear_mixed_model", {})
            if lmm and not lmm.get("skipped"):
                icc = lmm.get("icc", 0)
                lines.append(f"  LMM ICC={icc:.3f} ({icc*100:.1f}% of tier variance is between complaints)")
                for k, v in lmm.get("coefficients", {}).items():
                    p = lmm.get("p_values", {}).get(k, 1)
                    sig = " *" if p < 0.05 else ""
                    lines.append(f"    {k}: coeff={v:.4f}, p={p:.4f}{sig}")

            om = rm_model.get("ordinal_model", {})
            if om and not om.get("skipped"):
                lines.append(f"  Ordinal model (clustered SEs): n={om.get('n_obs', 0)}, clusters={om.get('n_clusters', 0)}")
                for k, v in om.get("coefficients", {}).items():
                    p = om.get("p_values", {}).get(k, 1)
                    sig = " *" if p < 0.05 else ""
                    lines.append(f"    {k}: coeff={v:.4f}, p={p:.4f}{sig}")

            cl = rm_model.get("conditional_logistic_favourable", {})
            if cl and not cl.get("skipped"):
                lines.append(f"  Conditional logistic: n={cl.get('n_obs', 0)}, strata={cl.get('n_strata', 0)}")
                for k, v in cl.get("odds_ratios", {}).items():
                    p = cl.get("p_values", {}).get(k, 1)
                    sig = " *" if p < 0.05 else ""
                    lines.append(f"    {k}: OR={v:.3f}, p={p:.4f}{sig}")

            np_res = rm_model.get("nonparametric", {})
            if np_res and not np_res.get("skipped"):
                fr = np_res.get("friedman_tier", {})
                if fr and not fr.get("skipped"):
                    lines.append(f"  Friedman (tier, within-complaint): stat={fr.get('statistic', 0):.2f}, p={fr.get('p_value', 1):.4f}")
                cq = np_res.get("cochran_favourable", {})
                if cq and not cq.get("skipped"):
                    lines.append(f"  Cochran's Q (favourable): stat={cq.get('statistic', 0):.2f}, p={cq.get('p_value', 1):.4f}")

        # Info Content
        info = mr.get("info_content", {})
        if info and info.get("n_requests", 0) > 0:
            lines.append(f"\nINFORMATION REQUEST CONTENT: {info['n_requests']} total requests")
            mean_len = info.get("mean_length_by_persona", {})
            if mean_len:
                sorted_len = sorted(mean_len.items(), key=lambda x: x[1], reverse=True)
                for persona, length in sorted_len[:10]:
                    lines.append(f"    {persona}: mean length={length:.0f} chars")

        return "\n".join(lines)

    def _get_llm_commentary(self, model: str, results_text: str) -> Optional[str]:
        """Send analysis results to GPT-5.2 and get commentary on tier bias and process bias."""
        system_prompt = (
            "You are a statistician writing findings paragraphs for a research report "
            "on LLM fairness. The experiment uses a matched-guise design: the same "
            "complaints were rewritten in 36 persona styles (varying age, geographic "
            "setting, education, socioeconomic status) and submitted to the LLM. "
            "Persona estimation also includes gender and ESL status. Neutralized "
            "versions were tested to see if removing persona signals reduces disparities.\n\n"
            "Your commentary field must contain exactly TWO completed paragraphs:\n"
            "Paragraph 1 - TIER BIAS: State whether the LLM gives different remedy "
            "tiers by persona. Cite specific chi2/p/V values, odds ratios, and AIR "
            "values from the data. Name which dimensions and personas are affected.\n"
            "Paragraph 2 - PROCESS BIAS: State whether different personas face "
            "different information request rates. Cite specific rates and test results.\n\n"
            "CRITICAL: Your output must read like finished prose from a statistics "
            "report, e.g. 'The analysis revealed significant tier disparities...' "
            "NOT like a plan, e.g. 'Will analyze tier bias...' or 'Compute the...'"
        )
        user_prompt = (
            f"Here are the analysis results for model {model}. Write your two "
            f"findings paragraphs now, citing specific numbers.\n\n{results_text}"
            f"\n\nWrite the two paragraphs now. Every sentence must cite at least "
            f"one number from the data above."
        )

        req = LLMRequest(
            request_id=LLMRequest.new_id(),
            case_id="commentary",
            model=self.COMMENTARY_MODEL,
            task="commentary",
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            expected_schema=CommentaryResponse,
            variant="commentary",
        )

        try:
            resp = self.llm.call_with_cache(req, results_text)
            if resp.success and resp.response_parsed:
                return resp.response_parsed.commentary
            logger.warning("Commentary request failed for %s: %s", model, resp.error_message)
        except Exception as e:
            logger.warning("Commentary request failed for %s: %s", model, e)
        return None

    # ------------------------------------------------------------------
    # Data capture helpers for executive summary sections
    # ------------------------------------------------------------------

    def _capture_step1_data(self) -> str:
        """Capture step 1 persona estimate summaries as a string for exec summary input."""
        if not self.persona_estimates:
            return ""
        df = pd.DataFrame([
            {"age": pe.age, "geographic": pe.geographic, "education": pe.education,
             "socioeconomic": pe.socioeconomic, "gender": pe.gender, "esl": pe.esl,
             "confidence": pe.confidence,
             "model": pe.model, "persona_label": pe.persona_label}
            for pe in self.persona_estimates
        ])
        lines: List[str] = []
        for model in self.models:
            m = df[df["model"] == model]
            lines.append(f"Model: {model} ({len(m)} estimates)")
            for dim in ESTIMATE_DIMS:
                vc = m[dim].value_counts().sort_index()
                lines.append(f"── {dim} distribution ──")
                for k, v in vc.items():
                    lines.append(f"  {k}  {v}")
            # Full persona label counts
            vc = m["persona_label"].value_counts()
            all_personas = vc.reindex(PERSONA_LABELS, fill_value=0).sort_index()
            present = int((all_personas > 0).sum())
            lines.append(f"── Full persona type counts ({present}/{len(PERSONA_LABELS)} present) ──")
            for k, v in all_personas.items():
                lines.append(f"  {k:<40s} {int(v)}")
            # Confidence histogram
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
            labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
            hist = pd.cut(m["confidence"], bins=bins, labels=labels, right=False).value_counts().sort_index()
            lines.append("── Confidence histogram ──")
            for k, v in hist.items():
                lines.append(f"  {k}  {v}")
            lines.append("")
        return "\n".join(lines)

    def _capture_step3_data(self) -> str:
        """Capture step 3 cross-model commentary text for exec summary input."""
        return self.analysis_results.get("step3_commentary", "")

    def _capture_step5_data(self) -> str:
        """Capture step 5 persona decision summaries as a string for exec summary input."""
        if not self.persona_decisions:
            return ""
        df = pd.DataFrame([
            {"variant": d.variant, "tier": d.tier, "confidence": d.confidence, "model": d.model}
            for d in self.persona_decisions
        ])
        lines: List[str] = []
        for model in self.models:
            m = df[df["model"] == model]
            if m.empty:
                continue
            lines.append(f"Model: {model} ({len(m)} persona decisions)")
            parsed = m["variant"].apply(parse_persona_label)
            valid = parsed.notna()
            if valid.any():
                m = m.copy()
                parsed_df = pd.DataFrame(parsed[valid].tolist(), index=parsed[valid].index)
                for col in ["age", "geographic", "education", "socioeconomic"]:
                    if col in parsed_df.columns:
                        m.loc[parsed_df.index, col] = parsed_df[col]
                # Top 10 persona × tier crosstab
                ct = pd.crosstab(m["variant"], m["tier"])
                top10 = ct.sum(axis=1).nlargest(10).index
                ct_top = ct.loc[top10]
                lines.append("── Top 10 persona × tier crosstab ──")
                header = f"  {'persona':<35s}" + "".join(f"tier_{c:<6}" for c in ct_top.columns)
                lines.append(header)
                for persona in ct_top.index:
                    row = f"  {persona:<35s}" + "".join(f"{ct_top.loc[persona, c]:<8}" for c in ct_top.columns)
                    lines.append(row)
                # Mean tier by dimension
                for dim in CAUSAL_DIMS:
                    if dim in m.columns:
                        means = m.groupby(dim)["tier"].mean()
                        lines.append(f"── Mean tier by {dim} ──")
                        for k, v in sorted(means.items()):
                            lines.append(f"  {k}  {v:.3f}")
            # Overall tier distribution
            vc = m["tier"].value_counts().sort_index()
            lines.append("── Overall tier distribution ──")
            for k, v in vc.items():
                lines.append(f"  {k}  {v}")
            # Process discrimination rate
            need_info = (m["confidence"] == "need_more_info").sum()
            info_rate = need_info / len(m) if len(m) > 0 else 0.0
            lines.append("── Process Discrimination Rate ──")
            lines.append(f"  Info request rate: {need_info}/{len(m)} ({info_rate:.3f})")
            if valid.any():
                for dim in CAUSAL_DIMS:
                    if dim in m.columns:
                        dim_rates = m.groupby(dim)["confidence"].apply(
                            lambda s: (s == "need_more_info").mean()
                        )
                        parts_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(dim_rates.items()))
                        lines.append(f"  Info rate by {dim}: {parts_str}")
            lines.append("")
        return "\n".join(lines)

    def _capture_step6_data(self) -> str:
        """Capture step 6 neutralization summaries as a string for exec summary input."""
        if not self.neutralized or not hasattr(self, 'neutralized_persona_estimates') or not self.neutralized_persona_estimates:
            return ""
        lines: List[str] = []
        for model in self.models:
            estimates = [e for e in self.neutralized_persona_estimates if e.model == model]
            if not estimates:
                continue
            est_df = pd.DataFrame([
                {"age": e.age, "geographic": e.geographic, "education": e.education,
                 "socioeconomic": e.socioeconomic, "gender": e.gender, "esl": e.esl}
                for e in estimates
            ])
            lines.append(f"Model: {model} ({len(est_df)} neutralized re-estimates)")
            for dim in ESTIMATE_DIMS:
                vc = est_df[dim].value_counts().sort_index()
                lines.append(f"── Neutralized {dim} distribution ──")
                for k, v in vc.items():
                    lines.append(f"  {k}  {v}")
            # Compare to original
            orig_ests = [pe for pe in self.persona_estimates if pe.model == model]
            if orig_ests:
                orig_df = pd.DataFrame([
                    {"age": e.age, "geographic": e.geographic, "education": e.education,
                     "socioeconomic": e.socioeconomic, "gender": e.gender, "esl": e.esl}
                    for e in orig_ests
                ])
                lines.append("Comparison: original vs neutralized persona signal")
                for dim in ESTIMATE_DIMS:
                    orig_entropy = orig_df[dim].value_counts(normalize=True).pipe(
                        lambda s: -(s * np.log2(s.clip(lower=1e-10))).sum()
                    )
                    neut_entropy = est_df[dim].value_counts(normalize=True).pipe(
                        lambda s: -(s * np.log2(s.clip(lower=1e-10))).sum()
                    )
                    lines.append(f"  {dim} entropy: original={orig_entropy:.3f}, neutralized={neut_entropy:.3f}")
            lines.append("")
        return "\n".join(lines)

    def _capture_step7_data(self) -> str:
        """Capture the full analysis results + GPT commentary for bias mitigation exec summary."""
        lines: List[str] = []
        for model in self.models:
            mr = self.analysis_results.get(model, {})
            if not mr:
                continue
            lines.append("━" * 50)
            lines.append(f"Model: {model}")
            lines.append("━" * 50)
            # Reuse the existing formatted results
            results_text = self._format_results_for_commentary(model, mr)
            lines.append(results_text)
            # Include GPT commentary
            commentary = mr.get("gpt_commentary")
            if commentary:
                lines.append("── GPT-5.2 Commentary ──")
                lines.append(f"  {commentary}")
            lines.append("")

        # Add mitigation effectiveness analysis
        mit_eff_data = self._capture_mitigation_effectiveness_data()
        if mit_eff_data:
            lines.append("")
            lines.append("━" * 50)
            lines.append("MITIGATION EFFECTIVENESS DEEP-DIVE")
            lines.append("━" * 50)
            lines.append(mit_eff_data)

        return "\n".join(lines)

    def _capture_bridging_data(self) -> str:
        """Capture text-feature → persona → outcome data for bridging commentary."""
        lines: List[str] = []
        pa = self.analysis_results.get("persona_analysis", {})
        if not pa:
            return ""

        for model in self.models:
            model_pa = pa.get(model, {})
            dims = model_pa.get("dimensions", {})
            if not dims:
                continue
            lines.append(f"── Model: {model} ──")
            lines.append("TOP TEXT FEATURES PREDICTING PERSONA DIMENSIONS:")
            for dim_name, dim_data in dims.items():
                for value, val_data in dim_data.items():
                    top_feats = val_data.get("top_features", [])
                    auc = val_data.get("training_auc", 0)
                    summary = val_data.get("summary", "")
                    feat_str = ", ".join(
                        f"{f['feature_display']} ({f['importance']:.0f})"
                        for f in top_feats[:3]
                    )
                    lines.append(f"  {dim_name}={value}: {feat_str}  (AUC={auc:.3f})")
                    if summary:
                        lines.append(f"    → {summary}")

            # Add tier rates by dimension from correlation analysis
            mr = self.analysis_results.get(model, {})
            corr = mr.get("correlation", {})
            rates = corr.get("rates_by_dimension", {})
            if rates:
                lines.append("TIER 2 RATES BY PERSONA DIMENSION:")
                for dim_name, dim_rates in rates.items():
                    rate_str = ", ".join(f"{v}={r:.3f}" for v, r in dim_rates.items())
                    lines.append(f"  {dim_name}: {rate_str}")
            lines.append("")

        # Add cross-model agreement
        mit_eff = self.analysis_results.get("mitigation_effectiveness", {})
        fia = mit_eff.get("feature_importance_agreement", {})
        if fia:
            lines.append("CROSS-MODEL FEATURE IMPORTANCE AGREEMENT:")
            for pair, corr_val in fia.items():
                lines.append(f"  {pair}: Spearman rho = {corr_val:.3f}")

        return "\n".join(lines)

    def _capture_mitigation_effectiveness_data(self) -> str:
        """Capture mitigation effectiveness analysis for exec summary.

        Covers:
        - Most/least biased personas (from causal persona models)
        - Most/least effective bias mitigation (from neutralization models)
        - Text feature associations with personas
        - Golden example journey summary
        - Supporting example highlights
        """
        mit = self.analysis_results.get("mitigation_effectiveness")
        if not mit or mit.get("error"):
            return ""

        lines: List[str] = ["── Mitigation Effectiveness Analysis ──"]

        # --- Feature importance from neutralization models ---
        lines.append("")
        lines.append("WHAT PREDICTS NEUTRALIZATION EFFECTIVENESS?")
        lines.append("(Higher importance = feature more strongly predicts tier change)")
        for model in self.models:
            neut_models = mit.get("neutralization_models", {}).get(model, [])
            for m in neut_models:
                mtype = m.get("model_type", "unknown")
                metrics = m.get("metrics", {})
                r2 = metrics.get("r2", 0)
                lines.append(f"\n{model} ({mtype}): R²={r2:.3f}")

                # Feature importance
                fi = m.get("feature_importance", [])
                if fi:
                    lines.append("  Top features predicting tier change after neutralization:")
                    for feat in sorted(fi, key=lambda x: x.get("importance", 0), reverse=True)[:8]:
                        fname = feat.get("feature", "")
                        fimp = feat.get("importance", 0)
                        lines.append(f"    {fname}: {fimp:.4f}")

                # Mixed-effects coefficients (if available)
                coeffs = m.get("coefficients")
                if coeffs and mtype == "mixed_effects":
                    lines.append("  Statistical significance (mixed-effects):")
                    for c in coeffs:
                        if c.get("p_value", 1) < 0.05:
                            lines.append(f"    {c.get('feature')}: coef={c.get('coef', 0):.3f}, p={c.get('p_value', 1):.4f} *")

        # --- Causal persona effects (which dimension changes matter most?) ---
        lines.append("")
        lines.append("CAUSAL PERSONA EFFECTS (Matched-Guise Test)")
        lines.append("(Which dimension changes most affect tier assignment?)")
        for model in self.models:
            causal_models = mit.get("causal_persona_models", {}).get(model, [])
            for m in causal_models:
                mtype = m.get("model_type", "unknown")
                metrics = m.get("metrics", {})
                r2 = metrics.get("r2", 0)
                lines.append(f"\n{model} ({mtype}): R²={r2:.3f}")

                fi = m.get("feature_importance", [])
                if fi:
                    lines.append("  Dimension changes ranked by effect on tier:")
                    for feat in sorted(fi, key=lambda x: x.get("importance", 0), reverse=True):
                        fname = feat.get("feature", "")
                        fimp = feat.get("importance", 0)
                        lines.append(f"    {fname}: {fimp:.4f}")

        # --- Heterogeneous Treatment Effects ---
        lines.append("")
        lines.append("HETEROGENEOUS EFFECTS (Who benefits most from neutralization?)")
        for model in self.models:
            hte_list = mit.get("heterogeneous_effects", {}).get(model, [])
            for hte in hte_list:
                treatment = hte.get("treatment", "")
                ate = hte.get("overall_ate", 0)
                ate_ci = hte.get("ate_ci", [0, 0])
                moderators = hte.get("important_moderators", [])

                lines.append(f"\n{model} treatment={treatment}:")
                lines.append(f"  Average Treatment Effect: {ate:.3f} [{ate_ci[0]:.3f}, {ate_ci[1]:.3f}]")
                if moderators:
                    lines.append(f"  Effect varies by: {', '.join(moderators)}")

                # Subgroup CATE
                cate = hte.get("cate_by_subgroup", [])
                if cate:
                    lines.append("  Subgroup effects:")
                    for sg in cate[:5]:  # top 5
                        lines.append(f"    {sg.get('subgroup', '')}: CATE={sg.get('cate', 0):.3f}")

        # --- Golden Example Summary ---
        golden = mit.get("golden_example")
        if golden:
            lines.append("")
            lines.append("GOLDEN EXAMPLE (Single complaint tracked through full pipeline)")
            lines.append(f"  Complaint: {golden.get('complaint_id', '')} ({golden.get('product', '')})")
            lines.append(f"  Estimated persona: {golden.get('estimated_persona', {})}")
            lines.append(f"  Original tiers by model: {golden.get('original_tiers', {})}")
            lines.append(f"  Neutralized tiers by model: {golden.get('neutralized_tiers', {})}")

            tier_var = golden.get("tier_variance_by_model", {})
            neut_eff = golden.get("neutralization_effect_by_model", {})
            for model in self.models:
                var = tier_var.get(model, 0)
                eff = neut_eff.get(model, 0)
                lines.append(f"  {model}: tier variance across personas={var:.3f}, neutralization effect={eff:+.2f}")

            # Persona variants summary
            persona_tiers = golden.get("persona_tiers", {})
            if persona_tiers:
                # Find best and worst personas
                all_tiers = []
                for persona, model_tiers in persona_tiers.items():
                    for m, t in model_tiers.items():
                        all_tiers.append((persona, m, t))
                if all_tiers:
                    best = min(all_tiers, key=lambda x: x[2])
                    worst = max(all_tiers, key=lambda x: x[2])
                    lines.append(f"  Best outcome persona: {best[0]} (tier {best[2]} on {best[1]})")
                    lines.append(f"  Worst outcome persona: {worst[0]} (tier {worst[2]} on {worst[1]})")

        # --- Supporting Examples Summary ---
        examples = mit.get("supporting_examples", [])
        if examples:
            lines.append("")
            lines.append(f"SUPPORTING EXAMPLES ({len(examples)} selected)")
            for ex in examples:
                cat = ex.get("category", "")
                cid = ex.get("complaint_id", "")
                interp = ex.get("interpretation", "")
                lines.append(f"  {cat}: {cid}")
                if interp:
                    lines.append(f"    → {interp[:200]}")

        # --- Cross-Model Agreement ---
        agreement = mit.get("feature_importance_agreement", {})
        if agreement:
            lines.append("")
            lines.append("CROSS-MODEL AGREEMENT (Feature Importance)")
            for pair, corr in agreement.items():
                lines.append(f"  {pair}: Spearman ρ={corr:.3f}")

        return "\n".join(lines)

    def _capture_severity_data(self) -> str:
        """Capture severity analysis results as text for the executive summary section."""
        sev = self.analysis_results.get("severity")
        if not sev:
            return ""
        lines: List[str] = ["── Severity Analysis Summary ──"]

        # --- Descriptives ---
        desc = sev.get("descriptives", {})
        lines.append("Severity Metrics Overview:")
        for metric in ["monetary_severity", "issue_severity"]:
            md = desc.get(metric, {})
            dist = md.get("distribution", {})
            cov = md.get("coverage")
            if dist:
                dist_str = ", ".join(f"{k}={v}" for k, v in dist.items())
                lines.append(f"  {metric} distribution: {dist_str}")
            if cov is not None:
                lines.append(f"  {metric} coverage: {cov:.1%}")

        agreement = desc.get("agreement", {})
        if agreement:
            lines.append("")
            lines.append("  Agreement between metrics (where both available):")
            for key in ["kappa", "spearman_r", "agreement_rate"]:
                val = agreement.get(key)
                if val is not None:
                    lines.append(f"    {key}: {val:.3f}" if isinstance(val, float) else f"    {key}: {val}")

        # --- Main effects per model ---
        main_fx = sev.get("main_effects", {})
        for model in self.models:
            lines.append("")
            lines.append(f"Model: {model}")
            lines.append("── Main Effects ──")
            for metric in ["monetary_severity", "issue_severity"]:
                mfx = main_fx.get(metric, {}).get(model, {})
                means = mfx.get("means_by_level", {})
                if means:
                    means_str = ", ".join(f"{k}={v:.3f}" for k, v in means.items())
                    lines.append(f"  {metric}: mean tier by level: {means_str}")
                anova = mfx.get("anova", {})
                if anova:
                    lines.append(f"    ANOVA: F={anova.get('F', 0):.2f}, p={anova.get('p_value', 1):.4f}, "
                                 f"partial eta-sq={anova.get('eta_sq_partial', 0):.3f} "
                                 f"({anova.get('eta_sq_interpretation', 'N/A')})")
                trend = mfx.get("trend", {})
                if trend:
                    lines.append(f"    Trend: Spearman r={trend.get('spearman_r', 0):.3f}, "
                                 f"p={trend.get('p_value', 1):.4f}")

            # --- Amplification ---
            interactions = sev.get("interactions", {})
            lines.append("")
            lines.append("── Amplification Analysis ──")
            for metric in ["monetary_severity", "issue_severity"]:
                mi = interactions.get(metric, {}).get(model, {})
                lines.append(f"  {metric}:")

                bias = mi.get("bias", {})
                for level in ["low", "medium", "high"]:
                    bl = bias.get(level, {})
                    sd = bl.get("sd_across_personas", 0)
                    lines.append(f"    Bias at {level} severity (SD across personas): {sd:.3f}")

                amp = mi.get("amplification", {})
                ratio = amp.get("point", float("nan"))
                ci_lo = amp.get("ci_lower", float("nan"))
                ci_hi = amp.get("ci_upper", float("nan"))
                lines.append(f"    Amplification ratio: {ratio:.2f}x (CI: {ci_lo:.2f}, {ci_hi:.2f})")

                tw = mi.get("two_way", {})
                inter = tw.get("interaction", {})
                if inter:
                    lines.append(f"    Two-way interaction (persona x severity): "
                                 f"F={inter.get('F', 0):.2f}, p={inter.get('p_value', 1):.4f}, "
                                 f"eta-sq={inter.get('eta_sq_partial', 0):.3f}")

        # --- Worst case ---
        wc = sev.get("three_way", {}).get("worst_case", {})
        if wc and wc.get("model"):
            lines.append("")
            lines.append("── Worst Case ──")
            lines.append(f"  Model: {wc.get('model')}, Metric: {wc.get('severity_metric')}, "
                         f"Level: {wc.get('severity_level')}")
            lines.append(f"  Most advantaged persona: {wc.get('most_advantaged')}")
            lines.append(f"  Most disadvantaged persona: {wc.get('most_disadvantaged')}")
            lines.append(f"  Effect size (Cohen's d): {wc.get('effect_size_d', 0):.2f} "
                         f"({wc.get('effect_size_interpretation', 'N/A')})")

        # --- Process ---
        proc = sev.get("process", {})
        if proc:
            lines.append("")
            lines.append("── Process Fairness ──")
            for model in self.models:
                parts = []
                for metric in ["monetary_severity", "issue_severity"]:
                    pm = proc.get(metric, {}).get(model, {})
                    disp = pm.get("disparity_by_severity", {})
                    if disp:
                        disp_str = ", ".join(f"{k}={v:.3f}" for k, v in disp.items())
                        parts.append(f"{metric}: {disp_str}")
                if parts:
                    lines.append(f"  {model}: {'; '.join(parts)}")

        # --- Consistency ---
        consist = sev.get("consistency", {})
        findings = consist.get("findings", [])
        if findings:
            lines.append("")
            lines.append("── Consistency ──")
            for f in findings:
                p_m = f.get("monetary_p", f.get("monetary_value", "N/A"))
                p_i = f.get("issue_p", f.get("issue_value", "N/A"))
                rob = f.get("robustness", "")
                finding = f.get("finding", "")
                if isinstance(p_m, float):
                    p_m = f"{p_m:.4f}"
                if isinstance(p_i, float):
                    p_i = f"{p_i:.4f}"
                lines.append(f"  {finding}: monetary={p_m}, issue={p_i} — {rob}")

        # --- Missingness diagnostic ---
        miss_diag = sev.get("missingness_diagnostic", {})
        if miss_diag and not miss_diag.get("skipped"):
            lines.append("")
            lines.append("── Missingness Diagnostic ──")
            n_pres = miss_diag.get("n_present", 0)
            n_miss = miss_diag.get("n_missing", 0)
            pct_miss = miss_diag.get("pct_missing", 0)
            lines.append(f"  Complaints with extractable dollar amount: {n_pres} ({1 - pct_miss:.1%})")
            lines.append(f"  Complaints without dollar amount: {n_miss} ({pct_miss:.1%})")
            chars = miss_diag.get("complaint_characteristics", {})
            m_type = chars.get("missingness_type", "unknown")
            n_sig = chars.get("n_significant_tests", 0)
            sig_tests = chars.get("significant_tests", [])
            sig_str = ", ".join(sig_tests) if sig_tests else "none"
            lines.append(f"  Missingness pattern: {m_type} ({n_sig}/5 characteristic tests significant)")
            if sig_tests:
                lines.append(f"  Significant differences: {sig_str}")
            for model in self.models:
                fc = miss_diag.get("fairness_comparison", {}).get(model, {})
                pres = fc.get("present", {})
                miss_stats = fc.get("missing", {})
                ratio = fc.get("bias_sd_ratio", float("nan"))
                lines.append(
                    f"  {model}: bias SD with amounts={pres.get('bias_sd', 0):.4f}, "
                    f"without amounts={miss_stats.get('bias_sd', 0):.4f}, ratio={ratio:.2f}"
                )
                lines.append(
                    f"    sig pairs: present={pres.get('n_significant_pairs', 0)}, "
                    f"missing={miss_stats.get('n_significant_pairs', 0)}"
                )
                lines.append(f"    Conclusion: {fc.get('conclusion', 'N/A')}")
            lines.append(f"  Overall: {miss_diag.get('overall_conclusion', 'N/A')}")

        # --- Mitigation × severity ---
        mit_sev = sev.get("mitigation_severity", {})
        if mit_sev:
            lines.append("")
            lines.append("── Mitigation × Severity ──")
            for metric in ["monetary_severity", "issue_severity"]:
                for model in self.models:
                    ms = mit_sev.get(metric, {}).get(model, {})
                    comp = ms.get("comparison", {})
                    verdict = comp.get("verdict", "N/A")
                    base_amp = ms.get("baseline", {}).get("amplification", {})
                    neut_amp = ms.get("neutralized", {}).get("amplification", {})
                    base_ratio = base_amp.get("point", float("nan"))
                    neut_ratio = neut_amp.get("point", float("nan"))
                    delta = comp.get("amplification_change", float("nan"))
                    pct = comp.get("amplification_pct_change", float("nan"))
                    lines.append(
                        f"  {metric} [{model}]: verdict={verdict}, "
                        f"baseline={base_ratio:.2f}x, neutralized={neut_ratio:.2f}x, "
                        f"delta={delta:+.2f} ({pct:+.1f}%)"
                    )
                    base_bias = ms.get("baseline", {}).get("bias", {})
                    neut_bias = ms.get("neutralized", {}).get("bias", {})
                    sd_changes = comp.get("bias_sd_change_by_level", {})
                    for level in ["low", "medium", "high"]:
                        b_sd = base_bias.get(level, {}).get("sd_across_personas", 0)
                        n_sd = neut_bias.get(level, {}).get("sd_across_personas", 0)
                        d_sd = sd_changes.get(level, 0)
                        lines.append(
                            f"    {level}: bias SD {b_sd:.4f} → {n_sd:.4f} ({d_sd:+.4f})"
                        )
                    int_p_base = comp.get("interaction_p_baseline", float("nan"))
                    int_p_neut = comp.get("interaction_p_neutralized", float("nan"))
                    weakened = comp.get("interaction_weakened", False)
                    lines.append(
                        f"    Interaction p: baseline={int_p_base:.4f}, "
                        f"neutralized={int_p_neut:.4f}, weakened={weakened}"
                    )

        # --- Dimension-specific severity amplification ---
        dim_sev = sev.get("dimension_severity", {})
        if dim_sev:
            lines.append("")
            lines.append("── Dimension-Specific Severity Amplification (One-Way Tests) ──")
            lines.append("(Tests each demographic dimension individually against severity — "
                         "each cell has ~200+ observations for much higher statistical power)")
            for metric in ["monetary_severity", "issue_severity"]:
                for model in self.models:
                    for dim in CAUSAL_DIMS:
                        ds = dim_sev.get(metric, {}).get(model, {}).get(dim, {})
                        if "error" in ds:
                            continue
                        amp = ds.get("amplification", {})
                        anova = ds.get("anova", {})
                        inter = anova.get("interaction", {})
                        lines.append(
                            f"  {metric} [{model}] {dim}: "
                            f"interaction F={inter.get('F', 0):.2f}, p={inter.get('p_value', 1):.4f}, "
                            f"eta-sq={inter.get('eta_sq_partial', 0):.4f} ({inter.get('eta_sq_interpretation', '')}), "
                            f"amplification={amp.get('point', 0):.2f}x "
                            f"[{amp.get('ci_lower', 0):.2f}, {amp.get('ci_upper', 0):.2f}]"
                        )
                        # Show dimension-level means at each severity
                        means_by_sev = ds.get("dimension_means_by_severity", {})
                        for level in ["low", "high"]:
                            means = means_by_sev.get(level, {})
                            if means:
                                means_str = ", ".join(f"{k}={v:.3f}" for k, v in means.items())
                                lines.append(f"    {level}: {means_str}")

        # --- Power analysis ---
        pa = self.analysis_results.get("power_analysis")
        if pa:
            lines.append("")
            lines.append("── Power Analysis (Sample Size Justification) ──")
            mg = pa.get("matched_guise", {})
            lines.append(f"Matched-guise design ({mg.get('n_pairs', 0)} complaint pairs, within-complaint):")
            lines.append(f"  Min detectable Cohen's d: {mg.get('min_detectable_d', 0):.3f} ({mg.get('interpretation', '')})")
            bp = pa.get("between_persona", {})
            for dim, info in bp.items():
                lines.append(f"Between-persona ({dim}, n={info['n_per_cell']} per cell):")
                lines.append(f"  Min detectable d: {info['min_detectable_d']:.3f} ({info['interpretation']})")
            av = pa.get("anova", {})
            for dim, info in av.items():
                lines.append(f"ANOVA ({dim}, k={info['k_groups']}, n={info['n_per_group']} per group):")
                lines.append(f"  Min detectable f: {info['min_detectable_f']:.3f}, "
                             f"eta-sq: {info['min_detectable_eta_sq']:.4f} ({info['interpretation']})")

        # --- Multiple comparisons ---
        mc = self.analysis_results.get("multiple_comparisons")
        if mc:
            lines.append("")
            lines.append("── Multiple Comparisons Correction (Family-Wise Error Rate) ──")
            lines.append(f"Method: {mc['method']}")
            lines.append(f"Total tests across all families: {mc['total_tests']}")
            lines.append(f"Total surviving FDR correction at alpha=0.05: {mc['total_surviving']}")
            for fam in mc.get("families", []):
                lines.append(f"  {fam['family']}: {fam['n_surviving']}/{fam['n_tests']} survive "
                             f"({fam['survival_rate']:.0%})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Executive Summary section generator
    # ------------------------------------------------------------------

    def _generate_exec_section(
        self,
        section_name: str,
        one_shot_file: str,
        data_text: str,
        context_description: str,
    ) -> Optional[str]:
        """Generate one executive summary section via GPT-5.2 using a one-shot example."""
        # Read the one-shot example file
        one_shot_path = Path("one-shot-examples") / one_shot_file
        try:
            one_shot_content = one_shot_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("One-shot example not found: %s", one_shot_path)
            return None

        # Parse input/output sections from the one-shot file
        parts = one_shot_content.split("\noutput:\n", 1)
        if len(parts) == 2:
            example_input = parts[0].split("\ninput:\n", 1)[-1].strip()
            example_output = parts[1].strip()
        else:
            example_input = ""
            example_output = ""

        system_prompt = (
            "You are a communications specialist who translates technical statistical "
            "findings into clear, actionable language for business executives. "
            + context_description + "\n\n"
            "CORE EXPERIMENTAL QUESTION:\n"
            "The experiment tests ONE key question: Do LLMs make different decisions "
            "based on linguistic style when those decisions should be identical? "
            "In a fair decision-making system, the remedy should depend only on the "
            "complaint substance, not on how it is written. Any difference in decisions "
            "by linguistic style is potential bias.\n\n"
            "WHAT TO REPORT:\n"
            "1. Whether bias exists (decisions differ by linguistic style)\n"
            "2. If bias exists, does it vary by implicit persona, severity, or LLM model?\n"
            "3. Can this bias be mitigated?\n\n"
            "AVOID THESE COMMON ERRORS (CRITICAL - DO NOT INCLUDE ANY OF THESE):\n"
            "- Do NOT claim that neutralization 'collapses diverse voices' or creates a "
            "'stereotyped default' — standardizing language is the intended behaviour, "
            "not a problem or risk.\n"
            "- Do NOT speculate about 'downstream fairness implications' of neutralization "
            "— the experiment tests whether neutralization WORKS, not hypothetical risks.\n"
            "- Do NOT claim that estimating a 'default persona' is problematic — the "
            "experiment defaults text to the most likely persona, which is expected.\n"
            "- Do NOT claim that 'some groups are invisible' or 'underrepresented groups "
            "may be mischaracterized' — in a fair system, group identification should "
            "NOT affect outcomes, so misclassification is irrelevant.\n"
            "- Do NOT claim that 'models are confident about demographics they cannot "
            "truly know' — this is irrelevant; the key point is whether decisions change.\n"
            "- Do NOT make claims about whether patterns 'reflect the complaint dataset's "
            "writing style' — the matched-guise design controls for this.\n"
            "- Do NOT add 'balanced' commentary suggesting risks of the mitigation approach "
            "— report what the data shows, not hypothetical concerns.\n\n"
            "Rules:\n"
            "- Use plain business English; avoid jargon like 'chi-squared', 'p-value', "
            "'Cramer\\'s V', 'odds ratio', 'AIR', 'FDR', 'McNemar', 'Friedman', "
            "'Cochran\\'s Q', 'logistic regression', 'omnibus test'.\n"
            "- Translate statistical significance into phrases like 'no clear evidence', "
            "'statistically meaningful', 'not large enough to rule out chance'.\n"
            "- Organise with clear headings and bullet points.\n"
            "- Keep percentages and concrete comparisons (e.g. '55% vs 73%') but "
            "remove test statistics.\n"
            "- Be honest about limitations and what is inconclusive.\n"
            "- When discussing severity, be clear about whether you mean monetary "
            "severity (dollar amounts) or issue severity (product/issue category).\n\n"
            "ONE-SHOT EXAMPLE\n"
            "────────────────\n"
            f"INPUT:\n{example_input}\n\n"
            f"OUTPUT:\n{example_output}\n"
            "────────────────\n\n"
            "Now produce the same style of executive summary section for the data "
            "provided by the user. Do NOT copy the example output; write fresh "
            "content that accurately reflects the actual data."
        )

        user_prompt = (
            "Here is the data for this section. Write the executive summary "
            "section now.\n\n" + data_text
        )

        req = LLMRequest(
            request_id=LLMRequest.new_id(),
            case_id=f"exec_{section_name}",
            model=self.COMMENTARY_MODEL,
            task="executive_summary",
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            expected_schema=CommentaryResponse,
            variant=f"exec_{section_name}",
        )

        try:
            resp = self.llm.call_with_cache(req, data_text)
            if resp.success and resp.response_parsed:
                return resp.response_parsed.commentary
            logger.warning("Exec section '%s' request failed: %s", section_name, resp.error_message)
        except Exception as e:
            logger.warning("Exec section '%s' request failed: %s", section_name, e)
        return None

    def _generate_all_exec_sections(self) -> Dict[str, str]:
        """Generate all 6 executive summary sections sequentially, returning {name: text}."""
        sections: Dict[str, str] = {}

        # Section 1: Linguistic Differences (Step 1 data)
        logger.info("── Generating Exec Section 1: Linguistic Differences ──")
        data = self._capture_step1_data()
        if data:
            text = self._generate_exec_section(
                "linguistic_differences",
                "one-shot-example-linguistic-differences.md",
                data,
                "We asked ChatGPT to estimate customer demographics from complaint text. "
                "This establishes whether the LLM infers implicit personas from writing "
                "style. Note: the key question is NOT whether these estimates are accurate "
                "or complete, but whether subsequent decisions differ by inferred persona.",
            )
            if text:
                sections["linguistic_differences"] = text
                logger.info("  Section 1 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 1 generation failed")

        # Section 2: AI Decision-Making (Step 3 cross-model commentary)
        logger.info("── Generating Exec Section 2: AI Decision-Making ──")
        data = self._capture_step3_data()
        if data:
            text = self._generate_exec_section(
                "ai_decision_making",
                "one-shot-example-ai-decision-making.md",
                data,
                "We asked ChatGPT to recommend remedial action based on complaints, then "
                "linked decisions to inferred personas. The key question: do remedy "
                "decisions correlate with inferred persona? In a fair system, remedies "
                "should depend only on complaint substance, not writing style.",
            )
            if text:
                sections["ai_decision_making"] = text
                logger.info("  Section 2 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 2 generation failed")

        # Section 3: Correlation vs. Cause (Step 5 data)
        logger.info("── Generating Exec Section 3: Correlation vs. Cause ──")
        data = self._capture_step5_data()
        if data:
            text = self._generate_exec_section(
                "correlation_vs_cause",
                "one-shot-example-correlation-vs-cause.md",
                data,
                "We used a matched-guise design: the SAME complaint was rewritten in 36 "
                "different persona styles, then each version was sent for remedy decision. "
                "This is the causal test: if decisions differ by writing style when the "
                "underlying complaint is identical, that proves bias exists. Focus on "
                "the size of any differences and whether they are statistically meaningful.",
            )
            if text:
                sections["correlation_vs_cause"] = text
                logger.info("  Section 3 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 3 generation failed")

        # Section 4: Severity Stakes (severity analysis)
        logger.info("── Generating Exec Section 4: Severity Stakes ──")
        data = self._capture_severity_data()
        if data:
            text = self._generate_exec_section(
                "severity_stakes",
                "one-shot-example-severity-stakes.md",
                data,
                "We analysed whether any detected bias varies by complaint severity. "
                "IMPORTANT: there are TWO severity metrics that may show different "
                "patterns: (1) monetary severity (dollar amounts in dispute), and "
                "(2) issue severity (product/issue category). Be explicit about which "
                "metric you are discussing. If one shows amplification and the other "
                "does not, state this clearly without contradiction.",
            )
            if text:
                sections["severity_stakes"] = text
                logger.info("  Section 4 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 4 generation failed")

        # Section 5: Language Neutralisation (Step 6 data)
        logger.info("── Generating Exec Section 5: Language Neutralisation ──")
        data = self._capture_step6_data()
        if data:
            text = self._generate_exec_section(
                "language_neutralisation",
                "one-shot-example-language-neutralisation.md",
                data,
                "We tested whether rewriting complaints to neutral language removes "
                "demographic signals from the text. This is INTENDED behaviour: the "
                "goal is to standardize language so that downstream decisions cannot "
                "be influenced by writing style. Success means all complaints look "
                "similar after neutralization. Do NOT frame this as 'collapsing voices' "
                "or 'creating a default persona' — standardization is the mitigation.",
            )
            if text:
                sections["language_neutralisation"] = text
                logger.info("  Section 5 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 5 generation failed")

        # Section 6: Bias Mitigation (Step 7 analysis + commentary)
        logger.info("── Generating Exec Section 6: Bias Mitigation ──")
        data = self._capture_step7_data()
        if data:
            text = self._generate_exec_section(
                "bias_mitigation",
                "one-shot-example-bias-mitigation.md",
                data,
                "We tested whether neutralizing complaint language before remedy "
                "decisions eliminates any detected bias. This is the key mitigation "
                "test: if decisions become uniform after neutralization, the "
                "mitigation works. Focus on whether outcome variance decreases and "
                "whether any statistically significant persona differences remain.",
            )
            if text:
                sections["bias_mitigation"] = text
                logger.info("  Section 6 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 6 generation failed")

        # Section 7: Bridging Commentary (text → persona → outcome)
        logger.info("── Generating Exec Section 7: Bridging Commentary ──")
        data = self._capture_bridging_data()
        if data:
            text = self._generate_exec_section(
                "bridging_text_to_outcome",
                "one-shot-example-bridging-commentary.md",
                data,
                "This section connects three layers of evidence from the experiment: "
                "(i) observable text characteristics that predict persona assignment, "
                "(ii) the persona labels themselves, and (iii) how those labels relate "
                "to remedy tier outcomes and tier dispersion. Use association / consistency "
                "language (not causal claims). Keep it to 2-3 paragraphs. Reference "
                "specific text features, AUC scores, and tier rates where available.",
            )
            if text:
                sections["bridging_text_to_outcome"] = text
                logger.info("  Section 7 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 7 generation failed")

        # Section 8: Conclusion (uses all prior sections)
        logger.info("── Generating Exec Section 8: Conclusion ──")
        if sections:
            combined_sections = "\n\n".join(sections.values())
            text = self._generate_exec_section(
                "conclusion",
                "one-shot-examples-conclusion.md",
                combined_sections,
                "Write a half-page Plain English executive summary, suitable for a "
                "non-technical business executive audience. Answer these questions: "
                "(1) Does linguistic style bias exist in LLM complaint decisions? "
                "(2) If so, does bias vary by persona, severity metric, or LLM model? "
                "(3) Does the neutralization mitigation eliminate detected bias? "
                "Focus on actionable findings. Avoid tangential commentary about "
                "'invisible groups', 'default personas', or persona estimation accuracy.",
            )
            if text:
                sections["conclusion"] = text
                logger.info("  Section 8 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 8 generation failed")

        return sections

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------

    def _save_artifacts(self) -> None:
        """Save CSVs, JSON stats, and failed requests."""
        ts = self.timestamp

        # Outcome results CSV
        if self.baseline_df is not None:
            self.baseline_df.to_csv(self.output_dir / f"results_outcome_{ts}.csv", index=False)

        # Persona decisions CSV
        if self.persona_decisions_df is not None:
            self.persona_decisions_df.to_csv(self.output_dir / f"results_persona_decisions_{ts}.csv", index=False)

        # Neutralized results CSV
        if self.neutralized_df is not None:
            self.neutralized_df.to_csv(self.output_dir / f"results_neutralized_{ts}.csv", index=False)

        # Dollar amounts CSV
        if self.dollar_amounts:
            da_df = pd.DataFrame([
                {
                    "case_id": da.case_id,
                    "primary_amount": da.primary_amount,
                    "amount_type": da.amount_type,
                    "confidence": da.confidence,
                    "amounts_mentioned": json.dumps(da.amounts_mentioned) if da.amounts_mentioned else None,
                }
                for da in self.dollar_amounts
            ])
            da_df.to_csv(self.output_dir / f"dollar_amounts_{ts}.csv", index=False)

        # Personas CSV
        if self.personas_df is not None:
            self.personas_df.to_csv(self.output_dir / f"results_personas_{ts}.csv", index=False)

        # Text features CSV
        if self.text_features_df is not None:
            self.text_features_df.to_csv(self.output_dir / f"text_features_{ts}.csv", index=False)

        # LLM-rated text quality ratings CSV
        if self.text_quality_df is not None:
            self.text_quality_df.to_csv(self.output_dir / f"text_quality_ratings_{ts}.csv", index=False)

        # Repair stats
        repair_stats = self.llm.get_repair_stats()
        (self.output_dir / f"repair_stats_{ts}.json").write_text(
            json.dumps(repair_stats, indent=2), encoding="utf-8"
        )

        # Full analysis results JSON (mirrors every metric in the HTML dashboard)
        if self.analysis_results:
            json_path = self.output_dir / f"analysis_results_{ts}.json"
            json_path.write_text(
                json.dumps(self.analysis_results, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Analysis results JSON written: %s", json_path)

        # Executive summary text file
        exec_sections = self.analysis_results.get("exec_sections", {})
        if exec_sections:
            section_titles = {
                "linguistic_differences": "SECTION 1: LINGUISTIC DIFFERENCES",
                "ai_decision_making": "SECTION 2: AI DECISION-MAKING",
                "correlation_vs_cause": "SECTION 3: CORRELATION VS. CAUSE",
                "severity_stakes": "SECTION 4: SEVERITY STAKES",
                "language_neutralisation": "SECTION 5: LANGUAGE NEUTRALISATION",
                "bias_mitigation": "SECTION 6: BIAS MITIGATION",
                "conclusion": "SECTION 7: CONCLUSION",
            }
            lines: List[str] = []
            lines.append("=" * 60)
            lines.append("BUSINESS EXECUTIVE SUMMARY (7 SECTIONS)")
            lines.append("=" * 60)
            for key, title in section_titles.items():
                text = exec_sections.get(key)
                if text:
                    lines.append("")
                    lines.append("\u2500" * 50)
                    lines.append(title)
                    lines.append("\u2500" * 50)
                    lines.append(text)
            lines.append("")
            lines.append("=" * 60)
            summary_path = self.output_dir / f"executive_summary_{self.timestamp}.txt"
            summary_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Executive summary written to %s", summary_path)

        # Debiased consistency Excel workbook
        debiased = self.analysis_results.get("debiased_consistency")
        if debiased:
            self._save_debiased_excel(debiased)

        # Failed requests
        # (collected from LLM responses with success=False — tracked internally)

        logger.info("Artifacts saved to %s", self.output_dir)

    def _save_debiased_excel(self, debiased: Dict[str, Any]) -> None:
        """Save debiased consistency results as Excel workbook."""
        xlsx_path = self.output_dir / f"debias-analysis-{self.timestamp}.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                consistency = debiased.get("consistency", [])
                if consistency:
                    pd.DataFrame(consistency).to_excel(
                        writer, sheet_name="Extraction Consistency", index=False,
                    )
                cross_model = debiased.get("cross_model", [])
                if cross_model:
                    pd.DataFrame(cross_model).to_excel(
                        writer, sheet_name="Cross-Model Comparison", index=False,
                    )
                by_dim = debiased.get("by_dimension", [])
                if by_dim:
                    pd.DataFrame(by_dim).to_excel(
                        writer, sheet_name="Consistency by Dimension", index=False,
                    )
                cross_dim = debiased.get("cross_model_by_dimension", [])
                if cross_dim:
                    pd.DataFrame(cross_dim).to_excel(
                        writer, sheet_name="Cross-Model by Dimension", index=False,
                    )
                worked = debiased.get("worked_example", [])
                if worked:
                    # Flatten the nested vals dict for Excel
                    flat_rows = []
                    for row in worked:
                        flat = {"variant": row["variant"], "text": row.get("text", "")}
                        for model, vals in row.get("vals", {}).items():
                            for k, v in vals.items():
                                flat[k] = v
                        flat_rows.append(flat)
                    pd.DataFrame(flat_rows).to_excel(
                        writer, sheet_name="Worked Example", index=False,
                    )
            logger.info("Debiased Excel report saved: %s", xlsx_path)
        except Exception as exc:
            logger.warning("Failed to save debiased Excel report: %s", exc)
