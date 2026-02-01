"""Step orchestration for the 7-step LLM fairness experiment."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cache import ResponseCache
from src.data_loader import load_and_clean
from src.llm_client import LLMClient
from src.severity_analysis import run_severity_analysis
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
    format_persona_rewrite_user,
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
    PERSONA_GRID,
    PERSONA_LABELS,
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
        verbose: bool = False,
        api_key: str = "",
    ) -> None:
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
        self.cache = ResponseCache(cache_dir)
        if clear_cache:
            self.cache.clear()
            # Also clear the data pipeline cache
            pipeline_dir = Path(data_path).parent / ".pipeline"
            if pipeline_dir.exists():
                import shutil
                shutil.rmtree(pipeline_dir)
                logger.info("Pipeline cache cleared: %s", pipeline_dir)

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

        # DataFrames built during analysis
        self.personas_df: Optional[pd.DataFrame] = None
        self.baseline_df: Optional[pd.DataFrame] = None
        self.persona_decisions_df: Optional[pd.DataFrame] = None
        self.neutralized_df: Optional[pd.DataFrame] = None
        self.text_features_df: Optional[pd.DataFrame] = None

        # Analysis results
        self.analysis_results: Dict[str, Any] = {}

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full experiment pipeline."""
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
        fit_topic_model([c.narrative for c in self.complaints], n_topics=10, seed=self.seed)

        # Extract dollar amounts (cached, runs once)
        logger.info("-" * 40)
        logger.info("Dollar amount extraction starting")
        self._extract_dollar_amounts()
        logger.info("Dollar amount extraction complete")

        # Run steps
        for step in range(1, 8):
            if step in self.skip_steps:
                logger.info("Skipping Step %d", step)
                continue
            logger.info("-" * 40)
            logger.info("Step %d starting", step)
            getattr(self, f"_step_{step}")()
            logger.info("Step %d complete", step)

        # Build analysis
        self._run_analyses()

        # Save artifacts
        self._save_artifacts()

        logger.info("=" * 60)
        logger.info("Experiment complete. Output: %s", self.output_dir)
        logger.info(self.cache.stats_summary())
        return self.analysis_results

    # ------------------------------------------------------------------
    # Batch result validation
    # ------------------------------------------------------------------

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
        resp_by_idx = {i: resp for i, resp in enumerate(responses)}
        req_by_idx = {i: req for i, req in enumerate(requests)}
        narr_by_idx = {i: narr for i, narr in enumerate(narratives)}

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
            by_case.setdefault(req.case_id, []).append({
                "error": resp.error_message or "unknown error",
                "variant": getattr(req, "variant", ""),
                "narrative_preview": narr[:200].replace("\n", " "),
                "narrative_len": str(len(narr)),
            })

        summary_lines = [
            f"{step_desc}: {n_failed} / {n_total} requests failed ({failure_rate:.2%})"
        ]
        for case_id, details in sorted(by_case.items()):
            unique_errors = sorted(set(d["error"] for d in details))
            variants = sorted(set(d["variant"] for d in details if d["variant"]))
            narr_len = details[0]["narrative_len"]
            preview = details[0]["narrative_preview"]
            summary_lines.append(
                f"  case_id={case_id}: {len(details)} failure(s), "
                f"narrative_len={narr_len}"
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
        model = "gpt-4o-mini"
        requests: List[LLMRequest] = []
        narratives: List[str] = []

        for c in self.complaints:
            req = LLMRequest(
                request_id=LLMRequest.new_id(),
                case_id=c.case_id,
                model=model,
                task="dollar_extract",
                prompt_system=DOLLAR_EXTRACT_SYSTEM,
                prompt_user=DOLLAR_EXTRACT_USER.format(narrative=c.narrative),
                expected_schema=DollarAmountResponse,
            )
            requests.append(req)
            narratives.append(c.narrative)

        desc = "Dollar amount extraction [gpt-4o-mini]"
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
                    prompt_user=PERSONA_ESTIMATE_USER.format(narrative=c.narrative),
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
            all_personas = pd.Series(0, index=PERSONA_LABELS)
            all_personas.update(vc)
            all_personas = all_personas.sort_index()
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
                    prompt_user=REMEDY_DECISION_USER.format(narrative=c.narrative),
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
            need_info = (merged["confidence"] == "need_more_info").sum() if "confidence" in merged.columns else 0
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
            model="gpt-5.2",
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

            for resp, req in zip(responses, requests):
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    # Find the original complaint
                    orig = next((c for c in sample if c.case_id == resp.case_id), None)
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
        self.rewrite_persona_estimates: List[Tuple[str, str, PersonaEstimate]] = []  # (intended_label, model, estimate)

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
                    prompt_user=PERSONA_ESTIMATE_USER.format(narrative=rw.rewritten_narrative),
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
            intended_dims = []
            for rw in model_rewrites:
                parts = rw.persona_label.split("_")
                if len(parts) == 4:
                    intended_dims.append({"age": parts[0], "geographic": parts[1], "education": parts[2], "socioeconomic": parts[3]})

            matched = [(intended, est) for (lbl, m, est), intended in
                       zip(self.rewrite_persona_estimates, intended_dims)
                       if m == model] if intended_dims else []

            if not matched:
                continue

            logger.info("  Model: %s (%d re-estimates)", model, len(matched))
            for dim in CAUSAL_DIMS:
                correct = sum(1 for intended, est in matched if getattr(est, dim) == intended[dim])
                rate = correct / len(matched) if matched else 0
                logger.info("    %s match: %d/%d (%.1f%%)", dim, correct, len(matched), rate * 100)

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
                    prompt_user=REMEDY_DECISION_USER.format(narrative=rw.rewritten_narrative),
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
            parts = m["variant"].str.split("_", expand=True)
            if parts.shape[1] >= 4:
                m = m.copy()
                m["age"] = parts[0]
                m["geographic"] = parts[1]
                m["education"] = parts[2]
                m["socioeconomic"] = parts[3]

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
            if parts.shape[1] >= 4:
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
                    prompt_user=NEUTRALIZE_USER.format(narrative=c.narrative),
                    expected_schema=NeutralizeResponse,
                    variant="neutralized",
                )
                requests.append(req)
                narratives.append(c.narrative)

            desc = f"Step 6 [{model}] Neutralization"
            responses = self.llm.execute_batch(requests, narratives, desc=desc)
            responses = self._check_batch_results(responses, requests, narratives, desc)

            for resp in responses:
                if resp.response_parsed:
                    p = resp.response_parsed
                    orig = next((c for c in self.complaints if c.case_id == resp.case_id), None)
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
        self.neutralized_persona_estimates: List[PersonaEstimate] = []

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
                    prompt_user=PERSONA_ESTIMATE_USER.format(narrative=nc.neutralized_narrative),
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
                for dim in ESTIMATE_DIMS:
                    orig_entropy = orig_df[dim].value_counts(normalize=True).pipe(
                        lambda s: -(s * np.log2(s.clip(lower=1e-10))).sum()
                    )
                    neut_entropy = est_df[dim].value_counts(normalize=True).pipe(
                        lambda s: -(s * np.log2(s.clip(lower=1e-10))).sum()
                    )
                    logger.info("    %s entropy: original=%.3f, neutralized=%.3f", dim, orig_entropy, neut_entropy)

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
                    prompt_user=REMEDY_DECISION_USER.format(narrative=nc.neutralized_narrative),
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

        if self.neutralized_decisions:
            self.neutralized_df = pd.DataFrame([
                {
                    "case_id": d.case_id, "model": d.model,
                    "tier": d.tier, "confidence": d.confidence,
                    "reasoning": d.reasoning, "information_needed": d.information_needed,
                }
                for d in self.neutralized_decisions
            ])

    # ------------------------------------------------------------------
    # Run all analyses
    # ------------------------------------------------------------------

    def _data_diagnostic(self) -> None:
        """Log data suitability diagnostics before running statistical analyses."""
        logger.info("── Data Diagnostic ──")
        dims = ESTIMATE_DIMS

        for model in self.models:
            logger.info("  Model: %s", model)

            # Baseline tier variation
            if self.baseline_df is not None:
                b_m = self.baseline_df[self.baseline_df["model"] == model]
                tier_vc = b_m["tier"].value_counts()
                logger.info("    Baseline tiers: %s (n=%d)", dict(tier_vc), len(b_m))
                if b_m["tier"].nunique() < 2:
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
                        for level, sub in grp:
                            if sub.nunique() < 2:
                                logger.info(
                                    "    %s=%s: target is constant (%s) — will use regularised logistic regression",
                                    dim, level, sub.iloc[0],
                                )
                        min_cell = grp.size().min()
                        logger.info("    %s: min cell size = %d", dim, min_cell)

                    # Overall persona_label cell stats
                    persona_sizes = merged.groupby("persona_label").size()
                    logger.info(
                        "    Persona cells: %d personas, min=%d, median=%.0f, max=%d",
                        len(persona_sizes), persona_sizes.min(), persona_sizes.median(), persona_sizes.max(),
                    )

    def _run_analyses(self) -> None:
        """Run statistical analyses after all steps complete."""
        self._build_dataframes()
        self._data_diagnostic()

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
                model_results["correlation"] = correlation_analysis(baseline_m, personas_m)
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

        self.analysis_results = results
        self._run_severity_analysis()
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
            }
            for c in self.complaints
        ])
        sev_cols = severity_df[["case_id", "monetary_severity", "issue_severity"]]

        # Merge severity + persona labels onto baseline decisions
        baseline_sev = self.baseline_df.merge(sev_cols, on="case_id", how="left")
        baseline_sev = baseline_sev.merge(
            self.personas_df[["case_id", "model", "persona_label"]],
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
                self.personas_df[["case_id", "model", "persona_label"]],
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
        )
        self.analysis_results["severity"] = result

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
            model="gpt-5.2",
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
            all_personas = pd.Series(0, index=PERSONA_LABELS)
            all_personas.update(vc)
            all_personas = all_personas.sort_index()
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
            parts = m["variant"].str.split("_", expand=True)
            if parts.shape[1] >= 4:
                m = m.copy()
                m["age"] = parts[0]
                m["geographic"] = parts[1]
                m["education"] = parts[2]
                m["socioeconomic"] = parts[3]
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
            if parts.shape[1] >= 4:
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
                                 f"partial eta-sq={anova.get('eta_sq_partial', 0):.3f}")
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
            lines.append(f"  Effect size (Cohen's d): {wc.get('effect_size_d', 0):.2f}")

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
            "Rules:\n"
            "- Use plain business English; avoid jargon like 'chi-squared', 'p-value', "
            "'Cramer\\'s V', 'odds ratio', 'AIR', 'FDR', 'McNemar', 'Friedman', "
            "'Cochran\\'s Q', 'logistic regression', 'omnibus test'.\n"
            "- Translate statistical significance into phrases like 'no clear evidence', "
            "'statistically meaningful', 'not large enough to rule out chance'.\n"
            "- Organise with clear headings and bullet points.\n"
            "- Keep percentages and concrete comparisons (e.g. '55% vs 73%') but "
            "remove test statistics.\n"
            "- Be honest about limitations and what is inconclusive.\n\n"
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
            model="gpt-5.2",
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
                "We did an experiment that asked ChatGPT to estimate the demographic "
                "of customers based upon their complaints. The topic is whether ChatGPT "
                "may be implicitly profiling customers, and if so, what does it look like.",
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
                "We did an experiment that asked ChatGPT to recommend remedial action "
                "based upon customer complaints. Then we linked that to the estimated "
                "customer persona. The topic is whether ChatGPT may be biased based "
                "upon its implicit profiling of customers, and if so, what does it "
                "look like.",
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
                "We carried out an experiment that reworded the same complaints to "
                "resemble different personas. Our objective was to test whether ChatGPT "
                "shows bias by personas based upon their writing style. The generated "
                "text should explain the experiment, then discuss the results, and "
                "conclude whether ChatGPT is biased on average.",
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
                "We analysed whether AI bias in complaint resolution varies by the "
                "stakes involved — both the dollar amount at issue and the severity "
                "of the product/issue category. The question is whether unfair "
                "treatment worsens when the stakes are higher. The generated text "
                "should explain the analysis, present findings for each model, and "
                "conclude whether severity amplifies bias.",
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
                "We carried out an experiment that reworded the same complaints to a "
                "neutral persona. Our objective was to test whether ChatGPT can "
                "anonymise stakeholders by making their language persona-neutral. The "
                "generated text should explain the experiment, then discuss the results, "
                "and conclude whether ChatGPT is effective at anonymising by "
                "neutralising persona-specific wordings.",
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
                "We carried out an experiment that reworded complaints to anonymise "
                "the implicit personas. Our objective was to test whether using a "
                "common linguistic style removes any bias. The generated text should "
                "explain the experiment, then discuss the results, and conclude "
                "whether this risk mitigation approach works.",
            )
            if text:
                sections["bias_mitigation"] = text
                logger.info("  Section 6 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 6 generation failed")

        # Section 7: Conclusion (uses all 6 prior sections)
        logger.info("── Generating Exec Section 7: Conclusion ──")
        if sections:
            combined_sections = "\n\n".join(sections.values())
            text = self._generate_exec_section(
                "conclusion",
                "one-shot-examples-conclusion.md",
                combined_sections,
                "Write a half-page Plain English executive summary, suitable for a "
                "non-technical business executive audience, with 3-5 key findings "
                "and recommendations. You are synthesising the findings from 6 "
                "preceding executive summary sections into an overall conclusion.",
            )
            if text:
                sections["conclusion"] = text
                logger.info("  Section 7 generated (%d chars)", len(text))
            else:
                logger.warning("  Section 7 generation failed")

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

        # Repair stats
        repair_stats = self.llm.get_repair_stats()
        (self.output_dir / f"repair_stats_{ts}.json").write_text(
            json.dumps(repair_stats, indent=2), encoding="utf-8"
        )

        # Severity analysis JSON files
        sev = self.analysis_results.get("severity", {})
        if sev:
            for key in ("descriptives", "main_effects", "interactions", "model_comparison"):
                data = sev.get(key, {})
                if data:
                    (self.output_dir / f"severity_{key}_{ts}.json").write_text(
                        json.dumps(data, indent=2, default=str), encoding="utf-8"
                    )

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
            summary_path = self.output_dir / "executive_summary.txt"
            summary_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Executive summary written to %s", summary_path)

        # Failed requests
        # (collected from LLM responses with success=False — tracked internally)

        logger.info("Artifacts saved to %s", self.output_dir)
