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
from src.process_discrimination import (
    info_request_content_analysis,
    process_causal_analysis,
    process_correlation_analysis,
    process_mitigation_analysis,
)
from src.prompts import (
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
        self._display_analysis_results()

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

        # --- Business Executive Summary ---
        all_commentaries = {}
        for model in self.models:
            mr = self.analysis_results.get(model, {})
            c = mr.get("gpt_commentary")
            if c:
                all_commentaries[model] = c

        if all_commentaries:
            exec_summary = self._get_business_executive_summary(all_commentaries)
            if exec_summary:
                logger.info("=" * 60)
                logger.info("BUSINESS EXECUTIVE SUMMARY")
                logger.info("=" * 60)
                logger.info("%s", exec_summary)
                self.analysis_results["business_executive_summary"] = exec_summary
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

    def _get_business_executive_summary(self, commentaries: Dict[str, str]) -> Optional[str]:
        """Translate technical GPT commentaries into a business executive summary."""
        combined = "\n\n".join(
            f"=== {model} ===\n{text}" for model, text in commentaries.items()
        )

        system_prompt = (
            "You are a communications specialist who translates technical statistical "
            "findings into clear, actionable language for business executives. You "
            "receive technical bias-testing commentary from a statistician and rewrite "
            "it as a polished executive summary.\n\n"
            "Rules:\n"
            "- Use plain business English; avoid jargon like 'chi-squared', 'p-value', "
            "'Cramer\\'s V', 'odds ratio', 'AIR', 'FDR', 'McNemar', 'Friedman', "
            "'Cochran\\'s Q', 'logistic regression', 'omnibus test'.\n"
            "- Translate statistical significance into phrases like 'no clear evidence', "
            "'statistically meaningful', 'not large enough to rule out chance'.\n"
            "- Organise with clear headings and bullet points.\n"
            "- Include a 'Bottom Line' up front, then 'Outcome Fairness', "
            "'Process Fairness', 'Important Caveats', and 'Recommended Actions'.\n"
            "- Keep percentages and concrete comparisons (e.g. '55% vs 73%') but "
            "remove test statistics.\n"
            "- Be honest about limitations and what is inconclusive.\n\n"
            "ONE-SHOT EXAMPLE\n"
            "────────────────\n"
            "INPUT (technical commentary):\n"
            "Tier bias: Across persona language styles, there is no statistically "
            "reliable evidence that gpt-5-mini changes remedy tier as a function of "
            "persona in this dataset: overall persona×tier association is non-significant "
            "(χ²=15.65, p=0.738; Cramer's V=0.280) and the matched-guise causal tests "
            "likewise show no detectable differences (Friedman p=0.4726; 0/630 McNemar "
            "pairwise significant after FDR). Most single-dimension χ² tests are also "
            "non-significant (age, education, SES, gender, ESL all p>0.16). The one "
            "exception is the logistic regression where urban geography is associated "
            "with lower odds of a favorable tier (OR=0.292, p=0.037), aligning with "
            "the descriptive gap (suburban 0.732 vs urban 0.552 favorable). Gender shows "
            "a sizable descriptive difference (female 0.806 vs male 0.623) but is not "
            "statistically significant in the model (gender_male OR=0.403, p=0.103), so "
            "it is suggestive rather than confirmed. ESL shows the opposite pattern "
            "descriptively (ESL 0.750 vs native 0.674) and is not significant "
            "(p=0.357/0.469). Practically, while aggregate tests do not support "
            "systematic tier discrimination, several intersectional personas have "
            "materially low adverse impact ratios (AIR<0.80), including "
            "young_urban_college_middle (AIR=0.40) and old_urban_college_middle (0.50); "
            "one persona has a zero favorable rate (old_suburban_high_school_low, "
            "AIR=0.00), which is practically severe but likely driven by small cell "
            "counts/noise given the lack of corroborating significance in causal tests.\n\n"
            "Process bias: Procedural burden (requests for more information) does not "
            "show statistically significant variation by persona overall (χ²=6.02, "
            "p=0.8138; V=0.245) and the matched-guise causal process test is also "
            "non-significant (Cochran's Q p=0.1246), despite a moderate raw spread in "
            "need-info rates (0.26 to 0.54). Descriptively, higher burdens appear for "
            "young (0.400 vs 0.314 old), high-school (0.391 vs 0.325 college), male "
            "(0.377 vs 0.258 female), and low-SES (0.455 vs 0.326 middle); ESL "
            "complainants are asked for info less often (0.125 vs 0.359 native), which "
            "is the reverse of a common concern but may reflect the model simplifying "
            "responses or under-probing. Compared to outcome (tier) discrimination, the "
            "process disparities are similarly not statistically supported at conventional "
            "thresholds, and any apparent subgroup differences should be treated as "
            "exploratory. Key limitations: multiple comparisons and sparse cells can "
            "inflate apparent AIR outliers; non-significant omnibus/causal tests suggest "
            "limited power to detect small effects; and the mitigation/neutralization "
            "results show no reliable pre/post change (McNemar p>0.60) and essentially "
            "no variance reduction for outcomes (−0.9%), so conclusions about bias "
            "removal are weak.\n\n"
            "OUTPUT (executive summary):\n"
            "GPT-5.2 Bias Testing: Executive Summary\n\n"
            "Bottom Line\n"
            "We tested whether GPT-5.2 treats different types of complainants unfairly "
            "based on their demographics. Overall, we found no clear evidence of "
            "systematic bias, though a few patterns warrant monitoring.\n\n"
            "Outcome Fairness (Did certain groups receive worse remedies?)\n"
            "The good news: When we analyzed whether the AI recommended different remedy "
            "levels based on a complainant's age, education, income level, gender, or "
            "English proficiency, we found no statistically meaningful pattern of "
            "discrimination.\n"
            "One area to watch: Urban complainants appeared to receive favorable outcomes "
            "at a notably lower rate than suburban complainants (55% vs. 73%). This was "
            "the only factor that showed up as statistically significant.\n"
            "Inconclusive signals: Women appeared to receive favorable outcomes more "
            "often than men (81% vs. 62%), but this difference wasn't large enough to "
            "rule out chance. Similarly, non-native English speakers actually fared "
            "slightly better than native speakers—the opposite of what we might have "
            "expected.\n"
            "Small-sample outliers: A handful of very specific customer profiles showed "
            "concerning patterns on paper, but these likely reflect the small number of "
            "cases in those categories rather than true bias.\n\n"
            "Process Fairness (Were certain groups asked to jump through more hoops?)\n"
            "We also examined whether some complainants were more frequently asked to "
            "provide additional information—creating extra burden.\n"
            "Finding: No statistically significant differences. While raw numbers showed "
            "some variation (for example, younger complainants and lower-income "
            "complainants were asked for more information slightly more often), these "
            "patterns weren't strong enough to confirm real bias.\n\n"
            "Important Caveats\n"
            "- Small sample sizes in some subgroups can make differences look larger "
            "than they truly are\n"
            "- Our tests may not be sensitive enough to detect subtle bias\n"
            "- Attempts to reduce bias showed no measurable improvement, so we can't yet "
            "confirm that any mitigation approach is working\n\n"
            "Recommended Actions\n"
            "- Monitor urban/suburban disparity — investigate whether this reflects "
            "genuine bias or differences in complaint complexity\n"
            "- Increase sample sizes before drawing firm conclusions about specific "
            "demographic groups\n"
            "- Revisit bias mitigation approach — current methods aren't demonstrably "
            "effective\n"
            "────────────────\n\n"
            "Now produce the same style of executive summary for the technical "
            "commentary provided by the user. Do NOT copy the example output; write "
            "fresh content that accurately reflects the actual data."
        )

        user_prompt = (
            "Here is the technical bias-testing commentary for each model. "
            "Write a business executive summary following the format shown in "
            "the example.\n\n" + combined
        )

        req = LLMRequest(
            request_id=LLMRequest.new_id(),
            case_id="executive_summary",
            model="gpt-5.2",
            task="executive_summary",
            prompt_system=system_prompt,
            prompt_user=user_prompt,
            expected_schema=CommentaryResponse,
            variant="executive_summary",
        )

        try:
            resp = self.llm.call_with_cache(req, combined)
            if resp.success and resp.response_parsed:
                return resp.response_parsed.commentary
            logger.warning("Executive summary request failed: %s", resp.error_message)
        except Exception as e:
            logger.warning("Executive summary request failed: %s", e)
        return None

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

        # Failed requests
        # (collected from LLM responses with success=False — tracked internally)

        logger.info("Artifacts saved to %s", self.output_dir)
