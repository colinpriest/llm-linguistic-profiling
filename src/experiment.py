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
    Complaint,
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

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 1 [{model}] Persona estimation"
            )

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
             "socioeconomic": pe.socioeconomic, "confidence": pe.confidence, "model": pe.model}
            for pe in self.persona_estimates
        ])
        logger.info("── Step 1 Summary ──")
        for model in self.models:
            m = df[df["model"] == model]
            logger.info("  Model: %s (%d estimates)", model, len(m))
            for dim in ["age", "geographic", "education", "socioeconomic"]:
                _log_value_counts(f"{dim} distribution", m[dim])
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

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 2 [{model}] Baseline remedy"
            )

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
        """Console summary: tier + confidence category distribution."""
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
        """Console summary: persona×tier crosstab + mean tier by dimension."""
        if self.personas_df is None or self.baseline_df is None:
            return
        logger.info("── Step 3 Summary ──")
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

            # 1-way: mean tier by dimension
            _log_mean_by_dimension(merged, "tier", ["age", "geographic", "education", "socioeconomic"])

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
                    narratives.append(c.narrative)

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 4 [{model}] Persona rewriting"
            )

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

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 4 [{model}] Re-estimate persona on rewrites"
            )

            for resp, rw in zip(responses, model_rewrites):
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    est = PersonaEstimate(
                        case_id=resp.case_id, model=resp.model,
                        age=p.age, geographic=p.geographic, education=p.education,
                        socioeconomic=p.socioeconomic, confidence=p.confidence,
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
            for dim in ["age", "geographic", "education", "socioeconomic"]:
                correct = sum(1 for intended, est in matched if getattr(est, dim) == intended[dim])
                rate = correct / len(matched) if matched else 0
                logger.info("    %s match: %d/%d (%.1f%%)", dim, correct, len(matched), rate * 100)

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

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 5 [{model}] Persona remedy"
            )

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
        """Console summary: persona×tier crosstab + mean tier by dimension."""
        if not self.persona_decisions:
            return
        df = pd.DataFrame([
            {"variant": d.variant, "tier": d.tier, "model": d.model}
            for d in self.persona_decisions
        ])
        logger.info("── Step 5 Summary ──")
        for model in self.models:
            m = df[df["model"] == model]
            if m.empty:
                continue
            logger.info("  Model: %s (%d persona decisions)", model, len(m))

            # Parse variant into dimensions
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

                # 1-way: mean tier by dimension
                _log_mean_by_dimension(m, "tier", ["age", "geographic", "education", "socioeconomic"])

            _log_value_counts("Overall tier distribution", m["tier"])

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

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 6 [{model}] Neutralization"
            )

            failed = [r for r in responses if not r.success]
            if failed:
                sample = failed[0]
                raise RuntimeError(
                    f"Step 6 fatal: {len(failed)}/{len(requests)} neutralizations "
                    f"failed for {model}. First failure ({sample.case_id}): "
                    f"{sample.error_message}"
                )

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

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 6 [{model}] Re-estimate persona on neutralized"
            )

            estimates_this_model = []
            for resp in responses:
                if resp.success and resp.response_parsed:
                    p = resp.response_parsed
                    est = PersonaEstimate(
                        case_id=resp.case_id, model=resp.model,
                        age=p.age, geographic=p.geographic, education=p.education,
                        socioeconomic=p.socioeconomic, confidence=p.confidence,
                        linguistic_markers=p.linguistic_markers,
                        persona_label=persona_label(p.age, p.geographic, p.education, p.socioeconomic),
                    )
                    self.neutralized_persona_estimates.append(est)
                    estimates_this_model.append(est)

            if not estimates_this_model:
                continue

            # Log estimated persona distribution on neutralized text
            est_df = pd.DataFrame([
                {"age": e.age, "geographic": e.geographic, "education": e.education, "socioeconomic": e.socioeconomic}
                for e in estimates_this_model
            ])
            logger.info("  Model: %s (%d neutralized re-estimates)", model, len(est_df))
            for dim in ["age", "geographic", "education", "socioeconomic"]:
                _log_value_counts(f"Neutralized {dim} distribution", est_df[dim])

            # Compare to original persona estimates
            orig_ests = [pe for pe in self.persona_estimates if pe.model == model]
            if orig_ests:
                orig_df = pd.DataFrame([
                    {"age": e.age, "geographic": e.geographic, "education": e.education, "socioeconomic": e.socioeconomic}
                    for e in orig_ests
                ])
                logger.info("  Comparison: original vs neutralized persona signal")
                for dim in ["age", "geographic", "education", "socioeconomic"]:
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

            responses = self.llm.execute_batch(
                requests, narratives, desc=f"Step 7 [{model}] Neutralized remedy"
            )

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
        """Console summary: persona×neutralized tier + mean tier comparison."""
        if not self.neutralized_decisions or not self.persona_estimates:
            return
        logger.info("── Step 7 Summary ──")
        neut_df = pd.DataFrame([
            {"case_id": d.case_id, "model": d.model, "neut_tier": d.tier}
            for d in self.neutralized_decisions
        ])
        base_df = pd.DataFrame([
            {"case_id": d.case_id, "model": d.model, "orig_tier": d.tier}
            for d in self.baseline_decisions
        ]) if self.baseline_decisions else pd.DataFrame()

        persona_df = pd.DataFrame([
            {"case_id": pe.case_id, "model": pe.model, "age": pe.age,
             "geographic": pe.geographic, "education": pe.education,
             "socioeconomic": pe.socioeconomic, "persona_label": pe.persona_label}
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

            # 1-way: mean neutralized tier by dimension
            _log_mean_by_dimension(merged, "neut_tier", ["age", "geographic", "education", "socioeconomic"])

            # Compare original vs neutralized mean tier
            if not base_df.empty:
                b_m = base_df[base_df["model"] == model]
                comp = b_m.merge(n_m, on=["case_id", "model"], how="inner")
                if not comp.empty:
                    logger.info("    Mean tier: original=%.3f, neutralized=%.3f, delta=%.3f",
                                comp["orig_tier"].mean(), comp["neut_tier"].mean(),
                                comp["neut_tier"].mean() - comp["orig_tier"].mean())

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
        dims = ["age", "geographic", "education", "socioeconomic"]

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
