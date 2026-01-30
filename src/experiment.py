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
        self.models = models or ["gpt-4o-mini", "gpt-5.2-mini"]
        self.threads = threads
        self.seed = seed
        self.skip_steps = skip_steps or set()
        self.verbose = verbose

        # Cache
        self.cache = ResponseCache(cache_dir)
        if clear_cache:
            self.cache.clear()

        # LLM client
        config = LLMConfig(
            api_key=api_key,
            models=self.models,
            temperature=0.0,
            max_tokens=1000,
            timeout=60,
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

            for resp in responses:
                if resp.success and resp.response_parsed:
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

    def _run_analyses(self) -> None:
        """Run statistical analyses after all steps complete."""
        self._build_dataframes()

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
