"""Paper-ready table generation for the paper bundle.

Produces every table specified in the pre-registered analysis plan and
paper-bundle output spec in three formats: LaTeX (booktabs), Markdown,
and CSV.

Public API
----------
TableGenerator.generate_all_tables  Generate all main, appendix, and
                                    robustness tables.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _fmt_prop(v: float) -> str:
    """Format a proportion to 3 decimal places."""
    if not np.isfinite(v):
        return "---"
    return f"{v:.3f}"


def _fmt_pval(v: float) -> str:
    """Format a p-value: 2 sig figs, scientific if <1e-3."""
    if not np.isfinite(v):
        return "---"
    if v < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4f}"


def _fmt_effect(v: float) -> str:
    """Format an effect size to 3 decimal places."""
    if not np.isfinite(v):
        return "---"
    return f"{v:.3f}"


def _fmt_int(v) -> str:
    """Format an integer."""
    try:
        return f"{int(v):,}"
    except (ValueError, TypeError):
        return str(v)


def _fmt_pct(v: float, decimals: int = 1) -> str:
    """Format a percentage."""
    if not np.isfinite(v):
        return "---"
    return f"{v:.{decimals}f}%"


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(char, f"\\{char}")
    return s


# ------------------------------------------------------------------
# Multi-format output
# ------------------------------------------------------------------

def _build_filename(
    kind: str, scope: str, topic: str, model: str, subset: str,
    version: int, ext: str,
) -> str:
    """Build spec-compliant filename."""
    return f"{kind}__{scope}__{topic}__{model}__{subset}__v{version}.{ext}"


def _write_three_formats(
    df: pd.DataFrame,
    output_dir: Path,
    kind: str,
    scope: str,
    topic: str,
    model: str,
    subset: str,
    version: int,
    caption: str,
    label: str,
    analysis_type: str = "exploratory",
) -> List[Path]:
    """Write a DataFrame as .tex, .md, and .csv. Returns list of paths."""
    paths: List[Path] = []

    # --- LaTeX ---
    tex_name = _build_filename(kind, scope, topic, model, subset, version, "tex")
    tex_path = output_dir / tex_name
    tex_lines = _dataframe_to_booktabs(df, caption, label)
    tex_path.write_text(tex_lines, encoding="utf-8")
    paths.append(tex_path)

    # --- Markdown ---
    md_name = _build_filename(kind, scope, topic, model, subset, version, "md")
    md_path = output_dir / md_name
    md_lines = _dataframe_to_markdown(df, caption)
    md_path.write_text(md_lines, encoding="utf-8")
    paths.append(md_path)

    # --- CSV ---
    csv_name = _build_filename(kind, scope, topic, model, subset, version, "csv")
    csv_path = output_dir / csv_name
    df.to_csv(csv_path, index=False, encoding="utf-8")
    paths.append(csv_path)

    logger.info("Written table: %s (.tex/.md/.csv)", topic)
    return paths


def _dataframe_to_booktabs(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to booktabs LaTeX table."""
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    # Header
    headers = " & ".join(_escape_latex(str(c)) for c in df.columns) + r" \\"
    lines.append(headers)
    lines.append(r"\midrule")

    # Rows
    for _, row in df.iterrows():
        cells = " & ".join(_escape_latex(str(v)) for v in row.values) + r" \\"
        lines.append(cells)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _dataframe_to_markdown(df: pd.DataFrame, title: str) -> str:
    """Convert DataFrame to Markdown table."""
    lines = [f"**{title}**\n"]
    # Header
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines.append(headers)
    lines.append(separator)
    for _, row in df.iterrows():
        line = "| " + " | ".join(str(v) for v in row.values) + " |"
        lines.append(line)
    return "\n".join(lines)


# ------------------------------------------------------------------
# TableGenerator class
# ------------------------------------------------------------------

class TableGenerator:
    """Generates all paper-ready tables from analysis results."""

    def __init__(
        self,
        analysis_results: Dict[str, Any],
        complaints: list,
        run_config: Dict[str, Any],
        output_root: Path,
    ):
        self.ar = analysis_results
        self.complaints = complaints
        self.rc = run_config
        self.output_root = output_root
        self.models = analysis_results.get("models", [])

    def generate_all_tables(self) -> List[Path]:
        """Generate all main, appendix, and robustness tables."""
        all_paths: List[Path] = []
        all_paths.extend(self._generate_main_tables())
        all_paths.extend(self._generate_appendix_tables())
        all_paths.extend(self._generate_robustness_tables())
        return all_paths

    # ==================================================================
    # MAIN TABLES
    # ==================================================================

    def _generate_main_tables(self) -> List[Path]:
        paths: List[Path] = []
        main_dir = self.output_root / "analysis" / "tables" / "main"
        main_dir.mkdir(parents=True, exist_ok=True)

        paths.extend(self._table1_dataset_sampling(main_dir))
        paths.extend(self._table2_endpoint_definitions(main_dir))
        paths.extend(self._table3_persona_grid(main_dir))
        for model in self.models:
            paths.extend(self._table4_distributive_fairness(main_dir, model))
        for model in self.models:
            paths.extend(self._table5_neutralisation(main_dir, model))
        for model in self.models:
            for metric in ["monetary_severity", "issue_severity"]:
                paths.extend(self._table6_amplification(main_dir, model, metric))
        return paths

    def _table1_dataset_sampling(self, out: Path) -> List[Path]:
        """Table 1: Dataset and sampling summary."""
        n_total = len(self.complaints)
        sev_counts = {}
        for c in self.complaints:
            sev_counts[c.monetary_severity] = sev_counts.get(c.monetary_severity, 0) + 1
        n_amount_present = sum(v for k, v in sev_counts.items() if k in ("low", "medium", "high"))

        causal_n = int(n_total * self.rc.get("causal_sample_pct", 0.1))

        rows = [
            ("Total complaints sampled", _fmt_int(n_total), "100.0%", "Stratified by product x issue"),
            ("With extractable dollar amount", _fmt_int(n_amount_present), _fmt_pct(100 * n_amount_present / n_total if n_total else 0), "M=1 subset"),
            ("Without extractable amount", _fmt_int(n_total - n_amount_present), _fmt_pct(100 * (n_total - n_amount_present) / n_total if n_total else 0), "M=0 subset"),
            ("Matched-guise sample", _fmt_int(causal_n), _fmt_pct(100 * self.rc.get("causal_sample_pct", 0.1)), "x 36 personas each"),
            ("Total matched-guise decisions", _fmt_int(causal_n * 36 * len(self.models)), "", f"{len(self.models)} model(s)"),
            ("Seed", str(self.rc.get("seed", 42)), "", "For reproducibility"),
        ]

        df = pd.DataFrame(rows, columns=["Item", "Count", "%", "Notes"])
        return _write_three_formats(
            df, out, "table", "main", "dataset_sampling_summary", "na", "all", 1,
            caption="Table 1. Dataset and sampling summary.",
            label="tab:dataset_summary",
        )

    def _table2_endpoint_definitions(self, out: Path) -> List[Path]:
        """Table 2: Endpoint definitions."""
        rows = [
            ("Tier 0", "No action warranted", "Baseline category", "Primary (reference)"),
            ("Tier 1", "Non-monetary remediation or guidance", "Positive remediation", "Primary"),
            ("Tier 2", "Monetary remediation recommended", "Most consequential allocation", "Primary"),
            ("Y2 = 1[T=2]", "Binary: monetary remediation", "Primary distributive endpoint", "Confirmatory"),
            ("Y1+ = 1[T>=1]", "Binary: any positive remediation", "Sensitivity check", "Exploratory"),
            ("F = 1[need_more_info]", "Model requests additional information", "Operational friction (secondary)", "Exploratory"),
            ("M = 1[amount_extractable]", "Dollar amount present indicator", "Two-part severity handling", "Descriptive"),
            ("S in {L,M,H}", "Monetary severity band (within M=1)", "Severity amplification", "Confirmatory (M=1)"),
            ("Issue severity", "Product x issue derived score", "Severity amplification", "Confirmatory (all)"),
        ]
        df = pd.DataFrame(rows, columns=["Endpoint", "Definition", "Rationale", "Used as"])
        return _write_three_formats(
            df, out, "table", "main", "endpoint_definitions", "na", "all", 1,
            caption="Table 2. Endpoint definitions and analysis roles.",
            label="tab:endpoints",
        )

    def _table3_persona_grid(self, out: Path) -> List[Path]:
        """Table 3: Persona grid and rewrite design."""
        from src.schemas import AGE_VALUES, GEO_VALUES, EDU_VALUES, SES_VALUES

        n_total = len(self.complaints)
        causal_n = int(n_total * self.rc.get("causal_sample_pct", 0.1))

        rows = [
            ("Age", "2", ", ".join(AGE_VALUES), "Causal dimension"),
            ("Geography", "3", ", ".join(GEO_VALUES), "Causal dimension"),
            ("Education", "2", ", ".join(EDU_VALUES), "Causal dimension"),
            ("Socioeconomic", "3", ", ".join(SES_VALUES), "Causal dimension"),
            ("Total personas", "36", "2 x 3 x 2 x 3", "Full Cartesian product"),
            ("Matched-guise complaints", _fmt_int(causal_n), "", f"{self.rc.get('causal_sample_pct', 0.1)*100:.0f}% of sample"),
            ("Total decisions", _fmt_int(causal_n * 36 * len(self.models)), "", f"x {len(self.models)} model(s)"),
        ]
        df = pd.DataFrame(rows, columns=["Dimension", "Levels", "Values", "Notes"])
        return _write_three_formats(
            df, out, "table", "main", "persona_grid_design", "na", "matched", 1,
            caption="Table 3. Persona grid and matched-guise rewrite design.",
            label="tab:persona_grid",
        )

    def _table4_distributive_fairness(self, out: Path, model: str) -> List[Path]:
        """Table 4: Main distributive fairness results (Tier 2, matched-guise)."""
        model_r = self.ar.get(model, {})
        causal = model_r.get("causal", {})
        severity = self.ar.get("severity", {})
        dim_sev = severity.get("dimension_severity", {})

        rows = []
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            # Get dimension-level disparity from causal analysis
            rates = causal.get("rates_by_persona", {})
            effect_sizes = causal.get("effect_sizes", {})

            # Get dimension-specific interaction data
            dim_data = {}
            for metric in ["monetary_severity", "issue_severity"]:
                dd = dim_sev.get(metric, {}).get(model, {}).get(dim, {})
                if dd and not dd.get("error"):
                    dim_data[metric] = dd

            # Extract disparity metrics
            amp = dim_data.get("issue_severity", {}).get("amplification", {})
            max_min = amp.get("bias_high", float("nan")) if amp else float("nan")
            sd = amp.get("bias_low", float("nan")) if amp else float("nan")
            ci_lo = amp.get("ci_lower", float("nan")) if amp else float("nan")
            ci_hi = amp.get("ci_upper", float("nan")) if amp else float("nan")

            # Interaction p from GEE (confirmatory)
            gee = self.ar.get("severity", {}).get("gee_interactions_tier2", {})
            gee_dim = gee.get("issue_severity", {}).get(model, {}).get(dim, {})
            int_p = gee_dim.get("interaction_p", float("nan")) if not gee_dim.get("skipped") else float("nan")

            rows.append((
                dim.capitalize(),
                _fmt_effect(max_min),
                _fmt_effect(sd),
                f"[{_fmt_effect(ci_lo)}, {_fmt_effect(ci_hi)}]",
                _fmt_pval(int_p),
            ))

        # Full persona grid summary
        interactions_t2 = severity.get("interactions_tier2", {})
        full_int = interactions_t2.get("issue_severity", {}).get(model, {})
        full_amp = full_int.get("amplification", {})

        rows.append((
            "Full persona grid",
            _fmt_effect(full_amp.get("bias_high", float("nan")) - full_amp.get("bias_low", float("nan"))) if full_amp else "---",
            _fmt_effect(full_amp.get("bias_high", float("nan"))) if full_amp else "---",
            f"[{_fmt_effect(full_amp.get('ci_lower', float('nan')))}, {_fmt_effect(full_amp.get('ci_upper', float('nan')))}]" if full_amp else "---",
            "---",
        ))

        df = pd.DataFrame(rows, columns=["Dimension", "Max-Min Gap", "SD", "95% CI", "Interaction p"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "main", "distributive_fairness_tier2",
            model_clean, "matched", 1,
            caption=f"Table 4. Distributive fairness results (Tier 2, matched-guise) -- {model}.",
            label=f"tab:fairness_tier2_{model_clean}",
            analysis_type="confirmatory",
        )

    def _table5_neutralisation(self, out: Path, model: str) -> List[Path]:
        """Table 5: Neutralisation mitigation (primary endpoint)."""
        model_r = self.ar.get(model, {})
        mitigation = model_r.get("mitigation", {})

        rows = []
        # Tier 2 rate changes
        baseline_rates = mitigation.get("baseline_rates", {})
        neutralized_rates = mitigation.get("neutralized_rates", {})
        variance_change = mitigation.get("variance_change", {})

        baseline_sd = variance_change.get("baseline_sd", float("nan"))
        neutralized_sd = variance_change.get("neutralized_sd", float("nan"))
        pct_change = variance_change.get("pct_change", float("nan"))

        rows.append((
            "Tier 2 disparity (SD)",
            _fmt_effect(baseline_sd),
            _fmt_effect(neutralized_sd),
            _fmt_effect(neutralized_sd - baseline_sd) if np.isfinite(baseline_sd) and np.isfinite(neutralized_sd) else "---",
            _fmt_pct(pct_change) if np.isfinite(pct_change) else "---",
            "Reduced" if np.isfinite(pct_change) and pct_change < -5 else
            "Eliminated" if np.isfinite(pct_change) and pct_change < -95 else
            "Increased" if np.isfinite(pct_change) and pct_change > 5 else "Unchanged",
        ))

        # Process friction
        proc_mit = model_r.get("process_mitigation", {})
        proc_vc = proc_mit.get("variance_change", {})
        proc_baseline_sd = proc_vc.get("baseline_sd", float("nan"))
        proc_neut_sd = proc_vc.get("neutralized_sd", float("nan"))
        proc_pct = proc_vc.get("pct_change", float("nan"))

        rows.append((
            "Friction disparity (SD)",
            _fmt_effect(proc_baseline_sd),
            _fmt_effect(proc_neut_sd),
            _fmt_effect(proc_neut_sd - proc_baseline_sd) if np.isfinite(proc_baseline_sd) and np.isfinite(proc_neut_sd) else "---",
            _fmt_pct(proc_pct) if np.isfinite(proc_pct) else "---",
            "Reduced" if np.isfinite(proc_pct) and proc_pct < -5 else
            "Increased" if np.isfinite(proc_pct) and proc_pct > 5 else "Unchanged",
        ))

        df = pd.DataFrame(rows, columns=["Metric", "Baseline", "Neutralised", "Delta", "% Change", "Verdict"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "main", "neutralization_effects",
            model_clean, "baseline_vs_neutralized", 1,
            caption=f"Table 5. Neutralisation mitigation effects -- {model}.",
            label=f"tab:neutralization_{model_clean}",
        )

    def _table6_amplification(self, out: Path, model: str, metric: str) -> List[Path]:
        """Table 6: Severity amplification (confirmatory)."""
        severity = self.ar.get("severity", {})
        interactions_t2 = severity.get("interactions_tier2", {}).get(metric, {}).get(model, {})
        gee = severity.get("gee_interactions_tier2", {}).get(metric, {}).get(model, {})
        miss_diag = severity.get("missingness_diagnostic", {})

        rows = []
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            gee_dim = gee.get(dim, {})
            int_p = gee_dim.get("interaction_p", float("nan")) if not gee_dim.get("skipped") else float("nan")
            n_obs = gee_dim.get("n_obs", 0) if not gee_dim.get("skipped") else 0

            # Amplification from ANOVA tier2
            amp = interactions_t2.get("amplification", {})
            bias_low = amp.get("bias_low", float("nan"))
            bias_high = amp.get("bias_high", float("nan"))
            A = bias_high - bias_low if np.isfinite(bias_high) and np.isfinite(bias_low) else float("nan")
            R = amp.get("point", float("nan"))
            ci_lo = amp.get("ci_lower", float("nan"))
            ci_hi = amp.get("ci_upper", float("nan"))

            # Coverage for monetary
            coverage = ""
            if metric == "monetary_severity" and miss_diag:
                coverage = _fmt_pct(100 * (1 - miss_diag.get("pct_missing", 0)))

            # BH q-value (will be applied in paper_bundle)
            rows.append((
                dim.capitalize(),
                _fmt_pval(int_p),
                _fmt_effect(A),
                _fmt_effect(R),
                f"[{_fmt_effect(ci_lo)}, {_fmt_effect(ci_hi)}]",
                coverage,
            ))

        df = pd.DataFrame(rows, columns=["Dimension", "Interaction p", "A (high-low)", "R (ratio)", "95% CI", "Coverage"])

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        subset = "amount_present" if metric == "monetary_severity" else "all"

        return _write_three_formats(
            df, out, "table", "main", f"severity_amplification_confirmatory",
            model_clean, f"{metric_clean}__{subset}", 1,
            caption=f"Table 6. Severity amplification (confirmatory) -- {model}, {metric.replace('_', ' ')}.",
            label=f"tab:amplification_{model_clean}_{metric_clean}",
            analysis_type="confirmatory",
        )

    # ==================================================================
    # APPENDIX TABLES
    # ==================================================================

    def _generate_appendix_tables(self) -> List[Path]:
        paths: List[Path] = []
        app_dir = self.output_root / "analysis" / "tables" / "appendix"
        app_dir.mkdir(parents=True, exist_ok=True)

        for model in self.models:
            paths.extend(self._tableA1_persona_tier_dist(app_dir, model))
        for model in self.models:
            paths.extend(self._tableA2_pairwise_comparisons(app_dir, model))
        paths.extend(self._tableA3_missingness_diagnostics(app_dir))
        paths.extend(self._tableA5_prompt_inventory(app_dir))
        return paths

    def _tableA1_persona_tier_dist(self, out: Path, model: str) -> List[Path]:
        """Appendix Table A1: Full persona x tier distribution."""
        model_r = self.ar.get(model, {})
        causal = model_r.get("causal", {})
        rates = causal.get("rates_by_persona", {})

        if not rates:
            return []

        rows = []
        for persona, rate in sorted(rates.items()):
            rows.append((persona, _fmt_prop(rate)))

        df = pd.DataFrame(rows, columns=["Persona", "Tier2 Rate"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "appendix", "persona_by_tier_fullgrid",
            model_clean, "matched", 1,
            caption=f"Table A1. Full persona x tier distribution -- {model}.",
            label=f"tab:persona_tier_{model_clean}",
        )

    def _tableA2_pairwise_comparisons(self, out: Path, model: str) -> List[Path]:
        """Appendix Table A2: Pairwise persona comparisons (top-K)."""
        model_r = self.ar.get(model, {})
        causal = model_r.get("causal", {})
        pairwise = causal.get("pairwise_mcnemar", [])

        if not pairwise:
            return []

        # Sort by absolute effect and take top 20
        for p in pairwise:
            p["abs_effect"] = abs(p.get("rate_1", 0) - p.get("rate_2", 0))
        sorted_pairs = sorted(pairwise, key=lambda x: x.get("abs_effect", 0), reverse=True)[:20]

        rows = []
        for p in sorted_pairs:
            rows.append((
                p.get("persona_1", ""),
                p.get("persona_2", ""),
                _fmt_effect(p.get("rate_1", 0) - p.get("rate_2", 0)),
                _fmt_pval(p.get("p_value", float("nan"))),
                _fmt_pval(p.get("p_adjusted", float("nan"))),
            ))

        df = pd.DataFrame(rows, columns=["Persona 1", "Persona 2", "Risk Diff", "p-value", "BH q"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "appendix", "pairwise_persona_effects_topk",
            model_clean, "matched", 1,
            caption=f"Table A2. Pairwise persona comparisons (top 20 by effect size) -- {model}.",
            label=f"tab:pairwise_{model_clean}",
        )

    def _tableA3_missingness_diagnostics(self, out: Path) -> List[Path]:
        """Appendix Table A3: Monetary missingness diagnostics."""
        miss_diag = self.ar.get("severity", {}).get("missingness_diagnostic", {})
        if not miss_diag or miss_diag.get("skipped"):
            return []

        chars = miss_diag.get("complaint_characteristics", {})
        rows = []

        for test_name in ["issue_category", "product_category", "word_count", "tier_distribution", "mean_tier"]:
            test_data = chars.get(test_name, {})
            if not test_data:
                continue
            rows.append((
                test_name.replace("_", " ").title(),
                test_data.get("test", "---"),
                _fmt_effect(test_data.get("statistic", float("nan"))),
                _fmt_pval(test_data.get("p_value", float("nan"))),
                _fmt_effect(test_data.get("effect_size", float("nan"))),
            ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=["Variable", "Test", "Statistic", "p-value", "Effect Size"])
        return _write_three_formats(
            df, out, "table", "appendix", "monetary_missingness_diagnostics",
            "na", "all", 1,
            caption="Table A3. Monetary missingness diagnostics: M=1 vs M=0 comparison.",
            label="tab:missingness",
        )

    def _tableA5_prompt_inventory(self, out: Path) -> List[Path]:
        """Appendix Table A5: Prompt inventory and parameters."""
        from src.prompts import PROMPT_VERSIONS
        import hashlib

        rows = []
        for task, version in PROMPT_VERSIONS.items():
            rows.append((task, version, f"v{version}"))

        # Add model parameters
        for model in self.models:
            rows.append((f"Model: {model}", "---", "temperature=1.0 (default)"))

        df = pd.DataFrame(rows, columns=["Task/Parameter", "Version", "Notes"])
        return _write_three_formats(
            df, out, "table", "appendix", "prompt_inventory",
            "na", "all", 1,
            caption="Table A5. Prompt inventory and model parameters.",
            label="tab:prompts",
        )

    # ==================================================================
    # ROBUSTNESS TABLES
    # ==================================================================

    def _generate_robustness_tables(self) -> List[Path]:
        paths: List[Path] = []
        rob_dir = self.output_root / "analysis" / "tables" / "robustness"
        rob_dir.mkdir(parents=True, exist_ok=True)

        for model in self.models:
            paths.extend(self._tableA4_robustness(rob_dir, model))

        # Bayesian robustness tables
        bayes_dir = self.output_root / "analysis" / "tables" / "robustness" / "bayes"
        bayes_dir.mkdir(parents=True, exist_ok=True)
        paths.extend(self._tableB1_bayes_model_spec(bayes_dir))
        for model in self.models:
            for metric in ["monetary_severity", "issue_severity"]:
                paths.extend(self._tableB2_bayes_amplification(bayes_dir, model, metric))
                paths.extend(self._tableB3_bayes_ordinal_tiers(bayes_dir, model, metric))
            paths.extend(self._tableB4_bayes_diagnostics(bayes_dir, model))

        return paths

    def _tableA4_robustness(self, out: Path, model: str) -> List[Path]:
        """Appendix Table A4: Robustness summary (IPW, alt disparity, alt endpoint)."""
        severity = self.ar.get("severity", {})
        ipw = severity.get("ipw_robustness", {})
        model_ipw = ipw.get(model, {})

        rows = []

        # IPW vs unweighted
        if model_ipw and not model_ipw.get("skipped"):
            dims = model_ipw.get("dimensions", {})
            for dim, dim_r in dims.items():
                uw = dim_r.get("unweighted", {})
                w = dim_r.get("weighted", {})
                rows.append((
                    f"IPW: {dim}",
                    _fmt_effect(uw.get("A", float("nan"))),
                    _fmt_effect(w.get("A", float("nan"))),
                    _fmt_pval(uw.get("interaction", {}).get("interaction", {}).get("p_value", float("nan"))),
                    _fmt_pval(w.get("interaction", {}).get("interaction", {}).get("p_value", float("nan"))),
                    "Yes" if dim_r.get("material_change") else "No",
                ))

        # Alternative disparity (SD vs max-min) from existing analysis
        interactions = severity.get("interactions", {})
        interactions_t2 = severity.get("interactions_tier2", {})
        for metric in ["monetary_severity", "issue_severity"]:
            anova_tier = interactions.get(metric, {}).get(model, {})
            anova_t2 = interactions_t2.get(metric, {}).get(model, {})
            if anova_tier and anova_t2:
                amp_tier = anova_tier.get("amplification", {})
                amp_t2 = anova_t2.get("amplification", {})
                rows.append((
                    f"Alt endpoint: {metric} (tier vs tier2)",
                    _fmt_effect(amp_tier.get("point", float("nan"))),
                    _fmt_effect(amp_t2.get("point", float("nan"))),
                    "---", "---",
                    "Sensitivity",
                ))

        if not rows:
            rows.append(("No robustness data available", "---", "---", "---", "---", "---"))

        df = pd.DataFrame(rows, columns=["Check", "Unweighted/Tier", "IPW/Tier2", "p (uw)", "p (w)", "Verdict"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "robustness", "robustness_summary",
            model_clean, "all", 1,
            caption=f"Table A4. Robustness summary -- {model}.",
            label=f"tab:robustness_{model_clean}",
        )

    # ==================================================================
    # BAYESIAN ROBUSTNESS TABLES
    # ==================================================================

    def _tableB1_bayes_model_spec(self, out: Path) -> List[Path]:
        """Table B1: Bayesian model specification and priors."""
        rows = [
            (
                "B1 (logistic)",
                "Bernoulli-logit",
                "Y2 = 1[tier == 2]",
                "beta ~ N(0, 1.5); alpha ~ N(0, 3); sigma_u ~ Exp(1)",
                "Complaint (non-centred)",
                "Dim dummies + severity dummies + interactions",
            ),
            (
                "B2 (ordinal)",
                "Ordered-logistic",
                "tier in {0,1,2} -> y in {1,2,3}",
                "beta ~ N(0, 1.5); alpha ~ N(0, 3); c ~ N(0, 3); sigma_u ~ Exp(1)",
                "Complaint (non-centred)",
                "Same as B1",
            ),
        ]
        df = pd.DataFrame(rows, columns=[
            "Model", "Likelihood", "Endpoint", "Priors", "Random Effects", "Predictors",
        ])
        return _write_three_formats(
            df, out, "table", "robustness", "bayes_model_spec",
            "na", "all", 1,
            caption="Table B1. Bayesian model specification and priors.",
            label="tab:bayes_spec",
        )

    def _tableB2_bayes_amplification(self, out: Path, model: str, metric: str) -> List[Path]:
        """Table B2: Bayesian amplification robustness (Tier2)."""
        bayes = self.ar.get("severity", {}).get("bayesian_robustness", {})
        model_bayes = bayes.get(model, {})
        if not model_bayes or model_bayes.get("skipped") or model_bayes.get("error"):
            return []

        metric_data = model_bayes.get(metric, {})
        if not metric_data or metric_data.get("skipped"):
            return []

        rows = []
        for model_type, label in [("logistic", "B1"), ("ordinal", "B2")]:
            mt_data = metric_data.get(model_type, {})
            if not mt_data or mt_data.get("error"):
                continue
            status = mt_data.get("status", "FAIL")
            amp = mt_data.get("amplification", {})
            subset = metric_data.get("subset", "all")
            endpoint = "Y2 = 1[tier==2]" if model_type == "logistic" else "tier (ordinal)"

            for dim, dim_data in amp.items():
                rows.append((
                    f"{label} ({dim})",
                    metric.replace("_", " "),
                    subset,
                    endpoint,
                    _fmt_effect(dim_data.get("A_mean", float("nan"))),
                    _fmt_effect(dim_data.get("A_CrI_low", float("nan"))),
                    _fmt_effect(dim_data.get("A_CrI_high", float("nan"))),
                    _fmt_prop(dim_data.get("P_A_gt_0", float("nan"))),
                    _fmt_effect(dim_data.get("D_low_mean", float("nan"))),
                    _fmt_effect(dim_data.get("D_high_mean", float("nan"))),
                    status,
                ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "Model (dim)", "Severity metric", "Subset", "Endpoint",
            "A mean", "A CrI low", "A CrI high", "P(A>0)",
            "D low mean", "D high mean", "Status",
        ])

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        return _write_three_formats(
            df, out, "table", "robustness", "bayes_amplification_tier2",
            model_clean, metric_clean, 1,
            caption=f"Table B2. Bayesian amplification robustness -- {model}, {metric.replace('_', ' ')}.",
            label=f"tab:bayes_amp_{model_clean}_{metric_clean}",
        )

    def _tableB3_bayes_ordinal_tiers(self, out: Path, model: str, metric: str) -> List[Path]:
        """Table B3: Bayesian ordinal tier-shift summary."""
        bayes = self.ar.get("severity", {}).get("bayesian_robustness", {})
        model_bayes = bayes.get(model, {})
        if not model_bayes or model_bayes.get("skipped") or model_bayes.get("error"):
            return []

        metric_data = model_bayes.get(metric, {})
        if not metric_data or metric_data.get("skipped"):
            return []

        ordinal = metric_data.get("ordinal", {})
        if not ordinal or ordinal.get("error"):
            return []

        tier_probs = ordinal.get("tier_probs", {})
        if not tier_probs:
            return []

        rows = []
        for sev_band in ["low", "medium", "high"]:
            band_data = tier_probs.get(sev_band, {})
            if not band_data:
                continue
            t0 = band_data.get("P_tier0", {})
            t1 = band_data.get("P_tier1", {})
            t2 = band_data.get("P_tier2", {})

            def _fmt_cri(d: dict) -> str:
                m = d.get("mean", float("nan"))
                lo = d.get("ci_low", float("nan"))
                hi = d.get("ci_high", float("nan"))
                return f"{_fmt_prop(m)} [{_fmt_prop(lo)}, {_fmt_prop(hi)}]"

            rows.append((sev_band.capitalize(), _fmt_cri(t0), _fmt_cri(t1), _fmt_cri(t2)))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "Severity band", "P(tier=0) [95% CrI]", "P(tier=1) [95% CrI]", "P(tier=2) [95% CrI]",
        ])

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        return _write_three_formats(
            df, out, "table", "robustness", "bayes_ordinal_tier_distribution",
            model_clean, metric_clean, 1,
            caption=f"Table B3. Bayesian ordinal tier distribution -- {model}, {metric.replace('_', ' ')}.",
            label=f"tab:bayes_ordinal_{model_clean}_{metric_clean}",
        )

    def _tableB4_bayes_diagnostics(self, out: Path, model: str) -> List[Path]:
        """Table B4: Bayesian MCMC diagnostics."""
        bayes = self.ar.get("severity", {}).get("bayesian_robustness", {})
        model_bayes = bayes.get(model, {})
        if not model_bayes or model_bayes.get("skipped") or model_bayes.get("error"):
            return []

        rows = []
        for metric in ["monetary_severity", "issue_severity"]:
            metric_data = model_bayes.get(metric, {})
            if not metric_data or metric_data.get("skipped"):
                continue
            for model_type in ["logistic", "ordinal"]:
                mt_data = metric_data.get(model_type, {})
                if not mt_data or mt_data.get("error"):
                    continue
                diag = mt_data.get("diagnostics", {})
                rows.append((
                    mt_data.get("fit_id", f"{model_type}/{metric}"),
                    _fmt_effect(diag.get("max_rhat", float("nan"))),
                    _fmt_int(diag.get("min_ess_bulk", 0)),
                    _fmt_int(diag.get("divergences", 0)),
                    _fmt_int(diag.get("treedepth_hits", 0)),
                    mt_data.get("status", "FAIL"),
                ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "Fit ID", "Max Rhat", "Min ESS bulk", "Divergences", "Treedepth hits", "Status",
        ])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "robustness", "bayes_diagnostics",
            model_clean, "all", 1,
            caption=f"Table B4. Bayesian MCMC diagnostics -- {model}.",
            label=f"tab:bayes_diag_{model_clean}",
        )
