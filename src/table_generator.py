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

from src.estimable import NE, NEValue, is_estimable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _fmt_prop(v: Any) -> str:
    """Format a proportion to 3 decimal places."""
    if not is_estimable(v):
        return "---"
    return f"{float(v):.3f}"


def _fmt_pval(v: Any) -> str:
    """Format a p-value: 2 sig figs, scientific if <1e-3."""
    if not is_estimable(v):
        return "---"
    fv = float(v)
    if fv < 1e-3:
        return f"{fv:.2e}"
    return f"{fv:.4f}"


def _fmt_effect(v: Any) -> str:
    """Format an effect size to 3 decimal places."""
    if not is_estimable(v):
        return "---"
    return f"{float(v):.3f}"


def _fmt_int(v) -> str:
    """Format an integer."""
    try:
        return f"{int(v):,}"
    except (ValueError, TypeError):
        return str(v)


def _fmt_pct(v: Any, decimals: int = 1) -> str:
    """Format a percentage."""
    if not is_estimable(v):
        return "---"
    return f"{float(v):.{decimals}f}%"


def _resolve_variance_change(block: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize variance change payload across mitigation result objects."""
    vc = dict(block.get("variance_change", {}) or {})

    # Backfill from legacy keys when variance_change payload is partial.
    if "baseline_sd" not in vc and "variance_orig" in block:
        try:
            vc["baseline_sd"] = float(block["variance_orig"]) ** 0.5
        except (TypeError, ValueError):
            pass
    if "neutralized_sd" not in vc and "variance_neut" in block:
        try:
            vc["neutralized_sd"] = float(block["variance_neut"]) ** 0.5
        except (TypeError, ValueError):
            pass

    # Accept either stored percent change or legacy ratio reduction.
    if "pct_change" not in vc:
        if "variance_reduction" in block:
            try:
                vc["pct_change"] = -100.0 * float(block["variance_reduction"])
            except (TypeError, ValueError):
                pass
        elif "baseline_sd" in vc and "neutralized_sd" in vc:
            base = vc.get("baseline_sd")
            neut = vc.get("neutralized_sd")
            if is_estimable(base) and is_estimable(neut) and float(base) != 0:
                vc["pct_change"] = (float(neut) - float(base)) / float(base) * 100.0

    return vc


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
    col_spec: Optional[str] = None,
    use_tabularx: bool = False,
    font_size: Optional[str] = None,
) -> List[Path]:
    """Write a DataFrame as .tex, .md, and .csv. Returns list of paths."""
    paths: List[Path] = []

    # --- LaTeX ---
    tex_name = _build_filename(kind, scope, topic, model, subset, version, "tex")
    tex_path = output_dir / tex_name
    tex_lines = _dataframe_to_booktabs(
        df, caption, label,
        col_spec=col_spec, use_tabularx=use_tabularx, font_size=font_size,
    )
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


def _dataframe_to_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    col_spec: Optional[str] = None,
    use_tabularx: bool = False,
    font_size: Optional[str] = None,
) -> str:
    """Convert DataFrame to booktabs LaTeX table.

    Parameters
    ----------
    col_spec : optional override for column alignment string.
    use_tabularx : if True, emit ``tabularx`` (requires ``\\usepackage{tabularx}``
        in the preamble).  At least one ``X`` column is expected in *col_spec*.
    font_size : e.g. ``"small"`` or ``"footnotesize"``; wrapped around the table
        body so the table shrinks without affecting the caption.
    """
    n_cols = len(df.columns)
    if col_spec is None:
        col_spec = "l" + "r" * (n_cols - 1)

    env = "tabularx" if use_tabularx else "tabular"
    env_open = (
        r"\begin{tabularx}{\textwidth}{" + col_spec + "}"
        if use_tabularx
        else r"\begin{tabular}{" + col_spec + "}"
    )
    env_close = r"\end{" + env + "}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{" + label + "}",
    ]
    if font_size:
        lines.append("\\" + font_size)
    lines.append(env_open)
    lines.append(r"\toprule")

    # Header
    headers = " & ".join(_escape_latex(str(c)) for c in df.columns) + r" \\"
    lines.append(headers)
    lines.append(r"\midrule")

    # Rows
    for _, row in df.iterrows():
        cells = " & ".join(_escape_latex(str(v)) for v in row.values) + r" \\"
        lines.append(cells)

    lines.append(r"\bottomrule")
    lines.append(env_close)
    if font_size:
        lines.append(r"\normalsize")
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
        paths.extend(self._table7_persona_explainability(main_dir))
        paths.extend(self._table8_manipulation_rewrite_match(main_dir))
        paths.extend(self._table9_manipulation_signal_reduction(main_dir))
        for model in self.models:
            paths.extend(self._table10_operational_friction(main_dir, model))
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
            caption="Dataset and sampling summary.",
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

        # --- Custom LaTeX with ragged columns ---
        tex_rows = [
            (r"Tier 0", "No action warranted", "Baseline category", "Primary (reference)"),
            (r"Tier 1", "Non-monetary remediation or guidance", "Positive remediation", "Primary"),
            (r"Tier 2", "Monetary remediation recommended", "Most consequential allocation", "Primary"),
            (r"\path{Y2 = 1[T=2]}", "Binary: monetary remediation", "Primary distributive endpoint", "Confirmatory"),
            (r"\path{Y1+ = 1[T>=1]}", "Binary: any positive remediation", "Sensitivity check", "Exploratory"),
            (r"\path{F = 1[need_more_info]}", "Model requests additional information", "Operational friction (secondary)", "Exploratory"),
            (r"\path{M = 1[amount_extractable]}", "Dollar amount present indicator", "Two-part severity handling", "Descriptive"),
            (r"\path{S \in \{L,M,H\}}", r"Monetary severity band (within \path{M=1})", "Severity amplification", "Confirmatory (M=1)"),
            ("Issue severity", r"Product $\times$ issue derived score", "Severity amplification", "Confirmatory (all)"),
        ]
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Endpoint definitions and analysis roles.}",
            r"\label{tab:endpoints}",
            r"\begin{tabular}{>{\RaggedRight\arraybackslash}p{0.18\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.34\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.30\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.14\linewidth}}",
            r"\toprule",
            r"Endpoint & Definition & Rationale & Used as \\",
            r"\midrule",
        ]
        for endpoint, definition, rationale, used_as in tex_rows:
            tex_lines.append(f"{endpoint} & {definition} & {rationale} & {used_as} \\\\")
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append(r"\end{table}")

        tex_name = _build_filename("table", "main", "endpoint_definitions", "na", "all", 1, "tex")
        tex_path = out / tex_name
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV via standard path ---
        md_name = _build_filename("table", "main", "endpoint_definitions", "na", "all", 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, "Endpoint definitions and analysis roles."), encoding="utf-8")

        csv_name = _build_filename("table", "main", "endpoint_definitions", "na", "all", 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: endpoint_definitions (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

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
            caption="Persona grid and matched-guise rewrite design.",
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
            ne_amp = NE.missing_data(context=f"amp {dim}")
            amp = dim_data.get("issue_severity", {}).get("amplification", {})
            max_min = amp.get("bias_high", ne_amp) if amp else ne_amp
            sd = amp.get("bias_low", ne_amp) if amp else ne_amp
            ci_lo = amp.get("ci_lower", ne_amp) if amp else ne_amp
            ci_hi = amp.get("ci_upper", ne_amp) if amp else ne_amp

            # Interaction p from GEE (confirmatory)
            gee = self.ar.get("severity", {}).get("gee_interactions_tier2", {})
            gee_dim = gee.get("issue_severity", {}).get(model, {}).get(dim, {})
            int_p = gee_dim.get("interaction_p", ne_amp) if not gee_dim.get("skipped") else NE.skipped("GEE skipped")

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

        ne_full = NE.missing_data(context="full grid amp")
        if full_amp:
            bias_hi = full_amp.get("bias_high", ne_full)
            bias_lo = full_amp.get("bias_low", ne_full)
            gap = bias_hi - bias_lo if is_estimable(bias_hi) and is_estimable(bias_lo) else ne_full
        else:
            gap = ne_full
            bias_hi = ne_full
        rows.append((
            "Full persona grid",
            _fmt_effect(gap) if full_amp else "---",
            _fmt_effect(bias_hi) if full_amp else "---",
            f"[{_fmt_effect(full_amp.get('ci_lower', ne_full))}, {_fmt_effect(full_amp.get('ci_upper', ne_full))}]" if full_amp else "---",
            "---",
        ))

        df = pd.DataFrame(rows, columns=["Dimension", "Max-Min Gap", "SD", "95% CI", "Interaction p"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "main", "distributive_fairness_tier2",
            model_clean, "matched", 1,
            caption=f"Distributive fairness results (Tier 2, matched-guise) -- {model}.",
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
        variance_change = _resolve_variance_change(mitigation)

        ne_vc = NE.missing_data(context="variance change")
        baseline_sd = variance_change.get("baseline_sd", ne_vc)
        neutralized_sd = variance_change.get("neutralized_sd", ne_vc)
        pct_change = variance_change.get("pct_change", ne_vc)

        rows.append((
            "Tier 2 disparity (SD)",
            _fmt_effect(baseline_sd),
            _fmt_effect(neutralized_sd),
            _fmt_effect(neutralized_sd - baseline_sd) if is_estimable(baseline_sd) and is_estimable(neutralized_sd) else "---",
            _fmt_pct(pct_change) if is_estimable(pct_change) else "---",
            "Reduced" if is_estimable(pct_change) and pct_change < -5 else
            "Eliminated" if is_estimable(pct_change) and pct_change < -95 else
            "Increased" if is_estimable(pct_change) and pct_change > 5 else "Unchanged",
        ))

        # Process friction
        proc_mit = model_r.get("process_mitigation", {})
        proc_vc = _resolve_variance_change(proc_mit)
        proc_baseline_sd = proc_vc.get("baseline_sd", ne_vc)
        proc_neut_sd = proc_vc.get("neutralized_sd", ne_vc)
        proc_pct = proc_vc.get("pct_change", ne_vc)

        rows.append((
            "Friction disparity (SD)",
            _fmt_effect(proc_baseline_sd),
            _fmt_effect(proc_neut_sd),
            _fmt_effect(proc_neut_sd - proc_baseline_sd) if is_estimable(proc_baseline_sd) and is_estimable(proc_neut_sd) else "---",
            _fmt_pct(proc_pct) if is_estimable(proc_pct) else "---",
            "Reduced" if is_estimable(proc_pct) and proc_pct < -5 else
            "Increased" if is_estimable(proc_pct) and proc_pct > 5 else "Unchanged",
        ))

        df = pd.DataFrame(rows, columns=["Metric", "Baseline", "Neutralised", "Delta", "% Change", "Verdict"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "main", "neutralization_effects",
            model_clean, "baseline_vs_neutralized", 1,
            caption=f"Neutralisation mitigation effects -- {model}.",
            label=f"tab:neutralization_{model_clean}",
        )

    def _table6_amplification(self, out: Path, model: str, metric: str) -> List[Path]:
        """Table 6: Severity amplification (confirmatory)."""
        severity = self.ar.get("severity", {})
        interactions_t2 = severity.get("interactions_tier2", {}).get(metric, {}).get(model, {})
        dim_sev = severity.get("dimension_severity", {}).get(metric, {}).get(model, {})
        gee = severity.get("gee_interactions_tier2", {}).get(metric, {}).get(model, {})
        miss_diag = severity.get("missingness_diagnostic", {})

        rows = []
        ne_amp_t6 = NE.missing_data(context="amplification table")
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            gee_dim = gee.get(dim, {})
            int_p = gee_dim.get("interaction_p", ne_amp_t6) if not gee_dim.get("skipped") else NE.skipped("GEE skipped")
            n_obs = gee_dim.get("n_obs", 0) if not gee_dim.get("skipped") else 0

            # Amplification should be read from dimension-specific interaction output.
            # Fallback to the model-level interaction object for backward compatibility.
            dd = dim_sev.get(dim, {})
            amp = dd.get("amplification", {}) if dd and not dd.get("error") else {}
            if not amp:
                amp = interactions_t2.get("amplification", {})
            bias_low = amp.get("bias_low", ne_amp_t6)
            bias_high = amp.get("bias_high", ne_amp_t6)
            A = bias_high - bias_low if is_estimable(bias_high) and is_estimable(bias_low) else ne_amp_t6
            R = amp.get("point", ne_amp_t6)
            ci_lo = amp.get("ci_lower", ne_amp_t6)
            ci_hi = amp.get("ci_upper", ne_amp_t6)

            # Coverage for monetary
            coverage = ""
            if metric == "monetary_severity" and miss_diag:
                coverage = _fmt_pct(100 * (1 - miss_diag.get("pct_missing", 0)))

            # BH q-value (will be applied in paper_bundle)
            row = [
                dim.capitalize(),
                _fmt_pval(int_p),
                _fmt_effect(A),
                _fmt_effect(R),
                f"[{_fmt_effect(ci_lo)}, {_fmt_effect(ci_hi)}]",
            ]
            if metric == "monetary_severity":
                row.append(coverage)
            rows.append(tuple(row))

        columns = ["Dimension", "Interaction p", "A (high-low)", "R (ratio)", "95% CI"]
        if metric == "monetary_severity":
            columns.append("Coverage")
        df = pd.DataFrame(rows, columns=columns)

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        subset = "amount_present" if metric == "monetary_severity" else "all"

        return _write_three_formats(
            df, out, "table", "main", f"severity_amplification_confirmatory",
            model_clean, f"{metric_clean}__{subset}", 1,
            caption=f"Severity amplification (confirmatory) -- {model}, {metric.replace('_', ' ')}.",
            label=f"tab:amplification_{model_clean}_{metric_clean}",
            analysis_type="confirmatory",
        )

    def _table7_persona_explainability(self, out: Path) -> List[Path]:
        """Table 7: Top text features predicting persona dimensions (LightGBM)."""
        pa = self.ar.get("persona_analysis", {})
        if not pa:
            logger.warning("No persona_analysis data; skipping Table 7.")
            return []

        rows = []
        for model in self.models:
            model_pa = pa.get(model, {})
            dims = model_pa.get("dimensions", {})
            for dim in ["age", "geographic", "education", "socioeconomic", "gender", "esl"]:
                dim_data = dims.get(dim, {})
                for value, val_data in sorted(dim_data.items()):
                    if isinstance(val_data, dict) and "top_features" in val_data:
                        feats = val_data["top_features"]
                        auc = val_data.get("auc")
                        summary = val_data.get("summary", "")
                        feat_strs = []
                        for f in feats[:3]:
                            display = f.get("feature_display", f.get("feature", "?"))
                            imp = f.get("importance", 0)
                            feat_strs.append(f"{display} ({imp:.0f})")
                        while len(feat_strs) < 3:
                            feat_strs.append("---")
                        rows.append((
                            dim.capitalize(),
                            value,
                            model,
                            "; ".join(feat_strs),
                            _fmt_prop(auc) if auc is not None else "---",
                            summary,
                        ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "Dimension", "Value", "Model",
            "Top 3 Features",
            "AUC", "Interpretation",
        ])

        # Custom LaTeX with ragged columns for wide table
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Top text features predicting persona dimensions (LightGBM classifiers).}",
            r"\label{tab:persona_explainability}",
            r"\begin{tabular}{>{\RaggedRight\arraybackslash}p{0.14\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.10\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.10\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.28\linewidth}",
            r"                r",
            r"                >{\RaggedRight\arraybackslash}p{0.24\linewidth}}",
            r"\toprule",
            r"Dimension & Value & Model & Top 3 Features & AUC & Interpretation \\",
            r"\midrule",
        ]
        for _, row in df.iterrows():
            dim_esc = _escape_latex(str(row["Dimension"]))
            val_esc = _escape_latex(str(row["Value"]))
            model_esc = _escape_latex(str(row["Model"]))
            # Split features on "; " and rejoin with \newline for LaTeX line breaks
            feat_parts = str(row["Top 3 Features"]).split("; ")
            feats_latex = r" \newline ".join(_escape_latex(p) for p in feat_parts)
            auc_esc = _escape_latex(str(row["AUC"]))
            interp_esc = _escape_latex(str(row["Interpretation"]))
            tex_lines.append(
                f"{dim_esc} & {val_esc} & {model_esc} & {feats_latex} & {auc_esc} & {interp_esc} \\\\"
            )
        tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        tex_name = _build_filename("table", "main", "persona_explainability", "na", "all", 1, "tex")
        tex_path = out / tex_name
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        md_name = _build_filename("table", "main", "persona_explainability", "na", "all", 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, "Top text features predicting persona dimensions."), encoding="utf-8")

        csv_name = _build_filename("table", "main", "persona_explainability", "na", "all", 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: persona_explainability (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

    def _table8_manipulation_rewrite_match(self, out: Path) -> List[Path]:
        """Table 8: Manipulation check -- persona re-estimation match rates after rewriting."""
        mc = self.ar.get("manipulation_checks", {})
        match_rates = mc.get("rewrite_match_rates", {})
        if not match_rates:
            logger.info("No rewrite_match_rates data; skipping Table 8 (requires re-run with updated code).")
            return []

        # Collect all dimensions across models
        all_dims = set()
        for model_rates in match_rates.values():
            all_dims.update(model_rates.keys())
        dims_sorted = sorted(all_dims)

        rows = []
        for dim in dims_sorted:
            row = [dim.capitalize()]
            for model in self.models:
                rate = match_rates.get(model, {}).get(dim)
                row.append(_fmt_pct(rate * 100, 1) if rate is not None else "---")
            rows.append(tuple(row))

        cols = ["Dimension"] + [f"{m} Match %" for m in self.models]
        df = pd.DataFrame(rows, columns=cols)
        return _write_three_formats(
            df, out, "table", "main", "manipulation_rewrite_match", "na", "all", 1,
            caption="Manipulation check: persona re-estimation match rates after style rewriting.",
            label="tab:manipulation_rewrite",
        )

    def _table9_manipulation_signal_reduction(self, out: Path) -> List[Path]:
        """Table 9: Manipulation check -- entropy signal reduction after neutralisation."""
        mc = self.ar.get("manipulation_checks", {})
        signal = mc.get("signal_reduction", {})
        if not signal:
            logger.info("No signal_reduction data; skipping Table 9 (requires re-run with updated code).")
            return []

        rows = []
        for model in self.models:
            model_signal = signal.get(model, {})
            for dim in sorted(model_signal.keys()):
                d = model_signal[dim]
                rows.append((
                    dim.capitalize(),
                    model,
                    _fmt_effect(d.get("original_entropy")),
                    _fmt_effect(d.get("neutralized_entropy")),
                    _fmt_pct(d.get("reduction_pct", 0), 1),
                ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "Dimension", "Model", "Orig Entropy", "Neut Entropy", "Reduction %",
        ])
        return _write_three_formats(
            df, out, "table", "main", "manipulation_signal_reduction", "na", "all", 1,
            caption="Manipulation check: entropy signal reduction after language neutralisation.",
            label="tab:manipulation_signal",
        )

    def _table10_operational_friction(self, out: Path, model: str) -> List[Path]:
        """Table 10: Operational friction (need_more_info) rates by dimension."""
        model_r = self.ar.get(model, {})
        proc_corr = model_r.get("process_correlation", {})
        info_rates = proc_corr.get("info_rates_by_dimension", {})
        if not info_rates:
            logger.warning("No info_rates_by_dimension for %s; skipping Table 10.", model)
            return []

        # Also try to get causal process rates if available
        proc_causal = model_r.get("process_causal", {})
        causal_rates = proc_causal.get("info_rates_by_dimension", {}) if proc_causal else {}

        rows = []
        for dim in sorted(info_rates.keys()):
            dim_rates = info_rates[dim]
            causal_dim = causal_rates.get(dim, {})
            for value in sorted(dim_rates.keys()):
                corr_rate = dim_rates[value]
                caus_rate = causal_dim.get(value)
                rows.append((
                    dim.capitalize(),
                    value,
                    _fmt_pct(corr_rate * 100, 1) if corr_rate is not None else "---",
                    _fmt_pct(caus_rate * 100, 1) if caus_rate is not None else "---",
                ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "Dimension", "Value", "Correlational Rate", "Causal Rate",
        ])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "main", "operational_friction",
            model_clean, "all", 1,
            caption=f"Operational friction (need_more_info) rates by dimension -- {model}.",
            label=f"tab:friction_{model_clean}",
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
        paths.extend(self._tableA3b_missingness_ipw_summary(app_dir))
        paths.extend(self._tableA5_prompt_inventory(app_dir))
        paths.extend(self._tableA6_text_feature_inventory(app_dir))
        for model in self.models:
            paths.extend(self._tableA7_persona_distribution_diagnostics(app_dir, model))
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
            caption=f"Full persona x tier distribution -- {model}.",
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
        ne_pair = NE.missing_data(context="pairwise comparison")
        for p in sorted_pairs:
            rows.append((
                p.get("persona_1", ""),
                p.get("persona_2", ""),
                _fmt_effect(p.get("rate_1", 0) - p.get("rate_2", 0)),
                _fmt_pval(p.get("p_value", ne_pair)),
                _fmt_pval(p.get("p_adjusted", ne_pair)),
            ))

        df = pd.DataFrame(rows, columns=["Persona 1", "Persona 2", "Risk Diff", "p-value", "BH q"])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "appendix", "pairwise_persona_effects_topk",
            model_clean, "matched", 1,
            caption=f"Pairwise persona comparisons (top 20 by effect size) -- {model}.",
            label=f"tab:pairwise_{model_clean}",
        )

    def _tableA3_missingness_diagnostics(self, out: Path) -> List[Path]:
        """Appendix Table A3: Monetary missingness diagnostics."""
        miss_diag = self.ar.get("severity", {}).get("missingness_diagnostic", {})
        if not miss_diag or miss_diag.get("skipped"):
            return []

        chars = miss_diag.get("complaint_characteristics", {})
        rows = []

        # Each test stores results with different key names; map to
        # standardised (test_label, statistic_key, effect_size_key) triples.
        _TEST_MAP = {
            "issue_category":    ("Chi-squared",    "chi2",        "cramers_v"),
            "product_category":  ("Chi-squared",    "chi2",        "cramers_v"),
            "word_count":        ("Mann-Whitney U", "U_statistic", "rank_biserial"),
            "tier_distribution": ("Chi-squared",    "chi2",        "cramers_v"),
            "mean_tier":         ("Welch t-test",   "t_statistic", "cohens_d"),
        }

        ne_test = NE.missing_data(context="missingness test")
        for test_name in ["issue_category", "product_category", "word_count", "tier_distribution", "mean_tier"]:
            test_data = chars.get(test_name, {})
            if not test_data:
                continue
            label, stat_key, es_key = _TEST_MAP[test_name]
            rows.append((
                test_name.replace("_", " ").title(),
                label,
                _fmt_effect(test_data.get(stat_key, ne_test)),
                _fmt_pval(test_data.get("p_value", ne_test)),
                _fmt_effect(test_data.get(es_key, ne_test)),
            ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=["Variable", "Test", "Statistic", "p-value", "Effect Size"])
        return _write_three_formats(
            df, out, "table", "appendix", "monetary_missingness_diagnostics",
            "na", "all", 1,
            caption="Monetary missingness diagnostics: M=1 vs M=0 comparison.",
            label="tab:missingness",
        )

    def _tableA3b_missingness_ipw_summary(self, out: Path) -> List[Path]:
        """Appendix Table A3b: Monetary missingness and IPW diagnostics summary."""
        miss_diag = self.ar.get("severity", {}).get("missingness_diagnostic", {})
        ipw_result = self.ar.get("severity", {}).get("ipw_robustness", {})
        if not miss_diag or miss_diag.get("skipped"):
            return []

        n_present = miss_diag.get("n_present", 0)
        n_missing = miss_diag.get("n_missing", 0)
        pct_present = (1 - miss_diag.get("pct_missing", 0)) * 100

        # Collect absolute effect sizes from complaint characteristic tests
        # to compute Max |SMD| and Mean |SMD|.
        chars = miss_diag.get("complaint_characteristics", {})
        _ES_KEY_MAP = {
            "issue_category": "cramers_v",
            "product_category": "cramers_v",
            "word_count": "rank_biserial",
            "tier_distribution": "cramers_v",
            "mean_tier": "cohens_d",
        }
        abs_effects: List[float] = []
        for test_name, es_key in _ES_KEY_MAP.items():
            td = chars.get(test_name, {})
            es_val = td.get(es_key)
            if es_val is not None and is_estimable(es_val):
                abs_effects.append(abs(float(es_val)))

        max_smd = max(abs_effects) if abs_effects else None
        mean_smd = float(np.mean(abs_effects)) if abs_effects else None

        # IPW diagnostics (propensity model is shared across models)
        prop_diag = ipw_result.get("propensity_diagnostics", {})
        ess = prop_diag.get("effective_sample_size_M1")

        rows = []
        for model in self.models:
            model_ipw = ipw_result.get(model, {})
            # Build status string from available indicators
            status_parts: List[str] = []
            if model_ipw and not model_ipw.get("skipped"):
                status_parts.append("W")  # Weighted analysis completed
                # Acceptable balance: Max |SMD| < 0.25
                if max_smd is not None and max_smd < 0.25:
                    status_parts.append("A")
                # Reasonable ESS retention
                if ess is not None and n_present > 0 and (ess / n_present) > 0.10:
                    status_parts.append("R")
            else:
                status_parts.append("S")  # Skipped

            rows.append((
                model,
                _fmt_int(n_present),
                _fmt_int(n_missing),
                _fmt_pct(pct_present),
                _fmt_effect(max_smd) if max_smd is not None else "---",
                _fmt_effect(mean_smd) if mean_smd is not None else "---",
                _fmt_int(int(ess)) if ess is not None else "---",
                "".join(status_parts),
            ))

        df = pd.DataFrame(rows, columns=[
            "Model", "Amt present", "Amt missing", "Present %",
            "Max |SMD|", "Mean |SMD|", "IPW ESS", "Stat",
        ])

        # Custom compact LaTeX to fit within page width
        note_lines = [
            r"W = IPW analysis completed; "
            r"A = acceptable covariate balance ($\max|\mathrm{SMD}|<0.25$); "
            r"R = reasonable ESS retention ($>10\%$); "
            r"S = skipped.",
        ]
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\footnotesize",
            r"\caption{Monetary missingness and IPW diagnostics (canonical run).}",
            r"\label{tab:missingness_ipw}",
            r"\begin{tabular}{l r r r r r r l}",
            r"\toprule",
            r"Model & Amt present & Amt missing & Present\,\% "
            r"& Max\,$|\mathrm{SMD}|$ & Mean\,$|\mathrm{SMD}|$ & IPW ESS & Stat \\",
            r"\midrule",
        ]
        for _, row in df.iterrows():
            cells = " & ".join(_escape_latex(str(v)) for v in row.values)
            tex_lines.append(cells + r" \\")
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append(r"\par\smallskip")
        tex_lines.append(r"{\scriptsize\textit{Note:}\enspace " + note_lines[0] + "}")
        tex_lines.append(r"\end{table}")

        tex_name = _build_filename(
            "table", "appendix", "missingness_ipw_diagnostics", "na", "all", 1, "tex",
        )
        tex_path = out / tex_name
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # Markdown + CSV via standard helpers
        md_name = _build_filename(
            "table", "appendix", "missingness_ipw_diagnostics", "na", "all", 1, "md",
        )
        md_path = out / md_name
        md_path.write_text(
            _dataframe_to_markdown(df, "Monetary missingness and IPW diagnostics (canonical run)."),
            encoding="utf-8",
        )

        csv_name = _build_filename(
            "table", "appendix", "missingness_ipw_diagnostics", "na", "all", 1, "csv",
        )
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: missingness_ipw_diagnostics (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

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
            caption="Prompt inventory and model parameters.",
            label="tab:prompts",
        )

    def _tableA6_text_feature_inventory(self, out: Path) -> List[Path]:
        """Appendix Table A6: Text feature inventory (static reference)."""
        rows = [
            # Length features
            ("Length", "word_count", "Word Count", "Total words in complaint text"),
            ("Length", "char_count", "Character Count", "Total characters in complaint text"),
            ("Length", "sentence_count", "Sentence Count", "Total sentences in complaint text"),
            ("Length", "avg_word_length", "Avg Word Length", "Mean characters per word"),
            ("Length", "avg_sentence_length", "Avg Sentence Length", "Mean words per sentence"),
            # Readability features
            ("Readability", "flesch_reading_ease", "Flesch Reading Ease", "Readability score (0--100, higher = easier)"),
            ("Readability", "flesch_kincaid_grade", "Flesch-Kincaid Grade", "US grade level required to understand"),
            ("Readability", "gunning_fog", "Gunning Fog Index", "Complexity estimate based on polysyllabic words"),
            ("Readability", "smog_index", "SMOG Index", "Years of education needed to comprehend"),
            ("Readability", "ari", "Automated Readability", "Character-based readability index"),
            # Vocabulary diversity
            ("Vocabulary", "type_token_ratio", "Type-Token Ratio", "Unique words / total words"),
            ("Vocabulary", "hapax_ratio", "Hapax Ratio", "Words appearing once / total words"),
            ("Vocabulary", "mtld", "MTLD", "Measure of textual lexical diversity"),
            ("Vocabulary", "hdd", "HD-D", "Hypergeometric distribution D diversity"),
            # Sentiment features
            ("Sentiment", "polarity", "TextBlob Polarity", "Sentiment polarity (--1 to +1)"),
            ("Sentiment", "subjectivity", "TextBlob Subjectivity", "Subjectivity score (0 to 1)"),
            ("Sentiment", "vader_compound", "VADER Compound", "Normalised composite sentiment (--1 to +1)"),
            ("Sentiment", "vader_pos", "VADER Positive", "Proportion of positive sentiment"),
            ("Sentiment", "vader_neg", "VADER Negative", "Proportion of negative sentiment"),
            ("Sentiment", "vader_neu", "VADER Neutral", "Proportion of neutral sentiment"),
            ("Sentiment", "emotional_intensity", "Emotional Intensity", "Absolute value of VADER compound"),
            # Topic features
            ("Topic", "dominant_topic", "Dominant Topic", "LDA-assigned primary topic (0--4)"),
            ("Topic", "topic_coherence", "Topic Coherence", "Maximum topic probability (concentration)"),
        ]

        df = pd.DataFrame(rows, columns=["Group", "Feature ID", "Display Name", "Definition"])

        # Custom LaTeX with ragged columns
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Text feature inventory: 23 linguistic metrics extracted per complaint.}",
            r"\label{tab:text_features}",
            r"\begin{tabular}{>{\RaggedRight\arraybackslash}p{0.10\linewidth}",
            r"                >{\ttfamily\RaggedRight\arraybackslash\hyphenchar\font=`\_}p{0.18\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.18\linewidth}",
            r"                >{\RaggedRight\arraybackslash}p{0.44\linewidth}}",
            r"\toprule",
            r"Group & Feature ID & Display Name & Definition \\",
            r"\midrule",
        ]
        prev_group = None
        for group, fid, display, defn in rows:
            g = _escape_latex(group) if group != prev_group else ""
            tex_lines.append(f"{g} & {_escape_latex(fid)} & {_escape_latex(display)} & {_escape_latex(defn)} \\\\")
            prev_group = group
            if g and prev_group != rows[-1][0]:
                pass  # no extra rule needed between groups in booktabs style
        tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        tex_name = _build_filename("table", "appendix", "text_feature_inventory", "na", "all", 1, "tex")
        tex_path = out / tex_name
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        md_name = _build_filename("table", "appendix", "text_feature_inventory", "na", "all", 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, "Text feature inventory."), encoding="utf-8")

        csv_name = _build_filename("table", "appendix", "text_feature_inventory", "na", "all", 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: text_feature_inventory (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

    def _tableA7_persona_distribution_diagnostics(self, out: Path, model: str) -> List[Path]:
        """Appendix Table A7: Persona distribution diagnostics per dimension."""
        model_r = self.ar.get(model, {})
        corr = model_r.get("correlation", {})
        rates = corr.get("rates_by_dimension", {})
        if not rates:
            logger.warning("No rates_by_dimension for %s; skipping Table A7.", model)
            return []

        rows = []
        for dim in sorted(rates.keys()):
            dim_rates = rates[dim]
            if not dim_rates:
                continue

            values = list(dim_rates.values())
            n_values = len(values)

            # Compute entropy
            total = sum(values) if values else 1
            probs = [v / total for v in values] if total > 0 else []
            entropy = -sum(p * np.log2(p) for p in probs if p > 0) if probs else 0.0

            # Top value and its share
            top_value = max(dim_rates, key=dim_rates.get) if dim_rates else "---"
            top_share = max(values) / total if total > 0 and values else 0.0

            rows.append((
                dim.capitalize(),
                f"{entropy:.3f}",
                top_value,
                _fmt_pct(top_share * 100, 1),
                str(n_values),
            ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "Dimension", "Entropy", "Top Value", "Top Share", "N Values",
        ])

        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "appendix", "persona_distribution_diagnostics",
            model_clean, "all", 1,
            caption=f"Persona distribution diagnostics -- {model}.",
            label=f"tab:persona_dist_{model_clean}",
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
        ne_ipw_tbl = NE.missing_data(context="IPW table")
        if model_ipw and not model_ipw.get("skipped"):
            dims = model_ipw.get("dimensions", {})
            for dim, dim_r in dims.items():
                uw = dim_r.get("unweighted", {})
                w = dim_r.get("weighted", {})
                rows.append((
                    f"IPW: {dim}",
                    _fmt_effect(uw.get("A", ne_ipw_tbl)),
                    _fmt_effect(w.get("A", ne_ipw_tbl)),
                    _fmt_pval(uw.get("interaction", {}).get("p_value", ne_ipw_tbl)),
                    _fmt_pval(w.get("interaction", {}).get("p_value", ne_ipw_tbl)),
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
                    _fmt_effect(amp_tier.get("point", ne_ipw_tbl)),
                    _fmt_effect(amp_t2.get("point", ne_ipw_tbl)),
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
            caption=f"Robustness summary -- {model}.",
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
            caption="Bayesian model specification and priors.",
            label="tab:bayes_spec",
            col_spec=r"l l >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X l l",
            use_tabularx=True,
            font_size="footnotesize",
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

            ne_bayes_amp = NE.missing_data(context="Bayes amp table")
            for dim, dim_data in amp.items():
                rows.append((
                    f"{label} ({dim})",
                    metric.replace("_", " "),
                    subset,
                    endpoint,
                    _fmt_effect(dim_data.get("A_mean", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("A_CrI_low", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("A_CrI_high", ne_bayes_amp)),
                    _fmt_prop(dim_data.get("P_A_gt_0", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("D_low_mean", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("D_high_mean", ne_bayes_amp)),
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
            caption=f"Bayesian amplification robustness -- {model}, {metric.replace('_', ' ')}.",
            label=f"tab:bayes_amp_{model_clean}_{metric_clean}",
            col_spec=(
                r">{\raggedright\arraybackslash}X "
                r"l l "
                r">{\raggedright\arraybackslash}X "
                r"r r r r r r l"
            ),
            use_tabularx=True,
            font_size="footnotesize",
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

            ne_cri = NE.missing_data(context="CrI")
            def _fmt_cri(d: dict) -> str:
                m = d.get("mean", ne_cri)
                lo = d.get("ci_low", ne_cri)
                hi = d.get("ci_high", ne_cri)
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
            caption=f"Bayesian ordinal tier distribution -- {model}, {metric.replace('_', ' ')}.",
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
                ne_diag_tbl = NE.missing_data(context="Bayes diag table")
                rows.append((
                    mt_data.get("fit_id", f"{model_type}/{metric}"),
                    _fmt_effect(diag.get("max_rhat", ne_diag_tbl)),
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
            caption=f"Bayesian MCMC diagnostics -- {model}.",
            label=f"tab:bayes_diag_{model_clean}",
            col_spec=r">{\raggedright\arraybackslash}X r r r r l",
            use_tabularx=True,
            font_size="footnotesize",
        )
