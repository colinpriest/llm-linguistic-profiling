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

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.estimable import NE, NECode, NEValue, is_estimable

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _fmt_prop(v: Any) -> str:
    """Format a proportion to 3 decimal places."""
    if not is_estimable(v):
        return "---"
    return f"{float(v):.3f}"


def _log_p_exponent(log_p: float) -> int:
    """Convert a natural-log p-value to a base-10 exponent floor.

    E.g. log_p = -500  →  floor(-500 / ln(10)) = floor(-217.1) = -218.
    """
    return int(np.floor(log_p / np.log(10)))


def _fmt_pval(v: Any, log_p: Optional[float] = None) -> str:
    """Format a p-value: 3 dp, '< 0.001' for smaller values.

    Parameters
    ----------
    v : p-value (may be 0.0 due to underflow).
    log_p : optional log(p) computed in log-space (kept for API compat).
    """
    if not is_estimable(v):
        if isinstance(v, NEValue) and v.code == NECode.SINGULAR_MATRIX:
            return "Sep.$^{\\dagger}$"
        return "---"
    fv = float(v)
    if fv < 0.001:
        return "< 0.001"
    return f"{fv:.3f}"


def _fmt_pval_latex(v: Any, log_p: Optional[float] = None) -> str:
    """Format a p-value for raw LaTeX (no escape pass).

    Returns ready-to-embed LaTeX: ``$< 0.001$`` for tiny values, dagger
    for separation, or plain decimal (3 dp).

    Parameters
    ----------
    v : p-value (may be 0.0 due to underflow).
    log_p : optional log(p) computed in log-space (kept for API compat).
    """
    if not is_estimable(v):
        if isinstance(v, NEValue) and v.code == NECode.SINGULAR_MATRIX:
            return r"N/A$^{\dagger}$"
        return "---"
    fv = float(v)
    if fv < 0.001:
        return r"$< 0.001$"
    return f"{fv:.3f}"


def _validate_pval_string(s: str) -> None:
    """Guardrail: ensure a formatted p-value is not misleading.

    Raises ValueError if the string looks like a zero p-value
    (``0.00 × 10^0`` or literal ``"0"``).
    """
    import re
    stripped = s.strip().replace("$", "").replace("\\!", "")
    # Reject "0.00\\times10^{0}" or "0.00e+00" or plain "0"/"0.00"
    if re.match(r"^-?0\.?0*$", stripped):
        raise ValueError(f"p-value formatted as literal zero: {s!r}")
    if re.search(r"0\.00.*(?:times|×).*10\^.*\{0\}", stripped):
        raise ValueError(f"p-value formatted as 0.00×10^0: {s!r}")
    if re.search(r"0\.00e\+?0+$", stripped):
        raise ValueError(f"p-value formatted as 0.00e+00: {s!r}")


def _assert_probability_scale(
    point: float,
    ci_lo: float,
    ci_hi: float,
    *,
    context: str = "",
) -> None:
    """Fail fast if values are not on probability scale [0, 1].

    Catches unit mismatches — statistic on probability scale but
    CI on ratio/OR scale (or vice-versa).

    Raises
    ------
    ValueError
        If any value is outside [0, 1] (with small tolerance) or
        the gap is negative.
    """
    if not np.isfinite(point):
        return
    if point < -1e-6:
        raise ValueError(
            f"Negative gap {point:.6f} for {context}. "
            f"Max-min gap must be non-negative."
        )
    for val, label in [(point, "point"), (ci_lo, "CI lower"), (ci_hi, "CI upper")]:
        if np.isfinite(val) and (val < -0.01 or val > 1.01):
            raise ValueError(
                f"{label}={val:.6f} outside probability scale [0, 1] "
                f"for {context}. This indicates a unit mismatch "
                f"(e.g., odds ratio vs probability gap)."
            )


def _assert_point_in_ci(
    point: float,
    ci_lo: float,
    ci_hi: float,
    *,
    context: str = "",
    tol: float = 5e-4,
) -> None:
    """Fail fast if the displayed point estimate falls outside the CI.

    Tolerance is set at display precision (3 dp -> 5e-4) so rounding
    artefacts do not trigger false positives.

    Raises
    ------
    ValueError
        If the point estimate is outside the CI beyond tolerance.
    """
    if np.isfinite(ci_lo) and np.isfinite(ci_hi):
        if point < ci_lo - tol or point > ci_hi + tol:
            raise ValueError(
                f"Point estimate {point:.6f} outside CI "
                f"[{ci_lo:.6f}, {ci_hi:.6f}] for {context}. "
                f"This indicates the displayed point estimate and CI "
                f"are from different distributions."
            )


def _warn_point_outside_ci(
    point: float,
    ci_lo: float,
    ci_hi: float,
    *,
    context: str = "",
    tol: float = 5e-4,
) -> None:
    """Log a warning if a non-smooth statistic falls outside its CI.

    Used for max-min range, where the full-sample value can legitimately
    sit below the bootstrap CI due to positive resampling bias.
    """
    if np.isfinite(ci_lo) and np.isfinite(ci_hi):
        if point < ci_lo - tol or point > ci_hi + tol:
            logger.warning(
                "Range estimate %.6f outside CI [%.6f, %.6f] for %s "
                "(expected for non-smooth range statistic; descriptive only).",
                point, ci_lo, ci_hi, context,
            )


def _fmt_effect(v: Any) -> str:
    """Format an effect size to 3 decimal places."""
    if not is_estimable(v):
        return "---"
    return f"{float(v):.3f}"


def _fmt_pp(v: Any, decimals: int = 1) -> str:
    """Format a probability as percentage points (pp).

    Multiplies a [0, 1] probability by 100 for display.
    """
    if not is_estimable(v):
        return "---"
    return f"{float(v) * 100:.{decimals}f}"


def _fmt_int(v) -> str:
    """Format an integer."""
    try:
        return f"{int(v):,}"
    except (ValueError, TypeError):
        return str(v)


def _fmt_pct(v: Any, decimals: int = 1) -> str:
    """Format a percentage (avoids '-0.0%')."""
    if not is_estimable(v):
        return "---"
    rounded = round(float(v), decimals)
    if rounded == 0:
        rounded = 0.0  # eliminate negative zero
    return f"{rounded:.{decimals}f}%"


def _fmt_pct_tex(v: Any, decimals: int = 1) -> str:
    """Format a percentage with escaped ``%`` for LaTeX (avoids '-0.0\\%')."""
    if not is_estimable(v):
        return "---"
    rounded = round(float(v), decimals)
    if rounded == 0:
        rounded = 0.0  # eliminate negative zero
    return f"{rounded:.{decimals}f}\\%"


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

    # Derive percent change from SDs (preferred) or legacy variance ratio.
    if "pct_change" not in vc:
        if "baseline_sd" in vc and "neutralized_sd" in vc:
            base = vc.get("baseline_sd")
            neut = vc.get("neutralized_sd")
            if is_estimable(base) and is_estimable(neut) and float(base) != 0:
                vc["pct_change"] = (float(neut) - float(base)) / float(base) * 100.0
        elif "variance_reduction" in block:
            # Legacy fallback: convert variance ratio to SD percent change.
            try:
                vr = float(block["variance_reduction"])
                vc["pct_change"] = (np.sqrt(max(0.0, 1 - vr)) - 1) * 100.0
            except (TypeError, ValueError):
                pass

    return vc


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    s = s.replace("\\", "\\textbackslash{}")
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        s = s.replace(char, f"\\{char}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


# ------------------------------------------------------------------
# Provenance comment
# ------------------------------------------------------------------

def _build_source_comment(**fields: str) -> str:
    """Build a ``% SOURCE: ...`` LaTeX comment for table provenance.

    The comment is invisible in PDF output but makes it trivial to trace
    which dataframe / code-path produced a given ``.tex`` file.

    Example::

        % SOURCE: severity_df | model=gpt-5-mini | endpoint=tier2_gap | units=pp
    """
    if not fields:
        return ""
    parts: list[str] = []
    # ``source`` always comes first when present.
    if "source" in fields:
        parts.append(fields["source"])
    for k, v in fields.items():
        if k != "source":
            parts.append(f"{k}={v}")
    return "% SOURCE: " + " | ".join(parts)


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
    source_meta: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Write a DataFrame as .tex, .md, and .csv. Returns list of paths."""
    paths: List[Path] = []

    # --- LaTeX ---
    tex_name = _build_filename(kind, scope, topic, model, subset, version, "tex")
    tex_path = output_dir / tex_name
    comment = _build_source_comment(**(source_meta or {}))
    tex_lines = _dataframe_to_booktabs(
        df, caption, label,
        col_spec=col_spec, use_tabularx=use_tabularx, font_size=font_size,
        source_comment=comment,
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
    source_comment: str = "",
) -> str:
    """Convert DataFrame to booktabs LaTeX table.

    Parameters
    ----------
    col_spec : optional override for column alignment string.
    use_tabularx : if True, emit ``tabularx`` (requires ``\\usepackage{tabularx}``
        in the preamble).  At least one ``X`` column is expected in *col_spec*.
    font_size : e.g. ``"small"`` or ``"footnotesize"``; wrapped around the table
        body so the table shrinks without affecting the caption.
    source_comment : ``% SOURCE: ...`` provenance comment prepended before
        ``\\begin{table}``.
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

    lines: list[str] = []
    if source_comment:
        lines.append(source_comment)
    lines.extend([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{" + label + "}",
    ])
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
        complaints: List[Any],
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
        all_paths.extend(self._generate_debiased_tables())
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
        for model in self.models:
            paths.extend(self._table11_robustness_ladder(main_dir, model))
        paths.extend(self._table12_endpoint_summary(main_dir))
        paths.extend(self._table13_permutation_test(main_dir))
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
            source_meta={"source": "run_config", "model": "na", "endpoint": "na", "units": "count"},
        )

    def _table2_endpoint_definitions(self, out: Path) -> List[Path]:
        """Table 2: Endpoint definitions."""
        rows = [
            ("T in {0,1,2}", "Ordered tier: no action (0), non-monetary (1), monetary (2)", "Primary distributive endpoint", "Primary"),
            ("Y2 = 1[T=2]", "Binary: monetary remediation", "Threshold summary of T", "Secondary"),
            ("Y1+ = 1[T>=1]", "Binary: any positive remediation", "Sensitivity check", "Exploratory"),
            ("F = 1[need_more_info]", "Model requests additional information", "Operational friction", "Exploratory"),
            ("M = 1[amount_extractable]", "Dollar amount present indicator", "Two-part severity handling", "Descriptive"),
            ("S in {L,M,H}", "Monetary severity band (within M=1)", "Severity amplification", "Confirmatory (M=1)"),
            ("Issue severity", "Product x issue derived score", "Severity amplification", "Confirmatory (all)"),
        ]
        df = pd.DataFrame(rows, columns=["Endpoint", "Definition", "Rationale", "Used as"])

        # --- Custom LaTeX with ragged columns ---
        tex_rows = [
            (r"$T\in\{0,1,2\}$", r"Ordered tier: no action (0), non-monetary (1), monetary (2)", "Primary distributive endpoint", "Primary"),
            (r"$Y_2=\mathbf{1}[T=2]$", "Binary: monetary remediation", r"Threshold summary of $T$", "Secondary"),
            (r"$Y_1^{+}=\mathbf{1}[T\ge 1]$", "Binary: any positive remediation", "Sensitivity check", "Exploratory"),
            (r"\indcode{F}{need_more_info}", "Model requests additional information", "Operational friction", "Exploratory"),
            (r"\indcode{M}{amount_extractable}", "Dollar amount present indicator", "Two-part severity handling", "Descriptive"),
            (r"$S\in\{L,M,H\}$", r"Monetary severity band (within $M=1$)", "Severity amplification", r"Confirmatory ($M=1$)"),
            ("Issue severity", r"Product $\times$ issue derived score", "Severity amplification", "Confirmatory (all)"),
        ]
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            r"\caption{Endpoint definitions and analysis roles.}",
            r"\label{tab:endpoints}",
            "",
            r"\begin{tabularx}{\linewidth}{@{}",
            r"  >{\RaggedRight\arraybackslash}p{0.20\linewidth}",
            r"  >{\RaggedRight\arraybackslash\hsize=1.08\hsize}X",
            r"  >{\RaggedRight\arraybackslash\hsize=0.92\hsize}X",
            r"  >{\RaggedRight\arraybackslash}p{0.18\linewidth}",
            r"@{}}",
            r"\toprule",
            r"Endpoint & Definition & Rationale & Used as \\",
            r"\midrule",
        ]
        for endpoint, definition, rationale, used_as in tex_rows:
            tex_lines.append(f"{endpoint} & {definition} & {rationale} & {used_as} \\\\")
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabularx}")
        tex_lines.append(r"\end{table}")

        tex_name = _build_filename("table", "main", "endpoint_definitions", "na", "all", 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="static", model="na", endpoint="na"))
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
            source_meta={"source": "static", "model": "na", "endpoint": "na"},
        )

    def _table4_distributive_fairness(self, out: Path, model: str) -> List[Path]:
        """Table 4: Distributive fairness results (matched-guise).

        Two panels:
          A — Monetary remediation (Tier = 2)
          B — Acknowledgement (Tier >= 1)

        Per-dimension rows show SD (smooth, CI-bearing) and max-min
        (descriptive).  Full-grid row likewise.  All probability-scale
        values rendered in percentage points (pp).
        """
        model_r = self.ar.get(model, {})
        causal = model_r.get("causal", {})
        gaps_tier2 = causal.get("dimension_gaps_tier2", {})
        gaps_ack = causal.get("dimension_gaps_ack", {})

        if not gaps_tier2 and not gaps_ack:
            logger.warning(
                "No dimension_gaps in causal results for %s; "
                "skipping distributive_fairness table.",
                model,
            )
            return []

        # LaTeX dimension labels
        _dim_tex = {
            "age": "Age",
            "geographic": "Geographic",
            "education": "Education",
            "socioeconomic": r"Socio\-economic",
        }
        _dims = ["age", "geographic", "education", "socioeconomic"]
        _blank = ("---", "---", "---", "---")

        def _build_panel_rows(dim_gaps, endpoint_label):
            """Build row tuples for one endpoint panel."""
            plain = []
            tex = []
            dagger = False
            for dim in _dims:
                dg = dim_gaps.get(dim)
                if not dg:
                    plain.append((dim.capitalize(), *_blank))
                    tex.append((_dim_tex[dim], *_blank))
                    continue
                gap = dg["gap"]
                sd = dg["sd"]
                sd_ci_lo = dg["sd_ci_lower"]
                sd_ci_hi = dg["sd_ci_upper"]
                p_val = dg["p_value"]
                _assert_probability_scale(
                    sd, sd_ci_lo, sd_ci_hi,
                    context=f"distributive_fairness/{model}/{endpoint_label}/{dim}/sd",
                )
                _assert_point_in_ci(
                    sd, sd_ci_lo, sd_ci_hi,
                    context=f"distributive_fairness/{model}/{endpoint_label}/{dim}/sd",
                )
                ci_str = f"[{_fmt_pp(sd_ci_lo)}, {_fmt_pp(sd_ci_hi)}]"
                p_latex = _fmt_pval_latex(p_val)
                if r"\dagger" in p_latex:
                    dagger = True
                plain.append((dim.capitalize(), _fmt_pp(sd), ci_str, _fmt_pp(gap), _fmt_pval(p_val)))
                tex.append((_dim_tex[dim], _fmt_pp(sd), ci_str, _fmt_pp(gap), p_latex))

            fg = dim_gaps.get("full_grid")
            if fg:
                fg_sd = fg.get("sd", float("nan"))
                fg_sd_ci_lo = fg.get("sd_ci_lower", float("nan"))
                fg_sd_ci_hi = fg.get("sd_ci_upper", float("nan"))
                fg_gap = fg["gap"]
                if np.isfinite(fg_sd):
                    _assert_probability_scale(
                        fg_sd, fg_sd_ci_lo, fg_sd_ci_hi,
                        context=f"distributive_fairness/{model}/{endpoint_label}/full_grid/sd",
                    )
                    _assert_point_in_ci(
                        fg_sd, fg_sd_ci_lo, fg_sd_ci_hi,
                        context=f"distributive_fairness/{model}/{endpoint_label}/full_grid/sd",
                    )
                    fg_ci_str = f"[{_fmt_pp(fg_sd_ci_lo)}, {_fmt_pp(fg_sd_ci_hi)}]"
                    frow = ("Full persona grid", _fmt_pp(fg_sd), fg_ci_str, _fmt_pp(fg_gap), "---")
                else:
                    frow = ("Full persona grid", *_blank)
            else:
                frow = ("Full persona grid", *_blank)
            plain.append(frow)
            tex.append(frow)
            return plain, tex, dagger

        rows_t2, tex_t2, dag_t2 = _build_panel_rows(gaps_tier2, "tier2")
        rows_ack, tex_ack, dag_ack = _build_panel_rows(gaps_ack, "ack")
        needs_dagger = dag_t2 or dag_ack

        # Combined plain-text rows for CSV/markdown
        rows = [("Panel A: Monetary remediation (Tier = 2)", "", "", "", "")]
        rows.extend(rows_t2)
        rows.append(("Panel B: Acknowledgement (Tier >= 1)", "", "", "", ""))
        rows.extend(rows_ack)

        # --- Custom LaTeX ---
        model_clean = model.replace("-", "").replace("_", "")
        caption = f"Distributive fairness results (matched-guise) -- {model}."
        label = f"tab:fairness_{model_clean}"

        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{" + _escape_latex(caption) + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{@{}p{0.26\linewidth}r p{0.22\linewidth}r p{0.15\linewidth}@{}}",
            r"\toprule",
            (r"Dimension & $\sigma$ (pp)"
             r" & 95\% CI$^{\ast}$ & Max--Min (pp) & $p$ \\"),
            r"\midrule",
            r"\multicolumn{5}{@{}l}{\textit{Panel A: Monetary remediation (Tier\,=\,2)}} \\",
        ]
        for dim_label, smooth_val, ci, mm, p in tex_t2:
            tex_lines.append(f"{dim_label} & {smooth_val} & {ci} & {mm} & {p} \\\\")
        tex_lines.append(r"\addlinespace")
        tex_lines.append(
            r"\multicolumn{5}{@{}l}{\textit{Panel B: Acknowledgement (Tier\,$\geq$\,1)}} \\",
        )
        for dim_label, smooth_val, ci, mm, p in tex_ack:
            tex_lines.append(f"{dim_label} & {smooth_val} & {ci} & {mm} & {p} \\\\")
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        # Footnotes
        tex_lines.append("")
        tex_lines.append(r"\vspace{2pt}")
        tex_lines.append(
            r"\parbox{0.94\linewidth}{\footnotesize"
            r" $\sigma$: SD of persona-level estimates (ddof\,=\,0)."
            r" $^{\ast}$ 95\% CI (log-scale bootstrap;"
            r" complaint-level resampling)."
            r" Max--Min: full range (descriptive, no CI)."
            r" Panel A uses $\mathbf{1}[\text{tier}=2]$;"
            r" Panel B uses $\mathbf{1}[\text{tier}\geq 1]$."
        )
        if needs_dagger:
            tex_lines.append(
                r" $^{\dagger}$ $p$-value unavailable / not estimable."
            )
        tex_lines.append("}")
        tex_lines.append(r"\end{table}")

        tex_name = _build_filename("table", "main", "distributive_fairness", model_clean, "matched", 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="causal.dimension_gaps_tier2+ack", model=model, endpoint="tier2+ack", units="pp", ci="log_scale_bootstrap"))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV via DataFrame ---
        df = pd.DataFrame(rows, columns=["Dimension", "SD (pp)", "95% CI (log-scale bootstrap)", "Max-Min (pp)", "p"])

        md_name = _build_filename("table", "main", "distributive_fairness", model_clean, "matched", 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, caption), encoding="utf-8")

        csv_name = _build_filename("table", "main", "distributive_fairness", model_clean, "matched", 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: distributive_fairness (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

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
            "Outcome disparity (SD)",
            _fmt_effect(baseline_sd),
            _fmt_effect(neutralized_sd),
            _fmt_effect(neutralized_sd - baseline_sd) if is_estimable(baseline_sd) and is_estimable(neutralized_sd) else "---",
            _fmt_pct(pct_change) if is_estimable(pct_change) else "---",
            "Eliminated" if is_estimable(pct_change) and pct_change < -95 else
            "Reduced" if is_estimable(pct_change) and pct_change < -5 else
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
            source_meta={"source": "neutralisation_results", "model": model, "endpoint": "tier2_mcnemar", "units": "pp"},
        )

    def _table6_amplification(self, out: Path, model: str, metric: str) -> List[Path]:
        """Table 6: Severity amplification (confirmatory)."""
        severity = self.ar.get("severity", {})
        interactions_t2 = severity.get("interactions_tier2", {}).get(metric, {}).get(model, {})
        dim_sev = severity.get("dimension_severity", {}).get(metric, {}).get(model, {})
        gee = severity.get("gee_interactions_tier2", {}).get(metric, {}).get(model, {})
        miss_diag = severity.get("missingness_diagnostic", {})

        rows = []          # plain-text for md/csv
        tex_rows = []      # LaTeX-ready
        has_separation = False
        ne_amp_t6 = NE.missing_data(context="amplification table")
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            gee_dim = gee.get(dim, {})
            if gee_dim.get("skipped"):
                int_p = NE.skipped("GEE skipped")
            elif gee_dim.get("separation_detected"):
                int_p = NE.singular_matrix(
                    detail="separation",
                    context=f"GEE interaction {dim}",
                )
            else:
                int_p = gee_dim.get("interaction_p", ne_amp_t6)

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
            coverage_tex = ""
            if metric == "monetary_severity" and miss_diag:
                coverage = _fmt_pct(100 * (1 - miss_diag.get("pct_missing", 0)))
                coverage_tex = _fmt_pct_tex(100 * (1 - miss_diag.get("pct_missing", 0)))

            p_plain = _fmt_pval(int_p)
            p_latex = _fmt_pval_latex(int_p)
            if r"\dagger" in p_latex:
                has_separation = True

            ci_str = f"[{_fmt_effect(ci_lo)}, {_fmt_effect(ci_hi)}]"
            row_plain = [dim.capitalize(), p_plain, _fmt_effect(A), _fmt_effect(R), ci_str]
            row_tex = [dim.capitalize(), p_latex, _fmt_effect(A), _fmt_effect(R), ci_str]
            if metric == "monetary_severity":
                row_plain.append(coverage)
                row_tex.append(coverage_tex)
            rows.append(tuple(row_plain))
            tex_rows.append(tuple(row_tex))

        columns = ["Dimension", "Interaction p", "A (high-low)", "R (ratio)", "95% CI (percentile bootstrap)"]
        if metric == "monetary_severity":
            columns.append("Coverage")
        df = pd.DataFrame(rows, columns=columns)

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        subset = "amount_present" if metric == "monetary_severity" else "all"
        caption = f"Severity amplification (confirmatory) -- {model}, {metric.replace('_', ' ')}."
        label = f"tab:amplification_{model_clean}_{metric_clean}"

        # --- Custom LaTeX ---
        col_spec = "lrrrr"
        if metric == "monetary_severity":
            col_spec += "r"
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + _escape_latex(caption) + "}",
            r"\label{" + label + "}",
            r"\begin{tabular}{" + col_spec + "}",
            r"\toprule",
        ]
        # Build LaTeX header — replace long CI label with footnote marker
        tex_columns = list(columns)
        ci_idx = next(i for i, c in enumerate(tex_columns) if c.startswith("95% CI"))
        tex_columns[ci_idx] = r"95\% CI$^{\ast}$"
        header = " & ".join(tex_columns) + r" \\"
        tex_lines.append(header)
        tex_lines.append(r"\midrule")
        for row_tex in tex_rows:
            tex_lines.append(" & ".join(str(c) for c in row_tex) + r" \\")
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        # Footnotes
        footnotes: List[str] = []
        footnotes.append(
            r"$^{\ast}$ 95\% CI (percentile bootstrap)."
        )
        if has_separation:
            footnotes.append(
                r"$^\dagger$ Sep."
                r" = quasi-complete separation in persona $\times$ severity"
                r" cells; Wald test not identified."
            )
        tex_lines.append("")
        tex_lines.append(r"\vspace{2pt}")
        tex_lines.append(
            r"\parbox{0.95\linewidth}{\footnotesize "
            + r"\\".join(footnotes)
            + "}"
        )
        tex_lines.append(r"\end{table}")

        topic = "severity_amplification_confirmatory"
        tex_name = _build_filename("table", "main", topic, model_clean, f"{metric_clean}__{subset}", 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="severity_analysis", model=model, endpoint="amplification", units="ratio", metric=metric))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV via DataFrame ---
        md_name = _build_filename("table", "main", topic, model_clean, f"{metric_clean}__{subset}", 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, caption), encoding="utf-8")

        csv_name = _build_filename("table", "main", topic, model_clean, f"{metric_clean}__{subset}", 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: %s (.tex/.md/.csv)", topic)
        return [tex_path, md_path, csv_path]

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
                        auc = val_data.get("training_auc")
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

        # Custom LaTeX – longtable with ragged columns for wide table
        # Column widths: 0.11+0.08+0.09+0.32+0.05+0.25 = 0.90 (leaves room for \tabcolsep)
        tex_lines = [
            r"% ---------- Table ----------",
            r"\small",
            r"\begin{longtable}{@{}",
            r"    >{\RaggedRight\arraybackslash}p{0.11\linewidth}",
            r"    >{\RaggedRight\arraybackslash}p{0.08\linewidth}",
            r"    >{\RaggedRight\arraybackslash}p{0.09\linewidth}",
            r"    >{\RaggedRight\arraybackslash}p{0.32\linewidth}",
            r"    >{\RaggedLeft\arraybackslash}p{0.05\linewidth}",
            r"    >{\RaggedRight\arraybackslash}p{0.25\linewidth}@{}}",
            r"\caption{Top text features predicting persona dimensions (LightGBM classifiers).}",
            r"\label{tab:persona_explainability}\\",
            r"\toprule",
            r"Dimension & Value & Model & Top 3 Features & AUC & Interpretation \\",
            r"\midrule",
            r"\endfirsthead",
            r"",
            r"\multicolumn{6}{@{}l}{\small\itshape Table \thetable\ (continued)}\\",
            r"\toprule",
            r"Dimension & Value & Model & Top 3 Features & AUC & Interpretation \\",
            r"\midrule",
            r"\endhead",
            r"",
            r"\midrule",
            r"\multicolumn{6}{r@{}}{\small\itshape Continued on next page}\\",
            r"\endfoot",
            r"",
            r"\bottomrule",
            r"\endlastfoot",
            r"",
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
                f"{dim_esc} & {val_esc} & {model_esc} & {feats_latex} & {auc_esc} & {interp_esc} \\\\",
            )
        tex_lines.append(r"\end{longtable}")

        tex_name = _build_filename("table", "main", "persona_explainability", "na", "all", 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="text_feature_importance", model="all", endpoint="logistic_coeff", units="mixed"))
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
            source_meta={"source": "manipulation_checks", "model": "all", "endpoint": "match_rate", "units": "proportion"},
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
            source_meta={"source": "signal_reduction", "model": "all", "endpoint": "entropy", "units": "bits"},
        )

    def _table10_operational_friction(self, out: Path, model: str) -> List[Path]:
        """Table 10: Operational friction (need_more_info) rates by dimension."""
        from src.schemas import CAUSAL_DIMS

        model_r = self.ar.get(model, {})
        proc_corr = model_r.get("process_correlation", {})
        info_rates = proc_corr.get("info_rates_by_dimension", {})
        if not info_rates:
            logger.warning("No info_rates_by_dimension for %s; skipping Table 10.", model)
            return []

        # Also try to get causal process rates if available
        proc_causal = model_r.get("process_causal", {})
        causal_rates = proc_causal.get("info_rates_by_dimension", {}) if proc_causal else {}

        rows = []          # plain-text for md/csv
        tex_rows = []      # LaTeX-ready
        dagger_dims_seen: set = set()   # first occurrence gets the dagger
        has_estimation_only = False

        for dim in sorted(info_rates.keys()):
            dim_rates = info_rates[dim]
            causal_dim = causal_rates.get(dim, {})
            is_estimation_only = dim not in CAUSAL_DIMS
            for value in sorted(dim_rates.keys()):
                corr_rate = dim_rates[value]
                caus_rate = causal_dim.get(value)
                corr_str = _fmt_pct(corr_rate * 100, 1) if corr_rate is not None else "---"
                caus_str = _fmt_pct(caus_rate * 100, 1) if caus_rate is not None else "---"
                corr_tex = _fmt_pct_tex(corr_rate * 100, 1) if corr_rate is not None else "---"
                caus_tex = _fmt_pct_tex(caus_rate * 100, 1) if caus_rate is not None else "---"

                rows.append((dim.capitalize(), value, corr_str, caus_str))

                # LaTeX: add dagger on first row of each estimation-only dim
                dim_label = dim.capitalize()
                if is_estimation_only and dim not in dagger_dims_seen:
                    dim_label = dim_label + r"$^{\dagger}$"
                    dagger_dims_seen.add(dim)
                    has_estimation_only = True
                tex_rows.append((dim_label, value, corr_tex, caus_tex))

        if not rows:
            return []

        # --- Custom LaTeX ---
        model_clean = model.replace("-", "").replace("_", "")
        caption = f"Operational friction (need_more_info) rates by dimension -- {model}."
        label = f"tab:friction_{model_clean}"

        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Operational friction (\texttt{need\_more\_info}) rates by dimension -- "
            + _escape_latex(model) + ".}",
            r"\label{" + label + "}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Dimension & Value & Correlational Rate & Causal Rate \\",
            r"\midrule",
        ]
        for dim_label, value, corr, caus in tex_rows:
            tex_lines.append(
                f"{dim_label} & {_escape_latex(value)} & {corr} & {caus} \\\\"
            )
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        if has_estimation_only:
            tex_lines.append("")
            tex_lines.append(r"\vspace{2pt}")
            tex_lines.append(
                r"\parbox{0.95\linewidth}{\footnotesize $^{\dagger}$"
                r" ESL and Gender were estimation-only dimensions and were not"
                r" included in the causal analysis; therefore, causal rates are"
                r" reported as `---'.}"
            )
        tex_lines.append(r"\end{table}")

        tex_name = _build_filename("table", "main", "operational_friction", model_clean, "all", 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="operational_friction", model=model, endpoint="info_request_rate", units="proportion"))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV via DataFrame ---
        df = pd.DataFrame(rows, columns=[
            "Dimension", "Value", "Correlational Rate", "Causal Rate",
        ])

        md_name = _build_filename("table", "main", "operational_friction", model_clean, "all", 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, caption), encoding="utf-8")

        csv_name = _build_filename("table", "main", "operational_friction", model_clean, "all", 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: operational_friction (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

    # ------------------------------------------------------------------
    # Table 11: Robustness ladder (per model)
    # ------------------------------------------------------------------

    def _table11_robustness_ladder(self, out: Path, model: str) -> List[Path]:
        """Table 11: Robustness ladder — effect sizes across data subsets."""
        rob = self.ar.get("robustness_subsets", {})
        if not rob or rob.get("skipped"):
            return []

        rob_dir = Path(rob.get("output_dir", ""))
        ep_path = rob_dir / "endpoint_summary.csv"
        om_path = rob_dir / "omnibus_tests.csv"
        if not ep_path.exists() or not om_path.exists():
            return []

        ep_df = pd.read_csv(ep_path)
        om_df = pd.read_csv(om_path)

        # Guard: require SD columns (generated by updated robustness code v3)
        _required_cols = [
            "sd_persona_p_tier2", "sd_persona_mean_tier",
            "ci_lower_sd_persona_p_tier2", "ci_upper_sd_persona_p_tier2",
            "ci_lower_sd_persona_mean_tier", "ci_upper_sd_persona_mean_tier",
        ]
        missing = [c for c in _required_cols if c not in ep_df.columns]
        if missing:
            logger.error(
                "endpoint_summary.csv missing columns %s; "
                "re-run robustness subset analysis to regenerate. "
                "Skipping robustness_ladder table for %s.",
                missing, model,
            )
            return []

        # Filter to this model
        ep = ep_df[ep_df["model_name"] == model]
        om = om_df[om_df["model_name"] == model]
        if ep.empty:
            return []

        # HF pair rate from slot audit (for footnote)
        sa = self.ar.get("slot_audit", {})
        hf_metrics = sa.get("metrics", {}).get("global", {}).get("high_fidelity", {})
        hf_rate = hf_metrics.get("pair_rate")
        hf_pct = f"{hf_rate * 100:.1f}" if hf_rate is not None else "?"

        # Subset order (skip HF_NO_POL — duplicates HF in this dataset)
        subset_order = ["FULL", "HF", "NO_POL", "NO_NUM"]
        subset_labels = {
            "FULL": "FULL",
            "HF": r"HF$^{a}$",
            "NO_POL": r"NO\_POL$^{a}$",
            "NO_NUM": r"NO\_NUM$^{a}$",
        }

        tex_rows = []
        md_rows = []
        for subset in subset_order:
            row = ep[ep["subset"] == subset]
            if row.empty:
                continue
            r = row.iloc[0]

            ret = r["pair_retention"]

            # SD (inferential, CI-bearing — normal CI centered on plug-in)
            sd_t2 = r["sd_persona_p_tier2"]
            sd_ci_lo_t2 = r["ci_lower_sd_persona_p_tier2"]
            sd_ci_hi_t2 = r["ci_upper_sd_persona_p_tier2"]
            sd_mt = r["sd_persona_mean_tier"]
            sd_ci_lo_mt = r["ci_lower_sd_persona_mean_tier"]
            sd_ci_hi_mt = r["ci_upper_sd_persona_mean_tier"]

            # Q90-Q10 (descriptive point only)
            q_t2 = r.get("q9010_p_tier2", float("nan"))
            q_mt = r.get("q9010_mean_tier", float("nan"))

            # Max-min (descriptive point only)
            mm_t2 = r["max_persona_gap_p_tier2"]
            mm_mt = r["max_persona_gap_mean_tier"]

            # Strict assertions on SD (containment holds by construction)
            _assert_probability_scale(
                sd_t2, sd_ci_lo_t2, sd_ci_hi_t2,
                context=f"robustness_ladder/{model}/{subset}/sd_tier2",
            )
            _assert_point_in_ci(
                sd_t2, sd_ci_lo_t2, sd_ci_hi_t2,
                context=f"robustness_ladder/{model}/{subset}/sd_tier2",
            )
            _assert_point_in_ci(
                sd_mt, sd_ci_lo_mt, sd_ci_hi_mt,
                context=f"robustness_ladder/{model}/{subset}/sd_mean_tier",
            )

            # Omnibus p-values (with log-space fallback for underflow)
            om_t2 = om[(om["subset"] == subset) & (om["endpoint"] == "tier2")]
            om_mt = om[(om["subset"] == subset) & (om["endpoint"] == "tier_numeric")]
            p_t2 = om_t2.iloc[0]["p_value"] if not om_t2.empty else None
            p_mt = om_mt.iloc[0]["p_value"] if not om_mt.empty else None
            log_p_t2 = om_t2.iloc[0].get("log_p_value") if not om_t2.empty else None
            log_p_mt = om_mt.iloc[0].get("log_p_value") if not om_mt.empty else None

            # Format: SD [CI] in pp for tier2; raw for mean_tier;
            # Q90-Q10 and Max-Min as descriptive points
            tex_rows.append((
                subset_labels.get(subset, subset),
                f"{ret * 100:.1f}\\%",
                f"{_fmt_pp(sd_t2)} [{_fmt_pp(sd_ci_lo_t2)}, {_fmt_pp(sd_ci_hi_t2)}]",
                f"{sd_mt:.4f} [{sd_ci_lo_mt:.4f}, {sd_ci_hi_mt:.4f}]",
                _fmt_pp(q_t2),
                f"{q_mt:.4f}",
                _fmt_pp(mm_t2),
                f"{mm_mt:.4f}",
                _fmt_pval_latex(p_t2, log_p=log_p_t2),
                _fmt_pval_latex(p_mt, log_p=log_p_mt),
            ))
            md_rows.append((
                subset, f"{ret * 100:.1f}%",
                f"{_fmt_pp(sd_t2)} [{_fmt_pp(sd_ci_lo_t2)}, {_fmt_pp(sd_ci_hi_t2)}]",
                f"{sd_mt:.4f} [{sd_ci_lo_mt:.4f}, {sd_ci_hi_mt:.4f}]",
                _fmt_pp(q_t2),
                f"{q_mt:.4f}",
                _fmt_pp(mm_t2),
                f"{mm_mt:.4f}",
                _fmt_pval(p_t2, log_p=log_p_t2),
                _fmt_pval(p_mt, log_p=log_p_mt),
            ))

        if not tex_rows:
            return []

        model_clean = model.replace("-", "").replace("_", "")

        # --- Custom LaTeX ---
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            r"\caption{Robustness ladder: effect sizes across data subsets -- "
            + _escape_latex(model) + r".}",
            r"\label{tab:robustness_ladder_" + model_clean + "}",
            r"\begin{tabular}{@{}l r ll rr rr cc@{}}",
            r"\toprule",
            r"Subset & Ret."
            r" & $\sigma\;P(T{=}2)$$^{b}$ [CI] (pp)"
            r" & $\sigma\;\bar{\tau}$$^{b}$ [CI]"
            r" & $\Delta_{90\text{-}10}$$^{b}$ (pp)"
            r" & $\Delta_{90\text{-}10}$$^{b}$"
            r" & $\Delta_{\max}$$^{b}$ (pp)"
            r" & $\Delta_{\max}$$^{b}$"
            r" & $p_W(Y_2)$$^{c}$ & $p_W(\bar{\tau})$$^{c}$ \\",
            r"\midrule",
        ]
        for row_tuple in tex_rows:
            lbl, ret, sdt2, sdmt, qt2, qmt, mmt2, mmmt, pt2, pmt = row_tuple
            tex_lines.append(
                f"{lbl} & {ret} & {sdt2} & {sdmt}"
                f" & {qt2} & {qmt} & {mmt2} & {mmmt}"
                f" & {pt2} & {pmt} \\\\"
            )
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append("")
        tex_lines.append(r"\vspace{2pt}")
        tex_lines.append(
            r"\parbox{0.95\linewidth}{\footnotesize"
        )
        tex_lines.append(
            r"$^{a}$ HF = high-fidelity rewrite pairs ("
            + hf_pct
            + r"\% of all pairs); NO\_POL = no polarity contradictions;"
            r" NO\_NUM = no numeric mutations.\\"
        )
        tex_lines.append(
            r"$^{b}$ $\sigma$: SD of persona-level estimates."
            r" 95\% CI (log-scale bootstrap;"
            r" complaint-level resampling)."
            r" $\Delta_{90\text{-}10}$: 90th--10th percentile spread (descriptive)."
            r" $\Delta_{\max}$: max--min range (descriptive).\\"
        )
        tex_lines.append(
            r"$^{c}$ $p_W$: GEE omnibus Wald test, exchangeable working correlation."
            r" $Y_2 = P(T{=}2)$; $\bar{\tau}$ = mean tier."
        )
        tex_lines.append(r"}")
        tex_lines.append(r"\end{table}")

        tex_name = _build_filename(
            "table", "main", "robustness_ladder", model_clean, "all", 1, "tex"
        )
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="robustness_subsets", model=model, endpoint="tier2_gap", units="pp"))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV ---
        df = pd.DataFrame(md_rows, columns=[
            "Subset", "Retention",
            "SD P(T=2) [95% CI, log-scale bootstrap] (pp)", "SD mean tier [95% CI, log-scale bootstrap]",
            "Q90-Q10 P(T=2) (pp)", "Q90-Q10 mean tier",
            "Max-Min P(T=2) (pp)", "Max-Min mean tier",
            "p_W (tier2)", "p_W (mean tier)",
        ])
        md_name = _build_filename(
            "table", "main", "robustness_ladder", model_clean, "all", 1, "md"
        )
        md_path = out / md_name
        caption_md = f"Robustness ladder: effect sizes across data subsets -- {model}."
        md_path.write_text(_dataframe_to_markdown(df, caption_md), encoding="utf-8")

        csv_name = _build_filename(
            "table", "main", "robustness_ladder", model_clean, "all", 1, "csv"
        )
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: robustness_ladder (%s) (.tex/.md/.csv)", model)
        return [tex_path, md_path, csv_path]

    # ------------------------------------------------------------------
    # Table 12: Endpoint summary (compact, both models)
    # ------------------------------------------------------------------

    def _table12_endpoint_summary(self, out: Path) -> List[Path]:
        """Table 12: Compact primary + secondary endpoint dispersion summary."""
        rob = self.ar.get("robustness_subsets", {})
        if not rob or rob.get("skipped"):
            return []

        rob_dir = Path(rob.get("output_dir", ""))
        ep_path = rob_dir / "endpoint_summary.csv"
        om_path = rob_dir / "omnibus_tests.csv"
        if not ep_path.exists() or not om_path.exists():
            return []

        ep_df = pd.read_csv(ep_path)
        om_df = pd.read_csv(om_path)

        # Guard: require SD columns (generated by updated robustness code v3)
        _sd_cols = ["sd_persona_p_tier2", "sd_persona_mean_tier"]
        missing = [c for c in _sd_cols if c not in ep_df.columns]
        if missing:
            logger.error(
                "endpoint_summary.csv missing columns %s; "
                "re-run robustness subset analysis to regenerate. "
                "Skipping endpoint_summary table.",
                missing,
            )
            return []

        tex_data_rows = []
        md_rows = []

        for model in self.models:
            ep_full = ep_df[(ep_df["model_name"] == model) & (ep_df["subset"] == "FULL")]
            ep_hf = ep_df[(ep_df["model_name"] == model) & (ep_df["subset"] == "HF")]
            if ep_full.empty:
                continue
            f = ep_full.iloc[0]
            h = ep_hf.iloc[0] if not ep_hf.empty else None

            # Primary: SD of mean tier
            de_full = f["sd_persona_mean_tier"]
            de_hf = (
                h["sd_persona_mean_tier"] if h is not None else None
            )

            # Secondary: SD of P(T=2)
            dp_full = f["sd_persona_p_tier2"]
            dp_hf = (
                h["sd_persona_p_tier2"] if h is not None else None
            )

            # Omnibus p (ordered): Friedman test from repeated_measures
            rm = self.ar.get("repeated_measures", {}).get(model, {})
            np_res = rm.get("nonparametric", {})
            fr = np_res.get("friedman_tier", {})
            p_ord = fr.get("p_value") if fr and not fr.get("skipped") else None

            # Omnibus p (Tier 2): GEE Wald from omnibus_tests
            om_t2 = om_df[
                (om_df["model_name"] == model)
                & (om_df["subset"] == "FULL")
                & (om_df["endpoint"] == "tier2")
            ]
            p_t2 = om_t2.iloc[0]["p_value"] if not om_t2.empty else None
            log_p_t2 = om_t2.iloc[0].get("log_p_value") if not om_t2.empty else None

            # HF direction label: compare FULL vs HF dispersion, qualified
            # by omnibus significance so noise-level models are not called
            # "amplified".  Rule is stated in table footnote c.
            # Band: stable = [0.9, 1.1]; amplified = >1.1; attenuated = <0.9.
            either_sig = (
                (p_ord is not None and p_ord < 0.05)
                or (p_t2 is not None and p_t2 < 0.05)
            )
            if h is not None and de_full > 0 and dp_full > 0:
                ratio_primary = de_hf / de_full if de_full else 1.0
                ratio_secondary = dp_hf / dp_full if dp_full else 1.0
                avg_ratio = (ratio_primary + ratio_secondary) / 2
                if either_sig:
                    if avg_ratio > 1.1:
                        hf_verdict_tex = "amplified"
                        hf_verdict_md = "amplified"
                    elif avg_ratio < 0.9:
                        hf_verdict_tex = "attenuated"
                        hf_verdict_md = "attenuated"
                    else:
                        hf_verdict_tex = "stable"
                        hf_verdict_md = "stable"
                else:
                    # Non-significant: signal is below noise floor
                    if avg_ratio > 1.1:
                        hf_verdict_tex = r"stable / slight$\uparrow$"
                        hf_verdict_md = "stable / slight\u2191"
                    elif avg_ratio < 0.9:
                        hf_verdict_tex = r"stable / slight$\downarrow$"
                        hf_verdict_md = "stable / slight\u2193"
                    else:
                        hf_verdict_tex = "stable"
                        hf_verdict_md = "stable"
            else:
                hf_verdict_tex = "---"
                hf_verdict_md = "---"

            model_tex = r"\texttt{" + _escape_latex(model) + "}"
            tex_data_rows.append((
                model_tex,
                f"{de_full:.4f}", f"{de_hf:.4f}" if de_hf is not None else "---",
                _fmt_pp(dp_full), _fmt_pp(dp_hf) if dp_hf is not None else "---",
                _fmt_pval_latex(p_ord),
                _fmt_pval_latex(p_t2, log_p=log_p_t2),
                hf_verdict_tex,
            ))
            md_rows.append((
                model,
                f"{de_full:.4f}", f"{de_hf:.4f}" if de_hf is not None else "---",
                _fmt_pp(dp_full), _fmt_pp(dp_hf) if dp_hf is not None else "---",
                _fmt_pval(p_ord), _fmt_pval(p_t2, log_p=log_p_t2),
                hf_verdict_md,
            ))

        if not tex_data_rows:
            return []

        # --- Custom LaTeX ---
        tex_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Primary and secondary endpoint dispersion summary"
            r" (matched-guise panel).",
            r"$\sigma\,E[T]$: SD of persona-level mean tier (tier units)"
            r" across 36 guises.",
            r"$\sigma\,P(T{=}2)$: SD of persona-level Tier-2 rate"
            r" (percentage points, pp).",
            r"FULL: all pairs; HF: high-fidelity pairs only.",
            r"Primary endpoint: $\sigma\,E[T]$."
            r" Secondary endpoint: $\sigma\,P(T{=}2)$.}",
            r"\label{tab:endpoint_summary}",
            r"\begin{tabular}{@{}l cc cc cc l@{}}",
            r"\toprule",
            r"& \multicolumn{2}{c}{$\sigma\,E[T]$}",
            r"& \multicolumn{2}{c}{$\sigma\,P(T{=}2)$ (pp)}"
            r" & & & \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
            r"Model & FULL & HF & FULL & HF"
            r" & $p_{\mathrm{ord}}$$^{a}$"
            r" & $p_{Y_2}$$^{b}$"
            r" & HF direction$^{c}$ \\",
            r"\midrule",
        ]
        for mtex, de_f, de_h, dp_f, dp_h, po, pt, hfv in tex_data_rows:
            tex_lines.append(
                f"{mtex} & {de_f} & {de_h} & {dp_f} & {dp_h}"
                f" & {po} & {pt} & {hfv} \\\\"
            )
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append("")
        tex_lines.append(r"\vspace{2pt}")
        tex_lines.append(r"\parbox{0.95\linewidth}{\footnotesize")
        tex_lines.append(
            r"$^{a}$~$p_{\mathrm{ord}}$: Friedman test"
            r" (within-complaint blocking; ordinal $T$;"
            r" 36 persona treatments)."
            r" (Supportive / descriptive.)\\"
        )
        tex_lines.append(
            r"$^{b}$~$p_{Y_2}$: GEE Wald test on"
            r" $Y_2 = \mathbf{1}[T{=}2]$ with exchangeable"
            r" working correlation."
            r" (Confirmatory omnibus for Tier-2 invariance.)\\"
        )
        tex_lines.append(
            r"$^{c}$~\textit{HF direction} (heuristic descriptor):"
            r" Let $r_E = \sigma_E^{\mathrm{HF}} / \sigma_E^{\mathrm{FULL}}$"
            r" and $r_P = \sigma_P^{\mathrm{HF}} / \sigma_P^{\mathrm{FULL}}$,"
            r" where $\sigma_E$ is the SD of persona-level mean tier and"
            r" $\sigma_P$ the SD of persona-level Tier-2 rate."
            r" Define $\bar{r} = \tfrac{1}{2}(r_E + r_P)$."
            r" \emph{Stable}: $0.9 \le \bar{r} \le 1.1$;"
            r" \emph{amplified}: $\bar{r} > 1.1$;"
            r" \emph{attenuated}: $\bar{r} < 0.9$."
            r" Significance gate: when neither omnibus test is significant"
            r" ($p_{\mathrm{ord}} \ge 0.05$ and $p_{Y_2} \ge 0.05$),"
            r" the label is \emph{stable} with a directional qualifier"
            r" (slight$\uparrow$ or slight$\downarrow$)"
            r" rather than \emph{amplified}/\emph{attenuated}."
            r" Thresholds are descriptive; this is not a formal test."
        )
        tex_lines.append(r"}")
        tex_lines.append(r"\end{table}")

        tex_name = _build_filename(
            "table", "main", "endpoint_summary", "both", "matched", 1, "tex"
        )
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="endpoint_dispersion", model="both", endpoint="delta_ET_and_PT2", units="pp"))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV ---
        df = pd.DataFrame(md_rows, columns=[
            "Model", "SD E[T] FULL", "SD E[T] HF",
            "SD P(T=2) FULL (pp)", "SD P(T=2) HF (pp)",
            "p_ord (Friedman on T)", "p_Y2 (GEE Wald on Y2)",
            "HF direction",
        ])
        md_name = _build_filename(
            "table", "main", "endpoint_summary", "both", "matched", 1, "md"
        )
        md_path = out / md_name
        md_path.write_text(
            _dataframe_to_markdown(
                df, "Primary and secondary endpoint dispersion summary (matched-guise panel)."
            ),
            encoding="utf-8",
        )

        csv_name = _build_filename(
            "table", "main", "endpoint_summary", "both", "matched", 1, "csv"
        )
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: endpoint_summary (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

    def _table13_permutation_test(self, out: Path) -> List[Path]:
        """Table 13: Within-complaint permutation test for σ_E invariance."""
        # Read bootstrap CIs from endpoint_summary
        rob = self.ar.get("robustness_subsets", {})
        rob_dir = Path(rob.get("output_dir", "")) if rob and not rob.get("skipped") else None
        ep_df = None
        if rob_dir:
            ep_path = rob_dir / "endpoint_summary.csv"
            if ep_path.exists():
                ep_df = pd.read_csv(ep_path)

        tex_data_rows = []
        md_rows = []

        for model in self.models:
            perm = self.ar.get(model, {}).get("permutation_sigma_e", {})
            sigma_obs = perm.get("sigma_obs")
            perm_p = perm.get("perm_p")
            if sigma_obs is None or (isinstance(sigma_obs, float) and np.isnan(sigma_obs)):
                continue

            # Bootstrap CI from endpoint_summary (FULL subset)
            ci_str_tex = "---"
            ci_str_md = "---"
            if ep_df is not None and "ci_lower_sd_persona_mean_tier" in ep_df.columns:
                row = ep_df[
                    (ep_df["model_name"] == model) & (ep_df["subset"] == "FULL")
                ]
                if not row.empty:
                    lo = row.iloc[0]["ci_lower_sd_persona_mean_tier"]
                    hi = row.iloc[0]["ci_upper_sd_persona_mean_tier"]
                    ci_str_tex = f"[{lo:.4f},\\;{hi:.4f}]"
                    ci_str_md = f"[{lo:.4f}, {hi:.4f}]"

            # Conclusion
            if perm_p < 0.001:
                conclusion_tex = r"Reject $H_0$"
                conclusion_md = "Reject H0"
            elif perm_p < 0.05:
                conclusion_tex = r"Reject $H_0$"
                conclusion_md = "Reject H0"
            else:
                conclusion_tex = r"Fail to reject"
                conclusion_md = "Fail to reject"

            model_tex = r"\texttt{" + _escape_latex(model) + "}"
            tex_data_rows.append((
                model_tex,
                f"{sigma_obs:.4f}",
                ci_str_tex,
                _fmt_pval_latex(perm_p),
                conclusion_tex,
            ))
            md_rows.append((
                model,
                f"{sigma_obs:.4f}",
                ci_str_md,
                _fmt_pval(perm_p),
                conclusion_md,
            ))

        if not tex_data_rows:
            return []

        B = self.ar.get(self.models[0], {}).get("permutation_sigma_e", {}).get("B", 10_000)
        B_str = f"{B:,}"

        tex_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Within-complaint randomisation test for"
            r" ordered-endpoint invariance"
            rf" ($B = {B_str}$ permutations).",
            r"The test statistic is $\sigma_E$ (SD of persona-level mean tier).",
            r"Cluster-bootstrap 95\% CI from Table~\ref{tab:endpoint_summary}"
            r" shown for comparison.}",
            r"\label{tab:permutation_sigma_e}",
            r"\begin{tabular}{@{}l c c c l@{}}",
            r"\toprule",
            r"Model & $\sigma_E^{\mathrm{obs}}$"
            r" & Bootstrap 95\% CI"
            r" & Permutation $p$"
            r" & Conclusion \\",
            r"\midrule",
        ]
        for mtex, sigma, ci, p, conc in tex_data_rows:
            tex_lines.append(
                f"{mtex} & {sigma} & ${ci}$ & {p} & {conc} \\\\"
            )
        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\vspace{2pt}",
            r"\parbox{0.95\linewidth}{\footnotesize",
            r"\textit{Note.}~Under $H_0$ persona labels are exchangeable"
            r" within each complaint; the tier vector is permuted"
            r" independently per complaint.",
            r"$p$-value: proportion of permuted $\sigma_E \ge"
            r" \sigma_E^{\mathrm{obs}}$ (conservative: $(k+1)/(B+1)$).}",
            r"\end{table}",
        ])

        tex_name = _build_filename(
            "table", "main", "permutation_sigma_e", "both", "matched", 1, "tex"
        )
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(
            source="permutation_test", model="both",
            endpoint="sigma_E", units="tier",
        ))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV ---
        df = pd.DataFrame(md_rows, columns=[
            "Model", "sigma_E_obs", "Bootstrap 95% CI",
            "Permutation p", "Conclusion",
        ])
        md_name = _build_filename(
            "table", "main", "permutation_sigma_e", "both", "matched", 1, "md"
        )
        md_path = out / md_name
        md_path.write_text(
            _dataframe_to_markdown(
                df, "Within-complaint randomisation test for ordered-endpoint invariance."
            ),
            encoding="utf-8",
        )

        csv_name = _build_filename(
            "table", "main", "permutation_sigma_e", "both", "matched", 1, "csv"
        )
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: permutation_sigma_e (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

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
            source_meta={"source": "persona_tier_counts", "model": model, "endpoint": "tier_distribution", "units": "proportion"},
        )

    def _tableA2_pairwise_comparisons(self, out: Path, model: str) -> List[Path]:
        """Appendix Table A2: Pairwise persona comparisons (top-K)."""
        model_r = self.ar.get(model, {})
        causal = model_r.get("causal", {})
        pairwise = causal.get("pairwise_mcnemar", [])

        if not pairwise:
            return []

        # Sort by absolute effect and take top 20
        sorted_pairs = sorted(pairwise, key=lambda p: abs(p.get("rate_1", 0) - p.get("rate_2", 0)), reverse=True)[:20]

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
            source_meta={"source": "pairwise_mcnemar", "model": model, "endpoint": "tier2", "units": "risk_diff"},
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
            source_meta={"source": "missingness_diagnostic", "model": "na", "endpoint": "M1_vs_M0", "units": "mixed"},
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

        tex_lines.insert(0, _build_source_comment(source="missingness_ipw", model="all", endpoint="smd_balance", units="smd"))
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

        rows = []
        for task, version in PROMPT_VERSIONS.items():
            rows.append((task, version, f"v{version}"))

        # Add model parameters
        for model in self.models:
            is_reasoning = model.startswith(("gpt-5", "o1", "o3", "o4"))
            temp_note = ("temperature=N/A (reasoning model)"
                         if is_reasoning else "temperature=0.0")
            rows.append((f"Model: {model}", "---", temp_note))

        df = pd.DataFrame(rows, columns=["Task/Parameter", "Version", "Notes"])
        return _write_three_formats(
            df, out, "table", "appendix", "prompt_inventory",
            "na", "all", 1,
            caption="Prompt inventory and model parameters.",
            label="tab:prompts",
            source_meta={"source": "static", "model": "na", "endpoint": "na"},
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
        tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        tex_name = _build_filename("table", "appendix", "text_feature_inventory", "na", "all", 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="static", model="na", endpoint="text_features", units="mixed"))
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
            source_meta={"source": "persona_estimation", "model": model, "endpoint": "distribution", "units": "entropy_and_proportion"},
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

        # Repeated-measures model diagnostics table
        paths.extend(self._tableA8_rm_model_diagnostics(rob_dir))

        # Bayesian robustness tables
        bayes_dir = self.output_root / "analysis" / "tables" / "robustness" / "bayes"
        bayes_dir.mkdir(parents=True, exist_ok=True)
        paths.extend(self._tableB1_bayes_model_spec(bayes_dir))
        for model in self.models:
            for metric in ["monetary_severity", "issue_severity"]:
                paths.extend(self._tableB2_bayes_amplification(bayes_dir, model, metric))
                paths.extend(self._tableB3_bayes_ordinal_tiers(bayes_dir, model, metric))
            paths.extend(self._tableB4_bayes_diagnostics(bayes_dir, model))

        # Slot-preservation audit tables
        paths.extend(self._tableS1_slot_audit_summary(rob_dir))
        paths.extend(self._tableS2_slot_family_breakdown(rob_dir))
        paths.extend(self._tableS3_slot_model_comparison(rob_dir))

        # Slot-preservation robustness subset tables
        paths.extend(self._tableR1_subset_retention(rob_dir))
        paths.extend(self._tableR2_effect_stability(rob_dir))
        paths.extend(self._tableR3_outcome_composition_shift(rob_dir))

        return paths

    # ------------------------------------------------------------------
    # Human-readable labels for repeated-measures analysis keys
    # ------------------------------------------------------------------
    _RM_ANALYSIS_LABELS: Dict[str, str] = {
        "ordinal_model": "Ordinal Mixed (CLMM)",
        "conditional_logistic_favourable": "Conditional Logit (Favourable)",
        "logistic_gee_favourable": "GEE Logistic (Favourable)",
        "logistic_gee_info_request": "GEE Logistic (Info Request)",
        "linear_mixed_model": "Linear Mixed Model (Tier)",
    }

    def _tableA8_rm_model_diagnostics(self, out: Path) -> List[Path]:
        """Appendix Table A8: Repeated-measures model pre-flight checks and fallback actions."""
        rm = self.ar.get("repeated_measures", {})
        if not rm or rm.get("skipped"):
            return []

        _ANALYSIS_KEYS = [
            "ordinal_model", "conditional_logistic_favourable",
            "logistic_gee_favourable", "logistic_gee_info_request",
            "linear_mixed_model",
        ]

        rows = []
        for model_key in sorted(rm.keys()):
            model_data = rm[model_key]
            if not isinstance(model_data, dict):
                continue
            model_clean = model_key.replace("-", " ").replace("_", " ")

            for analysis_key in _ANALYSIS_KEYS:
                result = model_data.get(analysis_key, {})
                if not isinstance(result, dict):
                    continue

                diag = result.get("diagnostics", {})
                status = diag.get("status", "N/A")
                preflight = diag.get("preflight_issues", [])
                fallbacks = diag.get("fallback_actions", [])
                orig_preds = diag.get("original_predictors", [])
                fitted_preds = diag.get("fitted_predictors", [])

                analysis_label = self._RM_ANALYSIS_LABELS.get(analysis_key, analysis_key)

                # Summarise preflight issues
                if preflight:
                    preflight_summary = "; ".join(
                        p.get("type", "unknown") if isinstance(p, dict) else str(p)
                        for p in preflight
                    )
                else:
                    preflight_summary = "None"

                # Summarise fallback actions
                if fallbacks:
                    fallback_summary = "; ".join(str(f) for f in fallbacks)
                else:
                    fallback_summary = "None"

                # Predictor changes
                if orig_preds and fitted_preds and set(orig_preds) != set(fitted_preds):
                    dropped = sorted(set(orig_preds) - set(fitted_preds))
                    predictor_note = f"Dropped: {', '.join(dropped)}"
                else:
                    predictor_note = "---"

                rows.append((
                    model_clean,
                    analysis_label,
                    status,
                    preflight_summary,
                    fallback_summary,
                    predictor_note,
                ))

        if not rows:
            return []

        df = pd.DataFrame(rows, columns=[
            "LLM", "Analysis", "Status",
            "Pre-flight Issues", "Fallback Actions", "Predictor Changes",
        ])

        # Custom LaTeX with ragged columns for wide table
        tex_lines = [
            r"% Required packages: booktabs, array, ragged2e, tabularx",
            r"\begingroup\small",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Repeated-measures model pre-flight diagnostics and fallback actions.}",
            r"\label{tab:rm_model_diagnostics}",
            (r"\begin{tabularx}{\textwidth}{"
             r">{\RaggedRight\arraybackslash}p{0.10\linewidth}"
             r">{\RaggedRight\arraybackslash}p{0.14\linewidth}"
             r"c"
             r">{\RaggedRight\arraybackslash}X"
             r">{\RaggedRight\arraybackslash}X"
             r">{\RaggedRight\arraybackslash}p{0.14\linewidth}"
             r"}"),
            r"\toprule",
            (r"LLM & Analysis & Status & Pre-flight Issues "
             r"& Fallback Actions & Predictor Changes \\"),
            r"\midrule",
        ]

        prev_model = ""
        for _, row in df.iterrows():
            m = _escape_latex(str(row["LLM"])) if str(row["LLM"]) != prev_model else ""
            prev_model = str(row["LLM"])
            analysis = _escape_latex(str(row["Analysis"]))
            status = _escape_latex(str(row["Status"]))
            preflight_esc = _escape_latex(str(row["Pre-flight Issues"]))
            fallback_esc = _escape_latex(str(row["Fallback Actions"]))
            pred_esc = _escape_latex(str(row["Predictor Changes"]))
            tex_lines.append(
                f"{m} & {analysis} & {status} & "
                f"{preflight_esc} & {fallback_esc} & {pred_esc} \\\\"
            )

        tex_lines.extend([
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            r"\endgroup",
        ])

        # Write .tex manually (custom layout)
        tex_lines.insert(0, _build_source_comment(source="repeated_measures_diagnostics", model="all", endpoint="rm_model_fit", units="mixed"))
        tex_name = _build_filename(
            "table", "robustness", "rm_model_diagnostics", "all", "all", 1, "tex",
        )
        tex_path = out / tex_name
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # Write .md and .csv via helpers
        md_name = _build_filename(
            "table", "robustness", "rm_model_diagnostics", "all", "all", 1, "md",
        )
        md_path = out / md_name
        md_path.write_text(
            _dataframe_to_markdown(
                df,
                "Repeated-measures model pre-flight diagnostics and fallback actions.",
            ),
            encoding="utf-8",
        )

        csv_name = _build_filename(
            "table", "robustness", "rm_model_diagnostics", "all", "all", 1, "csv",
        )
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: rm_model_diagnostics (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

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
            source_meta={"source": "robustness_summary", "model": model, "endpoint": "ipw_alt_disparity_alt_endpoint", "units": "mixed"},
        )

    # ==================================================================
    # BAYESIAN ROBUSTNESS TABLES
    # ==================================================================

    def _tableB1_bayes_model_spec(self, out: Path) -> List[Path]:
        """Table B1: Bayesian model specification and priors."""
        # Plain-text rows for md/csv
        rows = [
            (
                "B1 (logistic)",
                "Bernoulli-logit",
                "Y2 = 1[tier == 2]",
                "beta ~ N(0, 1.5); alpha ~ N(0, 3); sigma_u ~ Exp(1)",
                "Complaint (non-centred)",
                "Dimension dummies, severity dummies, and interactions",
            ),
            (
                "B2 (ordinal)",
                "Ordered-logistic",
                "tier in {0,1,2} -> y in {1,2,3}",
                "beta ~ N(0, 1.5); alpha ~ N(0, 3); c_k ~ N(0, 3); sigma_u ~ Exp(1)",
                "Complaint (non-centred)",
                "Same as B1",
            ),
        ]
        df = pd.DataFrame(rows, columns=[
            "Model", "Likelihood", "Endpoint", "Priors", "Random Effects", "Predictors",
        ])

        caption = "Bayesian model specification and priors."
        label = "tab:bayes_spec"

        # --- Custom LaTeX ---
        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + _escape_latex(caption) + "}",
            r"\label{" + label + "}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.12}",
            "",
            r"\begin{tabularx}{\linewidth}{@{}",
            r"  L{0.11\linewidth}",
            r"  L{0.14\linewidth}",
            r"  L{0.16\linewidth}",
            r"  Y",
            r"  L{0.15\linewidth}",
            r"  Y",
            r"@{}}",
            r"\toprule",
            r"Model & Likelihood & Endpoint & Priors & Random Effects & Predictors \\",
            r"\midrule",
            r"B1 (logistic) &",
            r"Bernoulli--logit &",
            r"$Y_2=\mathbf{1}[\mathrm{tier}=2]$ &",
            r"$\beta \sim \mathcal{N}(0,1.5)$; $\alpha \sim \mathcal{N}(0,3)$;"
            r" $\sigma_u \sim \operatorname{Exp}(1)$ &",
            r"Complaint (non-centred) &",
            r"Dimension dummies, severity dummies, and interactions \\",
            r"B2 (ordinal) &",
            r"Ordered--logistic &",
            r"$\text{tier}\in\{0,1,2\}\rightarrow y\in\{1,2,3\}$ &",
            r"$\beta \sim \mathcal{N}(0,1.5)$; $\alpha \sim \mathcal{N}(0,3)$;"
            r" $c_k \sim \mathcal{N}(0,3)$; $\sigma_u \sim \operatorname{Exp}(1)$ &",
            r"Complaint (non-centred) &",
            r"Same as B1 \\",
            r"\bottomrule",
            r"\end{tabularx}",
            r"\normalsize",
            r"\end{table}",
        ]

        tex_name = _build_filename("table", "robustness", "bayes_model_spec", "na", "all", 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="static", model="na", endpoint="model_specification", units="na"))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV via DataFrame ---
        md_name = _build_filename("table", "robustness", "bayes_model_spec", "na", "all", 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, caption), encoding="utf-8")

        csv_name = _build_filename("table", "robustness", "bayes_model_spec", "na", "all", 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: bayes_model_spec (.tex/.md/.csv)")
        return [tex_path, md_path, csv_path]

    def _tableB2_bayes_amplification(self, out: Path, model: str, metric: str) -> List[Path]:
        """Table B2: Bayesian amplification robustness (Tier2)."""
        bayes = self.ar.get("severity", {}).get("bayesian_robustness", {})
        model_bayes = bayes.get(model, {})
        if not model_bayes or model_bayes.get("skipped") or model_bayes.get("error"):
            return []

        metric_data = model_bayes.get(metric, {})
        if not metric_data or metric_data.get("skipped"):
            return []

        rows = []          # plain-text for md/csv
        tex_rows = []      # LaTeX-ready
        has_wide_subset = False
        for model_type, label in [("logistic", "B1"), ("ordinal", "B2")]:
            mt_data = metric_data.get(model_type, {})
            if not mt_data or mt_data.get("error"):
                continue
            status = mt_data.get("status", "FAIL")
            amp = mt_data.get("amplification", {})
            subset = metric_data.get("subset", "all")
            if len(subset) > 5:
                has_wide_subset = True
            # LaTeX subset: escape underscores and allow line-break
            subset_tex = subset.replace("_", r"\_\allowbreak ")
            endpoint_plain = "Y2 = 1[tier==2]" if model_type == "logistic" else "tier (ordinal)"
            endpoint_tex = (
                r"$Y_2=\mathbf{1}[\mathrm{tier}=2]$"
                if model_type == "logistic"
                else "tier (ordinal)"
            )

            ne_bayes_amp = NE.missing_data(context="Bayes amp table")
            for dim, dim_data in amp.items():
                common = (
                    _fmt_effect(dim_data.get("A_mean", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("A_CrI_low", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("A_CrI_high", ne_bayes_amp)),
                    _fmt_prop(dim_data.get("P_A_gt_0", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("D_low_mean", ne_bayes_amp)),
                    _fmt_effect(dim_data.get("D_high_mean", ne_bayes_amp)),
                    status,
                )
                rows.append((
                    f"{label} ({dim})", metric.replace("_", " "),
                    subset, endpoint_plain, *common,
                ))
                tex_rows.append((
                    f"{label} ({dim})", metric.replace("_", " "),
                    subset_tex, endpoint_tex, *common,
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
        caption = f"Bayesian amplification robustness -- {model}, {metric.replace('_', ' ')}."
        label = f"tab:bayes_amp_{model_clean}_{metric_clean}"

        # --- Custom LaTeX ---
        # Wider text columns when subset contains long values (e.g. "amount_present")
        if has_wide_subset:
            col_sev, col_sub, col_end, col_stat = "0.10", "0.10", "0.10", "0.08"
        else:
            col_sev, col_sub, col_end, col_stat = "0.08", "0.05", "0.09", "0.09"

        tex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{" + _escape_latex(caption) + "}",
            r"\label{" + label + "}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{2.5pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            "",
            r"\begin{tabular}{@{}",
            r"L{0.12\linewidth}",
            r"L{" + col_sev + r"\linewidth}",
            r"L{" + col_sub + r"\linewidth}",
            r"L{" + col_end + r"\linewidth}",
            r"R{0.06\linewidth}",
            r"R{0.06\linewidth}",
            r"R{0.06\linewidth}",
            r"R{0.06\linewidth}",
            r"R{0.06\linewidth}",
            r"R{0.06\linewidth}",
            r"L{" + col_stat + r"\linewidth}",
            r"@{}}",
            r"\toprule",
            r"Model (dim) &",
            r"Severity &",
            r"Subset &",
            r"Endpoint &",
            r"\shortstack[r]{A\\mean} &",
            r"\shortstack[r]{A CrI\\low} &",
            r"\shortstack[r]{A CrI\\high} &",
            r"\shortstack[r]{$P(A>0)$} &",
            r"\shortstack[r]{D low\\mean} &",
            r"\shortstack[r]{D high\\mean} &",
            r"Status \\",
            r"\midrule",
        ]
        for row_tex in tex_rows:
            tex_lines.append(" & ".join(str(c) for c in row_tex) + r" \\")
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append(r"\normalsize")
        tex_lines.append(r"\end{table}")

        topic = "bayes_amplification_tier2"
        tex_name = _build_filename("table", "robustness", topic, model_clean, metric_clean, 1, "tex")
        tex_path = out / tex_name
        tex_lines.insert(0, _build_source_comment(source="bayes_amplification", model=model, endpoint="tier2", units="log_odds", ci="95pct_CrI", metric=metric))
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        # --- Markdown + CSV via DataFrame ---
        md_name = _build_filename("table", "robustness", topic, model_clean, metric_clean, 1, "md")
        md_path = out / md_name
        md_path.write_text(_dataframe_to_markdown(df, caption), encoding="utf-8")

        csv_name = _build_filename("table", "robustness", topic, model_clean, metric_clean, 1, "csv")
        csv_path = out / csv_name
        df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info("Written table: %s (.tex/.md/.csv)", topic)
        return [tex_path, md_path, csv_path]

    def _tableB3_bayes_ordinal_tiers(self, out: Path, model: str, metric: str) -> List[Path]:
        """Table B3: Bayesian ordinal tier-shift summary.

        Always produces an artifact: real tier probabilities when
        available, or a diagnostic failure table when all models failed.
        """
        bayes = self.ar.get("severity", {}).get("bayesian_robustness", {})
        model_bayes = bayes.get(model, {})
        metric_data = model_bayes.get(metric, {}) if model_bayes else {}

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        caption_base = f"Bayesian ordinal tier distribution -- {model}, {metric.replace('_', ' ')}"
        label = f"tab:bayes_ordinal_{model_clean}_{metric_clean}"

        # --- Try to build a real tier-probability table ---
        ordinal = metric_data.get("ordinal", {}) if metric_data else {}
        tier_probs = ordinal.get("tier_probs", {})

        rows = []
        for sev_band in ["low", "medium", "high"]:
            band_data = tier_probs.get(sev_band, {})
            if not band_data:
                continue
            t0 = band_data.get("P_tier0", {})
            t1 = band_data.get("P_tier1", {})
            t2 = band_data.get("P_tier2", {})

            ne_cri = NE.missing_data(context="CrI")
            def _fmt_cri(d: Dict[str, Any]) -> str:
                m = d.get("mean", ne_cri)
                lo = d.get("ci_low", ne_cri)
                hi = d.get("ci_high", ne_cri)
                return f"{_fmt_prop(m)} [{_fmt_prop(lo)}, {_fmt_prop(hi)}]"

            rows.append((sev_band.capitalize(), _fmt_cri(t0), _fmt_cri(t1), _fmt_cri(t2)))

        if rows:
            method = ordinal.get("method", "MCMC")
            caption = f"{caption_base} (method: {method})."
            df = pd.DataFrame(rows, columns=[
                "Severity band", "P(tier=0) [95% CrI]",
                "P(tier=1) [95% CrI]", "P(tier=2) [95% CrI]",
            ])
            return _write_three_formats(
                df, out, "table", "robustness", "bayes_ordinal_tier_distribution",
                model_clean, metric_clean, 1,
                caption=caption, label=label,
                source_meta={"source": "bayes_ordinal", "model": model, "endpoint": "tier_probabilities", "units": "probability", "ci": "95pct_CrI", "metric": metric},
            )

        # --- Failure path: produce diagnostic artifact ---
        # Collect failure reasons from postmortem / metric_data
        failure_rows: List[Tuple[str, str]] = []
        pm = metric_data.get("postmortem", {})

        # -- Section 1: Overall status --
        ordinal_status = ordinal.get("status", "N/A") if ordinal else "N/A"
        fallback_step = metric_data.get("fallback_step_b2", metric_data.get("fallback_step", "N/A"))
        failure_rows.append(("B2 ordinal status", str(ordinal_status)))
        failure_rows.append(("Final fallback step", str(fallback_step)))

        # Skipped entirely?
        if not model_bayes or model_bayes.get("skipped"):
            failure_rows.append(("Reason", f"Model skipped: {model_bayes.get('reason', 'unknown')}"))
        elif not metric_data or metric_data.get("skipped"):
            failure_rows.append(("Reason", f"Metric skipped: {metric_data.get('reason', 'unknown')}"))
        elif metric_data.get("status") == "SKIP":
            failure_rows.append(("Reason", metric_data.get("reason", "unknown")))

        # -- Section 2: Data summary (from postmortem) --
        ds = pm.get("data_summary", {})
        if ds:
            failure_rows.append(("---", "--- Data summary ---"))
            failure_rows.append(("Observations (N)", f"{ds.get('N', '?')}"))
            failure_rows.append(("Parameters (K)", f"{ds.get('K', '?')}"))
            failure_rows.append(("Complaint groups (J)", f"{ds.get('J', '?')}"))
            epp = ds.get("events_per_param", 0)
            epp_str = f"{epp:.1f}" if isinstance(epp, (int, float)) else str(epp)
            epp_flag = " [LOW]" if isinstance(epp, (int, float)) and epp < 10 else ""
            failure_rows.append(("Events per parameter", f"{epp_str}{epp_flag}"))
            t2_rate = ds.get("tier2_rate", 0)
            failure_rows.append(("Tier-2 rate", f"{t2_rate:.1%}" if isinstance(t2_rate, (int, float)) else str(t2_rate)))
            pct_sing = ds.get("pct_singletons", 0)
            failure_rows.append(("Singleton complaints", f"{pct_sing:.0%}" if isinstance(pct_sing, (int, float)) else str(pct_sing)))
            failure_rows.append(("Random intercepts", "Yes" if ds.get("use_random_intercept") else "No (too many singletons)"))
            tier_dist = ds.get("tier_distribution", {})
            if tier_dist:
                failure_rows.append(("Tier distribution", ", ".join(f"tier {k}: {v}" for k, v in sorted(tier_dist.items()))))

        # -- Section 3: Design matrix diagnostics --
        dm = pm.get("design_matrix_diagnostics", {})
        if dm:
            failure_rows.append(("---", "--- Design matrix ---"))
            failure_rows.append(("Matrix rank", f"{dm.get('rank', '?')}/{dm.get('K', '?')}"))
            cond = dm.get("condition_number", 0)
            cond_str = f"{cond:.2e}" if isinstance(cond, (int, float)) else str(cond)
            cond_flag = " [ILL-CONDITIONED]" if dm.get("ill_conditioned") else ""
            failure_rows.append(("Condition number", f"{cond_str}{cond_flag}"))
            failure_rows.append(("Main-effect columns", f"{dm.get('n_main_effect_cols', '?')}"))
            failure_rows.append(("Interaction columns", f"{dm.get('n_interaction_cols', '?')}"))

        # -- Section 4: Cross-tab sparsity --
        sp = pm.get("cross_tab_sparsity", {})
        if sp and sp.get("n_sparse_cells", 0) > 0:
            failure_rows.append(("---", "--- Cell sparsity ---"))
            failure_rows.append(("Sparse cells (<5 obs)", f"{sp['n_sparse_cells']}/{sp.get('total_cells', '?')} ({sp.get('sparsity_ratio', 0):.0%})"))

        # -- Section 5: Root causes --
        root_causes = pm.get("root_causes", [])
        if root_causes:
            failure_rows.append(("---", "--- Root causes ---"))
            for i, cause in enumerate(root_causes):
                failure_rows.append((f"Root cause {i + 1}", cause))

        # -- Section 6: Fallback chain with per-step diagnostics --
        per_step = pm.get("per_step_diagnostics", [])
        if per_step:
            failure_rows.append(("---", "--- Fallback chain (B2 ordinal) ---"))
        for step_info in per_step:
            b2_status = step_info.get("b2_status", "N/A")
            detail = step_info.get("description", step_info.get("label", "?"))
            diag_parts = []
            if "b2_rhat" in step_info:
                diag_parts.append(f"rhat={step_info['b2_rhat']:.4f}")
            if "b2_ess" in step_info:
                diag_parts.append(f"ess={step_info['b2_ess']:.0f}")
            if "b2_div" in step_info:
                diag_parts.append(f"div={step_info['b2_div']}")
            diag_str = f" ({', '.join(diag_parts)})" if diag_parts else ""
            failure_rows.append((
                f"Step {step_info.get('step', '?')} [{step_info.get('label', '?')}]",
                f"B2={b2_status}{diag_str} -- {detail}",
            ))

        # -- Section 7: Recommendations --
        recs = pm.get("recommendations", [])
        if recs:
            failure_rows.append(("---", "--- Recommendations ---"))
            for i, rec in enumerate(recs):
                failure_rows.append((f"Recommendation {i + 1}", rec))

        if not failure_rows:
            failure_rows.append(("Status", "No Bayesian ordinal results available"))

        df = pd.DataFrame(failure_rows, columns=["Diagnostic", "Detail"])
        caption = (
            f"{caption_base}. "
            f"\\textbf{{All Bayesian ordinal tier models for {model} {metric.replace('_', ' ')} failed.}} "
            "This table documents the fallback chain and root causes."
        )
        return _write_three_formats(
            df, out, "table", "robustness", "bayes_ordinal_tier_distribution",
            model_clean, metric_clean, 1,
            caption=caption, label=label,
            source_meta={"source": "bayes_ordinal_failure_diagnostic", "model": model, "endpoint": "fallback_chain", "metric": metric},
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
            source_meta={"source": "bayes_mcmc_diagnostics", "model": model, "endpoint": "rhat_ess_divergences", "units": "mixed"},
        )

    # ==================================================================
    # SLOT-PRESERVATION AUDIT TABLES
    # ==================================================================

    def _tableS1_slot_audit_summary(self, out: Path) -> List[Path]:
        """Table S1: Slot-preservation audit global summary and viability gate."""
        sa = self.ar.get("slot_audit", {})
        if not sa:
            return []

        metrics = sa.get("metrics", {})
        g = metrics.get("global", {})
        if not g:
            return []

        hf = g.get("high_fidelity", {})
        vg = g.get("viability_gate", {})
        checks = vg.get("checks", {})

        rows = [
            ("Total pairs", _fmt_int(g.get("n_pairs", 0)), "---", "---"),
            ("Total complaints", _fmt_int(g.get("n_complaints", 0)), "---", "---"),
            (
                "Strict pair pass rate",
                _fmt_prop(g.get("strict_pair_pass_rate")),
                f"[{g['strict_pair_pass_ci'][0]:.4f}, {g['strict_pair_pass_ci'][1]:.4f}]"
                if "strict_pair_pass_ci" in g else "---",
                "---",
            ),
            (
                "High-fidelity pair rate",
                _fmt_prop(hf.get("pair_rate")),
                f"[{hf['pair_rate_ci'][0]:.4f}, {hf['pair_rate_ci'][1]:.4f}]"
                if "pair_rate_ci" in hf else "---",
                checks.get("high_fidelity_pair_rate", {}).get("threshold", "---"),
            ),
            (
                "High-fidelity complaint rate",
                _fmt_prop(hf.get("complaint_rate")),
                "---",
                checks.get("high_fidelity_complaint_rate", {}).get("threshold", "---"),
            ),
            (
                "Polarity contradiction rate",
                _fmt_prop(g.get("polarity_contradiction_rate")),
                "---",
                checks.get("polarity_contradiction_rate", {}).get("threshold", "---"),
            ),
            (
                "Numeric mutation rate",
                _fmt_prop(g.get("numeric_mutation_rate")),
                "---",
                checks.get("numeric_mutation_rate", {}).get("threshold", "---"),
            ),
        ]

        gate_label = "PASS" if vg.get("pass") else "FAIL"
        rows.append(("Viability gate", gate_label, "---", "---"))

        df = pd.DataFrame(rows, columns=["Metric", "Value", "95% CI (Wilson)", "Threshold"])
        return _write_three_formats(
            df, out, "table", "robustness", "slot_audit_summary",
            "all", "all", 1,
            caption=f"Matched-guise slot-preservation audit summary (viability gate: {gate_label}).",
            label="tab:slot_audit_summary",
            source_meta={"source": "slot_audit_global", "model": "all", "endpoint": "viability_gate", "units": "proportion", "ci": "bootstrap_95"},
        )

    def _tableS2_slot_family_breakdown(self, out: Path) -> List[Path]:
        """Table S2: Per-family slot preservation rates."""
        sa = self.ar.get("slot_audit", {})
        if not sa:
            return []

        metrics = sa.get("metrics", {})
        per_family = metrics.get("per_family", {})
        if not per_family:
            return []

        rows = []
        for fam in ["numeric", "temporal", "polarity", "anchor"]:
            fm = per_family.get(fam, {})
            if not fm:
                continue

            pres_rate = fm.get("presence_preservation_rate")
            pres_ci = fm.get("presence_preservation_ci")
            pres_str = _fmt_prop(pres_rate) if pres_rate is not None else "---"
            ci_str = (f"[{pres_ci[0]:.4f}, {pres_ci[1]:.4f}]"
                      if pres_ci else "---")

            row = {
                "Family": fam.title(),
                "Pres. Rate": pres_str,
                "Pres. 95% CI (Wilson)": ci_str,
                "Omission": _fmt_prop(fm.get("omission_rate")),
                "Addition": _fmt_prop(fm.get("addition_rate")),
                "Mutation": _fmt_prop(fm.get("mutation_rate")),
            }
            if fam == "polarity":
                row["Contradiction"] = _fmt_prop(fm.get("contradiction_rate"))
            else:
                row["Contradiction"] = "---"
            rows.append(row)

        df = pd.DataFrame(rows)
        return _write_three_formats(
            df, out, "table", "robustness", "slot_family_breakdown",
            "all", "all", 1,
            caption="Slot-preservation rates by family (presence preservation = source slot present and correctly reproduced).",
            label="tab:slot_family_breakdown",
            source_meta={"source": "slot_audit_per_family", "model": "all", "endpoint": "preservation_rates", "units": "proportion", "ci": "bootstrap_95"},
        )

    def _tableS3_slot_model_comparison(self, out: Path) -> List[Path]:
        """Table S3: Per-model slot audit comparison."""
        sa = self.ar.get("slot_audit", {})
        if not sa:
            return []

        metrics = sa.get("metrics", {})
        subgroups = metrics.get("subgroups", {})
        model_data = subgroups.get("model_name", {})
        if not model_data:
            return []

        rows = []
        for model_name, md in sorted(model_data.items()):
            rows.append({
                "Model": model_name,
                "Pairs": _fmt_int(md.get("n_pairs", 0)),
                "Strict Pass": _fmt_prop(md.get("strict_pair_pass_rate")),
                "Hi-Fi Rate": _fmt_prop(md.get("high_fidelity_pair_rate")),
                "Contradiction": _fmt_prop(md.get("contradiction_rate")),
            })

        df = pd.DataFrame(rows)
        return _write_three_formats(
            df, out, "table", "robustness", "slot_model_comparison",
            "all", "all", 1,
            caption="Slot-preservation audit results by model.",
            label="tab:slot_model_comparison",
            source_meta={"source": "slot_audit_subgroups", "model": "all", "endpoint": "model_comparison", "units": "proportion"},
        )

    # ==================================================================
    # SLOT-PRESERVATION ROBUSTNESS SUBSET TABLES
    # ==================================================================

    def _tableR1_subset_retention(self, out: Path) -> List[Path]:
        """Table R1: Subset sizes and retention rates (cross-model)."""
        rob = self.ar.get("robustness_subsets", {})
        if not rob or rob.get("skipped"):
            return []

        rob_dir = Path(rob.get("output_dir", ""))
        manifest_path = rob_dir / "subset_manifest.csv"
        if not manifest_path.exists():
            return []

        df = pd.read_csv(manifest_path)
        aliases = rob.get("aliases", {})

        # Add alias annotation
        df["Subset"] = df["subset"].apply(
            lambda s: f"{s} (= {aliases[s]})" if s in aliases else s
        )
        display = df[["Subset", "model_name", "n_pairs", "n_complaints",
                       "pair_retention", "complaint_retention"]].copy()
        display.columns = ["Subset", "Model", "Pairs", "Complaints",
                           "Pair Ret.", "Compl. Ret."]
        display["Pair Ret."] = display["Pair Ret."].apply(lambda v: f"{v:.3f}")
        display["Compl. Ret."] = display["Compl. Ret."].apply(lambda v: f"{v:.3f}")
        display["Pairs"] = display["Pairs"].apply(lambda v: f"{int(v):,}")
        display["Complaints"] = display["Complaints"].apply(lambda v: f"{int(v):,}")

        return _write_three_formats(
            display, out, "table", "robustness", "slot_subset_retention",
            "all", "all", 1,
            caption="Matched-guise robustness subset retention rates.",
            label="tab:slot_subset_retention",
            font_size="small",
            source_meta={"source": "subset_manifest", "model": "all", "endpoint": "retention_rate", "units": "proportion"},
        )

    def _tableR2_effect_stability(self, out: Path) -> List[Path]:
        """Table R2: Effect stability vs FULL (cross-model)."""
        rob = self.ar.get("robustness_subsets", {})
        if not rob or rob.get("skipped"):
            return []

        rob_dir = Path(rob.get("output_dir", ""))
        comp_path = rob_dir / "robustness_vs_full.csv"
        if not comp_path.exists():
            return []

        df = pd.read_csv(comp_path)
        aliases = rob.get("aliases", {})

        display_rows = []
        for _, r in df.iterrows():
            subset = r["subset"]
            alias_note = f" (= {aliases[subset]})" if subset in aliases else ""
            display_rows.append({
                "Subset": f"{subset}{alias_note}",
                "Model": r["model_name"],
                "Effect": r["effect"].replace("max_persona_gap_", ""),
                "Full": f"{r['effect_full']:.4f}",
                "Subset Val.": f"{r['effect_subset']:.4f}",
                "Atten.": f"{r['attenuation']:.3f}" if not np.isnan(r.get("attenuation", float("nan"))) else "---",
                "Direction": r.get("effect_change_class", ""),
                "CI Overlap": "Yes" if r.get("ci_overlap", True) else "No",
                "Pass": "Y" if r.get("pass_attenuation", True) and r.get("pass_ci_overlap", True) else "N",
            })

        display = pd.DataFrame(display_rows)
        return _write_three_formats(
            display, out, "table", "robustness", "slot_effect_stability",
            "all", "all", 1,
            caption="Matched-guise effect stability across slot-quality subsets.",
            label="tab:slot_effect_stability",
            font_size="footnotesize",
            source_meta={"source": "robustness_vs_full", "model": "all", "endpoint": "effect_attenuation", "units": "proportion"},
        )

    def _tableR3_outcome_composition_shift(self, out: Path) -> List[Path]:
        """Table R3: Outcome-composition shift between kept and dropped pairs."""
        rob = self.ar.get("robustness_subsets", {})
        if not rob or rob.get("skipped"):
            return []

        rob_dir = Path(rob.get("output_dir", ""))
        shift_path = rob_dir / "outcome_composition_shift.csv"
        if not shift_path.exists():
            return []

        df = pd.read_csv(shift_path)
        display_rows = []
        for _, r in df.iterrows():
            tp = r.get("tier_ttest_p", float("nan"))
            zp = r.get("tier2_ztest_p", float("nan"))
            log_tp = r.get("log_tier_ttest_p", None)
            log_zp = r.get("log_tier2_ztest_p", None)
            # Use _fmt_pval (not _fmt_pval_latex) because this table goes
            # through _write_three_formats → _escape_latex; raw LaTeX math
            # would be double-escaped.
            display_rows.append({
                "Model": r["model_name"],
                "N Kept": f"{int(r['n_kept']):,}",
                "N Dropped": f"{int(r['n_dropped']):,}",
                "Mean Tier (K)": f"{r['mean_tier_kept']:.3f}",
                "Mean Tier (D)": f"{r['mean_tier_dropped']:.3f}" if not np.isnan(r.get("mean_tier_dropped", float("nan"))) else "---",
                "Diff": f"{r['tier_diff']:.3f}",
                "p (tier)": _fmt_pval(tp, log_p=log_tp) if not np.isnan(tp) else "---",
                "Tier2 (K)": f"{r['p_tier2_kept']:.3f}",
                "Tier2 (D)": f"{r['p_tier2_dropped']:.3f}" if not np.isnan(r.get("p_tier2_dropped", float("nan"))) else "---",
                "p (tier2)": _fmt_pval(zp, log_p=log_zp) if not np.isnan(zp) else "---",
                "Bias Dir.": r.get("bias_direction", ""),
            })

        display = pd.DataFrame(display_rows)
        return _write_three_formats(
            display, out, "table", "robustness", "slot_composition_shift",
            "all", "all", 1,
            caption="Outcome-composition shift: kept vs dropped pairs under high-fidelity filtering.",
            label="tab:slot_composition_shift",
            font_size="footnotesize",
            source_meta={"source": "outcome_composition_shift", "model": "all", "endpoint": "kept_vs_dropped", "units": "proportion"},
        )

    # ==================================================================
    # DEBIASED EXTRACTION CONSISTENCY TABLES
    # ==================================================================

    def _generate_debiased_tables(self) -> List[Path]:
        """Generate debiased extraction consistency tables."""
        debiased = self.ar.get("debiased_consistency")
        if not debiased:
            return []

        paths: List[Path] = []
        main_dir = self.output_root / "analysis" / "tables" / "main"
        main_dir.mkdir(parents=True, exist_ok=True)
        app_dir = self.output_root / "analysis" / "tables" / "appendix"
        app_dir.mkdir(parents=True, exist_ok=True)

        # Table: Extraction Consistency (per model)
        for model in self.models:
            paths.extend(self._table_debiased_extraction_consistency(main_dir, model, debiased))

        # Table: Cross-Model Comparison
        if len(self.models) >= 2:
            paths.extend(self._table_debiased_cross_model(main_dir, debiased))

        # Table: Consistency by Dimension (per model)
        for model in self.models:
            paths.extend(self._table_debiased_consistency_by_dim(main_dir, model, debiased))

        # Table: Cross-Model by Dimension
        if len(self.models) >= 2:
            paths.extend(self._table_debiased_cross_model_by_dim(main_dir, debiased))

        # Table: Worked Example (appendix)
        paths.extend(self._table_debiased_worked_example(app_dir, debiased))

        return paths

    def _table_debiased_extraction_consistency(
        self, out: Path, model: str, debiased: Dict[str, Any],
    ) -> List[Path]:
        """Extraction consistency: original vs. 36 persona rewrites (per model)."""
        consistency = debiased.get("consistency", [])
        model_rows = [r for r in consistency if r["model"] == model]
        if not model_rows:
            return []

        rows = []
        for r in model_rows:
            is_avg = r["case_id"] == "AVERAGE"
            rows.append({
                "Case ID": r["case_id"] if not is_avg else "AVERAGE",
                "Orig": str(r["orig_tier"]) if r["orig_tier"] != "" else "",
                "Tier %": _fmt_pct(float(r["tier"]) * 100) if r["tier"] != "" else "",
                "Entity %": _fmt_pct(float(r["entity"]) * 100) if r["entity"] != "" else "",
                "Nexus %": _fmt_pct(float(r["nexus"]) * 100) if r["nexus"] != "" else "",
                "Conduct %": _fmt_pct(float(r["conduct"]) * 100) if r["conduct"] != "" else "",
                "Harm %": _fmt_pct(float(r["harm"]) * 100) if r["harm"] != "" else "",
                "Evid %": _fmt_pct(float(r["evid"]) * 100) if r["evid"] != "" else "",
                "N": str(r["n"]) if r["n"] != "" else "",
            })

        df = pd.DataFrame(rows)
        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "main", "debiased_extraction_consistency",
            model_clean, "matched", 1,
            caption=f"Debiased extraction consistency: original vs. 36 persona rewrites -- {model}.",
            label=f"tab:debiased_consistency_{model_clean}",
            use_tabularx=True,
            col_spec=r"l >{\raggedright\arraybackslash}X c r r r r r r r",
            font_size="footnotesize",
            source_meta={"source": "debiased_consistency", "model": model, "endpoint": "extraction_match_rate", "units": "proportion"},
        )

    def _table_debiased_cross_model(
        self, out: Path, debiased: Dict[str, Any],
    ) -> List[Path]:
        """Cross-model comparison on originals + rewrites."""
        cross_model = debiased.get("cross_model", [])
        if not cross_model:
            return []

        m1 = self.models[0] if self.models else ""
        m2 = self.models[1] if len(self.models) >= 2 else ""

        rows = []
        for r in cross_model:
            def _fmt_agree(v: Any) -> str:
                if isinstance(v, bool):
                    return "Y" if v else "N"
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return _fmt_pct(float(v) * 100)
                if v == "":
                    return ""
                return str(v)

            rows.append({
                "Section": r["section"],
                "Case ID": r["case_id"],
                f"T {m1}": str(r.get("tier_m1", "")),
                f"T {m2}": str(r.get("tier_m2", "")),
                "Tier": _fmt_agree(r["tier_agree"]),
                "Entity": _fmt_agree(r["entity_agree"]),
                "Nexus": _fmt_agree(r["nexus_agree"]),
                "Conduct": _fmt_agree(r["conduct_agree"]),
                "Harm": _fmt_agree(r["harm_agree"]),
                "Evid": _fmt_agree(r.get("evid_agree", "")),
            })

        df = pd.DataFrame(rows)
        return _write_three_formats(
            df, out, "table", "main", "debiased_cross_model",
            "na", "matched", 1,
            caption=f"Cross-model comparison: {_escape_latex(m1)} vs. {_escape_latex(m2)}.",
            label="tab:debiased_cross_model",
            use_tabularx=True,
            col_spec=r"l >{\raggedright\arraybackslash}X c c r r r r r r",
            font_size="footnotesize",
            source_meta={"source": "debiased_cross_model", "model": "all", "endpoint": "agreement_rate", "units": "proportion"},
        )

    def _table_debiased_consistency_by_dim(
        self, out: Path, model: str, debiased: Dict[str, Any],
    ) -> List[Path]:
        """Extraction consistency grouped by persona dimension (per model)."""
        by_dim = debiased.get("by_dimension", [])
        model_rows = [r for r in by_dim if r["model"] == model]
        if not model_rows:
            return []

        rows = []
        for r in model_rows:
            rows.append({
                "Dim": r["dimension"],
                "Value": r["value"],
                "Tier %": _fmt_pct(float(r["tier"]) * 100),
                "Entity %": _fmt_pct(float(r["entity"]) * 100),
                "Nexus %": _fmt_pct(float(r["nexus"]) * 100),
                "Conduct %": _fmt_pct(float(r["conduct"]) * 100),
                "Harm %": _fmt_pct(float(r["harm"]) * 100),
                "Evid %": _fmt_pct(float(r["evid"]) * 100),
                "N": _fmt_int(r["n"]),
            })

        df = pd.DataFrame(rows)
        model_clean = model.replace("-", "").replace("_", "")
        return _write_three_formats(
            df, out, "table", "main", "debiased_consistency_by_dim",
            model_clean, "matched", 1,
            caption=f"Extraction consistency by persona dimension -- {model}.",
            label=f"tab:debiased_dim_{model_clean}",
            use_tabularx=True,
            col_spec=r"l l r r r r r r r",
            font_size="footnotesize",
            source_meta={"source": "debiased_by_dimension", "model": model, "endpoint": "extraction_match_rate", "units": "proportion"},
        )

    def _table_debiased_cross_model_by_dim(
        self, out: Path, debiased: Dict[str, Any],
    ) -> List[Path]:
        """Cross-model agreement by persona dimension."""
        cross_dim = debiased.get("cross_model_by_dimension", [])
        if not cross_dim:
            return []

        rows = []
        for r in cross_dim:
            rows.append({
                "Dim": r["dimension"],
                "Value": r["value"],
                "Tier %": _fmt_pct(float(r["tier"]) * 100),
                "Entity %": _fmt_pct(float(r["entity"]) * 100),
                "Nexus %": _fmt_pct(float(r["nexus"]) * 100),
                "Conduct %": _fmt_pct(float(r["conduct"]) * 100),
                "Harm %": _fmt_pct(float(r["harm"]) * 100),
                "Evid %": _fmt_pct(float(r["evid"]) * 100),
                "N": _fmt_int(r["n"]),
            })

        df = pd.DataFrame(rows)
        return _write_three_formats(
            df, out, "table", "main", "debiased_cross_model_by_dim",
            "na", "matched", 1,
            caption="Cross-model agreement by persona dimension.",
            label="tab:debiased_cross_dim",
            use_tabularx=True,
            col_spec=r"l l r r r r r r r",
            font_size="footnotesize",
            source_meta={"source": "debiased_cross_model_by_dimension", "model": "all", "endpoint": "agreement_rate", "units": "proportion"},
        )

    def _table_debiased_worked_example(
        self, out: Path, debiased: Dict[str, Any],
    ) -> List[Path]:
        """Worked example: fact extraction across persona rewrites (appendix)."""
        worked = debiased.get("worked_example", [])
        if not worked or len(self.models) < 2:
            return []

        m1, m2 = self.models[0], self.models[1]

        # Build custom LaTeX (multi-row header with cmidrule)
        m1_h = _escape_latex(m1)
        m2_h = _escape_latex(m2)
        tex_lines = [
            r"% Required packages: tabularx, booktabs, array, multirow",
            r"% \newcolumntype{Y}{>{\raggedright\arraybackslash}X}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Worked Example: Fact Extraction Across Persona Rewrites}",
            r"\label{tab:debiased_worked_example}",
            r"\begin{tabularx}{\textwidth}{l Y c l c Y Y c l c Y Y}",
            r"\toprule",
            (
                r" & "
                rf"& \multicolumn{{5}}{{c}}{{\textbf{{{m1_h}}}}} "
                rf"& \multicolumn{{5}}{{c}}{{\textbf{{{m2_h}}}}} \\"
            ),
            r"\cmidrule(lr){3-7} \cmidrule(lr){8-12}",
            (
                r"\textbf{Variant} & \textbf{Text} "
                r"& \textbf{T} & \textbf{Ent} & \textbf{Nexus} "
                r"& \textbf{Conduct} & \textbf{Harm} "
                r"& \textbf{T} & \textbf{Ent} & \textbf{Nexus} "
                r"& \textbf{Conduct} & \textbf{Harm} \\"
            ),
            r"\midrule",
        ]

        for r in worked:
            full_text = str(r.get("text", ""))
            text_trunc = full_text[:100] + ("..." if len(full_text) > 100 else "")
            vals = r.get("vals", {})
            v1 = vals.get(m1, {})
            v2 = vals.get(m2, {})
            row = [
                _escape_latex(r["variant"]),
                _escape_latex(text_trunc),
                str(v1.get(f"{m1}_tier", "")),
                _escape_latex(str(v1.get(f"{m1}_entity", ""))),
                str(v1.get(f"{m1}_nexus", "")),
                _escape_latex(str(v1.get(f"{m1}_conduct", ""))),
                _escape_latex(str(v1.get(f"{m1}_harm", ""))),
                str(v2.get(f"{m2}_tier", "")),
                _escape_latex(str(v2.get(f"{m2}_entity", ""))),
                str(v2.get(f"{m2}_nexus", "")),
                _escape_latex(str(v2.get(f"{m2}_conduct", ""))),
                _escape_latex(str(v2.get(f"{m2}_harm", ""))),
            ]
            tex_lines.append(" & ".join(row) + r" \\")

        tex_lines += [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
        ]

        # Write .tex manually (custom multi-row header)
        tex_lines.insert(0, _build_source_comment(source="debiased_worked_example", model="all", endpoint="fact_extraction", units="mixed"))
        tex_name = _build_filename(
            "table", "appendix", "debiased_worked_example", "na", "matched", 1, "tex"
        )
        tex_path = out / tex_name
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
        paths = [tex_path]

        # Also write .md and .csv (flat format)
        flat_rows = []
        for r in worked:
            full_text = str(r.get("text", ""))
            text_trunc = full_text[:100] + ("..." if len(full_text) > 100 else "")
            vals = r.get("vals", {})
            v1 = vals.get(m1, {})
            v2 = vals.get(m2, {})
            flat_rows.append({
                "Variant": r["variant"],
                "Text": text_trunc,
                f"{m1} Tier": v1.get(f"{m1}_tier", ""),
                f"{m1} Entity": v1.get(f"{m1}_entity", ""),
                f"{m1} Nexus": v1.get(f"{m1}_nexus", ""),
                f"{m1} Conduct": v1.get(f"{m1}_conduct", ""),
                f"{m1} Harm": v1.get(f"{m1}_harm", ""),
                f"{m2} Tier": v2.get(f"{m2}_tier", ""),
                f"{m2} Entity": v2.get(f"{m2}_entity", ""),
                f"{m2} Nexus": v2.get(f"{m2}_nexus", ""),
                f"{m2} Conduct": v2.get(f"{m2}_conduct", ""),
                f"{m2} Harm": v2.get(f"{m2}_harm", ""),
            })

        if flat_rows:
            df = pd.DataFrame(flat_rows)
            md_name = _build_filename(
                "table", "appendix", "debiased_worked_example", "na", "matched", 1, "md"
            )
            md_path = out / md_name
            md_path.write_text(
                _dataframe_to_markdown(df, "Worked Example: Fact Extraction Across Persona Rewrites"),
                encoding="utf-8",
            )
            paths.append(md_path)

            csv_name = _build_filename(
                "table", "appendix", "debiased_worked_example", "na", "matched", 1, "csv"
            )
            csv_path = out / csv_name
            df.to_csv(csv_path, index=False, encoding="utf-8")
            paths.append(csv_path)

        logger.info("Written table: debiased_worked_example (.tex/.md/.csv)")
        return paths
