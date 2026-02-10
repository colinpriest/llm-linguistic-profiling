"""Paper bundle orchestrator.

Generates the complete paper-ready output bundle under
``artifacts/runs/{RUN_ID}/`` following the folder structure and
file naming conventions specified in the paper-bundle output spec.

Public API
----------
PaperBundleGenerator.generate  Generate the full paper bundle.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("$", "\\$"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("%", "\\%"),
        ("_", "\\_"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ------------------------------------------------------------------
# Folder structure definition
# ------------------------------------------------------------------

_REQUIRED_DIRS = [
    "manifest",
    "logs",
    "data/sampled",
    "data/derived",
    "prompts",
    "api",
    "analysis/reports",
    "analysis/tables/main",
    "analysis/tables/appendix",
    "analysis/tables/robustness",
    "analysis/tables/robustness/bayes",
    "analysis/figures/main",
    "analysis/figures/appendix",
    "analysis/figures/appendix/bayes",
    "analysis/stats",
    "analysis/stats/bayes",
    "analysis/panels",
    "qualitative",
    "paper_pack/latex/tables",
    "paper_pack/latex/figures",
    "paper_pack/latex/appendix_tables",
    "paper_pack/latex/appendix_bayes",
    "paper_pack/latex/appendix_worked_examples",
    "paper_pack/docx/tables",
    "paper_pack/docx/figures",
    "paper_pack/markdown/tables",
    "paper_pack/markdown/figures",
]


# ------------------------------------------------------------------
# Acceptance checklist items (spec section 8)
# ------------------------------------------------------------------

def _acceptance_checklist(output_root: Path, models: List[str]) -> List[str]:
    """Return list of missing required files (empty = pass)."""
    missing = []

    # Pre-reg compliance report
    reports = list((output_root / "analysis" / "reports").glob("report__main__prereg_compliance*"))
    if not reports:
        missing.append("Pre-reg compliance report")

    # Table 4 for each model
    for model in models:
        mc = model.replace("-", "").replace("_", "")
        t4 = list((output_root / "analysis" / "tables" / "main").glob(f"table__main__distributive_fairness_tier2__{mc}*"))
        if not t4:
            missing.append(f"Table 4 (distributive fairness) for {model}")

    # Table 6 for each model x metric
    for model in models:
        mc = model.replace("-", "").replace("_", "")
        for metric in ["monetaryseverity", "issueseverity"]:
            t6 = list((output_root / "analysis" / "tables" / "main").glob(
                f"table__main__severity_amplification_confirmatory__{mc}*{metric}*"))
            if not t6:
                missing.append(f"Table 6 (amplification) for {model}/{metric}")

    # Fig 7 (missingness)
    f7 = list((output_root / "analysis" / "figures" / "main").glob("fig__main__monetary_missingness*"))
    if not f7:
        missing.append("Figure 7 (missingness diagnostic)")

    # Fig 8 (amplification)
    f8 = list((output_root / "analysis" / "figures" / "main").glob("fig__main__amplification_interaction*"))
    if not f8:
        missing.append("Figure 8 (amplification interaction)")

    # BH correction stats
    bh = list((output_root / "analysis" / "stats").glob("bh_fdr*"))
    if not bh:
        missing.append("BH FDR correction output")

    # IPW robustness
    ipw = list((output_root / "analysis" / "stats").glob("ipw_robustness*"))
    if not ipw:
        missing.append("IPW robustness output")

    return missing


# ------------------------------------------------------------------
# PaperBundleGenerator
# ------------------------------------------------------------------

class PaperBundleGenerator:
    """Orchestrates generation of the complete paper bundle."""

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
        self.models = analysis_results.get("models", [])

        # Build RUN_ID
        self.run_id = self._build_run_id()
        self.rc["run_id"] = self.run_id

        # Output root: artifacts/runs/{RUN_ID}/
        self.output_root = Path(output_root) / self.run_id
        logger.info("Paper bundle output: %s", self.output_root)

    def _build_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.rc.get("seed", 0)
        n = self.rc.get("num_complaints", 0)
        model_str = "-".join(
            m.replace("-", "").replace("_", "") for m in self.models
        )
        return f"{ts}__seed{seed}__n{n}__models{model_str}"

    def generate(self) -> Dict[str, Any]:
        """Generate the full paper bundle. Returns summary dict."""
        logger.info("=" * 60)
        logger.info("Generating paper bundle: %s", self.run_id)

        # 1. Create folder structure
        self._create_folders()

        # 2. Reproducibility bundle
        self._generate_reproducibility()

        # 3. Save stats artifacts
        self._save_stats_artifacts()

        # 4. Tables
        self._generate_tables()

        # 5. Figures
        self._generate_figures()

        # 6. Qualitative cards
        self._generate_qualitative_cards()

        # 7. Reports (prereg compliance + summary)
        self._generate_reports()

        # 8. Paper pack
        self._populate_paper_pack()

        # 9. Index files
        self._generate_index_files()

        # 10. Final manifest (must be last — hashes all outputs)
        self._generate_final_manifest()

        # 11. Acceptance checklist
        missing = _acceptance_checklist(self.output_root, self.models)
        if missing:
            logger.warning("Acceptance checklist: %d items missing:", len(missing))
            for item in missing:
                logger.warning("  - %s", item)
        else:
            logger.info("Acceptance checklist: ALL PASS")

        logger.info("Paper bundle complete: %s", self.output_root)
        logger.info("=" * 60)

        return {
            "run_id": self.run_id,
            "output_root": str(self.output_root),
            "acceptance_missing": missing,
        }

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _create_folders(self) -> None:
        for d in _REQUIRED_DIRS:
            (self.output_root / d).mkdir(parents=True, exist_ok=True)
        logger.info("Created %d directories", len(_REQUIRED_DIRS))

    def _generate_reproducibility(self) -> None:
        from src.reproducibility import generate_reproducibility_bundle
        generate_reproducibility_bundle(
            run_config=self.rc,
            complaints=self.complaints,
            output_root=self.output_root,
        )

    def _save_stats_artifacts(self) -> None:
        """Save statistical outputs as JSON for reproducibility."""
        stats_dir = self.output_root / "analysis" / "stats"

        # BH FDR results
        severity = self.ar.get("severity", {})
        gee = severity.get("gee_interactions_tier2", {})
        if gee:
            bh_results = self._collect_bh_results(gee)
            (stats_dir / "bh_fdr__confirmatory__all__v1.json").write_text(
                json.dumps(bh_results, indent=2, default=str), encoding="utf-8",
            )

        # IPW robustness results
        ipw = severity.get("ipw_robustness", {})
        if ipw and not ipw.get("error"):
            (stats_dir / "ipw_robustness__monetary__all__v1.json").write_text(
                json.dumps(ipw, indent=2, default=str), encoding="utf-8",
            )

            # Generate panel payload for validation contract
            from src.ipw_robustness import ipw_to_panel_payload
            panels_dir = self.output_root / "analysis" / "panels"
            panels_dir.mkdir(parents=True, exist_ok=True)
            panel_payload = ipw_to_panel_payload(
                ipw_result=ipw,
                run_id=self.run_id,
                models=self.models,
            )
            (panels_dir / "panel12_ipw_robustness_monetary.json").write_text(
                json.dumps(panel_payload, indent=2, default=str), encoding="utf-8",
            )
            logger.info("IPW panel payload saved to analysis/panels/")

        # Claim classifications
        from src.prereg_compliance import generate_claim_classifications
        claims = generate_claim_classifications(self.ar)
        (stats_dir / "claim_classifications__all__v1.json").write_text(
            json.dumps(claims, indent=2), encoding="utf-8",
        )

        # Bayesian robustness results
        bayes = severity.get("bayesian_robustness", {})
        if bayes and not bayes.get("error"):
            bayes_dir = stats_dir / "bayes"
            bayes_dir.mkdir(parents=True, exist_ok=True)
            # Write summary JSON (strip large draw arrays for compact storage)
            bayes_summary = {}
            for model_key, model_data in bayes.items():
                if not isinstance(model_data, dict):
                    continue
                bayes_summary[model_key] = model_data
            (bayes_dir / "bayesian_robustness__all__v1.json").write_text(
                json.dumps(bayes_summary, indent=2, default=str), encoding="utf-8",
            )

        logger.info("Stats artifacts saved")

    def _collect_bh_results(self, gee: Dict) -> Dict[str, Any]:
        """Collect all confirmatory p-values and apply BH correction."""
        from src.statistics import fdr_correction
        import numpy as np

        tests = []
        for metric in ["monetary_severity", "issue_severity"]:
            for model in self.models:
                model_gee = gee.get(metric, {}).get(model, {})
                for dim in ["age", "geographic", "education", "socioeconomic"]:
                    dim_data = model_gee.get(dim, {})
                    if not dim_data.get("skipped", False):
                        p = dim_data.get("interaction_p", float("nan"))
                        if np.isfinite(p):
                            tests.append({
                                "test": f"{model}/{metric}/{dim}",
                                "p_value": p,
                                "is_confirmatory": True,
                            })

        if tests:
            p_values = [t["p_value"] for t in tests]
            adjusted = fdr_correction(p_values)
            for t, q in zip(tests, adjusted):
                t["q_value"] = float(q)
                t["significant"] = float(q) < 0.05
        else:
            adjusted = []

        return {
            "method": "Benjamini-Hochberg",
            "alpha": 0.05,
            "n_tests": len(tests),
            "n_significant": sum(1 for t in tests if t.get("significant", False)),
            "tests": tests,
        }

    def _generate_tables(self) -> None:
        from src.table_generator import TableGenerator
        tg = TableGenerator(
            analysis_results=self.ar,
            complaints=self.complaints,
            run_config=self.rc,
            output_root=self.output_root,
        )
        tg.generate_all_tables()

    def _generate_figures(self) -> None:
        from src.figure_generator import FigureGenerator
        fg = FigureGenerator(
            analysis_results=self.ar,
            complaints=self.complaints,
            run_config=self.rc,
            output_root=self.output_root,
        )
        fg.generate_all_figures()

    def _generate_qualitative_cards(self) -> None:
        """Generate worked example cards as Markdown and LaTeX."""
        qual_dir = self.output_root / "qualitative"
        if not self.complaints:
            return

        # Select up to 5 cases with different severity levels
        cases = []
        seen_severities = set()
        for c in self.complaints:
            key = (c.monetary_severity, c.issue_severity)
            if key not in seen_severities and len(cases) < 5:
                cases.append(c)
                seen_severities.add(key)

        for case in cases:
            card_id = case.case_id[:8]

            # Markdown version
            card_path = qual_dir / f"card__appendix__worked_example_{card_id}__both__all__v1.md"
            card_lines = [
                f"# Worked Example: {card_id}",
                "",
                f"**Product:** {case.product_category}",
                f"**Issue:** {case.issue_category}",
                f"**Monetary severity:** {case.monetary_severity}",
                f"**Issue severity:** {case.issue_severity}",
                "",
                "## Original Complaint",
                "",
                case.narrative[:500] + ("..." if len(case.narrative) > 500 else ""),
                "",
            ]
            card_path.write_text("\n".join(card_lines), encoding="utf-8")

            # LaTeX version
            tex_path = qual_dir / f"card__appendix__worked_example_{card_id}__both__all__v1.tex"
            narrative_escaped = _latex_escape(
                case.narrative[:500] + ("..." if len(case.narrative) > 500 else "")
            )
            tex_lines = [
                f"\\subsection{{Worked Example: {card_id}}}",
                f"\\label{{sec:worked-example-{card_id}}}",
                "",
                "\\begin{description}",
                f"  \\item[Product] {_latex_escape(case.product_category)}",
                f"  \\item[Issue] {_latex_escape(case.issue_category)}",
                f"  \\item[Monetary severity] {_latex_escape(case.monetary_severity)}",
                f"  \\item[Issue severity] {_latex_escape(case.issue_severity)}",
                "\\end{description}",
                "",
                "\\paragraph{Original Complaint}",
                "",
                "\\begin{quote}",
                f"\\small {narrative_escaped}",
                "\\end{quote}",
                "",
            ]
            tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

        logger.info("Generated %d qualitative cards (Markdown + LaTeX)", len(cases))

    def _generate_reports(self) -> None:
        """Generate pre-reg compliance and summary reports."""
        reports_dir = self.output_root / "analysis" / "reports"

        # Pre-reg compliance report
        from src.prereg_compliance import generate_prereg_compliance_report
        prereg_text = generate_prereg_compliance_report(self.ar, self.rc)
        prereg_path = reports_dir / "report__main__prereg_compliance__na__all__v1.md"
        prereg_path.write_text(prereg_text, encoding="utf-8")
        logger.info("Pre-reg compliance report written")

        # Summary report
        summary_lines = [
            "# Paper Summary Report",
            "",
            f"**Run ID:** {self.run_id}",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Models:** {', '.join(self.models)}",
            f"**N complaints:** {self.rc.get('num_complaints', 0)}",
            f"**Seed:** {self.rc.get('seed', 0)}",
            "",
            "## Analysis Outputs",
            "",
        ]

        # List generated tables
        tables_dir = self.output_root / "analysis" / "tables"
        tex_files = sorted(tables_dir.rglob("*.tex"))
        if tex_files:
            summary_lines.append("### Tables")
            for f in tex_files:
                summary_lines.append(f"- `{f.relative_to(self.output_root)}`")
            summary_lines.append("")

        # List generated figures
        figs_dir = self.output_root / "analysis" / "figures"
        pdf_files = sorted(figs_dir.rglob("*.pdf"))
        if pdf_files:
            summary_lines.append("### Figures")
            for f in pdf_files:
                summary_lines.append(f"- `{f.relative_to(self.output_root)}`")
            summary_lines.append("")

        summary_path = reports_dir / "report__main__paper_summary__na__all__v1.md"
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        # Index report
        index_lines = [
            "# Full Results Index",
            "",
            f"**Run ID:** {self.run_id}",
            "",
        ]
        for f in sorted(self.output_root.rglob("*")):
            if f.is_file():
                index_lines.append(f"- `{f.relative_to(self.output_root)}`")
        index_path = reports_dir / "report__appendix__full_results_index__na__all__v1.md"
        index_path.write_text("\n".join(index_lines), encoding="utf-8")

        logger.info("Reports written")

    def _populate_paper_pack(self) -> None:
        """Copy main tables and figures to paper_pack/ for easy access."""
        # LaTeX tables
        src_tables = self.output_root / "analysis" / "tables" / "main"
        dst_tables = self.output_root / "paper_pack" / "latex" / "tables"
        for f in src_tables.glob("*.tex"):
            shutil.copy2(f, dst_tables / f.name)

        # LaTeX figures
        src_figs = self.output_root / "analysis" / "figures" / "main"
        dst_figs = self.output_root / "paper_pack" / "latex" / "figures"
        for f in src_figs.glob("*.pdf"):
            shutil.copy2(f, dst_figs / f.name)

        # Markdown tables
        dst_md_tables = self.output_root / "paper_pack" / "markdown" / "tables"
        for f in src_tables.glob("*.md"):
            shutil.copy2(f, dst_md_tables / f.name)

        # Markdown figures (PNG)
        dst_md_figs = self.output_root / "paper_pack" / "markdown" / "figures"
        for f in src_figs.glob("*.png"):
            shutil.copy2(f, dst_md_figs / f.name)

        # Appendix tables → dedicated appendix_tables/ folder
        dst_app_tables = self.output_root / "paper_pack" / "latex" / "appendix_tables"
        src_app_tables = self.output_root / "analysis" / "tables" / "appendix"
        if src_app_tables.exists():
            for f in src_app_tables.glob("*.tex"):
                shutil.copy2(f, dst_app_tables / f.name)
            for f in src_app_tables.glob("*.md"):
                shutil.copy2(f, dst_md_tables / f.name)

        # Non-Bayes robustness tables → appendix_tables/ (Bayes goes to appendix_bayes/)
        src_rob_tables = self.output_root / "analysis" / "tables" / "robustness"
        if src_rob_tables.exists():
            for f in src_rob_tables.glob("*.tex"):
                shutil.copy2(f, dst_app_tables / f.name)
            for f in src_rob_tables.glob("*.md"):
                shutil.copy2(f, dst_md_tables / f.name)

        # Appendix figures
        src_app_figs = self.output_root / "analysis" / "figures" / "appendix"
        if src_app_figs.exists():
            for f in src_app_figs.rglob("*.pdf"):
                shutil.copy2(f, dst_figs / f.name)
            for f in src_app_figs.rglob("*.png"):
                shutil.copy2(f, dst_md_figs / f.name)

        # Worked examples → appendix_worked_examples/
        dst_worked = self.output_root / "paper_pack" / "latex" / "appendix_worked_examples"
        src_qual = self.output_root / "qualitative"
        if src_qual.exists():
            for f in src_qual.glob("*.tex"):
                shutil.copy2(f, dst_worked / f.name)

        # Bayesian appendix (dedicated folder for LaTeX \input convenience)
        dst_bayes = self.output_root / "paper_pack" / "latex" / "appendix_bayes"
        src_bayes_tables = self.output_root / "analysis" / "tables" / "robustness" / "bayes"
        if src_bayes_tables.exists():
            for f in src_bayes_tables.glob("*.tex"):
                shutil.copy2(f, dst_bayes / f.name)
        src_bayes_figs = self.output_root / "analysis" / "figures" / "appendix" / "bayes"
        if src_bayes_figs.exists():
            for f in src_bayes_figs.glob("*.pdf"):
                shutil.copy2(f, dst_bayes / f.name)

        logger.info("Paper pack populated (main + appendix_tables + worked_examples + bayes)")

    def _generate_index_files(self) -> None:
        """Generate stable pointer files for LaTeX/Markdown includes."""
        # LaTeX table index
        latex_tables = self.output_root / "paper_pack" / "latex" / "tables"
        tex_files = sorted(latex_tables.glob("*.tex"))
        if tex_files:
            index_lines = [
                "% Auto-generated table index for LaTeX \\input",
                f"% Run ID: {self.run_id}",
                "",
            ]
            for f in tex_files:
                index_lines.append(f"\\input{{tables/{f.name}}}")
            (latex_tables / "_index.tex").write_text("\n".join(index_lines), encoding="utf-8")

        # LaTeX appendix_tables index
        latex_app_tables = self.output_root / "paper_pack" / "latex" / "appendix_tables"
        app_tex = sorted(latex_app_tables.glob("*.tex"))
        if app_tex:
            index_lines = [
                "% Auto-generated appendix table index for LaTeX \\input",
                f"% Run ID: {self.run_id}",
                "",
            ]
            for f in app_tex:
                index_lines.append(f"\\input{{appendix_tables/{f.name}}}")
            (latex_app_tables / "_index.tex").write_text("\n".join(index_lines), encoding="utf-8")

        # LaTeX appendix_worked_examples index
        latex_worked = self.output_root / "paper_pack" / "latex" / "appendix_worked_examples"
        worked_tex = sorted(latex_worked.glob("*.tex"))
        if worked_tex:
            index_lines = [
                "% Auto-generated worked examples appendix index for LaTeX \\input",
                f"% Run ID: {self.run_id}",
                "",
            ]
            for f in worked_tex:
                index_lines.append(f"\\input{{appendix_worked_examples/{f.name}}}")
            (latex_worked / "_index.tex").write_text("\n".join(index_lines), encoding="utf-8")

        # LaTeX appendix_bayes index
        latex_bayes = self.output_root / "paper_pack" / "latex" / "appendix_bayes"
        bayes_tex = sorted(latex_bayes.glob("*.tex"))
        bayes_pdf = sorted(latex_bayes.glob("*.pdf"))
        if bayes_tex or bayes_pdf:
            index_lines = [
                "% Auto-generated Bayesian appendix index for LaTeX",
                f"% Run ID: {self.run_id}",
                "",
            ]
            for f in bayes_tex:
                index_lines.append(f"\\input{{appendix_bayes/{f.name}}}")
            for f in bayes_pdf:
                index_lines.append(f"\\includegraphics[width=\\textwidth]{{appendix_bayes/{f.name}}}")
            (latex_bayes / "_index.tex").write_text("\n".join(index_lines), encoding="utf-8")

        # LaTeX figure index
        latex_figs = self.output_root / "paper_pack" / "latex" / "figures"
        pdf_files = sorted(latex_figs.glob("*.pdf"))
        if pdf_files:
            index_lines = [
                "% Auto-generated figure index for LaTeX \\includegraphics",
                f"% Run ID: {self.run_id}",
                "",
            ]
            for f in pdf_files:
                index_lines.append(f"\\includegraphics[width=\\textwidth]{{figures/{f.name}}}")
            (latex_figs / "_index.tex").write_text("\n".join(index_lines), encoding="utf-8")

        # Markdown index
        md_dir = self.output_root / "paper_pack" / "markdown"
        md_tables = sorted((md_dir / "tables").glob("*.md"))
        md_figs = sorted((md_dir / "figures").glob("*.png"))
        index_lines = [
            f"# Paper Pack Index (Run: {self.run_id})",
            "",
            "## Tables",
            "",
        ]
        for f in md_tables:
            index_lines.append(f"- [{f.name}](tables/{f.name})")
        index_lines.append("")
        index_lines.append("## Figures")
        index_lines.append("")
        for f in md_figs:
            index_lines.append(f"- ![{f.stem}](figures/{f.name})")
        (md_dir / "_index.md").write_text("\n".join(index_lines), encoding="utf-8")

        logger.info("Index files generated")

    def _generate_final_manifest(self) -> None:
        """Generate manifest.json after all outputs exist."""
        from src.reproducibility import generate_manifest_json
        generate_manifest_json(
            run_config=self.rc,
            output_root=self.output_root,
            manifest_dir=self.output_root / "manifest",
        )
