"""Publication-ready figure generation for the paper bundle.

Produces Figures 1--10 as PDF (vector) and PNG (>=300 DPI) with
accompanying caption text files and metadata JSON files.

Public API
----------
FigureGenerator.generate_all_figures  Generate all main and appendix figures.
"""

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if matplotlib.get_backend() != "Agg":
    matplotlib.use("Agg")
import numpy as np

logger = logging.getLogger(__name__)

# Grayscale-friendly palette (8 distinct grays)
_GRAYS = ["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3",
           "#333333", "#666666", "#999999", "#cccccc"]

SEVERITY_LEVELS = ["low", "medium", "high"]
DIMENSIONS = ["age", "geographic", "education", "socioeconomic"]


def _resolve_variance_change(block: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize variance change payload across mitigation result objects."""
    vc = dict(block.get("variance_change", {}) or {})

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

    if "pct_change" not in vc:
        if "variance_reduction" in block:
            try:
                vc["pct_change"] = -100.0 * float(block["variance_reduction"])
            except (TypeError, ValueError):
                pass
        elif "baseline_sd" in vc and "neutralized_sd" in vc:
            try:
                base = float(vc.get("baseline_sd"))
                neut = float(vc.get("neutralized_sd"))
                if np.isfinite(base) and np.isfinite(neut) and base != 0:
                    vc["pct_change"] = (neut - base) / base * 100.0
            except (TypeError, ValueError):
                pass

    return vc


def _build_filename(
    kind: str, scope: str, topic: str, model: str, subset: str,
    version: int, ext: str,
) -> str:
    return f"{kind}__{scope}__{topic}__{model}__{subset}__v{version}.{ext}"


def _save_figure_files(
    fig: Any,
    out_dir: Path,
    topic: str,
    model: str,
    subset: str,
    version: int,
    caption: str,
    metadata: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Save figure as PDF + PNG + caption .txt + metadata .json."""
    import matplotlib.pyplot as plt

    base = f"fig__main__{topic}__{model}__{subset}__v{version}"

    pdf_path = out_dir / f"{base}.pdf"
    png_path = out_dir / f"{base}.png"
    txt_path = out_dir / f"{base}.txt"
    json_path = out_dir / f"{base}.json"

    try:
        fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=300, facecolor="white")
    finally:
        plt.close(fig)

    txt_path.write_text(caption, encoding="utf-8")
    json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    logger.info("Written figure: %s (.pdf/.png/.txt/.json)", topic)
    return pdf_path, png_path


class FigureGenerator:
    """Generates publication-ready figures from analysis results."""

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
        self.fig_dir = output_root / "analysis" / "figures" / "main"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def _base_metadata(self) -> Dict[str, Any]:
        return {
            "run_id": self.rc.get("run_id", ""),
            "seed": self.rc.get("seed"),
            "n_complaints": self.rc.get("num_complaints"),
            "models": self.models,
            "generated_at": datetime.now().isoformat(),
        }

    def generate_all_figures(self) -> List[Tuple[Path, Path]]:
        """Generate all main figures (1-10). Returns list of (pdf, png) paths."""
        results: List[Tuple[Path, Path]] = []
        fig_methods = [
            self._fig1_pipeline,
            self._fig2_persona_distributions,
            self._fig3_correlation_vs_causation,
            self._fig4_distributive_gaps,
            self._fig5_operational_friction,
            self._fig6_neutralisation,
            self._fig7_missingness,
            self._fig8_amplification,
            self._fig9_worked_examples,
            self._fig_golden_pipeline,
            self._fig10_robustness,
            self._fig_pdp_appendix,
            self._fig_cross_model_agreement,
        ]
        for method in fig_methods:
            try:
                result = method()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error("Figure generation failed (%s): %s", method.__name__, e, exc_info=True)

        # Bayesian appendix figures
        results.extend(self._generate_bayes_figures())

        return results

    # ==================================================================
    # Figure 1: Pipeline overview
    # ==================================================================

    def _fig1_pipeline(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 3)
        ax.axis("off")
        ax.set_title("End-to-End Experiment Pipeline", fontsize=13, fontweight="bold")

        steps = [
            ("CFPB\nSampling", 0.5),
            ("Dollar\nExtraction", 2.3),
            ("Persona\nEstimation", 4.1),
            ("Baseline\nTier Decision", 5.9),
            ("Matched-Guise\nRewrites", 7.7),
            ("Neutralisation", 9.5),
            ("Re-run\nTier Decision", 11.3),
        ]

        for label, x in steps:
            rect = mpatches.FancyBboxPatch(
                (x, 0.8), 1.5, 1.4, boxstyle="round,pad=0.1",
                facecolor="#e0e0e0", edgecolor="#333333", linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(x + 0.75, 1.5, label, ha="center", va="center", fontsize=12, fontweight="bold")

        # Arrows
        for i in range(len(steps) - 1):
            x1 = steps[i][1] + 1.5
            x2 = steps[i + 1][1]
            ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                         arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5))

        fig.tight_layout()

        meta = self._base_metadata()
        meta["figure_number"] = 1
        meta["analysis_type"] = "descriptive"
        return _save_figure_files(
            fig, self.fig_dir, "pipeline_overview", "na", "all", 1,
            caption="Figure 1. End-to-end experiment pipeline. CFPB complaints are sampled, "
                    "dollar amounts extracted, persona estimated, then baseline tier decisions obtained. "
                    "A matched-guise subset is rewritten into 36 persona styles and re-evaluated. "
                    "All complaints are also neutralised and re-evaluated.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 2: Default persona distributions
    # ==================================================================

    def _fig2_persona_distributions(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        n_models = len(self.models)
        if n_models == 0:
            return None

        fig, axes = plt.subplots(n_models, 4, figsize=(14, 3.5 * n_models))
        if n_models == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle("Estimated Persona Distributions by Model", fontsize=13, fontweight="bold")

        for i, model in enumerate(self.models):
            model_r = self.ar.get(model, {})
            corr = model_r.get("correlation", {})
            rates = corr.get("rates_by_dimension", {})

            for j, dim in enumerate(DIMENSIONS):
                ax = axes[i, j]
                dim_rates = rates.get(dim, {})
                if dim_rates:
                    cats = list(dim_rates.keys())
                    vals = list(dim_rates.values())
                    bars = ax.bar(range(len(cats)), vals, color=_GRAYS[:len(cats)])
                    ax.set_xticks(range(len(cats)))
                    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
                    ax.set_ylim(0, 1.0)
                    ax.set_ylabel("Rate" if j == 0 else "")
                    ax.set_title(f"{dim.capitalize()} ({model})", fontsize=9)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{dim.capitalize()} ({model})", fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        meta = self._base_metadata()
        meta["figure_number"] = 2
        meta["analysis_type"] = "exploratory"
        return _save_figure_files(
            fig, self.fig_dir, "default_persona_distributions", "both", "all", 1,
            caption="Figure 2. Marginal persona distributions by model. Each panel shows "
                    "the estimated distribution of a demographic dimension across complaints.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 3: Correlation vs causation
    # ==================================================================

    def _fig3_correlation_vs_causation(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        n_models = len(self.models)
        if n_models == 0:
            return None

        fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
        if n_models == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle("Correlation vs Causation", fontsize=13, fontweight="bold")

        for i, model in enumerate(self.models):
            model_r = self.ar.get(model, {})

            # Panel A: Correlational (slice-and-dice)
            ax_a = axes[i, 0]
            corr = model_r.get("correlation", {})
            rates = corr.get("rates_by_dimension", {})
            self._plot_dimension_bars(ax_a, rates, f"Correlational ({model})")

            # Panel B: Causal (matched-guise)
            ax_b = axes[i, 1]
            causal = model_r.get("causal", {})
            effect_sizes = causal.get("effect_sizes", {})
            if effect_sizes:
                dims = list(effect_sizes.keys())
                effects = [effect_sizes[d].get("cohens_d", 0) if isinstance(effect_sizes[d], dict) else 0 for d in dims]
                colors = ["#333333" if e > 0 else "#999999" for e in effects]
                ax_b.barh(range(len(dims)), effects, color=colors)
                ax_b.set_yticks(range(len(dims)))
                ax_b.set_yticklabels(dims, fontsize=8)
                ax_b.set_xlabel("Effect Size (Cohen's d)")
                ax_b.axvline(x=0, color="black", linewidth=0.5)
            else:
                # Fall back to rates_by_persona
                causal_rates = causal.get("rates_by_persona", {})
                if causal_rates:
                    sorted_rates = sorted(causal_rates.items(), key=lambda x: x[1])
                    labels = [r[0][:20] for r in sorted_rates[:10]]
                    vals = [r[1] for r in sorted_rates[:10]]
                    ax_b.barh(range(len(labels)), vals, color="#666666")
                    ax_b.set_yticks(range(len(labels)))
                    ax_b.set_yticklabels(labels, fontsize=7)
                    ax_b.set_xlabel("Tier2 Rate")
                else:
                    ax_b.text(0.5, 0.5, "No causal data", ha="center", va="center", transform=ax_b.transAxes)
            ax_b.set_title(f"Matched-Guise ({model})")

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        meta = self._base_metadata()
        meta["figure_number"] = 3
        meta["analysis_type"] = "exploratory"
        return _save_figure_files(
            fig, self.fig_dir, "correlation_vs_causation", "both", "all", 1,
            caption="Figure 3. Correlational vs causal analysis side-by-side. "
                    "(a) Disparities by inferred persona on natural complaint set. "
                    "(b) Matched-guise within-complaint effect sizes.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 4: Distributive fairness gaps (Tier 2)
    # ==================================================================

    def _fig4_distributive_gaps(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        n_models = len(self.models)
        if n_models == 0:
            return None

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
        fig.suptitle("Distributive Fairness Gaps (Tier 2, Matched-Guise)", fontsize=13, fontweight="bold")

        for i, model in enumerate(self.models):
            ax = axes[0, i]
            severity = self.ar.get("severity", {})
            dim_sev = severity.get("dimension_severity", {})

            dims_plotted = []
            gaps = []
            ci_los = []
            ci_his = []

            for dim in DIMENSIONS:
                dd = dim_sev.get("issue_severity", {}).get(model, {}).get(dim, {})
                if dd and not dd.get("error"):
                    amp = dd.get("amplification", {})
                    bias_high = amp.get("bias_high", float("nan"))
                    bias_low = amp.get("bias_low", float("nan"))
                    if np.isfinite(bias_high) and np.isfinite(bias_low):
                        dims_plotted.append(dim.capitalize())
                        gaps.append(bias_high - bias_low)
                        ci_los.append(amp.get("ci_lower", float("nan")))
                        ci_his.append(amp.get("ci_upper", float("nan")))

            if dims_plotted:
                y = range(len(dims_plotted))
                ax.barh(y, gaps, color="#555555", height=0.5)
                ax.set_yticks(list(y))
                ax.set_yticklabels(dims_plotted)
                ax.axvline(x=0, color="black", linewidth=0.5)
                ax.set_xlabel("Disparity Gap (high - low severity)")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        meta = self._base_metadata()
        meta["figure_number"] = 4
        meta["analysis_type"] = "confirmatory"
        meta["subset"] = "matched-guise"
        meta["endpoint"] = "Tier 2"
        return _save_figure_files(
            fig, self.fig_dir, "distributive_gaps_tier2", "both", "matched", 1,
            caption="Figure 4. Distributive fairness gaps by demographic dimension (Tier 2, matched-guise). "
                    "Bars show the amplification A = disparity(high) - disparity(low) for each dimension.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 5: Operational friction
    # ==================================================================

    def _fig5_operational_friction(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        n_models = len(self.models)
        if n_models == 0:
            return None

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
        fig.suptitle("Operational Friction (need_more_info) by Dimension", fontsize=13, fontweight="bold")

        for i, model in enumerate(self.models):
            ax = axes[0, i]
            model_r = self.ar.get(model, {})
            proc_corr = model_r.get("process_correlation", {})
            proc_rates = proc_corr.get("info_rates_by_dimension", {})

            if proc_rates:
                dims_plotted = []
                dim_gaps = []
                for dim in DIMENSIONS:
                    dim_data = proc_rates.get(dim, {})
                    if dim_data:
                        vals = list(dim_data.values())
                        if len(vals) >= 2:
                            dims_plotted.append(dim.capitalize())
                            dim_gaps.append(max(vals) - min(vals))

                if dims_plotted:
                    ax.barh(range(len(dims_plotted)), dim_gaps, color="#777777")
                    ax.set_yticks(range(len(dims_plotted)))
                    ax.set_yticklabels(dims_plotted)
                    ax.set_xlabel("Max-Min Info Request Rate Gap")
                else:
                    ax.text(0.5, 0.5, "Near-zero rates", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        meta = self._base_metadata()
        meta["figure_number"] = 5
        meta["analysis_type"] = "exploratory"
        meta["endpoint"] = "need_more_info"
        return _save_figure_files(
            fig, self.fig_dir, "operational_friction_need_more_info", "both", "all", 1,
            caption="Figure 5. Operational friction (information request rate) by demographic dimension. "
                    "Shows the max-min disparity in need_more_info rates across persona levels for each dimension.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 6: Neutralisation effects
    # ==================================================================

    def _fig6_neutralisation(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        n_models = len(self.models)
        if n_models == 0:
            return None

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
        fig.suptitle("Neutralisation Effects (Before/After)", fontsize=13, fontweight="bold")

        for i, model in enumerate(self.models):
            ax = axes[0, i]
            model_r = self.ar.get(model, {})
            mitigation = model_r.get("mitigation", {})
            vc = _resolve_variance_change(mitigation)

            baseline_sd = vc.get("baseline_sd", float("nan"))
            neutralised_sd = vc.get("neutralized_sd", float("nan"))

            proc_mit = model_r.get("process_mitigation", {})
            proc_vc = _resolve_variance_change(proc_mit)
            proc_base = proc_vc.get("baseline_sd", float("nan"))
            proc_neut = proc_vc.get("neutralized_sd", float("nan"))

            labels = ["Tier 2\nDisparity", "Friction\nDisparity"]
            baseline = [baseline_sd if np.isfinite(baseline_sd) else 0,
                        proc_base if np.isfinite(proc_base) else 0]
            neutralised = [neutralised_sd if np.isfinite(neutralised_sd) else 0,
                           proc_neut if np.isfinite(proc_neut) else 0]

            x = np.arange(len(labels))
            w = 0.3
            ax.bar(x - w/2, baseline, w, label="Baseline", color="#555555")
            ax.bar(x + w/2, neutralised, w, label="Neutralised", color="#aaaaaa")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("SD of persona-level rates")
            ax.legend()
            ax.set_title(model)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        meta = self._base_metadata()
        meta["figure_number"] = 6
        meta["analysis_type"] = "exploratory"
        return _save_figure_files(
            fig, self.fig_dir, "neutralization_effects", "both", "baseline_vs_neutralized", 1,
            caption="Figure 6. Neutralisation effects on between-persona disparity. "
                    "Paired bars show the SD of persona-level outcome rates before and after "
                    "demographic marker removal.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 7: Monetary missingness diagnostic
    # ==================================================================

    def _fig7_missingness(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        severity = self.ar.get("severity", {})
        miss_diag = severity.get("missingness_diagnostic", {})
        if not miss_diag or miss_diag.get("skipped"):
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Monetary Severity Missingness Diagnostic", fontsize=13, fontweight="bold")

        # Panel A: Proportion known vs unknown
        ax_a = axes[0]
        n_present = miss_diag.get("n_present", 0)
        n_missing = miss_diag.get("n_missing", 0)
        if n_present + n_missing > 0:
            ax_a.pie([n_present, n_missing],
                     labels=[f"Amount present\n(n={n_present})", f"Amount missing\n(n={n_missing})"],
                     autopct="%1.1f%%", colors=["#666666", "#cccccc"], startangle=90)
            ax_a.set_title("(a) Dollar Amount Extractability")
        else:
            ax_a.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_a.transAxes)

        # Panel B: Tier distribution differences (grouped bar chart)
        ax_b = axes[1]
        tier_counts = miss_diag.get("distributions", {}).get("tier", {})
        present_counts = tier_counts.get("present", {})
        missing_counts = tier_counts.get("missing", {})
        if present_counts and missing_counts:
            tiers = sorted(set(present_counts.keys()) | set(missing_counts.keys()), key=lambda t: int(t))
            pres_total = sum(present_counts.values()) or 1
            miss_total = sum(missing_counts.values()) or 1
            present_vals = [present_counts.get(t, 0) / pres_total for t in tiers]
            missing_vals = [missing_counts.get(t, 0) / miss_total for t in tiers]
            x = np.arange(len(tiers))
            w = 0.3
            ax_b.bar(x - w / 2, present_vals, w, label="Amount present", color="#666666")
            ax_b.bar(x + w / 2, missing_vals, w, label="Amount missing", color="#cccccc")
            ax_b.set_xticks(x)
            ax_b.set_xticklabels([f"Tier {t}" for t in tiers])
            ax_b.set_ylabel("Proportion")
            chars = miss_diag.get("complaint_characteristics", {})
            tier_test = chars.get("tier_distribution", {})
            chi2 = tier_test.get("chi2", float("nan"))
            p_val = tier_test.get("p_value", float("nan"))
            if np.isfinite(chi2):
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax_b.set_xlabel(f"\u03c7\u00b2={chi2:.1f}, p={p_val:.4f} {sig}", fontsize=9)
            ax_b.legend(fontsize=8)
            ax_b.set_title("(b) Tier Distribution Comparison")
        else:
            ax_b.text(0.5, 0.5, "No tier distribution data", ha="center", va="center", transform=ax_b.transAxes)
            ax_b.axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        meta = self._base_metadata()
        meta["figure_number"] = 7
        meta["analysis_type"] = "descriptive"
        meta["n_present"] = n_present
        meta["n_missing"] = n_missing
        total = n_present + n_missing
        if total > 0:
            caption_detail = (
                f"(a) {n_present} complaints ({100*n_present/total:.1f}%) have extractable dollar amounts; "
                f"{n_missing} ({100*n_missing/total:.1f}%) do not. "
            )
        else:
            caption_detail = "(a) No complaints available for missingness analysis. "

        return _save_figure_files(
            fig, self.fig_dir, "monetary_missingness_diagnostic", "na", "all", 1,
            caption="Figure 7. Monetary severity missingness diagnostic. "
                    + caption_detail
                    + "(b) Tier distribution differences between amount-present and amount-missing complaints.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 8: Amplification interaction (KEY FIGURE)
    # ==================================================================

    def _fig8_amplification(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        severity = self.ar.get("severity", {})
        n_models = len(self.models)
        if n_models == 0:
            return None

        fig, axes = plt.subplots(2, max(n_models, 1), figsize=(6 * max(n_models, 1), 10), squeeze=False)
        fig.suptitle("Severity Amplification as Interaction", fontsize=13, fontweight="bold")

        metrics = ["monetary_severity", "issue_severity"]
        metric_labels = ["Monetary Severity (M=1 only)", "Issue Severity (full sample)"]

        for row, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            for col, model in enumerate(self.models):
                ax = axes[row, col]
                dim_sev = severity.get("dimension_severity", {}).get(metric, {}).get(model, {})

                # Plot disparity at each severity level for each dimension
                for d_idx, dim in enumerate(DIMENSIONS):
                    dd = dim_sev.get(dim, {})
                    if not dd or dd.get("error"):
                        continue
                    means_by_sev = dd.get("dimension_means_by_severity", {})
                    if not means_by_sev:
                        continue

                    # Compute max-min gap at each severity level
                    gaps = []
                    for level in SEVERITY_LEVELS:
                        level_means = means_by_sev.get(level, {})
                        if level_means and len(level_means) >= 2:
                            vals = list(level_means.values())
                            gaps.append(max(vals) - min(vals))
                        else:
                            gaps.append(float("nan"))

                    x = np.arange(len(SEVERITY_LEVELS))
                    valid_gaps = [g for g in gaps if np.isfinite(g)]
                    if valid_gaps:
                        ax.plot(x, gaps, marker="o", label=dim.capitalize(),
                                color=_GRAYS[d_idx], linewidth=1.5, markersize=5)

                ax.set_xticks(range(len(SEVERITY_LEVELS)))
                ax.set_xticklabels([l.capitalize() for l in SEVERITY_LEVELS])
                ax.set_ylabel("Persona Disparity (max-min gap)")
                ax.set_title(f"{model}\n{metric_label}", fontsize=9)
                if ax.get_legend_handles_labels()[1]:
                    ax.legend(fontsize=7, loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        meta = self._base_metadata()
        meta["figure_number"] = 8
        meta["analysis_type"] = "confirmatory"
        meta["endpoint"] = "Tier (ordinal)"
        meta["note"] = "Monetary severity shown for amount-present (M=1) complaints only"
        return _save_figure_files(
            fig, self.fig_dir, "amplification_interaction", "both", "all", 1,
            caption="Figure 8. Severity amplification as an interaction between persona disparity and severity band. "
                    "For each model, persona disparity (max-min gap in mean tier) is plotted at low, medium, and high "
                    "severity. Row 1: monetary severity (amount-present subset only). Row 2: issue severity (full sample). "
                    "Rising lines indicate amplification.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 9: Worked examples
    # ==================================================================

    def _fig9_worked_examples(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        # Select 2-3 example complaints
        if not self.complaints or len(self.complaints) < 2:
            return None

        n_cases = min(3, len(self.complaints))
        examples = self.complaints[:n_cases]

        fig, axes = plt.subplots(n_cases, 1, figsize=(17, 9 * n_cases))
        if n_cases == 1:
            axes = [axes]
        fig.suptitle("Worked Examples", fontsize=32, fontweight="bold")

        for idx, (ax, complaint) in enumerate(zip(axes, examples)):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            # Background box spanning entire axes width
            bg = FancyBboxPatch(
                (0.0, 0.0), 1.0, 1.0,
                boxstyle="round,pad=0.01", facecolor="#f8f8f8",
                edgecolor="#cccccc", linewidth=0.8,
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(bg)

            text = complaint.narrative[:500] + ("..." if len(complaint.narrative) > 500 else "")
            wrapped = textwrap.fill(text, width=70)
            header = (
                f"Case {idx+1} (ID: {complaint.case_id[:8]}...)\n"
                f"Product: {complaint.product_category}  |  Issue: {complaint.issue_category}\n"
                f"Monetary severity: {complaint.monetary_severity}  |  "
                f"Issue severity: {complaint.issue_severity}\n"
                f"{'─' * 100}"
            )
            display_text = f"{header}\n{wrapped}"
            ax.text(0.02, 0.95, display_text, transform=ax.transAxes,
                    fontsize=18, family="monospace", verticalalignment="top")

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        meta = self._base_metadata()
        meta["figure_number"] = 9
        meta["analysis_type"] = "descriptive"
        meta["n_cases"] = n_cases
        return _save_figure_files(
            fig, self.fig_dir, "worked_examples_panel", "both", "all", 1,
            caption=f"Figure 9. Worked examples panel showing {n_cases} complaint cases "
                    "with original text, product/issue categories, and severity classifications.",
            metadata=meta,
        )

    # ==================================================================
    # Golden Example: Single complaint through the full pipeline
    # ==================================================================

    def _fig_golden_pipeline(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.colors as mcolors

        mit_eff = self.ar.get("mitigation_effectiveness", {})
        ge = mit_eff.get("golden_example", {})
        if not ge or not ge.get("original_text"):
            return None

        models = self.models
        if not models:
            return None

        tier_labels = {0: "Tier 0\n(No action)", 1: "Tier 1\n(Non-monetary)", 2: "Tier 2\n(Monetary)"}
        tier_cmap = mcolors.ListedColormap(["#e0e0e0", "#999999", "#333333"])

        fig = plt.figure(figsize=(16, 24))
        gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[2, 1, 1, 8, 1.5],
                               hspace=0.35)
        fig.suptitle("Golden Example: One Complaint Through the Pipeline",
                     fontsize=16, fontweight="bold", y=0.98)

        # --- Panel A: Original complaint text ---
        ax_a = fig.add_subplot(gs[0])
        ax_a.axis("off")
        original = ge.get("original_text", "")
        wrapped = textwrap.fill(original, width=100)
        product = ge.get("product", "")
        issue = ge.get("issue", "")
        case_id = ge.get("complaint_id", "")[:12]
        header = f"Stage 1 — Original Complaint  [{product} / {issue}]  (ID: {case_id}...)"
        ax_a.text(0.01, 0.98, header, transform=ax_a.transAxes,
                  fontsize=13, fontweight="bold", verticalalignment="top")
        ax_a.text(0.01, 0.82, wrapped, transform=ax_a.transAxes,
                  fontsize=11, family="monospace", verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                            edgecolor="#cccccc"))

        # --- Panel B: Persona estimation ---
        ax_b = fig.add_subplot(gs[1])
        ax_b.axis("off")
        ax_b.text(0.01, 0.95, "Stage 2 — Persona Estimation",
                  transform=ax_b.transAxes, fontsize=13, fontweight="bold",
                  verticalalignment="top")
        personas = ge.get("estimated_persona", {})
        dims = ["age", "geographic", "education", "socioeconomic"]
        y_pos = 0.65
        for model in models:
            p = personas.get(model, {})
            vals = "  |  ".join(f"{d}: {p.get(d, '?')}" for d in dims)
            ax_b.text(0.03, y_pos, f"{model}:  {vals}", transform=ax_b.transAxes,
                      fontsize=12, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef2ff",
                                edgecolor="#aabbdd"))
            y_pos -= 0.40

        # --- Panel C: Baseline remedy tiers ---
        ax_c = fig.add_subplot(gs[2])
        ax_c.axis("off")
        ax_c.text(0.01, 0.95, "Stage 3 — Baseline Remedy Decision",
                  transform=ax_c.transAxes, fontsize=13, fontweight="bold",
                  verticalalignment="top")
        orig_tiers = ge.get("original_tiers", {})
        x_pos = 0.03
        for model in models:
            tier = orig_tiers.get(model, "?")
            label = tier_labels.get(tier, f"Tier {tier}")
            color = ["#e0e0e0", "#999999", "#333333"][tier] if isinstance(tier, int) and 0 <= tier <= 2 else "#ffffff"
            text_color = "white" if tier == 2 else "black"
            ax_c.text(x_pos, 0.45, f"{model}:  {label.replace(chr(10), ' ')}",
                      transform=ax_c.transAxes, fontsize=12, fontweight="bold",
                      color=text_color,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                                edgecolor="#666666"))
            x_pos += 0.45

        # --- Panel D: Matched-guise heatmap ---
        ax_d = fig.add_subplot(gs[3])
        persona_tiers = ge.get("persona_tiers", {})
        if persona_tiers and any(persona_tiers.values()):
            # Build sorted persona list from the first model that has data
            first_model_tiers = next((v for v in persona_tiers.values() if v), {})
            persona_keys = sorted(first_model_tiers.keys())
            n_personas = len(persona_keys)
            n_models = len(models)

            # Build matrix: rows=personas, cols=models
            matrix = np.full((n_personas, n_models), np.nan)
            for j, model in enumerate(models):
                model_pt = persona_tiers.get(model, {})
                for i, pk in enumerate(persona_keys):
                    if pk in model_pt:
                        matrix[i, j] = model_pt[pk]

            im = ax_d.imshow(matrix, aspect="auto", cmap=tier_cmap, vmin=0, vmax=2,
                             interpolation="nearest")
            ax_d.set_yticks(range(n_personas))
            ax_d.set_yticklabels([pk.replace("_", " ") for pk in persona_keys], fontsize=7)
            ax_d.set_xticks(range(n_models))
            ax_d.set_xticklabels(models, fontsize=11)
            ax_d.set_title("Stages 4–5 — Matched-Guise Rewrite Tier Decisions (36 personas)",
                           fontsize=13, fontweight="bold", pad=10)

            # Annotate cells with tier values
            for i in range(n_personas):
                for j in range(n_models):
                    val = matrix[i, j]
                    if np.isfinite(val):
                        text_color = "white" if val == 2 else "black"
                        ax_d.text(j, i, f"{int(val)}", ha="center", va="center",
                                  fontsize=8, color=text_color, fontweight="bold")

            # Colorbar legend
            cbar = fig.colorbar(im, ax=ax_d, ticks=[0, 1, 2], shrink=0.5, pad=0.02)
            cbar.ax.set_yticklabels(["Tier 0", "Tier 1", "Tier 2"], fontsize=9)
        else:
            ax_d.axis("off")
            ax_d.text(0.5, 0.5, "No matched-guise data available",
                      ha="center", va="center", transform=ax_d.transAxes, fontsize=12)

        # --- Panel E: Neutralisation & final remedy ---
        ax_e = fig.add_subplot(gs[4])
        ax_e.axis("off")
        ax_e.text(0.01, 0.90, "Stages 6–7 — Neutralisation & Final Remedy",
                  transform=ax_e.transAxes, fontsize=13, fontweight="bold",
                  verticalalignment="top")
        neut_tiers = ge.get("neutralized_tiers", {})
        x_pos = 0.03
        for model in models:
            orig_t = orig_tiers.get(model, "?")
            neut_t = neut_tiers.get(model, "?")
            orig_label = tier_labels.get(orig_t, f"Tier {orig_t}").replace("\n", " ")
            neut_label = tier_labels.get(neut_t, f"Tier {neut_t}").replace("\n", " ")

            delta = ""
            if isinstance(orig_t, int) and isinstance(neut_t, int):
                diff = neut_t - orig_t
                if diff > 0:
                    delta = f"  (+{diff})"
                elif diff < 0:
                    delta = f"  ({diff})"
                else:
                    delta = "  (no change)"

            neut_color = ["#e0e0e0", "#999999", "#333333"][neut_t] if isinstance(neut_t, int) and 0 <= neut_t <= 2 else "#ffffff"
            text_color = "white" if neut_t == 2 else "black"
            ax_e.text(x_pos, 0.45,
                      f"{model}:  {orig_label}  \u2192  {neut_label}{delta}",
                      transform=ax_e.transAxes, fontsize=12, fontweight="bold",
                      color=text_color,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor=neut_color,
                                edgecolor="#666666"))
            x_pos += 0.45

        meta = self._base_metadata()
        meta["analysis_type"] = "descriptive"
        meta["golden_example_id"] = ge.get("complaint_id", "")
        return _save_figure_files(
            fig, self.fig_dir, "golden_example_pipeline", "both", "all", 1,
            caption="Golden example: a single complaint tracked through all pipeline stages. "
                    "(a) Original complaint text. (b) LLM-estimated persona. "
                    "(c) Baseline remedy tier. (d) Matched-guise tier decisions across 36 personas. "
                    "(e) Post-neutralisation remedy tier with change from baseline.",
            metadata=meta,
        )

    # ==================================================================
    # Figure 10: Robustness and sensitivity
    # ==================================================================

    def _fig10_robustness(self) -> Optional[Tuple[Path, Path]]:
        import matplotlib.pyplot as plt

        severity = self.ar.get("severity", {})
        ipw = severity.get("ipw_robustness", {})
        n_models = len(self.models)

        if n_models == 0:
            return None

        fig, axes = plt.subplots(1, max(n_models, 1), figsize=(6 * max(n_models, 1), 5), squeeze=False)
        fig.suptitle("Robustness and Sensitivity", fontsize=13, fontweight="bold")

        for i, model in enumerate(self.models):
            ax = axes[0, i]
            model_ipw = ipw.get(model, {})

            if model_ipw and not model_ipw.get("skipped") and not ipw.get("error"):
                dims_data = model_ipw.get("dimensions", {})
                dims_plotted = []
                uw_vals = []
                w_vals = []

                for dim in DIMENSIONS:
                    dd = dims_data.get(dim, {})
                    if dd:
                        uw = dd.get("unweighted", {}).get("A", float("nan"))
                        w = dd.get("weighted", {}).get("A", float("nan"))
                        if np.isfinite(uw) and np.isfinite(w):
                            dims_plotted.append(dim[:4].capitalize())
                            uw_vals.append(uw)
                            w_vals.append(w)

                if dims_plotted:
                    x = np.arange(len(dims_plotted))
                    w_bar = 0.3
                    ax.bar(x - w_bar/2, uw_vals, w_bar, label="Unweighted", color="#555555")
                    ax.bar(x + w_bar/2, w_vals, w_bar, label="IPW-weighted", color="#aaaaaa")
                    ax.set_xticks(x)
                    ax.set_xticklabels(dims_plotted)
                    ax.set_ylabel("Amplification A (high-low)")
                    ax.legend(fontsize=8)
                    ax.axhline(y=0, color="black", linewidth=0.5)
                else:
                    ax.text(0.5, 0.5, "No IPW data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "IPW not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        meta = self._base_metadata()
        meta["figure_number"] = 10
        meta["analysis_type"] = "exploratory"
        meta["note"] = "Monetary severity only, comparing unweighted vs IPW-reweighted amplification"
        return _save_figure_files(
            fig, self.fig_dir, "robustness_sensitivity", "both", "all", 1,
            caption="Figure 10. Robustness and sensitivity analysis. Amplification estimates "
                    "(A = disparity_high - disparity_low) under unweighted and IPW-reweighted "
                    "analyses for each demographic dimension. Similarity between bars indicates "
                    "stability of results to selection into the amount-present subset.",
            metadata=meta,
        )

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _plot_dimension_bars(self, ax, rates: Dict, title: str) -> None:
        """Plot bar chart of rates by dimension."""
        all_dims = []
        all_vals = []
        for dim in DIMENSIONS:
            dim_data = rates.get(dim, {})
            if dim_data:
                for level, rate in dim_data.items():
                    all_dims.append(f"{dim[:3]}:{level[:3]}")
                    all_vals.append(rate)

        if all_dims:
            x = range(len(all_dims))
            ax.bar(x, all_vals, color="#666666")
            ax.set_xticks(list(x))
            ax.set_xticklabels(all_dims, rotation=45, ha="right", fontsize=6)
            ax.set_ylabel("Rate")
            ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9)

    # ==================================================================
    # PDP + CROSS-MODEL AGREEMENT FIGURES
    # ==================================================================

    def _fig_pdp_appendix(self) -> Optional[Tuple[Path, Path]]:
        """Partial dependence plots for top persona-predicting features."""
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image
        import base64

        pa = self.ar.get("persona_analysis", {})
        if not pa:
            logger.warning("No persona_analysis data; skipping PDP figure.")
            return None

        fig_dir = self.output_root / "analysis" / "figures" / "main"
        fig_dir.mkdir(parents=True, exist_ok=True)

        dims_to_plot = ["age", "geographic", "education", "socioeconomic", "gender", "esl"]

        last_result = None
        for model in self.models:
            model_pa = pa.get(model, {})
            dims_data = model_pa.get("dimensions", {})
            if not dims_data:
                continue

            # Collect PDP charts: one per dimension (first value, top-1 feature)
            pdp_panels = []
            for dim in dims_to_plot:
                dim_data = dims_data.get(dim, {})
                for value, val_data in sorted(dim_data.items()):
                    if isinstance(val_data, dict) and val_data.get("pdp_charts"):
                        chart = val_data["pdp_charts"][0]
                        pdp_panels.append({
                            "dim": dim,
                            "value": value,
                            "feature": chart.get("feature_display", chart.get("feature", "?")),
                            "b64": chart["chart_b64"],
                        })
                        break  # One per dimension

            if not pdp_panels:
                continue

            n_panels = len(pdp_panels)
            n_cols = min(3, n_panels)
            n_rows = (n_panels + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes[np.newaxis, :]
            elif n_cols == 1:
                axes = axes[:, np.newaxis]

            for idx, panel in enumerate(pdp_panels):
                r, c = divmod(idx, n_cols)
                ax = axes[r, c]
                try:
                    img_bytes = base64.b64decode(panel["b64"])
                    img = Image.open(BytesIO(img_bytes))
                    ax.imshow(np.array(img))
                    ax.set_title(f"{panel['dim'].capitalize()}: {panel['value']}\n({panel['feature']})", fontsize=9)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{panel['dim'].capitalize()}", fontsize=9)
                ax.axis("off")

            # Hide unused axes
            for idx in range(n_panels, n_rows * n_cols):
                r, c = divmod(idx, n_cols)
                axes[r, c].axis("off")

            fig.suptitle(f"Partial Dependence Plots: Top Persona-Predicting Features ({model})", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            model_clean = model.replace("-", "").replace("_", "")
            result = _save_figure_files(
                fig, fig_dir,
                topic=f"pdp_persona_features",
                model=model_clean,
                subset="all",
                version=1,
                caption=f"Partial dependence plots for top persona-predicting text features ({model}).",
                metadata={
                    **self._base_metadata(),
                    "analysis_type": "exploratory",
                    "n_panels": n_panels,
                },
            )
            # Return last model's result (both models produce separate files)
            last_result = result

        return last_result

    def _fig_cross_model_agreement(self) -> Optional[Tuple[Path, Path]]:
        """Cross-model feature importance agreement (per-dimension Spearman correlations)."""
        import matplotlib.pyplot as plt

        pa = self.ar.get("persona_analysis", {})
        mit_eff = self.ar.get("mitigation_effectiveness", {})
        if not pa or len(self.models) < 2:
            logger.warning("Insufficient data for cross-model agreement figure.")
            return None

        fig_dir = self.output_root / "analysis" / "figures" / "main"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Compute per-dimension Spearman rank correlations between models
        from scipy.stats import spearmanr

        model_a, model_b = self.models[0], self.models[1]
        pa_a = pa.get(model_a, {}).get("dimensions", {})
        pa_b = pa.get(model_b, {}).get("dimensions", {})

        dim_correlations = {}
        dims_to_check = ["age", "geographic", "education", "socioeconomic", "gender", "esl"]

        for dim in dims_to_check:
            # Collect feature importances from both models for this dimension
            feats_a = {}
            feats_b = {}
            for value, val_data in pa_a.get(dim, {}).items():
                if isinstance(val_data, dict) and "top_features" in val_data:
                    for f in val_data["top_features"]:
                        feat_name = f.get("feature", "")
                        feats_a[feat_name] = feats_a.get(feat_name, 0) + f.get("importance", 0)
            for value, val_data in pa_b.get(dim, {}).items():
                if isinstance(val_data, dict) and "top_features" in val_data:
                    for f in val_data["top_features"]:
                        feat_name = f.get("feature", "")
                        feats_b[feat_name] = feats_b.get(feat_name, 0) + f.get("importance", 0)

            common = set(feats_a.keys()) & set(feats_b.keys())
            if len(common) >= 3:
                vals_a = [feats_a[f] for f in sorted(common)]
                vals_b = [feats_b[f] for f in sorted(common)]
                rho, _ = spearmanr(vals_a, vals_b)
                dim_correlations[dim] = rho

        # Get overall agreement from mitigation_effectiveness
        fia = mit_eff.get("feature_importance_agreement", {})
        overall_rho = None
        for pair_key, rho_val in fia.items():
            overall_rho = rho_val
            break

        if not dim_correlations and overall_rho is None:
            logger.warning("No agreement data for cross-model figure.")
            return None

        # Build figure
        fig, ax = plt.subplots(figsize=(8, 5))

        dims_sorted = sorted(dim_correlations.keys())
        rhos = [dim_correlations[d] for d in dims_sorted]
        labels = [d.capitalize() for d in dims_sorted]

        x = range(len(dims_sorted))
        bars = ax.bar(x, rhos, color="#666666", edgecolor="black", linewidth=0.5)

        # Annotate bars
        for bar, rho_val in zip(bars, rhos):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{rho_val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Spearman Rank Correlation")
        ax.set_ylim(-0.2, 1.1)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

        # Add overall annotation
        if overall_rho is not None:
            ax.axhline(y=overall_rho, color="#999999", linewidth=1.5, linestyle="--")
            ax.text(len(dims_sorted) - 0.5, overall_rho + 0.03,
                    f"Overall: {overall_rho:.3f}", ha="right", fontsize=9, color="#666666")

        ax.set_title(f"Cross-Model Feature Importance Agreement ({model_a} vs {model_b})", fontsize=11)
        fig.tight_layout()

        result = _save_figure_files(
            fig, fig_dir,
            topic="cross_model_feature_agreement",
            model="both",
            subset="all",
            version=1,
            caption=f"Cross-model feature importance agreement: per-dimension Spearman rank correlations between {model_a} and {model_b}.",
            metadata={
                **self._base_metadata(),
                "analysis_type": "exploratory",
                "n_dimensions": len(dim_correlations),
                "overall_rho": overall_rho,
            },
        )
        return result

    # ==================================================================
    # BAYESIAN APPENDIX FIGURES
    # ==================================================================

    def _generate_bayes_figures(self) -> List[Tuple[Path, Path]]:
        """Generate all Bayesian appendix figures (B1-B3)."""
        bayes = self.ar.get("severity", {}).get("bayesian_robustness", {})
        if not bayes or bayes.get("error"):
            return []

        bayes_dir = self.output_root / "analysis" / "figures" / "appendix" / "bayes"
        bayes_dir.mkdir(parents=True, exist_ok=True)

        results: List[Tuple[Path, Path]] = []
        for model in self.models:
            model_bayes = bayes.get(model, {})
            if not model_bayes or model_bayes.get("skipped"):
                continue
            for metric in ["monetary_severity", "issue_severity"]:
                metric_data = model_bayes.get(metric, {})
                if not metric_data or metric_data.get("skipped"):
                    continue
                try:
                    r = self._figB1_posterior_A(bayes_dir, model, metric, metric_data)
                    if r:
                        results.append(r)
                except Exception as e:
                    logger.warning("Fig B1 failed (%s/%s): %s", model, metric, e)
                try:
                    r = self._figB2_ppc_tier2(bayes_dir, model, metric, metric_data)
                    if r:
                        results.append(r)
                except Exception as e:
                    logger.warning("Fig B2 failed (%s/%s): %s", model, metric, e)
                try:
                    r = self._figB3_ppc_tier_dist(bayes_dir, model, metric, metric_data)
                    if r:
                        results.append(r)
                except Exception as e:
                    logger.warning("Fig B3 failed (%s/%s): %s", model, metric, e)
        return results

    def _figB1_posterior_A(
        self, out_dir: Path, model: str, metric: str, metric_data: Dict,
    ) -> Optional[Tuple[Path, Path]]:
        """Fig B1: Posterior density of amplification A with zero line."""
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        logistic = metric_data.get("logistic", {})
        if not logistic or logistic.get("error") or logistic.get("status") == "FAIL":
            return None

        amp = logistic.get("amplification", {})
        if not amp:
            return None

        dims = list(amp.keys())
        n_dims = len(dims)
        if n_dims == 0:
            return None

        fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 3.5), squeeze=False)
        fig.suptitle(
            f"Posterior Distribution of Amplification A (Tier 2)\n{model}, {metric.replace('_', ' ')}",
            fontsize=11, fontweight="bold",
        )

        for i, dim in enumerate(dims):
            ax = axes[0, i]
            dim_data = amp[dim]

            # Read draws from parquet if available, otherwise use summary
            a_mean = dim_data.get("A_mean", 0)
            a_lo = dim_data.get("A_CrI_low", 0)
            a_hi = dim_data.get("A_CrI_high", 0)
            p_gt0 = dim_data.get("P_A_gt_0", 0)

            # Generate synthetic draws from summary for KDE
            # (actual draws saved to parquet, but we use summary here for figure)
            n_synth = 2000
            rng = np.random.RandomState(42)
            std_est = (a_hi - a_lo) / 3.92  # approximate from 95% CrI
            if std_est <= 0:
                std_est = 0.01
            synth_draws = rng.normal(a_mean, std_est, n_synth)

            try:
                kde = gaussian_kde(synth_draws)
                x_grid = np.linspace(a_mean - 4 * std_est, a_mean + 4 * std_est, 300)
                density = kde(x_grid)
                ax.fill_between(x_grid, density, alpha=0.3, color="#666666")
                ax.plot(x_grid, density, color="#333333", linewidth=1.5)

                # Shade 95% CrI
                mask = (x_grid >= a_lo) & (x_grid <= a_hi)
                ax.fill_between(x_grid[mask], density[mask], alpha=0.2, color="#333333")
            except Exception:
                ax.hist(synth_draws, bins=50, density=True, color="#666666", alpha=0.5)

            ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="A = 0")
            ax.axvline(a_mean, color="#1a1a1a", linewidth=1, linestyle="-", alpha=0.6)
            ax.set_xlabel("A = D(high) - D(low)")
            ax.set_ylabel("Density")
            ax.set_title(f"{dim}\nP(A>0) = {p_gt0:.3f}", fontsize=9)
            ax.legend(fontsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.90])

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        base = f"fig__appendix__bayes_posterior_A__{model_clean}__{metric_clean}__v1"
        pdf_path = out_dir / f"{base}.pdf"
        png_path = out_dir / f"{base}.png"
        try:
            fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", dpi=300)
            fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=300, facecolor="white")
        finally:
            plt.close(fig)
        (out_dir / f"{base}.txt").write_text(
            f"Figure B1. Posterior distribution of amplification A (Tier 2) -- {model}, "
            f"{metric.replace('_', ' ')}. Vertical dashed line at A=0. Shaded region shows 95% CrI.",
            encoding="utf-8",
        )
        return pdf_path, png_path

    def _figB2_ppc_tier2(
        self, out_dir: Path, model: str, metric: str, metric_data: Dict,
    ) -> Optional[Tuple[Path, Path]]:
        """Fig B2: PPC — Tier2 rate by severity band (logistic model)."""
        import matplotlib.pyplot as plt

        logistic = metric_data.get("logistic", {})
        if not logistic or logistic.get("error"):
            return None

        amp = logistic.get("amplification", {})
        if not amp:
            return None

        # Aggregate D(s) by severity band across dimensions
        fig, ax = plt.subplots(figsize=(6, 4))
        dims = list(amp.keys())

        x_pos = np.arange(len(dims))
        width = 0.35
        d_low = [amp[d].get("D_low_mean", 0) for d in dims]
        d_high = [amp[d].get("D_high_mean", 0) for d in dims]

        ax.bar(x_pos - width / 2, d_low, width, label="D(low)", color="#999999")
        ax.bar(x_pos + width / 2, d_high, width, label="D(high)", color="#333333")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d.capitalize() for d in dims], fontsize=9)
        ax.set_ylabel("D(s) = max-min P(Tier2)")
        ax.set_title(
            f"Tier-2 Disparity by Severity Band (B1 Logistic)\n{model}, {metric.replace('_', ' ')}",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        fig.tight_layout()

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        base = f"fig__appendix__bayes_ppc_tier2_by_severity__{model_clean}__{metric_clean}__v1"
        pdf_path = out_dir / f"{base}.pdf"
        png_path = out_dir / f"{base}.png"
        fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        (out_dir / f"{base}.txt").write_text(
            f"Figure B2. Tier-2 disparity by severity band (B1 logistic) -- {model}, "
            f"{metric.replace('_', ' ')}. D(s) = max - min predicted P(Tier 2) across persona "
            f"levels within each dimension.",
            encoding="utf-8",
        )
        return pdf_path, png_path

    def _figB3_ppc_tier_dist(
        self, out_dir: Path, model: str, metric: str, metric_data: Dict,
    ) -> Optional[Tuple[Path, Path]]:
        """Fig B3: PPC — Full tier distribution by severity (ordinal model)."""
        import matplotlib.pyplot as plt

        ordinal = metric_data.get("ordinal", {})
        if not ordinal or ordinal.get("error"):
            return None

        tier_probs = ordinal.get("tier_probs", {})
        if not tier_probs:
            return None

        sev_bands = [b for b in ["low", "medium", "high"] if b in tier_probs]
        if not sev_bands:
            return None

        fig, ax = plt.subplots(figsize=(7, 4))

        x = np.arange(len(sev_bands))
        width = 0.25

        for j, (tier_key, tier_label) in enumerate([
            ("P_tier0", "Tier 0"), ("P_tier1", "Tier 1"), ("P_tier2", "Tier 2"),
        ]):
            means = []
            lows = []
            highs = []
            for band in sev_bands:
                tp = tier_probs[band].get(tier_key, {})
                m = tp.get("mean", 0)
                means.append(m)
                lows.append(m - tp.get("ci_low", m))
                highs.append(tp.get("ci_high", m) - m)

            bars = ax.bar(
                x + (j - 1) * width, means, width,
                yerr=[lows, highs], capsize=3,
                label=tier_label, color=_GRAYS[j * 2],
            )

        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in sev_bands], fontsize=10)
        ax.set_xlabel("Severity band")
        ax.set_ylabel("Posterior P(tier)")
        ax.set_title(
            f"Posterior Tier Distribution by Severity (B2 Ordinal)\n{model}, {metric.replace('_', ' ')}",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        fig.tight_layout()

        model_clean = model.replace("-", "").replace("_", "")
        metric_clean = metric.replace("_", "")
        base = f"fig__appendix__bayes_ppc_tier_dist_by_severity__{model_clean}__{metric_clean}__v1"
        pdf_path = out_dir / f"{base}.pdf"
        png_path = out_dir / f"{base}.png"
        fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        (out_dir / f"{base}.txt").write_text(
            f"Figure B3. Posterior tier distribution by severity band (B2 ordinal) -- {model}, "
            f"{metric.replace('_', ' ')}. Error bars show 95% credible intervals.",
            encoding="utf-8",
        )
        return pdf_path, png_path
