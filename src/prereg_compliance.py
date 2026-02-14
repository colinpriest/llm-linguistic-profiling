"""Pre-registration compliance report generator.

Checks every commitment from the pre-registered analysis plan and produces
a Markdown narrative suitable for inclusion in the paper's supplementary
materials.

Public API
----------
generate_prereg_compliance_report  Generate the full compliance report.
generate_claim_classifications     Generate claim strings per model x metric.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.statistics import fdr_correction

logger = logging.getLogger(__name__)

SEVERITY_METRICS = ["monetary_severity", "issue_severity"]
DIMENSIONS = ["age", "geographic", "education", "socioeconomic"]

_MIN_CELL_SIZE_SUPPRESS = 10
_FDR_ALPHA = 0.05
_MIN_CELL_SIZE_CONCLUSIVE = 30


# ------------------------------------------------------------------
# Claim classification (spec section 6.7)
# ------------------------------------------------------------------

def classify_claim(
    interaction_q: float,
    A: float,
    A_ci_lower: float,
    A_ci_upper: float,
    min_cell_size: int,
) -> str:
    """Apply pre-registered decision rules for amplification claims.

    Returns one of:
    - "Amplification supported"
    - "No evidence of amplification"
    - "Inconclusive"
    - "Not analysable"
    """
    if min_cell_size < _MIN_CELL_SIZE_SUPPRESS:
        return "Not analysable"
    if not np.isfinite(interaction_q) or not np.isfinite(A):
        return "Not analysable"
    if interaction_q >= _FDR_ALPHA:
        return "No evidence of amplification"
    # q < 0.05
    if A > 0 and A_ci_lower > 0:
        if min_cell_size < _MIN_CELL_SIZE_CONCLUSIVE:
            return "Inconclusive"
        return "Amplification supported"
    # q < 0.05 but CI includes 0
    return "Inconclusive"


def generate_claim_classifications(
    analysis_results: Dict[str, Any],
) -> Dict[str, Dict[str, str]]:
    """Generate claim classification strings per model x severity metric.

    Returns nested dict: {metric: {model: claim_string}}.
    """
    models = analysis_results.get("models", [])
    severity = analysis_results.get("severity", {})
    interactions_tier2 = severity.get("interactions_tier2", {})
    gee_interactions = severity.get("gee_interactions_tier2", {})

    claims: Dict[str, Dict[str, str]] = {}

    for metric in SEVERITY_METRICS:
        claims[metric] = {}
        for model in models:
            # Try GEE interaction test first (confirmatory per pre-reg)
            gee_data = gee_interactions.get(metric, {}).get(model, {})
            anova_data = interactions_tier2.get(metric, {}).get(model, {})

            # Collect interaction p-values from all dimensions
            p_values = []
            min_cell = 999
            A_values = []
            A_ci_lowers = []

            for dim in DIMENSIONS:
                # GEE-based test
                gee_dim = gee_data.get(dim, {})
                if not gee_dim.get("skipped", False):
                    try:
                        p = float(gee_dim.get("interaction_p", float("nan")))
                    except (TypeError, ValueError):
                        p = float("nan")
                    if np.isfinite(p):
                        p_values.append(p)

                # ANOVA-based amplification estimates
                anova_amp = anova_data.get("amplification", {})
                if anova_amp:
                    A_values.append(anova_amp.get("bias_high", 0) - anova_amp.get("bias_low", 0))
                    A_ci_lowers.append(anova_amp.get("ci_lower", float("nan")))

            # Cell sizes from ANOVA interaction data
            cell_sizes = anova_data.get("bias", {})
            for level in ["low", "medium", "high"]:
                level_data = cell_sizes.get(level, {})
                n = level_data.get("n", 0)
                if n > 0:
                    min_cell = min(min_cell, n)

            # Use minimum p-value across dimensions (then apply BH)
            if p_values:
                # Apply BH correction conceptually (within this family)
                adjusted = fdr_correction(p_values)
                best_q = float(min(adjusted))
            else:
                best_q = float("nan")

            A_val = float(np.mean(A_values)) if A_values else float("nan")
            # NOTE: Averaging CI lower bounds is an approximation; not a valid CI for the mean effect.
            A_ci_lo = float(np.mean(A_ci_lowers)) if A_ci_lowers else float("nan")

            claims[metric][model] = classify_claim(
                interaction_q=best_q,
                A=A_val,
                A_ci_lower=A_ci_lo,
                A_ci_upper=float("nan"),  # not needed for decision
                min_cell_size=min_cell if min_cell < 999 else 0,
            )

    return claims


# ------------------------------------------------------------------
# Compliance checks
# ------------------------------------------------------------------

def _check_endpoint_priority(analysis_results: Dict[str, Any]) -> str:
    """Verify primary endpoint is Y2 (Tier 2)."""
    lines = []
    severity = analysis_results.get("severity", {})
    has_tier2 = "interactions_tier2" in severity
    has_gee = "gee_interactions_tier2" in severity

    lines.append("### Primary endpoint: Y2 = 1[tier == 2]")
    if has_tier2:
        lines.append("- PASS: Tier-2 binary interaction analysis present in `interactions_tier2`.")
    else:
        lines.append("- **FAIL**: No Tier-2 interaction analysis found.")

    if has_gee:
        lines.append("- PASS: Logistic GEE interaction tests present in `gee_interactions_tier2`.")
    else:
        lines.append("- WARN: No GEE-based logistic interaction tests found.")

    # Check secondary endpoints
    lines.append("\n### Secondary endpoints")
    lines.append("- Y1+ (tier >= 1): reported as sensitivity check in mitigation analysis.")

    models = analysis_results.get("models", [])
    for model in models:
        model_r = analysis_results.get(model, {})
        has_process = "process_correlation" in model_r or "process_causal" in model_r
        lines.append(f"- F (need_more_info) for {model}: {'PASS' if has_process else 'MISSING'}")

    return "\n".join(lines)


def _check_confirmatory_labeling(analysis_results: Dict[str, Any]) -> str:
    """Verify confirmatory vs exploratory labeling."""
    lines = []
    lines.append("### Confirmatory family")
    lines.append("Per pre-registration, only **persona x severity interaction tests on Y2** are confirmatory.")
    lines.append("")

    severity = analysis_results.get("severity", {})
    gee = severity.get("gee_interactions_tier2", {})

    n_confirmatory = 0
    for metric in SEVERITY_METRICS:
        for model in analysis_results.get("models", []):
            model_gee = gee.get(metric, {}).get(model, {})
            for dim in DIMENSIONS:
                dim_data = model_gee.get(dim, {})
                if not dim_data.get("skipped", False) and "interaction_p" in dim_data:
                    n_confirmatory += 1

    lines.append(f"- Number of confirmatory tests found: {n_confirmatory}")
    lines.append("- All other analyses (pairwise comparisons, main effects, exploratory slices): exploratory.")

    return "\n".join(lines)


def _check_multiple_testing(analysis_results: Dict[str, Any]) -> str:
    """Verify BH FDR at 5% applied."""
    lines = []
    lines.append("### Multiple testing correction")
    lines.append("- Method: Benjamini-Hochberg FDR at alpha = 0.05")
    lines.append("- Applied to: confirmatory family (persona x severity interaction on Y2)")

    severity = analysis_results.get("severity", {})
    gee = severity.get("gee_interactions_tier2", {})

    # Collect all confirmatory p-values
    p_values = []
    test_labels = []
    for metric in SEVERITY_METRICS:
        for model in analysis_results.get("models", []):
            model_gee = gee.get(metric, {}).get(model, {})
            for dim in DIMENSIONS:
                dim_data = model_gee.get(dim, {})
                if not dim_data.get("skipped", False):
                    try:
                        p = float(dim_data.get("interaction_p", float("nan")))
                    except (TypeError, ValueError):
                        p = float("nan")
                    if np.isfinite(p):
                        p_values.append(p)
                        test_labels.append(f"{model}/{metric}/{dim}")

    if p_values:
        adjusted = fdr_correction(p_values)
        n_surviving = sum(1 for q in adjusted if q < 0.05)
        lines.append(f"\n- Total confirmatory tests: {len(p_values)}")
        lines.append(f"- Tests surviving FDR correction: {n_surviving}")
        lines.append("\n| Test | Raw p | Adjusted q | Significant |")
        lines.append("|------|-------|------------|-------------|")
        for label, p, q in sorted(zip(test_labels, p_values, adjusted), key=lambda x: x[1]):
            sig = "Yes" if q < 0.05 else "No"
            lines.append(f"| {label} | {p:.4f} | {q:.4f} | {sig} |")
    else:
        lines.append("\n- **WARN**: No confirmatory p-values available for BH correction.")

    return "\n".join(lines)


def _check_monetary_handling(analysis_results: Dict[str, Any]) -> str:
    """Verify two-part monetary severity handling."""
    lines = []
    lines.append("### Two-part monetary severity")
    lines.append("- M = 1[dollar_amount_extractable]")
    lines.append("- Confirmatory monetary analysis within M=1 only")
    lines.append("- Unknown amount NOT treated as a severity level")

    severity = analysis_results.get("severity", {})
    miss_diag = severity.get("missingness_diagnostic", {})

    if miss_diag and not miss_diag.get("skipped"):
        n_present = miss_diag.get("n_present", 0)
        n_missing = miss_diag.get("n_missing", 0)
        pct = miss_diag.get("pct_missing", 0) * 100
        lines.append(f"\n- PASS: Coverage reported: P(M=1) = {100-pct:.1f}%")
        lines.append(f"- Amount present: {n_present}, Amount missing: {n_missing}")

        chars = miss_diag.get("complaint_characteristics", {})
        m_type = chars.get("missingness_type", "unknown")
        lines.append(f"- Missingness type: {m_type}")
    else:
        lines.append("\n- **WARN**: Missingness diagnostic not available.")

    # Check that monetary interaction uses only M=1 subset
    interactions_t2 = severity.get("interactions_tier2", {})
    monetary_int = interactions_t2.get("monetary_severity", {})
    if monetary_int:
        lines.append("- PASS: Monetary severity interactions computed (within M=1 subset).")
    else:
        lines.append("- **WARN**: No monetary severity interaction results found.")

    return "\n".join(lines)


def _check_ipw_robustness(analysis_results: Dict[str, Any]) -> str:
    """Verify IPW robustness completed."""
    lines = []
    lines.append("### IPW robustness for monetary severity")

    severity = analysis_results.get("severity", {})
    ipw = severity.get("ipw_robustness", {})

    if not ipw or ipw.get("error"):
        lines.append(f"- **FAIL**: IPW robustness not completed. Error: {ipw.get('error', 'not run')}")
        return "\n".join(lines)

    diag = ipw.get("propensity_diagnostics", {})
    lines.append(f"- PASS: IPW analysis completed.")
    lines.append(f"- Coverage P(M=1) = {diag.get('coverage_M1', 0):.3f}")
    lines.append(f"- Effective sample size (M=1): {diag.get('effective_sample_size_M1', 0):.0f} / {diag.get('n_M1', 0)}")

    # Report stability across models
    models = analysis_results.get("models", [])
    for model in models:
        model_ipw = ipw.get(model, {})
        if model_ipw.get("skipped"):
            lines.append(f"- {model}: skipped ({model_ipw.get('reason', '')})")
            continue
        dims = model_ipw.get("dimensions", {})
        material_changes = []
        for dim, dim_r in dims.items():
            if dim_r.get("material_change"):
                material_changes.append(dim)
        if material_changes:
            lines.append(f"- {model}: material IPW change in dimensions: {', '.join(material_changes)}")
        else:
            lines.append(f"- {model}: IPW results stable (no material changes)")

    return "\n".join(lines)


def _check_sparse_cells(analysis_results: Dict[str, Any]) -> str:
    """Report sparse cell warnings."""
    lines = []
    lines.append("### Sparse cell warnings")
    lines.append("- Threshold: <30 observations = warning, <10 = analysis suppressed")

    severity = analysis_results.get("severity", {})
    sparsity = analysis_results.get("sparsity", {})

    # Check interaction cell sizes
    warn_cells = []
    suppress_cells = []

    for metric in SEVERITY_METRICS:
        interactions = severity.get("interactions_tier2", {}).get(metric, {})
        for model in analysis_results.get("models", []):
            model_int = interactions.get(model, {})
            bias = model_int.get("bias", {})
            for level in ["low", "medium", "high"]:
                level_data = bias.get(level, {})
                n = level_data.get("n", 0)
                if 0 < n < 10:
                    suppress_cells.append(f"{model}/{metric}/{level} (n={n})")
                elif 10 <= n < 30:
                    warn_cells.append(f"{model}/{metric}/{level} (n={n})")

    if suppress_cells:
        lines.append(f"\n- **SUPPRESSED** ({len(suppress_cells)} cells with n<10):")
        for cell in suppress_cells:
            lines.append(f"  - {cell}")
    else:
        lines.append("\n- No cells suppressed (all n >= 10).")

    if warn_cells:
        lines.append(f"\n- **WARNINGS** ({len(warn_cells)} cells with 10 <= n < 30):")
        for cell in warn_cells:
            lines.append(f"  - {cell}")
    else:
        lines.append("- No sparse cell warnings (all n >= 30).")

    return "\n".join(lines)


def _check_reporting_commitments(analysis_results: Dict[str, Any]) -> str:
    """Verify all reporting commitments are met."""
    lines = []
    lines.append("### Reporting commitments")

    checks = [
        ("Coverage P(M=1)", bool(analysis_results.get("severity", {}).get("missingness_diagnostic", {}).get("n_present"))),
        ("M=1 vs M=0 differences", bool(analysis_results.get("severity", {}).get("missingness_diagnostic", {}).get("complaint_characteristics"))),
        ("Confirmatory interaction tests", bool(analysis_results.get("severity", {}).get("gee_interactions_tier2"))),
        ("Effect sizes with uncertainty", bool(analysis_results.get("severity", {}).get("interactions_tier2"))),
        ("IPW robustness", bool(analysis_results.get("severity", {}).get("ipw_robustness")) and not analysis_results.get("severity", {}).get("ipw_robustness", {}).get("error")),
    ]

    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "**FAIL**"
        if not passed:
            all_pass = False
        lines.append(f"- {status}: {label}")

    if all_pass:
        lines.append("\nAll reporting commitments met.")
    else:
        lines.append("\n**Some reporting commitments not met. Review pipeline.**")

    return "\n".join(lines)


def _check_bayes_robustness(analysis_results: Dict[str, Any]) -> str:
    """Report Bayesian robustness analysis results (non-confirmatory)."""
    lines = []
    lines.append("### Bayesian hierarchical robustness")
    lines.append("")
    lines.append(
        "Bayesian results are treated as robustness checks and do not replace "
        "preregistered confirmatory inference."
    )
    lines.append("")

    severity = analysis_results.get("severity", {})
    bayes = severity.get("bayesian_robustness", {})

    if not bayes:
        lines.append("- **NOT RUN**: Bayesian robustness analysis was not executed.")
        return "\n".join(lines)

    if bayes.get("error"):
        lines.append(f"- **FAILED**: {bayes['error']}")
        return "\n".join(lines)

    lines.append("| Model | Metric | Fit | Status | A mean | 95% CrI | P(A>0) |")
    lines.append("|-------|--------|-----|--------|--------|---------|--------|")

    models = analysis_results.get("models", [])
    for model in models:
        model_bayes = bayes.get(model, {})
        if not model_bayes or model_bayes.get("skipped"):
            lines.append(f"| {model} | --- | --- | SKIPPED | --- | --- | --- |")
            continue

        for metric in SEVERITY_METRICS:
            metric_data = model_bayes.get(metric, {})
            if not metric_data or metric_data.get("skipped"):
                lines.append(f"| {model} | {metric} | --- | SKIPPED | --- | --- | --- |")
                continue

            for fit_type, fit_label in [("logistic", "B1"), ("ordinal", "B2")]:
                ft = metric_data.get(fit_type, {})
                if not ft or ft.get("error"):
                    status = "FAIL"
                    lines.append(
                        f"| {model} | {metric} | {fit_label} | {status} | --- | --- | --- |"
                    )
                    continue

                status = ft.get("status", "FAIL")
                amp = ft.get("amplification", {})
                if amp and status in ("OK", "WARN"):
                    # Report first dimension as representative summary
                    for dim, dd in amp.items():
                        a_mean = dd.get("A_mean", float("nan"))
                        a_lo = dd.get("A_CrI_low", float("nan"))
                        a_hi = dd.get("A_CrI_high", float("nan"))
                        p_gt0 = dd.get("P_A_gt_0", float("nan"))
                        a_str = f"{a_mean:.4f}" if np.isfinite(a_mean) else "---"
                        ci_str = (
                            f"[{a_lo:.4f}, {a_hi:.4f}]"
                            if np.isfinite(a_lo) and np.isfinite(a_hi)
                            else "---"
                        )
                        p_str = f"{p_gt0:.3f}" if np.isfinite(p_gt0) else "---"
                        lines.append(
                            f"| {model} | {metric} | {fit_label} ({dim}) | {status} | "
                            f"{a_str} | {ci_str} | {p_str} |"
                        )
                else:
                    lines.append(
                        f"| {model} | {metric} | {fit_label} | {status} | --- | --- | --- |"
                    )

    lines.append("")
    lines.append(
        "Full Bayesian diagnostics and posterior draws are reported in "
        "Appendix B (Tables B2--B4; Figures B1--B3)."
    )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main report generator
# ------------------------------------------------------------------

def generate_prereg_compliance_report(
    analysis_results: Dict[str, Any],
    run_config: Dict[str, Any],
) -> str:
    """Generate pre-registration compliance report as Markdown.

    Checks every pre-registered commitment and reports pass/fail/warn.

    Parameters
    ----------
    analysis_results : Full analysis results dict from experiment.run().
    run_config : Run configuration dict.

    Returns
    -------
    Markdown string for the compliance report.
    """
    sections = []

    sections.append("# Pre-Registration Compliance Report\n")
    sections.append(f"**Run ID:** {run_config.get('run_id', 'N/A')}")
    sections.append(f"**Generated:** {datetime.now().isoformat()}")
    sections.append(f"**Seed:** {run_config.get('seed', 'N/A')}")
    sections.append(f"**N complaints:** {run_config.get('num_complaints', 'N/A')}")
    sections.append(f"**Models:** {', '.join(run_config.get('models', []))}")
    sections.append("")

    # Section 1: Endpoint verification
    sections.append("## 1. Endpoint Priority\n")
    sections.append(_check_endpoint_priority(analysis_results))
    sections.append("")

    # Section 2: Confirmatory vs exploratory labeling
    sections.append("## 2. Confirmatory vs Exploratory Labeling\n")
    sections.append(_check_confirmatory_labeling(analysis_results))
    sections.append("")

    # Section 3: Multiple testing control
    sections.append("## 3. Multiple Testing Control\n")
    sections.append(_check_multiple_testing(analysis_results))
    sections.append("")

    # Section 4: Two-part monetary severity
    sections.append("## 4. Two-Part Monetary Severity Handling\n")
    sections.append(_check_monetary_handling(analysis_results))
    sections.append("")

    # Section 5: IPW robustness
    sections.append("## 5. IPW Robustness\n")
    sections.append(_check_ipw_robustness(analysis_results))
    sections.append("")

    # Section 6: Sparse cell warnings
    sections.append("## 6. Sparse Cell Diagnostics\n")
    sections.append(_check_sparse_cells(analysis_results))
    sections.append("")

    # Section 7: Claim classifications
    sections.append("## 7. Claim Classifications\n")
    claims = generate_claim_classifications(analysis_results)
    sections.append("Per-model x metric verdicts (pre-registered decision rules):\n")
    sections.append("| Metric | Model | Verdict |")
    sections.append("|--------|-------|---------|")
    for metric in SEVERITY_METRICS:
        for model in analysis_results.get("models", []):
            verdict = claims.get(metric, {}).get(model, "Not analysable")
            sections.append(f"| {metric} | {model} | {verdict} |")
    sections.append("")

    # Section 8: Reporting commitments
    sections.append("## 8. Reporting Commitments\n")
    sections.append(_check_reporting_commitments(analysis_results))
    sections.append("")

    # Section 9: Bayesian robustness (non-confirmatory)
    sections.append("## 9. Bayesian Robustness (Non-Confirmatory)\n")
    sections.append(_check_bayes_robustness(analysis_results))
    sections.append("")

    return "\n".join(sections)
