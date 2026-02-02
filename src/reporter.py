"""Self-contained HTML report generator for the LLM fairness experiment."""

from __future__ import annotations

import base64
import html
import io
import logging
import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.schemas import Complaint, ESTIMATE_DIMS, CAUSAL_DIMS, PERSONA_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

FLAG_COLORS = {
    "BOTH": "#e74c3c",       # Red
    "SIG_ONLY": "#e67e22",   # Orange
    "MAT_ONLY": "#f1c40f",   # Yellow
    "NONE": "#2ecc71",       # Green
}

FLAG_BG = {
    "BOTH": "#fdecea",
    "SIG_ONLY": "#fef5e7",
    "MAT_ONLY": "#fef9e7",
    "NONE": "#eafaf1",
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current


def _fmt(val: Any, precision: int = 4) -> str:
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if abs(val) < 0.0001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.{precision}f}"
    return str(val)


def _pct(val: Any) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(val)


def _flag_color(flag: str) -> str:
    return FLAG_COLORS.get(flag, "#95a5a6")


def _flag_bg(flag: str) -> str:
    return FLAG_BG.get(flag, "#f9f9f9")


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def _chart_persona_distribution(complaints: List[Complaint], analysis_results: Dict[str, Any]) -> str:
    """Pie chart showing estimated persona distribution across the first model."""
    models = analysis_results.get("models", [])
    if not models:
        return ""
    model = models[0]
    model_res = analysis_results.get(model, {})
    corr = model_res.get("correlation", {})
    rates = corr.get("rates_by_dimension", {})
    if not rates:
        return ""

    n_dims = len(ESTIMATE_DIMS)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    fig.suptitle("Persona Distribution (Estimated)", fontsize=14, fontweight="bold")

    for ax, dim in zip(axes.flat, ESTIMATE_DIMS):
        dim_rates = rates.get(dim, {})
        if dim_rates:
            labels = list(dim_rates.keys())
            values = list(dim_rates.values())
            colors = sns.color_palette("Set2", len(labels))
            ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
            ax.set_title(dim.capitalize())
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(dim.capitalize())

    # Hide unused axes
    for ax in axes.flat[len(ESTIMATE_DIMS):]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_base64(fig)


def _chart_tier_by_persona(analysis_results: Dict[str, Any]) -> str:
    """Stacked bar chart: tier distribution by persona dimension."""
    models = analysis_results.get("models", [])
    if not models:
        return ""
    model = models[0]
    model_res = analysis_results.get(model, {})
    corr = model_res.get("correlation", {})
    rates_dim = corr.get("rates_by_dimension", {})
    if not rates_dim:
        return ""

    n_dims = len(ESTIMATE_DIMS)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    fig.suptitle("Favorable Outcome Rate by Demographic Dimension", fontsize=14, fontweight="bold")

    for ax, dim in zip(axes.flat, ESTIMATE_DIMS):
        dim_data = rates_dim.get(dim, {})
        if dim_data:
            cats = list(dim_data.keys())
            fav_rates = [dim_data[c] for c in cats]
            unfav_rates = [1.0 - r for r in fav_rates]
            x = np.arange(len(cats))
            ax.bar(x, fav_rates, color="#2ecc71", label="Favorable (Tier >= 1)")
            ax.bar(x, unfav_rates, bottom=fav_rates, color="#e74c3c", label="Unfavorable (Tier 0)")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=45, ha="right")
            ax.set_title(dim.capitalize())
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Proportion")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dim.capitalize())

    # Hide unused axes
    for ax in axes.flat[len(ESTIMATE_DIMS):]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_base64(fig)


def _chart_air_heatmap(analysis_results: Dict[str, Any], model: str) -> str:
    """Heatmap of Adverse Impact Ratios by persona for a given model."""
    model_res = analysis_results.get(model, {})
    air_data = _safe_get(model_res, "correlation", "air_by_persona", default={})
    if not air_data:
        return ""

    personas = sorted(air_data.keys())
    if len(personas) < 2:
        return ""

    # Parse persona labels into dimension values
    rows = []
    for p in personas:
        parts = p.split("_")
        if len(parts) >= 4:
            rows.append({
                "persona": p,
                "age": parts[0],
                "geo": parts[1],
                "edu": parts[2],
                "ses": parts[3],
                "air": air_data[p].get("air", 1.0),
            })

    if not rows:
        return ""

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"Adverse Impact Ratio Heatmap ({model})", fontsize=14, fontweight="bold")

    # Create a matrix: rows = age+geo combos, cols = edu+ses combos
    df["row_key"] = df["age"] + "_" + df["geo"]
    df["col_key"] = df["edu"] + "_" + df["ses"]
    pivot = df.pivot_table(index="row_key", columns="col_key", values="air", aggfunc="first")

    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap=cmap, center=0.80,
        vmin=0.5, vmax=1.2, ax=ax, linewidths=0.5,
        cbar_kws={"label": "AIR (< 0.80 = material)"},
    )
    ax.set_xlabel("Education + SES")
    ax.set_ylabel("Age + Geography")
    ax.set_title("")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_base64(fig)


def _chart_causal_forest(analysis_results: Dict[str, Any], model: str) -> str:
    """Forest plot of causal effect sizes by persona."""
    model_res = analysis_results.get(model, {})
    effects = _safe_get(model_res, "causal", "effect_sizes", default={})
    if not effects:
        return ""

    # Sort by effect rate
    sorted_items = sorted(effects.items(), key=lambda x: x[1].get("rate", 0))
    if len(sorted_items) > 40:
        # Show top 20 and bottom 20
        sorted_items = sorted_items[:20] + sorted_items[-20:]

    labels = [item[0] for item in sorted_items]
    rates = [item[1].get("rate", 0) for item in sorted_items]
    ci_lo = [item[1].get("ci_lower", 0) for item in sorted_items]
    ci_hi = [item[1].get("ci_upper", 0) for item in sorted_items]
    errors_lo = [r - lo for r, lo in zip(rates, ci_lo)]
    errors_hi = [hi - r for r, hi in zip(rates, ci_hi)]

    fig_height = max(4, len(labels) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = np.arange(len(labels))

    ax.errorbar(
        rates, y_pos, xerr=[errors_lo, errors_hi],
        fmt="o", color="#2c3e50", ecolor="#bdc3c7", capsize=3, markersize=5,
    )
    overall_mean = np.mean(rates) if rates else 0
    ax.axvline(overall_mean, color="#e74c3c", linestyle="--", linewidth=1, label=f"Mean = {overall_mean:.3f}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Favorable Outcome Rate")
    ax.set_title(f"Causal Effect Sizes by Persona ({model})", fontweight="bold")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_mitigation_paired_bar(analysis_results: Dict[str, Any], model: str) -> str:
    """Paired bar chart showing pre/post neutralization AIR."""
    model_res = analysis_results.get(model, {})
    mitigation = _safe_get(model_res, "mitigation", "mitigation_by_persona", default={})
    if not mitigation:
        return ""

    # Show personas with the most bias
    sorted_personas = sorted(mitigation.items(), key=lambda x: x[1].get("orig_air", 1.0))
    show = sorted_personas[:20] if len(sorted_personas) > 20 else sorted_personas

    labels = [item[0] for item in show]
    orig_air = [item[1].get("orig_air", 1.0) for item in show]
    neut_air = [item[1].get("neut_air", 1.0) for item in show]

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.4)))
    y = np.arange(len(labels))
    bar_h = 0.35

    ax.barh(y - bar_h / 2, orig_air, bar_h, label="Original", color="#e74c3c", alpha=0.8)
    ax.barh(y + bar_h / 2, neut_air, bar_h, label="Neutralized", color="#2ecc71", alpha=0.8)
    ax.axvline(0.80, color="#f39c12", linestyle="--", linewidth=1.5, label="Materiality (0.80)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Adverse Impact Ratio")
    ax.set_title(f"Pre/Post Neutralization AIR ({model})", fontweight="bold")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_info_request_rate(analysis_results: Dict[str, Any], model: str) -> str:
    """Bar chart of information request rates by demographic dimension."""
    model_res = analysis_results.get(model, {})
    proc_corr = model_res.get("process_correlation", {})
    info_rates = proc_corr.get("info_rates_by_dimension", {})
    if not info_rates:
        return ""

    n_dims = len(ESTIMATE_DIMS)
    n_cols = 3
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    fig.suptitle(f"Information Request Rate by Dimension ({model})", fontsize=14, fontweight="bold")
    palette = sns.color_palette("Set2", 6)

    for ax, dim in zip(axes.flat, ESTIMATE_DIMS):
        rates = info_rates.get(dim, {})
        if rates:
            cats = list(rates.keys())
            vals = [rates[c] for c in cats]
            colors = [palette[i % len(palette)] for i in range(len(cats))]
            bars = ax.bar(cats, vals, color=colors)
            ax.set_title(dim.capitalize())
            ax.set_ylabel("Info Request Rate")
            ax.set_ylim(0, max(vals) * 1.3 if vals and max(vals) > 0 else 1.0)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dim.capitalize())

    # Hide unused axes
    for ax in axes.flat[len(ESTIMATE_DIMS):]:
        ax.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _fig_to_base64(fig)


def _chart_process_air_heatmap(analysis_results: Dict[str, Any], model: str) -> str:
    """Heatmap for process-discrimination AIR (information burden)."""
    model_res = analysis_results.get(model, {})
    proc_corr = model_res.get("process_correlation", {})
    air_data = proc_corr.get("process_air_by_persona", {})
    if not air_data:
        return ""

    rows = []
    for p, vals in air_data.items():
        parts = p.split("_")
        if len(parts) >= 4:
            rows.append({
                "row_key": f"{parts[0]}_{parts[1]}",
                "col_key": f"{parts[2]}_{parts[3]}",
                "air": vals.get("air", 1.0),
            })
    if not rows:
        return ""

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="row_key", columns="col_key", values="air", aggfunc="first")

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.diverging_palette(130, 10, as_cmap=True)  # reversed: green = low burden
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap=cmap, center=1.25,
        vmin=0.8, vmax=2.0, ax=ax, linewidths=0.5,
        cbar_kws={"label": "Process AIR (> 1.25 = material)"},
    )
    ax.set_xlabel("Education + SES")
    ax.set_ylabel("Age + Geography")
    ax.set_title(f"Process Discrimination AIR ({model})", fontweight="bold")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_feature_violins(text_features_df: Optional[pd.DataFrame]) -> str:
    """Violin plots of key text features by variant."""
    if text_features_df is None or text_features_df.empty:
        return ""

    features = ["flesch_reading_ease", "vader_compound", "word_count", "type_token_ratio"]
    available = [f for f in features if f in text_features_df.columns]
    if not available:
        return ""

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Text Feature Distributions by Variant", fontsize=14, fontweight="bold")

    for ax, feat in zip(axes, available):
        plot_df = text_features_df[["variant", feat]].dropna()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(feat)
            continue
        try:
            sns.violinplot(data=plot_df, x="variant", y=feat, ax=ax, inner="box",
                           palette="Set2", cut=0)
        except Exception:
            sns.boxplot(data=plot_df, x="variant", y=feat, ax=ax, palette="Set2")
        ax.set_title(feat.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_base64(fig)


def _chart_feature_importance(analysis_results: Dict[str, Any], model: str) -> str:
    """Bar chart of logistic regression coefficients (feature importance)."""
    model_res = analysis_results.get(model, {})
    logistic = _safe_get(model_res, "correlation", "logistic_favorable", default={})
    if not logistic or "error" in logistic:
        return ""

    coeffs = logistic.get("coefficients", {})
    pvals = logistic.get("p_values", {})
    if not coeffs:
        return ""

    # Filter out constant
    items = [(k, v) for k, v in coeffs.items() if k != "const"]
    if not items:
        return ""

    items.sort(key=lambda x: abs(x[1]), reverse=True)
    labels = [it[0] for it in items]
    values = [it[1] for it in items]
    colors = ["#e74c3c" if pvals.get(l, 1.0) < 0.05 else "#bdc3c7" for l in labels]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.35)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Logistic Regression Coefficients ({model})", fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="p < 0.05"),
        Patch(facecolor="#bdc3c7", label="p >= 0.05"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    ax.invert_yaxis()

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_topic_persona_heatmap(text_features_df: Optional[pd.DataFrame]) -> str:
    """Heatmap of dominant topic distribution across persona variants."""
    if text_features_df is None or text_features_df.empty:
        return ""
    if "dominant_topic" not in text_features_df.columns or "variant" not in text_features_df.columns:
        return ""

    df = text_features_df[text_features_df["variant"] != "original"].copy()
    if df.empty:
        df = text_features_df.copy()
    if df.empty:
        return ""

    ct = pd.crosstab(df["variant"], df["dominant_topic"], normalize="index")
    if ct.empty or ct.shape[0] < 2:
        return ""

    # Limit to manageable size
    if ct.shape[0] > 30:
        ct = ct.iloc[:30]

    fig, ax = plt.subplots(figsize=(10, max(4, ct.shape[0] * 0.3)))
    sns.heatmap(ct, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f", linewidths=0.5)
    ax.set_title("Topic Distribution by Variant", fontweight="bold")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Variant")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_model_comparison(analysis_results: Dict[str, Any]) -> str:
    """Grouped bar chart comparing key metrics across models."""
    models = analysis_results.get("models", [])
    if len(models) < 2:
        return ""

    metrics_labels = []
    metrics_values = {m: [] for m in models}

    for model in models:
        mr = analysis_results.get(model, {})

        # Overall chi2 Cramer's V
        cv = _safe_get(mr, "correlation", "overall_chi2", "cramers_v", default=None)
        if cv is not None and "Overall Cramer's V" not in metrics_labels:
            metrics_labels.append("Overall Cramer's V")
        metrics_values[model].append(cv if cv is not None else 0)

        # Pseudo R2
        r2 = _safe_get(mr, "correlation", "logistic_favorable", "pseudo_r2", default=None)
        if r2 is not None and "Pseudo R2" not in metrics_labels:
            metrics_labels.append("Pseudo R2")
        metrics_values[model].append(r2 if r2 is not None else 0)

        # Variance reduction
        vr = _safe_get(mr, "mitigation", "variance_reduction", default=None)
        if vr is not None and "Variance Reduction" not in metrics_labels:
            metrics_labels.append("Variance Reduction")
        metrics_values[model].append(vr if vr is not None else 0)

        # Process variance reduction
        pvr = _safe_get(mr, "process_mitigation", "variance_reduction", default=None)
        if pvr is not None and "Process Var. Reduction" not in metrics_labels:
            metrics_labels.append("Process Var. Reduction")
        metrics_values[model].append(pvr if pvr is not None else 0)

    # Make sure all lists have the right length
    expected_len = len(metrics_labels)
    if expected_len == 0:
        return ""
    for m in models:
        while len(metrics_values[m]) < expected_len:
            metrics_values[m].append(0)
        metrics_values[m] = metrics_values[m][:expected_len]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_labels))
    width = 0.8 / len(models)
    palette = sns.color_palette("Set1", len(models))

    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, metrics_values[model], width, label=model, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend()

    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _compute_flag(p_adj: Optional[float], air: Optional[float], threshold: float = 0.80) -> str:
    """Compute interpretation flag from p-value and AIR."""
    if p_adj is None or air is None:
        return "NONE"
    sig = p_adj < 0.05
    mat = air < threshold
    if sig and mat:
        return "BOTH"
    if sig:
        return "SIG_ONLY"
    if mat:
        return "MAT_ONLY"
    return "NONE"


def _get_dimension_flags(model_res: Dict[str, Any], threshold: float = 0.80) -> List[Dict[str, Any]]:
    """Compute per-dimension summary flags."""
    corr = model_res.get("correlation", {})
    air_data = corr.get("air_by_persona", {})
    rows = []

    for dim in ESTIMATE_DIMS:
        chi2_info = corr.get(f"{dim}_chi2", {})
        p_val = chi2_info.get("p_value")
        cramers_v = chi2_info.get("cramers_v")

        # Find min AIR for this dimension
        min_air = 1.0
        for persona, data in air_data.items():
            parts = persona.split("_")
            dim_idx = {"age": 0, "geographic": 1, "education": 2, "socioeconomic": 3}.get(dim)
            if dim_idx is not None and dim_idx < len(parts):
                air_val = data.get("air", 1.0)
                if air_val < min_air:
                    min_air = air_val

        flag = _compute_flag(p_val, min_air, threshold)
        rows.append({
            "dimension": dim.capitalize(),
            "p_value": p_val,
            "cramers_v": cramers_v,
            "min_air": min_air,
            "flag": flag,
        })

    return rows


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

def _build_html(
    body_content: str,
    title: str = "LLM Fairness Experiment Report",
) -> str:
    """Wrap body content in the full HTML document with inline CSS and JS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
/* Reset */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #2c3e50;
    background: #f8f9fa;
    padding: 20px;
    max-width: 1400px;
    margin: 0 auto;
}}

h1 {{ font-size: 1.8rem; margin-bottom: 10px; color: #2c3e50; }}
h2 {{ font-size: 1.4rem; margin: 20px 0 10px; color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
h3 {{ font-size: 1.15rem; margin: 15px 0 8px; color: #2c3e50; }}
h4 {{ font-size: 1.0rem; margin: 10px 0 6px; color: #555; }}

.header {{
    background: linear-gradient(135deg, #2c3e50, #3498db);
    color: white;
    padding: 25px 30px;
    border-radius: 10px;
    margin-bottom: 20px;
}}
.header h1 {{ color: white; border: none; font-size: 2rem; }}
.header .subtitle {{ opacity: 0.85; font-size: 0.95rem; margin-top: 5px; }}

/* Tab navigation */
.tab-nav {{
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    background: #ecf0f1;
    border-radius: 8px 8px 0 0;
    padding: 4px 4px 0;
    margin-top: 10px;
}}
.tab-btn {{
    padding: 10px 18px;
    border: none;
    background: transparent;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    color: #7f8c8d;
    border-radius: 6px 6px 0 0;
    transition: all 0.2s;
    white-space: nowrap;
}}
.tab-btn:hover {{ background: #dfe6e9; color: #2c3e50; }}
.tab-btn.active {{
    background: white;
    color: #3498db;
    font-weight: 600;
    box-shadow: 0 -2px 0 #3498db inset;
}}

.tab-content {{ display: none; background: white; padding: 25px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
.tab-content.active {{ display: block; }}

/* Collapsible sections */
.collapsible {{
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    margin: 12px 0;
    overflow: hidden;
}}
.collapsible-header {{
    padding: 12px 16px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    color: #2c3e50;
    user-select: none;
    transition: background 0.2s;
}}
.collapsible-header:hover {{ background: #ecf0f1; }}
.collapsible-header::after {{ content: "\\25BC"; font-size: 0.8rem; transition: transform 0.2s; }}
.collapsible.collapsed .collapsible-header::after {{ transform: rotate(-90deg); }}
.collapsible-body {{ padding: 0 16px 16px; }}
.collapsible.collapsed .collapsible-body {{ display: none; }}

/* Tables */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 0.9rem;
}}
th, td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #ecf0f1;
}}
th {{
    background: #f8f9fa;
    font-weight: 600;
    color: #2c3e50;
    cursor: pointer;
    white-space: nowrap;
    position: sticky;
    top: 0;
}}
th:hover {{ background: #dfe6e9; }}
th .sort-arrow {{ font-size: 0.7rem; margin-left: 4px; opacity: 0.5; }}
tr:hover {{ background: #f8f9fa; }}

/* Traffic light badges */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
}}
.badge-BOTH {{ background: {FLAG_COLORS["BOTH"]}; }}
.badge-SIG_ONLY {{ background: {FLAG_COLORS["SIG_ONLY"]}; }}
.badge-MAT_ONLY {{ background: {FLAG_COLORS["MAT_ONLY"]}; }}
.badge-NONE {{ background: {FLAG_COLORS["NONE"]}; }}

.flag-row-BOTH {{ background: {FLAG_BG["BOTH"]} !important; }}
.flag-row-SIG_ONLY {{ background: {FLAG_BG["SIG_ONLY"]} !important; }}
.flag-row-MAT_ONLY {{ background: {FLAG_BG["MAT_ONLY"]} !important; }}
.flag-row-NONE {{ background: {FLAG_BG["NONE"]} !important; }}

/* Metric cards */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 15px 0;
}}
.metric-card {{
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
}}
.metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: #3498db; }}
.metric-card .label {{ font-size: 0.8rem; color: #7f8c8d; margin-top: 4px; }}

/* Executive summary section cards */
.exec-card {{
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 24px 28px;
    margin: 18px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.exec-card h3 {{
    font-size: 1.2rem;
    color: #2c3e50;
    margin: 0 0 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #3498db;
}}
.exec-card p {{ margin: 8px 0; line-height: 1.7; }}
.exec-card ul, .exec-card ol {{ margin: 8px 0 8px 1.5em; line-height: 1.7; }}
.exec-card li {{ margin: 4px 0; }}
.exec-card strong {{ color: #2c3e50; }}
.exec-card em {{ color: #555; }}
.exec-card blockquote {{
    border-left: 3px solid #3498db;
    margin: 10px 0;
    padding: 6px 14px;
    background: #f8f9fa;
    border-radius: 0 4px 4px 0;
}}

/* Charts */
.chart-container {{ text-align: center; margin: 15px 0; }}
.chart-container img {{ max-width: 100%; height: auto; border-radius: 6px; }}

/* Utilities */
.muted {{ color: #95a5a6; font-size: 0.85rem; }}
.note {{ background: #eaf6ff; border-left: 4px solid #3498db; padding: 10px 15px; margin: 10px 0; border-radius: 0 4px 4px 0; font-size: 0.9rem; }}
.warning {{ background: #fef5e7; border-left: 4px solid #f39c12; padding: 10px 15px; margin: 10px 0; border-radius: 0 4px 4px 0; font-size: 0.9rem; }}

.table-scroll {{ overflow-x: auto; max-height: 600px; overflow-y: auto; }}

/* Responsive */
@media (max-width: 768px) {{
    body {{ padding: 10px; }}
    .tab-btn {{ padding: 8px 12px; font-size: 0.8rem; }}
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .header {{ padding: 15px 20px; }}
    .header h1 {{ font-size: 1.4rem; }}
}}

/* Print */
@media print {{
    body {{ background: white; padding: 0; max-width: 100%; }}
    .tab-nav {{ display: none; }}
    .tab-content {{ display: block !important; page-break-inside: avoid; box-shadow: none; border: 1px solid #ccc; margin-bottom: 20px; }}
    .collapsible-body {{ display: block !important; }}
    .collapsible.collapsed .collapsible-body {{ display: block !important; }}
    .header {{ background: #2c3e50 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .badge {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .flag-row-BOTH, .flag-row-SIG_ONLY, .flag-row-MAT_ONLY, .flag-row-NONE {{
        -webkit-print-color-adjust: exact; print-color-adjust: exact;
    }}
}}
</style>
</head>
<body>

{body_content}

<script>
// Tab switching
document.querySelectorAll('.tab-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
        var tabGroup = this.closest('.tab-group');
        tabGroup.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
        tabGroup.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
        this.classList.add('active');
        var target = document.getElementById(this.getAttribute('data-tab'));
        if (target) target.classList.add('active');
    }});
}});

// Collapsible sections
document.querySelectorAll('.collapsible-header').forEach(function(header) {{
    header.addEventListener('click', function() {{
        this.parentElement.classList.toggle('collapsed');
    }});
}});

// Sortable tables
document.querySelectorAll('table.sortable').forEach(function(table) {{
    var headers = table.querySelectorAll('th');
    headers.forEach(function(th, colIdx) {{
        th.addEventListener('click', function() {{
            var tbody = table.querySelector('tbody');
            if (!tbody) return;
            var rows = Array.from(tbody.querySelectorAll('tr'));
            var asc = th.getAttribute('data-sort-dir') !== 'asc';
            th.setAttribute('data-sort-dir', asc ? 'asc' : 'desc');

            // Reset arrows
            headers.forEach(function(h) {{
                var arrow = h.querySelector('.sort-arrow');
                if (arrow) arrow.textContent = '\\u2195';
            }});
            var arrow = th.querySelector('.sort-arrow');
            if (arrow) arrow.textContent = asc ? '\\u2191' : '\\u2193';

            rows.sort(function(a, b) {{
                var aVal = a.cells[colIdx] ? a.cells[colIdx].getAttribute('data-sort-value') || a.cells[colIdx].textContent : '';
                var bVal = b.cells[colIdx] ? b.cells[colIdx].getAttribute('data-sort-value') || b.cells[colIdx].textContent : '';
                var aNum = parseFloat(aVal);
                var bNum = parseFloat(bVal);
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return asc ? aNum - bNum : bNum - aNum;
                }}
                return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});
            rows.forEach(function(row) {{ tbody.appendChild(row); }});
        }});
    }});
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Tab content builders
# ---------------------------------------------------------------------------

def _md_to_html(md: str) -> str:
    """Convert a markdown string to HTML.

    Handles headings, bold, italic, bullet/numbered lists, blockquotes, and
    paragraphs.  Does *not* depend on any external library.
    """

    def _inline(text: str) -> str:
        """Convert inline markdown (bold, italic, links, code) within already-escaped HTML."""
        # Bold: **text** or __text__
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
        # Italic: *text* or _text_  (but not inside words for _)
        text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<em>\1</em>", text)
        text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", text)
        # Inline code: `code`
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
        # Links: [text](url)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        return text

    out: List[str] = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Blank line
        if not stripped:
            i += 1
            continue

        # Headings: # ... ######
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            level = len(m.group(1))
            # Render section headings one level down (h4-h6) since h3 is used for card title
            tag = f"h{min(level + 1, 6)}"
            out.append(f"<{tag}>{_inline(html.escape(m.group(2)))}</{tag}>")
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            bq_lines: List[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                bq_lines.append(html.escape(lines[i].strip().lstrip(">").strip()))
                i += 1
            out.append(f"<blockquote><p>{_inline(' '.join(bq_lines))}</p></blockquote>")
            continue

        # Unordered list (- or * or •)
        if re.match(r"^[-*•]\s", stripped):
            out.append("<ul>")
            while i < len(lines) and re.match(r"^\s*[-*•]\s", lines[i]):
                item = re.sub(r"^\s*[-*•]\s+", "", lines[i]).strip()
                out.append(f"<li>{_inline(html.escape(item))}</li>")
                i += 1
            out.append("</ul>")
            continue

        # Ordered list (1. 2. etc.)
        if re.match(r"^\d+[.)]\s", stripped):
            out.append("<ol>")
            while i < len(lines) and re.match(r"^\s*\d+[.)]\s", lines[i]):
                item = re.sub(r"^\s*\d+[.)]\s+", "", lines[i]).strip()
                out.append(f"<li>{_inline(html.escape(item))}</li>")
                i += 1
            out.append("</ol>")
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            out.append("<hr>")
            i += 1
            continue

        # Paragraph: collect consecutive non-blank, non-special lines
        para_lines: List[str] = []
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                break
            if re.match(r"^#{1,6}\s", s) or re.match(r"^[-*•]\s", s) or re.match(r"^\d+[.)]\s", s) or s.startswith(">") or re.match(r"^[-*_]{3,}\s*$", s):
                break
            para_lines.append(html.escape(s))
            i += 1
        if para_lines:
            out.append(f"<p>{_inline(' '.join(para_lines))}</p>")

    return "\n".join(out)


def _build_tab_business_executive_summary(
    analysis_results: Dict[str, Any],
) -> str:
    """Build the Business Executive Summary tab with each section in its own card."""
    parts: List[str] = []
    parts.append("<h2>Executive Summary</h2>")

    exec_sections: Optional[Dict[str, str]] = analysis_results.get("exec_sections")

    if exec_sections:
        section_meta = [
            ("linguistic_differences", "Linguistic Differences"),
            ("ai_decision_making", "AI Decision-Making"),
            ("correlation_vs_cause", "Correlation vs. Cause"),
            ("severity_stakes", "Severity &amp; Stakes"),
            ("language_neutralisation", "Language Neutralisation"),
            ("bias_mitigation", "Bias Mitigation"),
            ("conclusion", "Conclusion"),
        ]
        for key, title in section_meta:
            md_text = exec_sections.get(key)
            if not md_text:
                continue
            body_html = _md_to_html(md_text)
            parts.append(f'<div class="exec-card">')
            parts.append(f"<h3>{title}</h3>")
            parts.append(body_html)
            parts.append("</div>")
    elif analysis_results.get("business_executive_summary"):
        # Fallback: single combined markdown blob in one card
        md_text = analysis_results["business_executive_summary"]
        parts.append('<div class="exec-card">')
        parts.append(_md_to_html(md_text))
        parts.append("</div>")
    else:
        parts.append('<p class="muted">Business executive summary not available. '
                     "Run the experiment with GPT-5.2 commentary enabled to generate this section.</p>")

    return "\n".join(parts)


def _build_tab_executive_summary(
    analysis_results: Dict[str, Any],
    complaints: List[Complaint],
    models: List[str],
    num_complaints: int,
    seed: int,
    materiality_threshold: float,
    timestamp: str,
) -> str:
    """Build the Technical Summary tab HTML."""
    parts: List[str] = []

    parts.append("<h2>Technical Summary</h2>")

    # Experiment overview metrics
    parts.append('<div class="metrics-grid">')
    parts.append(f'<div class="metric-card"><div class="value">{num_complaints}</div><div class="label">Complaints Analyzed</div></div>')
    parts.append(f'<div class="metric-card"><div class="value">{len(models)}</div><div class="label">Models Tested</div></div>')
    parts.append(f'<div class="metric-card"><div class="value">{len(PERSONA_LABELS)}</div><div class="label">Persona Variants</div></div>')
    parts.append(f'<div class="metric-card"><div class="value">{materiality_threshold:.2f}</div><div class="label">AIR Threshold</div></div>')
    parts.append("</div>")

    # Per-model key findings
    for model in models:
        model_res = analysis_results.get(model, {})
        if not model_res:
            continue

        parts.append(f"<h3>Model: {html.escape(model)}</h3>")

        # Dimension flags table
        dim_flags = _get_dimension_flags(model_res, materiality_threshold)

        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Disparate Impact Assessment</div>')
        parts.append('<div class="collapsible-body">')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Dimension <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>p-value <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Cramer\'s V <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Min AIR <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Assessment <span class="sort-arrow">&#8597;</span></th>')
        parts.append("</tr></thead><tbody>")

        for row in dim_flags:
            flag = row["flag"]
            parts.append(f'<tr class="flag-row-{flag}">')
            parts.append(f'<td>{row["dimension"]}</td>')
            parts.append(f'<td data-sort-value="{row["p_value"] if row["p_value"] is not None else 999}">{_fmt(row["p_value"])}</td>')
            parts.append(f'<td data-sort-value="{row["cramers_v"] if row["cramers_v"] is not None else 0}">{_fmt(row["cramers_v"])}</td>')
            parts.append(f'<td data-sort-value="{row["min_air"]}">{_fmt(row["min_air"])}</td>')
            parts.append(f'<td><span class="badge badge-{flag}">{flag}</span></td>')
            parts.append("</tr>")

        parts.append("</tbody></table>")
        parts.append("</div></div>")

        # Mitigation effectiveness
        mit = model_res.get("mitigation", {})
        var_red = mit.get("variance_reduction")
        if var_red is not None:
            parts.append('<div class="collapsible">')
            parts.append('<div class="collapsible-header">Neutralization Effectiveness</div>')
            parts.append('<div class="collapsible-body">')
            parts.append('<div class="metrics-grid">')
            parts.append(f'<div class="metric-card"><div class="value">{_pct(var_red)}</div><div class="label">Outcome Variance Reduction</div></div>')
            proc_vr = _safe_get(model_res, "process_mitigation", "variance_reduction")
            if proc_vr is not None:
                parts.append(f'<div class="metric-card"><div class="value">{_pct(proc_vr)}</div><div class="label">Process Variance Reduction</div></div>')
            parts.append("</div></div></div>")

    # GPT-5.2 Commentary sections
    # Step 3 cross-model commentary
    step3_commentary = analysis_results.get("step3_commentary")
    if step3_commentary:
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">GPT-5.2 Step 3 Cross-Model Commentary</div>')
        parts.append('<div class="collapsible-body">')
        parts.append(f'<p>{html.escape(step3_commentary)}</p>')
        parts.append("</div></div>")

    # Per-model GPT commentary (tier bias + process bias)
    for model in models:
        model_res = analysis_results.get(model, {})
        gpt_commentary = model_res.get("gpt_commentary")
        if gpt_commentary:
            parts.append(f'<div class="collapsible">')
            parts.append(f'<div class="collapsible-header">GPT-5.2 Commentary: {html.escape(model)} (Tier Bias &amp; Process Bias)</div>')
            parts.append('<div class="collapsible-body">')
            # Split paragraphs for display
            for para in gpt_commentary.split("\n\n"):
                para = para.strip()
                if para:
                    parts.append(f'<p>{html.escape(para)}</p>')
            if not gpt_commentary.strip():
                parts.append(f'<p>{html.escape(gpt_commentary)}</p>')
            parts.append("</div></div>")

    # Color key
    parts.append('<div class="note">')
    parts.append("<strong>Traffic Light Key:</strong> ")
    parts.append('<span class="badge badge-BOTH">BOTH</span> = statistically significant AND materially adverse; ')
    parts.append('<span class="badge badge-SIG_ONLY">SIG_ONLY</span> = significant only; ')
    parts.append('<span class="badge badge-MAT_ONLY">MAT_ONLY</span> = material only; ')
    parts.append('<span class="badge badge-NONE">NONE</span> = neither')
    parts.append("</div>")

    parts.append(f'<p class="muted">Generated: {html.escape(timestamp)} | Seed: {seed}</p>')

    return "\n".join(parts)


def _build_tab_disparate_impact(
    analysis_results: Dict[str, Any],
    complaints: List[Complaint],
    models: List[str],
    materiality_threshold: float,
    text_features_df: Optional[pd.DataFrame],
) -> str:
    """Build the Disparate Impact tab HTML."""
    parts: List[str] = []
    parts.append("<h2>Disparate Impact Analysis</h2>")

    for model in models:
        model_res = analysis_results.get(model, {})
        if not model_res:
            continue
        parts.append(f"<h3>Model: {html.escape(model)}</h3>")

        # --- Correlation Analysis ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Correlation Analysis (Outcome)</div>')
        parts.append('<div class="collapsible-body">')

        corr = model_res.get("correlation", {})
        overall_chi2 = corr.get("overall_chi2", {})
        if overall_chi2:
            parts.append('<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>')
            parts.append(f'<tr><td>Chi-squared statistic</td><td>{_fmt(overall_chi2.get("chi2"))}</td></tr>')
            parts.append(f'<tr><td>p-value</td><td>{_fmt(overall_chi2.get("p_value"))}</td></tr>')
            parts.append(f'<tr><td>Degrees of freedom</td><td>{overall_chi2.get("dof", "N/A")}</td></tr>')
            parts.append(f'<tr><td>Cramer\'s V</td><td>{_fmt(overall_chi2.get("cramers_v"))}</td></tr>')
            parts.append("</tbody></table>")

        # Per-dimension chi2
        parts.append("<h4>Per-Dimension Tests</h4>")
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Dimension <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Chi2 <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>p-value <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Cramer\'s V <span class="sort-arrow">&#8597;</span></th>')
        parts.append("</tr></thead><tbody>")
        for dim in ESTIMATE_DIMS:
            d = corr.get(f"{dim}_chi2", {})
            parts.append(f'<tr><td>{dim.capitalize()}</td>')
            parts.append(f'<td data-sort-value="{d.get("chi2", 0)}">{_fmt(d.get("chi2"))}</td>')
            parts.append(f'<td data-sort-value="{d.get("p_value", 1)}">{_fmt(d.get("p_value"))}</td>')
            parts.append(f'<td data-sort-value="{d.get("cramers_v", 0)}">{_fmt(d.get("cramers_v"))}</td></tr>')
        parts.append("</tbody></table>")

        # Logistic regression
        logistic = corr.get("logistic_favorable", {})
        if logistic and "error" not in logistic:
            parts.append("<h4>Logistic Regression (Favorable Outcome)</h4>")
            parts.append(f'<p class="muted">Pseudo R2 = {_fmt(logistic.get("pseudo_r2"))} | AIC = {_fmt(logistic.get("aic"))}</p>')
            coeffs = logistic.get("coefficients", {})
            pvals = logistic.get("p_values", {})
            ors = logistic.get("odds_ratios", {})
            if coeffs:
                parts.append('<table class="sortable"><thead><tr>')
                parts.append('<th>Predictor <span class="sort-arrow">&#8597;</span></th>')
                parts.append('<th>Coefficient <span class="sort-arrow">&#8597;</span></th>')
                parts.append('<th>p-value <span class="sort-arrow">&#8597;</span></th>')
                parts.append('<th>Odds Ratio <span class="sort-arrow">&#8597;</span></th>')
                parts.append("</tr></thead><tbody>")
                for k in coeffs:
                    p = pvals.get(k, 1.0)
                    sig_class = ' style="font-weight:bold;"' if p is not None and p < 0.05 else ""
                    parts.append(f'<tr{sig_class}>')
                    parts.append(f'<td>{html.escape(str(k))}</td>')
                    parts.append(f'<td data-sort-value="{coeffs[k]}">{_fmt(coeffs[k])}</td>')
                    parts.append(f'<td data-sort-value="{p}">{_fmt(p)}</td>')
                    parts.append(f'<td data-sort-value="{ors.get(k, 0)}">{_fmt(ors.get(k))}</td>')
                    parts.append("</tr>")
                parts.append("</tbody></table>")

        # Tier distribution chart
        tier_chart = _chart_tier_by_persona(analysis_results)
        if tier_chart:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{tier_chart}" alt="Tier by persona dimension"></div>')

        # Feature importance chart
        fi_chart = _chart_feature_importance(analysis_results, model)
        if fi_chart:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{fi_chart}" alt="Feature importance"></div>')

        parts.append("</div></div>")

        # --- AIR Heatmap ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Adverse Impact Ratio Analysis</div>')
        parts.append('<div class="collapsible-body">')

        air_chart = _chart_air_heatmap(analysis_results, model)
        if air_chart:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{air_chart}" alt="AIR heatmap"></div>')

        # AIR table
        air_data = _safe_get(model_res, "correlation", "air_by_persona", default={})
        if air_data:
            parts.append('<div class="table-scroll">')
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Persona <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Rate <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>AIR <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Material <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")
            sorted_air = sorted(air_data.items(), key=lambda x: x[1].get("air", 1.0))
            for persona, data in sorted_air:
                is_mat = data.get("material", False)
                row_class = ' class="flag-row-BOTH"' if is_mat else ""
                parts.append(f"<tr{row_class}>")
                parts.append(f"<td>{html.escape(persona)}</td>")
                parts.append(f'<td data-sort-value="{data.get("rate", 0)}">{_fmt(data.get("rate"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("air", 1)}">{_fmt(data.get("air"))}</td>')
                parts.append(f'<td data-sort-value="{1 if is_mat else 0}">{"Yes" if is_mat else "No"}</td>')
                parts.append("</tr>")
            parts.append("</tbody></table></div>")

        parts.append("</div></div>")

        # --- Causal Analysis ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Causal Analysis (Matched-Guise)</div>')
        parts.append('<div class="collapsible-body">')

        causal = model_res.get("causal", {})
        friedman = causal.get("friedman", {})
        if friedman:
            parts.append(f'<p>Friedman test: statistic = {_fmt(friedman.get("statistic"))}, p = {_fmt(friedman.get("p_value"))}</p>')

        forest_chart = _chart_causal_forest(analysis_results, model)
        if forest_chart:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{forest_chart}" alt="Causal effect forest plot"></div>')

        # Effect sizes table
        effects = causal.get("effect_sizes", {})
        if effects:
            parts.append('<div class="table-scroll">')
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Persona <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Rate <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>CI Lower <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>CI Upper <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Diff from Mean <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")
            for persona, data in sorted(effects.items(), key=lambda x: x[1].get("diff_from_mean", 0)):
                parts.append(f"<tr>")
                parts.append(f"<td>{html.escape(persona)}</td>")
                parts.append(f'<td data-sort-value="{data.get("rate", 0)}">{_fmt(data.get("rate"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("ci_lower", 0)}">{_fmt(data.get("ci_lower"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("ci_upper", 0)}">{_fmt(data.get("ci_upper"))}</td>')
                diff = data.get("diff_from_mean", 0)
                color = "#e74c3c" if diff < -0.05 else ("#2ecc71" if diff > 0.05 else "#2c3e50")
                parts.append(f'<td data-sort-value="{diff}" style="color:{color}; font-weight:bold;">{_fmt(diff)}</td>')
                parts.append("</tr>")
            parts.append("</tbody></table></div>")

        parts.append("</div></div>")

        # --- Mitigation Analysis ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Mitigation Analysis (Neutralization)</div>')
        parts.append('<div class="collapsible-body">')

        mit = model_res.get("mitigation", {})
        mcnemar_pp = mit.get("mcnemar_pre_post", {})
        if mcnemar_pp:
            parts.append(f'<p>McNemar pre/post: statistic = {_fmt(mcnemar_pp.get("statistic"))}, p = {_fmt(mcnemar_pp.get("p_value"))}</p>')

        if "variance_orig" in mit:
            parts.append('<div class="metrics-grid">')
            parts.append(f'<div class="metric-card"><div class="value">{_fmt(mit.get("variance_orig"))}</div><div class="label">Original Variance</div></div>')
            parts.append(f'<div class="metric-card"><div class="value">{_fmt(mit.get("variance_neut"))}</div><div class="label">Neutralized Variance</div></div>')
            vr = mit.get("variance_reduction")
            parts.append(f'<div class="metric-card"><div class="value">{_pct(vr)}</div><div class="label">Variance Reduction</div></div>')
            parts.append("</div>")

        # Paired bar chart
        paired_chart = _chart_mitigation_paired_bar(analysis_results, model)
        if paired_chart:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{paired_chart}" alt="Pre/post neutralization AIR"></div>')

        # Tier changes
        tc = mit.get("tier_changes", {})
        if tc:
            parts.append("<h4>Tier Change Summary</h4>")
            parts.append('<table><thead><tr><th>Transition</th><th>Count</th></tr></thead><tbody>')
            for transition, count in sorted(tc.items()):
                parts.append(f"<tr><td>{html.escape(transition)}</td><td>{count}</td></tr>")
            parts.append("</tbody></table>")

        parts.append("</div></div>")

    return "\n".join(parts)


def _build_tab_process_discrimination(
    analysis_results: Dict[str, Any],
    models: List[str],
) -> str:
    """Build the Process Discrimination tab HTML."""
    parts: List[str] = []
    parts.append("<h2>Process Discrimination Analysis</h2>")
    parts.append('<p class="note">Process discrimination measures disparities in <em>procedural burden</em> &mdash; '
                 'specifically, whether the LLM requests additional information at different rates depending on '
                 'perceived demographic characteristics.</p>')

    for model in models:
        model_res = analysis_results.get(model, {})
        if not model_res:
            continue
        parts.append(f"<h3>Model: {html.escape(model)}</h3>")

        # --- Information Request Rates ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Information Request Rates</div>')
        parts.append('<div class="collapsible-body">')

        info_chart = _chart_info_request_rate(analysis_results, model)
        if info_chart:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{info_chart}" alt="Info request rate by dimension"></div>')

        proc_corr = model_res.get("process_correlation", {})
        info_rates = proc_corr.get("info_rates_by_dimension", {})
        if info_rates:
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Dimension <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Category <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Info Request Rate <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")
            for dim in ESTIMATE_DIMS:
                dim_rates = info_rates.get(dim, {})
                for cat, rate in sorted(dim_rates.items()):
                    parts.append(f'<tr><td>{dim.capitalize()}</td><td>{html.escape(cat)}</td>')
                    parts.append(f'<td data-sort-value="{rate}">{_fmt(rate)}</td></tr>')
            parts.append("</tbody></table>")

        parts.append("</div></div>")

        # --- Process AIR ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Process AIR Heatmap</div>')
        parts.append('<div class="collapsible-body">')

        proc_air_chart = _chart_process_air_heatmap(analysis_results, model)
        if proc_air_chart:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{proc_air_chart}" alt="Process AIR heatmap"></div>')

        # Process chi2 stats
        proc_overall = proc_corr.get("overall_chi2", {})
        if proc_overall:
            parts.append("<h4>Overall Process Association</h4>")
            parts.append(f'<p>Chi2 = {_fmt(proc_overall.get("chi2"))}, p = {_fmt(proc_overall.get("p_value"))}, '
                         f'Cramer\'s V = {_fmt(proc_overall.get("cramers_v"))}</p>')

        parts.append("</div></div>")

        # --- Content Analysis ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Information Request Content Analysis</div>')
        parts.append('<div class="collapsible-body">')

        info_content = model_res.get("info_content", {})
        n_req = info_content.get("n_requests", 0)
        parts.append(f"<p>Total information requests: <strong>{n_req}</strong></p>")

        # Mean request length by persona (if available)
        mean_lengths = info_content.get("mean_length_by_persona", {})
        if mean_lengths:
            parts.append("<h4>Mean Request Length by Persona (top 20)</h4>")
            sorted_lengths = sorted(mean_lengths.items(), key=lambda x: x[1], reverse=True)[:20]
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Persona <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Mean Length <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")
            for persona, length in sorted_lengths:
                parts.append(f'<tr><td>{html.escape(persona)}</td><td data-sort-value="{length}">{_fmt(length, 1)}</td></tr>')
            parts.append("</tbody></table>")

        # ANOVA results
        for dim in ESTIMATE_DIMS:
            anova = info_content.get(f"length_anova_{dim}", {})
            if anova:
                parts.append(f'<p class="muted">ANOVA ({dim}): F = {_fmt(anova.get("f_statistic"))}, p = {_fmt(anova.get("p_value"))}</p>')

        # Process causal
        proc_causal = model_res.get("process_causal", {})
        cochrans = proc_causal.get("cochrans_q", {})
        if cochrans:
            parts.append(f'<p>Cochran\'s Q: statistic = {_fmt(cochrans.get("statistic"))}, p = {_fmt(cochrans.get("p_value"))}</p>')

        parts.append("</div></div>")

        # --- Process Mitigation ---
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Process Mitigation (Neutralization)</div>')
        parts.append('<div class="collapsible-body">')

        proc_mit = model_res.get("process_mitigation", {})
        proc_mcnemar = proc_mit.get("mcnemar_pre_post", {})
        if proc_mcnemar:
            parts.append(f'<p>McNemar pre/post: statistic = {_fmt(proc_mcnemar.get("statistic"))}, p = {_fmt(proc_mcnemar.get("p_value"))}</p>')

        if "variance_orig" in proc_mit:
            parts.append('<div class="metrics-grid">')
            parts.append(f'<div class="metric-card"><div class="value">{_fmt(proc_mit.get("variance_orig"))}</div><div class="label">Original Variance</div></div>')
            parts.append(f'<div class="metric-card"><div class="value">{_fmt(proc_mit.get("variance_neut"))}</div><div class="label">Neutralized Variance</div></div>')
            pvr = proc_mit.get("variance_reduction")
            parts.append(f'<div class="metric-card"><div class="value">{_pct(pvr)}</div><div class="label">Variance Reduction</div></div>')
            parts.append("</div>")

        parts.append("</div></div>")

    return "\n".join(parts)


def _build_tab_text_features(
    analysis_results: Dict[str, Any],
    text_features_df: Optional[pd.DataFrame],
    models: List[str],
) -> str:
    """Build the Text Features tab HTML."""
    parts: List[str] = []
    parts.append("<h2>Text Feature Analysis</h2>")

    if text_features_df is None or text_features_df.empty:
        parts.append('<p class="warning">No text feature data available.</p>')
        return "\n".join(parts)

    # Summary stats
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Feature Summary Statistics</div>')
    parts.append('<div class="collapsible-body">')

    numeric_cols = text_features_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [c for c in numeric_cols if c.startswith("topic_")]
    summary_cols = [c for c in numeric_cols if c not in exclude and c != "dominant_topic"]

    if summary_cols:
        desc = text_features_df[summary_cols].describe().round(3)
        parts.append('<div class="table-scroll">')
        parts.append('<table class="sortable"><thead><tr><th>Statistic</th>')
        for col in desc.columns:
            parts.append(f'<th>{html.escape(str(col))} <span class="sort-arrow">&#8597;</span></th>')
        parts.append("</tr></thead><tbody>")
        for stat in desc.index:
            parts.append(f"<tr><td><strong>{stat}</strong></td>")
            for col in desc.columns:
                val = desc.loc[stat, col]
                parts.append(f'<td data-sort-value="{val}">{_fmt(val, 3)}</td>')
            parts.append("</tr>")
        parts.append("</tbody></table></div>")

    parts.append("</div></div>")

    # Feature distributions
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Feature Distributions by Variant</div>')
    parts.append('<div class="collapsible-body">')

    violin_chart = _chart_feature_violins(text_features_df)
    if violin_chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{violin_chart}" alt="Feature violin plots"></div>')
    else:
        parts.append('<p class="muted">Insufficient data for violin plots.</p>')

    parts.append("</div></div>")

    # Feature importance (logistic regression coefficients)
    for model in models:
        fi_chart = _chart_feature_importance(analysis_results, model)
        if fi_chart:
            parts.append(f'<div class="collapsible">')
            parts.append(f'<div class="collapsible-header">Feature Importance ({html.escape(model)})</div>')
            parts.append('<div class="collapsible-body">')
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{fi_chart}" alt="Feature importance"></div>')
            parts.append("</div></div>")

    # Topic model
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Topic Model Analysis</div>')
    parts.append('<div class="collapsible-body">')

    # Try to get topic top words
    try:
        from src.text_features import get_topic_top_words
        topic_words = get_topic_top_words(n_words=10)
        if topic_words:
            parts.append("<h4>Topic Top Words</h4>")
            parts.append('<table><thead><tr><th>Topic</th><th>Top Words</th></tr></thead><tbody>')
            for i, words in enumerate(topic_words):
                parts.append(f"<tr><td>Topic {i}</td><td>{html.escape(', '.join(words))}</td></tr>")
            parts.append("</tbody></table>")
    except Exception:
        pass

    topic_heatmap = _chart_topic_persona_heatmap(text_features_df)
    if topic_heatmap:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{topic_heatmap}" alt="Topic-persona heatmap"></div>')
    else:
        parts.append('<p class="muted">Insufficient data for topic-persona heatmap.</p>')

    parts.append("</div></div>")

    return "\n".join(parts)


def _build_tab_model_comparison(
    analysis_results: Dict[str, Any],
    models: List[str],
    materiality_threshold: float,
) -> str:
    """Build the Model Comparison tab HTML."""
    parts: List[str] = []
    parts.append("<h2>Model Comparison</h2>")

    if len(models) < 2:
        parts.append('<p class="note">Model comparison requires at least two models.</p>')
        # Still show single model summary
        if models:
            model = models[0]
            parts.append(f"<h3>{html.escape(model)}</h3>")
            _append_model_summary_table(parts, analysis_results, model, materiality_threshold)
        return "\n".join(parts)

    # Comparison chart
    comp_chart = _chart_model_comparison(analysis_results)
    if comp_chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{comp_chart}" alt="Model comparison"></div>')

    # Side-by-side metrics table
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Side-by-Side Metrics</div>')
    parts.append('<div class="collapsible-body">')

    parts.append('<table class="sortable"><thead><tr>')
    parts.append('<th>Metric <span class="sort-arrow">&#8597;</span></th>')
    for model in models:
        parts.append(f'<th>{html.escape(model)} <span class="sort-arrow">&#8597;</span></th>')
    parts.append("</tr></thead><tbody>")

    metric_rows = _build_comparison_rows(analysis_results, models, materiality_threshold)
    for row in metric_rows:
        parts.append("<tr>")
        parts.append(f'<td><strong>{html.escape(row["label"])}</strong></td>')
        for model in models:
            val = row.get(model, "N/A")
            parts.append(f"<td>{val}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table>")
    parts.append("</div></div>")

    # Per-dimension comparison
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Per-Dimension Assessment Comparison</div>')
    parts.append('<div class="collapsible-body">')

    for dim in ESTIMATE_DIMS:
        parts.append(f"<h4>{dim.capitalize()}</h4>")
        parts.append("<table><thead><tr><th>Model</th><th>Chi2</th><th>p-value</th><th>Cramer's V</th></tr></thead><tbody>")
        for model in models:
            mr = analysis_results.get(model, {})
            d = _safe_get(mr, "correlation", f"{dim}_chi2", default={})
            parts.append(f'<tr><td>{html.escape(model)}</td>')
            parts.append(f'<td>{_fmt(d.get("chi2"))}</td>')
            parts.append(f'<td>{_fmt(d.get("p_value"))}</td>')
            parts.append(f'<td>{_fmt(d.get("cramers_v"))}</td></tr>')
        parts.append("</tbody></table>")

    parts.append("</div></div>")

    return "\n".join(parts)


def _append_model_summary_table(
    parts: List[str],
    analysis_results: Dict[str, Any],
    model: str,
    threshold: float,
) -> None:
    """Append a summary table for a single model."""
    mr = analysis_results.get(model, {})
    parts.append('<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>')

    cv = _safe_get(mr, "correlation", "overall_chi2", "cramers_v")
    parts.append(f'<tr><td>Overall Cramer\'s V</td><td>{_fmt(cv)}</td></tr>')

    r2 = _safe_get(mr, "correlation", "logistic_favorable", "pseudo_r2")
    parts.append(f'<tr><td>Pseudo R2</td><td>{_fmt(r2)}</td></tr>')

    vr = _safe_get(mr, "mitigation", "variance_reduction")
    parts.append(f'<tr><td>Outcome Variance Reduction</td><td>{_pct(vr)}</td></tr>')

    pvr = _safe_get(mr, "process_mitigation", "variance_reduction")
    parts.append(f'<tr><td>Process Variance Reduction</td><td>{_pct(pvr)}</td></tr>')

    parts.append("</tbody></table>")


def _build_comparison_rows(
    analysis_results: Dict[str, Any],
    models: List[str],
    threshold: float,
) -> List[Dict[str, str]]:
    """Build rows for the model comparison table."""
    rows = []

    def _add_row(label: str, extractor):
        row = {"label": label}
        for m in models:
            mr = analysis_results.get(m, {})
            val = extractor(mr)
            row[m] = _fmt(val) if isinstance(val, (int, float)) else (str(val) if val is not None else "N/A")
        rows.append(row)

    _add_row("Overall Chi2 p-value", lambda mr: _safe_get(mr, "correlation", "overall_chi2", "p_value"))
    _add_row("Overall Cramer's V", lambda mr: _safe_get(mr, "correlation", "overall_chi2", "cramers_v"))
    _add_row("Logistic Pseudo R2", lambda mr: _safe_get(mr, "correlation", "logistic_favorable", "pseudo_r2"))
    _add_row("Logistic AIC", lambda mr: _safe_get(mr, "correlation", "logistic_favorable", "aic"))
    _add_row("Friedman p-value (Causal)", lambda mr: _safe_get(mr, "causal", "friedman", "p_value"))
    _add_row("McNemar p-value (Mitigation)", lambda mr: _safe_get(mr, "mitigation", "mcnemar_pre_post", "p_value"))
    _add_row("Outcome Variance (Original)", lambda mr: _safe_get(mr, "mitigation", "variance_orig"))
    _add_row("Outcome Variance (Neutralized)", lambda mr: _safe_get(mr, "mitigation", "variance_neut"))
    _add_row("Outcome Variance Reduction", lambda mr: _safe_get(mr, "mitigation", "variance_reduction"))
    _add_row("Process Chi2 p-value", lambda mr: _safe_get(mr, "process_correlation", "overall_chi2", "p_value"))
    _add_row("Process Cramer's V", lambda mr: _safe_get(mr, "process_correlation", "overall_chi2", "cramers_v"))
    _add_row("Process Variance Reduction", lambda mr: _safe_get(mr, "process_mitigation", "variance_reduction"))
    _add_row("Cochran's Q p-value", lambda mr: _safe_get(mr, "process_causal", "cochrans_q", "p_value"))

    # Repeated-measures model results (per-model rows)
    rm = analysis_results.get("repeated_measures", {})
    if rm and not rm.get("skipped"):
        def _rm_extractor(*keys):
            def extractor(mr_unused):
                # mr_unused is the per-model analysis_results[model] dict,
                # but repeated-measures results are stored under analysis_results["repeated_measures"][model].
                # We iterate models in the row builder, so just return None here;
                # the actual per-model values are added below.
                return None
            return extractor

        # Add per-model ICC row manually
        row_icc = {"label": "LMM ICC (complaint clustering)"}
        for m in models:
            icc = _safe_get(rm, m, "linear_mixed_model", "icc")
            row_icc[m] = _fmt(icc) if isinstance(icc, (int, float)) else "N/A"
        rows.append(row_icc)

        row_clogit = {"label": "Cond. Logistic persona_score OR"}
        for m in models:
            or_val = _safe_get(rm, m, "conditional_logistic_favourable", "odds_ratios", "persona_score")
            row_clogit[m] = _fmt(or_val) if isinstance(or_val, (int, float)) else "N/A"
        rows.append(row_clogit)

        row_pxs = {"label": "Ordinal persona×severity p"}
        for m in models:
            p_val = _safe_get(rm, m, "ordinal_model", "p_values", "persona_x_severity")
            row_pxs[m] = _fmt(p_val) if isinstance(p_val, (int, float)) else "N/A"
        rows.append(row_pxs)

    return rows


def _build_tab_detailed_results(
    analysis_results: Dict[str, Any],
    complaints: List[Complaint],
    models: List[str],
    num_complaints: int,
    seed: int,
    materiality_threshold: float,
    timestamp: str,
) -> str:
    """Build the Detailed Results tab HTML."""
    parts: List[str] = []
    parts.append("<h2>Detailed Results</h2>")

    # Reproducibility info
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Reproducibility Information</div>')
    parts.append('<div class="collapsible-body">')
    parts.append("<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>")
    parts.append(f"<tr><td>Timestamp</td><td>{html.escape(timestamp)}</td></tr>")
    parts.append(f"<tr><td>Random Seed</td><td>{seed}</td></tr>")
    parts.append(f"<tr><td>Number of Complaints</td><td>{num_complaints}</td></tr>")
    parts.append(f"<tr><td>Models</td><td>{html.escape(', '.join(models))}</td></tr>")
    parts.append(f"<tr><td>Materiality Threshold (AIR)</td><td>{materiality_threshold}</td></tr>")
    parts.append(f"<tr><td>Persona Grid Size</td><td>{len(PERSONA_LABELS)}</td></tr>")
    parts.append("</tbody></table>")
    parts.append("</div></div>")

    # Persona distribution chart
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Persona Distribution</div>')
    parts.append('<div class="collapsible-body">')

    dist_chart = _chart_persona_distribution(complaints, analysis_results)
    if dist_chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{dist_chart}" alt="Persona distribution"></div>')
    else:
        parts.append('<p class="muted">No persona distribution data available.</p>')

    parts.append("</div></div>")

    # Full AIR tables per model
    for model in models:
        model_res = analysis_results.get(model, {})

        # Outcome AIR
        air_data = _safe_get(model_res, "correlation", "air_by_persona", default={})
        if air_data:
            parts.append(f'<div class="collapsible collapsed">')
            parts.append(f'<div class="collapsible-header">Full AIR Table &mdash; {html.escape(model)} (Outcome)</div>')
            parts.append('<div class="collapsible-body">')
            parts.append('<div class="table-scroll">')
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Persona <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Age <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Geography <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Education <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>SES <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Favorable Rate <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>AIR <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Material <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")

            for persona in sorted(air_data.keys()):
                data = air_data[persona]
                parts_p = persona.split("_")
                age = parts_p[0] if len(parts_p) > 0 else ""
                geo = parts_p[1] if len(parts_p) > 1 else ""
                edu = parts_p[2] if len(parts_p) > 2 else ""
                ses = parts_p[3] if len(parts_p) > 3 else ""
                is_mat = data.get("material", False)
                row_class = f' class="flag-row-{"BOTH" if is_mat else "NONE"}"'
                parts.append(f"<tr{row_class}>")
                parts.append(f"<td>{html.escape(persona)}</td>")
                parts.append(f"<td>{html.escape(age)}</td>")
                parts.append(f"<td>{html.escape(geo)}</td>")
                parts.append(f"<td>{html.escape(edu)}</td>")
                parts.append(f"<td>{html.escape(ses)}</td>")
                parts.append(f'<td data-sort-value="{data.get("rate", 0)}">{_fmt(data.get("rate"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("air", 1)}">{_fmt(data.get("air"))}</td>')
                parts.append(f'<td>{"Yes" if is_mat else "No"}</td>')
                parts.append("</tr>")

            parts.append("</tbody></table></div></div></div>")

        # Process AIR
        proc_air = _safe_get(model_res, "process_correlation", "process_air_by_persona", default={})
        if proc_air:
            parts.append(f'<div class="collapsible collapsed">')
            parts.append(f'<div class="collapsible-header">Full AIR Table &mdash; {html.escape(model)} (Process)</div>')
            parts.append('<div class="collapsible-body">')
            parts.append('<div class="table-scroll">')
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Persona <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Info Request Rate <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Process AIR <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Material <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")

            for persona in sorted(proc_air.keys()):
                data = proc_air[persona]
                is_mat = data.get("material", False)
                row_class = f' class="flag-row-{"BOTH" if is_mat else "NONE"}"'
                parts.append(f"<tr{row_class}>")
                parts.append(f"<td>{html.escape(persona)}</td>")
                parts.append(f'<td data-sort-value="{data.get("rate", 0)}">{_fmt(data.get("rate"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("air", 1)}">{_fmt(data.get("air"))}</td>')
                parts.append(f'<td>{"Yes" if is_mat else "No"}</td>')
                parts.append("</tr>")

            parts.append("</tbody></table></div></div></div>")

        # Causal effect sizes
        effects = _safe_get(model_res, "causal", "effect_sizes", default={})
        if effects:
            parts.append(f'<div class="collapsible collapsed">')
            parts.append(f'<div class="collapsible-header">Full Causal Effects &mdash; {html.escape(model)}</div>')
            parts.append('<div class="collapsible-body">')
            parts.append('<div class="table-scroll">')
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Persona <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Rate <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>CI Lower <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>CI Upper <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Diff from Mean <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")
            for persona in sorted(effects.keys()):
                data = effects[persona]
                diff = data.get("diff_from_mean", 0)
                color = "#e74c3c" if diff < -0.05 else ("#2ecc71" if diff > 0.05 else "#2c3e50")
                parts.append(f"<tr>")
                parts.append(f"<td>{html.escape(persona)}</td>")
                parts.append(f'<td data-sort-value="{data.get("rate", 0)}">{_fmt(data.get("rate"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("ci_lower", 0)}">{_fmt(data.get("ci_lower"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("ci_upper", 0)}">{_fmt(data.get("ci_upper"))}</td>')
                parts.append(f'<td data-sort-value="{diff}" style="color:{color};font-weight:bold;">{_fmt(diff)}</td>')
                parts.append("</tr>")
            parts.append("</tbody></table></div></div></div>")

        # Mitigation by persona
        mit_by_persona = _safe_get(model_res, "mitigation", "mitigation_by_persona", default={})
        if mit_by_persona:
            parts.append(f'<div class="collapsible collapsed">')
            parts.append(f'<div class="collapsible-header">Mitigation Effectiveness by Persona &mdash; {html.escape(model)}</div>')
            parts.append('<div class="collapsible-body">')
            parts.append('<div class="table-scroll">')
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Persona <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Original AIR <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Neutralized AIR <span class="sort-arrow">&#8597;</span></th>')
            parts.append('<th>Bias Reduction <span class="sort-arrow">&#8597;</span></th>')
            parts.append("</tr></thead><tbody>")
            for persona in sorted(mit_by_persona.keys()):
                data = mit_by_persona[persona]
                reduction = data.get("bias_reduction_pct", 0)
                color = "#2ecc71" if reduction > 0.5 else ("#f39c12" if reduction > 0 else "#e74c3c")
                parts.append(f"<tr>")
                parts.append(f"<td>{html.escape(persona)}</td>")
                parts.append(f'<td data-sort-value="{data.get("orig_air", 1)}">{_fmt(data.get("orig_air"))}</td>')
                parts.append(f'<td data-sort-value="{data.get("neut_air", 1)}">{_fmt(data.get("neut_air"))}</td>')
                parts.append(f'<td data-sort-value="{reduction}" style="color:{color};font-weight:bold;">{_pct(reduction)}</td>')
                parts.append("</tr>")
            parts.append("</tbody></table></div></div></div>")

    # Complaint sample
    parts.append('<div class="collapsible collapsed">')
    parts.append('<div class="collapsible-header">Complaint Sample (first 20)</div>')
    parts.append('<div class="collapsible-body">')
    if complaints:
        parts.append('<div class="table-scroll">')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Case ID <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Product <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Issue <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Date <span class="sort-arrow">&#8597;</span></th>')
        parts.append('<th>Narrative (truncated) <span class="sort-arrow">&#8597;</span></th>')
        parts.append("</tr></thead><tbody>")
        for c in complaints[:20]:
            narrative_short = c.narrative[:150] + "..." if len(c.narrative) > 150 else c.narrative
            parts.append(f"<tr>")
            parts.append(f"<td>{html.escape(c.case_id)}</td>")
            parts.append(f"<td>{html.escape(c.product_category)}</td>")
            parts.append(f"<td>{html.escape(c.issue_category)}</td>")
            parts.append(f"<td>{html.escape(c.date_received)}</td>")
            parts.append(f"<td>{html.escape(narrative_short)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table></div>")
    else:
        parts.append('<p class="muted">No complaints available.</p>')

    parts.append("</div></div>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Severity Analysis tab
# ---------------------------------------------------------------------------

def _chart_severity_dist(sev: Dict[str, Any]) -> Optional[str]:
    """Side-by-side bar charts for severity distributions."""
    m_dist = _safe_get(sev, "descriptives", "monetary_severity", "distribution", default={})
    i_dist = _safe_get(sev, "descriptives", "issue_severity", "distribution", default={})
    if not m_dist and not i_dist:
        return None

    levels = ["low", "medium", "high", "unknown"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Monetary
    vals = [m_dist.get(l, 0) for l in levels]
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]
    ax1.bar(levels, vals, color=colors)
    ax1.set_title("Monetary Severity", fontweight="bold")
    ax1.set_ylabel("Count")
    for i, v in enumerate(vals):
        if v > 0:
            ax1.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=9)

    # Issue
    i_levels = ["low", "medium", "high"]
    i_vals = [i_dist.get(l, 0) for l in i_levels]
    ax2.bar(i_levels, i_vals, color=colors[:3])
    ax2.set_title("Issue Severity", fontweight="bold")
    ax2.set_ylabel("Count")
    for i, v in enumerate(i_vals):
        if v > 0:
            ax2.text(i, v + max(i_vals) * 0.02, str(v), ha="center", fontsize=9)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_severity_crosstab(sev: Dict[str, Any]) -> Optional[str]:
    """Heatmap of monetary × issue severity cross-tabulation (including unknown)."""
    # Prefer the full cross-tab that includes "unknown" monetary severity
    ct_data = _safe_get(sev, "descriptives", "agreement", "cross_tab_full", default={})
    if not ct_data:
        ct_data = _safe_get(sev, "descriptives", "agreement", "cross_tab", default={})
    if not ct_data:
        return None
    try:
        row_order = ["low", "medium", "high", "unknown"]
        col_order = ["low", "medium", "high"]
        ct = pd.DataFrame(ct_data)
        # Keep only rows that exist in the data
        row_order = [r for r in row_order if r in ct.index]
        ct = ct.reindex(index=row_order, columns=col_order).fillna(0)
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Count"})
    ax.set_xlabel("Issue Severity")
    ax.set_ylabel("Monetary Severity")
    ax.set_title("Severity Cross-Tabulation", fontweight="bold")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_severity_main_effects(sev: Dict[str, Any], models: List[str]) -> Optional[str]:
    """Line chart: mean tier by severity per model with CI bands."""
    me = _safe_get(sev, "main_effects", default={})
    if not me:
        return None

    levels = ["low", "medium", "high"]
    x = np.arange(len(levels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, metric in enumerate(["monetary_severity", "issue_severity"]):
        ax = axes[idx]
        metric_data = me.get(metric, {})
        for mi, model in enumerate(models):
            model_data = metric_data.get(model, {})
            ci_data = model_data.get("ci_by_level", {})
            means = [ci_data.get(l, {}).get("mean", float("nan")) for l in levels]
            ci_lo = [ci_data.get(l, {}).get("ci_lower", float("nan")) for l in levels]
            ci_hi = [ci_data.get(l, {}).get("ci_upper", float("nan")) for l in levels]

            valid = [i for i, m in enumerate(means) if np.isfinite(m)]
            if not valid:
                continue
            vx = [x[i] for i in valid]
            vm = [means[i] for i in valid]
            vlo = [ci_lo[i] for i in valid]
            vhi = [ci_hi[i] for i in valid]

            ax.plot(vx, vm, marker="o", label=model)
            ax.fill_between(vx, vlo, vhi, alpha=0.15)

        ax.set_xticks(x)
        ax.set_xticklabels([l.title() for l in levels])
        ax.set_ylabel("Mean Remedy Tier")
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Severity Main Effects on Outcomes", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_base64(fig)


def _chart_amplification_bars(
    sev: Dict[str, Any],
    models: List[str],
    interactions_key: str = "interactions",
    ylabel: str = "Bias (SD of persona means)",
    title: str = "Bias Magnitude by Severity Level",
) -> Optional[str]:
    """Bar chart: bias at each severity level with CI error bars."""
    interactions = _safe_get(sev, interactions_key, default={})
    if not interactions:
        return None

    levels = ["low", "medium", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, metric in enumerate(["monetary_severity", "issue_severity"]):
        ax = axes[idx]
        metric_int = interactions.get(metric, {})
        bar_width = 0.35
        for mi, model in enumerate(models):
            bias_data = metric_int.get(model, {}).get("bias", {})
            sds = [bias_data.get(l, {}).get("sd_across_personas", 0) for l in levels]
            positions = np.arange(len(levels)) + mi * bar_width
            ax.bar(positions, sds, bar_width, label=model, alpha=0.8)

        ax.set_xticks(np.arange(len(levels)) + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels([l.title() for l in levels])
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_base64(fig)


def _chart_model_amplification(sev: Dict[str, Any], models: List[str]) -> Optional[str]:
    """Grouped bar chart: amplification ratio by model."""
    mc = _safe_get(sev, "model_comparison", default={})
    if not mc:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.35

    m_ratios = [
        _safe_get(mc, m, "monetary_severity_amplification", "point", default=float("nan"))
        for m in models
    ]
    i_ratios = [
        _safe_get(mc, m, "issue_severity_amplification", "point", default=float("nan"))
        for m in models
    ]

    ax.bar(x - width / 2, m_ratios, width, label="Monetary", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, i_ratios, width, label="Issue", color="#e74c3c", alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No amplification")
    ax.axhline(y=1.5, color="#f39c12", linestyle=":", alpha=0.5)
    ax.axhline(y=2.5, color="#e74c3c", linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Amplification Ratio")
    ax.set_title("Amplification Ratio by Model", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_process_severity(sev: Dict[str, Any], models: List[str]) -> Optional[str]:
    """Info request disparity by severity level."""
    proc = _safe_get(sev, "process", default={})
    if not proc:
        return None

    levels = ["low", "medium", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, metric in enumerate(["monetary_severity", "issue_severity"]):
        ax = axes[idx]
        metric_proc = proc.get(metric, {})
        for model in models:
            model_data = metric_proc.get(model, {})
            rates = model_data.get("rates_by_severity", {})
            disparities = [rates.get(l, {}).get("disparity", 0) for l in levels]
            valid = [(i, d) for i, d in enumerate(disparities) if d > 0 or True]
            if valid:
                ax.plot([levels[i].title() for i, _ in valid],
                        [d for _, d in valid], marker="o", label=model)

        ax.set_ylabel("Info Request Disparity")
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Process Discrimination by Severity", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_base64(fig)


def _traffic_light(ratio: float) -> Tuple[str, str, str]:
    """Return (color_hex, bg_hex, label) for amplification ratio."""
    if not np.isfinite(ratio):
        return "#95a5a6", "#f0f0f0", "N/A"
    if ratio < 1.5:
        return "#2ecc71", "#eafaf1", "Minimal"
    if ratio <= 2.5:
        return "#f39c12", "#fef9e7", "Moderate"
    return "#e74c3c", "#fdecea", "Severe"


def _chart_missingness_distributions(distributions: Dict[str, Any]) -> Optional[str]:
    """Grouped bar charts showing value distributions split by monetary present/missing."""
    if not distributions:
        return None

    # Define display groups with labels
    chart_groups = [
        ("tier", "Remedy Tier"),
        ("age", "Age"),
        ("gender", "Gender"),
        ("geographic", "Geography"),
        ("education", "Education"),
        ("socioeconomic", "Socio-Economic"),
        ("esl", "ESL Status"),
        ("product_category", "Product"),
        ("issue_category", "Issue"),
        ("issue_severity", "Issue Severity"),
    ]

    # Filter to variables that exist
    available = [(key, label) for key, label in chart_groups if key in distributions]
    if not available:
        return None

    n_charts = len(available)
    n_cols = min(3, n_charts)
    n_rows = (n_charts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.0 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (key, label) in enumerate(available):
        ax = axes[idx]
        dist = distributions[key]
        pres = dist.get("present", {})
        miss = dist.get("missing", {})

        # Union of all category values, sorted
        all_cats = sorted(set(list(pres.keys()) + list(miss.keys())),
                          key=lambda x: (isinstance(x, str), x))

        # Convert to percentages within each group
        pres_total = max(sum(pres.values()), 1)
        miss_total = max(sum(miss.values()), 1)
        pres_pcts = [pres.get(c, 0) / pres_total * 100 for c in all_cats]
        miss_pcts = [miss.get(c, 0) / miss_total * 100 for c in all_cats]

        x = np.arange(len(all_cats))
        width = 0.38
        ax.bar(x - width / 2, pres_pcts, width, label="$ Present", color="#3498db", alpha=0.85)
        ax.bar(x + width / 2, miss_pcts, width, label="$ Missing", color="#e74c3c", alpha=0.85)

        # Truncate long labels
        cat_labels = [str(c)[:18] for c in all_cats]
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=45 if len(all_cats) > 3 else 0,
                           ha="right" if len(all_cats) > 3 else "center", fontsize=8)
        ax.set_ylabel("% within group", fontsize=8)
        ax.set_title(label, fontweight="bold", fontsize=10)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distributions by Monetary Amount Presence", fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _build_tab_severity(
    analysis_results: Dict[str, Any],
    complaints: list,
    models: List[str],
) -> str:
    """Build the Severity Analysis tab HTML (6 panels)."""
    parts: List[str] = []
    sev = analysis_results.get("severity", {})

    if not sev:
        parts.append("<h2>Severity Analysis</h2>")
        parts.append("<p>Severity analysis was not run. Ensure baseline decisions and persona estimates are available.</p>")
        return "\n".join(parts)

    parts.append("<h2>Severity Amplification Analysis</h2>")
    parts.append("<p>Tests whether LLM bias in complaint resolution is amplified in high-severity cases.</p>")

    # ── Panel 1: Overview ──
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Panel 1: Severity Overview</div>')
    parts.append('<div class="collapsible-body">')

    chart = _chart_severity_dist(sev)
    if chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{chart}" alt="Severity Distribution"></div>')

    # Coverage stats
    desc = sev.get("descriptives", {})
    m_cov = _safe_get(desc, "monetary_severity", "coverage", default=0)
    i_cov = _safe_get(desc, "issue_severity", "coverage", default=0)
    parts.append('<table><thead><tr><th>Metric</th><th>Coverage</th></tr></thead><tbody>')
    parts.append(f'<tr><td>Monetary Severity</td><td>{_pct(m_cov)}</td></tr>')
    parts.append(f'<tr><td>Issue Severity</td><td>{_pct(i_cov)}</td></tr>')
    parts.append('</tbody></table>')

    # Cross-tabulation heatmap
    chart = _chart_severity_crosstab(sev)
    if chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{chart}" alt="Cross-Tabulation"></div>')

    # Agreement stats
    agreement = _safe_get(desc, "agreement", default={})
    if agreement:
        kappa = agreement.get("kappa", float("nan"))
        interp = "High" if kappa > 0.6 else ("Moderate" if kappa > 0.4 else "Low")
        parts.append('<table><thead><tr><th>Agreement Metric</th><th>Value</th><th>Interpretation</th></tr></thead><tbody>')
        parts.append(f'<tr><td>Cohen\'s Kappa</td><td>{_fmt(kappa)}</td><td>{interp} agreement</td></tr>')
        parts.append(f'<tr><td>Spearman r</td><td>{_fmt(agreement.get("spearman_r"))}</td><td>p = {_fmt(agreement.get("spearman_p"))}</td></tr>')
        parts.append(f'<tr><td>Agreement Rate</td><td>{_pct(agreement.get("agreement_rate"))}</td><td></td></tr>')
        parts.append('</tbody></table>')

    # Coverage: small cells warning
    cov = _safe_get(desc, "coverage", default={})
    for metric in ["monetary_severity", "issue_severity"]:
        small = _safe_get(cov, metric, "small_cells", default=[])
        if small:
            parts.append(f'<p style="color:#e67e22;"><strong>{metric}:</strong> {len(small)} persona×severity cells have &lt; 30 observations</p>')

    # ── Missingness diagnostic sub-section ──
    miss_diag = sev.get("missingness_diagnostic", {})
    if miss_diag and not miss_diag.get("skipped"):
        parts.append('<h3>Monetary Severity: Missingness Diagnostic</h3>')
        parts.append('<p>About {:.0%} of complaints have no extractable dollar amount. '
                     'This diagnostic tests whether that subset differs systematically '
                     'from complaints with known amounts.</p>'.format(miss_diag.get("pct_missing", 0)))

        chars = miss_diag.get("complaint_characteristics", {})
        m_type = chars.get("missingness_type", "unknown")
        n_sig = chars.get("n_significant_tests", 0)

        # Verdict card
        if m_type == "MCAR":
            v_color, v_bg = "#27ae60", "#eafaf1"
            v_label = "Random (MCAR)"
            v_desc = "Missingness appears random. Monetary severity findings likely generalise."
        else:
            v_color, v_bg = "#e74c3c", "#fdedec"
            v_label = "Systematic (MAR)"
            v_desc = f"{n_sig}/5 tests significant — complaints with dollar amounts differ from those without."

        parts.append(f'<div class="metric-card" style="border-left: 4px solid {v_color}; background: {v_bg}; '
                     f'margin: 10px 0; padding: 12px;">')
        parts.append(f'<div class="value" style="color: {v_color}; font-size: 1.2rem;">{v_label}</div>')
        parts.append(f'<div style="font-size: 0.85rem;">{v_desc}</div>')
        parts.append('</div>')

        # Distribution histograms
        dist_chart = _chart_missingness_distributions(miss_diag.get("distributions", {}))
        if dist_chart:
            parts.append('<h4>Distribution Profiles by Missingness Group</h4>')
            parts.append('<p style="font-size: 0.85rem;">Percentage distributions within each group '
                         '(blue = dollar amount present, red = dollar amount missing). '
                         'Differences indicate systematic patterns in which complaints '
                         'contain extractable dollar amounts.</p>')
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{dist_chart}" '
                         f'alt="Missingness distribution profiles"></div>')

        # Characteristic comparison table
        parts.append('<table><thead><tr>')
        parts.append('<th>Test</th><th>Statistic</th><th>p-value</th><th>Effect Size</th><th>Significant?</th>')
        parts.append('</tr></thead><tbody>')

        sig_tests = chars.get("significant_tests", [])

        def _miss_row(name: str, label: str, stat_key: str, p_key: str, effect_key: str, effect_label: str) -> str:
            d = chars.get(name, {})
            if d.get("error"):
                return f'<tr><td>{label}</td><td colspan="4">Error</td></tr>'
            stat_val = d.get(stat_key, d.get("chi2", float("nan")))
            p_val = d.get(p_key, d.get("p_value", float("nan")))
            eff = d.get(effect_key, d.get("cramers_v", float("nan")))
            is_sig = name in sig_tests
            sig_style = ' style="color: #e74c3c; font-weight: bold;"' if is_sig else ""
            return (f'<tr{sig_style}><td>{label}</td>'
                    f'<td>{_fmt(stat_val)}</td>'
                    f'<td>{_fmt(p_val)}</td>'
                    f'<td>{_fmt(eff)} ({effect_label})</td>'
                    f'<td>{"Yes" if is_sig else "No"}</td></tr>')

        parts.append(_miss_row("issue_category", "Issue Category", "chi2", "p_value", "cramers_v", "Cram\u00e9r's V"))
        parts.append(_miss_row("product_category", "Product Category", "chi2", "p_value", "cramers_v", "Cram\u00e9r's V"))
        parts.append(_miss_row("word_count", "Word Count", "U_statistic", "p_value", "rank_biserial", "rank-biserial r"))
        parts.append(_miss_row("mean_tier", "Mean Remedy Tier", "t_statistic", "p_value", "cohens_d", "Cohen\u2019s d"))
        parts.append(_miss_row("tier_distribution", "Tier Distribution", "chi2", "p_value", "cramers_v", "Cram\u00e9r's V"))

        parts.append('</tbody></table>')

        # Word count means (if available)
        wc = chars.get("word_count", {})
        if wc and not wc.get("error"):
            parts.append(f'<p style="font-size: 0.85rem;">Mean word count: '
                         f'present = {wc.get("mean_present", 0):.0f}, '
                         f'missing = {wc.get("mean_missing", 0):.0f}</p>')

        # Fairness comparison table
        fairness = miss_diag.get("fairness_comparison", {})
        if fairness:
            parts.append('<h4>Fairness: Amount Present vs Missing</h4>')
            parts.append('<table><thead><tr>')
            parts.append('<th>Model</th><th>Subset</th><th>N</th><th>Bias SD</th><th>Bias Range</th><th>Sig. Pairs</th>')
            parts.append('</tr></thead><tbody>')
            for model in models:
                fc = fairness.get(model, {})
                if not fc:
                    continue
                for subset_label, subset_key in [("Present", "present"), ("Missing", "missing")]:
                    s = fc.get(subset_key, {})
                    parts.append(f'<tr><td>{html.escape(model)}</td><td>{subset_label}</td>'
                                 f'<td>{s.get("n", 0)}</td>'
                                 f'<td>{_fmt(s.get("bias_sd"))}</td>'
                                 f'<td>{_fmt(s.get("bias_range"))}</td>'
                                 f'<td>{s.get("n_significant_pairs", 0)}</td></tr>')
                # Ratio row
                ratio = fc.get("bias_sd_ratio", float("nan"))
                ratio_color = "#e74c3c" if np.isfinite(ratio) and abs(ratio - 1.0) > 0.2 else "#27ae60"
                parts.append(f'<tr style="font-weight: bold;"><td>{html.escape(model)}</td>'
                             f'<td>Ratio (missing/present)</td><td></td>'
                             f'<td style="color: {ratio_color};">{_fmt(ratio, 2)}</td>'
                             f'<td colspan="2">{html.escape(fc.get("conclusion", ""))}</td></tr>')
            parts.append('</tbody></table>')

        # Overall conclusion
        overall = miss_diag.get("overall_conclusion", "")
        if overall:
            parts.append(f'<p style="margin-top: 10px; padding: 8px; background: #f8f9fa; '
                         f'border-radius: 4px; font-size: 0.85rem;"><strong>Summary:</strong> {html.escape(overall)}</p>')

    parts.append('</div></div>')

    # ── Panel 2: Main Effects ──
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Panel 2: Severity Main Effects</div>')
    parts.append('<div class="collapsible-body">')

    chart = _chart_severity_main_effects(sev, models)
    if chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{chart}" alt="Main Effects"></div>')

    # ANOVA / chi-square results table
    me = sev.get("main_effects", {})
    parts.append('<table class="sortable"><thead><tr>')
    parts.append('<th>Metric</th><th>Model</th><th>ANOVA F</th><th>ANOVA p</th>')
    parts.append('<th>Partial &eta;&sup2;</th><th>&eta;&sup2; Size</th><th>Chi&sup2; p</th><th>Trend r</th><th>Trend p</th>')
    parts.append('</tr></thead><tbody>')
    for metric in ["monetary_severity", "issue_severity"]:
        for model in models:
            md = _safe_get(me, metric, model, default={})
            anova = md.get("anova", {})
            chi2 = md.get("chi2_favorable", {})
            trend = md.get("trend", {})
            parts.append(f'<tr><td>{metric.replace("_"," ").title()}</td><td>{html.escape(model)}</td>')
            parts.append(f'<td>{_fmt(anova.get("F"))}</td><td>{_fmt(anova.get("p_value"))}</td>')
            parts.append(f'<td>{_fmt(anova.get("eta_sq_partial"))}</td>')
            parts.append(f'<td>{anova.get("eta_sq_interpretation", "")}</td>')
            parts.append(f'<td>{_fmt(chi2.get("p_value"))}</td>')
            parts.append(f'<td>{_fmt(trend.get("spearman_r"))}</td><td>{_fmt(trend.get("p_value"))}</td></tr>')
    parts.append('</tbody></table>')

    parts.append('</div></div>')

    # ── Panel 3: Amplification ──
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Panel 3: Severity Amplification (Core Finding)</div>')
    parts.append('<div class="collapsible-body">')

    # Key metric cards with traffic lights
    interactions = sev.get("interactions", {})
    parts.append('<div class="metrics-grid">')
    for metric in ["monetary_severity", "issue_severity"]:
        for model in models:
            amp = _safe_get(interactions, metric, model, "amplification", default={})
            ratio = amp.get("point", float("nan"))
            ci_lo = amp.get("ci_lower", float("nan"))
            ci_hi = amp.get("ci_upper", float("nan"))
            color, bg, label = _traffic_light(ratio)
            parts.append(f'<div class="metric-card" style="border-left: 4px solid {color}; background: {bg};">')
            parts.append(f'<div class="value" style="color: {color};">{_fmt(ratio, 2)}x</div>')
            parts.append(f'<div class="label">{metric.replace("_"," ").title()} ({model})</div>')
            parts.append(f'<div style="font-size:0.8rem;">CI: [{_fmt(ci_lo, 2)}, {_fmt(ci_hi, 2)}] — {label}</div>')
            parts.append('</div>')
    parts.append('</div>')

    chart = _chart_amplification_bars(sev, models)
    if chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{chart}" alt="Amplification"></div>')

    # Bias detail by severity level
    main_effects = sev.get("main_effects", {})
    parts.append('<h3>Bias by Severity Level</h3>')
    parts.append('<p>Shows how bias (persona-based variation in outcomes) changes across severity levels. '
                 'Higher bias SD and range at high severity indicates amplification.</p>')
    parts.append('<table class="sortable"><thead><tr>')
    parts.append('<th>Metric</th><th>Model</th><th>Severity</th><th>N</th>'
                 '<th>Mean Tier</th><th>Bias SD</th><th>Bias Range</th>'
                 '<th>Persona &eta;&sup2;</th><th>Largest d</th><th>Sig. Pairs</th>')
    parts.append('</tr></thead><tbody>')
    for metric in ["monetary_severity", "issue_severity"]:
        for model in models:
            mi = _safe_get(interactions, metric, model, default={})
            bias_data = mi.get("bias", {})
            se_data = mi.get("simple_effects", {})
            mfx = _safe_get(main_effects, metric, model, default={})
            means = mfx.get("means_by_level", {})
            # Collect SDs for highlighting the max
            sds = {lvl: bias_data.get(lvl, {}).get("sd_across_personas", 0) for lvl in ["low", "medium", "high"]}
            max_sd_level = max(sds, key=lambda k: sds[k]) if any(sds.values()) else None
            for level in ["low", "medium", "high"]:
                bl = bias_data.get(level, {})
                sl = se_data.get(level, {})
                n = bl.get("n", 0)
                sd = bl.get("sd_across_personas", float("nan"))
                rng = bl.get("range", float("nan"))
                mean_tier = means.get(level, float("nan"))
                eta = sl.get("eta_sq_partial", float("nan"))
                largest_d = sl.get("largest_pairwise_d", float("nan"))
                n_sig = sl.get("n_significant", 0)
                # Highlight the row with the highest bias
                style = ' style="background:#fdecea;font-weight:bold;"' if level == max_sd_level else ""
                parts.append(f'<tr{style}>')
                parts.append(f'<td>{metric.replace("_"," ").title()}</td>')
                parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                parts.append(f'<td>{n}</td><td>{_fmt(mean_tier, 3)}</td>')
                parts.append(f'<td>{_fmt(sd, 4)}</td><td>{_fmt(rng, 4)}</td>')
                parts.append(f'<td>{_fmt(eta, 4)}</td><td>{_fmt(largest_d, 3)}</td>')
                parts.append(f'<td>{n_sig}</td></tr>')
    parts.append('</tbody></table>')

    # Most/least advantaged personas at each severity level
    parts.append('<h3>Most &amp; Least Advantaged Personas by Severity</h3>')
    parts.append('<table class="sortable"><thead><tr>')
    parts.append('<th>Metric</th><th>Model</th><th>Severity</th>'
                 '<th>Most Advantaged</th><th>Mean Tier</th>'
                 '<th>Most Disadvantaged</th><th>Mean Tier</th><th>Gap</th>')
    parts.append('</tr></thead><tbody>')
    for metric in ["monetary_severity", "issue_severity"]:
        for model in models:
            mi = _safe_get(interactions, metric, model, default={})
            bias_data = mi.get("bias", {})
            for level in ["low", "medium", "high"]:
                pm = bias_data.get(level, {}).get("persona_means", {})
                if len(pm) < 2:
                    continue
                sorted_pm = sorted(pm.items(), key=lambda x: x[1])
                worst_name, worst_val = sorted_pm[0]
                best_name, best_val = sorted_pm[-1]
                gap = best_val - worst_val
                style = ' style="background:#fdecea;"' if level == "high" and gap > 0.05 else ""
                parts.append(f'<tr{style}>')
                parts.append(f'<td>{metric.replace("_"," ").title()}</td>')
                parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                parts.append(f'<td>{html.escape(str(best_name))}</td><td>{_fmt(best_val, 3)}</td>')
                parts.append(f'<td>{html.escape(str(worst_name))}</td><td>{_fmt(worst_val, 3)}</td>')
                parts.append(f'<td>{_fmt(gap, 3)}</td></tr>')
    parts.append('</tbody></table>')

    # Two-way ANOVA interaction table
    parts.append('<h3>Persona × Severity Interaction Tests</h3>')
    parts.append('<table class="sortable"><thead><tr>')
    parts.append('<th>Metric</th><th>Model</th><th>F</th><th>p-value</th><th>Partial &eta;&sup2;</th><th>Interpretation</th>')
    parts.append('</tr></thead><tbody>')
    for metric in ["monetary_severity", "issue_severity"]:
        for model in models:
            tw = _safe_get(interactions, metric, model, "two_way", default={})
            inter = tw.get("interaction", {})
            p = inter.get("p_value", float("nan"))
            eta = inter.get("eta_sq_partial", 0)
            if np.isfinite(p) and p < 0.05:
                interp = "Significant"
                row_style = ' style="background:#fdecea;"'
            else:
                interp = "Not significant"
                row_style = ""
            if eta >= 0.14:
                interp += " (Large)"
            elif eta >= 0.06:
                interp += " (Medium)"
            elif eta >= 0.01:
                interp += " (Small)"
            parts.append(f'<tr{row_style}><td>{metric.replace("_"," ").title()}</td>')
            parts.append(f'<td>{html.escape(model)}</td>')
            parts.append(f'<td>{_fmt(inter.get("F"))}</td><td>{_fmt(p)}</td>')
            parts.append(f'<td>{_fmt(eta)}</td><td>{interp}</td></tr>')
    parts.append('</tbody></table>')

    # Simple effects summary
    parts.append('<h3>Simple Effects: Persona Effect at Each Severity Level</h3>')
    parts.append('<table class="sortable"><thead><tr>')
    parts.append('<th>Metric</th><th>Model</th><th>Severity</th><th>Partial &eta;&sup2;</th><th>&eta;&sup2; Size</th>')
    parts.append('<th>Largest d</th><th>d Size</th><th>N significant pairs</th><th>N</th>')
    parts.append('</tr></thead><tbody>')
    for metric in ["monetary_severity", "issue_severity"]:
        for model in models:
            se = _safe_get(interactions, metric, model, "simple_effects", default={})
            for level in ["low", "medium", "high"]:
                ld = se.get(level, {})
                parts.append(f'<tr><td>{metric.replace("_"," ").title()}</td>')
                parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                parts.append(f'<td>{_fmt(ld.get("eta_sq_partial"))}</td>')
                parts.append(f'<td>{ld.get("eta_sq_interpretation", "")}</td>')
                parts.append(f'<td>{_fmt(ld.get("largest_pairwise_d"))}</td>')
                parts.append(f'<td>{ld.get("largest_pairwise_d_interpretation", "")}</td>')
                parts.append(f'<td>{ld.get("n_significant", 0)}</td>')
                parts.append(f'<td>{ld.get("n", 0)}</td></tr>')
    parts.append('</tbody></table>')

    # ── Tier 2 Rate Amplification sub-section ──
    interactions_t2 = sev.get("interactions_tier2", {})
    if interactions_t2:
        parts.append('<hr style="margin: 25px 0;">')
        parts.append('<h3>Tier 2 (Escalated) Rate Amplification</h3>')
        parts.append('<p>Parallel amplification analysis using <strong>proportion of complaints '
                     'receiving a Tier 2 (escalated/monetary) recommendation</strong> as the '
                     'fairness measure, instead of mean tier. Bias is measured as the SD of '
                     'Tier 2 rates across personas at each severity level.</p>')

        # Tier 2 metric cards
        parts.append('<div class="metrics-grid">')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                amp = _safe_get(interactions_t2, metric, model, "amplification", default={})
                ratio = amp.get("point", float("nan"))
                ci_lo = amp.get("ci_lower", float("nan"))
                ci_hi = amp.get("ci_upper", float("nan"))
                color, bg, label = _traffic_light(ratio)
                parts.append(f'<div class="metric-card" style="border-left: 4px solid {color}; background: {bg};">')
                parts.append(f'<div class="value" style="color: {color};">{_fmt(ratio, 2)}x</div>')
                parts.append(f'<div class="label">Tier 2 Rate — {metric.replace("_"," ").title()} ({model})</div>')
                parts.append(f'<div style="font-size:0.8rem;">CI: [{_fmt(ci_lo, 2)}, {_fmt(ci_hi, 2)}] — {label}</div>')
                parts.append('</div>')
        parts.append('</div>')

        # Tier 2 amplification bar chart
        chart_t2 = _chart_amplification_bars(
            sev, models, interactions_key="interactions_tier2",
            ylabel="Bias (SD of Tier 2 rates)",
            title="Tier 2 Rate Bias by Severity Level",
        )
        if chart_t2:
            parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{chart_t2}" '
                         f'alt="Tier 2 Rate Amplification"></div>')

        # Tier 2 bias detail table
        parts.append('<h4>Tier 2 Rate Bias by Severity Level</h4>')
        parts.append('<p>Shows how disparity in Tier 2 recommendation rates across personas changes '
                     'with severity. Higher SD at high severity indicates amplification.</p>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Metric</th><th>Model</th><th>Severity</th><th>N</th>'
                     '<th>Tier 2 Rate SD</th><th>Tier 2 Rate Range</th>'
                     '<th>Persona &eta;&sup2;</th><th>Largest d</th><th>Sig. Pairs</th>')
        parts.append('</tr></thead><tbody>')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                mi = _safe_get(interactions_t2, metric, model, default={})
                bias_data = mi.get("bias", {})
                se_data = mi.get("simple_effects", {})
                sds_t2 = {lvl: bias_data.get(lvl, {}).get("sd_across_personas", 0) for lvl in ["low", "medium", "high"]}
                max_sd_level = max(sds_t2, key=lambda k: sds_t2[k]) if any(sds_t2.values()) else None
                for level in ["low", "medium", "high"]:
                    bl = bias_data.get(level, {})
                    sl = se_data.get(level, {})
                    n = bl.get("n", 0)
                    sd = bl.get("sd_across_personas", float("nan"))
                    rng = bl.get("range", float("nan"))
                    eta = sl.get("eta_sq_partial", float("nan"))
                    largest_d = sl.get("largest_pairwise_d", float("nan"))
                    n_sig = sl.get("n_significant", 0)
                    style = ' style="background:#fdecea;font-weight:bold;"' if level == max_sd_level else ""
                    parts.append(f'<tr{style}>')
                    parts.append(f'<td>{metric.replace("_"," ").title()}</td>')
                    parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                    parts.append(f'<td>{n}</td><td>{_fmt(sd, 4)}</td><td>{_fmt(rng, 4)}</td>')
                    parts.append(f'<td>{_fmt(eta, 4)}</td><td>{_fmt(largest_d, 3)}</td>')
                    parts.append(f'<td>{n_sig}</td></tr>')
        parts.append('</tbody></table>')

        # Tier 2 most/least advantaged personas
        parts.append('<h4>Most &amp; Least Advantaged Personas by Tier 2 Rate</h4>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Metric</th><th>Model</th><th>Severity</th>'
                     '<th>Highest Tier 2 Rate</th><th>Rate</th>'
                     '<th>Lowest Tier 2 Rate</th><th>Rate</th><th>Gap</th>')
        parts.append('</tr></thead><tbody>')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                mi = _safe_get(interactions_t2, metric, model, default={})
                bias_data = mi.get("bias", {})
                for level in ["low", "medium", "high"]:
                    pm = bias_data.get(level, {}).get("persona_means", {})
                    if len(pm) < 2:
                        continue
                    sorted_pm = sorted(pm.items(), key=lambda x: x[1])
                    worst_name, worst_val = sorted_pm[0]
                    best_name, best_val = sorted_pm[-1]
                    gap = best_val - worst_val
                    style = ' style="background:#fdecea;"' if level == "high" and gap > 0.02 else ""
                    parts.append(f'<tr{style}>')
                    parts.append(f'<td>{metric.replace("_"," ").title()}</td>')
                    parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                    parts.append(f'<td>{html.escape(str(best_name))}</td><td>{_fmt(best_val, 3)}</td>')
                    parts.append(f'<td>{html.escape(str(worst_name))}</td><td>{_fmt(worst_val, 3)}</td>')
                    parts.append(f'<td>{_fmt(gap, 3)}</td></tr>')
        parts.append('</tbody></table>')

        # Tier 2 two-way ANOVA interaction
        parts.append('<h4>Persona × Severity Interaction Tests (Tier 2 Rate)</h4>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Metric</th><th>Model</th><th>F</th><th>p-value</th><th>Partial &eta;&sup2;</th><th>Interpretation</th>')
        parts.append('</tr></thead><tbody>')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                tw = _safe_get(interactions_t2, metric, model, "two_way", default={})
                inter = tw.get("interaction", {})
                p = inter.get("p_value", float("nan"))
                eta = inter.get("eta_sq_partial", 0)
                if np.isfinite(p) and p < 0.05:
                    interp = "Significant"
                    row_style = ' style="background:#fdecea;"'
                else:
                    interp = "Not significant"
                    row_style = ""
                if eta >= 0.14:
                    interp += " (Large)"
                elif eta >= 0.06:
                    interp += " (Medium)"
                elif eta >= 0.01:
                    interp += " (Small)"
                parts.append(f'<tr{row_style}><td>{metric.replace("_"," ").title()}</td>')
                parts.append(f'<td>{html.escape(model)}</td>')
                parts.append(f'<td>{_fmt(inter.get("F"))}</td><td>{_fmt(p)}</td>')
                parts.append(f'<td>{_fmt(eta)}</td><td>{interp}</td></tr>')
        parts.append('</tbody></table>')

        # Tier 2 simple effects
        parts.append('<h4>Simple Effects: Persona Effect on Tier 2 Rate at Each Severity</h4>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Metric</th><th>Model</th><th>Severity</th><th>Partial &eta;&sup2;</th><th>&eta;&sup2; Size</th>')
        parts.append('<th>Largest d</th><th>d Size</th><th>N significant pairs</th><th>N</th>')
        parts.append('</tr></thead><tbody>')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                se = _safe_get(interactions_t2, metric, model, "simple_effects", default={})
                for level in ["low", "medium", "high"]:
                    ld = se.get(level, {})
                    parts.append(f'<tr><td>{metric.replace("_"," ").title()}</td>')
                    parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                    parts.append(f'<td>{_fmt(ld.get("eta_sq_partial"))}</td>')
                    parts.append(f'<td>{ld.get("eta_sq_interpretation", "")}</td>')
                    parts.append(f'<td>{_fmt(ld.get("largest_pairwise_d"))}</td>')
                    parts.append(f'<td>{ld.get("largest_pairwise_d_interpretation", "")}</td>')
                    parts.append(f'<td>{ld.get("n_significant", 0)}</td>')
                    parts.append(f'<td>{ld.get("n", 0)}</td></tr>')
        parts.append('</tbody></table>')

    parts.append('</div></div>')

    # ── Panel 4: Model Comparison ──
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Panel 4: Model Comparison</div>')
    parts.append('<div class="collapsible-body">')

    chart = _chart_model_amplification(sev, models)
    if chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{chart}" alt="Model Comparison"></div>')

    # Three-way ANOVA results
    tw3 = _safe_get(sev, "three_way", "anova", default={})
    if tw3:
        parts.append('<h3>Three-Way ANOVA (Model × Persona × Severity)</h3>')
        for metric in ["monetary_severity", "issue_severity"]:
            md = tw3.get(metric, {})
            if "error" in md:
                parts.append(f'<p>{metric}: {html.escape(str(md["error"]))}</p>')
                continue
            parts.append(f'<h4>{metric.replace("_"," ").title()}</h4>')
            parts.append('<table><thead><tr><th>Effect</th><th>F</th><th>p-value</th><th>Partial &eta;&sup2;</th></tr></thead><tbody>')
            for effect in ["model", "persona", "severity", "model_x_persona",
                           "model_x_severity", "persona_x_severity", "model_x_persona_x_severity"]:
                ed = md.get(effect, {})
                label = effect.replace("_x_", " × ").replace("_", " ").title()
                parts.append(f'<tr><td>{label}</td><td>{_fmt(ed.get("F"))}</td>')
                parts.append(f'<td>{_fmt(ed.get("p_value"))}</td><td>{_fmt(ed.get("eta_sq_partial"))}</td></tr>')
            parts.append('</tbody></table>')

    parts.append('</div></div>')

    # ── Panel 5: Worst Cases ──
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Panel 5: Worst Cases</div>')
    parts.append('<div class="collapsible-body">')

    wc = _safe_get(sev, "three_way", "worst_case", default={})
    if wc and wc.get("model"):
        parts.append('<h3>Maximum Discrimination Scenario</h3>')
        parts.append('<table><thead><tr><th>Attribute</th><th>Value</th></tr></thead><tbody>')
        parts.append(f'<tr><td>Model</td><td>{html.escape(str(wc.get("model")))}</td></tr>')
        parts.append(f'<tr><td>Severity Metric</td><td>{wc.get("severity_metric", "").replace("_"," ").title()}</td></tr>')
        parts.append(f'<tr><td>Severity Level</td><td>{wc.get("severity_level", "").title()}</td></tr>')
        parts.append(f'<tr><td>Bias Magnitude</td><td>{_fmt(wc.get("bias_magnitude"))}</td></tr>')
        parts.append(f'<tr><td>Most Advantaged Persona</td><td>{html.escape(str(wc.get("most_advantaged")))}</td></tr>')
        parts.append(f'<tr><td>Most Disadvantaged Persona</td><td>{html.escape(str(wc.get("most_disadvantaged")))}</td></tr>')
        parts.append(f'<tr><td>Effect Size (Cohen\'s d)</td><td>{_fmt(wc.get("effect_size_d"))} '
                     f'({html.escape(wc.get("effect_size_interpretation", ""))})</td></tr>')
        parts.append('</tbody></table>')
    else:
        parts.append('<p>No worst-case data available.</p>')

    parts.append('</div></div>')

    # ── Panel 6: Process Fairness ──
    parts.append('<div class="collapsible">')
    parts.append('<div class="collapsible-header">Panel 6: Process Fairness × Severity</div>')
    parts.append('<div class="collapsible-body">')

    chart = _chart_process_severity(sev, models)
    if chart:
        parts.append(f'<div class="chart-container"><img src="data:image/png;base64,{chart}" alt="Process Severity"></div>')

    proc = sev.get("process", {})
    parts.append('<table><thead><tr><th>Metric</th><th>Model</th><th>Process Amplification</th></tr></thead><tbody>')
    for metric in ["monetary_severity", "issue_severity"]:
        for model in models:
            pa = _safe_get(proc, metric, model, "process_amplification", default={})
            ratio = pa.get("point", float("nan"))
            color, _, label = _traffic_light(ratio)
            parts.append(f'<tr><td>{metric.replace("_"," ").title()}</td>')
            parts.append(f'<td>{html.escape(model)}</td>')
            parts.append(f'<td style="color:{color};font-weight:bold;">{_fmt(ratio, 2)}x ({label})</td></tr>')
    parts.append('</tbody></table>')

    # Consistency check
    consist = sev.get("consistency", {})
    findings = consist.get("findings", [])
    if findings:
        parts.append('<h3>Robustness: Consistency Across Severity Metrics</h3>')
        parts.append('<table><thead><tr><th>Finding</th><th>Monetary p/value</th><th>Issue p/value</th><th>Robustness</th></tr></thead><tbody>')
        for f in findings:
            rob = f.get("robustness", "")
            color = "#2ecc71" if rob == "Robust" else ("#e74c3c" if rob == "Metric-specific" else "#95a5a6")
            p_m = f.get("monetary_p", f.get("monetary_value", ""))
            p_i = f.get("issue_p", f.get("issue_value", ""))
            parts.append(f'<tr><td>{html.escape(f.get("finding", ""))}</td>')
            parts.append(f'<td>{_fmt(p_m)}</td><td>{_fmt(p_i)}</td>')
            parts.append(f'<td style="color:{color};font-weight:bold;">{rob}</td></tr>')
        parts.append('</tbody></table>')

    parts.append('</div></div>')

    # ── Panel 7: Mitigation × Severity ──
    mit_sev = sev.get("mitigation_severity", {})
    if mit_sev:
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Panel 7: Severity Amplification After Mitigation</div>')
        parts.append('<div class="collapsible-body">')
        parts.append('<p>Compares severity amplification before and after text neutralization (bias mitigation). '
                     'Verdicts: <strong>eliminated</strong> = neutralized CI includes 1.0 (baseline didn\'t), '
                     '<strong>reduced</strong> = amplification decreased, '
                     '<strong>increased</strong> = amplification worsened, '
                     '<strong>unchanged</strong> = no meaningful change.</p>')

        # Verdict cards
        parts.append('<div class="metrics-grid">')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                ms = _safe_get(mit_sev, metric, model, default={})
                comp = ms.get("comparison", {})
                verdict = comp.get("verdict", "N/A")
                base_amp = ms.get("baseline", {}).get("amplification", {})
                neut_amp = ms.get("neutralized", {}).get("amplification", {})
                base_ratio = base_amp.get("point", float("nan"))
                neut_ratio = neut_amp.get("point", float("nan"))
                delta = comp.get("amplification_change", float("nan"))
                pct = comp.get("amplification_pct_change", float("nan"))
                # Color by verdict
                v_colors = {
                    "eliminated": ("#2ecc71", "#eafaf1"),
                    "reduced": ("#27ae60", "#eafaf1"),
                    "unchanged": ("#f39c12", "#fef9e7"),
                    "increased": ("#e74c3c", "#fdedec"),
                    "insufficient_data": ("#95a5a6", "#f2f3f4"),
                }
                color, bg = v_colors.get(verdict, ("#95a5a6", "#f2f3f4"))
                parts.append(f'<div class="metric-card" style="border-left: 4px solid {color}; background: {bg};">')
                parts.append(f'<div class="value" style="color: {color};">{verdict.upper()}</div>')
                parts.append(f'<div class="label">{metric.replace("_"," ").title()} ({model})</div>')
                parts.append(f'<div style="font-size:0.8rem;">Baseline: {_fmt(base_ratio, 2)}x → '
                             f'Neutralized: {_fmt(neut_ratio, 2)}x</div>')
                if np.isfinite(delta) and np.isfinite(pct):
                    parts.append(f'<div style="font-size:0.8rem;">Delta: {delta:+.2f} ({pct:+.1f}%)</div>')
                parts.append('</div>')
        parts.append('</div>')

        # Before/after bias table
        parts.append('<h3>Bias by Severity Level: Before vs After Mitigation</h3>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Metric</th><th>Model</th><th>Severity</th>'
                     '<th>Baseline Bias SD</th><th>Neutralized Bias SD</th>'
                     '<th>Change</th><th>% Change</th>')
        parts.append('</tr></thead><tbody>')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                ms = _safe_get(mit_sev, metric, model, default={})
                base_bias = ms.get("baseline", {}).get("bias", {})
                neut_bias = ms.get("neutralized", {}).get("bias", {})
                sd_changes = ms.get("comparison", {}).get("bias_sd_change_by_level", {})
                for level in ["low", "medium", "high"]:
                    b_sd = base_bias.get(level, {}).get("sd_across_personas", float("nan"))
                    n_sd = neut_bias.get(level, {}).get("sd_across_personas", float("nan"))
                    d_sd = sd_changes.get(level, float("nan"))
                    pct_chg = (d_sd / b_sd * 100) if np.isfinite(b_sd) and b_sd > 0 and np.isfinite(d_sd) else float("nan")
                    # Highlight rows where bias increased
                    style = ' style="background:#fdedec;"' if np.isfinite(d_sd) and d_sd > 0.001 else ""
                    parts.append(f'<tr{style}>')
                    parts.append(f'<td>{metric.replace("_"," ").title()}</td>')
                    parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                    parts.append(f'<td>{_fmt(b_sd, 4)}</td><td>{_fmt(n_sd, 4)}</td>')
                    parts.append(f'<td>{_fmt(d_sd, 4)}</td>')
                    parts.append(f'<td>{_fmt(pct_chg, 1)}%</td></tr>')
        parts.append('</tbody></table>')

        # Interaction test comparison
        parts.append('<h3>Interaction Test: Before vs After Mitigation</h3>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Metric</th><th>Model</th>'
                     '<th>Baseline F</th><th>Baseline p</th>'
                     '<th>Neutralized F</th><th>Neutralized p</th><th>Weakened?</th>')
        parts.append('</tr></thead><tbody>')
        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                ms = _safe_get(mit_sev, metric, model, default={})
                comp = ms.get("comparison", {})
                base_int = ms.get("baseline", {}).get("interaction", {})
                neut_int = ms.get("neutralized", {}).get("interaction", {})
                weakened = comp.get("interaction_weakened", False)
                w_label = "Yes" if weakened else "No"
                w_color = "#2ecc71" if weakened else "#e74c3c"
                parts.append(f'<tr><td>{metric.replace("_"," ").title()}</td>')
                parts.append(f'<td>{html.escape(model)}</td>')
                parts.append(f'<td>{_fmt(base_int.get("F"))}</td><td>{_fmt(base_int.get("p_value"))}</td>')
                parts.append(f'<td>{_fmt(neut_int.get("F"))}</td><td>{_fmt(neut_int.get("p_value"))}</td>')
                parts.append(f'<td style="color:{w_color};font-weight:bold;">{w_label}</td></tr>')
        parts.append('</tbody></table>')

        # ── Tier 2 Rate Mitigation sub-section ──
        mit_sev_t2 = sev.get("mitigation_severity_tier2", {})
        if mit_sev_t2:
            parts.append('<hr style="margin: 25px 0;">')
            parts.append('<h3>Tier 2 Rate: Before vs After Mitigation</h3>')
            parts.append('<p>Parallel mitigation analysis using Tier 2 recommendation rate as the fairness measure.</p>')

            # Verdict cards
            parts.append('<div class="metrics-grid">')
            for metric in ["monetary_severity", "issue_severity"]:
                for model in models:
                    ms = _safe_get(mit_sev_t2, metric, model, default={})
                    comp = ms.get("comparison", {})
                    verdict = comp.get("verdict", "N/A")
                    base_amp = ms.get("baseline", {}).get("amplification", {})
                    neut_amp = ms.get("neutralized", {}).get("amplification", {})
                    base_ratio = base_amp.get("point", float("nan"))
                    neut_ratio = neut_amp.get("point", float("nan"))
                    delta = comp.get("amplification_change", float("nan"))
                    pct = comp.get("amplification_pct_change", float("nan"))
                    v_colors = {
                        "eliminated": ("#2ecc71", "#eafaf1"),
                        "reduced": ("#27ae60", "#eafaf1"),
                        "unchanged": ("#f39c12", "#fef9e7"),
                        "increased": ("#e74c3c", "#fdedec"),
                        "insufficient_data": ("#95a5a6", "#f2f3f4"),
                    }
                    color, bg = v_colors.get(verdict, ("#95a5a6", "#f2f3f4"))
                    parts.append(f'<div class="metric-card" style="border-left: 4px solid {color}; background: {bg};">')
                    parts.append(f'<div class="value" style="color: {color};">{verdict.upper()}</div>')
                    parts.append(f'<div class="label">Tier 2 Rate — {metric.replace("_"," ").title()} ({model})</div>')
                    parts.append(f'<div style="font-size:0.8rem;">Baseline: {_fmt(base_ratio, 2)}x → '
                                 f'Neutralized: {_fmt(neut_ratio, 2)}x</div>')
                    if np.isfinite(delta) and np.isfinite(pct):
                        parts.append(f'<div style="font-size:0.8rem;">Delta: {delta:+.2f} ({pct:+.1f}%)</div>')
                    parts.append('</div>')
            parts.append('</div>')

            # Tier 2 before/after bias table
            parts.append('<h4>Tier 2 Rate Bias by Severity: Before vs After Mitigation</h4>')
            parts.append('<table class="sortable"><thead><tr>')
            parts.append('<th>Metric</th><th>Model</th><th>Severity</th>'
                         '<th>Baseline Bias SD</th><th>Neutralized Bias SD</th>'
                         '<th>Change</th><th>% Change</th>')
            parts.append('</tr></thead><tbody>')
            for metric in ["monetary_severity", "issue_severity"]:
                for model in models:
                    ms = _safe_get(mit_sev_t2, metric, model, default={})
                    base_bias = ms.get("baseline", {}).get("bias", {})
                    neut_bias = ms.get("neutralized", {}).get("bias", {})
                    sd_changes = ms.get("comparison", {}).get("bias_sd_change_by_level", {})
                    for level in ["low", "medium", "high"]:
                        b_sd = base_bias.get(level, {}).get("sd_across_personas", float("nan"))
                        n_sd = neut_bias.get(level, {}).get("sd_across_personas", float("nan"))
                        d_sd = sd_changes.get(level, float("nan"))
                        pct_chg = (d_sd / b_sd * 100) if np.isfinite(b_sd) and b_sd > 0 and np.isfinite(d_sd) else float("nan")
                        style = ' style="background:#fdedec;"' if np.isfinite(d_sd) and d_sd > 0.001 else ""
                        parts.append(f'<tr{style}>')
                        parts.append(f'<td>{metric.replace("_"," ").title()}</td>')
                        parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                        parts.append(f'<td>{_fmt(b_sd, 4)}</td><td>{_fmt(n_sd, 4)}</td>')
                        parts.append(f'<td>{_fmt(d_sd, 4)}</td>')
                        parts.append(f'<td>{_fmt(pct_chg, 1)}%</td></tr>')
            parts.append('</tbody></table>')

        parts.append('</div></div>')

    # ── Panel 8: Power Analysis ──
    pa = analysis_results.get("power_analysis", {})
    if pa:
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Panel 8: Statistical Power Analysis</div>')
        parts.append('<div class="collapsible-body">')
        parts.append('<p>Post-hoc power analysis showing the minimum effect sizes detectable '
                     'with the current sample at 80% power, &alpha;=0.05. This documents '
                     'what effects the study is adequately powered to find.</p>')

        # Sample summary cards
        ss = pa.get("sample_summary", {})
        parts.append('<div class="metrics-grid">')
        for label, key in [("Total Complaints", "total_complaints"),
                           ("Persona Variants", "n_personas"),
                           ("Total Decisions", "total_persona_decisions"),
                           ("Causal Sample", "causal_sample")]:
            parts.append(f'<div class="metric-card"><div class="value">{ss.get(key, 0):,}</div>'
                         f'<div class="label">{label}</div></div>')
        parts.append('</div>')

        # Matched-guise
        mg = pa.get("matched_guise", {})
        parts.append('<h3>Matched-Guise (Within-Complaint) Design</h3>')
        parts.append('<p>Each complaint is evaluated under multiple persona framings, enabling '
                     'paired within-complaint comparisons.</p>')
        parts.append('<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>')
        parts.append(f'<tr><td>N complaint pairs</td><td>{mg.get("n_pairs", 0):,}</td></tr>')
        parts.append(f'<tr><td>Min Detectable Cohen\'s d</td><td>{_fmt(mg.get("min_detectable_d"))} '
                     f'({html.escape(mg.get("interpretation", ""))})</td></tr>')
        parts.append(f'<tr><td>Design</td><td>{html.escape(mg.get("design", ""))}</td></tr>')
        parts.append('</tbody></table>')

        # Between-persona table
        bp = pa.get("between_persona", {})
        if bp:
            parts.append('<h3>Between-Persona Comparisons (per Dimension)</h3>')
            parts.append('<table><thead><tr><th>Dimension</th><th>Min Cell Size</th>'
                         '<th>Min Detectable d</th><th>Interpretation</th></tr></thead><tbody>')
            for dim, info in sorted(bp.items()):
                parts.append(f'<tr><td>{dim.title()}</td><td>{info["n_per_cell"]:,}</td>'
                             f'<td>{_fmt(info["min_detectable_d"])}</td>'
                             f'<td>{info["interpretation"].title()}</td></tr>')
            parts.append('</tbody></table>')

        # ANOVA power
        av = pa.get("anova", {})
        if av:
            parts.append('<h3>ANOVA Power (k-Group Comparisons)</h3>')
            parts.append('<table><thead><tr><th>Dimension</th><th>k Groups</th><th>N per Group</th>'
                         '<th>Min f</th><th>Min &eta;&sup2;</th><th>Interpretation</th></tr></thead><tbody>')
            for dim, info in sorted(av.items()):
                parts.append(f'<tr><td>{dim.title()}</td><td>{info["k_groups"]}</td>'
                             f'<td>{info["n_per_group"]:,}</td>'
                             f'<td>{_fmt(info["min_detectable_f"])}</td>'
                             f'<td>{_fmt(info["min_detectable_eta_sq"])}</td>'
                             f'<td>{info["interpretation"].title()}</td></tr>')
            parts.append('</tbody></table>')

        parts.append('</div></div>')

    # ── Panel 9: Multiple Comparisons Correction ──
    mc = analysis_results.get("multiple_comparisons", {})
    if mc:
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Panel 9: Multiple Comparisons Correction (Family-Wise Error Rate)</div>')
        parts.append('<div class="collapsible-body">')
        parts.append(f'<p>{html.escape(mc.get("method_description", ""))}</p>')

        # Summary cards
        parts.append('<div class="metrics-grid">')
        parts.append(f'<div class="metric-card"><div class="value">{mc.get("total_tests", 0):,}</div>'
                     f'<div class="label">Total Tests</div></div>')
        parts.append(f'<div class="metric-card"><div class="value">{mc.get("total_surviving", 0):,}</div>'
                     f'<div class="label">Surviving FDR</div></div>')
        rate = mc["total_surviving"] / mc["total_tests"] if mc.get("total_tests", 0) > 0 else 0
        parts.append(f'<div class="metric-card"><div class="value">{rate:.0%}</div>'
                     f'<div class="label">Survival Rate</div></div>')
        parts.append(f'<div class="metric-card"><div class="value">{html.escape(mc.get("method", ""))}</div>'
                     f'<div class="label">Method</div></div>')
        parts.append('</div>')

        # Family detail table
        parts.append('<h3>Test Families</h3>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Test Family</th><th>N Tests</th><th>N Surviving</th><th>Survival Rate</th>')
        parts.append('</tr></thead><tbody>')
        for fam in mc.get("families", []):
            sr = fam.get("survival_rate", 0)
            color = "#e74c3c" if sr > 0.5 else ("#f39c12" if sr > 0.1 else "#27ae60")
            parts.append(f'<tr><td>{html.escape(fam["family"])}</td>'
                         f'<td>{fam["n_tests"]}</td><td>{fam["n_surviving"]}</td>'
                         f'<td style="color:{color};font-weight:bold;">{sr:.0%}</td></tr>')
        parts.append('</tbody></table>')

        parts.append('</div></div>')

    # ── Panel 10: Dimension-Specific Severity Amplification ──
    dim_sev = sev.get("dimension_severity", {})
    if dim_sev:
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Panel 10: Dimension-Specific Severity Amplification (One-Way)</div>')
        parts.append('<div class="collapsible-body">')
        parts.append('<p>Tests severity amplification for each demographic dimension individually '
                     '(age, geographic, education, socioeconomic). Each cell has ~200+ observations, '
                     'providing much better statistical power than the full 36-persona grid. '
                     'The tradeoff: intersectional effects (e.g. old + rural + low-SES) are not captured, '
                     'but these tests can definitively establish whether severity amplifies bias at all.</p>')

        # Interaction summary table
        parts.append('<h3>Severity &times; Dimension Interaction Tests</h3>')
        parts.append('<table class="sortable"><thead><tr>')
        parts.append('<th>Metric</th><th>Model</th><th>Dimension</th>')
        parts.append('<th>Interaction F</th><th>Interaction p</th><th>Partial &eta;&sup2;</th>')
        parts.append('<th>&eta;&sup2; Size</th><th>Amplification</th><th>95% CI</th><th>Min Cell</th>')
        parts.append('</tr></thead><tbody>')

        for metric in ["monetary_severity", "issue_severity"]:
            for model in models:
                for dim in ["age", "geographic", "education", "socioeconomic"]:
                    ds = _safe_get(dim_sev, metric, model, dim, default={})
                    if "error" in ds:
                        continue
                    anova = ds.get("anova", {})
                    inter = anova.get("interaction", {})
                    amp = ds.get("amplification", {})
                    p = inter.get("p_value", float("nan"))
                    row_style = ' style="background:#fdecea;"' if np.isfinite(p) and p < 0.05 else ""
                    parts.append(f'<tr{row_style}>')
                    parts.append(f'<td>{metric.replace("_"," ").title()}</td>')
                    parts.append(f'<td>{html.escape(model)}</td>')
                    parts.append(f'<td>{dim.title()}</td>')
                    parts.append(f'<td>{_fmt(inter.get("F"))}</td>')
                    parts.append(f'<td>{_fmt(p)}</td>')
                    parts.append(f'<td>{_fmt(inter.get("eta_sq_partial"))}</td>')
                    parts.append(f'<td>{inter.get("eta_sq_interpretation", "")}</td>')
                    ratio = amp.get("point", float("nan"))
                    color, _, label = _traffic_light(ratio)
                    parts.append(f'<td style="color:{color};font-weight:bold;">{_fmt(ratio, 2)}x</td>')
                    ci_lo = amp.get("ci_lower", float("nan"))
                    ci_hi = amp.get("ci_upper", float("nan"))
                    parts.append(f'<td>[{_fmt(ci_lo, 2)}, {_fmt(ci_hi, 2)}]</td>')
                    parts.append(f'<td>{ds.get("min_cell_size", 0)}</td>')
                    parts.append('</tr>')
        parts.append('</tbody></table>')

        # Amplification ratio cards per dimension
        parts.append('<h3>Amplification Ratios by Dimension</h3>')
        parts.append('<div class="metrics-grid">')
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            for model in models[:1]:  # Show first model as representative
                for metric in ["issue_severity"]:
                    ds = _safe_get(dim_sev, metric, model, dim, default={})
                    if "error" in ds:
                        continue
                    amp = ds.get("amplification", {})
                    ratio = amp.get("point", float("nan"))
                    color, bg, label = _traffic_light(ratio)
                    parts.append(f'<div class="metric-card" style="border-left: 4px solid {color}; background: {bg};">')
                    parts.append(f'<div class="value" style="color: {color};">{_fmt(ratio, 2)}x</div>')
                    parts.append(f'<div class="label">{dim.title()} ({model})</div>')
                    ci_lo = amp.get("ci_lower", float("nan"))
                    ci_hi = amp.get("ci_upper", float("nan"))
                    parts.append(f'<div style="font-size:0.8rem;">CI: [{_fmt(ci_lo, 2)}, {_fmt(ci_hi, 2)}] &mdash; {label}</div>')
                    parts.append('</div>')
        parts.append('</div>')

        # Detail: mean tier by dimension level at each severity
        parts.append('<h3>Mean Tier by Dimension Level and Severity</h3>')
        for dim in ["age", "geographic", "education", "socioeconomic"]:
            parts.append(f'<h4>{dim.title()}</h4>')
            # Determine dimension levels from the data
            ds_sample = _safe_get(dim_sev, "issue_severity", models[0] if models else "", dim, default={})
            dim_levels = sorted(set(
                k for sv in ds_sample.get("dimension_means_by_severity", {}).values()
                for k in sv.keys()
            )) if ds_sample.get("dimension_means_by_severity") else []
            if not dim_levels:
                parts.append('<p>No data available.</p>')
                continue

            parts.append('<table><thead><tr><th>Metric</th><th>Model</th><th>Severity</th>')
            for dl in dim_levels:
                parts.append(f'<th>{html.escape(str(dl))}</th>')
            parts.append('<th>Gap</th></tr></thead><tbody>')

            for metric in ["monetary_severity", "issue_severity"]:
                for model in models:
                    ds = _safe_get(dim_sev, metric, model, dim, default={})
                    means_by_sev = ds.get("dimension_means_by_severity", {})
                    for level in ["low", "medium", "high"]:
                        means = means_by_sev.get(level, {})
                        parts.append(f'<tr><td>{metric.replace("_"," ").title()}</td>')
                        parts.append(f'<td>{html.escape(model)}</td><td>{level.title()}</td>')
                        vals = []
                        for dl in dim_levels:
                            v = means.get(dl, float("nan"))
                            vals.append(v)
                            parts.append(f'<td>{_fmt(v, 3)}</td>')
                        gap = max(vals) - min(vals) if vals and all(np.isfinite(v) for v in vals) else float("nan")
                        style = ' style="font-weight:bold;"' if level == "high" else ""
                        parts.append(f'<td{style}>{_fmt(gap, 3)}</td></tr>')
            parts.append('</tbody></table>')

        parts.append('</div></div>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tab: Sparsity-Aware Analysis
# ---------------------------------------------------------------------------

def _build_tab_sparsity(
    analysis_results: Dict[str, Any],
    models: List[str],
) -> str:
    """Build the Sparsity-Aware Analysis tab content."""
    parts: List[str] = []
    sparsity = analysis_results.get("sparsity", {})
    if not sparsity:
        parts.append("<p>Sparsity analysis not available.</p>")
        return "\n".join(parts)

    parts.append("<h2>Sparsity-Aware Fairness Analysis</h2>")
    parts.append(
        "<p>This analysis applies a defensible sparsity policy to handle small-cell "
        "instability in observed/inferred group membership. Rates, AIR values, and "
        "disparity flags are annotated with support counts, confidence intervals, "
        "and stability indicators.</p>"
    )
    parts.append(
        '<p><b>Thresholds:</b> '
        '<span style="color:#e74c3c">n&lt;20 = insufficient support</span> | '
        '<span style="color:#e67e22">20&le;n&lt;100 = screening only</span> | '
        '<span style="color:#2ecc71">n&ge;100 = reliable</span></p>'
    )

    for model in models:
        mr = sparsity.get(model, {})
        if not mr:
            continue

        parts.append(f"<h3>Model: {html.escape(model)}</h3>")

        # ---- Panel 1: Support Summary ----
        parts.append('<div class="collapsible">')
        parts.append('<div class="collapsible-header">Support Summary &amp; Stability Flags</div>')
        parts.append('<div class="collapsible-body">')

        obs = mr.get("observed_persona", {})
        support = obs.get("support", {})
        cells = support.get("cells", {})
        if cells:
            parts.append(
                f'<p>Total groups: {support.get("n_groups", 0)} | '
                f'<span style="color:#e74c3c">Insufficient: {support.get("n_insufficient", 0)}</span> | '
                f'<span style="color:#e67e22">Screening: {support.get("n_screening", 0)}</span> | '
                f'<span style="color:#2ecc71">Reliable: {support.get("n_reliable", 0)}</span></p>'
            )
            parts.append('<table><thead><tr>'
                         '<th>Persona</th><th>n</th><th>Stability</th>'
                         '</tr></thead><tbody>')
            for persona in sorted(cells, key=lambda p: cells[p]["n"]):
                cell = cells[persona]
                stab = cell["stability"]
                color = {"insufficient_support": "#e74c3c", "screening_only": "#e67e22", "reliable": "#2ecc71"}.get(stab, "#333")
                parts.append(
                    f'<tr><td>{html.escape(persona)}</td>'
                    f'<td>{cell["n"]}</td>'
                    f'<td style="color:{color};font-weight:bold">{html.escape(stab)}</td></tr>'
                )
            parts.append("</tbody></table>")
        else:
            parts.append("<p>No observed persona support data available.</p>")

        parts.append("</div></div>")  # collapsible

        # ---- Panel 2: Annotated Rates with Intervals ----
        parts.append('<div class="collapsible collapsed">')
        parts.append('<div class="collapsible-header">Annotated Rates with Confidence Intervals</div>')
        parts.append('<div class="collapsible-body">')

        ann_rates = obs.get("annotated_rates", {})
        if ann_rates:
            parts.append(
                "<p>Each rate is shown with its Wilson score interval and Jeffreys "
                "credible interval. Groups marked <em>insufficient_support</em> "
                "should not be used for inferential claims.</p>"
            )
            parts.append('<table><thead><tr>'
                         '<th>Persona</th><th>n</th><th>Rate</th>'
                         '<th>Wilson 95% CI</th><th>Jeffreys 95% CI</th>'
                         '<th>Stability</th>'
                         '</tr></thead><tbody>')
            for persona in sorted(ann_rates, key=lambda p: ann_rates[p].get("rate", 0)):
                r = ann_rates[persona]
                stab = r.get("stability", "")
                color = {"insufficient_support": "#e74c3c", "screening_only": "#e67e22", "reliable": "#2ecc71"}.get(stab, "#333")
                w_ci = r.get("wilson_ci", (0, 0))
                j_ci = r.get("jeffreys_ci", (0, 0))
                parts.append(
                    f'<tr><td>{html.escape(persona)}</td>'
                    f'<td>{r.get("n", 0)}</td>'
                    f'<td>{_fmt(r.get("rate"), 3)}</td>'
                    f'<td>[{_fmt(w_ci[0], 3)}, {_fmt(w_ci[1], 3)}]</td>'
                    f'<td>[{_fmt(j_ci[0], 3)}, {_fmt(j_ci[1], 3)}]</td>'
                    f'<td style="color:{color}">{html.escape(stab)}</td></tr>'
                )
            parts.append("</tbody></table>")

        parts.append("</div></div>")

        # ---- Panel 3: Bayesian Interval AIR ----
        parts.append('<div class="collapsible collapsed">')
        parts.append('<div class="collapsible-header">Bayesian Interval AIR (Jeffreys Prior)</div>')
        parts.append('<div class="collapsible-body">')

        bair = mr.get("bayesian_air_persona", {})
        if bair:
            parts.append(
                "<p>AIR computed on smoothed rates using Jeffreys Beta(0.5, 0.5) prior. "
                "P(AIR&lt;0.8) gives a graded risk probability instead of binary flagging.</p>"
            )
            parts.append('<table><thead><tr>'
                         '<th>Persona</th><th>n</th><th>Rate</th>'
                         '<th>Point AIR</th><th>Median AIR</th>'
                         '<th>95% Credible Interval</th>'
                         '<th>P(AIR&lt;0.8)</th><th>Stability</th>'
                         '</tr></thead><tbody>')
            for persona in sorted(bair, key=lambda p: bair[p].get("bayesian_air", {}).get("median_air", 999)):
                info = bair[persona]
                ba = info.get("bayesian_air", {})
                stab = ba.get("stability_g", "")
                p_below = ba.get("p_below_threshold", float("nan"))
                color = {"insufficient_support": "#e74c3c", "screening_only": "#e67e22", "reliable": "#2ecc71"}.get(stab, "#333")
                # Color-code P(AIR<0.8) risk
                if isinstance(p_below, float) and not np.isnan(p_below):
                    if p_below > 0.5:
                        risk_color = "#e74c3c"
                    elif p_below > 0.1:
                        risk_color = "#e67e22"
                    else:
                        risk_color = "#2ecc71"
                    risk_str = f'<span style="color:{risk_color};font-weight:bold">{_pct(p_below)}</span>'
                else:
                    risk_str = "N/A"
                parts.append(
                    f'<tr><td>{html.escape(persona)}</td>'
                    f'<td>{info.get("n", 0)}</td>'
                    f'<td>{_fmt(info.get("rate"), 3)}</td>'
                    f'<td>{_fmt(info.get("point_air"), 3)}</td>'
                    f'<td>{_fmt(ba.get("median_air"), 3)}</td>'
                    f'<td>[{_fmt(ba.get("ci_lower"), 3)}, {_fmt(ba.get("ci_upper"), 3)}]</td>'
                    f'<td>{risk_str}</td>'
                    f'<td style="color:{color}">{html.escape(stab)}</td></tr>'
                )
            parts.append("</tbody></table>")

            # Risk summary chart
            try:
                fig, ax = plt.subplots(figsize=(10, max(4, len(bair) * 0.25)))
                personas_sorted = sorted(bair, key=lambda p: bair[p].get("bayesian_air", {}).get("median_air", 999))
                y_pos = range(len(personas_sorted))
                medians = [bair[p].get("bayesian_air", {}).get("median_air", 0) for p in personas_sorted]
                ci_lo = [bair[p].get("bayesian_air", {}).get("ci_lower", 0) for p in personas_sorted]
                ci_hi = [bair[p].get("bayesian_air", {}).get("ci_upper", 0) for p in personas_sorted]
                errors = [[m - lo for m, lo in zip(medians, ci_lo)],
                          [hi - m for m, hi in zip(medians, ci_hi)]]
                colors = []
                for p in personas_sorted:
                    pb = bair[p].get("bayesian_air", {}).get("p_below_threshold", 0)
                    if isinstance(pb, float) and not np.isnan(pb):
                        colors.append("#e74c3c" if pb > 0.5 else ("#e67e22" if pb > 0.1 else "#2ecc71"))
                    else:
                        colors.append("#95a5a6")
                ax.barh(y_pos, medians, xerr=errors, color=colors, alpha=0.8, height=0.6, capsize=3)
                ax.axvline(x=0.8, color="#333", linestyle="--", linewidth=1, label="AIR = 0.80 threshold")
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels([p.replace("_", " ") for p in personas_sorted], fontsize=7)
                ax.set_xlabel("Bayesian AIR (median + 95% CI)")
                ax.set_title(f"Bayesian AIR by Persona ({html.escape(model)})")
                ax.legend(fontsize=8)
                fig.tight_layout()
                img_b64 = _fig_to_base64(fig)
                parts.append(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%">')
            except Exception as e:
                logger.warning("Bayesian AIR chart failed: %s", e)

        parts.append("</div></div>")

        # ---- Panel 4: Partial Pooling ----
        parts.append('<div class="collapsible collapsed">')
        parts.append('<div class="collapsible-header">Partial Pooling (Hierarchical Model)</div>')
        parts.append('<div class="collapsible-body">')

        pooling = mr.get("partial_pooling", {})
        if pooling.get("error"):
            parts.append(f'<p>Partial pooling: {html.escape(str(pooling["error"]))}</p>')
        elif pooling.get("pooled_rates"):
            shrinkage = pooling.get("shrinkage_summary", {})
            parts.append(
                "<p>Hierarchical model shrinks small-cell estimates toward the grand mean, "
                "stabilising extreme rates from tiny groups. "
                f'Grand mean: {_fmt(pooling.get("grand_mean"), 3)} | '
                f'Random effect variance: {_fmt(pooling.get("random_effect_variance"), 4)}</p>'
            )
            if shrinkage:
                parts.append(
                    f'<p>Mean |shrinkage|: {_fmt(shrinkage.get("mean_abs_shrinkage"), 4)} | '
                    f'Max |shrinkage|: {_fmt(shrinkage.get("max_abs_shrinkage"), 4)} | '
                    f'Cells shrunk &gt;5%: {shrinkage.get("n_cells_shrunk_gt_5pct", 0)}</p>'
                )

            pr = pooling["pooled_rates"]
            parts.append('<table><thead><tr>'
                         '<th>Persona</th><th>n</th><th>Raw Rate</th>'
                         '<th>Pooled Estimate</th><th>Shrinkage</th><th>Stability</th>'
                         '</tr></thead><tbody>')
            for persona in sorted(pr, key=lambda p: abs(pr[p].get("shrinkage", 0)), reverse=True):
                r = pr[persona]
                stab = r.get("stability", "")
                color = {"insufficient_support": "#e74c3c", "screening_only": "#e67e22", "reliable": "#2ecc71"}.get(stab, "#333")
                shr = r.get("shrinkage", 0)
                shr_color = "#e74c3c" if abs(shr) > 0.05 else "#333"
                parts.append(
                    f'<tr><td>{html.escape(persona)}</td>'
                    f'<td>{r.get("raw_n", 0)}</td>'
                    f'<td>{_fmt(r.get("raw_rate"), 3)}</td>'
                    f'<td>{_fmt(r.get("pooled_estimate"), 3)}</td>'
                    f'<td style="color:{shr_color}">{_fmt(shr, 4)}</td>'
                    f'<td style="color:{color}">{html.escape(stab)}</td></tr>'
                )
            parts.append("</tbody></table>")

        parts.append("</div></div>")

        # ---- Panel 5: Sensitivity Analysis ----
        parts.append('<div class="collapsible collapsed">')
        parts.append('<div class="collapsible-header">Sensitivity Analysis (Robustness Check)</div>')
        parts.append('<div class="collapsible-body">')

        sens = mr.get("sensitivity_persona", {})
        verdict = sens.get("verdict", {})
        if verdict:
            robust = verdict.get("robust", False)
            badge_color = "#2ecc71" if robust else "#e74c3c"
            badge_text = "ROBUST" if robust else "CAUTION"
            parts.append(
                f'<p><span style="background:{badge_color};color:#fff;padding:2px 8px;'
                f'border-radius:4px;font-weight:bold">{badge_text}</span> '
                f'{html.escape(verdict.get("summary", ""))}</p>'
            )

        # Baseline vs filtered comparison
        baseline = sens.get("baseline", {})
        filtered = sens.get("exclude_small_cells", {})
        if baseline.get("air_by_group") and filtered.get("air_by_group"):
            n_excluded = filtered.get("n_groups_excluded", 0)
            excluded_groups = filtered.get("groups_excluded", [])
            parts.append(
                f"<p><b>Rule 1 (Exclude n&lt;20):</b> {n_excluded} group(s) excluded"
            )
            if excluded_groups:
                parts.append(f' ({", ".join(html.escape(g) for g in excluded_groups[:10])})')
            parts.append("</p>")

            parts.append(
                f'<p>Material disparities: baseline={baseline.get("n_material", 0)}, '
                f'after filtering={filtered.get("n_material", 0)}</p>'
            )

        # Collapse results
        collapse = sens.get("collapse_sparse", {})
        if collapse:
            parts.append("<p><b>Rule 2 (Collapse sparse categories):</b></p>")
            parts.append("<ul>")
            for dim, info in sorted(collapse.items()):
                if isinstance(info, dict) and info.get("collapsed"):
                    parts.append(
                        f'<li><b>{html.escape(dim)}:</b> merged '
                        f'{", ".join(html.escape(str(l)) for l in info.get("sparse_levels", []))} '
                        f'into {html.escape(str(info.get("merge_target", "N/A")))}</li>'
                    )
                elif isinstance(info, dict) and not info.get("collapsed"):
                    parts.append(f'<li><b>{html.escape(dim)}:</b> no sparse levels</li>')
            parts.append("</ul>")

        parts.append("</div></div>")

        # ---- Panel 6: Per-Dimension Analysis ----
        parts.append('<div class="collapsible collapsed">')
        parts.append('<div class="collapsible-header">Per-Dimension Sparsity Analysis</div>')
        parts.append('<div class="collapsible-body">')

        per_dim = mr.get("per_dimension_observed", {})
        for dim in ESTIMATE_DIMS:
            dim_data = per_dim.get(dim, {})
            if not dim_data:
                continue
            parts.append(f"<h4>{html.escape(dim)}</h4>")
            bair_dim = dim_data.get("bayesian_air", {})
            if bair_dim:
                parts.append('<table><thead><tr>'
                             '<th>Level</th><th>n</th><th>Rate</th>'
                             '<th>Median AIR</th><th>P(AIR&lt;0.8)</th><th>Stability</th>'
                             '</tr></thead><tbody>')
                for level in sorted(bair_dim):
                    info = bair_dim[level]
                    ba = info.get("bayesian_air", {})
                    stab = ba.get("stability_g", "")
                    color = {"insufficient_support": "#e74c3c", "screening_only": "#e67e22", "reliable": "#2ecc71"}.get(stab, "#333")
                    p_below = ba.get("p_below_threshold", float("nan"))
                    risk_str = _pct(p_below) if isinstance(p_below, float) and not np.isnan(p_below) else "N/A"
                    parts.append(
                        f'<tr><td>{html.escape(level)}</td>'
                        f'<td>{info.get("n", 0)}</td>'
                        f'<td>{_fmt(info.get("rate"), 3)}</td>'
                        f'<td>{_fmt(ba.get("median_air"), 3)}</td>'
                        f'<td>{risk_str}</td>'
                        f'<td style="color:{color}">{html.escape(stab)}</td></tr>'
                    )
                parts.append("</tbody></table>")

        parts.append("</div></div>")

        # ---- Panel 7: Mitigation Comparison (if available) ----
        mit = mr.get("mitigation_sensitivity", {})
        if mit:
            parts.append('<div class="collapsible collapsed">')
            parts.append('<div class="collapsible-header">Mitigation Sparsity-Aware Comparison</div>')
            parts.append('<div class="collapsible-body">')

            comparison = mit.get("comparison", {})
            if comparison:
                parts.append(
                    "<p>Compares Bayesian AIR before and after text neutralisation. "
                    "P(AIR&lt;0.8) gives a graded risk measure.</p>"
                )
                parts.append('<table><thead><tr>'
                             '<th>Persona</th>'
                             '<th>Pre Median AIR</th><th>Post Median AIR</th>'
                             '<th>Pre P(AIR&lt;0.8)</th><th>Post P(AIR&lt;0.8)</th>'
                             '<th>Stability</th>'
                             '</tr></thead><tbody>')
                for persona in sorted(comparison):
                    c = comparison[persona]
                    stab = c.get("pre_stability", "")
                    color = {"insufficient_support": "#e74c3c", "screening_only": "#e67e22", "reliable": "#2ecc71"}.get(stab, "#333")
                    parts.append(
                        f'<tr><td>{html.escape(persona)}</td>'
                        f'<td>{_fmt(c.get("pre_median_air"), 3)}</td>'
                        f'<td>{_fmt(c.get("post_median_air"), 3)}</td>'
                        f'<td>{_pct(c.get("pre_p_below_08"))}</td>'
                        f'<td>{_pct(c.get("post_p_below_08"))}</td>'
                        f'<td style="color:{color}">{html.escape(stab)}</td></tr>'
                    )
                parts.append("</tbody></table>")

            parts.append("</div></div>")

    # ---- Process discrimination sparsity ----
    proc_sparsity = sparsity.get("process_sparsity", {})
    if proc_sparsity:
        parts.append('<div class="collapsible collapsed">')
        parts.append('<div class="collapsible-header">Process Discrimination: Sparsity-Aware Info Request AIR</div>')
        parts.append('<div class="collapsible-body">')

        for model_name in models:
            pm = proc_sparsity.get(model_name, {})
            bair_proc = pm.get("bayesian_air", {})
            if not bair_proc:
                continue
            parts.append(f"<h4>Model: {html.escape(model_name)}</h4>")
            parts.append('<table><thead><tr>'
                         '<th>Persona</th><th>n</th><th>Info Request Rate</th>'
                         '<th>Median AIR</th><th>P(AIR&lt;0.8)</th><th>Stability</th>'
                         '</tr></thead><tbody>')
            for persona in sorted(bair_proc, key=lambda p: bair_proc[p].get("bayesian_air", {}).get("median_air", 999)):
                info = bair_proc[persona]
                ba = info.get("bayesian_air", {})
                stab = ba.get("stability_g", "")
                color = {"insufficient_support": "#e74c3c", "screening_only": "#e67e22", "reliable": "#2ecc71"}.get(stab, "#333")
                p_below = ba.get("p_below_threshold", float("nan"))
                risk_str = _pct(p_below) if isinstance(p_below, float) and not np.isnan(p_below) else "N/A"
                parts.append(
                    f'<tr><td>{html.escape(persona)}</td>'
                    f'<td>{info.get("n", 0)}</td>'
                    f'<td>{_fmt(info.get("rate"), 3)}</td>'
                    f'<td>{_fmt(ba.get("median_air"), 3)}</td>'
                    f'<td>{risk_str}</td>'
                    f'<td style="color:{color}">{html.escape(stab)}</td></tr>'
                )
            parts.append("</tbody></table>")

        parts.append("</div></div>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_tab_repeated_measures(
    analysis_results: Dict[str, Any],
    models: List[str],
) -> str:
    """Build the Repeated-Measures Causal Analysis tab HTML."""
    parts: List[str] = []
    rm = analysis_results.get("repeated_measures", {})

    parts.append("<h2>Repeated-Measures Causal Analysis</h2>")
    parts.append(
        '<p class="note">This section presents dependence-aware models that explicitly '
        "account for repeated measures within each complaint. The same underlying complaint "
        "is observed under multiple persona framings, so <code>complaint_id</code>-level "
        "clustering is non-negotiable for valid inference. Models include: "
        "(A)&nbsp;ordinal model with clustered SEs, "
        "(B)&nbsp;conditional logistic / GEE for binary outcomes, "
        "(C)&nbsp;GEE for info-request burden, "
        "(D)&nbsp;linear mixed model (robustness), and "
        "(E)&nbsp;nonparametric repeated-measures tests (Friedman, Cochran&rsquo;s Q).</p>"
    )

    if rm.get("skipped"):
        parts.append(f'<p class="warning">Repeated-measures analysis skipped: {html.escape(str(rm.get("reason", "")))}</p>')
        return "\n".join(parts)

    # Data summary
    ds = rm.get("data_summary", {})
    if ds:
        parts.append('<div class="metrics-grid">')
        parts.append(f'<div class="metric-card"><div class="value">{ds.get("n_rows", 0):,}</div><div class="label">Observations</div></div>')
        parts.append(f'<div class="metric-card"><div class="value">{ds.get("n_complaints", 0):,}</div><div class="label">Complaints (clusters)</div></div>')
        parts.append(f'<div class="metric-card"><div class="value">{ds.get("n_personas", 0)}</div><div class="label">Persona Variants</div></div>')
        parts.append(f'<div class="metric-card"><div class="value">{ds.get("n_models", 0)}</div><div class="label">LLM Models</div></div>')
        parts.append("</div>")

    for model in models:
        mr = rm.get(model, {})
        if not mr or mr.get("skipped"):
            continue
        parts.append(f"<h3>Model: {html.escape(model)}</h3>")

        # --- A) Ordinal model ---
        om = mr.get("ordinal_model", {})
        if om and not om.get("skipped"):
            parts.append('<div class="collapsible">')
            parts.append('<div class="collapsible-header">A) Ordinal Model (Outcome Tier)</div>')
            parts.append('<div class="collapsible-body">')
            mi = om.get("model_info", {})
            parts.append(f'<p><strong>Method:</strong> {html.escape(mi.get("method", ""))}</p>')
            parts.append(f'<p><strong>Clustering:</strong> {html.escape(mi.get("clustering", ""))}</p>')
            parts.append(f'<p>N = {om.get("n_obs", 0):,}, clusters = {om.get("n_clusters", 0):,}</p>')
            _render_coefficient_table(parts, om)
            parts.append("</div></div>")

        # --- B) Conditional logistic ---
        cl = mr.get("conditional_logistic_favourable", {})
        if cl and not cl.get("skipped"):
            parts.append('<div class="collapsible">')
            parts.append('<div class="collapsible-header">B1) Conditional Logistic (Favourable)</div>')
            parts.append('<div class="collapsible-body">')
            mi = cl.get("model_info", {})
            parts.append(f'<p><strong>Method:</strong> {html.escape(mi.get("method", ""))}</p>')
            parts.append(f'<p>N = {cl.get("n_obs", 0):,}, strata = {cl.get("n_strata", 0):,}, informative = {cl.get("n_informative_strata", 0):,}</p>')
            _render_coefficient_table(parts, cl, show_or=True)
            parts.append("</div></div>")

        # --- B2) GEE favourable ---
        gee_fav = mr.get("logistic_gee_favourable", {})
        if gee_fav and not gee_fav.get("skipped"):
            parts.append('<div class="collapsible">')
            parts.append('<div class="collapsible-header">B2) Logistic GEE (Favourable)</div>')
            parts.append('<div class="collapsible-body">')
            mi = gee_fav.get("model_info", {})
            parts.append(f'<p><strong>Method:</strong> {html.escape(mi.get("method", ""))}</p>')
            parts.append(f'<p>N = {gee_fav.get("n_obs", 0):,}, clusters = {gee_fav.get("n_clusters", 0):,}</p>')
            _render_coefficient_table(parts, gee_fav, show_or=True)
            parts.append("</div></div>")

        # --- C) GEE info request ---
        gee_info = mr.get("logistic_gee_info_request", {})
        if gee_info and not gee_info.get("skipped"):
            parts.append('<div class="collapsible">')
            parts.append('<div class="collapsible-header">C) Logistic GEE (Info Request Burden)</div>')
            parts.append('<div class="collapsible-body">')
            mi = gee_info.get("model_info", {})
            parts.append(f'<p><strong>Method:</strong> {html.escape(mi.get("method", ""))}</p>')
            parts.append(f'<p>N = {gee_info.get("n_obs", 0):,}, clusters = {gee_info.get("n_clusters", 0):,}</p>')
            _render_coefficient_table(parts, gee_info, show_or=True)
            parts.append("</div></div>")

        # --- D) Linear mixed model ---
        lmm = mr.get("linear_mixed_model", {})
        if lmm and not lmm.get("skipped"):
            parts.append('<div class="collapsible">')
            parts.append('<div class="collapsible-header">D) Linear Mixed Model (Robustness)</div>')
            parts.append('<div class="collapsible-body">')
            mi = lmm.get("model_info", {})
            parts.append(f'<p><strong>Method:</strong> {html.escape(mi.get("method", ""))}</p>')
            icc = lmm.get("icc", 0)
            parts.append(f'<p><strong>ICC:</strong> {icc:.3f} &mdash; {html.escape(mi.get("icc_interpretation", ""))}</p>')
            parts.append('<div class="metrics-grid">')
            parts.append(f'<div class="metric-card"><div class="value">{_fmt(lmm.get("random_effects_variance"))}</div><div class="label">Random Effect Var</div></div>')
            parts.append(f'<div class="metric-card"><div class="value">{_fmt(lmm.get("residual_variance"))}</div><div class="label">Residual Var</div></div>')
            parts.append(f'<div class="metric-card"><div class="value">{_fmt(icc)}</div><div class="label">ICC</div></div>')
            parts.append("</div>")
            _render_coefficient_table(parts, lmm)
            parts.append("</div></div>")

        # --- E) Nonparametric ---
        np_res = mr.get("nonparametric", {})
        if np_res and not np_res.get("skipped"):
            parts.append('<div class="collapsible">')
            parts.append('<div class="collapsible-header">E) Nonparametric Repeated-Measures Tests</div>')
            parts.append('<div class="collapsible-body">')

            for test_key, test_label in [
                ("friedman_tier", "Friedman Test (Outcome Tier)"),
                ("cochran_favourable", "Cochran's Q (Favourable)"),
                ("cochran_info_request", "Cochran's Q (Info Request)"),
            ]:
                t = np_res.get(test_key, {})
                if t and not t.get("skipped"):
                    sig = " *" if t.get("p_value", 1) < 0.05 else ""
                    parts.append(f'<h4>{test_label}</h4>')
                    parts.append(f'<p>{t.get("test", "")}: statistic = {_fmt(t.get("statistic"))}, '
                                 f'p = {_fmt(t.get("p_value"))}{sig}, '
                                 f'n_complaints = {t.get("n_complaints", 0)}, '
                                 f'n_personas = {t.get("n_personas", 0)}</p>')
                    posthoc = t.get("posthoc", {})
                    if posthoc:
                        parts.append(f'<p>Post-hoc: {posthoc.get("n_significant", 0)}/{posthoc.get("n_pairs_tested", 0)} '
                                     f'significant pairs (FDR-adjusted p &lt; 0.05)</p>')

            parts.append("</div></div>")

    # Pooled results
    pooled = rm.get("pooled", {})
    if pooled and not pooled.get("skipped"):
        parts.append("<h3>Pooled (All Models)</h3>")
        lmm = pooled.get("linear_mixed_model", {})
        if lmm and not lmm.get("skipped"):
            icc = lmm.get("icc", 0)
            parts.append(f'<p><strong>Pooled LMM ICC:</strong> {icc:.3f} ({icc*100:.1f}% of tier variance is between complaints)</p>')
            _render_coefficient_table(parts, lmm)

    return "\n".join(parts)


def _render_coefficient_table(
    parts: List[str],
    model_results: Dict[str, Any],
    show_or: bool = False,
) -> None:
    """Render a coefficient table for any model result dict."""
    coeffs = model_results.get("coefficients", {})
    if not coeffs:
        return
    se = model_results.get("std_errors", {})
    pv = model_results.get("p_values", {})
    ci = model_results.get("conf_int", {})
    ors = model_results.get("odds_ratios", {}) if show_or else {}

    parts.append('<div class="table-scroll">')
    parts.append('<table class="sortable"><thead><tr>')
    parts.append('<th>Parameter <span class="sort-arrow">&#8597;</span></th>')
    parts.append('<th>Estimate <span class="sort-arrow">&#8597;</span></th>')
    parts.append('<th>SE <span class="sort-arrow">&#8597;</span></th>')
    if show_or:
        parts.append('<th>OR <span class="sort-arrow">&#8597;</span></th>')
    parts.append('<th>p-value <span class="sort-arrow">&#8597;</span></th>')
    parts.append('<th>95% CI <span class="sort-arrow">&#8597;</span></th>')
    parts.append("</tr></thead><tbody>")

    for name in coeffs:
        p = pv.get(name, float("nan"))
        sig_marker = " *" if isinstance(p, (int, float)) and p < 0.05 else ""
        color = "#e74c3c" if sig_marker else "#2c3e50"
        ci_vals = ci.get(name, [])
        ci_str = f"[{_fmt(ci_vals[0])}, {_fmt(ci_vals[1])}]" if len(ci_vals) == 2 else "N/A"

        parts.append("<tr>")
        parts.append(f"<td>{html.escape(str(name))}{sig_marker}</td>")
        parts.append(f'<td data-sort-value="{coeffs[name]}">{_fmt(coeffs[name])}</td>')
        parts.append(f'<td data-sort-value="{se.get(name, 0)}">{_fmt(se.get(name))}</td>')
        if show_or:
            parts.append(f'<td data-sort-value="{ors.get(name, 0)}">{_fmt(ors.get(name))}</td>')
        parts.append(f'<td data-sort-value="{p}" style="color:{color}; font-weight:bold;">{_fmt(p)}</td>')
        parts.append(f"<td>{ci_str}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></div>")


def generate_report(
    analysis_results: Dict[str, Any],
    text_features_df: Optional[pd.DataFrame],
    complaints: List[Complaint],
    output_path: str,
    models: List[str],
    num_complaints: int,
    seed: int,
    materiality_threshold: float,
    timestamp: str,
) -> str:
    """Generate a self-contained HTML report and write it to *output_path*.

    Parameters
    ----------
    analysis_results : Full analysis results dict from experiment.py
    text_features_df : Optional DataFrame of extracted text features
    complaints : List of Complaint objects used in the experiment
    output_path : File path for the output HTML report
    models : List of model identifiers used
    num_complaints : Number of complaints analyzed
    seed : Random seed for reproducibility
    materiality_threshold : AIR threshold for materiality determination
    timestamp : Experiment timestamp string

    Returns
    -------
    str : The output file path
    """
    logger.info("Generating HTML report -> %s", output_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Build each tab's content
        tab_exec = _build_tab_executive_summary(
            analysis_results, complaints, models,
            num_complaints, seed, materiality_threshold, timestamp,
        )
        tab_disparate = _build_tab_disparate_impact(
            analysis_results, complaints, models,
            materiality_threshold, text_features_df,
        )
        tab_process = _build_tab_process_discrimination(
            analysis_results, models,
        )
        tab_text = _build_tab_text_features(
            analysis_results, text_features_df, models,
        )
        tab_model = _build_tab_model_comparison(
            analysis_results, models, materiality_threshold,
        )
        tab_detailed = _build_tab_detailed_results(
            analysis_results, complaints, models,
            num_complaints, seed, materiality_threshold, timestamp,
        )
        tab_severity = _build_tab_severity(analysis_results, complaints, models)
        tab_repeated = _build_tab_repeated_measures(analysis_results, models)
        tab_sparsity = _build_tab_sparsity(analysis_results, models)

    # Assemble the body
    body_parts: List[str] = []

    # Header
    body_parts.append('<div class="header">')
    body_parts.append("<h1>LLM Fairness Experiment Report</h1>")
    body_parts.append(f'<div class="subtitle">Models: {html.escape(", ".join(models))} | '
                      f'Complaints: {num_complaints} | Seed: {seed} | Generated: {html.escape(timestamp)}</div>')
    body_parts.append("</div>")

    # Tab group
    body_parts.append('<div class="tab-group">')
    body_parts.append('<div class="tab-nav">')
    tabs = [
        ("tab-biz-exec", "Executive Summary"),
        ("tab-exec", "Technical Summary"),
        ("tab-disparate", "Disparate Impact"),
        ("tab-process", "Process Discrimination"),
        ("tab-text", "Text Features"),
        ("tab-model", "Model Comparison"),
        ("tab-severity", "Severity Analysis"),
        ("tab-repeated", "Repeated Measures"),
        ("tab-sparsity", "Sparsity Analysis"),
        ("tab-detailed", "Detailed Results"),
    ]
    for i, (tid, label) in enumerate(tabs):
        active = " active" if i == 0 else ""
        body_parts.append(f'<button class="tab-btn{active}" data-tab="{tid}">{label}</button>')
    body_parts.append("</div>")

    tab_biz_exec = _build_tab_business_executive_summary(analysis_results)
    tab_contents = [
        ("tab-biz-exec", tab_biz_exec),
        ("tab-exec", tab_exec),
        ("tab-disparate", tab_disparate),
        ("tab-process", tab_process),
        ("tab-text", tab_text),
        ("tab-model", tab_model),
        ("tab-severity", tab_severity),
        ("tab-repeated", tab_repeated),
        ("tab-sparsity", tab_sparsity),
        ("tab-detailed", tab_detailed),
    ]
    for i, (tid, content) in enumerate(tab_contents):
        active = " active" if i == 0 else ""
        body_parts.append(f'<div id="{tid}" class="tab-content{active}">')
        body_parts.append(content)
        body_parts.append("</div>")

    body_parts.append("</div>")  # tab-group

    body = "\n".join(body_parts)
    full_html = _build_html(body)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info("Report written: %s (%d bytes)", output_path, len(full_html))
    return output_path
