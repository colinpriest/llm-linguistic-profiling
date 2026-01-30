"""Self-contained HTML report generator for the LLM fairness experiment."""

from __future__ import annotations

import base64
import html
import io
import logging
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

from src.schemas import Complaint, PERSONA_LABELS

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

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Persona Distribution (Estimated)", fontsize=14, fontweight="bold")

    dims = ["age", "geographic", "education", "socioeconomic"]
    for ax, dim in zip(axes.flat, dims):
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Favorable Outcome Rate by Demographic Dimension", fontsize=14, fontweight="bold")

    dims = ["age", "geographic", "education", "socioeconomic"]
    tier_colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    for ax, dim in zip(axes.flat, dims):
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

    dims = ["age", "geographic", "education", "socioeconomic"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Information Request Rate by Dimension ({model})", fontsize=14, fontweight="bold")
    palette = sns.color_palette("Set2", 6)

    for ax, dim in zip(axes.flat, dims):
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

    for dim in ["age", "geographic", "education", "socioeconomic"]:
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

def _build_tab_executive_summary(
    analysis_results: Dict[str, Any],
    complaints: List[Complaint],
    models: List[str],
    num_complaints: int,
    seed: int,
    materiality_threshold: float,
    timestamp: str,
) -> str:
    """Build the Executive Summary tab HTML."""
    parts: List[str] = []

    parts.append("<h2>Executive Summary</h2>")

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
        for dim in ["age", "geographic", "education", "socioeconomic"]:
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
            for dim in ["age", "geographic", "education", "socioeconomic"]:
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
        for dim in ["age", "geographic", "education", "socioeconomic"]:
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

    for dim in ["age", "geographic", "education", "socioeconomic"]:
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
# Main entry point
# ---------------------------------------------------------------------------

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
        ("tab-exec", "Executive Summary"),
        ("tab-disparate", "Disparate Impact"),
        ("tab-process", "Process Discrimination"),
        ("tab-text", "Text Features"),
        ("tab-model", "Model Comparison"),
        ("tab-detailed", "Detailed Results"),
    ]
    for i, (tid, label) in enumerate(tabs):
        active = " active" if i == 0 else ""
        body_parts.append(f'<button class="tab-btn{active}" data-tab="{tid}">{label}</button>')
    body_parts.append("</div>")

    tab_contents = [
        ("tab-exec", tab_exec),
        ("tab-disparate", tab_disparate),
        ("tab-process", tab_process),
        ("tab-text", tab_text),
        ("tab-model", tab_model),
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
