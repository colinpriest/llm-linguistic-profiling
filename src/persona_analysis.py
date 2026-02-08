"""Persona linguistic analysis using LightGBM and partial dependence plots.

This module builds classification models to understand which text features
the LLM uses to assign each persona dimension value, then generates
partial dependence plots and summary sentences for each.
"""

from __future__ import annotations

import logging
import io
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.inspection import partial_dependence

from src.schemas import (
    ESTIMATE_DIMS,
    AGE_VALUES,
    GEO_VALUES,
    EDU_VALUES,
    SES_VALUES,
    GENDER_VALUES,
    ESL_VALUES,
)

logger = logging.getLogger(__name__)

# Mapping of dimension name to its possible values
DIM_VALUES: Dict[str, List[str]] = {
    "age": AGE_VALUES,
    "geographic": GEO_VALUES,
    "education": EDU_VALUES,
    "socioeconomic": SES_VALUES,
    "gender": GENDER_VALUES,
    "esl": ESL_VALUES,
}

# Text features to use (excluding case_id, variant, topic distribution list, and topic columns)
TEXT_FEATURE_COLS = [
    "word_count",
    "char_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "ari",
    "type_token_ratio",
    "hapax_ratio",
    "mtld",
    "hdd",
    "polarity",
    "subjectivity",
    "vader_compound",
    "vader_pos",
    "vader_neg",
    "vader_neu",
    "emotional_intensity",
    "dominant_topic",
    "topic_coherence",
]

# Human-readable names for features
FEATURE_DISPLAY_NAMES = {
    "word_count": "Word Count",
    "char_count": "Character Count",
    "sentence_count": "Sentence Count",
    "avg_word_length": "Avg Word Length",
    "avg_sentence_length": "Avg Sentence Length",
    "flesch_reading_ease": "Flesch Reading Ease",
    "flesch_kincaid_grade": "Flesch-Kincaid Grade",
    "gunning_fog": "Gunning Fog Index",
    "smog_index": "SMOG Index",
    "ari": "Automated Readability",
    "type_token_ratio": "Type-Token Ratio",
    "hapax_ratio": "Hapax Ratio",
    "mtld": "MTLD (Lexical Diversity)",
    "hdd": "HD-D (Lexical Diversity)",
    "polarity": "Polarity",
    "subjectivity": "Subjectivity",
    "vader_compound": "VADER Compound",
    "vader_pos": "VADER Positive",
    "vader_neg": "VADER Negative",
    "vader_neu": "VADER Neutral",
    "emotional_intensity": "Emotional Intensity",
    "dominant_topic": "Dominant Topic",
    "topic_coherence": "Topic Coherence",
}


def _get_feature_name(feat: str) -> str:
    """Get human-readable feature name."""
    return FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())


def _extract_pdp_values(pd_result) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract values and average from partial_dependence result.

    Handles both old-style (namedtuple/list) and new-style (Bunch) returns.
    """
    # Try Bunch-style access (sklearn >= 0.24)
    if hasattr(pd_result, "values") and hasattr(pd_result, "average"):
        return pd_result.values[0], pd_result.average[0]

    # Try dict-style access
    if isinstance(pd_result, dict):
        if "values" in pd_result and "average" in pd_result:
            return pd_result["values"][0], pd_result["average"][0]
        # Some versions use 'grid_values'
        if "grid_values" in pd_result and "average" in pd_result:
            return pd_result["grid_values"][0], pd_result["average"][0]

    # Old-style tuple return (sklearn < 0.24): (average, values)
    if isinstance(pd_result, tuple) and len(pd_result) >= 2:
        avg, vals = pd_result[0], pd_result[1]
        if isinstance(vals, list) and len(vals) > 0:
            return vals[0], avg[0]

    return None, None


def _generate_pdp_chart(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    feature_name: str,
    persona_value: str,
) -> str:
    """Generate a partial dependence plot for a single feature.

    Returns base64-encoded PNG image.
    """
    try:
        feature_idx = list(X.columns).index(feature_name)

        # Compute partial dependence
        pd_result = partial_dependence(
            model,
            X,
            features=[feature_idx],
            kind="average",
            grid_resolution=50,
        )

        values, avg_pred = _extract_pdp_values(pd_result)
        if values is None or avg_pred is None:
            logger.warning("Could not extract PDP values for %s", feature_name)
            return ""

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(values, avg_pred, color="#3498db", linewidth=2)
        ax.fill_between(values, avg_pred, alpha=0.2, color="#3498db")
        ax.set_xlabel(_get_feature_name(feature_name), fontsize=9)
        ax.set_ylabel(f"P({persona_value})", fontsize=9)
        ax.set_title(f"Partial Dependence", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        logger.warning("Failed to generate PDP for %s: %s", feature_name, e)
        return ""


def _describe_feature_effect(
    feature_name: str,
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    persona_value: str,
) -> str:
    """Generate a description of how a feature affects persona prediction.

    Returns a phrase like "higher word count" or "lower reading ease".
    """
    try:
        feature_idx = list(X.columns).index(feature_name)

        # Compute partial dependence at extremes
        pd_result = partial_dependence(
            model,
            X,
            features=[feature_idx],
            kind="average",
            percentiles=(0.1, 0.9),
            grid_resolution=20,
        )

        values, avg_pred = _extract_pdp_values(pd_result)
        if values is None or avg_pred is None:
            return _get_feature_name(feature_name).lower()

        # Determine direction: does prediction increase or decrease with feature?
        if len(avg_pred) >= 2:
            start_pred = avg_pred[0]
            end_pred = avg_pred[-1]

            display_name = _get_feature_name(feature_name).lower()

            if end_pred > start_pred + 0.02:
                return f"higher {display_name}"
            elif end_pred < start_pred - 0.02:
                return f"lower {display_name}"
            else:
                return f"moderate {display_name}"

        return _get_feature_name(feature_name).lower()

    except Exception as e:
        logger.warning("Failed to describe feature effect for %s: %s", feature_name, e)
        return _get_feature_name(feature_name).lower()


def _build_summary_sentence(
    persona_dim: str,
    persona_value: str,
    feature_effects: List[str],
) -> str:
    """Build a summary sentence describing the persona's linguistic style."""
    if len(feature_effects) == 0:
        return f"No clear linguistic markers distinguish {persona_value} text."

    # Format dimension and value for readability
    dim_display = persona_dim.replace("_", " ")

    if len(feature_effects) == 1:
        effects_str = feature_effects[0]
    elif len(feature_effects) == 2:
        effects_str = f"{feature_effects[0]} and {feature_effects[1]}"
    else:
        effects_str = f"{feature_effects[0]}, {feature_effects[1]}, and {feature_effects[2]}"

    return f"Text classified as {persona_value} ({dim_display}) tends to exhibit {effects_str}."


def analyze_persona_for_model(
    personas_df: pd.DataFrame,
    text_features_df: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    """Analyze persona text features for a single model.

    Parameters
    ----------
    personas_df : DataFrame with columns [case_id, model, age, geographic, ...]
    text_features_df : DataFrame with columns [case_id, variant, word_count, ...]
    model_name : Name of the LLM model to analyze

    Returns
    -------
    dict : Results keyed by dimension, then by value within dimension
    """
    results: Dict[str, Any] = {"model": model_name, "dimensions": {}}

    # Filter personas to this model
    personas_m = personas_df[personas_df["model"] == model_name].copy()
    if personas_m.empty:
        logger.warning("No persona data for model %s", model_name)
        return results

    # Filter text features to original variant only
    tf_orig = text_features_df[text_features_df["variant"] == "original"].copy()
    if tf_orig.empty:
        logger.warning("No original text features found")
        return results

    # Merge personas with text features
    merged = personas_m.merge(tf_orig, on="case_id", how="inner")
    if merged.empty:
        logger.warning("No matched data after merge for model %s", model_name)
        return results

    # Get available text feature columns
    available_features = [c for c in TEXT_FEATURE_COLS if c in merged.columns]
    if len(available_features) < 3:
        logger.warning("Insufficient text features available: %d", len(available_features))
        return results

    # Prepare feature matrix
    X = merged[available_features].copy()

    # Handle missing values
    X = X.fillna(X.median())

    # Replace any remaining NaN/inf with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Convert all columns to float64 for partial_dependence compatibility
    # (sklearn PDP generates float grid values that fail with int64 dtypes)
    X = X.astype(np.float64)

    logger.info("Analyzing %d complaints with %d features for model %s",
                len(merged), len(available_features), model_name)

    # Analyze each dimension
    for dim in ESTIMATE_DIMS:
        if dim not in merged.columns:
            continue

        dim_values = DIM_VALUES.get(dim, [])
        if not dim_values:
            continue

        dim_results: Dict[str, Any] = {}

        for value in dim_values:
            # Create binary target: 1 if this value, 0 otherwise
            y = (merged[dim] == value).astype(int)

            n_positive = y.sum()
            n_negative = len(y) - n_positive

            if n_positive < 10 or n_negative < 10:
                logger.info("Skipping %s=%s: insufficient samples (pos=%d, neg=%d)",
                           dim, value, n_positive, n_negative)
                continue

            try:
                # Build LightGBM model
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    num_leaves=15,
                    min_child_samples=max(10, n_positive // 10),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True,
                )

                lgb_model.fit(X, y)

                # Get feature importances
                importances = lgb_model.feature_importances_
                feature_importance = list(zip(available_features, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                # Top 3 features
                top_3 = feature_importance[:3]

                # Generate PDPs and effect descriptions
                pdp_charts = []
                feature_effects = []

                for feat, imp in top_3:
                    if imp > 0:
                        # Generate PDP chart
                        pdp_b64 = _generate_pdp_chart(lgb_model, X, feat, value)
                        if pdp_b64:
                            pdp_charts.append({
                                "feature": feat,
                                "feature_display": _get_feature_name(feat),
                                "importance": float(imp),
                                "chart_b64": pdp_b64,
                            })

                        # Get effect description
                        effect = _describe_feature_effect(feat, lgb_model, X, value)
                        feature_effects.append(effect)

                # Generate summary sentence
                summary = _build_summary_sentence(dim, value, feature_effects)

                # Calculate model performance (AUC on training data, just for reference)
                try:
                    from sklearn.metrics import roc_auc_score
                    y_pred_proba = lgb_model.predict_proba(X)[:, 1]
                    auc = roc_auc_score(y, y_pred_proba)
                except Exception:
                    auc = None

                dim_results[value] = {
                    "n_samples": int(n_positive),
                    "n_total": len(y),
                    "prevalence": float(n_positive / len(y)),
                    "top_features": [
                        {
                            "feature": feat,
                            "feature_display": _get_feature_name(feat),
                            "importance": float(imp),
                        }
                        for feat, imp in top_3
                    ],
                    "pdp_charts": pdp_charts,
                    "summary": summary,
                    "auc": float(auc) if auc is not None else None,
                }

                logger.info("Analyzed %s=%s: top features = %s",
                           dim, value, [f[0] for f in top_3])

            except Exception as e:
                logger.warning("Failed to build model for %s=%s: %s", dim, value, e)
                continue

        if dim_results:
            results["dimensions"][dim] = dim_results

    return results


def run_persona_analysis(
    personas_df: pd.DataFrame,
    text_features_df: pd.DataFrame,
    models: List[str],
) -> Dict[str, Any]:
    """Run persona linguistic analysis for all models.

    Parameters
    ----------
    personas_df : DataFrame with persona estimates for all models
    text_features_df : DataFrame with text features
    models : List of model names to analyze

    Returns
    -------
    dict : Results keyed by model name
    """
    logger.info("Running persona linguistic analysis for %d models", len(models))

    results: Dict[str, Any] = {}

    for model_name in models:
        logger.info("Analyzing persona features for model: %s", model_name)
        model_results = analyze_persona_for_model(
            personas_df,
            text_features_df,
            model_name,
        )
        results[model_name] = model_results

    return results
