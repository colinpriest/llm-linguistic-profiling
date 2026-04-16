"""
Text Rewriting Analysis Module.

Analyzes how text characteristics change when complaints are rewritten
to different persona styles. Provides statistical tests and effect sizes
for before/after comparisons.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from src.schemas import parse_persona_label
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Features to analyze (most interpretable/important)
ANALYSIS_FEATURES = [
    # Readability
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    # Length
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "avg_word_length",
    # Vocabulary
    "type_token_ratio",
    "hapax_ratio",
    # Sentiment
    "vader_compound",
    "vader_pos",
    "vader_neg",
    "polarity",
    "subjectivity",
]

# Human-readable feature names
FEATURE_LABELS = {
    "flesch_reading_ease": "Flesch Reading Ease",
    "flesch_kincaid_grade": "Flesch-Kincaid Grade",
    "gunning_fog": "Gunning Fog Index",
    "smog_index": "SMOG Index",
    "word_count": "Word Count",
    "sentence_count": "Sentence Count",
    "avg_sentence_length": "Avg Sentence Length",
    "avg_word_length": "Avg Word Length",
    "type_token_ratio": "Type-Token Ratio",
    "hapax_ratio": "Hapax Ratio",
    "vader_compound": "VADER Compound",
    "vader_pos": "VADER Positive",
    "vader_neg": "VADER Negative",
    "polarity": "Polarity",
    "subjectivity": "Subjectivity",
    # LLM-rated text quality (available for original variants only)
    "convincingness": "Convincingness",
    "persuasiveness": "Persuasiveness",
    "evidence_blame": "Evidence of Blame",
    "evidence_harm": "Evidence of Harm",
    "tq_confidence": "Complainant Confidence",
    "legal_knowledge": "Legal Knowledge",
    "formality": "Formality",
    "coherence": "Coherence",
    "specificity": "Specificity",
    "financial_literacy": "Financial Literacy",
    "emotional_restraint": "Emotional Restraint",
    "grammar_quality": "Grammar Quality",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class FeatureComparison:
    """Statistical comparison of a single feature between two groups."""

    feature: str
    feature_label: str

    # Descriptive statistics
    original_mean: float
    original_std: float
    original_median: float
    rewritten_mean: float
    rewritten_std: float
    rewritten_median: float

    # Change metrics
    mean_difference: float
    pct_change: float  # percentage change from original

    # Statistical tests
    t_statistic: float
    t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float

    # Effect size
    cohens_d: float
    cohens_d_ci_lower: float
    cohens_d_ci_upper: float

    # Sample size
    n_pairs: int

    # Significance flags
    is_significant_t: bool = False  # p < 0.05
    is_significant_wilcoxon: bool = False
    is_significant_fdr: bool = False  # after FDR correction


@dataclass
class PersonaFeatureProfile:
    """Feature profile for a specific persona."""

    persona_label: str
    n_samples: int

    # Mean values for key features
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)

    # Comparisons vs original
    comparisons: List[FeatureComparison] = field(default_factory=list)


@dataclass
class DimensionEffect:
    """Effect of changing a single persona dimension on text features."""

    dimension: str  # age, geographic, education, socioeconomic
    from_value: str
    to_value: str

    # Feature effects (feature -> effect size)
    feature_effects: Dict[str, float] = field(default_factory=dict)
    feature_pvalues: Dict[str, float] = field(default_factory=dict)

    # Summary
    largest_effect_feature: str = ""
    largest_effect_size: float = 0.0


@dataclass
class TextRewritingAnalysisResults:
    """Complete results from text rewriting analysis."""

    # Overall comparison: original vs all rewrites
    overall_comparisons: List[FeatureComparison] = field(default_factory=list)

    # Per-persona profiles and comparisons
    persona_profiles: Dict[str, PersonaFeatureProfile] = field(default_factory=dict)

    # Dimension-level effects
    dimension_effects: List[DimensionEffect] = field(default_factory=list)

    # Summary statistics
    n_original: int = 0
    n_rewritten: int = 0
    n_personas: int = 0

    # Features with significant changes (after FDR)
    significant_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_comparisons": [
                self._comparison_to_dict(c) for c in self.overall_comparisons
            ],
            "persona_profiles": {
                k: self._profile_to_dict(v)
                for k, v in self.persona_profiles.items()
            },
            "dimension_effects": [
                self._dimension_to_dict(d) for d in self.dimension_effects
            ],
            "n_original": self.n_original,
            "n_rewritten": self.n_rewritten,
            "n_personas": self.n_personas,
            "significant_features": self.significant_features,
        }

    def _comparison_to_dict(self, c: FeatureComparison) -> Dict[str, Any]:
        return {
            "feature": c.feature,
            "feature_label": c.feature_label,
            "original_mean": c.original_mean,
            "original_std": c.original_std,
            "original_median": c.original_median,
            "rewritten_mean": c.rewritten_mean,
            "rewritten_std": c.rewritten_std,
            "rewritten_median": c.rewritten_median,
            "mean_difference": c.mean_difference,
            "pct_change": c.pct_change,
            "t_statistic": c.t_statistic,
            "t_pvalue": c.t_pvalue,
            "wilcoxon_statistic": c.wilcoxon_statistic,
            "wilcoxon_pvalue": c.wilcoxon_pvalue,
            "cohens_d": c.cohens_d,
            "cohens_d_ci_lower": c.cohens_d_ci_lower,
            "cohens_d_ci_upper": c.cohens_d_ci_upper,
            "n_pairs": c.n_pairs,
            "is_significant_t": c.is_significant_t,
            "is_significant_wilcoxon": c.is_significant_wilcoxon,
            "is_significant_fdr": c.is_significant_fdr,
        }

    def _profile_to_dict(self, p: PersonaFeatureProfile) -> Dict[str, Any]:
        return {
            "persona_label": p.persona_label,
            "n_samples": p.n_samples,
            "feature_means": p.feature_means,
            "feature_stds": p.feature_stds,
            "comparisons": [self._comparison_to_dict(c) for c in p.comparisons],
        }

    def _dimension_to_dict(self, d: DimensionEffect) -> Dict[str, Any]:
        return {
            "dimension": d.dimension,
            "from_value": d.from_value,
            "to_value": d.to_value,
            "feature_effects": d.feature_effects,
            "feature_pvalues": d.feature_pvalues,
            "largest_effect_feature": d.largest_effect_feature,
            "largest_effect_size": d.largest_effect_size,
        }


# ---------------------------------------------------------------------------
# Statistical Functions
# ---------------------------------------------------------------------------


def compute_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    paired: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute Cohen's d effect size with 95% CI.

    For paired data, uses the standard deviation of differences.
    Returns (d, ci_lower, ci_upper).
    """
    n1, n2 = len(group1), len(group2)

    if paired and n1 == n2:
        # Paired Cohen's d (dz)
        diff = group1 - group2
        mean = np.mean(diff)
        std = np.std(diff, ddof=1)
        if std > 0:
            d = mean / std
        elif mean != 0:
            d = float("nan")  # effect exists but std is 0
        else:
            d = 0.0  # no effect
        # SE for paired d
        se = np.sqrt((1 / n1) + (d ** 2 / (2 * n1)))
    else:
        # Independent Cohen's d
        mean_diff = np.mean(group1) - np.mean(group2)
        # Pooled standard deviation
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std > 0:
            d = mean_diff / pooled_std
        elif mean_diff != 0:
            d = float("nan")  # effect exists but std is 0
        else:
            d = 0.0  # no effect
        # SE for independent d
        se = np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2 / (2 * (n1 + n2))))

    # 95% CI
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    return d, ci_lower, ci_upper


def benjamini_hochberg_correction(pvalues: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Returns list of booleans indicating significance after correction.
    """
    n = len(pvalues)
    if n == 0:
        return []

    # Sort p-values with original indices
    sorted_pairs = sorted(enumerate(pvalues), key=lambda x: x[1])

    # Compute critical values and determine significance
    significant = [False] * n
    for rank, (orig_idx, pval) in enumerate(sorted_pairs, start=1):
        critical = (rank / n) * alpha
        if pval <= critical:
            # This and all smaller p-values are significant
            for j in range(rank):
                significant[sorted_pairs[j][0]] = True

    return significant


def compare_feature(
    original: np.ndarray,
    rewritten: np.ndarray,
    feature: str,
    paired: bool = True,
) -> FeatureComparison:
    """
    Compare a single feature between original and rewritten text.

    Performs paired t-test (or independent if not paired), Wilcoxon test,
    and computes Cohen's d effect size.
    """
    # Remove NaN pairs
    if paired:
        mask = ~(np.isnan(original) | np.isnan(rewritten))
        orig_clean = original[mask]
        rew_clean = rewritten[mask]
    else:
        orig_clean = original[~np.isnan(original)]
        rew_clean = rewritten[~np.isnan(rewritten)]

    n_pairs = len(orig_clean) if paired else min(len(orig_clean), len(rew_clean))

    if n_pairs < 3:
        # Not enough data
        return FeatureComparison(
            feature=feature,
            feature_label=FEATURE_LABELS.get(feature, feature),
            original_mean=0.0,
            original_std=0.0,
            original_median=0.0,
            rewritten_mean=0.0,
            rewritten_std=0.0,
            rewritten_median=0.0,
            mean_difference=0.0,
            pct_change=0.0,
            t_statistic=0.0,
            t_pvalue=1.0,
            wilcoxon_statistic=0.0,
            wilcoxon_pvalue=1.0,
            cohens_d=0.0,
            cohens_d_ci_lower=0.0,
            cohens_d_ci_upper=0.0,
            n_pairs=n_pairs,
        )

    # Descriptive statistics
    orig_mean = np.mean(orig_clean)
    orig_std = np.std(orig_clean, ddof=1)
    orig_median = np.median(orig_clean)
    rew_mean = np.mean(rew_clean)
    rew_std = np.std(rew_clean, ddof=1)
    rew_median = np.median(rew_clean)

    mean_diff = rew_mean - orig_mean
    pct_change = (mean_diff / orig_mean * 100) if orig_mean != 0 else 0.0

    # Statistical tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if paired:
            t_stat, t_pval = stats.ttest_rel(orig_clean, rew_clean)
        else:
            t_stat, t_pval = stats.ttest_ind(orig_clean, rew_clean)

        # Wilcoxon signed-rank test (paired) or Mann-Whitney U (unpaired)
        try:
            if paired:
                diff = rew_clean - orig_clean
                # Wilcoxon requires non-zero differences
                nonzero_diff = diff[diff != 0]
                if len(nonzero_diff) >= 3:
                    w_stat, w_pval = stats.wilcoxon(nonzero_diff)
                else:
                    w_stat, w_pval = 0.0, 1.0
            else:
                w_stat, w_pval = stats.mannwhitneyu(orig_clean, rew_clean, alternative="two-sided")
        except ValueError:
            w_stat, w_pval = 0.0, 1.0

    # Handle NaN p-values
    t_pval = t_pval if np.isfinite(t_pval) else 1.0
    w_pval = w_pval if np.isfinite(w_pval) else 1.0
    t_stat = t_stat if np.isfinite(t_stat) else 0.0
    w_stat = w_stat if np.isfinite(w_stat) else 0.0

    # Effect size
    d, d_lower, d_upper = compute_cohens_d(orig_clean, rew_clean, paired=paired)

    return FeatureComparison(
        feature=feature,
        feature_label=FEATURE_LABELS.get(feature, feature),
        original_mean=orig_mean,
        original_std=orig_std,
        original_median=orig_median,
        rewritten_mean=rew_mean,
        rewritten_std=rew_std,
        rewritten_median=rew_median,
        mean_difference=mean_diff,
        pct_change=pct_change,
        t_statistic=t_stat,
        t_pvalue=t_pval,
        wilcoxon_statistic=w_stat,
        wilcoxon_pvalue=w_pval,
        cohens_d=d,
        cohens_d_ci_lower=d_lower,
        cohens_d_ci_upper=d_upper,
        n_pairs=n_pairs,
        is_significant_t=t_pval < 0.05,
        is_significant_wilcoxon=w_pval < 0.05,
    )


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------


def analyze_overall_rewriting_effects(
    text_features_df: pd.DataFrame,
) -> Tuple[List[FeatureComparison], int, int]:
    """
    Compare original text features vs all rewritten versions (aggregated).

    For each original complaint, we compare against the mean of its rewritten versions.
    """
    comparisons = []

    # Separate original and rewritten
    original_df = text_features_df[text_features_df["variant"] == "original"].copy()
    rewritten_df = text_features_df[
        (text_features_df["variant"] != "original") &
        (text_features_df["variant"] != "neutralized")
    ].copy()

    if len(original_df) == 0 or len(rewritten_df) == 0:
        logger.warning("Insufficient data for overall rewriting analysis")
        return comparisons, 0, 0

    # Get available features
    available_features = [f for f in ANALYSIS_FEATURES if f in text_features_df.columns]

    if not available_features:
        logger.warning("No analyzable features found in text_features_df")
        return comparisons, len(original_df), len(rewritten_df)

    # Compute mean rewritten features per case_id
    rewritten_means = rewritten_df.groupby("case_id")[available_features].mean().reset_index()

    # Merge with original
    merged = original_df[["case_id"] + available_features].merge(
        rewritten_means,
        on="case_id",
        suffixes=("_orig", "_rew"),
    )

    if len(merged) < 10:
        logger.warning("Only %d paired observations for rewriting analysis", len(merged))
        return comparisons, len(original_df), len(rewritten_df)

    # Compare each feature
    for feature in available_features:
        orig_col = f"{feature}_orig"
        rew_col = f"{feature}_rew"

        if orig_col not in merged.columns or rew_col not in merged.columns:
            continue

        comparison = compare_feature(
            merged[orig_col].values,
            merged[rew_col].values,
            feature,
            paired=True,
        )
        comparisons.append(comparison)

    # Apply FDR correction
    if comparisons:
        pvalues = [c.t_pvalue for c in comparisons]
        fdr_significant = benjamini_hochberg_correction(pvalues)
        for i, comp in enumerate(comparisons):
            comp.is_significant_fdr = fdr_significant[i]

    return comparisons, len(original_df), len(rewritten_df)


def analyze_per_persona_effects(
    text_features_df: pd.DataFrame,
) -> Dict[str, PersonaFeatureProfile]:
    """
    Analyze text feature changes for each specific persona.
    """
    profiles = {}

    original_df = text_features_df[text_features_df["variant"] == "original"].copy()

    # Get unique persona labels (excluding original, neutralized)
    persona_labels = text_features_df[
        ~text_features_df["variant"].isin(["original", "neutralized", "persona_rewrite"])
    ]["variant"].unique()

    # Also check persona_label column if it exists
    if "persona_label" in text_features_df.columns:
        persona_col_labels = text_features_df[
            text_features_df["variant"] == "persona_rewrite"
        ]["persona_label"].dropna().unique()
        persona_labels = list(set(list(persona_labels) + list(persona_col_labels)))

    available_features = [f for f in ANALYSIS_FEATURES if f in text_features_df.columns]

    for persona in persona_labels:
        if not persona or persona in ["original", "neutralized", "persona_rewrite"]:
            continue

        # Get rewritten data for this persona
        if "persona_label" in text_features_df.columns:
            persona_df = text_features_df[
                text_features_df["persona_label"] == persona
            ].copy()
        else:
            persona_df = text_features_df[
                text_features_df["variant"] == persona
            ].copy()

        if len(persona_df) < 5:
            continue

        # Compute mean features for this persona
        feature_means = {}
        feature_stds = {}
        for feat in available_features:
            if feat in persona_df.columns:
                feature_means[feat] = float(persona_df[feat].mean())
                feature_stds[feat] = float(persona_df[feat].std())

        # Compare against original (paired by case_id)
        merged = original_df[["case_id"] + available_features].merge(
            persona_df[["case_id"] + available_features],
            on="case_id",
            suffixes=("_orig", "_rew"),
        )

        comparisons = []
        if len(merged) >= 5:
            for feature in available_features:
                orig_col = f"{feature}_orig"
                rew_col = f"{feature}_rew"
                if orig_col in merged.columns and rew_col in merged.columns:
                    comparison = compare_feature(
                        merged[orig_col].values,
                        merged[rew_col].values,
                        feature,
                        paired=True,
                    )
                    comparisons.append(comparison)

            # FDR correction within persona
            if comparisons:
                pvalues = [c.t_pvalue for c in comparisons]
                fdr_significant = benjamini_hochberg_correction(pvalues)
                for i, comp in enumerate(comparisons):
                    comp.is_significant_fdr = fdr_significant[i]

        profiles[persona] = PersonaFeatureProfile(
            persona_label=persona,
            n_samples=len(persona_df),
            feature_means=feature_means,
            feature_stds=feature_stds,
            comparisons=comparisons,
        )

    return profiles


def analyze_dimension_effects(
    text_features_df: pd.DataFrame,
) -> List[DimensionEffect]:
    """
    Analyze how changing each persona dimension affects text features.

    Compares personas that differ only in one dimension.
    """
    effects = []

    # Define dimension values
    dimensions = {
        "age": ["young", "old"],
        "geographic": ["urban", "suburban", "rural"],
        "education": ["high_school", "college"],
        "socioeconomic": ["low", "middle", "high"],
    }

    # Get persona profiles
    profiles = analyze_per_persona_effects(text_features_df)
    if not profiles:
        return effects

    available_features = [f for f in ANALYSIS_FEATURES if f in text_features_df.columns]

    # Parse persona labels into components
    def parse_persona(label: str) -> Optional[Dict[str, str]]:
        return parse_persona_label(label)

    parsed_profiles = {}
    for label, profile in profiles.items():
        parsed = parse_persona(label)
        if parsed:
            parsed_profiles[label] = (parsed, profile)

    # For each dimension, compare personas that differ only in that dimension
    for dim, values in dimensions.items():
        for i, from_val in enumerate(values):
            for to_val in values[i + 1:]:
                # Find persona pairs that differ only in this dimension
                feature_diffs = {f: [] for f in available_features}
                n_comparisons = 0

                for label1, (parsed1, profile1) in parsed_profiles.items():
                    if parsed1.get(dim) != from_val:
                        continue

                    # Find matching persona with to_val
                    target_parsed = parsed1.copy()
                    target_parsed[dim] = to_val
                    target_label = "_".join([
                        target_parsed["age"],
                        target_parsed["geographic"],
                        target_parsed["education"],
                        target_parsed["socioeconomic"],
                    ])

                    if target_label in parsed_profiles:
                        _, profile2 = parsed_profiles[target_label]

                        # Compare feature means
                        for feat in available_features:
                            mean1 = profile1.feature_means.get(feat)
                            mean2 = profile2.feature_means.get(feat)
                            if mean1 is not None and mean2 is not None:
                                feature_diffs[feat].append(mean2 - mean1)

                        n_comparisons += 1

                if n_comparisons < 2:
                    continue

                # Compute average effect for each feature
                feature_effects = {}
                feature_pvalues = {}

                for feat, diffs in feature_diffs.items():
                    if len(diffs) >= 2:
                        mean_effect = np.mean(diffs)
                        # One-sample t-test against 0
                        if np.std(diffs) > 0:
                            _, pval = stats.ttest_1samp(diffs, 0)
                        else:
                            pval = 1.0
                        feature_effects[feat] = mean_effect
                        feature_pvalues[feat] = pval if np.isfinite(pval) else 1.0

                if not feature_effects:
                    continue

                # Find largest effect
                largest_feat = max(feature_effects.keys(), key=lambda x: abs(feature_effects[x]))
                largest_effect = feature_effects[largest_feat]

                effects.append(DimensionEffect(
                    dimension=dim,
                    from_value=from_val,
                    to_value=to_val,
                    feature_effects=feature_effects,
                    feature_pvalues=feature_pvalues,
                    largest_effect_feature=largest_feat,
                    largest_effect_size=largest_effect,
                ))

    return effects


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def run_text_rewriting_analysis(
    text_features_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Run complete text rewriting analysis.

    This is the main entry point called from experiment.py.

    Parameters
    ----------
    text_features_df : DataFrame with columns including:
        - case_id: complaint identifier
        - variant: "original", "neutralized", persona label, or "persona_rewrite"
        - persona_label: (optional) specific persona for rewritten variants
        - Text feature columns (flesch_reading_ease, word_count, etc.)

    Returns
    -------
    Dictionary with analysis results suitable for JSON serialization.
    """
    if text_features_df is None or text_features_df.empty:
        logger.warning("No text features data available for rewriting analysis")
        return {"error": "No text features data available"}

    logger.info("Running text rewriting analysis...")

    results = TextRewritingAnalysisResults()

    # 1. Overall comparison: original vs all rewrites
    logger.info("  Analyzing overall rewriting effects...")
    overall_comparisons, n_orig, n_rew = analyze_overall_rewriting_effects(text_features_df)
    results.overall_comparisons = overall_comparisons
    results.n_original = n_orig
    results.n_rewritten = n_rew

    # Track significant features
    results.significant_features = [
        c.feature for c in overall_comparisons if c.is_significant_fdr
    ]

    # 2. Per-persona analysis
    logger.info("  Analyzing per-persona effects...")
    results.persona_profiles = analyze_per_persona_effects(text_features_df)
    results.n_personas = len(results.persona_profiles)

    # 3. Dimension-level effects
    logger.info("  Analyzing dimension-level effects...")
    results.dimension_effects = analyze_dimension_effects(text_features_df)

    # Log summary
    n_sig = len(results.significant_features)
    n_total = len(overall_comparisons)
    logger.info(
        "Text rewriting analysis complete: %d/%d features significantly "
        "changed after rewriting (FDR-corrected)",
        n_sig, n_total,
    )

    if results.significant_features:
        logger.info("  Significant features: %s", ", ".join(results.significant_features))

    return results.to_dict()
