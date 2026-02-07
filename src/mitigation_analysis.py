"""
Mitigation Effectiveness Analysis Module.

Analyzes when and why neutralization and persona-based interventions
are effective at reducing bias in LLM complaint processing.

Uses:
- LightGBM for predictive modeling with feature importance
- Mixed-effects models for statistical inference
- SHAP for local explanations
- Causal Forest / CATE for heterogeneous treatment effects
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports with graceful degradation
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("lightgbm not installed - using sklearn RandomForest fallback")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("shap not installed - SHAP explanations will be unavailable")

try:
    from econml.dml import CausalForestDML
    from econml.cate_interpreter import SingleTreeCateInterpreter
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    logger.warning("econml not installed - heterogeneous treatment effects unavailable")

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed - mixed-effects models unavailable")

# Sklearn fallback always available
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class ModelFitResults:
    """Results from a single model fit."""

    model_type: str  # 'lightgbm' | 'mixed_effects' | 'causal_forest' | 'sklearn_rf'
    target: str  # 'dispersion_delta' | 'gap_closure' | 'tier_delta'
    llm_model: str  # 'gpt-4o-mini' | 'gpt-5-mini'
    feature_importance: pd.DataFrame  # feature, importance, rank
    metrics: Dict[str, float]  # r2, rmse, cv_score
    shap_values: Optional[np.ndarray] = None
    shap_base_value: Optional[float] = None
    model_object: Any = None  # fitted model for prediction
    coefficients: Optional[pd.DataFrame] = None  # for mixed-effects


@dataclass
class WorkedExample:
    """A single worked example with SHAP explanation."""

    complaint_id: str
    category: str  # 'golden' | 'best_case' | 'worst_case' | etc.
    complaint_snippet: str  # truncated, anonymized
    product: str
    issue: str
    severity_percentile: float

    # Analysis-specific outcomes
    outcome_summary: Dict[str, Any] = field(default_factory=dict)

    # SHAP explanation
    shap_values: Optional[np.ndarray] = None
    shap_features: List[str] = field(default_factory=list)
    shap_base_value: Optional[float] = None

    # Causal interpretation
    interpretation: str = ""


@dataclass
class GoldenExample:
    """Extended example tracked through full pipeline."""

    complaint_id: str
    original_text: str
    complaint_snippet: str
    product: str
    issue: str
    severity_percentile: float

    # Step 1: Estimated persona
    estimated_persona: Dict[str, str] = field(default_factory=dict)

    # Step 2: Original tier assignment
    original_tiers: Dict[str, int] = field(default_factory=dict)  # model -> tier

    # Step 4-5: Persona variants
    persona_variants: Dict[str, str] = field(default_factory=dict)  # persona_key -> text
    persona_tiers: Dict[str, Dict[str, int]] = field(default_factory=dict)  # persona -> {model: tier}

    # Step 6-7: Neutralization
    neutralized_text: str = ""
    neutralized_tiers: Dict[str, int] = field(default_factory=dict)  # model -> tier

    # SHAP at each stage (per model)
    shap_persona_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    shap_neutralization_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Summary statistics
    tier_variance_by_model: Dict[str, float] = field(default_factory=dict)
    neutralization_effect_by_model: Dict[str, float] = field(default_factory=dict)


@dataclass
class HeterogeneousEffects:
    """Results from causal forest / CATE analysis."""

    treatment: str  # 'neutralization' | 'persona_age' | etc.
    llm_model: str
    cate_by_subgroup: pd.DataFrame = field(default_factory=pd.DataFrame)  # subgroup, cate, ci_lower, ci_upper
    important_moderators: List[str] = field(default_factory=list)
    overall_ate: float = 0.0
    ate_ci: Tuple[float, float] = (0.0, 0.0)
    policy_tree_rules: Optional[str] = None  # interpretable treatment rules


@dataclass
class MitigationAnalysisResults:
    """Complete results from mitigation effectiveness analysis."""

    # Model fits by LLM model
    neutralization_models: Dict[str, List[ModelFitResults]] = field(default_factory=dict)
    causal_persona_models: Dict[str, List[ModelFitResults]] = field(default_factory=dict)

    # Heterogeneous effects
    heterogeneous_effects: Dict[str, List[HeterogeneousEffects]] = field(default_factory=dict)

    # Examples
    golden_example: Optional[GoldenExample] = None
    supporting_examples: List[WorkedExample] = field(default_factory=list)

    # Summary statistics
    feature_importance_agreement: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "neutralization_models": {
                model: [self._model_fit_to_dict(m) for m in fits]
                for model, fits in self.neutralization_models.items()
            },
            "causal_persona_models": {
                model: [self._model_fit_to_dict(m) for m in fits]
                for model, fits in self.causal_persona_models.items()
            },
            "heterogeneous_effects": {
                model: [self._hte_to_dict(h) for h in effects]
                for model, effects in self.heterogeneous_effects.items()
            },
            "golden_example": self._golden_to_dict(self.golden_example) if self.golden_example else None,
            "supporting_examples": [self._example_to_dict(e) for e in self.supporting_examples],
            "feature_importance_agreement": self.feature_importance_agreement,
        }

    def _model_fit_to_dict(self, m: ModelFitResults) -> Dict[str, Any]:
        return {
            "model_type": m.model_type,
            "target": m.target,
            "llm_model": m.llm_model,
            "feature_importance": m.feature_importance.to_dict("records") if m.feature_importance is not None else [],
            "metrics": m.metrics,
            "coefficients": m.coefficients.to_dict("records") if m.coefficients is not None else None,
        }

    def _hte_to_dict(self, h: HeterogeneousEffects) -> Dict[str, Any]:
        return {
            "treatment": h.treatment,
            "llm_model": h.llm_model,
            "cate_by_subgroup": h.cate_by_subgroup.to_dict("records") if not h.cate_by_subgroup.empty else [],
            "important_moderators": h.important_moderators,
            "overall_ate": h.overall_ate,
            "ate_ci": list(h.ate_ci),
            "policy_tree_rules": h.policy_tree_rules,
        }

    def _example_to_dict(self, e: WorkedExample) -> Dict[str, Any]:
        return {
            "complaint_id": e.complaint_id,
            "category": e.category,
            "complaint_snippet": e.complaint_snippet,
            "product": e.product,
            "issue": e.issue,
            "severity_percentile": e.severity_percentile,
            "outcome_summary": e.outcome_summary,
            "shap_features": e.shap_features,
            "shap_values": e.shap_values.tolist() if e.shap_values is not None else None,
            "shap_base_value": e.shap_base_value,
            "interpretation": e.interpretation,
        }

    def _golden_to_dict(self, g: GoldenExample) -> Dict[str, Any]:
        return {
            "complaint_id": g.complaint_id,
            "original_text": g.original_text[:500] + "..." if len(g.original_text) > 500 else g.original_text,
            "complaint_snippet": g.complaint_snippet,
            "product": g.product,
            "issue": g.issue,
            "severity_percentile": g.severity_percentile,
            "estimated_persona": g.estimated_persona,
            "original_tiers": g.original_tiers,
            "persona_variants": {k: v[:300] + "..." if len(v) > 300 else v for k, v in g.persona_variants.items()},
            "persona_tiers": g.persona_tiers,
            "neutralized_text": g.neutralized_text[:500] + "..." if len(g.neutralized_text) > 500 else g.neutralized_text,
            "neutralized_tiers": g.neutralized_tiers,
            "tier_variance_by_model": g.tier_variance_by_model,
            "neutralization_effect_by_model": g.neutralization_effect_by_model,
        }


# -----------------------------------------------------------------------------
# Feature Engineering
# -----------------------------------------------------------------------------


def compute_neutralization_features(
    baseline_df: pd.DataFrame,
    neutralized_df: pd.DataFrame,
    personas_df: pd.DataFrame,
    persona_decisions_df: Optional[pd.DataFrame],
    complaints_df: pd.DataFrame,
    text_features_df: Optional[pd.DataFrame],
    model: str,
) -> pd.DataFrame:
    """
    Compute features for neutralization effectiveness analysis.

    Returns DataFrame with one row per complaint, containing:
    - Target variables: dispersion_delta, gap_closure_ratio
    - Features: product, issue, severity, text features, persona estimates
    """
    # Filter to specific model
    baseline_m = baseline_df[baseline_df["model"] == model].copy()
    neutralized_m = neutralized_df[neutralized_df["model"] == model].copy()
    personas_m = personas_df[personas_df["model"] == model].copy()

    # Merge baseline and neutralized
    merged = baseline_m[["case_id", "tier"]].rename(columns={"tier": "tier_original"})
    merged = merged.merge(
        neutralized_m[["case_id", "tier"]].rename(columns={"tier": "tier_neutralized"}),
        on="case_id",
        how="inner",
    )

    # Add persona estimates
    persona_cols = ["case_id", "age", "geographic", "education", "socioeconomic"]
    if "gender" in personas_m.columns:
        persona_cols.append("gender")
    if "esl" in personas_m.columns:
        persona_cols.append("esl")

    available_cols = [c for c in persona_cols if c in personas_m.columns]
    merged = merged.merge(personas_m[available_cols], on="case_id", how="left")

    # Add complaint metadata
    complaint_cols = ["case_id", "product_category", "issue_category", "monetary_severity", "issue_severity"]
    available_complaint_cols = [c for c in complaint_cols if c in complaints_df.columns]
    merged = merged.merge(complaints_df[available_complaint_cols], on="case_id", how="left")

    # Compute tier dispersion from persona decisions if available
    if persona_decisions_df is not None and len(persona_decisions_df) > 0:
        persona_dec_m = persona_decisions_df[persona_decisions_df["model"] == model]

        # Compute per-complaint tier variance across personas
        tier_stats = persona_dec_m.groupby("case_id")["tier"].agg(
            tier_variance="var",
            tier_range=lambda x: x.max() - x.min(),
            tier_mean="mean",
            n_distinct_tiers="nunique",
        ).reset_index()

        merged = merged.merge(tier_stats, on="case_id", how="left")
        merged["tier_variance"] = merged["tier_variance"].fillna(0)
        merged["tier_range"] = merged["tier_range"].fillna(0)
    else:
        merged["tier_variance"] = 0
        merged["tier_range"] = 0
        merged["tier_mean"] = merged["tier_original"]
        merged["n_distinct_tiers"] = 1

    # Compute target: dispersion delta (positive = neutralization reduced dispersion)
    # For now, use tier change as proxy if no persona decisions
    merged["tier_delta"] = merged["tier_neutralized"] - merged["tier_original"]
    merged["dispersion_delta"] = -merged["tier_variance"]  # placeholder, will be refined

    # Add text features if available
    if text_features_df is not None and len(text_features_df) > 0:
        # Get original text features
        orig_features = text_features_df[text_features_df["variant"] == "original"].copy()
        if len(orig_features) > 0:
            text_cols = [c for c in orig_features.columns if c not in ["case_id", "variant"]]
            orig_features = orig_features[["case_id"] + text_cols]
            # Prefix columns
            orig_features.columns = ["case_id"] + [f"text_{c}" for c in text_cols]
            merged = merged.merge(orig_features, on="case_id", how="left")

    return merged


def compute_causal_persona_features(
    persona_decisions_df: pd.DataFrame,
    personas_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    text_features_df: Optional[pd.DataFrame],
    model: str,
) -> pd.DataFrame:
    """
    Compute features for causal persona analysis.

    Returns DataFrame with one row per (complaint, persona_transition), containing:
    - Target: tier_delta (tier change from baseline persona to target persona)
    - Features: dimension changes, text feature deltas, complaint features
    """
    persona_dec_m = persona_decisions_df[persona_decisions_df["model"] == model].copy()
    personas_m = personas_df[personas_df["model"] == model].copy()

    if len(persona_dec_m) == 0:
        return pd.DataFrame()

    # Get baseline tier for each complaint (variant == "baseline" or estimated persona)
    baseline_dec = persona_dec_m[persona_dec_m["variant"] == "baseline"][["case_id", "tier"]].copy()
    baseline_dec = baseline_dec.rename(columns={"tier": "tier_baseline"})

    # If no explicit baseline, use original persona estimate
    if len(baseline_dec) == 0:
        # Fall back to first observation per case
        baseline_dec = persona_dec_m.groupby("case_id")["tier"].first().reset_index()
        baseline_dec = baseline_dec.rename(columns={"tier": "tier_baseline"})

    # Get persona variant tiers
    variant_dec = persona_dec_m[persona_dec_m["variant"] != "baseline"].copy()

    # Parse persona dimensions from variant label
    # Format: "young_urban_college_low" -> age=young, geo=urban, edu=college, ses=low
    def parse_persona(variant: str) -> Dict[str, str]:
        parts = variant.split("_")
        if len(parts) >= 4:
            return {
                "target_age": parts[0],
                "target_geo": parts[1],
                "target_edu": parts[2],
                "target_ses": parts[3],
            }
        return {}

    variant_dec = variant_dec.assign(**{
        "target_age": variant_dec["variant"].apply(lambda x: parse_persona(x).get("target_age", "")),
        "target_geo": variant_dec["variant"].apply(lambda x: parse_persona(x).get("target_geo", "")),
        "target_edu": variant_dec["variant"].apply(lambda x: parse_persona(x).get("target_edu", "")),
        "target_ses": variant_dec["variant"].apply(lambda x: parse_persona(x).get("target_ses", "")),
    })

    # Merge with baseline
    merged = variant_dec.merge(baseline_dec, on="case_id", how="left")
    merged["tier_delta"] = merged["tier"] - merged["tier_baseline"]

    # Add original persona estimates
    persona_cols = ["case_id", "age", "geographic", "education", "socioeconomic"]
    available_cols = [c for c in persona_cols if c in personas_m.columns]
    merged = merged.merge(
        personas_m[available_cols].rename(columns={
            "age": "source_age",
            "geographic": "source_geo",
            "education": "source_edu",
            "socioeconomic": "source_ses",
        }),
        on="case_id",
        how="left",
    )

    # Compute dimension changes (binary: did this dimension change?)
    merged["age_changed"] = (merged["source_age"] != merged["target_age"]).astype(int)
    merged["geo_changed"] = (merged["source_geo"] != merged["target_geo"]).astype(int)
    merged["edu_changed"] = (merged["source_edu"] != merged["target_edu"]).astype(int)
    merged["ses_changed"] = (merged["source_ses"] != merged["target_ses"]).astype(int)

    # Compute directional changes (signed: which direction?)
    # Age: young=0, old=1; change = target - source
    age_map = {"young": 0, "old": 1}
    merged["age_direction"] = merged["target_age"].map(age_map).fillna(0.5) - merged["source_age"].map(age_map).fillna(0.5)

    # SES: low=0, middle=1, high=2
    ses_map = {"low": 0, "middle": 1, "high": 2}
    merged["ses_direction"] = merged["target_ses"].map(ses_map).fillna(1) - merged["source_ses"].map(ses_map).fillna(1)

    # Education: high_school=0, college=1
    edu_map = {"high_school": 0, "college": 1}
    merged["edu_direction"] = merged["target_edu"].map(edu_map).fillna(0.5) - merged["source_edu"].map(edu_map).fillna(0.5)

    # Add complaint metadata
    complaint_cols = ["case_id", "product_category", "issue_category", "monetary_severity", "issue_severity"]
    available_complaint_cols = [c for c in complaint_cols if c in complaints_df.columns]
    merged = merged.merge(complaints_df[available_complaint_cols], on="case_id", how="left")

    return merged


# -----------------------------------------------------------------------------
# Model Fitting
# -----------------------------------------------------------------------------


def fit_lightgbm_model(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    target_name: str,
    llm_model: str,
    n_cv_folds: int = 5,
) -> ModelFitResults:
    """Fit LightGBM model with cross-validation and feature importance."""

    if not HAS_LIGHTGBM:
        # Fallback to sklearn
        return fit_sklearn_rf_model(X, y, feature_names, target_name, llm_model, n_cv_folds)

    # Encode categorical features
    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=["object", "category"]).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        label_encoders[col] = le

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_encoded, label=y, feature_name=feature_names)

    # Parameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    # Cross-validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=200,
            nfold=n_cv_folds,
            callbacks=[lgb.early_stopping(stopping_rounds=20)],
            return_cvbooster=False,
        )

    best_rounds = len(cv_results["valid rmse-mean"])
    cv_rmse = cv_results["valid rmse-mean"][-1]

    # Fit final model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_rounds,
    )

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    # Compute R-squared
    y_pred = model.predict(X_encoded)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return ModelFitResults(
        model_type="lightgbm",
        target=target_name,
        llm_model=llm_model,
        feature_importance=importance_df,
        metrics={
            "r2": r2,
            "rmse": cv_rmse,
            "cv_folds": n_cv_folds,
            "best_rounds": best_rounds,
        },
        model_object=model,
    )


def fit_sklearn_rf_model(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    target_name: str,
    llm_model: str,
    n_cv_folds: int = 5,
) -> ModelFitResults:
    """Fit sklearn RandomForest as fallback."""

    # Encode categorical features
    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # Handle missing values
    X_encoded = X_encoded.fillna(0)

    # Fit model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_encoded, y, cv=n_cv_folds, scoring="neg_root_mean_squared_error")
    cv_rmse = -np.mean(cv_scores)

    # Fit final model
    model.fit(X_encoded, y)

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    # R-squared
    y_pred = model.predict(X_encoded)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return ModelFitResults(
        model_type="sklearn_rf",
        target=target_name,
        llm_model=llm_model,
        feature_importance=importance_df,
        metrics={
            "r2": r2,
            "rmse": cv_rmse,
            "cv_folds": n_cv_folds,
        },
        model_object=model,
    )


def fit_mixed_effects_model(
    df: pd.DataFrame,
    target: str,
    fixed_effects: List[str],
    random_effect: str,
    llm_model: str,
) -> ModelFitResults:
    """Fit mixed-effects regression model."""

    if not HAS_STATSMODELS:
        logger.warning("statsmodels not available, skipping mixed-effects model")
        return ModelFitResults(
            model_type="mixed_effects",
            target=target,
            llm_model=llm_model,
            feature_importance=pd.DataFrame(),
            metrics={"error": "statsmodels not installed"},
        )

    # Check for sufficient data
    if len(df) < 50:
        return ModelFitResults(
            model_type="mixed_effects",
            target=target,
            llm_model=llm_model,
            feature_importance=pd.DataFrame(),
            metrics={"error": "insufficient data"},
        )

    # Check for variance in target
    if df[target].nunique() < 2:
        return ModelFitResults(
            model_type="mixed_effects",
            target=target,
            llm_model=llm_model,
            feature_importance=pd.DataFrame(),
            metrics={"error": "no variance in target"},
        )

    # Build formula
    formula = f"{target} ~ " + " + ".join(fixed_effects)

    try:
        # Fit model with warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*Hessian.*")
            warnings.filterwarnings("ignore", message=".*positive definite.*")

            model = mixedlm(formula, df, groups=df[random_effect])
            result = model.fit(method="lbfgs", maxiter=100)

        # Check for valid standard errors
        if not result.converged or np.any(np.isnan(result.bse_fe)):
            return ModelFitResults(
                model_type="mixed_effects",
                target=target,
                llm_model=llm_model,
                feature_importance=pd.DataFrame(),
                metrics={"error": "model did not converge properly"},
            )

        # Extract coefficients with safe handling of NaN
        std_err = result.bse_fe.values
        std_err = np.where(np.isnan(std_err), 1e-10, std_err)  # Replace NaN with small value

        coef_df = pd.DataFrame({
            "feature": result.fe_params.index,
            "coefficient": result.fe_params.values,
            "std_err": std_err,
            "z_value": result.fe_params.values / std_err,
            "p_value": result.pvalues.values[:len(result.fe_params)] if len(result.pvalues) >= len(result.fe_params) else [np.nan] * len(result.fe_params),
        })

        # Feature importance from |coefficient / std_err|
        coef_df["importance"] = np.abs(coef_df["z_value"])
        coef_df["importance"] = coef_df["importance"].replace([np.inf, -np.inf], 0).fillna(0)
        coef_df = coef_df.sort_values("importance", ascending=False)
        coef_df["rank"] = range(1, len(coef_df) + 1)

        return ModelFitResults(
            model_type="mixed_effects",
            target=target,
            llm_model=llm_model,
            feature_importance=coef_df[["feature", "importance", "rank"]],
            metrics={
                "aic": result.aic if np.isfinite(result.aic) else None,
                "bic": result.bic if np.isfinite(result.bic) else None,
                "llf": result.llf if np.isfinite(result.llf) else None,
                "converged": result.converged,
            },
            coefficients=coef_df,
            model_object=result,
        )

    except Exception as e:
        logger.debug(f"Mixed-effects model failed: {e}")
        return ModelFitResults(
            model_type="mixed_effects",
            target=target,
            llm_model=llm_model,
            feature_importance=pd.DataFrame(),
            metrics={"error": str(e)},
        )


# -----------------------------------------------------------------------------
# SHAP Analysis
# -----------------------------------------------------------------------------


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    max_samples: int = 1000,
) -> Tuple[np.ndarray, float, List[str]]:
    """Compute SHAP values for a fitted model."""

    if not HAS_SHAP:
        return np.array([]), 0.0, []

    # Subsample if needed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    # Encode categorical features
    X_encoded = X_sample.copy()
    categorical_cols = X_encoded.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    X_encoded = X_encoded.fillna(0)

    try:
        # Use TreeExplainer for tree-based models
        if HAS_LIGHTGBM and hasattr(model, "feature_name"):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback to KernelExplainer (slower)
            explainer = shap.KernelExplainer(model.predict, X_encoded.iloc[:100])

        shap_values = explainer.shap_values(X_encoded)
        base_value = explainer.expected_value

        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])

        return shap_values, base_value, feature_names

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return np.array([]), 0.0, []


def generate_shap_waterfall_data(
    shap_values: np.ndarray,
    base_value: float,
    feature_names: List[str],
    feature_values: np.ndarray,
    top_k: int = 8,
) -> Dict[str, Any]:
    """Generate data for SHAP waterfall plot."""

    if len(shap_values) == 0:
        return {}

    # Sort by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-top_k:][::-1]

    return {
        "base_value": base_value,
        "features": [feature_names[i] for i in top_indices],
        "feature_values": [float(feature_values[i]) for i in top_indices],
        "shap_values": [float(shap_values[i]) for i in top_indices],
        "prediction": base_value + np.sum(shap_values),
    }


# -----------------------------------------------------------------------------
# Heterogeneous Treatment Effects
# -----------------------------------------------------------------------------


def estimate_heterogeneous_effects(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: List[str],
    llm_model: str,
) -> HeterogeneousEffects:
    """Estimate heterogeneous treatment effects using Causal Forest."""

    if not HAS_ECONML:
        logger.debug("econml not available, skipping HTE analysis")
        return HeterogeneousEffects(
            treatment=treatment_col,
            llm_model=llm_model,
        )

    # Prepare data
    df_clean = df.dropna(subset=[treatment_col, outcome_col] + feature_cols)

    if len(df_clean) < 100:
        logger.debug(f"Insufficient data for HTE: {len(df_clean)} rows")
        return HeterogeneousEffects(treatment=treatment_col, llm_model=llm_model)

    # Check treatment variance - CausalForest requires variation in treatment
    T_raw = df_clean[treatment_col].values
    if len(np.unique(T_raw)) < 2:
        logger.debug(f"No treatment variation for HTE ({treatment_col}): all values are {T_raw[0]}")
        return HeterogeneousEffects(treatment=treatment_col, llm_model=llm_model)

    # Check treatment balance - need sufficient observations in both groups
    if np.min(np.bincount(T_raw.astype(int))) < 20:
        logger.debug(f"Insufficient treatment balance for HTE ({treatment_col})")
        return HeterogeneousEffects(treatment=treatment_col, llm_model=llm_model)

    # Encode features
    X = df_clean[feature_cols].copy()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(0)

    T = T_raw
    Y = df_clean[outcome_col].values

    # Check outcome variance
    if np.std(Y) < 1e-10:
        logger.debug(f"No outcome variation for HTE: std={np.std(Y)}")
        return HeterogeneousEffects(treatment=treatment_col, llm_model=llm_model)

    try:
        # Fit Causal Forest with warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*divide by zero.*")
            warnings.filterwarnings("ignore", message=".*invalid value.*")

            cf = CausalForestDML(
                model_y=GradientBoostingRegressor(n_estimators=50, max_depth=5),
                model_t=GradientBoostingRegressor(n_estimators=50, max_depth=5),
                n_estimators=100,
                random_state=42,
            )
            cf.fit(Y, T, X=X)

        # Get CATE estimates
        cate = cf.effect(X)

        # Overall ATE
        ate = np.mean(cate)
        ate_std = np.std(cate) / np.sqrt(len(cate))
        ate_ci = (ate - 1.96 * ate_std, ate + 1.96 * ate_std)

        # Feature importance for treatment effect heterogeneity
        # Use variance of CATE conditional on feature
        importance = []
        for i, col in enumerate(feature_cols):
            unique_vals = np.unique(X.iloc[:, i])
            if len(unique_vals) > 1:
                cate_by_val = [np.mean(cate[X.iloc[:, i] == v]) for v in unique_vals]
                var = np.var(cate_by_val)
            else:
                var = 0
            importance.append((col, var))

        importance.sort(key=lambda x: x[1], reverse=True)
        important_moderators = [x[0] for x in importance[:5]]

        # CATE by subgroup (top moderator)
        if important_moderators:
            top_mod = important_moderators[0]
            top_mod_idx = feature_cols.index(top_mod)
            unique_vals = np.unique(X.iloc[:, top_mod_idx])

            subgroup_data = []
            for val in unique_vals:
                mask = X.iloc[:, top_mod_idx] == val
                cate_sub = cate[mask]
                subgroup_data.append({
                    "subgroup": f"{top_mod}={val}",
                    "cate": np.mean(cate_sub),
                    "ci_lower": np.percentile(cate_sub, 2.5),
                    "ci_upper": np.percentile(cate_sub, 97.5),
                    "n": int(np.sum(mask)),
                })

            cate_by_subgroup = pd.DataFrame(subgroup_data)
        else:
            cate_by_subgroup = pd.DataFrame()

        # Policy tree interpretation
        policy_rules = None
        try:
            intrp = SingleTreeCateInterpreter(max_depth=3)
            intrp.interpret(cf, X)
            # Extract rules (simplified)
            policy_rules = "See policy tree visualization"
        except Exception:
            pass

        return HeterogeneousEffects(
            treatment=treatment_col,
            llm_model=llm_model,
            cate_by_subgroup=cate_by_subgroup,
            important_moderators=important_moderators,
            overall_ate=float(ate),
            ate_ci=(float(ate_ci[0]), float(ate_ci[1])),
            policy_tree_rules=policy_rules,
        )

    except np.linalg.LinAlgError as e:
        # Singular matrix usually means insufficient treatment variation
        logger.debug(f"HTE estimation failed (singular matrix): {e}")
        return HeterogeneousEffects(treatment=treatment_col, llm_model=llm_model)
    except Exception as e:
        logger.debug(f"HTE estimation failed: {e}")
        return HeterogeneousEffects(treatment=treatment_col, llm_model=llm_model)


# -----------------------------------------------------------------------------
# Example Selection
# -----------------------------------------------------------------------------


def select_golden_example(
    baseline_df: pd.DataFrame,
    neutralized_df: pd.DataFrame,
    persona_decisions_df: Optional[pd.DataFrame],
    personas_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    models: List[str],
) -> Optional[GoldenExample]:
    """Select a single complaint to track through the full pipeline."""

    # Build candidate set
    candidates = []

    for _, complaint in complaints_df.iterrows():
        case_id = complaint["case_id"]

        # Check we have data for both models
        baseline_rows = baseline_df[baseline_df["case_id"] == case_id]
        neutralized_rows = neutralized_df[neutralized_df["case_id"] == case_id]

        if len(baseline_rows) < len(models) or len(neutralized_rows) < len(models):
            continue

        # Compute persona sensitivity (if persona decisions available)
        if persona_decisions_df is not None:
            persona_rows = persona_decisions_df[persona_decisions_df["case_id"] == case_id]
            tier_variance = {}
            for model in models:
                model_tiers = persona_rows[persona_rows["model"] == model]["tier"]
                tier_variance[model] = model_tiers.var() if len(model_tiers) > 1 else 0
        else:
            tier_variance = {m: 0 for m in models}

        # Compute neutralization effect
        neut_effect = {}
        for model in models:
            orig_tier = baseline_rows[baseline_rows["model"] == model]["tier"].iloc[0]
            neut_tier = neutralized_rows[neutralized_rows["model"] == model]["tier"].iloc[0]
            neut_effect[model] = neut_tier - orig_tier

        # Compute model divergence
        if len(models) >= 2:
            model_divergence = abs(tier_variance.get(models[0], 0) - tier_variance.get(models[1], 0))
        else:
            model_divergence = 0

        # Score: prefer medium severity, common product, persona-sensitive, neutralization effect
        severity = complaint.get("issue_severity", "medium")
        severity_score = 1 if severity == "medium" else 0.5

        product = complaint.get("product_category", "")
        product_score = 1 if product in ["Credit card", "Checking or savings account"] else 0.5

        sensitivity_score = sum(tier_variance.values())
        neut_score = sum(abs(v) for v in neut_effect.values())

        total_score = (
            severity_score * 0.2 +
            product_score * 0.2 +
            sensitivity_score * 0.3 +
            neut_score * 0.2 +
            model_divergence * 0.1
        )

        candidates.append({
            "case_id": case_id,
            "score": total_score,
            "tier_variance": tier_variance,
            "neut_effect": neut_effect,
            "complaint": complaint,
        })

    if not candidates:
        return None

    # Select top candidate
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    case_id = best["case_id"]
    complaint = best["complaint"]

    # Build golden example
    golden = GoldenExample(
        complaint_id=case_id,
        original_text=complaint.get("narrative", ""),
        complaint_snippet=complaint.get("narrative", "")[:200] + "...",
        product=complaint.get("product_category", ""),
        issue=complaint.get("issue_category", ""),
        severity_percentile=50.0,  # Placeholder
        tier_variance_by_model=best["tier_variance"],
        neutralization_effect_by_model=best["neut_effect"],
    )

    # Add persona estimates
    for model in models:
        persona_row = personas_df[
            (personas_df["case_id"] == case_id) & (personas_df["model"] == model)
        ]
        if len(persona_row) > 0:
            golden.estimated_persona[model] = {
                "age": persona_row.iloc[0].get("age", ""),
                "geographic": persona_row.iloc[0].get("geographic", ""),
                "education": persona_row.iloc[0].get("education", ""),
                "socioeconomic": persona_row.iloc[0].get("socioeconomic", ""),
            }

    # Add tier assignments
    for model in models:
        orig_row = baseline_df[(baseline_df["case_id"] == case_id) & (baseline_df["model"] == model)]
        if len(orig_row) > 0:
            golden.original_tiers[model] = int(orig_row.iloc[0]["tier"])

        neut_row = neutralized_df[(neutralized_df["case_id"] == case_id) & (neutralized_df["model"] == model)]
        if len(neut_row) > 0:
            golden.neutralized_tiers[model] = int(neut_row.iloc[0]["tier"])

    # Add persona variant tiers
    if persona_decisions_df is not None:
        for model in models:
            persona_rows = persona_decisions_df[
                (persona_decisions_df["case_id"] == case_id) &
                (persona_decisions_df["model"] == model)
            ]
            golden.persona_tiers[model] = {}
            for _, row in persona_rows.iterrows():
                variant = row.get("variant", "")
                if variant and variant != "baseline":
                    golden.persona_tiers[model][variant] = int(row["tier"])

    return golden


def select_supporting_examples(
    feature_df: pd.DataFrame,
    shap_values: np.ndarray,
    shap_base_value: float,
    feature_names: List[str],
    target_col: str,
    complaints_df: pd.DataFrame,
    n_examples: int = 3,
    category_prefix: str = "neutralization",
) -> List[WorkedExample]:
    """Select supporting examples based on effect direction and magnitude."""

    examples = []

    if len(feature_df) == 0 or len(shap_values) == 0:
        return examples

    # Align indices
    if len(shap_values) != len(feature_df):
        logger.warning(f"SHAP values length {len(shap_values)} != feature_df length {len(feature_df)}")
        return examples

    # Get target values
    target_values = feature_df[target_col].values

    # Categories to select
    categories = [
        ("best_case", lambda x: x, True),  # Highest positive effect
        ("worst_case", lambda x: x, False),  # Lowest (most negative) effect
        ("typical", lambda x: abs(x - np.median(target_values)), False),  # Closest to median
    ]

    selected_indices = set()

    for category_name, score_fn, descending in categories:
        if len(examples) >= n_examples:
            break

        # Compute scores
        scores = [score_fn(v) for v in target_values]
        sorted_indices = np.argsort(scores)
        if descending:
            sorted_indices = sorted_indices[::-1]

        # Find first unselected index
        for idx in sorted_indices:
            if idx not in selected_indices:
                selected_indices.add(idx)

                # Get complaint info
                case_id = feature_df.iloc[idx].get("case_id", f"unknown_{idx}")
                complaint_row = complaints_df[complaints_df["case_id"] == case_id]

                if len(complaint_row) > 0:
                    product = complaint_row.iloc[0].get("product_category", "")
                    issue = complaint_row.iloc[0].get("issue_category", "")
                    snippet = complaint_row.iloc[0].get("narrative", "")[:150] + "..."
                else:
                    product = ""
                    issue = ""
                    snippet = ""

                # Generate SHAP explanation
                shap_data = generate_shap_waterfall_data(
                    shap_values[idx],
                    shap_base_value,
                    feature_names,
                    feature_df.iloc[idx].values,
                )

                # Generate interpretation
                interpretation = _generate_interpretation(
                    category_name,
                    target_col,
                    target_values[idx],
                    shap_data,
                )

                examples.append(WorkedExample(
                    complaint_id=str(case_id),
                    category=f"{category_prefix}_{category_name}",
                    complaint_snippet=snippet,
                    product=product,
                    issue=issue,
                    severity_percentile=50.0,  # Placeholder
                    outcome_summary={
                        target_col: float(target_values[idx]),
                    },
                    shap_values=shap_values[idx],
                    shap_features=feature_names,
                    shap_base_value=shap_base_value,
                    interpretation=interpretation,
                ))

                break

    return examples


def _generate_interpretation(
    category: str,
    target: str,
    value: float,
    shap_data: Dict[str, Any],
) -> str:
    """Generate plain-language causal interpretation."""

    if not shap_data:
        return f"This complaint shows a {target} of {value:.2f}."

    top_features = shap_data.get("features", [])[:3]
    top_shap = shap_data.get("shap_values", [])[:3]

    if not top_features:
        return f"This complaint shows a {target} of {value:.2f}."

    # Build interpretation
    parts = []
    for feat, sv in zip(top_features, top_shap):
        direction = "increases" if sv > 0 else "decreases"
        parts.append(f"{feat} {direction} the effect by {abs(sv):.3f}")

    base_text = {
        "best_case": "This complaint benefited most from the intervention.",
        "worst_case": "This complaint was negatively affected by the intervention.",
        "typical": "This complaint shows a typical response to the intervention.",
    }.get(category, "")

    return f"{base_text} Key drivers: {'; '.join(parts)}."


# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------


class MitigationAnalyzer:
    """Orchestrates mitigation effectiveness analysis."""

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        neutralized_df: pd.DataFrame,
        personas_df: pd.DataFrame,
        complaints_df: pd.DataFrame,
        persona_decisions_df: Optional[pd.DataFrame] = None,
        text_features_df: Optional[pd.DataFrame] = None,
    ):
        self.baseline_df = baseline_df
        self.neutralized_df = neutralized_df
        self.personas_df = personas_df
        self.complaints_df = complaints_df
        self.persona_decisions_df = persona_decisions_df
        self.text_features_df = text_features_df

        self.results = MitigationAnalysisResults()

    def run(
        self,
        models: List[str],
        n_examples: int = 7,
    ) -> MitigationAnalysisResults:
        """Run full mitigation effectiveness analysis."""

        logger.info("Starting mitigation effectiveness analysis...")

        # 1. Select golden example (shared across models)
        logger.info("Selecting golden example...")
        self.results.golden_example = select_golden_example(
            self.baseline_df,
            self.neutralized_df,
            self.persona_decisions_df,
            self.personas_df,
            self.complaints_df,
            models,
        )

        if self.results.golden_example:
            logger.info(f"Golden example selected: {self.results.golden_example.complaint_id}")

        # 2. Run analysis for each model
        for model in models:
            logger.info(f"Running analysis for {model}...")

            # 2a. Neutralization effectiveness
            neut_results = self._analyze_neutralization(model)
            self.results.neutralization_models[model] = neut_results

            # 2b. Causal persona analysis
            if self.persona_decisions_df is not None:
                causal_results = self._analyze_causal_persona(model)
                self.results.causal_persona_models[model] = causal_results

            # 2c. Heterogeneous treatment effects
            hte_results = self._analyze_heterogeneous_effects(model)
            self.results.heterogeneous_effects[model] = hte_results

        # 3. Compute feature importance agreement across models
        self._compute_importance_agreement(models)

        # 4. Select supporting examples (from first model's results)
        if models and self.results.neutralization_models.get(models[0]):
            n_supporting = n_examples - 1  # Reserve 1 for golden
            # Split between neutralization and causal
            n_neut = n_supporting // 2
            n_causal = n_supporting - n_neut

            # Get supporting examples from neutralization analysis
            neut_model_results = self.results.neutralization_models[models[0]]
            if neut_model_results:
                self._select_supporting_examples(models[0], n_neut, n_causal)

        logger.info("Mitigation effectiveness analysis complete.")
        return self.results

    def _analyze_neutralization(self, model: str) -> List[ModelFitResults]:
        """Analyze neutralization effectiveness for a single model."""

        results = []

        # Compute features
        feature_df = compute_neutralization_features(
            self.baseline_df,
            self.neutralized_df,
            self.personas_df,
            self.persona_decisions_df,
            self.complaints_df,
            self.text_features_df,
            model,
        )

        if len(feature_df) < 50:
            logger.warning(f"Insufficient data for neutralization analysis: {len(feature_df)} rows")
            return results

        # Define feature columns (exclude targets and IDs)
        exclude_cols = ["case_id", "tier_original", "tier_neutralized", "tier_delta",
                       "dispersion_delta", "tier_variance", "tier_range", "tier_mean", "n_distinct_tiers"]
        feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

        # Remove columns with too many missing values
        valid_cols = []
        for col in feature_cols:
            missing_pct = feature_df[col].isna().mean()
            if missing_pct < 0.5:
                valid_cols.append(col)
        feature_cols = valid_cols

        if not feature_cols:
            logger.warning("No valid feature columns for neutralization analysis")
            return results

        X = feature_df[feature_cols]

        # Target 1: Tier delta
        y_delta = feature_df["tier_delta"].fillna(0).values

        # Fit LightGBM
        lgb_result = fit_lightgbm_model(
            X, y_delta, feature_cols, "tier_delta", model
        )

        # Compute SHAP
        if lgb_result.model_object is not None:
            shap_vals, base_val, feat_names = compute_shap_values(
                lgb_result.model_object, X, feature_cols
            )
            lgb_result.shap_values = shap_vals
            lgb_result.shap_base_value = base_val

        results.append(lgb_result)

        # Fit mixed-effects model (if complaint grouping available)
        if "case_id" in feature_df.columns and len(feature_df["case_id"].unique()) > 10:
            # Select numeric features only for formula
            numeric_cols = feature_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                me_result = fit_mixed_effects_model(
                    feature_df.dropna(subset=numeric_cols + ["tier_delta"]),
                    "tier_delta",
                    numeric_cols[:10],  # Limit features
                    "case_id",
                    model,
                )
                results.append(me_result)

        return results

    def _analyze_causal_persona(self, model: str) -> List[ModelFitResults]:
        """Analyze causal persona effects for a single model."""

        results = []

        if self.persona_decisions_df is None or len(self.persona_decisions_df) == 0:
            return results

        # Compute features
        feature_df = compute_causal_persona_features(
            self.persona_decisions_df,
            self.personas_df,
            self.complaints_df,
            self.text_features_df,
            model,
        )

        if len(feature_df) < 50:
            logger.warning(f"Insufficient data for causal persona analysis: {len(feature_df)} rows")
            return results

        # Define feature columns
        feature_cols = [
            "age_changed", "geo_changed", "edu_changed", "ses_changed",
            "age_direction", "ses_direction", "edu_direction",
        ]

        # Add complaint features if available
        for col in ["product_category", "issue_category", "monetary_severity", "issue_severity"]:
            if col in feature_df.columns:
                feature_cols.append(col)

        # Filter to existing columns
        feature_cols = [c for c in feature_cols if c in feature_df.columns]

        if not feature_cols:
            logger.warning("No valid feature columns for causal analysis")
            return results

        X = feature_df[feature_cols]
        y = feature_df["tier_delta"].fillna(0).values

        # Fit LightGBM
        lgb_result = fit_lightgbm_model(
            X, y, feature_cols, "tier_delta", model
        )

        # Compute SHAP
        if lgb_result.model_object is not None:
            shap_vals, base_val, feat_names = compute_shap_values(
                lgb_result.model_object, X, feature_cols
            )
            lgb_result.shap_values = shap_vals
            lgb_result.shap_base_value = base_val

        results.append(lgb_result)

        return results

    def _analyze_heterogeneous_effects(self, model: str) -> List[HeterogeneousEffects]:
        """Analyze heterogeneous treatment effects."""

        results = []

        # Note: Neutralization HTE is skipped because all complaints receive the
        # same treatment (neutralization), so there's no treatment variation.
        # CausalForest requires variation in treatment assignment.
        # Instead, we analyze which complaint FEATURES predict larger neutralization effects
        # by using the tier_delta as the outcome and features as moderators.
        # This is captured in the neutralization models (LightGBM/mixed-effects) above.

        # Persona dimension HTEs - these have natural treatment variation
        # (some complaints get age_changed=1, others get age_changed=0)
        if self.persona_decisions_df is not None:
            causal_df = compute_causal_persona_features(
                self.persona_decisions_df,
                self.personas_df,
                self.complaints_df,
                self.text_features_df,
                model,
            )

            if len(causal_df) >= 100:
                for dim in ["age_changed", "ses_changed", "edu_changed"]:
                    if dim in causal_df.columns:
                        feature_cols = [c for c in ["product_category", "issue_category", "issue_severity"]
                                       if c in causal_df.columns]
                        if feature_cols:
                            hte = estimate_heterogeneous_effects(
                                causal_df,
                                dim,
                                "tier_delta",
                                feature_cols,
                                model,
                            )
                            results.append(hte)

        return results

    def _compute_importance_agreement(self, models: List[str]) -> None:
        """Compute agreement in feature importance rankings across models."""

        if len(models) < 2:
            return

        # Compare neutralization model importances
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                neut1 = self.results.neutralization_models.get(m1, [])
                neut2 = self.results.neutralization_models.get(m2, [])

                if neut1 and neut2:
                    # Get LightGBM results
                    lgb1 = next((r for r in neut1 if r.model_type in ["lightgbm", "sklearn_rf"]), None)
                    lgb2 = next((r for r in neut2 if r.model_type in ["lightgbm", "sklearn_rf"]), None)

                    if lgb1 and lgb2 and not lgb1.feature_importance.empty and not lgb2.feature_importance.empty:
                        # Compute rank correlation
                        merged = lgb1.feature_importance[["feature", "rank"]].merge(
                            lgb2.feature_importance[["feature", "rank"]],
                            on="feature",
                            suffixes=("_1", "_2"),
                        )

                        if len(merged) > 2:
                            from scipy.stats import spearmanr
                            corr, _ = spearmanr(merged["rank_1"], merged["rank_2"])
                            self.results.feature_importance_agreement[f"{m1}_vs_{m2}"] = corr

    def _select_supporting_examples(self, model: str, n_neut: int, n_causal: int) -> None:
        """Select supporting examples with SHAP explanations."""

        # Get neutralization model results
        neut_results = self.results.neutralization_models.get(model, [])
        lgb_result = next((r for r in neut_results if r.model_type in ["lightgbm", "sklearn_rf"]), None)

        if lgb_result and lgb_result.shap_values is not None and len(lgb_result.shap_values) > 0:
            # Recompute feature df for example selection
            feature_df = compute_neutralization_features(
                self.baseline_df,
                self.neutralized_df,
                self.personas_df,
                self.persona_decisions_df,
                self.complaints_df,
                self.text_features_df,
                model,
            )

            # Need to align feature_df with SHAP values
            # For now, select based on available data
            exclude_cols = ["case_id", "tier_original", "tier_neutralized", "tier_delta",
                           "dispersion_delta", "tier_variance", "tier_range", "tier_mean", "n_distinct_tiers"]
            feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
            valid_cols = [c for c in feature_cols if feature_df[c].isna().mean() < 0.5]

            if valid_cols and len(feature_df) == len(lgb_result.shap_values):
                neut_examples = select_supporting_examples(
                    feature_df,
                    lgb_result.shap_values,
                    lgb_result.shap_base_value or 0.0,
                    valid_cols,
                    "tier_delta",
                    self.complaints_df,
                    n_examples=n_neut,
                    category_prefix="neutralization",
                )
                self.results.supporting_examples.extend(neut_examples)

        # Get causal model results
        causal_results = self.results.causal_persona_models.get(model, [])
        lgb_causal = next((r for r in causal_results if r.model_type in ["lightgbm", "sklearn_rf"]), None)

        if lgb_causal and lgb_causal.shap_values is not None and len(lgb_causal.shap_values) > 0:
            feature_df = compute_causal_persona_features(
                self.persona_decisions_df,
                self.personas_df,
                self.complaints_df,
                self.text_features_df,
                model,
            )

            feature_cols = [
                "age_changed", "geo_changed", "edu_changed", "ses_changed",
                "age_direction", "ses_direction", "edu_direction",
            ]
            feature_cols = [c for c in feature_cols if c in feature_df.columns]

            if feature_cols and len(feature_df) == len(lgb_causal.shap_values):
                causal_examples = select_supporting_examples(
                    feature_df,
                    lgb_causal.shap_values,
                    lgb_causal.shap_base_value or 0.0,
                    feature_cols,
                    "tier_delta",
                    self.complaints_df,
                    n_examples=n_causal,
                    category_prefix="causal",
                )
                self.results.supporting_examples.extend(causal_examples)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def run_mitigation_effectiveness_analysis(
    baseline_df: pd.DataFrame,
    neutralized_df: pd.DataFrame,
    personas_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    models: List[str],
    persona_decisions_df: Optional[pd.DataFrame] = None,
    text_features_df: Optional[pd.DataFrame] = None,
    n_examples: int = 7,
) -> Dict[str, Any]:
    """
    Run mitigation effectiveness analysis and return results as dictionary.

    This is the main entry point called from experiment.py.
    """
    analyzer = MitigationAnalyzer(
        baseline_df=baseline_df,
        neutralized_df=neutralized_df,
        personas_df=personas_df,
        complaints_df=complaints_df,
        persona_decisions_df=persona_decisions_df,
        text_features_df=text_features_df,
    )

    results = analyzer.run(models=models, n_examples=n_examples)
    return results.to_dict()
