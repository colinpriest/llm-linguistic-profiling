# Scikit-learn Workflows Reference

## Contents
- IPW Propensity Modeling Workflow
- Ridge Logistic Regression Fallback Workflow
- Topic Model Feature Extraction Workflow
- Model Caching Pattern

---

## IPW Propensity Modeling Workflow

Used in `src/ipw_robustness.py` for inverse probability weighting:

### Complete Pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import warnings

def fit_propensity_model(df, treatment_col, cat_cols, num_cols):
    """Fit propensity model P(treatment=1|covariates)."""
    parts = []
    
    # 1. Encode categorical features
    if cat_cols:
        enc = OneHotEncoder(
            sparse_output=False,
            drop="first",
            handle_unknown="infrequent_if_exist",
        )
        cat_X = enc.fit_transform(df[cat_cols].fillna("Unknown"))
        parts.append(cat_X)
    
    # 2. Process numeric features
    if num_cols:
        num_X = df[num_cols].fillna(0).values.astype(float)
        parts.append(num_X)
    
    # 3. Stack features
    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    
    # 4. Fit logistic regression (suppress convergence warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
        model.fit(X, df[treatment_col])
    
    # 5. Get propensity scores with clipping
    propensity = np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99)
    
    # 6. Calculate and trim weights
    weights = 1.0 / propensity
    trim_val = np.percentile(weights, 99)
    weights = np.minimum(weights, trim_val)
    
    return propensity, weights
```

### Workflow Checklist

Copy this checklist and track progress:
- [ ] Step 1: Identify categorical and numeric covariate columns
- [ ] Step 2: Handle missing values (fillna for categoricals, 0 for numerics)
- [ ] Step 3: Fit OneHotEncoder with drop="first"
- [ ] Step 4: Stack encoded + numeric arrays with np.hstack
- [ ] Step 5: Fit LogisticRegression with max_iter=2000
- [ ] Step 6: Clip propensity to [0.01, 0.99]
- [ ] Step 7: Calculate weights as 1/propensity
- [ ] Step 8: Trim weights at 99th percentile

---

## Ridge Logistic Regression Fallback Workflow

Used in `src/statistics.py` when statsmodels MLE fails:

### When to Use Ridge Fallback

```python
def run_logistic_regression(df, outcome_col, predictor_cols, alpha=1.0):
    """Try MLE first, fall back to ridge if it fails."""
    import statsmodels.api as sm
    
    X = pd.get_dummies(df[predictor_cols], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    y = df[outcome_col]
    
    try:
        # Primary: statsmodels MLE
        model = sm.Logit(y, X).fit(disp=0, maxiter=100)
        if not model.mle_retvals["converged"]:
            raise ValueError("MLE did not converge")
        return extract_results(model)
    except Exception:
        # Fallback: sklearn ridge
        return _ridge_logit(X, y, alpha=alpha)
```

### Ridge Implementation

```python
from sklearn.linear_model import LogisticRegression
from src.regression_diagnostics import standardise_predictors

def _ridge_logit(X, y, alpha=1.0):
    """Ridge logistic regression fallback."""
    # 1. Standardize predictors for stable regularization
    X_std, std_stats = standardise_predictors(X)
    param_names = list(X.columns)
    
    # 2. Fit ridge logit
    lr = LogisticRegression(
        l1_ratio=0,           # Pure L2
        C=1.0 / alpha,        # sklearn uses C = 1/lambda
        solver="lbfgs",
        max_iter=500,
        fit_intercept=False,  # Constant already in X
    )
    lr.fit(X_std.values, y.values)
    coef = lr.coef_.flatten()
    
    # 3. Back-transform to original scale
    coef_orig = np.copy(coef)
    for i, col in enumerate(param_names):
        if col in std_stats:
            mean, sd = std_stats[col]
            coef_orig[i] = coef[i] / sd
    
    return {
        "coef": dict(zip(param_names, coef_orig.tolist())),
        "odds_ratios": dict(zip(param_names, np.exp(coef_orig).tolist())),
        "method": "ridge_fallback",
    }
```

### Validation Loop

1. Fit statsmodels MLE
2. Check convergence flag
3. If failed, fit sklearn ridge on standardized data
4. Back-transform coefficients to original scale
5. Return results with method indicator

---

## Topic Model Feature Extraction Workflow

Used in `src/text_features.py` for NLP features:

### Global Model Caching Pattern

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Module-level cache
_lda_model = None
_vectorizer = None

def fit_topic_model(texts, n_topics=10):
    """Fit and cache LDA model on corpus."""
    global _lda_model, _vectorizer
    
    _vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=5000,
        stop_words="english",
    )
    dtm = _vectorizer.fit_transform(texts)
    
    _lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="online",
    )
    _lda_model.fit(dtm)
    
    return _lda_model, _vectorizer


def extract_topic_features(text: str) -> dict:
    """Extract topic features for single document."""
    if _lda_model is None or _vectorizer is None:
        return {"dominant_topic": 0, "topic_coherence": 0.0}
    
    vec = _vectorizer.transform([text])
    dist = _lda_model.transform(vec)[0]
    
    return {
        "topic_distribution": dist.tolist(),
        "dominant_topic": int(np.argmax(dist)),
        "topic_coherence": float(np.max(dist)),
    }
```

### Workflow Checklist

Copy this checklist and track progress:
- [ ] Step 1: Call fit_topic_model(corpus_texts) once at startup
- [ ] Step 2: For each document, call extract_topic_features(text)
- [ ] Step 3: Verify model is cached before extraction (guard against None)

---

## Integration with pandas/numpy

### DataFrame → sklearn → DataFrame

```python
def add_propensity_weights(df, treatment_col, covariates):
    """Add propensity weights to DataFrame."""
    # sklearn processing
    propensity, weights = fit_propensity_model(
        df, treatment_col,
        cat_cols=[c for c in covariates if df[c].dtype == "object"],
        num_cols=[c for c in covariates if df[c].dtype in ["int64", "float64"]],
    )
    
    # Back to DataFrame
    df = df.copy()
    df["propensity"] = propensity
    df["ipw_weight"] = weights
    return df
```

### Combining with scipy Statistics

```python
from scipy import stats

def weighted_ttest(df, outcome_col, group_col, weight_col):
    """Weighted t-test using IPW weights."""
    g1 = df[df[group_col] == 1]
    g0 = df[df[group_col] == 0]
    
    # Weighted means using numpy
    mean1 = np.average(g1[outcome_col], weights=g1[weight_col])
    mean0 = np.average(g0[outcome_col], weights=g0[weight_col])
    
    # Standard t-test on weighted residuals
    return stats.ttest_ind(
        g1[outcome_col] * g1[weight_col],
        g0[outcome_col] * g0[weight_col],
    )
```

---

## Related Skills

- See the **pandas** skill for DataFrame operations
- See the **numpy** skill for array computations and weighted statistics
- See the **scipy** skill for statistical tests
- See the **statsmodels** skill for primary regression methods