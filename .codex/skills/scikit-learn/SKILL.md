---
name: scikit-learn
description: |
  Applies machine learning algorithms and classification models.
  Use when: fitting logistic regression for propensity scores, applying ridge regularization,
  encoding categorical features with OneHotEncoder, extracting text features with LDA/CountVectorizer,
  or building ML components for statistical analysis pipelines.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Scikit-learn Skill

This project uses scikit-learn sparingly for specific ML tasks: propensity score modeling (IPW), ridge-regularized logistic regression as a fallback for statsmodels, and topic modeling for text features. No sklearn Pipelines or cross-validation—custom preprocessing with pandas/numpy instead.

## Quick Start

### Logistic Regression for Propensity Scores

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Encode categoricals, stack with numerics
enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="infrequent_if_exist")
cat_X = enc.fit_transform(df[cat_cols].fillna("Unknown"))
num_X = df[num_cols].fillna(0).values.astype(float)
X = np.hstack([cat_X, num_X])

# Fit propensity model
model = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
model.fit(X, df["treatment"])
propensity = np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99)
```

### Ridge Logistic Regression (Fallback)

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    l1_ratio=0,           # Pure L2 (Ridge)
    C=1.0 / alpha,        # Inverse regularization
    solver="lbfgs",
    max_iter=500,
    fit_intercept=False,  # If constant already in X
)
lr.fit(X_standardized, y)
coef = lr.coef_.flatten()
```

### Topic Modeling with LDA

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=5000, stop_words="english")
dtm = vectorizer.fit_transform(texts)

lda = LatentDirichletAllocation(n_components=10, random_state=42, max_iter=20, learning_method="online")
lda.fit(dtm)

# Get topic distribution for new text
dist = lda.transform(vectorizer.transform([text]))[0]
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| OneHotEncoder | Categorical → numeric | `enc.fit_transform(df[cols])` |
| LogisticRegression | Propensity/classification | `model.predict_proba(X)[:, 1]` |
| Ridge (L2) | Regularized fallback | `C=1/alpha, l1_ratio=0` |
| CountVectorizer | Text → DTM | `max_df=0.95, min_df=2` |
| LDA | Topic extraction | `n_components=10` |

## Common Patterns

### Propensity Score Clipping

**When:** Using propensity for inverse probability weighting

```python
# Avoid extreme weights from near-0 or near-1 propensities
propensity = np.clip(propensity, 0.01, 0.99)
weights = 1.0 / propensity
weights = np.minimum(weights, np.percentile(weights, 99))  # Trim extremes
```

### Back-Transform Standardized Coefficients

**When:** Ridge logit on standardized predictors needs interpretable coefficients

```python
# After fitting on standardized X
for i, col in enumerate(param_names):
    if col in std_stats:
        mean, sd = std_stats[col]
        coef_orig[i] = coef[i] / sd  # Back to original scale
```

## See Also

- [patterns](references/patterns.md) - Encoding, regularization, and text patterns
- [workflows](references/workflows.md) - IPW pipeline, fallback regression, topic extraction

## Related Skills

- **pandas** skill for DataFrame preprocessing before sklearn
- **numpy** skill for array operations and numerical stability
- **scipy** skill for statistical tests alongside sklearn models
- **statsmodels** skill for primary regression (sklearn is fallback)