# Scikit-learn Patterns Reference

## Contents
- Categorical Encoding Patterns
- Logistic Regression Patterns
- Text Feature Extraction Patterns
- Numerical Stability Patterns
- Anti-Patterns

---

## Categorical Encoding Patterns

### OneHotEncoder for Mixed Data

Used in `src/ipw_robustness.py:72-85` for propensity model features:

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

enc = OneHotEncoder(
    sparse_output=False,              # Dense array for hstack
    drop="first",                     # Avoid multicollinearity
    handle_unknown="infrequent_if_exist",  # Robust to new categories
)
cat_X = enc.fit_transform(df[cat_cols].fillna("Unknown"))

# Combine with numeric features
num_X = df[num_cols].fillna(0).values.astype(float)
X = np.hstack([cat_X, num_X]) if cat_cols else num_X
```

### pandas get_dummies Alternative

Used in `src/statistics.py` for simpler cases:

```python
# GOOD - When all predictors are categorical
X = pd.get_dummies(df[predictor_cols], drop_first=True, dtype=float)
```

**When to use which:**
- `OneHotEncoder`: Mixed categorical + numeric, need fitted encoder for new data
- `get_dummies`: Pure categorical, one-off transformation

---

## Logistic Regression Patterns

### Standard MLE (No Regularization)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=2000,      # Increase from default 100
    random_state=42,    # Reproducibility
    solver="lbfgs",     # Good for small-to-medium datasets
)
model.fit(X, y)
proba = model.predict_proba(X)[:, 1]  # P(y=1)
```

### Ridge Regularization (L2 Fallback)

From `src/statistics.py:620-669` - used when statsmodels MLE fails:

```python
lr = LogisticRegression(
    l1_ratio=0,           # Pure L2 (no sparsity)
    C=1.0 / alpha,        # C = 1/lambda in sklearn notation
    solver="lbfgs",
    max_iter=500,
    fit_intercept=False,  # If X includes constant column
)
lr.fit(X_standardized.values, y.values)

# Extract coefficients
coef = lr.coef_.flatten()
odds_ratios = np.exp(coef)
```

**Why L2 over L1:** Ridge preserves all pre-registered predictors. LASSO would zero out coefficients, violating confirmatory analysis requirements.

---

## Text Feature Extraction Patterns

### CountVectorizer Configuration

From `src/text_features.py:234-250`:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_df=0.95,          # Ignore terms in >95% of docs (too common)
    min_df=2,             # Ignore terms in <2 docs (too rare)
    max_features=5000,    # Cap vocabulary
    stop_words="english", # Remove stopwords
)
dtm = vectorizer.fit_transform(texts)
```

### LDA Topic Modeling

```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(
    n_components=10,           # Number of topics
    random_state=42,
    max_iter=20,
    learning_method="online",  # Efficient for large corpora
)
lda.fit(dtm)

# Extract features for single document
def topic_features(text: str) -> dict:
    vec = vectorizer.transform([text])
    dist = lda.transform(vec)[0]
    return {
        "dominant_topic": int(np.argmax(dist)),
        "topic_coherence": float(max(dist)),
    }
```

---

## Numerical Stability Patterns

### Propensity Clipping

```python
# GOOD - Prevent extreme IPW weights
propensity = np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99)
```

### Weight Trimming

```python
# GOOD - Cap at 99th percentile to limit outlier influence
weights = 1.0 / propensity
trim_val = np.percentile(weights, 99)
weights_trimmed = np.minimum(weights, trim_val)
```

### Coefficient Back-Transformation

```python
# After fitting on standardized data
coef_orig = np.copy(coef)
for i, col in enumerate(param_names):
    if col in std_stats:
        mean, sd = std_stats[col]
        coef_orig[i] = coef[i] / sd  # Rescale to original units
```

---

## Anti-Patterns

### WARNING: Using Default max_iter

**The Problem:**

```python
# BAD - Default max_iter=100 often insufficient
model = LogisticRegression()
model.fit(X, y)  # ConvergenceWarning
```

**Why This Breaks:**
1. Convergence warnings indicate unreliable coefficients
2. Propensity scores become inaccurate
3. Results are not reproducible across runs

**The Fix:**

```python
# GOOD - Increase iterations
model = LogisticRegression(max_iter=2000, solver="lbfgs")
```

---

### WARNING: Sparse Output with hstack

**The Problem:**

```python
# BAD - sparse_output=True (default) breaks np.hstack
enc = OneHotEncoder()
cat_X = enc.fit_transform(df[cat_cols])  # Sparse matrix
X = np.hstack([cat_X, num_X])  # ValueError
```

**The Fix:**

```python
# GOOD - Explicit dense output
enc = OneHotEncoder(sparse_output=False)
```

---

### WARNING: Missing handle_unknown

**The Problem:**

```python
# BAD - Fails on unseen categories at transform time
enc = OneHotEncoder()
enc.fit(train_df[cols])
enc.transform(test_df[cols])  # ValueError if new category
```

**The Fix:**

```python
# GOOD - Handle unknown categories gracefully
enc = OneHotEncoder(handle_unknown="infrequent_if_exist")
```

---

### WARNING: Forgetting random_state

**The Problem:**

```python
# BAD - Non-reproducible results
lda = LatentDirichletAllocation(n_components=10)
```

**The Fix:**

```python
# GOOD - Reproducible
lda = LatentDirichletAllocation(n_components=10, random_state=42)
```

---

## Related Skills

- See the **numpy** skill for array operations used with sklearn
- See the **pandas** skill for DataFrame preprocessing
- See the **statsmodels** skill for primary regression (sklearn is fallback)