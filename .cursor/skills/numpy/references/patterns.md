# Numpy Patterns Reference

## Contents
- Array Creation and Manipulation
- Numerical Validation
- Statistical Calculations
- Linear Algebra for Regression
- Random Number Generation
- Anti-Patterns

---

## Array Creation and Manipulation

### Pandas to Numpy Extraction

```python
# GOOD - Extract values for vectorized operations
favorable = df["favorable"].values
weights = df["weight"].values
result = np.average(favorable, weights=weights)

# GOOD - Boolean array construction from contingency
a = merged["favorable_1"].values
b = merged["favorable_2"].values
table = np.array([
    [(a & b).sum(), (a & ~b).sum()],
    [(~a & b).sum(), (~a & ~b).sum()]
])
```

### Feature Matrix Stacking

```python
from sklearn.preprocessing import OneHotEncoder

# Stack multiple feature types horizontally
parts = []
if cat_features is not None:
    parts.append(encoder.fit_transform(df[cat_cols]))
if num_features is not None:
    parts.append(df[num_cols].values)
X = np.hstack(parts) if parts else np.zeros((len(df), 0))
```

---

## Numerical Validation

### WARNING: Missing Finite Checks

**The Problem:**

```python
# BAD - No validation before display/storage
cohens_d = (mean1 - mean2) / pooled_std
return {"effect_size": cohens_d}  # May be inf or nan
```

**Why This Breaks:**
1. Division by zero produces `inf`, JSON serialization fails
2. Empty groups produce `nan`, downstream calculations corrupt
3. HTML reports show "nan" instead of meaningful values

**The Fix:**

```python
# GOOD - Validate before returning
cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
if np.isfinite(cohens_d):
    return {"effect_size": float(cohens_d)}
return {"effect_size": None}  # Explicit missing value
```

### Comprehensive Validation Pattern

```python
def safe_metric(value: float, default: float = 0.0) -> float:
    """Return value if finite, else default."""
    return float(value) if np.isfinite(value) else default

# Usage throughout codebase
air = safe_metric(rate_protected / rate_baseline, default=1.0)
```

---

## Statistical Calculations

### Cohen's d (Effect Size)

```python
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute standardized mean difference."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1 = np.var(group1, ddof=1)  # Bessel's correction
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

### Cohen's Kappa (Inter-rater Agreement)

```python
def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute agreement beyond chance."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    
    # Build confusion matrix
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, lt in enumerate(labels):
        for j, lp in enumerate(labels):
            cm[i, j] = ((y_true == lt) & (y_pred == lp)).sum()
    
    po = np.trace(cm) / n  # Observed agreement
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n * n)  # Expected
    
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
```

---

## Linear Algebra for Regression

### Firth Logistic Regression

```python
def firth_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Bias-reduced logistic regression via penalized likelihood."""
    n, p = X.shape
    beta = np.zeros(p)
    
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.diag(mu * (1 - mu))
        
        # Fisher information matrix
        try:
            fisher = X.T @ W @ X
            fisher_inv = np.linalg.inv(fisher)
        except np.linalg.LinAlgError:
            return None  # Singular matrix
        
        # Firth adjustment (hat matrix diagonal)
        H_diag = np.diag(X @ fisher_inv @ X.T @ W)
        y_adj = y + H_diag * (0.5 - mu)
        
        # Newton-Raphson update
        score = X.T @ (y_adj - mu)
        beta_new = beta + fisher_inv @ score
        
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            break
        beta = beta_new
    
    return beta
```

---

## Random Number Generation

### Seeded RNG for Reproducibility

```python
# GOOD - Create seeded RNG instance
rng = np.random.RandomState(seed=42)
sample_idx = rng.choice(len(df), size=n_sample, replace=False)

# GOOD - Different seeds for different purposes
cache_seed = seed + 1000  # Separate cache invalidation
bootstrap_rng = np.random.RandomState(seed + 100)
```

### WARNING: Global Random State

**The Problem:**

```python
# BAD - Modifies global state, non-reproducible
np.random.seed(42)
idx = np.random.randint(0, n, size=n)  # Uses global RNG
```

**Why This Breaks:**
1. Other code may change global seed between calls
2. Parallel execution produces non-deterministic results
3. Tests become flaky and hard to debug

**The Fix:**

```python
# GOOD - Isolated RNG instance
rng = np.random.RandomState(42)
idx = rng.randint(0, n, size=n)
```

---

## Anti-Patterns

### WARNING: Unclipped Propensity Scores

**The Problem:**

```python
# BAD - Extreme weights cause variance explosion
weights = 1.0 / propensity_scores
weighted_mean = np.average(outcomes, weights=weights)
```

**Why This Breaks:**
1. Propensity near 0 produces weights approaching infinity
2. Single extreme weight dominates the entire average
3. Confidence intervals become meaningless

**The Fix:**

```python
# GOOD - Clip propensity scores and trim weights
propensity_clipped = np.clip(propensity_scores, 0.01, 0.99)
weights = 1.0 / propensity_clipped
weights_trimmed = np.minimum(weights, np.percentile(weights, 99))
weighted_mean = np.average(outcomes, weights=weights_trimmed)
```

### WARNING: Missing ddof in Variance

**The Problem:**

```python
# BAD - Population variance, biased for samples
sample_var = np.var(data)  # ddof=0 by default
```

**The Fix:**

```python
# GOOD - Sample variance with Bessel's correction
sample_var = np.var(data, ddof=1)
sample_std = np.std(data, ddof=1)
```