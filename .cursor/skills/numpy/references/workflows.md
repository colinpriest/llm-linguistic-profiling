# Numpy Workflows Reference

## Contents
- Bootstrap Confidence Interval Workflow
- Propensity Score Weighting Workflow
- Effect Size Analysis Workflow
- Matrix Feature Engineering Workflow

---

## Bootstrap Confidence Interval Workflow

### When to Use
Computing confidence intervals for custom statistics where parametric assumptions may not hold.

### Implementation

```python
import numpy as np

def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for any statistic."""
    rng = np.random.RandomState(seed)
    n = len(data)
    
    # Bootstrap resampling
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_sample = data[idx]
        boot_stats.append(statistic_fn(boot_sample))
    
    boot_stats = np.array(boot_stats)
    
    # Percentile method
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    point_estimate = statistic_fn(data)
    
    return point_estimate, ci_low, ci_high

# Usage
estimate, low, high = bootstrap_ci(
    data=outcomes,
    statistic_fn=np.mean,
    n_boot=2000,
    seed=42
)
```

### Workflow Checklist

Copy this checklist and track progress:
- [ ] Step 1: Extract numpy array from DataFrame
- [ ] Step 2: Define statistic function (mean, median, custom)
- [ ] Step 3: Create seeded RandomState
- [ ] Step 4: Run bootstrap loop with replacement
- [ ] Step 5: Compute percentile CI
- [ ] Step 6: Validate results with `np.isfinite()`

---

## Propensity Score Weighting Workflow

### When to Use
Causal inference when treatment assignment is non-random. See the **pandas** skill for data preparation.

### Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def compute_ipw_weights(
    df: pd.DataFrame,
    treatment_col: str,
    covariate_cols: list[str],
    trim_percentile: float = 99.0
) -> np.ndarray:
    """Compute inverse probability weights with stabilization."""
    # Fit propensity model
    X = df[covariate_cols].values
    y = df[treatment_col].values
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    propensity = model.predict_proba(X)[:, 1]
    
    # Clip extreme propensities
    propensity_clipped = np.clip(propensity, 0.01, 0.99)
    
    # Compute weights
    weights = np.where(
        y == 1,
        1.0 / propensity_clipped,
        1.0 / (1 - propensity_clipped)
    )
    
    # Trim extreme weights
    trim_val = np.percentile(weights, trim_percentile)
    weights_trimmed = np.minimum(weights, trim_val)
    
    # Stabilize (optional)
    p_treat = y.mean()
    stabilized = np.where(
        y == 1,
        p_treat / propensity_clipped,
        (1 - p_treat) / (1 - propensity_clipped)
    )
    stabilized = np.minimum(stabilized, np.percentile(stabilized, trim_percentile))
    
    return weights_trimmed, stabilized

# Weighted outcome estimation
def ipw_estimate(outcomes: np.ndarray, weights: np.ndarray) -> float:
    """Compute IPW-weighted average outcome."""
    if not np.all(np.isfinite(weights)):
        return float("nan")
    return np.average(outcomes, weights=weights)
```

### Validation Loop

1. Compute propensity scores
2. Check balance: `np.mean(weights[treated])` vs `np.mean(weights[control])`
3. If extreme imbalance (ratio > 10), increase trimming
4. Repeat until weights are stable

---

## Effect Size Analysis Workflow

### When to Use
Quantifying magnitude of differences between groups beyond p-values.

### Implementation

```python
import numpy as np
from typing import NamedTuple

class EffectSizeResult(NamedTuple):
    cohens_d: float
    cohens_d_ci: tuple[float, float]
    interpretation: str

def effect_size_analysis(
    group1: np.ndarray,
    group2: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> EffectSizeResult:
    """Compute Cohen's d with bootstrap CI and interpretation."""
    
    def compute_d(g1: np.ndarray, g2: np.ndarray) -> float:
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2:
            return 0.0
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) 
            / (n1 + n2 - 2)
        )
        if pooled_std == 0:
            return 0.0
        return (np.mean(g1) - np.mean(g2)) / pooled_std
    
    # Point estimate
    d = compute_d(group1, group2)
    
    # Bootstrap CI
    rng = np.random.RandomState(seed)
    boot_d = []
    for _ in range(n_boot):
        idx1 = rng.randint(0, len(group1), size=len(group1))
        idx2 = rng.randint(0, len(group2), size=len(group2))
        boot_d.append(compute_d(group1[idx1], group2[idx2]))
    
    ci_low, ci_high = np.percentile(boot_d, [2.5, 97.5])
    
    # Interpretation (Cohen's conventions)
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"
    
    return EffectSizeResult(d, (ci_low, ci_high), interp)
```

---

## Matrix Feature Engineering Workflow

### When to Use
Preparing feature matrices for regression models. See the **scikit-learn** skill for model fitting.

### Implementation

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_feature_matrix(
    df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    fit_encoders: bool = True,
    encoder: OneHotEncoder = None,
    scaler: StandardScaler = None
) -> tuple[np.ndarray, OneHotEncoder, StandardScaler]:
    """Build design matrix from mixed features."""
    parts = []
    
    # Categorical features
    if cat_cols:
        if fit_encoders:
            encoder = OneHotEncoder(sparse_output=False, drop="first")
            cat_encoded = encoder.fit_transform(df[cat_cols])
        else:
            cat_encoded = encoder.transform(df[cat_cols])
        parts.append(cat_encoded)
    
    # Numeric features
    if num_cols:
        if fit_encoders:
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(df[num_cols])
        else:
            num_scaled = scaler.transform(df[num_cols])
        parts.append(num_scaled)
    
    # Stack horizontally
    if parts:
        X = np.hstack(parts)
    else:
        X = np.zeros((len(df), 0))
    
    return X, encoder, scaler

# Usage
X_train, encoder, scaler = build_feature_matrix(
    df_train, cat_cols=["age", "geo"], num_cols=["word_count"]
)
X_test, _, _ = build_feature_matrix(
    df_test, cat_cols=["age", "geo"], num_cols=["word_count"],
    fit_encoders=False, encoder=encoder, scaler=scaler
)
```

### Validation Checklist

Copy this checklist and track progress:
- [ ] Step 1: Identify categorical vs numeric columns
- [ ] Step 2: Fit encoders on training data only
- [ ] Step 3: Transform both train and test with same encoders
- [ ] Step 4: Verify shape: `X.shape[0] == len(df)`
- [ ] Step 5: Check for NaN: `assert not np.any(np.isnan(X))`
- [ ] Step 6: Verify feature count matches expectation