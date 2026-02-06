---
name: numpy
description: |
  Performs numerical computing and statistical array operations.
  Use when: computing effect sizes (Cohen's d/kappa), matrix operations for regression,
  bootstrap confidence intervals, weighted aggregations, propensity score calculations,
  or validating numerical results with isfinite checks.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Numpy Skill

Numpy powers the statistical computing layer in this fairness experiment. Primary uses: effect size calculations (Cohen's d, kappa), matrix operations for Firth logistic regression, bootstrap resampling, propensity score weighting, and defensive validation with `np.isfinite()`. All random operations use seeded `RandomState` for reproducibility.

## Quick Start

### Statistical Effect Sizes

```python
import numpy as np

# Cohen's d for two groups
mean1, mean2 = group1.mean(), group2.mean()
n1, n2 = len(group1), len(group2)
pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                      (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
```

### Weighted Aggregation (IPW)

```python
# Inverse probability weighted average
weights = np.clip(1.0 / propensity_scores, 0.01, 100)  # Clip to prevent extremes
weights_trimmed = np.minimum(weights, np.percentile(weights, 99))
weighted_outcome = np.average(outcomes, weights=weights_trimmed)
```

### Bootstrap Confidence Intervals

```python
rng = np.random.RandomState(seed)
boot_means = []
for _ in range(1000):
    idx = rng.randint(0, len(data), size=len(data))
    boot_means.append(np.mean(data[idx]))
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Seeded RNG | Reproducibility | `rng = np.random.RandomState(42)` |
| Value validation | Defensive checks | `if np.isfinite(result):` |
| Weight clipping | Numerical stability | `np.clip(weights, 0.01, 0.99)` |
| Pooled variance | Effect sizes | `np.var(x, ddof=1)` with Bessel correction |
| Matrix ops | Firth logit | `X.T @ W @ X` for Fisher information |

## Common Patterns

### Pandas-Numpy Integration

```python
# Extract values, compute, return to pandas-compatible format
values = df["favorable"].values  # numpy array
result = np.mean(values[mask])
return float(result) if np.isfinite(result) else None  # JSON-safe
```

### Feature Matrix Construction

```python
from sklearn.preprocessing import OneHotEncoder

# Stack categorical and numeric features
cat_encoded = encoder.fit_transform(df[cat_cols])
num_array = df[num_cols].values
X = np.hstack([cat_encoded, num_array])
```

## See Also

- [patterns](references/patterns.md) - Array operations, validation, linear algebra
- [workflows](references/workflows.md) - Effect sizes, bootstrap, IPW analysis

## Related Skills

- See the **pandas** skill for DataFrame operations before numpy extraction
- See the **scipy** skill for statistical tests that consume numpy arrays
- See the **scikit-learn** skill for feature engineering pipelines