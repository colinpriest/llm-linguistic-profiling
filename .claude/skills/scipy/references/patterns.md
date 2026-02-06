# Scipy Patterns Reference

## Contents
- Test Selection Decision Tree
- Catastrophic Cancellation Handling
- Effect Size Interpretation
- P-Value Validation
- FDR Correction Families
- Distribution Functions

---

## Test Selection Decision Tree

| Data Type | Groups | Paired? | Test |
|-----------|--------|---------|------|
| Categorical × Categorical | Any | No | `chi2_contingency` |
| Categorical 2×2 (small n) | 2 | No | `fisher_exact` |
| Continuous | 2 | No | `ttest_ind(equal_var=False)` |
| Continuous | 2 | Yes | `wilcoxon` |
| Continuous | 3+ | No | `f_oneway` |
| Ordinal/Ranked | 3+ | Yes | `friedmanchisquare` |
| Binary | 2 | Yes | McNemar (via statsmodels) |
| Ordinal correlation | 2 vars | - | `spearmanr` |

---

## Catastrophic Cancellation Handling

**The Problem:** Low-variance ordinal data (tier ∈ {0,1,2}) triggers scipy RuntimeWarnings.

```python
# BAD - Warnings pollute logs and may mask real issues
f_stat, p = sp_stats.f_oneway(*groups)
```

**The Fix:**

```python
# GOOD - Suppress known benign warnings
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", 
        category=RuntimeWarning,
        message=".*Precision loss.*catastrophic cancellation.*"
    )
    f_stat, p = sp_stats.f_oneway(*groups)
```

**Applied to:** `f_oneway`, `ttest_ind`, `mannwhitneyu` in this codebase.

---

## Effect Size Interpretation

### Cohen's d (Continuous Comparisons)

```python
def interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"
```

### Eta-Squared (ANOVA)

```python
def interpret_eta_sq(eta_sq: float) -> str:
    if eta_sq < 0.01: return "negligible"
    if eta_sq < 0.06: return "small"
    if eta_sq < 0.14: return "medium"
    return "large"
```

### Cramér's V (Chi-Squared)

Thresholds depend on degrees of freedom:

```python
def interpret_cramers_v(v: float, df_min: int) -> str:
    thresholds = {
        1: (0.10, 0.30, 0.50),
        2: (0.07, 0.21, 0.35),
        3: (0.06, 0.17, 0.29),
    }.get(df_min, (0.05, 0.15, 0.25))
    
    if v < thresholds[0]: return "negligible"
    if v < thresholds[1]: return "small"
    if v < thresholds[2]: return "medium"
    return "large"
```

---

## P-Value Validation

### WARNING: Unchecked P-Values

**The Problem:**

```python
# BAD - Numerical instability can produce NaN or values outside [0,1]
_, p = sp_stats.ttest_ind(a, b, equal_var=False)
if p < 0.05:  # May raise or silently fail with NaN
    ...
```

**The Fix:**

```python
# GOOD - Validate before interpreting
_, p = sp_stats.ttest_ind(a, b, equal_var=False)
if not np.isfinite(p) or not (0 <= p <= 1):
    raise ValueError(f"Invalid p-value: {p!r}")
```

---

## FDR Correction Families

Apply BH correction **within** test families, not globally:

```python
from scipy.stats import false_discovery_control

def fdr_correction_with_tracking(
    p_values: list[float], 
    family_label: str,
    alpha: float = 0.05
) -> dict:
    arr = np.asarray(p_values, dtype=float)
    p_adj = false_discovery_control(arr, method="bh")
    n_surviving = int((p_adj < alpha).sum())
    return {
        "p_adjusted": p_adj,
        "n_tests": len(arr),
        "n_surviving": n_surviving,
        "family": family_label,
    }

# Usage: separate families
persona_vs_baseline = fdr_correction_with_tracking(p_persona, "persona_baseline")  # 36 tests
pairwise_persona = fdr_correction_with_tracking(p_pairwise, "pairwise")  # 630 tests
dimension_tests = fdr_correction_with_tracking(p_dim, "dimensions")  # 4 tests
```

---

## Distribution Functions

### Normal CDF for Wald P-Values

```python
z_score = coef / std_err
p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))
```

### Beta Distribution for Jeffreys Intervals

```python
def jeffreys_interval(successes: int, n: int, alpha: float = 0.05) -> tuple:
    a = successes + 0.5
    b = n - successes + 0.5
    lower = sp_stats.beta.ppf(alpha / 2, a, b)
    upper = sp_stats.beta.ppf(1 - alpha / 2, a, b)
    return lower, upper
```

### Wilson Score Interval

```python
def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple:
    p_hat = successes / n
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin
```

---

## Edge Case Patterns

### Zero Variance Check Before ANOVA

```python
# Check each group has variance
if any(np.var(g) == 0 for g in groups):
    return {"f_stat": np.nan, "p": np.nan, "eta_squared": np.nan}
```

### Fisher Exact Fallback for Small Tables

```python
def contingency_test(table: np.ndarray) -> dict:
    if table.shape == (2, 2) and table.sum() < 25:
        odds_ratio, p = sp_stats.fisher_exact(table)
        return {"test": "fisher", "odds_ratio": odds_ratio, "p": p}
    chi2, p, dof, _ = sp_stats.chi2_contingency(table)
    return {"test": "chi2", "chi2": chi2, "p": p, "dof": dof}
```