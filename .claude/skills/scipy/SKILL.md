---
name: scipy
description: |
  Executes statistical tests and numerical analysis computations.
  Use when: running chi-squared tests, ANOVA, t-tests, Mann-Whitney U, Wilcoxon, Friedman tests,
  computing FDR-corrected p-values, calculating effect sizes, or performing correlation analysis.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Scipy Skill

Statistical testing backbone for the fairness experiment. All scipy calls are wrapped in utility functions in `src/statistics.py` with edge-case handling, effect size computation, and FDR correction. Primary module: `scipy.stats` (imported as `sp_stats`).

## Quick Start

### Chi-Squared Test with Effect Size

```python
from scipy import stats as sp_stats
import numpy as np

def chi_squared_test(contingency: pd.DataFrame) -> dict:
    chi2, p, dof, expected = sp_stats.chi2_contingency(contingency.values)
    n = contingency.values.sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
    return {"chi2": chi2, "p": p, "dof": dof, "cramers_v": cramers_v}
```

### FDR Correction (Benjamini-Hochberg)

```python
from scipy.stats import false_discovery_control

def fdr_correction(p_values: list[float]) -> np.ndarray:
    arr = np.asarray(p_values, dtype=float)
    return false_discovery_control(arr, method="bh")
```

### One-Way ANOVA with Eta-Squared

```python
import warnings
from scipy import stats as sp_stats

def one_way_anova(groups: list[np.ndarray]) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message=".*Precision loss.*catastrophic cancellation.*")
        f_stat, p_val = sp_stats.f_oneway(*groups)
    # Compute eta-squared
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_within = sum(((g - g.mean())**2).sum() for g in groups)
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else np.nan
    return {"f_stat": f_stat, "p": p_val, "eta_squared": eta_sq}
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Catastrophic cancellation | Suppress warnings for low-variance ordinal data | `warnings.filterwarnings("ignore", message=".*catastrophic.*")` |
| FDR correction | Control false discovery rate across test families | `false_discovery_control(p_values, method="bh")` |
| Effect sizes | Always pair tests with interpretable metrics | Cramér's V, Cohen's d, eta-squared |
| P-value validation | Guard against NaN/inf from numerical instability | `np.isfinite(p) and 0 <= p <= 1` |

## Common Patterns

### Spearman Correlation for Ordinal Data

```python
rho, p_value = sp_stats.spearmanr(ordinal_x, ordinal_y)
```

### Mann-Whitney U with Effect Size

```python
u_stat, p = sp_stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
n1, n2 = len(group_a), len(group_b)
rank_biserial = 1 - (2 * u_stat) / (n1 * n2)  # Effect size
```

### Wilcoxon Post-Hoc After Friedman

```python
stat, p = sp_stats.friedmanchisquare(*groups)
if p < 0.05:
    # Pairwise Wilcoxon with FDR correction
    pairwise_p = []
    for i, j in combinations(range(len(groups)), 2):
        _, p_ij = sp_stats.wilcoxon(groups[i], groups[j], zero_method="wilcox")
        pairwise_p.append(p_ij)
    adjusted = false_discovery_control(pairwise_p, method="bh")
```

## See Also

- [patterns](references/patterns.md) - Test selection, edge cases, effect size interpretation
- [workflows](references/workflows.md) - End-to-end analysis pipelines

## Related Skills

- See the **statsmodels** skill for mixed models, GEE, and ordinal regression
- See the **numpy** skill for array operations and random sampling
- See the **pandas** skill for DataFrame-to-array conversion patterns