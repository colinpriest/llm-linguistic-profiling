---
name: statsmodels
description: |
  Runs regression analysis, ANOVA, and repeated measures statistical models.
  Use when: fitting logistic/OLS regression, running ANOVA, GEE models, 
  mixed effects models, McNemar tests, power analysis, or handling 
  convergence failures in statistical models.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Statsmodels Skill

Statistical modeling in this project uses formula-based and matrix-based APIs with robust fallback chains for convergence failures. Models cluster on `complaint_id` for within-subject repeated measures.

## Quick Start

### Logistic Regression with Fallback

```python
import statsmodels.api as sm
from statsmodels.tools import add_constant

X = add_constant(pd.get_dummies(df[predictor_cols], drop_first=True, dtype=float))
y = df["outcome"]

try:
    fit = sm.Logit(y, X).fit(disp=0, maxiter=100)
    if not fit.mle_retvals["converged"]:
        raise RuntimeError("Did not converge")
except Exception:
    # Fallback to sklearn ridge
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500)
    lr.fit(X.values, y.values)
```

### Formula-Based ANOVA

```python
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

fit = smf.ols("outcome ~ C(persona) + C(severity) + C(persona):C(severity)", data=df).fit()
table = anova_lm(fit, typ=2)  # Type II SS
f_val = float(table.loc["C(persona)", "F"])
p_val = float(table.loc["C(persona)", "PR(>F)"])
```

### GEE for Clustered Binary Outcomes

```python
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable, Independence

gee = smf.gee(
    "favourable ~ C(persona)",
    groups="complaint_id",
    data=df,
    family=Binomial(),
    cov_struct=Exchangeable(),
)
fit = gee.fit(maxiter=100)
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Formula API | R-style model specification | `smf.ols("y ~ C(x)")` |
| Matrix API | Explicit design matrices | `sm.Logit(y, X).fit()` |
| `add_constant` | Add intercept column | `X = add_constant(X)` |
| `C()` | Categorical factor in formula | `C(persona)` |
| `typ=2` | Type II ANOVA (order-invariant) | `anova_lm(fit, typ=2)` |
| `disp=0` | Suppress optimizer output | `.fit(disp=0)` |

## Result Extraction

```python
fit.params            # Coefficients (Series)
fit.pvalues           # P-values (Series)
fit.bse               # Standard errors (Series)
fit.conf_int()        # CI DataFrame [lower, upper]
fit.prsquared         # Pseudo R² (logit only)
fit.aic               # AIC
fit.mle_retvals["converged"]  # Convergence status
np.exp(fit.params)    # Odds ratios for logit
```

## See Also

- [patterns](references/patterns.md) - Model specification, fallback chains, error handling
- [workflows](references/workflows.md) - End-to-end analysis workflows

## Related Skills

- See the **pandas** skill for DataFrame preparation and groupby operations
- See the **scipy** skill for chi-squared tests and FDR correction
- See the **scikit-learn** skill for regularized fallback models