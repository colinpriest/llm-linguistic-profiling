# Statsmodels Patterns Reference

## Contents
- Formula vs Matrix API
- Convergence Fallback Chain
- Clustered Standard Errors
- Result Serialization
- Anti-Patterns

---

## Formula vs Matrix API

### Formula API (Preferred for Readability)

Use `statsmodels.formula.api` when columns are in a DataFrame:

```python
import statsmodels.formula.api as smf

# Logistic with categorical predictors
fit = smf.logit(
    "need_info ~ C(persona) + C(severity) + C(persona):C(severity)",
    data=df,
).fit(disp=0, maxiter=100)

# GEE with grouping
fit = smf.gee(
    "favourable ~ C(dim_level)",
    groups="complaint_id",
    data=df,
    family=Binomial(),
    cov_struct=Exchangeable(),
).fit(maxiter=100)
```

### Matrix API (When Control Needed)

Use `statsmodels.api` for explicit design matrices:

```python
import statsmodels.api as sm
from statsmodels.tools import add_constant

X = pd.get_dummies(df[predictor_cols], drop_first=True, dtype=float)
X = add_constant(X)  # CRITICAL: add intercept
y = df["outcome"]

fit = sm.Logit(y, X).fit(disp=0, maxiter=100)
```

---

## Convergence Fallback Chain

This project uses a 4-step fallback for binary models:

```python
def fit_logistic_with_fallback(X, y, model_label=""):
    # Step 1: Standard MLE
    try:
        fit = sm.Logit(y, X).fit(disp=0, maxiter=100)
        if fit.mle_retvals["converged"]:
            return fit, "MLE"
    except Exception:
        pass
    
    # Step 2: Firth penalized (for separation)
    try:
        return _firth_logit(X, y), "Firth"
    except Exception:
        pass
    
    # Step 3: Ridge regularization (sklearn)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500)
    lr.fit(X.values, y.values)
    return lr, "Ridge"
```

### GEE Fallback Pattern

```python
def fit_gee_with_fallback(formula, df, groups):
    # Try Exchangeable first (accounts for within-cluster correlation)
    try:
        fit = smf.gee(formula, groups=groups, data=df,
                      family=Binomial(), cov_struct=Exchangeable()).fit(maxiter=100)
        if fit.converged:
            return fit, "Exchangeable"
    except Exception:
        pass
    
    # Fallback to Independence (ignores correlation)
    fit = smf.gee(formula, groups=groups, data=df,
                  family=Binomial(), cov_struct=Independence()).fit(maxiter=100)
    return fit, "Independence"
```

---

## Clustered Standard Errors

### For OrderedModel (Cumulative Link)

```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

om = OrderedModel(endog, exog, distr="logit")
fit = om.fit(method="bfgs", disp=False, maxiter=300)

# Get clustered SEs
robust_fit = fit.get_robustcov_results(
    cov_type="cluster",
    groups=df["complaint_id"].values,
)
```

### For GEE (Built-in Clustering)

GEE handles clustering via `groups` parameter—no separate adjustment needed:

```python
gee = smf.gee("y ~ C(x)", groups="complaint_id", data=df, family=Binomial())
```

---

## Result Serialization

All results must be JSON-serializable. Convert statsmodels objects:

```python
def serialize_fit_results(fit):
    """Convert statsmodels fit to JSON-safe dict."""
    return {
        "coefficients": {k: float(v) for k, v in fit.params.items()},
        "p_values": {k: float(v) for k, v in fit.pvalues.items()},
        "odds_ratios": {k: float(v) for k, v in np.exp(fit.params).items()},
        "std_errors": {k: float(v) for k, v in fit.bse.items()},
        "aic": float(fit.aic) if hasattr(fit, "aic") else None,
        "converged": bool(fit.mle_retvals.get("converged", True)),
    }
```

---

## WARNING: Missing Intercept

**The Problem:**

```python
# BAD - No intercept, biased estimates
X = pd.get_dummies(df[cols], drop_first=True, dtype=float)
fit = sm.Logit(y, X).fit()  # Missing intercept!
```

**Why This Breaks:**
1. Model assumes mean of y is zero at baseline
2. All coefficient estimates are biased
3. Predictions are systematically wrong

**The Fix:**

```python
# GOOD - Always add constant for matrix API
from statsmodels.tools import add_constant
X = add_constant(pd.get_dummies(df[cols], drop_first=True, dtype=float))
fit = sm.Logit(y, X).fit()

# Formula API adds intercept automatically
fit = smf.logit("y ~ x1 + x2", data=df).fit()  # OK
```

---

## WARNING: Silent Convergence Failure

**The Problem:**

```python
# BAD - Ignoring convergence status
fit = sm.Logit(y, X).fit(disp=0)
return fit.params  # May be garbage if didn't converge
```

**Why This Breaks:**
1. Non-converged estimates are unreliable
2. Standard errors and p-values are meaningless
3. Downstream analysis based on bad coefficients

**The Fix:**

```python
# GOOD - Always check convergence
fit = sm.Logit(y, X).fit(disp=0, maxiter=100)
if not fit.mle_retvals["converged"]:
    raise RuntimeError(f"Model did not converge after {100} iterations")
```

---

## WARNING: Perfect Separation in Logistic Regression

**The Problem:**

```python
# When a predictor perfectly predicts outcome
# Coefficients explode to ±infinity
fit = sm.Logit(y, X).fit()  # Warning: Perfect separation detected
```

**Why This Breaks:**
1. MLE doesn't exist—coefficients diverge
2. Standard errors are undefined
3. Model is uninterpretable

**The Fix:**

```python
# GOOD - Use Firth penalization or ridge
try:
    fit = sm.Logit(y, X).fit(disp=0, maxiter=100)
except Exception as e:
    if "separation" in str(e).lower():
        # Fallback to regularized logistic
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty="l2", C=1.0)
        lr.fit(X.values, y.values)
```

---

## ANOVA Type Selection

| Type | Use When | Example |
|------|----------|---------|
| Type I | Sequential SS, order matters | Exploratory, hierarchical models |
| Type II | Main effects, no interaction | `anova_lm(fit, typ=2)` (default) |
| Type III | Interaction present | `anova_lm(fit, typ=3)` |

This project uses **Type II** for main effect tests:

```python
from statsmodels.stats.anova import anova_lm

fit = smf.ols("outcome ~ C(persona) + C(severity)", data=df).fit()
table = anova_lm(fit, typ=2)  # Order-invariant
```