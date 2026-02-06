# Statsmodels Workflows Reference

## Contents
- Disparate Impact Analysis Workflow
- Repeated Measures Analysis Workflow
- Severity Interaction Analysis Workflow
- Power Analysis Workflow
- Validation Checklist

---

## Disparate Impact Analysis Workflow

Tests whether persona predicts remedy tier (correlation phase).

### Step-by-Step

```python
import statsmodels.api as sm
import pandas as pd
import numpy as np

def run_disparate_impact_analysis(df: pd.DataFrame) -> dict:
    """
    Workflow:
    1. Build contingency table
    2. Run chi-squared test
    3. Fit logistic regression
    4. Calculate adverse impact ratios
    """
    
    # 1. Contingency table
    contingency = pd.crosstab(df["persona"], df["tier"])
    
    # 2. Chi-squared (use scipy - see scipy skill)
    from scipy.stats import chi2_contingency
    chi2, p, dof, expected = chi2_contingency(contingency.values)
    
    # 3. Logistic regression
    X = pd.get_dummies(df[["age", "geo", "edu", "ses"]], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    y = (df["tier"] >= 2).astype(int)  # Binary: favorable vs not
    
    fit = sm.Logit(y, X).fit(disp=0, maxiter=100)
    
    # 4. Adverse impact ratios
    baseline_rate = y.mean()
    air = {}
    for persona in df["persona"].unique():
        group_rate = y[df["persona"] == persona].mean()
        air[persona] = group_rate / baseline_rate if baseline_rate > 0 else np.nan
    
    return {
        "chi2": chi2,
        "chi2_p": p,
        "coefficients": fit.params.to_dict(),
        "p_values": fit.pvalues.to_dict(),
        "air": air,
    }
```

### Validation Loop

```
1. Prepare data (drop missing, encode categoricals)
2. Run analysis
3. Check: fit.mle_retvals["converged"] == True
4. If not converged, apply fallback chain
5. Only report results when model is valid
```

---

## Repeated Measures Analysis Workflow

Tests causal effect using within-complaint variation (matched-guise design).

### GEE for Binary Outcome

```python
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable, Independence

def run_repeated_measures_gee(df: pd.DataFrame) -> dict:
    """
    Each complaint has 36 rewrites (one per persona).
    Tests whether persona affects favorable outcome within same complaint.
    """
    
    # Sort by cluster (required for GEE)
    df = df.sort_values("complaint_id").reset_index(drop=True)
    
    # Fit GEE with exchangeable correlation
    formula = "favourable ~ C(age) + C(geo) + C(edu) + C(ses)"
    
    try:
        gee = smf.gee(
            formula,
            groups="complaint_id",
            data=df,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        fit = gee.fit(maxiter=100)
        method = "GEE-Exchangeable"
    except Exception:
        # Fallback to independence
        gee = smf.gee(
            formula,
            groups="complaint_id",
            data=df,
            family=Binomial(),
            cov_struct=Independence(),
        )
        fit = gee.fit(maxiter=100)
        method = "GEE-Independence"
    
    return {
        "method": method,
        "coefficients": fit.params.to_dict(),
        "p_values": fit.pvalues.to_dict(),
        "odds_ratios": np.exp(fit.params).to_dict(),
        "converged": fit.converged,
    }
```

### Conditional Logit (Pure Within-Complaint)

```python
from statsmodels.discrete.conditional_models import ConditionalLogit

def run_conditional_logit(df: pd.DataFrame) -> dict:
    """
    Conditions on complaint, removes all between-complaint variation.
    Only complaints with outcome variation contribute.
    """
    
    # Filter to informative complaints (outcome varies within complaint)
    var_check = df.groupby("complaint_id")["favourable"].nunique()
    informative = var_check[var_check > 1].index
    df = df[df["complaint_id"].isin(informative)]
    
    # Build design matrix
    X = pd.get_dummies(df[["age", "geo", "edu", "ses"]], drop_first=True, dtype=float)
    y = df["favourable"].astype(float)
    groups = df["complaint_id"]
    
    clogit = ConditionalLogit(y, X, groups=groups)
    fit = clogit.fit(disp=False, maxiter=200)
    
    return {
        "method": "ConditionalLogit",
        "coefficients": fit.params.to_dict(),
        "p_values": fit.pvalues.to_dict(),
        "n_informative_complaints": len(informative),
    }
```

---

## Severity Interaction Analysis Workflow

Tests whether bias amplifies with complaint severity.

### Two-Way ANOVA

```python
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def run_severity_interaction_anova(df: pd.DataFrame) -> dict:
    """
    Tests persona × severity interaction effect on remedy tier.
    """
    
    # Fit OLS with interaction
    formula = "tier ~ C(persona) + C(severity) + C(persona):C(severity)"
    fit = smf.ols(formula, data=df).fit()
    
    # Type II ANOVA table
    table = anova_lm(fit, typ=2)
    
    # Extract interaction F-test
    interaction_row = "C(persona):C(severity)"
    f_stat = float(table.loc[interaction_row, "F"])
    p_val = float(table.loc[interaction_row, "PR(>F)"])
    
    # Handle invalid F values
    if not np.isfinite(f_stat) or f_stat < 0:
        f_stat, p_val = 0.0, 1.0
    
    return {
        "interaction_F": f_stat,
        "interaction_p": p_val,
        "main_persona_F": float(table.loc["C(persona)", "F"]),
        "main_severity_F": float(table.loc["C(severity)", "F"]),
        "r_squared": fit.rsquared,
    }
```

### GEE for Binary Severity Interaction

```python
def run_severity_interaction_gee(df: pd.DataFrame) -> dict:
    """
    Tests whether persona effect on Tier-2 rate changes with severity.
    """
    
    df = df.sort_values("complaint_id").reset_index(drop=True)
    formula = "tier2 ~ C(persona) + C(severity) + C(persona):C(severity)"
    
    gee = smf.gee(
        formula,
        groups="complaint_id",
        data=df,
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    fit = gee.fit(maxiter=100)
    
    # Extract interaction terms
    interaction_coefs = {
        k: float(v) for k, v in fit.params.items()
        if ":" in k
    }
    interaction_pvals = {
        k: float(v) for k, v in fit.pvalues.items()
        if ":" in k
    }
    
    return {
        "interaction_coefficients": interaction_coefs,
        "interaction_p_values": interaction_pvals,
        "converged": fit.converged,
    }
```

---

## Power Analysis Workflow

Calculate minimum detectable effect sizes.

```python
from statsmodels.stats.power import TTestPower, FTestAnovaPower

def run_power_analysis(n_complaints: int, n_personas: int, alpha: float = 0.05) -> dict:
    """
    Calculate detectable effect sizes for planned sample.
    """
    
    # Paired t-test power (McNemar-style)
    tt = TTestPower()
    min_d = tt.solve_power(
        nobs=n_complaints,
        alpha=alpha,
        power=0.80,
        alternative="two-sided",
    )
    
    # ANOVA power (persona main effect)
    ft = FTestAnovaPower()
    n_per_cell = n_complaints * n_personas // n_personas
    min_f = ft.solve_power(
        nobs=n_per_cell,
        k_groups=n_personas,
        alpha=alpha,
        power=0.80,
    )
    
    return {
        "min_detectable_d": float(min_d),
        "min_detectable_f": float(min_f),
        "n_complaints": n_complaints,
        "power": 0.80,
        "alpha": alpha,
    }
```

---

## Validation Checklist

Copy this checklist when running statsmodels analyses:

```
- [ ] Design matrix has intercept (add_constant or formula)
- [ ] Categorical variables use C() in formula or get_dummies
- [ ] drop_first=True to avoid dummy trap
- [ ] Check fit.mle_retvals["converged"] for MLE models
- [ ] Check fit.converged for GEE models
- [ ] Sort data by group before GEE/MixedLM
- [ ] Filter to informative groups for ConditionalLogit
- [ ] Convert results to JSON-safe types (no numpy/pandas)
- [ ] Apply fallback chain on convergence failure
- [ ] Suppress warnings only after understanding them
```

---

## Error Handling Pattern

```python
import warnings

def fit_with_diagnostics(model_func, *args, **kwargs):
    """Standard wrapper for statsmodels fitting."""
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            fit = model_func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "singular" in msg or "rank" in msg:
                return None, {"error": "rank_deficient", "detail": str(e)}
            if "separation" in msg:
                return None, {"error": "separation", "detail": str(e)}
            if "converge" in msg:
                return None, {"error": "convergence", "detail": str(e)}
            raise
        
        # Log relevant warnings
        warns = [str(x.message) for x in w if "statsmodels" in str(x.category)]
        
        return fit, {"warnings": warns}
```

---

## Integration with Other Skills

- **pandas skill**: DataFrame preparation, `groupby`, `crosstab`
- **scipy skill**: `chi2_contingency`, `false_discovery_control` for FDR
- **scikit-learn skill**: `LogisticRegression` fallback for separation/convergence
- **numpy skill**: `np.exp()` for odds ratios, array operations