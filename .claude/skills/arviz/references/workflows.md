# ArviZ Workflows Reference

## Contents
- Model Validation Workflow
- Diagnostic Interpretation
- Posterior Analysis Workflow
- Integration with Project Architecture

---

## Model Validation Workflow

### Checklist: Validate MCMC Fit Before Analysis

Copy this checklist and track progress:
- [ ] Step 1: Convert fit to InferenceData
- [ ] Step 2: Check Rhat (all params < 1.01)
- [ ] Step 3: Check ESS (all params > 400)
- [ ] Step 4: Check divergences (must be 0)
- [ ] Step 5: Visual trace inspection
- [ ] Step 6: Proceed with posterior analysis

### Validation Code

```python
import arviz as az
import numpy as np

def validate_fit(fit, var_names=None):
    """Validate MCMC fit before analysis."""
    idata = az.from_cmdstanpy(fit)
    
    # Step 2: Rhat
    rhat = az.rhat(idata, var_names=var_names)
    max_rhat = float(rhat.to_array().max())
    if max_rhat > 1.01:
        raise RuntimeError(f"Convergence failure: max Rhat={max_rhat:.4f} > 1.01")
    
    # Step 3: ESS
    ess = az.ess(idata, var_names=var_names, method="bulk")
    min_ess = float(ess.to_array().min())
    if min_ess < 400:
        raise RuntimeError(f"Insufficient samples: min ESS={min_ess:.0f} < 400")
    
    # Step 4: Divergences
    n_div = 0
    if "diverging" in idata.sample_stats:
        n_div = int(idata.sample_stats.diverging.sum())
    if n_div > 0:
        raise RuntimeError(f"{n_div} divergent transitions detected")
    
    return idata  # Safe to proceed

# Usage
idata = validate_fit(fit, var_names=["alpha", "beta"])
```

---

## Diagnostic Interpretation

### Rhat Thresholds

| Rhat Value | Interpretation | Action |
|------------|----------------|--------|
| < 1.01 | Converged | Proceed |
| 1.01 - 1.05 | Marginal | Increase warmup/samples |
| > 1.05 | Not converged | Reparameterize or simplify model |

### ESS Thresholds

| ESS Value | Interpretation | Action |
|-----------|----------------|--------|
| > 400 | Sufficient | Proceed |
| 100 - 400 | Marginal | May need more samples for tail quantiles |
| < 100 | Insufficient | Increase samples or fix mixing issues |

### Divergence Interpretation

**Any divergences indicate problems:**
- Often caused by sharp curvature in posterior
- Fix: Increase `adapt_delta` (0.95 → 0.99)
- Fix: Use non-centered parameterization

Project example (`src/bayes_robustness.py:57-61`):
```python
_ADAPT_DELTA = 0.99   # High adapt_delta reduces divergences
_MAX_TREEDEPTH = 13   # Allow deeper trees for complex posteriors
```

---

## Posterior Analysis Workflow

### Iterate-Until-Pass Pattern

1. Fit model
2. Validate: Check diagnostics
3. If validation fails:
   - Adjust priors, reparameterize, or use fallback
   - Return to step 1
4. Only proceed when validation passes

### Project's 5-Step Fallback Chain

The project implements this pattern for `monetary_severity` (`src/bayes_robustness.py:1014-1186`):

```python
# Step 1: Standard hierarchical fit
result = _try_b1_b2(...)
if result is not None:
    return result

# Step 2: Category-level random intercept
result = _try_b1_b2(..., method_label="category-RE")
if result is not None:
    return result

# Step 3: Per-dimension models (simpler)
result = _try_per_dimension(...)
if result is not None:
    return result

# Step 4: Stronger priors
prior_sd = np.array([0.5 if ":" in c else 1.0 for c in col_names])
result = _try_b1_b2(..., prior_sd=prior_sd)
if result is not None:
    return result

# Step 5: Aggressive sampler settings
result = _try_b1_b2(..., adapt_delta=0.99, max_treedepth=15)
if result is None:
    raise RuntimeError("All fallbacks exhausted")
```

---

## Integration with Project Architecture

### Current Architecture (No ArviZ)

The project extracts diagnostics directly from CmdStanPy:

```
CmdStanModel.sample() → CmdStanMCMC
    ↓
fit.summary()         → DataFrame (Rhat, ESS, mean, sd)
fit.method_variables() → Dict (divergent__, treedepth__)
fit.stan_variable()    → ndarray (posterior draws)
    ↓
_check_diagnostics()  → Dict (max_rhat, min_ess, divergences)
_gate_fit()           → "OK" | "WARN" | "FAIL"
```

### Potential ArviZ Integration Points

```
CmdStanModel.sample() → CmdStanMCMC
    ↓
az.from_cmdstanpy()   → InferenceData
    ↓
az.summary()          → DataFrame (enhanced statistics)
az.plot_trace()       → Figure (visual diagnostics)
az.loo() / az.waic()  → Model comparison metrics
```

### Adding ArviZ Visualization

To add diagnostic plots to the project:

```python
# In src/bayes_robustness.py
import arviz as az
import matplotlib.pyplot as plt

def _save_diagnostic_plots(fit, fit_id, output_dir):
    """Generate and save ArviZ diagnostic plots."""
    idata = az.from_cmdstanpy(fit)
    
    # Trace plot
    fig, axes = plt.subplots(figsize=(12, 8))
    az.plot_trace(idata, var_names=["alpha", "beta"])
    plt.savefig(output_dir / f"trace_{fit_id}.png", dpi=150)
    plt.close()
    
    # Posterior distributions
    az.plot_posterior(idata, var_names=["beta"], hdi_prob=0.95)
    plt.savefig(output_dir / f"posterior_{fit_id}.png", dpi=150)
    plt.close()
```

### Artifact Storage Pattern

Project stores posterior draws as Parquet (`src/bayes_robustness.py:568-608`):

```python
def _save_artifacts(fit_id, model_type, amplification, diagnostics, status, stan_file, output_dir):
    """Save posterior draws (Parquet) and model code."""
    bayes_dir = output_dir / "analysis" / "stats" / "bayes"
    
    # Save amplification draws as Parquet
    draws_records = []
    for dim, dim_data in amplification.items():
        a_draws = dim_data.get("A_draws", [])
        for i, val in enumerate(a_draws):
            draws_records.append({"dimension": dim, "draw": i, "A": val})
    
    draws_df = pd.DataFrame(draws_records)
    draws_df.to_parquet(bayes_dir / f"amplification_draws__{fit_id}.parquet")
```

**ArviZ alternative:** Save InferenceData directly:

```python
# Save complete posterior (NetCDF format)
idata.to_netcdf(bayes_dir / f"posterior__{fit_id}.nc")

# Reload later
idata = az.from_netcdf(bayes_dir / f"posterior__{fit_id}.nc")
```

---

## Model Comparison Workflow

### When to Compare Models

Use model comparison when:
- Testing hierarchical vs fixed effects
- Comparing different prior specifications
- Selecting between model complexity levels

### Comparison Checklist

- [ ] All models have `log_lik` in generated quantities
- [ ] All models converged (passed validation)
- [ ] Same data used for all models
- [ ] Check Pareto k values for LOO stability

```python
# Compare hierarchical vs fixed effect
models = {
    "hierarchical": az.from_cmdstanpy(fit_hier),
    "fixed": az.from_cmdstanpy(fit_fixed),
}

comparison = az.compare(models, ic="loo")
print(comparison[["elpd_loo", "p_loo", "weight"]])

# Check if comparison is reliable
for name, idata in models.items():
    loo = az.loo(idata, pointwise=True)
    bad_k = (loo.pareto_k > 0.7).sum()
    if bad_k > 0:
        print(f"WARNING: {name} has {bad_k} high Pareto k values")
```