---
name: arviz
description: |
  Analyzes and visualizes Bayesian model diagnostics and results.
  Use when: creating trace plots, posterior distributions, forest plots,
  computing convergence diagnostics (rhat, ESS), model comparison (WAIC, LOO-CV),
  or converting CmdStanPy fits to InferenceData format.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# ArviZ Skill

ArviZ provides exploratory analysis of Bayesian models. **This project declares ArviZ but extracts diagnostics directly from CmdStanPy** — see `src/bayes_robustness.py` for the current approach using `fit.summary()` and `fit.method_variables()`.

## Quick Start

### Convert CmdStanPy Fit to InferenceData

```python
import arviz as az
from cmdstanpy import CmdStanModel

# After fitting with CmdStanPy
fit = model.sample(data=stan_data, chains=4, iter_sampling=1000)

# Convert to ArviZ InferenceData
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive="y_rep",  # if generated in Stan
    log_likelihood="log_lik",      # for model comparison
)
```

### Extract Diagnostics (Current Project Pattern)

```python
# src/bayes_robustness.py:288-326 — direct CmdStanPy extraction
def _check_diagnostics(fit: Any) -> Dict[str, Any]:
    summary = fit.summary()
    rhat_vals = summary["R_hat"].dropna()
    ess_vals = summary["N_Eff"].dropna()
    
    sampler_params = fit.method_variables()
    n_divergent = int(sampler_params["divergent__"].sum())
    
    return {
        "max_rhat": float(rhat_vals.max()),
        "min_ess_bulk": float(ess_vals.min()),
        "divergences": n_divergent,
    }
```

## Key Concepts

| Concept | CmdStanPy Native | ArviZ Equivalent |
|---------|------------------|------------------|
| Rhat | `fit.summary()["R_hat"]` | `az.rhat(idata)` |
| ESS | `fit.summary()["N_Eff"]` | `az.ess(idata)` |
| Divergences | `fit.method_variables()["divergent__"]` | `idata.sample_stats.diverging` |
| Posterior draws | `fit.stan_variable("beta")` | `idata.posterior["beta"]` |
| Summary table | `fit.summary()` | `az.summary(idata)` |

## Common Patterns

### Diagnostic Visualization

```python
import arviz as az

# Trace plot: chains should mix well (no separation)
az.plot_trace(idata, var_names=["beta", "alpha"])

# Posterior distributions with HDI
az.plot_posterior(idata, var_names=["beta"], hdi_prob=0.95)

# Forest plot for comparing coefficients
az.plot_forest(idata, var_names=["beta"], combined=True)
```

### Model Comparison

```python
# Requires log_lik in generated quantities
loo = az.loo(idata)
waic = az.waic(idata)

# Compare multiple models
comparison = az.compare({"model_a": idata_a, "model_b": idata_b})
```

## See Also

- [patterns](references/patterns.md) — InferenceData creation, diagnostic extraction
- [workflows](references/workflows.md) — Model validation, comparison workflows

## Related Skills

- See the **cmdstanpy** skill for Stan model fitting and sampling
- See the **numpy** skill for posterior array operations
- See the **matplotlib** skill for custom diagnostic plots
- See the **pandas** skill for summary DataFrame handling