# ArviZ Patterns Reference

## Contents
- InferenceData Creation
- Diagnostic Extraction Patterns
- Posterior Summarization
- Model Comparison
- Anti-Patterns

---

## InferenceData Creation

### From CmdStanPy (Recommended)

```python
import arviz as az

# Full conversion with all components
idata = az.from_cmdstanpy(
    posterior=fit,
    posterior_predictive="y_rep",      # generated quantity name
    log_likelihood="log_lik",          # for WAIC/LOO
    observed_data={"y": y_observed},   # original data
    coords={"predictor": col_names},   # dimension labels
    dims={"beta": ["predictor"]},      # dimension mapping
)
```

### From Posterior Draws (Manual)

```python
# If you need custom structure
import xarray as xr

# From fit.stan_variable() draws
beta_draws = fit.stan_variable("beta")  # (n_draws, K)

# Reshape to (chains, draws, K)
n_chains, n_draws = 4, 1000
beta_reshaped = beta_draws.reshape(n_chains, n_draws, -1)

posterior = xr.Dataset({
    "beta": (["chain", "draw", "predictor"], beta_reshaped),
})
idata = az.InferenceData(posterior=posterior)
```

---

## Diagnostic Extraction Patterns

### Current Project Approach (CmdStanPy Native)

The project extracts diagnostics directly without ArviZ conversion:

```python
# src/bayes_robustness.py:288-326
def _check_diagnostics(fit: Any) -> Dict[str, Any]:
    diag = fit.diagnose()
    summary = fit.summary()
    
    rhat_vals = summary["R_hat"].dropna()
    ess_vals = summary["N_Eff"].dropna()
    
    max_rhat = float(rhat_vals.max())
    min_ess = float(ess_vals.min())
    
    # Divergences from sampler variables
    sampler_params = fit.method_variables()
    n_divergent = int(sampler_params["divergent__"].sum())
    
    return {"max_rhat": max_rhat, "min_ess_bulk": min_ess, "divergences": n_divergent}
```

### ArviZ Diagnostic Extraction

```python
import arviz as az

# Compute diagnostics
rhat = az.rhat(idata)
ess_bulk = az.ess(idata, method="bulk")
ess_tail = az.ess(idata, method="tail")

# Check for problems
max_rhat = float(rhat.to_array().max())
min_ess = float(ess_bulk.to_array().min())

# Divergences from sample_stats
if "diverging" in idata.sample_stats:
    n_divergent = int(idata.sample_stats.diverging.sum())
```

---

## Posterior Summarization

### Summary Table

```python
# Full summary with all statistics
summary = az.summary(
    idata,
    var_names=["alpha", "beta"],
    hdi_prob=0.95,
    round_to=4,
)
# Returns DataFrame: mean, sd, hdi_2.5%, hdi_97.5%, mcse_mean, ess_bulk, ess_tail, r_hat
```

### Credible Intervals

```python
# HDI (Highest Density Interval) — preferred
hdi = az.hdi(idata, var_names=["beta"], hdi_prob=0.95)

# Quantile-based intervals
import numpy as np
beta_draws = idata.posterior["beta"].values.flatten()
ci_low, ci_high = np.percentile(beta_draws, [2.5, 97.5])
```

### Posterior Probability Calculations

```python
# P(A > 0) — project's amplification metric
A_draws = idata.posterior["A"].values.flatten()
P_A_gt_0 = float(np.mean(A_draws > 0))

# ROPE (Region of Practical Equivalence)
rope = az.plot_posterior(idata, var_names=["beta"], rope=[-0.1, 0.1])
```

---

## Model Comparison

### LOO-CV (Leave-One-Out Cross-Validation)

```python
# Requires log_lik in generated quantities
loo = az.loo(idata, pointwise=True)
print(f"ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

# Check for problematic observations
bad_k = loo.pareto_k > 0.7
if bad_k.any():
    print(f"WARNING: {bad_k.sum()} observations with high Pareto k")
```

### WAIC

```python
waic = az.waic(idata)
print(f"WAIC: {waic.waic:.2f}")
```

### Compare Models

```python
comparison = az.compare(
    {"hierarchical": idata_hier, "fixed": idata_fixed},
    ic="loo",
)
print(comparison[["elpd_loo", "weight"]])
```

---

## Anti-Patterns

### WARNING: Ignoring Divergences

**The Problem:**

```python
# BAD — fitting and ignoring divergences
fit = model.sample(data=stan_data)
idata = az.from_cmdstanpy(fit)
summary = az.summary(idata)  # Proceeds despite divergences
```

**Why This Breaks:**
1. Divergences indicate the sampler failed to explore the posterior
2. Posterior estimates may be biased toward regions the sampler could reach
3. Credible intervals will be unreliable

**The Fix:**

```python
# GOOD — check divergences first
idata = az.from_cmdstanpy(fit)
n_div = int(idata.sample_stats.diverging.sum()) if "diverging" in idata.sample_stats else 0
if n_div > 0:
    raise RuntimeError(f"{n_div} divergent transitions — posterior unreliable")
```

### WARNING: Using Mean Instead of Median for Skewed Posteriors

**The Problem:**

```python
# BAD — mean can be misleading for skewed posteriors
effect = float(idata.posterior["sigma"].mean())
```

**The Fix:**

```python
# GOOD — use median or report full interval
sigma_draws = idata.posterior["sigma"].values.flatten()
effect_median = float(np.median(sigma_draws))
effect_hdi = az.hdi(idata, var_names=["sigma"], hdi_prob=0.95)
```

### WARNING: Forgetting log_lik for Model Comparison

**The Problem:**

```python
# Stan model without log_lik
model {
  y ~ normal(mu, sigma);
}
# Then trying:
loo = az.loo(idata)  # ERROR: log_likelihood not found
```

**The Fix:**

```python
# Stan model WITH log_lik in generated quantities
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
```

See the project's Stan models for examples: `models/stan/tier2_hier_logit.stan:49-53`