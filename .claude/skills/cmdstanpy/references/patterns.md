# CmdStanPy Patterns Reference

## Contents
- Model Compilation
- Sampling Configuration
- Posterior Extraction
- Diagnostic Checks
- Non-Centred Parameterisation
- Prior Specification

## Model Compilation

### Standard Compilation

```python
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file=str(stan_file))
```

Compilation takes 1-2 minutes on first run. Compiled executables are cached in `~/.cmdstan/`.

### WARNING: Missing Stan File

```python
# BAD - No existence check
model = CmdStanModel(stan_file="models/stan/missing.stan")  # Cryptic error

# GOOD - Check and provide actionable error
if not stan_file.exists():
    raise RuntimeError(
        f"Stan model file not found: {stan_file}\n"
        f"Ensure the models/stan/ directory contains the required .stan files."
    )
```

## Sampling Configuration

### Production Settings (This Project)

```python
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_sampling=1000,
    iter_warmup=1000,      # 1000 needed for mass matrix adaptation
    adapt_delta=0.99,      # Tighter step size for hierarchical models
    max_treedepth=13,      # Allow deeper exploration
    seed=42,
    show_progress=True,
)
```

### WARNING: Insufficient Warmup

**The Problem:**

```python
# BAD - 500 warmup insufficient for ~30-40 column design matrix
fit = model.sample(data=data, iter_warmup=500, iter_sampling=1000)
```

**Why This Breaks:**
1. Mass matrix adaptation incomplete — chains fail to converge
2. Rhat > 1.2, ESS < 15 on difficult posteriors (issue_severity)
3. Posterior draws are unreliable for inference

**The Fix:**

```python
# GOOD - 1000 warmup for large design matrices
fit = model.sample(data=data, iter_warmup=1000, iter_sampling=1000,
                   adapt_delta=0.99, max_treedepth=13)
```

## Posterior Extraction

### Extracting Parameters

```python
# Single parameter (scalar per draw)
alpha_draws = fit.stan_variable("alpha")   # shape: (n_draws,)

# Vector parameter
beta_draws = fit.stan_variable("beta")     # shape: (n_draws, K)

# Ordered cutpoints
c_draws = fit.stan_variable("c")           # shape: (n_draws, 2)
```

### Computing Derived Quantities

```python
def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# Predicted probability from logistic model
eta = alpha_draws + beta_draws @ x_template
p_tier2 = _expit(eta)  # shape: (n_draws,)

# Credible intervals
ci_low = float(np.percentile(p_tier2, 2.5))
ci_high = float(np.percentile(p_tier2, 97.5))
```

## Diagnostic Checks

### Extracting Diagnostics

```python
def _check_diagnostics(fit) -> dict:
    diag = fit.diagnose()
    summary = fit.summary()
    
    rhat_vals = summary["R_hat"].dropna()
    ess_vals = summary["N_Eff"].dropna()
    
    # Count divergences from sampler diagnostics
    sampler_params = fit.method_variables()
    n_divergent = int(sampler_params["divergent__"].sum())
    n_treedepth = int((sampler_params["treedepth__"] >= 13).sum())
    
    return {
        "max_rhat": float(rhat_vals.max()),
        "min_ess_bulk": float(ess_vals.min()),
        "divergences": n_divergent,
        "treedepth_hits": n_treedepth,
    }
```

### Acceptance Gate

```python
def _gate_fit(diagnostics: dict) -> str:
    """Returns 'OK', 'WARN', or 'FAIL'."""
    rhat = diagnostics.get("max_rhat", float("nan"))
    ess = diagnostics.get("min_ess_bulk", 0)
    div = diagnostics.get("divergences", 0)
    
    # Hard fail: any divergences or non-convergence
    if rhat > 1.01 or ess < 400 or div > 0:
        return "FAIL"
    if diagnostics.get("treedepth_hits", 0) > 0:
        return "WARN"
    return "OK"
```

## Non-Centred Parameterisation

### Why Non-Centred?

Centred parameterisation causes funnel geometry when random effect variance is small:

```stan
// BAD - Centred (causes divergences)
parameters {
  vector[J] u;
  real<lower=0> sigma_u;
}
model {
  u ~ normal(0, sigma_u);  // Funnel when sigma_u → 0
}
```

```stan
// GOOD - Non-centred
parameters {
  vector[J] z_u;           // Standard normal
  real<lower=0> sigma_u;
}
transformed parameters {
  vector[J] u = z_u * sigma_u;  // Scale after sampling
}
model {
  z_u ~ std_normal();
}
```

## Prior Specification

### Default Priors (This Project)

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| `beta` | `Normal(0, 1)` | OR ≈ 2.7 at 1 SD — large but plausible |
| `alpha` | `Normal(0, 2)` | Wider for intercept |
| `sigma_u` | `Normal(0, 0.5)` | Half-normal; 95% below ~1.0 on logit scale |
| `c` (cutpoints) | `Normal(0, 2)` | Keeps cutpoints in ±4 range |

### Per-Column Priors (Flex Models)

```stan
data {
  vector<lower=0>[K] prior_sd;  // Per-column prior SD
}
model {
  beta ~ normal(0, prior_sd);   // Tighter on interactions
}
```

```python
# Python: 1.0 for main effects, 0.5 for interactions
prior_sd = np.array([0.5 if "__x__" in c else 1.0 for c in col_names])