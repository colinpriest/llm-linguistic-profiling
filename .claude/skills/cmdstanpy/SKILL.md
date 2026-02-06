---
name: cmdstanpy
description: |
  Executes Bayesian robustness checks using Stan probabilistic models.
  Use when: fitting hierarchical logistic/ordinal models, running MCMC sampling,
  extracting posterior draws, checking convergence diagnostics (rhat, ESS, divergences),
  or implementing fallback strategies for difficult posteriors.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# CmdStanPy Skill

This project uses CmdStanPy for Bayesian robustness checks on severity amplification analysis. Two model families are fitted per severity metric: B1 (hierarchical logistic for Tier-2 indicator) and B2 (hierarchical ordinal cumulative-logit for Tier 0/1/2). Non-centred parameterisations handle complaint-level random effects.

## Quick Start

### Compile and Fit a Model

```python
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file="models/stan/tier2_hier_logit.stan")
fit = model.sample(
    data={"N": len(df), "K": X.shape[1], "X": X.tolist(),
          "J": J, "jj": jj.tolist(), "y": y.tolist()},
    chains=4,
    iter_sampling=1000,
    iter_warmup=1000,
    adapt_delta=0.99,
    max_treedepth=13,
    seed=42,
)
```

### Extract Posterior Draws

```python
alpha_draws = fit.stan_variable("alpha")   # (n_draws,)
beta_draws = fit.stan_variable("beta")     # (n_draws, K)
```

### Check Diagnostics

```python
diag_text = fit.diagnose()
summary_df = fit.summary()
max_rhat = summary_df["R_hat"].max()
min_ess = summary_df["N_Eff"].min()
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Non-centred RE | Avoid divergences with random effects | `u = z_u * sigma_u` in transformed parameters |
| adapt_delta | Tighter step size for difficult posteriors | `0.99` for complex hierarchical models |
| max_treedepth | Allow deeper exploration | `13-15` for models with interactions |
| Singleton threshold | Switch to fixed-effect if >50% singletons | Random intercepts unidentifiable |

## Common Patterns

### Gating Fit Results

```python
def _gate_fit(diagnostics: dict) -> str:
    if diagnostics["max_rhat"] > 1.01 or diagnostics["min_ess_bulk"] < 400:
        return "FAIL"
    if diagnostics["divergences"] > 0:
        return "FAIL"
    if diagnostics["treedepth_hits"] > 0:
        return "WARN"
    return "OK"
```

## See Also

- [patterns](references/patterns.md) - Model fitting patterns and diagnostics
- [workflows](references/workflows.md) - Installation, fallback chains, artifact storage

## Related Skills

- See the **arviz** skill for posterior visualization and LOO-CV
- See the **numpy** skill for design matrix construction
- See the **pandas** skill for data preparation