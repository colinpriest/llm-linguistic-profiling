# CmdStanPy Workflows Reference

## Contents
- Installation
- Stan Data Preparation
- Fallback Chain Strategy
- Artifact Storage
- Troubleshooting

## Installation

### Windows with Conda

```python
# install-stan.py
import os
from cmdstanpy import install_cmdstan

CONDA_PREFIX = os.environ.get("CONDA_PREFIX")
mingw_bin = os.path.join(CONDA_PREFIX, "Library", "mingw-w64", "bin")
library_bin = os.path.join(CONDA_PREFIX, "Library", "bin")

os.environ["PATH"] = mingw_bin + ";" + library_bin + ";" + os.environ["PATH"]
os.environ["MAKE"] = os.path.join(mingw_bin, "mingw32-make.exe")

install_cmdstan(version="2.35.0")  # Pin version for reproducibility
```

### Installation Checklist

Copy this checklist and track progress:
- [ ] Activate conda environment
- [ ] Run: `python install-stan.py`
- [ ] Verify: `cmdstanpy.cmdstan_path()` returns valid path
- [ ] Test compile: `CmdStanModel(stan_file="models/stan/tier2_hier_logit.stan")`

## Stan Data Preparation

### Building Design Matrix

```python
def _build_design_matrix(df: pd.DataFrame, severity_metric: str) -> tuple:
    """Build X matrix with persona dummies, severity dummies, interactions."""
    col_names = []
    parts = []
    
    # 1. Persona dimension dummies (k-1 per dimension)
    for dim in ["age", "geographic", "education", "socioeconomic"]:
        ref = {"age": "young", "geographic": "urban", ...}[dim]
        for level in df[dim].unique():
            if level == ref:
                continue
            parts.append((df[dim] == level).astype(float).values[:, None])
            col_names.append(f"{dim}__{level}")
    
    # 2. Severity dummies (low is reference)
    # 3. Interactions: persona × severity
    
    X = np.hstack(parts)
    return X, col_names
```

### Complaint ID Mapping

```python
# Map case_id strings to 1-indexed integers for Stan
complaint_ids = df["case_id"].values
unique_complaints = np.unique(complaint_ids)
complaint_map = {cid: i + 1 for i, cid in enumerate(unique_complaints)}
jj = np.array([complaint_map[cid] for cid in complaint_ids])
J = len(unique_complaints)
```

### WARNING: Missing case_id

```python
# GOOD - Validate before mapping
if "case_id" not in df.columns:
    raise RuntimeError("Missing case_id column for Bayesian robustness.")
if df["case_id"].isna().sum() > 0:
    raise RuntimeError("Rows with missing case_id cannot have random effects.")
```

## Fallback Chain Strategy

This project uses a 5-step fallback chain for monetary_severity (which often has difficult posteriors):

### Step-by-Step Fallback

```python
# Step 1: Standard hierarchical fit
result = _try_b1_b2(...)

# Step 2: Category-level random intercept (coarser grouping)
if result is None and "product_category" in df.columns:
    # Use product_category instead of complaint ID
    result = _try_b1_b2(J=len(unique_categories), jj=category_index, ...)

# Step 3: Per-dimension models (smaller design matrix)
if result is None:
    for dim in ["age", "geographic", "education", "socioeconomic"]:
        X_dim, _ = _build_design_matrix(df, severity_metric, dimensions=[dim])
        # Fit each dimension separately, then merge results

# Step 4: Stronger priors (flex models)
if result is None:
    prior_sd = np.array([0.5 if "__x__" in c else 1.0 for c in col_names])
    # Use tier2_hier_logit_flex.stan with prior_sd data

# Step 5: Aggressive sampler settings
if result is None:
    result = _try_b1_b2(..., adapt_delta=0.99, max_treedepth=15)
```

### Fallback Validation Loop

1. Attempt fit with current strategy
2. Extract diagnostics: `fit.diagnose()`, `fit.summary()`
3. Gate: if rhat > 1.01, ESS < 400, or divergences > 0 → FAIL
4. If FAIL, proceed to next fallback step
5. If all steps fail, raise RuntimeError with attempted strategies

## Artifact Storage

### Saving Posterior Draws

```python
def _save_artifacts(fit_id: str, amplification: dict, output_dir: Path):
    bayes_dir = output_dir / "analysis" / "stats" / "bayes"
    bayes_dir.mkdir(parents=True, exist_ok=True)
    
    # Save draws as Parquet (efficient for large arrays)
    draws_records = []
    for dim, dim_data in amplification.items():
        for i, val in enumerate(dim_data["A_draws"]):
            draws_records.append({"dimension": dim, "draw": i, "A": val})
    
    draws_df = pd.DataFrame(draws_records)
    draws_df.to_parquet(
        bayes_dir / f"stats__bayes__amplification_draws__{fit_id}__v1.parquet",
        index=False
    )
```

### Saving Diagnostics

```python
import json
import shutil

# Copy Stan source for reproducibility
shutil.copy2(stan_file, bayes_dir / f"stats__bayes__model_code__{fit_id}__v1.stan")

# Save diagnostics JSON
diag_path = bayes_dir / f"stats__bayes__diagnostics__{fit_id}__v1.json"
diag_path.write_text(json.dumps({
    "fit_id": fit_id,
    "status": status,
    "diagnostics": diagnostics,
}, indent=2))
```

## Troubleshooting

### Divergent Transitions

**Symptom:** `divergent__` count > 0 in sampler diagnostics

**Causes:**
1. Posterior geometry has sharp curvature (funnels, ridges)
2. Step size too large for local curvature
3. Prior-data conflict

**Fixes:**
- Increase `adapt_delta` to 0.99 or 0.999
- Use non-centred parameterisation for random effects
- Tighten priors on variance parameters

### Low ESS

**Symptom:** `N_Eff` < 400 for some parameters

**Causes:**
1. Insufficient warmup for mass matrix adaptation
2. High autocorrelation in chains
3. Poorly identified parameters

**Fixes:**
- Increase `iter_warmup` to 1000+
- Increase `max_treedepth` to 13-15
- For singletons: switch to fixed-effect model

### Rhat > 1.01

**Symptom:** Chains have not converged

**Causes:**
1. Multimodal posterior
2. Unidentified parameters (singletons)
3. Insufficient warmup

**Fixes:**
- Run longer chains
- Check singleton threshold: if >50% complaints have 1 observation, random intercepts are unidentifiable
- Use per-dimension fallback (smaller model)

### Singleton Threshold Logic

```python
obs_per_complaint = pd.Series(complaint_ids).value_counts()
pct_singletons = (obs_per_complaint == 1).sum() / len(obs_per_complaint)

if pct_singletons > 0.50:
    # Random intercepts unidentifiable — use fixed-effect models
    use_random_intercept = False
    logger.info("%.0f%% singletons — switching to fixed-effect", 100 * pct_singletons)