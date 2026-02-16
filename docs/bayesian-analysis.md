# Bayesian Robustness Analysis

## Overview

The Bayesian robustness analysis serves as a **non-confirmatory** robustness check on the severity amplification findings from Steps 4-5 of the experiment. While the primary statistical tests (chi-squared, logistic regression, GEE) establish whether demographic disparities exist in LLM remedy decisions, the Bayesian layer asks a more nuanced question: **does the magnitude of demographic disparity grow as complaint severity increases?**

This analysis fits Stan-based probabilistic models to the matched-guise data, estimates posterior distributions over "amplification" parameters, and reports credible intervals and posterior probabilities. Because it is non-confirmatory, the results are treated as supplementary evidence rather than hypothesis tests.

The analysis is implemented in `src/bayes_robustness.py` and uses six Stan model files in `models/stan/`.

## Why Bayesian Methods Here?

The frequentist severity analysis (Step 8) uses GEE models and bootstrap confidence intervals, which provide valid point estimates but have known limitations in this context:

1. **Small-cell sparsity.** When crossing 4 demographic dimensions x 3 severity bands x 3 remedy tiers, many cells have fewer than 5 observations. Maximum-likelihood estimates become unstable or exhibit complete separation.
2. **Partial pooling.** Bayesian hierarchical models can borrow strength across complaints, stabilising estimates where data is thin, through complaint-level random intercepts.
3. **Direct probability statements.** Instead of p-values, the Bayesian analysis reports P(A > 0) — the posterior probability that amplification is positive — which is more directly interpretable for a policy audience.
4. **Quantified uncertainty.** Full posterior distributions provide credible intervals that naturally widen when data is sparse, rather than producing point estimates with infinite standard errors.

## Models

The analysis fits two model types for each (LLM model x severity metric) combination, yielding up to 4 fits per run (2 models x 2 severity metrics x N LLMs):

### B1: Hierarchical Logistic Regression (Tier-2 Indicator)

**File:** `models/stan/tier2_hier_logit.stan`

**Purpose:** Models the probability of the most severe remedy outcome (Tier 2 = full monetary relief) as a function of demographic persona dimensions, severity bands, and their interactions.

**Specification:**

```
y_i ~ Bernoulli(logit^{-1}(alpha + X_i * beta + u[complaint_i]))
beta ~ Normal(0, 0.8)
alpha ~ Normal(0, 2)
u_j = z_j * sigma_u          (non-centred parameterisation)
z_j ~ Normal(0, 1)
sigma_u ~ HalfNormal(0, 0.5)
```

**Why this model:** A binary logistic model isolates the "highest tier" question — is the complaint deemed worthy of full relief? This is the most policy-relevant outcome and the one most sensitive to demographic bias. The binary response avoids the cutpoint estimation problems that plague ordinal models (see B2 below).

**Design matrix columns (~20-40):**
- Persona dimension dummies (k-1 per dimension, reference level dropped): `age__old`, `geographic__rural`, `geographic__suburban`, `education__college`, `socioeconomic__high`, `socioeconomic__middle`
- Severity band dummies (k-1, reference = low): `sev__medium`, `sev__high`
- Interactions: each persona dummy x each severity dummy (e.g., `age__old__x__sev__high`)

**Non-centred parameterisation:** The complaint-level random intercept u_j is parameterised as z_j * sigma_u rather than u_j ~ Normal(0, sigma_u). This breaks the funnel geometry between sigma_u and u_j that causes divergent transitions in centred parameterisations, especially when J (number of complaints) is large and many complaints have only one observation.

### B2: Hierarchical Ordinal Cumulative-Logit (Tier 0/1/2)

**File:** `models/stan/tier3_hier_ordinal.stan`

**Purpose:** Models the full three-tier ordinal outcome (Tier 0 = no relief, Tier 1 = partial relief, Tier 2 = full relief) to capture whether demographic bias operates differently across the entire remedy spectrum.

**Specification:**

```
y_i ~ OrderedLogistic(X_i * beta + u[complaint_i], c)
beta ~ Normal(0, 0.8)
c[1] = c_base
c[2] = c_base + softplus(c_diff_raw) + 0.1    (guaranteed c[2] > c[1])
c_base ~ Normal(0, 1.5)
c_diff_raw ~ Normal(0, 1.0)
sigma_u ~ HalfNormal(0, 0.5)
```

**Why this model:** The ordinal model uses all three tiers of information rather than collapsing to binary. This can detect patterns where, for example, a demographic group is shifted from Tier 0 to Tier 1 (rather than to Tier 2), which the logistic model would miss.

**Note on the intercept:** Unlike B1, the ordinal model has **no separate alpha parameter**. The cutpoints c[1] and c[2] absorb the intercept — adding a separate alpha would create a non-identifiable redundancy with the cutpoints.

**Cutpoint reparameterisation:** Stan's `ordered_logistic` distribution requires c[1] < c[2]. Rather than using Stan's built-in `ordered` type with an implicit log-transform (which creates boundary issues), the model uses a softplus reparameterisation:

```
c[2] = c[1] + log(1 + exp(c_diff_raw)) + 0.1
```

The softplus function is always positive, and the +0.1 minimum gap prevents the cutpoints from collapsing to the same value. This eliminates the "Cut-points not valid ordered vector" errors that occurred with naive parameterisations.

### Fixed-Effect Variants

**Files:** `tier2_fixed_logit.stan`, `tier3_fixed_ordinal.stan`

These are identical to the hierarchical models but without the complaint-level random intercept (u_j, z_j, sigma_u parameters are removed). They are used when complaint-level random intercepts are unidentifiable — see the singleton threshold discussion below.

### Flex-Prior Variants

**Files:** `tier2_hier_logit_flex.stan`, `tier3_hier_ordinal_flex.stan`

These accept a `prior_sd` vector as data, allowing per-column prior standard deviations. Used in fallback step 4, they apply tighter priors to interaction terms (prior_sd = 0.4) than to main effects (prior_sd = 0.8), reflecting the prior belief that interaction effects are smaller than main effects and helping regularise the many near-zero interaction coefficients.

## Prior Justification

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| beta (coefficients) | Normal(0, 0.8) | At 1 SD, the odds ratio is exp(0.8) ≈ 2.2, which represents a meaningful but not extreme demographic effect. This is tighter than the common default of Normal(0, 1) and provides stronger shrinkage for the many interaction terms that are likely near zero. |
| alpha (logistic intercept) | Normal(0, 2) | Allows the baseline log-odds to range freely; at 2 SD the probability ranges from about 2% to 98%. |
| c_base (first cutpoint) | Normal(0, 1.5) | Weakly informative on the logit scale. |
| c_diff_raw (cutpoint gap) | Normal(0, 1.0) | softplus(0) ≈ 0.69, so the typical gap between cutpoints is ~0.8 on the logit scale, producing moderate separation between adjacent tiers. |
| sigma_u (RE std dev) | HalfNormal(0, 0.5) | 95% of mass below ~1.0 on the logit scale. An earlier version used Exponential(1) (mean = 1.0), but this was too diffuse — singleton complaints (one observation per complaint) have nearly unidentifiable random effects, which drifted to extreme values and caused non-convergence (rhat > 1.2, ESS < 15). |

## Amplification Estimands

The key quantities computed from posterior draws are:

**D(s)** — the **disparity** at severity level s:

```
D(s) = max_level P(Tier2 | level, severity=s) - min_level P(Tier2 | level, severity=s)
```

This is the maximum gap in predicted P(Tier 2) across all levels of a demographic dimension, holding severity fixed. For example, if at high severity, the predicted Tier-2 probability ranges from 15% (rural) to 25% (urban), then D(high) = 0.10 for the geographic dimension.

**A** — the **amplification**:

```
A = D(high) - D(low)
```

If A > 0, the demographic disparity is larger at high severity than at low severity — meaning the LLM's demographic bias is *amplified* for more severe complaints. The analysis reports:

- **A_mean**: Posterior mean of amplification
- **A_CrI**: 95% credible interval [2.5th, 97.5th percentiles]
- **P(A > 0)**: Posterior probability that amplification is positive

For the ordinal model, P(Tier 2) is computed as:

```
P(Tier2) = P(y = 3) = 1 - logistic(c[2] - eta)
```

where eta = X * beta (no separate alpha in the ordinal model).

## Pre-Flight Diagnostics

Before any Stan model is fitted, the pipeline runs a battery of pre-flight checks on the data and design matrix. These checks predict whether MCMC will converge and trigger automatic mitigations when problems are detected.

### Why Pre-Flight Checks Are Needed

MCMC sampling is expensive — a single Stan fit takes 5-15 minutes with 4 chains x 2000 iterations. Without pre-flight checks, the pipeline would waste hours attempting fits that are mathematically doomed to fail, only to discover the failure after the fact. Pre-flight diagnostics identify structural problems *before* sampling begins and either fix them automatically or skip MCMC entirely in favour of faster approximate methods.

### Check 1: Events Per Parameter (EPP)

**What it checks:** The ratio of minority-class events to the number of predictor columns.

```
EPP = min(n_tier2, n_non_tier2) / K
```

**Why:** Logistic and ordinal regression require sufficient events per parameter to estimate coefficients reliably. The classical rule of thumb is EPP ≥ 10; below this, maximum-likelihood estimates become biased, confidence intervals have poor coverage, and MCMC chains struggle to explore the posterior.

**What it triggers:**
- EPP < 10: Recommend dropping interaction terms (auto-simplification if difficulty is "very hard")
- EPP < 5: Flag as very low; convergence unlikely even with main-effects-only model

### Check 2: Design Matrix Rank and Condition Number

**What it checks:** Whether the design matrix X has full column rank and whether the Gram matrix X^T X is well-conditioned.

**Why:** Rank deficiency means some coefficients are perfectly collinear and unidentifiable. An ill-conditioned Gram matrix (condition number > 10^10) means near-collinearity, which causes the posterior to have narrow ridges that MCMC struggles to traverse efficiently.

**What it triggers:**
- Rank deficient: Flag; the model will have degenerate parameters
- Ill-conditioned: Recommend stronger priors or dimensionality reduction

### Check 3: Cross-Tabulation Cell Sparsity

**What it checks:** For every combination of (dimension level x severity band x tier), counts the number of observations and flags cells with fewer than 5.

**Why:** Sparse cells mean the data provides almost no information about certain interaction effects. The model can still be fit, but the posterior will be dominated by the prior for those cells, and MCMC may struggle because the likelihood surface is nearly flat in those regions.

**What it triggers:** Informational warning. If more sparse cells than parameters, contributes to difficulty score.

### Check 4: Logistic Separation

**What it checks:** For each binary dummy column in X, counts how many positive (Tier 2) and negative (non-Tier 2) events occur when that column equals 1.

**Why:** Complete separation occurs when a dummy column perfectly predicts the outcome — e.g., when `age__old__x__sev__high` is 1, every observation is Tier 2 (or every observation is non-Tier 2). This makes the logistic likelihood flat in that dimension: the MLE for that coefficient is ±infinity, and MCMC chains get stuck at their initial values because there is no gradient to guide them. The result is constant draws within each chain but different values between chains (high rhat, ESS equal to the number of chains, zero divergences).

Near-separation (fewer than 2 minority-class events per cell) produces the same symptoms to a lesser degree.

**What it triggers:**
- Complete separation in any column, or near-separation in >25% of columns: Set `b1_skip_mcmc = True`, skipping all MCMC steps (1-5) for the B1 logistic model and deferring directly to approximate methods (ADVI, Pathfinder, MAP).

**Observed in practice:** In the experiment log, 4/20 dummy columns showed complete separation for gpt-4o-mini/monetary_severity, causing all MCMC steps to be skipped for B1.

### Check 5: Ordinal Tier Imbalance

**What it checks:** Whether any ordinal response level (Tier 0, 1, or 2) contains fewer than 10% of the total observations.

**Why:** The ordinal cumulative-logit model estimates two cutpoints that partition the logit scale into three regions. When one tier is very rare (e.g., Tier 2 at 8.3% of observations), the corresponding cutpoint is pushed to an extreme value, and the gap between cutpoints becomes highly asymmetric. This creates a narrow **posterior ridge** — a region where the posterior is concentrated along a thin manifold in the (c_base, c_diff_raw, beta) space. NUTS (No-U-Turn Sampler) uses leapfrog integration with a fixed step size adapted during warmup, and it cannot efficiently traverse a ridge whose curvature changes sharply. The result is chains that get stuck in different parts of the ridge: constant draws within each chain (zero divergences), but different locations between chains (rhat >> 1, ESS ≈ number of chains).

This failure mode is particularly insidious because the **zero divergences** make it look superficially healthy. Unlike funnel geometries that produce divergences as a warning signal, the ridge geometry simply traps chains without triggering Stan's divergence detector.

**What it triggers:**
- Min tier rate < 10%: Set `b2_skip_mcmc = True`, skipping all MCMC steps for B2 and deferring to approximate methods.

**Observed in practice:** For gpt-4o-mini/issue_severity, Tier 2 had only 8.3% of observations. Every MCMC attempt on the ordinal model (standard, category-RE, per-dimension) failed with rhat > 1.03 and ESS < 120, always with zero divergences — the signature of a ridge posterior.

### Check 6: Singleton Threshold

**What it checks:** What fraction of complaints have exactly one observation (one persona row).

**Why:** The complaint-level random intercept u_j is meant to capture within-complaint correlation (multiple persona rewrites of the same complaint should be correlated in their remedy outcomes). When a complaint has only one observation, its random effect u_j is completely unidentifiable from the data — it can take any value that, combined with the fixed effects, produces the observed binary outcome. With diffuse priors on sigma_u, these singleton random effects drift to extreme values, destabilising the posterior. When >50% of complaints are singletons, the random intercept structure absorbs more variance than it explains, causing non-convergence.

**What it triggers:**
- Singleton rate > 50%: Use fixed-effect models (no complaint-level random intercept) instead of hierarchical models.

**Observed in practice:** In the experiment data, 100% of complaints were singletons (each complaint has exactly one observation per model), so fixed-effect models were always used.

### Check 7: Rare Severity Band Collapse

**What it checks:** Whether any severity band (low, medium, high) has fewer than 10 observations.

**Why:** With fewer than 10 observations in a severity band, the corresponding severity dummy and all its interaction terms are estimated from almost no data. The likelihood contribution is negligible, so the posterior for those coefficients is essentially the prior, adding noise to the model without information.

**What it triggers:** Merges the rare band into the adjacent band (high→medium, then low→medium).

### Check 8: Zero-Variance Column Removal

**What it checks:** Whether any column in the design matrix has near-zero variance (< 10^-12).

**Why:** A zero-variance column (e.g., an interaction dummy that is always 0 because no observations fall in that cell) creates a degenerate parameter — the data provides no information, but the sampler must still explore the parameter's prior. This wastes computational effort and can interact badly with mass matrix adaptation.

**What it triggers:** Drops the column from the design matrix and logs the removal.

### Difficulty Score

The individual checks are combined into a composite difficulty score:

| Condition | Points |
|-----------|--------|
| EPP < 5 | +3 |
| EPP 5-10 | +2 |
| EPP 10-20 | +1 |
| Rank deficient | +2 |
| Ill-conditioned | +1 |
| More sparse cells than parameters | +1 |
| Singleton rate > 50% | +1 |
| B1 logistic separation (skip_mcmc) | +2 |
| B2 ordinal imbalance (skip_mcmc) | +2 |

| Score | Label | Action |
|-------|-------|--------|
| 0 | easy | Standard fit |
| 1 | moderate | Standard fit with warnings |
| 2-3 | hard | Warnings; expect some fallbacks |
| 4+ | very_hard | Auto-simplify design matrix (drop interactions) |

## The Fallback Chain

When the standard model fit fails diagnostic checks, the pipeline progressively tries more robust configurations. B1 (logistic) and B2 (ordinal) are tracked **independently**: if one succeeds at step N, only the other is retried at step N+1.

### Diagnostic Gate

Each fit is evaluated against convergence criteria:

**Strict gate (steps 1-3.5):**
- **FAIL** if rhat > 1.01, ESS < 400, or any divergent transitions
- **WARN** if treedepth hits only
- **OK** otherwise

**Relaxed gate (steps 4-5):**
- **FAIL** if rhat > 1.05, ESS < 200, or > 2 divergent transitions
- **MARGINAL** if passes relaxed but would fail strict thresholds
- **WARN** if treedepth hits only
- **OK** if passes strict thresholds too

The relaxed gate reflects a pragmatic compromise: a fit with rhat = 1.02 and ESS = 350 may not meet gold-standard convergence criteria, but its posterior is likely close enough to be informative as a robustness check (not a confirmatory test).

### Step 1: Standard Fit

**Configuration:** Hierarchical or fixed-effect model (depending on singleton threshold) with all predictors, standard priors, standard sampler settings (4 chains, 1000+1000 iterations, adapt_delta = 0.99, max_treedepth = 13).

**Why it often fails:** The full design matrix has ~20 columns (8 main effects + 12 interactions) but many interaction cells are sparse. The ordinal model is especially vulnerable because the cutpoint ridge compounds the small-cell problem.

### Step 2: Category-Level Random Intercept

**Configuration:** Replaces complaint-level random intercept with a product-category-level intercept (J ≈ 5 categories instead of J ≈ 6700 complaints).

**Rationale:** If the standard model fails because of too many random effect levels, reducing J from thousands to single digits may help. The product category (e.g., "credit cards", "mortgages") captures coarse-grained heterogeneity that might otherwise destabilise the complaint-level intercept.

**When it helps:** When the complaint-level RE is unidentifiable due to singletons but some between-category variation exists. In practice, this step rarely rescues the ordinal model because the fundamental cutpoint ridge problem persists regardless of the RE structure.

### Step 3: Per-Dimension Models

**Configuration:** Instead of fitting one model with all 4 demographic dimensions, fits 4 separate models — one per dimension (age, geographic, education, socioeconomic), each with that dimension's levels x severity interactions.

**Rationale:** Each per-dimension model has only ~4-8 columns instead of ~20+, dramatically increasing EPP and reducing the chance of separation. Results are merged across dimensions to produce the same amplification estimates.

**Partial credit:** If some dimensions succeed and others fail, the pipeline returns partial results (with a `partial=True` flag and WARN status). Failed dimensions are retried once with 8 chains — but only if the first failure was due to a sampling exception (not a diagnostic gate failure), since more chains cannot fix bad posterior geometry.

**Observed in practice:** For gpt-4o-mini/monetary_severity, the geographic dimension succeeded (rhat = 1.002, ESS = 1421) while age, education, and socioeconomic failed (rhat > 1.03, ESS < 112). The partial result from the geographic dimension was accepted as a WARN.

### Step 3.5: Main-Effects-Only Model

**Configuration:** Drops all interaction terms, fitting only main effects (demographic dummies + severity dummies). Reduces K from ~20 to ~8.

**Rationale:** If interaction effects are the source of instability (sparse cells, near-separation), removing them may allow the main effects to be estimated cleanly. The amplification estimand becomes cruder (no dimension x severity interaction), but it can still detect whether demographic disparities differ between high and low severity through the severity main effects.

### Step 4: Stronger Priors (Flex Model) with Relaxed Gate

**Configuration:** Uses the flex-prior Stan model with per-column prior standard deviations: 0.8 for main effects, 0.4 for interactions. Applies the relaxed diagnostic gate.

**Rationale:** Tighter priors on interactions (allowing odds ratios up to ~exp(0.4) ≈ 1.5 at 1 SD) prevent near-zero interaction coefficients from drifting to extreme values during sampling, which can destabilise the mass matrix. The relaxed gate accepts MARGINAL fits that are close enough for a non-confirmatory robustness check.

### Step 5: Aggressive Sampler with Relaxed Gate

**Configuration:** Same as step 4 but with max_treedepth = 15 (instead of 13).

**Rationale:** A higher treedepth allows NUTS to take longer trajectories, which can help traverse narrow ridges in the posterior. The cost is roughly 4x longer per transition at the maximum treedepth, but this is the last MCMC attempt before falling back to approximate methods.

### Step 6: Variational Inference (ADVI)

**Configuration:** Uses CmdStan's ADVI (Automatic Differentiation Variational Inference) with a mean-field approximation and fixed-effect models (no random intercepts).

**Rationale:** When MCMC fails, variational inference provides an approximate posterior by optimising a simpler distribution (factored Gaussian) to be close to the true posterior in KL divergence. ADVI converges in seconds rather than minutes and doesn't suffer from the mixing problems that plague MCMC on ridge posteriors. The trade-off is that the mean-field approximation cannot capture posterior correlations between parameters, so credible intervals may be too narrow.

**Status label:** Fits that succeed via ADVI receive status "ADVI" rather than "OK" to signal that the posterior is approximate.

**Known issue:** In the current CmdStanPy version, ADVI's `stan_variable()` sometimes returns scalars instead of arrays, causing downstream computation failures. When this occurs, the pipeline falls through to Pathfinder.

### Step 7: Pathfinder

**Configuration:** Uses CmdStan's Pathfinder algorithm (requires CmdStan ≥ 2.33) with 4 paths and 1000 draws.

**Rationale:** Pathfinder is a newer approximate inference method that traces the L-BFGS optimisation path and uses the curvature information to construct an approximate posterior. It handles correlated posteriors better than mean-field ADVI and is typically faster. It provides draws that can be used with the same amplification computation functions.

**Status label:** "PATHFINDER".

**Observed in practice:** Pathfinder was the most common resolution method in the experiment log, succeeding for all 4 (model x metric) B1 fits and 2 of 4 B2 fits. It typically completed in 10-30 seconds.

### Step 8: MAP Point Estimate

**Configuration:** L-BFGS optimisation to find the maximum a posteriori (MAP) point estimate. Fixed-effect models only.

**Rationale:** The last resort. MAP provides a single best-fit parameter vector with no uncertainty quantification. The amplification computation produces point estimates where the credible interval collapses to a single value (CrI_low = CrI_high = point estimate). Results should be treated as "directional evidence only."

**Status label:** "MAP".

## Why MCMC Frequently Fails on This Data

The matched-guise experiment data presents a particularly challenging posterior geometry for MCMC, for several reinforcing reasons:

### 1. Extreme Tier Imbalance

The three-tier remedy outcome is typically distributed approximately 33%/49%/18% (Tier 0/1/2). Tier 2 (full relief) is the most policy-relevant but also the rarest category. For the ordinal model, this means:

- The second cutpoint c[2] is pushed far to the right on the logit scale
- The gap between cutpoints becomes large and asymmetric
- The posterior concentrates on a **thin ridge** in (c_base, c_diff_raw, beta) space
- NUTS chains initialise at different points on the ridge and stay there
- Result: rhat >> 1, ESS ≈ number of chains, **zero divergences**

When additional subsetting by severity metric reduces sample size (e.g., monetary_severity requires the complaint to have a dollar amount), the imbalance can worsen to 8% Tier 2 or below, making the ridge even narrower.

### 2. Pervasive Complete Separation

The design matrix crosses 4 demographic dimensions with 2-3 levels each against 3 severity bands, producing ~12 interaction columns. Many of these interaction cells contain zero or one Tier-2 events. For the logistic model (B1), this is **complete or near-complete separation**: the MLE for the corresponding coefficient is ±infinity.

Bayesian priors technically prevent infinite coefficients, but the prior Normal(0, 0.8) is not strong enough to dominate the flat likelihood in a separated cell. The posterior becomes bimodal — one mode near the prior, one far away — and MCMC chains split between modes.

### 3. Singleton Complaint Structure

Each base complaint generates one observation per LLM model in the baseline dataset. This means 100% of complaints are singletons — each complaint_id has exactly one row. A hierarchical model with complaint-level random intercepts would have as many random effect parameters as observations (J = N), making them completely unidentifiable. The pipeline detects this via the singleton threshold check and falls back to fixed-effect models.

### 4. High Dimensionality Relative to Events

With ~20 columns and sometimes only ~1200 Tier-2 events (for the logistic model) or ~500 Tier-2 events in a per-severity-metric subset (for the ordinal model), the EPP ranges from 25 to 60. While this is above the classical threshold of 10, the combination with separation and imbalance makes it marginal in practice.

### 5. The Ridge-Without-Divergences Problem

The most insidious failure mode is that the ordinal model's cutpoint ridge produces **zero divergences**. In standard Stan diagnostics, divergent transitions are the primary warning signal for problematic posterior geometry (funnels, banana shapes). But the ordinal ridge is smooth — it has no sharp curvature changes that would cause the leapfrog integrator to diverge. Instead, chains simply fail to mix: each chain explores a small region of the ridge and never moves to a different region. This means:

- `stansummary` reports rhat >> 1 and ESS < 50 (indicating non-mixing)
- But divergences = 0 (indicating no geometry problem detected by the integrator)
- The user might be tempted to increase adapt_delta or max_treedepth, but these adjustments won't help because the integrator is already accurately following the local posterior — it just can't reach distant parts of the ridge

This is why the pre-flight ordinal imbalance check is critical: it detects the ridge geometry from the data *before* MCMC is attempted and routes directly to approximate methods.

## Post-Mortem Diagnostic Reports

Regardless of whether fits succeed, partially succeed, or completely fail, the pipeline always generates a structured post-mortem report for each (model, metric) combination. This report is:

1. **Logged** to the experiment log at WARNING level for immediate visibility
2. **Saved** as a JSON artifact in `output/analysis/stats/bayes/`

The post-mortem contains:

| Section | Contents |
|---------|----------|
| Data summary | N, K, J, EPP, tier distribution, severity distribution, singleton rate |
| Outcome distribution | Full severity x tier cross-tabulation |
| Cross-tab sparsity | Count of cells with < 5 observations |
| Design matrix diagnostics | Rank, condition number, interaction/main-effect column counts |
| Per-step diagnostics | For each fallback step attempted: label, B1/B2 status, rhat, ESS, divergences |
| Root causes | Algorithmic analysis of why fits failed (overparameterisation, separation, imbalance, etc.) |
| Recommendations | Context-aware suggestions that account for which strategies were already tried |
| Final resolution | Which method resolved each model type (B1/B2) and at which step |

### Context-Aware Recommendations

The recommendations section avoids generic advice like "try more iterations" and instead checks what was actually attempted. For example:

- If per-dimension models were tried and failed: "Per-dimension models were attempted (B1=FAIL, B2=WARN); EPP=62.1 remains low — data may be fundamentally overparameterized for this sample size"
- If Pathfinder resolved both models: "Approximate inference attempted: Pathfinder: B1=PATHFINDER, B2=PATHFINDER; B1 resolved via pathfinder; B2 resolved via pathfinder"
- If everything failed: "MAP point estimates can provide directional evidence even without full posterior uncertainty"

## Consolidated Summary Table

At the end of all (model x metric) fits, the pipeline logs a consolidated summary table showing every combination's pre-flight difficulty prediction, the final B1 and B2 resolution (status and fallback step), and the number of fallback steps attempted.

Example from an actual experiment run:

```
BAYESIAN ROBUSTNESS — CONSOLIDATED SUMMARY
========================================================================
  Model        Metric             Pre-flight            B1 Logistic              B2 Ordinal                      Steps
  -----------  -----------------  --------------------  -----------------------  ------------------------------  -----
  gpt-4o-mini  monetary_severity  hard (EPP=62.2)       PATHFINDER @ pathfinder  WARN @ per-dimension (partial)  5
  gpt-4o-mini  issue_severity     very_hard (EPP=87.1)  PATHFINDER @ pathfinder  PATHFINDER @ pathfinder         3
  gpt-5-mini   monetary_severity  moderate (EPP=85.2)   PATHFINDER @ pathfinder  PATHFINDER @ pathfinder         8
  gpt-5-mini   issue_severity     hard (EPP=128.4)      PATHFINDER @ pathfinder  WARN @ per-dimension (partial)  5
  -----------  -----------------  --------------------  -----------------------  ------------------------------  -----
  Totals (4 combos):  B1: none  |  B2: 2 WARN
  Fallback order: standard -> category-RE -> per-dimension -> main-effects-only ->
                  flex-prior -> aggressive-sampler -> variational-advi -> pathfinder ->
                  map-estimate
========================================================================
```

Key observations from this run:
- **B1 logistic** was always resolved via Pathfinder because separation in dummy columns prevented all MCMC attempts
- **B2 ordinal** was split: two metric-model combos were resolved via per-dimension partial results (only some demographic dimensions converged), while two were resolved via Pathfinder
- Despite EPP values that look healthy (62-128), the combination of separation and tier imbalance made MCMC infeasible for nearly all configurations
- The pipeline spent 5-8 steps per combination but ultimately produced usable amplification estimates for all 4 combos

## Interpreting Results

Because the Bayesian analysis is non-confirmatory and frequently relies on approximate inference methods, results should be interpreted with appropriate caveats:

| Resolution Status | Interpretation |
|-------------------|----------------|
| OK | Full MCMC posterior; credible intervals are trustworthy |
| WARN | MCMC converged with minor issues (treedepth hits) or partial per-dimension results; credible intervals are likely reliable but may be slightly too narrow |
| MARGINAL | MCMC converged under relaxed thresholds; treat as supportive evidence, not definitive |
| ADVI | Approximate posterior via variational inference; credible intervals may be too narrow due to mean-field approximation |
| PATHFINDER | Approximate posterior via Pathfinder; better than ADVI for correlated posteriors but still approximate |
| MAP | Point estimate only; no posterior uncertainty. Treat as directional evidence only |
| FAIL | All methods failed; no Bayesian evidence available for this combination |

In all cases, the Bayesian amplification estimates should be compared with the frequentist results from the severity analysis (Step 8) for consistency. Agreement between frequentist and Bayesian approaches strengthens the evidence; disagreement warrants caution and further investigation.

## Sampling Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chains | 4 | Standard; allows rhat computation |
| Warmup iterations | 1000 | Mass matrix adaptation for ~30-40 parameters + J random effects. 750 was insufficient in early testing (rhat > 1.2 on issue_severity runs). |
| Sampling iterations | 1000 | 4000 total post-warmup draws across chains; sufficient for stable posterior summaries |
| adapt_delta | 0.99 | High target acceptance rate to reduce divergent transitions in difficult posteriors |
| max_treedepth | 13 (15 for aggressive) | Allows long trajectories to traverse narrow posterior ridges. Default of 10 was insufficient. |

## Artifacts

The analysis produces the following artifacts in `output/analysis/stats/bayes/`:

| File Pattern | Contents |
|--------------|----------|
| `stats__bayes__amplification_draws__*__v1.parquet` | Full posterior draws of the amplification estimand A, by dimension |
| `stats__bayes__model_code__*__v1.stan` | Copy of the Stan source used for each fit |
| `stats__bayes__diagnostics__*__v1.json` | MCMC diagnostics (rhat, ESS, divergences, stuck parameters) |
| `stats__bayes__postmortem__*__v1.json` | Full post-mortem diagnostic report |

## Source Files

| File | Purpose |
|------|---------|
| `src/bayes_robustness.py` | Main module: pre-flight checks, fallback chain, amplification computation, post-mortem |
| `models/stan/tier2_hier_logit.stan` | B1 hierarchical logistic model |
| `models/stan/tier3_hier_ordinal.stan` | B2 hierarchical ordinal model |
| `models/stan/tier2_fixed_logit.stan` | B1 fixed-effect logistic model |
| `models/stan/tier3_fixed_ordinal.stan` | B2 fixed-effect ordinal model |
| `models/stan/tier2_hier_logit_flex.stan` | B1 flex-prior logistic model |
| `models/stan/tier3_hier_ordinal_flex.stan` | B2 flex-prior ordinal model |
