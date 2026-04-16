# Step 5: Remedy Decision (Persona-Rewritten)

## Overview

Step 5 assigns remedy tiers to the persona-rewritten complaints from Step 4. Because each original complaint was rewritten into 36 persona styles while preserving all factual content, any differences in remedy decisions across personas constitute **causal evidence of linguistic bias**. This is the core causal test of the experiment.

**Source:** `src/experiment.py` → `_step_5()` (lines 1160–1214), `_summarize_step_5()` (lines 1216–1268)
**Statistics:** `src/repeated_measures.py` → 10 statistical models
**Bayesian:** `src/bayes_robustness.py` → Stan-based robustness checks

---

## What This Step Does

1. **Applies the same remedy decision** prompt to each rewritten complaint (identical to Step 2)
2. **Collects tier assignments** for all persona × complaint combinations
3. **Extracts text features** from rewritten complaints (26 metrics)
4. **Runs repeated-measures causal analysis** to test whether persona style affects decisions

---

## Execution

### Request Construction

For each model and each persona rewrite:

```python
req = LLMRequest(
    request_id=LLMRequest.new_id(),
    case_id=rw.case_id,
    model=model,
    task="remedy_decision",
    prompt_system=REMEDY_DECISION_SYSTEM,
    prompt_user=REMEDY_DECISION_USER.format(narrative=rw.rewritten_narrative),
    expected_schema=RemedyDecisionResponse,
    variant=rw.persona_label,
)
```

The prompt, schema, and calibration anchors are **identical** to Step 2. The only difference is the input text (rewritten vs. original) and the variant label (persona label vs. `"baseline"`). This ensures any tier differences are attributable to the text changes, not to prompt variation.

### Result Storage

Responses are stored as `RemedyDecision` objects with the persona label as the variant:

```python
RemedyDecision(
    case_id=resp.case_id,
    model=resp.model,
    variant=req.variant,       # e.g., "young_urban_college_high"
    tier=p.tier,
    confidence=p.confidence,
    reasoning=p.reasoning,
    information_needed=p.information_needed,
)
```

### Text Feature Extraction

After collecting decisions, Step 5 extracts 26 text features from all rewritten complaints and appends them to `self.text_features_df` with a `persona_label` column:

```python
rewrite_features = extract_features_batch(
    [r.rewritten_narrative for r in self.persona_rewrites],
    [r.case_id for r in self.persona_rewrites],
    "persona_rewrite",
)
rw_df = features_to_dataframe(rewrite_features)
rw_df["persona_label"] = [r.persona_label for r in self.persona_rewrites]
```

---

## Console Summary

`_summarize_step_5()` logs per model:

1. **Persona × tier crosstab** — top 10 personas by count, showing tier distribution
2. **Mean tier by causal dimension** — average tier for each level of age, geographic, education, socioeconomic
3. **Overall tier distribution** — value counts across all personas
4. **Process discrimination rate** — overall `need_more_info` rate and per-dimension info request rates

The dimension values are parsed from the variant label:

```python
parts = m["variant"].str.split("_", expand=True)
m["age"] = parts[0]
m["geographic"] = parts[1]
m["education"] = parts[2]
m["socioeconomic"] = parts[3]
```

---

## Causal Analysis: Repeated-Measures Models

The primary causal analysis is implemented in `src/repeated_measures.py`. Because the same complaint is observed under multiple persona framings, the data has a **repeated-measures** structure that must be modelled explicitly.

### Long-Format Data Assembly

`build_long_format()` (lines 62–152) constructs the canonical analysis dataset:

| Column | Description |
|--------|-------------|
| `complaint_id` | Original CFPB complaint ID (clustering unit) |
| `model` | LLM identifier |
| `condition` | `"original"` or `"neutralised"` |
| `persona_id` | One of 36 persona labels |
| `severity_bin` | `"low"` / `"medium"` / `"high"` |
| `outcome_tier` | Ordinal remedy tier (0, 1, 2) |
| `favourable` | Binary: `tier >= 1` |
| `info_request` | Binary: `confidence == "need_more_info"` |

Severity is determined by `issue_severity` with fallback to `monetary_severity`, defaulting to `"medium"` when neither is available.

### Model Suite

Ten statistical models are run, organised by outcome variable and modelling approach:

#### A. Ordinal Outcome (outcome_tier)

| Model | Function | Method | Purpose |
|-------|----------|--------|---------|
| Ordinal mixed model | `ordinal_mixed_model()` | OrderedModel (logit link) + clustered SEs | Respects ordinal nature of tier |
| Linear mixed model | `linear_mixed_model_tier()` | MixedLM with complaint_id random intercept | Robustness check treating tier as continuous |
| Friedman test | `repeated_measures_nonparametric()` | Nonparametric within-complaint | Distribution-free confirmatory test |

#### B. Binary Outcome (favourable: tier >= 1)

| Model | Function | Method | Purpose |
|-------|----------|--------|---------|
| Conditional logistic | `conditional_logistic_favourable()` | ConditionalLogit with complaint_id strata | Gold standard within-complaint comparison |
| Logistic GEE | `logistic_mixed_model_favourable()` | GEE Binomial + Exchangeable correlation | GLMM approximation with flexibility |
| Cochran's Q | `repeated_measures_nonparametric()` | Multi-condition binary test | Distribution-free omnibus |
| Pairwise McNemar | Post-hoc after Cochran's Q | Paired binary comparisons | Identify which personas differ |

#### C. Process Outcome (info_request)

| Model | Function | Method | Purpose |
|-------|----------|--------|---------|
| Logistic GEE (info) | `logistic_mixed_model_info_request()` | GEE Binomial + Exchangeable | Process discrimination model |
| Cochran's Q (info) | `repeated_measures_nonparametric()` | Multi-condition binary test | Omnibus process test |
| Pairwise McNemar (info) | Post-hoc after Cochran's Q | Paired binary comparisons | Which personas face higher burden |

### Fixed Effects (All Regression Models)

All parametric models include the same fixed-effect specification:

| Predictor | Construction | Interpretation |
|-----------|-------------|----------------|
| `sev_ord` | Ordinal severity (low=0, medium=1, high=2) | Severity main effect |
| `condition_code` | 1 if neutralised, 0 otherwise | Neutralisation effect |
| `persona_score` | Composite disadvantage score from persona dimensions | Persona main effect |
| `persona_x_severity` | `persona_score × sev_ord` | Severity amplification |
| `mdl_*` | Model dummies (if multiple LLMs) | Cross-model differences |

The `persona_score` is a numeric summary constructed by `_add_persona_score()` from the four causal dimension dummy variables, enabling parsimony in interaction terms.

---

## Fallback Chains

Each parametric model implements a multi-step fallback chain to handle problematic data:

### Ordinal Mixed Model

| Step | Condition | Action |
|------|-----------|--------|
| 1 | Standard fit | OrderedModel (logit) + clustered SEs on complaint_id |
| 2 | Rank deficiency | Sanitise design matrix, re-fit |
| 3 | Convergence failure | Standardise predictors, increase iterations |
| 4 | All failures | Skip with diagnostic (exploratory model) |

### Conditional Logistic (Confirmatory)

| Step | Condition | Action |
|------|-----------|--------|
| 1 | Standard fit | ConditionalLogit with complaint_id strata |
| 2 | Rank deficiency | Sanitise design matrix, re-fit |
| 3 | Convergence failure | Standardise predictors, re-fit |
| 4 | All failures | **Fatal error** (confirmatory estimand) |

### Logistic GEE

| Step | Condition | Action |
|------|-----------|--------|
| 1 | Standard fit | GEE Binomial + Exchangeable correlation |
| 2 | Correlation failure | GEE Binomial + Independence correlation |
| 3 | All failures | **Fatal error** (confirmatory estimand) |

### Linear Mixed Model

| Step | Condition | Action |
|------|-----------|--------|
| 1 | REML | Standard REML estimation |
| 2 | Singular REML | ML estimation |
| 3 | Singular ML | Powell optimiser with zero start params |
| 4 | All failures | Fatal error with diagnostics |

The distinction between "skip" (exploratory models) and "fatal error" (confirmatory models) reflects the pre-registered analysis plan: conditional logistic and GEE models are confirmatory estimands that must produce results or fail loudly.

---

## Nonparametric Tests

### Friedman Test (Ordinal Tier)

Tests whether the distribution of `outcome_tier` differs across persona variants within each complaint:

1. Pivot data: rows = complaint_id, columns = persona_id, values = outcome_tier
2. Require ≥3 persona columns and ≥5 complete cases
3. Run Friedman test (nonparametric repeated-measures ANOVA)
4. If significant (p < 0.05), run post-hoc pairwise Wilcoxon signed-rank tests with FDR correction

### Cochran's Q (Binary Outcomes)

Tests whether `favourable` or `info_request` rates differ across persona variants:

1. Pivot data: rows = complaint_id, columns = persona_id, values = binary outcome
2. Require ≥3 persona columns and ≥5 complete cases
3. Run Cochran's Q (generalised McNemar for >2 conditions)
4. If significant (p < 0.05), run post-hoc pairwise McNemar tests with FDR correction

### Post-Hoc Pair Limiting

To avoid combinatorial explosion (36 personas → 630 pairs), post-hoc tests are limited to:
- The 10 personas with highest mean outcome + 10 with lowest mean outcome
- Maximum 100 pairs tested
- FDR correction applied across all tested pairs

---

## Bayesian Robustness Checks

`src/bayes_robustness.py` provides Stan-based Bayesian models as sensitivity analysis:

### Stan Models

Six Stan models in `models/stan/`:

| Model | Outcome | Random Effects |
|-------|---------|----------------|
| `tier2_hierarchical` | Binary (tier ≥ 1) | Complaint random intercepts |
| `tier2_fixed` | Binary (tier ≥ 1) | No random effects |
| `tier2_flex` | Binary (tier ≥ 1) | Flexible per-column priors |
| `tier3_hierarchical` | Ordinal (3-level) | Complaint random intercepts |
| `tier3_fixed` | Ordinal (3-level) | No random effects |
| `tier3_flex` | Ordinal (3-level) | Flexible per-column priors |

### Priors

- `beta ~ Normal(0, 0.8)` for main effects
- `cutpoints ~ Normal(0, 1.5)` for ordinal thresholds
- Flex models use per-column prior SD: main = 0.8, interaction = 0.4
- Ordinal cutpoint reparameterization: `c[1] = c_base, c[2] = c_base + softplus(c_diff_raw) + 0.1` to guarantee ordering

### Diagnostic Gate

Results are flagged FAIL if:
- `rhat > 1.01` (convergence)
- `ESS < 400` (effective sample size)
- Any divergent transitions

### Singleton Threshold

If >50% of complaints have exactly 1 observation, hierarchical models switch to fixed-effect models (no complaint random intercepts) to avoid over-parameterisation.

### 5-Step Fallback Chain

1. Standard fit
2. Category-level random effects
3. Per-dimension random effects
4. Flexible priors
5. Aggressive sampler settings

---

## Multiple Comparison Correction

All p-values from pairwise post-hoc tests are corrected using **Benjamini-Hochberg FDR**:

- Wilcoxon post-hoc pairs (up to 100 tests)
- McNemar post-hoc pairs (up to 100 tests)
- Applied separately within each test family

---

## Orchestrator

`run_repeated_measures_analysis()` (lines 1156–1199) orchestrates the full suite:

1. Checks analysis cache for previously computed results
2. Builds the long-format dataset via `build_long_format()`
3. Runs all 10 models (4 parametric + 3 nonparametric outcomes × 2 for pooled/per-model)
4. Caches results for future runs

Results are returned as a nested dictionary keyed by model name and analysis type.

---

## Interpretation

Step 5 results answer the experiment's core causal question:

- **Significant persona effects** in the conditional logistic model or Cochran's Q indicate that linguistic style causally affects remedy decisions
- **Effect sizes** from odds ratios and mean tier differences quantify the magnitude of bias
- **Persona × severity interactions** reveal whether bias amplifies for high-stakes complaints
- **Info request rates** by persona identify process discrimination (differential procedural burden)

The matched-guise design controls for all complaint-level confounders because the same complaint appears under every persona style, making this a within-subject comparison.
