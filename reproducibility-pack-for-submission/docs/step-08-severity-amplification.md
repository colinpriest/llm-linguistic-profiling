# Step 8: Severity Amplification Analysis

## Overview

Step 8 tests whether LLM bias **amplifies with complaint severity** — that is, whether high-stakes complaints face larger disparities than low-stakes complaints. Two independent severity metrics are analysed: monetary severity (dollar amounts in dispute) and issue severity (product × issue category). The analysis includes 12 sections spanning descriptive statistics, ANOVA, bootstrap amplification ratios, and mitigation effectiveness by severity level.

**Source:** `src/severity_analysis.py` (2,438 lines) → `run_severity_analysis()`
**Scoring:** `src/data_loader.py` → `compute_issue_severity()` (lines 299–320)
**Dollar extraction:** `src/prompts.py` → `DOLLAR_EXTRACT_SYSTEM`, `DOLLAR_EXTRACT_USER`

---

## Severity Metrics

### 1. Monetary Severity

Derived from dollar amounts mentioned in complaint narratives, extracted via GPT-4o-mini:

| Level | Dollar Range | Description |
|-------|-------------|-------------|
| low | $0–$100 | Minor fees and small charges |
| medium | $101–$1,000 | Moderate financial impact |
| high | $1,001+ | Significant financial harm |
| unknown | No dollar amount found | ~40–60% of complaints |

#### Dollar Amount Extraction

```
DOLLAR_EXTRACT_SYSTEM:
You are a financial information extraction assistant.
Extract dollar amounts from consumer complaints accurately.

DOLLAR_EXTRACT_USER:
Extract any dollar amounts mentioned in this complaint. Return JSON with:
- amounts_mentioned: list of dollar values found
- primary_amount: the main disputed/affected amount (or null if unclear)
- amount_type: one of ["fee", "balance", "payment", "damage", "other"]
- confidence: high/medium/low
```

The `DollarAmountResponse` Pydantic schema validates the output with coercion for `amount_type` values.

### 2. Issue Severity

A composite score derived from the product category and issue category of each complaint using a deterministic scoring system in `src/data_loader.py`.

#### Scoring System

**Product Base Score (Tier 1):**

| Score | Products | Rationale |
|-------|----------|-----------|
| 3 (High) | Mortgage, Student loan, Debt collection | Life-altering consequences |
| 2 (Medium) | Credit card, Vehicle loan, Payday loan | Significant but recoverable |
| 1 (Low) | Checking/savings account, Prepaid card, Credit reporting | Smaller amounts, procedural |

**Issue Modifier (Tier 2):**

| Modifier | Examples | Rationale |
|----------|----------|-----------|
| +2 (Critical) | Struggling to pay mortgage, Threatened with jail | Asset/service loss |
| +1 (Elevated) | Attempts to collect debt not owed, Incorrect credit report | Active collection or credit damage |
| 0 (Standard) | Managing an account, Deposits and withdrawals | Billing/service disputes |
| -1 (Lower) | Getting information, Privacy issues | Informational/procedural |

**Composite Score:** `score = product_base + issue_modifier`, clamped to [1, 5]:
- Score 1–2 → `"low"`
- Score 3 → `"medium"`
- Score 4–5 → `"high"`

**Product × Issue Overrides:** ~50 specific combinations where the default composite would be misleading (e.g., "Mortgage" + "Struggling to pay mortgage" is always score 5 = `"high"`).

---

## Analysis Sections

The severity analysis comprises 12 sections, implemented as individual functions in `src/severity_analysis.py`:

### Section 1: Descriptive Analysis

**Function:** `severity_descriptives()`

- Distribution of severity levels per metric (low/medium/high/unknown)
- Coverage rate (% of complaints with known severity)
- Agreement between monetary and issue severity (Spearman correlation, Cohen's kappa, agreement rate)
- Cross-tabulation of monetary × issue severity
- Breakdown of product/issue categories within each severity level

### Section 2: Severity Main Effects

**Function:** `severity_main_effects()`

Tests whether severity affects outcomes independently of persona:

| Test | Method | Outcome |
|------|--------|---------|
| Mean tier by severity level | Group means + 95% CI | Do higher-severity complaints get higher tiers? |
| One-way ANOVA | `tier ~ severity` | Is the severity effect significant? |
| Chi-squared | `favourable × severity` | Does the favourable rate vary by severity? |
| Spearman trend | Ordinal severity → tier | Is there a monotonic trend? |

Run for both metrics × all models.

### Section 3a: Severity × Persona Interaction (Full Grid)

**Function:** `bias_by_severity()` + `two_way_anova_persona_severity()`

The core amplification test using the full 36-persona grid:

- **Bias at each severity level**: SD of mean outcome across personas (higher SD = more disparity)
- **Two-way ANOVA**: `tier ~ persona + severity + persona×severity` (Type II SS)
  - **Persona main effect**: Do personas differ on average?
  - **Severity main effect**: Does severity affect tier?
  - **Interaction**: Does persona disparity change with severity? (amplification test)
- **Simple effects**: One-way ANOVA of persona effect at each severity level separately
- **Pairwise Cohen's d**: Largest effect size between any two personas at each severity level
- **FDR correction**: Applied across all pairwise t-tests within each severity level

**Status:** Exploratory (two-way ANOVA uses OLS, not the pre-registered GEE model).

### Section 3b: Amplification Ratio (Bootstrap)

**Function:** `amplification_ratio()`

The key summary statistic:

```
amplification = bias_high / bias_low
```

Where `bias` = SD of mean tier across personas at a given severity level.

- **Amplification > 1.0**: Bias increases with severity (bad)
- **Amplification = 1.0**: Bias is constant across severity (neutral)
- **Amplification < 1.0**: Bias decreases with severity (better)

**Bootstrap 95% CI**: 1,000 iterations resampling at the complaint (case_id) level, with operation-level caching to avoid recomputation.

### Section 3c: Dimension-Specific Severity Interaction

**Function:** `dimension_severity_interaction()`

Tests severity amplification for each causal dimension **individually** (rather than the full 36-persona grid), yielding much better-powered tests (~200+ observations per cell vs. ~15–20 for the full grid):

- Two-way ANOVA: `tier ~ dimension + severity + dimension:severity`
- Per-dimension amplification ratio with bootstrap CI
- Mean tier by dimension level at each severity level

Run for all 4 causal dimensions × 2 severity metrics × all models.

### Section 3d: Logistic GEE Interaction (Confirmatory)

**Function:** `logistic_interaction_tier2()`

The pre-registered confirmatory test:

```
logit P(tier >= 2) = dimension + severity + dimension:severity
```

- GEE with exchangeable correlation clustered by `case_id`
- Wald test for interaction terms (joint test)
- Coefficients with odds ratios and confidence intervals
- Run per dimension × severity metric × model

This is the **confirmatory** interaction test specified in the pre-registration plan.

### Section 4: Three-Way ANOVA

**Function:** `three_way_anova()`

```
tier ~ model × persona × severity
```

Tests whether severity amplification differs by LLM model (requires multiple models). Exploratory; provides the three-way interaction `model × persona × severity`.

### Section 5: Worst-Case Analysis

**Function:** `worst_case_analysis()`

Finds the most discriminatory `model × severity_metric × severity_level` combination:
- Identifies the persona with the highest and lowest mean tier
- Computes Cohen's d between the most and least advantaged personas
- Reports the single most concerning combination across all models and metrics

### Section 6: Process × Severity

**Function:** `process_severity_analysis()`

Tests whether **process discrimination** (info request burden) amplifies with severity:
- `need_more_info` rate by severity level
- Persona disparity in info request rates at each severity level
- Logistic regression: `need_info ~ persona + severity + persona:severity`
- Process amplification ratio: `disparity_high / disparity_low`

### Section 7: Consistency Check

**Function:** `consistency_check()`

Compares findings across the two severity metrics:
- Do both monetary and issue severity show significant main effects?
- Do both show significant persona × severity interactions?
- Are amplification ratios directionally consistent?
- Classification: "Robust" (both significant), "Null" (neither), "Metric-specific" (one only)

### Section 8: Mitigation × Severity

**Function:** `mitigation_severity_comparison()`

Tests whether language neutralisation (Step 6) **removes severity-dependent amplification**:
- Computes amplification ratios before and after neutralisation
- Computes bias SD at each severity level before/after
- Compares two-way ANOVA interaction p-values
- Verdict: "eliminated", "reduced", "unchanged", or "increased"

### Section 9: Model-Specific Amplification

**Function:** `model_specific_amplification()`

Amplification ratio per model per severity metric. Uses ThreadPoolExecutor for parallel bootstrap computation.

### Section 10: Monetary Missingness Diagnostic

Analyses whether missing monetary severity (`"unknown"`) is related to persona, which could bias the monetary severity analysis.

### Section 11: IPW Robustness (if applicable)

Inverse probability weighting to adjust for severity selection into the analysis sample.

### Section 12: Orchestrator

**Function:** `run_severity_analysis()`

Runs all sections with caching, parallel execution, and error handling. Returns a nested dictionary keyed by section name.

---

## Pre-Flight Diagnostics

Every ANOVA analysis uses the `anova_preflight()` function (lines 60–229) which performs 7 checks before fitting:

1. Required columns exist
2. Minimum sample size (typically n ≥ 30)
3. Outcome dtype is numeric or coercible
4. No Inf values in outcome
5. Outcome has variance (not constant)
6. Each factor has ≥ 2 levels
7. Cell counts for interactions (no empty cells)

Failures return structured diagnostic objects rather than raising exceptions.

---

## Not-Estimable (NE) Handling

When a statistic cannot be computed (e.g., zero denominator, insufficient data), the analysis returns `NEValue` objects from `src/estimable.py` instead of NaN or None. This enables:
- Downstream code to distinguish "not computed" from "computed as zero"
- Structured error messages in reports
- Safe rendering in HTML tables (NE values display as "N/E" with tooltips)

---

## Relationship to Other Steps

| Step | Relationship |
|------|-------------|
| Steps 2, 5, 7 | Provides the remedy decisions analysed by severity |
| Step 1 | Provides persona labels for cross-tabulation |
| Data Loader | Provides issue severity scores and product/issue categories |
| Dollar Extraction | Provides monetary severity from complaint narratives |
| Step 5 | Repeated-measures models include severity × persona interactions |
| Step 7 | Mitigation × severity tests neutralisation effectiveness by severity |
