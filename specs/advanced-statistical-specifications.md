## Revised Statistical Specifications (with Severity Measures)

### Overview

This document specifies the statistical analyses required for the LLM fairness study. All analyses must accommodate two severity metrics:

* **monetary_severity** : Four levels (≤$100, $100-$1000, >$1000, unknown)
* **issue_severity** : Three levels (Low, Medium, High)

The "unknown" category in monetary_severity requires special handling as specified below.

---

## 1. Power Analysis

### 1.1 Purpose

Demonstrate that the experimental design has sufficient statistical power to detect effects of interest, including main effects, two-way interactions (persona × severity), and three-way interactions (model × persona × severity).

### 1.2 Where to Apply

* Run once during experimental design phase
* Report results in Methodology section of paper
* Add summary to dashboard under new "Statistical Power" panel

### 1.3 Algorithm

**Step 1: Define parameters for each analysis**

| Analysis                                    | α   | Target Power | Effect Size (f) | Numerator df            |
| ------------------------------------------- | ---- | ------------ | --------------- | ----------------------- |
| Persona main effect                         | 0.05 | 0.80         | 0.10            | (P - 1)                 |
| Severity main effect (monetary, known only) | 0.05 | 0.80         | 0.10            | 2                       |
| Severity main effect (issue)                | 0.05 | 0.80         | 0.10            | 2                       |
| Persona × monetary_severity interaction    | 0.05 | 0.80         | 0.10            | (P - 1) × 2            |
| Persona × issue_severity interaction       | 0.05 | 0.80         | 0.10            | (P - 1) × 2            |
| Model × Persona × monetary_severity       | 0.05 | 0.80         | 0.10            | (M - 1) × (P - 1) × 2 |
| Model × Persona × issue_severity          | 0.05 | 0.80         | 0.10            | (M - 1) × (P - 1) × 2 |

Where P = number of persona levels, M = number of models

**Step 2: For each analysis, calculate**

```
Required N per cell = solve for N where:
  Power(N, α, f, df_numerator, df_denominator) = 0.80
  
Using F-test power formula:
  df_denominator = N_total - number_of_cells
  Non-centrality parameter λ = f² × N_total
  Power = 1 - F_cdf(F_critical, df_num, df_denom, λ)
```

**Step 3: Calculate achieved power**

```
For each analysis:
  actual_N_per_cell = min(cell sizes in design)
  actual_N_total = sum of all cell sizes
  achieved_power = Power(actual_N, α, f=0.10, df_num, df_denom)
```

**Step 4: Calculate minimum detectable effect**

```
For each analysis:
  MDE = solve for f where:
    Power(actual_N, α, f, df_num, df_denom) = 0.80
```

### 1.4 Special Handling for monetary_severity

* **Primary analysis:** Exclude "unknown" category. Use only three known levels (≤$100, $100-$1000, >$1000)
* **Secondary analysis:** Include "unknown" as fourth category to test whether unknown-severity complaints show different patterns
* Report coverage: percentage of complaints with known monetary severity
* Flag if coverage < 50% (may limit generalizability of monetary severity findings)

### 1.5 Metrics to Calculate

For each of the 7 analyses listed above:

* Required N per cell for 80% power at f = 0.10
* Actual N per cell (minimum across cells)
* Actual N per cell (median across cells)
* Achieved power given actual cell sizes
* Minimum detectable effect size at 80% power given actual N
* Number of cells with N < 30 (insufficient for stable estimates)

### 1.6 Dashboard Addition

**New collapsible section: "Statistical Power Analysis"**

**Summary metrics (prominent display):**

* "Design achieves [X]% power to detect persona × severity interactions of size f = 0.10"
* "Minimum detectable interaction effect at 80% power: f = [X]"

**Power table:**

| Analysis                              | Required N/cell | Actual N/cell (min) | Achieved Power | MDE (f) | Status |
| ------------------------------------- | --------------- | ------------------- | -------------- | ------- | ------ |
| Persona main effect                   |                 |                     |                |         |        |
| monetary_severity main effect         |                 |                     |                |         |        |
| issue_severity main effect            |                 |                     |                |         |        |
| Persona × monetary_severity          |                 |                     |                |         |        |
| Persona × issue_severity             |                 |                     |                |         |        |
| Model × Persona × monetary_severity |                 |                     |                |         |        |
| Model × Persona × issue_severity    |                 |                     |                |         |        |

**Traffic light in Status column:**

* Green: Achieved power ≥ 0.80
* Yellow: Achieved power 0.60-0.80
* Red: Achieved power < 0.60

**Coverage warning (if applicable):**

* Display: "Note: monetary_severity could only be determined for [X]% of complaints. Power calculations for monetary severity analyses are based on this subset."

### 1.7 Output File: power_analysis.json

```json
{
  "parameters": {
    "alpha": 0.05,
    "target_power": 0.80,
    "effect_size_f": 0.10
  },
  "monetary_severity_coverage": {
    "known_count": N,
    "unknown_count": N,
    "coverage_rate": 0.XX
  },
  "issue_severity_coverage": {
    "known_count": N,
    "coverage_rate": 1.00
  },
  "analyses": {
    "persona_main": {
      "df_numerator": X,
      "required_n_per_cell": X,
      "actual_n_per_cell_min": X,
      "actual_n_per_cell_median": X,
      "achieved_power": 0.XX,
      "minimum_detectable_effect": 0.XX,
      "underpowered_cells": X
    },
    "monetary_severity_main": {
      // same structure
    },
    "issue_severity_main": {
      // same structure
    },
    "persona_x_monetary": {
      // same structure
    },
    "persona_x_issue": {
      // same structure
    },
    "model_x_persona_x_monetary": {
      // same structure
    },
    "model_x_persona_x_issue": {
      // same structure
    }
  },
  "warnings": [
    "monetary_severity coverage below 50%",
    "X cells have fewer than 30 observations"
  ],
  "timestamp": "..."
}
```

---

## 2. Bootstrap Confidence Intervals

### 2.1 Purpose

Provide distribution-free confidence intervals for all key bias estimates, avoiding parametric assumptions. Enable determination of statistical significance without relying on normality.

### 2.2 Where to Apply

**Group A: Single-dimension persona comparisons**

* Education: College vs High school
* Age: Young vs Old
* Geography: Urban vs Suburban, Urban vs Rural, Suburban vs Rural
* Socioeconomic: High vs Middle, High vs Low, Middle vs Low
* ESL: Native vs ESL
* Gender: Male vs Female

**Group B: Model comparisons**

* gpt-4o-mini vs gpt-5-mini (overall bias)
* gpt-4o-mini vs gpt-5-mini (within each severity level)

**Group C: Severity comparisons (monetary_severity, known only)**

* ≤$100 vs $100-$1000
* ≤$100 vs >$1000
* $100-$1000 vs >$1000
* Known vs Unknown (separate analysis)

**Group D: Severity comparisons (issue_severity)**

* Low vs Medium
* Low vs High
* Medium vs High

**Group E: Pre/post neutralization**

* Overall pre vs post
* Pre vs post within each persona
* Pre vs post within each severity level

**Group F: Severity amplification ratios**

* Amplification ratio for monetary_severity
* Amplification ratio for issue_severity

**Group G: Interaction contrasts**

* Persona effect at Low severity vs Persona effect at High severity
* Model difference at Low severity vs Model difference at High severity

### 2.3 Algorithm

**For mean differences (Groups A, B, C, D, E):**

```
Input: group1_data (array), group2_data (array)
Parameters: B = 10000, ci_level = 0.95

1. Compute point_estimate = mean(group1_data) - mean(group2_data)
2. Initialize bootstrap_diffs = empty array of size B
3. For b = 1 to B:
   a. sample1 = random sample with replacement from group1_data, size = len(group1_data)
   b. sample2 = random sample with replacement from group2_data, size = len(group2_data)
   c. bootstrap_diffs[b] = mean(sample1) - mean(sample2)
4. Sort bootstrap_diffs
5. alpha = 1 - ci_level
6. ci_lower = percentile(bootstrap_diffs, 100 * alpha/2)
7. ci_upper = percentile(bootstrap_diffs, 100 * (1 - alpha/2))
8. se_bootstrap = std(bootstrap_diffs)
9. significant = (ci_lower > 0) OR (ci_upper < 0)  # CI excludes zero

Output: {point_estimate, ci_lower, ci_upper, se_bootstrap, significant}
```

**For proportion differences (info request rates):**

```
Input: group1_successes, group1_n, group2_successes, group2_n
Parameters: B = 10000, ci_level = 0.95

1. p1 = group1_successes / group1_n
2. p2 = group2_successes / group2_n
3. point_estimate = p1 - p2
4. Initialize bootstrap_diffs = empty array of size B
5. For b = 1 to B:
   a. sample1 = binomial(n=group1_n, p=p1) / group1_n
   b. sample2 = binomial(n=group2_n, p=p2) / group2_n
   c. bootstrap_diffs[b] = sample1 - sample2
6. Continue as above for CI calculation

Output: {point_estimate, ci_lower, ci_upper, se_bootstrap, significant}
```

**For amplification ratios (Group F):**

```
Input: bias_low (array of persona means at low severity), 
       bias_high (array of persona means at high severity)
Parameters: B = 10000, ci_level = 0.95

1. Compute point_estimate:
   sd_low = std(bias_low)
   sd_high = std(bias_high)
   point_estimate = sd_high / sd_low

2. For b = 1 to B:
   a. Resample bias_low with replacement → boot_low
   b. Resample bias_high with replacement → boot_high
   c. bootstrap_ratios[b] = std(boot_high) / std(boot_low)
   
3. Handle edge cases:
   - If std(boot_low) == 0, set ratio to NaN for that iteration
   - Remove NaN values before percentile calculation
   
4. Compute CI from bootstrap_ratios

Output: {point_estimate, ci_lower, ci_upper, significant_amplification}
Where significant_amplification = ci_lower > 1.0
```

### 2.4 Metrics to Calculate

**Complete list of comparisons requiring bootstrap CIs:**

| ID                      | Comparison                 | Outcome           | Type      |
| ----------------------- | -------------------------- | ----------------- | --------- |
| edu_tier                | College vs High school     | remedy_tier       | mean_diff |
| edu_fav                 | College vs High school     | favorable_rate    | prop_diff |
| edu_info                | College vs High school     | info_request_rate | prop_diff |
| age_tier                | Young vs Old               | remedy_tier       | mean_diff |
| age_fav                 | Young vs Old               | favorable_rate    | prop_diff |
| age_info                | Young vs Old               | info_request_rate | prop_diff |
| geo_urban_suburban_tier | Urban vs Suburban          | remedy_tier       | mean_diff |
| geo_urban_rural_tier    | Urban vs Rural             | remedy_tier       | mean_diff |
| geo_suburban_rural_tier | Suburban vs Rural          | remedy_tier       | mean_diff |
| geo_urban_suburban_fav  | Urban vs Suburban          | favorable_rate    | prop_diff |
| geo_urban_rural_fav     | Urban vs Rural             | favorable_rate    | prop_diff |
| geo_suburban_rural_fav  | Suburban vs Rural          | favorable_rate    | prop_diff |
| ses_high_mid_tier       | High vs Middle             | remedy_tier       | mean_diff |
| ses_high_low_tier       | High vs Low                | remedy_tier       | mean_diff |
| ses_mid_low_tier        | Middle vs Low              | remedy_tier       | mean_diff |
| esl_tier                | Native vs ESL              | remedy_tier       | mean_diff |
| esl_fav                 | Native vs ESL              | favorable_rate    | prop_diff |
| esl_info                | Native vs ESL              | info_request_rate | prop_diff |
| gender_tier             | Male vs Female             | remedy_tier       | mean_diff |
| gender_fav              | Male vs Female             | favorable_rate    | prop_diff |
| gender_info             | Male vs Female             | info_request_rate | prop_diff |
| model_tier              | gpt-4o-mini vs gpt-5-mini  | remedy_tier       | mean_diff |
| model_bias              | gpt-4o-mini vs gpt-5-mini  | bias_magnitude    | mean_diff |
| mon_sev_1v2             | ≤$100 vs $100-$1000       | remedy_tier       | mean_diff |
| mon_sev_1v3             | ≤$100 vs >$1000           | remedy_tier       | mean_diff |
| mon_sev_2v3             | $100-$1000 vs >$1000       | remedy_tier       | mean_diff |
| mon_sev_known_unk       | Known vs Unknown           | remedy_tier       | mean_diff |
| iss_sev_lvm             | Low vs Medium              | remedy_tier       | mean_diff |
| iss_sev_lvh             | Low vs High                | remedy_tier       | mean_diff |
| iss_sev_mvh             | Medium vs High             | remedy_tier       | mean_diff |
| neutral_overall         | Pre vs Post neutralization | remedy_tier       | mean_diff |
| amp_monetary            | Amplification ratio        | monetary_severity | ratio     |
| amp_issue               | Amplification ratio        | issue_severity    | ratio     |

**Stratified comparisons (run for each model separately):**

* All persona comparisons above × 2 models = additional rows
* All severity comparisons above × 2 models = additional rows

**Stratified by severity (run for each severity level):**

* All persona comparisons × 3 monetary levels (known) = additional rows
* All persona comparisons × 3 issue levels = additional rows

### 2.5 Special Handling for monetary_severity

* Exclude "unknown" category from primary severity analyses
* Run separate analysis: "Known severity vs Unknown severity" to test whether unknown cases differ systematically
* Report sample sizes for known vs unknown
* If unknown > 30% of data, add warning to dashboard

### 2.6 Dashboard Additions

**Modification to all existing tables:**

Current format:

```
| Comparison | Mean Difference | p-value |
```

New format:

```
| Comparison | Difference | 95% CI | Significant | Effect Size |
| College vs HS | 0.036 | [0.028, 0.044] | Yes | d = 0.24 |
```

**CI visualization on charts:**

For bar charts:

* Add error bars showing 95% CI
* Error bar style: capped lines, width = 0.2 × bar width

For line charts:

* Add shaded bands showing 95% CI
* Band style: semi-transparent fill (alpha = 0.2)

**New "Confidence Interval Summary" panel:**

Collapsible section containing table with columns:

* Comparison ID
* Comparison Description
* Point Estimate
* 95% CI Lower
* 95% CI Upper
* CI Width
* Significant (Yes/No)
* Sample Sizes (n1, n2)

Sortable by any column. Default sort: by CI width (narrowest first = most precise).

**Filter options:**

* Show only significant comparisons
* Filter by comparison type (persona, model, severity, neutralization)
* Filter by outcome (tier, favorable rate, info request)

### 2.7 Output File: bootstrap_results.json

```json
{
  "parameters": {
    "n_bootstrap": 10000,
    "ci_level": 0.95,
    "random_seed": 42
  },
  "comparisons": {
    "edu_tier_overall": {
      "description": "College vs High school - remedy tier",
      "group1": "college",
      "group2": "high_school",
      "outcome": "remedy_tier",
      "n1": X,
      "n2": X,
      "point_estimate": 0.036,
      "ci_lower": 0.028,
      "ci_upper": 0.044,
      "ci_width": 0.016,
      "se_bootstrap": 0.004,
      "significant": true
    },
    "edu_tier_gpt4omini": {
      "description": "College vs High school - remedy tier (gpt-4o-mini only)",
      // same structure, stratified by model
    },
    "edu_tier_monetary_low": {
      "description": "College vs High school - remedy tier (monetary severity ≤$100)",
      // same structure, stratified by severity
    },
    // ... all other comparisons
    "amp_monetary": {
      "description": "Severity amplification ratio - monetary",
      "type": "ratio",
      "numerator": "bias at >$1000",
      "denominator": "bias at ≤$100",
      "point_estimate": 2.34,
      "ci_lower": 1.87,
      "ci_upper": 2.91,
      "significant_amplification": true
    },
    "amp_issue": {
      "description": "Severity amplification ratio - issue",
      // same structure
    }
  },
  "summary": {
    "total_comparisons": X,
    "significant_comparisons": X,
    "percent_significant": X,
    "median_ci_width": X
  },
  "timestamp": "..."
}
```

---

## 3. Effect Size Reporting

### 3.1 Purpose

Provide standardized measures of practical significance that are comparable across studies and independent of sample size. Enable readers to assess whether statistically significant effects are meaningful in practice.

### 3.2 Where to Apply

* Every pairwise comparison in bootstrap analysis
* Every ANOVA main effect and interaction
* Executive summary findings
* All tables displaying group comparisons

### 3.3 Algorithms

**Cohen's d (for continuous outcomes: remedy_tier)**

```
Input: group1_data, group2_data

1. n1 = len(group1_data)
2. n2 = len(group2_data)
3. mean1 = mean(group1_data)
4. mean2 = mean(group2_data)
5. var1 = variance(group1_data, ddof=1)  # sample variance
6. var2 = variance(group2_data, ddof=1)
7. pooled_sd = sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
8. d = (mean1 - mean2) / pooled_sd

Output: d

Interpretation:
  |d| < 0.20: negligible
  0.20 ≤ |d| < 0.50: small
  0.50 ≤ |d| < 0.80: medium
  |d| ≥ 0.80: large
```

**Cohen's h (for proportions: favorable_rate, info_request_rate)**

```
Input: p1, p2 (proportions between 0 and 1)

1. phi1 = 2 * arcsin(sqrt(p1))
2. phi2 = 2 * arcsin(sqrt(p2))
3. h = phi1 - phi2

Output: h

Interpretation:
  |h| < 0.20: negligible
  0.20 ≤ |h| < 0.50: small
  0.50 ≤ |h| < 0.80: medium
  |h| ≥ 0.80: large
```

**Partial eta-squared (for ANOVA effects)**

```
Input: SS_effect (sum of squares for effect), SS_error (sum of squares for error)

1. eta_sq_partial = SS_effect / (SS_effect + SS_error)

Output: eta_sq_partial

Interpretation:
  η²p < 0.01: negligible
  0.01 ≤ η²p < 0.06: small
  0.06 ≤ η²p < 0.14: medium
  η²p ≥ 0.14: large
```

**Bootstrap CI for effect sizes**

```
Input: group1_data, group2_data, effect_type ("d" or "h")
Parameters: B = 10000, ci_level = 0.95

1. Compute point_estimate using appropriate formula above
2. For b = 1 to B:
   a. Resample both groups with replacement
   b. Compute effect size on resampled data
   c. Store in bootstrap_effects[b]
3. Compute CI from percentiles

Output: {point_estimate, ci_lower, ci_upper}
```

### 3.4 Metrics to Calculate

**For each pairwise comparison:**

| Comparison               | Outcome           | Effect Size Measure   |
| ------------------------ | ----------------- | --------------------- |
| All persona comparisons  | remedy_tier       | Cohen's d with 95% CI |
| All persona comparisons  | favorable_rate    | Cohen's h with 95% CI |
| All persona comparisons  | info_request_rate | Cohen's h with 95% CI |
| All severity comparisons | remedy_tier       | Cohen's d with 95% CI |
| All model comparisons    | remedy_tier       | Cohen's d with 95% CI |
| Pre/post neutralization  | remedy_tier       | Cohen's d with 95% CI |

**For ANOVA effects:**

| Effect                                   | Partial η² |
| ---------------------------------------- | ------------ |
| Persona main effect                      |              |
| monetary_severity main effect            |              |
| issue_severity main effect               |              |
| Model main effect                        |              |
| Persona × monetary_severity interaction |              |
| Persona × issue_severity interaction    |              |
| Model × Persona interaction             |              |
| Model × monetary_severity interaction   |              |
| Model × issue_severity interaction      |              |
| Model × Persona × monetary_severity    |              |
| Model × Persona × issue_severity       |              |

**Stratified effect sizes:**

Calculate Cohen's d for each persona comparison separately:

* Within each model (2 models)
* Within each monetary_severity level (3 known levels)
* Within each issue_severity level (3 levels)

Total: Each persona comparison × (2 + 3 + 3) = 8 stratified effect sizes

### 3.5 Special Handling for Severity

**Severity-stratified effect sizes:**

For each persona comparison, calculate d at each severity level:

```
| Comparison | d (≤$100) | d ($100-$1000) | d (>$1000) | d (Low) | d (Medium) | d (High) |
| College vs HS | 0.18 | 0.24 | 0.52 | 0.15 | 0.22 | 0.48 |
```

**Amplification effect size:**

Calculate the change in effect size from low to high severity:

```
effect_amplification = d_high_severity - d_low_severity

If effect_amplification > 0: bias effect increases with severity
If effect_amplification < 0: bias effect decreases with severity
```

Report amplification for both monetary and issue severity.

### 3.6 Dashboard Additions

**Modification to comparison tables:**

Add columns:

* Effect Size: numerical value (d or h)
* Effect Category: text label (negligible/small/medium/large)
* Effect CI: confidence interval for effect size

**Color coding for effect size cells:**

| Category   | Color      | Hex Code |
| ---------- | ---------- | -------- |
| Negligible | Light grey | #E0E0E0  |
| Small      | Yellow     | #FFF9C4  |
| Medium     | Orange     | #FFCC80  |
| Large      | Red        | #EF9A9A  |

**New "Effect Size Distribution" panel:**

Contains:

1. **Summary statistics:**
   * Total comparisons: X
   * Negligible: X (Y%)
   * Small: X (Y%)
   * Medium: X (Y%)
   * Large: X (Y%)
2. **Histogram:**
   * X-axis: Effect size (d or h)
   * Y-axis: Count of comparisons
   * Vertical lines at 0.2, 0.5, 0.8 thresholds
   * Color bands matching category colors
3. **Key insight highlight:**
   * "[X]% of persona comparisons show medium or large effects"
   * "[X]% of severity-stratified comparisons show amplification"

**New "Effect Size by Severity" panel:**

Heatmap showing effect sizes across severity levels:

```
                | ≤$100 | $100-$1K | >$1000 | Low | Medium | High |
College vs HS   |  0.18 |   0.24   |  0.52  | 0.15|  0.22  | 0.48 |
Young vs Old    |  0.12 |   0.19   |  0.41  | 0.10|  0.18  | 0.39 |
...
```

Color intensity reflects effect size magnitude.

**ANOVA Effect Size panel:**

Table showing partial η² for all main effects and interactions:

```
| Effect | Partial η² | Category | F | p-value |
| Persona | 0.08 | Medium | 12.4 | <0.001 |
| monetary_severity | 0.03 | Small | 8.2 | <0.001 |
| Persona × monetary | 0.05 | Small | 3.1 | 0.002 |
```

### 3.7 Interpretation Guide (Dashboard Component)

**Collapsible section: "How to Interpret Effect Sizes"**

Content:

```
EFFECT SIZE INTERPRETATION GUIDE

What are effect sizes?
Effect sizes measure the magnitude of a difference, independent of sample size. 
A statistically significant result (p < 0.05) tells you a difference exists; 
the effect size tells you whether that difference matters in practice.

Cohen's d (for remedy tier differences):
- Measures the difference between two group means in standard deviation units
- d = 0.50 means the groups differ by half a standard deviation
- Benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large

Cohen's h (for rate differences):
- Measures the difference between two proportions
- Similar interpretation to Cohen's d
- Benchmarks: 0.2 = small, 0.5 = medium, 0.8 = large

Partial η² (for ANOVA effects):
- Measures the proportion of variance explained by an effect
- η² = 0.06 means the effect explains 6% of the variance
- Benchmarks: 0.01 = small, 0.06 = medium, 0.14 = large

Why does this matter?
With large samples (like this study), even tiny differences can be statistically 
significant. Effect sizes help distinguish:
- Trivial effects that are significant only due to sample size
- Meaningful effects with practical implications

In this context:
- Small effects (d ≈ 0.2): May indicate systematic bias worth monitoring
- Medium effects (d ≈ 0.5): Meaningful disparity requiring attention
- Large effects (d ≈ 0.8): Severe discrimination with clear harm
```

### 3.8 Output File: effect_sizes.json

```json
{
  "pairwise_comparisons": {
    "edu_tier_overall": {
      "description": "College vs High school - remedy tier",
      "effect_type": "cohens_d",
      "point_estimate": 0.24,
      "ci_lower": 0.18,
      "ci_upper": 0.30,
      "category": "small",
      "n1": X,
      "n2": X
    },
    "edu_tier_monetary_low": {
      "description": "College vs High school - remedy tier (≤$100)",
      "effect_type": "cohens_d",
      "point_estimate": 0.18,
      "ci_lower": 0.10,
      "ci_upper": 0.26,
      "category": "negligible",
      "severity_metric": "monetary",
      "severity_level": "<=100"
    },
    "edu_tier_monetary_high": {
      "description": "College vs High school - remedy tier (>$1000)",
      "effect_type": "cohens_d",
      "point_estimate": 0.52,
      "ci_lower": 0.38,
      "ci_upper": 0.66,
      "category": "medium",
      "severity_metric": "monetary",
      "severity_level": ">1000"
    },
    // ... similar for all comparisons
  },
  "anova_effects": {
    "persona_main": {
      "partial_eta_sq": 0.08,
      "category": "medium",
      "F": 12.4,
      "p_value": 0.0001,
      "df_effect": 35,
      "df_error": 10000
    },
    "monetary_severity_main": {
      "partial_eta_sq": 0.03,
      "category": "small",
      // ...
    },
    "issue_severity_main": {
      // ...
    },
    "persona_x_monetary": {
      "partial_eta_sq": 0.05,
      "category": "small",
      // ...
    },
    "persona_x_issue": {
      // ...
    },
    "model_x_persona_x_monetary": {
      // ...
    },
    "model_x_persona_x_issue": {
      // ...
    }
  },
  "amplification": {
    "monetary": {
      "d_at_low": 0.18,
      "d_at_high": 0.52,
      "amplification": 0.34,
      "interpretation": "Effect size nearly triples from lowest to highest monetary severity"
    },
    "issue": {
      "d_at_low": 0.15,
      "d_at_high": 0.48,
      "amplification": 0.33,
      "interpretation": "Effect size more than triples from low to high issue severity"
    }
  },
  "summary": {
    "total_pairwise": X,
    "by_category": {
      "negligible": X,
      "small": X,
      "medium": X,
      "large": X
    },
    "percent_medium_or_large": X,
    "mean_effect_size": X,
    "max_effect_size": X,
    "comparison_with_max": "description of comparison"
  },
  "timestamp": "..."
}
```

---

## 4. Integration Across Specifications

### 4.1 Unified Analysis Pipeline

The three statistical components should be run in sequence:

1. **Power analysis** (run once, at start)
   * Determines whether sample sizes are adequate
   * Flags underpowered analyses before they are run
2. **Bootstrap CIs** (run for each comparison)
   * Provides CIs for all point estimates
   * Determines statistical significance
3. **Effect sizes** (run alongside bootstrap)
   * Computed for same comparisons as bootstrap
   * Stored together in output

### 4.2 Handling the "Unknown" Monetary Severity

Across all three specification areas:

1. **Primary analyses:** Exclude "unknown" category
   * Power analysis based on known-severity subset
   * Bootstrap CIs computed on known-severity subset
   * Effect sizes computed on known-severity subset
2. **Secondary analyses:** Test known vs unknown
   * Single comparison: known severity vs unknown severity
   * Bootstrap CI and effect size for this comparison
   * Interpretation: Does excluding unknown bias results?
3. **Reporting:**
   * Always report coverage rate (% with known monetary severity)
   * If coverage < 70%, add prominent warning
   * If known vs unknown shows significant difference (d > 0.2), discuss implications

### 4.3 Dashboard Summary Panel

**New "Statistical Summary" panel (prominent placement):**

```
STATISTICAL RIGOR SUMMARY

Power Analysis:
✓ Adequate power (>80%) for main effects and two-way interactions
⚠ Limited power (65%) for three-way interactions

Confidence Intervals:
- 147 pairwise comparisons computed
- 89 (61%) show significant differences
- Median CI width: 0.032 (precise estimates)

Effect Sizes:
- 12% of comparisons show medium or large effects
- Largest effect: Urban vs Rural at high severity (d = 0.72)
- Severity amplification confirmed: effects 2.3× larger at high vs low severity

Severity Coverage:
- monetary_severity: 68% known (warning: 32% unknown)
- issue_severity: 100% coverage
```

### 4.4 Reporting Template for Paper

**Methods section (~200 words):**

"Statistical power was assessed using F-test power analysis for factorial ANOVA designs. With [N] observations distributed across [X] cells, the design achieves [Y]% power to detect small-to-medium interaction effects (f = 0.10) at α = .05.

All pairwise comparisons report 95% bootstrap confidence intervals based on 10,000 resamples, providing distribution-free inference. Statistical significance was determined by whether confidence intervals excluded zero.

Effect sizes are reported throughout using Cohen's d for continuous outcomes and Cohen's h for proportions, with interpretation following conventional benchmarks (small: 0.2, medium: 0.5, large: 0.8). For ANOVA effects, partial eta-squared (η²p) quantifies variance explained.

Two severity metrics were employed: monetary severity based on disputed dollar amounts (≤$100, $100-$1000, >$1000) and issue severity based on product category and complaint type (low, medium, high). Monetary severity could be determined for [X]% of complaints; analyses are reported for the known-severity subset with sensitivity checks comparing known versus unknown cases."
