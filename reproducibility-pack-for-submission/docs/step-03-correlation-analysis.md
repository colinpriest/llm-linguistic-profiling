# Step 3: Correlation Analysis

## Overview

Step 3 tests whether the persona estimates from Step 1 correlate with the remedy outcomes from Step 2. This is an **observational analysis** — it establishes association, not causation. Causal testing comes in Steps 4–5.

**Source:** `src/experiment.py` → `_step_3()` (lines 855–870), `_summarize_step_3()` (lines 872–1022)
**Statistics:** `src/statistics.py` → `correlation_analysis()` (lines 721–786)

---

## What This Step Does

1. **Builds DataFrames** from the collected persona estimates and baseline decisions
2. **Extracts text features** (26 metrics) from original complaint narratives
3. **Runs statistical tests** to detect persona-outcome associations
4. **Generates GPT-5.2 commentary** comparing patterns across models

---

## Text Feature Extraction

Before running statistical tests, Step 3 extracts 26 text metrics from all original complaints using `src/text_features.py`:

| Category | Metrics | Library |
|----------|---------|---------|
| Length (5) | word_count, char_count, sentence_count, avg_word_length, avg_sentence_length | nltk |
| Readability (5) | flesch_reading_ease, flesch_kincaid_grade, gunning_fog, smog_index, ari | textstat |
| Vocabulary (4) | type_token_ratio, hapax_ratio, mtld, hdd | lexicalrichness |
| Sentiment (7) | polarity, subjectivity, vader_compound, vader_pos, vader_neg, vader_neu, emotional_intensity | TextBlob, VADER |
| Topics (3+10) | dominant_topic, topic_coherence, topic_distribution (10 dims via LDA) | scikit-learn |

These features are stored in `self.text_features_df` and used throughout later analyses.

---

## Statistical Tests

The `correlation_analysis()` function in `src/statistics.py` runs five families of tests:

### 1. Overall Chi-Squared Test

Tests independence in the `persona_label × tier` contingency table.

- **Function:** `chi_squared_test()` (lines 302–311)
- **Returns:** chi2 statistic, p-value, degrees of freedom, Cramer's V, interpretation
- **Interpretation:** Cramer's V is adjusted for degrees of freedom:
  - V < 0.10: negligible
  - V < 0.30: small
  - V < 0.50: medium
  - V ≥ 0.50: large

### 2. Per-Dimension Chi-Squared Tests

Six separate tests — one for each dimension × tier:

| Dimension | Test |
|-----------|------|
| age × tier | Chi-squared with Cramer's V |
| geographic × tier | Chi-squared with Cramer's V |
| education × tier | Chi-squared with Cramer's V |
| socioeconomic × tier | Chi-squared with Cramer's V |
| gender × tier | Chi-squared with Cramer's V |
| esl × tier | Chi-squared with Cramer's V |

### 3. Logistic Regression

Models the probability of a favourable outcome (`tier >= 1`) as a function of persona dimensions:

```
logit P(favourable) = β₀ + β₁·age + β₂·geographic + β₃·education + β₄·socioeconomic + β₅·gender + β₆·esl
```

- **Function:** `logistic_regression_analysis()` (lines 413–564)
- **Encoding:** Categorical dimensions are one-hot encoded (reference category dropped)
- **Returns:** coefficients, p-values, odds ratios, pseudo R², AIC, method used

#### Fallback Chain for Convergence

The logistic regression uses a 4-step fallback chain to handle problematic data:

| Step | Condition | Method |
|------|-----------|--------|
| 1 | Standard MLE | `sm.Logit().fit()` |
| 2 | Rank deficiency | Drop aliased columns, re-fit |
| 3 | Separation | **Firth penalised logit** — bias-reduced via Jeffreys prior (lines 567–648) |
| 4 | Convergence failure | Standardise predictors + **Ridge (L2) logit** (lines 651–699) |

The Firth logit is particularly important: when one persona combination perfectly predicts the outcome (complete separation), standard MLE gives infinite coefficients. Firth's method adds a small bias correction that produces finite estimates.

### 4. Adverse Impact Ratios (AIR)

Computes the ratio of favourable outcome rates between each persona and the best-performing persona:

```
AIR(persona) = rate(persona) / rate(best_persona)
```

- **Function:** `_compute_air_by_persona()` (lines 789–803)
- **80% rule:** AIR < 0.80 indicates material adverse impact
- **Returns:** per-persona rate, AIR value, and materiality flag

### 5. Per-Dimension Favourable Rates

Computes the favourable outcome rate (`tier >= 1`) for each level of each dimension:

```
rate(age=young) = count(tier >= 1 & age == young) / count(age == young)
```

- **Function:** `_rates_by_dimension()` (lines 806–814)
- **Returns:** `{dimension: {level: favourable_rate}}`

---

## Multiple Comparison Correction

All p-values are corrected within test families using **Benjamini-Hochberg FDR correction**:

- **Function:** `fdr_correction()` (lines 38–48)
- Applied within:
  - Per-dimension tests (6 tests)
  - Per-persona AIR tests (up to 36 tests)

The correction prevents false positives from inflating significance when many tests are run simultaneously.

---

## Cross-Model Commentary

When multiple models are tested, Step 3 generates **GPT-5.2 commentary** comparing persona-tier associations across models.

### Inputs to Commentary

Per model, the following statistics are gathered:

- Record count and unique persona count
- **Persona entropy** — Shannon entropy of the persona distribution (higher = more diverse assignments)
- **Overall chi-squared** — persona × tier (chi2, p, Cramer's V)
- **Per-dimension chi-squared** — each dimension × tier
- **Mean tier** by dimension level

### Commentary Prompt

The system prompt instructs GPT-5.2 to write a 4–8 sentence paragraph as a statistician, reporting:

1. Whether the overall persona × tier chi-squared is significant for each model (with p-values)
2. Which dimensions have the largest Cramer's V and whether they reach p < 0.05
3. Which model has more unique personas / higher entropy (less anchoring)
4. Which model shows stronger persona-tier associations (compare V values)

Every sentence must contain at least one specific number from the data.

### Critical Instruction

> Your output must read like a finished paragraph from a statistics report, e.g. 'The analysis revealed that...' NOT like a to-do list or plan such as 'Compute per-model significance...'

---

## Console Summary

The Step 3 console summary reports per model:

1. **2-way crosstab** — `persona_label × tier` (top 10 personas by count)
2. **Mean tier by dimension** — average tier for each level of all 6 dimensions
3. **Process discrimination rate** — count of `confidence == "need_more_info"`
4. **Association tests** — overall and per-dimension chi-squared with Cramer's V
5. **Persona diversity metrics** — unique persona count and Shannon entropy

---

## Interpretation

Step 3 results should be interpreted with caution:

- **Correlation ≠ causation**: Persona-outcome associations may reflect genuine complaint differences (complaints from lower-SES writers may describe less severe bank errors) rather than LLM bias
- **Confounding**: Writing style correlates with complaint content, product type, and severity — all of which legitimately affect remedy decisions
- **Purpose**: Step 3 identifies which dimensions are associated with outcomes and quantifies the association strength, motivating the causal test in Steps 4–5

The matched-guise experiment (Steps 4–5) is required to isolate writing style as the causal variable, controlling for complaint content by rewriting the same complaint in different persona styles.
