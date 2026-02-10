# Step 7: Remedy Decision (Neutralised)

## Overview

Step 7 assigns remedy tiers to the neutralised complaints from Step 6. By comparing these decisions to the baseline decisions from Step 2 (same complaints, original text), we measure whether **language neutralisation reduces discrimination**. This is the key mitigation test: if neutralisation eliminates persona-based outcome differences, it validates the bias mitigation strategy.

**Source:** `src/experiment.py` → `_step_7()` (lines 1391–1442), `_summarize_step_7()` (lines 1444–1476+)
**Statistics:** `src/statistics.py` → `mitigation_analysis()`, `src/process_discrimination.py` → `process_mitigation_analysis()`

---

## What This Step Does

1. **Applies the same remedy decision** prompt to each neutralised complaint (identical to Steps 2 and 5)
2. **Collects tier assignments** for all neutralised complaints
3. **Extracts text features** from neutralised complaints (26 metrics)
4. **Compares decisions** to baseline (Step 2) to measure mitigation effectiveness
5. **Tests residual bias** — whether any persona-based disparities remain after neutralisation

---

## Execution

### Request Construction

For each model and each neutralised complaint:

```python
req = LLMRequest(
    request_id=LLMRequest.new_id(),
    case_id=nc.case_id,
    model=model,
    task="remedy_decision",
    prompt_system=REMEDY_DECISION_SYSTEM,
    prompt_user=REMEDY_DECISION_USER.format(narrative=nc.neutralized_narrative),
    expected_schema=RemedyDecisionResponse,
    variant="neutralized",
)
```

The prompt and schema are **identical** to Steps 2 and 5. The variant is set to `"neutralized"` to distinguish these decisions in the analysis.

### Result Storage

```python
RemedyDecision(
    case_id=resp.case_id,
    model=resp.model,
    variant="neutralized",
    tier=p.tier,
    confidence=p.confidence,
    reasoning=p.reasoning,
    information_needed=p.information_needed,
)
```

### Text Feature Extraction

After collecting decisions, Step 7 extracts 26 text features from all neutralised complaints:

```python
neut_features = extract_features_batch(
    [n.neutralized_narrative for n in self.neutralized],
    [n.case_id for n in self.neutralized],
    "neutralized",
)
```

These features are appended to `self.text_features_df` for cross-variant analysis.

---

## Console Summary

`_summarize_step_7()` merges neutralised decisions with the original persona estimates and baseline decisions, then logs per model:

1. **Persona × neutralised tier crosstab** — top 10 personas, showing whether tier assignments vary by original persona even after neutralisation
2. **Mean neutralised tier by dimension** — whether dimension-level patterns persist
3. **Comparison to baseline** — original vs. neutralised tier distributions
4. **Process discrimination** — `need_more_info` rates on neutralised text vs. original

---

## Mitigation Analysis

### Pre/Post McNemar Test

The most direct mitigation test compares baseline and neutralised decisions **per complaint**:

- **Paired comparison**: For each complaint, compare `tier(original)` vs. `tier(neutralised)`
- **McNemar's test**: Tests whether the proportion of complaints that change tier is symmetric
- **Interpretation**: If neutralisation is effective, we expect decisions to converge toward the same outcome regardless of original writing style

### Variance Reduction

For each persona, the information request rate is computed before and after neutralisation:

```python
orig_rates = orig_m.groupby("persona_label")["need_info"].mean()
neut_rates = neut_m.groupby("persona_label")["need_info"].mean()

variance_orig = orig_rates.var()
variance_neut = neut_rates.var()
variance_reduction = 1 - variance_neut / variance_orig
```

A positive variance reduction means that persona-based disparities in info request rates have decreased.

### AIR Improvement

The Adverse Impact Ratio is compared before and after neutralisation:

```python
for persona in orig_rates.index:
    orig_air = process_adverse_impact_ratio(orig_rates[persona], min_orig)
    neut_air = process_adverse_impact_ratio(neut_rates[persona], min_neut)
    bias_reduction = (orig_air - neut_air) / (orig_air - 1.0)
```

If the original AIR exceeded the 1.25 materiality threshold for process discrimination, a successful mitigation should bring it closer to 1.0.

### Residual Bias Analysis

After neutralisation, the full correlation analysis from Step 3 is re-run on the neutralised decisions to check for **residual bias**:

```python
results["residual_correlation"] = process_correlation_analysis(
    neutralized_decisions, personas_df
)
```

This tests whether any statistically significant persona-outcome associations remain after the mitigation intervention.

---

## Repeated-Measures Integration

Neutralised decisions are incorporated into the long-format dataset used by `src/repeated_measures.py`:

```python
# In build_long_format():
ndf = neutralized_df.copy()
ndf["condition"] = "neutralised"
ndf = ndf.merge(personas_df[["case_id", "model", "persona_label"]], ...)
```

This enables the `condition_code` predictor in all parametric models (Step 5) to estimate the neutralisation effect:

| Predictor | Interpretation |
|-----------|---------------|
| `condition_code` | Main effect of neutralisation on outcome |
| `condition_code × persona_score` | Whether neutralisation reduces persona bias |
| `condition_code × sev_ord` | Whether neutralisation interacts with severity |

A significant negative coefficient on `condition_code × persona_score` would indicate that neutralisation successfully attenuates persona-based disparities.

---

## Process Discrimination Mitigation

`process_mitigation_analysis()` in `src/process_discrimination.py` (lines 235–299) specifically tests whether neutralisation reduces procedural burden disparities:

| Metric | Method | Interpretation |
|--------|--------|----------------|
| Pre/post McNemar | `mcnemar_test()` | Are info request rates different before/after? |
| Variance reduction | `var(orig_rates) vs var(neut_rates)` | Do persona-based rate differences narrow? |
| AIR improvement | Per-persona process AIR comparison | Do specific personas see reduced burden? |
| Residual correlation | Re-run process correlation analysis | Any remaining persona-burden associations? |

---

## Interpreting Mitigation Effectiveness

The mitigation is assessed on three criteria:

### 1. Outcome Equalisation
Do tier assignments become more uniform across personas after neutralisation?
- **Metric**: Variance of mean tier across personas
- **Success**: Lower variance in neutralised decisions

### 2. Disparate Impact Reduction
Do Adverse Impact Ratios improve toward 1.0?
- **Metric**: AIR change per persona
- **Success**: Fewer personas below the 0.80 threshold (outcome) or above 1.25 (process)

### 3. Statistical Significance
Do previously significant persona-outcome associations become non-significant?
- **Metric**: Chi-squared p-values and Cramer's V
- **Success**: p > 0.05 or smaller effect sizes after neutralisation

### Limitations

- Neutralisation may affect overall tier distribution (e.g., more neutral language may be perceived as less urgent), so the comparison is between **persona-based disparity** rather than absolute tier levels
- If the LLM assigns a "default persona" to all neutralised text, the comparison is to a single reference style rather than true persona-blindness
- Partial mitigation (reduced but not eliminated bias) is a valid and useful finding

---

## Relationship to Other Steps

| Step | Relationship |
|------|-------------|
| Step 2 | Baseline decisions — the "before" in pre/post comparison |
| Step 5 | Established causal bias that mitigation targets |
| Step 6 | Produced the neutralised text used here |
| Step 8 | Tests whether mitigation effectiveness varies by severity |
| Step 9 | Tests mitigation of process discrimination specifically |
