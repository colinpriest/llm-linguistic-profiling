ONE-SHOT EXAMPLE "CONCLUSION"

input:
# Does the LLM Infer Customer Personas from Writing Style?

## What We Tested
We asked ChatGPT to estimate the demographic profile of **5,000 customers** based solely on the text of their complaints. The goal was to establish whether the LLM infers implicit personas from how complaints are written — a necessary first step before testing whether those inferences affect decisions.

## Key Finding: The LLM Infers Personas from Writing Style
When analysing complaint text, the LLM consistently infers demographic characteristics. This confirms the model responds to stylistic cues, which is relevant background for testing decision bias.

| Characteristic | Most Common Inference |
|---|---|
| Age | Older (68%) |
| Location | Suburban (72%) |
| Education | College-educated (72%) |
| Income | Middle class (92%) |
| Gender | Male (73%) |
| Language background | Native English speaker (90%) |

The single most common profile — **older, suburban, college-educated, middle-class male** — accounted for **45%** of all estimates.

# Do Remedy Decisions Correlate with Inferred Persona?

## What We Tested
After establishing that ChatGPT infers customer demographics from complaint text, we tested the critical question: **Do those inferences correlate with the remedies it recommends?**

We compared two versions of ChatGPT (**GPT-4o-mini** and **GPT-5-mini**) across **5,000 complaints each**.

## Key Finding: Yes, Decisions Correlate with Inferred Persona
Both versions of ChatGPT show statistically significant correlations between inferred persona and recommended remedy. This suggests the LLM's implicit assumptions about customers may affect outcomes.

| Bias Dimension | GPT-4o-mini (Older) | GPT-5-mini (Newer) |
|---|---|---|
| Gender | Strong correlation | Not statistically significant |
| Language background (ESL) | Significant correlation | Not statistically significant |
| Geographic | Present | Present (largest remaining factor) |

The newer model shows meaningful improvement: gender and ESL correlations have been reduced to non-significant levels.

# Does Writing Style CAUSE Different Decisions? (The Causal Test)

## The Experiment
Correlation does not prove causation. To isolate whether ChatGPT's bias stems from **writing style alone**, we designed a rigorous test: we took the same underlying complaints and rewrote each one in **36 different ways** to mimic how various customer personas might express themselves.

This "matched-guise" approach lets us test whether the AI responds to **how something is said** rather than **what is said**. We tested **18,000 complaint-persona combinations** on each model.

## Key Finding: The Newer Model Shows Virtually No Causal Bias

### GPT-5-mini: Remarkably Consistent
When the same complaint was rewritten in different persona styles, the newer model recommended essentially identical remedies regardless of writing style:

| Characteristic | Variation in Remedy Quality |
|---|---|
| Age (old vs. young) | 0.0% difference |
| Education (college vs. high school) | 0.2% difference |
| Geography (rural/suburban/urban) | 1.0% spread |
| Income level | 0.8% spread |

### GPT-4o-mini: Some Causal Bias Detected
The older model showed more variation, particularly around education:

| Characteristic | Variation in Remedy Quality |
|---|---|
| Education (college vs. high school) | 4.9% difference |
| Income (high vs. low) | 4.1% difference |
| Age (old vs. young) | 2.5% difference |

# Does Severity Amplify Any Detected Bias?

## Key Finding: Different Patterns for Different Severity Metrics
We tested whether bias varies by complaint severity. **Important:** monetary severity (dollar amounts) and issue severity (product/issue category) show different patterns.

### Monetary Severity
- **GPT-4o-mini:** Bias nearly triples for high-dollar complaints (2.8x amplification)
- **GPT-5-mini:** No meaningful amplification (1.25x, not statistically significant)

### Issue Severity
- Neither model shows statistically significant amplification by issue severity

## Interpretation
Severity amplification is **metric-specific**: it appears for dollar amounts in the older model but not for issue categories or in the newer model.

# Does Neutralization Remove Demographic Signals?

## Key Finding: Neutralization Successfully Removes Most Signals
Rewriting complaints to neutral, professional language effectively removes demographic signals for:
- **Education:** 87-94% signal reduction
- **ESL status:** 92-99% signal reduction
- **Income level:** 72-82% signal reduction

Geographic signals remain partially detectable (9-15% reduction).

# Does Neutralization Eliminate Decision Bias?

## Key Finding: Results Depend on Which Model You Use

### GPT-5-mini: Neutralization Works
After neutralization:
- No statistically significant differences between personas
- Zero pairwise comparisons showed significant bias (0 out of 630)
- Outcome variance reduced by 5.4%

### GPT-4o-mini: Neutralization Fails
Despite neutralization:
- 147 out of 630 pairwise comparisons still showed significant bias
- Outcome variance increased by 2.2%
- Strong disparities remain across multiple dimensions

output:
## Overview
We conducted experiments to answer one key question: **Do LLMs make different remedy decisions based on linguistic style when those decisions should be identical?** We tested two versions of ChatGPT (GPT-4o-mini and GPT-5-mini) across thousands of complaints.

## Key Findings

### 1. Writing Style Bias Exists in the Older Model
The matched-guise experiment proves that **GPT-4o-mini changes decisions based on writing style alone**. When the same complaint was rewritten in different persona styles, remedy recommendations varied by up to 4.9% (education) and 4.1% (income level). This is causal bias: the complaint substance was identical, only the wording changed.

### 2. The Newer Model Shows No Detectable Bias
**GPT-5-mini treats identical complaints consistently** regardless of how they are written. When we varied only the writing style, remedy recommendations differed by less than 1% across all persona dimensions. This represents substantial improvement.

### 3. Severity Amplification Is Limited and Metric-Specific
- **Monetary severity (dollar amounts):** GPT-4o-mini shows 2.8x amplification of bias for high-dollar complaints. GPT-5-mini shows no amplification.
- **Issue severity (product/issue category):** Neither model shows amplification.

Do not assume "bigger dollar dispute = higher bias" — this pattern is specific to monetary amounts in the older model only.

### 4. Neutralization Works for the Newer Model Only
Rewriting complaints to neutral language before processing:
- **GPT-5-mini:** Eliminates detectable causal bias. After neutralization, decisions are uniform across all persona styles.
- **GPT-4o-mini:** Fails to eliminate bias. Significant disparities persist even after neutralization.

## Recommendations

1. **Use GPT-5-mini for complaint handling where fairness matters.** It shows no detectable bias based on writing style in our causal tests.

2. **Do not use GPT-4o-mini in bias-sensitive applications.** It makes systematically different decisions based on how complaints are written, and this bias persists even after language neutralization.

3. **For GPT-5-mini, neutralization is a validated mitigation approach.** Pre-processing complaints through a neutralization step provides an additional safeguard.

4. **Monitor high-dollar complaints if using older models.** Monetary severity amplifies bias specifically in GPT-4o-mini.

5. **Continue monitoring as models evolve.** Bias patterns can shift with updates.
