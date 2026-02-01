ONE-SHOT EXAMPLE "CONCLUSION"

input:
# Does ChatGPT Profile Customers? Executive Summary

## What We Tested
We ran an experiment asking ChatGPT to estimate the demographic profile of **5,000 customers** based solely on the text of their complaints. The goal was to understand whether—and how—the AI forms assumptions about who customers are from how they communicate.

## Key Finding: ChatGPT Has a “Default Customer” in Mind
When guessing demographics from complaint text, ChatGPT overwhelmingly imagines a very specific type of person.

| Characteristic | ChatGPT’s Dominant Assumption |
|---|---|
| Age | Older (68%) |
| Location | Suburban (72%) |
| Education | College-educated (72%) |
| Income | Middle class (92%) |
| Gender | Male (73%) |
| Language background | Native English speaker (90%) |

The single most common profile—**older, suburban, college-educated, middle-class male**—accounted for **45%** of all **5,000** estimates.

## What’s Missing from the AI’s Mental Picture
The model’s assumptions show stark blind spots:

- **Rural customers are nearly invisible:** Only **0.5%** of estimates (**26 out of 5,000**)
- **High-income customers essentially don’t exist:** Just **4** estimates total (**0.08%**)
- **Women are significantly underrepresented:** Despite likely comprising close to half of actual complainants
- **36** possible demographic combinations exist; only **23** ever appeared

## How Confident Is the AI in These Guesses?
Very confident—perhaps too confident. In **93%** of cases, ChatGPT reported **60–80% certainty** in its demographic estimates, suggesting the model readily forms firm assumptions about customers based on limited textual cues.

## Why This Matters
If ChatGPT is used anywhere in your complaint-handling process, these embedded assumptions could influence:

- How complaints are prioritized or routed
- The tone and formality of automated responses
- Risk assessments or escalation decisions
- Downstream analytics about your customer base

The AI appears to mentally “picture” a narrow slice of customers and may process complaints differently based on implicit profiling—even when no demographic information is provided.

## Recommended Actions
1. Audit any AI-assisted complaint workflows for unintended demographic assumptions
2. Avoid feeding demographic estimates into downstream decisions without validation
3. Consider whether these patterns reflect training data biases that could affect your customers

# Does Implicit Customer Profiling Affect Remedies? Executive Summary

## What We Tested
After discovering that ChatGPT forms assumptions about customer demographics from complaint text, we examined the critical follow-up question: **Do those assumptions influence the remedies it recommends?**

We compared two versions of ChatGPT (**the older GPT-4o-mini** and **the newer GPT-5-mini**) across **5,000 complaints each**.

## Bottom Line: Yes, Bias Exists—But It’s Improving
Both versions of ChatGPT show **statistically significant bias**, meaning the remedy recommended does depend on the customer profile the AI imagines. However, the newer model has made meaningful progress on some of the most concerning forms of bias.

## How the Two Models Compare

| Bias Dimension | GPT-4o-mini (Older) | GPT-5-mini (Newer) |
|---|---|---|
| Gender bias | Strong and significant | Not statistically significant |
| Language background (ESL) bias | Significant | Not statistically significant |
| Geographic bias | Present | Present (now the largest factor) |
| Overall bias strength | Moderate | Slightly higher overall, but more diffuse |

## Key Findings

### The Good News: Gender and Language Bias Reduced
The older model showed a pronounced tendency to recommend different remedies based on perceived gender—this was its strongest bias by a wide margin. The newer model has essentially eliminated this pattern to the point where it’s no longer statistically detectable.

Similarly, bias based on perceived English-as-a-second-language status has been reduced to non-significant levels.

### The Remaining Concern: Geographic Bias Persists
In the newer model, the strongest remaining bias relates to where the AI thinks the customer lives. Urban, suburban, and rural customers (as imagined by the AI) receive systematically different remedy recommendations.

### The Newer Model “Imagines” More Diverse Customers
GPT-5-mini shows less tendency to default to a single customer stereotype. It generated **23 distinct customer profiles** versus **21** for the older model, and its estimates were more evenly spread across those profiles—suggesting reduced anchoring to a “default customer” assumption.

## What This Means in Practice
Even without knowing anything about a customer’s actual demographics, these AI systems form assumptions from complaint text—and those assumptions influence outcomes. While the newer model shows meaningful improvement, the fundamental pattern remains: **imagined identity affects recommended remedy**.

## Recommended Actions
- If using **GPT-4o-mini** in complaint handling, prioritize upgrading—its gender bias is substantial
- Investigate geographic disparities in the newer model’s recommendations
- Consider masking or standardizing complaint text before AI processing to reduce profiling cues
- Monitor remedies by actual customer demographics to detect whether AI-assisted decisions create real-world disparate outcomes

# Does Writing Style Trigger AI Bias? A Controlled Experiment

## The Experiment
To isolate whether ChatGPT’s bias stems from **writing style alone**, we designed a rigorous test: we took the same underlying complaints and rewrote each one in **36 different ways** to mimic how various customer personas might express themselves.

For example, the same billing dispute might be written:

- In the formal, detailed style of an older, college-educated professional  
- In the casual, direct style of a younger urban customer  
- In the straightforward language of someone with a high school education  

This “matched-guise” approach—borrowed from linguistics research—lets us test whether the AI responds to **how something is said** rather than **what is said**. We tested **18,000 complaint–persona combinations** on each of two ChatGPT models.

## Key Finding: The Newer Model Shows Virtually No Bias

### GPT-5-mini: Remarkably Consistent
When the same complaint was rewritten in different persona styles, the newer model recommended essentially identical remedies regardless of writing style:

| Characteristic | Variation in Remedy Quality |
|---|---|
| Age (old vs. young) | 0.0% difference |
| Education (college vs. high school) | 0.2% difference |
| Geography (rural/suburban/urban) | 1.0% spread |
| Income level | 0.8% spread |

This is a strong result: the AI appears to evaluate the **substance** of complaints, not the perceived sophistication of the writer.

### GPT-4o-mini: Some Bias Detected
The older model showed more variation, particularly around education:

| Characteristic | Variation in Remedy Quality |
|---|---|
| Education (college vs. high school) | 4.9% difference |
| Income (high vs. low) | 4.1% difference |
| Age (old vs. young) | 2.5% difference |
| Geography | 0.5% spread |

Complaints written in a college-educated style received modestly better remedy recommendations than those written in a high-school style—even though the underlying issue was identical.

## Process Fairness: No Evidence of Discriminatory Hurdles
We also examined whether certain personas were asked for additional information more often (creating extra burden).

- **GPT-4o-mini:** Never requested additional information from any persona  
- **GPT-5-mini:** Requested information **44.5%** of the time, but this rate was virtually identical across all demographic groups (**within 1.5 percentage points**)  

Neither model created discriminatory procedural barriers based on writing style.

## Conclusion: Is ChatGPT Biased Based on Writing Style?
- **For the newer GPT-5-mini:** No meaningful bias detected. When we control for complaint content and vary only the writing style, remedy recommendations are essentially identical across all personas.  
- **For the older GPT-4o-mini:** Modest bias exists, particularly favoring writing styles associated with higher education and socioeconomic status.  

This represents genuine progress. The newer model appears to have substantially reduced its sensitivity to stylistic cues that might correlate with customer demographics—an important improvement for fair AI-assisted customer service.

## Recommended Actions
- Organizations using **GPT-4o-mini** should consider upgrading to reduce education/income-related bias
- This controlled test provides reassurance that **GPT-5-mini** evaluates complaint substance over style
- Continue monitoring as models evolve—bias patterns can shift with updates


# Can ChatGPT Anonymize Complaints? Executive Summary

## The Experiment
We tested whether ChatGPT can serve as a preprocessing filter to remove demographic signals from complaint text before downstream AI processing.

The goal: if all complaints are rewritten in a uniform style, subsequent AI systems cannot infer customer demographics from writing patterns—eliminating the opportunity for style-based bias in remedy recommendations.

Here’s how we tested it:

- Took **5,000** original complaints with diverse writing styles
- Asked ChatGPT to rewrite each in **“neutral, professional”** language
- Asked ChatGPT to guess the customer’s demographic profile from the neutralized text
- Measured whether demographic diversity collapsed to uniformity (indicating successful signal removal)

## Key Finding: Neutralization Works Well for Most Dimensions
Both models successfully homogenized complaint language, dramatically reducing the AI’s ability to distinguish between customer personas.

### Signal Removal Effectiveness
We measured **“entropy”**—a technical term for diversity. **Lower entropy after neutralization** means the technique is working (all complaints now look the same to the AI).

### GPT-4o-mini Results

| Dimension | Original Diversity | After Neutralization | Signal Removed? |
|---|---:|---:|---|
| Education | High (0.81) | Very low (0.11) | ✅ 87% reduction |
| ESL status | Moderate (0.47) | Near-zero (0.003) | ✅ 99% reduction |
| Income | Moderate (0.47) | Low (0.13) | ✅ 72% reduction |
| Geography | Moderate (0.40) | Low (0.34) | ⚠️ 15% reduction |
| Gender | High (0.91) | High (0.83) | ⚠️ 8% reduction |
| Age | Low (0.14) | Low (0.14) | ➖ No change |

### GPT-5-mini Results

| Dimension | Original Diversity | After Neutralization | Signal Removed? |
|---|---:|---:|---|
| Education | High (0.85) | Near-zero (0.05) | ✅ 94% reduction |
| ESL status | Moderate (0.49) | Near-zero (0.04) | ✅ 92% reduction |
| Income | Moderate (0.40) | Very low (0.07) | ✅ 82% reduction |
| Age | High (0.90) | Moderate (0.45) | ✅ 51% reduction |
| Gender | High (0.84) | Moderate (0.45) | ✅ 46% reduction |
| Geography | High (0.90) | High (0.82) | ⚠️ 9% reduction |

## What Works Well
Three dimensions show excellent anonymization in both models:

- **Education:** 87–94% signal reduction — the AI can no longer distinguish college-educated from high-school-educated writing styles  
- **ESL status:** 92–99% signal reduction — non-native English patterns are effectively normalized  
- **Income level:** 72–82% signal reduction — socioeconomic markers in language are largely removed  

These are meaningful results. Education, language background, and income level are among the most concerning dimensions for potential discrimination, and neutralization effectively masks them.

## Where Gaps Remain
**Geographic signals persist** in both models (only **9–15% reduction**). Something about how customers reference locations, services, or contexts continues to reveal urban/suburban/rural patterns.

**Gender and age show mixed results:**
- **GPT-5-mini** achieves moderate anonymization (**46–51% reduction**)
- **GPT-4o-mini** struggles with these dimensions (**0–8% reduction**)

## Conclusion
ChatGPT is largely effective at anonymizing complaint language, particularly for the highest-risk dimensions of **education**, **language background**, and **socioeconomic status**. After neutralization, downstream AI systems cannot reliably infer these characteristics from writing style.

**GPT-5-mini** delivers more consistent anonymization across all dimensions and is the better choice for this application. **Geographic signals remain partially detectable** in both models and represent an area for further refinement.

## Recommended Actions
- Neutralization is a viable bias mitigation strategy, particularly for **education**, **ESL**, and **income-related** bias
- **GPT-5-mini** is preferred for more complete anonymization
- Consider supplementary measures for geographic signals if location-based bias is a concern
- Validate end-to-end: confirm that reduced persona signals translate to reduced outcome disparities in your specific downstream processes

# Does Anonymizing Complaint Language Remove Bias? Executive Summary

## The Experiment
Having established that ChatGPT can successfully neutralize persona-specific writing styles, we tested the critical question: **Does this neutralization actually eliminate bias in remedy recommendations?**

We ran the complete pipeline:

- Took **5,000** original complaints with diverse writing styles  
- Neutralized them to remove demographic signals  
- Asked ChatGPT to recommend remedies for the neutralized complaints  
- Measured whether outcome disparities were reduced or eliminated  

This is the definitive test of whether linguistic anonymization works as a bias mitigation strategy.

## Key Finding: Results Depend Heavily on Which Model You Use

### GPT-5-mini: Neutralization Works
**Causal bias eliminated.** When we controlled for complaint content and tested whether writing style affected outcomes, we found:

- No statistically significant differences between personas (**Friedman test p = 0.42**)  
- Zero pairwise comparisons showed significant bias (**0 out of 630 tested**)  
- Outcome variance reduced by **5.4%** after neutralization  
- Process burden variance reduced by **9.8%**

The remaining correlational differences (e.g., rural vs. suburban outcomes) reflect differences in the underlying complaints themselves, not bias based on how they’re written.

| Bias Dimension | Before Neutralization | After Neutralization |
|---|---|---|
| Gender | Not significant | Not significant ✅ |
| ESL status | Not significant | Not significant ✅ |
| Causal style bias | Not tested | None detected ✅ |

### GPT-4o-mini: Neutralization Fails
**Significant causal bias persists.** Despite neutralization:

- **147 out of 630** pairwise persona comparisons showed significant bias  
- Outcome variance **increased by 2.2%** after neutralization  
- Strong disparities remain across multiple dimensions  

| Dimension | Favorable Outcome Rate |
|---|---|
| Female vs. Male | 69% vs. 49% |
| Young vs. Old | 63% vs. 48% |
| College vs. High School | 65% vs. 56% |
| Native vs. ESL | 64% vs. 53% |

Even after complaints were rewritten in neutral language, this model continues to produce systematically different outcomes. The neutralization step does not solve the problem.

## Why the Difference?
GPT-4o-mini appears to have deeper embedded biases that persist even when surface-level writing style is standardized. The model may be responding to:

- Subtle content differences that survive neutralization  
- Complaint structure or framing patterns  
- Topic or issue-type correlations with demographics  

GPT-5-mini, by contrast, evaluates neutralized complaints without detectable persona-based discrimination.

## Process Fairness Results
We also examined whether certain groups face higher procedural burden (being asked for more information).

- **GPT-4o-mini:** Never requests additional information from any customer, so no process discrimination exists—but also no opportunity to measure mitigation.  
- **GPT-5-mini:** Shows meaningful improvement after neutralization:
  - Information request disparities reduced by **9.8%**
  - Pre/post change is statistically significant (**p = 0.015**)
  - ESL and male customers, who previously faced higher request rates, see reduced burden

## Conclusion: Does This Risk Mitigation Approach Work?
- **For GPT-5-mini:** Yes. Linguistic neutralization successfully eliminates detectable causal bias. After anonymization, the model evaluates complaints based on substance rather than writing style. This represents a viable bias mitigation strategy.  
- **For GPT-4o-mini:** No. Neutralization fails to remove bias—outcome disparities persist at statistically significant levels, and variance actually increases. This model requires different mitigation approaches or should not be used in bias-sensitive applications.

## Recommended Actions
- Use **GPT-5-mini with neutralization** as a validated bias mitigation pipeline
- Do **not** use **GPT-4o-mini** for complaint handling where fairness is a concern—neutralization does not solve its bias issues
- Implement the two-step process: **neutralize complaints first**, then process for **remedy recommendations**
- Continue monitoring as models are updated—bias patterns can shift with new versions

output:
## Overview
We conducted a series of experiments to determine whether ChatGPT exhibits bias when recommending remedies for customer complaints—and whether such bias can be eliminated. Our research tested two versions of ChatGPT (**the older GPT-4o-mini** and **newer GPT-5-mini**) across thousands of complaints to understand how these systems form assumptions about customers and whether those assumptions affect outcomes.

## Key Findings
- **ChatGPT profiles customers from their writing style.** Without any demographic information, the AI forms confident assumptions about who customers are based solely on how they write. It defaults to imagining an **older, suburban, college-educated, middle-class male**—a profile that accounted for **45%** of all estimates. Rural customers, women, and non-native English speakers are systematically underrepresented in the AI’s mental picture.

- **These assumptions influence remedy recommendations.** Both models showed statistically significant relationships between imagined customer demographics and recommended remedies. The older model exhibited pronounced gender and language-background bias; the newer model has largely eliminated these but retains geographic bias.

- **Linguistic neutralization is an effective mitigation strategy—but only with the right model.** When complaints are first rewritten in neutral, professional language before processing, **GPT-5-mini** shows no detectable causal bias: remedy recommendations become independent of writing style. However, the same approach fails with **GPT-4o-mini**, where significant disparities persist even after neutralization.

## Recommendations
- **Adopt GPT-5-mini with a two-step neutralization pipeline** for complaint handling where fairness matters. This combination represents a validated approach to eliminating style-based discrimination.

- **Do not use GPT-4o-mini in bias-sensitive applications.** Its embedded biases resist neutralization and produce systematically different outcomes based on perceived customer demographics.

- **Monitor outcomes continuously.** Bias patterns shift as models are updated. Ongoing measurement against actual customer demographics is essential to ensure AI-assisted decisions remain fair.
