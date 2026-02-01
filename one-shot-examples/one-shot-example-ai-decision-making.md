ONE_SHOT EXAMPLE FOR "AI DECISION-MAKING"

input:
For gpt-4o-mini (n=5000), the overall persona×tier association was significant (chi2=105.59, p=0.0000, V=0.103), and for gpt-5-mini (n=5000) it was also significant with a larger effect (chi2=176.36, p=0.0000, V=0.133). In gpt-4o-mini, the largest dimension-level association was gender×tier (V=0.192, p=0.0000; chi2=184.84), while in gpt-5-mini the largest was geographic×tier (V=0.080, p=0.0000; chi2=64.56). Notably, gpt-5-mini showed non-significant associations for esl×tier (p=0.1347, V=0.028) and gender×tier (p=0.1120, V=0.030), whereas gpt-4o-mini showed significant effects for both esl×tier (p=0.0000, V=0.073) and gender×tier (p=0.0000, V=0.192). Persona diversity was higher for gpt-5-mini with 23 unique personas and entropy=2.749 versus 21 unique personas and entropy=1.597 for gpt-4o-mini, indicating less anchoring in the former. Comparing strength of persona-tier associations, gpt-5-mini had a higher overall V (0.133 vs 0.103), but gpt-4o-mini exhibited markedly stronger single-dimension sensitivity in gender (V=0.192) than any gpt-5-mini dimension (max V=0.080).

output:
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
