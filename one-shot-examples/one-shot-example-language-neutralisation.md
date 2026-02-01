ONE-SHOT EXAMPLE "LANGUAGE NEUTRALISATION"

input:
Model: gpt-4o-mini (4999 neutralized re-estimates)
── Neutralized age distribution ──
  old    100
  young  4899
── Neutralized geographic distribution ──
  rural     2
  suburban  312
  urban     4685
── Neutralized education distribution ──
  college      4928
  high_school  71
── Neutralized socioeconomic distribution ──
  high    6
  low     82
  middle  4911
── Neutralized gender distribution ──
  female  3678
  male    1321
── Neutralized esl distribution ──
  esl     1
  native  4998
Comparison: original vs neutralized persona signal
  age entropy: original=0.136, neutralized=0.141
  geographic entropy: original=0.404, neutralized=0.342
  education entropy: original=0.809, neutralized=0.108
  socioeconomic entropy: original=0.472, neutralized=0.134
  gender entropy: original=0.909, neutralized=0.833
  esl entropy: original=0.473, neutralized=0.003


Model: gpt-5-mini (4999 neutralized re-estimates)
── Neutralized age distribution ──
  old    4536
  young  463
── Neutralized geographic distribution ──
  rural     15
  suburban  3791
  urban     1193
── Neutralized education distribution ──
  college      4972
  high_school  27
── Neutralized socioeconomic distribution ──
  high    8
  low     31
  middle  4960
── Neutralized gender distribution ──
  female  472
  male    4527
── Neutralized esl distribution ──
  esl     20
  native  4979
Comparison: original vs neutralized persona signal
  age entropy: original=0.902, neutralized=0.445
  geographic entropy: original=0.898, neutralized=0.821
  education entropy: original=0.852, neutralized=0.048
  socioeconomic entropy: original=0.396, neutralized=0.072
  gender entropy: original=0.836, neutralized=0.451
  esl entropy: original=0.485, neutralized=0.038

output:
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
