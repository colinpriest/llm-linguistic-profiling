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
# Does Neutralization Remove Demographic Signals from Complaint Text?

## The Experiment
We tested whether ChatGPT can rewrite complaints to remove demographic signals from the text. The goal is straightforward: if all complaints are standardized to neutral, professional language, subsequent AI systems cannot infer customer demographics from writing style — eliminating the opportunity for style-based bias in remedy decisions.

Here's how we tested it:

- Took **5,000** original complaints with diverse writing styles
- Asked ChatGPT to rewrite each in **"neutral, professional"** language
- Asked ChatGPT to guess the customer's demographic profile from the neutralized text
- Measured whether diversity collapsed to uniformity (indicating successful signal removal)

**Important:** Standardization is the intended outcome. When neutralization works, the LLM should infer similar personas from all complaints — that means the demographic signals have been successfully removed.

## Key Finding: Neutralization Successfully Removes Most Demographic Signals
Both models effectively standardized complaint language, dramatically reducing the ability to distinguish between customer personas.

### How We Measured Success
We measured **"entropy"** — a measure of diversity. **Lower entropy after neutralization** means the technique is working (complaints now look similar to the AI).

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
Three dimensions show excellent signal removal in both models:

- **Education:** 87-94% reduction — the AI can no longer distinguish college-educated from high-school-educated writing styles
- **ESL status:** 92-99% reduction — non-native English patterns are effectively normalized
- **Income level:** 72-82% reduction — socioeconomic markers in language are largely removed

These are the dimensions most likely to correlate with how complaints are written, and neutralization effectively standardizes them.

## Where Gaps Remain
**Geographic signals persist** in both models (only 9-15% reduction). Something about how customers reference locations, services, or contexts continues to reveal urban/suburban/rural patterns.

**Gender and age show mixed results:**
- **GPT-5-mini** achieves moderate signal removal (46-51% reduction)
- **GPT-4o-mini** struggles with these dimensions (0-8% reduction)

## Conclusion
Neutralization is largely effective at standardizing complaint language. The technique removes demographic signals for **education**, **language background**, and **income level** in both models. **GPT-5-mini** provides more consistent signal removal across all dimensions.

The next step is to test whether removing these signals actually eliminates bias in remedy decisions.
