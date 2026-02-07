ONE-SHOT EXAMPLE FOR "LINGUISTIC DIFFERENCES"

input:
Model: gpt-5-mini (5000 estimates)
── age distribution ──
  old    3409
  young  1591
── geographic distribution ──
  rural     26
  suburban  3582
  urban     1392
── education distribution ──
  college      3613
  high_school  1387
── socioeconomic distribution ──
  high    4
  low     378
  middle  4618
── gender distribution ──
  female  1330
  male    3670
── esl distribution ──
  esl     525
  native  4475
── Full persona type counts (23/36 present) ──
  old_rural_college_high             0
  old_rural_college_low              0
  old_rural_college_middle           2
  old_rural_high_school_high         0
  old_rural_high_school_low          6
  old_rural_high_school_middle       5
  old_suburban_college_high          0
  old_suburban_college_low           2
  old_suburban_college_middle        2237
  old_suburban_high_school_high      0
  old_suburban_high_school_low       52
  old_suburban_high_school_middle    403
  old_urban_college_high             4
  old_urban_college_low              1
  old_urban_college_middle           583
  old_urban_high_school_high         0
  old_urban_high_school_low          43
  old_urban_high_school_middle       71
  young_rural_college_high           0
  young_rural_college_low            0
  young_rural_college_middle         1
  young_rural_high_school_high       0
  young_rural_high_school_low        10
  young_rural_high_school_middle     2
  young_suburban_college_high        0
  young_suburban_college_low         6
  young_suburban_college_middle      483
  young_suburban_high_school_high    0
  young_suburban_high_school_low     52
  young_suburban_high_school_middle  347
  young_urban_college_high           0
  young_urban_college_low            10
  young_urban_college_middle         284
  young_urban_high_school_high       0
  young_urban_high_school_low        196
  young_urban_high_school_middle     200
── Confidence histogram ──
  0.0-0.2  0
  0.2-0.4  8
  0.4-0.6  323
  0.6-0.8  4658
  0.8-1.0  11

output:
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

## Distribution Patterns
The model's inferences cluster around certain profiles:

- **Age:** 68% classified as older vs 32% younger
- **Location:** 72% suburban, 28% urban, 0.5% rural
- **Education:** 72% college-educated, 28% high school
- **Income:** 92% middle class, 8% low income, <1% high income
- **36** possible persona combinations exist; **23** appeared in the estimates

## Confidence Levels
The model reported moderate-to-high confidence in its estimates:
- **93%** of estimates had 60-80% confidence
- Very few estimates showed low (<40%) or very high (>80%) confidence

## Why This Matters for the Experiment
The key question is NOT whether these persona estimates are accurate or complete. What matters is whether the LLM's remedy decisions differ based on the persona it infers. If decisions vary by inferred persona when the complaint substance is identical, that would indicate bias.

## Next Step
The experiment proceeds to test whether these inferred personas correlate with — and causally affect — remedy decisions.
