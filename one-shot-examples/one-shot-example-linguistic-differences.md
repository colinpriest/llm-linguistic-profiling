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
