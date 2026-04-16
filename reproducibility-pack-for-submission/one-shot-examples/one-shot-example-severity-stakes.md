ONE-SHOT EXAMPLE "SEVERITY STAKES"

input:
── Severity Analysis Summary ──
Severity Metrics Overview:
  monetary_severity coverage: 47.2% (236/500 complaints had extractable dollar amounts)
  issue_severity coverage: 100.0% (all complaints classified)
  monetary_severity distribution: low=89, medium=102, high=45, unknown=264
  issue_severity distribution: low=148, medium=207, high=145

  Agreement between metrics (where both available):
    Cohen's kappa: 0.18
    Spearman r: 0.22
    Agreement rate: 38.1%
  Interpretation: Low agreement — metrics capture different severity dimensions

Model: gpt-4o-mini
── Main Effects ──
  monetary_severity: mean tier by level: low=0.742, medium=0.781, high=0.811
    ANOVA: F=3.21, p=0.042, partial eta-sq=0.013
    Trend: Spearman r=0.14, p=0.031
  issue_severity: mean tier by level: low=0.758, medium=0.762, high=0.770
    ANOVA: F=0.41, p=0.664, partial eta-sq=0.001
    Trend: Spearman r=0.03, p=0.412

── Amplification Analysis ──
  monetary_severity:
    Bias at low severity (SD across personas): 0.032
    Bias at medium severity: 0.048
    Bias at high severity: 0.089
    Amplification ratio: 2.78x (CI: 1.42, 5.13)
    Two-way interaction (persona x severity): F=1.87, p=0.038, eta-sq=0.024
  issue_severity:
    Bias at low severity: 0.041
    Bias at medium severity: 0.044
    Bias at high severity: 0.051
    Amplification ratio: 1.24x (CI: 0.71, 2.08)
    Two-way interaction (persona x severity): F=0.92, p=0.487, eta-sq=0.008

── Worst Case ──
  Model: gpt-4o-mini, Metric: monetary_severity, Level: high
  Most advantaged persona: young_suburban_college_high
  Most disadvantaged persona: old_rural_high_school_low
  Effect size (Cohen's d): 0.41

Model: gpt-5-mini
── Main Effects ──
  monetary_severity: mean tier by level: low=0.831, medium=0.838, high=0.842
    ANOVA: F=0.52, p=0.595, partial eta-sq=0.002
    Trend: Spearman r=0.04, p=0.528
  issue_severity: mean tier by level: low=0.832, medium=0.836, high=0.837
    ANOVA: F=0.12, p=0.887, partial eta-sq=0.000
    Trend: Spearman r=0.02, p=0.718

── Amplification Analysis ──
  monetary_severity:
    Bias at low severity: 0.028
    Bias at medium severity: 0.031
    Bias at high severity: 0.035
    Amplification ratio: 1.25x (CI: 0.68, 2.15)
    Two-way interaction (persona x severity): F=0.71, p=0.612, eta-sq=0.006
  issue_severity:
    Bias at low severity: 0.025
    Bias at medium severity: 0.027
    Bias at high severity: 0.029
    Amplification ratio: 1.16x (CI: 0.59, 1.94)
    Two-way interaction (persona x severity): F=0.54, p=0.742, eta-sq=0.004

── Process Fairness ──
  gpt-4o-mini: Info request disparity by severity: low=0.000, medium=0.000, high=0.000
  gpt-5-mini: Info request disparity by severity: low=0.012, medium=0.015, high=0.022

── Consistency ──
  Severity main effect on outcomes: monetary p=0.042, issue p=0.664 — Metric-specific
  Persona x severity interaction: monetary p=0.038, issue p=0.487 — Metric-specific
  Amplification ratio: monetary=2.78x, issue=1.24x — Metric-specific

output:
# Do Higher Stakes Lead to Greater Bias? Severity Analysis

## The Question
Not all complaints are equal. A disputed $50 fee and a $15,000 mortgage dispute carry very different stakes for the customer. We investigated whether AI bias **increases when the financial or issue severity is higher** — in other words, whether unfair treatment worsens precisely when fair treatment matters most.

We measured severity in two independent ways:
- **Dollar severity:** Based on the actual monetary amounts mentioned in complaints (available for about 47% of cases)
- **Issue severity:** Based on the type of product and issue involved (available for all cases)

These two measures showed low agreement with each other, meaning they capture genuinely different aspects of how serious a complaint is.

## Key Finding: Financial Stakes Amplify Bias in the Older Model

### GPT-4o-mini: Bias Nearly Triples for High-Dollar Complaints
When we examined complaints involving large dollar amounts (over $1,000), the spread in remedy quality across different customer personas was **2.8 times larger** than for low-dollar complaints. This amplification was statistically meaningful.

| Dollar Severity | Variation in Remedy Quality Across Personas |
|---|---|
| Low (under $100) | 3.2% spread |
| Medium ($100 - $1,000) | 4.8% spread |
| High (over $1,000) | 8.9% spread |

The worst-case scenario: for high-dollar complaints on GPT-4o-mini, the gap between the most and least favoured customer personas reached a moderate effect size — meaning the bias was not just detectable but practically meaningful.

### GPT-5-mini: No Evidence of Severity Amplification
The newer model showed no meaningful increase in bias at higher severity levels. Whether the complaint involved $50 or $15,000, persona-based variation in outcomes remained consistently small (around 2.5-3.5% spread).

## Issue-Based Severity: No Clear Amplification for Either Model
When severity was measured by product/issue type rather than dollar amount, neither model showed a statistically meaningful increase in bias at higher severity levels. This suggests the amplification effect is specifically tied to **financial stakes** rather than the general seriousness of the complaint category.

## Process Fairness
Neither model created meaningful differences in procedural hurdles across severity levels. Information request rates were consistent regardless of complaint severity.

## Robustness Note
The severity amplification finding is **metric-specific**: it appeared for dollar-based severity but not for issue-based severity. This means the pattern should be interpreted with some caution — it may reflect how AI models respond to financial amounts in complaint text rather than a universal severity effect.

## Implications
- **GPT-4o-mini poses higher risk for high-value complaints:** Organizations handling large-dollar disputes should be particularly cautious about bias when using this model
- **GPT-5-mini shows improvement:** The newer model does not amplify bias with stakes, suggesting architectural progress
- **Monitor dollar-amount sensitivity specifically:** Standard fairness audits should include a severity breakdown, particularly for monetary amounts
- **The finding that bias concentrates in high-stakes cases strengthens the case for human review** of large-dollar complaint resolutions
