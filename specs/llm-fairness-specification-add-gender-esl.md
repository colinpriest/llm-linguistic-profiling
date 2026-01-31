# Adding Gender and English as a Second Language (ESL)

## Why These Variables Add Value

**Gender**

* Protected characteristic under discrimination law (directly relevant to your fairness research)
* Well-documented gender bias in financial services literature
* Linguistic gender markers are subtle but detectable (hedging language, assertiveness patterns, etc.)
* Highly policy-relevant finding if discrimination is detected

**English as a Second Language (ESL)**

* Could serve as proxy for national origin discrimination (protected class)
* Extends existing dialect discrimination literature (AAE vs. SAE) to broader linguistic patterns
* Non-native grammar/syntax patterns are exactly the kind of subtle cues your research is designed to detect
* Practical implications for financial services serving diverse populations

## Recommended Implementation

Add these to **Step 1 (Persona Estimation)** and **Step 3 (Correlation Analysis)** only — not to the causal rewriting in Steps 4-5. This keeps your factorial design manageable:

| Component                       | Current Design         | With Addition     |
| ------------------------------- | ---------------------- | ----------------- |
| Persona estimation dimensions   | 4 (age, geo, edu, ses) | 6 (+ gender, ESL) |
| Causal rewrite combinations     | 36                     | 36 (unchanged)    |
| Correlation analysis dimensions | 4                      | 6                 |

## Rationale for Not Expanding Rewrites

1. **Practical** : Expanding to 36 × 2 × 2 = 144 rewrites per complaint would be prohibitively expensive
2. **Methodological** : Gender and ESL are harder to authentically "rewrite into" than age or education — an LLM's stereotyped notion of "how women write" or "how ESL speakers write" may introduce more noise than signal
3. **Sufficient for correlation** : You can still establish whether *estimated* gender/ESL correlates with outcomes, which is the primary discrimination concern
