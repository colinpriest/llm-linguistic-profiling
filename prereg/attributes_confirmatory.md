# Pre-registration 2, amendment 2b: confirmatory attribute hypotheses (holdout)

**Frozen:** 20 October 2026, before any holdout attribute scores exist. The pilot passed G1–G3 (commit a7e920f). These hypotheses are derived from the pilot and are tested once on the 1,000-complaint holdout (same 19 attribute questions, `jev_attributes.py holdout --tag n1000`) using `prereg/attributes_confirmatory.py --panel holdout`.

**Setup:**
- Complaint-block bootstrap with 2,000 resamples, seed 20261021.
- Within-complaint (complaint fixed effects) throughout.
- Attributes are the 19 Jev expected scores plus 3 objective measures. Families and normative relevance are as in `prereg/attributes_prereg.md`.
- Outcomes are GPT decisions on the same holdout guises (`holdout_out/decisions_n1000.csv`) and Jev E[tier] (option order o012).

| # | Hypothesis | Supported if |
|---|---|---|
| **H-A1** | **Content invariance.** Persona rewrites leave content unchanged. | For all 6 content attributes in both banks, the one-sided 95% upper bound of the within-complaint persona η² is < 0.05. |
| **H-A2** | **Delivery drives GPT-4o-mini's education and SES gaps.** Outcome: GPT-4o-mini tier on GPT-4o-mini guises. Order-invariant Gelbach decomposition of the college−high-school and high−low SES gaps. | For each dimension: share explained ≥ 0.50, delivery contribution 95% CI > 0, and (delivery − content) 95% CI > 0. Both dimensions required. The monetary outcome is reported as secondary. |
| **H-A3** | **The education effect works through recognisable cues.** GPT-4o-mini tier, college vs high school: the complaint-demeaned gap among guises where Jev recognises the intended education level (P ≥ 0.60) exceeds the gap among ambiguous guises. | 95% CI of (recognised gap − ambiguous gap) > 0. |
| **H-A4** | **Jev reads persona from delivery.** For Jev's P(old), P(college) and P(high) − P(low) SES, in both banks: within-complaint R² of delivery attributes alone exceeds R² of content attributes alone. | All 6 differences have 95% CI > 0. |
| **H-A5** | **Jev's own remedy judgment is content-driven.** For Jev E[tier], in both banks: R²(content alone) > R²(delivery alone). This is secondary-strength evidence, because Jev also scores the attributes. | Both differences have 95% CI > 0. |

**Not tested confirmatorily:** GPT persona estimates. They don't exist for holdout guises, and generating them would need new OpenAI calls.

**Code check, disclosed:** the script was first run on the already-seen pilot panel (`--panel pilot`), purely to debug it. Any code fix made after this commit will be committed and disclosed before the holdout run.
