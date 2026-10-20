# Amendment 1: extend the holdout to 4,000 complaints for H6

**Frozen:** 20 October 2026, before any of holdout complaints 1,001–4,000 were generated. This amendment and `prereg/extension_h6.py` are committed together. The original pre-registration (`prereg/jev_facct_prereg.md`) and its one-shot results at n = 1,000 (`prereg/holdout_confirmatory_results.*`, commit 21e856e) are unchanged and remain the confirmatory record for H1–H5.

## 1. Reason
At n = 1,000, H6 (Jev's systematic monetary disparity is below GPT-4o-mini's at matched monetary volume) was inconclusive:
- Δ = D(GPT) − D(Jev) = 0.010, 95% CI −0.006 to 0.019.

A resampling power analysis, based on all 3,000 complaints scored so far, estimates the power for Δ ≈ 0.010 as:

| n | Power |
|---|---|
| 1,000 | 0.27 |
| 4,000 | 0.78 |
| 6,000 | 0.95 |

The extension to 4,000 was chosen for cost and time reasons. It was decided after seeing the n = 1,000 result, so it is handled as a group-sequential design.

## 2. Sample
- Same draw as the original pre-registration: the first 4,000 of the seeded shuffle of the 18,000 fresh complaints. The first 1,000 are the existing holdout.
- Generation is the same as for the first 1,000: prompts, models, client settings, and Jev questions (6 option orders plus persona).

| Fingerprint (SHA-256, newline-joined case_ids in draw order) | Hash |
|---|---|
| First 1,000 (unchanged) | `2753897c44576a8201e4bd2fdab0486a817c808b0c2728a678e53f6911365dc9` |
| All 4,000 | `cb9fe6ea04f4d375670b2259f7d110b62be47e8d2f432f6e03a3cbea7abb7e84` |

## 3. Confirmatory test
- **Hypothesis:** only the GPT-4o-mini part of H6, with Δ = D(GPT-4o-mini monetary) − D(Jev monetary). Jev's monetary decisions use p2 at or above the quantile that matches GPT-4o-mini's monetary rate on the same guises. D is the noise-corrected standardised persona disparity from `confirmatory_analysis.py`.
- **Design:** two looks, Lan–DeMets O'Brien–Fleming alpha spending, overall two-sided alpha 0.05.

| Look | n | Information fraction | Alpha spent | Boundary |
|---|---|---|---|---|
| 1 (done) | 1,000 | 0.25 | 8.86e-05 | \|z\| 3.920, not crossed |
| 2 (final) | 4,000 | 1 | nominal two-sided 0.04996 | \|z\| 1.9603 |

- **Decision rule:** H6 (GPT-4o-mini) is supported if the lower bound of the 95.004% complaint-bootstrap CI (5,000 resamples, seed 20261020) for Δ is above 0.
- **Dropped:** the GPT-5-mini part of H6 ("Jev not lower") is no longer a confirmatory claim. GPT-5-mini's own systematic monetary disparity is estimated at about 0, so no directional test is meaningful at feasible n. It is reported descriptively.

## 4. Other analyses at n = 4,000
H1–H5 are recomputed on n = 4,000 as secondary, descriptive updates. Their confirmatory status remains that recorded at n = 1,000.
