# Pre-registration and analysis record: Jev / FAccT 2027

Every confirmatory claim maps to a document and a script frozen in git **before** the data it tests existed, and to a committed results file. Commits are on branch `claude/project-thread-6y5sk1`. All times are AEST.

## Timeline
| When | Commit | What |
|---|---|---|
| 19 Oct 3:15 pm | 7958a6e | **Pre-registration 1** (`jev_facct_prereg.md`, `confirmatory_analysis.py`): H1–H6, holdout draw and fingerprint |
| 19 Oct 4:04–6:09 pm | ad0eecb, 3f36e7e, 7f8f9e2 | Non-substantive `holdout_panel.py` fixes (per-model files, schema warm-up, tolerance of transient failures) |
| 19 Oct 11:19 pm | — | One-shot confirmatory run on 1,000 holdout complaints |
| 20 Oct 9:29 am | 21e856e | Holdout results recorded (`holdout_confirmatory_results.*`) |
| 20 Oct 10:44 am | 764b1c5 | **Amendment 1** (`amendment_1_extension.md`, `extension_h6.py`): extend to 4,000 for H6 |
| 20 Oct 10:56 am | cfbd833 | Amendment 1 **not executed** after a materiality review; post hoc `materiality_analysis.py` |
| 20 Oct 11:02 am | 88d6899 | Post hoc H6 in rate units (`h6_rate_units.py`: 90/95% CIs, minimum-effect test, TOST vs 1 pp) |
| 20 Oct 11:26 am | 9abbd00 | **Pre-registration 2** (`attributes_prereg.md`, `../jev_attributes.py`, `attributes_pilot_analysis.py`): delivery / structure / content attributes, pilot go/no-go |
| 20 Oct 11:47 am | fcc1e82 | **Amendment 2a**: persona uncertainty (Q-E); exact sequence disclosed in `attributes_prereg.md` §7 |
| 20 Oct 11:52 am | 362cb90 | Corrected-bootstrap sensitivity (`corrected_bootstrap.py`) |
| 20 Oct 12:41 pm | a7e920f | Attribute pilot results: GO (`attributes_pilot_results.json`) |
| 20 Oct 1:13 pm | e847152 | **Amendment 2b** (`attributes_confirmatory.md`, `attributes_confirmatory.py`): H-A1–H-A5 |
| 20 Oct 2:51 pm | f1dfdca | One-shot confirmatory attribute results (`attributes_holdout_confirmatory_results.json`) |
| 20 Oct 4:03 pm | 1d78bcd | Secondary analyses (pre-registered as "reported") and H4 on a difference scale (`secondary_analysis.py`) |

## Status of each claim
| Claim | Status | File |
|---|---|---|
| H1, H2, H3a, H4 | confirmatory, supported (H2: 9/12 contrasts) | `holdout_confirmatory_results.json` |
| H3b | estimation only (no reduction) | same; corrected CIs in `corrected_bootstrap_results.json` |
| H5 | confirmatory, not supported | same |
| H6 | confirmatory, inconclusive at n = 1,000; extension not executed | same, plus `amendment_1_extension.md` §5 |
| Materiality, rate-unit H6 | **post hoc** | `materiality_results.json`, `h6_rate_units_results.json` |
| Corrected bootstrap | **post hoc** sensitivity | `corrected_bootstrap_results.json` |
| H-A1–H-A5 | confirmatory, all supported | `attributes_holdout_confirmatory_results.json` |
| Secondary S0–S4 | pre-registered as reported, not confirmatory | `secondary_results.json` |
| Attribute descriptives D1–D4 | descriptive | `attributes_descriptives.json` |

## Known limitations, disclosed
- Bootstrap CIs for the noise-corrected variance-component metric D were biased upward in the pre-registered script. The corrected bootstrap changes no conclusion.
- Jev 1.13 is not deterministic: rescoring the same text gives a mean TV of 0.020, and 2.5% of chosen tiers flip.
- There is no ground truth for remedy correctness. Claims concern sensitivity and disparity, not accuracy.
- GPT persona estimates exist only for exploratory-panel guises.
