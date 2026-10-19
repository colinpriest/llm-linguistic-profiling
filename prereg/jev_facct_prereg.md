# Pre-registration: style sensitivity in probabilistic vs single-label remedy decisions

**Frozen:** 19 October 2026, before any holdout text was generated. The commit containing this file and `prereg/confirmatory_analysis.py` is the registration. Neither file may be changed afterwards. Any deviation will be reported as such in the paper, and further analyses will be labelled exploratory.

## 1. Background and what has already been seen
- An exploratory panel has been fully inspected: 2,000 CFPB complaints x 36 persona guises x 2 rewrite models (gpt-4o-mini, gpt-5-mini), scored by Jev 1.13 under two option orders, with GPT decisions on the same texts. Results are in jev_pilot_out/dist_audit/findings.md and jev_pilot_out/confirmatory_results.json.
- The hypotheses below were formed from that panel and are tested once, on a fresh holdout.

## 2. Holdout sample
- **Pool:** data/.pipeline/step10_seed42_n20000.parquet (20,000 cleaned complaints), minus the 2,000 in the exploratory matched-guise panel. That leaves 18,000 complaints that were never rewritten, never persona-decided and never scored by Jev. They were used only in the earlier study's observational analysis of original texts.
- **Draw:** `fresh = sorted(fresh_ids); random.Random(20261019).shuffle(fresh); holdout = fresh[:N]` (holdout_panel.py). Because it is a prefix, a larger N extends a smaller one.
- **Size:** N = 1,000. If N is raised before any holdout text is generated, that change will be committed as an amendment to this file, and the first 1,000 complaints stay identical.
- **Fingerprints:** SHA-256 of the newline-joined case_ids, in draw order.

| N | SHA-256 |
|---|---|
| 1,000 | `2753897c44576a8201e4bd2fdab0486a817c808b0c2728a678e53f6911365dc9` |
| 2,000 | `f2bf0ed85680290b70c6a9041999aa58f3ca2481ba4dd5cd1fb905f86a9429bd` |
- **Generation:** the same rewrite prompt (PERSONA_REWRITE_SYSTEM and format_persona_rewrite_user, current PROMPT_VERSIONS), the same models and the same client settings as the original Steps 4–5 (temperature 0, max_tokens 16384). The remedy-decision prompt is likewise unchanged.
- **Jev scoring:** jev-1.13.0 (the version string is recorded per response). All 6 orders of the three remedy options are asked in one call per text, plus the four persona-estimate questions.
- **Exclusions:** a complaint enters the analysis only if all 36 guises succeed in both rewrite banks (complete cases). The number excluded and the reasons will be reported.

## 3. Primary measures
| Measure | Definition |
|---|---|
| p = (p0, p1, p2) | Jev probabilities for no action / non-monetary / monetary, primary order o012 |
| E[tier] | p1 + 2·p2 |
| argmax tier | Jev's hard tier |
| Normalised entropy | Basis for referral |
| TV(p, q) | ½ Σ \|p_k − q_k\| |
| D (standardised systematic persona disparity) | Two-way complaint x persona ANOVA decomposition: sqrt(var_persona / var_complaint), each variance corrected for the residual mean square (unbiased estimators). D does not reward a constant output, and it does not penalise a rule merely for being noisier. |
| Residual share (arbitrariness) | Residual (complaint x persona interaction) variance / (residual + between-complaint variance) |

**Inference:**
- Complaint-cluster bootstrap with 2,000 resamples and within-complaint permutation with 2,000 permutations; seed 20261019.
- Benjamini–Hochberg correction within the H2 family (12 tests).
- alpha = 0.05.
- "No disparity" is never inferred from p > 0.05.

## 4. Hypotheses
| # | Hypothesis | Confirmed if |
|---|---|---|
| H1 | **Profiling vs arbitrariness.** The persona-specific share of within-complaint distributional variation is small. It is computed cross-bank as (TV between different-persona rewrites − TV between same-persona rewrites) / TV same-persona. | Upper 95% CI < 0.05 |
| H2 | **Directional persona effects.** Within complaints, old > young, college > high school and high SES > low SES on E[tier] and on the argmax tier, in each rewrite bank. That is 12 directional contrasts. Geography is reported without a directional prediction. | Each contrast: BH q < 0.05 and estimate > 0. Reported individually. |
| H3a | **Graded outputs reduce arbitrariness.** The residual share is lower for E[tier] than for the argmax tier, in both banks. | Lower 95% CI of the difference > 0 in both banks |
| H3b | **Graded outputs and systematic disparity.** Estimation only: D(argmax) − D(E[tier]) is reported with a 95% CI. No directional claim is made. | — |
| H4 | **Probability averaging is not superior to equal-budget hard voting** (ties broken to the lower tier). Compared on D for (a) the two same-style paraphrases and (b) all six option orders within each bank. | Prediction refuted if, for any comparison, the upper bound of the 90% CI of D_avg / D_vote is < 1. Equivalence within ±20% is reported as secondary. |
| H5 | **Uncertainty referral creates disparity.** Referring the 20% highest-entropy guises gives a persona-patterned referral rate in the gpt-5-mini bank. The gpt-4o-mini bank is reported descriptively. | Permutation p < 0.05 for the persona SD of referral |
| H6 | **Jev vs GPT at matched monetary volume.** Jev's monetary decisions use a p2 threshold set to match each GPT model's monetary rate on the same guises. D(GPT) − D(Jev) > 0 for gpt-4o-mini, and not > 0 for gpt-5-mini. Requires holdout GPT decisions; skipped if they are not collected. | 95% CI excludes 0 for gpt-4o-mini and does not for gpt-5-mini |

## 5. Secondary and exploratory analyses (not confirmatory)
- Directional effects on p2 and entropy.
- Hidden-variation summaries: complaints with a constant argmax, the range of p2 within them, and mean-invisible distributional differences.
- Option-order effects and their interaction with persona.
- The repeat-scoring determinism check.
- The structured-facts arm.
- Persona-recovery (manipulation) checks.
- Results on the exploratory panel.

## 6. Known limitations, stated in advance
- **No ground truth for remedy correctness.** Claims concern sensitivity and disparity, not accuracy or calibration.
- **Single System One model and version** (Jev 1.13). The comparison is between decision interfaces as deployed; it does not attribute differences to architecture.
- **Power.** With a 1,000-complaint holdout, the smallest exploratory effects may not reach significance, notably the age effect on the gpt-5-mini bank's argmax tier. A non-significant result will be reported as inconclusive, not as absence.
