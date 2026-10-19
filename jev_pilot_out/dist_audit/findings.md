# Do probabilities introduce, or help remove, style-linked bias? (Jev, exploratory)

> **Correction (19 Oct, 3 pm AEST).** The "persona SD / between-complaint SD" metric used in the Q2 table below includes a sampling-noise floor that is larger for noisier rules. With the noise-corrected ANOVA estimator (prereg/confirmatory_analysis.py, `D`), E[tier] does **not** reliably reduce systematic persona disparity:
> - gpt-4o-mini bank: 0.0265 vs 0.0295 for argmax, CI spanning 0.
> - gpt-5-mini bank: 0.0112 vs 0.0091.
>
> What it reduces is arbitrariness: the residual share falls from 0.20 to 0.05 and from 0.13 to 0.02. Probability averaging vs hard voting remains a tie (0.0213 vs 0.0219). The "7–14% / about 32% reduction" statements below are superseded. See jev_pilot_out/confirmatory_results.json for the corrected exploratory numbers.

Source: jev_pilot_out/jev_scores.csv (2,000 complaints x 36 guises x 2 rewrite banks, jev-1.13.0, two option orders per text) plus frozen GPT decisions. No new API calls. Script: jev_pilot_out/jev_distributional_audit.py; full numbers: dist_audit.json. Complaint-cluster bootstrap (1,000) and within-complaint permutation (1,000). **Exploratory only: this panel has already been inspected, so confirmatory claims need fresh complaints.**

## Q1. Do probabilities introduce new forms of bias?

**1. Single predictions hide variation, but almost none of it is persona-specific.**
- Jev's argmax is identical across all 36 guises for 64% (4o-mini bank) / 72% (5-mini bank) of complaints; the GPT tier is constant for only 49% / 36%.
- Inside those "invariant" complaints, p2 still ranges by 0.064 / 0.052 on average (by more than 0.10 in 26% / 22% of them), and normalised entropy ranges by 0.27 / 0.19.
- Two controls show what that variation is:

| Comparison | Mean TV |
|---|---|
| Same text, options reversed (no style change) | 0.055 |
| Cross-bank, same persona (two paraphrases, same style) | 0.0910 |
| Cross-bank, different persona | 0.0916 |
| Cross-bank, one dimension changed (age / geo / edu / ses) | 0.0915 / 0.0910 / 0.0915 / 0.0912 |

  Changing the persona adds about 0.0007 TV on top of generic paraphrase sensitivity. The hidden distributional variation is overwhelmingly **instability** (rewrite and option-order sensitivity), not systematic persona bias. That is a fairness-relevant arbitrariness finding, but it is not linguistic profiling.
- The "same mode, same mean, different distribution" case from the critique (|delta E[tier]| < 0.05 with TV > 0.15) occurs in 0.14% / 0.04% of guise pairs. With three ordered tiers and a small p2, Jev's distributions are close to one-dimensional, so expected tier captures almost all of the distributional difference in practice.

**2. Systematic persona effects on probabilities are real but small, and point the same way as the hard labels.**
- p2 (monetary probability) rises with older, college-educated and higher-SES guises. Gaps are 0.001–0.002 absolute (on a mean of about 0.09–0.10), with bootstrap CIs excluding 0 for edu and SES. These are the same directions GPT shows, but much smaller.
- Uncertainty is barely persona-patterned. Entropy and max-probability gaps are at most 0.0016, and most CIs include 0. The exception is a 0.0013 entropy gap by SES in the 5-mini bank.

**3. Policies that use probabilities mostly reshuffle the disparity rather than create a new one.**
- Monetary decision by p2 threshold at the **same monetary budget** as argmax:
  - 4o-mini bank: persona SD 0.0039 vs 0.0044 for argmax; edu and SES gaps roughly halve.
  - 5-mini bank: 0.0032 vs 0.0029; edu gap 0.0028 vs 0.0016, SES gap 0.0017 vs 0.0031.
  - Net: no consistent gain or loss, only a shift in which dimension carries the disparity.
- **Uncertainty-based referral** (refer the most uncertain cases):
  - Top 10%: no detectable persona disparity in either bank (perm p = 0.34 / 0.64).
  - Top 20%, 5-mini bank: detectable but small (p = 0.023). High-SES and college guises are referred about 0.4–0.5 percentage points more often.
  - Referral is a candidate channel for new disparity, but it is weak here.
- Compared with GPT at **GPT's own monetary volume**, as standardised persona SD (divided by the between-complaint SD):
  - 4o-mini bank: Jev 0.016 vs GPT-4o-mini 0.027 (SES gap 0.003 vs 0.0105 absolute).
  - 5-mini bank: Jev 0.012 vs GPT-5-mini 0.013, and GPT-5-mini's own monetary disparity is not significant (p = 0.62) while Jev's is (p = 0.001).
  - So Jev is not uniformly "fairer": that depends on the comparator.

**Answer to Q1 (provisional):** probabilities surface a lot of variation that single labels hide, but it is mostly arbitrariness from paraphrase and option order, not persona-specific bias. The persona-specific parts of p2 and of uncertainty are small. Threshold and referral policies built on them neither clearly add nor clearly remove disparity at matched budgets; they move it between dimensions. The clearest probability-specific risk is uncertainty referral at larger budgets, which needs confirmation.

## Q2. Do probabilities open opportunities for reducing bias?

The scale-free metric is the persona SD divided by the between-complaint SD. It is lower when a rule is more invariant **and still discriminates between complaints**, so a constant output does not score well.

| Rule (inference calls) | 4o-mini bank | 5-mini bank |
|---|---|---|
| argmax, 1 call | 0.0315 | 0.0127 |
| E[tier], 1 call | 0.0271 | 0.0118 |
| argmax of order-averaged probs, 2 calls | 0.0344 | 0.0135 |
| hard vote of the 2 orders, 2 calls | 0.0318 | 0.0137 |
| E[tier] order-averaged, 2 calls | 0.0264 | 0.0119 |

Pooling both banks, with two same-style paraphrases per guise:

| Rule (inference calls) | Pooled score |
|---|---|
| argmax of averaged probs, 2 calls | 0.0229 |
| hard vote, 2 calls | 0.0240 |
| mean of the 2 hard labels, 2 calls | 0.0198 |
| argmax of 4-way averaged probs, 4 calls | 0.0249 |
| hard vote of 4 labels, 4 calls | 0.0256 |
| E[tier] 4-way averaged, 4 calls | 0.0173 |

- **Averaging probabilities and then taking argmax is no better than hard voting at the same budget.** The ensembling, not the probabilities, does the work, as the critique predicted.
- **Using the continuous output (E[tier]) instead of a hard tier helps.** It cuts the within-complaint share of variance from 0.20 to 0.06 (4o-mini) and from 0.14 to 0.03 (5-mini). Standardised persona disparity falls by about 7–14% at 1 call and about 32% at 4 calls (0.0173 vs 0.0256), while keeping between-complaint signal.
- The systematic age, edu and SES gaps survive every rule (CIs exclude 0). Nothing here removes them.
- **Option-order averaging fixes position bias (the mean shift of +0.057 E[tier]) but not persona bias.**
- **Neutralising the originals moves Jev more than persona style does.** Mean TV is 0.090 / 0.093, and the argmax changes on 12% / 14% of complaints, versus about 0.05–0.07 TV between guises. Guises were never neutralised, so the matched-guise neutralisation test still needs to be run.

**Answer to Q2 (provisional):** yes, but modestly. Graded outputs plus several rewrites reduce style sensitivity relative to signal by up to about a third, and they reduce arbitrariness much more. Neither removes the systematic persona gap, and the benefit over equal-budget hard ensembling comes from using a graded decision, not from averaging the probabilities.

## What this means for the critique
- It is right that "probabilities" bundles measurement, policy and model differences. The data support keeping them separate.
- Its hypothetical (same mode and mean, different distribution) is mathematically valid but empirically rare for Jev on this task. TV and p2 endpoints are worth reporting, but E[tier] already carries most of the distributional signal.
- Its claim that Jev shows hard-label style effects is confirmed at n = 2,000 (argmax persona SD p = 0.001 in both banks).
- The strongest new finding is in the paraphrase control: about 99% of within-complaint distributional variation is generic rewrite sensitivity. That supports making paraphrase and option-order controls central.

## Cheapest next steps (confirmatory, on fresh complaints)
1. **Repeat-scoring control** (same text twice): a few cents. It establishes whether Jev is deterministic.
2. **All 6 option orders** in one call per text: about +$5–8 for 150k texts.
3. **Structured-facts arm:** fact sets for every guise already exist in cache (Step 8 fact_extraction, about 193k entries), so there is no OpenAI cost. Render each fact set as canonical text and score it with Jev (at most about $10, less after deduplication). This also measures directly whether style changes the extracted facts.
4. **Neutralise every guise:** about 144k OpenAI calls plus about $10 of Jev. This is the only expensive item.
5. **An adjudicated subset and fact-changing positive controls** are still needed before any claim about decision quality.
