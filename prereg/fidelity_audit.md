# Pre-registration 5: independent fact-fidelity audit and same-day baseline

**Frozen:** 22 October 2026, before any analysis-set pair is audited and before any baseline decision is made. This file and `prereg/fidelity_audit.py` (sample, auditor prompt and schema, baseline pipeline, frozen analysis) are committed together. The code was checked on holdout complaints 501–505, which are outside every analysis set.

## Why
An adversarial review (21 Oct) raised two points this study cannot answer from existing data.
1. **Fact preservation.** Low persona variance in six content attributes, scored by Jev, does not show that individual rewrites kept the facts. If rewrites sometimes change facts, part of the individual decision instability attributed to arbitrariness could be a legitimate response to changed facts.
2. **Drift.** M0 decisions were made on 19 Oct and M1/M2 on 21 Oct, so provider drift could bias the mitigation contrasts.

## Design
### A. Fact-fidelity audit
**Auditors.** Two LLMs from outside the audited systems:
- Claude Opus 5.5 (Anthropic; effort low);
- GPT-5.2 (OpenAI; reasoning effort low).

Neither is GPT-4o-mini, GPT-5-mini or Jev, and neither sees any decision. GPT-5.2 is from the same provider as two audited models; this is disclosed. A Google model was planned but the available key was invalid.

**Task.** Each auditor compares the original (A) with the rewrite (B). It lists every factual difference with a category, a kind (added / omitted / changed) and whether it is material, and gives an overall verdict (preserved / minor / material). The categories are: amount, date or time, chronology, responsibility, negation, certainty, requested relief, event or entity, and other. A difference is *material* if a careful complaints handler could reasonably choose a different remedy because of it. The prompt and the JSON schema are in the script.

**Exact code checks.** Every dollar amount and four-digit year is extracted from both texts and compared as sets.

**Sample** (seed 20261022; GPT-4o-mini rewrite bank; first 500 holdout complaints):

| Set | What is compared | Sample | Weights |
|---|---|---|---|
| `guise` | original vs persona guise | 40 complaints × all 36 guises (1,440 pairs): 20 complaints whose GPT-4o-mini monetary decision is unanimous across guises, 20 where it is not | Stratum population share |
| `std` | guise vs its standardised (style A) rewrite | 150 pairs: 100 where GPT-4o-mini's tier did not change between M0 and M1, 50 where it did | Stratum population share |
| `edit` | original vs fact edit (positive control) | 50 `strengthen` + 50 `weaken` from the first 200 complaints | — |

**Cost.** The smoke run cost about $0.0125 per pair for both auditors together, so the 1,690 pairs are expected to cost about $21. The auditor step stops automatically at US$45 of estimated cost. The baseline is expected to cost about $2–3. All of this is within the $55 Colin approved on 21 Oct.

### B. Same-day baseline
- **Sample.** 100 complaints drawn at random from the first 500 holdout complaints (seed 20261023), each with all 36 guises in two arms: as written (M0) and standardised into style A (M1). That is 7,200 texts.
- **Fresh decisions, interleaved.** The arms are shuffled together and decided in one run:
  - by GPT-4o-mini, with the same prompt, client and settings as the original runs but a fresh cache, so every call is new;
  - by Jev, on the tier question under all six option orders.

## Hypotheses and pre-specified analyses
Inference uses a complaint-cluster bootstrap with 2,000 resamples (seed 20261022). Weighted estimates use the stratum weights above.

| # | Question | Pre-specified criterion / output |
|---|---|---|
| **FA1** | Auditor validity: do the auditors detect real fact changes? | An auditor is *valid* if it rates ≥ 80% of `strengthen` and `weaken` edits as material. Later steps use only valid auditors; if neither is valid, the audit is reported as inconclusive. |
| **FA2** | How often do persona guises change facts materially? | Weighted rate of material change (each auditor; either valid auditor; both), plus code-check mismatch rates and the categories involved. Estimation, no threshold. |
| **FA3** | Does individual instability survive among *verified* fact-preserving guises? | A guise is *verified* if no valid auditor rates it material and the dollar amounts match. Within each sampled complaint, compute the pairwise monetary-decision disagreement among all 36 guises and among verified guises only, for GPT-4o-mini (tier 2) and Jev (argmax tier 2, order o012); also report the MPAD of tier and E[tier]. **Supported** (instability is not mainly due to fact changes) if the weighted verified/all ratio is ≥ 0.5 for both GPT-4o-mini and Jev monetary disagreement. CIs are reported. |
| **FA4** | Are guises whose GPT-4o-mini monetary decision departs from the complaint's majority more often flagged? | Descriptive: the flag rate for departing vs agreeing guises. |
| **FA5** | Do the standardising rewrites change facts? | Weighted material-change rate for the `std` set (as for FA2). |
| **FA6** | Agreement | Raw agreement and Cohen's κ between the two auditors on "material"; the share of code-check mismatches that the auditors rate material. |
| **DB1** | Drift | Fresh vs original decisions on identical texts: GPT-4o-mini exact agreement and mean tier difference (M0 vs 19 Oct; M1 vs 21 Oct), and Jev mean TV and E[tier] difference. Reported alongside Jev's known rescoring noise (TV 0.021). |
| **DB2** | Is MH1 robust to drift? | On the 100 complaints, re-estimate the M0 − M1 education, SES and age gap reductions and the MPAD difference using all-fresh decisions, and with the original decisions for comparison. **Robust** if the fresh reduction CI is above 0 for both education and SES (GPT-4o-mini). Jev is reported. |

## Caveats stated in advance
- Two LLM auditors are not a human gold standard. FA1 tests their sensitivity to known fact edits, and FA6 their agreement. No human annotation was done.
- The audit covers the GPT-4o-mini rewrite bank only, and 40 complaints for FA3. The GPT-5-mini bank is not audited.
- Over-flagging by an auditor makes FA3 conservative: fewer guises count as verified, and the verified-only disagreement is estimated on fewer pairs.
- DB2 uses 100 of the 500 mitigation complaints, so its precision is lower than the main MH1 test.
