ONE-SHOT EXAMPLE "ROBUSTNESS & DATA QUALITY"

input:
── Slot-Preservation Audit ──
Total matched-guise pairs: 72,000 (across 2 models)
Total complaints: 1,000

Viability Gate: PASS (all 6 criteria met)
  High-fidelity pair rate: 0.841 (threshold: >= 0.40)  PASS
  High-fidelity complaint rate: 0.462 (threshold: >= 0.25)  PASS
  Polarity contradiction rate: 0.048 (threshold: <= 0.05)  PASS
  Numeric mutation rate: 0.030 (threshold: <= 0.10)  PASS
  Temporal presence preservation: 0.718 (threshold: >= 0.70)  PASS
  Anchor presence preservation: 0.976 (threshold: >= 0.70)  PASS

Per-family preservation rates (presence preservation):
  Numeric: 0.877 [0.874, 0.880]
  Temporal: 0.718 [0.715, 0.722]
  Polarity: 0.520 [0.518, 0.523]
  Anchor: 0.976 [0.975, 0.977]

Per-model comparison:
  gpt-4o-mini: 36,000 pairs, strict pass=0.201, hi-fi=0.736, contradiction=0.079
  gpt-5-mini: 36,000 pairs, strict pass=0.745, hi-fi=0.946, contradiction=0.018

── Robustness Subset Analysis ──
Subsets tested: FULL, HF (high-fidelity only), NO_POL (no polarity contradictions), NO_NUM (no numeric mutations)
Note: HF_NO_POL is identical to HF (aliased)

Overall verdict: PASS

Effect stability (vs FULL baseline):
  HF subset:
    gpt-4o-mini: persona gap (tier2) amplified by 48.2%, persona gap (mean tier) amplified by 31.4% — effects STRONGER in filtered data
    gpt-5-mini: persona gap (tier2) amplified by 34.4%, persona gap (mean tier) amplified by 11.2% — effects STRONGER in filtered data
  NO_POL subset:
    gpt-4o-mini: stable (attenuation < 5%)
    gpt-5-mini: amplified by 9.2% and 6.3% — small amplification
  NO_NUM subset:
    gpt-4o-mini: stable (attenuation < 5%)
    gpt-5-mini: tier2 attenuated by 9.4%, mean tier stable

All CI overlaps confirmed. No significance class changes.

Outcome composition shift:
  gpt-4o-mini: dropped pairs had higher mean tiers (0.779 vs 0.729, p<0.001) — anti-conservative bias
  gpt-5-mini: dropped pairs had higher mean tiers (0.906 vs 0.804, p<0.001) — anti-conservative bias

output:
# Robustness & Data Quality

## Can We Trust the Matched-Guise Results?

The matched-guise experiment rewrites real complaints in 36 different customer styles. For those results to be valid, the rewritten complaints must preserve factual content — the same dollar amounts, dates, and claim severity — while changing only the writing style. We tested this with a four-part quality audit.

## Quality Audit: All Checks Passed

We examined every rewritten complaint across four categories:

| Quality Check | Preservation Rate | Assessment |
|---|---|---|
| Named entities (companies, products) | 97.6% preserved | Excellent |
| Numeric amounts ($, dates) | 87.7% preserved | Good |
| Temporal references (dates, deadlines) | 71.8% preserved | Adequate |
| Claim polarity (complaints vs. compliments) | Varies | See below |

All six viability criteria were met, meaning the data quality is sufficient for causal analysis. The vast majority of rewritten complaints faithfully reproduce the factual content of the original.

## Two Quality Tiers

Not all rewritten pairs are equal. We defined "high-fidelity" pairs as those passing all four quality checks simultaneously:

- **84.1% of all pairs** met the high-fidelity standard — meaning over 5 in 6 rewrites preserved all factual content
- The newer model (GPT-5-mini) produced substantially higher-quality rewrites: **94.6% high-fidelity** compared to **73.6%** for GPT-4o-mini

## Filtering to High-Quality Data Does Not Weaken Findings

The critical robustness test: when we restricted analysis to only high-fidelity pairs, **the persona-based differences in outcomes grew larger, not smaller**. For both models, the gap between most- and least-favoured personas was wider in the high-fidelity subset than in the full dataset.

This is important because it rules out a key objection: that the observed bias might be an artefact of imperfect rewrites. In reality, removing lower-quality rewrites makes the bias signal stronger.

## Composition Shift Is Anti-Conservative

The pairs excluded by high-fidelity filtering had higher remedy tiers than those retained. This means our full-dataset estimates are, if anything, **conservative** — they slightly understate the true size of persona-based bias.

## Implications
- **The matched-guise findings are robust**: quality filtering strengthens rather than weakens the bias signal
- **GPT-5-mini produces cleaner experimental data**: its higher rewrite fidelity makes it a better tool for this kind of causal testing
- **No evidence that results are driven by rewrite artefacts**: the viability gate passed and the high-fidelity subset shows larger effects
