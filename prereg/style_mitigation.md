# Pre-registration 4: style standardisation as a mitigation

**Frozen:** 21 October 2026, before any analysis-set text is standardised. This file and `prereg/style_mitigation.py` (prompts, pipeline and frozen analysis) are committed together.

## Question
Persona-associated *delivery* drives GPT-4o-mini's education and SES gaps (H-A2), and the gaps appear only when the style cue is recognisable (H-A3). Does rewriting every complaint into a **standard style** before deciding:
- remove that profiling,
- reduce arbitrariness, and
- preserve sensitivity to remedy-relevant facts?

And does averaging over **several standard styles** improve on a single style?

## Design
- **Sample:** the first 500 complaints of the registered holdout draw × 36 persona guises from the GPT-4o-mini rewrite bank (18,000 guises).
- **Arms:**

| Arm | What is decided | Cost |
|---|---|---|
| M0 | the guise as written | none; existing decisions and Jev scores |
| M1 | the guise rewritten by gpt-4o-mini (temperature 0) into one canonical style, A (plain, neutral, polite, chronological) | new rewrites and decisions |
| M2 | the guise rewritten into three standard styles (A plain; B formal business; C simple everyday), each decided, outcomes averaged | new rewrites and decisions |

  Every rewrite is instructed to preserve every fact and change only the writing (prompts are in the script).
- **Deciders:** GPT-4o-mini (the same remedy prompt and client as the original experiment), giving the tier; and Jev (6 option orders), giving E[tier] from order o012. Jev also scores the 19 attributes on all M1 texts.
- **Fact sensitivity:** the first 200 complaints' positive-control texts (`strengthen` and `weaken` edits, from pre-registration 3) are standardised with the same three styles and decided.
- **Code check:** the full pipeline was run on holdout complaints 501–505 (outside the analysis set, about $0.30). That run fixed three code-only bugs: a pandas `.style` name clash, missing output folders, and a column name. No analysis-set text was standardised before this commit.
- **Cost estimate:** about $45–55. That covers about 56k rewrites (about $20), about 56k GPT-4o-mini decisions (about $16) and Jev scoring of about 56k texts (about $10–15).

## Hypotheses
Complaint-block bootstrap, 2,000 resamples, seed 20261025.

| # | Hypothesis | Supported if |
|---|---|---|
| **MH1** | **M1 removes GPT-4o-mini's profiling.** For the education (college − high school) and SES (high − low) tier gaps. | The 95% CI of (gap_M0 − gap_M1) is > 0 **and** \|gap_M1\| ≤ 0.5·\|gap_M0\|, for both dimensions. Age, Jev's gaps and the M2 gaps are reported. |
| **MH2** | **M1 reduces arbitrariness.** Arbitrariness is the per-complaint mean absolute pairwise difference in outcome across the 36 guises. | The 95% CI of (MPAD_M0 − MPAD_M1) is > 0, for both GPT-4o-mini tier and Jev E[tier]. |
| **MH3** | **Standardisation preserves fact sensitivity.** Retention = fact span (strengthen − weaken) after standardisation ÷ fact span without it. | The lower 95% bound of retention is ≥ 0.5, for GPT and Jev, under both M1 and M2. |
| **MH4** | **Fidelity** (descriptive). | Under M1, content attributes keep their complaint-level values (correlation with M0) and persona η² stays about 0. Delivery attributes' persona η² should collapse relative to M0. |
| **MH5** | **Averaging styles reduces arbitrariness further.** | The 95% CI of (MPAD_M1 − MPAD_M2) is > 0, for GPT and Jev. Directional; reported for both systems. |

## Caveats stated in advance
- The standardiser is itself an LLM, so it may leak persona cues or alter facts. MH4 and MH3 measure both.
- M2 costs about 3× M1 at inference.
- M0 decisions come from the 19 Oct run, while M1/M2 are decided now with the same model name and settings. Model drift would bias M0 vs M1 comparisons, so it is disclosed.
