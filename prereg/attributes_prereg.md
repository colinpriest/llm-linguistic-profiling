# Pre-registration 2: interpretable attributes behind persona effects (delivery / structure / content)

**Frozen:** 20 October 2026, before any attribute scores exist. This file, `jev_attributes.py` (question wording) and `prereg/attributes_pilot_analysis.py` (analysis and go/no-go rule) are committed together and will not be edited afterwards. Deviations will be reported.

## 1. Aim
Explain *why* writing style moves remedy decisions. We score every guise on interpretable attributes in three pre-specified families, then ask four questions.

**Families, with normative relevance to the remedy decision fixed in advance:**

| Family | Meaning | Normative relevance | Attributes |
|---|---|---|---|
| Delivery | how it is said | **irrelevant** | vocabulary sophistication, grammar/spelling, formality, confidence/assertiveness, politeness, rudeness/insults/hostility, emotional intensity, entitlement/demandingness; plus objective sentence length, Flesch–Kincaid grade, \|VADER compound\| |
| Structure | how it is organised | **procedurally relevant only:** it aids assessment but should not change merit for the same facts | logical coherence, organisation/chronology, argument quality, clarity of the request, concision |
| Content | what is said | **relevant** | evidence/documentation cited, severity of financial harm, clarity of the alleged bank error, explicit monetary demand, legal/regulatory basis cited, prior attempts to resolve |

Because rewrites are meant to preserve facts, **content attributes should be invariant across guises.** Content movement measures fact leakage.

Each Jev attribute is a 3-level choice question (wording in `jev_attributes.py`). The expected score is E = 0.5·P(medium) + P(high).

## 2. Questions
| | Question | Analysis |
|---|---|---|
| **Q-A** | Manipulation and fidelity: which attributes do persona rewrites change? | Within-complaint persona η² per attribute and standardised contrast effects (old−young, college−high school, high−low SES, urban−rural), per rewrite bank. Expectation: delivery and structure move, content does not. |
| **Q-B** | Persona inference: which attributes do the LLMs use to decide who is writing? | Within-complaint regressions of Jev's persona estimates and **GPT's own persona estimates** (cached Step 4 estimates on the same guises) on all attributes. Standardised betas; R² by family alone. GPT's self-reported linguistic markers are summarised descriptively. |
| **Q-C** | Certainty: which attributes is Jev most certain about, and is that certainty reliable? | Mean normalised entropy per attribute, plus test-retest Spearman and choice agreement on 500 re-scored texts. |
| **Q-D** | Outcomes and relevance: which attributes affect decisions, and are they decision-relevant? | Within-complaint regressions of GPT tier, GPT monetary (the primary, independent outcomes) and Jev E[tier] (secondary; shares a model with the attribute scorer). Order-invariant Gelbach decomposition of the age, education and SES gaps into attribute and family contributions, with complaint-bootstrap CIs. **Relevance:** the share of each explained gap carried by delivery (irrelevant) vs content (relevant) attributes. |

## 3. Pilot (exploratory, go/no-go)
- **Sample:** the same 200 complaints as the original Jev pilot (`random.Random(42).sample` of the sorted matched-guise case_ids), originals plus all guises in both banks. That is 14,600 texts (13,930 unique), about $2.2 of Jev. Test-retest: 500 random pilot texts, about $0.10.
- **Go** if all three hold:
  - **G1 (discrimination):** at least 10 of the 19 Jev attributes have mean normalised entropy < 0.90 and within-complaint SD of E ≥ 0.05.
  - **G2 (manipulation):** at least 3 delivery or structure measures have within-complaint persona η² ≥ 0.05 in at least one bank.
  - **G3 (explanation):** for GPT-4o-mini tier, attributes explain ≥ 25% of the education or SES gap (Gelbach total / raw gap).
- Attributes with test-retest Spearman < 0.50 are dropped from the confirmatory phase.

## 4. Confirmatory phase (only if GO)
Before any holdout attribute scores exist, a short amendment will fix the confirmatory hypotheses, derived from the pilot (directions, families, minimum shares). The holdout (1,000 complaints, about $12 of Jev) is then scored and analysed once with a frozen script.

## 5. Caveats stated in advance
- The decomposition is explanatory, not causal: attributes are measured, not manipulated, and there may be unmeasured style channels. The matched-guise design holds facts fixed only to the extent that rewrites preserve them, which Q-A's content check measures.
- Jev both scores attributes and makes one of the outcomes, so Jev-on-Jev results may be inflated by shared representations. GPT outcomes are primary for this reason.
- Attribute judgments may themselves be biased by persona (e.g. "rural style reads as less logical"). Agreement with the objective measures is reported, and divergence is treated as a finding about the instrument.

## 6. Pipeline smoke test (disclosed)
Before freezing, the scoring and analysis code was run once on the first 300 pilot texts (5 complaints, $0.04) to catch code errors. This was a code check only: the outputs were inspected for crashes and empty merges, not for substantive results, and no wording, threshold or analysis choice was changed afterwards. Those 281 responses are reused by the full pilot. One observation, made while checking why some regressions returned NaN: GPT's cached persona estimates vary very little across guises. On a 300-guise sample, GPT-4o-mini labels 96% "young", 92% "college" and 96% "middle" SES. Within-complaint regressions for a target with no within-complaint variation therefore return NaN by design.
