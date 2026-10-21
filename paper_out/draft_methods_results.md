# Profiling or Arbitrariness? Auditing Style Sensitivity in Probabilistic Judgments of Consumer Complaints

*Draft, Methods and Results only (21 Oct 2026). Every number comes from committed result files; see `prereg/README.md` for the claim-to-commit map. Labels: **[C]** confirmatory (pre-registered, tested once on held-out data); **[S]** pre-registered secondary; **[PH]** post hoc; **[E]** exploratory. `[CITE]` marks references to add and verify.*

---

## 3. Methods

### 3.1 Task and decision systems
We study the assignment of remedies to US consumer financial complaints from the CFPB public complaint database [CITE]. There are three ordered tiers: 0 = no action, 1 = non-monetary action, 2 = monetary remedy. Two kinds of decision interface are compared, as each is typically deployed:

- **Generative LLMs that return a single label.** GPT-4o-mini (temperature 0) and GPT-5-mini (default reasoning). Both use the remedy prompt and structured-output client of a prior matched-guise study [CITE, third person].
- **A probabilistic "System One" model.** Jev 1.13.0 (TypeSafe), which returns a probability distribution over the three tiers for a multiple-choice question whose options carry the same tier definitions.

From Jev's distribution p = (p0, p1, p2) we derive:
- the expected tier, E[tier] = p1 + 2·p2;
- the argmax tier;
- the monetary probability, p2;
- the normalised entropy.

Each text is scored under all six orderings of the three options; order o012 is primary. We make no causal claim about model architecture: the comparison is between interfaces as deployed.

### 3.2 Matched-guise panels
- **Complaints.** Complaints were drawn from a cleaned pool of 20,000 narratives (the prior study's pipeline).
- **Guises.** Each complaint was rewritten into 36 sociolinguistic guises: age (young / old) × geography (urban / suburban / rural) × education (high school / college) × socioeconomic status (low / middle / high). The rewrites were made independently by two models (the GPT-4o-mini bank and the GPT-5-mini bank), with an instruction to preserve every fact.
- **Exploratory panel.** 2,000 complaints (144,000 guises), analysed first and reported as exploratory.
- **Holdout panel.** 1,000 fresh complaints that had never been rewritten or scored (72,000 guises). The sample was drawn with a fixed seed and fingerprinted (SHA-256) before any holdout text existed. It was generated with identical prompts and settings, and the frozen confirmatory analysis was run on it once.
- **GPT decisions.** Each GPT model decided the guises in its own bank. GPT decisions on the original narratives come from the prior study's frozen run.

### 3.3 Controls for arbitrariness
Three non-persona sources of variation are measured directly:
1. **Inference noise:** 1,000 identical texts were re-scored.
2. **Option order:** all six orderings.
3. **Paraphrase:** the two rewrite banks give two independent, same-persona paraphrases of every guise.

Cross-bank comparisons isolate persona-specific variation. The persona-specific share is the difference between cross-bank total-variation (TV) distance for different-persona pairs and for same-persona pairs, divided by the same-persona TV.

### 3.4 Disparity and arbitrariness measures
- **Systematic persona disparity (D).** For a complaint × persona matrix of outcomes, we use a two-way additive decomposition. D is the noise-corrected SD of the 36 persona effects divided by the noise-corrected SD of complaint effects (unbiased ANOVA estimators). D does not reward a constant output and does not penalise a rule merely for being noisier.
- **Arbitrariness.** The residual (complaint × persona interaction) variance share, plus individual-level disagreement rates.
- **Directional contrasts.** Within-complaint differences between persona levels (old − young, college − high school, high − low SES), in tiers or in percentage points of the monetary-remedy rate.
- **Materiality [PH].** Adverse impact ratios (AIR) and Cohen's h. Equivalence (TOST) is tested against AIR > 0.80, |h| < 0.20 and |gap| < 1 pp, alongside minimum-effect tests.

### 3.5 Interpretable attributes
Jev also scored every guise on 19 attributes (3-level choice questions; expected score in [0, 1]), plus three objective measures (mean sentence length, Flesch–Kincaid grade, |VADER compound|) [CITE]. The attributes fall into three families, whose normative relevance to the remedy decision was fixed in advance:

| Family | Relevance | Attributes |
|---|---|---|
| Delivery (how it is said) | irrelevant | vocabulary, grammar/spelling, formality, confidence, politeness, rudeness/insults, emotional intensity, entitlement, plus the objective measures |
| Structure (how it is organised) | procedurally relevant only | coherence, chronology, argument quality, clarity of the request, concision |
| Content (what is said) | relevant; should not change under fact-preserving rewrites | evidence cited, harm severity, alleged bank error, monetary demand, legal basis, prior attempts |

**Decomposition.** Within-complaint persona gaps are decomposed into attribute and family contributions with Gelbach's order-invariant decomposition [CITE].

**Recognition.** Moderation by recognisability uses Jev's probability of the intended persona level; a guise counts as "recognised" if that probability is ≥ 0.60.

**Pilot.** A 200-complaint pilot with pre-specified go/no-go rules preceded the confirmatory holdout run.

### 3.6 Positive controls
For the first 200 holdout complaints, gpt-4o-mini produced three fact edits, each instructed to keep wording, tone and errors unchanged:
- `strengthen`: adds a documented bank error with a dollar loss;
- `weaken`: the problem becomes the writer's own mistake or agreed terms;
- `harm_x10`: dollar amounts are multiplied by 10.

All three systems decided or scored the edits. Effects are paired edit-minus-original differences.

### 3.7 Inference and pre-registration
- **Resampling.** Complaint-cluster (block) bootstrap and within-complaint permutation. Benjamini–Hochberg correction within the directional family.
- **Pre-registration.** Three pre-registrations and two amendments were committed to version control before the data they test existed. They cover the main hypotheses (H1–H6), the attribute hypotheses (H-A1–H-A5) and the positive controls (PC1–PC5).
- **Planned extension, not run.** A pre-registered extension of the holdout to 4,000 complaints for H6 was not executed after a materiality review, before any new data were generated.
- **Disclosures.** All deviations are disclosed: two code-only fixes and one amendment requested after a freeze but committed before any result was inspected.
- **Bootstrap correction.** The pre-registered bootstrap of D under-corrected the noise term in resamples with repeated complaints. We report a corrected bootstrap (noise term s²·Σw²/C²) as a sensitivity analysis; no conclusion changes.

---

## 4. Results

### 4.1 Single labels hide variation, but almost none of it is persona-specific
- **Hidden variation [S].** On the holdout, Jev's argmax tier is identical across all 36 guises for 63% of complaints (GPT-4o-mini bank) and 74% (GPT-5-mini bank). Yet within those complaints the monetary probability still varies, by more than 0.10 in about a quarter of them.
- **What drives it (Fig. 1).**

| Comparison | Mean TV |
|---|---|
| Same text, scored twice | 0.021 (tier flips in 2.5% of re-scores) |
| Same text, options reversed | 0.058 |
| Two paraphrases, same persona | 0.091 |
| Two paraphrases, different persona | 0.092 |

- **H1 [C], supported.** The persona-specific share is 1.1% (95% CI 0.7–1.5%), far below the pre-registered 5% bound. About 99% of the style-induced variation in Jev's distributions is arbitrariness (inference noise, option order, paraphrase), not profiling.

### 4.2 Systematic persona effects are real but immaterial
- **H2 [C]: 9 of 12 directional contrasts supported.** Older, college-educated and higher-SES styles receive slightly more remedy. On E[tier], all six contrasts hold in both banks; for example, in the GPT-4o-mini bank:
  - old − young: +0.014 tier [0.010, 0.019];
  - college − high school: +0.006 [0.003, 0.009];
  - high − low SES: +0.008 [0.004, 0.011].

  On the argmax tier, 3 of 6 hold. Geography shows no effect.
- **Materiality [PH].**
  - In monetary-rate units (Fig. 2), every gap for every system lies within ±1 percentage point.
  - Every AIR is above the four-fifths bound by equivalence test, and every |h| is below 0.2.
  - The largest disparity is GPT-4o-mini's SES gap (pooled n = 3,000): +0.84 pp [0.56, 1.11], AIR 0.917, h = 0.028. It is the only gap that cannot be bounded below 1 pp, and the only one that fails a stricter AIR > 0.90 bound.
- **Individual arbitrariness [PH].** It is far larger (Fig. 6). The probability that two style versions of the same complaint receive different monetary decisions is:
  - GPT-4o-mini: 4.1%;
  - GPT-5-mini: 6.3%;
  - Jev: 2.8–3.6%.

  Monetary outcomes are not unanimous across the 36 styles for 15.7%, 23.9% and 10.7–12.5% of complaints respectively.

### 4.3 What probabilistic outputs do and do not change
- **H3a [C], supported.** Using E[tier] instead of the argmax cuts arbitrariness: the residual share falls from 0.20 to 0.06 (GPT-4o-mini bank) and from 0.13 to 0.03 (GPT-5-mini bank).
- **H3b (estimation).** It does not reduce systematic disparity: D(argmax) − D(E[tier]) = −0.006, corrected 95% CI [−0.021, 0.005].
- **H4 [C], supported.** Averaging probabilities is not superior to equal-budget hard voting:
  - two paraphrases: D difference −0.005, 90% CI [−0.014, 0.005];
  - six option orders, GPT-4o-mini bank: +0.005 [0.000, 0.013], i.e. averaging is slightly worse.
- **H5 [C], not supported.** Referring the 20% most uncertain cases does not create a persona disparity (p = 0.32 in the primary bank). Jev's uncertainty itself is barely persona-patterned [S].
- **Option order [S/E].**
  - There is no general first-position bias. There is a small last-position penalty (−0.013).
  - The fully reversed order is an outlier: E[tier] 0.586 vs 0.510–0.554 for the others (exploratory).
  - The chosen tier is identical across all six orders for 85% of texts.
  - Order does not interact with education or SES effects.

### 4.4 Generative LLM vs System One
- **H6 [C], inconclusive at n = 1,000.** Jev's systematic monetary disparity is below GPT-4o-mini's at matched monetary volume: ΔD = 0.010, corrected 95% CI [−0.009, 0.023]. This is significant only in the pooled n = 3,000 estimate [PH]: 0.012 [0.002, 0.023].
- **In rate units [PH].** Jev's SES gap is 0.64 pp smaller than GPT-4o-mini's (pooled [0.27, 0.99]). It is 0.34 pp larger than GPT-5-mini's ([0.03, 0.67]). Every Jev-vs-GPT difference is equivalent to zero within ±1 pp (TOST p ≤ 0.03), and none exceeds 1 pp (minimum-effect tests, p ≥ 0.97).
- **Conclusion.** The three systems are practically equivalent in group disparity. A pre-registered extension that could have confirmed the GPT-4o-mini contrast was abandoned, because the difference would be immaterial.

### 4.5 Why style moves decisions: delivery, structure and content
- **H-A1 [C], supported: the rewrites preserve facts.** The persona explains at most 2.3% of within-complaint variance in each content attribute; every one-sided upper bound is ≤ 0.032.
- **What the persona rewrites do change (Fig. 5):**
  - delivery, strongly: vocabulary η² = 0.73, formality 0.71, grammar 0.58, politeness 0.37;
  - structure, moderately: chronology 0.26, argument quality 0.22.
- **H-A2 [C], supported: delivery drives GPT-4o-mini's education and SES gaps (Fig. 3).**

| Gap | Size | Explained by attributes | Delivery | Structure | Content |
|---|---|---|---|---|---|
| College − high school | +0.032 tier [0.025, 0.039] | 79% | +0.014 [0.007, 0.021] | +0.010 [0.005, 0.014] | +0.002 [0.000, 0.003] |
| High − low SES | +0.026 [0.018, 0.034] | 96% | +0.012 [0.005, 0.019] | +0.010 | +0.003 |

  Irrelevant delivery outweighs relevant content by roughly 5–8×. Individual delivery attributes are collinear, so attribution is reliable at family level, not attribute level.
- **H-A3 [C], supported: the bias appears only where the style cue is recognisable (Fig. 4).**
  - GPT-4o-mini's education gap where Jev recognises the cue: +0.053 [0.039, 0.068].
  - Where the cue is ambiguous: +0.009 [−0.004, 0.022].
  - Difference: +0.045 [0.025, 0.063].
- **H-A4 [C], supported: Jev infers who is writing from delivery, not content.** Within-complaint R² is 0.24–0.79 for delivery vs 0.01–0.05 for content. By contrast, GPT's own persona labels barely vary [E]: GPT-4o-mini labels 96% of guises "young" and 92% "college", yet its decisions respond to style.
- **H-A5 [C], supported: Jev's own remedy judgment tracks relevant content.** R² is 0.20–0.21 for content vs 0.04 for delivery. Jev also scores the attributes, so this is weaker evidence than H-A2.

### 4.6 Positive controls: the systems respond to facts that matter
- **PC1 and PC2 [C], supported (Fig. 7).**

| System | `strengthen` effect | Share moved up | `weaken` effect |
|---|---|---|---|
| GPT-4o-mini | +1.17 tier [1.08, 1.26] | 89% | −0.41 |
| GPT-5-mini | +0.89 [0.78, 1.00] | 69% | −0.39 |
| Jev E[tier] | +0.76 [0.70, 0.82] | 98% | −0.30 |

  All `weaken` CIs are below 0.
- **PC4 [C], supported.** The fact span (strengthen − weaken) is 49× (GPT-4o-mini), 387× (GPT-5-mini) and 75× (Jev) the largest persona gap of the same system. The pre-registered bar was 5×.
- **Graded output detects changes better.** Jev's graded E[tier] registered the strengthened facts in 98% of complaints, versus 69% for its argmax.
- **`harm_x10` [E].** It mainly moves the monetary dimension (Jev p2 +0.04; GPT monetary +0.06 to +0.12).
- **PC5 (descriptive).** Content attributes moved as intended:
  - `strengthen`: evidence +5.6 SD and bank error +3.9 SD (in units of within-complaint guise SD);
  - `weaken`: bank error −2.1 SD.

  The edits were fact-led but not fact-only. `weaken` also lowered perceived confidence and coherence, and `strengthen` raised argument quality.

---

*To do: intro, related work (arbitrariness/predictive multiplicity, matched-guise and dialect-bias audits of LLMs, probabilistic and calibrated judgment), discussion (implications for audit practice: separate profiling from arbitrariness; evaluate decision policies, not only scores; materiality), limitations (single System One model and version; no remedy ground truth; LLM-generated rewrites and edits; Jev non-determinism; GPT persona estimates only on the exploratory panel), and the ethics/adverse-impact page.*
