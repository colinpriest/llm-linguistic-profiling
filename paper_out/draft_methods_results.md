# Same Facts, Different Remedy: Linguistic Profiling and Arbitrariness in LLM and System One Decisions

*Draft: Abstract, Introduction, Related Work, Methods, Results, Discussion, Limitations, Ethics (21 Oct 2026). Every number comes from committed result files; see `prereg/README.md` for the claim-to-commit map. Labels: **[C]** confirmatory (pre-registered, tested once on held-out data); **[S]** pre-registered secondary; **[PH]** post hoc; **[E]** exploratory. `[CITE]` marks references to add and verify.*

## Abstract (draft)
Large language models are increasingly used to triage and resolve consumer complaints, and the way complainants write varies systematically with age, education and socioeconomic status. We audit three remedy-decision systems on fact-preserving sociolinguistic rewrites of US consumer financial complaints:
- two generative LLMs that return a single label (GPT-4o-mini, GPT-5-mini);
- a probabilistic "System One" model that returns a distribution over remedies (Jev).

Using pre-registered hypotheses tested once on held-out complaints, we separate two phenomena that standard single-label audits conflate.
- **Profiling at the group scale.** Older, college-educated and higher-SES writing styles receive systematically more remedy. These effects are real and replicated, but every one is below one percentage point of the monetary-remedy rate.
- **Arbitrariness at the individual scale.** The probability that two style versions of the same complaint receive different monetary decisions (2.8–6.3%) is far larger than the largest group gap of the same system (0.22–0.84 pp). About 99% of the variation in the probabilistic model's output across versions is arbitrariness (rewording, option order, inference noise), not persona.

Decomposing the profiling, it runs through persona-associated delivery (vocabulary, grammar, politeness) rather than content, and appears only where the style cue is recognisable. Positive controls show all three systems respond 49–387× more to remedy-relevant facts than to style. Graded probabilistic outputs cut arbitrariness by 70–80% but do not reduce profiling, and averaging probabilities is no better than majority voting. In a pre-registered mitigation test, rewriting every complaint into one standard style before deciding removed GPT-4o-mini's education and SES gaps (by 94% and 87%) and kept 96–102% of the systems' sensitivity to facts. It did not reduce arbitrariness; averaging decisions over three standard styles reduced it by about a fifth. We argue that fairness audits of LLM decision systems should report profiling and arbitrariness separately, test materiality rather than significance alone, and target the delivery channel when mitigating.

---

## 1. Introduction
LLMs are moving into consequential administrative decisions, including the triage and resolution of consumer complaints. Complainants differ in how they write, and those differences are not random: vocabulary, grammar, formality and politeness vary with age, education and socioeconomic status.

A system that responds to *how* a complaint is written rather than *what* it says can disadvantage some groups without ever seeing a demographic attribute. Sociolinguistics has studied this kind of *linguistic profiling* since the matched-guise experiments of Lambert et al. [1960]. Recent work shows that LLMs exhibit covert dialect prejudice that standard alignment does not remove [Hofmann et al., 2024], and that LLM underwriting recommendations can differ across otherwise-identical applicants [Bowen et al., working paper].

Most audits of such systems compare a single predicted label across demographic groups. This misses two things.

*First*, a single label sensitive to wording, option order or sampling noise produces disagreements that look like disparities, but are better described as **arbitrariness**. It is well established that prediction variance can be large enough to make individual decisions effectively arbitrary, and that it confounds fairness measurement [Cooper et al., 2024; Marx et al., 2020]. Arbitrariness carries its own normative weight [Creel & Hellman, 2022]. An audit that does not separate it from **profiling** cannot say which problem it has found.

*Second*, audits rarely ask whether a statistically detectable disparity is **material**, or *through what channel* it operates. Both matter for deciding whether, and how, to intervene.

Probabilistic decision models make these questions sharper. When a system returns a distribution over remedies rather than a label:
- the distribution may reveal differences that labels hide;
- decision rules built on it (monetary thresholds, uncertainty-based referral) may create new disparities, as abstention has been shown to do [Jones et al., 2021];
- graded outputs and averaging might offer mitigation.

**We study all of this in one design.** Complaints from the CFPB public database are rewritten into 36 fact-preserving sociolinguistic guises (age × geography × education × SES) by two independent rewrite models. Remedies are then decided by two single-label LLMs and one probabilistic System One model. The design adds direct controls for each non-persona source of variation: repeated scoring, all six option orders, and independent same-persona paraphrases. All main hypotheses were pre-registered and tested once on fresh complaints.

**Contributions.**
1. **Profiling vs arbitrariness, quantified.** About 99% of the probabilistic model's style sensitivity is arbitrariness (inference noise, option order, paraphrase). The persona-specific share is 1.1% (95% CI 0.7–1.5%).
2. **Two scales.** At the group scale, persona-associated profiling is real, replicated and directional, but immaterial: every monetary-rate gap is below 1 pp and passes the four-fifths rule by equivalence test. At the individual scale, arbitrariness is far larger: two style versions of the same complaint receive different monetary decisions 2.8–6.3% of the time, against largest group gaps of 0.22–0.84 pp.
3. **Mechanism.** GPT-4o-mini's education and SES gaps are 79–96% explained by interpretable attributes. Irrelevant *delivery* outweighs relevant *content* by roughly 5–8×, and the gaps appear only where the style cue is recognisable.
4. **Positive controls.** All three systems respond 49–387× more to remedy-relevant fact changes than to persona style, so the small profiling effects are not a sign of insensitivity.
5. **What probabilities do.** Graded outputs cut arbitrariness by 70–80% but do not reduce profiling. Probability averaging is no better than equal-budget hard voting, and uncertainty-based referral did not create a persona disparity.
6. **Mitigation.** Rewriting complaints into one standard style before deciding removes GPT-4o-mini's education and SES profiling while keeping its sensitivity to facts. It does not remove arbitrariness. Averaging over several standard styles reduces arbitrariness by about a fifth, at three times the inference cost.
7. **Practice.** A pre-registered, materiality-aware audit protocol with paraphrase, order and repeat controls, applicable beyond this task.

---

## 2. Related work

**Linguistic profiling and matched guises.** The matched-guise technique holds content constant while varying how it is expressed, to elicit evaluations of speakers [Lambert et al., 1960]. Applied to LLMs, dialect-based matched-guise probing reveals covert prejudice, including harsher judgments for speakers of African American English that do not surface when race is mentioned explicitly [Hofmann et al., 2024]. In finance, audit studies of LLM loan underwriting report disparities for otherwise-identical applicants [Bowen et al., working paper]. A prior study, which we cite in the third person, applied matched guises to remedy decisions on consumer complaints with single-label LLMs [CITE]. We extend this line in three ways: from labels to probabilistic outputs; from detecting disparity to decomposing its mechanism; and from significance to materiality and mitigation.

**Arbitrariness and predictive multiplicity.** Predictive multiplicity describes competing, equally accurate models that assign conflicting predictions to the same individual [Marx et al., 2020]. Variance across retrained models can make decisions effectively arbitrary, and accounting for it removes much of the apparent benefit of common fairness interventions [Cooper et al., 2024]. Arbitrariness carries a distinct moral weight, especially when one system is applied at scale [Creel & Hellman, 2022]. These works concern variance across *models*. We study *input-side* arbitrariness for a fixed deployed model (inference noise, option order and meaning-preserving rewording), and measure it on the same scale as group profiling.

**Prompt and option sensitivity in LLMs.** LLM outputs are highly sensitive to meaning-preserving prompt formatting [Sclar et al., 2024] and to the order of answer options in multiple-choice questions [Pezeshkpour & Hruschka, 2024]. We treat these sensitivities as fairness-relevant arbitrariness in a decision task. We measure them for a probabilistic judgment model and test whether they interact with persona; for education and SES, they do not.

**Probabilistic outputs, abstention and decision policies.** Abstaining on uncertain cases can magnify disparities between groups [Jones et al., 2021]. With access to a full remedy distribution, we evaluate decision policies, not only scores:
- monetary-probability thresholds at matched budgets;
- uncertainty-based referral;
- graded outputs;
- probability averaging versus equal-budget hard voting.

**Materiality and decomposition.** Regulatory practice in the US screens for adverse impact with the four-fifths rule [29 CFR 1607.4(D)]. Equivalence testing provides positive evidence that an effect is smaller than a smallest effect of interest [Lakens, 2017]. We use both to judge whether detected disparities matter. To attribute persona gaps to interpretable attributes we use Gelbach's order-invariant decomposition [Gelbach, 2016].

---

## References (verification status, 21 Oct 2026)
- Bowen, D. E. III, Price, S. M., Stein, L. C. D., & Yang, K. Racial disparities in LLM mortgage underwriting. **Working paper**; title varies by version (2024–2025). *Cite the exact version used.*
- Cooper, A. F., Lee, K., Choksi, M. Z., Barocas, S., De Sa, C., Grimmelmann, J., Kleinberg, J., Sen, S., & Zhang, B. (2024). Arbitrariness and social prediction: The confounding role of variance in fair classification. *AAAI-24*, 38(20), 22004–22012. doi:10.1609/aaai.v38i20.30203. *Verified.*
- Creel, K., & Hellman, D. (2022). The algorithmic Leviathan: Arbitrariness, fairness, and opportunity in algorithmic decision-making systems. *Canadian Journal of Philosophy*, 52(1), 26–43. *Verified; confirm the article DOI on the Cambridge page.*
- Gelbach, J. B. (2016). When do covariates matter? And which ones, and how much? *Journal of Labor Economics*, 34(2), 509–543. doi:10.1086/683668. *Verified.*
- Hofmann, V., Kalluri, P. R., Jurafsky, D., & King, S. (2024). AI generates covertly racist decisions about people based on their dialect. *Nature*, 633, 147–154. doi:10.1038/s41586-024-07856-5. *Verified.*
- Jones, E., Sagawa, S., Koh, P. W., Kumar, A., & Liang, P. (2021). Selective classification can magnify disparities across groups. *ICLR 2021*. *Verified.*
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362. doi:10.1177/1948550617697177. *Verified.*
- Lambert, W. E., Hodgson, R. C., Gardner, R. C., & Fillenbaum, S. (1960). Evaluational reactions to spoken languages. *Journal of Abnormal and Social Psychology*, 60(1), 44–51. *Venue verified; DOI to confirm.*
- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML 2020, PMLR* 119, 6765–6774. *Verified.*
- Pezeshkpour, P., & Hruschka, E. (2024). Large language models sensitivity to the order of options in multiple-choice questions. *Findings of NAACL 2024* (arXiv:2308.11483). *Venue per authors' lab page; confirm the ACL Anthology entry.*
- Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying language models' sensitivity to spurious features in prompt design. *ICLR 2024*. *Verified.*
- U.S. Code of Federal Regulations, 29 CFR §1607.4(D) (Uniform Guidelines on Employee Selection Procedures, four-fifths rule). *Verified.*
- *Still to add:* CFPB Consumer Complaint Database; TypeSafe Jev documentation; VADER; Flesch–Kincaid; the prior matched-guise study (third person, anonymised).

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
- **Pre-registration.** Four pre-registrations and two amendments were committed to version control before the data they test existed. They cover the main hypotheses (H1–H6), the attribute hypotheses (H-A1–H-A5), the positive controls (PC1–PC5) and the style-standardisation mitigation (MH1–MH5).
- **Planned extension, not run.** A pre-registered extension of the holdout to 4,000 complaints for H6 was not executed after a materiality review, before any new data were generated.
- **Disclosures.** All deviations are disclosed: two code-only fixes and one amendment requested after a freeze but committed before any result was inspected.
- **Bootstrap correction.** The pre-registered bootstrap of D under-corrected the noise term in resamples with repeated complaints. We report a corrected bootstrap (noise term s²·Σw²/C²) as a sensitivity analysis; no conclusion changes.

### 3.8 Style standardisation as a mitigation
The mitigation targets the channel identified in §4.5: if persona-associated *delivery* drives the gaps, rewriting every complaint into a standard style before deciding should remove them.

- **Sample.** The first 500 holdout complaints × 36 guises from the GPT-4o-mini bank (18,000 guises).
- **Arms.**
  - **M0:** the guise as written (existing decisions and scores).
  - **M1:** the guise rewritten by gpt-4o-mini (temperature 0) into one canonical style, A (plain, neutral, polite, chronological), then decided.
  - **M2:** the guise rewritten into three standard styles (A plain; B formal business; C simple everyday), each decided, with the outcomes averaged.

  Every standardising rewrite is instructed to preserve every fact, claim and request, including its strength and certainty, and to change only how the complaint is written.
- **Deciders.** GPT-4o-mini (the same remedy prompt and client) gives the tier. Jev gives E[tier] under order o012 and also scores the 19 attributes on every M1 text.
- **Fact sensitivity.** The `strengthen` and `weaken` texts of the first 200 complaints (§3.6) were standardised with the same three styles and decided. Retention is the fact span (strengthen − weaken) after standardisation divided by the span without it.
- **Arbitrariness.** The per-complaint mean absolute pairwise difference (MPAD) in outcome across the 36 guises.
- **Inference.** Complaint-block bootstrap, 2,000 resamples. Hypotheses MH1–MH5 were frozen with the code before any analysis-set text was standardised. The pipeline was first checked on five complaints outside the analysis set. The analysis was run once; no code changed after the freeze.
- **Scale and cost.** 55,800 standardising rewrites, 55,800 GPT-4o-mini decisions and Jev scoring of 55,001 unique texts, for about US$29 in total. 799 standardised texts were word-for-word identical to another text and were scored once.

---

## 4. Results

### 4.1 Single labels hide variation, but almost none of it is persona-specific
- **Hidden variation [S].** On the holdout, Jev's argmax tier is identical across all 36 guises for 63% of complaints (GPT-4o-mini bank) and 74% (GPT-5-mini bank). Yet within those complaints the monetary probability still varies, by more than 0.10 in about a quarter of them.
- **What drives it (Fig. 1, Jev only).**

| Comparison | Mean TV |
|---|---|
| Same text, scored twice | 0.021 (tier flips in 2.5% of re-scores) |
| Same text, options reversed | 0.058 |
| Two paraphrases, same persona | 0.091 |
| Two paraphrases, different persona | 0.092 |

- **H1 [C], supported.** The persona-specific share is 1.1% (95% CI 0.7–1.5%), far below the pre-registered 5% bound. About 99% of the style-induced variation in Jev's distributions is arbitrariness (inference noise, option order, paraphrase), not profiling.

### 4.2 Two scales: group profiling is real but immaterial; individual arbitrariness is material
- **H2 [C]: 9 of 12 directional contrasts supported.** Older, college-educated and higher-SES styles receive slightly more remedy. On E[tier], all six contrasts hold in both banks; for example, in the GPT-4o-mini bank:
  - old − young: +0.014 tier [0.010, 0.019];
  - college − high school: +0.006 [0.003, 0.009];
  - high − low SES: +0.008 [0.004, 0.011].

  On the argmax tier, 3 of 6 hold. Geography shows no effect.
- **Materiality [PH].**
  - In monetary-rate units (Fig. 2a), every gap for every system lies within ±1 percentage point.
  - Every AIR is above the four-fifths bound by equivalence test, and every |h| is below 0.2.
  - The largest disparity is GPT-4o-mini's SES gap (pooled n = 3,000): +0.84 pp [0.56, 1.11], AIR 0.917, h = 0.028. It is the only gap that cannot be bounded below 1 pp, and the only one that fails a stricter AIR > 0.90 bound.
- **Individual arbitrariness [PH].** It is far larger than the largest group gap of the same system (Fig. 2b). The probability that two style versions of the same complaint receive different monetary decisions is:
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

### 4.7 Mitigation: style standardisation removes profiling, not arbitrariness
Results are on 500 holdout complaints × 36 guises (GPT-4o-mini bank). Gaps are within-complaint contrasts in tiers (GPT-4o-mini) or E[tier] (Jev), with 95% bootstrap CIs.

- **MH1 [C], supported: one standard style removes GPT-4o-mini's profiling.**

| GPT-4o-mini gap | M0 (as written) | M1 (one style) | M2 (three styles) | Reduction M0 − M1 |
|---|---|---|---|---|
| College − high school | +0.031 [0.021, 0.042] | +0.002 [−0.008, 0.011] | +0.003 [−0.003, 0.010] | 0.029 [0.017, 0.041] |
| High − low SES | +0.024 [0.012, 0.035] | +0.003 [−0.008, 0.014] | +0.006 [−0.001, 0.014] | 0.021 [0.007, 0.034] |
| Old − young | +0.014 [−0.002, 0.028] | +0.007 [−0.008, 0.020] | +0.007 [−0.003, 0.018] | 0.007 [−0.008, 0.023] |

  The education gap falls by 94% and the SES gap by 87%. Both meet the pre-registered bar (reduction CI above 0 and |gap_M1| ≤ half of |gap_M0|).
- **Jev's gaps (reported).** They are removed too, but one style overshoots. Jev's M0 gaps were all positive: age +0.015, education +0.008, SES +0.010. Under M1, age and education reverse sign: −0.012 [−0.017, −0.006] and −0.007 [−0.011, −0.003]. Under M2, all three are within ±0.001 of zero.
- **MH2 [C], not supported: one standard style does not reduce arbitrariness.**
  - GPT-4o-mini MPAD: 0.147 (M0) vs 0.138 (M1); difference 0.009 [−0.001, 0.020].
  - Jev MPAD: 0.084 vs 0.085; difference −0.001 [−0.003, 0.001].

  A standardising rewrite is itself a new paraphrase. It removes the persona's direction but not the decision's sensitivity to wording.
- **MH5 [C], supported: averaging over three styles reduces arbitrariness.**
  - M1 − M2: GPT-4o-mini 0.017 [0.010, 0.025]; Jev 0.017 [0.016, 0.019].
  - Against the original guises (M0 − M2), arbitrariness falls by 18% for GPT-4o-mini (0.147 → 0.120) and 19% for Jev (0.084 → 0.068).
  - Part of this gain is mechanical: M2 averages three decisions per guise, and an average of three decisions varies less than one decision even without different styles (compare H4). The design does not separate style diversity from the number of draws.
- **MH3 [C], supported: standardisation preserves fact sensitivity.**

| System | Fact span, M0 | Retention, M1 | Retention, M2 |
|---|---|---|---|
| GPT-4o-mini (tier) | 1.58 | 0.99 [0.95, 1.03] | 0.96 [0.92, 0.99] |
| Jev E[tier] | 1.06 | 1.02 [0.99, 1.05] | 0.96 [0.94, 0.99] |

  Every lower bound is far above the pre-registered 0.5. M2 loses 4%, which is small but statistically detectable.
- **MH4 (descriptive): fidelity.**
  - **Content.** It is preserved: for every content attribute, the complaint-level correlation between M0 and M1 is 0.99–1.00 and persona η² is ≤ 0.007 under M1.
  - **Delivery.** Persona η² collapses for most cues: grammar 0.55 → 0.04, formality 0.70 → 0.12, politeness 0.34 → 0.07, emotional intensity 0.17 → 0.02.
  - **Vocabulary.** It collapses only partly (0.72 → 0.32), so the standardiser still passes on part of the persona cue.
  - **Structure.** Chronology (0.21 → 0.004) and argument quality (0.23 → 0.01) collapse. Concision does not (0.12 → 0.13).
  - **Mean shifts.** Standardised texts read as more formal (+0.38 on the [0, 1] scale) and less emotional (−0.14) than the guises.

---

## 5. Discussion

### 5.1 Two problems that need two different remedies
The central result is that profiling and arbitrariness are different problems with different causes, and each responds to a different intervention.

**Profiling.** In these systems, profiling is a *channel* problem. Persona-associated delivery (vocabulary, grammar, formality, politeness) carries the education and SES gaps (H-A2), and only where the style cue is recognisable (H-A3). Removing the channel removes the gap: one standardising rewrite cut GPT-4o-mini's education and SES gaps by 87–94% (MH1).

**Arbitrariness.** Arbitrariness is a *variance* problem: the decision depends on incidental features of any particular wording, option order or inference draw. Swapping one wording for another does not reduce it (MH2). Only averaging over draws does: averaging over standard styles (MH5), or graded probabilistic outputs in place of an argmax (H3a).

Neither intervention fixes both problems. A system that standardises style still gives the same complaint a different remedy depending on how it happens to be worded; it is simply no longer worded differently *by group*. A system that averages probabilities is less arbitrary, but its profiling is unchanged (H3b). An audit that reports one aggregate disparity number, or a mitigation evaluated on one, can hide that it has solved only half the problem.

### 5.2 Materiality at two scales
Judged by the group-rate standards used in adverse-impact screening, all three systems pass. Every monetary-rate gap is below one percentage point, and every adverse impact ratio is bounded above 0.80 by equivalence test.

Judged at the individual scale, they do not look so benign. Two fact-identical versions of the same complaint receive different monetary decisions 2.8–6.3% of the time, far more often than the largest group gap of the same system (0.22–0.84 pp) would suggest. For the complainant, it matters little whether a different remedy came from a group-patterned bias or a wording lottery. Creel and Hellman [2022] argue that arbitrariness at scale has moral weight of its own. Our results suggest that, for LLM decision systems, it can be the larger of the two harms.

Group-rate tests remain necessary: they caught a real, replicated and directional profiling signal. They should not be the only test.

### 5.3 Implications for audit practice
We suggest five practices that generalise beyond complaint handling.

1. **Separate profiling from arbitrariness.** Measure the non-persona sources of variation directly (repeated scoring, option orders, independent same-persona paraphrases) and report the persona-specific share. Without these controls, a matched-guise audit cannot tell which problem it has found.
2. **Test materiality, not just significance.** Equivalence and minimum-effect tests against a pre-stated bound make "small" a finding rather than a failure to reject.
3. **Include positive controls.** A system can show small persona gaps because it ignores the text altogether. Fact edits establish that it responds to what matters; here the response was 49–387× the largest persona gap.
4. **Evaluate decision policies, not only scores.** Thresholds, referral rules and averaging change disparity and arbitrariness in different ways. Referral did not create a disparity here (H5), but that was an empirical question.
5. **Identify the channel before mitigating.** The attribute decomposition predicted that standardising delivery would work, and it did. A mitigation aimed at content, or a generic "debiasing" instruction, would have targeted the wrong channel.

### 5.4 Style standardisation in practice
Standardisation is cheap and, by our measures, faithful. One rewrite costs about US$0.0002 with gpt-4o-mini. Content attributes are preserved almost exactly, and sensitivity to remedy-relevant facts is kept within a few percent. A deployer can apply it without changing the decision model. It also composes with graded outputs, which address the variance that standardisation leaves.

It has three caveats.
- **The standardiser is itself a model that needs auditing.** It passed on part of the vocabulary cue (η² 0.32). For Jev, a single canonical style overshot and reversed the age and education gaps. A mitigation can create a new directional bias, so it must be audited with the same matched-guise design as the decision system.
- **Averaging buys arbitrariness reduction at a cost.** Three styles cost three times as much at inference, and the 18–19% reduction is modest next to the 70–80% from graded outputs (H3a). Where a probabilistic decider is available, graded outputs are the cheaper lever against arbitrariness; standardisation is the lever against profiling.
- **Standardisation removes signals as well as noise.** It made complaints less emotional (−0.14) and more formal. For most remedy decisions that is the intent. But distress can signal consumer vulnerability, which some regulators require firms to recognise [CITE: verify, e.g. UK FCA FG21/1]. A standardised text should feed the remedy decision, not replace the original in the case record or in human review.

### 5.5 Generative LLMs and System One models
The three systems were practically equivalent in group disparity (§4.4), and all three responded strongly to facts (§4.6). The probabilistic interface still mattered:
- it exposed variation that single labels hide (§4.1);
- it allowed graded outputs that cut arbitrariness by 70–80%;
- its distribution supported the decision-policy analyses (§4.3).

It did not reduce profiling, and averaging probabilities was no better than voting. We make no claim about architecture: the comparison is between interfaces as deployed, and a single-label LLM sampled repeatedly could recover some of the same benefits.

---

## 6. Limitations
- **No ground truth.** There is no correct remedy for these complaints. Our claims concern sensitivity, disparity and arbitrariness, not accuracy: a system could be unbiased and consistently wrong.
- **Synthetic guises.** The persona styles were written by LLMs following persona instructions. They are stylised representations of how groups write, not samples of real writers, and may exaggerate or miss real sociolinguistic variation. Effects on real complainants' own writing may differ in size. We did not study dialects, race, gender or English as a second language as manipulated factors.
- **LLM-generated edits and standardisation.** The positive-control edits and the standardising rewrites were produced by gpt-4o-mini. The edits were fact-led but not fact-only (§4.6). The standardiser partly passed on persona vocabulary (§4.7).
- **Shared model families.** GPT-4o-mini wrote one rewrite bank, the standardised texts and the edits, and it is also one of the deciders. Jev both scores the attributes and decides. Self-preference or shared representations could inflate fidelity or alignment estimates. The second rewrite bank (GPT-5-mini) mitigates this for the main results but not for the mitigation, which used only the GPT-4o-mini bank.
- **One System One model, one version.** Jev 1.13.0 is not deterministic (mean TV 0.020 on re-scoring; 2.5% of argmax tiers flip). Results may not transfer to other probabilistic judgment models or later versions.
- **Model drift.** M0 decisions come from the 19 October run; M1 and M2 were decided on 21 October with the same model name and settings. Unannounced provider updates would bias M0-versus-M1 comparisons. The M1-versus-M2 comparisons share a date and are unaffected.
- **Scope.** The study covers US consumer financial complaints in English, one remedy prompt and a coarse three-tier outcome. The mitigation was tested on 500 complaints and one rewrite bank.
- **Post hoc materiality.** The materiality bounds (1 pp, AIR 0.80 and 0.90, |h| 0.2) were chosen after the confirmatory results were seen and are labelled post hoc. The four-fifths rule comes from US employment selection and has no formal status for complaint remedies.
- **Averaging confound.** M2 combines style diversity with more decisions per guise, so its arbitrariness reduction cannot be attributed to style diversity alone.

---

## 7. Ethical considerations and adverse impact

**Data.** We use narratives from the CFPB public Consumer Complaint Database, which consumers consented to publish and which the CFPB scrubs of personal information before release [CITE]. No new data were collected from people, and no attempt was made to re-identify complainants. Rewritten guises and edits are synthetic.

**Representing groups through stereotyped style.** Matched-guise designs necessarily write in the stereotyped style of a group. We use these texts as measurement instruments, not as portraits, and release them with that caveat. We recommend they not be used to train or prompt systems that infer demographics from writing.

**Adverse impacts we anticipate.**
- **Misreading "immaterial."** A finding that group gaps fall below adverse-impact thresholds could be cited as clearance to deploy. It is not one. The same systems are materially arbitrary for individuals (§5.2), and our results cover one task, three systems and stylised guises.
- **Loss of voice.** Standardisation changes how a complainant's own words reach the decision-maker. Used carelessly, it could erase signals of distress or vulnerability, or let an organisation describe a complaint in terms the complainant did not choose. We recommend standardising only the input to an automated decision, keeping the original text for the record and for human review, and auditing the standardiser itself.
- **Gaming.** Knowing that delivery moves decisions, or which facts move them most, could help complainants present their cases more effectively. This arguably narrows existing advantages, but it could also help firms tune systems towards outcomes that favour them while passing group-rate audits. Reporting arbitrariness alongside profiling makes the second harder.
- **Compute.** The study used inexpensive models; total API spend across all experiments was modest [AUTHOR: give total].

**Pre-registration and disclosure.** Every confirmatory claim was pre-registered and tested once, and every deviation is disclosed in the analysis record (`prereg/README.md`).

**Competing interests and positionality.** [AUTHOR TO COMPLETE: any relationship with TypeSafe, the provider of Jev, or with OpenAI; funding; researcher positionality statement.]
