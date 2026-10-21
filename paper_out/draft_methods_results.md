# Same Facts, Different Remedy: Linguistic Profiling and Arbitrariness in LLM and System One Decisions

*Draft: Abstract, Introduction, Related Work, Methods, Results, Discussion, Limitations, Ethics, Conclusion (21 Oct 2026). Every number comes from committed result files; see* `prereg/README.md` *for the claim-to-commit map. Labels: **\[C\]** confirmatory (pre-registered, tested once on held-out data); **\[S\]** pre-registered secondary; **\[PH\]** post hoc; **\[E\]** exploratory.* `[CITE]` *marks references to add and verify.*

## Abstract (draft)

Large language models are increasingly used to triage and resolve consumer complaints, and the way complainants write varies systematically with age, education and socioeconomic status. We audit three remedy-decision systems using fact-preserving sociolinguistic rewrites of US consumer financial complaints:

- two generative LLMs that return a single label (GPT-4o-mini, GPT-5-mini);

- a probabilistic "System One" model that returns a distribution over remedies (Jev).

Using pre-registered hypotheses tested once on held-out complaints, we separate two phenomena that standard single-label audits conflate.

- **Profiling at the group scale.** Older, college-educated and higher-SES writing styles receive systematically more generous remedies. These effects are reliable and replicate, but each is smaller than one percentage point in the monetary-remedy rate.

- **Arbitrariness at the individual scale.** The probability that two style versions of the same complaint receive different monetary decisions (2.8–6.3%) is far larger than the largest group gap of the same system (0.22–0.84 pp). Approximately 99% of the cross-version variation in the probabilistic model's output is attributable to arbitrariness (rewording, option order and inference noise) rather than to persona.

Decomposition shows that profiling operates through persona-associated delivery (vocabulary, grammar and politeness) rather than content, and that it appears only where the style cue is recognisable. Positive controls show that all three systems respond 49–387× more strongly to remedy-relevant facts than to style. Graded probabilistic outputs reduce arbitrariness by 70–80% but leave profiling unchanged, and averaging probabilities performs no better than majority voting. In a pre-registered mitigation test, rewriting every complaint into a single standard style before the decision largely removed GPT-4o-mini's education and SES gaps (reductions of 94% and 87%, respectively) while preserving 96–102% of the systems' sensitivity to facts. It did not, however, reduce arbitrariness; averaging decisions across three standard styles reduced it by approximately one fifth. We argue that fairness audits of LLM decision systems should report profiling and arbitrariness separately, assess materiality rather than statistical significance alone, and target the delivery channel when designing mitigations.

------------------------------------------------------------------------

## 1. Introduction

Large language models (LLMs) are increasingly deployed in consequential administrative decisions, including the triage and resolution of consumer complaints. Complainants differ in how they write, and these differences are not random: vocabulary, grammar, formality and politeness vary with age, education and socioeconomic status.

A system that responds to *how* a complaint is written rather than *what* it says can disadvantage particular groups without ever observing a demographic attribute. Sociolinguists have studied such *linguistic profiling* since the matched-guise experiments of Lambert et al. \[1960\]. Recent work shows that LLMs exhibit covert dialect prejudice that standard alignment does not remove \[Hofmann et al., 2024\], and that LLM underwriting recommendations can differ across otherwise-identical applicants \[Bowen et al., working paper\].

Most audits of such systems compare a single predicted label across demographic groups. This approach overlooks two issues.

*First*, when a single label is sensitive to wording, option order or sampling noise, it produces disagreements that resemble disparities but are better described as **arbitrariness**. It is well established that prediction variance can be large enough to make individual decisions effectively arbitrary, and that it confounds fairness measurement \[Cooper et al., 2024; Marx et al., 2020\]. Arbitrariness carries its own normative weight \[Creel & Hellman, 2022\]. An audit that does not separate arbitrariness from **profiling** cannot determine which problem it has identified.

*Second*, audits rarely ask whether a statistically detectable disparity is **material**, or *through which channel it operates. Both questions bear directly on* whether, and how, to intervene.

Probabilistic decision models sharpen these questions. When a system returns a distribution over remedies rather than a label:

- the distribution may reveal differences that labels hide;

- decision rules built on it (monetary thresholds, uncertainty-based referral) may create new disparities, as abstention has been shown to do \[Jones et al., 2021\];

- graded outputs and averaging might offer mitigation.

**We address these questions within a single design.** Complaints from the CFPB public database are rewritten into 36 fact-preserving sociolinguistic guises (age × geography × education × SES) by two independent rewrite models. Remedies are then determined by two single-label LLMs and one probabilistic System One model. The design incorporates direct controls for each non-persona source of variation: repeated scoring, all six option orders, and independent same-persona paraphrases. All main hypotheses were pre-registered and tested once on held-out complaints.

**Our contributions are as follows.**

1.  **Profiling versus arbitrariness, quantified. Approximately** 99% of the probabilistic model's style sensitivity is arbitrariness (inference noise, option order, paraphrase). The persona-specific share is 1.1% (95% CI 0.7–1.5%).

2.  **Two scales.** At the group scale, persona-associated profiling is real, replicated and directional, but immaterial: every monetary-rate gap is below 1 pp and passes the four-fifths rule by equivalence test. At the individual scale, arbitrariness is far larger: two style versions of the same complaint receive different monetary decisions 2.8–6.3% of the time, compared with largest group gaps of 0.22–0.84 pp.

3.  **Mechanism.** GPT-4o-mini's education and SES gaps are 79–96% explained by interpretable attributes. Irrelevant *delivery* outweighs relevant *content* by roughly 5–8×, and the gaps appear only where the style cue is recognisable.

4.  **Positive controls.** All three systems respond 49–387× more to remedy-relevant fact changes than to persona style, indicating that the small profiling effects do not reflect general insensitivity.

5.  **The role of probabilistic outputs. Graded outputs reduce arbitrariness by 70–80% but do not reduce profiling. Probability averaging performs no better than equal-budget hard voting, and uncertainty-based referral does not introduce** a persona disparity.

6.  **Mitigation.** Rewriting complaints into one standard style before deciding removes GPT-4o-mini's education and SES profiling while preserving its sensitivity to facts. It does not remove arbitrariness. Averaging over several standard styles reduces arbitrariness by about a fifth, at three times the inference cost.

7.  **Practice.** We provide a pre-registered, materiality-aware audit protocol, incorporating paraphrase, order and repeat controls, that is applicable beyond this task.

------------------------------------------------------------------------

## 2. Related work

**Linguistic profiling and matched guises.** The matched-guise technique holds content constant while varying how it is expressed, to elicit evaluations of speakers \[Lambert et al., 1960\]. Applied to LLMs, dialect-based matched-guise probing reveals covert prejudice, including harsher judgments for speakers of African American English that do not surface when race is mentioned explicitly \[Hofmann et al., 2024\]. In finance, audit studies of LLM loan underwriting report disparities for otherwise-identical applicants \[Bowen et al., working paper\]. We extend this line of work to remedy decisions on consumer complaints in three ways: from single labels to probabilistic outputs; from detecting disparity to decomposing its mechanism; and from significance to materiality and mitigation.

**Arbitrariness and predictive multiplicity.** Predictive multiplicity describes competing, equally accurate models that assign conflicting predictions to the same individual \[Marx et al., 2020\]. Variance across retrained models can make decisions effectively arbitrary, and accounting for it removes much of the apparent benefit of common fairness interventions \[Cooper et al., 2024\]. Arbitrariness carries a distinct moral weight, especially when one system is applied at scale \[Creel & Hellman, 2022\]. These studies concern variance across *models*. We study *input-side* arbitrariness for a fixed deployed model (inference noise, option order and meaning-preserving rewording), and measure it on the same scale as group profiling.

**Prompt and option sensitivity in LLMs.** LLM outputs are highly sensitive to meaning-preserving prompt formatting \[Sclar et al., 2024\] and to the order of answer options in multiple-choice questions \[Pezeshkpour & Hruschka, 2024\]. We treat these sensitivities as fairness-relevant arbitrariness in a decision task. We measure them for a probabilistic judgment model and test whether they interact with persona; for education and SES, they do not.

**Probabilistic outputs, abstention and decision policies.** Abstaining on uncertain cases can magnify disparities between groups \[Jones et al., 2021\]. Because we observe the full remedy distribution, we can evaluate decision policies rather than scores alone:

- monetary-probability thresholds at matched budgets;

- uncertainty-based referral;

- graded outputs;

- probability averaging versus equal-budget hard voting.

**Materiality and decomposition.** In the US, regulatory practice screens for adverse impact using the four-fifths rule \[29 CFR 1607.4(D)\]. Equivalence testing provides positive evidence that an effect is smaller than a smallest effect of interest \[Lakens, 2017\]. We use both to assess whether detected disparities are material. To attribute persona gaps to interpretable attributes, we use Gelbach's order-invariant decomposition \[Gelbach, 2016\].

------------------------------------------------------------------------

## References (verification status, 21 Oct 2026)

- Consumer Financial Protection Bureau (2015). *Narrative Scrubbing Standard*. Office of Consumer Response. https://files.consumerfinance.gov/a/assets/201503_cfpb_Narrative-Scrubbing-Standard.pdf. *Verified.*

- Consumer Financial Protection Bureau (2023). *Privacy Impact Assessment: Publication of Consumer Response Complaint Narratives* (2 June 2023). https://files.consumerfinance.gov/f/documents/06022023_Publication-of-Consumer-Response-Complaint-Narratives.pdf. *Verified.*

- Consumer Financial Protection Bureau (n.d.). *Consumer Complaint Database* \[dataset\]. https://www.consumerfinance.gov/data-research/consumer-complaints/. *Add the download date.*

- Bowen, D. E. III, Stein, L. C. D., Price, S. M., & Yang, K. (2025). Measuring and mitigating racial disparities in LLMs: Evidence from a mortgage underwriting experiment. Working paper, version of 17 September 2025 (earlier versions were titled "Racial bias in large language model mortgage underwriting"). *Verified.*

- Cooper, A. F., Lee, K., Choksi, M. Z., Barocas, S., De Sa, C., Grimmelmann, J., Kleinberg, J., Sen, S., & Zhang, B. (2024). Arbitrariness and social prediction: The confounding role of variance in fair classification. *AAAI-24*, 38(20), 22004–22012. doi:10.1609/aaai.v38i20.30203. *Verified.*

- Creel, K., & Hellman, D. (2022). The algorithmic Leviathan: Arbitrariness, fairness, and opportunity in algorithmic decision-making systems. *Canadian Journal of Philosophy*, 52(1), 26–43. *Verified; confirm the article DOI on the Cambridge page.*

- Financial Conduct Authority (2021). *FG21/1: Guidance for firms on the fair treatment of vulnerable customers* (23 February 2021). *Verified.*

- Gelbach, J. B. (2016). When do covariates matter? And which ones, and how much? *Journal of Labor Economics*, 34(2), 509–543. doi:10.1086/683668. *Verified.*

- Hofmann, V., Kalluri, P. R., Jurafsky, D., & King, S. (2024). AI generates covertly racist decisions about people based on their dialect. *Nature*, 633, 147–154. doi:10.1038/s41586-024-07856-5. *Verified.*

- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of ICWSM*, 8(1), 216–225. doi:10.1609/icwsm.v8i1.14550. *Verified.*

- Jones, E., Sagawa, S., Koh, P. W., Kumar, A., & Liang, P. (2021). Selective classification can magnify disparities across groups. *ICLR 2021*. *Verified.*

- Kincaid, J. P., Fishburne, R. P., Rogers, R. L., & Chissom, B. S. (1975). *Derivation of new readability formulas (Automated Readability Index, Fog Count and Flesch Reading Ease formula) for Navy enlisted personnel*. Research Branch Report 8-75, Naval Technical Training Command. DTIC ADA006655. *Verified.*

- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362. doi:10.1177/1948550617697177. *Verified.*

- Lambert, W. E., Hodgson, R. C., Gardner, R. C., & Fillenbaum, S. (1960). Evaluational reactions to spoken languages. *Journal of Abnormal and Social Psychology*, 60(1), 44–51. *Venue verified; DOI to confirm.*

- Marx, C., Calmon, F., & Ustun, B. (2020). Predictive multiplicity in classification. *ICML 2020, PMLR* 119, 6765–6774. *Verified.*

- Pezeshkpour, P., & Hruschka, E. (2024). Large language models sensitivity to the order of options in multiple-choice questions. *Findings of NAACL 2024* (arXiv:2308.11483). *Venue per authors' lab page; confirm the ACL Anthology entry.*

- Watkins, E. A., & Chen, J. (2024). The four-fifths rule is not disparate impact: A woeful tale of epistemic trespassing in algorithmic fairness. *FAccT '24*. Preprint arXiv:2202.09519 (with M. McKenna). *Verified.*

- Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying language models' sensitivity to spurious features in prompt design. *ICLR 2024*. *Verified.*

- U.S. Code of Federal Regulations, 29 CFR §1607.4(D) (Uniform Guidelines on Employee Selection Procedures, four-fifths rule). *Verified.*

- TypeSafe (2026). *Jev: a System One model* (version 1.13.0 used). Documentation, https://docs.typesafe.ai/concepts/system-one. *Verified.*

------------------------------------------------------------------------

## 3. Methods

### 3.1 Task and decision systems

We study the assignment of remedies to US consumer financial complaints from the CFPB public complaint database \[CFPB, n.d.\]. Remedies fall into three ordered tiers: 0 = no action, 1 = non-monetary action and 2 = monetary remedy. We compare two kinds of decision interface, each configured as it is typically deployed:

- **Generative LLMs that return a single label.** We use GPT-4o-mini (temperature 0) and GPT-5-mini (default reasoning). Both receive the same remedy prompt and return a validated structured response (tier, confidence, information needed); the prompt is given in Appendix A.

- **A probabilistic "System One" model.** We use Jev 1.13.0 (TypeSafe) \[TypeSafe, 2026\]. We use "System One" in its provider's sense: a model that returns a probability distribution over the options of a typed question rather than generated text. Here the question's options carry the same three tier definitions as the GPT prompt.

From Jev's distribution p = (p0, p1, p2) we derive:

- the expected tier, E\[tier\] = p1 + 2·p2;

- the argmax tier;

- the monetary probability, p2;

- the normalised entropy.

Each text is scored under all six orderings of the three options, with order o012 designated as primary. We make no causal claim about model architecture; the comparison is between interfaces as deployed.

### 3.2 Matched-guise panels

- **Complaints.** Complaints were drawn from a cleaned pool of 20,000 English-language narratives from the CFPB database, after language detection, deduplication and length filtering (Appendix A).

- **Guises.** Each complaint was rewritten into 36 sociolinguistic guises: age (young / old) × geography (urban / suburban / rural) × education (high school / college) × socioeconomic status (low / middle / high). The rewrites were produced independently by two models (the GPT-4o-mini bank and the GPT-5-mini bank), with an instruction to preserve every fact.

- **Exploratory panel.** This panel comprises 2,000 complaints (144,000 guises); it was analysed first and is reported as exploratory.

- **Holdout panel.** This panel comprises 1,000 new complaints that had never previously been rewritten or scored (72,000 guises). The sample was drawn with a fixed seed and fingerprinted (SHA-256) before any holdout text existed. The panel was generated with identical prompts and settings, and the frozen confirmatory analysis was run on it exactly once.

- **GPT decisions.** Each GPT model decided the guises in its own bank. Comparisons between the two GPT models therefore combine differences in the decider with differences in the rewrites it saw; comparisons between Jev and a GPT model use the same bank and are not affected.

### 3.3 Controls for arbitrariness

We directly measure three non-persona sources of variation:

1.  **Inference noise, estimated by re-scoring** 1,000 identical texts;

2.  **Option order, estimated across** all six orderings;

3.  **Paraphrase, estimated from** the two rewrite banks, which provide two independent, same-persona paraphrases of every guise.

Cross-bank comparisons isolate persona-specific variation. The pre-registered H1 statistic is the relative excess of cross-bank total-variation (TV) distance for different-persona pairs over same-persona pairs. Because it is a ratio of distances rather than a decomposition, a consistent persona effect can be masked by differences between the rewrite banks. We therefore also fit a variance-component model \[PH\]: for Jev's monetary probability and E\[tier\], a fully crossed random-effects ANOVA over complaint × persona × rewrite bank × option order (one observation per cell, solved by expected mean squares). It splits within-complaint variance into five parts: (i) the persona main effect; (ii) complaint-specific persona effects that replicate across both independent paraphrases; (iii) systematic rewrite-model differences, where one bank shifts every guise of a complaint; (iv) guise-specific paraphrase; and (v) option order and inference noise. Part (ii) can contain heterogeneous persona effects, so we do not count it as arbitrariness.

### 3.4 Disparity and arbitrariness measures

- **Systematic persona disparity (D).** For a complaint × persona matrix of outcomes, we use a two-way additive decomposition. D is the noise-corrected SD of the 36 persona effects divided by the noise-corrected SD of complaint effects (unbiased ANOVA estimators). D neither rewards a constant output nor penalises a rule merely for being noisier.

- **Arbitrariness.** By arbitrariness we mean instability of an individual decision under changes that should not matter: rewording, option order and re-scoring. We measure it directly as the probability that two style versions of the same complaint receive different monetary decisions, and as the per-complaint mean absolute pairwise difference (MPAD). The pre-registered residual (complaint × persona interaction) share used in H3a mixes such instability with heterogeneous persona effects; we report it as registered but do not read it as arbitrariness alone.

- **Directional contrasts.** These are within-complaint differences between persona levels (old − young, college − high school, high − low SES), expressed in tiers or in percentage points of the monetary-remedy rate.

- **Materiality benchmarks \[PH\].** We report adverse impact ratios (AIR) and Cohen's h, and test equivalence (TOST) against AIR \> 0.80, \|h\| \< 0.20 and \|gap\| \< 1 pp, alongside minimum-effect tests. These bounds were chosen after the confirmatory results were seen. The four-fifths rule itself allows that smaller differences may constitute adverse impact \[29 CFR 1607.4(D)\], and its use as a general fairness criterion has been criticised \[Watkins & Chen, 2024\]. We therefore treat the bounds as descriptive benchmarks of absolute size, not as tests of whether a disparity matters.

### 3.5 Interpretable attributes

Jev additionally scored every guise on 19 attributes (3-level choice questions; expected score in \[0, 1\]), plus three objective measures (mean sentence length, Flesch–Kincaid grade, \|VADER compound\|) \[Kincaid et al., 1975; Hutto & Gilbert, 2014\]. The attributes fall into three families, whose normative relevance to the remedy decision was fixed in advance:

| Family | Relevance | Attributes |
|----|----|----|
| Delivery (how it is said) | irrelevant | vocabulary, grammar/spelling, formality, confidence, politeness, rudeness/insults, emotional intensity, entitlement, plus the objective measures |
| Structure (how it is organised) | procedurally relevant only | coherence, chronology, argument quality, clarity of the request, concision |
| Content (what is said) | relevant; should not change under fact-preserving rewrites | evidence cited, harm severity, alleged bank error, monetary demand, legal basis, prior attempts |

**Decomposition.** Within-complaint persona gaps are decomposed into attribute and family contributions with Gelbach's order-invariant decomposition \[Gelbach, 2016\].

**Recognition.** Moderation by recognisability is based on Jev's probability of the intended persona level; a guise is classed as "recognised" if this probability is at least 0.60.

**Pilot.** A 200-complaint pilot with pre-specified go/no-go rules preceded the confirmatory holdout run.

### 3.6 Positive controls

For the first 200 holdout complaints, GPT-4o-mini produced three fact edits, each under an instruction to leave wording, tone and errors unchanged:

- `strengthen`, which adds a documented bank error with a dollar loss;

- `weaken`, which recasts the problem as the writer's own mistake or as agreed terms;

- `harm_x10`, which multiplies all dollar amounts by 10.

All three systems decided or scored the edited texts. Effects are computed as paired edit-minus-original differences.

### 3.7 Inference and pre-registration

- **Resampling.** We use a complaint-cluster (block) bootstrap and within-complaint permutation tests, with Benjamini–Hochberg correction applied within the directional family.

- **Pre-registration.** Four pre-registrations and two amendments were committed to version control before the data they test existed. They cover the main hypotheses (H1–H6), the attribute hypotheses (H-A1–H-A5), the positive controls (PC1–PC5) and the style-standardisation mitigation (MH1–MH5).

- **Planned extension, not run.** Following a materiality review, and before any new data were generated, we did not execute a pre-registered extension of the holdout to 4,000 complaints for H6.

- **Disclosures.** All deviations are disclosed. They comprise two code-only fixes and one amendment that was requested after a freeze but committed before any result was inspected.

- **Bootstrap correction.** The pre-registered bootstrap of D under-corrected the noise term in resamples with repeated complaints. All confidence intervals for D reported here use the corrected bootstrap (noise term s²·Σw²/C²); the pre-registered intervals are given in Appendix B for transparency, and no conclusion depends on the choice.

### 3.8 Style standardisation as a mitigation

The mitigation targets the channel identified in §4.5: if persona-associated *delivery* drives the gaps, rewriting every complaint into a standard style before the decision should remove them.

- **Sample.** The sample comprises the first 500 holdout complaints × 36 guises from the GPT-4o-mini bank (18,000 guises).

- **Arms.** The design has three arms:

  - **M0:** the guise as originally written (using existing decisions and scores);

  - **M1:** the guise rewritten by GPT-4o-mini (temperature 0) into a single canonical style, A (plain, neutral, polite and chronological), and then decided;

  - **M2:** the guise rewritten into three standard styles (A, plain; B, formal business; C, simple everyday), each decided separately, with the outcomes averaged.

<!-- -->

- Every standardising rewrite is instructed to preserve every fact, claim and request, including its strength and certainty, and to change only how the complaint is written.

<!-- -->

- **Deciders.** GPT-4o-mini, using the same remedy prompt and client, assigns the tier. Jev provides E\[tier\] under order o012 and also scores the 19 attributes on every M1 text.

- **Fact sensitivity.** The `strengthen` and `weaken` texts of the first 200 complaints (§3.6) were standardised into the same three styles and decided. Retention is defined as the fact span (strengthen − weaken) after standardisation divided by the span without it.

- **Arbitrariness.** Arbitrariness is measured as the per-complaint mean absolute pairwise difference (MPAD) in outcome across the 36 guises.

- **Inference.** We use a complaint-block bootstrap with 2,000 resamples. Hypotheses MH1–MH5 were frozen with the code before any analysis-set text was standardised. The pipeline was first checked on five complaints outside the analysis set. The analysis was run once, and no code was changed after the freeze.

- **Scale.** The 18,000 guises and the 600 positive-control texts (the 200 originals and their `strengthen` and `weaken` edits) were each standardised into three styles, giving 55,800 standardising rewrites, 55,800 GPT-4o-mini decisions and Jev scoring of 55,001 unique texts. Of the standardised texts, 799 were word-for-word identical to another text and were therefore scored only once.

------------------------------------------------------------------------

## 4. Results

### 4.1 Single labels hide variation, and little of it is attributable to persona

- **Hidden variation \[S\].** On the holdout, Jev's argmax tier is identical across all 36 guises for 63% of complaints (GPT-4o-mini bank) and 74% (GPT-5-mini bank). Yet within those complaints the monetary probability still varies, by more than 0.10 in approximately a quarter of cases.

- **Sources of variation (Fig. 1; Jev only).**

| Comparison | Mean TV |
|----|----|
| Same text, scored twice | 0.021 (tier flips in 2.5% of re-scores) |
| Same text, options reversed | 0.058 |
| Two paraphrases, same persona | 0.091 |
| Two paraphrases, different persona | 0.092 |

- **H1 \[C\], supported as registered.** Different-persona cross-bank pairs are 1.1% (95% CI 0.7–1.5%) further apart in TV than same-persona pairs, below the pre-registered 5% bound. This is a relative difference in distance, not a share of variance.

- **Variance components \[PH\] (Table 9).** For Jev's monetary probability, the persona main effect accounts for 0.04% of within-complaint variance \[0.00, 0.11\], and complaint-specific persona effects that replicate across both paraphrases for 1.5% \[0.7, 2.4\]. The rest comes from systematic differences between the two rewrite models (47% \[42, 52\]), guise-specific paraphrase (39% \[35, 43\]), and option order and inference noise (12% \[11, 14\]). E\[tier\] gives the same picture (persona 0.2% and 1.5%). Persona-attributable variance is therefore small. But about half of the rest reflects which model wrote the rewrite, shifting all of a complaint's guises together, rather than the wording of any single guise.

### 4.2 Two scales: small average group gaps, larger individual instability

- **H2 \[C\]: 9 of 12 directional contrasts are** **supported.** Older, college-educated and higher-SES styles receive slightly more generous remedies. On E\[tier\], all six contrasts hold in both banks; in the GPT-4o-mini bank, for example:

  - old − young: +0.014 tier \[0.010, 0.019\];

  - college − high school: +0.006 \[0.003, 0.009\];

  - high − low SES: +0.008 \[0.004, 0.011\].

<!-- -->

- On the argmax tier, 3 of 6 contrasts hold. Geography has no detectable effect.

<!-- -->

- **Materiality \[PH\].**

  - On the held-out panel (n = 1,000), every monetary-rate gap for every system is within ±0.5 pp, and every 95% upper bound is below 1 pp. The largest is GPT-4o-mini's education gap, +0.47 pp \[0.08, 0.83\] (AIR 0.947).

  - Equivalence tests place every AIR above 0.80, and every \|h\| is below 0.2.

  - Pooling the exploratory and held-out panels (n = 3,000; post hoc and descriptive), the largest gap is GPT-4o-mini's SES gap: +0.84 pp \[0.56, 1.11\], AIR 0.917, h = 0.028. That AIR is an 8.3% relative shortfall. The interval does not exclude a gap above 1 pp, and this gap fails a stricter AIR \> 0.90 bound.

  - These gaps are small in absolute terms. Whether a gap of this size matters depends on the decision and its consequences, which the benchmarks do not settle.

- **Individual instability \[PH\].** On the held-out panel, the probability that two style versions of the same complaint receive different monetary decisions (Fig. 2b) is:

  - GPT-4o-mini: 3.8%;

  - GPT-5-mini: 6.3%;

  - Jev (p2 threshold matched to each GPT model's monetary volume): 2.6–3.4%.

<!-- -->

- Monetary outcomes are not unanimous across the 36 styles for 14.6%, 22.9% and 10.0–12.6% of complaints, respectively.

- **Two different quantities.** A group gap is a net directional difference, in which shifts in opposite directions cancel. Pairwise disagreement is unsigned, so such shifts accumulate. The two quantities measure different things. Their ratio does not show which harm is larger, only that small average gaps can coexist with substantial individual instability.

### 4.3 What probabilistic outputs do and do not change

- **H3a \[C\], supported as registered.** The residual share is lower for E\[tier\] than for the argmax: 0.20 → 0.06 (GPT-4o-mini bank) and 0.13 → 0.03 (GPT-5-mini bank). This shows that the graded score varies less across versions. It does not show that remedies do.

- **Remedy rules \[PH\] (Table 10).** We compared explicit monetary-decision rules at matched monetary volume. In the GPT-4o-mini bank, the probability that two versions receive different monetary decisions is:
  - 3.4% for a threshold on Jev's p2 (order o012);
  - 3.2% when p2 is averaged over the six option orders (difference 0.14 pp \[−0.04, 0.31\]);
  - 3.8% for GPT-4o-mini's own label (difference from the Jev threshold rule 0.4 pp \[−0.4, 1.2\]).

  In the GPT-5-mini bank, Jev's threshold rule disagrees less often than GPT-5-mini's label (2.6% vs 6.3%; difference 3.7 pp \[2.9, 4.6\]). That is a difference between deciders, not an effect of graded output. Group gaps in monetary decisions stay within ±0.5 pp under every rule. GPT-4o-mini's label responds more to the positive-control facts than any Jev rule: the monetary-rate span between `strengthen` and `weaken` is 91.5 pp for GPT-4o-mini and 72–78 pp for the Jev rules. Smoother scores therefore did not translate into materially more stable remedies at the same monetary volume.

- **H3b (estimation).** No reduction in systematic disparity was detected: D(argmax) − D(E\[tier\]) = −0.006, corrected 95% CI \[−0.021, 0.005\]. No equivalence margin was pre-specified, so this does not show that disparity is unchanged.

- **H4 \[C\], supported.** On the pre-registered disparity measure D, averaging probabilities was not superior to equal-budget hard voting (one-sided test). This comparison concerns systematic disparity, not individual stability:

  - two paraphrases: D difference −0.005, 90% CI \[−0.014, 0.005\];

  - six option orders, GPT-4o-mini bank: +0.005 \[0.000, 0.013\]; that is, averaging performs slightly worse.

- **H5 \[C\], not supported.** No persona disparity was detected when the 20% most uncertain cases were referred (p = 0.32 in the primary bank). This is a failure to detect a disparity, not evidence that none exists. Jev's uncertainty itself shows little persona patterning \[S\].

- **Option order \[S/E\].**

  - There is no general first-position bias, although there is a small last-position penalty (−0.013).

  - The fully reversed order is an outlier, with E\[tier\] of 0.586 compared with 0.510–0.554 for the other orders (exploratory).

  - The chosen tier is identical across all six orders for 85% of texts.

  - Order does not interact with education or SES effects.

### 4.4 Generative LLM vs System One

- **H6 \[C\], inconclusive at n = 1,000.** Jev's systematic monetary disparity is below GPT-4o-mini's at matched monetary volume: ΔD = 0.010, corrected 95% CI \[−0.009, 0.023\]. The difference is significant only in the pooled n = 3,000 estimate \[PH\]: 0.012 \[0.002, 0.023\].

- **In rate units \[PH\].** Jev's SES gap is 0.64 pp smaller than GPT-4o-mini's (pooled \[0.27, 0.99\]). It is, however, 0.34 pp larger than GPT-5-mini's (\[0.03, 0.67\]). Every Jev-vs-GPT difference is equivalent to zero within ±1 pp (TOST p ≤ 0.03), and none exceeds 1 pp (minimum-effect tests, p ≥ 0.97).

- **Conclusion.** The three systems are practically equivalent in group disparity. We abandoned a pre-registered extension that could have confirmed the GPT-4o-mini contrast, because the difference would be immaterial.

### 4.5 Why style moves decisions: delivery, structure and content

- **H-A1 \[C\], supported: the measured content attributes show little systematic variation by persona.** Persona explains at most 2.3% of within-complaint variance in each content attribute; every one-sided upper bound is ≤ 0.032. Broad content scores could miss changes to individual facts, so fidelity is tested independently in §4.8.

- **The persona rewrites do, however, change the following (Fig. 5):**

  - delivery, strongly: vocabulary η² = 0.73, formality 0.71, grammar 0.58, politeness 0.37;

  - structure, moderately: chronology 0.26, argument quality 0.22.

- **H-A2 \[C\], supported: the education and SES gaps are associated mainly with delivery (Fig. 3).**

| Gap | Size | Explained by attributes | Delivery | Structure | Content |
|----|----|----|----|----|----|
| College − high school | +0.032 tier \[0.025, 0.039\] | 79% | +0.014 \[0.007, 0.021\] | +0.010 \[0.005, 0.014\] | +0.002 \[0.000, 0.003\] |
| High − low SES | +0.026 \[0.018, 0.034\] | 96% | +0.012 \[0.005, 0.019\] | +0.010 | +0.003 |

The delivery contribution is roughly 5–8× the content contribution. Gelbach's method describes how the persona coefficient changes when attributes are added. It does not identify causal mediators: the attributes are measured after the rewrite, are correlated, and are scored by Jev. Because individual delivery attributes are collinear, we interpret the decomposition at the family level only.

- **H-A3 \[C\], supported: the gap is larger where the style cue is recognisable (Fig. 4).**

  - GPT-4o-mini's education gap where Jev recognises the cue is +0.053 \[0.039, 0.068\];

  - where the cue is ambiguous, it is +0.009 \[−0.004, 0.022\];

  - the difference is +0.045 \[0.025, 0.063\]. The interval for ambiguous cues still allows a small positive gap.

- **H-A4 \[C\], supported: Jev's persona inferences track delivery, not content.** This concerns how recognisable the experimental style cue is, not the identity of real writers. Within-complaint R² is 0.24–0.79 for delivery, compared with 0.01–0.05 for content. By contrast, GPT's own persona labels vary very little \[E\]: GPT-4o-mini labels 96% of guises "young" and 92% "college", yet its decisions respond to style.

- **H-A5 \[C\], supported: Jev's own remedy judgment tracks relevant content.** R² is 0.20–0.21 for content, compared with 0.04 for delivery. Because Jev also scores the attributes, this evidence is weaker than that for H-A2.

### 4.6 Positive controls: the systems respond to facts that matter

- **PC1 and PC2 \[C\], supported (Fig. 7).**

| System        | `strengthen` effect       | Share moved up | `weaken` effect |
|---------------|---------------------------|----------------|-----------------|
| GPT-4o-mini   | +1.17 tier \[1.08, 1.26\] | 89%            | −0.41           |
| GPT-5-mini    | +0.89 \[0.78, 1.00\]      | 69%            | −0.39           |
| Jev E\[tier\] | +0.76 \[0.70, 0.82\]      | 98%            | −0.30           |

All confidence intervals for the `weaken` effect lie below zero.

- **PC4 \[C\], supported.** The fact span (strengthen − weaken) is 49× (GPT-4o-mini), 387× (GPT-5-mini) and 75× (Jev) the largest persona gap of the same system, against a pre-registered threshold of 5×.

- **Graded output is more sensitive to changes.** Jev's graded E\[tier\] registered the strengthened facts in 98% of complaints, compared with 69% for its argmax.

- `harm_x10` **\[E\].** This edit mainly affects the monetary dimension (Jev p2 +0.04; GPT monetary +0.06 to +0.12).

- **PC5 (descriptive).** Content attributes moved as intended:

  - `strengthen`: evidence +5.6 SD and bank error +3.9 SD (in units of within-complaint guise SD);

  - `weaken`: bank error −2.1 SD.

<!-- -->

- The edits were fact-led but not exclusively factual: `weaken` also lowered perceived confidence and coherence, and `strengthen` raised argument quality.

### 4.7 Mitigation: style standardisation reduces selected average gaps, not individual instability

Results are based on 500 holdout complaints × 36 guises (GPT-4o-mini bank); see Fig. 8 and Table 8. Gaps are within-complaint contrasts in tiers (GPT-4o-mini) or E\[tier\] (Jev), with 95% bootstrap CIs.

- **MH1 \[C\], supported: one standard style reduces GPT-4o-mini's education and SES gaps.**

| GPT-4o-mini gap | M0 (as written) | M1 (one style) | M2 (three styles) | Reduction M0 − M1 |
|----|----|----|----|----|
| College − high school | +0.031 \[0.021, 0.042\] | +0.002 \[−0.008, 0.011\] | +0.003 \[−0.003, 0.010\] | 0.029 \[0.017, 0.041\] |
| High − low SES | +0.024 \[0.012, 0.035\] | +0.003 \[−0.008, 0.014\] | +0.006 \[−0.001, 0.014\] | 0.021 \[0.007, 0.034\] |
| Old − young | +0.014 \[−0.002, 0.028\] | +0.007 \[−0.008, 0.020\] | +0.007 \[−0.003, 0.018\] | 0.007 \[−0.008, 0.023\] |

The point estimates fall by 94% (education) and 87% (SES). Both reductions meet the pre-registered criterion (reduction CI above 0 and \|gap_M1\| ≤ half of \|gap_M0\|). The M1 intervals (about ±0.01 tier) do not exclude small residual gaps. In monetary-decision units \[PH\], GPT-4o-mini's gaps move from +0.63 pp to +0.14 pp (education) and from +0.80 pp to +0.42 pp (SES). All M1 intervals include zero.

- **Jev's gaps (reported): reversed, not removed, by one style.** Jev's M0 gaps were all positive: age +0.015, education +0.008, SES +0.010. Under M1, age and education reverse sign, with magnitudes close to the originals and intervals that exclude zero: −0.012 \[−0.017, −0.006\] and −0.007 \[−0.011, −0.003\]. A standardiser can therefore reverse the direction of a disparity. Under M2, all three gaps are within ±0.001 of zero.

- **MH2 \[C\], not supported: no reduction in arbitrariness was established for one standard style.**

  - GPT-4o-mini MPAD is 0.147 (M0) versus 0.138 (M1), a difference of 0.009 \[−0.001, 0.020\]. The interval allows a modest reduction;

  - Jev MPAD is 0.084 versus 0.085, a difference of −0.001 \[−0.003, 0.001\].

<!-- -->

- A standardising rewrite is itself a new paraphrase: it removes the directional effect of persona but not the decision's sensitivity to wording.

<!-- -->

- **MH5 \[C\], supported: averaging over three styles reduces arbitrariness.**

  - M1 − M2: GPT-4o-mini 0.017 \[0.010, 0.025\]; Jev 0.017 \[0.016, 0.019\].

  - Relative to the original guises (M0 − M2), arbitrariness falls by 18% for GPT-4o-mini (0.147 → 0.120) and 19% for Jev (0.084 → 0.068).

  - Part of this gain is mechanical: M2 averages three decisions per guise, and an average of three decisions varies less than one decision even without different styles (compare H4). The design does not separate style diversity from the number of draws.

  - **At the level of monetary decisions \[PH\]**, a majority vote across the three styles changes the probability that two versions differ from 4.1% to 3.7% for GPT-4o-mini (difference 0.5 pp \[−0.2, 1.2\]) and from 2.3% to 1.9% for Jev (0.4 pp \[0.0, 0.8\]). The score-level reduction therefore translates into at most a small reduction in disagreeing remedies.

- **MH3 \[C\], supported: standardisation preserves fact sensitivity.**

| System             | Fact span, M0 | Retention, M1       | Retention, M2       |
|--------------------|---------------|---------------------|---------------------|
| GPT-4o-mini (tier) | 1.58          | 0.99 \[0.95, 1.03\] | 0.96 \[0.92, 0.99\] |
| Jev E\[tier\]      | 1.06          | 1.02 \[0.99, 1.05\] | 0.96 \[0.94, 0.99\] |

Every lower bound lies well above the pre-registered threshold of 0.5. M2 loses 4% of fact sensitivity, a small but statistically detectable reduction.

- **MH4 (descriptive): fidelity.**

  - **Content.** The measured content attributes are stable: for every content attribute, the complaint-level correlation between M0 and M1 is 0.99–1.00, and persona η² is ≤ 0.007 under M1. The independent fact audit of standardised texts is reported in §4.8.

  - **Delivery.** Persona η² collapses for most cues: grammar 0.55 → 0.04, formality 0.70 → 0.12, politeness 0.34 → 0.07, emotional intensity 0.17 → 0.02.

  - **Vocabulary.** Persona η² for vocabulary falls only partially (0.72 → 0.32), indicating that the standardiser still transmits part of the persona cue.

  - **Structure.** Persona η² collapses for chronology (0.21 → 0.004) and argument quality (0.23 → 0.01), but not for concision (0.12 → 0.13).

  - **Mean shifts.** Standardised texts are rated as more formal (+0.38 on the \[0, 1\] scale) and less emotional (−0.14) than the guises.

------------------------------------------------------------------------

## 5. Discussion

### 5.1 Two problems that need two different remedies

The central result is that profiling and arbitrariness are different problems with different causes, and each responds to a different intervention.

**Profiling.** In these systems, profiling looks like a *channel* problem. The education and SES gaps are statistically associated mainly with persona-associated delivery (vocabulary, grammar, formality and politeness; H-A2), and they are larger where the style cue is recognisable (H-A3). Standardising the writing reduced GPT-4o-mini's education and SES gaps by 87–94% in point estimates (MH1), which supports this reading. The intervention changes delivery and structure together, so it does not identify which feature matters.

**Arbitrariness.** Arbitrariness is a *variance* problem: the decision depends on incidental features of a particular wording, option order or inference draw. Substituting one wording for another did not detectably reduce it (MH2). Averaging reduced score-level variation, whether over standard styles (MH5) or by using graded probabilistic outputs in place of an argmax (H3a). In actual monetary decisions at matched volume, however, neither made remedies materially more stable (§4.3, §4.7).

Neither intervention resolves both problems. A system that standardises style can still give the same complaint a different remedy depending on how it happens to be worded; it is simply no longer worded differently *by group*. A system that averages probabilities produces smoother scores, but no reduction in profiling was detected (H3b). An audit that reports one aggregate disparity figure, or a mitigation evaluated against one, can hide the fact that only part of the problem has been addressed.

### 5.2 Materiality at two scales

Measured against group-rate benchmarks, all three systems' average gaps are small: on the held-out panel, every monetary-rate gap is below 0.5 pp, and every AIR is bounded above 0.80. These benchmarks are post hoc and borrowed from employment selection, and smaller differences can still constitute adverse impact \[29 CFR 1607.4(D)\]. So we read them as describing absolute size, not as settling whether the gaps matter.

At the individual scale the picture is different. Two style versions of the same complaint receive different monetary decisions 2.6–6.3% of the time. The two numbers measure different things: a net group difference, in which opposite shifts cancel, and an unsigned individual disagreement, in which they accumulate. Their ratio does not show which harm is larger. What the comparison does show is that small average gaps can conceal substantial individual instability. For a complainant, a different remedy is a different remedy whether it arises from a group pattern or from a wording lottery. Creel and Hellman \[2022\] argue that arbitrariness at scale carries moral weight of its own.

Group-rate tests remain necessary: they detected a real, replicated and directional signal. They should not be the only test.

### 5.3 Implications for audit practice

We suggest five practices that generalise beyond complaint handling.

1.  **Separate profiling from arbitrariness.** Measure the non-persona sources of variation directly (repeated scoring, option orders, independent same-persona paraphrases) and estimate variance components rather than ratios of distances. Report average persona effects, complaint-specific persona effects and instability separately. Without these controls, a matched-guise audit cannot determine which problem it has identified.

2.  **Test materiality, not just significance.** Equivalence and minimum-effect tests against a pre-stated bound make "small" a finding rather than a failure to reject.

3.  **Include positive controls.** A system may show small persona gaps simply because it ignores the text altogether. Fact edits establish that it responds to what matters; here, the response was 49–387× the largest persona gap.

4.  **Evaluate decision policies, not only scores.** Thresholds, referral rules and averaging change disparity and arbitrariness in different ways, and a gain in a score need not reach the remedy: graded outputs smoothed Jev's score but barely changed how often monetary decisions disagreed at matched volume. No referral disparity was detected here (H5), but this could not have been assumed in advance.

5.  **Identify the channel before mitigating, and audit the mitigation.** The attribute decomposition suggested that standardising delivery would reduce the gaps, and for GPT-4o-mini it did. For Jev, the same standardiser reversed two gaps. A mitigation must be audited with the same design as the system it corrects.

### 5.4 Style standardisation in practice

Standardisation is cheap and, by our measures, faithful. A standardising rewrite with GPT-4o-mini averaged about 390 input and 220 output tokens, one extra call of roughly the size of the remedy decision itself. Content attributes are preserved almost exactly, and sensitivity to remedy-relevant facts is retained to within a few per cent. A deployer can apply it without changing the decision model. It can also be combined with graded outputs, which address the variance that standardisation leaves untouched.

Three caveats apply.

- **The standardiser is itself a model that needs auditing.** It transmitted part of the vocabulary cue (η² 0.32). For Jev, a single canonical style overshot and reversed the age and education gaps. A mitigation can create a new directional bias, so it must be audited with the same matched-guise design as the decision system.

- **Averaging buys little stability in actual decisions.** Three styles triple the inference cost. The 18–19% reduction in score-level MPAD became a 0.4–0.5 pp reduction in the probability that two versions receive different monetary decisions. Graded outputs did no better at matched volume (§4.3). We found no inexpensive lever that makes individual remedies materially more stable; this remains open.

- **Standardisation removes signals as well as noise.** It made complaints less emotional (−0.14) and more formal. For most remedy decisions, that is the intent. However, distress can signal consumer vulnerability, which some regulators expect firms to recognise \[e.g. FCA, 2021\]. A standardised text should feed the remedy decision, not replace the original in the case record or in human review.

### 5.5 Generative LLMs and System One models

The three systems were practically equivalent in group disparity (§4.4), and all three responded strongly to facts (§4.6). The probabilistic interface nonetheless mattered:

- it exposed variation that single labels hide (§4.1);

- it enabled graded scores that vary less across versions, although at matched volume this did not make monetary decisions materially more stable;

- its distribution supported the decision-policy analyses (§4.3).

No reduction in profiling was detected, and on the disparity measure D averaging probabilities was not superior to voting. We make no claim about architecture: the comparison is between interfaces as deployed, and a single-label LLM sampled repeatedly could recover some of the same benefits.

------------------------------------------------------------------------

## 6. Limitations

- **No ground truth.** There is no known correct remedy for these complaints. Our claims concern sensitivity, disparity and arbitrariness, not accuracy: a system could be unbiased and consistently wrong.

- **Synthetic guises.** The persona styles were generated by LLMs following persona instructions. They are stylised representations of how groups write, not samples of real writers, and may exaggerate or miss real sociolinguistic variation. Effects on real complainants' own writing may differ in size. We did not study dialects, race, gender or English as a second language as manipulated factors.

- **LLM-generated edits and standardisation.** The positive-control edits and the standardising rewrites were produced by GPT-4o-mini. The edits were fact-led but not exclusively factual (§4.6), and the standardiser partly transmitted persona vocabulary (§4.7).

- **Shared model families.** GPT-4o-mini wrote one rewrite bank, the standardised texts and the edits, and it is also one of the deciders. Jev both scores the attributes and makes decisions. Self-preference or shared representations could inflate fidelity or alignment estimates. The second rewrite bank (GPT-5-mini) mitigates this for the main results but not for the mitigation, which used only the GPT-4o-mini bank.

- **One System One model, one version.** Jev 1.13.0 is not deterministic (mean TV 0.020 on re-scoring; 2.5% of argmax tiers flip). Results may not transfer to other probabilistic judgment models or later versions.

- **Model drift.** M0 decisions come from the 19 October run; M1 and M2 were decided on 21 October with the same model name and settings. Unannounced provider updates would bias M0-versus-M1 comparisons. A same-day baseline (§4.8) measures this directly. M1 and M2 were decided on the same day, which reduces but does not eliminate exposure to temporal or batch effects.

- **Scope.** The study covers US consumer financial complaints in English, one remedy prompt and a coarse three-tier outcome. The mitigation was tested on only 500 complaints and a single rewrite bank.

- **Post hoc analyses.** The materiality bounds (1 pp, AIR 0.80 and 0.90, \|h\| 0.2), the variance-component model and the remedy-rule analyses were all specified after the confirmatory results were seen, and are labelled post hoc. The four-fifths rule comes from US employment selection and has no formal status for complaint remedies.

- **LLM auditors.** The fact-fidelity audit uses two LLM auditors, one from the same provider as two of the audited models, validated against known fact edits but not against human annotation.

- **Averaging confound.** M2 combines style diversity with more decisions per guise, so its arbitrariness reduction cannot be attributed to style diversity alone.

------------------------------------------------------------------------

## 7. Ethical considerations and adverse impact

**Data.** We use narratives from the CFPB public Consumer Complaint Database, which consumers consented to publish and which the CFPB scrubs of personal information before release \[CFPB, 2015; CFPB, 2023\]. The CFPB states that scrubbing minimises, but does not eliminate, re-identification risk. No new data were collected from people, and no attempt was made to re-identify complainants. Rewritten guises and edits are synthetic.

**Representing groups through stereotyped style.** Matched-guise designs necessarily write in the stereotyped style of a group. We use these texts as measurement instruments, not as portraits, and release them with that caveat. We recommend that they not be used to train or prompt systems that infer demographics from writing.

**Anticipated adverse impacts. We identify three:**

- **Misreading "small."** A finding that average group gaps are small could be cited as clearance to deploy. It is not. The same systems are unstable for individuals (§5.2), and our results cover one task, three systems and stylised guises.

- **Loss of voice.** Standardisation changes how a complainant's own words reach the decision-maker. Used carelessly, it could erase signals of distress or vulnerability, or let an organisation describe a complaint in terms the complainant did not choose. We recommend standardising only the input to an automated decision, keeping the original text for the record and for human review, and auditing the standardiser itself.

- **Gaming.** Knowing that delivery moves decisions, or which facts move them most, could help complainants to present their cases more effectively. This arguably narrows existing advantages, but it could also help firms tune systems towards outcomes that favour them while passing group-rate audits. Reporting arbitrariness alongside profiling makes the latter more difficult.

**Pre-registration and disclosure.** Every confirmatory claim was pre-registered and tested once, and every deviation is disclosed in the analysis record (`prereg``/README.md`).

**Competing interests.** The author declares no competing interests.

**Funding.** This research received no external funding.

------------------------------------------------------------------------

## 8. Conclusion

When an LLM decides remedies from complainants' own words, writing style matters in two ways. At the group scale, older, college-educated and higher-SES styles receive slightly more generous remedies. This profiling is reliable, is associated mainly with how complaints are written rather than what they say, and in these systems is small in absolute terms. At the individual scale, the same complaint worded differently often receives a different monetary decision, and little of that variation is attributable to persona.

The two problems respond differently to intervention. Rewriting complaints into a standard style reduced GPT-4o-mini's education and SES gaps while preserving its sensitivity to the facts, but reversed two of Jev's gaps and did not make individual decisions more stable. Averaging smoothed scores but barely changed how often actual remedies disagreed. Audits that report a single disparity figure cannot tell these problems apart, and mitigations evaluated against one may leave the other in place. We recommend that fairness audits of LLM decision systems report average disparities and individual instability as separate outcomes, evaluate them on the decisions people actually receive, audit any mitigation with the same design, and confirm with positive controls and an independent fact audit that the system responds to the facts that should matter.
