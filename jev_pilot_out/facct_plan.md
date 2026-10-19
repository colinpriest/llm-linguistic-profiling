# FAccT 2027: Jev paper, revised plan (19 Oct)

**Deadlines (official CFP, facctconference.org/2027/cfp.html; Brisbane time):**

| Step | Brisbane time |
|---|---|
| Abstract registration (mandatory) | Wed 28 Oct, 9:59 pm AEST |
| Full paper | Wed 4 Nov, 9:59 pm AEST |

**Format:** 14 pages excluding references, plus 1 page for ethics and adverse-impact statements. ACM TAPS. Fully anonymised; the earlier GPT matched-guise study is cited in the third person.

**Working title:** *Profiling or Arbitrariness? Auditing Style Sensitivity in Probabilistic Judgments of Consumer Complaints*

## Framing
Two decision interfaces, compared as each is deployed, on identical inputs:
- **Generative LLMs** (GPT-4o-mini; GPT-5-mini with reasoning) that return one remedy label. The earlier paper's results are reused here.
- **A System One model** (Jev 1.13) that returns a probability distribution over remedies.

No causal claim about architecture. There are no log-probability arms: token log-probabilities measure something different from Jev's output, and GPT-5-mini only returns them with reasoning switched off, which would change the decision process being evaluated.

## Research questions
- **RQ1 (measurement).** Do fact-preserving style changes shift remedy probabilities and uncertainty when the chosen tier does not change? How much is persona-specific (profiling) and how much is generic sensitivity to paraphrase or option order (arbitrariness)?
- **RQ2 (policy).** At matched budgets, do decision rules that use probabilities (a monetary-probability threshold, uncertainty-based referral) create, move or reduce persona disparity, relative to hard labels and to GPT?
- **RQ3 (mitigation).** Do graded outputs, option-order averaging, multi-rewrite ensembles or structured-fact inputs reduce disparity relative to the between-complaint signal? Is averaging probabilities better than hard voting with the same number of calls?

## Data
1. **Exploratory panel (done):** 2,000 complaints x 36 guises x 2 rewrite banks, both option orders, GPT decisions on the same texts. Already analysed (dist_audit/findings.md), so it is reported as exploratory.
2. **Holdout panel (new):** fresh complaints drawn from the 18,000 cleaned complaints that were never rewritten or scored by Jev. They appeared only in the earlier paper's observational arm, which used original texts. Same rewrite prompts, models and settings as the exploratory panel, so the cross-bank paraphrase control carries over. Rewrites were at temperature 0, so a second gpt-4o-mini rewrite would not give an independent paraphrase; the second bank therefore stays gpt-5-mini.
3. **Pre-registration.** Before any holdout text is generated, freeze the analysis script, endpoints, thresholds and practical bounds as a timestamped commit (and optionally on OSF). The holdout is analysed once with that frozen script.

## Confirmatory hypotheses (holdout)
> **Superseded by prereg/jev_facct_prereg.md**, which is authoritative. There, H3 is split into H3a (E[tier] reduces arbitrariness) and H3b (systematic disparity, estimation only), and H4 is a falsifiable "not superior" prediction. Both changes follow the noise-floor correction to the disparity metric.

| # | Hypothesis | Test / bound |
|---|---|---|
| H1 | The persona-specific share of within-complaint distributional variation is small | (cross-bank different-persona TV minus same-persona TV) / same-persona TV < 5% |
| H2 | Systematic persona effects on E[tier] and the hard tier exist, in the exploratory direction (older, college and higher SES receive more) | complaint-bootstrap CI excludes 0; within-complaint permutation |
| H3 | E[tier] has lower standardised persona disparity than argmax (persona SD divided by between-complaint SD) | paired complaint bootstrap |
| H4 | Probability averaging is no better than equal-budget hard voting | equivalence margin +/-10% on the standardised metric |
| H5 | Referring the 20% most uncertain cases yields a persona disparity in referral rate | permutation p; gap with CI |
| H6 | At matched monetary volume, Jev's disparity is below GPT-4o-mini's and not below GPT-5-mini's | paired bootstrap |

Benjamini-Hochberg correction within the confirmatory family. "No disparity" claims require the CI to sit inside a stated practical bound, never just p > 0.05. Distributional p2 effects are secondary: at about 0.001, they need far more complaints than a holdout can afford.

## Arms
| Arm | Panel | New calls | Approx. cost |
|---|---|---|---|
| Repeat scoring (determinism) | 1,000 exploratory texts | Jev 1k | < $0.10 |
| All 6 option orders: the 4 orders not yet scored, asked in one call per text | exploratory, 150k texts | Jev 150k, about 2.8k tokens each | ~$18 |
| Structured facts (cached GPT fact sets rendered as text) | exploratory | Jev up to 150k, fewer after deduplication | <= ~$10 |
| **Holdout rewrites** | 1,000 complaints x 36 x 2 banks | gpt-4o-mini 36k, gpt-5-mini 36k | ~$10 + ~$76 |
| **Holdout GPT decisions** (keeps the GPT-vs-Jev contrast confirmatory) | own-bank guises | gpt-4o-mini 36k, gpt-5-mini 36k | ~$10 + ~$76 |
| **Holdout Jev scoring**, 6 orders plus persona questions | 75k texts | Jev 75k, about 4.4k tokens each | ~$14 |

Per-call GPT costs are the averages from the February runs ($0.00029 for gpt-4o-mini, $0.0021 for gpt-5-mini). A 5-complaint smoke test will measure the real cost before the full holdout runs.

**Totals:**
- Exploratory-panel Jev arms: about $28.
- Holdout with GPT decisions: about $186. Without GPT decisions: about $100 (the GPT contrast then stays exploratory).
- Grand total: about $215, or about $128 without holdout GPT decisions. A 500-complaint holdout costs half as much but has too little power for the SES and education gaps.

## Engineering
- A new script, holdout_panel.py, samples fresh complaints with a fixed seed. It excludes the 2,000 already used and calls the existing src/ prompt and rewrite functions (same PROMPT_VERSIONS). It stores texts directly so the slow cache reconstruction isn't needed. Outputs go to a separate folder, holdout_out/.
- jev_pilot.py gains the 6-order question set. The existing outputs stay untouched.

## Schedule (Brisbane)
| Date | Work |
|---|---|
| Mon 19 Oct | Write and freeze the pre-registration and analysis script; write holdout_panel.py; 5-complaint smoke test (about $1); run the repeat-scoring and 6-order arms on the exploratory panel |
| Tue 20 Oct | Holdout rewrites and GPT decisions (a few hours); structured-facts arm |
| Wed 21 Oct | Jev scoring of the holdout; run the frozen confirmatory analysis once |
| Thu 22 – Sun 25 Oct | Figures, tables, full first draft |
| Wed 28 Oct, 9:59 pm | Abstract registration |
| Mon 26 Oct – Tue 3 Nov | Revise, anonymity check, ethics and adverse-impact page |
| Wed 4 Nov, 9:59 pm | Submit |

## Dropped
- Log-probability arm.
- Neutralising every guise. The neutralised-original scores already exist; the rest is a possible extension.
- Positive-control fact edits. The structured-facts arm partly covers this.

## Risks
- **Single vendor model.** Mitigated by framing the paper as a comparison of two decision interfaces, with a general method.
- **First-option bias in Jev 1.13.** Becomes an explicit factor through the 6 permutations.
- **No ground truth for remedies.** Claims are about sensitivity and disparity, not correctness.
- **Overlap with the GPT paper.** If that paper is resubmitted before 23 Mar, each paper must disclose the other.
