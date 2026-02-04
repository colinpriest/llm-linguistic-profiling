## 1) Pre-registered analysis plan subsection (paste-ready, ~1–2 pages)

### Pre-registered analysis plan: endpoints, estimands, and severity amplification

This study audits whether a commercial large language model (LLM) produces **unequal remedy recommendations** for semantically equivalent consumer complaints when the only manipulated signal is  **linguistic style** . To reduce researcher degrees of freedom and prevent post-hoc framing of severity amplification as exploratory, we pre-specify (i) primary and secondary endpoints, (ii) causal and associational estimands, (iii) a two-part approach to partially observed monetary severity, and (iv) decision rules for inference and robustness.

#### Primary and secondary endpoints

**Primary endpoint (distributive fairness).** The primary ethical endpoint is the **remedy recommendation** produced by the LLM, recorded as an ordinal tier (T \in {0,1,2}), where Tier 0 indicates no action warranted, Tier 1 indicates non-monetary remediation or guidance, and Tier 2 indicates monetary remediation is recommended. Because Tier 2 is the most consequential allocation in this setting, the primary binary endpoint for inferential tests is:
[
Y_{2} = \mathbb{1}[T=2]
]
We additionally report the ordinal tier as a descriptive outcome and, where appropriate, an auxiliary binary endpoint (Y_{1+}=\mathbb{1}[T\ge 1]) to capture “any positive remediation” as a robustness check.

**Secondary endpoint (operational friction).** The secondary endpoint captures workflow burden rather than headline fairness: whether the model requests additional information, (F=\mathbb{1}[\texttt{need_more_info}=1]). This is analysed as an indicator of consumer effort and operational cost, not as the primary fairness outcome.

#### Experimental units and design

The unit of analysis is a **(complaint, persona-style, model)** decision. The study includes (a) correlational analyses over naturally occurring complaints, and (b) a **matched-guise within-complaint** design in which each complaint is rewritten into multiple persona-coded linguistic styles under strict fact-preservation constraints. The matched-guise subset provides the primary basis for causal interpretation because it holds complaint content fixed while varying style.

To account for repeated measures induced by the matched-guise design, all inferential tests for persona effects and interactions use complaint-clustered methods (mixed effects or GEE; see below).

#### Pre-specified estimands

##### (A) Causal estimand: style/persona effect on remedy recommendation (matched-guise)

Let (i) index complaints, (p) index persona-style variants, and (m) index models. For the matched-guise subset, the causal estimand is the **within-complaint difference** in Tier-2 probability between persona styles:
[
\Delta_{p,p'}^{(m)} ;=; \mathbb{E}\big[ Y_{2}(i,p,m) - Y_{2}(i,p',m) \big]
]
We report: (i) max–min gaps across persona styles, (ii) dimension-level gaps (age, geography, education, socioeconomic status), and (iii) a global dispersion measure (SD of persona-specific predicted probabilities), all with bootstrap confidence intervals clustered by complaint.

##### (B) Severity amplification estimand: interaction between persona and severity

Severity amplification is defined as  **effect modification** : whether the persona disparity in Tier-2 recommendations differs across severity levels. The primary amplification estimand is the interaction term in a pre-specified model, tested separately by model (m).

For monetary severity (defined below), let (S\in{\text{Low},\text{Medium},\text{High}}). Define the persona disparity function (D(S)) as the max–min gap in predicted probability of Tier-2 across persona styles (or across a specified persona dimension). The amplification estimand is:
[
A ;=; D(S=\text{High}) - D(S=\text{Low})
]
We also report the ratio (R = D(\text{High})/D(\text{Low})) when (D(\text{Low})>0), with bootstrap CIs.

**Primary hypothesis test for amplification:** a joint test of persona×severity interaction terms in the pre-specified model (below). Descriptive “amplification ratios” are not treated as confirmatory unless supported by this interaction test.

#### Two-part monetary severity handling (pre-specified)

Monetary amounts are not extractable for a substantial proportion of complaints. We pre-specify that “unknown amount” is **not** treated as the lowest severity level because missingness is expected to be systematic by complaint type and is plausibly related to the remedy decision. We therefore model monetary severity using a two-part structure:

1. **Amount presence indicator:**
   [
   M=\mathbb{1}[\text{extractable dollar amount present}]
   ]
2. **Conditional monetary severity band:** for complaints with (M=1), assign severity (S\in{\text{Low},\text{Medium},\text{High}}) using pre-defined dollar ranges, with all range boundaries fixed before analysis.

**Confirmatory monetary amplification analysis is conducted only within (M=1)** (conditional estimand). We additionally report a pre-specified robustness analysis to address selection into (M=1) (below).

#### Statistical models (pre-specified)

##### Primary outcome model (Tier-2; matched-guise and non-matched analyses)

For each model (m), we estimate a complaint-clustered logistic model for (Y_2). In the matched-guise subset, we use either a mixed-effects logistic regression with a random intercept for complaint or a GEE with exchangeable correlation by complaint:
[
\text{logit};P(Y_2=1) = \alpha + \text{Persona} + \text{Severity} + (\text{Persona}\times\text{Severity}) + \mathbf{X} + u_i
]
where (u_i) is a complaint random intercept (mixed model) or the complaint cluster (GEE), and (\mathbf{X}) includes pre-specified complaint-level controls used only in correlational and robustness analyses (e.g., issue category, product category, word count). In the matched-guise analysis, (\mathbf{X}) may be omitted because complaint identity already blocks time-invariant complaint content; we retain (\mathbf{X}) only for sensitivity checks.

**Amplification is tested via the joint significance of Persona×Severity terms.**

##### Secondary friction model (need_more_info)

We estimate an analogous complaint-clustered logistic model for (F), reported descriptively and as secondary inference.

#### Multiple testing and error control

We pre-specify a single primary family for confirmatory inference:  **persona×severity interaction tests on (Y_2)** , by model and by severity metric (monetary within (M=1); issue severity on full sample). Within each family, we control the false discovery rate at 5% using Benjamini–Hochberg. All other analyses (pairwise persona comparisons, exploratory slices, additional outcomes) are labelled exploratory.

#### Robustness analyses (pre-specified)

1. **Selection into amount-present subset (IPW).** Because (M) may be systematically related to complaint type and remedy tier, we estimate (P(M=1|\mathbf{Z})) using complaint-level predictors (\mathbf{Z}) (issue category, product category, word count, and any pre-specified text-derived covariates available pre-rewrite). We then reweight the (M=1) subset by (w=1/P(M=1|\mathbf{Z})) and re-estimate the monetary amplification model. We report whether the amplification estimand (A) materially changes under IPW.
2. **Alternative disparity functional.** We repeat amplification reporting using (i) max–min gap and (ii) SD of persona-specific predicted probabilities. Interaction tests remain the confirmatory criterion.
3. **Outcome definition sensitivity.** We repeat the amplification analysis using (Y_{1+}) (Tier ≥ 1) as a secondary distributive endpoint to test whether any interaction is specific to monetary remediation.

#### Reporting commitments

We commit to reporting: (i) coverage of monetary severity ((P(M=1))); (ii) descriptive differences between (M=1) and (M=0) complaints; (iii) confirmatory interaction tests for amplification; (iv) effect size estimates (disparities by severity) with uncertainty; and (v) a clear distinction between conditional monetary amplification claims (within (M=1)) and population-level claims (supported only if robustness checks indicate stability).

---

## 2) Script outputs to generate (paper + appendices) with specs

Below is a **shopping list of auto-generated artifacts** your Python pipeline should emit each run, designed for copy-paste into a JBE paper.

### A) “Paper-ready” tables (main text)

**Table 1 — Dataset and sampling summary**

* **Purpose:** Describe corpus, filters, and representativeness.
* **Rows:** total complaints, included complaints, excluded, language filters, word-count range, stratification variables.
* **Columns:** count, %, notes.
* **Format:** 3-line APA-ish table; no vertical rules; 1 decimal for %.

**Table 2 — Endpoint definitions**

* **Rows:** remedy tier (0/1/2), tier≥1, tier=2, need_more_info, monetary severity (M and bands), issue severity.
* **Columns:** operational definition, rationale/ethical interpretation, used as (primary/secondary/exploratory).

**Table 3 — Persona grid and rewrite design**

* **Rows:** each persona dimension with levels (age, geo, edu, SES) and total combinations; N matched-guise complaints; total decisions.
* **Columns:** levels, values, N, notes on constraints.

**Table 4 — Main distributive fairness results (matched-guise, primary outcome = Tier 2)**

* **Separate panels by model** (or separate tables if big).
* **Rows:** each persona dimension (age, geo, edu, SES) + “full persona grid”.
* **Columns:** disparity metric (max–min predicted P(Tier2)), SD, 95% CI, interaction p (if applicable), interpretation.

**Table 5 — Neutralisation mitigation (primary endpoint)**

* **Rows:** model × metric.
* **Columns:** baseline disparity, neutralised disparity, Δ, % change, CI, verdict (reduced/eliminated/increased/unchanged).

**Table 6 — Severity amplification (confirmatory)**

* **Separate for:** issue severity (full sample) and monetary severity (within M=1).
* **Rows:** model × severity metric.
* **Columns:** interaction test (F/χ², p, q), amplification A (high–low disparity), ratio R, CI, coverage (for monetary: P(M=1)), caveat note.

### B) “Paper-ready” figures (main text)

For each figure, output:

* a **PDF** (vector) and a **PNG** (300+ dpi),
* an accompanying **caption text file** (`.txt`) with 2–4 sentences,
* and a machine-readable **metadata JSON** (for reproducibility: N, seed, model names, filters).

(See fixed Figure plan in section 3 below.)

### C) Appendices / Supplementary tables (auto-generated)

**Appendix Table A1 — Full persona × tier distribution**

* 36 persona cells, per model: counts and % for each tier.
* Include cell-size flags (<30, <50).

**Appendix Table A2 — Pairwise persona comparisons (matched-guise)**

* Only include:
  * effect size (risk difference or odds ratio on Tier2),
  * CI,
  * BH-adjusted q.
* Provide a filtered “top-k” view (e.g., top 20 by absolute effect).

**Appendix Table A3 — Monetary missingness diagnostics**

* Differences between M=1 vs M=0:
  * issue category distribution,
  * product category distribution,
  * word count distribution,
  * tier distribution,
  * mean tier.
* Include test used, statistic, p, effect size.

**Appendix Table A4 — Robustness checks**

* IPW vs unweighted amplification,
* alternative disparity functionals,
* alternative endpoint (Tier≥1).

**Appendix A5 — Prompt inventory and parameters**

* Hash of each prompt,
* model, temperature, top_p, seed strategy,
* caching keys and reproducibility notes.

### D) Worked examples pack (supplementary qualitative)

**Figure/Appendix — Case studies (2–5 complaints)**
Each case should output a single-page “card”:

* Original complaint (truncated to a fixed max length),
* 1–2 persona rewrites,
* neutralised version,
* tier outcomes by model,
* short rationale snippets.

### E) Reproducibility bundle (for you + peer review)

* `run_config.json` (seed, sampling strata, model versions, date/time)
* `data_manifest.csv` (hashes of input files, output files)
* `analysis_notebook.ipynb` or `analysis_report.md` with tables inserted

**Formatting spec:** Every table output should support:

* Markdown table (`.md`) for quick preview
* LaTeX table (`.tex`) using `booktabs`
* Word-friendly table (`.docx`) optional but very helpful for JBE workflows

---

## 3) Fixed figure plan text (cleaned)

**Figure plan (what to draw)**

1. **Figure 1 — End-to-end experiment pipeline**
   Flowchart: CFPB sampling → dollar extraction → persona estimation → baseline tier decision → matched-guise persona rewrites → neutralisation → re-run tier decision; annotate which model is used at each step.
2. **Figure 2 — “Default customer” persona distributions (RQ1)**
   Two panels (one per model): marginal distributions for age / geography / education / SES / gender / ESL status; plus a heatmap of the most common full persona combination.
3. **Figure 3 — Correlation vs causation (RQ2)**
   Side-by-side: (a) slice-and-dice disparities by inferred persona on the natural complaint set, (b) matched-guise within-complaint effect sizes (same complaint, different persona-coded writing style).
4. **Figure 4 — Distributive fairness gaps by dimension (matched-guise)**
   For each dimension (age, geography, education, SES), plot predicted probability of each tier (or Tier 2 as primary) with confidence intervals; highlight the largest gaps.
5. **Figure 5 — Operational friction (need_more_info) by dimension**
   For each model: raw disparities and matched-guise results; include a note where the metric is degenerate (e.g., near-zero requests).
6. **Figure 6 — Neutralisation effects (outcomes and friction)**
   Before/after plots: change in between-persona disparity for Tier 2 (primary) and change in need_more_info (secondary); include an inset showing whether neutralised text shifts toward a particular demographic-coded “house style” (e.g., age/gender leaning).
7. **Figure 7 — Monetary severity missingness diagnostic**
   Two-part figure: (a) proportion unknown vs known dollar amount, (b) distribution shifts by issue category and product category, plus tier distribution differences between amount-present and amount-missing complaints.
8. **Figure 8 — Severity amplification as an interaction**
   For each model: persona disparity metric vs severity band (low/medium/high), shown separately for (i) monetary severity among amount-present complaints only, and (ii) issue severity on the full sample; include a note on sparse cells and the planned 20k rerun.
9. **Figure 9 — Worked examples (qualitative)**
   A 2–3 case panel showing: original complaint, one persona rewrite, neutralised version, and resulting tiers (and short rationales).
10. **Figure 10 — Robustness and sensitivity**
    Amplification estimates under alternative disparity metrics (max–min vs SD vs AIR where applicable), and results with and without reweighting for selection into the amount-present subset (IPW).

---

If you want, I can also give you a **folder/file naming convention** for all these outputs so your script produces a clean “paper bundle” per run (e.g., `run_id/tables/main/Table4_matched_guise_tier2.tex`, `run_id/figures/Fig8_amplification.pdf`, etc.).
