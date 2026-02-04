# Paper Bundle Output Spec (Folder/File Naming Convention + Analysis Plan Compliance)

This document specifies the **required outputs**, **folder structure**, **file naming conventions**, and **analysis-plan compliance rules** for the Python experiment pipeline so results can be pasted directly into the paper (JBE target) and appended as supplementary materials.

---

## 0) Run identity and reproducibility

### 0.1 Run ID

**Format**

- `RUN_ID = YYYYMMDD_HHMMSS__seed{SEED}__n{N_COMPLAINTS}__models{MODELSET}`

**Example**

- `20260204_091500__seed42__n20000__modelsgpt4omini-gpt5mini`

### 0.2 Canonical output root

All outputs for a run must be written to:

- `artifacts/runs/{RUN_ID}/`

### 0.3 Required reproducibility files

Every run must include:

- `manifest/manifest.json` (all inputs/outputs + hashes)
- `manifest/run_config.json` (all config parameters + model settings)
- `manifest/data_manifest.csv` (input file hashes + sampling stats)
- `manifest/git_state.json` (commit hash, dirty flag, branch)
- `manifest/env_freeze.txt` (pip/conda freeze)
- `manifest/prompts_fingerprints.json` (prompt templates + sha256)

---

## 1) Folder structure (must create all directories)

artifacts/runs/{RUN_ID}/
manifest/
logs/
data/
sampled/
derived/
prompts/
api/
analysis/
reports/
tables/
main/
appendix/
robustness/
figures/
main/
appendix/
stats/
qualitative/
paper_pack/
latex/
tables/
figures/
docx/
tables/
figures/
markdown/
tables/
figures/

markdown
Copy code

### 1.1 Directory purposes (brief)

- `data/sampled/` : sampled complaints (IDs + text + strata labels)
- `data/derived/` : extracted dollar amounts, severity bins, rewrite metadata
- `prompts/` : prompt templates actually used (verbatim)
- `api/` : request/response cache summaries (no raw PII)
- `analysis/reports/` : narrative report(s) and pre-reg plan verification report
- `analysis/tables/*` : machine + paper-ready tables
- `analysis/figures/*` : PDF/PNG figures + captions
- `analysis/stats/` : model objects, bootstrap draws, test results, BH results
- `qualitative/` : worked examples “cards”
- `paper_pack/` : copy-paste bundles in LaTeX/Word/Markdown

---

## 2) File naming rules (global)

### 2.1 General naming pattern

Use:

- `{kind}__{scope}__{topic}__{model}__{subset}__v{N}.ext`

Where:

- `kind`: `table` | `fig` | `report` | `stats` | `data` | `card`
- `scope`: `main` | `appendix` | `robustness` | `manifest`
- `topic`: short snake_case descriptor
- `model`: `gpt4omini` | `gpt5mini` | `both` | `na`
- `subset`: `all` | `matched` | `amount_present` | `amount_missing` | `neutralized` | `baseline` | etc.
- `vN`: integer version within run (start at `v1`)

**Example**

- `table__main__distributive_fairness_tier2__gpt5mini__matched__v1.tex`
- `fig__main__amplification_interaction__both__amount_present__v1.pdf`

### 2.2 Allowed extensions

- Tables: `.tex`, `.md`, `.csv`
- Figures: `.pdf`, `.png`, plus `.txt` caption
- Reports: `.md`, `.pdf`
- Stats: `.json`, `.pkl` (if needed), `.parquet`
- Qual: `.md`, `.pdf`

### 2.3 Paper-pack duplication rule

Every “main” figure/table must also be copied into:

- `paper_pack/latex/...` and `paper_pack/markdown/...`
  Optionally also:
- `paper_pack/docx/...` if you generate docx-friendly tables.

---

## 3) Required outputs (minimum deliverables)

### 3.1 Reports

1) `analysis/reports/report__main__paper_summary__na__all__v1.md`
2) `analysis/reports/report__main__prereg_compliance__na__all__v1.md`  ✅ (new)
3) `analysis/reports/report__appendix__full_results_index__na__all__v1.md`

### 3.2 Paper-ready main tables (each in .tex + .md + .csv)

- `Table1`: dataset + sampling summary`table__main__dataset_sampling_summary__na__all__v1.*`
- `Table2`: endpoint definitions`table__main__endpoint_definitions__na__all__v1.*`
- `Table3`: persona grid + rewrite design`table__main__persona_grid_design__na__matched__v1.*`
- `Table4`: main distributive fairness results (Tier 2, matched-guise)`table__main__distributive_fairness_tier2__{model}__matched__v1.*`
- `Table5`: neutralisation effects (Tier 2 primary + friction secondary)`table__main__neutralization_effects__{model}__baseline_vs_neutralized__v1.*`
- `Table6`: severity amplification (confirmatory)
  `table__main__severity_amplification_confirmatory__{model}__{severity_metric}__v1.*`

### 3.3 Appendix tables (minimum)

- A1 full persona × tier distribution`table__appendix__persona_by_tier_fullgrid__{model}__matched__v1.*`
- A2 pairwise persona comparisons (top-K + full)`table__appendix__pairwise_persona_effects_topk__{model}__matched__v1.*``table__appendix__pairwise_persona_effects_full__{model}__matched__v1.*`
- A3 monetary missingness diagnostics`table__appendix__monetary_missingness_diagnostics__na__all__v1.*`
- A4 robustness summary`table__robustness__robustness_summary__{model}__all__v1.*`
- A5 prompt inventory + parameters
  `table__appendix__prompt_inventory__na__all__v1.*`

### 3.4 Main figures (each as PDF + PNG + caption TXT)

- Fig1 pipeline`fig__main__pipeline_overview__na__all__v1.(pdf|png)` + caption
- Fig2 default persona distributions`fig__main__default_persona_distributions__{model}__all__v1.*`
- Fig3 correlation vs causation`fig__main__correlation_vs_causation__{model}__all__v1.*`
- Fig4 distributive fairness gaps (Tier 2)`fig__main__distributive_gaps_tier2__{model}__matched__v1.*`
- Fig5 operational friction`fig__main__operational_friction_need_more_info__{model}__all__v1.*`
- Fig6 neutralisation effects`fig__main__neutralization_effects__{model}__baseline_vs_neutralized__v1.*`
- Fig7 monetary missingness diagnostic`fig__main__monetary_missingness_diagnostic__na__all__v1.*`
- Fig8 amplification interaction`fig__main__amplification_interaction__{model}__{severity_metric}__v1.*`
- Fig9 worked examples panel`fig__main__worked_examples_panel__both__all__v1.*`
- Fig10 robustness/sensitivity
  `fig__main__robustness_sensitivity__{model}__all__v1.*`

### 3.5 Qualitative “cards”

- `qualitative/card__appendix__worked_example_{CASE_ID}__both__all__v1.md`
- Optional: `.pdf` render.

---

## 4) Table formatting specs (for JBE-friendly copy/paste)

### 4.1 LaTeX tables

- Use `booktabs` (`\toprule \midrule \bottomrule`)
- No vertical lines
- Numerical formatting:
  - proportions: 3 decimals
  - p-values: 2 sig figs (scientific if <1e-3)
  - effect sizes: 3 decimals
- Must include a `\caption{...}` and `\label{tab:...}`
- Use consistent column headers across models.

### 4.2 Markdown tables

- Same column order as LaTeX
- Include a one-line title above the table as `**Table X. ...**`

### 4.3 CSV

- Plain machine-readable; no formatting symbols
- Include columns for `run_id`, `model`, `subset`, `metric`, `value`.

---

## 5) Figure formatting specs

- Output both:
  - `PDF` (vector; primary for LaTeX)
  - `PNG` (>=300 DPI; for Word / quick preview)
- No hard-coded colours unless specified; ensure grayscale legibility.
- Each figure must have:
  - `fig__...__v1.txt` caption file (2–4 sentences)
  - `fig__...__v1.json` metadata (N, subset definition, model(s), seed)

---

## 6) Additional specs to enforce the pre-registered analysis plan

### 6.1 Endpoint priority (must be enforced)

**Primary distributive endpoint**

- `Y2 = 1[tier == 2]` (Tier 2 probability)

**Secondary distributive endpoint**

- `Y1p = 1[tier >= 1]` (reported as sensitivity)

**Secondary operational endpoint (friction)**

- `F = 1[need_more_info == 1]` (report as operational friction, not headline fairness)

The script must label every result as:

- `confirmatory` or `exploratory`

Only the following are confirmatory:

- persona×severity interaction tests on `Y2` (by model, by severity metric), with stated multiplicity correction.

### 6.2 Severity amplification is an interaction, not a main effect

For each `model` and each `severity_metric`:

- Fit a complaint-clustered model for `Y2` including:
  - `persona`, `severity`, and `persona × severity`
- The script must output:
  - joint test p-value for `persona × severity`
  - amplification estimand `A = disparity_high - disparity_low`
  - amplification ratio `R = disparity_high / disparity_low` when defined
  - bootstrap 95% CIs clustered by complaint
- The script must explicitly avoid using “bias SD increases” alone as the headline amplification claim.

### 6.3 Two-part monetary severity handling (must be implemented)

Define:

- `M = 1[dollar_amount_extractable]`
- `S ∈ {low, medium, high}` only for `M==1`

Rules:

- Confirmatory monetary amplification uses only `subset = amount_present (M==1)`
- “Unknown amount” is **not** treated as a severity level in the confirmatory trend.
- The script must always output:
  - `coverage = mean(M)`
  - a missingness diagnostic table (issue/product/wordcount/tier differences)

### 6.4 Missingness robustness (required)

Implement **IPW** for monetary amplification:

1) Fit `P(M=1 | Z)` using complaint-level predictors `Z`:
   - issue_category, product_category, word_count (minimum)
2) Weighted analysis within `M==1`:
   - weight `w = 1 / P(M=1 | Z)`
3) Output side-by-side comparison:
   - unweighted vs IPW amplification results (A, R, CIs, interaction p/q)

### 6.5 Repeated-measures handling (required)

For matched-guise analyses, the script must use one of:

- mixed-effects logistic with random intercept for complaint, or
- GEE with clustering by complaint

And must output:

- the method used
- cluster sizes distribution
- any convergence warnings

### 6.6 Multiple testing control (required)

- For each confirmatory “test family” (by model × severity metric):
  - apply Benjamini–Hochberg FDR at 5%
- Output:
  - raw p-values, adjusted q-values
  - number of tests in family
  - number surviving

### 6.7 Decision rules for claims (must auto-generate)

The script must generate a short “claim classification” string per model × metric:

- **Amplification supported**: interaction q < 0.05 AND A > 0 with CI excluding 0
- **No evidence of amplification**: interaction q ≥ 0.05
- **Inconclusive**: interaction q < 0.05 but A CI includes 0, OR sparse cells warning
- **Not analysable**: insufficient cell sizes / degenerate outcomes

### 6.8 Sparse cell warnings (required)

If any persona×severity cell has:

- `< 30` observations → warning
- `< 10` observations → analysis suppressed for that grid; require dimension-level aggregation

The script must output:

- a “cell size heatmap” and a table of flagged cells.

### 6.9 Output labelling and provenance (required)

Every table/figure must embed in metadata:

- `run_id`, `seed`, `n_complaints`, `n_matched`, model version identifiers, date/time.
  Every caption must include:
- subset definition (e.g., “amount-present only”)
- endpoint (Tier2 vs Tier>=1)
- confirmatory vs exploratory label.

---

## 7) Suggested stable “paper handles” (optional but recommended)

Create a stable pointer file for easy LaTeX includes:

- `paper_pack/latex/tables/_index.tex`  (\\input lines for all main tables)
- `paper_pack/latex/figures/_index.tex` (\\includegraphics lines)
- `paper_pack/markdown/_index.md`        (links to all outputs)

---

## 8) Minimal acceptance checklist (script should fail if missing)

The run is **invalid** unless all exist:

- prereg compliance report (`report__main__prereg_compliance__...`)
- Table4 (Tier2 matched-guise) for each model
- Table6 (severity amplification confirmatory) for each model×metric
- Fig7 (missingness diagnostic)
- Fig8 (amplification interaction)
- BH correction outputs (`analysis/stats/bh_fdr__...json`)
- IPW robustness outputs for monetary severity

---
