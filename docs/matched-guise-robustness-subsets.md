# Matched-Guise Robustness Subset Analysis

## Overview

The robustness subset analysis tests whether the matched-guise causal findings (Steps 4-5) are robust to data quality filtering. It runs the core persona-dispersion analysis on five overlapping subsets of the matched-guise data — from the full unfiltered dataset to increasingly strict quality filters — and compares the results. If filtering to high-quality pairs weakens or eliminates the observed persona effects, the findings may be driven by rewrite artefacts rather than genuine LLM bias.

**Source:** `matched_guise_robustness_subsets.py` (~830 lines)
**Entry point:** `run_robustness_analysis()`
**Dependencies:** Slot audit pair-level CSV + persona decisions CSV

---

## Why This Analysis Exists

The slot-preservation audit (see [slot-preservation-audit.md](matched-guise-slot-preservation-audit.md)) identifies pairs where the rewriting process introduced factual mutations — altered dollar amounts, dropped dates, or flipped claim polarity. These mutations could create spurious differences in LLM remedy decisions that look like persona-based bias but are actually content-based.

The robustness subset analysis addresses this by asking: **if we exclude pairs with known quality problems, do the persona-dispersion effects persist, weaken, or strengthen?**

Three possible outcomes:
1. **Effects persist or strengthen** → findings are robust; mutations are not driving the signal
2. **Effects weaken but remain significant** → findings are directionally robust but overestimated in the full dataset
3. **Effects disappear** → findings may be artefactual; the observed bias was driven by content differences rather than style-based discrimination

---

## Subsets

Five subsets are tested, each applying a different quality filter:

| Subset | Filter | Rationale |
|--------|--------|-----------|
| `FULL` | None (all matched-guise pairs) | Baseline for comparison |
| `HF` | High-fidelity pairs only (`high_fidelity_pass == True`) | Removes numeric mutations, polarity contradictions, and temporal/anchor mutations |
| `NO_POL` | No polarity contradictions (`has_contradiction == False`) | Tests sensitivity to the most serious quality failure |
| `NO_NUM` | No numeric mutations (`numeric_status != "MUTATION"`) | Tests sensitivity to altered dollar amounts |
| `HF_NO_POL` | High-fidelity AND no contradictions | Most restrictive filter; may alias to `HF` if high-fidelity already excludes contradictions |

### Alias Detection

If the `HF_NO_POL` subset is identical to `HF` (because the high-fidelity definition already requires no polarity contradictions), the analysis detects this automatically and marks `HF_NO_POL` as an alias. This prevents redundant computation and reporting.

---

## Analysis Pipeline

The analysis proceeds through 8 stages:

### 1-2. Load and Map Columns

Loads two CSV files and maps their columns to canonical names:

| Canonical Name | Decisions CSV Column | Audit CSV Column |
|----------------|---------------------|-----------------|
| `complaint_id` | `case_id` | `complaint_id` |
| `model_name` | `model` | `model_name` |
| `persona_label` | `variant` | `persona_label` |
| `tier` | `tier` | — |
| `tier2` | derived: `(tier == 2)` | — |
| `need_more_info` | derived: `(confidence == "need_more_info")` | — |
| `high_fidelity_pair_pass` | — | `high_fidelity_pass` |
| `polarity_contradiction_flag` | — | `has_contradiction` |
| `numeric_mutation_flag` | — | derived: `(numeric_status == "MUTATION")` |

**Validation checks:**
- Tier values must be in {0, 1, 2}
- At least 1 model present
- At least 2 persona levels present
- Required columns exist in both files

### 3. Join

Merges the decisions and audit DataFrames on `[complaint_id, model_name, persona_label]`. Writes `join_diagnostics.json` with row counts and coverage rate. Fails if coverage drops below 95%.

### 4. Subset Construction

Applies the five filter masks to the joined DataFrame. Each mask is a boolean Series indexing the rows that belong to that subset.

### 5. Metric Computation

Per subset x model, computes:

| Metric | Description |
|--------|-------------|
| `n_pairs` | Number of matched-guise pairs in this subset |
| `n_complaints` | Number of unique complaints |
| `pair_retention` | `n_pairs / n_pairs_FULL` — fraction of FULL pairs retained |
| `complaint_retention` | `n_complaints / n_complaints_FULL` |
| `mean_tier` | Average remedy tier (0-2 scale) |
| `p_tier2` | Proportion receiving tier 2 (monetary relief) |
| `p_need_more_info` | Proportion receiving "need more info" response |
| `sd_persona_p_tier2` | Standard deviation of persona-level p(tier2) — **primary CI-bearing metric** |
| `sd_persona_mean_tier` | Standard deviation of persona-level mean tier — **primary CI-bearing metric** |
| `q9010_p_tier2` | Q90-Q10 of persona-level p(tier2) — descriptive spread |
| `q9010_mean_tier` | Q90-Q10 of persona-level mean tier — descriptive spread |
| `max_persona_gap_p_tier2` | Max minus min of persona-level p(tier2) — descriptive worst-case |
| `max_persona_gap_mean_tier` | Max minus min of persona-level mean tier — descriptive worst-case |

SD (standard deviation of persona-level estimates) is the primary CI-bearing dispersion statistic. Its bootstrap distribution is unbiased (unlike order-statistic-based spread measures), and the normal CI (`SD ± 1.96 × SE_boot`) is centred on the plug-in estimate by construction. Q90-Q10 and max-min gap are retained as descriptive companions but carry no CI or containment assertion (order-statistic spread measures are positively biased under resampling). If SD shrinks meaningfully in filtered subsets, the findings may not be robust.

### 6. Bootstrap Confidence Intervals

Cluster bootstrap 95% CI (complaint-level resampling):

1. Sample complaint IDs with replacement (default: 1,000 iterations)
2. Take all rows for each sampled complaint
3. Recompute eight key metrics: `mean_tier`, `p_tier2`, `max_persona_gap_p_tier2`, `max_persona_gap_mean_tier`, `q9010_p_tier2`, `q9010_mean_tier`, `sd_persona_p_tier2`, `sd_persona_mean_tier`
4. Compute confidence intervals — method depends on the metric:
   - **SD metrics (`sd_persona_*`) — log-scale normal CI:** Inference on `log(SD + ε)`, back-transformed: `[exp(y − 1.96·SE_y) − ε, exp(y + 1.96·SE_y) − ε]`. Guarantees non-negative bounds and handles right-skew when dispersion is small.
   - **Location metrics (`mean_tier`, `p_tier2`) — BCa:** Bias-corrected and accelerated interval with jackknife acceleration, falling back to symmetric percentile if BCa produces a degenerate interval.
   - **Spread metrics (`max_persona_gap_*`, `q9010_*`) — BCa:** Computed for completeness but no containment assertion (order-statistic spread measures are positively biased under resampling).

Complaint-level (rather than pair-level) resampling respects the clustering structure — all persona rewrites of the same complaint are included together. A diagnostic check verifies that every bootstrap replicate retains the full set of persona labels (since complaints are the resampling unit and each complaint contributes all 36 personas, persona counts should be consistent across replicates).

### 7. GEE Omnibus Tests

Per subset x model, fits two Generalized Estimating Equations models:

| Model | Outcome | Family | Purpose |
|-------|---------|--------|---------|
| Binomial GEE | `tier2` (0/1) | Binomial | Tests whether persona affects probability of monetary relief |
| Gaussian GEE | `tier` (0-2) | Gaussian | Tests whether persona affects average remedy tier |

Both models use `persona_label` dummies as predictors and cluster by `complaint_id`. The working correlation structure falls back from Exchangeable to Independence if the former fails to converge.

The omnibus Wald test evaluates whether the persona block is jointly significant (p < 0.05). Skipped if fewer than 30 complaints or fewer than 2 persona levels.

### 8. Comparison vs FULL + Verdicts

For each non-FULL subset x model x primary effect (`sd_persona_p_tier2`, `sd_persona_mean_tier`):

**Effect stability metrics:**
- `abs_ratio = |effect_subset| / |effect_full|`
- `attenuation = 1 - abs_ratio`
- `effect_change_class` (±10% band, i.e. ratio in [0.9, 1.1]):
  - `"stable"` if attenuation between -0.10 and +0.10
  - `"amplified"` if attenuation < -0.10 (effect is larger in subset)
  - `"attenuated"` if attenuation > +0.10 (effect is smaller in subset)

**CI overlap check:** Do the BCa bootstrap CIs for FULL and the subset overlap? Non-overlapping CIs indicate a statistically meaningful change.

**Omnibus significance class:** Did the GEE p-value change class (significant ↔ non-significant) between FULL and the subset?

---

## Verdict Logic

Each subset x model receives a verdict:

### Hard Gates (immediate FAIL)

| Gate | Threshold | Default |
|------|-----------|---------|
| Pair retention | >= 0.40 | 40% of FULL pairs must survive filtering |
| Complaint retention | >= 0.25 | 25% of FULL complaints must survive |

### Effect Stability

| Condition | Outcome |
|-----------|---------|
| All effects pass attenuation (<=0.40) AND no CI gaps AND no significance class changes | **PASS** |
| Some effects flagged but not all, no hard gate failures | **PASS_WITH_CAUTION** |
| Multiple attenuation failures (>= 2x the number of effects) | **FAIL** |
| Hard gate failure | **FAIL** |

### Overall Verdict

The overall analysis verdict is:
- **PASS** if all subset x model verdicts are PASS
- **FAIL** if any verdict is FAIL
- **PASS_WITH_CAUTION** otherwise

**Default thresholds** (from `DEFAULT_THRESHOLDS`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_pair_retention` | 0.40 | Minimum fraction of pairs retained |
| `min_complaint_retention` | 0.25 | Minimum fraction of complaints retained |
| `max_attenuation` | 0.40 | Maximum allowable effect attenuation |
| `alpha` | 0.05 | Significance level for GEE tests |
| `min_complaints_for_tests` | 30 | Skip GEE if fewer complaints |
| `min_persona_levels_for_tests` | 2 | Skip GEE if fewer persona levels |

---

## Retention-Balance Analysis

An additional diagnostic checks whether high-fidelity filtering disproportionately excludes certain groups:

**Persona balance:** For each model, computes the HF inclusion rate per persona label and runs a chi-squared test for independence between persona and HF status. A significant result (p < 0.05) indicates that some personas produce systematically lower-quality rewrites.

**Severity balance:** If severity data is available, computes HF inclusion rate per severity band per model. This checks whether high-severity complaints are disproportionately filtered out.

---

## Outcome Composition Shift

Compares the outcomes (mean tier, p(tier2)) of HF-retained pairs vs HF-dropped pairs:

| Direction | Meaning |
|-----------|---------|
| Dropped pairs have **higher** mean tiers | **Anti-conservative**: the full dataset slightly overstates effects; filtered results are conservative |
| Dropped pairs have **lower** mean tiers | **Conservative**: the full dataset understates effects |
| No significant difference | **Neutral**: filtering does not change the outcome distribution |

A two-sample t-test and proportion z-test determine statistical significance.

---

## Output Files

The analysis produces the following in its output directory:

| File | Description |
|------|-------------|
| `subset_manifest.csv` | Per-subset x model: pair counts, complaint counts, retention rates |
| `endpoint_summary.csv` | Per-subset x model: metrics + BCa bootstrap CIs |
| `omnibus_tests.csv` | GEE results per subset x model x endpoint |
| `robustness_vs_full.csv` | Attenuation, CI overlap, significance class comparison |
| `robustness_flags.json` | Hierarchical verdicts (per subset x model, overall) |
| `robustness_report.md` | Human-readable markdown summary |
| `join_diagnostics.json` | Join coverage statistics |
| `analysis_fingerprint.json` | Provenance: input file hashes, timestamps, parameter values |

---

## Interpreting Results

### The Ideal Outcome

The best-case result for the experiment's validity is:

1. **Overall verdict: PASS** — all subsets and models pass
2. **Effects amplified in HF subset** — persona gaps are **larger** after filtering to high-quality data, meaning rewrite artefacts were diluting (not inflating) the bias signal
3. **All CIs overlap** — no statistically meaningful change in effects across subsets
4. **No significance class changes** — GEE remains significant (or non-significant) across all subsets
5. **Anti-conservative composition shift** — dropped pairs had higher tiers, meaning the full-dataset estimates were conservative

### Warning Signs

- **Attenuation > 40%** — the effect shrinks substantially when quality filtering is applied
- **CI non-overlap** — the BCa bootstrap CI for a filtered subset does not overlap with FULL
- **Significance class changes** — a formerly significant effect becomes non-significant (or vice versa) in a filtered subset
- **Pair retention < 40%** — the filter removed too many pairs for reliable comparison
- **Differential persona balance** — high-fidelity filtering disproportionately excludes specific persona styles

### Reading the Robustness Report

The markdown report (`robustness_report.md`) provides:

1. **Data summary** — join coverage, pair/complaint counts per model
2. **Subset retention** — how many pairs survive each filter
3. **Effect stability** — per-subset x model comparison table with attenuation, CI overlap, and change class
4. **Omnibus GEE results** — significance of persona effects per subset
5. **Composition shift** — direction and significance of outcome differences between kept and dropped pairs
6. **Verdicts** — per-subset x model and overall

---

## CLI Usage

```bash
python matched_guise_robustness_subsets.py \
    --decisions-csv output/results_persona_decisions_*.csv \
    --pair-audit-csv output/slot_audit/pair_slot_audit.csv \
    --output-dir robustness_out/ \
    --bootstrap-reps 1000 \
    --seed 42
```

| Option | Default | Description |
|--------|---------|-------------|
| `--decisions-csv` | (required) | Path to persona decisions CSV from Step 5 |
| `--pair-audit-csv` | (required) | Path to pair-level slot audit CSV |
| `--output-dir` | `robustness_out/` | Output directory for all files |
| `--bootstrap-reps` | 1000 | Number of bootstrap iterations |
| `--seed` | 42 | Random seed for reproducibility |
| `--alpha` | 0.05 | Significance level |
| `--min-pair-retention` | 0.40 | Hard gate: minimum pair retention |
| `--min-complaint-retention` | 0.25 | Hard gate: minimum complaint retention |
| `--max-attenuation` | 0.40 | Maximum allowable effect attenuation |
| `--no-hf-no-pol` | false | Disable the HF_NO_POL combined subset |

The analysis runs as part of the main experiment automatically after the slot audit completes and persona decisions are available.

---

## Relationship to Other Steps

| Component | Relationship |
|-----------|-------------|
| Slot-Preservation Audit | Produces the `pair_slot_audit.csv` used as input |
| Step 5 (Rewritten Decisions) | Produces the `results_persona_decisions_*.csv` used as input |
| Paper Bundle | Generates LaTeX tables R1 (retention), R2 (effect stability), R3 (composition shift) |
| Executive Summary | Section 7 (Robustness & Data Quality) includes robustness subset findings |
| Bayesian Robustness | Independent robustness check; complements the subset analysis from a different angle |
