# Matched-Guise Slot-Preservation Audit

## Overview

The slot-preservation audit verifies whether persona-style rewrites in the matched-guise experiment (Step 4) faithfully preserve the factual content of the original complaint. For the causal inference to be valid, a rewrite must change **only the writing style** while keeping dollar amounts, dates, named entities, and claim polarity intact.

The audit extracts structured "slots" from both the original and rewritten text using regex-based extractors, compares them family by family, and produces per-pair preservation statuses, aggregate metrics, and a viability gate that determines whether the data quality is sufficient for causal analysis.

**Source:** `slot_audit/` package (7 modules + tests)
**Entry point:** `slot_audit/slot_audit.py` → `run_audit()`
**Aggregation:** `slot_audit/aggregate_report.py` → `compute_metrics()`

---

## Why This Audit Exists

The matched-guise experiment rewrites each complaint into 36 different persona styles (2 ages x 3 geographies x 2 educations x 3 socioeconomic levels). If the rewriting process inadvertently changes factual content — adding dollar amounts, dropping dates, or flipping the polarity of a claim — then observed differences in LLM remedy decisions might reflect content differences rather than style-based discrimination.

The slot audit provides:
1. **Per-pair quality flags** used to filter the analysis to high-fidelity data
2. **Aggregate quality metrics** reported in the paper's robustness section
3. **A viability gate** that must pass before causal findings are reported

---

## Slot Families

The audit checks four families of factual content:

### 1. Numeric Slots

**Extractor:** `slot_audit/extract_numeric.py` → `extract_numeric()`
**Slot type:** `NumericSlot(subtype, value, unit, operator, raw)`

Extracts dollar amounts, percentages, APRs, counts, and numeric bounds from complaint text using regex patterns.

| Subtype | Examples | Canonical Form |
|---------|----------|----------------|
| `money` | $1,500, $45.99, 1500 dollars | `money:1500.00` |
| `percent` | 15%, 3.5 percent | `percent:15.00` |
| `apr` | 24.99% APR | `apr:24.99` |
| `count` | 3 times, 5 payments | `count:3:times` |
| `bound` | over $500, at least $1,000 | `bound:>:500.00` |

**Comparison logic:** Source and rewrite canonical sets must be identical for PRESERVED_PRESENT. Any difference in values constitutes a MUTATION.

### 2. Temporal Slots

**Extractor:** `slot_audit/extract_temporal.py` → `extract_temporal()`
**Slot type:** `TemporalSlot(subtype, value, unit, relation, raw)`

Extracts durations, frequencies, and year references.

| Subtype | Examples | Canonical Form |
|---------|----------|----------------|
| `duration` | 3 months, 2 years | `duration:3:months` |
| `frequency` | twice a month, 4 times | `frequency:2:month` |
| `year` | 2023, in 2024 | `year:2024` |

**Comparison logic:** Same set-equality comparison as numeric slots.

### 3. Polarity Slots

**Extractor:** `slot_audit/extract_polarity.py` → `extract_polarity()`
**Output:** Dictionary mapping event descriptions to polarity labels (`positive`, `negative`, `neutral`)

Extracts complaint/compliment polarity using a configurable event dictionary. Polarity extraction masks anchor spans to avoid double-counting named entities.

**Comparison logic (special):**
- If both maps are identical → `PRESERVED_PRESENT`
- If a shared key has **opposite** polarity (e.g., `negative` → `positive`) → `MUTATION_POLARITY_CONTRADICTION`
- Other mismatches → `MUTATION`

Polarity contradictions are the most serious quality failure because they could flip the direction of the complaint (e.g., turning a grievance into praise).

### 4. Anchor Slots

**Extractor:** `slot_audit/extract_anchors.py` → `extract_anchors()`
**Output:** Set of canonical anchor strings + list of character spans

Extracts named entities: company names, product names, and other domain-specific anchors using a configurable anchor dictionary. The anchor extractor also returns character spans that are used by the polarity extractor for masking.

**Comparison logic:** Same set-equality comparison as numeric slots.

---

## Slot Statuses

Every family comparison produces one of six statuses, defined in `slot_audit/__init__.py`:

| Status | Meaning | Pass? |
|--------|---------|-------|
| `PRESERVED_ABSENCE` | Neither source nor rewrite has this slot type | Yes (strict) |
| `PRESERVED_PRESENT` | Both have identical canonical slot sets | Yes (strict) |
| `OMISSION` | Source has slots, rewrite does not | No (strict) |
| `ADDITION` | Source has no slots, rewrite adds them | No (strict) |
| `MUTATION` | Both have slots but sets differ | No (strict) |
| `MUTATION_POLARITY_CONTRADICTION` | Polarity-specific: opposite polarity on shared event | No (strict, contradiction) |

---

## Pass Criteria

### Strict Pass

A pair passes the strict test if **all four families** have status `PRESERVED_ABSENCE` or `PRESERVED_PRESENT`. This is a demanding standard — the pair must be perfectly faithful across all slot types.

```
strict_pass = all(family_status in {PRESERVED_ABSENCE, PRESERVED_PRESENT} for each family)
```

### High-Fidelity Pass

A pair passes the high-fidelity test under a more lenient standard that tolerates minor temporal/anchor additions or omissions but still requires numeric exactness and no polarity contradictions:

```
high_fidelity = (
    no polarity contradictions
    AND numeric status in {PRESERVED_ABSENCE, PRESERVED_PRESENT}
    AND temporal status != MUTATION
    AND anchor status != MUTATION
)
```

The rationale: temporal and anchor additions/omissions (e.g., a rewrite mentioning "2024" when the source didn't, or vice versa) are less concerning than numeric mutations (changing $500 to $5,000) or polarity flips.

### Weighted Pair Score

Each pair also receives a continuous score from 0.0 to 1.0:

```
score = 0.30 * numeric_pass + 0.25 * temporal_pass + 0.30 * polarity_pass + 0.15 * anchor_pass
```

If a polarity contradiction is detected, a 0.30 penalty is subtracted (floored at 0.0).

**Default weights** (from `slot_audit/__init__.py`):

| Family | Weight | Rationale |
|--------|--------|-----------|
| Numeric | 0.30 | Dollar amounts directly affect remedy decisions |
| Temporal | 0.25 | Dates/durations provide complaint context |
| Polarity | 0.30 | Flipped polarity invalidates the entire pair |
| Anchor | 0.15 | Entity names are less likely to affect remedy tier |

---

## Viability Gate

The viability gate determines whether data quality is sufficient for the matched-guise causal analysis. All six thresholds must be met:

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| High-fidelity pair rate | >= 0.40 | At least 40% of pairs must be usable |
| High-fidelity complaint rate | >= 0.25 | At least 25% of complaints must have usable rewrites |
| Polarity contradiction rate | <= 0.05 | No more than 5% of pairs may have flipped polarity |
| Numeric mutation rate | <= 0.10 | No more than 10% of pairs may have altered numbers |
| Temporal presence preservation | >= 0.70 | At least 70% of temporal slots preserved |
| Anchor presence preservation | >= 0.70 | At least 70% of anchor slots preserved |

**Thresholds are defined in** `slot_audit/aggregate_report.py` → `REAL_DATA_THRESHOLDS`.

If the viability gate fails, the causal findings from Steps 4-5 should not be reported as confirmatory evidence.

---

## Aggregate Metrics

The `compute_metrics()` function produces a nested dictionary with three levels:

### Global Metrics

Top-level quality statistics across all pairs:
- `n_pairs`, `n_complaints`
- `strict_pair_pass_rate` (with Wilson CI)
- `polarity_contradiction_rate`, `numeric_mutation_rate`
- `high_fidelity` sub-dict: pair rate, complaint rate (with CIs)
- `viability_gate` sub-dict: pass/fail + per-check details

### Per-Family Metrics

For each of the four families:
- `presence_preservation_rate` — proportion of pairs where a source slot was correctly reproduced (with CI)
- `overall_preservation_rate` — includes preserved-absence pairs
- `omission_rate`, `addition_rate`, `mutation_rate`
- `contradiction_rate` (polarity family only)

### Subgroup Metrics

Broken down by:
- **Model name** — allows per-model quality comparison (e.g., GPT-5-mini vs GPT-4o-mini)
- **Persona label** — checks whether certain personas produce systematically lower-quality rewrites

---

## Complaint-Level Aggregation

The `aggregate_complaints()` function rolls pair-level results up to one row per complaint:
- Per-family pass rates across all of that complaint's rewrites
- Strict pass count/rate across rewrites
- High-fidelity count/rate
- Contradiction count/rate
- Boolean flags: `meets_strict`, `meets_practical`, `meets_high_fidelity`

A complaint `meets_high_fidelity` only if **all** of its rewrites are high-fidelity (strict complaint-level criterion).

---

## Output Files

The audit produces the following files in its output directory:

| File | Description |
|------|-------------|
| `pair_slot_audit.csv` | Per-pair statuses, pass flags, and scores |
| `complaint_slot_audit.csv` | Per-complaint aggregated quality metrics |
| `preserved_complaints.csv` | Subset of complaints meeting strict criteria |
| `high_fidelity_pairs.csv` | Pairs passing the high-fidelity test |
| `high_fidelity_complaints.csv` | Complaints where all rewrites are high-fidelity |
| `slot_preservation_metrics.json` | Full metrics dictionary (global, per-family, subgroups) |
| `slot_preservation_report.md` | Human-readable markdown summary |

---

## Interpreting Results

### Reading the Metrics JSON

The key numbers to check first:

1. **`global.viability_gate.pass`** — Did the overall quality gate pass? If `false`, the matched-guise findings should be treated with caution.

2. **`global.high_fidelity.pair_rate`** — What fraction of pairs are high-fidelity? Higher is better. In our experiment, 84.1% of pairs passed high-fidelity.

3. **`global.polarity_contradiction_rate`** — The most serious quality failure. Should be well below 5%. Values above 5% indicate the rewriting process is systematically altering claim direction.

4. **`per_family.*.presence_preservation_rate`** — How well each slot type was preserved. Anchor slots are typically best preserved (>95%), while polarity and temporal slots may have lower rates.

5. **`subgroups.model_name`** — Compare rewrite quality across LLM models. A model with higher strict pass and high-fidelity rates produces more reliable experimental data.

### What "Good" Results Look Like

- Viability gate: PASS (all 6 checks met)
- High-fidelity pair rate: >80%
- Polarity contradiction rate: <3%
- Numeric mutation rate: <5%
- No systematic variation by persona (check `subgroups.persona_label`)

### What "Concerning" Results Look Like

- High polarity contradiction rate (>5%) — rewrites are flipping complaints into compliments
- Low numeric preservation (<70%) — dollar amounts are being altered
- Large model-to-model quality gap — one model may not be suitable for matched-guise experiments
- Systematic persona variation — if certain persona styles consistently produce worse rewrites, this could confound the causal analysis

---

## Relationship to Other Steps

| Component | Relationship |
|-----------|-------------|
| Step 4 (Persona Rewriting) | Produces the source/rewrite pairs audited here |
| Step 5 (Rewritten Decisions) | Uses the pair-level flags for robustness filtering |
| Robustness Subset Analysis | Filters matched-guise data by high-fidelity pass to test effect stability |
| Paper Bundle | Copies audit metrics and report to the evidence folder; generates LaTeX tables |
| Executive Summary | Section 7 (Robustness & Data Quality) includes slot audit findings |

---

## Running the Audit Standalone

```bash
# From cache files (typical usage within the experiment)
python -m slot_audit --from-cache cache/ --output-dir output/slot_audit/

# From a pre-built pairs CSV
python -m slot_audit --pairs-csv path/to/pairs.csv --output-dir output/slot_audit/
```

The audit runs as part of the main experiment automatically when persona rewrites are available.
