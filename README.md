# LLM Fairness in Financial Complaint Resolution

Tests whether large language models exhibit discriminatory behavior when processing financial complaints. The experiment follows a 7-step methodology to detect linguistic profiling, establish causation through a matched-guise experiment, and measure bias reduction via language neutralization.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env with your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Download CFPB complaints CSV from:
#   https://www.consumerfinance.gov/data-research/consumer-complaints/
# Place at ./data/complaints.csv

# Run a small test (100 complaints, ~5 min)
python llm_fairness_experiment.py --num-complaints 100 --causal-sample-pct 0.20

# Full run (5000 complaints, ~2-4 hours with caching)
python llm_fairness_experiment.py
```

## What the Experiment Does

`llm_fairness_experiment.py` orchestrates all 7 steps end-to-end:

| Step | Name | What happens |
|------|------|-------------|
| 1 | Persona Estimation | LLM classifies each complaint's likely demographics (age, geography, education, socioeconomic, gender, ESL) |
| 2 | Remedy Decision (Original) | LLM assigns a remedy tier (no action / non-monetary / monetary) to the original text |
| 3 | Correlation Analysis | Statistical tests for persona-outcome associations (chi-squared, logistic regression, AIR) |
| 4 | Persona-Style Rewriting | A sample of complaints is rewritten into all 36 persona styles, preserving factual content |
| 5 | Remedy Decision (Rewritten) | LLM assigns tiers to rewritten complaints to test causation via within-complaint comparison |
| 6 | Language Neutralization | All complaints are rewritten into demographically neutral language |
| 7 | Remedy Decision (Neutralized) | Measure whether neutralization reduces discrimination |

After the 7 steps, additional analyses run automatically:

- **Slot-preservation audit** — Verifies that persona rewrites preserve factual content (dollar amounts, dates, entities, polarity) across four slot families
- **Robustness subset analysis** — Re-runs persona-dispersion analysis on quality-filtered subsets (HF, NO_POL, NO_NUM) to confirm findings are not driven by rewrite artefacts
- **Severity amplification** — Does bias increase with complaint severity (dollar amounts, issue category)?
- **Process discrimination** — Do certain personas face higher procedural burdens (information requests)?
- **Repeated-measures causal inference** — CLMM, conditional logit, GEE models for within-complaint effects
- **Bayesian robustness** — Stan-based hierarchical models as robustness checks
- **Executive summary** — GPT-generated 8-section business-audience commentary

## CLI Options

```bash
python llm_fairness_experiment.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--num-complaints` | 5000 | Number of base complaints to process |
| `--causal-sample-pct` | 0.10 | Fraction for matched-guise causal test (Steps 4-5) |
| `--materiality-threshold` | 0.80 | Adverse impact ratio threshold (80% rule) |
| `--models` | gpt-4o-mini gpt-5-mini | Models to test |
| `--threads` | 100 | Concurrent LLM request threads |
| `--seed` | 42 | Random seed for reproducibility |
| `--output-dir` | ./output | Directory for reports and artifacts |
| `--data-path` | ./data/complaints.csv | Path to CFPB complaints CSV |
| `--cache-dir` | ./cache | Directory for LLM response cache |
| `--skip-steps` | (none) | Comma-separated steps to skip, e.g. `4,5` |
| `--verbose` | false | Enable debug logging |
| `--clear-cache` | false | Clear all caches (LLM, text features, pipeline, analysis) |
| `--clear-analysis-cache` | false | Clear severity + Bayesian caches only |
| `--clear-bayes-cache` | false | Clear Bayesian model cache only |
| `--clear-exec-cache` | false | Clear executive summary cache only |

### Examples

```bash
# Run only correlation analysis (Steps 1-3)
python llm_fairness_experiment.py --skip-steps 4,5,6,7

# Single model with verbose logging
python llm_fairness_experiment.py --models gpt-4o-mini --verbose

# Re-run with fresh cache and stricter threshold
python llm_fairness_experiment.py --clear-cache --materiality-threshold 0.75
```

## Output

Each run produces the following in `--output-dir`:

| File | Description |
|------|-------------|
| `fairness_report_{timestamp}.html` | Interactive 8-tab HTML dashboard (executive summary, disparate impact, severity analysis, etc.) |
| `executive_summary.txt` | GPT-generated 7-section business summary |
| `worked-examples-{timestamp}.html` | Side-by-side complaint comparisons showing persona rewrites |
| `results_*.csv` | Raw results (persona estimates, remedy decisions, text features) |
| `analysis_results_*.json` | Complete statistical analysis output |
| `experiment_{timestamp}.log` | Full run log with timestamps |

Token usage and cost estimates are logged to `token-usage/token_usage_{timestamp}.json`.

## Persona Dimensions

The experiment tests bias across 36 persona combinations (2 x 3 x 2 x 3) from four causal dimensions:

| Dimension | Values |
|-----------|--------|
| Age | young (25-35), old (65+) |
| Geographic | urban, suburban, rural |
| Education | high school, college |
| Socioeconomic | low, middle, high |

Two additional dimensions (gender, ESL status) are estimated from text but not used in causal rewrites.

## Remedy Tiers

The LLM assigns each complaint to one of three tiers, calibrated to historical CFPB resolution rates:

| Tier | Label | Historical Rate |
|------|-------|----------------|
| 0 | No Action Required | 65.6% |
| 1 | Non-Monetary Action | 31.8% |
| 2 | Monetary Action | 2.6% |

## Caching

LLM responses are cached to disk using Blake2b hashes of `(model, task, narrative, prompt_version)`. This allows interrupted runs to resume automatically and analysis code to iterate without re-calling the API. Cache invalidates automatically when prompt templates change (see `PROMPT_VERSIONS` in [prompts.py](src/prompts.py)).

Data pipeline steps are cached as parquet files in `data/.pipeline/`, invalidated when the source CSV changes.

## Setup Details

### Prerequisites

- Python 3.10+
- OpenAI API key
- CFPB complaints CSV (~500MB)

### Installation

```bash
pip install -r requirements.txt
```

For exact reproducibility, use the pinned lockfile:

```bash
pip install -r requirements.lock
```

### Fasttext Language Model

The data cleaning pipeline uses fasttext for language detection. Pre-download the model to avoid a silent download on first run:

```bash
python -c "from src.data_loader import ensure_fasttext_model; ensure_fasttext_model()"
```

### Data

Download the CFPB Consumer Complaint Database CSV from https://www.consumerfinance.gov/data-research/consumer-complaints/ and place it at `./data/complaints.csv`. Only complaints with a non-empty `Consumer complaint narrative` field are used.

## Project Structure

```
.
├── llm_fairness_experiment.py          # Main entry point — runs the full experiment
├── matched_guise_robustness_subsets.py # Robustness subset analysis (FULL, HF, NO_POL, NO_NUM)
├── requirements.txt
├── .env                                # API keys (not in version control)
│
├── slot_audit/                         # Matched-guise slot-preservation audit package
│   ├── slot_audit.py                  # Main audit pipeline + CLI
│   ├── extract_numeric.py             # Dollar/percent/count extraction
│   ├── extract_temporal.py            # Duration/frequency/year extraction
│   ├── extract_polarity.py            # Complaint polarity extraction
│   ├── extract_anchors.py             # Named entity extraction
│   ├── compare_slots.py               # Family-level comparison logic
│   ├── aggregate_report.py            # Metrics aggregation + viability gate
│   └── tests/                         # Unit tests for all extractors
│
├── src/
│   ├── experiment.py                   # 7-step orchestration + executive summary
│   ├── llm_client.py                   # OpenAI client with threading, retry, circuit breaker
│   ├── cache.py                        # Lock-free disk-based response cache
│   ├── analysis_cache.py              # Analysis-level caching (severity, Bayesian)
│   ├── data_loader.py                  # CFPB data ingestion and 10-step cleaning pipeline
│   ├── schemas.py                      # Pydantic models, dataclasses, persona grid
│   ├── prompts.py                      # LLM prompt templates (versioned)
│   ├── json_utils.py                   # JSON extraction, repair, validation
│   ├── text_features.py               # 26 text metrics extraction
│   ├── token_usage.py                  # Token accounting and cost estimation
│   ├── statistics.py                   # Disparate impact analysis (AIR, chi-squared, logistic)
│   ├── process_discrimination.py       # Information request burden analysis
│   ├── severity_analysis.py            # Severity amplification (7 sections, multithreaded)
│   ├── repeated_measures.py            # CLMM, GEE, conditional logit models
│   ├── bayes_robustness.py             # Stan-based Bayesian robustness checks
│   ├── ipw_robustness.py              # Inverse probability weighting robustness
│   ├── mitigation_analysis.py         # Bias mitigation effectiveness
│   ├── reporter.py                     # HTML report generation (8 tabs, 20+ charts)
│   ├── worked_examples.py             # Side-by-side complaint comparisons
│   ├── table_generator.py             # LaTeX/CSV table generation
│   ├── figure_generator.py            # Publication-quality figures
│   └── paper_bundle.py                # Academic paper artifact bundling
│
├── one-shot-examples/                  # Executive summary examples (7 files)
├── specs/                              # Experiment specifications and methodology
├── docs/                               # Technical documentation
├── contracts/                          # Publication validation gate
├── data/                               # CFPB CSV + fasttext model (gitignored)
├── cache/                              # LLM response cache (auto-created)
├── models/stan/                        # Compiled Stan models (auto-created)
├── token-usage/                        # Token usage logs (auto-created)
└── output/                             # Reports and artifacts (auto-created)
```

## Further Documentation

- [LLM API Calls](docs/llm-api-calls.md) — API call reference
- [LLM Prompts](docs/llm-prompts.md) — Prompt templates and versioning
- [Tiers and Severities](docs/tiers-and-severities.md) — Remedy tiers and severity scoring
- [CFPB Data](docs/cfpb-data.md) — Data source, category mappings, exclusions
- [Slot-Preservation Audit](docs/matched-guise-slot-preservation-audit.md) — Rewrite quality verification (4 slot families, viability gate)
- [Robustness Subset Analysis](docs/matched-guise-robustness-subsets.md) — Effect stability under quality filtering
- [Severity Amplification](docs/step-08-severity-amplification.md) — Severity x persona interaction analysis
- [Bayesian Robustness](docs/bayesian-analysis.md) — Stan-based hierarchical model robustness checks
- [Experiment Spec v1.2](specs/llm_fairness_experiment_specification_v1.2.md) — Full methodology
- [Severity Amplification Spec](specs/severity-amplification-specification.md) — Severity analysis design
- [Advanced Statistics Spec](specs/advanced-statistical-specifications.md) — Repeated measures and Bayesian methods
