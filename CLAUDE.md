# LLM Fairness in Financial Complaint Resolution

A research experiment testing whether large language models exhibit discriminatory behavior when processing financial complaints. Uses a 7-step methodology to detect linguistic profiling, establish causation through matched-guise experiments, and analyze bias mitigation through language neutralization.

## Tech Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Runtime | Python | 3.10+ | Scientific computing with strong typing support |
| AI Framework | OpenAI API | 1.x | LLM inference (GPT-4o-mini, GPT-5-mini) |
| Validation | Pydantic + instructor | 2.x / 1.x | Structured JSON responses from LLMs |
| Data Processing | pandas + numpy | 2.x / 1.x | DataFrame operations and numerical computing |
| Statistics | scipy + statsmodels + scikit-learn | 1.x / 0.14+ / 1.x | Statistical tests and machine learning |
| Bayesian | CmdStanPy + ArviZ | 1.x / 0.16+ | Bayesian robustness checks |
| NLP | fasttext + nltk + textstat + VADER | Various | Language detection and text feature extraction |
| Visualization | matplotlib + seaborn + plotly | 3.x / 0.12+ / 5.x | Charts and interactive HTML reports |
| Templating | Jinja2 | 3.x | HTML report generation |

## Quick Start

```bash
# Prerequisites
# - Python 3.10+
# - OpenAI API key
# - ~500MB disk for fasttext language model (auto-downloads)

# Installation
git clone <repo-url>
cd LLM-linguistic-profiling
pip install -r requirements.txt

# Environment setup
# Create .env file with:
OPENAI_API_KEY=sk-...

# Download CFPB data
# Get complaints.csv from https://www.consumerfinance.gov/data-research/consumer-complaints/
# Place at ./data/complaints.csv

# Run experiment (small test)
python llm_fairness_experiment.py --num-complaints 100 --causal-sample-pct 0.20

# Full run (5000 complaints, ~2-4 hours with caching)
python llm_fairness_experiment.py
```

## Project Structure

```
.
├── llm_fairness_experiment.py      # CLI entry point
├── requirements.txt                # Python dependencies
├── .env                            # API keys (gitignored)
│
├── src/
│   ├── schemas.py                  # Pydantic models, dataclasses, persona grid
│   ├── prompts.py                  # LLM prompt templates (versioned)
│   ├── experiment.py               # 7-step orchestration + executive summary
│   ├── llm_client.py               # OpenAI client with threading, retry, circuit breaker
│   ├── cache.py                    # Thread-safe disk-based response cache
│   ├── data_loader.py              # CFPB data ingestion, cleaning (10-step pipeline)
│   ├── text_features.py            # 26 text metrics extraction
│   ├── statistics.py               # Disparate impact analysis (AIR, chi-squared, logistic)
│   ├── process_discrimination.py   # Information request burden analysis
│   ├── severity_analysis.py        # Severity amplification (7 analysis sections)
│   ├── repeated_measures.py        # CLMM, GEE, conditional logit models
│   ├── bayes_robustness.py         # Stan-based Bayesian robustness checks
│   ├── reporter.py                 # HTML report (8 tabs, 20+ charts)
│   ├── worked_examples.py          # Side-by-side complaint comparison generator
│   ├── table_generator.py          # LaTeX/CSV table generation for papers
│   ├── figure_generator.py         # Publication-quality figure generation
│   └── paper_bundle.py             # Academic paper artifact bundling
│
├── one-shot-examples/              # GPT executive summary examples (7 sections)
├── specs/                          # Experiment specifications and methodology
├── data/                           # CFPB CSV + fasttext model (gitignored)
├── cache/                          # LLM response cache (gitignored)
├── output/                         # Reports and artifacts (gitignored)
└── models/stan/                    # Compiled Stan models (gitignored)
```

## Architecture Overview

The experiment follows a 7-step methodology to detect and analyze linguistic profiling:

1. **Persona Estimation** — LLM classifies complaints across 6 demographic dimensions
2. **Remedy Decision (Original)** — Assign remedy tier to original complaint text
3. **Correlation Analysis** — Test persona-outcome associations (chi-squared, logistic regression)
4. **Persona-Style Rewriting** — Rewrite complaints into 36 persona styles (matched-guise)
5. **Remedy Decision (Rewritten)** — Test causation via within-complaint comparisons
6. **Language Neutralization** — Remove demographic markers from all complaints
7. **Remedy Decision (Neutralized)** — Measure bias reduction from neutralization

### Key Modules

| Module | Location | Purpose |
|--------|----------|---------|
| Experiment | `src/experiment.py` | Main orchestrator; runs all 7 steps sequentially |
| LLM Client | `src/llm_client.py` | Thread-safe OpenAI wrapper with instructor validation |
| Cache | `src/cache.py` | Lock-free file-based caching keyed by content hash |
| Data Loader | `src/data_loader.py` | 10-step cleaning pipeline with parquet checkpoints |
| Statistics | `src/statistics.py` | Disparate impact metrics (AIR, chi-squared, FDR correction) |
| Severity | `src/severity_analysis.py` | Tests whether bias amplifies with complaint severity |
| Reporter | `src/reporter.py` | Generates self-contained HTML report with embedded charts |

### Persona Dimensions

| Dimension | Values | Used In |
|-----------|--------|---------|
| Age | young (25-35), old (65+) | Estimation + Causal |
| Geographic | urban, suburban, rural | Estimation + Causal |
| Education | high_school, college | Estimation + Causal |
| Socioeconomic | low, middle, high | Estimation + Causal |
| Gender | male, female | Estimation only |
| ESL | native, esl | Estimation only |

The 4 causal dimensions produce 36 persona combinations (2 × 3 × 2 × 3).

## Development Guidelines

### Code Style
- File names: `snake_case.py` (e.g., `llm_client.py`, `data_loader.py`)
- Class names: `PascalCase` (e.g., `Experiment`, `ResponseCache`)
- Functions/variables: `snake_case` (e.g., `run_severity_analysis`, `cache_dir`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `PERSONA_GRID`, `PROMPT_VERSIONS`)
- Private methods: `_prefixed` (e.g., `_call_once`, `_ensure_dirs`)

### Import Order
1. `from __future__ import annotations` (always first)
2. Standard library (logging, pathlib, typing, etc.)
3. Third-party packages (pandas, numpy, openai, etc.)
4. Local imports (`from src.schemas import ...`)

### Type Hints
- Use type hints on all public functions
- Use `Optional[X]` for nullable parameters
- Use `List`, `Dict`, `Tuple` from typing (Python 3.10 compatibility)

### Dataclasses vs Pydantic
- **Pydantic models** (`BaseModel`): LLM response schemas with validation/coercion
- **Dataclasses** (`@dataclass`): Internal data containers without validation overhead

## Available Commands

| Command | Description |
|---------|-------------|
| `python llm_fairness_experiment.py` | Run full experiment with defaults |
| `python llm_fairness_experiment.py --num-complaints 100` | Small test run |
| `python llm_fairness_experiment.py --clear-cache` | Clear cache before run |
| `python llm_fairness_experiment.py --skip-steps 4,5` | Skip specific steps |
| `python llm_fairness_experiment.py --models gpt-4o-mini` | Single model test |
| `python llm_fairness_experiment.py --verbose` | Enable debug logging |

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-complaints` | 5000 | Number of base complaints to process |
| `--causal-sample-pct` | 0.10 | Fraction for matched-guise test (Steps 4-5) |
| `--materiality-threshold` | 0.80 | Threshold for 80% rule (adverse impact) |
| `--models` | gpt-4o-mini gpt-5-mini | Models to test |
| `--threads` | 100 | Concurrent LLM request threads |
| `--output-dir` | ./output | Report and artifact directory |
| `--data-path` | ./data/complaints.csv | Path to CFPB CSV |
| `--cache-dir` | ./cache | LLM response cache directory |
| `--seed` | 42 | Random seed for reproducibility |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM calls |

## Caching System

LLM responses are cached to disk using Blake2b hashes of `(model, task, narrative, prompt_version)`:

- **Cache directory:** `./cache/` with subdirectories per task type
- **Invalidation:** Automatic when prompt version changes (see `PROMPT_VERSIONS` in `prompts.py`)
- **Thread safety:** Lock-free design using atomic file operations
- **Resume:** Interrupted runs resume automatically from cache

Data pipeline steps are cached as parquet files in `data/.pipeline/`, invalidated on source CSV modification.

## Output Files

| File Pattern | Description |
|--------------|-------------|
| `fairness_report_{timestamp}.html` | Interactive 8-tab HTML report |
| `executive_summary.txt` | GPT-generated business summary (7 sections) |
| `worked-examples-{timestamp}.html` | Side-by-side complaint comparisons |
| `results_*.csv` | Raw results (personas, decisions, text features) |
| `analysis_results_*.json` | Complete statistical analysis output |
| `experiment_*.log` | Full experiment log with timestamps |

## Statistical Methods

### Primary Tests
- **Adverse Impact Ratio (AIR):** Rate comparison with 80% rule threshold
- **Chi-squared:** Persona-outcome association testing
- **Logistic regression:** Tier ~ persona dimensions (with regularization)
- **McNemar's test:** Paired pre/post comparisons
- **Friedman test:** Within-complaint repeated measures

### Multiple Comparison Correction
Benjamini-Hochberg FDR correction applied within test families:
- Persona vs. baseline (36 tests)
- Pairwise persona comparisons (630 tests)
- Per-dimension tests (4 tests)

### Severity Analysis
- Two metrics: monetary (dollar amounts) and issue (product × issue category)
- Two-way ANOVA for persona × severity interaction
- Bootstrap confidence intervals for amplification ratios

## Additional Resources

- Specifications: See `specs/` directory for detailed methodology
- CFPB Data: https://www.consumerfinance.gov/data-research/consumer-complaints/
- One-shot examples: See `one-shot-examples/` for GPT summary format
