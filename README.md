# LLM Fairness in Financial Complaint Resolution

Tests whether large language models exhibit discriminatory behavior when processing financial complaints. Uses a 7-step methodology to detect linguistic profiling, establish causation through a matched-guise experiment, and test mitigation via language neutralization.

## Research Design

The experiment evaluates whether LLMs assign different remedy outcomes to complaints based on inferred demographic characteristics of the writer, using 36 persona combinations across four dimensions:

| Dimension | Values |
|-----------|--------|
| Age | young (25-35), old (65+) |
| Geographic | urban, suburban, rural |
| Education | high school, college |
| Socioeconomic | low, middle, high |

### 7-Step Methodology

1. **Persona Estimation** - Classify each complaint's likely demographic persona from linguistic cues
2. **Remedy Decision (Original)** - Assign remedy tier (no action / non-monetary / monetary) to original text
3. **Correlation Analysis** - Test whether estimated persona correlates with remedy outcomes
4. **Persona-Style Rewriting** - Rewrite a sample into all 36 persona styles, preserving factual content
5. **Remedy Decision (Rewritten)** - Assign remedy tiers to rewritten complaints to test causation
6. **Language Neutralization** - Rewrite all complaints into demographically neutral language
7. **Remedy Decision (Neutralized)** - Measure whether neutralization reduces discrimination

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### Data

Download the CFPB Consumer Complaint Database CSV and place it at `./data/complaints.csv`. The dataset is publicly available from [consumerfinance.gov](https://www.consumerfinance.gov/data-research/consumer-complaints/).

Required columns: `Consumer complaint narrative`, `Product`, `Issue`, `Date received`, `Complaint ID`.

## Usage

```bash
# Full run with defaults (5000 complaints, 2 models, 100 threads)
python llm_fairness_experiment.py

# Small test run
python llm_fairness_experiment.py --num-complaints 100 --causal-sample-pct 0.20

# Clear cache and run with stricter threshold
python llm_fairness_experiment.py --clear-cache --materiality-threshold 0.75

# Run only correlation analysis (Steps 1-3)
python llm_fairness_experiment.py --skip-steps 4,5,6,7

# Single model, verbose logging
python llm_fairness_experiment.py --models gpt-4o-mini --verbose
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-complaints` | 5000 | Number of base complaints to process |
| `--causal-sample-pct` | 0.10 | Fraction for matched-guise causal test (Steps 4-5) |
| `--materiality-threshold` | 0.80 | Threshold for adverse impact ratio (80% rule) |
| `--clear-cache` | false | Clear all cached LLM responses before run |
| `--models` | gpt-4o-mini gpt-5.2-mini | Models to test |
| `--threads` | 100 | Concurrent LLM request threads |
| `--output-dir` | ./output | Directory for report and artifacts |
| `--data-path` | ./data/complaints.csv | Path to CFPB CSV |
| `--cache-dir` | ./cache | Directory for response cache |
| `--seed` | 42 | Random seed for reproducibility |
| `--skip-steps` | (none) | Comma-separated steps to skip, e.g. "4,5" |
| `--verbose` | false | Enable detailed logging |

## Output

The experiment produces:

- **HTML Report** (`fairness_report_{timestamp}.html`) - Self-contained report with 6 tabs:
  - Executive Summary
  - Disparate Impact (outcome analysis)
  - Process Discrimination (procedural burden analysis)
  - Text Features (linguistic feature analysis)
  - Model Comparison
  - Detailed Results
- **CSV files** - Raw results for outcome decisions, persona estimates, text features
- **JSON** - Repair statistics tracking JSON validation/repair success rates
- **Log file** - Full experiment log

## Project Structure

```
.
├── llm_fairness_experiment.py     # Main entry point
├── requirements.txt
├── .env                           # API keys (not in version control)
│
├── src/
│   ├── schemas.py                 # Pydantic models and dataclasses
│   ├── prompts.py                 # LLM prompt templates
│   ├── json_utils.py              # JSON extraction, repair, validation
│   ├── cache.py                   # Disk-based response cache
│   ├── data_loader.py             # CFPB data ingestion and cleaning
│   ├── text_features.py           # Text feature extraction (26 metrics)
│   ├── llm_client.py              # OpenAI API client with threading
│   ├── statistics.py              # Disparate impact statistical analysis
│   ├── process_discrimination.py  # Process discrimination analysis
│   ├── experiment.py              # 7-step orchestration
│   └── reporter.py                # HTML report generation
│
├── data/
│   └── complaints.csv             # CFPB data (not in version control)
│
├── cache/                         # LLM response cache (auto-created)
├── output/                        # Reports and artifacts (auto-created)
├── docs/                          # Project documentation
└── specs/                         # Experiment specification
```

## Data Pipeline

The cleaning pipeline processes the raw CFPB data through 8 sequential steps:

1. Whitespace normalization
2. Length filter (300-2000 characters)
3. Language detection (English only via langid)
4. ASCII ratio filter (>= 85%)
5. Null handling (fill missing product/issue with "Unknown")
6. Deduplication (Blake2b hash)
7. ID generation (UUID4, year extraction)
8. Stratified sampling by product/issue category

## Statistical Methods

### Disparate Impact Analysis
- Chi-squared tests for persona-outcome association
- Logistic regression (tier ~ age + geographic + education + socioeconomic)
- Adverse Impact Ratios with 80% rule threshold
- McNemar's test for paired pre/post comparisons
- Friedman test for within-complaint repeated measures
- Bootstrap confidence intervals

### Process Discrimination Analysis
- Information request rate disparities (confidence = "need_more_info")
- Cochran's Q test for matched conditions
- Process AIR (inverted: higher burden = worse treatment, threshold > 1.25)
- Information request content analysis

### Multiple Comparison Correction
Benjamini-Hochberg FDR correction applied within each test family:
- Persona vs. baseline comparisons (36 tests)
- Pairwise persona comparisons (630 tests)
- Per-dimension tests (4 tests)
- Per-model comparisons
- Process discrimination tests

### Text Feature Analysis
26 metrics across 5 categories:
- **Length**: word count, character count, sentence count, average lengths
- **Complexity**: Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG, ARI
- **Vocabulary**: type-token ratio, hapax ratio, MTLD, HD-D
- **Sentiment**: TextBlob polarity/subjectivity, VADER compound/pos/neg/neu
- **Topics**: LDA (10 topics, unsupervised)

## Caching

LLM responses are cached to disk using Blake2b hashes of (model, task, narrative, prompt_version). This allows:
- Resuming interrupted runs without re-calling the API
- Iterating on analysis code without re-running LLM calls
- Automatic invalidation when prompt templates change (via version string)

Clear the cache with `--clear-cache` or delete the `./cache/` directory.

## Remedy Tier Definitions

| Tier | Label | Historical Rate | Description |
|------|-------|----------------|-------------|
| 0 | No Action Required | 65.6% | No bank error identified |
| 1 | Non-Monetary Action | 31.8% | Process correction, no financial compensation |
| 2 | Monetary Action | 2.6% | Financial compensation for bank error |
