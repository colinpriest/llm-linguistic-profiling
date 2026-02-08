# LLM Fairness in Financial Complaint Resolution

Tests whether large language models exhibit discriminatory behavior when processing financial complaints. Uses a 7-step methodology to detect linguistic profiling, establish causation through a matched-guise experiment, test mitigation via language neutralization, and analyze whether bias amplifies with complaint severity. Results are synthesized into a GPT-generated business executive summary.

## Research Design

The experiment evaluates whether LLMs assign different remedy outcomes to complaints based on inferred demographic characteristics of the writer, using 36 persona combinations across four causal dimensions:

| Dimension | Values |
|-----------|--------|
| Age | young (25-35), old (65+) |
| Geographic | urban, suburban, rural |
| Education | high school, college |
| Socioeconomic | low, middle, high |

Two additional dimensions (gender and ESL status) are estimated from text but not used in causal rewrites, to test whether models infer protected characteristics from writing style.

### 7-Step Methodology

1. **Persona Estimation** — Classify each complaint's likely demographic persona from linguistic cues (6 dimensions: age, geographic, education, socioeconomic, gender, ESL)
2. **Remedy Decision (Original)** — Assign remedy tier (no action / non-monetary / monetary) to original text
3. **Correlation Analysis** — Test whether estimated persona correlates with remedy outcomes
4. **Persona-Style Rewriting** — Rewrite a sample into all 36 persona styles, preserving factual content
5. **Remedy Decision (Rewritten)** — Assign remedy tiers to rewritten complaints to test causation
6. **Language Neutralization** — Rewrite all complaints into demographically neutral language
7. **Remedy Decision (Neutralized)** — Measure whether neutralization reduces discrimination

### Additional Analyses

- **Severity Amplification** — Tests whether bias increases with complaint severity (monetary dollar amounts and issue category), including whether mitigation removes severity-dependent amplification
- **Dollar Amount Extraction** — LLM-based extraction of monetary amounts from complaint narratives, classified into low/medium/high severity
- **Issue Severity Scoring** — Product × issue category mapping with 50+ manual overrides for severity classification
- **GPT-5.2 Executive Summary** — Automated business-audience commentary across 7 sections, generated from statistical results using one-shot examples
- **Process Discrimination** — Analysis of whether certain personas face higher procedural burdens (information requests)

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

For reproducibility, a pinned `requirements.lock` is provided (generated from `pip freeze`) and can be used to install exact versions:

```bash
pip install -r requirements.lock
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### Fasttext language model (explicit download)

The language-detection step uses the fasttext `lid.176.ftz` model. To avoid a silent runtime download during the first pipeline run, you can pre-download it explicitly:

```bash
python -c "from src.data_loader import ensure_fasttext_model; ensure_fasttext_model()"
```

This will store the model at `./data/lid.176.ftz`.

### Data

See [Data Source](#data-source) below for full details on the dataset, category mappings, and remedy tiers.

Download the CFPB Consumer Complaint Database CSV and place it at `./data/complaints.csv`:

```
https://www.consumerfinance.gov/data-research/consumer-complaints/
```

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
| `--clear-cache` | false | Clear everything: LLM response cache, text-feature cache, data pipeline cache, and analysis caches |
| `--clear-analysis-cache` | false | Clear only severity amplification + Bayesian analysis caches (keeps LLM/text-feature caches) |
| `--clear-bayes-cache` | false | Clear only the Bayesian model cache (retains severity amplification cache and all LLM caches) |
| `--clear-exec-cache` | false | Clear only executive summary cache (regenerate summaries without re-running LLM calls) |
| `--models` | gpt-4o-mini gpt-5-mini | Models to test |
| `--threads` | 100 | Concurrent LLM request threads |
| `--output-dir` | ./output | Directory for report and artifacts |
| `--data-path` | ./data/complaints.csv | Path to CFPB CSV |
| `--cache-dir` | ./cache | Directory for response cache |
| `--seed` | 42 | Random seed for reproducibility |
| `--skip-steps` | (none) | Comma-separated steps to skip, e.g. "4,5" |
| `--verbose` | false | Enable detailed logging |

## Output

The experiment produces:

- **HTML Report** (`fairness_report_{timestamp}.html`) — Self-contained interactive report with 8 tabs:
  - Executive Summary (GPT-5.2 generated, business audience)
  - Technical Summary (experiment parameters, sample statistics)
  - Disparate Impact (outcome analysis with AIR, chi-squared, logistic regression)
  - Process Discrimination (procedural burden analysis)
  - Text Features (linguistic feature distributions and correlations)
  - Model Comparison (cross-model performance and bias patterns)
  - Severity Analysis (7 panels: overview, main effects, amplification, model comparison, worst cases, process fairness, mitigation × severity)
  - Detailed Results (raw statistics, crosstabs, test results)
- **Executive Summary** (`executive_summary.txt`) — 7-section business-audience summary as plain text
- **CSV files** — Raw results for outcome decisions, persona estimates, text features, dollar amounts
- **JSON files** — Complete analysis results and repair statistics
- **Log file** — Full experiment log with timestamps
- **Token usage log** (`token-usage/token_usage_{timestamp}.json`) — Aggregated prompt/completion tokens and estimated cost by model

## Project Structure

```
.
├── llm_fairness_experiment.py          # Main entry point (CLI)
├── requirements.txt
├── .env                                # API keys (not in version control)
│
├── src/
│   ├── schemas.py                      # Pydantic models and dataclasses
│   ├── prompts.py                      # LLM prompt templates (versioned)
│   ├── json_utils.py                   # JSON extraction, repair, validation
│   ├── cache.py                        # Thread-safe disk-based response cache
│   ├── data_loader.py                  # CFPB data ingestion, cleaning, severity scoring
│   ├── text_features.py                # Text feature extraction (26 metrics)
│   ├── llm_client.py                   # OpenAI API client with threading and circuit breaker
│   ├── statistics.py                   # Disparate impact statistical analysis
│   ├── process_discrimination.py       # Process discrimination analysis
│   ├── severity_analysis.py            # Severity amplification analysis (7 sections, multithreaded)
│   ├── repeated_measures.py            # Repeated-measures causal inference (CLMM, GEE, multithreaded)
│   ├── experiment.py                   # 7-step orchestration + executive summary generation
│   └── reporter.py                     # HTML report generation (8 tabs, 20+ charts)
│
├── one-shot-examples/                  # GPT-5.2 executive summary examples (7 files)
│   ├── one-shot-example-linguistic-differences.md
│   ├── one-shot-example-ai-decision-making.md
│   ├── one-shot-example-correlation-vs-cause.md
│   ├── one-shot-example-severity-stakes.md
│   ├── one-shot-example-language-neutralisation.md
│   ├── one-shot-example-bias-mitigation.md
│   └── one-shot-examples-conclusion.md
│
├── specs/                              # Experiment specifications
│   ├── llm_fairness_experiment_specification_v1.2.md
│   ├── llm-fairness-specification-add-gender-esl.md
│   ├── severity-amplification-specification.md
│   └── advanced-statistical-specifications.md
│
├── data/
│   └── complaints.csv                  # CFPB data (not in version control)
│
├── cache/                              # LLM response cache (auto-created)
│   ├── persona_estimate/
│   ├── remedy_decision/
│   ├── persona_rewrite/
│   ├── neutralize/
│   └── dollar_extract/
│
└── output/                             # Reports and artifacts (auto-created)
```

## Data Pipeline

The cleaning pipeline processes the raw CFPB data through 10 sequential steps, each cached as parquet for resumption:

1. Whitespace normalization
2. Length filter (300-1500 characters)
3. Language detection (English only via fasttext with langid fallback)
4. ASCII ratio filter (>= 85%)
5. Null handling (fill missing product/issue with "Unknown")
6. Deduplication (Blake2b hash)
7. Repetition filter (unique-word ratio >= 20%)
8. Redaction filter (X characters < 20% of text)
9. ID generation (UUID4, year extraction) + category assignment + exclusion of non-bank category pairs (see [Excluded Category Pairs](#excluded-category-pairs)) + issue severity scoring
10. Stratified sampling by product × issue category

**Dollar amount extraction** runs as a separate cached step using `gpt-4o-mini` to extract monetary values from narratives and classify them into severity bins (low ≤ $100, medium ≤ $1,000, high > $1,000).

## Statistical Methods

### Disparate Impact Analysis
- Chi-squared tests for persona-outcome association
- Logistic regression (tier ~ age + geographic + education + socioeconomic) with regularization for separation
- Adverse Impact Ratios with 80% rule threshold
- McNemar's test for paired pre/post comparisons
- Friedman test for within-complaint repeated measures
- Bootstrap confidence intervals (1,000 iterations)
- One-way ANOVA with partial eta-squared for dimension-level effects

### Process Discrimination Analysis
- Information request rate disparities (confidence = "need_more_info")
- Cochran's Q test for matched conditions
- Process AIR (inverted: higher burden = worse treatment, threshold > 1.25)
- Pairwise McNemar tests with FDR correction
- Information request content analysis (length ANOVA, word frequency)
- Pre/post mitigation comparison (variance reduction)

### Severity Amplification Analysis
- Two independent severity metrics: monetary (dollar amounts) and issue (product × issue category)
- Severity descriptives with coverage analysis and cross-metric agreement (Cohen's kappa, Spearman correlation)
- Main effects: one-way ANOVA, chi-squared, Spearman trend tests per severity metric and model
- Persona × severity interaction: two-way ANOVA with partial eta-squared
- Simple effects: per-level persona one-way ANOVA with pairwise t-tests and FDR correction
- Amplification ratio: bias_high / bias_low with bootstrap 95% CI (progress bars on bootstrap loops)
- Dimension-specific severity interaction: per-dimension two-way ANOVA + bootstrap (parallelised across metric × model × dimension)
- Three-way ANOVA: model × persona × severity
- Worst-case analysis: identifies the model × severity × persona combination with largest effect size
- Mitigation × severity comparison: before/after neutralization amplification ratios, per-level bias SD changes, interaction weakening, and summary verdict (eliminated / reduced / increased / unchanged)
- Process fairness × severity: information request disparity by severity level

All severity and repeated-measures analyses are parallelised using `ThreadPoolExecutor` across metric × model combinations, with `tqdm` progress bars at both the orchestrator and bootstrap-iteration level.

### Repeated-Measures Causal Inference
- Builds a long-format dataset with `complaint_id` as the repeated-measures unit across conditions and personas
- Ordinal mixed model (CLMM approximation via `OrderedModel` with clustered sandwich SEs)
- Conditional logistic regression (`ConditionalLogit` with `complaint_id` as stratum) for within-complaint comparisons
- Logistic GEE (exchangeable correlation, `complaint_id` clustering) for favourable outcome and information-request burden
- Linear mixed model (`MixedLM` with random intercept) as robustness check treating tier as continuous
- Nonparametric repeated-measures: Friedman test with post-hoc Wilcoxon signed-rank; Cochran's Q with post-hoc McNemar (all FDR-corrected)
- Per-model suites and pooled analysis run in parallel via `ThreadPoolExecutor`

### Multiple Comparison Correction
Benjamini-Hochberg FDR correction applied within each test family:
- Persona vs. baseline comparisons (36 tests)
- Pairwise persona comparisons (630 tests)
- Per-dimension tests (4 tests)
- Per-model comparisons
- Process discrimination tests
- Severity simple effects pairwise tests

### Text Feature Analysis
26 metrics across 5 categories, extracted for original, persona-rewritten, and neutralized complaints:
- **Length**: word count, character count, sentence count, average lengths
- **Complexity**: Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, SMOG, ARI
- **Vocabulary**: type-token ratio, hapax ratio, MTLD, HD-D
- **Sentiment**: TextBlob polarity/subjectivity, VADER compound/pos/neg/neu, emotional intensity
- **Topics**: LDA (10 topics, unsupervised)

## Executive Summary Generation

After all statistical analysis completes, the experiment sends structured results to GPT-5.2 to generate a 7-section business executive summary:

| Section | Topic |
|---------|-------|
| 1. Linguistic Differences | How AI perceives demographic traits from writing style |
| 2. AI Decision-Making | Whether remedy outcomes vary by perceived demographics |
| 3. Correlation vs. Cause | Matched-guise experiment isolating writing style as causal factor |
| 4. Severity Stakes | Whether bias amplifies with financial or issue severity |
| 5. Language Neutralisation | Effectiveness of removing demographic markers |
| 6. Bias Mitigation | Quantified bias reduction from neutralization |
| 7. Conclusion | Synthesis and recommendations |

Each section is generated using a one-shot example (in `one-shot-examples/`) that demonstrates the expected format and analytical depth. The full summary is displayed in the console, saved to `output/executive_summary.txt`, and rendered in the HTML report's Executive Summary tab.

## Caching

LLM responses are cached to disk using Blake2b hashes of (model, task, narrative, prompt_version). This allows:
- Resuming interrupted runs without re-calling the API
- Iterating on analysis code without re-running LLM calls
- Automatic invalidation when prompt templates change (via version string in `prompts.py`)

The cache is thread-safe (atomic writes via temp files + `os.replace`) and tracks hit/miss statistics.

Data pipeline steps are separately cached as parquet files in `data/.pipeline/`, invalidated on source CSV modification.

Clear the cache with `--clear-cache` or delete the `./cache/` directory.

## Data Source

### CFPB Consumer Complaint Database

The experiment uses the [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) published by the Consumer Financial Protection Bureau (CFPB). This is a public dataset of consumer complaints about financial products and services that were sent to companies for response. The database is updated daily and contains millions of records dating back to 2011.

Download the full CSV export from:

```
https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data
```

Place the file at `./data/complaints.csv`. Only complaints with a non-empty `Consumer complaint narrative` field are used (roughly 30% of all records include the consumer's narrative text).

### Product Categories

Raw CFPB product names are mapped to four high-level categories for stratified sampling:

| Category | Raw CFPB Product Values |
|----------|------------------------|
| **Credit Reporting** | Credit reporting, credit repair services, or other personal consumer reports; Credit reporting |
| **Debt Collection** | Debt collection |
| **Lending** | Mortgage; Student loan; Vehicle loan or lease; Payday loan, title loan, or personal loan; Payday loan; Consumer Loan |
| **Banking** | Checking or savings account; Bank account or service; Prepaid card; Credit card or prepaid card; Credit card; Money transfer, virtual currency, or money service; Money transfers |

Products not in this mapping are grouped as "Other".

### Issue Categories

Raw CFPB issue descriptions are mapped to four high-level categories:

| Category | Raw CFPB Issue Values |
|----------|-----------------------|
| **Credit Report Errors** | Incorrect information on your report; Improper use of your report; Problem with a credit reporting company's investigation into an existing problem |
| **Debt Disputes** | Attempts to collect debt not owed; Written notification about debt; Communication tactics; Took or threatened to take negative or legal action |
| **Account Management** | Managing an account; Problem with a purchase shown on your statement |
| **Payment Issues** | Struggling to pay mortgage; Trouble during payment process; Problem with a lender or other company charging your account |

Issues not in this mapping are grouped as "Other". The product × issue category combination is used as the stratification key during sampling, ensuring the final sample is proportionally representative of complaint types in the full dataset.

### Excluded Category Pairs

Two product × issue combinations are filtered out before sampling because they are dominated by non-bank actors and are less informative about bank discretion in remedy decisions:

| Product Category | Issue Category | Reason |
|-----------------|---------------|--------|
| Credit Reporting | Credit Report Errors | Bias is primarily at the bureau/system level; individual banks have constrained reporting obligations, so complaint patterns are less informative about bank discretion |
| Debt Collection | Debt Disputes | Discriminatory behavior is driven by third-party collectors rather than the originating bank, making it a weaker signal for inter-bank behavioral differences |

These pairs typically account for a large share of the raw CFPB data. Excluding them focuses the analysis on complaint types where the bank itself has meaningful discretion over the remedy outcome.

### Issue Severity Scoring

Each complaint is assigned an issue severity (low/medium/high) based on its product × issue combination:

- **Base score** (1-3): Assigned by product category (e.g., mortgage = 3, checking account = 1)
- **Issue modifier** (-1 to +2): Adjusts based on issue type (e.g., foreclosure = +2, account opening = -1)
- **Manual overrides**: 50+ specific product × issue combinations with explicit severity scores (e.g., mortgage foreclosure = 5 → high)
- **Thresholds**: score ≤ 2 = low, score ≤ 3 = medium, score ≥ 4 = high

### Remedy Tiers

The LLM assigns each complaint to one of three remedy tiers. These are modeled after the actual CFPB resolution categories (company responses), with historical base rates provided to the model as calibration anchors:

| Tier | Label | Historical Rate | Description |
|------|-------|----------------|-------------|
| 0 | No Action Required | 65.6% | No bank error identified |
| 1 | Non-Monetary Action | 31.8% | Process correction, no financial compensation |
| 2 | Monetary Action | 2.6% | Financial compensation for bank error |

**Tier 0 — No Action Required** corresponds to CFPB responses like "Closed with explanation" where the company investigated but found no error. This includes customer misunderstandings of terms, issues outside the bank's control, and complaints lacking factual basis.

**Tier 1 — Non-Monetary Action** corresponds to CFPB responses like "Closed with non-monetary relief." The bank fixes a process, corrects customer data, or improves procedures, but no financial compensation is provided.

**Tier 2 — Monetary Action** corresponds to CFPB responses like "Closed with monetary relief." The bank reimburses the customer for losses, reverses fees, adjusts interest, or pays damages caused by a confirmed bank error.

The model also assigns a confidence category to each decision:

| Confidence | Meaning |
|------------|---------|
| `confident_no_action` | High confidence that no remedial action is warranted |
| `confident_action_needed` | High confidence that some action (tier 1 or 2) is warranted |
| `need_more_info` | Insufficient information to make a confident determination |

The `need_more_info` rate is used as the primary metric for **process discrimination** analysis — it measures whether the LLM imposes a higher procedural burden (requesting additional information) on complaints written in certain linguistic styles.
