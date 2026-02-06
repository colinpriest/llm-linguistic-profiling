---
name: textstat
description: |
  Extracts text readability metrics and linguistic complexity measures.
  Use when: computing Flesch Reading Ease, Flesch-Kincaid Grade Level, 
  Gunning Fog Index, SMOG Index, or Automated Readability Index (ARI) 
  for text analysis, comparing text complexity across groups, or building 
  feature pipelines for NLP tasks.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Textstat Skill

Textstat provides readability formulas that estimate text difficulty. This codebase uses 5 metrics in `src/text_features.py` as part of a 26-feature extraction pipeline for analyzing financial complaints.

## Quick Start

### Extract Readability Metrics

```python
import textstat

def _complexity_features(text: str) -> Dict[str, float]:
    """Extract all readability metrics for a document."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress edge-case warnings
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog_index": textstat.smog_index(text),
            "ari": textstat.automated_readability_index(text),
        }
```

### Batch Processing with Caching

```python
from src.text_features import extract_features_batch, features_to_dataframe

# Extract features for all complaints
features = extract_features_batch(
    texts=df["narrative"].tolist(),
    case_ids=df["case_id"].tolist(),
    variant="original",
)

# Convert to DataFrame for analysis
features_df = features_to_dataframe(features)
```

## Key Metrics

| Metric | Function | Scale | Interpretation |
|--------|----------|-------|----------------|
| Flesch Reading Ease | `flesch_reading_ease()` | 0-100 | Higher = easier (60-70 = standard) |
| Flesch-Kincaid Grade | `flesch_kincaid_grade()` | 0-18 | U.S. grade level |
| Gunning Fog | `gunning_fog()` | 0-20+ | Years of education needed |
| SMOG Index | `smog_index()` | 0-20+ | Years of education (syllable-based) |
| ARI | `automated_readability_index()` | -10-60 | Grade level (character-based) |

## Common Patterns

### Compare Readability Across Groups

```python
# Group by persona and compare average readability
readability_by_persona = features_df.groupby("variant")["flesch_reading_ease"].agg(
    ["mean", "std", "count"]
)
```

### Filter by Complexity Threshold

```python
# Find highly complex complaints (grade level > 12)
complex_complaints = features_df[features_df["flesch_kincaid_grade"] > 12]
```

## See Also

- [patterns](references/patterns.md) — Integration patterns and anti-patterns
- [workflows](references/workflows.md) — Feature pipeline and analysis workflows

## Related Skills

- See the **pandas** skill for DataFrame operations on extracted features
- See the **numpy** skill for statistical aggregations
- See the **nltk** skill for sentence tokenization used in length features
- See the **scipy** skill for statistical tests on readability distributions