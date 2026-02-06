---
name: vader
description: |
  Analyzes sentiment polarity in financial complaint narratives using VADER lexicon.
  Use when: extracting sentiment features from text, computing compound/positive/negative/neutral
  scores, measuring emotional intensity, or comparing sentiment across complaint variants.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) extracts sentiment scores from financial complaint text. In this project, VADER runs as part of a 26-feature text extraction pipeline, computing 5 sentiment metrics per complaint variant (original, rewritten, neutralized).

## Quick Start

### Basic Sentiment Extraction

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
scores = vader.polarity_scores("The bank refused to help with my dispute!")

# Returns: {'neg': 0.267, 'neu': 0.733, 'pos': 0.0, 'compound': -0.4019}
```

### Project Pattern (Lazy Singleton)

```python
_vader_analyzer = None

def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer

def _sentiment_features(text: str) -> Dict[str, float]:
    vader = _get_vader()
    vs = vader.polarity_scores(text)
    return {
        "vader_compound": vs["compound"],
        "vader_pos": vs["pos"],
        "vader_neg": vs["neg"],
        "vader_neu": vs["neu"],
        "emotional_intensity": abs(vs["compound"]),
    }
```

## Key Concepts

| Score | Range | Meaning |
|-------|-------|---------|
| `compound` | -1.0 to +1.0 | Normalized weighted composite |
| `pos` | 0.0 to 1.0 | Proportion of positive words |
| `neg` | 0.0 to 1.0 | Proportion of negative words |
| `neu` | 0.0 to 1.0 | Proportion of neutral words |
| `emotional_intensity` | 0.0 to 1.0 | `abs(compound)` - magnitude regardless of valence |

## Common Thresholds

```python
# Standard interpretation
if compound >= 0.05:
    sentiment = "positive"
elif compound <= -0.05:
    sentiment = "negative"
else:
    sentiment = "neutral"
```

## Integration Point

VADER is called in `src/text_features.py:314` within `extract_features()`:

```python
sentiment = _sentiment_features(text)  # Returns dict with 5 VADER metrics
```

Results stored in `TextFeatures` dataclass (see **pydantic** skill for schema patterns).

## See Also

- [patterns](references/patterns.md) - Code patterns and anti-patterns
- [workflows](references/workflows.md) - Integration workflows

## Related Skills

- **textstat** skill for readability metrics computed alongside VADER
- **pandas** skill for DataFrame operations on extracted features
- **numpy** skill for statistical operations on sentiment scores