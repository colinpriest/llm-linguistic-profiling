# Textstat Patterns

## Contents
- Lazy Import Pattern
- Warning Suppression Pattern
- Integration with Feature Pipeline
- Anti-Patterns
- Metric Interpretation Guidelines

## Lazy Import Pattern

Textstat is imported inside the function to defer module load until needed.

```python
# GOOD - Lazy loading defers heavy import
def _complexity_features(text: str) -> Dict[str, float]:
    import textstat  # Import only when called
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        # ...
    }
```

```python
# BAD - Top-level import slows startup for all code paths
import textstat  # Loaded even if never used

def _complexity_features(text: str) -> Dict[str, float]:
    return {"flesch_reading_ease": textstat.flesch_reading_ease(text)}
```

**Why:** The experiment may skip feature extraction entirely (e.g., cache hit), so deferring the import saves ~100ms startup time.

## Warning Suppression Pattern

Textstat generates warnings for edge cases (empty text, very short text). Suppress them in a scoped context manager.

```python
# GOOD - Scoped suppression
import warnings

def _complexity_features(text: str) -> Dict[str, float]:
    import textstat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
            "smog_index": textstat.smog_index(text),
            "ari": textstat.automated_readability_index(text),
        }
```

```python
# BAD - Global suppression hides all warnings
warnings.filterwarnings("ignore")  # Affects entire application
```

**Why:** Global suppression hides legitimate warnings from other libraries. Context manager scopes the suppression tightly.

## Integration with Feature Pipeline

The codebase uses a dataclass to store all features including textstat metrics.

```python
# src/schemas.py
@dataclass
class TextFeatures:
    case_id: str
    variant: str  # "original", "neutralized", or persona label
    
    # Complexity (textstat metrics)
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    ari: float = 0.0
    # ... other feature categories
```

```python
# src/text_features.py - Extraction pipeline
def extract_features(text: str, case_id: str, variant: str) -> TextFeatures:
    length = _length_features(text)
    complexity = _complexity_features(text)  # <-- textstat
    vocab = _vocabulary_features(text)
    sentiment = _sentiment_features(text)
    topics = _topic_features(text)
    
    all_features = {**length, **complexity, **vocab, **sentiment, **topics}
    return TextFeatures(case_id=case_id, variant=variant, **all_features)
```

## Anti-Patterns

### WARNING: Calling Textstat on Empty Text

**The Problem:**

```python
# BAD - No input validation
metrics = textstat.flesch_reading_ease("")  # Returns 206.835 (nonsensical)
```

**Why This Breaks:**
1. Empty text returns mathematically valid but meaningless values
2. Very short text (<30 words) produces unreliable SMOG scores
3. Downstream analysis treats garbage values as legitimate data

**The Fix:**

```python
# GOOD - Validate before extraction
def _complexity_features(text: str) -> Dict[str, float]:
    import textstat
    
    word_count = len(text.split())
    if word_count < 3:
        return {k: 0.0 for k in ["flesch_reading_ease", "flesch_kincaid_grade", 
                                  "gunning_fog", "smog_index", "ari"]}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            # ...
        }
```

**When You Might Be Tempted:** Processing user-generated content with variable lengths.

### WARNING: Interpreting SMOG for Short Texts

**The Problem:**

```python
# SMOG requires 30+ sentences for accuracy
short_text = "This is a short complaint."
smog = textstat.smog_index(short_text)  # Unreliable
```

**Why This Breaks:**
1. SMOG formula assumes 30 sentences minimum
2. Short texts produce inflated or zero scores
3. Comparing SMOG across different-length documents is invalid

**The Fix:**

```python
# Use Flesch-Kincaid for short texts, SMOG for long texts
def get_best_grade_metric(text: str) -> float:
    import textstat
    sentence_count = textstat.sentence_count(text)
    if sentence_count >= 30:
        return textstat.smog_index(text)
    return textstat.flesch_kincaid_grade(text)
```

## Metric Interpretation Guidelines

| Flesch Reading Ease | Grade Level | Audience |
|---------------------|-------------|----------|
| 90-100 | 5th grade | Very easy |
| 80-89 | 6th grade | Easy |
| 70-79 | 7th grade | Fairly easy |
| 60-69 | 8-9th grade | Standard |
| 50-59 | 10-12th grade | Fairly difficult |
| 30-49 | College | Difficult |
| 0-29 | College graduate | Very difficult |

Financial complaints in this corpus typically score 40-60 (difficult to fairly difficult).