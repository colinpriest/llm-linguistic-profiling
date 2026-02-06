# NLTK Patterns Reference

## Contents
- Lazy Loading Pattern
- Sentence Tokenization
- Resource Management
- Anti-Patterns

---

## Lazy Loading Pattern

### Module-Level Singleton

NLTK downloads are slow. Use a flag to ensure resources load once per process.

```python
# src/text_features.py:115-132
_nltk_ready = False

def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _nltk_ready = True
```

### Why This Pattern

1. **Startup time** - NLTK imports are heavy; defer until needed
2. **Network calls** - Downloads only happen if resource missing
3. **Idempotent** - Safe to call multiple times
4. **Quiet mode** - No console spam during batch processing

---

## Sentence Tokenization

### Basic Usage

```python
# src/text_features.py:147-162
def _length_features(text: str) -> Dict[str, Any]:
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    words = text.split()  # DO: Simple split for words
    sentences = sent_tokenize(text)  # NLTK for sentences
    sentence_count = max(len(sentences), 1)  # Guard division by zero

    return {
        "word_count": len(words),
        "sentence_count": sentence_count,
        "avg_sentence_length": len(words) / sentence_count,
    }
```

### Why `sent_tokenize` Only

| Method | Use Case | Performance |
|--------|----------|-------------|
| `sent_tokenize` | Sentence boundaries | ~1ms per doc |
| `word_tokenize` | Linguistic analysis | ~5ms per doc |
| `str.split()` | Simple word counts | ~0.01ms per doc |

This project uses `str.split()` for words because:
- Financial complaints don't need linguistic tokenization
- 500x faster at scale (5000+ documents)
- Sufficient for word counts and readability metrics

---

## Resource Management

### Required Resources

```python
# These are the only NLTK resources needed for this project
NLTK_RESOURCES = ["punkt", "punkt_tab"]
```

### Check Before Download

```python
# DO: Check if resource exists first
try:
    nltk.data.find(f"tokenizers/{resource}")
except LookupError:
    nltk.download(resource, quiet=True)

# DON'T: Download unconditionally
nltk.download("punkt")  # Slow, prints to console
```

### Download Location

NLTK stores data in `~/nltk_data/` by default. This project doesn't customize the path.

---

## Anti-Patterns

### WARNING: Top-Level NLTK Import

**The Problem:**

```python
# BAD - Slows down module import
import nltk
from nltk.tokenize import sent_tokenize

def process(text):
    return sent_tokenize(text)
```

**Why This Breaks:**
1. NLTK import takes 200-500ms
2. All scripts that import this module pay the cost
3. CLI startup becomes sluggish

**The Fix:**

```python
# GOOD - Lazy import inside function
def process(text):
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)
```

---

### WARNING: Downloading Without Checking

**The Problem:**

```python
# BAD - Downloads every time, prints to console
nltk.download("punkt")
sentences = sent_tokenize(text)
```

**Why This Breaks:**
1. Network call on every invocation
2. Console output disrupts progress bars
3. Slow in batch processing

**The Fix:**

```python
# GOOD - Check first, download only if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
```

---

### WARNING: Using word_tokenize for Counts

**The Problem:**

```python
# BAD - Overkill for simple word counting
from nltk.tokenize import word_tokenize
words = word_tokenize(text)
word_count = len(words)
```

**Why This Breaks:**
1. `word_tokenize` handles punctuation, contractions, etc.
2. 500x slower than `str.split()`
3. Unnecessary for basic word counts

**The Fix:**

```python
# GOOD - Simple split for word counts
words = text.split()
word_count = len(words)
```

**When You Might Be Tempted:**
When you need precise linguistic tokenization (handling "don't" → ["do", "n't"]). This project doesn't need that level of precision.

---

## Integration with Other Libraries

### With textstat

See the **textstat** skill for readability metrics. NLTK provides sentence counts; textstat provides grade levels.

```python
# Combined pattern from src/text_features.py
def _length_features(text: str) -> Dict[str, Any]:
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize
    return {"sentence_count": len(sent_tokenize(text))}

def _complexity_features(text: str) -> Dict[str, float]:
    import textstat
    return {"flesch_reading_ease": textstat.flesch_reading_ease(text)}
```

### With VADER

See the **vader** skill for sentiment analysis. Both are lazy-loaded singletons.

```python
# Pattern: separate lazy loaders for each heavy module
_nltk_ready = False
_vader_analyzer = None

def _ensure_nltk() -> None:
    # NLTK setup...

def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer
```