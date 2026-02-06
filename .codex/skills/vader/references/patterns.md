# VADER Patterns Reference

## Contents
- Singleton Initialization
- Feature Extraction Pattern
- DataFrame Integration
- WARNING: Reinitializing Analyzer
- WARNING: Missing Compound Interpretation
- Caching Pattern

---

## Singleton Initialization

VADER's lexicon loads ~7500 words. Initialize once per process:

```python
# GOOD - Lazy singleton (src/text_features.py:135-140)
_vader_analyzer = None

def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer
```

```python
# BAD - Creates new instance per call
def get_sentiment(text: str) -> dict:
    vader = SentimentIntensityAnalyzer()  # 7500-word lexicon loads EVERY call
    return vader.polarity_scores(text)
```

**Why singleton matters:** Loading the lexicon takes ~50ms. With 5000 complaints × 3 variants = 15000 calls, that's 12+ minutes of unnecessary overhead.

---

## Feature Extraction Pattern

Extract all VADER scores plus derived metrics:

```python
# src/text_features.py:212-227
def _sentiment_features(text: str) -> Dict[str, float]:
    from textblob import TextBlob

    blob = TextBlob(text)
    vader = _get_vader()
    vs = vader.polarity_scores(text)

    return {
        "polarity": blob.sentiment.polarity,          # TextBlob (different algorithm)
        "subjectivity": blob.sentiment.subjectivity,  # TextBlob
        "vader_compound": vs["compound"],
        "vader_pos": vs["pos"],
        "vader_neg": vs["neg"],
        "vader_neu": vs["neu"],
        "emotional_intensity": abs(vs["compound"]),   # Derived: magnitude
    }
```

**Why both TextBlob and VADER?** They use different lexicons and algorithms. TextBlob provides subjectivity (fact vs opinion); VADER excels at social media/informal text with emoji/punctuation awareness.

---

## DataFrame Integration

Convert features to analysis-ready format:

```python
# src/text_features.py:346-381
def features_to_dataframe(features: List[TextFeatures]) -> pd.DataFrame:
    records = []
    for f in features:
        d = {
            "case_id": f.case_id,
            "variant": f.variant,
            "vader_compound": f.vader_compound,
            "vader_pos": f.vader_pos,
            "vader_neg": f.vader_neg,
            "vader_neu": f.vader_neu,
            "emotional_intensity": f.emotional_intensity,
            # ... 20+ other features
        }
        records.append(d)
    return pd.DataFrame(records)
```

See the **pandas** skill for DataFrame operations.

---

## WARNING: Reinitializing Analyzer

**The Problem:**

```python
# BAD - Performance killer
def process_batch(texts: List[str]) -> List[dict]:
    results = []
    for text in texts:
        vader = SentimentIntensityAnalyzer()  # NEW instance per iteration
        results.append(vader.polarity_scores(text))
    return results
```

**Why This Breaks:**
1. 50ms lexicon load × N texts = O(N × 50ms) overhead
2. Memory pressure from repeated allocations
3. GC pauses from discarded instances

**The Fix:**

```python
# GOOD - Single instance
def process_batch(texts: List[str]) -> List[dict]:
    vader = SentimentIntensityAnalyzer()  # Once, outside loop
    return [vader.polarity_scores(text) for text in texts]
```

---

## WARNING: Missing Compound Interpretation

**The Problem:**

```python
# BAD - Raw compound without interpretation
def is_negative(text: str) -> bool:
    scores = vader.polarity_scores(text)
    return scores["compound"] < 0  # Too sensitive, catches near-neutral
```

**Why This Breaks:**
1. Compound of -0.01 is effectively neutral
2. Binary threshold ignores VADER's designed thresholds
3. Misclassifies neutral complaints as negative

**The Fix:**

```python
# GOOD - Use standard thresholds
def classify_sentiment(text: str) -> str:
    compound = vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"
```

**When You Might Be Tempted:** When building binary classifiers. The 0.05 threshold is research-validated for VADER.

---

## Caching Pattern

VADER scores are deterministic. Cache to avoid recomputation:

```python
# src/text_features.py:68-76
def put(self, text: str, variant: str, features: Dict[str, Any]) -> None:
    key = self._make_key(text, variant)
    path = self._path_for(key)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(features, f)
    tmp.replace(path)  # Atomic rename
```

Cache key includes `(cache_version, variant, text)` hashed with Blake2b. See `src/cache.py` for thread-safe implementation.

---

## No Preprocessing Required

VADER handles raw text including:
- Punctuation emphasis ("This is TERRIBLE!!!")
- Emoji (":)" increases positive)
- Capitalization (ALL CAPS = emphasis)
- Degree modifiers ("extremely", "barely")

```python
# GOOD - Pass raw text
scores = vader.polarity_scores(complaint.narrative)

# UNNECESSARY - Don't preprocess for VADER
text = complaint.narrative.lower().strip()  # Loses emphasis signals
scores = vader.polarity_scores(text)
```