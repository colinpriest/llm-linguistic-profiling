# NLTK Workflows Reference

## Contents
- Text Feature Extraction Pipeline
- Batch Processing
- Caching Strategy
- Testing and Validation

---

## Text Feature Extraction Pipeline

### Complete Workflow

The NLTK sentence tokenizer is one step in a multi-library feature extraction pipeline.

```python
# src/text_features.py:304-326
def extract_features(text: str, case_id: str, variant: str) -> TextFeatures:
    """Extract all text features for a single document."""
    # 1. Check cache first
    cached = text_feature_cache.get(text, variant)
    if cached is not None:
        return TextFeatures(case_id=case_id, variant=variant, **cached)

    # 2. Extract features from each library
    length = _length_features(text)      # NLTK: sent_tokenize
    complexity = _complexity_features(text)  # textstat
    vocab = _vocabulary_features(text)       # collections.Counter + lexicalrichness
    sentiment = _sentiment_features(text)    # VADER + TextBlob
    topics = _topic_features(text)           # sklearn LDA

    # 3. Merge and cache
    all_features = {**length, **complexity, **vocab, **sentiment, **topics}
    text_feature_cache.put(text, variant, all_features)

    return TextFeatures(case_id=case_id, variant=variant, **all_features)
```

### Pipeline Checklist

Copy this checklist when modifying the pipeline:
- [ ] Ensure `_ensure_nltk()` called before `sent_tokenize`
- [ ] Handle empty text (guard against division by zero)
- [ ] Cache features after extraction
- [ ] Return `TextFeatures` dataclass with all fields

---

## Batch Processing

### Workflow

```python
# src/text_features.py:329-343
def extract_features_batch(
    texts: List[str],
    case_ids: List[str],
    variant: str,
) -> List[TextFeatures]:
    """Extract features for a list of texts."""
    results = []
    for text, cid in tqdm(zip(texts, case_ids), total=len(texts),
                          desc=f"Text features ({variant})", unit="doc"):
        try:
            results.append(extract_features(text, cid, variant))
        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", cid, e)
            results.append(TextFeatures(case_id=cid, variant=variant))
    return results
```

### Error Handling Pattern

**Iterate-Until-Pass:**

1. Attempt feature extraction
2. If NLTK fails (missing resource), let `_ensure_nltk()` handle it
3. If still fails, log warning and return default `TextFeatures`
4. Continue processing remaining documents

```python
try:
    results.append(extract_features(text, cid, variant))
except Exception as e:
    logger.warning("Feature extraction failed for %s: %s", cid, e)
    results.append(TextFeatures(case_id=cid, variant=variant))  # Safe default
```

---

## Caching Strategy

### How Caching Works

Text features are cached by content hash, keyed on `(text, variant)`.

```python
# src/text_features.py:99-101
def _key(self, text: str, variant: str) -> str:
    content = f"{_CACHE_VERSION}|{variant}|{text}"
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()
```

### Cache Invalidation

```python
_CACHE_VERSION = "1"  # Bump to invalidate all cached entries
```

**When to Bump:**
- Changed NLTK tokenization logic
- Added new feature extractors
- Modified feature calculations

### Workflow with Cache

```
1. Call extract_features(text, case_id, variant)
2. Check cache: text_feature_cache.get(text, variant)
3. If hit: return cached TextFeatures
4. If miss:
   a. _ensure_nltk() - load NLTK if needed
   b. sent_tokenize(text) - get sentences
   c. Run other extractors
   d. Cache result
   e. Return TextFeatures
```

---

## Testing and Validation

### Manual Validation

```python
# Quick test in Python REPL
from nltk.tokenize import sent_tokenize

text = "Hello world. This is a test. Multiple sentences here!"
sentences = sent_tokenize(text)
print(f"Sentences: {len(sentences)}")  # Expected: 3
print(sentences)
```

### Edge Cases to Test

| Input | Expected Behavior |
|-------|-------------------|
| Empty string `""` | `[]` - empty list |
| Single sentence | `["The sentence."]` |
| No period | `["Text without period"]` |
| Abbreviations | `"Dr. Smith..."` stays together |
| Numbers | `"$1,234.56"` handled correctly |

### Validation Checklist

Copy this checklist for testing NLTK changes:
- [ ] Empty text returns empty list
- [ ] Single sentence returns one item
- [ ] Abbreviations (Dr., Mr., etc.) don't split
- [ ] Numbers with decimals don't split
- [ ] Unicode text handled correctly
- [ ] Very long text (10k+ words) doesn't timeout

---

## Adding New NLTK Features

### Workflow for New Tokenizers

If you need to add POS tagging or other NLTK features:

1. **Add resource to lazy loader:**

```python
def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger"]:  # New
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _nltk_ready = True
```

2. **Create new feature extractor:**

```python
def _pos_features(text: str) -> Dict[str, Any]:
    _ensure_nltk()
    from nltk import word_tokenize, pos_tag
    
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    # Extract features...
```

3. **Integrate into extract_features:**

```python
def extract_features(text: str, case_id: str, variant: str) -> TextFeatures:
    # ... existing code ...
    pos = _pos_features(text)  # New
    all_features = {**length, **complexity, **vocab, **sentiment, **topics, **pos}
```

4. **Bump cache version:**

```python
_CACHE_VERSION = "2"  # Invalidate old cache
```

---

## Integration with Experiment Pipeline

### Where NLTK Fits

```
Data Loading (data_loader.py)
    ↓
FastText language detection
    ↓
Filter to English
    ↓
Text Features (text_features.py)  ← NLTK here
    ↓
Statistical Analysis (statistics.py)
    ↓
Report Generation (reporter.py)
```

### Experiment Flow

See the **pandas** skill for DataFrame operations after feature extraction.

```python
# In experiment.py
features = extract_features_batch(texts, case_ids, variant="original")
df = features_to_dataframe(features)
# Continue with statistical analysis...
```