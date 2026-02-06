# Fasttext Patterns Reference

## Contents
- Model Loading Patterns
- Batch Prediction Patterns
- Error Recovery Patterns
- Anti-Patterns

---

## Model Loading Patterns

### Lazy Singleton with Deferred Import

**Why:** Avoids import-time failures and ~500MB memory until needed.

```python
_ft_model = None

def _get_fasttext_model():
    """Lazy-load the fasttext language-identification model."""
    global _ft_model
    if _ft_model is None:
        import fasttext  # Deferred import
        model_path = Path(__file__).parent.parent / "data" / "lid.176.ftz"
        if not model_path.exists():
            _download_fasttext_model(model_path)
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model
```

### Atomic Model Download

**Why:** Prevents corrupted partial downloads from breaking subsequent runs.

```python
def _download_fasttext_model(model_path: Path) -> None:
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(f"{model_path.suffix}.tmp")
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(tmp_path, "wb") as fh:
                fh.write(response.read())
        os.replace(tmp_path, model_path)  # Atomic rename
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()  # Clean up on failure
        raise
```

---

## Batch Prediction Patterns

### Vectorized Detection

**Why:** FastText's `predict()` is optimized for batch input—one call for all texts.

```python
def _detect_lang_batch(texts: List[str]) -> List[str]:
    if not texts:
        return []
    model = _get_fasttext_model()
    cleaned = [t.replace("\n", " ") for t in texts]
    predictions = model.predict(cleaned)
    return [lbl[0].replace("__label__", "") for lbl in predictions[0]]
```

### Extracting Language Codes

**Why:** FastText returns `__label__en` format; strip the prefix.

```python
# DO - Extract clean ISO 639-1 codes
labels = [lbl[0].replace("__label__", "") for lbl in predictions[0]]
# Returns: ["en", "es", "fr", ...]

# DON'T - Use raw labels
labels = predictions[0]  # ["__label__en", "__label__es", ...]
```

---

## Error Recovery Patterns

### Model Load Recovery

**Why:** Corrupted model files cause cryptic C-level errors.

```python
try:
    _ft_model = fasttext.load_model(str(model_path))
except Exception as exc:
    logger.warning("Fasttext model load failed (%s); re-downloading...", exc)
    if model_path.exists():
        model_path.unlink()  # Delete corrupted file
    _download_fasttext_model(model_path)
    _ft_model = fasttext.load_model(str(model_path))  # Retry once
```

### Fallback to langid

**Why:** Pure Python fallback when fasttext fails entirely.

```python
def _detect_lang_fallback(texts: List[str]) -> List[str]:
    if not texts:
        return []
    import langid
    return [langid.classify(t)[0] for t in texts]
```

---

## Anti-Patterns

### WARNING: Processing One Text at a Time

**The Problem:**

```python
# BAD - Loop defeats batch optimization
languages = []
for text in texts:
    lang = model.predict([text])[0][0].replace("__label__", "")
    languages.append(lang)
```

**Why This Breaks:**
1. 10-100x slower than batch prediction
2. Python loop overhead dominates
3. Defeats FastText's internal vectorization

**The Fix:**

```python
# GOOD - Single batch call
cleaned = [t.replace("\n", " ") for t in texts]
predictions = model.predict(cleaned)
languages = [lbl[0].replace("__label__", "") for lbl in predictions[0]]
```

### WARNING: Importing at Module Level

**The Problem:**

```python
# BAD - Fails import if fasttext not installed
import fasttext
_ft_model = fasttext.load_model("data/lid.176.ftz")
```

**Why This Breaks:**
1. Import fails before any code runs
2. Cannot gracefully fall back to langid
3. ~500MB memory allocated even if never used

**The Fix:**

```python
# GOOD - Deferred import inside function
def _get_fasttext_model():
    import fasttext  # Only imported when needed
    # ... rest of lazy loading
```

### WARNING: Not Normalizing Newlines

**The Problem:**

```python
# BAD - Raw text with newlines
predictions = model.predict(texts)
```

**Why This Breaks:**
1. FastText tokenizes on newlines
2. Multi-line text returns multiple predictions per input
3. Array length mismatch causes index errors

**The Fix:**

```python
# GOOD - Normalize newlines to spaces
cleaned = [t.replace("\n", " ") for t in texts]
predictions = model.predict(cleaned)
```

### WARNING: Ignoring Empty Batches

**The Problem:**

```python
# BAD - Crashes on empty input
predictions = model.predict(texts)  # texts = []
```

**Why This Breaks:**
1. FastText returns malformed tuple on empty input
2. Index access fails with cryptic error

**The Fix:**

```python
# GOOD - Guard clause for empty input
if not texts:
    return []
predictions = model.predict(texts)
```