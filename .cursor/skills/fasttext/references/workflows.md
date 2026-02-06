# Fasttext Workflows Reference

## Contents
- Data Pipeline Integration
- Model Setup Workflow
- Debugging Language Detection
- Caching Strategy

---

## Data Pipeline Integration

### Language Filtering in Cleaning Pipeline

FastText runs as step 3 in the 10-step data cleaning pipeline (`src/data_loader.py:355-368`).

```python
def _step_3_language(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Language detection – keep English only."""
    n = len(df)
    logger.info("[Data Clean 3/10] Language detection (%d texts)...", n)
    try:
        df = df.copy()
        df["_lang"] = _detect_lang_batch(df["narrative"].tolist())
    except Exception as e:
        logger.warning("  fasttext unavailable (%s), falling back to langid...", e)
        df = df.copy()
        df["_lang"] = _detect_lang_fallback(df["narrative"].tolist())
    df = df[df["_lang"] == "en"].drop(columns=["_lang"]).copy()
    logger.info("  After English filter: %d", len(df))
    return df
```

**Workflow checklist:**

Copy this checklist and track progress:
- [ ] Input: DataFrame with `narrative` column
- [ ] Extract narratives to list: `df["narrative"].tolist()`
- [ ] Detect languages: `_detect_lang_batch()` or fallback
- [ ] Filter English: `df[df["_lang"] == "en"]`
- [ ] Drop temp column: `.drop(columns=["_lang"])`
- [ ] Return cleaned DataFrame

---

## Model Setup Workflow

### First-Time Model Download

On first run, the model auto-downloads (~500MB).

```
INFO:src.data_loader:Downloading fasttext language-id model (once)...
INFO:src.data_loader:Model saved to D:\dev\LLM-linguistic-profiling\data\lid.176.ftz
```

**Validation feedback loop:**

1. Run experiment: `python llm_fairness_experiment.py --num-complaints 100`
2. Check logs for model download
3. If download fails, manually download from:
   `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz`
4. Place at `data/lid.176.ftz`
5. Re-run experiment

### Model Recovery

If model is corrupted:

```python
# Automatic recovery (built-in)
try:
    _ft_model = fasttext.load_model(str(model_path))
except Exception as exc:
    logger.warning("Fasttext model load failed (%s); re-downloading...", exc)
    if model_path.exists():
        model_path.unlink()
    _download_fasttext_model(model_path)
    _ft_model = fasttext.load_model(str(model_path))
```

**Manual recovery:**

```bash
# Delete corrupted model
rm data/lid.176.ftz

# Re-run (will auto-download)
python llm_fairness_experiment.py --num-complaints 100
```

---

## Debugging Language Detection

### Inspect Detection Results

```python
import pandas as pd
from src.data_loader import _detect_lang_batch

# Sample texts
texts = [
    "I have a problem with my credit card.",
    "Tengo un problema con mi tarjeta de crédito.",
    "J'ai un problème avec ma carte de crédit."
]

# Detect languages
languages = _detect_lang_batch(texts)
print(list(zip(texts[:50], languages)))
# [('I have a problem...', 'en'), ('Tengo un problema...', 'es'), ("J'ai un problème...", 'fr')]
```

### Check Language Distribution

```python
df = pd.read_csv("data/complaints.csv", low_memory=False)
df["_lang"] = _detect_lang_batch(df["Consumer complaint narrative"].dropna().tolist())
print(df["_lang"].value_counts().head(10))
# en    45000
# es      500
# fr       50
# ...
```

### Compare with Fallback

```python
from src.data_loader import _detect_lang_batch, _detect_lang_fallback

texts = df["narrative"].head(100).tolist()

ft_langs = _detect_lang_batch(texts)
langid_langs = _detect_lang_fallback(texts)

# Compare disagreements
for i, (ft, lid) in enumerate(zip(ft_langs, langid_langs)):
    if ft != lid:
        print(f"Row {i}: fasttext={ft}, langid={lid}")
        print(f"  Text: {texts[i][:100]}...")
```

---

## Caching Strategy

### Pipeline Checkpoint Caching

Language detection is cached as part of the cleaning pipeline. See the **pandas** skill for parquet checkpointing.

**Cache location:** `data/.pipeline/step_3_language.parquet`

**Cache invalidation:** Automatic when source CSV modified (checks mtime).

```python
# Cache hit (fast)
[Data Clean 3/10] Language detection... (loaded from cache)

# Cache miss (runs fasttext)
[Data Clean 3/10] Language detection (50000 texts)...
  After English filter: 45000
```

### Forcing Re-Detection

```bash
# Delete pipeline cache
rm -rf data/.pipeline/

# Or delete specific step
rm data/.pipeline/step_3_language.parquet

# Re-run
python llm_fairness_experiment.py
```

---

## Dependency Installation

### Using Pre-Compiled Wheels

**Why:** `fasttext-wheel` provides pre-compiled binaries; `fasttext` requires C++ compiler.

```bash
# GOOD - Pre-compiled
pip install fasttext-wheel>=0.9.2

# BAD - Requires compilation
pip install fasttext  # Fails without C++ toolchain
```

### Verifying Installation

```python
import fasttext
print(fasttext.__version__)  # Should print version

# Test model loading
model = fasttext.load_model("data/lid.176.ftz")
print(model.predict(["Hello world"]))
# (('__label__en',), array([0.99...]))
```

---

## Performance Considerations

| Aspect | fasttext | langid (fallback) |
|--------|----------|-------------------|
| Speed | ~100k texts/sec | ~1k texts/sec |
| Memory | ~500MB model | ~50MB |
| Accuracy | Higher | Lower |
| Dependencies | C library | Pure Python |

**When fallback triggers:**
- fasttext not installed
- Model file missing/corrupted
- C library loading fails

**Recommendation:** Always prefer fasttext; fallback exists for CI/testing environments without C dependencies.