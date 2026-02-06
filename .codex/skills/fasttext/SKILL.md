---
name: fasttext
description: |
  Detects language identification in complaint text narratives.
  Use when: filtering non-English complaints, implementing language detection,
  loading fasttext models, handling language detection fallbacks, or
  processing batch text for language classification.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Fasttext Skill

Language identification for filtering English-only complaints in the data pipeline. Uses Facebook's `lid.176.ftz` model with lazy loading, automatic download, and `langid` fallback.

## Quick Start

### Batch Language Detection

```python
def _detect_lang_batch(texts: List[str]) -> List[str]:
    """Detect language for a batch of texts using fasttext."""
    if not texts:
        return []
    model = _get_fasttext_model()
    cleaned = [t.replace("\n", " ") for t in texts]
    predictions = model.predict(cleaned)
    return [lbl[0].replace("__label__", "") for lbl in predictions[0]]
```

### Lazy Model Loading

```python
_ft_model = None

def _get_fasttext_model():
    """Lazy-load the fasttext language-identification model."""
    global _ft_model
    if _ft_model is None:
        import fasttext
        model_path = Path(__file__).parent.parent / "data" / "lid.176.ftz"
        if not model_path.exists():
            _download_fasttext_model(model_path)
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Model path | `data/lid.176.ftz` at project root | `Path("data") / "lid.176.ftz"` |
| Prediction API | Returns `(labels, scores)` tuple | `model.predict(texts)` |
| Label format | ISO 639-1 with `__label__` prefix | `"__label__en"` → `"en"` |
| Batch input | List of strings, newlines normalized | `[t.replace("\n", " ") for t in texts]` |
| Fallback | `langid.classify()` on any exception | Pure Python, slower |

## Common Patterns

### Filter English Only

**When:** Cleaning complaint data before LLM processing

```python
df["_lang"] = _detect_lang_batch(df["narrative"].tolist())
df = df[df["_lang"] == "en"].drop(columns=["_lang"])
```

### Handle Detection Failures

**When:** fasttext unavailable or model corrupted

```python
try:
    df["_lang"] = _detect_lang_batch(df["narrative"].tolist())
except Exception as e:
    logger.warning("fasttext unavailable (%s), falling back to langid...", e)
    df["_lang"] = _detect_lang_fallback(df["narrative"].tolist())
```

## See Also

- [patterns](references/patterns.md) - Model loading, batch processing, error recovery
- [workflows](references/workflows.md) - Pipeline integration, caching, fallbacks

## Related Skills

- See the **pandas** skill for DataFrame operations in the cleaning pipeline
- See the **python** skill for type hints and import patterns
- See the **nltk** skill for complementary text tokenization