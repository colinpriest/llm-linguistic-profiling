---
name: nltk
description: |
  Performs natural language processing and text tokenization tasks.
  Use when: tokenizing text into sentences, downloading NLTK data resources,
  ensuring NLTK dependencies are loaded before NLP operations, or
  integrating sentence boundaries with other text analysis pipelines.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# NLTK Skill

Handles sentence tokenization and NLTK resource management for the linguistic profiling pipeline. This project uses NLTK minimally—only `sent_tokenize` for sentence boundary detection. Word tokenization uses simple `.split()` for performance.

## Quick Start

### Lazy Loading with Resource Download

```python
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

### Sentence Tokenization

```python
def _length_features(text: str) -> Dict[str, Any]:
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    words = text.split()  # Simple word tokenization
    sentences = sent_tokenize(text)  # NLTK sentence tokenization
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": word_count / sentence_count,
    }
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Lazy initialization | Module-level flag prevents repeated downloads | `_nltk_ready = False` |
| Resource check | Verify data exists before downloading | `nltk.data.find("tokenizers/punkt")` |
| Quiet download | Suppress console output | `nltk.download(resource, quiet=True)` |
| Sentence tokenization | Split text at sentence boundaries | `sent_tokenize(text)` |

## NLTK Resources Used

| Resource | Purpose | Location |
|----------|---------|----------|
| `punkt` | Sentence tokenizer model | `tokenizers/punkt` |
| `punkt_tab` | Updated punkt format | `tokenizers/punkt_tab` |

## Common Patterns

### Guard Against Empty Sentences

**When:** Processing may yield zero sentences (empty text, whitespace only).

```python
sentences = sent_tokenize(text)
sentence_count = max(len(sentences), 1)  # Prevent division by zero
```

### Import Inside Function

**When:** Heavy modules should only load when needed.

```python
def _length_features(text: str) -> Dict[str, Any]:
    _ensure_nltk()  # Ensure resources first
    from nltk.tokenize import sent_tokenize  # Import after ensuring
    # ...
```

## See Also

- [patterns](references/patterns.md) - Common NLTK patterns and anti-patterns
- [workflows](references/workflows.md) - Integration with other NLP tools

## Related Skills

- **textstat** skill for readability metrics (Flesch, Gunning Fog)
- **vader** skill for sentiment analysis
- **pandas** skill for DataFrame operations with extracted features
- **python** skill for lazy loading patterns and type hints