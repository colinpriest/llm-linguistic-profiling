# VADER Workflows Reference

## Contents
- Full Feature Extraction Workflow
- Batch Processing Workflow
- Variant Comparison Workflow
- Sentiment Validation Checklist
- Integration with Statistical Analysis

---

## Full Feature Extraction Workflow

VADER runs within the broader text feature pipeline:

```python
# src/text_features.py:304-326
def extract_features(text: str, case_id: str, variant: str) -> TextFeatures:
    """Extract all text features for a single document (cached)."""
    # Check cache first
    cached = text_feature_cache.get(text, variant)
    if cached is not None:
        return TextFeatures(case_id=case_id, variant=variant, **cached)

    # Extract feature groups
    length = _length_features(text)
    complexity = _complexity_features(text)
    vocab = _vocabulary_features(text)
    sentiment = _sentiment_features(text)  # <-- VADER here
    topics = _topic_features(text)

    all_features = {**length, **complexity, **vocab, **sentiment, **topics}

    # Cache for future runs
    text_feature_cache.put(text, variant, all_features)

    return TextFeatures(case_id=case_id, variant=variant, **all_features)
```

**Workflow Steps:**
1. Check cache by `(text, variant)` hash
2. If miss: extract length → complexity → vocabulary → **sentiment** → topics
3. Merge all feature dicts
4. Cache combined features
5. Return `TextFeatures` dataclass

---

## Batch Processing Workflow

Process multiple complaints efficiently:

```python
# src/text_features.py:329-343
def extract_features_batch(
    texts: List[str],
    case_ids: List[str],
    variant: str,
    desc: str = "Extracting text features",
) -> List[TextFeatures]:
    """Extract features for multiple texts with progress bar."""
    return [
        extract_features(text, cid, variant)
        for text, cid in tqdm(
            zip(texts, case_ids),
            total=len(texts),
            desc=desc,
        )
    ]
```

**Called from experiment.py for each variant:**

```python
# Original complaints (src/experiment.py:799-803)
orig_features = extract_features_batch(
    [c.narrative for c in self.complaints],
    [c.case_id for c in self.complaints],
    "original",
)

# Rewritten complaints (src/experiment.py:1139-1143)
rewrite_features = extract_features_batch(
    [r.rewritten_narrative for r in self.persona_rewrites],
    [r.case_id for r in self.persona_rewrites],
    "persona_rewrite",
)

# Neutralized complaints (src/experiment.py:1370-1373)
neut_features = extract_features_batch(
    [n.neutralized_narrative for n in self.neutralized],
    [n.case_id for n in self.neutralized],
    "neutralized",
)
```

---

## Variant Comparison Workflow

Compare sentiment across complaint transformations:

```python
import pandas as pd

# Convert to DataFrames (see pandas skill)
orig_df = features_to_dataframe(orig_features)
rewrite_df = features_to_dataframe(rewrite_features)
neutral_df = features_to_dataframe(neutral_features)

# Tag variants
orig_df["variant"] = "original"
rewrite_df["variant"] = "rewritten"
neutral_df["variant"] = "neutralized"

# Combine for analysis
all_features = pd.concat([orig_df, rewrite_df, neutral_df])

# Compare compound scores by variant
sentiment_by_variant = all_features.groupby("variant")["vader_compound"].agg([
    "mean", "std", "median"
])
```

---

## Sentiment Validation Checklist

Copy this checklist when adding sentiment analysis:

```
- [ ] Singleton pattern used for SentimentIntensityAnalyzer
- [ ] Cache check before extraction
- [ ] All 4 VADER scores extracted (compound, pos, neg, neu)
- [ ] Derived metrics computed (emotional_intensity)
- [ ] Results stored in typed dataclass
- [ ] Cache updated after extraction
- [ ] DataFrame conversion handles all fields
```

---

## Integration with Statistical Analysis

VADER scores feed into statistical tests (see **scipy** and **statsmodels** skills):

```python
# src/reporter.py:394 - Features included in report
features = ["flesch_reading_ease", "vader_compound", "word_count", "type_token_ratio"]
available = [f for f in features if f in text_features_df.columns]
```

**Common analyses using VADER:**

```python
from scipy import stats

# Compare sentiment between personas (see scipy skill)
group_a = df[df["persona"] == "young_urban"]["vader_compound"]
group_b = df[df["persona"] == "old_rural"]["vader_compound"]
stat, pvalue = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

# Correlation with remedy tier (see statsmodels skill)
import statsmodels.api as sm
X = sm.add_constant(df[["vader_compound", "emotional_intensity"]])
y = df["remedy_tier"]
model = sm.OLS(y, X).fit()
```

---

## Feedback Loop: Sentiment Feature Extraction

1. Run extraction: `extract_features_batch(texts, ids, variant)`
2. Validate: Check returned `TextFeatures` have non-zero sentiment values
3. If `vader_compound == 0.0` for all records:
   - Check VADER installation: `pip show vaderSentiment`
   - Verify text not empty: `assert all(len(t) > 0 for t in texts)`
   - Check for encoding issues: `text.encode("utf-8")`
4. Only proceed when sentiment values vary across corpus

---

## Error Handling

VADER rarely fails, but handle edge cases:

```python
def safe_sentiment(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,  # Empty = neutral
            "emotional_intensity": 0.0,
        }
    
    try:
        vader = _get_vader()
        vs = vader.polarity_scores(text)
        return {
            "vader_compound": vs["compound"],
            "vader_pos": vs["pos"],
            "vader_neg": vs["neg"],
            "vader_neu": vs["neu"],
            "emotional_intensity": abs(vs["compound"]),
        }
    except Exception as e:
        logging.warning(f"VADER failed on text: {e}")
        return {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,
            "emotional_intensity": 0.0,
        }
```