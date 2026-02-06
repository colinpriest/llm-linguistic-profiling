# Textstat Workflows

## Contents
- Feature Extraction Pipeline
- Batch Processing Workflow
- Readability Analysis Workflow
- Caching Strategy
- Integration with Statistical Analysis

## Feature Extraction Pipeline

The complete workflow from raw text to DataFrame with readability metrics.

```python
from src.text_features import (
    text_feature_cache,
    fit_topic_model,
    extract_features_batch,
    features_to_dataframe,
)

# 1. Configure cache (once per experiment)
text_feature_cache.configure("./cache")

# 2. Fit topic model on corpus (optional, for topic features)
fit_topic_model(df["narrative"].tolist(), n_topics=10, seed=42)

# 3. Extract features for all documents
features = extract_features_batch(
    texts=df["narrative"].tolist(),
    case_ids=df["case_id"].tolist(),
    variant="original",
)

# 4. Convert to DataFrame
features_df = features_to_dataframe(features)

# 5. Check cache performance
print(text_feature_cache.stats_summary())
# Output: "Text-feature cache: 4500 hits, 500 misses (90% hit rate)"
```

**Copy this checklist:**
- [ ] Configure cache directory
- [ ] Fit topic model if using topic features
- [ ] Extract features with correct variant label
- [ ] Convert to DataFrame
- [ ] Verify cache hit rate is reasonable

## Batch Processing Workflow

For processing multiple text variants (original, rewritten, neutralized).

```python
# Process all three variants
variants = ["original", "rewritten", "neutralized"]
all_features = []

for variant in variants:
    column = f"{variant}_narrative"
    if column in df.columns:
        features = extract_features_batch(
            texts=df[column].tolist(),
            case_ids=df["case_id"].tolist(),
            variant=variant,
        )
        all_features.extend(features)

# Combine into single DataFrame
combined_df = features_to_dataframe(all_features)

# Pivot for comparison
comparison = combined_df.pivot(
    index="case_id",
    columns="variant",
    values="flesch_reading_ease"
)
```

## Readability Analysis Workflow

Compare readability across persona groups. See the **scipy** skill for statistical tests.

```python
import pandas as pd
from scipy import stats

# Merge features with persona data
analysis_df = features_df.merge(personas_df, on="case_id")

# Group statistics
readability_stats = analysis_df.groupby("socioeconomic").agg({
    "flesch_reading_ease": ["mean", "std", "count"],
    "flesch_kincaid_grade": ["mean", "std"],
}).round(2)

# Statistical test: do readability scores differ by socioeconomic status?
groups = [
    analysis_df[analysis_df["socioeconomic"] == level]["flesch_reading_ease"]
    for level in ["low", "middle", "high"]
]
stat, p_value = stats.kruskal(*groups)  # Non-parametric ANOVA
print(f"Kruskal-Wallis H={stat:.2f}, p={p_value:.4f}")
```

## Caching Strategy

The codebase caches text features to disk using Blake2b hashes.

**Cache key structure:**
```
{CACHE_VERSION}|{variant}|{text}  →  Blake2b hash  →  {hash}.json
```

**Cache invalidation:**
- Bump `_CACHE_VERSION` in `src/text_features.py` to invalidate all cached entries
- Cache is per-variant: same text with different variant labels = separate entries

```python
# Force cache clear
from src.text_features import text_feature_cache

count = text_feature_cache.clear()
print(f"Cleared {count} cached entries")
```

**Validation loop:**
1. Run feature extraction
2. Check cache stats with `stats_summary()`
3. If hit rate is low on re-runs, verify variant labels match
4. Only proceed when cache is working correctly

## Integration with Statistical Analysis

Readability metrics feed into the disparate impact analysis. See the **statsmodels** skill for regression.

```python
import statsmodels.api as sm

# Does readability predict remedy tier?
X = features_df[["flesch_reading_ease", "word_count", "vader_compound"]]
X = sm.add_constant(X)
y = features_df["tier_numeric"]

model = sm.OLS(y, X).fit()
print(model.summary())
```

### Visualizing Readability Distributions

See the **seaborn** skill for visualization patterns.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Violin plot by persona
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(
    data=analysis_df,
    x="socioeconomic",
    y="flesch_reading_ease",
    ax=ax,
)
ax.set_xlabel("Socioeconomic Status")
ax.set_ylabel("Flesch Reading Ease")
ax.set_title("Readability Distribution by Socioeconomic Group")
plt.tight_layout()
```

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Cache miss despite same text | Different variant label | Use consistent variant naming |
| Zero SMOG values | Text too short (<30 sentences) | Use Flesch-Kincaid instead |
| Extreme Flesch scores (>100 or <0) | Edge case in formula | Cap values or filter outliers |
| Slow extraction | No cache configured | Call `text_feature_cache.configure()` first |