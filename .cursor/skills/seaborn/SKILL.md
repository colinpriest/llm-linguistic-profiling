---
name: seaborn
description: |
  Creates statistical data visualizations and heatmaps.
  Use when: generating AIR heatmaps, violin plots for text features,
  topic distribution matrices, creating diverging color palettes,
  or building publication-quality statistical visualizations for HTML reports.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Seaborn Skill

Seaborn in this project is isolated to `src/reporter.py` for generating statistical visualizations embedded in HTML reports. All plots are rendered headlessly via matplotlib's Agg backend and converted to base64 PNG for embedding.

## Quick Start

### Diverging Heatmap with Threshold

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(
    pivot_df, annot=True, fmt=".2f", cmap=cmap, center=0.80,
    vmin=0.5, vmax=1.2, ax=ax, linewidths=0.5,
    cbar_kws={"label": "AIR (< 0.80 = material)"},
)
```

### Violin Plot with Fallback

```python
try:
    sns.violinplot(data=df, x="variant", y="flesch_reading_ease",
                   ax=ax, inner="box", palette="Set2", cut=0)
except Exception:
    sns.boxplot(data=df, x="variant", y="flesch_reading_ease",
                ax=ax, palette="Set2")
```

### Color Palette Generation

```python
# Categorical palette for discrete groups
colors = sns.color_palette("Set2", n_colors=6)

# Diverging palette for threshold-centered heatmaps
cmap = sns.diverging_palette(130, 10, as_cmap=True)  # Green to Red
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Diverging palette | Threshold-centered heatmaps | `sns.diverging_palette(10, 130, as_cmap=True)` |
| Categorical palette | Discrete group colors | `sns.color_palette("Set2", n)` |
| Heatmap center | Statistical threshold | `center=0.80` for AIR materiality |
| Violin inner | Distribution + summary | `inner="box"` shows box inside violin |
| Cut parameter | Limit violin extent | `cut=0` constrains to data range |

## Common Patterns

### Heatmap with Statistical Threshold

**When:** Visualizing AIR, effect sizes, or metrics with meaningful cutoffs.

```python
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap=cmap, center=0.80,
    vmin=0.5, vmax=1.2, ax=ax, linewidths=0.5,
    cbar_kws={"label": "AIR (< 0.80 = material)"},
)
```

### Count Heatmap

**When:** Showing frequency cross-tabulations.

```python
sns.heatmap(ct, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
            linewidths=0.5, cbar_kws={"label": "Count"})
```

## See Also

- [patterns](references/patterns.md) - Heatmap, violin, and color patterns
- [workflows](references/workflows.md) - End-to-end chart generation

## Related Skills

- **matplotlib** skill for figure creation and saving
- **pandas** skill for pivot tables and crosstabs
- **numpy** skill for numerical operations in visualizations