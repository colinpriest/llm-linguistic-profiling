# Seaborn Patterns Reference

## Contents
- Heatmap Patterns
- Distribution Plots
- Color Palette Selection
- Anti-Patterns

---

## Heatmap Patterns

### Diverging Heatmap for Statistical Thresholds

Used for AIR (Adverse Impact Ratio) visualizations centered at materiality threshold:

```python
# src/reporter.py:223-228
cmap = sns.diverging_palette(10, 130, as_cmap=True)  # Red-to-Green
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap=cmap, center=0.80,
    vmin=0.5, vmax=1.2, ax=ax, linewidths=0.5,
    cbar_kws={"label": "AIR (< 0.80 = material)"},
)
```

**Key parameters:**
- `center=0.80`: EEOC's 80% rule threshold
- `vmin/vmax`: Defines color range boundaries
- `linewidths=0.5`: Grid separation for readability

### Reversed Diverging Palette

For metrics where higher = worse (e.g., process burden):

```python
# src/reporter.py:375-380
cmap = sns.diverging_palette(130, 10, as_cmap=True)  # Green-to-Red reversed
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap=cmap, center=1.25,
    vmin=0.8, vmax=2.0, ax=ax, linewidths=0.5,
    cbar_kws={"label": "Process AIR (> 1.25 = material)"},
)
```

### Sequential Heatmap for Counts

```python
# src/reporter.py:490
sns.heatmap(ct, cmap="YlOrRd", ax=ax, annot=True, fmt=".2f", linewidths=0.5)
```

Use `fmt=".0f"` for integer counts, `fmt=".2f"` for proportions.

---

## Distribution Plots

### Violin with Box Fallback

```python
# src/reporter.py:411-415
try:
    sns.violinplot(data=plot_df, x="variant", y=feat, ax=ax,
                   inner="box", palette="Set2", cut=0)
except Exception:
    sns.boxplot(data=plot_df, x="variant", y=feat, ax=ax, palette="Set2")
```

**Why fallback?** Violin plots fail on sparse data. Boxplots are more robust.

**Parameters:**
- `inner="box"`: Shows quartiles inside violin shape
- `cut=0`: Prevents violin extending beyond data range
- `palette="Set2"`: Consistent categorical colors

---

## Color Palette Selection

### Categorical Palettes

```python
# For pie charts and bar charts with discrete categories
colors = sns.color_palette("Set2", len(categories))

# More saturated version for emphasis
colors = sns.color_palette("Set1", len(models))
```

| Palette | Use Case |
|---------|----------|
| `Set2` | Default categorical (softer colors) |
| `Set1` | Multi-model comparison (saturated) |
| `YlOrRd` | Sequential counts/frequencies |

### Diverging Palettes

```python
# Green = good, Red = bad (AIR < 0.80 is bad)
cmap = sns.diverging_palette(10, 130, as_cmap=True)

# Reversed: Green = good at HIGH values (burden > 1.25 is bad)
cmap = sns.diverging_palette(130, 10, as_cmap=True)
```

**Hue values:** 10 = red, 130 = green in HSL space.

---

## Anti-Patterns

### WARNING: Using Seaborn Without Axis Parameter

**The Problem:**

```python
# BAD - Creates new figure, can't control placement
sns.heatmap(data)
plt.savefig("output.png")
```

**Why This Breaks:**
1. Creates implicit figure - can't embed in subplots
2. No control over figure size or DPI
3. Memory leaks from unclosed figures

**The Fix:**

```python
# GOOD - Explicit figure and axis control
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data, ax=ax)
fig.tight_layout()
# Convert or save explicitly
```

### WARNING: Hardcoded Color Values in Heatmaps

**The Problem:**

```python
# BAD - Magic numbers for colors
sns.heatmap(data, vmin=0, vmax=1, center=0.5)
```

**Why This Breaks:**
1. No semantic meaning to threshold
2. Different metrics need different centers
3. Hard to maintain consistency

**The Fix:**

```python
# GOOD - Meaningful thresholds with documentation
MATERIALITY_THRESHOLD = 0.80  # EEOC 80% rule
sns.heatmap(
    data, center=MATERIALITY_THRESHOLD,
    vmin=0.5, vmax=1.2,
    cbar_kws={"label": f"AIR (< {MATERIALITY_THRESHOLD} = material)"},
)
```

### WARNING: Not Handling Empty Data

**The Problem:**

```python
# BAD - Crashes on empty DataFrame
sns.violinplot(data=df, x="group", y="value")
```

**Why This Breaks:**
1. ValueError on empty data
2. No graceful degradation in reports

**The Fix:**

```python
# GOOD - Guard clause with fallback
if df.empty or df[["group", "value"]].dropna().empty:
    ax.text(0.5, 0.5, "No data", ha="center", va="center",
            transform=ax.transAxes)
    return
sns.violinplot(data=df, x="group", y="value", ax=ax)
```

---

## Integration with Matplotlib

All seaborn plots integrate with **matplotlib** for figure management. See the **matplotlib** skill for:
- Figure creation (`plt.subplots()`)
- Base64 conversion for HTML embedding
- Backend configuration (`matplotlib.use("Agg")`)

## Integration with Pandas

Heatmaps require pivot tables from **pandas**:

```python
# Create pivot for heatmap
pivot = df.pivot_table(index="row_key", columns="col_key", values="air")

# Create crosstab for topic distribution
ct = pd.crosstab(df["variant"], df["dominant_topic"], normalize="index")
```

See the **pandas** skill for DataFrame operations.