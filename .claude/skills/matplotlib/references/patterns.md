# Matplotlib Patterns Reference

## Contents
- Backend and Import Configuration
- Color Schemes
- Chart Types
- Axis and Layout Configuration
- Text and Annotations
- Anti-Patterns

---

## Backend and Import Configuration

### Agg Backend for Headless Rendering

```python
# src/reporter.py:14-20, src/figure_generator.py:101-102
import matplotlib
matplotlib.use("Agg")  # MUST be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import seaborn as sns  # noqa: E402
```

**Why Agg:** Server environments have no display. Agg renders to memory/file without requiring X11 or a GUI backend. The `# noqa: E402` pragma bypasses flake8's import order rule since backend must be set first.

---

## Color Schemes

### Publication Grayscale Palette

```python
# src/figure_generator.py:26-28
_GRAYS = ["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3",
          "#333333", "#666666", "#999999", "#cccccc"]
_HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "**"]
```

**When:** Publication figures that must be readable in grayscale. Combine with hatching for additional differentiation.

### Seaborn Diverging Palettes

```python
# AIR heatmap: red = bad (< 0.80), green = good
cmap = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(pivot, cmap=cmap, center=0.80, vmin=0.5, vmax=1.2)

# Process burden: reversed (green = low burden)
cmap = sns.diverging_palette(130, 10, as_cmap=True)
sns.heatmap(pivot, cmap=cmap, center=1.25, vmin=0.8, vmax=2.0)
```

### Semantic Colors for Flags

```python
# src/reporter.py:30-35
FLAG_COLORS = {
    "BOTH": "#e74c3c",       # Red - significant AND material
    "SIG_ONLY": "#e67e22",   # Orange - significant only
    "MAT_ONLY": "#f1c40f",   # Yellow - material only
    "NONE": "#2ecc71",       # Green - neither
}
```

---

## Chart Types

### Grouped Bar Charts

```python
# src/figure_generator.py:454-455
x = np.arange(len(labels))
w = 0.3
ax.bar(x - w/2, baseline, w, label="Baseline", color="#555555")
ax.bar(x + w/2, neutralised, w, label="Neutralised", color="#aaaaaa")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
```

### Horizontal Bar with Reference Line

```python
# src/figure_generator.py:336
y = range(len(dims_plotted))
ax.barh(y, gaps, color="#555555", height=0.5)
ax.set_yticks(list(y))
ax.set_yticklabels(dims_plotted)
ax.axvline(x=0, color="black", linewidth=0.5)
ax.invert_yaxis()  # Top-to-bottom reading order
```

### Seaborn Heatmap with Annotations

```python
# src/reporter.py:223-228
sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap=cmap, center=0.80,
    vmin=0.5, vmax=1.2, ax=ax, linewidths=0.5,
    cbar_kws={"label": "AIR (< 0.80 = material)"},
)
```

### Forest Plot (Error Bars)

```python
# src/reporter.py:261-266
ax.errorbar(
    rates, y_pos, xerr=[errors_lo, errors_hi],
    fmt="o", color="#2c3e50", ecolor="#bdc3c7", capsize=3, markersize=5,
)
ax.axvline(overall_mean, color="#e74c3c", linestyle="--", linewidth=1)
```

### KDE Posterior Distribution

```python
# src/figure_generator.py:839-851
from scipy.stats import gaussian_kde

kde = gaussian_kde(synth_draws)
x_grid = np.linspace(a_mean - 4 * std_est, a_mean + 4 * std_est, 300)
density = kde(x_grid)
ax.fill_between(x_grid, density, alpha=0.3, color="#666666")
ax.plot(x_grid, density, color="#333333", linewidth=1.5)

# Shade 95% credible interval
mask = (x_grid >= a_lo) & (x_grid <= a_hi)
ax.fill_between(x_grid[mask], density[mask], alpha=0.2, color="#333333")
ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="A = 0")
```

---

## Axis and Layout Configuration

### Subplot Creation with Consistent Indexing

```python
# GOOD - squeeze=False ensures axes is always 2D
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
ax = axes[0, i]  # Always works

# GOOD - Manual newaxis for single-row case
fig, axes = plt.subplots(n_models, 4, figsize=(14, 3.5 * n_models))
if n_models == 1:
    axes = axes[np.newaxis, :]
```

### Tight Layout with Suptitle Space

```python
fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave 5% for title
```

### Reference Lines for Thresholds

```python
ax.axvline(0.80, color="#f39c12", linestyle="--", linewidth=1.5,
           label="Materiality (0.80)")
ax.axhline(0, color="black", linewidth=0.5)
```

---

## Text and Annotations

### Bar Value Labels

```python
# src/reporter.py:336-338
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
```

### Styled Text Box

```python
# src/figure_generator.py:511-518
ax.text(0.5, 0.5,
        f"Test: {test}\nStatistic: {stat:.2f}\np-value: {p_val:.4f}",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0"))
```

### Custom Legend Patches

```python
# src/reporter.py:456-461
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="p < 0.05"),
    Patch(facecolor="#bdc3c7", label="p >= 0.05"),
]
ax.legend(handles=legend_elements, fontsize=8)
```

---

## Anti-Patterns

### WARNING: Backend After Import

**The Problem:**

```python
# BAD - Backend set too late
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # UserWarning: backend already set
```

**Why This Breaks:** pyplot initializes the backend on import. Changing it afterward is a no-op on some systems, causes warnings on others.

**The Fix:**

```python
# GOOD - Backend before pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

### WARNING: Inconsistent Subplot Indexing

**The Problem:**

```python
# BAD - Breaks when n_models == 1
fig, axes = plt.subplots(n_models, 4)
ax = axes[0, 0]  # IndexError when n_models == 1
```

**Why This Breaks:** Single-row subplots return 1D array, multi-row returns 2D.

**The Fix:**

```python
# GOOD - Always 2D
fig, axes = plt.subplots(n_models, 4)
if n_models == 1:
    axes = axes[np.newaxis, :]
```

### WARNING: Missing plt.close()

**The Problem:**

```python
# BAD - Memory leak
fig, ax = plt.subplots()
ax.plot(data)
fig.savefig("plot.png")
# Figure never closed
```

**Why This Breaks:** Matplotlib keeps figures in memory. In loops, this causes memory exhaustion.

**The Fix:**

```python
# GOOD - Explicit close
fig.savefig("plot.png")
plt.close(fig)
```
