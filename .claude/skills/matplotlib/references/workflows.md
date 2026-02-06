# Matplotlib Workflows Reference

## Contents
- Publication Export Workflow
- HTML Report Embedding
- Multi-Model Comparison Figures
- Error-Resilient Plotting
- Figure Metadata Management

---

## Publication Export Workflow

### Dual-Format Export (PDF + PNG)

```python
# src/figure_generator.py:61-66
def _save_figure(fig: plt.Figure, base_path: Path, caption: str, metadata: dict):
    """Save figure in both vector and raster formats with metadata."""
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    txt_path = base_path.with_suffix(".txt")
    json_path = base_path.with_suffix(".json")

    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)

    # Save caption and metadata
    txt_path.write_text(caption, encoding="utf-8")
    json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
```

**Why dual format:**
- PDF: Vector graphics for journal submissions, scales without pixelation
- PNG: Raster for web/presentations, universally viewable
- 300 DPI: Standard for print publication

### Publication Workflow Checklist

Copy this checklist and track progress:
- [ ] Step 1: Set Agg backend before imports
- [ ] Step 2: Use grayscale palette `_GRAYS` for accessibility
- [ ] Step 3: Add hatching patterns if >4 categories
- [ ] Step 4: Save as PDF with `bbox_inches="tight"`
- [ ] Step 5: Save as PNG with `facecolor="white"`
- [ ] Step 6: Call `plt.close(fig)` to free memory
- [ ] Step 7: Write caption to `.txt` file

---

## HTML Report Embedding

### Base64 Encoding for Self-Contained HTML

```python
# src/reporter.py:49-56
import io
import base64

def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """Render matplotlib figure to base64 PNG for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
```

### Embedding in HTML Template

```python
# Generate figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(categories, values)
ax.set_title("Distribution")

# Convert to embeddable string
img_data = _fig_to_base64(fig)

# Use in Jinja2 template
template_vars = {
    "distribution_chart": f"data:image/png;base64,{img_data}"
}
```

```html
<!-- In Jinja2 template -->
<img src="{{ distribution_chart }}" alt="Distribution Chart">
```

**Why base64:** Single self-contained HTML file, no external image dependencies, works offline.

---

## Multi-Model Comparison Figures

### Dynamic Grid Layout

```python
# src/figure_generator.py:194
n_models = len(self.models)
fig, axes = plt.subplots(n_models, 4, figsize=(14, 3.5 * n_models))

# Handle single-model case
if n_models == 1:
    axes = axes[np.newaxis, :]  # Convert to 2D array

for i, model in enumerate(models):
    for j, metric in enumerate(["AIR", "Chi-sq", "Effect", "Severity"]):
        ax = axes[i, j]
        plot_metric(ax, data[model][metric])
        ax.set_title(f"{model}: {metric}")

fig.tight_layout(rect=[0, 0, 1, 0.95])
```

### Calculated Grid Dimensions

```python
# src/reporter.py:155
n_dims = len(ESTIMATE_DIMS)
n_cols = 3
n_rows = (n_dims + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes_flat = axes.flatten()

for idx, dim in enumerate(ESTIMATE_DIMS):
    ax = axes_flat[idx]
    plot_dimension(ax, dim)

# Hide unused axes
for idx in range(len(ESTIMATE_DIMS), len(axes_flat)):
    axes_flat[idx].axis("off")
```

---

## Error-Resilient Plotting

### Fallback Chart Types

```python
# src/reporter.py:411-415
def plot_distribution(ax, data, feature):
    """Plot distribution with violin -> boxplot fallback."""
    try:
        sns.violinplot(data=data, x="variant", y=feature, ax=ax,
                       inner="box", palette="Set2", cut=0)
    except Exception:
        # Violin fails with few data points or constant values
        sns.boxplot(data=data, x="variant", y=feature, ax=ax,
                    palette="Set2")
```

### KDE with Histogram Fallback

```python
# src/figure_generator.py:838-849
def plot_posterior(ax, draws, label):
    """Plot posterior density with histogram fallback."""
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(draws)
        x_grid = np.linspace(draws.min(), draws.max(), 300)
        density = kde(x_grid)
        ax.fill_between(x_grid, density, alpha=0.3, color="#666666")
        ax.plot(x_grid, density, color="#333333", linewidth=1.5)
    except Exception:
        # KDE fails with too few unique values
        ax.hist(draws, bins=50, density=True, color="#666666", alpha=0.5)

    ax.set_xlabel(label)
    ax.set_ylabel("Density")
```

### Empty Data Handling

```python
def plot_with_empty_check(ax, data, title):
    """Handle empty or invalid data gracefully."""
    if data is None or len(data) == 0:
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#999999")
        ax.axis("off")
        return

    # Proceed with actual plotting
    ax.bar(range(len(data)), data)
    ax.set_title(title)
```

### Error Handling Workflow

1. Attempt primary visualization (e.g., violin plot)
2. If exception: fallback to simpler chart (e.g., boxplot)
3. If still fails: show "No data" message
4. Always call `plt.close(fig)` even on error

---

## Figure Metadata Management

### Structured Metadata with Figure

```python
# src/figure_generator.py
def generate_air_heatmap(self, model: str) -> Path:
    """Generate AIR heatmap with full metadata."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot logic...
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f")

    # Metadata for paper
    metadata = {
        "figure_type": "heatmap",
        "model": model,
        "metric": "adverse_impact_ratio",
        "threshold": 0.80,
        "n_personas": len(pivot),
        "timestamp": datetime.now().isoformat(),
    }

    caption = (
        f"Adverse Impact Ratio (AIR) heatmap for {model}. "
        f"Values below 0.80 (red) indicate potential disparate impact. "
        f"Based on {len(pivot)} persona combinations."
    )

    base_path = self.output_dir / f"air_heatmap_{model}"
    _save_figure(fig, base_path, caption, metadata)
    return base_path.with_suffix(".pdf")
```

### Figure Registry Pattern

```python
class FigureGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.generated_figures: List[dict] = []

    def _register_figure(self, path: Path, fig_type: str, metadata: dict):
        """Track all generated figures for manifest."""
        self.generated_figures.append({
            "path": str(path),
            "type": fig_type,
            **metadata
        })

    def write_manifest(self):
        """Write JSON manifest of all generated figures."""
        manifest_path = self.output_dir / "figures_manifest.json"
        manifest_path.write_text(
            json.dumps(self.generated_figures, indent=2, default=str)
        )
```

---

## Integration with seaborn

### When to Use Each

| Task | Use matplotlib | Use seaborn |
|------|---------------|-------------|
| Custom diagram | Yes | No |
| Statistical heatmap | ax only | `sns.heatmap(ax=ax)` |
| Distribution comparison | No | `sns.violinplot` |
| Color palettes | No | `sns.color_palette()` |
| Publication styling | `plt.rcParams` | `sns.set_theme()` |

### Combined Workflow

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Create figure with matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: seaborn statistical plot
sns.violinplot(data=df, x="group", y="value", ax=axes[0])
axes[0].set_title("Distribution by Group")

# Right: matplotlib custom chart
axes[1].bar(categories, values, color=_GRAYS[:len(categories)])
axes[1].axhline(threshold, color="red", linestyle="--")
axes[1].set_title("Custom Threshold Chart")

fig.tight_layout()
```

See the **seaborn** skill for advanced statistical visualization patterns.
