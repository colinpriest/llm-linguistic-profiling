# Seaborn Workflows Reference

## Contents
- Creating Embedded HTML Charts
- Building Multi-Panel Figures
- Statistical Heatmap Workflow
- Validation Checklist

---

## Creating Embedded HTML Charts

### Complete Workflow

```python
import matplotlib
matplotlib.use("Agg")  # MUST be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

def create_chart_base64(data: pd.DataFrame) -> str:
    """Generate chart and return base64 PNG."""
    # 1. Create figure with explicit size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2. Generate seaborn plot
    sns.heatmap(data, annot=True, fmt=".2f", ax=ax)
    
    # 3. Configure axes (matplotlib)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_title("Chart Title", fontweight="bold")
    
    # 4. Tighten layout
    fig.tight_layout()
    
    # 5. Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)  # Prevent memory leak
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
```

### Iterate-Until-Pass Pattern

1. Generate chart
2. Validate output is non-empty string
3. If empty, check data guards and retry with fallback
4. Only embed in HTML when validation passes

```python
chart_b64 = create_heatmap(data)
if not chart_b64:
    logger.warning("Heatmap generation failed, using fallback")
    chart_b64 = create_simple_table(data)  # Text fallback
```

---

## Building Multi-Panel Figures

### Grid of Dimension Charts

```python
# src/reporter.py pattern for 6 demographic dimensions
n_dims = len(ESTIMATE_DIMS)
n_cols = 3
n_rows = (n_dims + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
fig.suptitle("Title", fontsize=14, fontweight="bold")

for ax, dim in zip(axes.flat, ESTIMATE_DIMS):
    # Plot for each dimension
    dim_data = data.get(dim, {})
    if dim_data:
        colors = sns.color_palette("Set2", len(dim_data))
        ax.pie(list(dim_data.values()), labels=list(dim_data.keys()),
               autopct="%1.1f%%", colors=colors)
        ax.set_title(dim.capitalize())
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")

# Hide unused axes
for ax in axes.flat[len(ESTIMATE_DIMS):]:
    ax.set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
```

### Side-by-Side Comparison (2 metrics)

```python
# src/reporter.py:1986-2017 pattern
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, metric in enumerate(["monetary_severity", "issue_severity"]):
    ax = axes[idx]
    # Plot metric-specific data
    ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Comparison Title", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
```

---

## Statistical Heatmap Workflow

### AIR Heatmap from Raw Data

```python
def create_air_heatmap(air_data: Dict[str, Dict]) -> str:
    """Complete workflow: raw data -> embedded heatmap."""
    
    # 1. Parse persona keys into dimensions
    rows = []
    for persona, vals in air_data.items():
        parts = persona.split("_")
        if len(parts) >= 4:
            rows.append({
                "row_key": f"{parts[0]}_{parts[1]}",  # age_geo
                "col_key": f"{parts[2]}_{parts[3]}",  # edu_ses
                "air": vals.get("air", 1.0),
            })
    
    if not rows:
        return ""
    
    # 2. Create pivot table (pandas)
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="row_key", columns="col_key",
        values="air", aggfunc="first"
    )
    
    # 3. Generate heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap=cmap, center=0.80,
        vmin=0.5, vmax=1.2, ax=ax, linewidths=0.5,
        cbar_kws={"label": "AIR (< 0.80 = material)"},
    )
    ax.set_xlabel("Education + SES")
    ax.set_ylabel("Age + Geography")
    
    # 4. Convert and return
    fig.tight_layout()
    return _fig_to_base64(fig)
```

---

## Validation Checklist

Copy this checklist when implementing seaborn visualizations:

- [ ] Import seaborn AFTER `matplotlib.use("Agg")`
- [ ] Create explicit `fig, ax` before plotting
- [ ] Pass `ax=ax` to all seaborn functions
- [ ] Add data guards for empty/missing data
- [ ] Use fallback plot type if primary fails (violin -> box)
- [ ] Call `fig.tight_layout()` before saving
- [ ] Call `plt.close(fig)` after conversion to prevent leaks
- [ ] Return empty string `""` for failed charts, not `None`

---

## Common Parameters Quick Reference

### Heatmap Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `annot` | `True` | Show values in cells |
| `fmt` | `".2f"` / `".0f"` | Decimal / integer format |
| `cmap` | Colormap or string | Color scheme |
| `center` | Float | Diverging palette center |
| `vmin/vmax` | Float | Color range limits |
| `linewidths` | `0.5` | Cell border width |
| `cbar_kws` | Dict | Colorbar options |

### Violin/Box Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `inner` | `"box"` | Show quartiles inside |
| `cut` | `0` | Constrain to data range |
| `palette` | `"Set2"` | Categorical colors |

---

## Related Skills

- **matplotlib** skill for figure management and base64 conversion
- **pandas** skill for pivot_table and crosstab operations
- **numpy** skill for numerical calculations in visualizations