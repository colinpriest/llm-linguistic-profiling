# Jinja2 Workflows Reference

## Contents
- Migration from F-Strings Workflow
- Report Generation Workflow
- Template Development Workflow
- Testing Templates Workflow

---

## Migration from F-Strings Workflow

This codebase currently uses f-strings in `src/reporter.py` (~3000 lines). Migration to Jinja2 improves maintainability.

### Checklist

Copy this checklist and track progress:
- [ ] Step 1: Create `templates/` directory structure
- [ ] Step 2: Extract base HTML skeleton to `templates/base.html`
- [ ] Step 3: Create macros for repeated components (badges, cards, tables)
- [ ] Step 4: Convert one tab at a time, starting with simplest
- [ ] Step 5: Add custom filters for number formatting
- [ ] Step 6: Update `reporter.py` to use Environment
- [ ] Step 7: Verify output matches original

### Step 1: Directory Structure

```
templates/
├── base.html
├── fairness_report.html
├── worked_examples.html
├── macros/
│   ├── badges.html
│   ├── cards.html
│   └── tables.html
└── partials/
    ├── executive_summary.html
    ├── disparate_impact.html
    └── model_section.html
```

### Step 2: Extract Repeated Patterns

**Before (f-string in reporter.py:1058-1063):**

```python
parts.append('<div class="metrics-grid">')
parts.append(f'<div class="metric-card"><div class="value">{num_complaints}</div><div class="label">Complaints Analyzed</div></div>')
parts.append(f'<div class="metric-card"><div class="value">{len(models)}</div><div class="label">Models Tested</div></div>')
parts.append("</div>")
```

**After (Jinja2 macro + template):**

```html
{# macros/cards.html #}
{% macro metric_card(value, label) %}
<div class="metric-card">
    <div class="value">{{ value }}</div>
    <div class="label">{{ label }}</div>
</div>
{% endmacro %}

{# usage in template #}
{% from "macros/cards.html" import metric_card %}
<div class="metrics-grid">
    {{ metric_card(num_complaints, "Complaints Analyzed") }}
    {{ metric_card(models|length, "Models Tested") }}
</div>
```

### Step 3: Convert Reporter Class

```python
# src/reporter.py
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path

class ReportGenerator:
    def __init__(self, template_dir: Path | None = None):
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / "templates"
        
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._register_filters()
    
    def _register_filters(self) -> None:
        self.env.filters["pvalue"] = self._format_pvalue
        self.env.filters["pct"] = self._format_percent
    
    @staticmethod
    def _format_pvalue(val: float | None) -> str:
        if val is None:
            return "N/A"
        return f"{val:.2e}" if val < 0.0001 else f"{val:.4f}"
    
    @staticmethod
    def _format_percent(val: float | None) -> str:
        return "N/A" if val is None else f"{val * 100:.1f}%"
    
    def generate_report(
        self,
        analysis_results: dict,
        complaints: list,
        **kwargs,
    ) -> str:
        template = self.env.get_template("fairness_report.html")
        return template.render(
            analysis_results=analysis_results,
            complaints=complaints,
            **kwargs,
        )
```

---

## Report Generation Workflow

### Integration with Pandas DataFrames

See the **pandas** skill for DataFrame operations.

```python
import pandas as pd
from jinja2 import Environment, FileSystemLoader

def render_table_from_df(df: pd.DataFrame, template_name: str) -> str:
    """Render a DataFrame as an HTML table using Jinja2."""
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template(template_name)
    
    return template.render(
        columns=df.columns.tolist(),
        rows=df.to_dict("records"),
    )
```

```html
{# templates/data_table.html #}
<table class="sortable">
    <thead>
        <tr>
        {% for col in columns %}
            <th>{{ col }}</th>
        {% endfor %}
        </tr>
    </thead>
    <tbody>
    {% for row in rows %}
        <tr>
        {% for col in columns %}
            <td>{{ row[col] }}</td>
        {% endfor %}
        </tr>
    {% endfor %}
    </tbody>
</table>
```

### Embedding Matplotlib Charts

See the **matplotlib** skill for chart generation.

```python
import base64
import io
import matplotlib.pyplot as plt

def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

# In template context
context = {
    "chart_img": fig_to_base64(my_figure),
}
```

```html
{# Template #}
<div class="chart-container">
    <img src="data:image/png;base64,{{ chart_img }}" alt="{{ chart_title }}">
</div>
```

---

## Template Development Workflow

### Iterate-Until-Pass Pattern

1. Create template file
2. Run: `python -c "from src.reporter import ReportGenerator; r = ReportGenerator(); print(r.generate_report({}, []))"`
3. If Jinja2 raises `TemplateSyntaxError` or `UndefinedError`, fix and repeat step 2
4. Visually inspect output HTML
5. Only proceed when template renders correctly

### Common Errors and Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `UndefinedError: 'foo' is undefined` | Variable not passed to context | Add to `render()` kwargs or use `default` filter |
| `TemplateSyntaxError: unexpected '}'` | Missing `{%` or `{{` | Check block tag syntax |
| `TypeError: 'NoneType' object is not iterable` | Looping over None | Use `{% for x in items or [] %}` |

### Default Value Pattern

```html
{# Prevent UndefinedError when optional values are missing #}
<td>{{ result.cramers_v|default("N/A")|round(4) }}</td>

{# For loops with potentially empty lists #}
{% for item in items|default([]) %}
    {{ item }}
{% endfor %}
```

---

## Testing Templates Workflow

### Unit Testing Template Rendering

```python
import pytest
from jinja2 import Environment, FileSystemLoader

@pytest.fixture
def jinja_env():
    return Environment(
        loader=FileSystemLoader("templates"),
        autoescape=True,
    )

def test_metric_card_renders(jinja_env):
    template = jinja_env.get_template("macros/cards.html")
    # Import the macro
    module = template.module
    
    result = module.metric_card(5000, "Complaints")
    assert "5000" in result
    assert "Complaints" in result
    assert "metric-card" in result

def test_flag_badge_colors(jinja_env):
    template = jinja_env.from_string("""
        {% from "macros/badges.html" import flag_badge %}
        {{ flag_badge("BOTH") }}
    """)
    result = template.render()
    assert "#e74c3c" in result  # Red color for BOTH
```

### Snapshot Testing

```python
def test_report_snapshot(jinja_env, snapshot):
    template = jinja_env.get_template("fairness_report.html")
    html = template.render(
        models=["gpt-4o-mini"],
        analysis_results={"gpt-4o-mini": {}},
        num_complaints=100,
        timestamp="2026-02-06",
    )
    assert html == snapshot
```

### Checklist for Template Changes

Copy this checklist and track progress:
- [ ] Template renders without syntax errors
- [ ] All variables have defaults or are guaranteed in context
- [ ] Autoescaping is enabled for HTML templates
- [ ] Custom filters handle None gracefully
- [ ] Visual output matches expected design
- [ ] Tests pass