# Jinja2 Patterns Reference

## Contents
- Environment Configuration
- Template Inheritance for Reports
- Macros for UI Components
- Loop Patterns with Statistics Data
- Filter Patterns
- Security Patterns

---

## Environment Configuration

### Production-Safe Setup

```python
from jinja2 import Environment, FileSystemLoader, select_autoescape

def create_report_environment(template_dir: str = "templates") -> Environment:
    """Create Jinja2 environment with security defaults."""
    return Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,      # Remove first newline after block tags
        lstrip_blocks=True,    # Strip leading whitespace before block tags
    )
```

### WARNING: Disabled Autoescaping

**The Problem:**

```python
# BAD - XSS vulnerability
env = Environment(loader=FileSystemLoader("templates"))
template.render(user_input=complaint_narrative)  # Unescaped!
```

**Why This Breaks:**
1. Complaint narratives may contain `<script>` tags from malicious users
2. HTML entities render incorrectly, breaking layout
3. Audit failures for security-sensitive applications

**The Fix:**

```python
# GOOD - Always enable autoescape for HTML
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"]),
)
```

---

## Template Inheritance for Reports

### Base Template Structure

```html
{# templates/base_report.html #}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}Report{% endblock %}</title>
    <style>{% block styles %}{% endblock %}</style>
</head>
<body>
    <header class="header">
        {% block header %}
        <h1>{{ report_title }}</h1>
        {% endblock %}
    </header>
    
    <nav class="tab-nav">
        {% block navigation %}{% endblock %}
    </nav>
    
    <main>
        {% block content %}{% endblock %}
    </main>
    
    <script>{% block scripts %}{% endblock %}</script>
</body>
</html>
```

### Child Template (Fairness Report)

```html
{# templates/fairness_report.html #}
{% extends "base_report.html" %}

{% block title %}LLM Fairness Report - {{ timestamp }}{% endblock %}

{% block header %}
<h1>LLM Fairness Experiment Report</h1>
<div class="subtitle">Generated: {{ timestamp }} | Seed: {{ seed }}</div>
{% endblock %}

{% block content %}
{% for model in models %}
    {% include "partials/model_section.html" %}
{% endfor %}
{% endblock %}
```

---

## Macros for UI Components

### Metric Card Macro

```html
{# templates/macros/cards.html #}
{% macro metric_card(value, label, format="default") %}
<div class="metric-card">
    <div class="value">
        {% if format == "percent" %}
            {{ (value * 100)|round(1) }}%
        {% elif format == "decimal" %}
            {{ value|round(4) }}
        {% else %}
            {{ value }}
        {% endif %}
    </div>
    <div class="label">{{ label }}</div>
</div>
{% endmacro %}
```

### Traffic Light Badge Macro

```html
{% macro flag_badge(flag) %}
{% set colors = {
    "BOTH": "#e74c3c",
    "SIG_ONLY": "#e67e22", 
    "MAT_ONLY": "#f1c40f",
    "NONE": "#2ecc71"
} %}
<span class="badge" style="background: {{ colors.get(flag, '#95a5a6') }}">
    {{ flag }}
</span>
{% endmacro %}
```

### Using Macros

```html
{% from "macros/cards.html" import metric_card, flag_badge %}

<div class="metrics-grid">
    {{ metric_card(num_complaints, "Complaints Analyzed") }}
    {{ metric_card(variance_reduction, "Variance Reduction", format="percent") }}
</div>

<td>{{ flag_badge(row.flag) }}</td>
```

---

## Loop Patterns with Statistics Data

### Iterating Analysis Results

```html
{% for model in models %}
<h3>Model: {{ model }}</h3>
{% set model_res = analysis_results[model] %}

<table class="sortable">
    <thead>
        <tr>
            <th>Dimension</th>
            <th>p-value</th>
            <th>Cramer's V</th>
            <th>Assessment</th>
        </tr>
    </thead>
    <tbody>
    {% for dim in dimensions %}
        {% set d = model_res.correlation.get(dim ~ "_chi2", {}) %}
        <tr class="flag-row-{{ d.flag|default('NONE') }}">
            <td>{{ dim|title }}</td>
            <td>{{ d.p_value|round(4) if d.p_value is not none else "N/A" }}</td>
            <td>{{ d.cramers_v|round(4) if d.cramers_v is not none else "N/A" }}</td>
            <td>{{ flag_badge(d.flag|default('NONE')) }}</td>
        </tr>
    {% endfor %}
    </tbody>
</table>
{% endfor %}
```

### Loop Variables

```html
{% for row in air_data %}
<tr{% if loop.first %} class="first-row"{% endif %}
    {% if loop.last %} class="last-row"{% endif %}>
    <td>{{ loop.index }}.</td>  {# 1-indexed #}
    <td>{{ row.persona }}</td>
    <td>{{ row.air|round(2) }}</td>
</tr>
{% endfor %}
```

---

## Filter Patterns

### Custom Filter Registration

```python
def format_pvalue(val: float | None) -> str:
    """Format p-value with scientific notation for small values."""
    if val is None:
        return "N/A"
    if val < 0.0001:
        return f"{val:.2e}"
    return f"{val:.4f}"

def format_percent(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"

env = Environment(loader=FileSystemLoader("templates"))
env.filters["pvalue"] = format_pvalue
env.filters["pct"] = format_percent
```

### Using Custom Filters

```html
<td>{{ result.p_value|pvalue }}</td>
<td>{{ result.rate|pct }}</td>
```

---

## Security Patterns

### WARNING: Using `|safe` on User Input

**The Problem:**

```html
{# BAD - Never do this with user content #}
<div class="narrative">{{ complaint.narrative|safe }}</div>
```

**Why This Breaks:**
1. Allows XSS attacks via malicious complaint text
2. Breaks HTML structure if narrative contains unescaped `<` or `>`

**The Fix:**

```html
{# GOOD - Let autoescaping handle it #}
<div class="narrative">{{ complaint.narrative }}</div>

{# GOOD - Only use safe for content YOU generated #}
<div class="chart">{{ chart_base64_img|safe }}</div>
```

### When `|safe` IS Appropriate

```python
# Pre-sanitized HTML you generated (not user input)
chart_html = f'<img src="data:image/png;base64,{base64_data}" alt="Chart">'
```

```html
{{ chart_html|safe }}  {# OK - you generated this, not the user #}