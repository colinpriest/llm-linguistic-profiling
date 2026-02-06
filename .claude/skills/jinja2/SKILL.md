---
name: jinja2
description: |
  Generates templated HTML reports with dynamic content injection.
  Use when: creating HTML reports, rendering templates with variables,
  generating multi-tab dashboards, building email templates, or any
  task requiring separation of presentation from logic.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Jinja2 Skill

Jinja2 is Python's premier templating engine for generating HTML, XML, or text files with dynamic content. This codebase uses Jinja2 3.1+ for report generation.

## WARNING: Current Codebase Uses F-String Templating

**Detected:** `src/reporter.py` and `src/worked_examples.py` use Python f-strings instead of Jinja2 templates despite Jinja2 being in `requirements.txt`.

**Impact:** 3000+ lines of HTML embedded in Python code. Hard to maintain, no template inheritance, difficult for non-Python developers to modify styling.

**Recommendation:** Migrate to Jinja2 templates for separation of concerns.

## Quick Start

### Basic Template Rendering

```python
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Create environment with autoescaping for HTML
env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html", "xml"]),
)

template = env.get_template("report.html")
html = template.render(
    title="LLM Fairness Report",
    models=["gpt-4o-mini", "gpt-5-mini"],
    results=analysis_results,
)
```

### String Template (No Files)

```python
from jinja2 import Environment

env = Environment(autoescape=True)
template = env.from_string("""
<div class="metric-card">
    <div class="value">{{ value }}</div>
    <div class="label">{{ label }}</div>
</div>
""")
html = template.render(value=5000, label="Complaints Analyzed")
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Variables | Output values | `{{ model_name }}` |
| Filters | Transform values | `{{ name\|title }}`, `{{ rate\|round(4) }}` |
| Control flow | Conditionals | `{% if flag == "BOTH" %}...{% endif %}` |
| Loops | Iteration | `{% for model in models %}...{% endfor %}` |
| Inheritance | Base templates | `{% extends "base.html" %}` |
| Blocks | Override sections | `{% block content %}...{% endblock %}` |
| Macros | Reusable components | `{% macro badge(text, color) %}...{% endmacro %}` |
| Includes | Partial templates | `{% include "partials/header.html" %}` |

## Common Patterns

### Escaping User Content

**When:** Rendering complaint narratives or any user-provided text

```html
{# Autoescaping handles this, but be explicit for clarity #}
<p>{{ complaint.narrative|e }}</p>

{# For content you've already sanitized and trust #}
<div class="exec-card">{{ summary_html|safe }}</div>
```

### Conditional CSS Classes

**When:** Applying traffic-light styling based on flags

```html
<tr class="flag-row-{{ flag }}">
    <td>{{ dimension }}</td>
    <td><span class="badge badge-{{ flag }}">{{ flag }}</span></td>
</tr>
```

### Number Formatting

**When:** Displaying p-values, AIR ratios, percentages

```html
<td>{{ p_value|round(4) if p_value is not none else "N/A" }}</td>
<td>{{ (rate * 100)|round(1) }}%</td>
```

## See Also

- [patterns](references/patterns.md) - Template patterns for HTML reports
- [workflows](references/workflows.md) - Migration and integration workflows

## Related Skills

- See the **python** skill for environment setup and type hints
- See the **pandas** skill for DataFrame-to-template data preparation
- See the **matplotlib** skill for embedding base64 chart images
- See the **plotly** skill for interactive chart HTML injection