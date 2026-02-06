---
name: pydantic
description: |
  Validates structured JSON responses and enforces data model schemas.
  Use when: defining LLM response schemas, validating API responses, coercing messy input data, or creating type-safe data containers.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Pydantic Skill

Pydantic v2 is used in this codebase for LLM response validation via the **instructor** library. The key design principle: **aggressive coercion with sensible defaults** rather than strict validation. LLM outputs contain typos and variations—validators normalize them instead of failing.

## Quick Start

### LLM Response Schema

```python
from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal, List

class RemedyDecisionResponse(BaseModel):
    tier: Literal[0, 1, 2]
    confidence: Literal["confident_no_action", "confident_action_needed", "need_more_info"]
    reasoning: str = ""
    information_needed: Optional[str] = None

    @field_validator("tier", mode="before")
    @classmethod
    def coerce_tier(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        return v if v in (0, 1, 2) else 0
```

### Using with Instructor

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())
result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=RemedyDecisionResponse,  # Pydantic schema
    messages=[{"role": "user", "content": prompt}],
)
# result is a validated RemedyDecisionResponse instance
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| `Literal[]` | Constrain to specific values | `tier: Literal[0, 1, 2]` |
| `Field()` | Constraints, defaults, descriptions | `Field(ge=0.0, le=1.0)` |
| `field_validator` | Pre-validation coercion | `mode="before"` for input normalization |
| `default_factory` | Mutable defaults | `Field(default_factory=list)` |
| `model_dump()` | Serialize to dict | For caching, DataFrame creation |

## Common Patterns

### Coercing LLM Output Variations

**When:** LLM returns "highschool" but schema expects "high_school"

```python
@field_validator("education", mode="before")
@classmethod
def coerce_education(cls, v: Any) -> str:
    v = str(v).strip().lower()
    if v in ("high_school", "college"):
        return v
    if v in ("highschool", "high school", "hs", "secondary"):
        return "high_school"
    if v in ("university", "graduate", "bachelor"):
        return "college"
    return "college"  # default fallback
```

### Optional Fields with Validation

```python
@field_validator("primary_amount", mode="before")
@classmethod
def coerce_primary_amount(cls, v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return abs(float(v))
    except (TypeError, ValueError):
        return None
```

## See Also

- [patterns](references/patterns.md) — Field validators, Literal constraints, coercion
- [workflows](references/workflows.md) — Instructor integration, caching, error handling

## Related Skills

- See the **instructor** skill for LLM-specific extraction patterns
- See the **openai** skill for API client configuration
- See the **python** skill for dataclass vs Pydantic guidance