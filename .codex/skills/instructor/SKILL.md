---
name: instructor
description: |
  Integrates Pydantic validation with LLM responses for structured outputs.
  Use when: defining response schemas for LLM calls, validating LLM JSON output,
  handling type coercion from unpredictable LLM responses, or building reliable
  LLM pipelines with structured data extraction.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Instructor Skill

Instructor patches the OpenAI client to extract and validate structured JSON responses using Pydantic models. This codebase uses `instructor.from_openai()` with aggressive field validators to handle LLM output inconsistencies. All schemas use `mode="before"` validators to coerce values before Pydantic's type system enforces constraints.

## Quick Start

### Patching the OpenAI Client

```python
import instructor
from openai import OpenAI

raw_client = OpenAI(api_key=config.api_key)
client = instructor.from_openai(raw_client)
```

### Calling with a Response Model

```python
from pydantic import BaseModel, Field
from typing import Literal

class PersonaEstimateResponse(BaseModel):
    age: Literal["young", "old"]
    confidence: float = Field(ge=0.0, le=1.0)

result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=PersonaEstimateResponse,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)
# result is a validated PersonaEstimateResponse instance
print(result.age, result.confidence)
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Patching | Wrap OpenAI client | `instructor.from_openai(client)` |
| Response model | Pass Pydantic class | `response_model=MySchema` |
| Coercion validators | Handle LLM variations | `@field_validator("age", mode="before")` |
| Safe defaults | Never fail validation | `return "young"` as fallback |

## Common Patterns

### Robust Field Validation

**When:** LLM outputs unexpected values for constrained fields

```python
@field_validator("tier", mode="before")
@classmethod
def coerce_tier(cls, v: Any) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return 0
    return v if v in (0, 1, 2) else 0
```

### Clamping Numeric Ranges

**When:** LLM returns confidence scores outside [0, 1]

```python
@field_validator("confidence", mode="before")
@classmethod
def clamp_confidence(cls, v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))
```

## See Also

- [patterns](references/patterns.md) - Schema design and validation patterns
- [workflows](references/workflows.md) - Batch execution and caching workflows

## Related Skills

- See the **pydantic** skill for BaseModel and field validation details
- See the **openai** skill for raw API client configuration
- See the **python** skill for type hints and dataclass conventions