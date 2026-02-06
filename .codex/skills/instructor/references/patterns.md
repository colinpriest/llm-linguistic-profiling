# Instructor Patterns Reference

## Contents
- Schema Design Patterns
- Field Validator Patterns
- Response Model Usage
- Error Handling Patterns
- Anti-Patterns

---

## Schema Design Patterns

### Literal Fields with Coercion

LLMs frequently output variations like "elderly" instead of "old". Always coerce to valid values:

```python
class PersonaEstimateResponse(BaseModel):
    age: Literal["young", "old"]
    geographic: Literal["urban", "suburban", "rural"]
    
    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> str:
        v = str(v).strip().lower()
        if v in ("young", "old"):
            return v
        if v in ("middle", "middle-aged", "adult"):
            return "young"
        if v in ("elderly", "senior"):
            return "old"
        return "young"  # safe default
```

### Optional Fields with Type Coercion

```python
class DollarAmountResponse(BaseModel):
    primary_amount: Optional[float] = None
    
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

### Constrained Numeric Fields

```python
class RemedyDecisionResponse(BaseModel):
    tier: Literal[0, 1, 2]
    confidence: float = Field(ge=0.0, le=1.0)
    
    @field_validator("tier", mode="before")
    @classmethod
    def coerce_tier(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        return v if v in (0, 1, 2) else 0
    
    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))
```

---

## Field Validator Patterns

### DO: Use `mode="before"` for All Coercion

```python
# GOOD - intercepts raw LLM output before Pydantic type coercion
@field_validator("education", mode="before")
@classmethod
def coerce_education(cls, v: Any) -> str:
    v = str(v).strip().lower()
    if "college" in v or "university" in v or "degree" in v:
        return "college"
    return "high_school"
```

### WARNING: Never Use `mode="after"` for LLM Output

**The Problem:**

```python
# BAD - validation runs AFTER Pydantic type coercion
@field_validator("age", mode="after")
@classmethod
def validate_age(cls, v: str) -> str:
    # Too late - Pydantic already raised ValidationError
    return v
```

**Why This Breaks:**
1. Pydantic's type coercion fails before your validator runs
2. You cannot catch and fix LLM formatting errors
3. Entire response validation fails instead of gracefully degrading

**The Fix:** Always use `mode="before"` and handle all coercion yourself.

---

## Response Model Usage

### Passing Schema as Class (Not Instance)

```python
# GOOD - pass the class itself
result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=PersonaEstimateResponse,  # Class, not instance
    messages=[...],
)

# BAD - do not instantiate
result = client.chat.completions.create(
    response_model=PersonaEstimateResponse(),  # Wrong!
)
```

### Storing Both Parsed and Raw Responses

The codebase stores both for debugging and cache integrity:

```python
raw_json = validated_obj.model_dump_json()

return LLMResponse(
    response_raw=raw_json,           # For debugging
    response_parsed=validated_obj,   # For type-safe access
)
```

---

## Error Handling Patterns

### Disable Instructor's Internal Retries

This codebase implements custom retry logic with exponential backoff:

```python
validated_obj = self._client.chat.completions.create(
    model=request.model,
    response_model=request.expected_schema,
    max_retries=0,  # Application handles retries
    messages=[...],
)
```

### Re-validate Cached Responses

Cache stores dicts, not Pydantic objects. Re-validate on retrieval:

```python
cached = self.cache.get(model, task, narrative)
if cached is not None:
    try:
        validated = request.expected_schema(**cached)
        return validated
    except Exception:
        self.cache.delete(model, task, narrative)
        # Fall through to re-fetch
```

---

## Anti-Patterns

### WARNING: Trusting LLM Output Types

**The Problem:**

```python
# BAD - assumes LLM returns correct type
class Response(BaseModel):
    count: int  # No validator - LLM might return "five"
```

**Why This Breaks:**
1. LLM might return `"3"` (string) instead of `3` (int)
2. LLM might return `"three"` or `3.0`
3. Validation fails, losing the entire response

**The Fix:**

```python
@field_validator("count", mode="before")
@classmethod
def coerce_count(cls, v: Any) -> int:
    try:
        return int(float(v))  # Handle "3.0" and "3"
    except (TypeError, ValueError):
        return 0
```

### WARNING: No Fallback for Literal Fields

**The Problem:**

```python
# BAD - no fallback if LLM returns unexpected value
@field_validator("status", mode="before")
@classmethod
def validate_status(cls, v: Any) -> str:
    v = str(v).strip().lower()
    if v not in ("active", "inactive"):
        raise ValueError(f"Invalid status: {v}")  # Fails entire response!
    return v
```

**The Fix:**

```python
@field_validator("status", mode="before")
@classmethod
def coerce_status(cls, v: Any) -> str:
    v = str(v).strip().lower()
    return v if v in ("active", "inactive") else "inactive"  # Safe default
```

---

## Integration with Pydantic

See the **pydantic** skill for:
- `BaseModel` configuration
- `Field()` constraints (`ge`, `le`, `min_length`)
- `model_dump()` and `model_dump_json()` usage
- Nested model validation