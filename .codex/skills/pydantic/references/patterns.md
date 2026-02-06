# Pydantic Patterns Reference

## Contents
- Field Validators and Coercion
- Literal Type Constraints
- Optional Fields with Defaults
- Dataclass vs Pydantic Decision
- Serialization Patterns
- Common Errors and Solutions

---

## Field Validators and Coercion

This codebase uses `field_validator` with `mode="before"` for all LLM response fields. The validator runs before Pydantic's type validation, allowing normalization of messy input.

### Pattern: Normalize String Variations

```python
@field_validator("geographic", mode="before")
@classmethod
def coerce_geographic(cls, v: Any) -> str:
    v = str(v).strip().lower()
    if v in ("urban", "suburban", "rural"):
        return v
    return "suburban"  # default fallback
```

### Pattern: Clamp Numeric Values

```python
@field_validator("confidence", mode="before")
@classmethod
def clamp_confidence(cls, v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))  # Clamp to [0, 1]
```

### Pattern: Map Synonyms to Canonical Values

```python
@field_validator("amount_type", mode="before")
@classmethod
def coerce_amount_type(cls, v: Any) -> Optional[str]:
    if v is None:
        return None
    v = str(v).strip().lower()
    allowed = {"fee", "balance", "payment", "damage", "other"}
    if v in allowed:
        return v
    mapping = {
        "refund": "payment", "charge": "fee",
        "debt": "balance", "fraud": "damage",
    }
    return mapping.get(v, "other")
```

---

## WARNING: Raising ValidationError in Validators

**The Problem:**

```python
# BAD - Raises on unexpected input
@field_validator("tier", mode="before")
@classmethod
def validate_tier(cls, v: Any) -> int:
    if v not in (0, 1, 2):
        raise ValueError(f"Invalid tier: {v}")  # Will crash on "tier": "1"
    return v
```

**Why This Breaks:**
1. LLMs return strings for numbers (`"1"` not `1`)
2. LLMs return variations (`"Tier 1"`, `"low"`, etc.)
3. Production runs fail on outlier responses

**The Fix:**

```python
# GOOD - Coerce and fallback
@field_validator("tier", mode="before")
@classmethod
def coerce_tier(cls, v: Any) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return 0
    return v if v in (0, 1, 2) else 0
```

**When You Might Be Tempted:** When you expect clean, typed data. With LLM outputs, always expect garbage.

---

## Literal Type Constraints

Use `Literal` to constrain fields to a discrete set of values. Combined with `field_validator`, this provides both validation and documentation.

```python
class PersonaEstimateResponse(BaseModel):
    age: Literal["young", "old"]
    geographic: Literal["urban", "suburban", "rural"]
    education: Literal["high_school", "college"]
    socioeconomic: Literal["low", "middle", "high"]
    tier: Literal[0, 1, 2]  # Integer literals work too
```

### DO: Validator Before Literal

```python
# GOOD - Validator coerces BEFORE Literal validates
@field_validator("age", mode="before")
@classmethod
def coerce_age(cls, v: Any) -> str:
    v = str(v).strip().lower()
    if v in ("young", "old"):
        return v
    if v in ("middle", "middle-aged"):
        return "young"
    return "young"
```

### DON'T: Rely on Literal Alone

```python
# BAD - "Young" fails Literal["young", "old"] (case-sensitive)
age: Literal["young", "old"]  # No validator = crashes on "Young"
```

---

## Optional Fields with Defaults

### Pattern: Optional with None Default

```python
class DollarAmountResponse(BaseModel):
    primary_amount: Optional[float] = Field(
        default=None,
        description="The main disputed amount (or null if unclear)",
    )
```

### Pattern: List with default_factory

```python
# GOOD - Use default_factory for mutable defaults
linguistic_markers: List[str] = Field(default_factory=list)
style_changes_made: List[str] = Field(default_factory=list)
```

### WARNING: Mutable Default Arguments

```python
# BAD - Shared list across instances
linguistic_markers: List[str] = []

# GOOD - New list per instance
linguistic_markers: List[str] = Field(default_factory=list)
```

---

## Dataclass vs Pydantic Decision

This codebase uses a clear separation:

| Use Case | Type | Why |
|----------|------|-----|
| LLM response schemas | Pydantic `BaseModel` | Validation, coercion, instructor integration |
| Internal data transfer | `@dataclass` | No validation overhead, faster |

### Example: Pydantic for LLM Response

```python
# src/schemas.py - LLM returns this, needs validation
class RemedyDecisionResponse(BaseModel):
    tier: Literal[0, 1, 2]
    confidence: Literal[...]
```

### Example: Dataclass for Internal Use

```python
# src/schemas.py - Internal container, no validation needed
@dataclass
class RemedyDecision:
    case_id: str
    model: str
    variant: str
    tier: int
    confidence: str
```

---

## Serialization Patterns

### To Dict (for caching)

```python
# Store in cache
response.response_parsed.model_dump()

# Reconstruct from cache
validated = RemedyDecisionResponse(**cached_dict)
```

### To JSON String

```python
raw_json = validated_obj.model_dump_json()
```

### To DataFrame

```python
from dataclasses import asdict
df = pd.DataFrame([asdict(pe) for pe in self.persona_estimates])
```

---

## Common Errors and Solutions

| Error | Cause | Fix |
|-------|-------|-----|
| `validation error for X` | LLM returned unexpected value | Add `field_validator` with coercion |
| `value is not a valid integer` | LLM returned `"1"` not `1` | `int(v)` in validator |
| `Input should be 'x' or 'y'` | Case mismatch on Literal | `.strip().lower()` in validator |