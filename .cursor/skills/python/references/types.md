# Python Types Reference

## Contents
- Type Annotation Style
- Literal Types
- Optional and Union
- Collection Types
- Pydantic Field Constraints
- Type Aliases

---

## Type Annotation Style

### Always Use Future Annotations

```python
from __future__ import annotations  # Line 1 or after docstring

from typing import Dict, List, Optional, Tuple, Any
```

This enables forward references and postponed evaluation.

### Function Signatures

```python
def analyze_complaints(
    complaints: List[str],
    model: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Analyze complaints and return results with metrics."""
    ...
```

---

## Literal Types

### Constrained String Values

```python
from typing import Literal

# Use Literal instead of str for known value sets
AgeValue = Literal["young", "old"]
GeoValue = Literal["urban", "suburban", "rural"]
TierValue = Literal[0, 1, 2]  # Also works with ints

class PersonaResponse(BaseModel):
    age: Literal["young", "old"]
    geographic: Literal["urban", "suburban", "rural"]
    tier: Literal[0, 1, 2]
```

### WARNING: Using Plain str for Enums

**The Problem:**

```python
# BAD - No type checking, typos silently pass
def process(tier: str) -> None:
    if tier == "hgih":  # Typo not caught
        ...
```

**The Fix:**

```python
# GOOD - Literal catches typos at type-check time
from typing import Literal

Tier = Literal["low", "medium", "high"]

def process(tier: Tier) -> None:
    ...
```

---

## Optional and Union

### Optional for Nullable

```python
# GOOD - Clear nullable intent
from typing import Optional

def get_cache(key: str) -> Optional[str]:
    """Returns cached value or None if not found."""
    ...

# Equivalent but less readable
from typing import Union
def get_cache(key: str) -> Union[str, None]:
    ...
```

### Default None Pattern

```python
def process(
    data: List[str],
    timeout: Optional[int] = None,  # Explicit Optional
) -> Dict[str, Any]:
    timeout = timeout or 30  # Apply default
    ...
```

---

## Collection Types

### Use typing Module (Python 3.10 Compatibility)

```python
# GOOD - Works on Python 3.9+
from typing import Dict, List, Tuple, Set

def func(items: List[str]) -> Dict[str, int]:
    ...

# AVOID in this codebase - 3.10+ syntax
def func(items: list[str]) -> dict[str, int]:
    ...
```

### Complex Nested Types

```python
from typing import Dict, List, Any, Tuple

# Response mapping by model and complaint
ResponseMap = Dict[str, Dict[int, List[Any]]]

# Batch results with metadata
BatchResult = Tuple[List[Dict[str, Any]], Dict[str, float]]
```

---

## Pydantic Field Constraints

See the **pydantic** skill for full details.

### Numeric Bounds

```python
from pydantic import BaseModel, Field

class Response(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    tier: int = Field(ge=0, le=2)
```

### Factory Defaults

```python
from pydantic import BaseModel, Field
from typing import List

class Response(BaseModel):
    markers: List[str] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=lambda: [0.0] * 10)
```

### Field Validators

```python
from pydantic import BaseModel, field_validator

class PersonaResponse(BaseModel):
    age: str

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v):
        if isinstance(v, str):
            v = v.lower().strip()
            if v in ("young", "younger", "25-35"):
                return "young"
            if v in ("old", "older", "elderly", "65+"):
                return "old"
        return v
```

---

## Type Aliases

### Create Readable Aliases

```python
from typing import Dict, List, Any

# Domain-specific aliases
ComplaintId = int
ModelName = str
PersonaGrid = List[Dict[str, str]]
AnalysisResult = Dict[str, Any]

def run_analysis(
    complaints: Dict[ComplaintId, str],
    model: ModelName,
) -> AnalysisResult:
    ...
```

### WARNING: Any Everywhere

**The Problem:**

```python
# BAD - Defeats type checking
def process(data: Any) -> Any:
    return data["key"]["nested"]  # No type safety
```

**The Fix:**

```python
# GOOD - Specific types
from typing import Dict, List

def process(data: Dict[str, List[str]]) -> List[str]:
    return data["key"]
```