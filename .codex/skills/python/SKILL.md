---
name: python
description: |
  Manages Python 3.10+ runtime, type hints, and scientific computing patterns.
  Use when: writing or modifying Python files, adding type annotations, implementing concurrency, handling errors, or working with dataclasses/Pydantic models.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# Python Skill

Scientific computing Python with strong typing. This codebase uses `from __future__ import annotations` everywhere, explicit typing from the `typing` module (not Python 3.10+ union syntax), and a strict import order.

## Quick Start

### Type-Annotated Functions

```python
from __future__ import annotations
from typing import Dict, List, Optional, Any

def process_complaints(
    complaints: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """Process complaints and return counts by category."""
    config = config or {}
    # ...
```

### Pydantic vs Dataclass Decision

```python
# Pydantic: LLM responses needing validation/coercion
class PersonaEstimateResponse(BaseModel):
    age: Literal["young", "old"]
    confidence: float = Field(ge=0.0, le=1.0)

# Dataclass: Internal containers without validation overhead
@dataclass
class Complaint:
    case_id: str
    narrative: str
    monetary_severity: str = "unknown"
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Type hints | All public functions | `def func(x: List[str]) -> Optional[Dict]` |
| Literal types | Constrained values | `Literal["young", "old"]` |
| Future annotations | Always first | `from __future__ import annotations` |
| Private methods | Underscore prefix | `def _internal_method()` |
| Constants | SCREAMING_SNAKE | `MAX_RETRIES = 3` |

## Import Order

```python
from __future__ import annotations  # 1. Always first

import logging                      # 2. Standard library
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd                 # 3. Third-party
from pydantic import BaseModel

from src.schemas import Complaint   # 4. Local imports
```

## Common Patterns

### Thread-Safe State

```python
from threading import Lock

class Client:
    def __init__(self):
        self._stats_lock = Lock()
        self._success_count = 0

    def record_success(self):
        with self._stats_lock:
            self._success_count += 1
```

### Lock-Free File Caching

```python
import os
import uuid
from pathlib import Path

def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)  # Atomic on Windows & POSIX
```

## See Also

- [patterns](references/patterns.md) - Idioms, concurrency, error handling
- [types](references/types.md) - Type hints, Literal, generics
- [modules](references/modules.md) - Import conventions, lazy loading
- [errors](references/errors.md) - Exception handling, retry patterns

## Related Skills

- **pydantic** - Model validation for LLM responses
- **pandas** - DataFrame operations
- **numpy** - Numerical computing