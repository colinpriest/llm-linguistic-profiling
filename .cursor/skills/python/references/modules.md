# Python Modules Reference

## Contents
- Import Order Convention
- Absolute vs Relative Imports
- Lazy Module Loading
- Module-Level Initialization
- Circular Import Prevention

---

## Import Order Convention

Strict four-section order with blank lines between:

```python
"""Module docstring."""
from __future__ import annotations  # 1. Future (always first)

import json                          # 2. Standard library
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np                   # 3. Third-party
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.cache import ResponseCache  # 4. Local imports
from src.schemas import Complaint, LLMRequest
```

### WARNING: Mixed Import Order

**The Problem:**

```python
# BAD - Imports scattered
from src.cache import Cache
import json
from pydantic import BaseModel
import logging
from typing import Dict
```

**The Fix:**

```python
# GOOD - Sorted within sections
from __future__ import annotations

import json
import logging
from typing import Dict

from pydantic import BaseModel

from src.cache import Cache
```

---

## Absolute vs Relative Imports

### Always Use Absolute Imports

```python
# GOOD - Explicit, clear origin
from src.schemas import Complaint, LLMRequest
from src.llm_client import LLMClient
from src.cache import ResponseCache

# AVOID - Relative imports
from .schemas import Complaint      # Don't use
from ..utils import helper          # Don't use
```

**Why absolute imports:**
1. Clear where modules come from
2. Works regardless of how script is invoked
3. Better IDE support and refactoring

---

## Lazy Module Loading

### Pattern: Global Cache with None Check

```python
_expensive_model = None

def _get_model():
    """Load model lazily on first access."""
    global _expensive_model
    if _expensive_model is None:
        import expensive_library
        _expensive_model = expensive_library.load("large_model.bin")
    return _expensive_model
```

### When to Use Lazy Loading

- Heavy imports (ML models, large libraries)
- Optional dependencies (graceful degradation)
- Startup time optimization

```python
def detect_language(text: str) -> str:
    """Detect language, loading fasttext lazily."""
    try:
        model = _get_fasttext_model()
        return model.predict(text)[0][0]
    except ImportError:
        # Fallback if fasttext not installed
        import langid
        return langid.classify(text)[0]
```

---

## Module-Level Initialization

### Logger Setup

```python
"""Module docstring."""
from __future__ import annotations

import logging
# ... other imports ...

logger = logging.getLogger(__name__)  # After imports, before classes

class MyClass:
    ...
```

### Constants

```python
# Module-level constants (SCREAMING_SNAKE_CASE)
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60
SUPPORTED_MODELS = ["gpt-4o-mini", "gpt-5-mini"]

# Computed constants
PERSONA_GRID = generate_persona_grid()
```

### WARNING: Complex Logic at Module Level

**The Problem:**

```python
# BAD - Runs on import, hard to test
API_KEY = os.environ["API_KEY"]  # Crashes if missing
client = OpenAI(api_key=API_KEY)  # Side effect on import
```

**The Fix:**

```python
# GOOD - Defer to function/class
def get_client() -> OpenAI:
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not set")
    return OpenAI(api_key=api_key)
```

---

## Circular Import Prevention

### Problem: Mutual Imports

```python
# src/a.py
from src.b import B
class A:
    def method(self) -> B: ...

# src/b.py
from src.a import A  # Circular!
class B:
    def method(self) -> A: ...
```

### Solution 1: TYPE_CHECKING Guard

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.b import B  # Only imported for type hints

class A:
    def method(self) -> B:  # Works with future annotations
        from src.b import B  # Runtime import inside method
        return B()
```

### Solution 2: Restructure

Move shared types to a third module (`src/types.py` or `src/schemas.py`):

```python
# src/schemas.py
class A: ...
class B: ...

# src/module_a.py
from src.schemas import A, B

# src/module_b.py
from src.schemas import A, B
```

---

## Conditional Imports

### Platform-Specific

```python
import sys

if sys.platform == "win32":
    from ctypes import windll
else:
    windll = None
```

### Version-Specific

```python
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
```