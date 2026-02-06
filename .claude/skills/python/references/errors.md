# Python Error Handling Reference

## Contents
- Exception Hierarchy
- EAFP vs LBYL
- Retry Patterns
- Logging Errors
- Custom Exceptions
- Graceful Degradation

---

## Exception Hierarchy

### Catch Specific Exceptions

```python
# GOOD - Catch what you expect
try:
    data = json.loads(response_text)
except json.JSONDecodeError as e:
    logger.warning("Invalid JSON: %s", e)
    return None
```

### Order Matters (Most Specific First)

```python
from openai import BadRequestError, RateLimitError, APIError

try:
    response = client.chat.completions.create(...)
except BadRequestError as e:
    # Most specific - invalid request
    raise RuntimeError(f"Invalid request: {e}") from e
except RateLimitError:
    # Rate limited - retry
    time.sleep(60)
except APIError as e:
    # Catch-all for other API errors
    logger.error("API error: %s", e)
    raise
```

---

## EAFP vs LBYL

### EAFP (Easier to Ask Forgiveness than Permission)

Preferred Python style - try the operation, handle failure:

```python
# GOOD - EAFP style
def delete_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass  # Already gone, that's fine
```

### When LBYL Makes Sense

Check first when the check is cheap and failure is expensive:

```python
# LBYL appropriate here - avoid expensive operation
if not api_key:
    raise ValueError("API key required")
# Expensive operation follows...
client = OpenAI(api_key=api_key)
```

### WARNING: TOCTOU Race Condition

**The Problem:**

```python
# BAD - Race condition between check and use
if path.exists():
    data = path.read_text()  # File could be deleted between check and read!
```

**The Fix:**

```python
# GOOD - EAFP avoids race
try:
    data = path.read_text()
except FileNotFoundError:
    data = None
```

---

## Retry Patterns

### Exponential Backoff

```python
import time
from typing import TypeVar, Callable

T = TypeVar("T")

def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except (IOError, TimeoutError) as e:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed, retrying in %.1fs: %s", attempt, delay, e)
            time.sleep(delay)
```

### Distinguish Retryable Errors

```python
RETRYABLE_ERRORS = (RateLimitError, TimeoutError, ConnectionError)
FATAL_ERRORS = (BadRequestError, AuthenticationError)

try:
    return make_request()
except FATAL_ERRORS as e:
    raise RuntimeError(f"Non-retryable error: {e}") from e
except RETRYABLE_ERRORS:
    # Retry logic...
    pass
```

---

## Logging Errors

### Use %-Formatting (Not f-strings)

```python
# GOOD - Lazy evaluation, better for logging frameworks
logger.error("Failed to process %s: %s", item_id, error)
logger.warning("Cache miss for key %r", cache_key)

# AVOID - Eager evaluation even when log level disabled
logger.error(f"Failed to process {item_id}: {error}")
```

### Include Context

```python
try:
    process_complaint(complaint_id, narrative)
except Exception as e:
    logger.error(
        "Failed processing complaint %d (len=%d): %s",
        complaint_id,
        len(narrative),
        e,
    )
    raise
```

### Log Levels

| Level | Use For |
|-------|---------|
| DEBUG | Detailed flow, variable values |
| INFO | Progress, successful operations |
| WARNING | Recoverable issues, fallbacks |
| ERROR | Failures requiring attention |

---

## Custom Exceptions

### Domain-Specific Exceptions

```python
class ExperimentError(Exception):
    """Base exception for experiment errors."""
    pass

class CacheCorruptError(ExperimentError):
    """Cache entry is corrupted."""
    def __init__(self, path: Path, reason: str):
        self.path = path
        super().__init__(f"Corrupt cache at {path}: {reason}")

class CircuitBreakerError(ExperimentError):
    """Too many failures in batch."""
    def __init__(self, failed: int, total: int):
        self.failed = failed
        self.total = total
        super().__init__(f"Circuit breaker: {failed}/{total} failures")
```

---

## Graceful Degradation

### Fallback Chain

```python
def detect_language(text: str) -> str:
    """Detect language with fallback chain."""
    # Try primary method
    try:
        return _detect_with_fasttext(text)
    except Exception as e:
        logger.warning("fasttext failed: %s", e)

    # Fallback to secondary
    try:
        return _detect_with_langid(text)
    except Exception as e:
        logger.warning("langid failed: %s", e)

    # Final fallback
    return "unknown"
```

### WARNING: Silent Failures

**The Problem:**

```python
# BAD - Errors disappear
try:
    important_operation()
except Exception:
    pass
```

**The Fix:**

```python
# GOOD - Log and handle appropriately
try:
    important_operation()
except ExpectedError as e:
    logger.warning("Expected failure, using default: %s", e)
    return default_value
except Exception as e:
    logger.error("Unexpected error: %s", e)
    raise  # Re-raise unexpected errors
```

---

## Exception Chaining

### Preserve Original Traceback

```python
# GOOD - Chain with 'from'
try:
    data = json.loads(text)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid config format") from e

# Result: Shows both ValueError AND original JSONDecodeError
```

### Suppress Chain When Appropriate

```python
# Explicit suppression when original error is irrelevant
try:
    return cached_lookup(key)
except KeyError:
    raise CacheMiss(key) from None  # Suppress KeyError chain
```