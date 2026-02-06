# Python Patterns Reference

## Contents
- Concurrency Patterns
- Error Handling Conventions
- Lazy Loading
- Property Pattern
- Loop Patterns

---

## Concurrency Patterns

### ThreadPoolExecutor with FIRST_COMPLETED

```python
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

def batch_process(requests: List[Request]) -> List[Response]:
    pool = ThreadPoolExecutor(max_workers=100)
    try:
        future_to_idx = {
            pool.submit(process_one, req): i
            for i, req in enumerate(requests)
        }
        pending = set(future_to_idx.keys())
        results = [None] * len(requests)

        while pending:
            done, pending = wait(pending, timeout=300, return_when=FIRST_COMPLETED)
            for future in done:
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
    finally:
        pool.shutdown(wait=True)
```

### Circuit Breaker Pattern

Abort batch if early requests fail at high rate:

```python
CIRCUIT_BREAKER_THRESHOLD = 0.5
CIRCUIT_BREAKER_WINDOW = 10

completed, failed = 0, 0
for future in as_completed(futures):
    completed += 1
    if future.exception():
        failed += 1
    if completed == CIRCUIT_BREAKER_WINDOW:
        if failed > CIRCUIT_BREAKER_WINDOW * CIRCUIT_BREAKER_THRESHOLD:
            raise RuntimeError("Circuit breaker tripped: >50% failures")
```

### WARNING: Shared Mutable State Without Locks

**The Problem:**

```python
# BAD - Race condition
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1  # Not atomic!
```

**Why This Breaks:**
1. `+=` is read-modify-write, not atomic
2. Threads interleave, losing increments
3. Final count will be lower than expected

**The Fix:**

```python
# GOOD - Thread-safe with Lock
from threading import Lock

class Counter:
    def __init__(self):
        self._lock = Lock()
        self._count = 0

    def increment(self):
        with self._lock:
            self._count += 1
```

---

## Error Handling Conventions

### Distinguish Retryable vs Non-Retryable

```python
from openai import BadRequestError, RateLimitError

try:
    response = client.chat.completions.create(...)
except BadRequestError as e:
    # Non-retryable: invalid request, won't succeed on retry
    raise RuntimeError(f"Fatal: {e}") from e
except RateLimitError:
    # Retryable: wait and try again
    time.sleep(backoff_delay)
    continue
```

### Retry with Exponential Backoff

```python
MAX_ATTEMPTS = 3
BASE_DELAY = 1.0

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        return make_request()
    except RetryableError:
        if attempt < MAX_ATTEMPTS:
            time.sleep(BASE_DELAY * (2 ** (attempt - 1)))
            continue
        raise
```

### WARNING: Bare Except Clauses

**The Problem:**

```python
# BAD - Catches KeyboardInterrupt, SystemExit
try:
    process_data()
except:
    pass
```

**The Fix:**

```python
# GOOD - Catch specific exceptions
try:
    process_data()
except (ValueError, IOError) as e:
    logger.warning("Processing failed: %s", e)
```

---

## Lazy Loading

### Global Module Cache

```python
_ft_model = None

def _get_fasttext_model():
    global _ft_model
    if _ft_model is None:
        import fasttext
        model_path = Path("data/lid.176.bin")
        _ft_model = fasttext.load_model(str(model_path))
    return _ft_model
```

### Graceful Fallback

```python
try:
    df["lang"] = detect_with_fasttext(texts)
except Exception as e:
    logger.warning("fasttext unavailable (%s), using fallback", e)
    df["lang"] = detect_with_langid(texts)
```

---

## Property Pattern

Use `@property` for computed/cached values:

```python
class ResponseCache:
    def __init__(self):
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

---

## Loop Patterns

### Early Return Guard Clauses

```python
# GOOD - Flat structure
def process(item: Optional[Item]) -> Result:
    if item is None:
        return Result.empty()
    if not item.is_valid:
        return Result.invalid()
    return do_processing(item)
```

### List Comprehension for Cartesian Product

```python
AGES = ["young", "old"]
GEOS = ["urban", "suburban", "rural"]

personas = [
    {"age": a, "geo": g}
    for a in AGES
    for g in GEOS
]
```

### Dynamic Method Dispatch

```python
# Execute steps 1-7 dynamically
for step in range(1, 8):
    if step in skip_steps:
        continue
    method = getattr(self, f"_step_{step}")
    method()
```