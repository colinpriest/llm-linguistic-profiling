# OpenAI Patterns Reference

## Contents
- Client Configuration
- Response Schema Design
- Error Handling Patterns
- Caching Strategy
- Anti-Patterns

## Client Configuration

### Proper HTTP Transport Setup

```python
# src/llm_client.py:47-70
transport = httpx.HTTPTransport(retries=0)  # No transport retries
http_client = httpx.Client(
    transport=transport,
    timeout=httpx.Timeout(
        config.timeout,
        connect=10,
        pool=5,  # Fail fast if no connection in pool
    ),
    limits=httpx.Limits(
        max_connections=max_workers + 10,
        max_keepalive_connections=20,
    ),
)
raw_client = OpenAI(
    api_key=config.api_key,
    http_client=http_client,
    max_retries=0,  # We handle retries ourselves
)
self._client = instructor.from_openai(raw_client)
```

**Why `retries=0`:** Transport-level retries silently multiply the timeout (retries=2 means 3×60s=180s). Handle retries explicitly with backoff instead.

### LLMConfig Dataclass

```python
# src/schemas.py:443-451
@dataclass
class LLMConfig:
    api_key: str
    models: List[str]
    temperature: float = 0.0      # Deterministic for experiments
    max_tokens: int = 1000        # Output limit
    timeout: int = 60             # Per-request timeout
    max_retries: int = 3          # Retry attempts
    retry_delay: float = 1.0      # Initial backoff delay
```

## Response Schema Design

### Coercing LLM Output with Validators

LLMs sometimes return unexpected formats. Use `field_validator` to coerce gracefully. See the **pydantic** skill for details.

```python
# src/schemas.py:121-131
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
    return "young"  # Default fallback
```

### Numeric Clamping

```python
# src/schemas.py:189-196
@field_validator("confidence", mode="before")
@classmethod
def clamp_confidence(cls, v: Any) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, v))
```

## Error Handling Patterns

### Non-Retryable Errors

```python
# src/llm_client.py:164-184
except BadRequestError as e:
    # 400 errors are fatal — bad model params, invalid request
    short_err = str(e).split("\n", 1)[0][:200]
    raise RuntimeError(
        f"Fatal API error for model {request.model!r}: {short_err}"
    ) from e

# Quota exhaustion — never retries
if "insufficient_quota" in err_str:
    raise RuntimeError(
        "OpenAI API quota exceeded. Check billing at "
        "https://platform.openai.com/account/billing"
    ) from e
```

### Token Limit Handling

```python
# src/llm_client.py:188-247
is_token_limit = (
    "max_tokens" in err_str
    or "length limit" in err_str
    or "incomplete" in err_str
)
if is_token_limit:
    # Variable-input tasks can hit limits on outliers — skip this request
    # Fixed-length tasks signal systematic budget problem — abort batch
    _variable_input_tasks = {"neutralize", "rewrite", "remedy", "persona_estimate"}
    if is_reasoning and request.task not in _variable_input_tasks:
        raise RuntimeError(f"Output truncated for {request.model!r}")
```

### Exponential Backoff

```python
# src/llm_client.py:248-254
if attempt < MAX_ATTEMPTS:
    time.sleep(self.config.retry_delay * (2 ** (attempt - 1)))
    continue  # Retry 1s, 2s, 4s...
```

## Caching Strategy

### Cache Key Generation

```python
# src/cache.py:134-137
def _key(self, model: str, task: str, narrative: str) -> str:
    prompt_version = PROMPT_VERSIONS.get(task, "1.0")
    content = f"{model}|{task}|{prompt_version}|{narrative}"
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()
```

**Automatic invalidation:** Bump `PROMPT_VERSIONS[task]` in `src/prompts.py` when you change a prompt.

### Lock-Free Atomic Writes

```python
# src/cache.py:81-96
# Write to unique temp file, then atomic replace
tmp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
tmp_path.write_text(payload, encoding="utf-8")
os.replace(tmp_path, path)  # Atomic on NTFS and POSIX
```

## Anti-Patterns

### WARNING: Using Transport-Level Retries

**The Problem:**

```python
# BAD — hidden timeout multiplication
raw_client = OpenAI(api_key=key, max_retries=3)
```

**Why This Breaks:**
1. With timeout=60s and max_retries=3, worst case is 4×60s = 240s per request
2. You lose visibility into retry attempts
3. Exponential backoff is not customizable

**The Fix:**

```python
# GOOD — explicit retry handling
raw_client = OpenAI(api_key=key, max_retries=0)
# Handle retries in application code with logging and backoff
```

### WARNING: Missing Schema Coercion

**The Problem:**

```python
# BAD — crashes on unexpected LLM output
class Response(BaseModel):
    tier: Literal[0, 1, 2]  # LLM returns "zero" → ValidationError
```

**Why This Breaks:**
1. LLMs return creative formats: "0", "zero", "none", 0.0
2. ValidationError kills the entire batch
3. No graceful degradation

**The Fix:**

```python
# GOOD — coerce with fallback
@field_validator("tier", mode="before")
@classmethod
def coerce_tier(cls, v: Any) -> int:
    try:
        return int(v) if int(v) in (0, 1, 2) else 0
    except (TypeError, ValueError):
        return 0
```

### WARNING: Ignoring Reasoning Model Differences

**The Problem:**

```python
# BAD — reasoning models reject temperature param
response = client.chat.completions.create(
    model="gpt-5-mini",
    temperature=0.7,  # 400 Bad Request
    max_tokens=1000,
)
```

**Why This Breaks:**
1. Reasoning models (gpt-5, o1, o3, o4) use different API parameters
2. They require `max_completion_tokens` instead of `max_tokens`
3. They don't accept `temperature`

**The Fix:**

```python
# GOOD — detect and adjust
if request.model.startswith(("gpt-5", "o1", "o3", "o4")):
    kwargs["max_completion_tokens"] = 32_000
else:
    kwargs["max_tokens"] = 1000
    kwargs["temperature"] = 0.0