# Pydantic Workflows Reference

## Contents
- Instructor Integration Workflow
- Cache Integration Workflow
- Validation Error Handling
- Adding New LLM Response Schemas
- Testing Validators

---

## Instructor Integration Workflow

The **instructor** library wraps OpenAI's client to extract JSON and validate against Pydantic schemas automatically.

### Setup (src/llm_client.py)

```python
import instructor
from openai import OpenAI

raw_client = OpenAI(api_key=config.api_key)
client = instructor.from_openai(raw_client)
```

### Making Requests

```python
validated_obj = client.chat.completions.create(
    model=request.model,
    response_model=request.expected_schema,  # Pydantic class
    max_retries=0,  # Handle retries externally
    messages=[
        {"role": "system", "content": request.prompt_system},
        {"role": "user", "content": request.prompt_user},
    ],
)
# validated_obj is a Pydantic instance, not a dict
```

### Reasoning Model Handling

Reasoning models (GPT-5, o1, o3, o4) use different API parameters:

```python
is_reasoning = request.model.startswith(("gpt-5", "o1", "o3", "o4"))
if is_reasoning:
    kwargs["max_completion_tokens"] = max(config.max_tokens, 32_000)
else:
    kwargs["max_tokens"] = config.max_tokens
    kwargs["temperature"] = config.temperature
```

---

## Cache Integration Workflow

LLM responses are cached as dicts, then rehydrated to Pydantic instances.

### Store Response

```python
# Convert Pydantic to dict for JSON serialization
self.cache.put(
    request.model,
    request.task,
    narrative,
    response.response_parsed.model_dump()  # dict
)
```

### Retrieve and Rehydrate

```python
cached = self.cache.get(request.model, request.task, narrative)
if cached is not None:
    # Re-instantiate Pydantic model from dict
    validated = request.expected_schema(**cached)
    return LLMResponse(..., response_parsed=validated, ...)
```

---

## Validation Error Handling

### Fallback Validation (src/json_utils.py)

When instructor fails, manual validation catches the error:

```python
from pydantic import ValidationError

def validate_response(
    parsed_json: Dict[str, Any],
    expected_schema: Type[BaseModel],
) -> Tuple[bool, Optional[BaseModel], Optional[str]]:
    try:
        validated = expected_schema(**parsed_json)
        return True, validated, None
    except ValidationError as e:
        return False, None, str(e)
```

### Error Handling Pattern

```python
try:
    validated_obj = client.chat.completions.create(
        response_model=RemedyDecisionResponse,
        ...
    )
except BadRequestError as e:
    # Non-retryable API error
    raise RuntimeError(f"Fatal: {e}")
except Exception as e:
    # Retryable error - log and retry
    logger.warning(f"Attempt failed: {e}")
```

---

## Adding New LLM Response Schemas

Copy this checklist and track progress:
- [ ] Step 1: Define Pydantic model in `src/schemas.py`
- [ ] Step 2: Add `field_validator` for every field that could vary
- [ ] Step 3: Define matching dataclass for internal storage
- [ ] Step 4: Add prompt template to `src/prompts.py`
- [ ] Step 5: Update `PROMPT_VERSIONS` for cache invalidation
- [ ] Step 6: Test with real LLM output variations

### Step 1: Define Model

```python
class NewTaskResponse(BaseModel):
    result: Literal["accept", "reject", "review"]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list)
```

### Step 2: Add Validators

```python
@field_validator("result", mode="before")
@classmethod
def coerce_result(cls, v: Any) -> str:
    v = str(v).strip().lower()
    if v in ("accept", "reject", "review"):
        return v
    if v in ("approved", "yes", "pass"):
        return "accept"
    if v in ("denied", "no", "fail"):
        return "reject"
    return "review"  # default
```

### Step 3: Dataclass for Storage

```python
@dataclass
class NewTaskResult:
    case_id: str
    model: str
    result: str
    confidence: float
    notes: List[str]
```

---

## Testing Validators

### Validation Loop Pattern

1. Run with sample LLM outputs
2. Check logs for coercion warnings
3. If validation fails, add mapping to validator
4. Repeat until all variations handled

### Test Harness

```python
# Test various LLM output formats
test_inputs = [
    {"tier": 1},           # int
    {"tier": "1"},         # string
    {"tier": "Tier 1"},    # verbose
    {"tier": "low"},       # semantic
    {"tier": None},        # missing
]

for inp in test_inputs:
    result = RemedyDecisionResponse(**inp)
    print(f"{inp} -> tier={result.tier}")
```

---

## WARNING: Forgetting Prompt Version Bump

**The Problem:**

You update a Pydantic schema but forget to bump `PROMPT_VERSIONS`, causing stale cached responses to be used.

**Why This Breaks:**
1. Cache key doesn't include schema version
2. Old responses fail validation or have wrong structure
3. Debugging nightmare—code is correct but cache is stale

**The Fix:**

After any schema change, update `src/prompts.py`:

```python
PROMPT_VERSIONS = {
    "persona_estimate": "1.0",
    "remedy_decision": "1.1",  # Bumped after adding field
    "new_task": "1.0",
}
```

The cache key includes prompt version, so bumping invalidates old entries.

---

## Integration Checklist

Copy this checklist when adding a new LLM task:

```
New LLM Task Checklist:
- [ ] Pydantic response model in schemas.py
- [ ] field_validator for each field
- [ ] Dataclass for internal storage
- [ ] Prompt template in prompts.py
- [ ] Prompt version in PROMPT_VERSIONS
- [ ] LLMRequest construction in experiment.py
- [ ] Response parsing and dataclass conversion
- [ ] Manual test with 10+ real outputs