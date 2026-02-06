# Instructor Workflows Reference

## Contents
- Request Construction Workflow
- Batch Execution Workflow
- Cache Integration Workflow
- Validation Feedback Loop

---

## Request Construction Workflow

### Step 1: Define Schema

Create Pydantic model with validators for LLM output robustness:

```python
class RemedyDecisionResponse(BaseModel):
    tier: Literal[0, 1, 2]
    confidence: Literal["confident_no_action", "confident_action_needed", "need_more_info"]
    reasoning: str = ""
    
    @field_validator("tier", mode="before")
    @classmethod
    def coerce_tier(cls, v: Any) -> int:
        try:
            return int(v) if int(v) in (0, 1, 2) else 0
        except (TypeError, ValueError):
            return 0
```

### Step 2: Create Request Object

```python
from src.schemas import LLMRequest

req = LLMRequest(
    request_id=LLMRequest.new_id(),
    case_id=complaint.case_id,
    model="gpt-4o-mini",
    task="remedy_decision",
    prompt_system=REMEDY_DECISION_SYSTEM,
    prompt_user=REMEDY_DECISION_USER.format(narrative=complaint.narrative),
    expected_schema=RemedyDecisionResponse,
    variant="baseline",
)
```

### Step 3: Execute and Access Results

```python
response = llm_client.call_with_cache(req, complaint.narrative)

if response.success and response.response_parsed:
    p = response.response_parsed
    # Type-safe access to validated fields
    decision = RemedyDecision(
        tier=p.tier,         # Guaranteed Literal[0, 1, 2]
        confidence=p.confidence,
        reasoning=p.reasoning,
    )
```

---

## Batch Execution Workflow

### Concurrent Processing with Circuit Breaker

```python
requests = [
    LLMRequest(
        request_id=LLMRequest.new_id(),
        case_id=c.case_id,
        model=model,
        task="persona_estimate",
        prompt_system=PERSONA_ESTIMATE_SYSTEM,
        prompt_user=PERSONA_ESTIMATE_USER.format(narrative=c.narrative),
        expected_schema=PersonaEstimateResponse,
    )
    for c in complaints
]
narratives = [c.narrative for c in complaints]

responses = llm_client.execute_batch(
    requests=requests,
    narratives=narratives,
    desc="Step 1: Persona Estimation",
)
```

### Processing Batch Results

```python
for resp in responses:
    if resp.success and resp.response_parsed:
        p = resp.response_parsed
        self.persona_estimates.append(PersonaEstimate(
            case_id=resp.case_id,
            model=resp.model,
            age=p.age,
            geographic=p.geographic,
            education=p.education,
            socioeconomic=p.socioeconomic,
        ))
    else:
        logger.warning("Failed: %s - %s", resp.case_id, resp.error_message)
```

---

## Cache Integration Workflow

### How Caching Works

1. Request is hashed by `(model, task, narrative)`
2. On cache hit: dict is re-validated against schema
3. On validation failure: cache entry deleted, request re-fetched
4. On cache miss: call LLM, validate, store dict

```python
def call_with_cache(self, request: LLMRequest, narrative: str) -> LLMResponse:
    cached = self.cache.get(request.model, request.task, narrative)
    if cached is not None:
        try:
            validated = request.expected_schema(**cached)
            return LLMResponse(
                success=True,
                response_parsed=validated,
                attempt_number=0,  # 0 indicates cache hit
            )
        except Exception:
            self.cache.delete(request.model, request.task, narrative)
    
    response = self._call_once(request)
    
    if response.success and response.response_parsed:
        self.cache.put(
            request.model,
            request.task,
            narrative,
            response.response_parsed.model_dump(),
        )
    
    return response
```

### Cache Invalidation

Cache keys include prompt version from `PROMPT_VERSIONS` in `src/prompts.py`. Changing a prompt version automatically invalidates related cache entries.

---

## Validation Feedback Loop

When adding new schemas or modifying validators:

Copy this checklist and track progress:
- [ ] Step 1: Define schema with `Literal` fields and `Field()` constraints
- [ ] Step 2: Add `@field_validator(mode="before")` for every field
- [ ] Step 3: Include safe defaults (never raise in validators)
- [ ] Step 4: Test with edge cases: empty string, None, wrong type, unexpected values
- [ ] Step 5: Run small batch (100 complaints) to verify
- [ ] Step 6: Check logs for validation warnings

### Iterate-Until-Pass Pattern

1. Run experiment with small sample:
   ```bash
   python llm_fairness_experiment.py --num-complaints 100
   ```

2. Check logs for validation issues:
   ```
   WARNING: Invalid cache entry for gpt-4o-mini/persona_estimate
   ```

3. If validators are too strict, add more coercion cases:
   ```python
   # Before: only handles "young" and "old"
   # After: handles "middle-aged", "elderly", "senior", etc.
   ```

4. Clear cache and re-run:
   ```bash
   python llm_fairness_experiment.py --num-complaints 100 --clear-cache
   ```

5. Repeat until no validation warnings appear

---

## Multi-Model Workflow

The experiment runs against multiple models. Each model-task pair has its own cache namespace:

```python
for model in ["gpt-4o-mini", "gpt-5-mini"]:
    requests = [
        LLMRequest(
            model=model,
            task="remedy_decision",
            expected_schema=RemedyDecisionResponse,
            ...
        )
        for c in complaints
    ]
    responses = llm_client.execute_batch(requests, narratives, desc=f"Step 2: {model}")
```

---

## Debugging Validation Failures

### Check Raw Response

```python
if not response.success:
    logger.error("Raw: %s", response.response_raw)
    logger.error("Error: %s", response.error_message)
```

### Validate Manually

```python
from src.schemas import PersonaEstimateResponse

raw_dict = {"age": "middle-aged", "confidence": "0.8"}
try:
    validated = PersonaEstimateResponse(**raw_dict)
    print(validated.age)  # Should coerce to "young"
except Exception as e:
    print(f"Validation failed: {e}")
```

### Add Logging to Validators

```python
@field_validator("age", mode="before")
@classmethod
def coerce_age(cls, v: Any) -> str:
    original = v
    v = str(v).strip().lower()
    if v not in ("young", "old"):
        logger.debug("Coerced age %r -> 'young'", original)
    return v if v in ("young", "old") else "young"