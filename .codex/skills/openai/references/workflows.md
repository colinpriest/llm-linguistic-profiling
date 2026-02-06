# OpenAI Workflows Reference

## Contents
- Batch Processing Workflow
- Prompt Versioning Workflow
- Adding New LLM Tasks
- Debugging Failed Requests
- Performance Tuning

## Batch Processing Workflow

### Execute Batch with Circuit Breaker

```python
# src/llm_client.py:359-481
responses = client.execute_batch(
    requests=request_list,
    narratives=narrative_list,
    desc="Remedy decisions",
)
```

**Circuit breaker behavior:**
- Window: First 10 requests
- Threshold: >50% failures
- Action: Cancel remaining futures, raise RuntimeError

### Building Request Batches

```python
# Pattern from src/experiment.py
requests = []
narratives = []
for complaint in complaints:
    request = LLMRequest(
        request_id=LLMRequest.new_id(),
        case_id=complaint.case_id,
        model=model_name,
        task="remedy_decision",
        prompt_system=REMEDY_DECISION_SYSTEM,
        prompt_user=REMEDY_DECISION_USER.format(narrative=complaint.narrative),
        expected_schema=RemedyDecisionResponse,
    )
    requests.append(request)
    narratives.append(complaint.narrative)

responses = client.execute_batch(requests, narratives, desc="Step 2: Baseline decisions")
```

### Processing Responses

```python
for response in responses:
    if response.success and response.response_parsed:
        decision = RemedyDecision(
            case_id=response.case_id,
            model=response.model,
            variant="baseline",
            tier=response.response_parsed.tier,
            confidence=response.response_parsed.confidence,
            reasoning=response.response_parsed.reasoning,
            information_needed=response.response_parsed.information_needed,
        )
        decisions.append(decision)
    else:
        logger.warning("Failed: %s - %s", response.case_id, response.error_message)
```

## Prompt Versioning Workflow

### Step 1: Define Version in prompts.py

```python
# src/prompts.py:3-9
PROMPT_VERSIONS = {
    "persona_estimate": "1.0",
    "remedy_decision": "1.0",
    "persona_rewrite": "1.0",
    "neutralize": "1.0",
    "dollar_extract": "1.1",  # Bumped when prompt changed
}
```

### Step 2: Update Prompt Text

```python
# When you modify this prompt, bump the version above
REMEDY_DECISION_SYSTEM = """\
You are a fair and impartial bank complaint assessor...
"""
```

### Step 3: Cache Invalidates Automatically

The cache key includes `prompt_version`, so old cached responses are ignored:

```python
# src/cache.py:136
content = f"{model}|{task}|{prompt_version}|{narrative}"
```

Copy this checklist and track progress:
- [ ] Update prompt text in `src/prompts.py`
- [ ] Bump version in `PROMPT_VERSIONS` dict
- [ ] Test with `--num-complaints 10` to verify new responses
- [ ] Run full experiment when satisfied

## Adding New LLM Tasks

### Step 1: Define Response Schema

See the **pydantic** skill for schema design. See the **instructor** skill for LLM integration.

```python
# src/schemas.py
class NewTaskResponse(BaseModel):
    result: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))
```

### Step 2: Add Prompts

```python
# src/prompts.py
PROMPT_VERSIONS["new_task"] = "1.0"

NEW_TASK_SYSTEM = "You are a..."
NEW_TASK_USER = "Analyze: {narrative}"
```

### Step 3: Register Cache Directory

```python
# src/cache.py:29
TASK_DIRS = [..., "new_task"]
```

### Step 4: Build Requests

```python
request = LLMRequest(
    request_id=LLMRequest.new_id(),
    case_id=case_id,
    model=model,
    task="new_task",
    prompt_system=NEW_TASK_SYSTEM,
    prompt_user=NEW_TASK_USER.format(narrative=text),
    expected_schema=NewTaskResponse,
)
```

Copy this checklist and track progress:
- [ ] Create Pydantic response model with validators
- [ ] Add prompts and version to `src/prompts.py`
- [ ] Add task to `ResponseCache.TASK_DIRS`
- [ ] Create LLMRequest with correct `expected_schema`
- [ ] Test with small batch before full run

## Debugging Failed Requests

### Check Error Messages

```python
for response in responses:
    if not response.success:
        print(f"{response.case_id}: {response.error_message}")
        print(f"  Task: {response.task}, Attempt: {response.attempt_number}")
```

### Common Error Patterns

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `insufficient_quota` | Billing issue | Check OpenAI billing dashboard |
| `max_tokens` / `length limit` | Output too long | Increase `max_tokens` or simplify prompt |
| `BadRequestError` | Invalid model params | Check reasoning model detection |
| `Timed out after 300s` | Network or API issues | Reduce batch size, check connectivity |

### Inspect Cache

```bash
# View cache structure
ls ./cache/

# Check specific task
ls ./cache/remedy_decision/ | wc -l

# View a cached response
cat ./cache/remedy_decision/abc123def456.json | jq .
```

### Force Re-fetch

```python
# Delete specific cache entry
cache.delete(model="gpt-4o-mini", task="remedy_decision", narrative=text)

# Clear entire cache
cache.clear()
```

## Performance Tuning

### Thread Pool Sizing

```python
# Default: 100 workers — matches OpenAI's rate limits
LLMClient(config, cache, max_workers=100)

# For lower-tier API plans, reduce workers
LLMClient(config, cache, max_workers=20)
```

### Connection Pool Settings

```python
# src/llm_client.py:61-64
limits=httpx.Limits(
    max_connections=max_workers + 10,  # Headroom for retries
    max_keepalive_connections=20,      # Avoid stale sockets
)
```

### Timeout Tuning

```python
# Reasoning models need longer timeouts
config = LLMConfig(
    timeout=120,     # 2 min for regular models
    # Or pass per-request:
)
response = client.call_with_cache(request, narrative, timeout_override=240)
```

### Batch Size vs Memory

For large experiments, process in chunks:

```python
CHUNK_SIZE = 1000
for i in range(0, len(all_requests), CHUNK_SIZE):
    chunk_requests = all_requests[i:i + CHUNK_SIZE]
    chunk_narratives = all_narratives[i:i + CHUNK_SIZE]
    responses = client.execute_batch(chunk_requests, chunk_narratives)
    # Process responses immediately to free memory
```

Iterate-until-pass pattern for batch validation:
1. Run batch with `execute_batch()`
2. Check `response.success` for each response
3. Collect failed case_ids
4. If failures > threshold, investigate error patterns
5. Adjust prompts/schemas, bump version, re-run failed items only