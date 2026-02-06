---
name: openai
description: |
  Configures OpenAI API client with instructor validation, threading, caching, and retry logic.
  Use when: making LLM calls, configuring model parameters, handling structured responses, implementing batch processing, or troubleshooting API errors.
allowed-tools: Read, Edit, Write, Glob, Grep, Bash
---

# OpenAI Skill

This codebase wraps the OpenAI API with `instructor` for guaranteed Pydantic-validated JSON responses. The `LLMClient` class handles threading (100 workers), disk caching, exponential backoff retries, and a circuit breaker for batch operations.

## Quick Start

### Single Validated Call

```python
from src.llm_client import LLMClient
from src.cache import ResponseCache
from src.schemas import LLMConfig, LLMRequest, RemedyDecisionResponse

config = LLMConfig(
    api_key=os.getenv("OPENAI_API_KEY"),
    models=["gpt-4o-mini"],
    temperature=0.0,
    max_tokens=16384,
    timeout=120,
    max_retries=3,
)
client = LLMClient(config, ResponseCache("./cache"), max_workers=100)

request = LLMRequest(
    request_id=LLMRequest.new_id(),
    case_id="complaint_123",
    model="gpt-4o-mini",
    task="remedy_decision",
    prompt_system=REMEDY_DECISION_SYSTEM,
    prompt_user=REMEDY_DECISION_USER.format(narrative=text),
    expected_schema=RemedyDecisionResponse,
)
response = client.call_with_cache(request, narrative=text)
```

### Batch Processing with Circuit Breaker

```python
responses = client.execute_batch(
    requests=request_list,
    narratives=narrative_list,
    desc="Remedy decisions",
)
# Circuit breaker trips if >50% of first 10 requests fail
```

## Key Concepts

| Concept | Usage | Example |
|---------|-------|---------|
| Reasoning models | Use `max_completion_tokens` not `max_tokens` | `gpt-5-mini`, `o1`, `o3` |
| Instructor | Automatic JSON extraction + Pydantic validation | `response_model=YourSchema` |
| Cache key | Blake2b hash of `(model, task, narrative, prompt_version)` | Invalidates on prompt change |
| Circuit breaker | Aborts batch if >50% early failures | First 10 requests |

## Common Patterns

### Defining Response Schemas

See the **pydantic** skill for validation. See the **instructor** skill for LLM integration.

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

### Handling Reasoning Models

```python
# Reasoning models (gpt-5, o1, o3, o4) need different params
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

if request.model.startswith(_REASONING_PREFIXES):
    kwargs["max_completion_tokens"] = max(config.max_tokens, 32_000)
else:
    kwargs["max_tokens"] = config.max_tokens
    kwargs["temperature"] = config.temperature
```

## See Also

- [patterns](references/patterns.md) - Client configuration, error handling, caching
- [workflows](references/workflows.md) - Batch processing, prompt versioning

## Related Skills

- **pydantic** - Response schema validation with field coercion
- **instructor** - Structured JSON extraction from LLM responses
- **python** - Async patterns, threading, type hints