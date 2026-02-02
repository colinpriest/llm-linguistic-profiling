# LLM API Call Reference

## Overview

All OpenAI API calls are routed through `src/llm_client.py` via the `LLMClient` class, which wraps the OpenAI Python SDK with the **Instructor** library (v2) for automatic JSON extraction and Pydantic validation. There are two call patterns:

1. **Batch calls** (`execute_batch`) — concurrent thread-pool execution of many requests with progress bars, circuit breakers, and post-batch retry logic
2. **Single cached calls** (`call_with_cache`) — individual requests used for commentary and executive summary generation

Every call is cached to disk via `src/cache.py` using a Blake2b hash of `(model, task, prompt_version, narrative)`.

---

## Global API Parameters

Defined in `src/schemas.py` → `LLMConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | `0.0` | Deterministic output. Only sent to non-reasoning models. |
| `max_tokens` | `1000` | Maximum completion tokens. Overridden to `32,000` for reasoning models. |
| `timeout` | `60` seconds | Per-request HTTP timeout. |
| `max_retries` | `3` | Maximum attempts per request before failure. |
| `retry_delay` | `1.0` second | Base delay between retries (exponential backoff: `delay × 2^(attempt-1)`). |

### Reasoning model detection

Models with names starting with `gpt-5`, `o1`, `o3`, or `o4` are treated as **reasoning models**. For these:
- `temperature` is **not sent** (API rejects it)
- `max_tokens` is replaced with `max_completion_tokens = max(config.max_tokens, 32_000)` to leave room for internal reasoning tokens
- `max_tokens` parameter is **not sent** (API rejects it)

### HTTP transport configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| Transport retries | `0` | Disabled — `_call_once` handles retries with backoff |
| Connect timeout | `10`s | Fast failure on connection issues |
| Pool timeout | `5`s | Fast failure if no connection available in pool |
| Max connections | `max_workers + 10` | Sized to thread pool |
| Max keepalive connections | `20` | Limits idle sockets |
| OpenAI SDK retries | `0` | Disabled — handled at application level |

---

## Instructor Integration

The raw `OpenAI` client is patched via `instructor.from_openai()`. Every API call uses:

```python
self._client.chat.completions.create(
    model=request.model,
    response_model=request.expected_schema,  # Pydantic class
    max_retries=0,                           # instructor-level retries disabled
    timeout=effective_timeout,
    messages=[
        {"role": "system", "content": request.prompt_system},
        {"role": "user",   "content": request.prompt_user},
    ],
    **kwargs,  # temperature + max_tokens OR max_completion_tokens
)
```

Instructor handles:
- Requesting JSON mode from the API
- Extracting JSON from model responses
- Validating against the Pydantic `response_model`
- Returning a fully typed Pydantic object

The `json_utils.py` module provides **fallback** JSON repair utilities (used if Instructor's built-in extraction fails):

| Strategy | What it fixes |
|----------|--------------|
| `trailing_comma_removal` | Removes trailing commas before `}` or `]` |
| `single_quote_replacement` | Replaces `'` with `"` |
| `unquoted_key_fix` | Adds quotes around bare keys |
| `bracket_closing` | Closes unclosed `{` and `[` brackets |
| `newline_escape` | Escapes raw newlines inside JSON strings |
| `control_char_removal` | Strips non-printable characters |

Repairs are attempted individually first, then cumulatively if no single fix works.

---

## Caching

**Source:** `src/cache.py` → `ResponseCache`

| Property | Detail |
|----------|--------|
| Storage | File-based, one JSON file per response in `./cache/{task}/{hash}.json` |
| Key | Blake2b-16 hash of `"{model}|{task}|{prompt_version}|{narrative}"` |
| Invalidation | Automatic when `PROMPT_VERSIONS` in `src/prompts.py` changes |
| Thread safety | `threading.Lock` around reads and writes |
| Atomic writes | Write to `.tmp` file, then `os.replace()` |
| Task directories | `persona_estimate`, `remedy_decision`, `persona_rewrite`, `neutralize`, `dollar_extract` |
| Commentary/exec caching | Keyed by `variant` field (e.g. `step3_commentary`, `exec_linguistic_differences`) |

Cache hits return `attempt_number=0` in the `LLMResponse`. Invalid cache entries are silently discarded and the API is called.

---

## Call-by-Call Reference

### 1. Dollar Amount Extraction

| Property | Value |
|----------|-------|
| **Step** | Pre-step (runs once per complaint) |
| **Model** | `gpt-4o-mini` |
| **Task name** | `dollar_extract` |
| **Prompt version** | `1.1` |
| **System prompt** | `DOLLAR_EXTRACT_SYSTEM` from `src/prompts.py` |
| **User prompt** | `DOLLAR_EXTRACT_USER.format(narrative=...)` from `src/prompts.py` |
| **Pydantic schema** | `DollarAmountResponse` |
| **Temperature** | `0.0` |
| **Max tokens** | `1000` |
| **Execution** | `execute_batch` (concurrent) |
| **Batch retry** | Yes — timeout failures retried up to 3 rounds with escalating timeouts |
| **Failure handling** | Failed responses skipped (null values stored); fatal if >1% fail |

**Input JSON structure:** Single message pair — system + user containing the complaint narrative.

**Output JSON structure:**
```json
{
  "amounts_mentioned": [100.0, 250.0],
  "primary_amount": 250.0,
  "amount_type": "fee",
  "confidence": "high"
}
```

**Field validators:**
- `amount_type`: Coerces synonyms (e.g. `"refund"` → `"payment"`, `"charge"` → `"fee"`, `"fraud"` → `"damage"`). Falls back to `"other"`.
- `confidence`: Coerces to `"high"` / `"medium"` / `"low"`. Falls back to `"medium"`.
- `primary_amount`: Takes absolute value. Returns `null` on parse failure.
- `amounts_mentioned`: Filters non-numeric entries. Returns `null` if empty.

---

### 2. Persona Estimation (Step 1)

| Property | Value |
|----------|-------|
| **Step** | Step 1 |
| **Model** | Each model under test (e.g. `gpt-4o-mini`, `gpt-5-mini`) |
| **Task name** | `persona_estimate` |
| **Prompt version** | `1.0` |
| **System prompt** | `PERSONA_ESTIMATE_SYSTEM` from `src/prompts.py` |
| **User prompt** | `PERSONA_ESTIMATE_USER.format(narrative=...)` from `src/prompts.py` |
| **Pydantic schema** | `PersonaEstimateResponse` |
| **Temperature** | `0.0` (non-reasoning) or omitted (reasoning) |
| **Max tokens** | `1000` (non-reasoning) or `32,000` (reasoning) |
| **Execution** | `execute_batch` per model |
| **Batch retry** | Yes — timeout failures retried up to 3 rounds |
| **Failure handling** | Failed responses silently skipped; fatal if >1% fail |

**Output JSON structure:**
```json
{
  "age": "young",
  "geographic": "urban",
  "education": "college",
  "socioeconomic": "middle",
  "gender": "female",
  "esl": "native",
  "confidence": 0.85,
  "linguistic_markers": ["formal vocabulary", "complex sentences"]
}
```

**Field validators:**
- `age`: Coerces `"middle"`, `"middle-aged"`, `"adult"` → `"young"`; `"elderly"`, `"senior"` → `"old"`. Default: `"young"`.
- `geographic`: Default fallback: `"suburban"`.
- `education`: Coerces `"highschool"`, `"hs"`, `"secondary"` → `"high_school"`; `"university"`, `"graduate"`, `"phd"` etc. → `"college"`. Default: `"college"`.
- `socioeconomic`: Coerces `"lower"`, `"poverty"`, `"working_class"` → `"low"`; `"upper"`, `"wealthy"`, `"affluent"` → `"high"`. Default: `"middle"`.
- `gender`: Coerces `"man"`, `"boy"`, `"m"` → `"male"`; `"woman"`, `"girl"`, `"f"` → `"female"`. Default: `"male"`.
- `esl`: Coerces `"native_speaker"`, `"l1"`, `"no"` → `"native"`; `"non-native"`, `"l2"`, `"yes"` → `"esl"`. Default: `"native"`.
- `confidence`: Clamped to `[0.0, 1.0]`. Falls back to `0.5` on parse failure.

**Also used in:** Step 4 (re-estimate on rewritten complaints), Step 6 (re-estimate on neutralised complaints) — same schema and prompts, applied to transformed text.

---

### 3. Baseline Remedy Decision (Step 2)

| Property | Value |
|----------|-------|
| **Step** | Step 2 |
| **Model** | Each model under test |
| **Task name** | `remedy_decision` |
| **Prompt version** | `1.0` |
| **System prompt** | `REMEDY_DECISION_SYSTEM` from `src/prompts.py` |
| **User prompt** | `REMEDY_DECISION_USER.format(narrative=...)` from `src/prompts.py` |
| **Pydantic schema** | `RemedyDecisionResponse` |
| **Variant** | `"baseline"` |
| **Temperature** | `0.0` (non-reasoning) or omitted (reasoning) |
| **Max tokens** | `1000` (non-reasoning) or `32,000` (reasoning) |
| **Execution** | `execute_batch` per model |
| **Batch retry** | Yes — timeout failures retried up to 3 rounds |
| **Failure handling** | Failed responses silently skipped; fatal if >1% fail |

**Output JSON structure:**
```json
{
  "tier": 0,
  "confidence": "confident_no_action",
  "reasoning": "No bank error identified; customer misunderstood fee schedule.",
  "information_needed": null
}
```

**Field validators:**
- `tier`: Coerced to `int`. Values outside `{0, 1, 2}` fall back to `0`.

**Also used in:** Step 5 (on persona-rewritten complaints, variant = persona label), Step 7 (on neutralised complaints, variant = `"neutralized"`).

---

### 4. Persona-Style Rewriting (Step 4)

| Property | Value |
|----------|-------|
| **Step** | Step 4 |
| **Model** | Each model under test |
| **Task name** | `persona_rewrite` |
| **Prompt version** | `1.0` |
| **System prompt** | `PERSONA_REWRITE_SYSTEM` from `src/prompts.py` |
| **User prompt** | `format_persona_rewrite_user(narrative, age, geo, edu, ses)` from `src/prompts.py` |
| **Pydantic schema** | `PersonaRewriteResponse` |
| **Variant** | Persona label (e.g. `"young_urban_college_high"`) |
| **Temperature** | `0.0` (non-reasoning) or omitted (reasoning) |
| **Max tokens** | `1000` (non-reasoning) or `32,000` (reasoning) |
| **Execution** | `execute_batch` per model (36 personas × N complaints per batch) |
| **Batch retry** | Yes — timeout failures retried up to 3 rounds |
| **Failure handling** | Failed responses silently skipped; fatal if >1% fail |

**Output JSON structure:**
```json
{
  "rewritten_complaint": "Full rewritten complaint text...",
  "style_changes_made": ["simplified vocabulary", "added hedging language"]
}
```

**No field validators** — both fields accept raw strings/lists.

---

### 5. Language Neutralisation (Step 6)

| Property | Value |
|----------|-------|
| **Step** | Step 6 |
| **Model** | Each model under test |
| **Task name** | `neutralize` |
| **Prompt version** | `1.0` |
| **System prompt** | `NEUTRALIZE_SYSTEM` from `src/prompts.py` |
| **User prompt** | `NEUTRALIZE_USER.format(narrative=...)` from `src/prompts.py` |
| **Pydantic schema** | `NeutralizeResponse` |
| **Variant** | `"neutralized"` |
| **Temperature** | `0.0` (non-reasoning) or omitted (reasoning) |
| **Max tokens** | `1000` (non-reasoning) or `32,000` (reasoning) |
| **Execution** | `execute_batch` per model |
| **Batch retry** | Yes — timeout failures retried up to 3 rounds |
| **Failure handling** | Checks `resp.response_parsed` only (not `resp.success`); fatal if >1% fail |

**Output JSON structure:**
```json
{
  "neutralized_complaint": "Professionally rewritten complaint text...",
  "markers_removed": ["informal vocabulary", "emotional language", "regional expressions"]
}
```

**No field validators** — both fields accept raw strings/lists.

---

### 6. Cross-Model Commentary (Step 3)

| Property | Value |
|----------|-------|
| **Step** | Step 3 analysis |
| **Model** | `gpt-5.2` |
| **Task name** | `commentary` |
| **Prompt version** | Not versioned in `PROMPT_VERSIONS` (defaults to `1.0`) |
| **System prompt** | Inline in `src/experiment.py` (~line 874–889) |
| **User prompt** | Inline in `src/experiment.py` (~line 864–872) — statistical results + instruction |
| **Pydantic schema** | `CommentaryResponse` |
| **Variant** | `"step3_commentary"` |
| **Temperature** | Omitted (reasoning model) |
| **Max tokens** | `max_completion_tokens = 32,000` (reasoning model) |
| **Execution** | `call_with_cache` (single request) |
| **Timeout** | Default (`60`s) — no override |
| **Retry** | Default (`3` attempts with exponential backoff) |
| **Failure handling** | **Warning only** — logged at WARNING level, experiment continues |

**Output JSON structure:**
```json
{
  "commentary": "The analysis revealed that... [4-8 sentence paragraph]"
}
```

---

### 7. Per-Model Tier/Process Bias Commentary

| Property | Value |
|----------|-------|
| **Step** | Per-model analysis (after bias calculations) |
| **Model** | `gpt-5.2` |
| **Task name** | `commentary` |
| **Prompt version** | Not versioned (defaults to `1.0`) |
| **System prompt** | Inline in `src/experiment.py` (~line 2359–2375) |
| **User prompt** | Inline in `src/experiment.py` (~line 2376–2381) — model-specific results |
| **Pydantic schema** | `CommentaryResponse` |
| **Variant** | `"commentary"` |
| **Temperature** | Omitted (reasoning model) |
| **Max tokens** | `max_completion_tokens = 32,000` (reasoning model) |
| **Execution** | `call_with_cache` (single request) |
| **Timeout** | Default (`60`s) — no override |
| **Retry** | Default (`3` attempts with exponential backoff) |
| **Failure handling** | **Warning only** — logged at WARNING level, returns `None` |

**Output JSON structure:**
```json
{
  "commentary": "Paragraph 1 — TIER BIAS: ... Paragraph 2 — PROCESS BIAS: ..."
}
```

---

### 8. Executive Summary Sections (×7)

| Property | Value |
|----------|-------|
| **Step** | Report generation |
| **Model** | `gpt-5.2` |
| **Task name** | `executive_summary` |
| **Prompt version** | Not versioned (defaults to `1.0`) |
| **System prompt** | Inline in `src/experiment.py` (~line 2858–2880) — includes one-shot example |
| **User prompt** | Inline in `src/experiment.py` (~line 2882–2885) — section-specific statistical data |
| **Pydantic schema** | `CommentaryResponse` |
| **Variant** | `"exec_{section_name}"` (e.g. `exec_linguistic_differences`) |
| **Temperature** | Omitted (reasoning model) |
| **Max tokens** | `max_completion_tokens = 32,000` (reasoning model) |
| **Execution** | `call_with_cache` (single request per section) |
| **Timeout** | Default (`60`s) — no override |
| **Retry** | Default (`3` attempts with exponential backoff) |
| **Failure handling** | **Warning only** — logged at WARNING level, returns `None` |
| **One-shot examples** | Loaded from `one-shot-examples/` directory |

The 7 sections and their variants:

| # | Section | Variant |
|---|---------|---------|
| 1 | Linguistic Differences | `exec_linguistic_differences` |
| 2 | AI Decision-Making | `exec_ai_decision_making` |
| 3 | Correlation vs. Cause | `exec_correlation_vs_cause` |
| 4 | Severity Stakes | `exec_severity_stakes` |
| 5 | Language Neutralisation | `exec_language_neutralisation` |
| 6 | Bias Mitigation | `exec_bias_mitigation` |
| 7 | Conclusion | `exec_conclusion` |

Section 7 (Conclusion) receives the concatenated output of sections 1–6 as input rather than raw statistics.

**Output JSON structure:**
```json
{
  "commentary": "Executive summary text in plain business English..."
}
```

---

## Error Handling Summary

### Per-request errors (`_call_once`)

| Error type | Behaviour | Retryable? |
|------------|-----------|------------|
| `BadRequestError` (HTTP 400) | `RuntimeError` raised immediately — **fatal** | No |
| `insufficient_quota` | `RuntimeError` raised immediately — **fatal** | No |
| Token limit exceeded (variable-input tasks) | Warning logged, request returns `success=False` — **skipped** | No |
| Token limit exceeded (fixed-input tasks on reasoning models) | `RuntimeError` raised — **fatal** | No |
| All other exceptions | Retried with exponential backoff up to `max_retries` | Yes |
| All retries exhausted | Returns `LLMResponse(success=False)` — **soft failure** | — |

### Batch-level errors (`execute_batch` + `_check_batch_results`)

| Mechanism | Detail |
|-----------|--------|
| **Circuit breaker** | If >50% of the first 10 requests fail, the entire batch is cancelled with `RuntimeError` |
| **Future timeout** | If no futures complete within 300s, all pending requests are cancelled with `RuntimeError` |
| **Timeout retry** | Timeout failures are retried up to 3 rounds with escalating timeouts: 2×, 4×, 8× base timeout |
| **Retry execution** | Each retry uses `max_attempts=1` to avoid compounding wait times |
| **Final failure check** | If >1% of requests still fail after retries → `RuntimeError` (fatal). If ≤1% → warning logged, failures skipped |

### Commentary/summary errors (`call_with_cache`)

| Mechanism | Detail |
|-----------|--------|
| **API failure** | Warning logged, returns `None`, experiment continues |
| **Exception** | Caught in try/except, warning logged, returns `None` |
| **Impact** | Non-fatal — missing commentary sections are omitted from the report |

---

## Request/Response Data Flow

```
LLMRequest                      LLMResponse
┌──────────────────┐            ┌──────────────────┐
│ request_id       │            │ request_id       │
│ case_id          │            │ case_id          │
│ model            │            │ model            │
│ task             │            │ task             │
│ prompt_system    │            │ success          │
│ prompt_user      │──────────→ │ response_raw     │
│ expected_schema  │  API call  │ response_parsed  │ ← Pydantic object
│ variant          │  + cache   │ error_message    │
└──────────────────┘            │ latency_ms       │
                                │ attempt_number   │ ← 0 = cache hit
                                │ repair_applied   │
                                └──────────────────┘
```

### Cache flow

```
call_with_cache(request, narrative)
    │
    ├─ cache.get(model, task, narrative) → hit?
    │     ├─ Yes → rebuild Pydantic object → return LLMResponse(attempt=0)
    │     │         └─ Invalid cache entry → fall through to API
    │     └─ No  → call _call_once(request)
    │                 ├─ Success → cache.put(...) → return LLMResponse
    │                 └─ Failure → return LLMResponse(success=False)
    │
    └─ Cache key = blake2b("{model}|{task}|{prompt_version}|{narrative}")
```

---

## Models Used

| Model | Purpose | Call pattern |
|-------|---------|-------------|
| `gpt-4o-mini` | Dollar amount extraction (pre-step) | Batch |
| `gpt-4o-mini` | Primary experiment model (Steps 1–7) | Batch |
| `gpt-5-mini` | Primary experiment model (Steps 1–7) | Batch |
| `gpt-5.2` | Commentary generation (Steps 3, per-model) | Single cached |
| `gpt-5.2` | Executive summary generation (7 sections) | Single cached |

The experiment models are configurable via `LLMConfig.models`. Dollar extraction always uses `gpt-4o-mini`. Commentary and executive summaries always use `gpt-5.2`.
