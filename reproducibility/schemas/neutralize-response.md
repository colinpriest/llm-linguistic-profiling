# NeutralizeResponse

**Source:** `src/schemas.py` | **Task:** `neutralize`

Removes demographic and stylistic markers from a complaint narrative, producing a "language-neutralized" version. Used in Step 6 of the experiment to test whether removing linguistic cues reduces bias in remedy decisions.

## Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `neutralized_complaint` | string | min length 1 | The complaint text with demographic markers removed |
| `markers_removed` | array of strings | — | List of specific demographic or stylistic markers that were identified and removed |

## Example Response

```json
{
  "neutralized_complaint": "An unauthorized charge of $450 appeared on the credit card account on January 15, 2024. A dispute was filed on January 18, 2024. The bank has not resolved the dispute after 60 days despite multiple follow-up contacts.",
  "markers_removed": [
    "Informal colloquial expressions ('y'all', 'gonna')",
    "Age-indicating references ('back in my day')",
    "Socioeconomic indicators ('can barely afford')",
    "Emotional language ('absolutely furious', 'devastating')"
  ]
}
```

## Validation and Coercion Rules

- **neutralized_complaint:** Must be at least 1 character. No coercion applied — the LLM must return a non-empty string.
- **markers_removed:** Defaults to an empty list if not provided.
