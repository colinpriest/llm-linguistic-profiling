# PersonaRewriteResponse

**Source:** `src/schemas.py` | **Task:** `persona_rewrite`

Rewrites a complaint narrative into the linguistic style of a specified demographic persona while preserving the factual content. This is the core of the matched-guise experimental design: the same complaint is rewritten into all 36 persona styles to test whether linguistic style alone affects remedy decisions.

## Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `rewritten_complaint` | string | min length 1 | The complaint text rewritten in the target persona's linguistic style |
| `style_changes_made` | array of strings | — | List of specific stylistic modifications applied during rewriting |

## Example Response

```json
{
  "rewritten_complaint": "I have been a loyal customer of this bank for over forty years, and I am deeply troubled by what has transpired with my checking account. On January 15th, I noticed three charges I did not authorize...",
  "style_changes_made": [
    "Added formal register consistent with older educated speaker",
    "Included reference to long-term banking relationship",
    "Used measured, restrained emotional tone",
    "Maintained all original factual claims and dollar amounts"
  ]
}
```

## Validation and Coercion Rules

- **rewritten_complaint:** Must be at least 1 character. No coercion applied — the LLM must return a non-empty string.
- **style_changes_made:** Defaults to an empty list if not provided.
