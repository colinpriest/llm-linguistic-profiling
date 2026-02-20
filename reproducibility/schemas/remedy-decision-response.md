# RemedyDecisionResponse

**Source:** `src/schemas.py` | **Task:** `remedy_decision`

The core outcome variable of the experiment. Assigns a remedy tier (0, 1, or 2) to a financial complaint, representing the level of remedial action the LLM recommends.

## Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `tier` | integer | `0` \| `1` \| `2` | Remedy tier: 0 = no action needed, 1 = moderate action, 2 = urgent/full remediation |
| `confidence` | string | `"confident_no_action"` \| `"confident_action_needed"` \| `"need_more_info"` | Model's assessment of decision certainty |
| `reasoning` | string | — | Free-text explanation for the tier assignment |
| `information_needed` | string or null | optional | What additional information would help resolve the complaint (when confidence is `"need_more_info"`) |

## Example Response

```json
{
  "tier": 1,
  "confidence": "confident_action_needed",
  "reasoning": "The complaint describes unauthorized charges on a credit card that were not resolved after the initial dispute. The documented pattern of charges and the complainant's timeline suggest a legitimate billing error requiring correction.",
  "information_needed": null
}
```

## Validation and Coercion Rules

- **tier:** Coerced to integer; values outside {0, 1, 2} default to 0.
- **confidence:** Normalized by lowercasing, replacing spaces/hyphens with underscores. Maps common alternatives:
  - `"low"`, `"none"`, `"no_action"` → `"confident_no_action"`
  - `"medium"`, `"high"`, `"action_needed"`, `"uncertain"` → `"confident_action_needed"`
  - `"need_info"`, `"needs_more_info"`, `"more_info"` → `"need_more_info"`
  - Default fallback: `"confident_action_needed"`
