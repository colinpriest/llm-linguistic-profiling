# PersonaEstimateResponse

**Source:** `src/schemas.py` | **Task:** `persona_estimate`

Classifies the perceived demographic persona of a complaint author across 6 dimensions based on linguistic markers in the complaint text.

## Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `age` | string | `"young"` \| `"old"` | Estimated age bracket (young = 25-35, old = 65+) |
| `geographic` | string | `"urban"` \| `"suburban"` \| `"rural"` | Estimated geographic setting |
| `education` | string | `"high_school"` \| `"college"` | Estimated education level |
| `socioeconomic` | string | `"low"` \| `"middle"` \| `"high"` | Estimated socioeconomic status |
| `gender` | string | `"male"` \| `"female"` | Estimated gender |
| `esl` | string | `"native"` \| `"esl"` | Estimated English language background |
| `confidence` | float | 0.0 - 1.0 | Model's self-reported confidence in the classification |
| `linguistic_markers` | array of strings | — | Specific textual cues the model used to make its classifications |

## Example Response

```json
{
  "age": "old",
  "geographic": "rural",
  "education": "high_school",
  "socioeconomic": "low",
  "gender": "female",
  "esl": "native",
  "confidence": 0.72,
  "linguistic_markers": [
    "references to fixed income",
    "mentions of small-town bank branch",
    "informal register with colloquial expressions"
  ]
}
```

## Validation and Coercion Rules

All dimension fields apply `before`-mode validators that normalize LLM outputs to the allowed enum values:

- **age:** Maps `"middle"`, `"middle-aged"`, `"adult"` to `"young"`; maps `"elderly"`, `"senior"` to `"old"`. Default fallback: `"young"`.
- **geographic:** Default fallback: `"suburban"`.
- **education:** Maps `"highschool"`, `"high school"`, `"hs"`, `"secondary"` to `"high_school"`; maps `"university"`, `"graduate"`, `"postgraduate"`, `"bachelor"`, `"masters"`, `"phd"` to `"college"`. Default fallback: `"college"`.
- **socioeconomic:** Maps `"lower"`, `"poverty"`, `"working_class"` to `"low"`; maps `"upper"`, `"wealthy"`, `"affluent"` to `"high"`. Default fallback: `"middle"`.
- **gender:** Maps `"man"`, `"boy"`, `"m"` to `"male"`; maps `"woman"`, `"girl"`, `"f"` to `"female"`. Default fallback: `"male"`.
- **esl:** Maps `"native_speaker"`, `"l1"`, `"no"` to `"native"`; maps `"non-native"`, `"l2"`, `"yes"` to `"esl"`. Default fallback: `"native"`.
- **confidence:** Clamped to [0.0, 1.0]; defaults to 0.5 on parse failure.
