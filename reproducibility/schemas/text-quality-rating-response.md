# TextQualityRatingResponse

**Source:** `src/schemas.py` | **Task:** `text_quality_rating`

Rates the subjective quality of a complaint narrative across 12 dimensions on a 1-5 Likert scale. Used to measure whether persona-style rewriting affects perceived complaint quality, and whether quality perceptions correlate with remedy outcomes.

## Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `convincingness` | integer | 1-5 | How convincing the complaint sounds (1 = not at all, 5 = very) |
| `persuasiveness` | integer | 1-5 | How persuasive the complaint is (1 = not at all, 5 = very) |
| `evidence_blame` | integer | 1-5 | Clarity of evidence for which party is to blame (1 = unclear, 5 = very clear) |
| `evidence_harm` | integer | 1-5 | Clarity of evidence for harms suffered (1 = unclear, 5 = very clear) |
| `confidence` | integer | 1-5 | Confidence conveyed by the complainant (1 = not confident, 5 = very confident) |
| `legal_knowledge` | integer | 1-5 | Knowledge of legal/regulatory environment (1 = none apparent, 5 = extensive) |
| `formality` | integer | 1-5 | Formality of writing register (1 = very informal/casual, 5 = very formal/professional) |
| `coherence` | integer | 1-5 | Logical organization and structure (1 = disorganized/rambling, 5 = clear logical flow) |
| `specificity` | integer | 1-5 | Density of concrete facts, dates, amounts, names (1 = vague, 5 = highly specific) |
| `financial_literacy` | integer | 1-5 | Understanding of financial products and terminology (1 = none, 5 = expert-level) |
| `emotional_restraint` | integer | 1-5 | Measured tone vs. emotional charge (1 = highly emotional, 5 = calm and measured) |
| `grammar_quality` | integer | 1-5 | Grammar, spelling, and writing mechanics (1 = many errors, 5 = near-perfect) |
| `reasoning` | string | — | Brief justification for the ratings |

## Example Response

```json
{
  "convincingness": 4,
  "persuasiveness": 3,
  "evidence_blame": 4,
  "evidence_harm": 3,
  "confidence": 4,
  "legal_knowledge": 2,
  "formality": 3,
  "coherence": 4,
  "specificity": 5,
  "financial_literacy": 3,
  "emotional_restraint": 2,
  "grammar_quality": 4,
  "reasoning": "The complaint provides specific dates, amounts, and account details (high specificity) and clearly identifies the bank's failure to process the dispute (strong evidence of blame). However, the emotional language ('absolutely outrageous') reduces restraint, and there are no references to specific regulations or consumer protection laws."
}
```

## Validation and Coercion Rules

All 12 Likert dimensions share the same `coerce_likert` validator:
- Input is converted via `int(round(float(v)))`.
- Result is clamped to the range [1, 5].
- On parse failure, defaults to 3 (midpoint).
