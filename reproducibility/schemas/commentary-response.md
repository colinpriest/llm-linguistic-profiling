# CommentaryResponse

**Source:** `src/schemas.py` | **Task:** `commentary`

Generates analytical commentary paragraphs for sections of the experiment report. The LLM produces written analysis with specific findings, statistics, and conclusions based on the experiment's quantitative results.

## Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `commentary` | string | — | Complete written analysis paragraph(s) with specific findings, statistics, and conclusions. Must contain actual analytical claims (e.g., "The chi-squared test was significant at p=0.03"), not descriptions of what analysis to perform. |

## Example Response

```json
{
  "commentary": "The adverse impact ratio analysis reveals that complaints classified as originating from 'low' socioeconomic backgrounds received Tier 2 remedies at a rate 23% lower than those from 'high' socioeconomic backgrounds (AIR = 0.77, below the 0.80 threshold). The chi-squared test confirms this association is statistically significant (χ² = 14.3, p = 0.002, FDR-corrected p = 0.008). Logistic regression controlling for complaint severity and product category shows the socioeconomic coefficient remains significant (OR = 0.68, 95% CI [0.52, 0.89])."
}
```

## Validation and Coercion Rules

No field-level coercion is applied. The field accepts any string value.
