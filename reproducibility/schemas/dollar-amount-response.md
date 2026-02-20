# DollarAmountResponse

**Source:** `src/schemas.py` | **Task:** `dollar_extract`

Extracts monetary values mentioned in a complaint narrative. Used for computing monetary severity scores and for the hard-slot preservation audit (verifying that dollar amounts are preserved during persona rewriting).

## Fields

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `amounts_mentioned` | array of floats or null | optional | All dollar values found in the complaint text |
| `primary_amount` | float or null | optional, non-negative | The main disputed or affected amount |
| `amount_type` | string or null | `"fee"` \| `"balance"` \| `"payment"` \| `"damage"` \| `"other"` | Category of the primary amount |
| `confidence` | string or null | `"high"` \| `"medium"` \| `"low"` | Confidence in the extraction accuracy |

## Example Response

```json
{
  "amounts_mentioned": [450.00, 35.00, 25.00],
  "primary_amount": 450.00,
  "amount_type": "balance",
  "confidence": "high"
}
```

## Validation and Coercion Rules

- **amounts_mentioned:** Each element coerced to `abs(float(item))`; non-numeric entries are silently dropped. Returns `null` if the resulting list is empty.
- **primary_amount:** Coerced to `abs(float(v))`; returns `null` on parse failure.
- **amount_type:** Normalized to lowercase. Maps common synonyms to allowed categories:
  - `"refund"`, `"reimbursement"`, `"transfer"`, `"deposit"`, `"withdrawal"`, `"transaction"` → `"payment"`
  - `"charge"`, `"interest"`, `"penalty"`, `"overdraft"`, `"overcharge"` → `"fee"`
  - `"debt"`, `"loan"`, `"credit"`, `"mortgage"` → `"balance"`
  - `"loss"`, `"fraud"`, `"theft"` → `"damage"`
  - All other values → `"other"`
- **confidence:** Normalized to lowercase; values outside {`"high"`, `"medium"`, `"low"`} default to `"medium"`.
