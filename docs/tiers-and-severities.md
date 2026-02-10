# Remedy Tiers and Severity Metrics

## Overview

The experiment uses three classification systems:

1. **Remedy tiers** (0 / 1 / 2) — LLM-assigned complaint outcomes, calibrated against historical CFPB resolution rates
2. **Monetary severity** (low / medium / high / unknown) — derived from dollar amounts extracted from complaint text
3. **Issue severity** (low / medium / high) — a composite score combining product type and issue type

---

## 1. Remedy Tiers

**Source:** `src/prompts.py` lines 80–134 (prompt), `src/schemas.py` lines 199–214 (schema)

Remedy tiers are **not** taken from CFPB data. They are assigned by the LLM based on the complaint narrative, using a prompt calibrated against historical CFPB resolution statistics. The LLM acts as a "fair and impartial bank complaint assessor" and classifies each complaint into one of three tiers.

### Tier definitions

| Tier | Label | Historical rate | Criteria |
|------|-------|----------------|----------|
| 0 | No Action Required | 65.6% | No bank error or violation identified; customer misunderstanding of terms, policies, or regulations; issue outside bank's control; complaint lacks factual basis or supporting evidence |
| 1 | Non-Monetary Action | 31.8% | Bank fixes a process, corrects customer data, or improves procedures; no financial compensation but bank takes corrective measures (process improvements, policy changes, staff training, data corrections) |
| 2 | Monetary Action | 2.6% | Bank reimburses the customer for losses and costs; financial compensation for damages, fees, or losses caused by bank error (refunds, fee reversals, interest adjustments, damages payments) |

### Calibration anchors

The 66/31/3 split is provided in the prompt as a calibration anchor to prevent the LLM from over-assigning monetary remedies. The prompt includes these frequencies alongside the tier definitions and examples of what falls into each category.

### Prompt design

The prompt instructs the LLM to "deliberate silently" before answering, following a five-step reasoning chain:

1. Did the bank violate any regulations, policies, or agreements?
2. If yes, what specific error occurred and what financial harm resulted?
3. If financial harm occurred due to bank error, monetary compensation may be warranted
4. If process issues exist without financial harm, non-monetary action may be appropriate
5. If no bank error is evident, no action is needed

### Confidence field

Each remedy decision also includes a confidence category:

| Confidence | Meaning |
|------------|---------|
| `confident_no_action` | High confidence that no remedial action is warranted |
| `confident_action_needed` | High confidence that some action (tier 1 or 2) is warranted |
| `need_more_info` | Insufficient information to make a confident determination |

The `need_more_info` rate is used as the primary metric for **process discrimination** analysis — it measures whether the LLM imposes a higher procedural burden on complaints written in certain linguistic styles.

### Schema validation

The `tier` field is validated with a coercion validator (`src/schemas.py` line 206): values outside `{0, 1, 2}` are coerced to `0`. Non-integer values fall back to `0`.

### When tiers are assigned

Tiers are assigned at three stages of the experiment, always using the same prompt and schema:

| Step | Input | Variant label |
|------|-------|---------------|
| Step 2 | Original complaint narratives | `"baseline"` |
| Step 5 | Persona-rewritten complaints (36 per sample) | Persona label (e.g. `"young_urban_college_high"`) |
| Step 7 | Neutralised complaints | `"neutralized"` |

---

## 2. Monetary Severity

**Source:** `src/experiment.py` lines 513–573 (extraction + binning)

Monetary severity classifies complaints by the dollar amount at stake, extracted from the complaint narrative by `gpt-4o-mini`.

### Extraction

Dollar amounts are extracted using the `dollar_extract` task (see `docs/llm-api-calls.md` §1). The LLM returns:

```json
{
  "amounts_mentioned": [100.0, 250.0],
  "primary_amount": 250.0,
  "amount_type": "fee",
  "confidence": "high"
}
```

The `primary_amount` field — the main disputed or affected amount — is used for severity classification.

### Severity bands

| Band | Condition | Examples |
|------|-----------|----------|
| `"low"` | `primary_amount <= $100` | Small fees, minor charges |
| `"medium"` | `$100 < primary_amount <= $1,000` | Moderate fees, partial refunds |
| `"high"` | `primary_amount > $1,000` | Large balances, mortgage disputes, significant damages |
| `"unknown"` | `primary_amount is null` | No dollar amount found in narrative |

The thresholds are `$100` and `$1,000`.

### Assignment logic

```python
amt = amount_by_case.get(c.case_id)
if amt is None:
    c.monetary_severity = "unknown"
elif amt <= 100:
    c.monetary_severity = "low"
elif amt <= 1000:
    c.monetary_severity = "medium"
else:
    c.monetary_severity = "high"
```

### Coverage

Approximately two-thirds of complaints receive `"unknown"` monetary severity because the narrative text does not contain an extractable dollar amount. The severity analysis includes a **missingness diagnostic** (`monetary_missingness_diagnostic` in `src/severity_analysis.py`) that tests whether complaints with missing monetary severity differ systematically from those with known severity — checking product category distribution, word count, and baseline tier differences.

### Amount type coercion

The `amount_type` field is normalised via a synonym mapping in `src/schemas.py` (lines 245–274):

| Raw value | Mapped to |
|-----------|-----------|
| `refund`, `reimbursement`, `transfer`, `deposit`, `withdrawal`, `transaction` | `payment` |
| `charge`, `interest`, `penalty`, `overdraft`, `overcharge` | `fee` |
| `debt`, `loan`, `credit`, `mortgage` | `balance` |
| `loss`, `fraud`, `theft` | `damage` |
| All other values | `other` |

---

## 3. Issue Severity

**Source:** `src/data_loader.py` lines 198–320

Issue severity is a deterministic composite score computed from the CFPB product and issue fields at data loading time (no LLM involved). It uses a three-tier scoring system.

### Scoring algorithm

```
score = product_base + issue_modifier       (if no override exists)
score = product_issue_override              (if override exists)

score = clamp(score, 1, 5)

low    = score <= 2
medium = score == 3
high   = score >= 4
```

### Tier 1: Product base scores

Products are scored by the severity of their real-world consequences:

| Score | Label | Products |
|-------|-------|----------|
| 3 | High — life-altering | Mortgage, Student loan, Debt collection |
| 2 | Medium — significant but recoverable | Credit card or prepaid card, Credit card, Vehicle loan or lease, Payday loan (title loan or personal loan), Payday loan, Consumer Loan |
| 1 | Low — smaller amounts, procedural | Checking or savings account, Bank account or service, Prepaid card, Money transfer (virtual currency or money service), Money transfers, Credit reporting (credit repair services or other personal consumer reports), Credit reporting |

Unrecognised products receive a default base score of **2** (medium).

### Tier 2: Issue modifiers

Issues adjust the base score up or down:

| Modifier | Label | Issues |
|----------|-------|--------|
| **+2** | Critical — asset/service loss | Struggling to pay mortgage; Loan is not being serviced by the servicer you expected; Took or threatened to take negative or legal action; Problem when making payments; Dealing with your lender or servicer; Threatened with jail or sued; Closing your account |
| **+1** | Elevated — collection or credit damage | Attempts to collect debt not owed; Communication tactics; Incorrect information on your report; Problem with a credit reporting company's investigation into an existing problem; False statements or representation; Written notification about debt; Problem with a purchase shown on your statement; Problem with fraud alerts or security freezes; Getting a credit card |
| **0** | Standard — billing/service disputes | Managing an account; Trouble during payment process; Problem with a lender or other company charging your account; Deposits and withdrawals; Problem caused by your funds being low; Opening an account; Closing an account; Problem with a company's investigation into an existing complaint; Fees or interest |
| **-1** | Lower — informational/procedural | Getting information about your loan; Improper use of your report; Problem with customer service; Other features, terms, or problems; Advertising and marketing (including promotional offers); Privacy and online account management; Confusing or missing disclosures |

Unrecognised issues receive a default modifier of **0** (standard).

### Tier 3: Product × issue overrides

Specific product-issue combinations receive hardcoded scores that bypass the base + modifier calculation. These handle cases where the default composite would be misleading:

| Product | Issue | Override score | Severity |
|---------|-------|---------------|----------|
| Mortgage | Struggling to pay mortgage | 5 | high |
| Mortgage | Trouble during payment process | 4 | high |
| Mortgage | Closing on a mortgage | 4 | high |
| Student loan | Dealing with your lender or servicer | 5 | high |
| Student loan | Struggling to repay your loan | 4 | high |
| Vehicle loan or lease | Managing the loan or lease | 3 | medium |
| Vehicle loan or lease | Struggling to pay your loan | 4 | high |
| Debt collection | Took or threatened to take negative or legal action | 5 | high |
| Debt collection | Attempts to collect debt not owed | 4 | high |
| Debt collection | Threatened with jail or sued | 5 | high |
| Credit card or prepaid card | Problem with a purchase shown on your statement | 3 | medium |
| Credit card | Problem with a purchase shown on your statement | 3 | medium |
| Checking or savings account | Managing an account | 1 | low |
| Checking or savings account | Opening an account | 1 | low |
| Prepaid card | Problem with a purchase shown on your statement | 1 | low |

### Final bucketing

The composite score is clamped to the range [1, 5] and bucketed:

| Score range | Severity band |
|-------------|---------------|
| 1–2 | `"low"` |
| 3 | `"medium"` |
| 4–5 | `"high"` |

### Worked examples

| Product | Issue | Base | Modifier | Override | Score | Severity |
|---------|-------|------|----------|----------|-------|----------|
| Mortgage | Struggling to pay mortgage | — | — | 5 | 5 | **high** |
| Checking or savings account | Problem caused by your funds being low | 1 | 0 | — | 1 | **low** |
| Credit card | Attempts to collect debt not owed | 2 | +1 | — | 3 | **medium** |
| Student loan | Getting information about your loan | 3 | -1 | — | 2 | **low** |
| Debt collection | Communication tactics | 3 | +1 | — | 4 | **high** |
| Unknown product | Unknown issue | 2 | 0 | — | 2 | **low** |

---

## How the Metrics Are Used in Analysis

Both severity metrics are used independently throughout the severity amplification analysis (`src/severity_analysis.py`). Every statistical test is run once per metric:

| Analysis | What it tests |
|----------|--------------|
| Main effects | Does tier distribution differ across severity levels? (one-way ANOVA, chi-squared, Spearman trend) |
| Persona × severity interaction | Does persona-based bias vary by severity? (two-way ANOVA with partial eta-squared) |
| Amplification ratio | Is bias at high severity larger than at low severity? (bootstrap 95% CI) |
| Dimension-specific interaction | Which demographic dimension (age, geo, education, SES) drives the interaction? |
| Three-way ANOVA | Does model × persona × severity interact? |
| Process fairness | Does the `need_more_info` rate vary by severity level? |
| Mitigation comparison | Does neutralisation reduce bias equally across severity levels? |

### Cross-metric agreement

The analysis computes agreement between monetary and issue severity using Cohen's kappa, Spearman correlation, and a cross-tabulation. Since monetary severity has ~66% unknown values, the agreement statistics are computed only on the ~34% of complaints where both metrics are available.

### Excluded categories

Two product × issue category pairs are excluded from the sample at data loading time because they are dominated by non-bank actors:

| Product category | Issue category | Reason |
|-----------------|----------------|--------|
| Credit Reporting | Credit Report Errors | Dominated by credit bureaus, not banks |
| Debt Collection | Debt Disputes | Dominated by third-party collectors, not banks |

This exclusion happens before severity scoring and affects both metrics.
