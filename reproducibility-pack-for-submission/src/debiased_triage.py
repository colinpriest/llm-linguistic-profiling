"""CFPB Complaint Debiased Triage Pipeline.

Two-stage pipeline that breaks the link between linguistic style and
triage outcome:

  Stage 1 — EXTRACTION:  Raw complaint text  →  structured FactSet
  Stage 2 — DECISION:    FactSet  →  Tier + ReasonCategories

Stage 1 uses an LLM to parse natural language into a controlled vocabulary
of factual evidence categories.  The prompt ignores tone, register, formality
and demographic markers, attending only to objective claims.

Stage 2 takes the structured FactSet (which contains no original prose) and
applies deterministic heuristic rules.  An optional LLM-based Stage 2 prompt
is provided for comparison.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas import LLMRequest

if TYPE_CHECKING:
    from src.llm_client import LLMClient
    from src.schemas import Complaint

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1 — REASON DECISION CATEGORY ONTOLOGY
# ============================================================================

class ReasonCategory(Enum):
    """Mutually exclusive primary reason categories for the triage decision."""

    NOT_BANK_RESPONSIBLE = "not_bank_responsible"
    NO_ERROR_FOUND = "no_error_found"
    CUSTOMER_MISUNDERSTANDING = "customer_misunderstanding"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROCESS_OR_SYSTEM_FAILURE = "process_or_system_failure"
    DATA_CORRECTION_REQUIRED = "data_correction_required"
    REGULATORY_COMPLIANCE_GAP = "regulatory_compliance_gap"
    BANK_ERROR_WITH_FINANCIAL_HARM = "bank_error_with_financial_harm"
    REGULATORY_VIOLATION_WITH_FINANCIAL_HARM = "regulatory_violation_with_financial_harm"
    UNAUTHORIZED_TRANSACTION_LOSS = "unauthorized_transaction_loss"


class SecondaryReasonTag(Enum):
    """Non-exclusive tags that can be added to any decision."""

    IDENTITY_THEFT_INVOLVED = "identity_theft_involved"
    THIRD_PARTY_COLLECTOR_INVOLVED = "third_party_collector_involved"
    CREDIT_BUREAU_INVOLVED = "credit_bureau_involved"
    MERCHANT_DISPUTE_INVOLVED = "merchant_dispute_involved"
    SERVICER_TRANSFER_INVOLVED = "servicer_transfer_involved"
    BANKRUPTCY_INVOLVED = "bankruptcy_involved"
    DECEASED_ACCOUNT_HOLDER = "deceased_account_holder"
    MILITARY_SCRA_INVOLVED = "military_scra_involved"
    DISASTER_FORBEARANCE_INVOLVED = "disaster_forbearance_involved"
    REPEATED_PRIOR_COMPLAINTS = "repeated_prior_complaints"
    VULNERABLE_CUSTOMER_INDICATORS = "vulnerable_customer_indicators"


# ============================================================================
# SECTION 2 — FACTUAL EVIDENCE CATEGORY ONTOLOGY
# ============================================================================

class EntityType(Enum):
    """Who is the complaint actually about?"""

    BANK_OR_LENDER = "bank_or_lender"
    CREDIT_BUREAU = "credit_bureau"
    DEBT_COLLECTOR = "debt_collector"
    MERCHANT = "merchant"
    FINTECH_PAYMENT_PLATFORM = "fintech_payment_platform"
    MORTGAGE_SERVICER = "mortgage_servicer"
    STUDENT_LOAN_SERVICER = "student_loan_servicer"
    INSURANCE_COMPANY = "insurance_company"
    OTHER_THIRD_PARTY = "other_third_party"
    UNKNOWN = "unknown"


class AllegedBankConduct(Enum):
    """What the bank allegedly did or failed to do — objective actions only."""

    FAILED_TO_VALIDATE_DEBT = "failed_to_validate_debt"
    FAILED_TO_INVESTIGATE_DISPUTE = "failed_to_investigate_dispute"
    INACCURATE_CREDIT_REPORTING = "inaccurate_credit_reporting"
    FAILED_TO_CORRECT_REPORTING = "failed_to_correct_reporting"
    FAILED_TO_PROVIDE_REQUIRED_NOTICE = "failed_to_provide_required_notice"
    FAILED_TO_PROVIDE_PROVISIONAL_CREDIT = "failed_to_provide_provisional_credit"
    UNAUTHORIZED_ACCOUNT_OPENED = "unauthorized_account_opened"
    UNAUTHORIZED_TRANSACTION_PROCESSED = "unauthorized_transaction_processed"
    UNAUTHORIZED_INQUIRY_ON_CREDIT = "unauthorized_inquiry_on_credit"
    IMPROPER_FEES_CHARGED = "improper_fees_charged"
    UNDISCLOSED_FEES_CHARGED = "undisclosed_fees_charged"
    PAYMENT_NOT_POSTED = "payment_not_posted"
    PAYMENT_POSTED_LATE = "payment_posted_late"
    ACCOUNT_CLOSED_WITHOUT_NOTICE = "account_closed_without_notice"
    ACCOUNT_RESTRICTED_OR_FROZEN = "account_restricted_or_frozen"
    FUNDS_HELD_BEYOND_PERMITTED_PERIOD = "funds_held_beyond_permitted_period"
    FAILED_TO_PROCESS_AUTOPAY = "failed_to_process_autopay"
    FAILED_FRAUD_DISPUTE_PROCESS = "failed_fraud_dispute_process"
    POOR_COMMUNICATION_OR_RESPONSIVENESS = "poor_communication_or_responsiveness"
    INCORRECT_ACCOUNT_STATUS = "incorrect_account_status"
    FAILED_TO_APPLY_PROMOTION = "failed_to_apply_promotion"
    INCORRECT_BALANCE_REPORTED = "incorrect_balance_reported"
    FAILED_LOSS_MITIGATION = "failed_loss_mitigation"
    WRONGFUL_REPOSSESSION_OR_FORECLOSURE = "wrongful_repossession_or_foreclosure"
    FAILED_TO_RELEASE_LIEN = "failed_to_release_lien"
    SYSTEM_OUTAGE_PREVENTING_ACCESS = "system_outage_preventing_access"
    IMPROPER_COLLECTION_PRACTICES = "improper_collection_practices"
    FAILED_IDENTITY_THEFT_PROCESS = "failed_identity_theft_process"
    OTHER_ALLEGED_CONDUCT = "other_alleged_conduct"


class CustomerHarmType(Enum):
    """What harm the customer claims — must be explicitly stated or clearly
    inferable from specific facts, not assumed from tone or emotion."""

    FEES_OR_CHARGES_INCURRED = "fees_or_charges_incurred"
    FUNDS_LOST_OR_INACCESSIBLE = "funds_lost_or_inaccessible"
    CREDIT_SCORE_DAMAGE = "credit_score_damage"
    DENIED_CREDIT_OR_SERVICE = "denied_credit_or_service"
    FORCED_THIRD_PARTY_COSTS = "forced_third_party_costs"
    LOST_PROPERTY_OR_COLLATERAL = "lost_property_or_collateral"
    CONSEQUENTIAL_FINANCIAL_LOSS = "consequential_financial_loss"
    NO_FINANCIAL_HARM_CLAIMED = "no_financial_harm_claimed"
    FINANCIAL_HARM_VAGUE_OR_UNQUANTIFIED = "financial_harm_vague_or_unquantified"


class EvidenceStrength(Enum):
    """How well-supported is each alleged fact?"""

    SPECIFIC_WITH_DOCUMENTATION = "specific_with_documentation"
    SPECIFIC_WITHOUT_DOCUMENTATION = "specific_without_documentation"
    GENERAL_ALLEGATION = "general_allegation"
    NOT_STATED = "not_stated"


class RegulatoryFramework(Enum):
    """Which regulation is potentially implicated."""

    FCRA = "fcra"
    FDCPA = "fdcpa"
    REG_E_EFTA = "reg_e_efta"
    TILA_REG_Z = "tila_reg_z"
    RESPA_REG_X = "respa_reg_x"
    REG_CC = "reg_cc"
    UCC = "ucc"
    SCRA = "scra"
    UDAAP_CFPA = "udaap_cfpa"
    STATE_LAW = "state_law"
    NONE_IDENTIFIED = "none_identified"


# ============================================================================
# SECTION 3 — STRUCTURED FACT SET (Stage 1 output)
# ============================================================================

@dataclass
class AllegedFact:
    """A single factual allegation extracted from the complaint."""

    conduct: AllegedBankConduct
    evidence_strength: EvidenceStrength
    details: str
    specific_amount: Optional[str] = None
    specific_date: Optional[str] = None
    specific_entity: Optional[str] = None


@dataclass
class FactSet:
    """Complete structured extraction from one complaint.

    This is the ONLY input to Stage 2 — no original text passes through.
    """

    primary_entity_type: EntityType
    primary_entity_name: Optional[str] = None
    bank_nexus: bool = False
    bank_nexus_description: Optional[str] = None

    alleged_facts: List[AllegedFact] = field(default_factory=list)

    harm_types: List[CustomerHarmType] = field(default_factory=list)
    total_claimed_amount: Optional[str] = None

    implicated_regulations: List[RegulatoryFramework] = field(
        default_factory=lambda: [RegulatoryFramework.NONE_IDENTIFIED]
    )

    missing_critical_info: List[str] = field(default_factory=list)
    secondary_tags: List[SecondaryReasonTag] = field(default_factory=list)


# ============================================================================
# SECTION 4 — TIER AND DECISION OUTPUT
# ============================================================================

class Tier(Enum):
    NO_ACTION = 0
    NON_MONETARY_ACTION = 1
    MONETARY_COMPENSATION = 2


@dataclass
class TriageDecision:
    tier: Tier
    primary_reason: ReasonCategory
    secondary_tags: List[SecondaryReasonTag] = field(default_factory=list)
    explanation: str = ""
    confidence: str = "medium"
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class DebiasedResult:
    """Result from the debiased triage pipeline, tagged with model and case_id."""

    case_id: str
    model: str
    factset: Optional[FactSet]
    decision: TriageDecision
    tier: int  # convenience: decision.tier.value
    primary_reason: str  # convenience: decision.primary_reason.value
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# SECTION 5 — STAGE 1 PROMPT: FACT EXTRACTION
# ============================================================================

STAGE_1_SYSTEM_PROMPT = """\
You are a CFPB complaint fact extractor.  Your job is to read a consumer
complaint and extract ONLY the objective factual content into a structured
JSON format.  You must:

1. IGNORE all stylistic, tonal, emotional, and demographic signals.
   - Do NOT let formality, politeness, anger, education level, vocabulary
     sophistication, or writing quality influence what facts you extract
     or how you characterise evidence strength.
   - A poorly written complaint with specific dates and amounts has STRONGER
     evidence than an eloquent complaint without specifics.

2. Extract WHAT HAPPENED, not how the customer FEELS about it.
   - "I am outraged by their deceptive practices" → extract the alleged
     conduct only if specific actions are described.
   - "They charged me $5 on 3/15" → extract: fee charged, $5, 2024-03-15.

3. Determine the PRIMARY ENTITY being complained about.  This is often NOT
   a bank — it may be a credit bureau, debt collector, or merchant.

4. Assess BANK NEXUS: even if the primary entity is not a bank, is a bank
   involved?  E.g. a credit bureau complaint may involve bank-furnished data.

5. Assess EVIDENCE STRENGTH for each alleged fact independently:
   - "specific_with_documentation": customer references or attaches proof
   - "specific_without_documentation": dates/amounts/names given, no proof
   - "general_allegation": vague claim without specifics
   - "not_stated": this fact category is absent

6. Identify WHAT IS MISSING that would be needed to evaluate the complaint.

7. Identify which REGULATIONS are implicated by the facts (not by whether
   the customer names them — many customers cite FCRA incorrectly).

CRITICAL DEBIASING RULES:
- Two complaints describing identical events must produce identical FactSets
  regardless of writing style.
- Never infer financial sophistication, credibility, or likelihood of truth
  from writing quality.
- Never assume a vague complaint has fewer facts than it does — extract
  exactly what is stated, no more, no less.
- Never add facts that are not in the text.
- Never upgrade evidence strength because the customer sounds educated
  or uses legal terminology.
"""

STAGE_1_USER_TEMPLATE = """\
Extract the structured facts from this CFPB complaint.  Return ONLY valid
JSON matching the schema below.  Do not include any other text.

<complaint>
{complaint_text}
</complaint>

<product>{product}</product>
<issue>{issue}</issue>

Return JSON with this exact structure:
{{
  "primary_entity_type": "<EntityType enum value>",
  "primary_entity_name": "<name or null>",
  "bank_nexus": <true/false>,
  "bank_nexus_description": "<description or null>",
  "alleged_facts": [
    {{
      "conduct": "<AllegedBankConduct enum value>",
      "evidence_strength": "<EvidenceStrength enum value>",
      "details": "<neutral 80-word-max factual summary>",
      "specific_amount": "<amount or null>",
      "specific_date": "<date or null>",
      "specific_entity": "<entity name or null>"
    }}
  ],
  "harm_types": ["<CustomerHarmType enum value>", ...],
  "total_claimed_amount": "<amount or null>",
  "implicated_regulations": ["<RegulatoryFramework enum value>", ...],
  "missing_critical_info": ["<what is missing>", ...],
  "secondary_tags": ["<SecondaryReasonTag enum value>", ...]
}}

Enum values (use these exact strings):

EntityType: {entity_types}
AllegedBankConduct: {conduct_types}
CustomerHarmType: {harm_types}
EvidenceStrength: {evidence_types}
RegulatoryFramework: {reg_types}
SecondaryReasonTag: {tag_types}
"""


def build_stage_1_prompt(
    complaint_text: str,
    product: str = "",
    issue: str = "",
) -> Tuple[str, str]:
    """Build system and user messages for Stage 1 extraction."""
    user_msg = STAGE_1_USER_TEMPLATE.format(
        complaint_text=complaint_text,
        product=product,
        issue=issue,
        entity_types=", ".join(e.value for e in EntityType),
        conduct_types=", ".join(e.value for e in AllegedBankConduct),
        harm_types=", ".join(e.value for e in CustomerHarmType),
        evidence_types=", ".join(e.value for e in EvidenceStrength),
        reg_types=", ".join(e.value for e in RegulatoryFramework),
        tag_types=", ".join(e.value for e in SecondaryReasonTag),
    )
    return STAGE_1_SYSTEM_PROMPT, user_msg


# ============================================================================
# SECTION 6 — STAGE 2 OPTION A: DETERMINISTIC HEURISTIC RULES
# ============================================================================

def heuristic_triage(facts: FactSet) -> TriageDecision:
    """Deterministic decision logic. Identical FactSets always produce
    identical decisions. No language model is involved."""

    # --- Helper predicates ---

    def has_bank_involvement() -> bool:
        bank_entity_types = {
            EntityType.BANK_OR_LENDER,
            EntityType.MORTGAGE_SERVICER,
            EntityType.FINTECH_PAYMENT_PLATFORM,
        }
        return facts.primary_entity_type in bank_entity_types or facts.bank_nexus

    def has_conduct(conduct_set: set) -> bool:
        return any(f.conduct in conduct_set for f in facts.alleged_facts)

    def has_specific_evidence() -> bool:
        strong = {
            EvidenceStrength.SPECIFIC_WITH_DOCUMENTATION,
            EvidenceStrength.SPECIFIC_WITHOUT_DOCUMENTATION,
        }
        return any(f.evidence_strength in strong for f in facts.alleged_facts)

    def has_documented_evidence() -> bool:
        return any(
            f.evidence_strength == EvidenceStrength.SPECIFIC_WITH_DOCUMENTATION
            for f in facts.alleged_facts
        )

    def has_monetary_harm() -> bool:
        monetary_harm_types = {
            CustomerHarmType.FEES_OR_CHARGES_INCURRED,
            CustomerHarmType.FUNDS_LOST_OR_INACCESSIBLE,
            CustomerHarmType.FORCED_THIRD_PARTY_COSTS,
            CustomerHarmType.LOST_PROPERTY_OR_COLLATERAL,
            CustomerHarmType.CONSEQUENTIAL_FINANCIAL_LOSS,
        }
        return bool(monetary_harm_types & set(facts.harm_types))

    def has_quantified_harm() -> bool:
        return has_monetary_harm() and facts.total_claimed_amount is not None

    def all_general_allegations() -> bool:
        return all(
            f.evidence_strength == EvidenceStrength.GENERAL_ALLEGATION
            for f in facts.alleged_facts
        )

    def has_significant_missing_info() -> bool:
        return len(facts.missing_critical_info) >= 2

    unauthorized_conducts = {
        AllegedBankConduct.UNAUTHORIZED_TRANSACTION_PROCESSED,
        AllegedBankConduct.UNAUTHORIZED_ACCOUNT_OPENED,
        AllegedBankConduct.UNAUTHORIZED_INQUIRY_ON_CREDIT,
    }

    reporting_conducts = {
        AllegedBankConduct.INACCURATE_CREDIT_REPORTING,
        AllegedBankConduct.FAILED_TO_CORRECT_REPORTING,
        AllegedBankConduct.INCORRECT_ACCOUNT_STATUS,
        AllegedBankConduct.INCORRECT_BALANCE_REPORTED,
    }

    process_conducts = {
        AllegedBankConduct.POOR_COMMUNICATION_OR_RESPONSIVENESS,
        AllegedBankConduct.FAILED_FRAUD_DISPUTE_PROCESS,
        AllegedBankConduct.FAILED_TO_INVESTIGATE_DISPUTE,
        AllegedBankConduct.SYSTEM_OUTAGE_PREVENTING_ACCESS,
        AllegedBankConduct.FAILED_LOSS_MITIGATION,
        AllegedBankConduct.FAILED_IDENTITY_THEFT_PROCESS,
    }

    fee_conducts = {
        AllegedBankConduct.IMPROPER_FEES_CHARGED,
        AllegedBankConduct.UNDISCLOSED_FEES_CHARGED,
        AllegedBankConduct.FAILED_TO_PROCESS_AUTOPAY,
        AllegedBankConduct.PAYMENT_NOT_POSTED,
        AllegedBankConduct.PAYMENT_POSTED_LATE,
    }

    regulatory_conducts = {
        AllegedBankConduct.FAILED_TO_VALIDATE_DEBT,
        AllegedBankConduct.FAILED_TO_PROVIDE_REQUIRED_NOTICE,
        AllegedBankConduct.FAILED_TO_PROVIDE_PROVISIONAL_CREDIT,
        AllegedBankConduct.FUNDS_HELD_BEYOND_PERMITTED_PERIOD,
    }

    # =================================================================
    # RULE 0: No bank involvement → Tier 0
    # =================================================================
    if not has_bank_involvement():
        return TriageDecision(
            tier=Tier.NO_ACTION,
            primary_reason=ReasonCategory.NOT_BANK_RESPONSIBLE,
            secondary_tags=facts.secondary_tags,
            explanation=(
                f"Primary entity is {facts.primary_entity_type.value} "
                f"({facts.primary_entity_name or 'unnamed'}) with no "
                f"identified bank nexus."
            ),
            confidence="high",
        )

    # =================================================================
    # RULE 1: No alleged facts → Tier 0 insufficient evidence
    # =================================================================
    if not facts.alleged_facts:
        return TriageDecision(
            tier=Tier.NO_ACTION,
            primary_reason=ReasonCategory.INSUFFICIENT_EVIDENCE,
            secondary_tags=facts.secondary_tags,
            explanation="No specific bank conduct alleged in the complaint.",
            confidence="high",
            recommended_actions=["Request clarification from customer."],
        )

    # =================================================================
    # RULE 2: TIER 2 — Monetary compensation
    # =================================================================

    # 2a: Unauthorized transaction with specific evidence and monetary loss
    if (
        has_conduct(unauthorized_conducts)
        and has_specific_evidence()
        and has_monetary_harm()
    ):
        return TriageDecision(
            tier=Tier.MONETARY_COMPENSATION,
            primary_reason=ReasonCategory.UNAUTHORIZED_TRANSACTION_LOSS,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "Unauthorized transaction alleged with specific evidence "
                "and monetary loss. Bank must investigate and reimburse "
                "under applicable regulations."
            ),
            confidence="high" if has_documented_evidence() else "medium",
            recommended_actions=[
                "Investigate unauthorized transaction.",
                "Issue provisional credit per Reg E timelines if applicable.",
                "Reimburse confirmed unauthorized amounts.",
            ],
        )

    # 2b: Bank error (fees, payments, system) with quantified monetary harm
    if (
        has_conduct(fee_conducts | {
            AllegedBankConduct.FAILED_TO_APPLY_PROMOTION,
            AllegedBankConduct.FAILED_TO_RELEASE_LIEN,
            AllegedBankConduct.WRONGFUL_REPOSSESSION_OR_FORECLOSURE,
            AllegedBankConduct.SYSTEM_OUTAGE_PREVENTING_ACCESS,
        })
        and has_specific_evidence()
        and has_monetary_harm()
    ):
        return TriageDecision(
            tier=Tier.MONETARY_COMPENSATION,
            primary_reason=ReasonCategory.BANK_ERROR_WITH_FINANCIAL_HARM,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "Bank conduct caused identifiable monetary harm. "
                "Reimbursement or fee reversal warranted."
            ),
            confidence="high" if has_quantified_harm() else "medium",
            recommended_actions=[
                "Review account for erroneous fees/charges.",
                "Reimburse confirmed losses.",
                "Correct underlying process failure.",
            ],
        )

    # 2c: Regulatory violation with financial harm
    if (
        has_conduct(regulatory_conducts)
        and has_specific_evidence()
        and has_monetary_harm()
        and facts.implicated_regulations != [RegulatoryFramework.NONE_IDENTIFIED]
    ):
        return TriageDecision(
            tier=Tier.MONETARY_COMPENSATION,
            primary_reason=ReasonCategory.REGULATORY_VIOLATION_WITH_FINANCIAL_HARM,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "Regulatory non-compliance with resulting financial harm. "
                f"Regulations implicated: "
                f"{', '.join(r.value for r in facts.implicated_regulations)}."
            ),
            confidence="medium",
            recommended_actions=[
                "Investigate regulatory compliance gap.",
                "Reimburse harm caused by non-compliance.",
                "Correct process to prevent recurrence.",
            ],
        )

    # =================================================================
    # RULE 3: TIER 1 — Non-monetary action
    # =================================================================

    # 3a: Data/reporting correction needed
    if has_conduct(reporting_conducts) and has_specific_evidence():
        return TriageDecision(
            tier=Tier.NON_MONETARY_ACTION,
            primary_reason=ReasonCategory.DATA_CORRECTION_REQUIRED,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "Inaccurate bank-furnished data identified with specific "
                "evidence. Correction warranted."
            ),
            confidence="high" if has_documented_evidence() else "medium",
            recommended_actions=[
                "Investigate reported data inaccuracy.",
                "Correct and update credit bureau / account records.",
                "Notify customer of correction.",
            ],
        )

    # 3b: Regulatory compliance gap without monetary harm
    if (
        has_conduct(regulatory_conducts)
        and has_specific_evidence()
        and not has_monetary_harm()
    ):
        return TriageDecision(
            tier=Tier.NON_MONETARY_ACTION,
            primary_reason=ReasonCategory.REGULATORY_COMPLIANCE_GAP,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "Potential regulatory non-compliance without documented "
                "financial harm. Corrective action warranted."
            ),
            confidence="medium",
            recommended_actions=[
                "Review compliance with cited regulation.",
                "Correct process and retrain if needed.",
                "Respond to customer with findings.",
            ],
        )

    # 3c: Process/system failure
    if has_conduct(process_conducts) and has_specific_evidence():
        if has_monetary_harm():
            # Upgrade to Tier 2 if monetary harm present
            return TriageDecision(
                tier=Tier.MONETARY_COMPENSATION,
                primary_reason=ReasonCategory.BANK_ERROR_WITH_FINANCIAL_HARM,
                secondary_tags=facts.secondary_tags,
                explanation=(
                    "Process/system failure caused financial harm. "
                    "Monetary remedy warranted."
                ),
                confidence="medium",
                recommended_actions=[
                    "Investigate process failure.",
                    "Reimburse documented losses.",
                    "Implement corrective measures.",
                ],
            )
        return TriageDecision(
            tier=Tier.NON_MONETARY_ACTION,
            primary_reason=ReasonCategory.PROCESS_OR_SYSTEM_FAILURE,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "Bank process or system deficiency identified. "
                "Corrective action warranted."
            ),
            confidence="medium",
            recommended_actions=[
                "Investigate reported process failure.",
                "Implement corrective measures.",
                "Improve customer communication.",
            ],
        )

    # 3d: Conduct alleged with only general evidence
    if has_conduct(
        reporting_conducts | process_conducts | fee_conducts | regulatory_conducts
    ) and all_general_allegations():
        if has_significant_missing_info():
            return TriageDecision(
                tier=Tier.NO_ACTION,
                primary_reason=ReasonCategory.INSUFFICIENT_EVIDENCE,
                secondary_tags=facts.secondary_tags,
                explanation=(
                    "Bank conduct alleged but evidence is entirely general "
                    "with significant information gaps. Cannot determine "
                    "error without specifics."
                ),
                confidence="medium",
                recommended_actions=[
                    f"Request from customer: "
                    f"{'; '.join(facts.missing_critical_info[:3])}.",
                ],
            )
        return TriageDecision(
            tier=Tier.NON_MONETARY_ACTION,
            primary_reason=ReasonCategory.PROCESS_OR_SYSTEM_FAILURE,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "Bank conduct alleged with limited specifics. "
                "Preliminary investigation warranted."
            ),
            confidence="low",
            recommended_actions=[
                "Conduct preliminary investigation.",
                "Request supporting documentation from customer.",
            ],
        )

    # =================================================================
    # RULE 4: TIER 0 — Catch-all
    # =================================================================

    # 4a: Customer misunderstanding
    if not facts.alleged_facts or all(
        f.conduct == AllegedBankConduct.OTHER_ALLEGED_CONDUCT
        for f in facts.alleged_facts
    ):
        return TriageDecision(
            tier=Tier.NO_ACTION,
            primary_reason=ReasonCategory.CUSTOMER_MISUNDERSTANDING,
            secondary_tags=facts.secondary_tags,
            explanation=(
                "No specific bank error conduct identified. "
                "Complaint may reflect misunderstanding of terms or process."
            ),
            confidence="low",
            recommended_actions=[
                "Send explanatory response to customer.",
            ],
        )

    # 4b: General fallback
    return TriageDecision(
        tier=Tier.NO_ACTION,
        primary_reason=ReasonCategory.INSUFFICIENT_EVIDENCE,
        secondary_tags=facts.secondary_tags,
        explanation=(
            "Unable to determine bank error from facts provided. "
            f"Missing: "
            f"{'; '.join(facts.missing_critical_info[:3]) or 'various specifics'}."
        ),
        confidence="low",
        recommended_actions=[
            "Request additional documentation from customer.",
        ],
    )


# ============================================================================
# SECTION 7 — STAGE 2 OPTION B: LLM-BASED DECISION (fallback/comparison)
# ============================================================================

STAGE_2_SYSTEM_PROMPT = """\
You are a CFPB complaint triage decision-maker.  You receive a structured
set of FACTS extracted from a consumer complaint — you do NOT see the
original complaint text.  This is intentional: your decision must be based
solely on the objective facts, not on how the complaint was written.

Your task: assign a remedy tier and primary reason category.

TIER DEFINITIONS:
- Tier 0 (NO_ACTION): No bank error identified, or complaint is about a
  non-bank entity with no bank nexus, or evidence is too sparse to evaluate.
- Tier 1 (NON_MONETARY_ACTION): Bank error or deficiency identified that
  requires correction (data fix, process improvement, compliance remediation)
  but no quantifiable financial loss to the customer.
- Tier 2 (MONETARY_COMPENSATION): Bank error identified AND quantifiable
  financial loss to the customer that warrants reimbursement.

DECISION RULES (apply strictly):

1. If primary_entity_type is NOT bank/lender/servicer/fintech AND
   bank_nexus is false → Tier 0, reason: not_bank_responsible.

2. If bank is involved but alleged_facts are empty or all evidence_strength
   is "not_stated" → Tier 0, reason: insufficient_evidence.

3. If bank is involved AND specific conduct is alleged AND monetary harm
   is established with specifics → Tier 2.

4. If bank is involved AND specific conduct is alleged AND no monetary
   harm OR harm is vague → Tier 1.

5. If bank is involved AND conduct is alleged but only as general
   allegations with significant missing info → Tier 0, reason:
   insufficient_evidence.

6. Never let the number of alleged facts alone determine tier — one
   well-evidenced fact outranks five vague allegations.

7. Never assume facts not present in the FactSet.

REASON CATEGORIES (use these exact values):
{reason_values}

SECONDARY TAGS (zero or more):
{tag_values}
"""

STAGE_2_USER_TEMPLATE = """\
Based on the following structured facts, determine the triage tier, primary
reason, and explanation.  Return ONLY valid JSON.

<facts>
{facts_json}
</facts>

<product>{product}</product>
<issue>{issue}</issue>

Return JSON:
{{
  "tier": <0, 1, or 2>,
  "primary_reason": "<ReasonCategory enum value>",
  "secondary_tags": ["<SecondaryReasonTag value>", ...],
  "explanation": "<1-3 sentence rationale based only on the facts above>",
  "confidence": "<high, medium, or low>",
  "recommended_actions": ["<action>", ...]
}}
"""


def build_stage_2_prompt(
    facts: FactSet,
    product: str = "",
    issue: str = "",
) -> Tuple[str, str]:
    """Build system and user messages for Stage 2 LLM decision."""
    system_msg = STAGE_2_SYSTEM_PROMPT.format(
        reason_values="\n".join(
            f"  - {e.value}: see ontology definition" for e in ReasonCategory
        ),
        tag_values=", ".join(e.value for e in SecondaryReasonTag),
    )
    user_msg = STAGE_2_USER_TEMPLATE.format(
        facts_json=json.dumps(asdict(facts), indent=2, default=str),
        product=product,
        issue=issue,
    )
    return system_msg, user_msg


# ============================================================================
# SECTION 8 — VALIDATION AND CONSISTENCY CHECKS
# ============================================================================

def validate_factset(facts: FactSet) -> List[str]:
    """Return a list of warnings/errors for a FactSet. Empty = valid."""
    issues: List[str] = []

    if not facts.alleged_facts and facts.harm_types:
        issues.append(
            "Harm types listed but no alleged facts — harm must be "
            "connected to specific conduct."
        )

    if (
        facts.primary_entity_type
        not in {
            EntityType.BANK_OR_LENDER,
            EntityType.MORTGAGE_SERVICER,
            EntityType.FINTECH_PAYMENT_PLATFORM,
        }
        and not facts.bank_nexus
        and facts.alleged_facts
    ):
        issues.append(
            "Alleged bank conduct listed but primary entity is not a bank "
            "and bank_nexus is False — review entity classification."
        )

    for i, fact in enumerate(facts.alleged_facts):
        if fact.evidence_strength == EvidenceStrength.SPECIFIC_WITH_DOCUMENTATION:
            if not fact.specific_date and not fact.specific_amount:
                issues.append(
                    f"Fact {i}: evidence rated 'specific_with_documentation' "
                    f"but no date or amount provided."
                )

    if (
        CustomerHarmType.NO_FINANCIAL_HARM_CLAIMED in facts.harm_types
        and len(facts.harm_types) > 1
    ):
        issues.append(
            "NO_FINANCIAL_HARM_CLAIMED is mutually exclusive with "
            "other harm types."
        )

    return issues


def validate_decision(decision: TriageDecision, facts: FactSet) -> List[str]:
    """Check decision consistency against the facts."""
    issues: List[str] = []

    if decision.tier == Tier.MONETARY_COMPENSATION:
        monetary_harms = {
            CustomerHarmType.FEES_OR_CHARGES_INCURRED,
            CustomerHarmType.FUNDS_LOST_OR_INACCESSIBLE,
            CustomerHarmType.FORCED_THIRD_PARTY_COSTS,
            CustomerHarmType.LOST_PROPERTY_OR_COLLATERAL,
            CustomerHarmType.CONSEQUENTIAL_FINANCIAL_LOSS,
        }
        if not (monetary_harms & set(facts.harm_types)):
            issues.append(
                "Tier 2 assigned but no monetary harm type in FactSet."
            )

    if decision.primary_reason == ReasonCategory.NOT_BANK_RESPONSIBLE:
        bank_types = {
            EntityType.BANK_OR_LENDER,
            EntityType.MORTGAGE_SERVICER,
            EntityType.FINTECH_PAYMENT_PLATFORM,
        }
        if facts.primary_entity_type in bank_types or facts.bank_nexus:
            issues.append(
                "NOT_BANK_RESPONSIBLE reason but bank involvement detected."
            )

    return issues


# ============================================================================
# SECTION 9 — PYDANTIC RESPONSE MODELS (instructor integration)
# ============================================================================

def _coerce_enum_value(value: Any, enum_cls: type, default: str) -> str:
    """Coerce an LLM output string to a valid enum value."""
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    valid = {e.value for e in enum_cls}
    if v in valid:
        return v
    return default


def _coerce_enum_list(
    values: Any,
    enum_cls: type,
    default: str,
) -> List[str]:
    """Coerce a list of LLM strings to valid enum values."""
    if not isinstance(values, list):
        return [default]
    return [_coerce_enum_value(v, enum_cls, default) for v in values]


class AllegedFactItem(BaseModel):
    """Pydantic model for a single alleged fact (Stage 1 LLM output)."""

    model_config = ConfigDict(use_enum_values=True)

    conduct: AllegedBankConduct
    evidence_strength: EvidenceStrength
    details: str = ""
    specific_amount: Optional[str] = None
    specific_date: Optional[str] = None
    specific_entity: Optional[str] = None

    @field_validator("conduct", mode="before")
    @classmethod
    def coerce_conduct(cls, v: Any) -> str:
        return _coerce_enum_value(
            v, AllegedBankConduct, AllegedBankConduct.OTHER_ALLEGED_CONDUCT.value
        )

    @field_validator("evidence_strength", mode="before")
    @classmethod
    def coerce_evidence(cls, v: Any) -> str:
        return _coerce_enum_value(
            v, EvidenceStrength, EvidenceStrength.GENERAL_ALLEGATION.value
        )


class FactSetResponse(BaseModel):
    """Pydantic model for Stage 1 LLM output (instructor validation)."""

    model_config = ConfigDict(use_enum_values=True)

    primary_entity_type: EntityType
    primary_entity_name: Optional[str] = None
    bank_nexus: bool = False
    bank_nexus_description: Optional[str] = None
    alleged_facts: List[AllegedFactItem] = Field(default_factory=list)
    harm_types: List[CustomerHarmType] = Field(default_factory=list)
    total_claimed_amount: Optional[str] = None
    implicated_regulations: List[RegulatoryFramework] = Field(default_factory=list)
    missing_critical_info: List[str] = Field(default_factory=list)
    secondary_tags: List[SecondaryReasonTag] = Field(default_factory=list)

    @field_validator("primary_entity_type", mode="before")
    @classmethod
    def coerce_entity_type(cls, v: Any) -> str:
        return _coerce_enum_value(v, EntityType, EntityType.UNKNOWN.value)

    @field_validator("harm_types", mode="before")
    @classmethod
    def coerce_harm_types(cls, v: Any) -> List[str]:
        return _coerce_enum_list(
            v, CustomerHarmType,
            CustomerHarmType.FINANCIAL_HARM_VAGUE_OR_UNQUANTIFIED.value,
        )

    @field_validator("implicated_regulations", mode="before")
    @classmethod
    def coerce_regulations(cls, v: Any) -> List[str]:
        return _coerce_enum_list(
            v, RegulatoryFramework,
            RegulatoryFramework.NONE_IDENTIFIED.value,
        )

    @field_validator("secondary_tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        result = []
        valid = {e.value for e in SecondaryReasonTag}
        for item in v:
            coerced = str(item).strip().lower().replace(" ", "_").replace("-", "_")
            if coerced in valid:
                result.append(coerced)
        return result


class DebiasedTriageResponse(BaseModel):
    """Pydantic model for Stage 2 LLM output (instructor validation)."""

    model_config = ConfigDict(use_enum_values=True)

    tier: Tier
    primary_reason: ReasonCategory
    secondary_tags: List[SecondaryReasonTag] = Field(default_factory=list)
    explanation: str = ""
    confidence: str = "medium"
    recommended_actions: List[str] = Field(default_factory=list)

    @field_validator("tier", mode="before")
    @classmethod
    def coerce_tier(cls, v: Any) -> int:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        return v if v in (0, 1, 2) else 0

    @field_validator("primary_reason", mode="before")
    @classmethod
    def coerce_reason(cls, v: Any) -> str:
        return _coerce_enum_value(
            v, ReasonCategory, ReasonCategory.INSUFFICIENT_EVIDENCE.value
        )

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> str:
        v = str(v).strip().lower()
        return v if v in ("high", "medium", "low") else "medium"

    @field_validator("secondary_tags", mode="before")
    @classmethod
    def coerce_tags(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        valid = {e.value for e in SecondaryReasonTag}
        return [
            s for item in v
            if (s := str(item).strip().lower().replace(" ", "_").replace("-", "_")) in valid
        ]


# ============================================================================
# SECTION 10 — PARSING (Pydantic → dataclass conversion)
# ============================================================================

def parse_factset(response: FactSetResponse) -> FactSet:
    """Convert a Pydantic FactSetResponse to a FactSet dataclass.

    With ``use_enum_values=True``, field values are stored as primitives
    (str/int), so we wrap them back into their Enum types here.
    """
    return FactSet(
        primary_entity_type=EntityType(response.primary_entity_type),
        primary_entity_name=response.primary_entity_name,
        bank_nexus=response.bank_nexus,
        bank_nexus_description=response.bank_nexus_description,
        alleged_facts=[
            AllegedFact(
                conduct=AllegedBankConduct(f.conduct),
                evidence_strength=EvidenceStrength(f.evidence_strength),
                details=f.details,
                specific_amount=f.specific_amount,
                specific_date=f.specific_date,
                specific_entity=f.specific_entity,
            )
            for f in response.alleged_facts
        ],
        harm_types=[CustomerHarmType(h) for h in response.harm_types],
        total_claimed_amount=response.total_claimed_amount,
        implicated_regulations=[
            RegulatoryFramework(r) for r in response.implicated_regulations
        ],
        missing_critical_info=response.missing_critical_info,
        secondary_tags=[
            SecondaryReasonTag(t) for t in response.secondary_tags
        ],
    )


def parse_triage_decision(response: DebiasedTriageResponse) -> TriageDecision:
    """Convert a Pydantic DebiasedTriageResponse to a TriageDecision dataclass."""
    return TriageDecision(
        tier=Tier(response.tier),
        primary_reason=ReasonCategory(response.primary_reason),
        secondary_tags=[
            SecondaryReasonTag(t) for t in response.secondary_tags
        ],
        explanation=response.explanation,
        confidence=response.confidence,
        recommended_actions=response.recommended_actions,
    )


# ============================================================================
# SECTION 11 — PIPELINE IMPLEMENTATION
# ============================================================================

def run_pipeline(
    complaint_text: str,
    case_id: str,
    model: str,
    llm_client: LLMClient,
    product: str = "",
    issue: str = "",
    use_llm_stage_2: bool = False,
) -> Tuple[Optional[FactSet], TriageDecision, List[str]]:
    """Full two-stage pipeline for a single complaint.

    Returns (factset, decision, warnings).  If Stage 1 fails, factset is
    None and decision defaults to Tier 0 / insufficient_evidence.
    """
    warnings: List[str] = []

    # --- Stage 1: Extract facts ---
    system_msg, user_msg = build_stage_1_prompt(complaint_text, product, issue)
    request = LLMRequest(
        request_id=LLMRequest.new_id(),
        case_id=case_id,
        model=model,
        task="fact_extraction",
        prompt_system=system_msg,
        prompt_user=user_msg,
        expected_schema=FactSetResponse,
    )
    response = llm_client.call_with_cache(request, complaint_text)

    if not response.success or response.response_parsed is None:
        decision = TriageDecision(
            tier=Tier.NO_ACTION,
            primary_reason=ReasonCategory.INSUFFICIENT_EVIDENCE,
            explanation="Fact extraction failed.",
            confidence="low",
        )
        warnings.append(f"Stage 1 failed: {response.error_message}")
        return None, decision, warnings

    facts = parse_factset(response.response_parsed)
    warnings.extend(validate_factset(facts))

    # --- Stage 2: Decide ---
    if use_llm_stage_2:
        sys2, usr2 = build_stage_2_prompt(facts, product, issue)
        req2 = LLMRequest(
            request_id=LLMRequest.new_id(),
            case_id=case_id,
            model=model,
            task="debiased_triage",
            prompt_system=sys2,
            prompt_user=usr2,
            expected_schema=DebiasedTriageResponse,
            variant="debiased",
        )
        cache_key_narrative = json.dumps(
            asdict(facts), sort_keys=True, default=str
        )
        resp2 = llm_client.call_with_cache(req2, cache_key_narrative)

        if resp2.success and resp2.response_parsed is not None:
            decision = parse_triage_decision(resp2.response_parsed)
        else:
            decision = heuristic_triage(facts)
            warnings.append(
                "Stage 2 LLM failed, fell back to heuristic triage."
            )
    else:
        decision = heuristic_triage(facts)

    warnings.extend(validate_decision(decision, facts))
    return facts, decision, warnings


def run_pipeline_batch(
    complaints: List[Complaint],
    model: str,
    llm_client: LLMClient,
    use_llm_stage_2: bool = False,
) -> List[Tuple[Optional[FactSet], TriageDecision, List[str]]]:
    """Full two-stage pipeline for a batch of complaints.

    Uses LLMClient.execute_batch for parallel Stage 1 extraction, then
    applies Stage 2 (heuristic or LLM) to each result.
    """
    if not complaints:
        return []

    # --- Stage 1: batch fact extraction ---
    requests: List[LLMRequest] = []
    narratives: List[str] = []
    for c in complaints:
        system_msg, user_msg = build_stage_1_prompt(
            c.narrative, c.product, c.issue
        )
        req = LLMRequest(
            request_id=LLMRequest.new_id(),
            case_id=c.case_id,
            model=model,
            task="fact_extraction",
            prompt_system=system_msg,
            prompt_user=user_msg,
            expected_schema=FactSetResponse,
        )
        requests.append(req)
        narratives.append(c.narrative)

    responses = llm_client.execute_batch(
        requests, narratives, desc=f"Debiased Stage 1 [{model}] Fact extraction"
    )

    # Parse Stage 1 results
    factsets: List[Optional[FactSet]] = []
    stage1_warnings: List[List[str]] = []
    for resp in responses:
        if resp.success and resp.response_parsed is not None:
            try:
                fs = parse_factset(resp.response_parsed)
                factsets.append(fs)
                stage1_warnings.append(validate_factset(fs))
            except (ValueError, KeyError) as exc:
                logger.warning("FactSet parse error for %s: %s", resp.case_id, exc)
                factsets.append(None)
                stage1_warnings.append([f"FactSet parse error: {exc}"])
        else:
            factsets.append(None)
            stage1_warnings.append(
                [f"Stage 1 failed: {resp.error_message}"]
            )

    # --- Stage 2 ---
    if use_llm_stage_2:
        # Build Stage 2 LLM requests for successful extractions
        s2_requests: List[LLMRequest] = []
        s2_narratives: List[str] = []
        s2_indices: List[int] = []  # map back to original index
        for i, (c, fs) in enumerate(zip(complaints, factsets)):
            if fs is None:
                continue
            sys2, usr2 = build_stage_2_prompt(fs, c.product, c.issue)
            req2 = LLMRequest(
                request_id=LLMRequest.new_id(),
                case_id=c.case_id,
                model=model,
                task="debiased_triage",
                prompt_system=sys2,
                prompt_user=usr2,
                expected_schema=DebiasedTriageResponse,
                variant="debiased",
            )
            s2_requests.append(req2)
            s2_narratives.append(
                json.dumps(asdict(fs), sort_keys=True, default=str)
            )
            s2_indices.append(i)

        s2_responses = llm_client.execute_batch(
            s2_requests, s2_narratives,
            desc=f"Debiased Stage 2 [{model}] LLM triage",
        )

        # Map Stage 2 responses back
        s2_decisions: Dict[int, Optional[TriageDecision]] = {}
        s2_extra_warnings: Dict[int, List[str]] = {}
        for idx, resp2 in zip(s2_indices, s2_responses):
            if resp2.success and resp2.response_parsed is not None:
                try:
                    s2_decisions[idx] = parse_triage_decision(resp2.response_parsed)
                    s2_extra_warnings[idx] = []
                except (ValueError, KeyError) as exc:
                    s2_decisions[idx] = None
                    s2_extra_warnings[idx] = [f"Stage 2 parse error: {exc}"]
            else:
                s2_decisions[idx] = None
                s2_extra_warnings[idx] = [
                    "Stage 2 LLM failed, fell back to heuristic triage."
                ]

    # Assemble final results
    results: List[Tuple[Optional[FactSet], TriageDecision, List[str]]] = []
    for i, (fs, w) in enumerate(zip(factsets, stage1_warnings)):
        if fs is None:
            decision = TriageDecision(
                tier=Tier.NO_ACTION,
                primary_reason=ReasonCategory.INSUFFICIENT_EVIDENCE,
                explanation="Fact extraction failed.",
                confidence="low",
            )
            results.append((None, decision, w))
            continue

        if use_llm_stage_2:
            decision = s2_decisions.get(i)
            extra_w = s2_extra_warnings.get(i, [])
            if decision is None:
                decision = heuristic_triage(fs)
                extra_w.append("Fell back to heuristic triage.")
            w = w + extra_w
        else:
            decision = heuristic_triage(fs)

        w.extend(validate_decision(decision, fs))
        results.append((fs, decision, w))

    return results


def run_debiased_analysis(
    complaints: List[Complaint],
    models: List[str],
    llm_client: LLMClient,
    use_llm_stage_2: bool = False,
) -> List[DebiasedResult]:
    """Run the debiased triage pipeline across multiple models.

    Iterates ``for model in models`` (matching the experiment's multi-model
    pattern), running batch fact extraction (Stage 1) and triage (Stage 2)
    for all complaints under each model.

    All LLM calls are cached — ``execute_batch`` calls ``call_with_cache``
    internally, keyed on ``(model, task, prompt_version, narrative)``.
    Re-running with the same complaints/models hits the disk cache.

    Parameters
    ----------
    complaints : list of Complaint
        Complaints to process (from data_loader).
    models : list of str
        Model identifiers to run, e.g. ``["gpt-4o-mini", "gpt-5-mini"]``.
    llm_client : LLMClient
        Configured client with cache attached.
    use_llm_stage_2 : bool
        If True, use the LLM prompt for Stage 2 instead of heuristics.

    Returns
    -------
    list of DebiasedResult
        Flat list of results, one per (complaint, model) pair.
    """
    all_results: List[DebiasedResult] = []

    for model in models:
        logger.info(
            "Debiased triage: processing %d complaints with %s",
            len(complaints), model,
        )
        batch_results = run_pipeline_batch(
            complaints, model, llm_client,
            use_llm_stage_2=use_llm_stage_2,
        )

        for complaint, (factset, decision, warnings) in zip(
            complaints, batch_results
        ):
            all_results.append(DebiasedResult(
                case_id=complaint.case_id,
                model=model,
                factset=factset,
                decision=decision,
                tier=decision.tier.value,
                primary_reason=decision.primary_reason.value,
                warnings=warnings,
            ))

    logger.info(
        "Debiased triage complete: %d results across %d model(s)",
        len(all_results), len(models),
    )
    return all_results


# ============================================================================
# SECTION 12 — BIAS DETECTION UTILITIES
# ============================================================================

def compare_factsets(fs1: FactSet, fs2: FactSet) -> dict:
    """Compare two FactSets (e.g. from original vs rewritten complaint) to
    detect extraction inconsistencies that would indicate residual bias.

    Returns a dict of differences. If the pipeline is unbiased, two
    paraphrases of the same complaint should produce near-identical FactSets.
    """
    diffs: dict = {}

    if fs1.primary_entity_type != fs2.primary_entity_type:
        diffs["entity_type"] = (
            fs1.primary_entity_type.value,
            fs2.primary_entity_type.value,
        )

    if fs1.bank_nexus != fs2.bank_nexus:
        diffs["bank_nexus"] = (fs1.bank_nexus, fs2.bank_nexus)

    conducts_1 = {f.conduct.value for f in fs1.alleged_facts}
    conducts_2 = {f.conduct.value for f in fs2.alleged_facts}
    if conducts_1 != conducts_2:
        diffs["conduct_diff"] = {
            "only_in_first": conducts_1 - conducts_2,
            "only_in_second": conducts_2 - conducts_1,
        }

    harms_1 = {h.value for h in fs1.harm_types}
    harms_2 = {h.value for h in fs2.harm_types}
    if harms_1 != harms_2:
        diffs["harm_diff"] = {
            "only_in_first": harms_1 - harms_2,
            "only_in_second": harms_2 - harms_1,
        }

    strengths_1 = sorted(f.evidence_strength.value for f in fs1.alleged_facts)
    strengths_2 = sorted(f.evidence_strength.value for f in fs2.alleged_facts)
    if strengths_1 != strengths_2:
        diffs["evidence_strength_diff"] = (strengths_1, strengths_2)

    return diffs


def bias_audit_report(
    pairs: List[Tuple[FactSet, FactSet, TriageDecision, TriageDecision]],
) -> dict:
    """Aggregate bias metrics across a set of complaint pairs.

    Each tuple: (facts_original, facts_rewritten, decision_original,
    decision_rewritten).

    Returns summary statistics on extraction and decision consistency.
    """
    n = len(pairs)
    if n == 0:
        return {"total_pairs": 0}

    entity_mismatches = 0
    nexus_mismatches = 0
    conduct_mismatches = 0
    harm_mismatches = 0
    tier_mismatches = 0
    reason_mismatches = 0

    for fs1, fs2, d1, d2 in pairs:
        diffs = compare_factsets(fs1, fs2)
        if "entity_type" in diffs:
            entity_mismatches += 1
        if "bank_nexus" in diffs:
            nexus_mismatches += 1
        if "conduct_diff" in diffs:
            conduct_mismatches += 1
        if "harm_diff" in diffs:
            harm_mismatches += 1
        if d1.tier != d2.tier:
            tier_mismatches += 1
        if d1.primary_reason != d2.primary_reason:
            reason_mismatches += 1

    return {
        "total_pairs": n,
        "extraction_consistency": {
            "entity_type_match_rate": (n - entity_mismatches) / n,
            "bank_nexus_match_rate": (n - nexus_mismatches) / n,
            "conduct_match_rate": (n - conduct_mismatches) / n,
            "harm_match_rate": (n - harm_mismatches) / n,
        },
        "decision_consistency": {
            "tier_match_rate": (n - tier_mismatches) / n,
            "reason_match_rate": (n - reason_mismatches) / n,
        },
    }
