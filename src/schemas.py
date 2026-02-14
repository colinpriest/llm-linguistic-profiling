"""Pydantic models and dataclasses for the LLM fairness experiment."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persona helpers
# ---------------------------------------------------------------------------

AGE_VALUES = ["young", "old"]
GEO_VALUES = ["urban", "suburban", "rural"]
EDU_VALUES = ["high_school", "college"]
SES_VALUES = ["low", "middle", "high"]
GENDER_VALUES = ["male", "female"]
ESL_VALUES = ["native", "esl"]

# Dimensions used in persona estimation (6 dimensions including gender and ESL)
ESTIMATE_DIMS = ["age", "geographic", "education", "socioeconomic", "gender", "esl"]

# Dimensions used in causal rewrites (4 dimensions, excluding gender and ESL)
CAUSAL_DIMS = ["age", "geographic", "education", "socioeconomic"]


def generate_persona_grid() -> List[Dict[str, str]]:
    """Generate the full 36-persona Cartesian product."""
    return [
        {"age": a, "geo": g, "edu": e, "ses": s}
        for a in AGE_VALUES
        for g in GEO_VALUES
        for e in EDU_VALUES
        for s in SES_VALUES
    ]


def persona_label(age: str, geo: str, edu: str, ses: str) -> str:
    return f"{age}_{geo}_{edu}_{ses}"


def parse_persona_label(label: str) -> Optional[Dict[str, str]]:
    """Parse a persona label back into its dimension values.

    Handles multi-word values like ``high_school`` by matching against known
    dimension values rather than naive underscore splitting.

    Returns ``None`` if the label cannot be parsed.
    """
    if not isinstance(label, str):
        return None
    for a in AGE_VALUES:
        if not label.startswith(a + "_"):
            continue
        rest = label[len(a) + 1:]
        for g in GEO_VALUES:
            if not rest.startswith(g + "_"):
                continue
            rest2 = rest[len(g) + 1:]
            for e in EDU_VALUES:
                if not rest2.startswith(e + "_"):
                    continue
                s = rest2[len(e) + 1:]
                if s in SES_VALUES:
                    return {"age": a, "geographic": g, "education": e, "socioeconomic": s}
    return None


PERSONA_GRID = generate_persona_grid()
PERSONA_LABELS = [persona_label(**p) for p in PERSONA_GRID]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Complaint:
    case_id: str
    complaint_id: int
    narrative: str
    product: str
    issue: str
    date_received: str
    year: str
    product_category: str
    issue_category: str
    monetary_severity: str = "unknown"   # low / medium / high / unknown
    issue_severity: str = "medium"       # low / medium / high


# ---------------------------------------------------------------------------
# LLM request / response
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    request_id: str
    case_id: str
    model: str
    task: str  # persona_estimate | remedy_decision | persona_rewrite | neutralize
    prompt_system: str
    prompt_user: str
    expected_schema: type  # Pydantic model class
    variant: str = "baseline"  # persona label, "baseline", or "neutralized"

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]


@dataclass
class LLMResponse:
    request_id: str
    case_id: str
    model: str
    task: str
    success: bool
    response_raw: Optional[str] = None
    response_parsed: Optional[Any] = None
    error_message: Optional[str] = None
    latency_ms: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempt_number: int = 1
    repair_applied: bool = False


# ---------------------------------------------------------------------------
# Pydantic schemas for LLM JSON responses
# ---------------------------------------------------------------------------

class PersonaEstimateResponse(BaseModel):
    age: Literal["young", "old"]
    geographic: Literal["urban", "suburban", "rural"]
    education: Literal["high_school", "college"]
    socioeconomic: Literal["low", "middle", "high"]
    gender: Literal["male", "female"]
    esl: Literal["native", "esl"]
    confidence: float = Field(ge=0.0, le=1.0)
    linguistic_markers: List[str] = Field(default_factory=list)

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> str:
        original = v
        v = str(v).strip().lower()
        if v in ("young", "old"):
            return v
        if v in ("middle", "middle-aged", "adult"):
            logger.debug("PersonaEstimate: coerced age '%s' -> 'young'", original)
            return "young"
        if v in ("elderly", "senior"):
            logger.debug("PersonaEstimate: coerced age '%s' -> 'old'", original)
            return "old"
        logger.debug("PersonaEstimate: coerced age '%s' -> 'young'", original)
        return "young"  # default fallback

    @field_validator("geographic", mode="before")
    @classmethod
    def coerce_geographic(cls, v: Any) -> str:
        original = v
        v = str(v).strip().lower()
        if v in ("urban", "suburban", "rural"):
            return v
        logger.debug("PersonaEstimate: coerced geographic '%s' -> 'suburban'", original)
        return "suburban"  # default fallback

    @field_validator("education", mode="before")
    @classmethod
    def coerce_education(cls, v: Any) -> str:
        original = v
        v = str(v).strip().lower()
        if v in ("high_school", "college"):
            return v
        if v in ("highschool", "high school", "hs", "secondary"):
            logger.debug("PersonaEstimate: coerced education '%s' -> 'high_school'", original)
            return "high_school"
        if v in ("university", "graduate", "postgraduate", "bachelor", "masters", "phd"):
            logger.debug("PersonaEstimate: coerced education '%s' -> 'college'", original)
            return "college"
        logger.debug("PersonaEstimate: coerced education '%s' -> 'college'", original)
        return "college"  # default fallback

    @field_validator("socioeconomic", mode="before")
    @classmethod
    def coerce_socioeconomic(cls, v: Any) -> str:
        original = v
        v = str(v).strip().lower()
        if v in ("low", "middle", "high"):
            return v
        if v in ("lower", "poverty", "working_class", "working class"):
            logger.debug("PersonaEstimate: coerced socioeconomic '%s' -> 'low'", original)
            return "low"
        if v in ("upper", "wealthy", "affluent", "upper_middle", "upper middle"):
            logger.debug("PersonaEstimate: coerced socioeconomic '%s' -> 'high'", original)
            return "high"
        logger.debug("PersonaEstimate: coerced socioeconomic '%s' -> 'middle'", original)
        return "middle"  # default fallback

    @field_validator("gender", mode="before")
    @classmethod
    def coerce_gender(cls, v: Any) -> str:
        original = v
        v = str(v).strip().lower()
        if v in ("male", "female"):
            return v
        if v in ("man", "boy", "m"):
            logger.debug("PersonaEstimate: coerced gender '%s' -> 'male'", original)
            return "male"
        if v in ("woman", "girl", "f"):
            logger.debug("PersonaEstimate: coerced gender '%s' -> 'female'", original)
            return "female"
        logger.debug("PersonaEstimate: coerced gender '%s' -> 'male'", original)
        return "male"  # default fallback

    @field_validator("esl", mode="before")
    @classmethod
    def coerce_esl(cls, v: Any) -> str:
        original = v
        v = str(v).strip().lower()
        if v in ("native", "esl"):
            return v
        if v in ("native_speaker", "native speaker", "l1", "first_language", "no"):
            logger.debug("PersonaEstimate: coerced esl '%s' -> 'native'", original)
            return "native"
        if v in ("non-native", "non_native", "l2", "second_language", "yes"):
            logger.debug("PersonaEstimate: coerced esl '%s' -> 'esl'", original)
            return "esl"
        logger.debug("PersonaEstimate: coerced esl '%s' -> 'native'", original)
        return "native"  # default fallback

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))


class RemedyDecisionResponse(BaseModel):
    tier: Literal[0, 1, 2]
    confidence: Literal["confident_no_action", "confident_action_needed", "need_more_info"]
    reasoning: str = ""
    information_needed: Optional[str] = None

    @field_validator("tier", mode="before")
    @classmethod
    def coerce_tier(cls, v: Any) -> int:
        original = v
        try:
            v = int(v)
        except (TypeError, ValueError):
            logger.debug("RemedyDecision: coerced tier %s -> 0", original)
            return 0
        if v not in (0, 1, 2):
            logger.debug("RemedyDecision: coerced tier %s -> 0", original)
            return 0
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> str:
        if isinstance(v, str):
            v_lower = v.strip().lower().replace(" ", "_").replace("-", "_")
            valid = {"confident_no_action", "confident_action_needed", "need_more_info"}
            if v_lower in valid:
                return v_lower
            # Map common alternatives
            if v_lower in ("low", "none", "no_action", "no_info", "n/a"):
                return "confident_no_action"
            if v_lower in ("medium", "high", "action_needed", "uncertain", "unsure"):
                return "confident_action_needed"
            if v_lower in ("need_info", "needs_more_info", "more_info",
                           "need_more_information"):
                return "need_more_info"
        logger.debug("RemedyDecision: coerced confidence %s -> 'confident_action_needed'", v)
        return "confident_action_needed"


class PersonaRewriteResponse(BaseModel):
    rewritten_complaint: str = Field(min_length=1)
    style_changes_made: List[str] = Field(default_factory=list)


class NeutralizeResponse(BaseModel):
    neutralized_complaint: str = Field(min_length=1)
    markers_removed: List[str] = Field(default_factory=list)


class DollarAmountResponse(BaseModel):
    amounts_mentioned: Optional[List[float]] = Field(
        default=None,
        description="List of dollar values found in the complaint",
    )
    primary_amount: Optional[float] = Field(
        default=None,
        description="The main disputed/affected amount (or null if unclear)",
    )
    amount_type: Optional[str] = Field(
        default=None,
        description="Category of the primary amount",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence in the extraction",
    )

    @field_validator("amount_type", mode="before")
    @classmethod
    def coerce_amount_type(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip().lower()
        allowed = {"fee", "balance", "payment", "damage", "other"}
        if v in allowed:
            return v
        mapping = {
            "refund": "payment",
            "reimbursement": "payment",
            "charge": "fee",
            "interest": "fee",
            "penalty": "fee",
            "overdraft": "fee",
            "debt": "balance",
            "loan": "balance",
            "credit": "balance",
            "mortgage": "balance",
            "transfer": "payment",
            "deposit": "payment",
            "withdrawal": "payment",
            "transaction": "payment",
            "loss": "damage",
            "fraud": "damage",
            "theft": "damage",
            "overcharge": "fee",
        }
        return mapping.get(v, "other")

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip().lower()
        if v in ("high", "medium", "low"):
            return v
        return "medium"

    @field_validator("primary_amount", mode="before")
    @classmethod
    def coerce_primary_amount(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return abs(float(v))
        except (TypeError, ValueError):
            return None

    @field_validator("amounts_mentioned", mode="before")
    @classmethod
    def coerce_amounts_mentioned(cls, v: Any) -> Optional[List[float]]:
        if v is None:
            return None
        if not isinstance(v, list):
            return None
        result = []
        for item in v:
            try:
                result.append(abs(float(item)))
            except (TypeError, ValueError):
                continue
        return result if result else None


@dataclass
class DollarAmount:
    case_id: str
    amounts_mentioned: Optional[List[float]]
    primary_amount: Optional[float]
    amount_type: Optional[str]
    confidence: Optional[str]


class CommentaryResponse(BaseModel):
    commentary: str = Field(
        description=(
            "The complete written analysis paragraph(s) with specific findings, "
            "statistics, and conclusions. Must contain actual analytical claims "
            "about the data (e.g. 'The chi-squared test was significant at p=0.03'), "
            "NOT a description of what analysis to perform."
        )
    )


class TextQualityRatingResponse(BaseModel):
    """LLM-rated subjective text quality on 1-5 Likert scale."""

    convincingness: int = Field(ge=1, le=5, description="How convincing the complaint sounds (1=not at all, 5=very)")
    persuasiveness: int = Field(ge=1, le=5, description="How persuasive the complaint is (1=not at all, 5=very)")
    evidence_blame: int = Field(ge=1, le=5, description="Clarity of evidence for which party is to blame (1=unclear, 5=very clear)")
    evidence_harm: int = Field(ge=1, le=5, description="Clarity of evidence for harms suffered (1=unclear, 5=very clear)")
    confidence: int = Field(ge=1, le=5, description="Confidence conveyed by the complainant (1=not confident, 5=very confident)")
    legal_knowledge: int = Field(ge=1, le=5, description="Knowledge of legal/regulatory environment (1=none apparent, 5=extensive)")
    formality: int = Field(ge=1, le=5, description="Formality of writing register (1=very informal/casual, 5=very formal/professional)")
    coherence: int = Field(ge=1, le=5, description="Logical organization and structure (1=disorganized/rambling, 5=clear logical flow)")
    specificity: int = Field(ge=1, le=5, description="Density of concrete facts, dates, amounts, names (1=vague, 5=highly specific)")
    financial_literacy: int = Field(ge=1, le=5, description="Understanding of financial products and terminology (1=none, 5=expert-level)")
    emotional_restraint: int = Field(ge=1, le=5, description="Measured tone vs emotional charge (1=highly emotional, 5=calm and measured)")
    grammar_quality: int = Field(ge=1, le=5, description="Grammar, spelling, and writing mechanics (1=many errors, 5=near-perfect)")
    reasoning: str = Field(default="", description="Brief justification for the ratings")

    @field_validator(
        "convincingness", "persuasiveness", "evidence_blame",
        "evidence_harm", "confidence", "legal_knowledge",
        "formality", "coherence", "specificity",
        "financial_literacy", "emotional_restraint", "grammar_quality",
        mode="before",
    )
    @classmethod
    def coerce_likert(cls, v: Any) -> int:
        """Clamp any numeric value to [1, 5]; default to 3 on parse failure."""
        try:
            v = int(round(float(v)))
        except (TypeError, ValueError):
            return 3
        return max(1, min(5, v))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PersonaEstimate:
    case_id: str
    model: str
    age: str
    geographic: str
    education: str
    socioeconomic: str
    gender: str
    esl: str
    confidence: float
    linguistic_markers: List[str]
    persona_label: str


@dataclass
class RemedyDecision:
    case_id: str
    model: str
    variant: str  # "baseline", "neutralized", or persona label
    tier: int
    confidence: str
    reasoning: str
    information_needed: Optional[str]


@dataclass
class PersonaRewrite:
    case_id: str
    model: str
    original_narrative: str
    persona_label: str
    rewritten_narrative: str
    style_changes_made: List[str]


@dataclass
class NeutralizedComplaint:
    case_id: str
    model: str
    original_narrative: str
    neutralized_narrative: str
    markers_removed: List[str]


@dataclass
class TextQualityRating:
    case_id: str
    model: str
    variant: str  # "original", persona label, or "neutralized"
    convincingness: int
    persuasiveness: int
    evidence_blame: int
    evidence_harm: int
    confidence: int
    legal_knowledge: int
    formality: int
    coherence: int
    specificity: int
    financial_literacy: int
    emotional_restraint: int
    grammar_quality: int


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------

@dataclass
class TextFeatures:
    case_id: str
    variant: str  # "original", "neutralized", or persona label

    # Length
    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0

    # Complexity
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    smog_index: float = 0.0
    ari: float = 0.0

    # Vocabulary diversity
    type_token_ratio: float = 0.0
    hapax_ratio: float = 0.0
    mtld: float = 0.0
    hdd: float = 0.0

    # Sentiment
    polarity: float = 0.0
    subjectivity: float = 0.0
    vader_compound: float = 0.0
    vader_pos: float = 0.0
    vader_neg: float = 0.0
    vader_neu: float = 0.0
    emotional_intensity: float = 0.0

    # Topics
    topic_distribution: List[float] = field(default_factory=lambda: [0.0] * 10)
    dominant_topic: int = 0
    topic_coherence: float = 0.0


# ---------------------------------------------------------------------------
# Repair statistics
# ---------------------------------------------------------------------------

@dataclass
class RepairStats:
    total_requests: int = 0
    successful_first_attempt: int = 0
    successful_after_repair: int = 0
    successful_after_retry: int = 0
    failed_after_all_attempts: int = 0
    repair_strategies_used: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    api_key: str
    models: List[str]
    temperature: float = 0.0
    max_tokens: int = 1000
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
