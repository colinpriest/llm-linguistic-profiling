"""Pydantic models and dataclasses for the LLM fairness experiment."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
    timestamp: datetime = field(default_factory=datetime.now)
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
        v = str(v).strip().lower()
        if v in ("young", "old"):
            return v
        if v in ("middle", "middle-aged", "adult"):
            return "young"
        if v in ("elderly", "senior"):
            return "old"
        return "young"  # default fallback

    @field_validator("geographic", mode="before")
    @classmethod
    def coerce_geographic(cls, v: Any) -> str:
        v = str(v).strip().lower()
        if v in ("urban", "suburban", "rural"):
            return v
        return "suburban"  # default fallback

    @field_validator("education", mode="before")
    @classmethod
    def coerce_education(cls, v: Any) -> str:
        v = str(v).strip().lower()
        if v in ("high_school", "college"):
            return v
        if v in ("highschool", "high school", "hs", "secondary"):
            return "high_school"
        if v in ("university", "graduate", "postgraduate", "bachelor", "masters", "phd"):
            return "college"
        return "college"  # default fallback

    @field_validator("socioeconomic", mode="before")
    @classmethod
    def coerce_socioeconomic(cls, v: Any) -> str:
        v = str(v).strip().lower()
        if v in ("low", "middle", "high"):
            return v
        if v in ("lower", "poverty", "working_class", "working class"):
            return "low"
        if v in ("upper", "wealthy", "affluent", "upper_middle", "upper middle"):
            return "high"
        return "middle"  # default fallback

    @field_validator("gender", mode="before")
    @classmethod
    def coerce_gender(cls, v: Any) -> str:
        v = str(v).strip().lower()
        if v in ("male", "female"):
            return v
        if v in ("man", "boy", "m"):
            return "male"
        if v in ("woman", "girl", "f"):
            return "female"
        return "male"  # default fallback

    @field_validator("esl", mode="before")
    @classmethod
    def coerce_esl(cls, v: Any) -> str:
        v = str(v).strip().lower()
        if v in ("native", "esl"):
            return v
        if v in ("native_speaker", "native speaker", "l1", "first_language", "no"):
            return "native"
        if v in ("non-native", "non_native", "l2", "second_language", "yes"):
            return "esl"
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
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        if v not in (0, 1, 2):
            return 0
        return v


class PersonaRewriteResponse(BaseModel):
    rewritten_complaint: str
    style_changes_made: List[str] = Field(default_factory=list)


class NeutralizeResponse(BaseModel):
    neutralized_complaint: str
    markers_removed: List[str] = Field(default_factory=list)


class CommentaryResponse(BaseModel):
    commentary: str = Field(
        description=(
            "The complete written analysis paragraph(s) with specific findings, "
            "statistics, and conclusions. Must contain actual analytical claims "
            "about the data (e.g. 'The chi-squared test was significant at p=0.03'), "
            "NOT a description of what analysis to perform."
        )
    )


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
