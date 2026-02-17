"""Matched-guise slot-preservation audit package."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple


class SlotStatus(str, Enum):
    """Preservation status for a slot family comparison."""
    PRESERVED_ABSENCE = "PRESERVED_ABSENCE"
    PRESERVED_PRESENT = "PRESERVED_PRESENT"
    OMISSION = "OMISSION"
    ADDITION = "ADDITION"
    MUTATION = "MUTATION"
    MUTATION_POLARITY_CONTRADICTION = "MUTATION_POLARITY_CONTRADICTION"


@dataclass
class NumericSlot:
    subtype: str  # money, percent, apr, count, bound
    value: float
    unit: str = ""
    operator: str = ""
    raw: str = ""

    def canonical(self) -> str:
        if self.subtype == "money":
            return f"money:{self.value:.2f}"
        if self.subtype in ("percent", "apr"):
            return f"{self.subtype}:{self.value:.2f}"
        if self.subtype == "count":
            return f"count:{int(self.value)}:{self.unit}"
        if self.subtype == "bound":
            return f"bound:{self.operator}:{self.value:.2f}"
        return f"{self.subtype}:{self.value}"


@dataclass
class TemporalSlot:
    subtype: str  # duration, frequency, year
    value: str
    unit: str = ""
    relation: str = ""
    raw: str = ""

    def canonical(self) -> str:
        parts = [self.subtype, self.value]
        if self.unit:
            parts.append(self.unit)
        if self.relation:
            parts.append(self.relation)
        return ":".join(parts)


@dataclass
class SlotComparisonResult:
    family: str
    status: SlotStatus
    source_canonicals: FrozenSet[str]
    rewrite_canonicals: FrozenSet[str]
    has_contradiction: bool = False


# Default weights for pair-level scoring
DEFAULT_WEIGHTS: Dict[str, float] = {
    "numeric": 0.30,
    "temporal": 0.25,
    "polarity": 0.30,
    "anchor": 0.15,
}

CONTRADICTION_PENALTY: float = 0.30
