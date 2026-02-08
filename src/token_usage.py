"""Token usage accounting and cost estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelPricing:
    prompt_per_1m: float
    completion_per_1m: float


# Update these rates as pricing changes. Values are USD per 1M tokens.
PRICING_PER_1M_TOKENS: Dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(prompt_per_1m=0.15, completion_per_1m=0.60),
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Tuple[Optional[float], Optional[ModelPricing]]:
    """Estimate USD cost for the given model and token counts."""
    pricing = PRICING_PER_1M_TOKENS.get(model)
    if pricing is None:
        return None, None
    prompt_cost = (prompt_tokens / 1_000_000) * pricing.prompt_per_1m
    completion_cost = (completion_tokens / 1_000_000) * pricing.completion_per_1m
    return prompt_cost + completion_cost, pricing
