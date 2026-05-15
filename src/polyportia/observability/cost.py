"""Cost estimation. Prefer ActualModel metadata, fall back to LiteLLM tables."""

from __future__ import annotations

from polyportia.config.models import ActualModel
from polyportia.providers.base import TokenUsage


def estimate_cost_usd(actual: ActualModel, usage: TokenUsage | None) -> float | None:
    if usage is None:
        return None
    pt = usage.prompt_tokens or 0
    ct = usage.completion_tokens or 0
    in_rate = actual.input_cost_per_1m_tokens
    out_rate = actual.output_cost_per_1m_tokens
    if in_rate is not None and out_rate is not None:
        return (pt * in_rate + ct * out_rate) / 1_000_000
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=actual.id,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        return prompt_cost + completion_cost
    except Exception:
        return None
