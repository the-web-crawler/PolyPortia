"""Normalised result envelopes returned from provider calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_litellm(cls, raw: Any) -> TokenUsage | None:
        if raw is None:
            return None
        try:
            pt = getattr(raw, "prompt_tokens", None) or (
                raw["prompt_tokens"] if isinstance(raw, dict) else None
            )
            ct = getattr(raw, "completion_tokens", None) or (
                raw["completion_tokens"] if isinstance(raw, dict) else None
            )
            tt = getattr(raw, "total_tokens", None) or (
                raw["total_tokens"] if isinstance(raw, dict) else None
            )
        except (AttributeError, TypeError):
            return None
        return cls(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)


@dataclass
class ProviderResult:
    """Result of a single underlying LiteLLM call.

    `raw` is the litellm ModelResponse so the HTTP layer can re-serialise to
    OpenAI shape without translating fields ourselves.
    """

    model_id: str
    content: str
    usage: TokenUsage | None = None
    raw: Any = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
