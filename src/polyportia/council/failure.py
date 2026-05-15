"""Council-level partial-failure policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polyportia.config.models import FailurePolicy
from polyportia.providers.base import ProviderResult


@dataclass
class MemberOutcome:
    member_repr: str
    result: ProviderResult | None
    error: BaseException | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None


class CouncilFailureError(RuntimeError):
    """Raised when the failure policy says we cannot proceed."""

    def __init__(self, message: str, *, outcomes: list[MemberOutcome]) -> None:
        super().__init__(message)
        self.outcomes = outcomes


def apply_failure_policy(
    outcomes: list[MemberOutcome],
    policy: FailurePolicy,
) -> list[MemberOutcome]:
    """Validate ``outcomes`` against ``policy`` and return them, or raise.

    Returns the same list unchanged on success so callers can chain.
    """
    successes = sum(1 for o in outcomes if o.ok)
    total = len(outcomes)

    if policy.on_failure == "fail" and successes < total:
        raise CouncilFailureError(
            f"failure policy 'fail': {total - successes}/{total} members failed",
            outcomes=outcomes,
        )

    min_required: int | None = None
    if policy.min_success is not None:
        min_required = policy.min_success
    if policy.min_success_fraction is not None:
        frac_required = max(1, int(round(policy.min_success_fraction * total)))
        min_required = frac_required if min_required is None else max(min_required, frac_required)

    if min_required is not None and successes < min_required:
        raise CouncilFailureError(
            f"council requires {min_required} successful members, got {successes}/{total}",
            outcomes=outcomes,
        )
    return outcomes


def outcomes_to_array(outcomes: list[MemberOutcome]) -> list[dict[str, Any]]:
    """Serialise outcomes for the array-envelope response."""
    out: list[dict[str, Any]] = []
    for o in outcomes:
        entry: dict[str, Any] = {"member": o.member_repr}
        if o.ok and o.result is not None:
            entry.update(
                {
                    "model": o.result.model_id,
                    "content": o.result.content,
                    "latency_ms": o.result.latency_ms,
                    "finish_reason": o.result.finish_reason,
                }
            )
            if o.result.usage is not None:
                entry["usage"] = {
                    "prompt_tokens": o.result.usage.prompt_tokens,
                    "completion_tokens": o.result.usage.completion_tokens,
                    "total_tokens": o.result.usage.total_tokens,
                }
        else:
            entry["error"] = str(o.error) if o.error else "unknown error"
        out.append(entry)
    return out
