"""Budget-related error types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CostBreakdownEntry:
    model_id: str
    calls: int
    input_tokens_est: int
    output_tokens_est: int
    cost_usd: float


@dataclass
class CostEstimate:
    total_usd: float
    breakdown: list[CostBreakdownEntry]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_usd": self.total_usd,
            "breakdown": [
                {
                    "model": e.model_id,
                    "calls": e.calls,
                    "input_tokens_est": e.input_tokens_est,
                    "output_tokens_est": e.output_tokens_est,
                    "cost_usd_est": e.cost_usd,
                }
                for e in self.breakdown
            ],
            "notes": self.notes,
        }


class BudgetExceededError(Exception):
    """Raised when a request's predicted or actual cost crosses the budget."""

    def __init__(
        self,
        message: str,
        *,
        stage: Literal["pre_flight", "mid_execution"],
        budget_usd: float,
        predicted_usd: float | None = None,
        actual_usd_so_far: float | None = None,
        breakdown: list[CostBreakdownEntry] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.budget_usd = budget_usd
        self.predicted_usd = predicted_usd
        self.actual_usd_so_far = actual_usd_so_far
        self.breakdown = breakdown or []

    def to_envelope(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "code": "budget_exceeded",
                "stage": self.stage,
                "message": str(self),
                "budget_usd": self.budget_usd,
            }
        }
        if self.predicted_usd is not None:
            body["error"]["predicted_usd"] = self.predicted_usd
        if self.actual_usd_so_far is not None:
            body["error"]["actual_usd_so_far"] = self.actual_usd_so_far
        if self.breakdown:
            body["error"]["breakdown"] = [
                {
                    "model": e.model_id,
                    "calls": e.calls,
                    "input_tokens_est": e.input_tokens_est,
                    "output_tokens_est": e.output_tokens_est,
                    "cost_usd_est": e.cost_usd,
                }
                for e in self.breakdown
            ]
        return body
