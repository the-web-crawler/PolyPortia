"""Mid-execution cost enforcer.

Tracks running real cost on an ExecutionContext. The executor calls
``check_before_call`` before issuing each provider call; if the running cost
plus the estimated cost of the next call exceeds the budget, raise
``BudgetExceededError`` with stage ``"mid_execution"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polyportia.budget.errors import BudgetExceededError, CostBreakdownEntry


@dataclass
class BudgetEnforcer:
    budget_usd: float | None
    spent_usd: float = 0.0
    last_predicted_usd: float | None = None
    breakdown_at_breach: list[CostBreakdownEntry] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.budget_usd is not None

    def record_spent(self, amount_usd: float | None) -> None:
        if amount_usd is None:
            return
        self.spent_usd += amount_usd

    def check_or_raise(self) -> None:
        if not self.enabled:
            return
        assert self.budget_usd is not None
        if self.spent_usd > self.budget_usd:
            raise BudgetExceededError(
                f"Spent ${self.spent_usd:.6f} exceeds budget ${self.budget_usd:.6f}",
                stage="mid_execution",
                budget_usd=self.budget_usd,
                actual_usd_so_far=self.spent_usd,
            )

    def check_next_or_raise(self, est_next_call_usd: float) -> None:
        if not self.enabled:
            return
        assert self.budget_usd is not None
        if self.spent_usd + est_next_call_usd > self.budget_usd:
            raise BudgetExceededError(
                f"Next call (~${est_next_call_usd:.6f}) would push total above "
                f"budget (${self.budget_usd:.6f}); already spent ${self.spent_usd:.6f}",
                stage="mid_execution",
                budget_usd=self.budget_usd,
                actual_usd_so_far=self.spent_usd,
                predicted_usd=self.spent_usd + est_next_call_usd,
            )
