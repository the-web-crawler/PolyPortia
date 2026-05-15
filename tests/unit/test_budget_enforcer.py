"""Budget enforcer: record + check semantics."""

from __future__ import annotations

import pytest

from polyportia.budget.enforcer import BudgetEnforcer
from polyportia.budget.errors import BudgetExceededError


def test_no_budget_means_no_op() -> None:
    e = BudgetEnforcer(budget_usd=None)
    e.record_spent(99.0)
    e.check_or_raise()
    e.check_next_or_raise(99.0)


def test_record_and_check_below_budget() -> None:
    e = BudgetEnforcer(budget_usd=1.0)
    e.record_spent(0.4)
    e.check_or_raise()
    assert e.spent_usd == pytest.approx(0.4)


def test_overshoot_raises() -> None:
    e = BudgetEnforcer(budget_usd=1.0)
    e.record_spent(1.5)
    with pytest.raises(BudgetExceededError) as info:
        e.check_or_raise()
    assert info.value.stage == "mid_execution"
    assert info.value.budget_usd == 1.0
    assert info.value.actual_usd_so_far == pytest.approx(1.5)


def test_check_next_or_raise() -> None:
    e = BudgetEnforcer(budget_usd=1.0)
    e.record_spent(0.7)
    e.check_next_or_raise(0.2)  # 0.9 < 1.0 ok
    with pytest.raises(BudgetExceededError):
        e.check_next_or_raise(0.5)  # 0.7 + 0.5 > 1.0


def test_none_spent_recordable_as_no_op() -> None:
    e = BudgetEnforcer(budget_usd=1.0)
    e.record_spent(None)
    assert e.spent_usd == 0.0
