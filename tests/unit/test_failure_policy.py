from __future__ import annotations

import pytest

from polyportia.config.models import FailurePolicy
from polyportia.council.failure import (
    CouncilFailureError,
    MemberOutcome,
    apply_failure_policy,
)
from polyportia.providers.base import ProviderResult


def _ok(name: str) -> MemberOutcome:
    return MemberOutcome(
        member_repr=name,
        result=ProviderResult(model_id=name, content="x"),
        error=None,
    )


def _err(name: str) -> MemberOutcome:
    return MemberOutcome(member_repr=name, result=None, error=RuntimeError("fail"))


def test_continue_keeps_partial():
    policy = FailurePolicy(on_failure="continue", min_success=1)
    out = apply_failure_policy([_ok("a"), _err("b")], policy)
    assert len(out) == 2


def test_fail_raises_on_any_failure():
    policy = FailurePolicy(on_failure="fail")
    with pytest.raises(CouncilFailureError):
        apply_failure_policy([_ok("a"), _err("b")], policy)


def test_min_success_count():
    policy = FailurePolicy(min_success=2)
    with pytest.raises(CouncilFailureError):
        apply_failure_policy([_ok("a"), _err("b")], policy)


def test_min_success_fraction():
    policy = FailurePolicy(min_success=None, min_success_fraction=0.66)
    with pytest.raises(CouncilFailureError):
        apply_failure_policy([_ok("a"), _err("b"), _err("c")], policy)
    out = apply_failure_policy([_ok("a"), _ok("b"), _err("c")], policy)
    assert len(out) == 3
