"""Edge-case sweep across the surface area.

Covers: cyclic councils, recursion depth, empty configs, env-var defaults,
failure-policy fractions, propose_review reviewer-no-verdict / insight-flag
interactions, debate judge termination, depth cap, malformed inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from polyportia.config.loader import load_config_from_string
from polyportia.config.models import (
    ActualModelRef,
    Debate,
    DebateConfig,
    DefinedModel,
    DefinedModelRef,
    FailurePolicy,
    PolyPortiaConfig,
    ProposeAndReview,
    ProviderConfig,
)
from polyportia.config.registry import Registry, RegistryError
from polyportia.council.context import ExecutionContext, RecursionDepthExceeded
from polyportia.council.executor import (
    CyclicDefinedModelError,
    FallbacksExhaustedError,
    execute_target,
    resolve_request_model,
)
from polyportia.council.failure import (
    CouncilFailureError,
    MemberOutcome,
    apply_failure_policy,
)
from polyportia.council.propose_review import run_propose_review
from polyportia.observability.trace import TraceBuilder
from tests.conftest import MockProvider, make_mock_response


def _ctx(reg: Registry, *, depth: int = 0, max_depth: int = 8) -> ExecutionContext:
    return ExecutionContext(
        registry=reg, trace=TraceBuilder({}), depth=depth, max_depth=max_depth
    )


# --- Config / YAML edge cases ---


def test_empty_yaml_produces_default_config() -> None:
    cfg = load_config_from_string("")
    assert cfg.providers == []
    assert cfg.actual_models == []


def test_yaml_env_var_default_used_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAYBE_KEY", raising=False)
    cfg = load_config_from_string(
        """
providers:
  - name: x
    api_key: ${MAYBE_KEY:-fallback-key}
"""
    )
    assert cfg.providers[0].api_key is not None
    assert cfg.providers[0].api_key.get_secret_value() == "fallback-key"


def test_yaml_env_var_missing_no_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNDEFINED_KEY_X", raising=False)
    cfg = load_config_from_string(
        """
providers:
  - name: x
    api_key: ${UNDEFINED_KEY_X}
"""
    )
    # Missing env var with no default expands to empty string — secret still present.
    assert cfg.providers[0].api_key is not None
    assert cfg.providers[0].api_key.get_secret_value() == ""


def test_actual_model_referencing_unknown_provider_rejected() -> None:
    cfg = PolyPortiaConfig.model_validate(
        {
            "providers": [{"name": "p1"}],
            "actual_models": [{"id": "p1/m", "provider": "ghost"}],
        }
    )
    with pytest.raises(RegistryError, match="ghost"):
        Registry(cfg)


def test_extra_keys_rejected_in_provider() -> None:
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(
            {"name": "p", "unknown_field": "boom"}
        )


# --- Request-model resolution ---


def test_resolve_request_model_falls_through_to_literal_id() -> None:
    cfg = load_config_from_string("providers: [{name: p, api_key: k}]")
    reg = Registry(cfg)
    target = resolve_request_model("unknown/passthrough", reg)
    assert isinstance(target, ActualModelRef)
    assert target.id == "unknown/passthrough"


def test_resolve_request_model_unknown_alias_raises() -> None:
    reg = Registry(load_config_from_string(""))
    with pytest.raises(ValueError, match="unknown model"):
        resolve_request_model("nameWithoutSlash", reg)


# --- Cyclic and depth cases ---


def test_cyclic_defined_model_fallback_chain_raises(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    # Add two defined models that point at each other as fallbacks.
    test_registry.register_defined_model(
        DefinedModel(
            name="A",
            target=ActualModelRef(id="anthropic/claude-opus-4-7"),
            fallbacks=[DefinedModelRef(name="B")],
        )
    )
    test_registry.register_defined_model(
        DefinedModel(
            name="B",
            target=ActualModelRef(id="anthropic/claude-haiku-4-5"),
            fallbacks=[DefinedModelRef(name="A")],
        )
    )
    mock_provider.set_error(
        "anthropic/claude-opus-4-7", lambda: TimeoutError("timeout")
    )
    mock_provider.set_error(
        "anthropic/claude-haiku-4-5", lambda: TimeoutError("timeout")
    )
    import asyncio

    # Both primary and fallback fail; cycle on A is caught and resolves to
    # final FallbacksExhaustedError on the outermost.
    with pytest.raises((FallbacksExhaustedError, CyclicDefinedModelError)):
        asyncio.run(
            execute_target(
                DefinedModelRef(name="A"), [{"role": "user", "content": "x"}],
                _ctx(test_registry),
            )
        )


def test_recursion_depth_cap_triggers(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.set_response("anthropic/claude-opus-4-7", "ok")
    import asyncio

    ctx = _ctx(test_registry, depth=99, max_depth=8)  # already over the cap
    with pytest.raises(RecursionDepthExceeded):
        asyncio.run(
            execute_target(
                ActualModelRef(id="anthropic/claude-opus-4-7"),
                [{"role": "user", "content": "x"}],
                ctx,
            )
        )


# --- Failure policy ---


def test_min_success_fraction_below_threshold_fails() -> None:
    outcomes = [
        MemberOutcome(member_repr="a", result=make_mock_response("ok"), error=None),
        MemberOutcome(member_repr="b", result=None, error=RuntimeError("x")),
        MemberOutcome(member_repr="c", result=None, error=RuntimeError("y")),
    ]
    pol = FailurePolicy(on_failure="continue", min_success_fraction=0.5)
    with pytest.raises(CouncilFailureError):
        apply_failure_policy(outcomes, pol)


def test_failure_policy_fail_when_any_member_errors() -> None:
    outcomes = [
        MemberOutcome(member_repr="a", result=make_mock_response("ok"), error=None),
        MemberOutcome(member_repr="b", result=None, error=RuntimeError("oops")),
    ]
    pol = FailurePolicy(on_failure="fail")
    with pytest.raises(CouncilFailureError):
        apply_failure_policy(outcomes, pol)


def test_failure_policy_continue_passes_when_min_met() -> None:
    outcomes = [
        MemberOutcome(member_repr="a", result=make_mock_response("ok"), error=None),
        MemberOutcome(member_repr="b", result=None, error=RuntimeError("oops")),
    ]
    pol = FailurePolicy(on_failure="continue", min_success=1)
    apply_failure_policy(outcomes, pol)  # no raise


# --- ProposeAndReview edge cases ---


async def test_reviewer_returns_no_tool_call_counts_as_none(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    def proposer(_: dict) -> object:
        return make_mock_response_with_tool_call("email", {}, call_id="p")

    def reviewer(_: dict) -> object:
        # Returns plain text — no tool_calls. Should be counted as "none" verdict.
        return make_mock_response("just some thoughts, no verdict")

    mock_provider.handlers["anthropic/claude-opus-4-7"] = proposer
    mock_provider.handlers["anthropic/claude-haiku-4-5"] = reviewer

    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[ActualModelRef(id="anthropic/claude-haiku-4-5")],
        consensus="all",
        max_revisions=0,
    )
    result = await run_propose_review(
        spec, [{"role": "user", "content": "x"}], _ctx(test_registry)
    )
    # No verdict cast → not approved → denial text returned.
    assert "not approved" in result.content.lower()


async def test_insight_counts_as_approval_when_flag_set(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.handlers["anthropic/claude-opus-4-7"] = lambda _: \
        make_mock_response_with_tool_call("email", {}, call_id="p")
    mock_provider.handlers["anthropic/claude-haiku-4-5"] = lambda _: \
        make_mock_response_with_tool_call("insight", {"text": "consider X"}, call_id="r")

    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[ActualModelRef(id="anthropic/claude-haiku-4-5")],
        consensus="all",
        max_revisions=0,
        insight_counts_as_approval=True,
    )
    result = await run_propose_review(
        spec, [{"role": "user", "content": "x"}], _ctx(test_registry)
    )
    # With insight counting as approval, proposal goes through.
    assert "not approved" not in result.content.lower()


# Helper: mock response with tool_calls.


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _Fn
    type: str = "function"


@dataclass
class _Msg:
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[_ToolCall] = field(default_factory=list)


def make_mock_response_with_tool_call(name: str, args: dict, call_id: str = "c1") -> object:
    from tests.conftest import MockChoice, MockResponse, MockUsage

    msg = _Msg(
        content=None,
        tool_calls=[_ToolCall(id=call_id, function=_Fn(name=name, arguments=json.dumps(args)))],
    )
    return MockResponse(choices=[MockChoice(message=msg)], usage=MockUsage())


# --- Debate edge cases ---


async def test_debate_with_one_member_still_runs(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    from polyportia.council.debate import run_debate

    mock_provider.set_response("anthropic/claude-opus-4-7", "alone")
    spec = Debate(
        members=[ActualModelRef(id="anthropic/claude-opus-4-7")],
        debate=DebateConfig(turns=2, termination="consensus"),
        output="array",
    )
    result = await run_debate(
        spec, [{"role": "user", "content": "x"}], _ctx(test_registry)
    )
    assert result.raw is not None


# --- Provider with no api_key still acceptable to register ---


def test_provider_with_no_api_key_is_valid() -> None:
    p = ProviderConfig(name="local-ollama")
    assert p.api_key is None
    Registry().register_provider(p)  # no error


# --- Trace builder edge cases ---


def test_trace_finalize_with_zero_spans_is_ok_status() -> None:
    tb = TraceBuilder({"test": True})
    record = tb.finalize()
    assert record.final_status == "ok"
    assert record.spans == []
