"""Propose-and-Review strategy: tool-call interception + consensus + revisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from polyportia.config.models import ActualModelRef, ProposeAndReview
from polyportia.config.registry import Registry
from polyportia.council.context import ExecutionContext
from polyportia.council.propose_review import run_propose_review
from polyportia.observability.trace import TraceBuilder
from tests.conftest import (
    MockChoice,
    MockProvider,
    MockResponse,
    MockUsage,
)


@dataclass
class _FnPayload:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _FnPayload
    type: str = "function"


@dataclass
class _ToolMessage:
    content: str | None = None
    role: str = "assistant"
    tool_calls: list[_ToolCall] = field(default_factory=list)


def _tool_response(name: str, args: dict[str, Any], call_id: str = "call_1") -> MockResponse:
    msg = _ToolMessage(
        content=None,
        tool_calls=[_ToolCall(id=call_id, function=_FnPayload(name=name, arguments=json.dumps(args)))],
    )
    # MockResponse expects MockChoice with a message attribute — duck-typed.
    return MockResponse(choices=[MockChoice(message=msg)], usage=MockUsage())


def _ctx(reg: Registry) -> ExecutionContext:
    return ExecutionContext(registry=reg, trace=TraceBuilder({}))


async def test_passthrough_when_proposer_has_no_tool_calls(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.set_response("anthropic/claude-opus-4-7", "plain text answer, no tool")
    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[ActualModelRef(id="anthropic/claude-haiku-4-5")],
        consensus="all",
    )
    result = await run_propose_review(
        spec, [{"role": "user", "content": "hi"}], _ctx(test_registry)
    )
    # Reviewers should never be invoked.
    models_called = [c["model"] for c in mock_provider.calls]
    assert "anthropic/claude-haiku-4-5" not in models_called
    assert result.content == "plain text answer, no tool"


async def test_all_approve_returns_proposer_response_with_tool_calls(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.handlers["anthropic/claude-opus-4-7"] = lambda kw: _tool_response(
        "email", {"to": "alice@example.com", "body": "hi"}, call_id="call_proposer"
    )
    mock_provider.handlers["anthropic/claude-haiku-4-5"] = lambda kw: _tool_response(
        "approve", {"reason": "looks good"}, call_id="call_reviewer_1"
    )
    mock_provider.handlers["openai/gpt-5-5"] = lambda kw: _tool_response(
        "approve", {}, call_id="call_reviewer_2"
    )

    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[
            ActualModelRef(id="anthropic/claude-haiku-4-5"),
            ActualModelRef(id="openai/gpt-5-5"),
        ],
        consensus="all",
        max_revisions=0,
    )
    result = await run_propose_review(
        spec, [{"role": "user", "content": "send the email"}], _ctx(test_registry)
    )
    # Proposer's raw response is returned unchanged.
    assert result.raw is not None
    proposer_call = mock_provider.calls[0]
    assert proposer_call["model"] == "anthropic/claude-opus-4-7"
    # Reviewers should have been called once each with the injected toolset.
    reviewer_calls = [
        c for c in mock_provider.calls if c["model"] != "anthropic/claude-opus-4-7"
    ]
    assert len(reviewer_calls) == 2
    for rc in reviewer_calls:
        names = {t["function"]["name"] for t in rc["tools"]}
        assert names == {"approve", "deny", "insight"}


async def test_deny_triggers_revision_loop_then_approval(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    proposer_calls = {"i": 0}

    def proposer(_: dict) -> MockResponse:
        proposer_calls["i"] += 1
        # First proposal: aggressive. Second proposal: revised.
        body = "DELETE_ALL" if proposer_calls["i"] == 1 else "PLEASE"
        return _tool_response("email", {"body": body}, call_id=f"c{proposer_calls['i']}")

    reviewer_calls = {"i": 0}

    def reviewer(_: dict) -> MockResponse:
        reviewer_calls["i"] += 1
        # First round: deny. Second round: approve.
        if reviewer_calls["i"] == 1:
            return _tool_response("deny", {"reason": "too aggressive"}, call_id="rev")
        return _tool_response("approve", {}, call_id="rev")

    mock_provider.handlers["anthropic/claude-opus-4-7"] = proposer
    mock_provider.handlers["anthropic/claude-haiku-4-5"] = reviewer

    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[ActualModelRef(id="anthropic/claude-haiku-4-5")],
        consensus="all",
        max_revisions=2,
    )
    await run_propose_review(
        spec, [{"role": "user", "content": "send"}], _ctx(test_registry)
    )
    assert proposer_calls["i"] == 2  # initial + one revision
    assert reviewer_calls["i"] == 2
    # The proposer's second call should have seen the tool feedback in its messages.
    second_proposer = [c for c in mock_provider.calls if c["model"] == "anthropic/claude-opus-4-7"][1]
    msgs = second_proposer["messages"]
    assert any(m.get("role") == "tool" for m in msgs)
    assert "DENY" in str(msgs)


async def test_max_revisions_exhausted_return_denial(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.handlers["anthropic/claude-opus-4-7"] = lambda kw: _tool_response(
        "email", {"body": "spam"}, call_id="p"
    )
    mock_provider.handlers["anthropic/claude-haiku-4-5"] = lambda kw: _tool_response(
        "deny", {"reason": "spam"}, call_id="r"
    )

    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[ActualModelRef(id="anthropic/claude-haiku-4-5")],
        consensus="all",
        max_revisions=1,
        on_denial="return_denial",
    )
    result = await run_propose_review(
        spec, [{"role": "user", "content": "send"}], _ctx(test_registry)
    )
    assert "not approved" in result.content.lower()
    assert "DENY" in result.content


async def test_on_denial_fail_raises(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.handlers["anthropic/claude-opus-4-7"] = lambda kw: _tool_response(
        "x", {}, call_id="p"
    )
    mock_provider.handlers["anthropic/claude-haiku-4-5"] = lambda kw: _tool_response(
        "deny", {"reason": "no"}, call_id="r"
    )
    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[ActualModelRef(id="anthropic/claude-haiku-4-5")],
        consensus="all",
        max_revisions=0,
        on_denial="fail",
    )
    with pytest.raises(RuntimeError, match="consensus not reached"):
        await run_propose_review(spec, [{"role": "user", "content": "x"}], _ctx(test_registry))


async def test_any_consensus_passes_with_one_approve(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.handlers["anthropic/claude-opus-4-7"] = lambda kw: _tool_response(
        "email", {}, call_id="p"
    )
    call_count = {"i": 0}

    def reviewer(_: dict) -> MockResponse:
        call_count["i"] += 1
        # First reviewer denies, second approves — "any" should pass.
        if call_count["i"] == 1:
            return _tool_response("deny", {"reason": "no"}, call_id="r1")
        return _tool_response("approve", {}, call_id="r2")

    mock_provider.handlers["anthropic/claude-haiku-4-5"] = reviewer
    mock_provider.handlers["openai/gpt-5-5"] = reviewer

    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        reviewers=[
            ActualModelRef(id="anthropic/claude-haiku-4-5"),
            ActualModelRef(id="openai/gpt-5-5"),
        ],
        consensus="any",
        max_revisions=0,
    )
    result = await run_propose_review(
        spec, [{"role": "user", "content": "x"}], _ctx(test_registry)
    )
    # Approved → proposer's response returned, not a denial text.
    assert "not approved" not in (result.content or "").lower()


async def test_integer_threshold_consensus(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.handlers["anthropic/claude-opus-4-7"] = lambda kw: _tool_response(
        "email", {}, call_id="p"
    )

    seq = ["approve", "approve", "deny"]
    idx = {"i": 0}

    def reviewer(_: dict) -> MockResponse:
        verdict = seq[idx["i"] % len(seq)]
        idx["i"] += 1
        if verdict == "approve":
            return _tool_response("approve", {}, call_id=f"r{idx['i']}")
        return _tool_response("deny", {"reason": "n"}, call_id=f"r{idx['i']}")

    mock_provider.handlers["anthropic/claude-haiku-4-5"] = reviewer
    mock_provider.handlers["openai/gpt-5-5"] = reviewer

    spec = ProposeAndReview(
        proposer=ActualModelRef(id="anthropic/claude-opus-4-7"),
        # Three reviewers; ints are not allowed as Union with discriminator so we
        # pass two reviewers and need ≥1 approval (consensus=1).
        reviewers=[
            ActualModelRef(id="anthropic/claude-haiku-4-5"),
            ActualModelRef(id="openai/gpt-5-5"),
        ],
        consensus=1,
        max_revisions=0,
    )
    result = await run_propose_review(
        spec, [{"role": "user", "content": "x"}], _ctx(test_registry)
    )
    assert "not approved" not in (result.content or "").lower()
