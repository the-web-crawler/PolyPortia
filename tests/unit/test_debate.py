"""Debate strategy: turn semantics + visibility modes + termination."""

from __future__ import annotations

from collections import Counter

import pytest

from polyportia.config.models import (
    ActualModelRef,
    Debate,
    DebateConfig,
    DebateVisibility,
)
from polyportia.config.registry import Registry
from polyportia.council.context import ExecutionContext
from polyportia.council.debate import _build_turn_messages, run_debate
from polyportia.observability.trace import TraceBuilder
from tests.conftest import MockProvider


def _ctx(reg: Registry) -> ExecutionContext:
    return ExecutionContext(registry=reg, trace=TraceBuilder({}))


def _members() -> list[ActualModelRef]:
    return [
        ActualModelRef(id="anthropic/claude-opus-4-7"),
        ActualModelRef(id="anthropic/claude-haiku-4-5"),
    ]


async def test_debate_fixed_turns_runs_every_member_every_turn(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.set_response("anthropic/claude-opus-4-7", "opus answer")
    mock_provider.set_response("anthropic/claude-haiku-4-5", "haiku answer")

    spec = Debate(
        members=_members(),
        debate=DebateConfig(turns=3, visibility=DebateVisibility.prompt_and_peer_responses),
        output="array",
    )
    result = await run_debate(
        spec,
        base_messages=[{"role": "user", "content": "what is consciousness?"}],
        ctx=_ctx(test_registry),
    )
    # Two members across three turns → six provider calls.
    calls_per_model = Counter(c["model"] for c in mock_provider.calls)
    assert calls_per_model["anthropic/claude-opus-4-7"] == 3
    assert calls_per_model["anthropic/claude-haiku-4-5"] == 3
    assert result.raw is not None and isinstance(result.raw, dict)
    assert len(result.raw["responses"]) == 2


async def test_debate_consensus_termination_stops_early(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    # Both models always return the same content -> consensus after turn 1.
    mock_provider.set_response("anthropic/claude-opus-4-7", "42")
    mock_provider.set_response("anthropic/claude-haiku-4-5", "42")

    spec = Debate(
        members=_members(),
        debate=DebateConfig(turns=5, termination="consensus"),
        output="array",
    )
    await run_debate(spec, [{"role": "user", "content": "x"}], _ctx(test_registry))

    # Consensus detected after turn 1 (history has two entries: turn 0 + turn 1).
    calls_per_model = Counter(c["model"] for c in mock_provider.calls)
    assert calls_per_model["anthropic/claude-opus-4-7"] == 2
    assert calls_per_model["anthropic/claude-haiku-4-5"] == 2


@pytest.mark.parametrize(
    "visibility,expected_substr_in_turn1_for_first_member",
    [
        (DebateVisibility.prompt_and_peer_responses, "haiku turn 0"),
        (DebateVisibility.full_history, "(you)"),
    ],
)
def test_build_turn_messages_visibility_shape(
    visibility: DebateVisibility, expected_substr_in_turn1_for_first_member: str
) -> None:
    members = _members()
    base = [{"role": "user", "content": "topic"}]
    history = [["opus turn 0", "haiku turn 0"]]
    msgs = _build_turn_messages(
        visibility=visibility,
        base_messages=base,
        history=history,
        self_idx=0,
        members=members,
        critique_tpl="own:{{own_prior}}\npeers:{{peer_responses}}",
    )
    blob = "".join(str(m.get("content", "")) for m in msgs)
    assert "opus turn 0" in blob  # member's own prior is included
    assert expected_substr_in_turn1_for_first_member in blob


def test_own_only_with_target_excludes_full_base_messages() -> None:
    members = _members()
    base = [
        {"role": "system", "content": "private system"},
        {"role": "user", "content": "the question"},
    ]
    history = [["opus prior", "haiku prior"]]
    msgs = _build_turn_messages(
        visibility=DebateVisibility.own_only_with_target,
        base_messages=base,
        history=history,
        self_idx=0,
        members=members,
        critique_tpl="own={{own_prior}};peers={{peer_responses}}",
    )
    blob = "".join(str(m.get("content", "")) for m in msgs)
    # System prompt is intentionally omitted in this visibility mode.
    assert "private system" not in blob
    assert "the question" in blob
    assert "haiku prior" in blob
    assert "opus prior" in blob


async def test_debate_synthesize_output_calls_synthesizer(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.set_response("anthropic/claude-opus-4-7", "opus")
    mock_provider.set_response("anthropic/claude-haiku-4-5", "haiku")
    mock_provider.set_response("openai/gpt-5-5", "SYNTH_OUT")

    spec = Debate(
        members=_members(),
        debate=DebateConfig(turns=2),
        output="synthesize",
        synthesizer=ActualModelRef(id="openai/gpt-5-5"),
    )
    result = await run_debate(
        spec, [{"role": "user", "content": "q"}], _ctx(test_registry)
    )
    assert result.content == "SYNTH_OUT"
    # 2 members * 2 turns + 1 synthesizer call = 5
    assert len(mock_provider.calls) == 5
    # Synthesizer received the panel responses.
    synth_call = next(c for c in mock_provider.calls if c["model"] == "openai/gpt-5-5")
    assert "opus" in str(synth_call["messages"]) or "haiku" in str(synth_call["messages"])


async def test_debate_synthesize_without_synthesizer_raises(
    test_registry: Registry, mock_provider: MockProvider
) -> None:
    mock_provider.set_response("anthropic/claude-opus-4-7", "x")
    mock_provider.set_response("anthropic/claude-haiku-4-5", "y")
    spec = Debate(
        members=_members(),
        debate=DebateConfig(turns=1),
        output="synthesize",
        synthesizer=None,
    )
    with pytest.raises(ValueError, match="no synthesizer"):
        await run_debate(spec, [{"role": "user", "content": "q"}], _ctx(test_registry))
