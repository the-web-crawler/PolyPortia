"""ParallelArray strategy."""

from __future__ import annotations

import asyncio

import pytest

from polyportia.config.models import (
    CouncilRef,
    FailurePolicy,
)
from polyportia.council.context import ExecutionContext
from polyportia.council.executor import execute_target
from polyportia.council.failure import CouncilFailureError
from polyportia.observability.trace import TraceBuilder


def _ctx(registry):
    return ExecutionContext(registry=registry, trace=TraceBuilder({}))


class FakeRateLimit(Exception):
    pass


FakeRateLimit.__name__ = "RateLimitError"


def test_parallel_returns_array_envelope(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-haiku-4-5", "fast-result")
    mock_provider.set_response("anthropic/claude-opus-4-7", "thinking-result")
    mock_provider.set_response("openai/gpt-5-5", "creative-result")

    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            CouncilRef(name="triad-raw"),
            [{"role": "user", "content": "hi"}],
            ctx,
        )
    )
    assert isinstance(result.raw, dict)
    contents = [r["content"] for r in result.raw["responses"] if "content" in r]
    assert sorted(contents) == sorted(["fast-result", "thinking-result", "creative-result"])


def test_parallel_partial_failure_kept(mock_provider, test_registry):
    """When a member exhausts its fallback chain and fails, the council still
    returns; other members appear with their content."""
    # Strip 'thinking' fallbacks so its failure isn't masked by them
    from polyportia.config.models import ActualModelRef, DefinedModel

    thinking = test_registry.get_defined_model("thinking")
    test_registry._defined["thinking"] = DefinedModel(
        name="thinking", target=ActualModelRef(id=thinking.target.id), fallbacks=[]
    )

    mock_provider.set_response("anthropic/claude-haiku-4-5", "fast-ok")
    mock_provider.set_error("anthropic/claude-opus-4-7", FakeRateLimit)
    mock_provider.set_response("openai/gpt-5-5", "creative-ok")

    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            CouncilRef(name="triad-raw"),
            [{"role": "user", "content": "hi"}],
            ctx,
        )
    )
    responses = result.raw["responses"]
    by_member = {r["member"]: r for r in responses}
    assert "content" in by_member["defined:fast"]
    assert "error" in by_member["defined:thinking"]


def test_parallel_min_success_violated(monkeypatch, mock_provider, test_registry):
    mock_provider.set_error("anthropic/claude-haiku-4-5", FakeRateLimit)
    mock_provider.set_error("anthropic/claude-opus-4-7", FakeRateLimit)
    mock_provider.set_error("openai/gpt-5-5", FakeRateLimit)

    monkeypatch.setattr(test_registry, "_failure", FailurePolicy(min_success=2))
    ctx = _ctx(test_registry)
    with pytest.raises(CouncilFailureError):
        asyncio.run(
            execute_target(
                CouncilRef(name="triad-raw"),
                [{"role": "user", "content": "hi"}],
                ctx,
            )
        )


def test_parallel_runs_concurrently(mock_provider, test_registry):
    """All members start before any complete (ordering proof via barrier)."""
    barrier = asyncio.Event()
    started = 0

    async def slow_response(content: str):
        nonlocal started
        started += 1
        if started == 3:
            barrier.set()
        await barrier.wait()
        from tests.conftest import make_mock_response

        return make_mock_response(content)

    mock_provider.handlers["anthropic/claude-haiku-4-5"] = lambda kw: slow_response("a")
    mock_provider.handlers["anthropic/claude-opus-4-7"] = lambda kw: slow_response("b")
    mock_provider.handlers["openai/gpt-5-5"] = lambda kw: slow_response("c")

    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            CouncilRef(name="triad-raw"),
            [{"role": "user", "content": "hi"}],
            ctx,
        )
    )
    contents = sorted(r["content"] for r in result.raw["responses"] if "content" in r)
    assert contents == ["a", "b", "c"]
