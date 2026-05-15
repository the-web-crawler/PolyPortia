"""Defined-model fallback walking, including transitive (defined→defined) chains."""

from __future__ import annotations

import asyncio

import pytest

from polyportia.council.context import ExecutionContext
from polyportia.council.executor import (
    CyclicDefinedModelError,
    FallbacksExhaustedError,
    execute_target,
)
from polyportia.config.models import DefinedModel, DefinedModelRef, ActualModelRef
from polyportia.observability.trace import TraceBuilder
from tests.conftest import make_mock_response


class FakeRateLimit(Exception):
    pass


FakeRateLimit.__name__ = "RateLimitError"


def _ctx(registry):
    return ExecutionContext(registry=registry, trace=TraceBuilder({}))


def test_primary_succeeds_no_fallback(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-opus-4-7", "primary-ok")
    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            DefinedModelRef(name="thinking"),
            [{"role": "user", "content": "hi"}],
            ctx,
        )
    )
    assert result.content == "primary-ok"
    # Only the primary was called
    called_models = [c["model"] for c in mock_provider.calls]
    assert called_models == ["anthropic/claude-opus-4-7"]


def test_falls_over_to_first_fallback(mock_provider, test_registry):
    mock_provider.set_error("anthropic/claude-opus-4-7", FakeRateLimit)
    mock_provider.set_response("openai/gpt-5-5", "fallback-ok")
    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            DefinedModelRef(name="thinking"),
            [{"role": "user", "content": "hi"}],
            ctx,
        )
    )
    assert result.content == "fallback-ok"
    # The trace should record the defined-model span with fallback_chain
    defined_span = next(s for s in ctx.trace.record.spans if s.kind == "defined")
    assert defined_span.fallback_chain[0] == "actual:anthropic/claude-opus-4-7"
    assert defined_span.fallback_chain[1] == "actual:openai/gpt-5-5"


def test_transitive_fallback_through_defined(mock_provider, test_registry):
    """thinking's fallbacks: [openai/gpt-5-5, defined:fast]; both openai fail,
    fast (haiku) succeeds via fast's primary target."""
    mock_provider.set_error("anthropic/claude-opus-4-7", FakeRateLimit)
    mock_provider.set_error("openai/gpt-5-5", FakeRateLimit)
    mock_provider.set_response("anthropic/claude-haiku-4-5", "haiku-via-fast")
    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            DefinedModelRef(name="thinking"),
            [{"role": "user", "content": "hi"}],
            ctx,
        )
    )
    assert result.content == "haiku-via-fast"


def test_all_fallbacks_exhausted_raises(mock_provider, test_registry):
    mock_provider.set_error("anthropic/claude-opus-4-7", FakeRateLimit)
    mock_provider.set_error("openai/gpt-5-5", FakeRateLimit)
    mock_provider.set_error("anthropic/claude-haiku-4-5", FakeRateLimit)
    ctx = _ctx(test_registry)
    with pytest.raises(FallbacksExhaustedError):
        asyncio.run(
            execute_target(
                DefinedModelRef(name="thinking"),
                [{"role": "user", "content": "hi"}],
                ctx,
            )
        )


def test_cyclic_defined_chain_raises(mock_provider):
    """Construct a cycle: A → fallback B → fallback A."""
    from polyportia.config.models import (
        ActualModel,
        DefinedModel,
        ProviderConfig,
        PolyPortiaConfig,
    )
    from polyportia.config.registry import Registry

    cfg = PolyPortiaConfig(
        providers=[ProviderConfig(name="p", api_key="k")],
        actual_models=[ActualModel(id="p/m", provider="p")],
        defined_models=[
            DefinedModel(
                name="A",
                target=ActualModelRef(id="p/m"),
                fallbacks=[DefinedModelRef(name="B")],
            ),
            DefinedModel(
                name="B",
                target=ActualModelRef(id="p/m"),
                fallbacks=[DefinedModelRef(name="A")],
            ),
        ],
    )
    reg = Registry(cfg)
    mock_provider.set_error("p/m", FakeRateLimit)
    ctx = _ctx(reg)
    with pytest.raises((CyclicDefinedModelError, FallbacksExhaustedError)):
        asyncio.run(
            execute_target(
                DefinedModelRef(name="A"),
                [{"role": "user", "content": "hi"}],
                ctx,
            )
        )
