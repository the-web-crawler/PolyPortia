"""Synthesize strategy and recursive (council-of-councils) cases."""

from __future__ import annotations

import asyncio

from polyportia.config.models import CouncilRef
from polyportia.council.context import ExecutionContext
from polyportia.council.executor import execute_target
from polyportia.observability.trace import TraceBuilder


def _ctx(registry):
    return ExecutionContext(registry=registry, trace=TraceBuilder({}))


def test_synthesize_calls_synthesizer_with_member_responses(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-haiku-4-5", "fast-says")
    mock_provider.set_response("openai/gpt-5-5", "creative-says")

    captured = {}

    def synthesizer_handler(kw):
        captured["messages"] = kw["messages"]
        from tests.conftest import make_mock_response

        return make_mock_response("synthesized-answer")

    mock_provider.handlers["anthropic/claude-opus-4-7"] = synthesizer_handler

    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            CouncilRef(name="triad"),
            [{"role": "user", "content": "explain CRDTs"}],
            ctx,
        )
    )
    assert result.content == "synthesized-answer"
    synth_user = captured["messages"][-1]["content"]
    assert "explain CRDTs" in synth_user
    assert "fast-says" in synth_user
    assert "creative-says" in synth_user


def test_meta_council_recurses(mock_provider, test_registry):
    """meta-council: synthesize over [council:triad, defined:creative],
    synthesizer = defined:thinking. The triad sub-council also synthesizes."""
    call_counter = {"n": 0}

    def increment_handler(content_template: str):
        def handler(kw):
            call_counter["n"] += 1
            from tests.conftest import make_mock_response

            return make_mock_response(content_template.format(n=call_counter["n"]))

        return handler

    mock_provider.handlers["anthropic/claude-haiku-4-5"] = increment_handler("haiku-{n}")
    mock_provider.handlers["openai/gpt-5-5"] = increment_handler("openai-{n}")
    mock_provider.handlers["anthropic/claude-opus-4-7"] = increment_handler("opus-{n}")

    ctx = _ctx(test_registry)
    result = asyncio.run(
        execute_target(
            CouncilRef(name="meta-council"),
            [{"role": "user", "content": "ping"}],
            ctx,
        )
    )
    assert result.content.startswith("opus-")
    # Sub-council triad fans out 3 members + synthesizer (opus) = 4 calls;
    # meta-council adds 1 more member (creative=openai) and 1 final synthesizer (opus).
    # Total = 4 + 1 + 1 = 6 underlying provider calls
    assert len(mock_provider.calls) == 6
