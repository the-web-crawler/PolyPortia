"""End-to-end SDK coverage: complete / acomplete / run_council, budgets,
registration, return_cost."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import polyportia
from polyportia.budget.errors import BudgetExceededError
from polyportia.config.loader import load_config_from_string
from polyportia.config.registry import Registry, set_default_registry
from polyportia.observability.store import TraceStore, set_default_store
from polyportia.sdk.client import acomplete, complete, run_council
from tests.conftest import MockProvider

_YAML = """
providers:
  - {name: anthropic, api_key: x}
  - {name: openai, api_key: y}
actual_models:
  - id: anthropic/claude-haiku-4-5
    provider: anthropic
    input_cost_per_1m_tokens: 1.0
    output_cost_per_1m_tokens: 2.0
  - id: openai/gpt-5-5
    provider: openai
    input_cost_per_1m_tokens: 2.0
    output_cost_per_1m_tokens: 4.0
defined_models:
  - name: fast
    target: {kind: actual, id: anthropic/claude-haiku-4-5}
councils:
  - name: panel
    strategy:
      kind: parallel_array
      members:
        - {kind: actual, id: anthropic/claude-haiku-4-5}
        - {kind: actual, id: openai/gpt-5-5}
"""


@pytest.fixture(autouse=True)
def _setup(monkeypatch: pytest.MonkeyPatch) -> MockProvider:
    cfg = load_config_from_string(_YAML)
    set_default_registry(Registry(cfg))
    set_default_store(TraceStore())
    provider = MockProvider()

    async def patched(**kwargs: Any) -> Any:
        return await provider(**kwargs)

    monkeypatch.setattr("polyportia.providers.litellm_adapter.acompletion", patched)
    return provider


def test_complete_sync(_setup: MockProvider) -> None:
    _setup.set_response("anthropic/claude-haiku-4-5", "hello!")
    result, trace_id = complete(
        model="fast", messages=[{"role": "user", "content": "x"}]
    )
    assert result.content == "hello!"
    assert isinstance(trace_id, str) and trace_id


def test_acomplete_async(_setup: MockProvider) -> None:
    _setup.set_response("anthropic/claude-haiku-4-5", "hi")

    async def _run() -> None:
        result, _ = await acomplete(
            model="fast", messages=[{"role": "user", "content": "x"}]
        )
        assert result.content == "hi"

    asyncio.run(_run())


def test_complete_with_return_cost(_setup: MockProvider) -> None:
    _setup.set_response("anthropic/claude-haiku-4-5", "ok")
    result, trace_id, estimate, actual = complete(
        model="fast",
        messages=[{"role": "user", "content": "x"}],
        return_cost=True,
        budget_usd="unlimited",
    )
    assert result.content == "ok"
    assert estimate.total_usd >= 0
    assert actual >= 0


def test_run_council_via_sdk(_setup: MockProvider) -> None:
    _setup.set_response("anthropic/claude-haiku-4-5", "a")
    _setup.set_response("openai/gpt-5-5", "b")

    async def _run() -> None:
        result, _ = await run_council(
            "panel",
            messages=[{"role": "user", "content": "x"}],
            budget_usd="unlimited",
        )
        assert result.raw is not None

    asyncio.run(_run())


def test_complete_budget_pre_flight_raises(_setup: MockProvider) -> None:
    _setup.set_response("openai/gpt-5-5", "ok")
    with pytest.raises(BudgetExceededError) as info:
        complete(
            model="openai/gpt-5-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1_000_000,
            budget_usd=0.0000001,
        )
    assert info.value.stage == "pre_flight"


def test_top_level_register_helpers_round_trip(_setup: MockProvider) -> None:
    from polyportia.config.models import ActualModel, DefinedModel, ProviderConfig

    polyportia.register_provider(ProviderConfig(name="newprov"))
    polyportia.register_actual_model(
        ActualModel(id="newprov/m", provider="newprov", max_output_tokens=10)
    )
    polyportia.register_defined_model(
        DefinedModel(name="newd", target={"kind": "actual", "id": "newprov/m"})
    )
    _setup.set_response("newprov/m", "registered ok")
    result, _ = complete(
        model="newd",
        messages=[{"role": "user", "content": "x"}],
        budget_usd="unlimited",
    )
    assert result.content == "registered ok"
