"""HTTP-level budget behaviour: pre-flight refusal, mid-execution stop, headers."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from polyportia.config.loader import load_config_from_string
from polyportia.config.registry import Registry
from polyportia.observability.store import TraceStore
from polyportia.server.app import create_app
from tests.conftest import MockProvider

_YAML = """
providers:
  - {name: anthropic, api_key: x}
  - {name: openai, api_key: y}
actual_models:
  - id: tiny
    provider: anthropic
    max_output_tokens: 100
    input_cost_per_1m_tokens: 1.0
    output_cost_per_1m_tokens: 2.0
  - id: huge
    provider: openai
    max_output_tokens: 100000
    input_cost_per_1m_tokens: 1000.0
    output_cost_per_1m_tokens: 5000.0
defined_models:
  - name: cheap
    target: {kind: actual, id: tiny}
  - name: expensive
    target: {kind: actual, id: huge}
councils:
  - name: panel
    strategy:
      kind: parallel_array
      members:
        - {kind: actual, id: tiny}
        - {kind: actual, id: tiny}
        - {kind: actual, id: tiny}
"""


@pytest.fixture
def app_and_provider(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, MockProvider]:
    cfg = load_config_from_string(_YAML)
    app = create_app()
    app.state.registry = Registry(cfg)
    app.state.trace_store = TraceStore()

    provider = MockProvider()

    async def patched(**kwargs: Any) -> Any:
        return await provider(**kwargs)

    monkeypatch.setattr("polyportia.providers.litellm_adapter.acompletion", patched)
    return app, provider


def test_request_under_budget_returns_with_cost_headers(
    app_and_provider: tuple[Any, MockProvider]
) -> None:
    app, provider = app_and_provider
    provider.set_response("tiny", "ok")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "cheap",
            "messages": [{"role": "user", "content": "hi"}],
            "polyportia": {"budget_usd": 1.0},
        },
    )
    assert r.status_code == 200
    assert "x-polyportia-cost-usd" in r.headers
    assert "x-polyportia-cost-predicted-usd" in r.headers
    assert float(r.headers["x-polyportia-cost-usd"]) >= 0.0
    assert float(r.headers["x-polyportia-cost-predicted-usd"]) >= 0.0


def test_pre_flight_refusal_returns_402(
    app_and_provider: tuple[Any, MockProvider]
) -> None:
    app, _ = app_and_provider
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "expensive",
            "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 100000,
            "polyportia": {"budget_usd": 0.01},
        },
    )
    assert r.status_code == 402
    body = r.json()
    assert body["error"]["code"] == "budget_exceeded"
    assert body["error"]["stage"] == "pre_flight"
    assert body["error"]["budget_usd"] == 0.01
    assert body["error"]["predicted_usd"] > 0.01
    assert "breakdown" in body["error"]


def test_unlimited_budget_disables_check(
    app_and_provider: tuple[Any, MockProvider]
) -> None:
    app, provider = app_and_provider
    provider.set_response("huge", "ok")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "expensive",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100000,
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200


def test_config_default_budget_applies_when_request_omits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    yaml = _YAML + "\nbudget_usd_default: 0.001\n"
    cfg = load_config_from_string(yaml)
    app = create_app()
    app.state.registry = Registry(cfg)
    app.state.trace_store = TraceStore()
    provider = MockProvider()
    provider.set_response("huge", "ok")

    async def patched(**kwargs: Any) -> Any:
        return await provider(**kwargs)

    monkeypatch.setattr("polyportia.providers.litellm_adapter.acompletion", patched)
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "expensive",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100000,
        },
    )
    # config default of 0.001 should reject expensive model
    assert r.status_code == 402
    assert r.json()["error"]["stage"] == "pre_flight"


def test_include_cost_body_extension(
    app_and_provider: tuple[Any, MockProvider]
) -> None:
    app, provider = app_and_provider
    provider.set_response("tiny", "ok")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "cheap",
            "messages": [{"role": "user", "content": "hi"}],
            "polyportia": {"include_cost": True},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "polyportia" in body
    assert "cost" in body["polyportia"]
    assert "actual_usd" in body["polyportia"]["cost"]
    assert "predicted_usd" in body["polyportia"]["cost"]
    assert isinstance(body["polyportia"]["cost"]["by_model"], list)


def test_council_endpoint_returns_cost_in_envelope(
    app_and_provider: tuple[Any, MockProvider]
) -> None:
    app, provider = app_and_provider
    provider.set_response("tiny", "ok")
    client = TestClient(app)
    r = client.post(
        "/v1/councils/panel/run",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "cost_usd" in body
    assert "cost_predicted_usd" in body
    assert "responses" in body
    assert len(body["responses"]) == 3
