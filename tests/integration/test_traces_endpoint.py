"""Traces endpoint coverage: GET /v1/traces, GET /v1/traces/{id}."""

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
providers: [{name: anthropic, api_key: x}]
actual_models:
  - id: anthropic/claude-haiku-4-5
    provider: anthropic
    input_cost_per_1m_tokens: 1.0
    output_cost_per_1m_tokens: 2.0
defined_models:
  - name: fast
    target: {kind: actual, id: anthropic/claude-haiku-4-5}
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


def test_trace_404_for_unknown_id(app_and_provider: tuple[Any, MockProvider]) -> None:
    app, _ = app_and_provider
    with TestClient(app) as client:
        r = client.get("/v1/traces/nonexistent")
    assert r.status_code == 404


def test_trace_persisted_after_request(
    app_and_provider: tuple[Any, MockProvider]
) -> None:
    app, provider = app_and_provider
    provider.set_response("anthropic/claude-haiku-4-5", "ok")
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "fast",
                "messages": [{"role": "user", "content": "hi"}],
                "polyportia": {"budget_usd": "unlimited"},
            },
        )
        trace_id = r.headers["x-polyportia-trace-id"]
        t = client.get(f"/v1/traces/{trace_id}")
    assert t.status_code == 200
    body = t.json()
    assert body["trace_id"] == trace_id
    # defined and actual spans both recorded
    kinds = {s["kind"] for s in body["spans"]}
    assert "defined" in kinds and "actual" in kinds


def test_trace_list_returns_recent_ids(
    app_and_provider: tuple[Any, MockProvider]
) -> None:
    app, provider = app_and_provider
    provider.set_response("anthropic/claude-haiku-4-5", "ok")
    with TestClient(app) as client:
        ids: list[str] = []
        for _ in range(3):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "fast",
                    "messages": [{"role": "user", "content": "hi"}],
                    "polyportia": {"budget_usd": "unlimited"},
                },
            )
            ids.append(r.headers["x-polyportia-trace-id"])
        listing = client.get("/v1/traces").json()
    returned_ids = {entry["trace_id"] for entry in listing["data"]}
    assert returned_ids >= set(ids)
