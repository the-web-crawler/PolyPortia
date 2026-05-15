"""HTTP integration: OpenAI-compatible /v1/chat/completions and /v1/models."""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from polyportia.server.app import create_app


def _client(registry):
    app = create_app(registry=registry)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def test_chat_completion_via_defined_model(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-opus-4-7", "hello there")

    async def run():
        async with _client(test_registry) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "thinking",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return r

    r = asyncio.run(run())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello there"
    assert body["model"] == "thinking"
    assert "x-polyportia-trace-id" in r.headers


def test_chat_completion_via_actual_model_literal(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-opus-4-7", "literal-id")

    async def run():
        async with _client(test_registry) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-opus-4-7",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return r

    r = asyncio.run(run())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "literal-id"


def test_unknown_model_404(mock_provider, test_registry):
    async def run():
        async with _client(test_registry) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "nonexistent",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return r

    r = asyncio.run(run())
    assert r.status_code == 404


def test_list_models(test_registry):
    async def run():
        async with _client(test_registry) as ac:
            r = await ac.get("/v1/models")
            return r

    r = asyncio.run(run())
    assert r.status_code == 200
    ids = [e["id"] for e in r.json()["data"]]
    assert "thinking" in ids
    assert "anthropic/claude-opus-4-7" in ids


def test_trace_endpoint_returns_record(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-opus-4-7", "x")

    async def run():
        async with _client(test_registry) as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "thinking",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            trace_id = r.headers["x-polyportia-trace-id"]
            t = await ac.get(f"/v1/traces/{trace_id}")
            return t

    t = asyncio.run(run())
    assert t.status_code == 200
    body = t.json()
    assert body["spans"][0]["kind"] in ("defined", "actual")
