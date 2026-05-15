"""POST /v1/councils/{name}/run array envelope endpoint."""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from polyportia.server.app import create_app


def _client(registry):
    app = create_app(registry=registry)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def test_council_run_returns_array(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-haiku-4-5", "h")
    mock_provider.set_response("anthropic/claude-opus-4-7", "o")
    mock_provider.set_response("openai/gpt-5-5", "g")

    async def run():
        async with _client(test_registry) as ac:
            return await ac.post(
                "/v1/councils/triad-raw/run",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    r = asyncio.run(run())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["trace_id"]
    assert len(body["responses"]) == 3
    contents = sorted(x["content"] for x in body["responses"])
    assert contents == ["g", "h", "o"]


def test_synthesize_council_via_openai_endpoint(mock_provider, test_registry):
    mock_provider.set_response("anthropic/claude-haiku-4-5", "h")
    mock_provider.set_response("openai/gpt-5-5", "g")
    mock_provider.set_response("anthropic/claude-opus-4-7", "synth-final")

    async def run():
        async with _client(test_registry) as ac:
            return await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "triad",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

    r = asyncio.run(run())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "synth-final"


def test_council_unknown_404(test_registry):
    async def run():
        async with _client(test_registry) as ac:
            return await ac.post(
                "/v1/councils/ghost/run",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    r = asyncio.run(run())
    assert r.status_code == 404
