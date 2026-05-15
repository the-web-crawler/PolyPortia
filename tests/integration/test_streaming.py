"""SSE streaming for single-model targets."""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from polyportia.server.app import create_app


def _client(registry):
    app = create_app(registry=registry)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def test_stream_single_model_sse(mock_stream_provider, test_registry):
    async def run():
        async with _client(test_registry) as ac:
            async with ac.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "thinking",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as r:
                body = ""
                async for chunk in r.aiter_text():
                    body += chunk
                return r.status_code, body

    status, body = asyncio.run(run())
    assert status == 200
    assert "data:" in body
    assert "[DONE]" in body
