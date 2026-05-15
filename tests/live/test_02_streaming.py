"""Live test #2 — SSE streaming for single-model targets.

What this validates
-------------------
- Hitting ``POST /v1/chat/completions`` with ``stream=true`` opens a
  text/event-stream response.
- Multiple ``data: {...}`` chunks arrive (Ollama generates more than one
  token), and the stream terminates with ``data: [DONE]``.

How to run
----------
    RUN_LIVE_TESTS=1 pytest tests/live/test_02_streaming.py -v -s

Failure modes
-------------
- 400 response → the model resolved to a non-ActualModel target. This v1 only
  supports streaming for actual+defined targets that resolve to a single
  ActualModel.
- Stream hangs → Ollama may be loading the model into memory on first call;
  give it 30s and retry.
- Stream terminates with no content chunks → check the server logs for an
  upstream error.
"""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from polyportia.server.app import create_app


def test_stream_yields_multiple_chunks(live_registry, require_models):
    require_models("llama3.2:1b")
    app = create_app(registry=live_registry)

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
            chunks: list[str] = []
            async with ac.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "fast",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Count from 1 to 5."},
                    ],
                },
            ) as r:
                assert r.status_code == 200, await r.aread()
                async for chunk in r.aiter_text():
                    chunks.append(chunk)
            return "".join(chunks)

    body = asyncio.run(go())
    print(f"\n[live#02] full SSE body length: {len(body)}")
    print(f"[live#02] first 400 chars:\n{body[:400]}")

    # Expect at least one data: frame and a [DONE] terminator
    data_frames = [line for line in body.splitlines() if line.startswith("data:")]
    assert len(data_frames) >= 2, f"expected multiple data frames, got {len(data_frames)}"
    assert any("[DONE]" in line for line in data_frames)
