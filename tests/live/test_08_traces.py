"""Live test #8 — trace endpoint returns the recorded spans.

What this validates
-------------------
- After making a request, the same process can fetch the trace via
  ``GET /v1/traces/{trace_id}`` and receive the full span tree.
- Token-usage fields are populated when the upstream provider reports them.
  Ollama may or may not — tests are tolerant.

How to run
----------
    RUN_LIVE_TESTS=1 pytest tests/live/test_08_traces.py -v -s
"""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from polyportia.server.app import create_app


def test_trace_endpoint_returns_record(live_registry, require_models):
    require_models("gemma4:e2b")
    app = create_app(registry=live_registry)

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
            r = await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "fast",
                    "messages": [{"role": "user", "content": "Say 'trace ok'."}],
                },
            )
            assert r.status_code == 200, r.text
            trace_id = r.headers["x-polyportia-trace-id"]
            t = await ac.get(f"/v1/traces/{trace_id}")
            return r, t, trace_id

    r, t, trace_id = asyncio.run(go())
    assert t.status_code == 200, t.text
    body = t.json()
    print(f"\n[live#08] trace_id: {trace_id}")
    print(f"[live#08] span count: {len(body['spans'])}")
    print(f"[live#08] final_status: {body['final_status']}")
    for span in body["spans"]:
        latency = span["latency_ms"] or 0
        print(
            f"  - {span['kind']:8s} {span['target_repr']:40s} "
            f"status={span['status']} latency_ms={latency:.1f}"
        )

    assert body["spans"]
    assert body["final_status"] == "ok"


def test_trace_list(live_registry, require_models):
    require_models("gemma4:e2b")
    app = create_app(registry=live_registry)

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
            await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "fast",
                    "messages": [{"role": "user", "content": "Say 'a'."}],
                },
            )
            return await ac.get("/v1/traces?limit=10")

    r = asyncio.run(go())
    assert r.status_code == 200
    body = r.json()
    print(f"\n[live#08-list] {len(body['data'])} traces in the ring")
    assert body["data"]
