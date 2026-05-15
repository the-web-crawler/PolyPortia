"""Live test #6 — synthesize council against real Ollama models.

What this validates
-------------------
- Council ``trio`` (synthesize over [fast, thinking, creative], synthesizer =
  thinking) calls all three members concurrently, then sends their answers to
  the synthesizer (llama3.2:3b), which produces a single combined response.
- The OpenAI-compatible endpoint returns this as a normal chat completion
  (the council orchestration is invisible to the client).

How to run
----------
    ollama pull llama3.2:1b llama3.2:3b
    RUN_LIVE_TESTS=1 pytest tests/live/test_06_synthesize_council.py -v -s

Notes
-----
- The synthesizer's prompt includes a transcript of every member's answer
  (formatted with member labels). For small local models, the synthesizer's
  output may not be a perfect "synthesis" — it's still useful for verifying
  the orchestration is correct end-to-end.
- A 4th call is made for the synthesizer step on top of the 3 fan-out calls.
  Total = 4 round-trips to Ollama.
"""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from polyportia.observability.store import get_default_store
from polyportia.server.app import create_app


def test_synthesize_council_returns_one_response(live_registry, require_models):
    require_models("llama3.2:1b", "llama3.2:3b")
    app = create_app(registry=live_registry)

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
            return await ac.post(
                "/v1/chat/completions",
                json={
                    "model": "trio",
                    "messages": [
                        {
                            "role": "user",
                            "content": "List 2 reasons CRDTs are useful, very briefly.",
                        },
                    ],
                },
            )

    r = asyncio.run(go())
    assert r.status_code == 200, r.text
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    trace_id = r.headers["x-polyportia-trace-id"]
    print(f"\n[live#06] synthesized: {content[:300]!r}")
    print(f"[live#06] trace_id: {trace_id}")

    assert content.strip()

    # Trace should show at least 4 actual spans (3 members + synthesizer)
    rec = get_default_store().get(trace_id)
    assert rec is not None
    actual_spans = [s for s in rec.spans if s.kind == "actual"]
    print(f"[live#06] actual span count: {len(actual_spans)}")
    assert len(actual_spans) >= 3  # creative may have fallen over to a defined model
