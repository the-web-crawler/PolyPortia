"""Live test #5 — parallel_array council against real Ollama models.

What this validates
-------------------
- Council ``trio-raw`` (parallel_array of [fast, thinking, creative])
  fans out three concurrent calls to Ollama and returns one response per
  member.
- The HTTP endpoint ``POST /v1/councils/trio-raw/run`` returns the array
  envelope shape: ``{trace_id, responses: [...]}`` with one entry per
  member.

How to run
----------
    ollama pull llama3.2:1b llama3.2:3b mistral:7b   # all three
    # OR
    ollama pull llama3.2:1b llama3.2:3b              # creative will fall back
    RUN_LIVE_TESTS=1 pytest tests/live/test_05_parallel_council.py -v -s

Notes on timing
---------------
- All three calls go in parallel (asyncio.gather); total wall-clock should be
  roughly the **slowest** of the three, not the sum.
- First-call latency is dominated by Ollama loading the model into memory.
  Re-run to see the warm-cache speed.
"""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from polyportia.server.app import create_app


def test_parallel_council_returns_one_per_member(live_registry, require_models):
    require_models("llama3.2:1b", "llama3.2:3b")
    app = create_app(registry=live_registry)

    async def go():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
            return await ac.post(
                "/v1/councils/trio-raw/run",
                json={
                    "messages": [
                        {"role": "user", "content": "Reply with exactly one short sentence."},
                    ],
                },
            )

    r = asyncio.run(go())
    assert r.status_code == 200, r.text
    body = r.json()
    print(f"\n[live#05] trace_id: {body['trace_id']}")
    print(f"[live#05] responses count: {len(body['responses'])}")
    for entry in body["responses"]:
        if "content" in entry:
            print(f"  - {entry['member']}: {entry['content'][:80]!r}")
        else:
            print(f"  - {entry['member']}: ERROR {entry.get('error')}")

    assert len(body["responses"]) == 3
    # At least the two llama-based members should succeed (mistral may fall back).
    successes = [e for e in body["responses"] if "content" in e]
    assert len(successes) >= 2
