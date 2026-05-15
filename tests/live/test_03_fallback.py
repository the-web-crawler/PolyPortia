"""Live test #3 — DefinedModel fallback chain fires on a real provider error.

What this validates
-------------------
- DefinedModel ``brittle`` is defined with a primary target of
  ``ollama_chat/this-model-does-not-exist`` (which doesn't exist in Ollama).
- Its fallback chain is ``[defined:fast]``, where ``fast`` resolves to
  llama3.2:1b (which IS pulled).
- Calling ``acomplete(model="brittle", ...)`` should:
    1. Attempt the broken primary, get an error from Ollama.
    2. Fall over to the fallback (``defined:fast`` → llama3.2:1b).
    3. Return llama's response.
- The trace records the failed primary attempt **and** the successful
  fallback attempt, with ``fallback_chain`` populated on the defined span.

How to run
----------
    ollama pull llama3.2:1b
    RUN_LIVE_TESTS=1 pytest tests/live/test_03_fallback.py -v -s

Failure modes
-------------
- The "broken" model id no longer errors → Ollama may have changed behaviour
  for unknown models. Adjust ``polyportia.live.yaml`` to use a different
  obviously-broken id.
- The fallback also fails → check ``ollama list`` to confirm llama3.2:1b is
  pulled.
"""

from __future__ import annotations

import asyncio

from polyportia.observability.store import get_default_store
from polyportia.sdk.client import acomplete


def test_fallback_kicks_in_when_primary_is_broken(live_registry, require_models):
    require_models("llama3.2:1b")

    async def go():
        return await acomplete(
            model="brittle",
            messages=[{"role": "user", "content": "Say 'fallback ok'"}],
            registry=live_registry,
        )

    result, trace_id = asyncio.run(go())
    print(f"\n[live#03] response: {result.content!r}")
    rec = get_default_store().get(trace_id)
    assert rec is not None

    # The defined-model span should record both targets in fallback_chain.
    defined_spans = [s for s in rec.spans if s.kind == "defined"]
    assert len(defined_spans) >= 1, "expected at least one defined-model span"
    chain = defined_spans[0].fallback_chain
    print(f"[live#03] fallback_chain: {chain}")
    assert any("this-model-does-not-exist" in entry for entry in chain), chain
    assert any("fast" in entry or "llama3.2:1b" in entry for entry in chain), chain

    # We expect 2 actual spans: one failed (the broken primary), one ok
    # (the fallback). Note: the broken one may be wrapped as an error span.
    actual_spans = [s for s in rec.spans if s.kind == "actual"]
    statuses = [s.status for s in actual_spans]
    assert "ok" in statuses, f"expected at least one successful actual span, got {statuses}"
    assert "error" in statuses, f"expected the broken primary to fail, got {statuses}"
