"""Live test #1 — basic single-model completion against Ollama.

What this validates
-------------------
- ``polyportia.acomplete(model="thinking", ...)`` resolves the DefinedModel
  ``thinking`` to its primary ActualModel (llama3.2:3b in the live YAML),
  builds a real LiteLLM ``acompletion`` call, hits Ollama, and gets a string
  response back.
- The trace builder records exactly one ``actual`` span (no fallback fired)
  with non-zero latency.

How to run
----------
    ollama serve &           # if not already running
    ollama pull llama3.2:3b
    RUN_LIVE_TESTS=1 pytest tests/live/test_01_single_model.py -v -s

Expected output
---------------
- Test passes
- With -s, you'll see the model's actual response printed to stdout, e.g.
  "PolyPortia is a model-agnostic LLM gateway..."

Failure modes to investigate
----------------------------
- ``Ollama not reachable`` → check ``ollama serve`` is running and that
  ``OLLAMA_BASE_URL`` matches.
- ``Ollama is missing required models`` → run the ``ollama pull`` command
  printed in the skip reason.
- The completion comes back but is empty — likely Ollama responded but the
  message field was malformed; check ``polyportia.providers.litellm_adapter``
  's _extract_content function.
"""

from __future__ import annotations

import asyncio

from polyportia.observability.store import get_default_store
from polyportia.sdk.client import acomplete


def test_single_defined_model_completes(live_registry, require_models):
    require_models("llama3.2:3b")

    async def go():
        return await acomplete(
            model="thinking",
            messages=[
                {"role": "user", "content": "Reply with exactly: PolyPortia is alive."},
            ],
            registry=live_registry,
        )

    result, trace_id = asyncio.run(go())
    print(f"\n[live#01] response: {result.content!r}")
    print(f"[live#01] trace_id: {trace_id}")

    assert result.content.strip(), "expected non-empty content from Ollama"

    # Trace inspection — there should be one or two spans (defined wrapping
    # actual), with the actual span having a positive latency.
    rec = get_default_store().get(trace_id)
    assert rec is not None
    actual_spans = [s for s in rec.spans if s.kind == "actual"]
    assert len(actual_spans) == 1
    assert (actual_spans[0].latency_ms or 0) > 0
    assert actual_spans[0].status == "ok"


def test_actual_model_id_passthrough(live_registry, require_models):
    """When the caller passes a literal provider/model id (instead of a
    defined-model name), PolyPortia should route it directly to that
    ActualModel without touching the defined-model layer.
    """
    require_models("llama3.2:1b")

    async def go():
        return await acomplete(
            model="ollama_chat/llama3.2:1b",
            messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
            registry=live_registry,
        )

    result, trace_id = asyncio.run(go())
    print(f"\n[live#01-passthrough] response: {result.content!r}")
    assert result.content.strip()

    rec = get_default_store().get(trace_id)
    assert rec is not None
    # No defined-model span this time — only the actual span.
    assert all(s.kind == "actual" for s in rec.spans)
