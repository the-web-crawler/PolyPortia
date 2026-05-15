"""Live test #7 — recursive council-of-councils against Ollama.

What this validates
-------------------
- Council ``meta`` is a ``synthesize`` whose first member is ``council:trio``
  (itself a synthesize council) and whose synthesizer is ``defined:thinking``.
- Calling ``meta`` triggers the recursion: the executor's single
  ``execute_target`` entry-point handles the nested council without any
  special-case code path.

How to run
----------
    ollama pull llama3.2:1b llama3.2:3b
    RUN_LIVE_TESTS=1 pytest tests/live/test_07_meta_council.py -v -s

Why this matters
----------------
- This is the design's load-bearing feature: a synthesizer can itself be a
  group of models, so users can build hierarchical councils without
  PolyPortia knowing or caring about the topology.
"""

from __future__ import annotations

import asyncio

from polyportia.observability.store import get_default_store
from polyportia.sdk.client import acomplete


def test_meta_council_runs_end_to_end(live_registry, require_models):
    require_models("gemma4:e2b", "lfm2.5-thinking:latest")

    async def go():
        return await acomplete(
            model="meta",
            messages=[{"role": "user", "content": "Say 'meta-ok' and nothing else."}],
            registry=live_registry,
        )

    result, trace_id = asyncio.run(go())
    print(f"\n[live#07] response: {result.content!r}")
    print(f"[live#07] trace_id: {trace_id}")

    assert result.content.strip()
    rec = get_default_store().get(trace_id)
    assert rec is not None
    # Trace span tree: meta synthesize span → trio synthesize span (nested) →
    # member spans → trio synthesizer → meta synthesizer
    synthesize_spans = [s for s in rec.spans if s.kind == "synthesize"]
    print(f"[live#07] synthesize spans: {len(synthesize_spans)}")
    assert len(synthesize_spans) >= 2  # meta + trio
