"""Live test #4 — retry policy fires on a transient error.

What this validates
-------------------
- A request-level retry policy is honoured.
- The trace's per-actual-span ``retry_attempts`` array records every attempt.

This test doesn't strictly *require* a transient failure to occur (Ollama is
usually reliable locally), so the assertions are conservative: we verify that
the retry loop ran and produced a sensible attempts record on success. The
retry behaviour on real failures is exercised by test_03_fallback when the
primary errors.

How to run
----------
    RUN_LIVE_TESTS=1 pytest tests/live/test_04_retry.py -v -s

Failure modes
-------------
- The trace span is missing retry_attempts entirely → check that
  ``call_with_retries`` is recording attempts via the on_attempt callback,
  even on first-try success.
"""

from __future__ import annotations

import asyncio

from polyportia.config.models import RetryPolicy
from polyportia.observability.store import get_default_store
from polyportia.sdk.client import acomplete


def test_first_try_success_records_one_attempt(live_registry, require_models):
    require_models("gemma4:e2b")

    async def go():
        return await acomplete(
            model="fast",
            messages=[{"role": "user", "content": "Say 'ok'."}],
            registry=live_registry,
            retry=RetryPolicy(max_retries=2, backoff_base_s=0.1, jitter=False),
        )

    result, trace_id = asyncio.run(go())
    print(f"\n[live#04] response: {result.content!r}")
    rec = get_default_store().get(trace_id)
    assert rec is not None
    actual_spans = [s for s in rec.spans if s.kind == "actual"]
    assert actual_spans
    # On first-try success, exactly one attempt should be recorded.
    assert len(actual_spans[0].retry_attempts) == 1
    assert actual_spans[0].retry_attempts[0].error is None
