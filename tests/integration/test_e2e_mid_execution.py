"""Mid-execution budget stop via the dummy.

The estimator is conservative, so the only realistic way to overshoot is via
usage that exceeds ``max_tokens`` (e.g., a model lying about counts) or via
fallbacks/retries. We force the dummy to report inflated token usage on the
first member of a synthesize council, then verify the second member or the
synthesizer never runs.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from polyportia.config.loader import load_config_from_string
from polyportia.config.registry import Registry
from polyportia.observability.store import TraceStore
from polyportia.server.app import create_app
from polyportia.testing.dummy_server import DummyHandlerResult, DummyServer

_YAML = """
providers:
  - {name: anthropic, api_key: x}
actual_models:
  - id: tiny
    provider: anthropic
    max_output_tokens: 100
    input_cost_per_1m_tokens: 10.0
    output_cost_per_1m_tokens: 20.0
councils:
  - name: triad
    strategy:
      kind: synthesize
      members:
        - {kind: actual, id: tiny}
        - {kind: actual, id: tiny}
      synthesizer: {kind: actual, id: tiny}
"""


@pytest.fixture
def dummy_app(monkeypatch: pytest.MonkeyPatch) -> tuple[DummyServer, Any]:
    dummy = DummyServer()
    transport = httpx.ASGITransport(app=dummy.app)

    async def routed(**kwargs: Any) -> Any:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://dummy"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages") or [],
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(f"dummy status {resp.status_code}")
            return resp.json()

    monkeypatch.setattr("polyportia.providers.litellm_adapter.acompletion", routed)
    cfg = load_config_from_string(_YAML)
    app = create_app()
    app.state.registry = Registry(cfg)
    app.state.trace_store = TraceStore()
    return dummy, app


def test_mid_execution_402_when_dummy_returns_inflated_usage(
    dummy_app: tuple[DummyServer, Any]
) -> None:
    dummy, app = dummy_app
    # Each call reports 1,000,000 output tokens — enough to single-handedly
    # blow a tight budget. The synthesizer (and second member) should not run.
    dummy.register(
        "tiny",
        lambda _: DummyHandlerResult(
            content="ok",
            usage={"prompt_tokens": 1, "completion_tokens": 1_000_000, "total_tokens": 1_000_001},
        ),
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/councils/triad/run",
            json={
                "messages": [{"role": "user", "content": "x"}],
                # Pre-flight estimate (1×in*3 + 100×out*3 = ~6e-3) fits below;
                # one inflated call (1M × $20/M = $20) overshoots easily.
                "polyportia": {"budget_usd": 0.01},
            },
        )
    assert r.status_code == 402
    body = r.json()
    assert body["error"]["stage"] == "mid_execution"
    assert body["error"]["actual_usd_so_far"] > 0
    # Only the first member ran; synthesizer never reached.
    assert len(dummy.calls) < 3


def test_mid_execution_actual_cost_recorded_in_envelope(
    dummy_app: tuple[DummyServer, Any]
) -> None:
    """A mid-execution 402 response includes the amount spent before stopping."""
    dummy, app = dummy_app
    dummy.register(
        "tiny",
        lambda _: DummyHandlerResult(
            content="ok",
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1_000_000,
                "total_tokens": 1_000_001,
            },
        ),
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/councils/triad/run",
            json={
                "messages": [{"role": "user", "content": "x"}],
                "polyportia": {"budget_usd": 0.01},
            },
        )
    body = r.json()
    assert r.status_code == 402
    assert body["error"]["stage"] == "mid_execution"
    # Cost headers populated on the 402 too.
    assert "x-polyportia-cost-usd" in r.headers
    assert float(r.headers["x-polyportia-cost-usd"]) > 0.0
