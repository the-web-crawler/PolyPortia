"""End-to-end coverage: PolyPortia → dummy completions endpoint.

Each test routes ``polyportia.providers.litellm_adapter.acompletion`` through
the in-process ``DummyServer`` (via ASGI transport), so the full PolyPortia
pipeline runs against realistic OpenAI-shape responses produced by the dummy.
Every council strategy and policy gets one focused test.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
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
  - {name: openai, api_key: y}
actual_models:
  - id: alpha
    provider: anthropic
    max_output_tokens: 64
    input_cost_per_1m_tokens: 1.0
    output_cost_per_1m_tokens: 2.0
  - id: beta
    provider: openai
    max_output_tokens: 64
    input_cost_per_1m_tokens: 1.0
    output_cost_per_1m_tokens: 2.0
  - id: gamma
    provider: anthropic
    max_output_tokens: 64
    input_cost_per_1m_tokens: 1.0
    output_cost_per_1m_tokens: 2.0
defined_models:
  - name: fast
    target: {kind: actual, id: alpha}
    fallbacks:
      - {kind: actual, id: beta}
  - name: thinking
    target: {kind: actual, id: alpha}
  - name: creative
    target: {kind: actual, id: beta}
councils:
  - name: trio-array
    strategy:
      kind: parallel_array
      members:
        - {kind: actual, id: alpha}
        - {kind: actual, id: beta}
        - {kind: actual, id: gamma}
  - name: trio-synth
    strategy:
      kind: synthesize
      members:
        - {kind: actual, id: alpha}
        - {kind: actual, id: beta}
      synthesizer: {kind: actual, id: gamma}
  - name: meta
    strategy:
      kind: synthesize
      members:
        - {kind: council, name: trio-synth}
        - {kind: actual, id: beta}
      synthesizer: {kind: actual, id: gamma}
  - name: debate
    strategy:
      kind: debate
      members:
        - {kind: actual, id: alpha}
        - {kind: actual, id: beta}
      debate: {turns: 2, visibility: prompt_and_peer_responses}
      output: array
  - name: review-AND
    strategy:
      kind: propose_review
      proposer: {kind: actual, id: alpha}
      reviewers:
        - {kind: actual, id: beta}
        - {kind: actual, id: gamma}
      consensus: all
      max_revisions: 1
      on_denial: return_denial
"""


@pytest.fixture
def dummy_and_app(monkeypatch: pytest.MonkeyPatch) -> tuple[DummyServer, Any]:
    dummy = DummyServer()
    transport = httpx.ASGITransport(app=dummy.app)

    async def routed(**kwargs: Any) -> Any:
        body: dict[str, Any] = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages") or [],
        }
        for k in ("stream", "max_tokens", "temperature", "top_p", "tools", "tool_choice"):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]
        if not body.get("stream"):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://dummy"
            ) as client:
                resp = await client.post("/v1/chat/completions", json=body)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"dummy returned status {resp.status_code}: {resp.text}"
                    )
                return resp.json()

        async def gen() -> AsyncIterator[dict[str, Any]]:
            client = httpx.AsyncClient(transport=transport, base_url="http://dummy")
            try:
                async with client.stream(
                    "POST", "/v1/chat/completions", json=body
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[len("data: ") :].strip()
                        if payload == "[DONE]":
                            break
                        yield json.loads(payload)
            finally:
                await client.aclose()

        return gen()

    monkeypatch.setattr("polyportia.providers.litellm_adapter.acompletion", routed)

    cfg = load_config_from_string(_YAML)
    app = create_app()
    app.state.registry = Registry(cfg)
    app.state.trace_store = TraceStore()
    return dummy, app


def _post(app: Any, path: str, body: dict[str, Any]) -> httpx.Response:
    with TestClient(app) as client:
        return client.post(path, json=body)


# --- Single model + DefinedModel ---


def test_single_actual_model_via_dummy(dummy_and_app: tuple[DummyServer, Any]) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "alpha-response")
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "alpha",
            "messages": [{"role": "user", "content": "hello"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "alpha-response"


def test_defined_model_via_dummy(dummy_and_app: tuple[DummyServer, Any]) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "fast-via-alpha")
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "fast",
            "messages": [{"role": "user", "content": "hi"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "fast-via-alpha"


def test_defined_model_falls_back_on_primary_error(
    dummy_and_app: tuple[DummyServer, Any]
) -> None:
    dummy, app = dummy_and_app
    dummy.register("alpha", lambda _: DummyHandlerResult(error_status=503))
    dummy.register_fixed("beta", "from-fallback")
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "fast",
            "messages": [{"role": "user", "content": "hi"}],
            "polyportia": {
                "budget_usd": "unlimited",
                "retry": {"max_retries": 0, "retry_on": ["server_error"]},
            },
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "from-fallback"


# --- Streaming ---


def test_streaming_single_model_via_dummy(
    dummy_and_app: tuple[DummyServer, Any]
) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "hello")
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "alpha",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
                "polyportia": {"budget_usd": "unlimited"},
            },
        ) as r:
            assert r.status_code == 200
            lines = [line for line in r.iter_lines() if line.startswith("data:")]
    # SSE chunks reach the client
    assert any("e" in line or "l" in line or "o" in line for line in lines)


# --- Council strategies ---


def test_parallel_array_via_dummy(dummy_and_app: tuple[DummyServer, Any]) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "A")
    dummy.register_fixed("beta", "B")
    dummy.register_fixed("gamma", "C")
    r = _post(
        app,
        "/v1/councils/trio-array/run",
        {
            "messages": [{"role": "user", "content": "x"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    contents = {x["content"] for x in body["responses"] if "content" in x}
    assert contents == {"A", "B", "C"}


def test_synthesize_via_dummy(dummy_and_app: tuple[DummyServer, Any]) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "A")
    dummy.register_fixed("beta", "B")
    dummy.register_fixed("gamma", "synthesized: A+B")
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "trio-synth",
            "messages": [{"role": "user", "content": "x"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    assert "synthesized" in r.json()["choices"][0]["message"]["content"]


def test_meta_council_recursive_via_dummy(
    dummy_and_app: tuple[DummyServer, Any]
) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "a")
    dummy.register_fixed("beta", "b")
    dummy.register_fixed("gamma", "META-RESULT")
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "meta",
            "messages": [{"role": "user", "content": "x"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    # Gamma is the outermost synthesizer
    assert r.json()["choices"][0]["message"]["content"] == "META-RESULT"


def test_debate_via_dummy(dummy_and_app: tuple[DummyServer, Any]) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "a")
    dummy.register_fixed("beta", "b")
    r = _post(
        app,
        "/v1/councils/debate/run",
        {
            "messages": [{"role": "user", "content": "topic"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "responses" in body
    # 2 members × 2 turns = 4 calls
    assert len([c for c in dummy.calls if c.model in {"alpha", "beta"}]) == 4


# --- ProposeAndReview via tool-call dummies ---


def test_propose_review_all_approve_via_dummy(
    dummy_and_app: tuple[DummyServer, Any]
) -> None:
    dummy, app = dummy_and_app
    dummy.register_tool("alpha", "email", {"to": "alice@example.com"})
    dummy.register_tool("beta", "approve", {"reason": "ok"})
    dummy.register_tool("gamma", "approve", {})
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "review-AND",
            "messages": [{"role": "user", "content": "send email"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    tool_calls = r.json()["choices"][0]["message"].get("tool_calls", [])
    assert tool_calls and tool_calls[0]["function"]["name"] == "email"


def test_propose_review_deny_then_revise_then_approve(
    dummy_and_app: tuple[DummyServer, Any]
) -> None:
    dummy, app = dummy_and_app

    # Proposer always emits same tool call; reviewer denies first then approves.
    dummy.register_tool("alpha", "email", {"to": "alice@example.com"})

    state = {"i": 0}

    def reviewer(_: dict) -> DummyHandlerResult:
        state["i"] += 1
        if state["i"] <= 2:
            # First round both reviewers deny.
            return DummyHandlerResult(
                tool_calls=[
                    {
                        "id": f"r{state['i']}",
                        "type": "function",
                        "function": {"name": "deny", "arguments": json.dumps({"reason": "no"})},
                    }
                ],
                finish_reason="tool_calls",
            )
        return DummyHandlerResult(
            tool_calls=[
                {
                    "id": f"r{state['i']}",
                    "type": "function",
                    "function": {"name": "approve", "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
        )

    dummy.register("beta", reviewer)
    dummy.register("gamma", reviewer)

    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "review-AND",
            "messages": [{"role": "user", "content": "send"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    # Two revision rounds: initial (denied) + revision (approved). Reviewers ran twice each.
    assert state["i"] == 4


def test_propose_review_denial_returns_text(
    dummy_and_app: tuple[DummyServer, Any]
) -> None:
    dummy, app = dummy_and_app
    dummy.register_tool("alpha", "email", {"to": "spam@example.com"})
    dummy.register_tool("beta", "deny", {"reason": "spam"})
    dummy.register_tool("gamma", "deny", {"reason": "spam"})
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "review-AND",
            "messages": [{"role": "user", "content": "send"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    assert "not approved" in content.lower()
    assert "DENY" in content


# --- Budget enforcement via dummy ---


def test_pre_flight_402_via_dummy(dummy_and_app: tuple[DummyServer, Any]) -> None:
    _, app = dummy_and_app
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "trio-array",
            "messages": [{"role": "user", "content": "x"}],
            "polyportia": {"budget_usd": 0.0000001},
        },
    )
    assert r.status_code == 402
    assert r.json()["error"]["stage"] == "pre_flight"


# --- Cost headers + body present on every success ---


def test_cost_headers_on_every_response(dummy_and_app: tuple[DummyServer, Any]) -> None:
    dummy, app = dummy_and_app
    dummy.register_fixed("alpha", "ok")
    r = _post(
        app,
        "/v1/chat/completions",
        {
            "model": "alpha",
            "messages": [{"role": "user", "content": "hi"}],
            "polyportia": {"budget_usd": "unlimited"},
        },
    )
    assert r.status_code == 200
    assert "x-polyportia-cost-usd" in r.headers
    assert "x-polyportia-cost-predicted-usd" in r.headers
    assert "x-polyportia-trace-id" in r.headers


# --- /v1/models lists everything ---


def test_models_endpoint_lists_defined_and_council_and_actual(
    dummy_and_app: tuple[DummyServer, Any]
) -> None:
    _, app = dummy_and_app
    with TestClient(app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    kinds = {entry["polyportia_kind"] for entry in r.json()["data"]}
    assert kinds == {"actual", "defined", "council"}
