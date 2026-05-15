"""Dummy completions server: pattern matching + handlers + streaming."""

from __future__ import annotations

import json

import httpx
import pytest

from polyportia.testing.dummy_server import DummyServer


@pytest.fixture
def dummy() -> DummyServer:
    return DummyServer()


def _post(dummy: DummyServer, body: dict) -> httpx.Response:
    from fastapi.testclient import TestClient

    with TestClient(dummy.app) as client:
        return client.post("/v1/chat/completions", json=body)


def test_default_pattern_returns_ok(dummy: DummyServer) -> None:
    r = _post(dummy, {"model": "anything", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "ok"


def test_echo_pattern(dummy: DummyServer) -> None:
    r = _post(
        dummy,
        {"model": "dummy/echo", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert r.json()["choices"][0]["message"]["content"] == "hello"


def test_fixed_pattern_decodes_urlencoded(dummy: DummyServer) -> None:
    r = _post(
        dummy,
        {"model": "dummy/fixed/Hello%20world", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.json()["choices"][0]["message"]["content"] == "Hello world"


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_error_pattern(dummy: DummyServer, status: int) -> None:
    r = _post(
        dummy,
        {
            "model": f"dummy/error/{status}",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == status


def test_tool_call_pattern(dummy: DummyServer) -> None:
    r = _post(
        dummy,
        {
            "model": "dummy/tool/email",
            "messages": [{"role": "user", "content": "send"}],
        },
    )
    body = r.json()
    tc = body["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "email"
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_usage_pattern_returns_specified_counts(dummy: DummyServer) -> None:
    r = _post(
        dummy,
        {"model": "dummy/usage/42/17", "messages": [{"role": "user", "content": "x"}]},
    )
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] == 42
    assert usage["completion_tokens"] == 17
    assert usage["total_tokens"] == 59


def test_register_fixed_overrides_pattern(dummy: DummyServer) -> None:
    dummy.register_fixed("dummy/echo", "I do not echo")
    r = _post(
        dummy,
        {"model": "dummy/echo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.json()["choices"][0]["message"]["content"] == "I do not echo"


def test_register_sequence_advances_per_call(dummy: DummyServer) -> None:
    dummy.register_sequence("seq", ["one", "two", "three"])
    contents = [
        _post(dummy, {"model": "seq", "messages": [{"role": "user", "content": "x"}]})
        .json()["choices"][0]["message"]["content"]
        for _ in range(4)
    ]
    assert contents == ["one", "two", "three", "three"]  # last item repeats


def test_register_tool_emits_structured_tool_call(dummy: DummyServer) -> None:
    dummy.register_tool("model", "approve", {"reason": "ok"})
    r = _post(
        dummy,
        {"model": "model", "messages": [{"role": "user", "content": "x"}]},
    )
    tc = r.json()["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "approve"
    assert json.loads(tc["function"]["arguments"]) == {"reason": "ok"}


def test_streaming_returns_sse_chunks(dummy: DummyServer) -> None:
    from fastapi.testclient import TestClient

    with TestClient(dummy.app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "dummy/stream/hello",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        ) as r:
            chunks = [line for line in r.iter_lines() if line.startswith("data:")]
    assert chunks[-1] == "data: [DONE]"
    content_blob = " ".join(chunks[:-1])
    assert "h" in content_blob and "o" in content_blob


def test_admin_register_then_call(dummy: DummyServer) -> None:
    from fastapi.testclient import TestClient

    with TestClient(dummy.app) as client:
        client.post("/admin/register", json={"model": "x", "content": "registered!"})
        r = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.json()["choices"][0]["message"]["content"] == "registered!"


def test_calls_recorded_for_introspection(dummy: DummyServer) -> None:
    _post(dummy, {"model": "x", "messages": [{"role": "user", "content": "x"}]})
    _post(dummy, {"model": "y", "messages": [{"role": "user", "content": "x"}]})
    assert len(dummy.calls) == 2
    assert {c.model for c in dummy.calls} == {"x", "y"}


def test_reset_clears_handlers_and_calls(dummy: DummyServer) -> None:
    dummy.register_fixed("m", "v")
    _post(dummy, {"model": "m", "messages": [{"role": "user", "content": "x"}]})
    dummy.reset()
    assert dummy.calls == []
    r = _post(dummy, {"model": "m", "messages": [{"role": "user", "content": "x"}]})
    # After reset, the registered handler is gone — should fall through to pattern default
    assert r.json()["choices"][0]["message"]["content"] == "ok"
