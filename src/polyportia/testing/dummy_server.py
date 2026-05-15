"""A configurable, OpenAI-compatible dummy completions server.

The dummy is used for end-to-end verification of PolyPortia: it speaks the
OpenAI chat-completions protocol, supports streaming, tool calls, deliberate
errors, deliberate delays, and per-call sequencing — without requiring any
real provider credentials.

There are two ways to use it:

1. **As an in-process FastAPI app** (for tests using ``httpx.AsyncClient``):

   ```python
   from polyportia.testing.dummy_server import DummyServer
   dummy = DummyServer()
   dummy.register_fixed("model-a", "hello world")
   async with httpx.AsyncClient(app=dummy.app, base_url="http://dummy") as client:
       r = await client.post("/v1/chat/completions", json={...})
   ```

2. **As a standalone HTTP server**:

   ```bash
   polyportia testing dummy --port 9999
   ```

   Then point a PolyPortia provider at ``http://localhost:9999`` and exercise
   it like any other OpenAI-compatible backend.

Behavioural rules (in priority order, first match wins):
  1. An explicit registered handler for the exact model id.
  2. The model id matches a behavioural pattern:
       - ``dummy/echo``                      → echoes the last user message
       - ``dummy/fixed/<urlencoded text>``   → returns the text
       - ``dummy/error/<status>``            → returns HTTP error status
       - ``dummy/tool/<name>``               → returns a tool call to ``name``
                                                with arguments from the JSON
                                                body of the user message
       - ``dummy/delay/<ms>``                → sleeps ms then returns "ok"
       - ``dummy/usage/<pt>/<ct>``           → "ok" with the given token usage
  3. Default: a plain "ok" response.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class DummyCall:
    model: str
    body: dict[str, Any]
    stream: bool
    ts: float = field(default_factory=time.monotonic)


Handler = Callable[[dict[str, Any]], "DummyHandlerResult | Any"]


@dataclass
class DummyHandlerResult:
    """Structured result a handler can return."""

    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"
    error_status: int | None = None
    error_body: dict[str, Any] | None = None
    delay_ms: int = 0
    usage: dict[str, int] | None = None
    stream_chunks: list[str] | None = None


def _now_id() -> str:
    return f"chatcmpl-dummy-{int(time.time() * 1000)}"


def _openai_response(model: str, result: DummyHandlerResult) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.content or ""}
    if result.tool_calls:
        message["content"] = None
        message["tool_calls"] = result.tool_calls
    usage = result.usage or {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    return {
        "id": _now_id(),
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if result.tool_calls else result.finish_reason,
            }
        ],
        "usage": usage,
    }


def _stream_chunks(model: str, chunks: list[str]) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        for c in chunks:
            payload = {
                "id": _now_id(),
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
        final = {
            "id": _now_id(),
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return gen()


def _coerce(handler_out: Any, default_content: str = "ok") -> DummyHandlerResult:
    if isinstance(handler_out, DummyHandlerResult):
        return handler_out
    if isinstance(handler_out, str):
        return DummyHandlerResult(content=handler_out)
    if isinstance(handler_out, dict):
        return DummyHandlerResult(**handler_out)
    return DummyHandlerResult(content=default_content)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
    return ""


def _pattern_handler(model: str) -> DummyHandlerResult:  # noqa: C901 — pattern table
    """Resolve behaviour from the model id when no explicit handler is set."""
    if model == "dummy/echo":
        return DummyHandlerResult(content="__ECHO__")
    if model.startswith("dummy/fixed/"):
        return DummyHandlerResult(content=urllib.parse.unquote(model[len("dummy/fixed/") :]))
    if model.startswith("dummy/error/"):
        try:
            status = int(model[len("dummy/error/") :])
        except ValueError:
            status = 500
        return DummyHandlerResult(
            error_status=status,
            error_body={"error": {"code": f"dummy_{status}", "message": "dummy error"}},
        )
    if model.startswith("dummy/tool/"):
        name = model[len("dummy/tool/") :]
        return DummyHandlerResult(
            tool_calls=[
                {
                    "id": "call_dummy_1",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
            ],
            finish_reason="tool_calls",
        )
    if model.startswith("dummy/delay/"):
        try:
            ms = int(model[len("dummy/delay/") :].split("/")[0])
        except ValueError:
            ms = 0
        return DummyHandlerResult(content="ok", delay_ms=ms)
    if model.startswith("dummy/usage/"):
        parts = model[len("dummy/usage/") :].split("/")
        try:
            pt, ct = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pt, ct = 5, 5
        return DummyHandlerResult(
            content="ok", usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}
        )
    if model.startswith("dummy/stream/"):
        text = urllib.parse.unquote(model[len("dummy/stream/") :])
        return DummyHandlerResult(content=text, stream_chunks=[c for c in text])
    return DummyHandlerResult(content="ok")


class DummyServer:
    """An OpenAI-compatible dummy. Hold one per test or per session."""

    def __init__(self) -> None:
        self.handlers: dict[str, Handler] = {}
        self.calls: list[DummyCall] = []
        self.app = self._build_app()

    def reset(self) -> None:
        self.handlers.clear()
        self.calls.clear()

    def register(self, model: str, handler: Handler) -> None:
        self.handlers[model] = handler

    def register_fixed(
        self,
        model: str,
        content: str,
        *,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.register(model, lambda _: DummyHandlerResult(content=content, usage=usage))

    def register_sequence(self, model: str, items: list[Any]) -> None:
        idx = {"i": 0}

        def handler(body: dict[str, Any]) -> DummyHandlerResult:
            i = min(idx["i"], len(items) - 1)
            idx["i"] += 1
            item = items[i]
            if isinstance(item, BaseException):
                raise item
            return _coerce(item)

        self.register(model, handler)

    def register_tool(
        self,
        model: str,
        function_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        def handler(_: dict[str, Any]) -> DummyHandlerResult:
            return DummyHandlerResult(
                tool_calls=[
                    {
                        "id": "call_dummy",
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments or {}),
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        self.register(model, handler)

    def _resolve(self, model: str, body: dict[str, Any]) -> DummyHandlerResult:
        if model in self.handlers:
            return _coerce(self.handlers[model](body))
        result = _pattern_handler(model)
        if result.content == "__ECHO__":
            result.content = _last_user_text(body.get("messages") or [])
        return result

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="PolyPortia Dummy Completions", version="0.1.0")

        @app.get("/healthz")
        async def healthz() -> dict[str, str]:  # pragma: no cover — trivial
            return {"status": "ok"}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Any:
            body = await request.json()
            model = body.get("model", "dummy/echo")
            stream = bool(body.get("stream"))
            self.calls.append(DummyCall(model=model, body=body, stream=stream))
            result = self._resolve(model, body)
            if result.delay_ms:
                await asyncio.sleep(result.delay_ms / 1000)
            if result.error_status:
                raise HTTPException(
                    status_code=result.error_status,
                    detail=result.error_body or {"error": "dummy error"},
                )
            if stream:
                chunks = result.stream_chunks or [c for c in (result.content or "ok")]
                return StreamingResponse(
                    _stream_chunks(model, chunks),
                    media_type="text/event-stream",
                )
            return JSONResponse(_openai_response(model, result))

        @app.post("/admin/register")
        async def admin_register(body: dict[str, Any]) -> dict[str, bool]:
            self.register_fixed(body["model"], body.get("content", "ok"))
            return {"registered": True}

        @app.post("/admin/reset")
        async def admin_reset() -> dict[str, bool]:
            self.reset()
            return {"reset": True}

        @app.get("/admin/calls")
        async def admin_calls() -> list[dict[str, Any]]:
            return [{"model": c.model, "stream": c.stream, "body": c.body} for c in self.calls]

        return app


_default_dummy = DummyServer()


def get_default_dummy() -> DummyServer:
    return _default_dummy


def app() -> FastAPI:
    """Module-level FastAPI app factory for ``uvicorn polyportia.testing.dummy_server:app``."""
    return _default_dummy.app
