"""OpenAI-compatible chat completions + model listing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse

from polyportia.config.models import ActualModelRef, DefinedModelRef
from polyportia.config.registry import Registry, get_default_registry
from polyportia.council.context import ExecutionContext
from polyportia.council.executor import FallbacksExhaustedError, execute_target
from polyportia.observability.store import TraceStore, get_default_store
from polyportia.observability.trace import TraceBuilder
from polyportia.providers.errors import RetryableExhaustedError
from polyportia.sdk.client import resolve_model_input
from polyportia.server.schemas import (
    ChatCompletionsRequest,
    ModelList,
    ModelListEntry,
    messages_to_dicts,
)
from polyportia.server.streaming import stream_single_model

router = APIRouter()


def _registry(request: Request) -> Registry:
    return getattr(request.app.state, "registry", get_default_registry())


def _store(request: Request) -> TraceStore:
    return getattr(request.app.state, "trace_store", get_default_store())


@router.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    reg = _registry(request)
    entries: list[ModelListEntry] = []
    for d in reg.list_defined_models():
        entries.append(ModelListEntry(id=d.name, polyportia_kind="defined"))
    for c in reg.list_councils():
        entries.append(ModelListEntry(id=c.name, polyportia_kind="council"))
    for a in reg.list_actual_models():
        entries.append(ModelListEntry(id=a.id, polyportia_kind="actual"))
    return ModelList(data=entries)


def _extract_overrides(req: ChatCompletionsRequest) -> dict[str, Any]:
    overrides = req.polyportia
    return {
        "retry": overrides.retry if overrides else None,
        "timeout_s": overrides.timeout_s if overrides else None,
    }


def _passthrough_params(req: ChatCompletionsRequest) -> dict[str, Any]:
    excluded = {"model", "messages", "stream", "polyportia"}
    return {k: v for k, v in req.model_dump(exclude_none=True).items() if k not in excluded}


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, request: Request) -> Response:
    reg = _registry(request)
    store = _store(request)
    try:
        target = resolve_model_input(req.model, reg)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    overrides = _extract_overrides(req)
    request_params = _passthrough_params(req)
    messages = messages_to_dicts(req.messages)

    if req.stream:
        if not isinstance(target, (ActualModelRef, DefinedModelRef)):
            raise HTTPException(
                status_code=400,
                detail="stream=true is supported only for single-model targets in v1",
            )

        async def gen() -> AsyncIterator[str]:
            async for chunk in stream_single_model(
                target=target,
                messages=messages,
                registry=reg,
                request_params=request_params,
                request_timeout_s=overrides["timeout_s"],
            ):
                yield chunk

        return EventSourceResponse(gen())

    trace = TraceBuilder({"model": req.model, "message_count": len(req.messages)})
    ctx = ExecutionContext(
        registry=reg,
        trace=trace,
        request_params=request_params,
        request_retry=overrides["retry"],
        request_timeout_s=overrides["timeout_s"],
    )
    try:
        result = await execute_target(target, messages, ctx)
    except (RetryableExhaustedError, FallbacksExhaustedError) as e:
        store.add(trace.finalize())
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception:
        store.add(trace.finalize())
        raise
    store.add(trace.finalize())

    body = _to_openai_response(result, model=req.model)
    headers = {"x-polyportia-trace-id": trace.trace_id}
    return Response(
        content=json.dumps(body),
        media_type="application/json",
        headers=headers,
    )


def _to_openai_response(result: Any, *, model: str) -> dict[str, Any]:
    """Best-effort serialise to OpenAI shape.

    If the underlying litellm raw response carries an ``id``, ``created``, etc.
    we surface them; otherwise we synthesize a minimal valid envelope.
    """
    raw = result.raw
    if raw is not None and hasattr(raw, "model_dump"):
        try:
            data = cast(dict[str, Any], raw.model_dump())
            data["model"] = model
            return data
        except Exception:
            pass
    if isinstance(raw, dict):
        data = dict(raw)
        data["model"] = model
        return data
    return {
        "id": "chatcmpl-polyportia",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.content},
                "finish_reason": result.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": (result.usage.prompt_tokens if result.usage else 0) or 0,
            "completion_tokens": (result.usage.completion_tokens if result.usage else 0) or 0,
            "total_tokens": (result.usage.total_tokens if result.usage else 0) or 0,
        },
    }
