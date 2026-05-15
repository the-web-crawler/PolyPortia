"""POST /v1/councils/{name}/run — always returns the array envelope."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from polyportia.config.models import CouncilRef
from polyportia.config.registry import Registry, get_default_registry
from polyportia.council.context import ExecutionContext
from polyportia.council.executor import FallbacksExhaustedError, execute_target
from polyportia.council.failure import CouncilFailureError
from polyportia.observability.store import TraceStore, get_default_store
from polyportia.observability.trace import TraceBuilder
from polyportia.providers.errors import RetryableExhaustedError
from polyportia.server.schemas import CouncilRunRequest, messages_to_dicts

router = APIRouter()


def _registry(request: Request) -> Registry:
    return getattr(request.app.state, "registry", get_default_registry())


def _store(request: Request) -> TraceStore:
    return getattr(request.app.state, "trace_store", get_default_store())


@router.post("/v1/councils/{name}/run")
async def run_council(name: str, req: CouncilRunRequest, request: Request) -> dict[str, Any]:
    reg = _registry(request)
    store = _store(request)
    if not reg.has_council(name):
        raise HTTPException(status_code=404, detail=f"council '{name}' not found")

    overrides = req.polyportia
    request_params: dict[str, Any] = {}

    trace = TraceBuilder({"council": name, "message_count": len(req.messages)})
    ctx = ExecutionContext(
        registry=reg,
        trace=trace,
        request_params=request_params,
        request_retry=overrides.retry if overrides else None,
        request_timeout_s=overrides.timeout_s if overrides else None,
    )

    target = CouncilRef(name=name)
    messages = messages_to_dicts(req.messages)
    try:
        result = await execute_target(target, messages, ctx)
    except CouncilFailureError as e:
        store.add(trace.finalize())
        raise HTTPException(status_code=422, detail=str(e)) from e
    except (RetryableExhaustedError, FallbacksExhaustedError) as e:
        store.add(trace.finalize())
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception:
        store.add(trace.finalize())
        raise
    store.add(trace.finalize())

    body: dict[str, Any] = {"trace_id": trace.trace_id}
    if isinstance(result.raw, dict) and result.raw.get("object") == "polyportia.council":
        body["responses"] = result.raw.get("responses", [])
    else:
        body["synthesized"] = {
            "model": result.model_id,
            "content": result.content,
            "finish_reason": result.finish_reason,
        }
        if result.usage is not None:
            body["synthesized"]["usage"] = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }
    return body
