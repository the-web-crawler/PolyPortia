"""OpenAI-compatible chat completions + model listing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse

from polyportia.budget.enforcer import BudgetEnforcer
from polyportia.budget.errors import BudgetExceededError
from polyportia.budget.estimator import estimate_cost
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
        "budget_usd": overrides.budget_usd if overrides else None,
        "include_cost": overrides.include_cost if overrides else False,
    }


def _passthrough_params(req: ChatCompletionsRequest) -> dict[str, Any]:
    excluded = {"model", "messages", "stream", "polyportia"}
    return {k: v for k, v in req.model_dump(exclude_none=True).items() if k not in excluded}


def _resolve_budget(request_value: Any, registry: Registry) -> float | None:
    """Apply the cascade: request-override > config default. ``"unlimited"`` disables."""
    if request_value == "unlimited":
        return None
    if isinstance(request_value, (int, float)):
        return float(request_value)
    return registry.budget_usd_default


def _sum_trace_cost(trace: TraceBuilder) -> float:
    total = 0.0
    for s in trace.record.spans:
        if s.cost_usd is not None:
            total += float(s.cost_usd)
    return total


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
    budget_value = _resolve_budget(overrides["budget_usd"], reg)
    estimate = estimate_cost(target, messages, request_params, reg)

    if budget_value is not None and estimate.total_usd > budget_value:
        exc = BudgetExceededError(
            f"Predicted ${estimate.total_usd:.6f} exceeds budget ${budget_value:.6f}",
            stage="pre_flight",
            budget_usd=budget_value,
            predicted_usd=estimate.total_usd,
            breakdown=estimate.breakdown,
        )
        return Response(
            content=json.dumps(exc.to_envelope()),
            status_code=402,
            media_type="application/json",
            headers={"x-polyportia-cost-predicted-usd": f"{estimate.total_usd:.6f}"},
        )

    enforcer = BudgetEnforcer(budget_usd=budget_value)
    ctx = ExecutionContext(
        registry=reg,
        trace=trace,
        request_params=request_params,
        request_retry=overrides["retry"],
        request_timeout_s=overrides["timeout_s"],
        budget=enforcer,
    )
    try:
        result = await execute_target(target, messages, ctx)
    except BudgetExceededError as e:
        store.add(trace.finalize())
        envelope = e.to_envelope()
        envelope["error"]["actual_usd_so_far"] = enforcer.spent_usd
        return Response(
            content=json.dumps(envelope),
            status_code=402,
            media_type="application/json",
            headers={
                "x-polyportia-trace-id": trace.trace_id,
                "x-polyportia-cost-usd": f"{enforcer.spent_usd:.6f}",
                "x-polyportia-cost-predicted-usd": f"{estimate.total_usd:.6f}",
            },
        )
    except (RetryableExhaustedError, FallbacksExhaustedError) as e:
        store.add(trace.finalize())
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception:
        store.add(trace.finalize())
        raise
    store.add(trace.finalize())

    actual_cost = _sum_trace_cost(trace)
    body = _to_openai_response(result, model=req.model)
    if overrides["include_cost"]:
        body.setdefault("polyportia", {})["cost"] = {
            "actual_usd": actual_cost,
            "predicted_usd": estimate.total_usd,
            "by_model": estimate.to_dict()["breakdown"],
        }
    headers = {
        "x-polyportia-trace-id": trace.trace_id,
        "x-polyportia-cost-usd": f"{actual_cost:.6f}",
        "x-polyportia-cost-predicted-usd": f"{estimate.total_usd:.6f}",
    }
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
