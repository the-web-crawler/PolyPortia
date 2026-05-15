"""POST /v1/councils/{name}/run — always returns the array envelope."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from polyportia.budget.enforcer import BudgetEnforcer
from polyportia.budget.errors import BudgetExceededError
from polyportia.budget.estimator import estimate_cost
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


def _resolve_budget(request_value: Any, registry: Registry) -> float | None:
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


@router.post("/v1/councils/{name}/run")
async def run_council(name: str, req: CouncilRunRequest, request: Request) -> Response:
    reg = _registry(request)
    store = _store(request)
    if not reg.has_council(name):
        raise HTTPException(status_code=404, detail=f"council '{name}' not found")

    overrides = req.polyportia
    request_params: dict[str, Any] = {}

    target = CouncilRef(name=name)
    messages = messages_to_dicts(req.messages)
    budget_value = _resolve_budget(
        overrides.budget_usd if overrides else None, reg
    )
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
    trace = TraceBuilder({"council": name, "message_count": len(req.messages)})
    ctx = ExecutionContext(
        registry=reg,
        trace=trace,
        request_params=request_params,
        request_retry=overrides.retry if overrides else None,
        request_timeout_s=overrides.timeout_s if overrides else None,
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

    actual_cost = _sum_trace_cost(trace)
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
    body["cost_usd"] = actual_cost
    body["cost_predicted_usd"] = estimate.total_usd
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
