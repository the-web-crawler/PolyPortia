"""Programmatic SDK for in-process use of the PolyPortia executor."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from polyportia.budget.enforcer import BudgetEnforcer
from polyportia.budget.errors import BudgetExceededError, CostEstimate
from polyportia.budget.estimator import estimate_cost
from polyportia.config.models import (
    ActualModelRef,
    CouncilRef,
    DefinedModelRef,
    PolyPortiaConfig,
    ResolvableTarget,
    RetryPolicy,
)
from polyportia.config.registry import Registry, get_default_registry
from polyportia.council.context import ExecutionContext
from polyportia.council.executor import execute_target
from polyportia.observability.store import TraceStore, get_default_store
from polyportia.observability.trace import TraceBuilder
from polyportia.providers.base import ProviderResult


def resolve_model_input(value: str, registry: Registry) -> ResolvableTarget:
    """Map an incoming ``model`` string to a ResolvableTarget.

    Resolution order:
        1. literal ``provider/model`` form (contains ``/`` and matches a known
           ActualModel.id), returned as ``ActualModelRef``.
        2. DefinedModel name.
        3. Council name.
        4. Otherwise treat as a literal ActualModel.id.
    """
    if "/" in value and registry.has_actual_model(value):
        return ActualModelRef(id=value)
    if registry.has_defined_model(value):
        return DefinedModelRef(name=value)
    if registry.has_council(value):
        return CouncilRef(name=value)
    if registry.has_actual_model(value):
        return ActualModelRef(id=value)
    raise KeyError(
        f"model '{value}' is not a registered defined_model, council, or actual_model"
    )


def _build_context(
    *,
    registry: Registry,
    request_summary: dict[str, Any],
    request_params: dict[str, Any],
    request_retry: RetryPolicy | None,
    request_timeout_s: float | None,
) -> ExecutionContext:
    trace = TraceBuilder(request_summary)
    return ExecutionContext(
        registry=registry,
        trace=trace,
        request_params=request_params,
        request_retry=request_retry,
        request_timeout_s=request_timeout_s,
    )


def _resolve_budget(
    budget: float | Literal["unlimited"] | None, registry: Registry
) -> float | None:
    if budget == "unlimited":
        return None
    if isinstance(budget, (int, float)):
        return float(budget)
    return registry.budget_usd_default


async def acomplete(
    *,
    model: str,
    messages: list[dict[str, Any]],
    retry: RetryPolicy | None = None,
    timeout_s: float | None = None,
    budget_usd: float | Literal["unlimited"] | None = None,
    config: PolyPortiaConfig | None = None,
    registry: Registry | None = None,
    trace_store: TraceStore | None = None,
    return_cost: bool = False,
    **params: Any,
) -> tuple[ProviderResult, str] | tuple[ProviderResult, str, CostEstimate, float]:
    """Run a completion through the executor.

    Returns ``(ProviderResult, trace_id)`` by default. With ``return_cost=True``
    returns ``(ProviderResult, trace_id, predicted_estimate, actual_cost_usd)``.

    Raises ``BudgetExceededError`` on pre-flight refusal or mid-execution stop.
    """
    if config is not None:
        registry = Registry(config)
    if registry is None:
        registry = get_default_registry()
    if trace_store is None:
        trace_store = get_default_store()

    target = resolve_model_input(model, registry)
    estimate = estimate_cost(target, messages, params, registry)
    budget_value = _resolve_budget(budget_usd, registry)
    if budget_value is not None and estimate.total_usd > budget_value:
        raise BudgetExceededError(
            f"Predicted ${estimate.total_usd:.6f} exceeds budget ${budget_value:.6f}",
            stage="pre_flight",
            budget_usd=budget_value,
            predicted_usd=estimate.total_usd,
            breakdown=estimate.breakdown,
        )

    enforcer = BudgetEnforcer(budget_usd=budget_value)
    ctx = ExecutionContext(
        registry=registry,
        trace=TraceBuilder({"model": model, "message_count": len(messages)}),
        request_params=params,
        request_retry=retry,
        request_timeout_s=timeout_s,
        budget=enforcer,
    )
    try:
        result = await execute_target(target, messages, ctx)
    finally:
        trace_store.add(ctx.trace.finalize())
    if return_cost:
        actual = sum(s.cost_usd or 0.0 for s in ctx.trace.record.spans)
        return result, ctx.trace.trace_id, estimate, actual
    return result, ctx.trace.trace_id


def complete(
    *,
    model: str,
    messages: list[dict[str, Any]],
    retry: RetryPolicy | None = None,
    timeout_s: float | None = None,
    budget_usd: float | Literal["unlimited"] | None = None,
    config: PolyPortiaConfig | None = None,
    registry: Registry | None = None,
    trace_store: TraceStore | None = None,
    return_cost: bool = False,
    **params: Any,
) -> tuple[ProviderResult, str] | tuple[ProviderResult, str, CostEstimate, float]:
    return asyncio.run(
        acomplete(
            model=model,
            messages=messages,
            retry=retry,
            timeout_s=timeout_s,
            budget_usd=budget_usd,
            config=config,
            registry=registry,
            trace_store=trace_store,
            return_cost=return_cost,
            **params,
        )
    )


async def run_council(
    spec_or_name: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> tuple[ProviderResult, str] | tuple[ProviderResult, str, CostEstimate, float]:
    """Execute a named council. Same signature as ``acomplete`` modulo the name."""
    return await acomplete(model=spec_or_name, messages=messages, **kwargs)
