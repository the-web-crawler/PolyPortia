"""Programmatic SDK for in-process use of the PolyPortia executor."""

from __future__ import annotations

import asyncio
from typing import Any

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


async def acomplete(
    *,
    model: str,
    messages: list[dict[str, Any]],
    retry: RetryPolicy | None = None,
    timeout_s: float | None = None,
    config: PolyPortiaConfig | None = None,
    registry: Registry | None = None,
    trace_store: TraceStore | None = None,
    **params: Any,
) -> tuple[ProviderResult, str]:
    """Run a completion through the executor.

    Returns ``(ProviderResult, trace_id)`` so callers can fetch the trace later.
    """
    if config is not None:
        registry = Registry(config)
    if registry is None:
        registry = get_default_registry()
    if trace_store is None:
        trace_store = get_default_store()

    target = resolve_model_input(model, registry)
    ctx = _build_context(
        registry=registry,
        request_summary={"model": model, "message_count": len(messages)},
        request_params=params,
        request_retry=retry,
        request_timeout_s=timeout_s,
    )
    try:
        result = await execute_target(target, messages, ctx)
    finally:
        trace_store.add(ctx.trace.finalize())
    return result, ctx.trace.trace_id


def complete(
    *,
    model: str,
    messages: list[dict[str, Any]],
    retry: RetryPolicy | None = None,
    timeout_s: float | None = None,
    config: PolyPortiaConfig | None = None,
    registry: Registry | None = None,
    trace_store: TraceStore | None = None,
    **params: Any,
) -> tuple[ProviderResult, str]:
    return asyncio.run(
        acomplete(
            model=model,
            messages=messages,
            retry=retry,
            timeout_s=timeout_s,
            config=config,
            registry=registry,
            trace_store=trace_store,
            **params,
        )
    )


async def run_council(
    spec_or_name: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> tuple[ProviderResult, str]:
    """Execute a named council. Same signature as ``acomplete`` modulo the name."""
    return await acomplete(model=spec_or_name, messages=messages, **kwargs)
