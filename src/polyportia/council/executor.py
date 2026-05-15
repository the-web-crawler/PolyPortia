"""Recursive executor for ResolvableTargets.

Single entry point ``execute_target`` dispatches by target kind. The single-
model + defined-model paths (with transitive fallback walking) are implemented
here. Council strategies (parallel/synthesize/debate/propose_review) are
implemented in their own modules and dispatched from ``execute_target``.
"""

from __future__ import annotations

from typing import Any

from polyportia.config.models import (
    ActualModel,
    ActualModelRef,
    CouncilRef,
    Debate,
    DefinedModel,
    DefinedModelRef,
    ParallelArray,
    ProposeAndReview,
    ResolvableTarget,
    Synthesize,
)
from polyportia.config.policy import resolve_retry, resolve_timeout
from polyportia.config.registry import Registry
from polyportia.config.resolver import resolve, resolve_for_model
from polyportia.council.context import ExecutionContext, RecursionDepthExceeded
from polyportia.observability.cost import estimate_cost_usd
from polyportia.providers.base import ProviderResult
from polyportia.providers.errors import RetryableExhaustedError
from polyportia.providers.litellm_adapter import call_with_retries


class CyclicDefinedModelError(RuntimeError):
    pass


class FallbacksExhaustedError(RuntimeError):
    def __init__(self, message: str, *, chain: list[str]) -> None:
        super().__init__(message)
        self.chain = chain


def _target_repr(target: object) -> str:
    if isinstance(target, ActualModelRef):
        return f"actual:{target.id}"
    if isinstance(target, DefinedModelRef):
        return f"defined:{target.name}"
    return type(target).__name__


async def execute_target(
    target: ResolvableTarget,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    if ctx.depth > ctx.max_depth:
        raise RecursionDepthExceeded(f"depth {ctx.depth} > max {ctx.max_depth}")
    resolved = resolve(target, ctx.registry)

    if isinstance(resolved, ActualModel):
        return await _call_actual(resolved, messages, ctx, defined=None)

    if isinstance(resolved, DefinedModel):
        return await _call_defined(resolved, messages, ctx)

    if isinstance(resolved, (ParallelArray, Synthesize, Debate, ProposeAndReview)):
        from polyportia.council import strategy_dispatch

        return await strategy_dispatch(resolved, messages, ctx)

    raise TypeError(f"unhandled resolved target: {type(resolved).__name__}")


async def _call_actual(
    actual: ActualModel,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
    *,
    defined: DefinedModel | None,
) -> ProviderResult:
    provider = ctx.registry.get_provider(actual.provider)
    retry = resolve_retry(
        request=ctx.request_retry, defined=defined, actual=actual, provider=provider
    )
    timeout = resolve_timeout(
        request=ctx.request_timeout_s, defined=defined, actual=actual, provider=provider
    )
    merged_params: dict[str, Any] = {**ctx.request_params}
    if defined is not None:
        merged_params = {**defined.params, **merged_params}

    if ctx.budget is not None and ctx.budget.enabled:
        ctx.budget.check_or_raise()

    with ctx.trace.span(kind="actual", target_repr=actual.id) as span:
        span.effective_retry_source = retry.source
        span.effective_timeout_source = timeout.source
        span.request_messages = messages
        try:
            result = await call_with_retries(
                actual=actual,
                provider=provider,
                messages=messages,
                params=merged_params,
                retry=retry.policy,
                timeout_s=timeout.value,
                on_attempt=lambda att: span.retry_attempts.append(att),
            )
        except RetryableExhaustedError as e:
            span.status = "error"
            span.error = str(e)
            raise
        span.response_content = result.content
        if result.usage is not None:
            span.prompt_tokens = result.usage.prompt_tokens
            span.completion_tokens = result.usage.completion_tokens
        span.cost_usd = estimate_cost_usd(actual, result.usage)
        if ctx.budget is not None and ctx.budget.enabled:
            ctx.budget.record_spent(span.cost_usd)
            ctx.budget.check_or_raise()
        return result


async def _call_defined(
    defined: DefinedModel,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    if defined.name in ctx.visited_defined:
        raise CyclicDefinedModelError(
            f"defined model '{defined.name}' is reached cyclically via fallback chain"
        )
    ctx.visited_defined.add(defined.name)
    try:
        chain = [defined.target, *defined.fallbacks]
        chain_repr: list[str] = []
        with ctx.trace.span(kind="defined", target_repr=f"defined:{defined.name}") as defined_span:
            defined_span.fallback_chain = chain_repr
            last_error: BaseException | None = None
            for entry in chain:
                chain_repr.append(_target_repr(entry))
                try:
                    return await _execute_chain_entry(entry, defined, messages, ctx)
                except (
                    RetryableExhaustedError,
                    FallbacksExhaustedError,
                    CyclicDefinedModelError,
                ) as e:
                    last_error = e
                    continue
            defined_span.status = "error"
            defined_span.error = str(last_error) if last_error else "fallbacks exhausted"
            raise FallbacksExhaustedError(
                f"all fallbacks exhausted for defined model '{defined.name}'",
                chain=chain_repr,
            )
    finally:
        ctx.visited_defined.discard(defined.name)


async def _execute_chain_entry(
    entry: object,
    defined: DefinedModel,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    """Execute one step of a defined-model fallback chain.

    If ``entry`` resolves to an ActualModel, call it with the parent
    DefinedModel's effective policy. If it resolves to another DefinedModel,
    delegate to ``_call_defined`` so that the nested chain is followed too.
    """
    from polyportia.config.models import ActualModelRef as _AMR
    from polyportia.config.models import DefinedModelRef as _DMR

    if not isinstance(entry, (_AMR, _DMR)):
        raise TypeError(f"fallback entry must be a ModelTarget, got {type(entry).__name__}")
    resolved = resolve_for_model(entry, ctx.registry)
    child = ctx.child()
    if isinstance(resolved, ActualModel):
        return await _call_actual(resolved, messages, child, defined=defined)
    if isinstance(resolved, DefinedModel):
        return await _call_defined(resolved, messages, child)
    raise TypeError(f"chain entry resolved to unexpected type: {type(resolved).__name__}")


def resolve_request_model(model: str, registry: Registry) -> ResolvableTarget:
    """Convert an incoming ``model`` string to a ResolvableTarget.

    Precedence: DefinedModel name → Council name → registered ActualModel id →
    literal ``provider/model`` id (passed through verbatim to LiteLLM, must be
    a known LiteLLM model).
    """
    if registry.has_defined_model(model):
        return DefinedModelRef(name=model)
    if registry.has_council(model):
        return CouncilRef(name=model)
    if registry.has_actual_model(model):
        return ActualModelRef(id=model)
    if "/" in model:
        return ActualModelRef(id=model)
    raise ValueError(
        f"unknown model or alias: '{model}'. Define it as a DefinedModel, "
        "Council, or ActualModel, or pass a literal 'provider/model' identifier."
    )
