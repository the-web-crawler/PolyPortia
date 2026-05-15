"""Parallel-array and synthesize council strategies."""

from __future__ import annotations

import asyncio
from typing import Any

from polyportia.config.models import (
    ActualModelRef,
    CouncilRef,
    DefinedModelRef,
    ParallelArray,
    ResolvableTarget,
    Synthesize,
)
from polyportia.council.context import ExecutionContext
from polyportia.council.failure import (
    MemberOutcome,
    apply_failure_policy,
    outcomes_to_array,
)
from polyportia.council.synthesis import build_synth_messages
from polyportia.providers.base import ProviderResult


def _member_repr(target: ResolvableTarget) -> str:
    if isinstance(target, ActualModelRef):
        return f"actual:{target.id}"
    if isinstance(target, DefinedModelRef):
        return f"defined:{target.name}"
    if isinstance(target, CouncilRef):
        return f"council:{target.name}"
    return type(target).__name__


async def _run_member(
    member: ResolvableTarget,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
    *,
    timeout_s: float | None,
) -> MemberOutcome:
    from polyportia.council.executor import execute_target

    repr_str = _member_repr(member)
    try:
        coro = execute_target(member, messages, ctx.child())
        if timeout_s is not None:
            result = await asyncio.wait_for(coro, timeout=timeout_s)
        else:
            result = await coro
        return MemberOutcome(member_repr=repr_str, result=result, error=None)
    except BaseException as e:
        return MemberOutcome(member_repr=repr_str, result=None, error=e)


async def _fan_out(
    members: list[ResolvableTarget],
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
    *,
    timeout_s: float | None,
) -> list[MemberOutcome]:
    return await asyncio.gather(
        *(_run_member(m, messages, ctx, timeout_s=timeout_s) for m in members)
    )


def _array_envelope_result(
    outcomes: list[MemberOutcome],
    *,
    label: str,
) -> ProviderResult:
    """Wrap fan-out outcomes into a synthetic ProviderResult.

    The HTTP layer's array endpoint reads ``raw["responses"]`` to render the
    array envelope; OpenAI-shape callers see a stringified summary in
    ``content``.
    """
    array = outcomes_to_array(outcomes)
    successful = [o for o in outcomes if o.ok]
    summary_lines = [f"({label}: {len(successful)}/{len(outcomes)} members)"]
    for entry in array:
        if "content" in entry:
            summary_lines.append(f"- {entry['member']}: {entry['content']}")
        else:
            summary_lines.append(f"- {entry['member']}: ERROR {entry.get('error')}")
    return ProviderResult(
        model_id=label,
        content="\n".join(summary_lines),
        raw={"object": "polyportia.council", "responses": array},
        finish_reason="stop",
    )


async def run_parallel_array(
    spec: ParallelArray,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    with ctx.trace.span(kind="parallel_array", target_repr="parallel_array") as span:
        outcomes = await _fan_out(spec.members, messages, ctx, timeout_s=spec.timeout_s)
        apply_failure_policy(outcomes, ctx.registry.failure)
        if not any(o.ok for o in outcomes):
            span.status = "error"
        elif not all(o.ok for o in outcomes):
            span.status = "fellback"
        return _array_envelope_result(outcomes, label="parallel_array")


async def run_synthesize(
    spec: Synthesize,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    from polyportia.council.executor import execute_target

    with ctx.trace.span(kind="synthesize", target_repr="synthesize") as span:
        outcomes = await _fan_out(spec.members, messages, ctx, timeout_s=spec.timeout_s)
        apply_failure_policy(outcomes, ctx.registry.failure)
        if not all(o.ok for o in outcomes):
            span.status = "fellback"
        synth_messages = build_synth_messages(
            base_messages=messages,
            members=spec.members,
            outcomes=outcomes,
            template=spec.synthesizer_prompt,
            include_names=spec.include_member_names,
        )
        return await execute_target(spec.synthesizer, synth_messages, ctx.child())
