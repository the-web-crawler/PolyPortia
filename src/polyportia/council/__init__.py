"""Council strategy dispatcher."""

from __future__ import annotations

from typing import Any

from polyportia.config.models import (
    Debate,
    ParallelArray,
    ProposeAndReview,
    Synthesize,
)
from polyportia.council.context import ExecutionContext
from polyportia.providers.base import ProviderResult


async def strategy_dispatch(
    strategy: ParallelArray | Synthesize | Debate | ProposeAndReview,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    if isinstance(strategy, ParallelArray):
        from polyportia.council.parallel import run_parallel_array

        return await run_parallel_array(strategy, messages, ctx)
    if isinstance(strategy, Synthesize):
        from polyportia.council.parallel import run_synthesize

        return await run_synthesize(strategy, messages, ctx)
    if isinstance(strategy, Debate):
        from polyportia.council.debate import run_debate

        return await run_debate(strategy, messages, ctx)
    if isinstance(strategy, ProposeAndReview):
        from polyportia.council.propose_review import run_propose_review

        return await run_propose_review(strategy, messages, ctx)
    raise TypeError(f"unhandled strategy: {type(strategy).__name__}")
