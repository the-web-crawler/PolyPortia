"""Council strategy dispatcher.

In M1 only the model paths (ActualModel/DefinedModel) are implemented; the
council strategies raise NotImplementedError until M2/M3 fills them in.
"""

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
    if isinstance(strategy, (ParallelArray, Synthesize)):
        raise NotImplementedError(
            "parallel/synthesize council strategies arrive in M2"
        )
    if isinstance(strategy, Debate):
        raise NotImplementedError("debate council strategy arrives in M3")
    if isinstance(strategy, ProposeAndReview):
        raise NotImplementedError("propose_review council strategy arrives in M3")
    raise TypeError(f"unhandled strategy: {type(strategy).__name__}")
