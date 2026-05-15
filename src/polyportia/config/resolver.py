"""Reference resolution for ResolvableTargets.

ActualModelRef → ActualModel, DefinedModelRef → DefinedModel, CouncilRef →
CouncilSpec.strategy, and inline strategies pass through unchanged. The
resolver does not itself walk DefinedModel.fallbacks — that is the executor's
responsibility (see council/executor.py).
"""

from __future__ import annotations

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
from polyportia.config.registry import Registry

ResolvedTarget = (
    ActualModel | DefinedModel | ParallelArray | Synthesize | Debate | ProposeAndReview
)


class CyclicReferenceError(RuntimeError):
    pass


class UnknownTargetTypeError(TypeError):
    pass


def resolve(target: ResolvableTarget, registry: Registry) -> ResolvedTarget:
    """Resolve a single reference layer.

    DefinedModelRef returns the DefinedModel (does NOT chase its .target — the
    executor does that, so it can apply per-defined params/retry/timeout at the
    right layer).
    """
    if isinstance(target, ActualModelRef):
        return registry.get_actual_model(target.id)
    if isinstance(target, DefinedModelRef):
        return registry.get_defined_model(target.name)
    if isinstance(target, CouncilRef):
        return registry.get_council(target.name).strategy
    if isinstance(target, (ParallelArray, Synthesize, Debate, ProposeAndReview)):
        return target
    raise UnknownTargetTypeError(f"cannot resolve target of type {type(target).__name__}")


def resolve_for_model(target: ResolvableTarget, registry: Registry) -> ActualModel | DefinedModel:
    """Resolve a target expected to be a single model (not a council).

    Used inside DefinedModel.fallbacks walking.
    """
    resolved = resolve(target, registry)
    if isinstance(resolved, (ActualModel, DefinedModel)):
        return resolved
    raise TypeError(
        f"expected actual/defined model in this position, got {type(resolved).__name__}"
    )
