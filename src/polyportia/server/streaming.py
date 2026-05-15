"""SSE adapter for streaming completions.

In v1 streaming is supported only for terminal single-model calls (and the
synthesizer step in M2). Per-member fan-out for councils is non-streamed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from polyportia.config.models import ActualModel, ActualModelRef, DefinedModel, ResolvableTarget
from polyportia.config.policy import resolve_retry, resolve_timeout
from polyportia.config.registry import Registry
from polyportia.config.resolver import resolve, resolve_for_model
from polyportia.providers.litellm_adapter import stream_completion


def _to_sse(chunk: Any) -> str:
    if hasattr(chunk, "model_dump_json"):
        body = chunk.model_dump_json()
    elif hasattr(chunk, "json"):
        try:
            body = chunk.json()
        except Exception:
            body = json.dumps(_jsonable(chunk))
    else:
        body = json.dumps(_jsonable(chunk))
    return f"data: {body}\n\n"


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _resolve_to_actual(
    target: ResolvableTarget, registry: Registry
) -> tuple[ActualModel, DefinedModel | None]:
    """For streaming we only support terminal-actual-model targets in v1."""
    resolved = resolve(target, registry)
    if isinstance(resolved, ActualModel):
        return resolved, None
    if isinstance(resolved, DefinedModel):
        # Walk the defined target only — fallbacks are not exercised on stream
        # in v1 (streaming + multi-attempt retries don't compose well).
        sub = resolve_for_model(resolved.target, registry)
        if isinstance(sub, ActualModel):
            return sub, resolved
        if isinstance(sub, DefinedModel):
            sub_target = sub.target
            if not isinstance(sub_target, ActualModelRef):
                raise ValueError(
                    "streaming only supports defined-model chains terminating in an actual model"
                )
            actual, _ = _resolve_to_actual(sub_target, registry)
            return actual, resolved
    raise ValueError("streaming only supported on single-model targets in v1")


async def stream_single_model(
    *,
    target: ResolvableTarget,
    messages: list[dict[str, Any]],
    registry: Registry,
    request_params: dict[str, Any],
    request_timeout_s: float | None,
) -> AsyncIterator[str]:
    actual, defined = _resolve_to_actual(target, registry)
    provider = registry.get_provider(actual.provider)
    timeout = resolve_timeout(
        request=request_timeout_s, defined=defined, actual=actual, provider=provider
    )
    merged_params: dict[str, Any] = {**request_params}
    if defined is not None:
        merged_params = {**defined.params, **merged_params}
    # Validate retry policy can be resolved (not used here, but errors early)
    resolve_retry(request=None, defined=defined, actual=actual, provider=provider)
    async for chunk in stream_completion(
        actual=actual,
        provider=provider,
        messages=messages,
        params=merged_params,
        timeout_s=timeout.value,
    ):
        yield _to_sse(chunk)
    yield "data: [DONE]\n\n"
