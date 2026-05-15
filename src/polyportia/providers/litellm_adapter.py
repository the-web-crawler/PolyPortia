"""The single seam between PolyPortia and LiteLLM.

Tests monkey-patch ``acompletion`` here to inject canned responses or errors.
Nothing else in PolyPortia is allowed to import ``litellm.acompletion`` directly.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from polyportia.config.models import ActualModel, ProviderConfig, RetryPolicy
from polyportia.providers.base import ProviderResult, TokenUsage
from polyportia.providers.errors import RetryableExhaustedError, classify

if TYPE_CHECKING:
    pass


async def acompletion(**kwargs: Any) -> Any:
    """Thin wrapper around ``litellm.acompletion`` for monkeypatching."""
    import litellm

    return await litellm.acompletion(**kwargs)


def _backoff_seconds(attempt: int, policy: RetryPolicy) -> float:
    if policy.backoff == "linear":
        base = policy.backoff_base_s * (attempt + 1)
    else:
        base = policy.backoff_base_s * (2**attempt)
    base = min(base, policy.backoff_max_s)
    if policy.jitter:
        base = base * (0.5 + random.random() * 0.5)
    return base


def _build_kwargs(
    actual: ActualModel,
    provider: ProviderConfig,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    stream: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": actual.id,
        "messages": messages,
        **provider.default_params,
        **actual.default_params,
        **params,
    }
    if stream:
        kwargs["stream"] = True
    if provider.api_key is not None:
        kwargs["api_key"] = provider.api_key.get_secret_value()
    if provider.api_base is not None:
        kwargs["api_base"] = provider.api_base
    if provider.extra_headers:
        kwargs.setdefault("extra_headers", {}).update(provider.extra_headers)
    return kwargs


def _extract_content(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return ""
        choice = choices[0]
        msg = getattr(choice, "message", None)
        if msg is None and isinstance(choice, dict):
            msg = choice.get("message")
        if msg is None:
            return ""
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return content or ""
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _extract_finish(response: Any) -> str | None:
    try:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None
        first = choices[0]
        value = getattr(first, "finish_reason", None)
        if value is None and isinstance(first, dict):
            value = first.get("finish_reason")
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return value if value is None or isinstance(value, str) else str(value)


def _extract_usage(response: Any) -> TokenUsage | None:
    raw = getattr(response, "usage", None)
    if raw is None and isinstance(response, dict):
        raw = response.get("usage")
    return TokenUsage.from_litellm(raw)


async def call_with_retries(
    *,
    actual: ActualModel,
    provider: ProviderConfig,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    retry: RetryPolicy,
    timeout_s: float | None,
    on_attempt: Any = None,
) -> ProviderResult:
    """Invoke the provider with the configured retry + timeout policy.

    ``on_attempt`` is an optional callback ``(RetryAttempt) -> None`` invoked
    after every attempt, used by the executor to record trace data.
    """
    from polyportia.providers.errors import ErrorCategory

    kwargs = _build_kwargs(actual, provider, messages, params, stream=False)
    last_exc: BaseException | None = None
    last_category: ErrorCategory = "unknown"
    attempts_total = retry.max_retries + 1
    for attempt in range(attempts_total):
        start = time.monotonic()
        try:
            coro = acompletion(**kwargs)
            if timeout_s is not None:
                response = await asyncio.wait_for(coro, timeout=timeout_s)
            else:
                response = await coro
            latency_ms = (time.monotonic() - start) * 1000
            if on_attempt is not None:
                from polyportia.observability.trace import RetryAttempt

                on_attempt(
                    RetryAttempt(
                        attempt=attempt,
                        latency_ms=latency_ms,
                        error=None,
                        error_category=None,
                        sleep_before_next_s=None,
                    )
                )
            return ProviderResult(
                model_id=actual.id,
                content=_extract_content(response),
                usage=_extract_usage(response),
                raw=response,
                finish_reason=_extract_finish(response),
                latency_ms=latency_ms,
            )
        except BaseException as e:
            latency_ms = (time.monotonic() - start) * 1000
            cat = classify(e)
            last_exc = e
            last_category = cat
            retryable = cat in retry.retry_on
            is_last = attempt == attempts_total - 1
            sleep_for = _backoff_seconds(attempt, retry) if retryable and not is_last else None
            if on_attempt is not None:
                from polyportia.observability.trace import RetryAttempt

                on_attempt(
                    RetryAttempt(
                        attempt=attempt,
                        latency_ms=latency_ms,
                        error=str(e),
                        error_category=cat,
                        sleep_before_next_s=sleep_for,
                    )
                )
            if not retryable or is_last:
                raise RetryableExhaustedError(
                    f"{type(e).__name__}: {e}", last_category=cat
                ) from e
            await asyncio.sleep(sleep_for or 0)
    assert last_exc is not None  # unreachable: loop always raises or returns
    raise RetryableExhaustedError(
        f"{type(last_exc).__name__}: {last_exc}", last_category=last_category
    ) from last_exc


async def stream_completion(
    *,
    actual: ActualModel,
    provider: ProviderConfig,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    timeout_s: float | None,
) -> AsyncIterator[Any]:
    """Yield raw streaming chunks from LiteLLM. No retry on stream calls in v1."""
    kwargs = _build_kwargs(actual, provider, messages, params, stream=True)
    coro = acompletion(**kwargs)
    if timeout_s is not None:
        stream = await asyncio.wait_for(coro, timeout=timeout_s)
    else:
        stream = await coro
    async for chunk in stream:
        yield chunk
