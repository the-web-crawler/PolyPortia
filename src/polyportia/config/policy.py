"""Cascade resolution for retry policy and timeout.

Precedence (highest → lowest):
    request override → DefinedModel → ActualModel → Provider → built-in default
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polyportia.config.models import ActualModel, DefinedModel, ProviderConfig, RetryPolicy

PolicySource = Literal["request", "defined", "actual", "provider", "default"]


@dataclass(frozen=True)
class ResolvedRetry:
    policy: RetryPolicy
    source: PolicySource


@dataclass(frozen=True)
class ResolvedTimeout:
    value: float | None
    source: PolicySource


_DEFAULT_RETRY = RetryPolicy()


def resolve_retry(
    *,
    request: RetryPolicy | None,
    defined: DefinedModel | None,
    actual: ActualModel | None,
    provider: ProviderConfig | None,
) -> ResolvedRetry:
    if request is not None:
        return ResolvedRetry(request, "request")
    if defined is not None and defined.retry is not None:
        return ResolvedRetry(defined.retry, "defined")
    if actual is not None and actual.retry is not None:
        return ResolvedRetry(actual.retry, "actual")
    if provider is not None:
        return ResolvedRetry(provider.default_retry, "provider")
    return ResolvedRetry(_DEFAULT_RETRY, "default")


def resolve_timeout(
    *,
    request: float | None,
    defined: DefinedModel | None,
    actual: ActualModel | None,
    provider: ProviderConfig | None,
    for_streaming: bool = False,
) -> ResolvedTimeout:
    if request is not None:
        return ResolvedTimeout(request, "request")
    if defined is not None and defined.timeout_s is not None:
        return ResolvedTimeout(defined.timeout_s, "defined")
    if actual is not None and actual.timeout_s is not None:
        return ResolvedTimeout(actual.timeout_s, "actual")
    # For streaming calls, skip the provider default: asyncio.wait_for() on the
    # initial acompletion() awaitable only guards until the provider sends HTTP
    # headers, not until generation is complete. When a provider like Ollama
    # queues requests (OLLAMA_NUM_PARALLEL=1), concurrent callers exhaust this
    # budget waiting in the queue rather than generating, causing premature
    # TimeoutError. Explicit request/model-level timeouts are still respected.
    if provider is not None and not for_streaming:
        return ResolvedTimeout(provider.default_timeout_s, "provider")
    return ResolvedTimeout(None, "default")
