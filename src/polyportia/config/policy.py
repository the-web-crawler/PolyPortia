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
) -> ResolvedTimeout:
    if request is not None:
        return ResolvedTimeout(request, "request")
    if defined is not None and defined.timeout_s is not None:
        return ResolvedTimeout(defined.timeout_s, "defined")
    if actual is not None and actual.timeout_s is not None:
        return ResolvedTimeout(actual.timeout_s, "actual")
    if provider is not None:
        return ResolvedTimeout(provider.default_timeout_s, "provider")
    return ResolvedTimeout(None, "default")
