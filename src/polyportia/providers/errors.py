"""Error taxonomy and classification helpers for provider calls."""

from __future__ import annotations

from typing import Literal

ErrorCategory = Literal[
    "timeout",
    "rate_limit",
    "server_error",
    "connection",
    "auth",
    "bad_request",
    "unknown",
]


class ProviderError(Exception):
    """Base class for normalised provider errors."""

    def __init__(self, message: str, *, category: ErrorCategory = "unknown") -> None:
        super().__init__(message)
        self.category: ErrorCategory = category


class RetryableExhaustedError(ProviderError):
    """All retry attempts have been used."""

    def __init__(self, message: str, *, last_category: ErrorCategory) -> None:
        super().__init__(message, category=last_category)


def classify(err: BaseException) -> ErrorCategory:
    """Map a raised exception to one of the configured retry_on categories.

    We probe litellm's exception class names by string because importing the
    full exception tree from litellm.exceptions creates a heavy import chain
    that we want to lazy-evaluate in the adapter, not here.
    """
    import asyncio

    if isinstance(err, asyncio.TimeoutError):
        return "timeout"
    name = type(err).__name__
    if name == "Timeout" or "Timeout" in name:
        return "timeout"
    if "RateLimit" in name or name == "RateLimitError":
        return "rate_limit"
    if name in {"APIConnectionError", "ConnectError", "ConnectionError"}:
        return "connection"
    if name in {"ServiceUnavailableError", "InternalServerError", "APIError"}:
        return "server_error"
    if name in {"AuthenticationError", "PermissionError", "PermissionDeniedError"}:
        return "auth"
    if name in {"BadRequestError", "InvalidRequestError", "ContentPolicyViolationError"}:
        return "bad_request"
    status = getattr(err, "status_code", None) or getattr(err, "http_status", None)
    if isinstance(status, int):
        if status == 408 or status == 504:
            return "timeout"
        if status == 429:
            return "rate_limit"
        if 500 <= status < 600:
            return "server_error"
        if status in (401, 403):
            return "auth"
        if 400 <= status < 500:
            return "bad_request"
    return "unknown"
