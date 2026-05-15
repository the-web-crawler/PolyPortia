from __future__ import annotations

import asyncio

import pytest

from polyportia.config.models import RetryPolicy
from polyportia.providers.errors import RetryableExhaustedError, classify
from polyportia.providers.litellm_adapter import _backoff_seconds


class FakeRateLimit(Exception):
    pass


FakeRateLimit.__name__ = "RateLimitError"


class FakeBadRequest(Exception):
    status_code = 400


def test_classify_known_categories():
    assert classify(asyncio.TimeoutError()) == "timeout"
    assert classify(FakeRateLimit()) == "rate_limit"
    assert classify(FakeBadRequest()) == "bad_request"


def test_backoff_exponential_grows():
    p = RetryPolicy(max_retries=5, backoff="exponential", backoff_base_s=1.0, jitter=False)
    assert _backoff_seconds(0, p) == 1.0
    assert _backoff_seconds(1, p) == 2.0
    assert _backoff_seconds(2, p) == 4.0


def test_backoff_capped():
    p = RetryPolicy(
        max_retries=10,
        backoff="exponential",
        backoff_base_s=1.0,
        backoff_max_s=3.0,
        jitter=False,
    )
    assert _backoff_seconds(10, p) == 3.0


def test_call_with_retries_succeeds_after_failure(mock_provider, test_registry):
    from polyportia.providers.litellm_adapter import call_with_retries

    actual = test_registry.get_actual_model("anthropic/claude-opus-4-7")
    provider = test_registry.get_provider(actual.provider)

    # First attempt fails (timeout), second succeeds.
    mock_provider.set_sequence(
        actual.id, [asyncio.TimeoutError(), object()]
    )

    # second call should return mock response from default handler — we'll
    # rebuild the sequence to use a real mock response on success.
    from tests.conftest import make_mock_response
    mock_provider.set_sequence(
        actual.id, [asyncio.TimeoutError(), make_mock_response("after-retry")]
    )

    result = asyncio.run(
        call_with_retries(
            actual=actual,
            provider=provider,
            messages=[{"role": "user", "content": "hi"}],
            params={},
            retry=RetryPolicy(max_retries=2, backoff_base_s=0, jitter=False),
            timeout_s=None,
        )
    )
    assert result.content == "after-retry"


def test_call_with_retries_exhausts_on_non_retryable(mock_provider, test_registry):
    from polyportia.providers.litellm_adapter import call_with_retries

    actual = test_registry.get_actual_model("anthropic/claude-opus-4-7")
    provider = test_registry.get_provider(actual.provider)
    mock_provider.set_error(actual.id, FakeBadRequest)

    with pytest.raises(RetryableExhaustedError):
        asyncio.run(
            call_with_retries(
                actual=actual,
                provider=provider,
                messages=[{"role": "user", "content": "hi"}],
                params={},
                retry=RetryPolicy(max_retries=5, backoff_base_s=0, jitter=False),
                timeout_s=None,
            )
        )
    # Bad request is not in retry_on by default → just one attempt
    assert len(mock_provider.calls) == 1
