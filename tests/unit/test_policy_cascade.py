from __future__ import annotations

from polyportia.config.models import (
    ActualModel,
    DefinedModel,
    ProviderConfig,
    RetryPolicy,
)
from polyportia.config.policy import resolve_retry, resolve_timeout


def make_provider() -> ProviderConfig:
    return ProviderConfig(
        name="x",
        default_retry=RetryPolicy(max_retries=1),
        default_timeout_s=10.0,
    )


def make_actual() -> ActualModel:
    return ActualModel(id="x/m", provider="x")


def make_defined(target_id: str = "x/m") -> DefinedModel:
    from polyportia.config.models import ActualModelRef

    return DefinedModel(name="d", target=ActualModelRef(id=target_id))


def test_cascade_request_wins():
    req = RetryPolicy(max_retries=7)
    res = resolve_retry(
        request=req,
        defined=make_defined(),
        actual=make_actual(),
        provider=make_provider(),
    )
    assert res.policy.max_retries == 7
    assert res.source == "request"


def test_cascade_defined_wins_over_actual_provider():
    defined = make_defined()
    defined.retry = RetryPolicy(max_retries=5)
    actual = make_actual()
    actual.retry = RetryPolicy(max_retries=3)
    res = resolve_retry(request=None, defined=defined, actual=actual, provider=make_provider())
    assert res.policy.max_retries == 5
    assert res.source == "defined"


def test_cascade_actual_wins_over_provider():
    actual = make_actual()
    actual.retry = RetryPolicy(max_retries=4)
    res = resolve_retry(request=None, defined=make_defined(), actual=actual, provider=make_provider())
    assert res.policy.max_retries == 4
    assert res.source == "actual"


def test_cascade_provider_default():
    res = resolve_retry(
        request=None, defined=make_defined(), actual=make_actual(), provider=make_provider()
    )
    assert res.policy.max_retries == 1
    assert res.source == "provider"


def test_cascade_built_in_default():
    res = resolve_retry(request=None, defined=None, actual=None, provider=None)
    assert res.source == "default"


def test_timeout_cascade():
    actual = make_actual()
    actual.timeout_s = 22.0
    res = resolve_timeout(
        request=None, defined=make_defined(), actual=actual, provider=make_provider()
    )
    assert res.value == 22.0
    assert res.source == "actual"


def test_timeout_provider_default_skipped_for_streaming():
    # provider.default_timeout_s must not apply to streaming calls: asyncio.wait_for
    # on the initial acompletion() awaitable only guards until HTTP headers arrive,
    # so queued providers (e.g. Ollama OLLAMA_NUM_PARALLEL=1) exhaust the budget
    # in the queue rather than during generation, causing spurious TimeoutErrors.
    res = resolve_timeout(
        request=None,
        defined=make_defined(),
        actual=make_actual(),
        provider=make_provider(),
        for_streaming=True,
    )
    assert res.value is None
    assert res.source == "default"


def test_timeout_explicit_request_still_applies_for_streaming():
    # Explicit request-level timeouts must still be respected even for streaming.
    res = resolve_timeout(
        request=5.0,
        defined=make_defined(),
        actual=make_actual(),
        provider=make_provider(),
        for_streaming=True,
    )
    assert res.value == 5.0
    assert res.source == "request"


def test_timeout_model_level_still_applies_for_streaming():
    # Explicit actual-model-level timeouts must still be respected for streaming.
    actual = make_actual()
    actual.timeout_s = 120.0
    res = resolve_timeout(
        request=None,
        defined=make_defined(),
        actual=actual,
        provider=make_provider(),
        for_streaming=True,
    )
    assert res.value == 120.0
    assert res.source == "actual"
