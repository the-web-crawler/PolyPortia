"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from polyportia.config.loader import load_config_from_string
from polyportia.config.models import PolyPortiaConfig
from polyportia.config.registry import Registry


_TEST_YAML = """
providers:
  - name: anthropic
    api_key: test-anthropic-key
  - name: openai
    api_key: test-openai-key

actual_models:
  - id: anthropic/claude-opus-4-7
    provider: anthropic
    input_cost_per_1m_tokens: 15.00
    output_cost_per_1m_tokens: 75.00
  - id: anthropic/claude-haiku-4-5
    provider: anthropic
  - id: openai/gpt-5-5
    provider: openai

defined_models:
  - name: fast
    target: {kind: actual, id: anthropic/claude-haiku-4-5}
    fallbacks:
      - {kind: actual, id: openai/gpt-5-5}
  - name: thinking
    target: {kind: actual, id: anthropic/claude-opus-4-7}
    fallbacks:
      - {kind: actual, id: openai/gpt-5-5}
      - {kind: defined, name: fast}
  - name: creative
    target: {kind: actual, id: openai/gpt-5-5}

councils:
  - name: triad-raw
    strategy:
      kind: parallel_array
      members:
        - {kind: defined, name: fast}
        - {kind: defined, name: thinking}
        - {kind: defined, name: creative}
  - name: triad
    strategy:
      kind: synthesize
      members:
        - {kind: defined, name: fast}
        - {kind: defined, name: thinking}
        - {kind: defined, name: creative}
      synthesizer: {kind: defined, name: thinking}
  - name: meta-council
    strategy:
      kind: synthesize
      members:
        - {kind: council, name: triad}
        - {kind: defined, name: creative}
      synthesizer: {kind: defined, name: thinking}
"""


@dataclass
class MockChoice:
    message: Any
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class MockMessage:
    content: str
    role: str = "assistant"


@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockResponse:
    choices: list[MockChoice]
    usage: MockUsage = field(default_factory=MockUsage)
    id: str = "chatcmpl-mock"
    object: str = "chat.completion"
    model: str = "mock"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": c.message.role, "content": c.message.content},
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


def make_mock_response(content: str = "ok") -> MockResponse:
    return MockResponse(choices=[MockChoice(message=MockMessage(content=content))])


@pytest.fixture
def test_config() -> PolyPortiaConfig:
    return load_config_from_string(_TEST_YAML)


@pytest.fixture
def test_registry(test_config: PolyPortiaConfig) -> Registry:
    return Registry(test_config)


class MockProvider:
    """Pluggable mock for litellm.acompletion behaviour.

    ``handlers`` maps model id to a callable returning either a MockResponse,
    raising an exception, or returning an awaitable resolving to those.
    """

    def __init__(self) -> None:
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.calls: list[dict[str, Any]] = []

    def set_response(self, model_id: str, content: str = "ok") -> None:
        self.handlers[model_id] = lambda kw: make_mock_response(content)

    def set_error(self, model_id: str, exc_factory: Callable[[], BaseException]) -> None:
        def raise_(kw: dict) -> None:
            raise exc_factory()

        self.handlers[model_id] = raise_

    def set_sequence(self, model_id: str, items: list[Any]) -> None:
        idx = {"i": 0}

        def next_item(kw: dict) -> Any:
            i = idx["i"]
            idx["i"] += 1
            item = items[min(i, len(items) - 1)]
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                return item()
            return item

        self.handlers[model_id] = next_item

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        model = kwargs.get("model", "")
        if model not in self.handlers:
            return make_mock_response(f"default-for-{model}")
        result = self.handlers[model](kwargs)
        if hasattr(result, "__await__"):
            return await result
        return result


@pytest.fixture
def mock_provider(monkeypatch: pytest.MonkeyPatch) -> MockProvider:
    mock = MockProvider()

    async def patched(**kwargs: Any) -> Any:
        return await mock(**kwargs)

    monkeypatch.setattr("polyportia.providers.litellm_adapter.acompletion", patched)
    return mock


class _MockStreamChunk:
    def __init__(self, content: str) -> None:
        self.content = content

    def model_dump(self) -> dict[str, Any]:
        return {
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": self.content}}],
        }


@pytest.fixture
def mock_stream_provider(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Patch acompletion so stream=True returns an async iterator of chunks."""
    calls: dict[str, list[str]] = {}

    async def patched(**kwargs: Any) -> Any:
        if not kwargs.get("stream"):
            return make_mock_response("not-streamed")
        chunks = ["hello", " ", "world"]
        calls.setdefault(kwargs["model"], []).extend(chunks)

        async def gen() -> AsyncIterator[_MockStreamChunk]:
            for c in chunks:
                yield _MockStreamChunk(c)

        return gen()

    monkeypatch.setattr("polyportia.providers.litellm_adapter.acompletion", patched)
    return calls
