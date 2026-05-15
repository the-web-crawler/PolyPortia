"""HTTP request/response Pydantic models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from polyportia.config.models import RetryPolicy


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class PolyPortiaOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retry: RetryPolicy | None = None
    timeout_s: float | None = None
    on_failure: Literal["continue", "fail", "retry"] | None = None
    response_format: Literal["openai", "array"] | None = None
    budget_usd: float | Literal["unlimited"] | None = None
    include_cost: bool = False


class ChatMessage(_Open):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionsRequest(_Open):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    polyportia: PolyPortiaOverrides | None = None


class CouncilRunRequest(_Open):
    messages: list[ChatMessage]
    response_format: Literal["openai", "array"] | None = None
    polyportia: PolyPortiaOverrides | None = None


class ModelListEntry(BaseModel):
    id: str
    object: Literal["model"] = "model"
    polyportia_kind: Literal["actual", "defined", "council"]


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelListEntry] = Field(default_factory=list)


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [m.model_dump(exclude_none=True) for m in messages]
