"""Trace records, spans, and the context manager helper for nesting spans."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from polyportia.config.policy import PolicySource
from polyportia.utils.ids import new_id

SpanKind = Literal[
    "actual",
    "defined",
    "parallel_array",
    "synthesize",
    "debate",
    "debate_turn",
]
SpanStatus = Literal["ok", "error", "timeout", "skipped", "fellback"]


@dataclass
class RetryAttempt:
    attempt: int
    latency_ms: float
    error: str | None
    error_category: str | None
    sleep_before_next_s: float | None


@dataclass
class TraceSpan:
    span_id: str
    parent_span_id: str | None
    kind: SpanKind
    target_repr: str
    started_at: datetime
    ended_at: datetime | None = None
    latency_ms: float | None = None
    status: SpanStatus = "ok"
    error: str | None = None
    request_messages: list[dict[str, Any]] | None = None
    response_content: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    retry_attempts: list[RetryAttempt] = field(default_factory=list)
    fallback_chain: list[str] = field(default_factory=list)
    effective_retry_source: PolicySource | None = None
    effective_timeout_source: PolicySource | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind,
            "target_repr": self.target_repr,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
            "request_messages": self.request_messages,
            "response_content": self.response_content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "retry_attempts": [
                {
                    "attempt": a.attempt,
                    "latency_ms": a.latency_ms,
                    "error": a.error,
                    "error_category": a.error_category,
                    "sleep_before_next_s": a.sleep_before_next_s,
                }
                for a in self.retry_attempts
            ],
            "fallback_chain": self.fallback_chain,
            "effective_retry_source": self.effective_retry_source,
            "effective_timeout_source": self.effective_timeout_source,
        }


@dataclass
class TraceRecord:
    trace_id: str
    created_at: datetime
    request_summary: dict[str, Any]
    spans: list[TraceSpan] = field(default_factory=list)
    final_status: Literal["ok", "partial", "error"] = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "request_summary": self.request_summary,
            "spans": [s.to_dict() for s in self.spans],
            "final_status": self.final_status,
        }


class TraceBuilder:
    """Per-request state collected via nested `span()` context managers."""

    def __init__(self, request_summary: dict[str, Any]) -> None:
        self.record = TraceRecord(
            trace_id=new_id(),
            created_at=datetime.now(UTC),
            request_summary=request_summary,
        )
        self._stack: list[TraceSpan] = []

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    @property
    def current_span(self) -> TraceSpan | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, *, kind: SpanKind, target_repr: str) -> Iterator[TraceSpan]:
        parent = self._stack[-1] if self._stack else None
        span = TraceSpan(
            span_id=new_id(),
            parent_span_id=parent.span_id if parent else None,
            kind=kind,
            target_repr=target_repr,
            started_at=datetime.now(UTC),
        )
        self.record.spans.append(span)
        self._stack.append(span)
        start = time.monotonic()
        try:
            yield span
        except BaseException as e:
            span.status = "error"
            span.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            span.ended_at = datetime.now(UTC)
            span.latency_ms = (time.monotonic() - start) * 1000
            self._stack.pop()

    def finalize(self) -> TraceRecord:
        statuses = {s.status for s in self.record.spans}
        if statuses == {"ok"} or not statuses:
            self.record.final_status = "ok"
        elif "ok" in statuses:
            self.record.final_status = "partial"
        else:
            self.record.final_status = "error"
        return self.record
