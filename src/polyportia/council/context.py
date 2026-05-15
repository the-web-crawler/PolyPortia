"""Per-request execution state shared across the recursive executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from polyportia.config.models import RetryPolicy
from polyportia.config.registry import Registry
from polyportia.observability.trace import TraceBuilder


@dataclass
class ExecutionContext:
    registry: Registry
    trace: TraceBuilder
    request_params: dict[str, Any] = field(default_factory=dict)
    request_retry: RetryPolicy | None = None
    request_timeout_s: float | None = None
    visited_defined: set[str] = field(default_factory=set)
    visited_council: set[str] = field(default_factory=set)
    depth: int = 0
    max_depth: int = 8

    def child(self) -> ExecutionContext:
        return ExecutionContext(
            registry=self.registry,
            trace=self.trace,
            request_params=self.request_params,
            request_retry=self.request_retry,
            request_timeout_s=self.request_timeout_s,
            visited_defined=self.visited_defined,
            visited_council=self.visited_council,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )


class RecursionDepthExceeded(RuntimeError):
    pass
