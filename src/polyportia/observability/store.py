"""In-memory ring buffer + optional JSONL file sink for trace records."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from threading import RLock

from polyportia.observability.trace import TraceRecord


class TraceStore:
    def __init__(self, maxlen: int = 1000, file_sink: str | None = None) -> None:
        self._lock = RLock()
        self._ring: deque[TraceRecord] = deque(maxlen=maxlen)
        self._index: dict[str, TraceRecord] = {}
        self._file_sink: Path | None = Path(file_sink) if file_sink else None

    def add(self, record: TraceRecord) -> None:
        with self._lock:
            if len(self._ring) == self._ring.maxlen:
                evicted = self._ring[0]
                self._index.pop(evicted.trace_id, None)
            self._ring.append(record)
            self._index[record.trace_id] = record
        if self._file_sink is not None:
            self._file_sink.parent.mkdir(parents=True, exist_ok=True)
            with self._file_sink.open("a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

    def get(self, trace_id: str) -> TraceRecord | None:
        with self._lock:
            return self._index.get(trace_id)

    def list(self, limit: int = 50) -> Iterable[TraceRecord]:
        with self._lock:
            return list(self._ring)[-limit:][::-1]


_default_store = TraceStore()


def get_default_store() -> TraceStore:
    return _default_store


def set_default_store(store: TraceStore) -> None:
    global _default_store
    _default_store = store
