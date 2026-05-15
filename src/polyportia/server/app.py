"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from polyportia.config.models import PolyPortiaConfig
from polyportia.config.registry import Registry, get_default_registry
from polyportia.observability.logging import configure_logging
from polyportia.observability.store import TraceStore, get_default_store
from polyportia.server.routes_health import router as health_router
from polyportia.server.routes_openai import router as openai_router
from polyportia.server.routes_traces import router as traces_router


def create_app(
    *,
    config: PolyPortiaConfig | None = None,
    registry: Registry | None = None,
    trace_store: TraceStore | None = None,
    log_level: str = "INFO",
) -> FastAPI:
    configure_logging(log_level)
    app = FastAPI(title="PolyPortia", version="0.1.0")

    if config is not None:
        registry = Registry(config)
    if registry is None:
        registry = get_default_registry()
    if trace_store is None:
        if config is not None and config.server.trace_ring_size:
            trace_store = TraceStore(
                maxlen=config.server.trace_ring_size,
                file_sink=config.server.trace_file_sink,
            )
        else:
            trace_store = get_default_store()

    app.state.registry = registry
    app.state.trace_store = trace_store

    app.include_router(health_router)
    app.include_router(openai_router)
    app.include_router(traces_router)
    return app
