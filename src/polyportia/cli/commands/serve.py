"""`polyportia serve` — boot the FastAPI app via uvicorn."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn

from polyportia.config.loader import load_config
from polyportia.config.registry import Registry, set_default_registry


def serve(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to a polyportia.yaml file"),
    ] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p")] = 8080,
    reload: Annotated[bool, typer.Option("--reload")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "info",
) -> None:
    """Boot the PolyPortia HTTP server."""
    if config is not None:
        cfg = load_config(config)
        set_default_registry(Registry(cfg))
        host = host or cfg.server.host
        port = port or cfg.server.port

    uvicorn.run(
        "polyportia.cli.commands.serve:_app_factory",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        factory=True,
    )


def _app_factory() -> Any:  # called by uvicorn when using factory=True
    from polyportia.config.registry import get_default_registry
    from polyportia.server.app import create_app

    return create_app(registry=get_default_registry())
