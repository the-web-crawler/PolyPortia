"""`polyportia run` — one-shot completion against an alias."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from polyportia.config.loader import load_config
from polyportia.config.registry import Registry, set_default_registry
from polyportia.sdk.client import acomplete


def run(
    name_or_id: Annotated[str, typer.Argument(help="DefinedModel name, council name, or ActualModel id")],
    message: Annotated[str | None, typer.Option("--message", "-m", help="User message")] = None,
    stdin: Annotated[bool, typer.Option("--stdin", help="Read user message from stdin")] = False,
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    system: Annotated[str | None, typer.Option("--system", help="Optional system prompt")] = None,
    show_trace_id: Annotated[bool, typer.Option("--trace-id")] = False,
) -> None:
    """Run one completion through PolyPortia and print the response."""
    if config is not None:
        set_default_registry(Registry(load_config(config)))

    if stdin:
        message = sys.stdin.read()
    if message is None:
        raise typer.BadParameter("provide --message or --stdin")

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})

    sdk_result = asyncio.run(
        acomplete(model=name_or_id, messages=messages, budget_usd="unlimited")
    )
    result, trace_id = sdk_result[0], sdk_result[1]
    console = Console()
    console.print(result.content or "")
    if show_trace_id:
        console.print(f"\n[dim]trace: {trace_id}[/dim]")
