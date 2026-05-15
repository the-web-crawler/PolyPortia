"""`polyportia models ls / show` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from polyportia.config.loader import load_config
from polyportia.config.policy import resolve_retry, resolve_timeout
from polyportia.config.registry import Registry, get_default_registry, set_default_registry

models_app = typer.Typer(no_args_is_help=True, help="Inspect actual + defined models")


def _maybe_load(config: Path | None) -> Registry:
    if config is not None:
        set_default_registry(Registry(load_config(config)))
    return get_default_registry()


@models_app.command("ls")
def ls(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    actual: Annotated[bool, typer.Option("--actual", help="Show only actual models")] = False,
    defined: Annotated[bool, typer.Option("--defined", help="Show only defined models")] = False,
) -> None:
    reg = _maybe_load(config)
    console = Console()
    show_both = not (actual or defined)
    if actual or show_both:
        table = Table(title="Actual models")
        table.add_column("id")
        table.add_column("provider")
        table.add_column("ctx")
        table.add_column("$ in / $ out (per 1M)")
        for a in reg.list_actual_models():
            table.add_row(
                a.id,
                a.provider,
                str(a.context_window or "-"),
                f"{a.input_cost_per_1m_tokens or '-'} / {a.output_cost_per_1m_tokens or '-'}",
            )
        console.print(table)
    if defined or show_both:
        table = Table(title="Defined models")
        table.add_column("name")
        table.add_column("target")
        table.add_column("fallbacks")
        for d in reg.list_defined_models():
            target_repr = (
                f"actual:{d.target.id}" if d.target.kind == "actual" else f"defined:{d.target.name}"
            )
            fb = ", ".join(
                f"actual:{f.id}" if f.kind == "actual" else f"defined:{f.name}" for f in d.fallbacks
            ) or "-"
            table.add_row(d.name, target_repr, fb)
        console.print(table)


@models_app.command("show")
def show(
    name: Annotated[str, typer.Argument(help="DefinedModel name (or actual model id)")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    reg = _maybe_load(config)
    console = Console()
    if reg.has_defined_model(name):
        d = reg.get_defined_model(name)
        tree = Tree(f"[bold]defined:{d.name}[/bold]")
        target_label = (
            f"primary → actual:{d.target.id}"
            if d.target.kind == "actual"
            else f"primary → defined:{d.target.name}"
        )
        tree.add(target_label)
        for i, fb in enumerate(d.fallbacks):
            label = f"fallback[{i}] → actual:{fb.id}" if fb.kind == "actual" else f"fallback[{i}] → defined:{fb.name}"
            tree.add(label)
        if d.target.kind == "actual" and reg.has_actual_model(d.target.id):
            actual = reg.get_actual_model(d.target.id)
            provider = reg.get_provider(actual.provider)
            retry = resolve_retry(request=None, defined=d, actual=actual, provider=provider)
            timeout = resolve_timeout(request=None, defined=d, actual=actual, provider=provider)
            tree.add(
                f"effective retry: {retry.policy.max_retries} retries, "
                f"backoff={retry.policy.backoff} (source: {retry.source})"
            )
            tree.add(f"effective timeout: {timeout.value}s (source: {timeout.source})")
        console.print(tree)
        return
    if reg.has_actual_model(name):
        a = reg.get_actual_model(name)
        console.print_json(data=a.model_dump())
        return
    raise typer.BadParameter(f"unknown model '{name}'")
