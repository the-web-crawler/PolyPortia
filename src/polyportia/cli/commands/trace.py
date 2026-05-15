"""`polyportia trace <id>` — fetch and pretty-print a trace from the in-memory store."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.tree import Tree

from polyportia.observability.store import get_default_store

trace_app = typer.Typer(no_args_is_help=True, help="Inspect request traces")


@trace_app.command("show")
def show(
    trace_id: Annotated[str, typer.Argument()],
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    store = get_default_store()
    rec = store.get(trace_id)
    if rec is None:
        raise typer.BadParameter(f"trace '{trace_id}' not in store")
    console = Console()
    if json_out:
        console.print_json(data=rec.to_dict())
        return
    root = Tree(f"[bold]trace[/bold] {rec.trace_id}  ({rec.final_status})")
    by_id: dict[str, Tree] = {}
    for span in rec.spans:
        node = Tree(
            f"{span.kind} {span.target_repr}  "
            f"[dim]{span.latency_ms or 0:.1f}ms  status={span.status}[/dim]"
        )
        by_id[span.span_id] = node
        if span.parent_span_id and span.parent_span_id in by_id:
            by_id[span.parent_span_id].add(node)
        else:
            root.add(node)
    console.print(root)


@trace_app.command("ls")
def ls(limit: Annotated[int, typer.Option("--limit", "-n")] = 20) -> None:
    store = get_default_store()
    console = Console()
    for rec in store.list(limit=limit):
        console.print(f"{rec.trace_id}  {rec.final_status}  {rec.request_summary}")
