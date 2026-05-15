"""`polyportia councils ls / show` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from polyportia.config.loader import load_config
from polyportia.config.models import (
    ActualModelRef,
    CouncilRef,
    Debate,
    DefinedModelRef,
    ParallelArray,
    ProposeAndReview,
    ResolvableTarget,
    Synthesize,
)
from polyportia.config.registry import Registry, get_default_registry, set_default_registry

councils_app = typer.Typer(no_args_is_help=True, help="Inspect councils")


def _maybe_load(config: Path | None) -> Registry:
    if config is not None:
        set_default_registry(Registry(load_config(config)))
    return get_default_registry()


def _ref_label(target: ResolvableTarget) -> str:
    if isinstance(target, ActualModelRef):
        return f"actual:{target.id}"
    if isinstance(target, DefinedModelRef):
        return f"defined:{target.name}"
    if isinstance(target, CouncilRef):
        return f"council:{target.name}"
    if isinstance(target, ParallelArray):
        return "inline parallel_array"
    if isinstance(target, Synthesize):
        return "inline synthesize"
    if isinstance(target, Debate):
        return "inline debate"
    if isinstance(target, ProposeAndReview):
        return "inline propose_review"
    return type(target).__name__


@councils_app.command("ls")
def ls(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    reg = _maybe_load(config)
    console = Console()
    table = Table(title="Councils")
    table.add_column("name")
    table.add_column("kind")
    table.add_column("members")
    for c in reg.list_councils():
        s = c.strategy
        if isinstance(s, ProposeAndReview):
            members = f"proposer={_ref_label(s.proposer)}, {len(s.reviewers)} reviewers"
        else:
            members = f"{len(s.members)} members"
        table.add_row(c.name, s.kind, members)
    console.print(table)


def _expand(target: ResolvableTarget, reg: Registry, parent: Tree) -> None:
    label = _ref_label(target)
    node = parent.add(label)
    if isinstance(target, CouncilRef) and reg.has_council(target.name):
        spec = reg.get_council(target.name)
        _expand_strategy(spec.strategy, reg, node)
    elif isinstance(target, (ParallelArray, Synthesize, Debate, ProposeAndReview)):
        _expand_strategy(target, reg, node)


def _expand_strategy(
    strategy: ParallelArray | Synthesize | Debate | ProposeAndReview,
    reg: Registry,
    parent: Tree,
) -> None:
    if isinstance(strategy, ProposeAndReview):
        proposer_node = parent.add(f"proposer: {_ref_label(strategy.proposer)}")
        if isinstance(strategy.proposer, CouncilRef) and reg.has_council(strategy.proposer.name):
            _expand(strategy.proposer, reg, proposer_node)
        for r in strategy.reviewers:
            _expand(r, reg, parent)
        return
    for m in strategy.members:
        _expand(m, reg, parent)
    if isinstance(strategy, Synthesize):
        synth_node = parent.add(f"synthesizer: {_ref_label(strategy.synthesizer)}")
        if isinstance(strategy.synthesizer, CouncilRef) and reg.has_council(strategy.synthesizer.name):
            _expand(strategy.synthesizer, reg, synth_node)


@councils_app.command("show")
def show(
    name: Annotated[str, typer.Argument()],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    reg = _maybe_load(config)
    if not reg.has_council(name):
        raise typer.BadParameter(f"council '{name}' not registered")
    spec = reg.get_council(name)
    tree = Tree(f"[bold]council:{name}[/bold] ({spec.strategy.kind})")
    _expand_strategy(spec.strategy, reg, tree)
    Console().print(tree)
