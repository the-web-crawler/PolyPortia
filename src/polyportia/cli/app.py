"""PolyPortia command-line entry point."""

from __future__ import annotations

import typer

from polyportia.cli.commands.councils import councils_app
from polyportia.cli.commands.models import models_app
from polyportia.cli.commands.run import run as run_command
from polyportia.cli.commands.serve import serve as serve_command
from polyportia.cli.commands.trace import trace_app

app = typer.Typer(no_args_is_help=True, add_completion=False, help="PolyPortia gateway CLI")
app.command(name="serve")(serve_command)
app.command(name="run")(run_command)
app.add_typer(models_app, name="models")
app.add_typer(councils_app, name="councils")
app.add_typer(trace_app, name="trace")


if __name__ == "__main__":
    app()
