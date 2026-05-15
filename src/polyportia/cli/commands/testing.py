"""``polyportia testing dummy`` — start a local dummy completions server.

The dummy speaks OpenAI's chat-completions protocol and is useful for
exercising PolyPortia end-to-end without burning provider credits.
"""

from __future__ import annotations

import typer
import uvicorn

testing_app = typer.Typer(no_args_is_help=True, add_completion=False)


@testing_app.command("dummy")
def dummy(
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(9999, help="Port to bind."),
    log_level: str = typer.Option("warning", help="uvicorn log level."),
) -> None:
    """Start the OpenAI-compatible dummy completions server.

    Model-id patterns drive behaviour. See
    ``polyportia.testing.dummy_server`` for the full table. Examples:

      • ``dummy/echo``                   — echo the last user message
      • ``dummy/fixed/Hello%20world``    — fixed content
      • ``dummy/error/429``              — return HTTP 429
      • ``dummy/tool/email``             — return a tool_call to ``email``
      • ``dummy/delay/250``              — sleep 250 ms before responding

    Use a PolyPortia provider with ``api_base: http://<host>:<port>`` to
    route LiteLLM through this dummy.
    """
    uvicorn.run(
        "polyportia.testing.dummy_server:app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level,
    )
