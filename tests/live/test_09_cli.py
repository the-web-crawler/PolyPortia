"""Live test #9 — `polyportia` CLI commands against a real Ollama config.

What this validates
-------------------
- ``polyportia models ls`` prints the actual + defined sections from the
  loaded YAML.
- ``polyportia models show <name>`` shows the resolved tree for a defined
  model, including its fallback chain.
- ``polyportia councils ls`` lists the councils.
- ``polyportia run fast -m "..."`` performs a real completion via Ollama
  and prints the response.

How to run
----------
    ollama pull llama3.2:1b
    RUN_LIVE_TESTS=1 pytest tests/live/test_09_cli.py -v -s
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from polyportia.cli.app import app

LIVE_YAML = Path(__file__).parent / "polyportia.live.yaml"


def test_cli_models_ls(live_registry):
    runner = CliRunner()
    r = runner.invoke(app, ["models", "ls", "--config", str(LIVE_YAML)])
    print(f"\n[live#09-models-ls]\n{r.stdout}")
    assert r.exit_code == 0, r.stdout
    assert "thinking" in r.stdout
    assert "ollama_chat/llama3.2:3b" in r.stdout


def test_cli_models_show_with_fallback(live_registry):
    runner = CliRunner()
    r = runner.invoke(app, ["models", "show", "brittle", "--config", str(LIVE_YAML)])
    print(f"\n[live#09-models-show-brittle]\n{r.stdout}")
    assert r.exit_code == 0
    assert "this-model-does-not-exist" in r.stdout
    assert "fallback" in r.stdout.lower()


def test_cli_councils_ls(live_registry):
    runner = CliRunner()
    r = runner.invoke(app, ["councils", "ls", "--config", str(LIVE_YAML)])
    print(f"\n[live#09-councils-ls]\n{r.stdout}")
    assert r.exit_code == 0
    assert "trio" in r.stdout
    assert "meta" in r.stdout


def test_cli_run(live_registry, require_models):
    require_models("llama3.2:1b")
    runner = CliRunner()
    r = runner.invoke(
        app,
        ["run", "fast", "-m", "Say 'cli-ok'.", "--config", str(LIVE_YAML)],
    )
    print(f"\n[live#09-run]\n{r.stdout}")
    assert r.exit_code == 0, r.stdout
    assert r.stdout.strip()  # Some response came back
