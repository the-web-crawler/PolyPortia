"""CLI smoke tests via Typer's CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from polyportia.cli.app import app
from polyportia.config.loader import load_config_from_string
from polyportia.config.registry import Registry, set_default_registry

_TEST_YAML = """
providers:
  - name: anthropic
    api_key: k
actual_models:
  - id: anthropic/claude-opus-4-7
    provider: anthropic
defined_models:
  - name: thinking
    target: {kind: actual, id: anthropic/claude-opus-4-7}
"""


def setup_function(_fn):
    set_default_registry(Registry(load_config_from_string(_TEST_YAML)))


def test_models_ls():
    runner = CliRunner()
    result = runner.invoke(app, ["models", "ls"])
    assert result.exit_code == 0, result.stdout
    assert "thinking" in result.stdout
    assert "anthropic/claude-opus-4-7" in result.stdout


def test_models_show():
    runner = CliRunner()
    result = runner.invoke(app, ["models", "show", "thinking"])
    assert result.exit_code == 0, result.stdout
    assert "defined:thinking" in result.stdout


def test_run_with_message(mock_provider, monkeypatch, tmp_path: Path):
    """`polyportia run thinking -m hi` returns the mocked content."""
    yaml_file = tmp_path / "polyportia.yaml"
    yaml_file.write_text(_TEST_YAML)
    mock_provider.set_response("anthropic/claude-opus-4-7", "cli-ok")
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "thinking", "-m", "hi", "--config", str(yaml_file)],
    )
    assert result.exit_code == 0, result.stdout
    assert "cli-ok" in result.stdout
