from __future__ import annotations

import os

from polyportia.config.loader import load_config_from_string


def test_loads_minimal_yaml():
    cfg = load_config_from_string(
        """
providers:
  - name: anthropic
    api_key: literal-key
actual_models:
  - id: anthropic/claude-opus-4-7
    provider: anthropic
"""
    )
    assert cfg.providers[0].name == "anthropic"
    assert cfg.providers[0].api_key.get_secret_value() == "literal-key"
    assert cfg.actual_models[0].id == "anthropic/claude-opus-4-7"


def test_env_var_expansion(monkeypatch):
    monkeypatch.setenv("MY_KEY", "from-env")
    cfg = load_config_from_string(
        """
providers:
  - name: x
    api_key: ${MY_KEY}
actual_models: []
"""
    )
    assert cfg.providers[0].api_key.get_secret_value() == "from-env"


def test_env_var_default():
    if "DOES_NOT_EXIST_XYZ" in os.environ:
        del os.environ["DOES_NOT_EXIST_XYZ"]
    cfg = load_config_from_string(
        """
providers:
  - name: x
    api_key: ${DOES_NOT_EXIST_XYZ:-fallback-default}
actual_models: []
"""
    )
    assert cfg.providers[0].api_key.get_secret_value() == "fallback-default"


def test_defined_model_with_fallbacks_parses():
    cfg = load_config_from_string(
        """
providers:
  - name: a
    api_key: k
  - name: b
    api_key: k
actual_models:
  - id: a/m1
    provider: a
  - id: b/m2
    provider: b
defined_models:
  - name: thinking
    target: {kind: actual, id: a/m1}
    fallbacks:
      - {kind: actual, id: b/m2}
      - {kind: defined, name: fast}
  - name: fast
    target: {kind: actual, id: a/m1}
"""
    )
    thinking = next(d for d in cfg.defined_models if d.name == "thinking")
    assert thinking.target.kind == "actual"
    assert thinking.fallbacks[0].kind == "actual"
    assert thinking.fallbacks[1].kind == "defined"
    assert thinking.fallbacks[1].name == "fast"
