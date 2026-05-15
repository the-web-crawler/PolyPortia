"""Load PolyPortiaConfig from a YAML file with ${ENV_VAR} expansion."""

import os
import re
from pathlib import Path

import yaml

from polyportia.config.models import PolyPortiaConfig

_ENV_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: object) -> object:
    if isinstance(value, str):

        def sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path) -> PolyPortiaConfig:
    """Read a YAML file and return a validated PolyPortiaConfig."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if raw is None:
        raw = {}
    expanded = _expand_env(raw)
    return PolyPortiaConfig.model_validate(expanded)


def load_config_from_string(yaml_text: str) -> PolyPortiaConfig:
    raw = yaml.safe_load(yaml_text) or {}
    return PolyPortiaConfig.model_validate(_expand_env(raw))
