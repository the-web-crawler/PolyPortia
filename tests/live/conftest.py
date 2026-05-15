"""Shared fixtures and gating for the live test suite.

The whole suite is opt-in via the env var ``RUN_LIVE_TESTS=1`` and an Ollama
server reachable at ``OLLAMA_BASE_URL`` (default http://localhost:11434).

How the gating works:

1. ``pytest_collection_modifyitems`` adds a ``skip`` marker to every test in
   this directory unless ``RUN_LIVE_TESTS=1``. This way, if you accidentally
   run ``pytest`` from the repo root, the live tests don't try to phone home.

2. The ``live_registry`` fixture loads ``polyportia.live.yaml`` and probes
   Ollama. If the probe fails (Ollama isn't running, or the configured
   models aren't pulled), the test is skipped with a clear message rather
   than failing.

You can also run a single file with: ``RUN_LIVE_TESTS=1 pytest tests/live/test_01_single_model.py -v -s``
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from polyportia.config.loader import load_config
from polyportia.config.registry import Registry, set_default_registry

LIVE_YAML = Path(__file__).parent / "polyportia.live.yaml"


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 — pytest hook
    """Skip every test in this directory unless RUN_LIVE_TESTS=1."""
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip_marker = pytest.mark.skip(
        reason="live tests are opt-in; set RUN_LIVE_TESTS=1 to enable"
    )
    for item in items:
        if "tests/live" in str(item.fspath):
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def ollama_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture(scope="session")
def _ollama_reachable(ollama_url: str) -> bool:
    """Probe the Ollama /api/tags endpoint. Returns True if it responds."""
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def _ollama_models(_ollama_reachable: bool, ollama_url: str) -> set[str]:
    """Return the set of model names currently pulled in Ollama."""
    if not _ollama_reachable:
        return set()
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=3.0)
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()


@pytest.fixture
def live_registry(_ollama_reachable: bool) -> Registry:
    """Load polyportia.live.yaml and skip if Ollama isn't reachable."""
    if not _ollama_reachable:
        pytest.skip("Ollama not reachable at OLLAMA_BASE_URL")
    cfg = load_config(LIVE_YAML)
    reg = Registry(cfg)
    set_default_registry(reg)
    return reg


@pytest.fixture
def require_models(_ollama_models: set[str]):
    """Returns a callable that skips the test if any required model isn't pulled.

    Usage:
        def test_something(require_models):
            require_models("llama3.2:1b", "llama3.2:3b")
            ...
    """

    def _check(*models: str) -> None:
        missing = [m for m in models if m not in _ollama_models]
        if missing:
            pytest.skip(f"Ollama is missing required models: {missing}. Run: ollama pull <name>")

    return _check
