from __future__ import annotations

import pytest

from polyportia.config.models import ActualModel, ActualModelRef, DefinedModel, DefinedModelRef
from polyportia.config.registry import Registry, RegistryError
from polyportia.config.resolver import resolve, resolve_for_model


def test_resolve_actual_ref(test_registry: Registry):
    out = resolve(ActualModelRef(id="anthropic/claude-opus-4-7"), test_registry)
    assert isinstance(out, ActualModel)
    assert out.provider == "anthropic"


def test_resolve_defined_ref_returns_defined(test_registry: Registry):
    out = resolve(DefinedModelRef(name="thinking"), test_registry)
    assert isinstance(out, DefinedModel)
    assert out.name == "thinking"


def test_resolve_unknown_raises(test_registry: Registry):
    with pytest.raises(RegistryError):
        resolve(ActualModelRef(id="ghost/x"), test_registry)


def test_resolve_for_model_rejects_council_strategy(test_registry: Registry):
    from polyportia.config.models import ParallelArray

    with pytest.raises(TypeError):
        resolve_for_model(
            ParallelArray(members=[ActualModelRef(id="anthropic/claude-haiku-4-5")]),  # type: ignore[arg-type]
            test_registry,
        )
