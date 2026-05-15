"""Runtime registry of providers, actual models, defined models, and councils.

The Registry is the mutable state behind both the YAML config and the SDK's
register_* helpers. There is a process-wide default registry used by the SDK
and HTTP server; tests/embedders can construct their own.
"""

from __future__ import annotations

from threading import RLock

from polyportia.config.models import (
    ActualModel,
    CouncilSpec,
    DefinedModel,
    FailurePolicy,
    PolyPortiaConfig,
    ProviderConfig,
    ServerConfig,
)


class RegistryError(KeyError):
    pass


class Registry:
    def __init__(self, config: PolyPortiaConfig | None = None) -> None:
        self._lock = RLock()
        self._providers: dict[str, ProviderConfig] = {}
        self._actual: dict[str, ActualModel] = {}
        self._defined: dict[str, DefinedModel] = {}
        self._councils: dict[str, CouncilSpec] = {}
        self._failure = FailurePolicy()
        self._server = ServerConfig()
        self._budget_usd_default: float | None = None
        if config is not None:
            self.load(config)

    def load(self, config: PolyPortiaConfig) -> None:
        with self._lock:
            self._providers.clear()
            self._actual.clear()
            self._defined.clear()
            self._councils.clear()
            for p in config.providers:
                self._providers[p.name] = p
            for a in config.actual_models:
                self._actual[a.id] = a
            for d in config.defined_models:
                self._defined[d.name] = d
            for c in config.councils:
                self._councils[c.name] = c
            self._failure = config.failure
            self._server = config.server
            self._budget_usd_default = config.budget_usd_default
            self._validate()

    def _validate(self) -> None:
        for a in self._actual.values():
            if a.provider not in self._providers:
                raise RegistryError(
                    f"actual_model '{a.id}' references unknown provider '{a.provider}'"
                )

    def register_provider(self, p: ProviderConfig) -> None:
        with self._lock:
            self._providers[p.name] = p

    def register_actual_model(self, m: ActualModel) -> None:
        with self._lock:
            if m.provider not in self._providers:
                raise RegistryError(f"unknown provider '{m.provider}'")
            self._actual[m.id] = m

    def register_defined_model(self, m: DefinedModel) -> None:
        with self._lock:
            self._defined[m.name] = m

    def register_council(self, c: CouncilSpec) -> None:
        with self._lock:
            self._councils[c.name] = c

    def get_provider(self, name: str) -> ProviderConfig:
        try:
            return self._providers[name]
        except KeyError as e:
            raise RegistryError(f"provider '{name}' not registered") from e

    def get_actual_model(self, model_id: str) -> ActualModel:
        try:
            return self._actual[model_id]
        except KeyError as e:
            raise RegistryError(f"actual_model '{model_id}' not registered") from e

    def get_defined_model(self, name: str) -> DefinedModel:
        try:
            return self._defined[name]
        except KeyError as e:
            raise RegistryError(f"defined_model '{name}' not registered") from e

    def get_council(self, name: str) -> CouncilSpec:
        try:
            return self._councils[name]
        except KeyError as e:
            raise RegistryError(f"council '{name}' not registered") from e

    def has_defined_model(self, name: str) -> bool:
        return name in self._defined

    def has_council(self, name: str) -> bool:
        return name in self._councils

    def has_actual_model(self, model_id: str) -> bool:
        return model_id in self._actual

    def list_actual_models(self) -> list[ActualModel]:
        return list(self._actual.values())

    def list_defined_models(self) -> list[DefinedModel]:
        return list(self._defined.values())

    def list_councils(self) -> list[CouncilSpec]:
        return list(self._councils.values())

    @property
    def failure(self) -> FailurePolicy:
        return self._failure

    @property
    def server(self) -> ServerConfig:
        return self._server

    @property
    def budget_usd_default(self) -> float | None:
        return self._budget_usd_default


_default_registry: Registry = Registry()


def get_default_registry() -> Registry:
    return _default_registry


def set_default_registry(r: Registry) -> None:
    global _default_registry
    _default_registry = r


def register_provider(p: ProviderConfig) -> None:
    _default_registry.register_provider(p)


def register_actual_model(m: ActualModel) -> None:
    _default_registry.register_actual_model(m)


def register_defined_model(m: DefinedModel) -> None:
    _default_registry.register_defined_model(m)


def register_council(c: CouncilSpec) -> None:
    _default_registry.register_council(c)
