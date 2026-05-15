"""Declarative Pydantic v2 models for PolyPortia configuration.

The three-layer model: Provider → ActualModel → DefinedModel. Councils are a
fourth, orthogonal concept that orchestrates ResolvableTarget members.
"""

from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


RetryCategory = Literal["timeout", "rate_limit", "server_error", "connection"]
_DEFAULT_RETRY_ON: list[RetryCategory] = ["timeout", "rate_limit", "server_error", "connection"]


class RetryPolicy(_Strict):
    max_retries: int = Field(default=2, ge=0, le=20)
    retry_on: list[RetryCategory] = Field(default_factory=lambda: list(_DEFAULT_RETRY_ON))
    backoff: Literal["linear", "exponential"] = "exponential"
    backoff_base_s: float = Field(default=0.5, ge=0)
    backoff_max_s: float = Field(default=30.0, ge=0)
    jitter: bool = True


class ProviderConfig(_Strict):
    name: str
    litellm_provider: str | None = None
    api_base: str | None = None
    api_key: SecretStr | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    default_params: dict[str, Any] = Field(default_factory=dict)
    default_retry: RetryPolicy = Field(default_factory=RetryPolicy)
    default_timeout_s: float | None = 60.0
    rpm: int | None = None
    tpm: int | None = None


class ActualModel(_Strict):
    """Canonical metadata for one real model. Declared once per provider/model."""

    id: str
    provider: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_1m_tokens: float | None = None
    output_cost_per_1m_tokens: float | None = None
    supports: list[Literal["streaming", "tools", "vision", "json_mode"]] = Field(
        default_factory=list
    )
    default_params: dict[str, Any] = Field(default_factory=dict)
    retry: RetryPolicy | None = None
    timeout_s: float | None = None


class ActualModelRef(_Strict):
    kind: Literal["actual"] = "actual"
    id: str


class DefinedModelRef(_Strict):
    kind: Literal["defined"] = "defined"
    name: str


class CouncilRef(_Strict):
    kind: Literal["council"] = "council"
    name: str


ModelTarget: TypeAlias = Annotated[
    ActualModelRef | DefinedModelRef,
    Field(discriminator="kind"),
]


class DefinedModel(_Strict):
    """User-named handle pointing at an ActualModel or another DefinedModel.

    `fallbacks` is walked transitively: a DefinedModel entry in the list follows
    its own fallback chain when reached.
    """

    name: str
    target: ModelTarget
    params: dict[str, Any] = Field(default_factory=dict)
    fallbacks: list[ModelTarget] = Field(default_factory=list)
    retry: RetryPolicy | None = None
    timeout_s: float | None = None
    description: str | None = None


class ParallelArray(_Strict):
    kind: Literal["parallel_array"] = "parallel_array"
    members: list["ResolvableTarget"]
    timeout_s: float | None = 60.0


class Synthesize(_Strict):
    kind: Literal["synthesize"] = "synthesize"
    members: list["ResolvableTarget"]
    synthesizer: "ResolvableTarget"
    synthesizer_prompt: str | None = None
    include_member_names: bool = True
    timeout_s: float | None = 60.0


class DebateVisibility(StrEnum):
    full_history = "full_history"
    prompt_and_peer_responses = "prompt_and_peer_responses"
    own_only_with_target = "own_only_with_target"


class DebateConfig(_Strict):
    visibility: DebateVisibility = DebateVisibility.prompt_and_peer_responses
    critique_prompt: str | None = None
    turns: int = Field(default=2, ge=1, le=10)
    termination: Literal["fixed_turns", "consensus", "judge"] = "fixed_turns"
    judge: "ResolvableTarget | None" = None


class Debate(_Strict):
    kind: Literal["debate"] = "debate"
    members: list["ResolvableTarget"]
    debate: DebateConfig = Field(default_factory=DebateConfig)
    output: Literal["array", "synthesize"] = "synthesize"
    synthesizer: "ResolvableTarget | None" = None
    timeout_s: float | None = 120.0


class ProposeAndReview(_Strict):
    """A proposer generates a response; reviewers approve/deny/comment.

    Reviewers express verdicts via standard tool calls — three tools are
    injected into the reviewer call: ``approve``, ``deny``, ``insight``. This
    avoids prompt-parsing fragility and works across providers via LiteLLM's
    unified tool-call surface.

    ``consensus`` combines verdicts: ``"all"`` requires every reviewer to
    approve, ``"any"`` requires one, an integer N requires at least N approvals.
    Insights count as approvals by default but the original message is still
    shown to the proposer in any revision round.

    On denial, ``on_denial`` controls the behaviour: ``return_denial`` (caller
    sees the denial + suggestions), ``revise`` (proposer re-runs up to
    ``max_revisions`` rounds with reviewer feedback), ``fail`` (raise an error).
    """

    kind: Literal["propose_review"] = "propose_review"
    proposer: "ResolvableTarget"
    reviewers: list["ResolvableTarget"]
    consensus: Literal["all", "any"] | int = "all"
    insight_counts_as_approval: bool = True
    review_prompt: str | None = None
    verdict_format: Literal["tool_calls", "keyword"] = "tool_calls"
    on_denial: Literal["return_denial", "revise", "fail"] = "return_denial"
    max_revisions: int = Field(default=1, ge=0, le=10)
    output: Literal["proposal", "envelope"] = "proposal"
    timeout_s: float | None = 120.0


CouncilStrategy: TypeAlias = Annotated[
    ParallelArray | Synthesize | Debate | ProposeAndReview,
    Field(discriminator="kind"),
]


class CouncilSpec(_Strict):
    name: str
    strategy: CouncilStrategy


ResolvableTarget: TypeAlias = Annotated[
    ActualModelRef | DefinedModelRef | CouncilRef | ParallelArray | Synthesize | Debate | ProposeAndReview,
    Field(discriminator="kind"),
]


class FailurePolicy(_Strict):
    on_failure: Literal["continue", "fail", "retry"] = "continue"
    min_success: int | None = 1
    min_success_fraction: float | None = None


class ServerConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8080
    trace_ring_size: int = 1000
    trace_file_sink: str | None = None


class PolyPortiaConfig(_Strict):
    providers: list[ProviderConfig] = Field(default_factory=list)
    actual_models: list[ActualModel] = Field(default_factory=list)
    defined_models: list[DefinedModel] = Field(default_factory=list)
    councils: list[CouncilSpec] = Field(default_factory=list)
    failure: FailurePolicy = Field(default_factory=FailurePolicy)
    server: ServerConfig = Field(default_factory=ServerConfig)


_ns: dict[str, Any] = {
    "ResolvableTarget": ResolvableTarget,
    "DebateConfig": DebateConfig,
}
ParallelArray.model_rebuild(_types_namespace=_ns)
Synthesize.model_rebuild(_types_namespace=_ns)
Debate.model_rebuild(_types_namespace=_ns)
DebateConfig.model_rebuild(_types_namespace=_ns)
ProposeAndReview.model_rebuild(_types_namespace=_ns)
CouncilSpec.model_rebuild(_types_namespace=_ns)


__all__ = [
    "ActualModel",
    "ActualModelRef",
    "CouncilRef",
    "CouncilSpec",
    "CouncilStrategy",
    "Debate",
    "DebateConfig",
    "DebateVisibility",
    "DefinedModel",
    "DefinedModelRef",
    "FailurePolicy",
    "ModelTarget",
    "ParallelArray",
    "PolyPortiaConfig",
    "ProposeAndReview",
    "ProviderConfig",
    "ResolvableTarget",
    "RetryPolicy",
    "ServerConfig",
    "Synthesize",
]
