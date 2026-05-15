"""Default synthesizer prompt + message-builder for Synthesize councils."""

from __future__ import annotations

from typing import Any

from polyportia.config.models import (
    ActualModelRef,
    CouncilRef,
    DefinedModelRef,
    ResolvableTarget,
)
from polyportia.council.failure import MemberOutcome

_DEFAULT_SYNTH_PROMPT = """You are synthesising the responses of multiple AI \
panelists who all answered the same prompt. Read every panelist's answer \
carefully, identify points of agreement and disagreement, weigh evidence and \
reasoning quality, and then produce a single best response that incorporates \
the strongest insights. Return only the final synthesised answer.

Original prompt:
{{user_prompt}}

Panelist responses:
{{responses}}
"""


def _short_repr(target: ResolvableTarget) -> str:
    if isinstance(target, ActualModelRef):
        return target.id
    if isinstance(target, DefinedModelRef):
        return f"defined:{target.name}"
    if isinstance(target, CouncilRef):
        return f"council:{target.name}"
    return type(target).__name__


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return ""


def _format_responses(
    members: list[ResolvableTarget],
    outcomes: list[MemberOutcome],
    include_names: bool,
) -> str:
    parts: list[str] = []
    for member, outcome in zip(members, outcomes, strict=True):
        if not outcome.ok or outcome.result is None:
            continue
        if include_names:
            parts.append(f"--- Panelist: {_short_repr(member)} ---\n{outcome.result.content}")
        else:
            parts.append(outcome.result.content)
    return "\n\n".join(parts)


def build_synth_messages(
    *,
    base_messages: list[dict[str, Any]],
    members: list[ResolvableTarget],
    outcomes: list[MemberOutcome],
    template: str | None,
    include_names: bool,
) -> list[dict[str, Any]]:
    """Build the message list passed to the synthesizer target."""
    tpl = template or _DEFAULT_SYNTH_PROMPT
    user_prompt = _last_user_text(base_messages)
    responses = _format_responses(members, outcomes, include_names)
    rendered = tpl.replace("{{user_prompt}}", user_prompt).replace("{{responses}}", responses)
    return [{"role": "user", "content": rendered}]
