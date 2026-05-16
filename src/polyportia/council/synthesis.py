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

_DEFAULT_SYNTH_PROMPT = """\
Below are responses from multiple AI panelists, each labelled with the model \
that produced it. They were all asked the same question.

Original question:
{{user_prompt}}

Panelist responses:
{{responses}}

Combine the panelists' answers into a single inclusive response to the \
original question. Include every distinct point or piece of information at \
least one panelist raised; do not filter based on your own judgement of \
which is correct.

When panelists contradict each other on a specific point, flag the \
disagreement explicitly and attribute the conflicting claims to the specific \
models that made them (e.g. "Model A says X, while Model B says Y"). Where \
the panelists agree, present the shared answer without attribution.

Write your output as a direct reply to the original question — not as a \
meta-summary about the panel. Do not say things like "the panelists agreed \
that…" except when explicitly flagging a contradiction.
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
