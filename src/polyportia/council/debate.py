"""Debate council strategy: N turns of cross-model critique.

Each visibility mode controls what every member sees on its turn:
- ``full_history``         — labelled transcript of every prior turn
- ``prompt_and_peer_responses`` — only peers' most-recent turn
- ``own_only_with_target`` — only the member's own prior answer, with peer
  responses aggregated into a single critique target

After all turns (or early termination via consensus/judge), the final per-member
responses are returned as an array envelope or synthesised through the
configured synthesizer target.
"""

from __future__ import annotations

import asyncio
from typing import Any

from polyportia.config.models import (
    ActualModelRef,
    CouncilRef,
    Debate,
    DebateVisibility,
    DefinedModelRef,
    ResolvableTarget,
)
from polyportia.council.context import ExecutionContext
from polyportia.council.failure import (
    MemberOutcome,
    apply_failure_policy,
    outcomes_to_array,
)
from polyportia.providers.base import ProviderResult

_DEFAULT_CRITIQUE_PROMPT = """\
The original question has been answered by multiple AI panelists. Below is \
the conversation so far. Other panelists are identified by their model name; \
"your prior answer" refers to your own previous output.

Your prior answer:
{{own_prior}}

Other panelists' answers:
{{peer_responses}}

You are now playing devil's advocate. Read every answer carefully — \
including your own — and look for loopholes: weak arguments, hidden \
assumptions, missing considerations, edge cases, counterexamples, factual \
errors, or alternative interpretations. Be adversarial; assume each answer \
is wrong somewhere and find where.

First, list the loopholes you found, indicating which answer each appears \
in (your own or a specific panelist's). Then produce a revised answer to \
the original question that survives your own critique.
"""

_JUDGE_PROMPT = """\
You are judging whether a panel of AI models has reached a useful stopping \
point in their debate. Below are the latest responses from each panelist. \
If they have converged or further turns are unlikely to help, reply with the \
single word DONE. Otherwise reply CONTINUE.

Latest responses:
{{responses}}
"""

_DEBATE_FINAL_SYNTH_PROMPT = """\
Multiple AI panelists have just finished debating the original question, \
critiquing each other across several turns. Below are their final answers, \
each labelled with the model that produced it. Each answer may contain \
both the panelist's loophole notes from its last critique and its final \
revised answer; treat the revised answer as the panelist's substantive \
position.

Original question:
{{user_prompt}}

Final panelist answers:
{{responses}}

Combine the panelists' final answers into a single inclusive response to \
the original question. Include every distinct point or piece of information \
at least one panelist raised; do not filter based on your own judgement of \
which is correct.

When panelists still contradict each other after the debate, flag the \
disagreement explicitly and attribute the conflicting claims to the \
specific models (e.g. "Model A says X, while Model B says Y"). Where they \
agree, present the shared answer without attribution.

Write your output as a direct reply to the original question — not as a \
meta-summary of the debate. Do not refer to "the panelists" or "the \
debate" except when flagging a contradiction.
"""


def _short_name(target: ResolvableTarget) -> str:
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


def _previous_own_response(history: list[list[str | None]], self_idx: int) -> str:
    for turn in reversed(history):
        v = turn[self_idx]
        if v is not None:
            return v
    return ""


def _build_turn_messages(
    *,
    visibility: DebateVisibility,
    base_messages: list[dict[str, Any]],
    history: list[list[str | None]],
    self_idx: int,
    members: list[ResolvableTarget],
    critique_tpl: str,
) -> list[dict[str, Any]]:
    own_prior = _previous_own_response(history, self_idx)

    if visibility == DebateVisibility.full_history:
        lines: list[str] = []
        for turn_idx, turn in enumerate(history):
            for i, content in enumerate(turn):
                if content is None:
                    continue
                who = "(you)" if i == self_idx else f"({_short_name(members[i])})"
                lines.append(f"[Turn {turn_idx}] {who}:\n{content}")
        peer_blob = "\n\n".join(lines)
        critique = critique_tpl.replace("{{own_prior}}", own_prior).replace(
            "{{peer_responses}}", peer_blob
        )
        return list(base_messages) + [{"role": "user", "content": critique}]

    if visibility == DebateVisibility.prompt_and_peer_responses:
        last = history[-1]
        parts: list[str] = []
        for i, content in enumerate(last):
            if i == self_idx or content is None:
                continue
            parts.append(f"({_short_name(members[i])}):\n{content}")
        peer_blob = "\n\n".join(parts)
        critique = critique_tpl.replace("{{own_prior}}", own_prior).replace(
            "{{peer_responses}}", peer_blob
        )
        return list(base_messages) + [{"role": "user", "content": critique}]

    if visibility == DebateVisibility.own_only_with_target:
        last = history[-1]
        peer_parts = [c for i, c in enumerate(last) if i != self_idx and c is not None]
        peer_blob = "\n\n".join(peer_parts)
        critique = critique_tpl.replace("{{own_prior}}", own_prior).replace(
            "{{peer_responses}}", peer_blob
        )
        prompt = _last_user_text(base_messages)
        return [
            {"role": "user", "content": prompt},
            {"role": "user", "content": critique},
        ]

    raise ValueError(f"unknown DebateVisibility: {visibility}")


async def _safe_call(
    member: ResolvableTarget,
    messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> tuple[str | None, BaseException | None]:
    from polyportia.council.executor import execute_target

    try:
        result = await execute_target(member, messages, ctx.child())
        return result.content, None
    except BaseException as e:
        return None, e


def _detect_consensus(turn: list[str | None]) -> bool:
    """Cheap consensus detector: all non-null responses are byte-identical."""
    non_null = [c for c in turn if c is not None]
    if len(non_null) < 2:
        return False
    first = non_null[0].strip()
    return all(c.strip() == first for c in non_null[1:])


def _build_judge_messages(
    history: list[list[str | None]], members: list[ResolvableTarget]
) -> list[dict[str, Any]]:
    last = history[-1]
    parts: list[str] = []
    for i, content in enumerate(last):
        if content is None:
            continue
        parts.append(f"({_short_name(members[i])}):\n{content}")
    body = _JUDGE_PROMPT.replace("{{responses}}", "\n\n".join(parts))
    return [{"role": "user", "content": body}]


def _final_outcomes(
    members: list[ResolvableTarget], history: list[list[str | None]]
) -> list[MemberOutcome]:
    outcomes: list[MemberOutcome] = []
    for i, member in enumerate(members):
        last_content: str | None = None
        for turn in reversed(history):
            if turn[i] is not None:
                last_content = turn[i]
                break
        repr_str = _short_name(member)
        if last_content is None:
            outcomes.append(
                MemberOutcome(
                    member_repr=repr_str,
                    result=None,
                    error=RuntimeError("debate member never produced a response"),
                )
            )
        else:
            outcomes.append(
                MemberOutcome(
                    member_repr=repr_str,
                    result=ProviderResult(model_id=repr_str, content=last_content),
                    error=None,
                )
            )
    return outcomes


def _array_envelope_result(outcomes: list[MemberOutcome]) -> ProviderResult:
    array = outcomes_to_array(outcomes)
    successful = [o for o in outcomes if o.ok]
    summary = [f"(debate: {len(successful)}/{len(outcomes)} members)"]
    for entry in array:
        if "content" in entry:
            summary.append(f"- {entry['member']}: {entry['content']}")
        else:
            summary.append(f"- {entry['member']}: ERROR {entry.get('error')}")
    return ProviderResult(
        model_id="debate",
        content="\n".join(summary),
        raw={"object": "polyportia.council", "responses": array},
        finish_reason="stop",
    )


async def run_debate(
    spec: Debate,
    base_messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    from polyportia.council.executor import execute_target

    members = spec.members
    critique_tpl = spec.debate.critique_prompt or _DEFAULT_CRITIQUE_PROMPT
    history: list[list[str | None]] = []

    with ctx.trace.span(kind="debate", target_repr="debate") as debate_span:
        with ctx.trace.span(kind="debate_turn", target_repr="turn:0"):
            turn0_pairs = await asyncio.gather(
                *(_safe_call(m, list(base_messages), ctx) for m in members)
            )
        history.append([p[0] for p in turn0_pairs])

        for t in range(1, spec.debate.turns):
            per_member_msgs = [
                _build_turn_messages(
                    visibility=spec.debate.visibility,
                    base_messages=base_messages,
                    history=history,
                    self_idx=i,
                    members=members,
                    critique_tpl=critique_tpl,
                )
                for i in range(len(members))
            ]
            with ctx.trace.span(kind="debate_turn", target_repr=f"turn:{t}"):
                pairs = await asyncio.gather(
                    *(
                        _safe_call(members[i], per_member_msgs[i], ctx)
                        for i in range(len(members))
                    )
                )
            history.append([p[0] for p in pairs])

            if (
                spec.debate.termination == "consensus"
                and _detect_consensus(history[-1])
            ):
                break
            if (
                spec.debate.termination == "judge"
                and spec.debate.judge is not None
            ):
                try:
                    verdict = await execute_target(
                        spec.debate.judge,
                        _build_judge_messages(history, members),
                        ctx.child(),
                    )
                    if "DONE" in (verdict.content or "").upper():
                        break
                except BaseException:
                    # If the judge fails we just continue the loop.
                    pass

        outcomes = _final_outcomes(members, history)
        apply_failure_policy(outcomes, ctx.registry.failure)
        if not all(o.ok for o in outcomes):
            debate_span.status = "fellback"

        if spec.output == "array":
            return _array_envelope_result(outcomes)

        if spec.synthesizer is None:
            raise ValueError(
                "Debate.output='synthesize' but no synthesizer is configured"
            )
        from polyportia.council.synthesis import build_synth_messages

        synth_messages = build_synth_messages(
            base_messages=base_messages,
            members=members,
            outcomes=outcomes,
            template=_DEBATE_FINAL_SYNTH_PROMPT,
            include_names=True,
        )
        return await execute_target(spec.synthesizer, synth_messages, ctx.child())
