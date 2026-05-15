"""Propose-and-Review council strategy.

The proposer model is run with the caller's tools. If its response contains
``tool_calls``, PolyPortia intercepts them and asks the reviewer panel to vote
by calling one of three synthetic tools — ``approve``, ``deny``, or
``insight``. Verdicts are combined per consensus rule. On non-approval the
combined feedback is fed back to the proposer as ``role: tool`` results and
the proposer is asked to revise. Loops up to ``max_revisions``; on exhaustion
the ``on_denial`` policy decides the final response.

This module never executes the caller's tools — it just gates them.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from polyportia.config.models import (
    ActualModelRef,
    CouncilRef,
    DefinedModelRef,
    ProposeAndReview,
    ResolvableTarget,
)
from polyportia.council.context import ExecutionContext
from polyportia.providers.base import ProviderResult

_REVIEWER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "approve",
            "description": "Approve the proposed tool call. Optional rationale.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deny",
            "description": "Deny the proposed tool call. Provide a reason.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insight",
            "description": (
                "Decline to approve and provide insight or feedback to the "
                "proposer. The proposer will see your text and may revise."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]

_DEFAULT_REVIEWER_PROMPT = """\
Another model has proposed a tool call (shown in the assistant message above) \
to fulfill the original request. You are a reviewer. Inspect the proposed \
tool call carefully. Decide whether to:
  - approve: the proposal is correct and safe to execute,
  - deny: the proposal is incorrect or unsafe; explain why,
  - insight: the proposal needs revision; explain what.

You MUST respond by calling exactly one of the tools: approve, deny, insight.
"""


def _short_name(target: ResolvableTarget) -> str:
    if isinstance(target, ActualModelRef):
        return target.id
    if isinstance(target, DefinedModelRef):
        return f"defined:{target.name}"
    if isinstance(target, CouncilRef):
        return f"council:{target.name}"
    return type(target).__name__


def _extract_tool_calls(raw: Any) -> list[dict[str, Any]] | None:
    try:
        choices = raw.choices if hasattr(raw, "choices") else raw["choices"]
        choice = choices[0]
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")
    if msg is None:
        return None
    tcs = getattr(msg, "tool_calls", None)
    if tcs is None and isinstance(msg, dict):
        tcs = msg.get("tool_calls")
    if not tcs:
        return None
    out: list[dict[str, Any]] = []
    for tc in tcs:
        if isinstance(tc, dict):
            out.append(tc)
            continue
        fn = getattr(tc, "function", None)
        out.append(
            {
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", "function"),
                "function": (
                    {
                        "name": getattr(fn, "name", None),
                        "arguments": getattr(fn, "arguments", None),
                    }
                    if fn is not None
                    else None
                ),
            }
        )
    return out


def _parse_verdict(tool_calls: list[dict[str, Any]] | None) -> tuple[str, str]:
    if not tool_calls:
        return "none", ""
    first = tool_calls[0]
    fn = first.get("function") or {}
    name = fn.get("name") or ""
    args_raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
    except json.JSONDecodeError:
        args = {}
    if name == "approve":
        return "approve", args.get("reason", "") or ""
    if name == "deny":
        return "deny", args.get("reason", "") or ""
    if name == "insight":
        return "insight", args.get("text", "") or ""
    return "none", ""


def _check_consensus(
    verdicts: list[str],
    consensus: str | int,
    total: int,
    *,
    insight_counts_as_approval: bool,
) -> bool:
    approving = {"approve"}
    if insight_counts_as_approval:
        approving.add("insight")
    approvals = sum(1 for v in verdicts if v in approving)
    if isinstance(consensus, int):
        return approvals >= consensus
    if consensus == "all":
        return approvals == total
    if consensus == "any":
        return approvals >= 1
    if consensus == "majority":
        return approvals * 2 > total
    return False


def _build_tool_results(
    proposer_tool_calls: list[dict[str, Any]],
    reviewer_verdicts: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    lines: list[str] = []
    for member, verdict, rationale in reviewer_verdicts:
        if verdict == "approve":
            tail = f" — {rationale}" if rationale else ""
            lines.append(f"- {member}: APPROVE{tail}")
        elif verdict == "deny":
            lines.append(f"- {member}: DENY — {rationale or '(no reason)'}")
        elif verdict == "insight":
            lines.append(f"- {member}: INSIGHT — {rationale}")
        else:
            lines.append(f"- {member}: (no verdict cast)")
    body = (
        "Your tool call was NOT approved by the reviewer panel. Revise and try "
        "again. Reviewer feedback:\n" + "\n".join(lines)
    )
    return [
        {"role": "tool", "tool_call_id": tc.get("id") or "", "content": body}
        for tc in proposer_tool_calls
    ]


def _denial_text(
    proposal: list[dict[str, Any]],
    last_verdicts: list[tuple[str, str, str]],
) -> str:
    lines = ["The proposed tool call was not approved by the reviewer panel."]
    lines.append("\nProposal:")
    for tc in proposal:
        fn = tc.get("function") or {}
        lines.append(f"  - {fn.get('name')}({fn.get('arguments')})")
    lines.append("\nReviewer verdicts:")
    for member, verdict, rationale in last_verdicts:
        if verdict == "approve":
            tail = f" — {rationale}" if rationale else ""
            lines.append(f"  - {member}: APPROVE{tail}")
        elif verdict == "deny":
            lines.append(f"  - {member}: DENY — {rationale or '(no reason)'}")
        elif verdict == "insight":
            lines.append(f"  - {member}: INSIGHT — {rationale}")
        else:
            lines.append(f"  - {member}: (no verdict cast)")
    return "\n".join(lines)


async def _run_reviewers(
    spec: ProposeAndReview,
    reviewer_messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> list[tuple[str, str, str]]:
    from polyportia.council.executor import execute_target

    reviewer_ctx = ctx.child()
    reviewer_ctx.request_params = {
        **reviewer_ctx.request_params,
        "tools": _REVIEWER_TOOLS,
        "tool_choice": "required",
    }

    async def run_one(r: ResolvableTarget) -> tuple[str, str, str]:
        member_repr = _short_name(r)
        try:
            result = await execute_target(r, reviewer_messages, reviewer_ctx.child())
        except BaseException as e:
            return member_repr, "none", f"reviewer error: {e!s}"
        verdict, rationale = _parse_verdict(_extract_tool_calls(result.raw))
        return member_repr, verdict, rationale

    return await asyncio.gather(*(run_one(r) for r in spec.reviewers))


async def run_propose_review(
    spec: ProposeAndReview,
    base_messages: list[dict[str, Any]],
    ctx: ExecutionContext,
) -> ProviderResult:
    from polyportia.council.executor import execute_target

    messages: list[dict[str, Any]] = list(base_messages)
    last_proposal_result: ProviderResult | None = None
    last_proposal_tool_calls: list[dict[str, Any]] | None = None
    last_verdicts: list[tuple[str, str, str]] = []

    with ctx.trace.span(kind="propose_review", target_repr="propose_review") as outer:
        for attempt in range(spec.max_revisions + 1):
            with ctx.trace.span(
                kind="propose_review_round",
                target_repr=f"round:{attempt}",
            ):
                proposer_result = await execute_target(spec.proposer, messages, ctx.child())
                last_proposal_result = proposer_result
                tool_calls = _extract_tool_calls(proposer_result.raw)

                if not tool_calls:
                    return proposer_result

                last_proposal_tool_calls = tool_calls
                reviewer_messages = list(base_messages) + [
                    {"role": "assistant", "content": None, "tool_calls": tool_calls},
                    {
                        "role": "user",
                        "content": spec.review_prompt or _DEFAULT_REVIEWER_PROMPT,
                    },
                ]
                verdicts = await _run_reviewers(spec, reviewer_messages, ctx)
                last_verdicts = verdicts

                simple = [v[1] for v in verdicts]
                if _check_consensus(
                    simple,
                    spec.consensus,
                    len(verdicts),
                    insight_counts_as_approval=spec.insight_counts_as_approval,
                ):
                    if spec.output == "envelope":
                        envelope = {
                            "object": "polyportia.propose_review",
                            "approved": True,
                            "revisions_used": attempt,
                            "proposal": tool_calls,
                            "reviews": [
                                {"reviewer": m, "verdict": v, "rationale": r}
                                for m, v, r in verdicts
                            ],
                        }
                        return ProviderResult(
                            model_id="propose_review",
                            content=proposer_result.content,
                            raw=envelope,
                            finish_reason="stop",
                        )
                    return proposer_result

                if attempt < spec.max_revisions:
                    messages = list(messages) + [
                        {"role": "assistant", "content": None, "tool_calls": tool_calls},
                        *_build_tool_results(tool_calls, verdicts),
                    ]

        outer.status = "fellback"

        if spec.on_denial == "fail":
            raise RuntimeError(
                "propose_review: consensus not reached after "
                f"{spec.max_revisions} revision(s); on_denial='fail'"
            )
        if spec.on_denial == "revise":
            if last_proposal_result is None:
                raise RuntimeError("propose_review: no proposal produced")
            return last_proposal_result

        denial = _denial_text(last_proposal_tool_calls or [], last_verdicts)
        if spec.output == "envelope":
            return ProviderResult(
                model_id="propose_review",
                content=denial,
                raw={
                    "object": "polyportia.propose_review",
                    "approved": False,
                    "revisions_used": spec.max_revisions,
                    "proposal": last_proposal_tool_calls,
                    "reviews": [
                        {"reviewer": m, "verdict": v, "rationale": r}
                        for m, v, r in last_verdicts
                    ],
                    "denial_text": denial,
                },
                finish_reason="stop",
            )
        return ProviderResult(
            model_id="propose_review",
            content=denial,
            raw={
                "object": "chat.completion",
                "model": "propose_review",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": denial},
                        "finish_reason": "stop",
                    }
                ],
            },
            finish_reason="stop",
        )
