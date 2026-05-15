"""Pre-flight worst-case cost estimator.

Walks the resolved target tree (single, parallel, synthesize, debate,
propose_review, defined model + fallback chain) and computes the maximum
plausible cost for the entire call graph. Conventions:

- ``max_tokens`` is treated as the actual output token count for each call
  (a hard upper bound the provider will respect).
- The initial input token count is computed from the messages via
  ``litellm.token_counter``; downstream calls grow this by the prior outputs
  they will see (synthesizer sees fan-out outputs, debate turn N sees prior
  turns' outputs, proposer's nth revision sees prior reviewer feedback).
- DefinedModel ``fallbacks`` are NOT walked at estimate time — they fire only
  when the primary fails. Mid-execution enforcement catches an over-budget
  fallback excursion.
- Retries are NOT included in the estimate; they're caught mid-execution.
- Reviewer outputs in propose_review are capped at ``REVIEWER_OUTPUT_CAP``
  tokens (tool calls are short).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polyportia.budget.errors import CostBreakdownEntry, CostEstimate
from polyportia.config.models import (
    ActualModel,
    Debate,
    DefinedModel,
    ParallelArray,
    ProposeAndReview,
    ResolvableTarget,
    Synthesize,
)
from polyportia.config.registry import Registry
from polyportia.config.resolver import resolve

REVIEWER_OUTPUT_CAP = 200
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def _count_tokens(messages: list[dict[str, Any]], model_hint: str | None) -> int:
    """Approximate input token count. ``litellm`` is used when available, else
    a coarse character-based fallback (1 token ≈ 4 chars).
    """
    try:
        import litellm

        m = model_hint or "gpt-4"
        return int(litellm.token_counter(model=m, messages=messages))
    except Exception:
        total = 0
        for msg in messages:
            for v in msg.values():
                if isinstance(v, str):
                    total += max(1, len(v) // 4)
        return total or 1


@dataclass
class _Acc:
    by_model: dict[str, CostBreakdownEntry]
    notes: list[str]

    def add(self, e: CostBreakdownEntry) -> None:
        existing = self.by_model.get(e.model_id)
        if existing is None:
            self.by_model[e.model_id] = e
        else:
            existing.calls += e.calls
            existing.input_tokens_est += e.input_tokens_est
            existing.output_tokens_est += e.output_tokens_est
            existing.cost_usd += e.cost_usd


def _max_output_for(
    actual: ActualModel,
    defined: DefinedModel | None,
    params: dict[str, Any],
) -> int:
    if "max_tokens" in params:
        return int(params["max_tokens"])
    if defined is not None and "max_tokens" in defined.params:
        return int(defined.params["max_tokens"])
    if "max_tokens" in actual.default_params:
        return int(actual.default_params["max_tokens"])
    if actual.max_output_tokens is not None:
        return int(actual.max_output_tokens)
    return DEFAULT_MAX_OUTPUT_TOKENS


def _estimate_call(
    actual: ActualModel,
    defined: DefinedModel | None,
    input_tokens: int,
    params: dict[str, Any],
    acc: _Acc,
    *,
    output_cap: int | None = None,
) -> int:
    out_tokens = _max_output_for(actual, defined, params)
    if output_cap is not None:
        out_tokens = min(out_tokens, output_cap)
    in_rate = actual.input_cost_per_1m_tokens
    out_rate = actual.output_cost_per_1m_tokens
    if in_rate is None or out_rate is None:
        acc.notes.append(
            f"{actual.id}: input/output rates unset; treating cost contribution as 0"
        )
        cost = 0.0
    else:
        cost = (input_tokens * in_rate + out_tokens * out_rate) / 1_000_000
    acc.add(
        CostBreakdownEntry(
            model_id=actual.id,
            calls=1,
            input_tokens_est=input_tokens,
            output_tokens_est=out_tokens,
            cost_usd=cost,
        )
    )
    return out_tokens


def _estimate_target(
    target: ResolvableTarget,
    input_tokens: int,
    params: dict[str, Any],
    registry: Registry,
    acc: _Acc,
    *,
    defined: DefinedModel | None = None,
    output_cap: int | None = None,
    visited_defined: frozenset[str] = frozenset(),
    visited_council: frozenset[str] = frozenset(),
) -> int:
    """Estimate cost for one resolvable position. Returns expected output tokens."""
    try:
        resolved = resolve(target, registry)
    except Exception as e:
        acc.notes.append(f"could not resolve target {target}: {e!s}")
        return 0

    if isinstance(resolved, ActualModel):
        return _estimate_call(
            resolved, defined, input_tokens, params, acc, output_cap=output_cap
        )

    if isinstance(resolved, DefinedModel):
        if resolved.name in visited_defined:
            acc.notes.append(f"defined cycle on '{resolved.name}'; stopping descent")
            return 0
        merged_params = {**resolved.params, **params}
        return _estimate_target(
            resolved.target,
            input_tokens,
            merged_params,
            registry,
            acc,
            defined=resolved,
            output_cap=output_cap,
            visited_defined=visited_defined | {resolved.name},
            visited_council=visited_council,
        )

    if isinstance(resolved, ParallelArray):
        max_out = 0
        for member in resolved.members:
            out = _estimate_target(
                member, input_tokens, params, registry, acc,
                visited_defined=visited_defined,
                visited_council=visited_council,
            )
            max_out = max(max_out, out)
        return max_out

    if isinstance(resolved, Synthesize):
        member_outs: list[int] = []
        for member in resolved.members:
            member_outs.append(
                _estimate_target(
                    member, input_tokens, params, registry, acc,
                    visited_defined=visited_defined,
                    visited_council=visited_council,
                )
            )
        synth_input = input_tokens + sum(member_outs)
        return _estimate_target(
            resolved.synthesizer, synth_input, params, registry, acc,
            visited_defined=visited_defined,
            visited_council=visited_council,
        )

    if isinstance(resolved, Debate):
        accumulated_output_tokens = 0
        last_turn_outs: list[int] = []
        for _t in range(resolved.debate.turns):
            last_turn_outs = []
            for member in resolved.members:
                turn_input = input_tokens + accumulated_output_tokens
                last_turn_outs.append(
                    _estimate_target(
                        member, turn_input, params, registry, acc,
                        visited_defined=visited_defined,
                        visited_council=visited_council,
                    )
                )
            accumulated_output_tokens += sum(last_turn_outs)
        if resolved.output == "synthesize" and resolved.synthesizer is not None:
            synth_input = input_tokens + accumulated_output_tokens
            return _estimate_target(
                resolved.synthesizer, synth_input, params, registry, acc,
                visited_defined=visited_defined,
                visited_council=visited_council,
            )
        return max(last_turn_outs) if last_turn_outs else 0

    if isinstance(resolved, ProposeAndReview):
        rounds = resolved.max_revisions + 1
        proposer_accumulated = 0  # input growth for proposer across rounds
        last_proposer_out = 0
        for _r in range(rounds):
            proposer_input = input_tokens + proposer_accumulated
            last_proposer_out = _estimate_target(
                resolved.proposer, proposer_input, params, registry, acc,
                visited_defined=visited_defined,
                visited_council=visited_council,
            )
            reviewer_outs_sum = 0
            for reviewer in resolved.reviewers:
                reviewer_input = input_tokens + last_proposer_out
                out = _estimate_target(
                    reviewer, reviewer_input, params, registry, acc,
                    output_cap=REVIEWER_OUTPUT_CAP,
                    visited_defined=visited_defined,
                    visited_council=visited_council,
                )
                reviewer_outs_sum += out
            proposer_accumulated += last_proposer_out + reviewer_outs_sum
        return last_proposer_out

    acc.notes.append(f"unhandled target type {type(resolved).__name__}; cost=0")
    return 0


def estimate_cost(
    target: ResolvableTarget,
    messages: list[dict[str, Any]],
    request_params: dict[str, Any],
    registry: Registry,
    *,
    initial_input_tokens: int | None = None,
    token_model_hint: str | None = None,
) -> CostEstimate:
    acc = _Acc(by_model={}, notes=[])
    in_tokens = (
        initial_input_tokens
        if initial_input_tokens is not None
        else _count_tokens(messages, token_model_hint)
    )
    _estimate_target(target, in_tokens, request_params, registry, acc)
    total = sum(e.cost_usd for e in acc.by_model.values())
    return CostEstimate(
        total_usd=total,
        breakdown=sorted(acc.by_model.values(), key=lambda e: e.model_id),
        notes=acc.notes,
    )
