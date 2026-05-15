"""Pre-flight cost estimator across every council strategy."""

from __future__ import annotations

import pytest

from polyportia.budget.estimator import REVIEWER_OUTPUT_CAP, estimate_cost
from polyportia.config.loader import load_config_from_string
from polyportia.config.models import (
    ActualModelRef,
    Debate,
    DebateConfig,
    DefinedModelRef,
    ParallelArray,
    ProposeAndReview,
    Synthesize,
)
from polyportia.config.registry import Registry

_YAML = """
providers:
  - {name: anthropic, api_key: x}
  - {name: openai, api_key: y}
actual_models:
  - id: cheap/A
    provider: anthropic
    max_output_tokens: 100
    input_cost_per_1m_tokens: 1.0
    output_cost_per_1m_tokens: 2.0
  - id: pricey/B
    provider: openai
    max_output_tokens: 200
    input_cost_per_1m_tokens: 10.0
    output_cost_per_1m_tokens: 20.0
  - id: free/C
    provider: anthropic
    max_output_tokens: 50
defined_models:
  - name: alias-a
    target: {kind: actual, id: cheap/A}
  - name: alias-b-ovr
    target: {kind: actual, id: pricey/B}
    params: {max_tokens: 50}
"""


@pytest.fixture
def reg() -> Registry:
    return Registry(load_config_from_string(_YAML))


def _msgs() -> list[dict]:
    return [{"role": "user", "content": "hello"}]


def test_single_actual_model(reg: Registry) -> None:
    est = estimate_cost(
        ActualModelRef(id="cheap/A"),
        _msgs(),
        {},
        reg,
        initial_input_tokens=100,
    )
    # input 100 * $1/M + output 100 * $2/M = $0.0001 + $0.0002 = $0.0003
    assert est.total_usd == pytest.approx(0.0003, rel=1e-3)
    assert len(est.breakdown) == 1
    assert est.breakdown[0].calls == 1


def test_unset_rates_recorded_as_zero_with_note(reg: Registry) -> None:
    est = estimate_cost(
        ActualModelRef(id="free/C"),
        _msgs(),
        {},
        reg,
        initial_input_tokens=100,
    )
    assert est.total_usd == 0.0
    assert any("free/C" in n for n in est.notes)


def test_defined_model_params_override_max_tokens(reg: Registry) -> None:
    # pricey/B with defined override max_tokens=50 vs unrestricted=200.
    est_restricted = estimate_cost(
        DefinedModelRef(name="alias-b-ovr"),
        _msgs(),
        {},
        reg,
        initial_input_tokens=100,
    )
    est_full = estimate_cost(
        ActualModelRef(id="pricey/B"),
        _msgs(),
        {},
        reg,
        initial_input_tokens=100,
    )
    assert est_restricted.total_usd < est_full.total_usd


def test_request_max_tokens_overrides_defined_and_actual(reg: Registry) -> None:
    est = estimate_cost(
        ActualModelRef(id="pricey/B"),
        _msgs(),
        {"max_tokens": 10},
        reg,
        initial_input_tokens=100,
    )
    # output cost = 10 * $20/M = $0.0002; input cost = 100 * $10/M = $0.001
    assert est.total_usd == pytest.approx(0.0012, rel=1e-3)


def test_parallel_array_sums_members(reg: Registry) -> None:
    spec = ParallelArray(
        members=[ActualModelRef(id="cheap/A"), ActualModelRef(id="pricey/B")]
    )
    est = estimate_cost(spec, _msgs(), {}, reg, initial_input_tokens=100)
    # cheap/A: 100*1 + 100*2 = $0.0003
    # pricey/B: 100*10 + 200*20 = $5e-3
    assert est.total_usd == pytest.approx(0.0003 + 0.005, rel=1e-3)


def test_synthesize_includes_grown_input_for_synthesizer(reg: Registry) -> None:
    spec = Synthesize(
        members=[ActualModelRef(id="cheap/A"), ActualModelRef(id="cheap/A")],
        synthesizer=ActualModelRef(id="pricey/B"),
    )
    est = estimate_cost(spec, _msgs(), {}, reg, initial_input_tokens=100)
    # members: 2 × (100*1 + 100*2)/1M = $0.0006
    # synth input = 100 + 100 + 100 = 300; output 200 -> 300*10 + 200*20 = 7e-3
    assert est.total_usd == pytest.approx(0.0006 + 0.007, rel=1e-3)
    # Synth model billed once
    by = {e.model_id: e for e in est.breakdown}
    assert by["pricey/B"].calls == 1
    assert by["cheap/A"].calls == 2


def test_debate_grows_input_across_turns(reg: Registry) -> None:
    spec = Debate(
        members=[ActualModelRef(id="cheap/A"), ActualModelRef(id="cheap/A")],
        debate=DebateConfig(turns=3),
        output="array",
    )
    est = estimate_cost(spec, _msgs(), {}, reg, initial_input_tokens=100)
    # turn 0: 2 calls each 100 in + 100 out = $0.0003 * 2 = $0.0006
    # turn 1: each sees 100 + 200 (prior turn) = 300 in + 100 out -> 300*1 + 100*2 / 1M = 5e-4 each * 2 = $0.001
    # turn 2: each sees 100 + 200 + 200 = 500 in + 100 out -> 500*1 + 100*2 = 7e-4 each * 2 = $0.0014
    # total cheap/A: $0.0006 + $0.001 + $0.0014 = $0.003
    assert est.total_usd == pytest.approx(0.003, rel=1e-3)


def test_debate_synthesize_adds_synthesizer(reg: Registry) -> None:
    spec = Debate(
        members=[ActualModelRef(id="cheap/A")],
        debate=DebateConfig(turns=2),
        output="synthesize",
        synthesizer=ActualModelRef(id="cheap/A"),
    )
    est = estimate_cost(spec, _msgs(), {}, reg, initial_input_tokens=100)
    # turn 0: 1 call 100in+100out = $0.0003
    # turn 1: 1 call 200in+100out = (200*1 + 100*2)/1M = $0.0004
    # synth input = 100 + (100+100) = 300, output 100; cost = (300*1 + 100*2)/1M = $0.0005
    # total = $0.0012, 3 calls on cheap/A
    assert est.total_usd == pytest.approx(0.0012, rel=1e-3)
    assert est.breakdown[0].calls == 3


def test_propose_review_worst_case_includes_revisions(reg: Registry) -> None:
    spec = ProposeAndReview(
        proposer=ActualModelRef(id="cheap/A"),
        reviewers=[ActualModelRef(id="cheap/A"), ActualModelRef(id="cheap/A")],
        consensus="all",
        max_revisions=2,  # so 3 rounds worst case
    )
    est = estimate_cost(spec, _msgs(), {}, reg, initial_input_tokens=100)
    # Worst case: 3 rounds × (1 proposer + 2 reviewers) = 9 calls on cheap/A.
    by = {e.model_id: e for e in est.breakdown}
    assert by["cheap/A"].calls == 9


def test_propose_review_reviewers_capped_at_reviewer_output_cap(reg: Registry) -> None:
    # If REVIEWER_OUTPUT_CAP is less than the model's max_output_tokens, the
    # estimator should use the smaller value for reviewer output.
    spec = ProposeAndReview(
        proposer=ActualModelRef(id="cheap/A"),
        reviewers=[ActualModelRef(id="pricey/B")],
        consensus="all",
        max_revisions=0,
    )
    est = estimate_cost(spec, _msgs(), {}, reg, initial_input_tokens=100)
    # Reviewer output is min(200, REVIEWER_OUTPUT_CAP) = 200 (cap and B's max coincide)
    # but reviewer input = 100 (initial) + 100 (proposer out worst case) = 200.
    # pricey/B output = REVIEWER_OUTPUT_CAP = 200 ⇒ (200*10 + 200*20)/1M = $0.006
    by = {e.model_id: e for e in est.breakdown}
    assert by["pricey/B"].output_tokens_est == REVIEWER_OUTPUT_CAP


def test_recursive_synthesize_council(reg: Registry) -> None:
    inner = Synthesize(
        members=[ActualModelRef(id="cheap/A"), ActualModelRef(id="cheap/A")],
        synthesizer=ActualModelRef(id="cheap/A"),
    )
    outer = Synthesize(
        members=[inner, ActualModelRef(id="cheap/A")],
        synthesizer=ActualModelRef(id="cheap/A"),
    )
    est = estimate_cost(outer, _msgs(), {}, reg, initial_input_tokens=100)
    # Inner: 2 member calls + 1 synth call = 3 cheap/A calls
    # Outer adds: 1 extra member + 1 outer synth = 2 cheap/A calls
    by = {e.model_id: e for e in est.breakdown}
    assert by["cheap/A"].calls == 5
