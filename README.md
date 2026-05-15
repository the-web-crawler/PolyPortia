# PolyPortia

A model-agnostic LLM gateway in the spirit of LiteLLM, with a first-class
**model council** layer for parallel synthesis, multi-turn debate, and
proposer/reviewer consensus — plus budget-aware cost estimation and
mid-execution enforcement.

PolyPortia exposes three surfaces over the same executor:

- **HTTP server** — drop-in OpenAI-compatible `/v1/chat/completions` (plus
  `/v1/councils/{name}/run` for rich array envelopes and `/v1/traces/{id}`
  for trace inspection).
- **Python SDK** — `polyportia.complete(...)` / `polyportia.acomplete(...)` /
  `polyportia.run_council(...)`.
- **CLI** — `polyportia serve`, `polyportia run`, `polyportia models show ...`,
  `polyportia councils show ...`, `polyportia testing dummy` for the local
  dummy completions endpoint.

## Three-layer model abstraction

| Layer | Identity | Role |
|---|---|---|
| `Provider` | `name` (e.g. `anthropic`) | API endpoint, auth, retry/timeout defaults |
| `ActualModel` | `id` (litellm format, e.g. `anthropic/claude-opus-4-7`) | Canonical metadata: context window, $/token, supported features. Declared once |
| `DefinedModel` | `name` (e.g. `thinking`) | User-named handle pointing at an `ActualModel` (or another `DefinedModel`); carries param overrides + a transitive **fallback chain** |

`Council` is a fourth, orthogonal concept that orchestrates members under one
of four strategies: `parallel_array`, `synthesize`, `debate`, `propose_review`.
Members are themselves `ResolvableTarget`s — recursion is uniform, so a
synthesizer can be another council, and a council member can be another
council.

## Quick start

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
polyportia serve --config polyportia.example.yaml
```

```bash
curl localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"thinking","messages":[{"role":"user","content":"hi"}]}'
```

```python
import polyportia

polyportia.load_config("polyportia.example.yaml")
result, trace_id = polyportia.complete(
    model="thinking",
    messages=[{"role": "user", "content": "Pros/cons of CRDTs?"}],
    budget_usd=0.50,
)
print(result.content)
```

## Council strategies

- **`parallel_array`** — every member answers the same prompt; the array
  envelope contains all responses.
- **`synthesize`** — members fan out in parallel; a synthesizer target (which
  can itself be a council) combines the answers into a single response.
- **`debate`** — N rounds where every member sees its peers' prior responses
  according to one of three visibility modes (`full_history`,
  `prompt_and_peer_responses`, `own_only_with_target`). Optional consensus or
  judge-driven early termination. Final output is either an array or a
  synthesizer pass.
- **`propose_review`** — proposer's `tool_calls` are intercepted, reviewer
  panel votes by calling synthetic `approve` / `deny` / `insight` tools.
  Combine verdicts with `all`, `any`, `majority`, or an integer threshold.
  On non-approval the combined feedback is fed back to the proposer via
  `role: tool` messages and it revises. On exhausted revisions the
  `on_denial` policy (`return_denial` / `revise` / `fail`) decides.

## Budgets

`polyportia.budget_usd` in any request bounds the call by a dollar amount.
Two checks run:

1. **Pre-flight**: PolyPortia walks the resolved target tree, summing
   worst-case `(input_tokens × in_rate + max_tokens × out_rate)` for every
   call that could happen — including every fan-out member, every debate
   turn, every proposer/reviewer revision round. Over budget → HTTP 402 with
   `stage: "pre_flight"` and a per-model breakdown.
2. **Mid-execution**: after every real provider call, running cost is
   compared against budget. Over budget → HTTP 402 with `stage:
   "mid_execution"` and `actual_usd_so_far` populated.

`polyportia.budget_usd: "unlimited"` disables the check.
`budget_usd_default` in `polyportia.yaml` sets a config-level fallback.

Every successful response carries `X-PolyPortia-Cost-USD` and
`X-PolyPortia-Cost-Predicted-USD` headers. Pass `polyportia.include_cost: true`
to nest a `polyportia.cost: {actual_usd, predicted_usd, by_model: [...]}`
object inside the body.

## Retries, timeouts, and fallbacks

Retry policy and timeout cascade by precedence:

```
request override > DefinedModel > ActualModel > Provider > built-in default
```

`RetryPolicy` controls `max_retries`, retryable categories
(`timeout`/`rate_limit`/`server_error`/`connection`), exponential or linear
backoff with jitter, and a backoff cap.

DefinedModels carry a `fallbacks: list[ModelTarget]` chain. After retries on
the primary are exhausted, PolyPortia tries each fallback in order. If a
fallback is itself a DefinedModel, **its** chain is followed transitively;
cycles are detected per-request and skipped.

## Observability

Every request produces a `TraceRecord` with one `TraceSpan` per layer
(defined, actual, parallel_array, synthesize, debate, debate_turn,
propose_review, propose_review_round). Each leaf span carries token usage,
cost, latency, every retry attempt, the fallback chain that was walked, and
the effective retry/timeout source. Traces live in an in-memory ring buffer
(default 1000) plus an optional JSONL file sink.

- `GET /v1/traces` — recent ring contents.
- `GET /v1/traces/{trace_id}` — full record.
- `X-PolyPortia-Trace-ID` header on every response.

## Dummy completions endpoint

For end-to-end verification without burning provider credits:

```bash
polyportia testing dummy --port 9999
```

The dummy speaks OpenAI's chat-completions protocol. Behaviour is driven by
model-id patterns (`dummy/echo`, `dummy/fixed/<text>`, `dummy/error/<status>`,
`dummy/tool/<name>`, `dummy/delay/<ms>`, `dummy/usage/<pt>/<ct>`,
`dummy/stream/<text>`) plus admin endpoints (`POST /admin/register`,
`POST /admin/reset`, `GET /admin/calls`) for runtime configuration. Set a
PolyPortia provider's `api_base: http://localhost:9999` to route requests
through it.

## LiteLLM relationship

PolyPortia depends on `litellm` for provider HTTP calls, token counting, and
cost-table fallbacks. We do **not** fork it. The orchestration layer
(councils, fallbacks, traces, budgets) is pure PolyPortia. Any
LiteLLM-recognised `provider/model` identifier passed in `model` is
forwarded through (subject to budget); registered ActualModels add canonical
metadata for accurate cost estimation.

## Development

```bash
python -m pytest          # 138 tests
python -m ruff check src/ tests/
python -m mypy src/polyportia
```

End-to-end tests against real Ollama models live under `tests/live/` and are
excluded from the default run; see `tests/live/README.md`.
