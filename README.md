# PolyPortia

A model-agnostic LLM gateway in the spirit of LiteLLM, with a first-class
**model council** layer for parallel synthesis, multi-turn debate, and
proposer/reviewer consensus.

PolyPortia exposes three surfaces over the same executor:

- **HTTP server** — drop-in OpenAI-compatible `/v1/chat/completions`
- **Python SDK** — `polyportia.complete(...)` / `polyportia.acomplete(...)`
- **CLI** — `polyportia serve`, `polyportia run`, `polyportia models show ...`

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
)
print(result.content)
```

## Status

- **M1 (current)** — three-layer config, single-model passthrough, transitive
  fallbacks, retries with exponential backoff + jitter, cascading
  retry/timeout policy, OpenAI-compatible HTTP, SDK, CLI, in-memory traces.
- **M2 (next)** — `parallel_array` and `synthesize` councils, `/v1/councils/{name}/run`.
- **M3** — `debate` council (with three visibility modes) and
  `propose_review` council (proposer + reviewers using tool-call verdicts).
- **M4** — Trace UX polish, file sink, optional response caching.

## LiteLLM relationship

PolyPortia depends on `litellm` for provider HTTP calls, token counting, and
cost tables. We do **not** fork it. The orchestration layer (councils,
fallbacks, traces) is pure PolyPortia.

## Development

```bash
python -m pytest          # 33 tests
python -m ruff check src/
python -m mypy src/polyportia
```
