# Live tests (run against real LLM providers)

These tests exercise PolyPortia end-to-end against **real** model providers —
no mocks. They are intentionally **excluded** from the default `pytest` run.
The unit + integration tests under `tests/unit/` and `tests/integration/`
already cover the same code paths with mocked providers; this directory is
for "does it actually work against a server?" validation.

The default target is **Ollama** (free, local, OpenAI-compatible) so you can
run everything without burning cloud credits. Each test file documents
exactly which model it pulls and any setup steps.

## Setup

1. **Install Ollama**: <https://ollama.com/download>
2. **Start the Ollama server** (usually starts automatically):
   ```bash
   ollama serve  # if not already running
   ```
3. **Pull the models referenced in `polyportia.live.yaml`**:
   ```bash
   ollama pull llama3.2:1b      # the "fast" model
   ollama pull llama3.2:3b      # the "thinking" model
   ollama pull mistral:7b       # the "creative" model — optional, larger
   ```
   You can swap these for any models you have locally — edit
   `polyportia.live.yaml` to match.
4. **Set the env var** so PolyPortia knows where Ollama lives:
   ```bash
   export OLLAMA_BASE_URL=http://localhost:11434
   ```
   (This is also Ollama's default, so the env var is mostly for explicitness.)

## Running

The live tests are gated by the `live` pytest marker and will **skip
automatically** if the env var `RUN_LIVE_TESTS=1` isn't set:

```bash
RUN_LIVE_TESTS=1 pytest tests/live -v
```

Run a single file:

```bash
RUN_LIVE_TESTS=1 pytest tests/live/test_01_single_model.py -v -s
```

`-s` is recommended on first run so you see the model output streaming back —
useful for sanity-checking that Ollama is generating sensible text.

## What each file covers

| File | Validates |
|---|---|
| `test_01_single_model.py` | Basic `complete()` against a defined model with a real provider |
| `test_02_streaming.py` | SSE streaming chunks for a single-model target |
| `test_03_fallback.py` | DefinedModel falls over when its primary target is broken |
| `test_04_retry.py` | Retry loop fires (or doesn't) per the configured policy |
| `test_05_parallel_council.py` | `parallel_array` returns one result per member |
| `test_06_synthesize_council.py` | `synthesize` produces a single combined response |
| `test_07_meta_council.py` | Council-of-councils (recursive synthesizer) |
| `test_08_traces.py` | Trace records contain spans, retries, costs |
| `test_09_cli.py` | `polyportia` CLI commands work end-to-end |
| `test_10_debate_council.py` | (M3) Debate strategy across 3 visibility modes |
| `test_11_propose_review.py` | (M3) Propose-review with tool-call verdicts |

## Notes

- Most cost/usage assertions are loose because Ollama doesn't always report
  token usage for every model, and pricing is zero locally. The tests assert
  *structure*, not exact values.
- If a test fails because Ollama is slow on first generation (model
  warmup), re-run after the model is loaded into memory.
- These tests are **safe to commit**: nothing here calls a paid API by
  default, and there are no API keys in the YAML.
