# stateforward.bot

`stateforward.bot` is a Python software-robot runtime: realtime, stateful,
event-driven bots with deterministic HSM-owned execution, typed events,
provider-injected capabilities, and OpenTelemetry-visible behavior.

Install as `stateforward.bot` and import the library as `bot`, following the
same Stateforward packaging convention as `stateforward.hsm` → `import hsm`.

It began as a way to decompose an existing `voice-bot` runtime into smaller
contracts that could be understood, tested, and reused independently. The
package now treats robots, devices, abilities, behaviors, memory, telemetry, and
provider integrations as explicit boundaries instead of letting realtime
behavior disappear into prompt glue or callback chains.

## Why stateforward.bot Exists

- Decompose voice-bot / software-robot behavior into explicit contracts so the
  runtime can be understood, tested, and evolved without one large application
  carrying every concern.
- Build the deterministic realtime execution model software robots need, rather
  than assuming general bot frameworks were designed for low-latency events,
  interruptions, provider lifecycle, device state, and missing-event handling.
- Apply lessons from bot work into one observable framework where state,
  events, abilities, devices, memory, providers, and proof boundaries are
  visible.

## Design Goals

- Model stateful behavior with `stateforward.hsm` so lifecycle, retries,
  timeouts, cancellation, and completion are explicit state-machine behavior.
- Use typed `hsm.Event` contracts backed by Pydantic schemas for data crossing
  bot, device, ability, provider, and observation boundaries.
- Keep realtime execution deterministic: events enter queues, machines process
  them through run-to-completion steps, and blocking work lives in modeled
  activities that report back through completion or failure events.
- Make observability part of the framework contract. HSM-visible behavior is
  observed through OpenTelemetry metrics and spans, with trace context carried
  through event metadata and low-cardinality attributes.
- Keep the core package provider-agnostic. Provider packages own SDKs and
  transport details while core stateforward.bot owns the bot/device/ability contracts.
- Keep devices bot-agnostic. Devices expose environment-facing affordances and
  events; agents decide whether an interrupt or observation deserves attention.

## Package Map

- `src/bot/bot.py` defines the body machine; `src/bot/abilities`, `behavior`, and
  `skills` define cognitive, ability, behavior, memory, and action contracts.
- `src/bot/address.py` and `src/bot/scope.py` define actor addressing and visibility;
  `src/bot/lifecycle.py` is the single started/stopped predicate.
- `src/bot/device` and `src/bot/devices` define device lifecycle,
  notifications, audio peripherals, and phone behavior.
- `src/bot/environment` defines the broadcast/parenting scope, typed stimuli
  (`events.py`), world snapshots (`snapshot.py`), and spatial placement (`space.py`).
- `src/bot/protocols` defines attachment, transport, and cross-boundary protocols.
- `src/bot/telemetry` records HSM observations as OpenTelemetry metrics and
  spans.
- `src/providers/*` contains optional provider packages: elevenlabs, gemini,
  livekit, mlx_audio, mlx_vlm, moonshine, openai_compat, postgres_memory,
  pyannote, sqlite_memory. Provider packages own SDKs and transport while core
  stays provider-agnostic.
- `examples/` contains runnable examples that stay outside the distributable
  core package.
- `tests/` mirrors `src/bot` one-to-one (see `docs/test_mapping.md` for indirect
  coverage); provider tests stay under `src/providers/<provider>/tests`.

## Quickstart

Clone, sync, run an example, and inspect the trace artifact:

```sh
git clone <repo-url> && cd bot.py
uv sync --locked --all-packages --all-groups
uv run --project examples/listen_speak_bot listen-speak-bot --help
uv run --locked python -m pytest -q -m 'not live'
ls otel-spans.jsonl otel-logs.jsonl
```

## Telemetry and Privacy (OTEL disclosure)

HSM-visible behavior is observed through OpenTelemetry metrics and spans with
low-cardinality attributes only (machine/component, event kind/name, stage,
outcome, normalized failure kind). Raw media, credentials, and
high-cardinality diagnostics never become metric/span attributes.

Local JSONL export writes `otel-spans.jsonl` / `otel-logs.jsonl` (and
`otel-*.jsonl` test dumps) in the repo root with `0600` permissions. These
files are git-ignored. Opt out with `BOT_OTEL_DISABLED=1`.

## Development

Install dependencies:

```sh
uv sync
```

Run tests:

```sh
uv run --group dev python -m pytest
```

Build the package:

```sh
uv build
```

Check release metadata:

```sh
uv run python scripts/check_release_version.py
```

See [docs/release.md](docs/release.md) for versioning and CI publishing notes.

## Provider Packages

Provider integrations live as workspace packages under `src/providers`.
Each provider package owns its SDK dependencies while the core `stateforward.bot`
package (import as `bot`) stays provider-agnostic.

Run the ElevenLabs provider tests:

```sh
uv run --package bot-provider-elevenlabs --group dev python -m pytest src/providers/elevenlabs/tests
```

Run the MLX Audio provider tests:

```sh
uv run --package bot-provider-mlx-audio --group dev python -m pytest src/providers/mlx_audio/tests
```

Run the LiveKit provider tests:

```sh
uv run --package bot-provider-livekit --group dev python -m pytest src/providers/livekit/tests
```

Run the MLX VLM provider tests:

```sh
uv run --package bot-provider-mlx-vlm --group dev python -m pytest src/providers/mlx_vlm/tests
```

Run the OpenAI-compatible provider tests:

```sh
uv run --package bot-provider-openai-compat --group dev python -m pytest src/providers/openai_compat/tests
```

Run the Gemini provider tests:

```sh
uv run --package bot-provider-gemini --group dev python -m pytest src/providers/gemini/tests
```

Run the SQLite memory provider tests:

```sh
uv run --package bot-provider-sqlite-memory --group dev python -m pytest src/providers/sqlite_memory/tests
```

Run the Postgres memory provider tests:

```sh
uv run --package bot-provider-postgres-memory --group dev python -m pytest src/providers/postgres_memory/tests
```

Run the Moonshine provider tests:

```sh
uv run --package bot-provider-moonshine --group dev python -m pytest src/providers/moonshine/tests
```

Run the pyannote provider tests:

```sh
uv run --package bot-provider-pyannote --group dev python -m pytest src/providers/pyannote/tests
```

> Note: `context.md` and `handoff.md` at the repo root are local working notes,
> not framework docs. They are intentionally left in place for the owner to
> decide (keep, move to `~/journal/`, or delete); CI and releases ignore them.
