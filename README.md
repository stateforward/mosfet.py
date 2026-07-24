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

- `src/bot/bot.py` defines the body machine; `src/bot/abilities`,  `behavior`, and
  `skills` define cognitive, ability, behavior, memory, and action contracts.
- `src/bot/device` and `src/bot/devices` define device lifecycle,
  notifications, audio peripherals, and phone behavior.
- `src/bot/telemetry` records HSM observations as OpenTelemetry metrics and
  spans.
- `src/providers/*` contains optional provider packages for SDK-backed audio,
  vision, memory, LiveKit, Gemini, and OpenAI-compatible integrations.
- `examples/` contains runnable examples that stay outside the distributable
  core package.

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
