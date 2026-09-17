# Development

Operational reference for contributors working on this repository. The root README is
the product document; everything that used to live there for programmers is consolidated
here.

Distribution name: `stateforward.bot`, imported as `bot`. Deterministic execution runs
on [`stateforward-hsm`](https://pypi.org/project/stateforward-hsm/) (`import hsm`).

## Composing a bot in code (today's composition surface)

```python
import bot
import hsm
from bot import Bot
from bot.abilities import listening, memory, speaking     # anatomy
from bot.devices import phone                              # peripherals
from bot.environment import Environment, SoundEvent, space # physics
from bot.providers.openai_compat import ChatClient         # mind transport
```

…then attach provider components, acquire abilities, and start the bot. This is the
lowest comfort layer and the seams the future self-assembling bot will exercise on
its own behalf.

## Experiences (examples)

```sh
uv run --project examples/listen_speak_bot listen-speak-bot
uv run --project examples/phone_bot phone-bot --json
uv run --project examples/phone_bot python examples/phone_bot/learning_phone_bot.py
```

Each example is its own `uv` package so example-only dependencies never leak into core.
Provider credentials come from ordinary environment variables read by the provider
packages (`BOT_MERCURY_API_KEY`, `BOT_GEMINI_API_KEY` / `GEMINI_API_KEY`,
`BOT_OPENAI_API_KEY`, …).

## Toolchain

```sh
uv sync                          # core + workspace providers
uv run --locked python -m pytest -q -m 'not live'   # full suite, no live providers
uv run --group dev ruff check --fix src tests
uv run --group dev basedpyright  # strict: private usage is an error, not a lint
uv build                         # wheels/CLI release builds (see dist/)
uv run python scripts/check_release_version.py
```

Scoped provider tests: `uv run --package bot-provider-<p> -m pytest src/providers/<p>/tests`.

See [release.md](release.md) for versioning and CI publishing notes.

## Observability

Every HSM-visible behavior is observed at runtime boundaries that opt in
(`hsm.observe`) and exported as OpenTelemetry metrics and spans with **low-cardinality
attributes only** (component, event kind/name, stage, outcome, normalized failure kind).
Raw media, credentials, and high-cardinality identifiers never become telemetry
attributes. Trace context rides `hsm.Event.metadata` end-to-end.

Local JSONL export writes `otel-spans.jsonl` / `otel-logs.jsonl` with `0600` permissions
(git-ignored). Disable collection with `BOT_OTEL_DISABLED=1`. Opt into the live model
studio (publishes topology into the `web/` studio for inspection) with `BOT_MODEL_PUBLISH=1`.

## What a Bot is made of

**Core package (`import bot`)**

| Layer | What ships |
|---|---|
| Body | `bot.Bot` — lifecycle, device attachment, stimulus fan-out, cognition handoff, reflex floor |
| Devices | `bot.devices.phone` (ringing, dialing, call lifecycle, busy/reorder tones), `bot.devices.audio` (microphone, speaker with placement) |
| Environment | broadcast scope, typed stimuli (`environment.sound`, …), world snapshots, `bot.environment.space` propagation law |
| Abilities | Listening (VAD · STT · sensitivity/self-sound), Hearing (voice identity), Vision, Reading, Language, Memory (STM/register, associative, consolidation, long-term), Communication (Conversation + turn detection), Speaking, Identity (name, recognition, values), Classifying, Decoding/Encoding |
| Cognition | Autonomy, Intuition, Reasoning, Reflection, Learning — all cancellable, all typed |
| Behavior | Starlark → HSM compiler, verified seed/native behaviors, durable storage, diagnostics |
| Telemetry | HSM observation → OpenTelemetry metrics + spans, low-cardinality by contract |

**Provider packages (`src/providers/*`, each owns its own SDK deps)**

| Package | Provides |
|---|---|
| `bot-provider-livekit` | realtime SFU transport + room audio for the Phone |
| `bot-provider-openai-compat` | chat clients / processors for any OpenAI-compatible endpoint |
| `bot-provider-gemini` | Gemini text generation & processing |
| `bot-provider-elevenlabs`, `bot-provider-mlx-audio`, `bot-provider-moonshine` | speech synthesis & recognition (cloud / Apple-Silicon-local / on-device) |
| `bot-provider-mlx-vlm` | local VLM reading (Apple Silicon) |
| `bot-provider-pyannote` | voice diarization & speaker identification |
| `bot-provider-sqlite-memory`, `bot-provider-postgres-memory` | turnkey SQLite / Postgres(+PGlite) durable memory |
| `bot-provider-typesafe` | label-tier selection (system_one Choice/Score/Noul) |
