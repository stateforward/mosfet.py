# Development

Operational reference for contributors working on this repository. The root README is
the product document; everything that used to live there for programmers is consolidated
here.

Distribution name: `stateforward.mosfet`, imported as `bot`. Deterministic execution runs
on [`stateforward-hsm`](https://pypi.org/project/stateforward-hsm/) (`import hsm`).

## Composing a bot in code (today's composition surface)

```python
import mosfet
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

Scoped provider tests: `uv run --package mosfet-provider-<p> -m pytest src/providers/<p>/tests`.

See [release.md](release.md) for versioning and CI publishing notes.

## Memory schema migrations

Memory storage schema (`src/mosfet/abilities/memory/schema.py` plus
`src/mosfet/behavior/storage.py`, sharing one `metadata`) is brought up by Alembic
revisions in `src/mosfet/abilities/memory/migrations/versions/` (package data, not an
importable package). `open_sqlite_engine` and `MemoryStore(engine=...)` call
`store.migrate(engine)`, which upgrades to head (and stamps a pre-migration `create_all`
database at `0001` first). There is no `alembic.ini`; `store.migration_config()` builds the
config in code. To add a revision:

1. Change the table definitions in `schema.py` / `behavior/storage.py`.
2. Autogenerate a draft against a database at the current head:

   ```sh
   uv run --locked python - <<'PY'
   from alembic import command
   from mosfet.abilities.memory import store
   engine = store.open_sqlite_engine()  # in-memory database at the current head
   config = store.migration_config()
   with engine.begin() as connection:
       config.attributes["connection"] = connection
       command.revision(config, message="<what changed>", autogenerate=True, rev_id="0002")
   PY
   ```

3. Review `versions/0002_<slug>.py`: keep it dialect-neutral (e.g. `sa.func.now()`, not
   SQLite text defaults), give new `NOT NULL` columns a `server_default`, and write a real
   `downgrade()`. `down_revision` must be the previous head.
4. Run `tests/bot/abilities/memory/test_store.py`; its drift test fails until the
   migrated schema matches `metadata` again. Never edit a revision that has shipped.

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

**Core package (`import mosfet`)**

| Layer | What ships |
|---|---|
| Body | `bot.Bot` — lifecycle, device attachment, stimulus fan-out, cognition handoff, reflex floor |
| Devices | `bot.devices.phone` (basic phone: ringing, dialing, call lifecycle, busy/reorder tones), `bot.devices.smart_phone` (a phone that also texts: incoming texts, notification ding, lock screen, sending texts), `bot.devices.audio` (microphone, speaker with placement) |
| Environment | broadcast scope, typed stimuli (`environment.sound`, …), world snapshots, `bot.environment.space` propagation law |
| Abilities | Listening (VAD · STT · sensitivity/self-sound), Hearing (voice identity), Vision, Reading, Language, Memory (STM/register, associative, consolidation, long-term), Communication (Conversation + turn detection), Speaking, Identity (name, recognition, values), Classifying, Decoding/Encoding |
| Cognition | Autonomy, Intuition, Reasoning, Reflection, Learning — all cancellable, all typed |
| Behavior | Starlark → HSM compiler, verified seed/native behaviors, durable storage, diagnostics |
| Telemetry | HSM observation → OpenTelemetry metrics + spans, low-cardinality by contract |

**Provider packages (`src/providers/*`, each owns its own SDK deps)**

| Package | Provides |
|---|---|
| `mosfet-provider-livekit` | realtime SFU transport + room audio for the Phone |
| `mosfet-provider-openai-compat` | chat clients / processors for any OpenAI-compatible endpoint |
| `mosfet-provider-gemini` | Gemini text generation & processing |
| `mosfet-provider-elevenlabs`, `mosfet-provider-mlx-audio`, `mosfet-provider-moonshine` | speech synthesis & recognition (cloud / Apple-Silicon-local / on-device) |
| `mosfet-provider-mlx-vlm` | local VLM reading (Apple Silicon) |
| `mosfet-provider-pyannote` | voice diarization & speaker identification |
| `mosfet-provider-sqlite-memory`, `mosfet-provider-postgres-memory` | turnkey SQLite / Postgres(+PGlite) durable memory |
| `mosfet-provider-typesafe` | label-tier selection (system_one Choice/Score/Noul) |
| `mosfet-provider-needle` | local Needle 3 tool calling for the Intuition reflex tier (Apache-2.0, on-device) |
