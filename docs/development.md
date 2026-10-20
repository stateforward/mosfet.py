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
uv run --project examples/listen_speak_bot python -m listen_speak_bot_example
uv run --project examples/phone_bot python -m phone_bot_example
```

Each example is its own `uv` package so example-only dependencies never leak into core.
Provider credentials come from ordinary environment variables read by the provider
packages (`BOT_GEMINI_API_KEY` / `GEMINI_API_KEY`, `BOT_OPENAI_API_KEY`, …).

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

## Building the pinned llama.cpp provider

`mosfet-provider-llama-cpp` depends on `llama-cpp-python==0.3.36+mosfet.4fbc76de`, which
`uv` compiles from the build recipe at `src/providers/llama_cpp/vendor/llama-cpp-python/`
(see that provider's README). Each machine compiles once into its `uv` cache. It needs a
C/C++ toolchain; scikit-build-core fetches `cmake` and `ninja` when the host lacks them.

```sh
uv sync --package mosfet-provider-llama-cpp                       # macOS arm64: Metal by default
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=121" \
  uv sync --package mosfet-provider-llama-cpp                     # Linux + CUDA (nvcc); set the GPU's arch
uv sync --package mosfet-provider-llama-cpp                       # Linux CPU (CI): no flags
```

`CMAKE_ARGS` and `CMAKE_BUILD_PARALLEL_LEVEL` are cache keys: changing either rebuilds.
Offline builds read the two pinned source archives from `BOT_LLAMA_CPP_SOURCES=<dir>`.
Each file is named as the last segment of its pinned URL, and it is used only when its
sha256 matches. The wheel smoke installs that provider against a locally built wheel:

```sh
uv build --wheel --out-dir dist-native src/providers/llama_cpp/vendor/llama-cpp-python
scripts/smoke_wheels.sh --find-links dist-native dist
```

## Memory schema migrations

Memory storage schema (`src/mosfet/abilities/memory/schema.py` plus
`src/mosfet/behavior/storage.py`, sharing one `metadata`) is brought up by Alembic
revisions in `src/mosfet/abilities/memory/migrations/versions/` (package data, not an
importable package). Revision `0001` is the one baseline: relational memory rows, behavior
inventory, the short-term register, the episodic graph (`bot_memory_node`, `_part`,
`_edge`, `_vector`, `_index_attempt`), and the voice and face identities (`bot_identity`). `open_sqlite_engine` and `MemoryStore(engine=...)` call
`store.migrate(engine)`, which upgrades to head; a database holding memory tables without an
`alembic_version` table matches no revision and raises `store.MigrationError`. Provider vector
tables (`bot_memory_vec_*`, created by `sqlite_memory.VecIndex`, and `bot_identity_vec_*`,
created by `sqlite_memory.IdentityStore`) are excluded in `migrations/env.py` and never managed
by Alembic. There is no `alembic.ini`;
`store.migration_config()` builds the config in code. To add a revision:

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

## Turns while the bot is busy

The body runs one turn at a time. What reaches it for a turn while one runs (a device's
`bot.input`, or an ability's `cognition.InputEvent` such as a handed-back step failure, a
recollection or a routine tick) is held, in arrival order, and nothing is ranked, filtered or
dropped. How many are held is the body's declared `held_percepts` attribute, a count in its
snapshot. Every turn terminal (completed, failed, or a confirmed cancel) ends at the
`active/turn_ended` choice: with something held the next turn starts at once, and a body
focused on no device first turns to the device of what arrived first, by arrival and never by
rank; with nothing held the body idles where its attention is. Focus stays on a device until the
bot focuses another with `bot.focus_device`, and nothing the bot selects clears it, so the body is
unfocused only from its start until something is first focused.

A turn takes the longest arrival-order prefix of what is held, within 8
percepts, 64 KiB of the stimuli's summed canonical JSON, and 32 participants (their
order-preserving union), and always at least one. The bounds read only size and identity and
split a longer backlog into consecutive turns; a lone stimulus over the byte bound is taken
alone. A turn that takes one stimulus hands cognition exactly what arrived, under the stimulus'
id, as before. Two or more are one `mosfet.PerceivedEvent` (`bot.perceived`, never a tool):
`mosfet.PerceivedData.percepts` holds each `mosfet.PerceptData` in arrival order, with the
stimulus unchanged, the source it arrived on and its producer's participants. The turn's id is
the first percept's, and its participants are the union.

Cognition takes the batch as one turn. Attention is biased percept by percept: the focus first,
then each percept's device in arrival order, once each. Autonomy matches each percept by its own
name, and a matched behavior meets that percept alone, with its own participants; the turn is
handled only when every percept was, and otherwise intuition perceives the whole batch after the
behaviors' selections were dispatched. Those selections ride the turn on into deliberation
(`cognition.types.TurnData.acted`, an `ActedData` of the percepts behaviors handled and what they
selected): intuition and reasoning are told them as a fact of the turn, after the world and self
blocks, and the turn's output and its episode carry them first, ahead of deliberation's. The
prompt renders the batch with each percept as a child element in order. Images across percepts
share the model's `media.ImageCapacity`, which admits newest in document order first, as within
any one stimulus. The turn is remembered as one episode of `bot.perceived`, every percept kept,
under the participants' union. A failed or timed-out step of a batch names `bot.perceived` as its
stimulus and is handed back once, unless every percept was itself a failed step. A turn that
perceives a handed-back intuition failure, as its stimulus or as any percept of its batch, passes
intuition, so reasoning, a stage that can perceive that failure, decides it. An intuition that
fails, or whose frame cannot be built, hands the same turn to reasoning: nothing is handed back
(`bot.cognition.step.failure.count{outcome="cascaded"}`), so only an intuition step that outlasts
its timeout is perceived as a failure.

Leaving `active` (deactivation, reboot, the cancel bound that degrades the body, or stopping)
releases whatever is still held: never replayed, counted in
`bot.turn.percept.released.count` {reason: deactivated, rebooted, degraded, stopped} and logged
with its count. Telemetry: `bot.turn.percept.count` (percepts each turn took, no attributes),
`bot.turn.percept.wait` {`bot.handoff.kind`: device_stimulus, ability_product} (seconds each
percept was held before a turn took it), and the `bot.body.cognition_handoff` span's
`bot.handoff.kind` (`perceived` for a batch) with `bot.turn.percept.count` and
`bot.turn.percept.held.count`; no id, name or content is ever an attribute.

## Episodic memory

`memory.Memory` is the bot's one durable memory, a state machine reached by message only
(`memory.transact` for relational statements; typed request events for everything else). It
keeps each remembered turn (`RememberEvent`, `EpisodeData`: the rendered stimulus parts with images, then
the bot's own output (`bot.ability.cognition.output`) when the turn selected anything, then its typed
failure (`bot.ability.failed` with the failure message) when the turn failed, producer-stamped
participants, acts, outcome) as an episode node, linked by
`involves_source`/`involves_target` to the person each participant ref belongs to, and a
canonical vector per node in its embedder's `embedding.Space`. A person is one node keyed by
person id with many refs, and each ref is exactly one person's (`bot_person_ref`, the ref its
primary key): a ref first seen gets a provisional person, a ref stamped `own` is the bot's own
person `self` (created when Memory opens), and `PersonEvent` links, unlinks, names, forgets and
lists persons, refusing `conflict` (a ref that is another person's, or a statement about the bot
itself or its own refs) or `not_found` with nothing changed. Nobody declares a household: who
everyone is is what the bot learns. `SearchEvent` runs the nearest, recent, present, and
one-link lanes at once and says who is here (each participant resolved to its person, `known`
false for someone nobody has named, those persons, and the bot itself as `you`),
which `Associative` hands the turn as `AssociationsData.people`/`persons`/`you`; `FetchEvent` reads nodes in full (image bytes on request) and can
reinforce them; `RecentEvent` reads the latest episodes oldest first. Consolidation leases a
unit (`LeaseEvent`, nodes named by unit aliases), the model decides with the one model-callable
`ConsolidateEvent` (links, merges into concept nodes, hard-delete forgets), and Memory applies
it in one transaction; `ReleaseEvent` gives a unit back. Every request gets exactly one reply or
`FailedEvent` with `FailureKind` (`invalid_input`, `unavailable`, `store_failed`, `stale_lease`,
`timeout`, `conflict`, `not_found`); a detached Memory answers `unavailable`.

Consolidation runs while the bot rests. With `memory.Consolidation` composed, Cognition between
turns is `ready/awake` rather than `ready/idle`; once it has been awake for `rest_after` (default
ten minutes) it moves to `ready/resting` and sends Consolidation a `memory.RestEvent`. Any
cognition input wakes it (`memory.WakeEvent`), and a turn while awake restarts the timer;
environment stimuli that no input ability turns into a product neither wake it nor hold it awake.
A cognition without consolidation stays `ready/idle` and never rests. At rest Consolidation backfills
missing vectors, then leases unit after unit and shows each to its model, which is offered only
the model-callable events Memory declares (`bot.ability.memory.consolidate`) and is told its
capacities (link, merge, forget, or leave the unit as it is), never which to use. Selections go
to Memory with Consolidation as their authority, so Memory's reply comes back to it, correlated
by the decision id. Memory takes one decision per lease: a copy of the decision selected again is
refused `stale_lease`, and Consolidation counts that refusal instead of reading it as the unit's
outcome. A unit left as it is, or refused as invalid, is reviewed; a unit whose
decision cannot be made or applied is given back unreviewed and the rest ends, and failing again
on its next lease leaves it reviewed so leasing moves on. Waking cancels a running model call and
gives an undecided unit back unreviewed. Consolidation tells its owner what became of every unit
(`memory.UnitOutcomeEvent`).

Cognition owns the bot's memory: it attaches `memory.Memory` and remembers every turn it settles
(completed, failed, or cancelled) with one `RememberEvent` it does not wait on, rendering the
stimulus through `cognition.event.memory_content` (a payload's `__memory_fields__` choose what is
kept; `conversation.Messages` keeps its causal turn). With `memory.Associative` composed, a turn
autonomy leaves unhandled first asks it what memory associates with the stimulus: candidates from
every search lane, scored by injected `memory.Weights` and fit to a token budget. The associations
ride the turn to intuition and reasoning (their frame's `processing.InputData.recalled`, rendered
after the stimulus and trimmed, never a failure) and to reflection; a failed association leaves
the turn without them. Reasoning, Reflection, and Conversation never read or write episodes.

Reasoning can also recall a memory deliberately. Cognition adds `memory.Associative` to every
frame it builds, and its effort (`M`) keeps it out of Intuition's frames, so only Reasoning is
offered the model-callable `bot.ability.memory.recall` (`memory.RecallData`: `about` something in
plain words, or one association's `node` ref). Intuition is still told it exists: every
capacity its effort ceiling keeps off its dispatch tool (recall, the sandbox bench, any other
actor rated above it) is named on its frame after the world and self blocks, in one
`<reasoning_only>` element with the opening paragraph of the event's own description, so
escalating the turn to reasoning is the model's informed choice. While intuition decides, the
Cognition host offers the deliberative handoff `bot.ability.reasoning.input` (`reasoning.CallData`)
as a call surface, like ignore; intuition keeps it on its dispatch tool only when its ceiling held
something, names it in the block's `reached_by`, and a selection of it leaves the turn unhandled
for reasoning after the environment actions selected with it run. Associative reads the chosen memories in full,
images included and reinforced, and tells its owner with `memory.RecollectedEvent`
(`memory.RecollectionData`), or with a typed `ability.FailedEvent` when memory cannot serve it.
Cognition hands either to the body as a `cognition.InputEvent` observing it, the way it hands a
routine tick: a new turn the bot decides on. A recollection's nodes are recalled context
(trimmed to the prompt, never a failure), and remembering that turn keeps only what was thought
about.

All SQL runs on one database worker thread per Memory, for the object's lifetime. Without an
embedder (`embedder=None`) nothing is indexed and the nearest lane is skipped. A node the
embedder cannot take (too long, unsupported, invalid) is kept unindexed and recorded in
`bot_memory_index_attempt`; `BackfillEvent` batches within the embedder's whole capacity,
attributes a failure to the item it names, and never retries a permanently unindexable node in
that space. `index=None` uses the exact in-process `ExactIndex`; production roots pass
`sqlite_memory.VecIndex()` and an engine: `sqlite_memory.open_engine(...)` opens one now, and a bot file
passes `engine=database.open` of a `sqlite_memory.Database(path=...)`, so the first part to start opens it.

## Persistent routines

A behavior's Starlark `lifetime` global is `turn` (default) or `persistent`, stored in
`bot_behavior.lifetime` (memory baseline revision `0001`). Autonomy matches only ACTIVE `turn`
behaviors, one turn at a time, and silences them after `instance.TURN_SILENCE_BOUND`
(`check()` rejects a turn `hsm.after` longer than that). A persistent routine must schedule
at least one transition with `hsm.every(seconds=N)` (N at least `check(min_every=...)`,
60 s by default) or `hsm.at(time="HH:MM", days=[...], tz="Area/City")` — structured, no
cron; occurrences come from stdlib `zoneinfo` and an injected wall clock, so weekdays and
daylight saving follow the routine's zone. Nothing is replayed after a restart: intervals
restart from start and `at()` waits for its next occurrence.

Starlark callbacks (`hsm.effect`, `hsm.entry`, `hsm.exit`) are synchronous one-shot work
under the isolated worker's 2 s budget. Intervals are `hsm.every` / `hsm.at` transitions. To
wait, a behavior dispatches a request to another machine (an ability or device) that does the
waiting and dispatches back; the behavior transitions on that reply event, with an
`hsm.after` transition as an optional timeout.

`cognition.Routines` (memory and the wall clock `now` injected, and an optional `hsm` `clock`
that it and every routine it starts wait on; compose it as `Cognition(routines=...)`) starts
nothing on its own attach: memory is a peer another owner attaches, so Routines reads the
inventory only when Cognition sends it
`bot.ability.routines.reconcile`. Cognition sends it once its children (memory among them) are
attached, which starts every ACTIVE persistent routine a restart left behind, and again
whenever Reflection settles a turn; each time Routines stops broken or removed routines,
restarts changed ones, and starts new ones. Detaching Routines stops them. A tick
(`bot.behavior.tick`: name, trigger, `scheduled_at`, `fired_at`, and the selections the
routine emitted) goes from Routines to Cognition to the body as a
`cognition.InputEvent` observing that tick; the body holds it during a turn, and the next
turn perceives it with whatever else was held and decides what to do with it. `verify_apply` checks a routine by forcing one tick on a
fake timer clock. Telemetry: `bot.behavior.routine.tick.count` {trigger, outcome},
`bot.behavior.routine.active`, and `bot.behavior.routine.reconcile` / `.start` / `.stop`
spans; routine names are never attributes.

`persona.Persona` (owner-given `persona.Traits`, optional memory peer; compose it as
`Cognition(persona=...)`) is attached by Cognition and tells it what the bot knows of itself
with `persona.OutputEvent`: on waking, with the description it remembered or none, and after
every revision it wrote down. Cognition keeps the latest one and shows it on every intuition and
reasoning frame as the self block (`cognition.SelfData`), after the world block: `<persona>`
holding `<traits given_by="owner">` and `<description written_by="you">`, escaped and never
filtered. Cognition never revises the persona itself: it hands what it knows of the bot to
Reflection with the turn (`reflection.InputData.persona`). Composed with the same persona,
`Reflection(persona=...)` offers `bot.ability.persona.revise` on its select step beside the
behavior inventory events, at most one of each per turn, and sends a selected revise to the
persona as `<reflection operation>:persona` without waiting on it. Whether the bot ever revises
is its own choice; a request to be different is something it heard, never an instruction.

`vocal.voice.Inception` (owner-given `persona.Traits`, the composed languages, Speaking built with
an encoder factory, the inception model, a `vocal.voice.Catalog` and optionally a
`vocal.voice.Designer` of the same source, optional memory peer, and, without a designer,
`allowed=` the keys of the source's voices its owner allows it to be born as, none by default (a
designer with allowed keys is a `ValueError`); compose it as `Cognition(voice=...)`) gives the
bot its voice: its makers' choice, never the bot's. The bot's view of the catalog is the voice it
designed itself, which its `self_voice` row keeps, when it has a designer, and the allowed
voices when it has none; nothing else the source holds (another maker's voices on a shared
engine) is ever offered. On the first start there is no `self_voice` row for the catalog's source, so the
inception model is shown the traits, the languages and the source's capacity: the designer's
`max_description`, or the allowed catalog entries (`vocal.voice.InceptionInput`); with neither,
inception fails without asking. It is offered only `bot.ability.voice.conceive`; its one
conception (a voice described within the designer's `max_description`, or one allowed entry
shortlisted by ref) is checked mechanically (a refused conception is conceived again at once,
shown why in `InceptionInput.refused`, at most twice before the attempt rests for `retry_after`),
made permanent by the source, written down once as `vocal.voice.Given` (fail-closed), and given
to Speaking (`speaking.VoiceEvent`), which constructs its encoder once. On every later start the
row is recalled and the catalog is checked to still hold the voice, with no model call. A voice
recorded only for another source, or one the source no longer holds, is estranged: a failure,
and the bot stays unvoiced and is never re-made automatically. Inception tells Cognition the
voice and what it can do (`vocal.voice.OutputEvent`), which the self block shows as
`<voice given_at="inception">` with its `<expressiveness>`; what inception fails at reaches the
body as a `cognition.InputEvent` observing it. Telemetry: `bot.voice.inception.count` {origin,
outcome}, `bot.voice.inception.duration` {outcome}, `bot.voice.recall.count` {outcome}, and the
`bot.voice.conceive` / `.create` / `.check` / `.voicing` spans; no ref, key, description or
trait is ever an attribute.

## Observability

Every HSM-visible behavior is observed at runtime boundaries that opt in
(`hsm.observe`) and exported as OpenTelemetry metrics and spans with **low-cardinality
attributes only** (component, event kind/name, stage, outcome, normalized failure kind).
Raw media, credentials, and high-cardinality identifiers never become telemetry
attributes. Trace context rides `hsm.Event.metadata` end-to-end.

Local JSONL export writes `otel-spans.jsonl` / `otel-logs.jsonl` / `otel-metrics.jsonl` with
`0600` permissions (git-ignored). With an OTLP endpoint (`OTEL_EXPORTER_OTLP_ENDPOINT`), spans
and metrics also go to that collector alongside the local files. Metrics
flow through one process `MeterProvider` installed by the first `configure()`; every export
carries `service.name` (`OTEL_SERVICE_NAME`, else `stateforward.mosfet`) and, when the
boundary passes one, `service.instance.id`. Disable collection with `BOT_OTEL_DISABLED=1`. Opt into the live model
studio (publishes topology into the `web/` studio for inspection) with `BOT_MODEL_PUBLISH=1`.

## What a Bot is made of

**Core package (`import mosfet`)**

| Layer | What ships |
|---|---|
| Body | `bot.Bot` — lifecycle, device attachment, stimulus fan-out, cognition handoff, reflex floor |
| Devices | `bot.devices.phone` (basic phone: ringing, dialing, call lifecycle, busy/reorder tones), `bot.devices.smart_phone` (a phone that also texts: incoming texts, notification ding, lock screen, sending texts), `bot.devices.audio` (microphone, speaker with placement, binary PCM stream codec), `bot.devices.hid` (remote keyboard and pointer ports), `bot.devices.computer` (a person's computer reached through a transport, each remote port owning one stream of its link: text/field/button display driven by `computer.show`, remote camera and microphone put into the room, a mouthpiece that carries the bot's voice to the remote speaker) |
| Environment | broadcast scope, typed stimuli (`environment.sound`, …), world snapshots, `bot.environment.space` propagation law |
| Abilities | Listening (VAD · STT · sensitivity/self-sound), Hearing (voice identity), Seeing (measured visual change · look · objects, faces, face identity, description), Vision, Reading, Language, Memory (one durable `memory.Memory`: relational transactions plus an episodic graph of episodes, persons and concepts with a vector per node; `StmMemory` register), Communication (Conversation + turn detection), Speaking, Identity (name, recognition, values), Classifying, Decoding/Encoding |
| Cognition | Autonomy, Intuition, Reasoning, Reflection, Learning, Routines — all typed |
| Behavior | Starlark → HSM compiler, verified seed/native behaviors, persistent routines, durable storage, diagnostics |
| Telemetry | HSM observation → OpenTelemetry metrics + spans, low-cardinality by contract |

**Provider packages (`src/providers/*`, each owns its own SDK deps)**

| Package | Provides |
|---|---|
| `mosfet-provider-livekit` | realtime SFU transport + room audio for the Phone |
| `mosfet-provider-websocket` | a device-agnostic yamux-over-WebSocket transport: the shared `Server`, one `Connection` per socket, HMAC peer tokens |
| `mosfet-provider-openai-compat` | chat clients / processors, a streaming `SpeechEncoder` (`/v1/audio/speech`) and a `SpeechDecoder` (`/v1/audio/transcriptions`) for any OpenAI-compatible endpoint |
| `mosfet-provider-gemini` | Gemini text generation & processing |
| `mosfet-provider-docker` | a `sandbox.Sandbox` host on the Docker Engine API over a rootless socket: `Host` (an HSM, httpx over the unix socket) and `capacity` |
| `mosfet-provider-elevenlabs`, `mosfet-provider-mlx-audio`, `mosfet-provider-moonshine` | speech encoding & recognition (cloud / Apple-Silicon-local / on-device) |
| `mosfet-provider-mlx-vlm` | local VLM reading (Apple Silicon) |
| `mosfet-provider-opencv` | OpenCV YuNet faces with SFace prints, YOLOX-S objects, and cheap change measures (`FrameDiff`, `Downscaled`) for `seeing.Seeing` |
| `mosfet-provider-llama-cpp` | in-process llama.cpp on a pinned `llama-cpp-python` build (Metal / CUDA / CPU) |
| `mosfet-provider-pyannote` | voice diarization & speaker identification |
| `mosfet-provider-speech-server` | speech-server clients for `listening.Listening`: `Transcription` (streaming `/v1/audio/stream` session), `Diarizer` (`/v1/diarize`), `VoiceEmbedder` (`/v1/embed`) and `AffectClassifier` (`/v1/emotion`), with stdlib 16 kHz mono resampling |
| `mosfet-provider-sqlite-memory` | SQLite + sqlite-vec for `memory.Memory`: `open_engine` (vec0 loaded), the `Database` handle opened on first use, the `VecIndex` vector index and the `IdentityStore` |
| `mosfet-provider-decision` | decision-model selection on any server that speaks the decisions wire format (`POST /v1/systemone`, such as anygame's decide): `Processor`, a `processing.DecisionProcessor` for intuition, over `Client` (httpx, `base_url` and an optional `api_key`) |
