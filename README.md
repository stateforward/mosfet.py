# Bot

**Bot** is a Python 3.13 runtime for *teachable software robots* — realtime, event-driven,
state-machine-native bots with a body that perceives and acts, a mind that decides, and
memory that turns experience into compiled reflexes.

Distribution: `stateforward.bot` ⟶ `import bot`. Deterministic execution runs on
[`stateforward-hsm`](https://pypi.org/project/stateforward-hsm/) ⟶ `import hsm`.

---

> The industry builds *agents*: an LLM in a loop, with chat history for memory, tool
> listings for hands, and a system prompt for a personality. Everything the bot will ever
> do is whatever the model feels like doing this turn — at model-call latency, at
> model-call cost, and with behavior that drifts when the prompt drifts.
>
> **Bot is structured the way an actual robot is.** It has devices wired into an
> environment with real physics. It has a body that fans sensory stimuli out to abilities
> while obeying a reflex floor it cannot disable. It has a layered mind that tries
> compiled reflexes first, then a fast model, then a deep model — and only pays for chips
> when no cheaper path fires. And it is *teachable*: you tell it what to do in plain
> English and it compiles your lesson into a deterministic state machine, stored in
> durable memory to run without you, without tokens, without drift. And even assembly is
> on the same schedule: today you wire the anatomy in code — when this ships, bots
> assemble themselves. You unbox one, teach it, and let it loose.

---

## What makes it different

| Concern | Agent frameworks (prompt loop) | Bot |
|---|---|---|
| Where behavior lives | system prompt + tool list | state-machine topology + taught reflexes |
| Changing behavior | edit the prompt, hope it holds | teach it; it compiles a durable machine |
| Known situations | full model call, every time | compiled reflex: no model call, no tokens, deterministic |
| Mid-run interruption | the turn is lost or stalls | typed cancel/completion events over an HSM |
| Memory | chat transcript | typed episodes + durable learned behavior |
| Execution model | asyncio glue around one model | realtime event-driven HSM (hierarchical) |
| Sound | bytes in, transcript, reply out | a physical space: dB SPL, distance, self-suppression |
| Observability | print/log | OpenTelemetry metrics & spans, first-class |

One principle sits under all of it, and it is the thing most "agent-like" libraries get
backwards:

> **A bot perceives, decides, and acts. The framework models capability — never the
> choice.** You give a bot anatomy and teachers; you do not script its reactions. A bot
> that hears a ring and does nothing is not broken: it was given nothing that earned a
> response. A missing behavior is a missing capability, and capabilities are what Bot
> supplies — *if* something in the bot's position could reasonably have done otherwise,
> the bot must be the one choosing.

If you want a workflow builder, a RAG stack, or a service that wires an LLM to a
chat channel, Bot is not it. If you want software robotic anatomy — something that
exists in a space, hears with peripheral hardware modeling, reacts faster than a model
round-trip, and gets smarter through instruction — read on.

---

## The anatomy

```
                   ENVIRONMENT (real space, physics, broadcast)

        ┌────────────────────────────────────────────────────────────┐
        │                         BODY                                │
        │  devices (Phone · Microphone · Speaker)                     │
        │  spatial presence · reflex floor · sensory fan-out          │
        └──────────────┬─────────────────────────────────────────────┘
                       │  typed stimuli (e.g. environment.sound)
                       ▼
                    INPUT ABILITIES (Listening · Reading · Vision …)
                       │  enriched cognition input, live body context
                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │                      COGNITION (the mind)                   │
        │                                                             │
        │   1. Autonomy    compiled reflexes — 0 tokens, no model     │
        │          │ unhandled                                       │
        │   2. Intuition   fast System-1 model (e.g. Mercury 2)       │
        │          │ unhandled                                       │
        │   3. Reasoning   deep System-2 model (e.g. Gemini / OpenAI) │
        │          │                                                 │
        │   4. Reflection  retain the turn, store the episode         │
        │   5. Learning    decode a lesson → ground it → author a     │
        │                  new compiled behavior (back to step 1)     │
        └────────────────────────────────────────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │      OUTPUT ABILITIES (Speaking · Communication …)          │
        │        effectors selected by the mind, driven by the body   │
        └────────────────────────────────────────────────────────────┘
```

Every box is a real, testable software-robot layer, not a convention:

- **Environment** is a broadcast/parenting scope with *physical fidelity*: sensors obey
  real acoustics (inverse-square dB SPL spreading — a sound at reference level heard twice
  as far away arrives 6 dB quieter, and something 1 cm from a source is already ~40 dB hot).
  A device cannot know how far away you are; the environment decides what reaches it.
- **Body** owns lifetime, device attachment, spatial presence, and stimulus fan-out —
  plus the *reflex floor*, the level of automatic behavior no bias can tune away, so a bot
  can never make itself permanently deaf or mute. The body never interprets; it hands
  enriched, typed ingots to cognition.
- **Devices** model real hardware: telephones that ring, dial, and seize; microphones and
  speakers with physical placement. A phone does not decide what a ring means — the bot
  does. (A dial that never completes still sounds the way it sounds to a human caller:
  busy and reorder tones at 20 dB under the ringer, only in the earpiece. *What the bot
  does about a busy tone is its own call.*)
- **Abilities** are the composition unit — typed input, typed products, typed failures,
  nestable. *Input* abilities (Listening, Reading, Vision, …) sit on the sensory fan-out;
  *output* abilities (Speaking, Communication, …) are effectors the mind selects. The bot
  suppresses its own voice using the same ballistic self-sound window an ear uses — not
  "self-recognition," just physics-adjacent bookkeeping of its own mouth — so a bot does
  not talk over itself or hallucinate that it heard itself.
- **Cognition** is a pipeline, not a while loop. Autonomy matches each turn against
  learned (or natively seeded) compiled behaviors by stimulus trigger; intuition and
  reasoning are separate, swappable model stages; reflection retains experience into
  memory; learning can return new behavior straight into step 1. Turns are cancellable
  operations — being interrupted is a typed event, not lost work.
- **Providers** (optional workspace packages) own SDKs, transport, and raw quirks. The
  core is provider-agnostic: cognition stages are any OpenAI-compatible endpoint, device
  transport is pluggable, memory runs on SQLite or Postgres.

---

## Teach, don't script

You never program a Bot's choices. You assemble it, then talk to it:

1. **Experience** — an ordinary stimulus arrives (say, a phone starts ringing). The body
   admits it through the real sensory fan-out into short-term memory. Nothing is
   pre-seeded; nothing is hand-fed.
2. **Instruction** — someone says, in the room, the way you'd talk to a person:
   *"When the phone rings, make sure you answer it."*
3. **Learning** — the (acquired) Learning ability offers itself in that turn's tool menu
   like any other capability; the model decides on its own that this input is a lesson.
   Learning decodes the instruction, *grounds* it against remembered turns and the
   observed stimulus register, and fails closed when it cannot name what the lesson is
   about.
4. **Compilation** — Revision authors a sandboxed **Starlark** program — a restricted,
   deterministic dialect — and Bot compiles it into an executable HSM behavior. The
   compiler validates every event name against the live model; compiled behaviors may
   only dispatch and process modeled events. No ability bindings, no imports, no network —
   the authoring sandbox cannot reach the filesystem or call out.
5. **Autonomy** — the next ring matches the new machine by stimulus trigger and runs it
   through one attach → input → terminal lifecycle. Deterministic, durably stored, token-free. The
   authored behavior persists with a practice lifecycle in the same durable memory, and a
   behavior authored mid-lifetime is usable immediately — Autonomy re-reads the learned
   inventory every turn.

That is *training*, in the plain-English sense. No fine-tuning, no fine-tune drift, no
prompt surgery, no prompt drift — a lesson compiles into something with `read`/write
semantics, versioned storage, and a runtime the bot itself executes.

---

## In the box

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

---

## Quickstart

```sh
git clone <repo-url> && cd bot.py
uv sync --locked --all-packages --all-groups

# a device-free ear/mouth bot: macOS `say` WAV → Listening → mixed-model
# cognition (Mercury 2 intuition / Gemini reasoning) → spoken reply WAV
uv run --project examples/listen_speak_bot listen-speak-bot

# a real phone against a LiveKit SFU: turnkey docker+JWT+join, full turn proof
uv run --project examples/phone_bot phone-bot --json

# the teachability proof: memory starts empty; the bot hears a lesson spoken
# in the room ("When the phone rings, make sure you answer it"), compiles it,
# and answers the *second* ring through Autonomy with zero model tokens
uv run --project examples/phone_bot python examples/phone_bot/learning_phone_bot.py
```

Today, assembly happens in code — real, typed contracts. That is scaffolding, not the
product; when this ships, bots assemble themselves:

```python
import bot
import hsm
from bot import Bot
from bot.abilities import listening, memory, speaking     # anatomy
from bot.devices import phone                              # peripherals
from bot.environment import Environment, SoundEvent, space # physics
from bot.providers.openai_compat import ChatClient         # mind transport
```

…then attach provider components, acquire abilities, and start the bot. Every seam
above — device, provider, ability — is a typed contract a bot will exercise on its own
behalf; what you assemble today is today's test of that, not tomorrow's API. See
`examples/` for complete, runnable compositions — each example is its own `uv` package so
example-only dependencies never leak into core.

Provider credentials are ordinary environment variables read by the provider packages
(`BOT_MERCURY_API_KEY`, `BOT_GEMINI_API_KEY` / `GEMINI_API_KEY`, `BOT_OPENAI_API_KEY`, …).
No live provider is required to run the test suite:

```sh
uv run --locked python -m pytest -q -m 'not live'
```

---

## Observability & privacy

Every HSM-visible behavior is observed at runtime boundaries that opt in
(`hsm.observe`) and exported as OpenTelemetry metrics and spans with **low-cardinality
attributes only** (component, event kind/name, stage, outcome, normalized failure kind).
Raw media, credentials, and high-cardinality identifiers never become telemetry
attributes. Trace context rides `hsm.Event.metadata` end-to-end, so a reflex on a phone
correlates with the model turn that teaches the next one.

Local JSONL export writes `otel-spans.jsonl` / `otel-logs.jsonl` with `0600` permissions
(git-ignored). Disable collection with `BOT_OTEL_DISABLED=1`. Opt into the live model
studio (publishes topology into the `web/` studio for inspection) with `BOT_MODEL_PUBLISH=1`.

---

## Development

```sh
uv sync                          # core + workspace providers
uv run --group dev python -m pytest        # focused runs: uv run --package bot-provider-<p> -m pytest src/providers/<p>/tests
uv run --group dev ruff check --fix src tests
uv run --group dev basedpyright  # strict: private usage is an error, not a lint
uv build                         # wheels/CLI release builds (see dist/)
uv run python scripts/check_release_version.py
```

See [docs/release.md](docs/release.md) for versioning and CI publishing notes.

---

## Status

Experimental, pre-1.0, under active development — but not a prototype: typed contracts,
a strict typechecker, OpenTelemetry contract, and provider isolation are enforced, and
the examples above are lived-in proofs, not demos.

One direction is already fixed: **bots that assemble themselves.** Composition-in-code
exists to harden the seams — devices, abilities, providers, teaching — as typed
contracts a bot can exercise on its own behalf. When this ships, you unbox a bot, teach
it, and never wire the body; what you assemble today is how those seams get tested.
Build an anatomy in the meantime, teach it something, and watch it never pay for tokens
on a known situation again.
