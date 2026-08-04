# phone_bot example

stateforward.bot phone bot against a real LiveKit SFU: core `Phone` + LiveKit `PhoneService`,
with Mercury 2 intuition, OpenAI Terra reasoning/reflection, **local Silero VAD**,
**local pyannote voice identity**, and **off-device Gemini** STT/TTS.

## Local run (turnkey)

```bash
cd examples/phone_bot
uv run phone-bot --json
```

That single command will:

1. Create `.env` from `.env.example` if missing (local LiveKit dev defaults)  
2. Start LiveKit via `docker compose` if nothing is listening on `:7880`  
3. Mint a join JWT (`devkey` / `secret`) when no token is configured  
4. Run the bot with `--connect-livekit` and print a readiness summary  

Exit code `2` if the room join did not get a local track SID.

### Prerequisites

- Docker (only if LiveKit is not already running)  
- `uv` + Python 3.13  
- Optional for full `can_talk`: `BOT_OPENAI_API_KEY`, `BOT_MERCURY_API_KEY`, and
  `BOT_GEMINI_API_KEY` in `.env` (speech is off-device Gemini)

### Manual pieces (optional)

```bash
# SFU only
docker compose up -d
# or: livekit-server --dev

# Token only
uv run mint-livekit-token --identity human --room bot-phone-bot

# App only (expects URL+token already in env)
uv run phone-bot-example --env .env --connect-livekit --json
```

### Second participant

`uv run phone-bot` (without `--once`) joins the LiveKit room and prints a **prefilled** Meet link:

`https://meet.livekit.io/custom?liveKitUrl=ws://127.0.0.1:7880&token=…`

Open that URL on **this machine** (local SFU is not reachable from other devices as-is).  
Bare `https://meet.livekit.io/custom` shows “Missing LiveKit URL” — Meet reads the query params only.

Joining the room makes you reachable; it does not ring the bot. A phone rings when somebody
**dials** it — call setup addressed to its participant identity over LiveKit RPC. The `caller`
the phone shows is the identity LiveKit authenticated for whoever sent that setup.

The phone **rings** as `environment.sound` (`kind=phone.ringing`, `caller` = who is calling,
`source` = phone id) into **Listening** input — not raw `phone.ringing` into cognition. The
provider's call session handle stays on the service plane; nothing outside firmware sees it.
After answer, room media can flow `ServiceAudioReceived` → speaker → `environment.sound` →
Listening. Local speaker uplink is published to the LiveKit track (remote delivery is
suppressed to avoid echo).

A dial that never becomes a call is heard too, as the tone the exchange would have put in the
caller's ear: `kind=phone.busy` when the line refused or is engaged, `kind=phone.reorder` (fast
busy) every other way the network fails to complete the call. Both come out of the earpiece at
`CALL_PROGRESS_DB`, 20 dB under the ringer, because they are for the one person holding the
handset. The other ways to end up with no call — nothing was ringing, the bot hung up mid-dial,
nobody answered — make **no sound**, because no telephone makes one for them; they stay on the
service plane as `phone.no_call`. What the bot does about a busy tone is its own call.

## Saying something to the bot

```bash
uv run phone-bot
# ...
Call Bob at 555-0142.              # type it, press enter
```

Every line you type is **said out loud** in the robot's room. A mouth — the same
`bot.devices.audio.Speaker` the robot uses for its own voice — stands a metre in front of it,
renders your words locally with macOS `say`, and broadcasts them through `Environment.broadcast`
at 60 dB from where it is standing. The robot hears them the way it hears anything: environment
sound → its ears (its placement's `threshold_db`) → Listening (VAD + pyannote voice embeddings for
`source_ids`) → participant-owned TurnDetector TurnStart/TurnUpdate/TurnPause/TurnEnd + STT while
the turn is open → TurnComplete Conversation product → cognition when the turn closes. Step far enough away,
or raise its hearing floor, and it genuinely does not hear you. Nothing is dispatched at the
body.

A number is digits, because that is what makes it sayable out loud and pressable on a keypad.
On this LiveKit fiction the **participant identity is the phone number** (normalized digits, e.g.
`5550141`). Set `BOT_LIVEKIT_IDENTITY` to that form so token mint and dial destination agree.
`BOT_LIVEKIT_DIAL_PLAN` (`number=identity`, comma separated) is **optional** and only for rare
aliases; empty means dial-by-number. Use the fictional `555-0100`–`555-0199` range so nothing here
can resemble a real subscriber.

Say it the way you would say it. The number goes out of your mouth as sound and comes back
through speech recognition, which chooses its own punctuation — so `555-0142`, `5550142`,
`555 0142` and `(555) 0142` are one number on the keypad and one identity on the wire. Spelled-out
digits ("five five five…") are **not** a number and the dial is refused; if your STT returns words
rather than figures, that is a real limit and you will see it as a rejected dial rather than a
wrong one.

Being spoken to is not being made to. The example never reads the words: it does not look for a
number in them, does not match on "call", and has no transition anywhere of the form "heard X →
do Y". A bot that hears `Call Bob at 555-0142.` and decides this is not the moment has
decided, and that is a legitimate outcome. Answering, declining, dialing, and hanging up all stay
the bot's decisions — nothing in the wiring makes them.

Requires macOS (`say` and `afconvert`); the readiness summary warns by name if they are missing.
Any other `Encoder[bytes, bytes]` drops in — the Moonshine `SpeechEncoder` in
`src/providers/moonshine` is the same contract.

## What “connected” means

| Summary field | Meaning |
|---|---|
| `livekit_room_audio_attempted` | `--connect-livekit` and URL+token present |
| `livekit_room_audio_connected` | Local audio track SID assigned after room join |
| `can_talk` | Room connected **and** OpenAI + Mercury cognition keys **and** Gemini speech key |

Room join alone does not require model keys; full agent talk does. VAD is local Silero
(`mlx-community/silero-vad` via `bot-provider-mlx-audio`). Voice identity is local pyannote
speaker embeddings (`pyannote/wespeaker-voxceleb-resnet34-LM` via `bot-provider-pyannote`) so
Listening speech products carry non-empty `source_ids` into Conversation. STT/TTS stay off-device
Gemini (`gemini-3.5-flash` STT + `gemini-3.1-flash-tts-preview` TTS by default; override with
`BOT_GEMINI_STT_MODEL` / `BOT_GEMINI_TTS_MODEL` / `BOT_SILERO_VAD_MODEL` /
`BOT_PYANNOTE_VOICE_IDENTITY_MODEL`). Live pyannote weights use the Hugging Face/pyannote auth
already required by that provider package.

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | Local LiveKit SFU (`--dev`) |
| `.env.example` | Env template |
| `scripts/mint_livekit_token.py` | JWT mint (no extra deps) |
| `scripts/phone-bot.sh` | Thin wrapper around `uv run phone-bot` |
| `src/phone_bot_example/` | Example app |

## Blackbox dual agent (LiveKit only)

Two agents meet **only** on the local SFU — no shared Environment and no direct `environment.sound` wiring.
Both sides use **off-device Gemini** for speech so dual talk does not load local MLX STT/TTS:

| Side | Process | Speech |
|---|---|---|
| Agent A | `phone-bot` | Silero VAD + pyannote identity (local) + Gemini STT + Gemini TTS |
| Agent B | blackbox / `caller_agent` | Gemini TTS (default; `--tts say` for offline) |

```bash
# terminal 1
uv run phone-bot -v

# terminal 2 — agent B publishes Gemini TTS audio via LiveKit RTC
uv run python scripts/blackbox_livekit_dual_agent.py \
  --line "Hello agent A, can you hear me through LiveKit?"
```

Or one-shot (spawns agent A):

```bash
uv run python scripts/blackbox_livekit_dual_agent.py --start-phone-bot \
  --line "Hello agent A, can you hear me through LiveKit?"
```

Requires `BOT_GEMINI_API_KEY` in `.env` (plus cognition keys for full agent A replies).
Verdict uses LiveKit media plus optional scrapes of `/tmp/phone-bot-live.log`
(DecodingSpeech). Silero VAD loads are expected; local Whisper/Qwen STT/TTS loads fail the harness.

## Blackbox two bots (caller and callee)

Two real `phone-bot` processes, two lines on one exchange. Identities are the line numbers
(normalized digits); no dial plan is required:

```bash
uv run python scripts/blackbox_livekit_two_bots.py \
  --caller-number 555-0141 --callee-number 555-0142
```

Somebody walks up to the caller and says `Call Bob at 555-0142.` out loud; nobody says anything to
the callee. The harness writes the sentence to the caller's stdin once it is awake, and the bot
hears it as sound. Both join the same room as their number identities and can dial each other by
number. Whether the caller dials, and whether the callee answers, stay theirs to decide. The
verdict prints `dialed=` for the caller, `rang=` for the callee and `decoded=` for whoever turned
audio into words, and both sides must be audible to pass. A run where the caller never dials is
reported, not failed.

## Unit tests (no LiveKit)

From repo root:

```bash
uv run python -m pytest tests/examples/test_phone_bot.py
```
