# phone_bot example

stateforward.bot phone bot against a real LiveKit SFU: core `Phone` + LiveKit `PhoneService`,
with Mercury 2 intuition, OpenAI Terra reasoning/reflection, **local Silero VAD**, and
**off-device Gemini** STT/TTS.

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

To let this bot place calls, give it a dial plan with `BOT_LIVEKIT_DIRECTORY`
(`name=identity`, or a bare identity dialable by its own name). With no dial plan the bot is
registered with no exchange: it can be called but cannot call. Answering, declining, and hanging
up all stay the bot's decisions — nothing in the wiring makes them.

## What “connected” means

| Summary field | Meaning |
|---|---|
| `livekit_room_audio_attempted` | `--connect-livekit` and URL+token present |
| `livekit_room_audio_connected` | Local audio track SID assigned after room join |
| `can_talk` | Room connected **and** OpenAI + Mercury cognition keys **and** Gemini speech key |

Room join alone does not require model keys; full agent talk does. VAD is local Silero
(`mlx-community/silero-vad` via `bot-provider-mlx-audio`). STT/TTS stay off-device Gemini
(`gemini-3.5-flash` STT + `gemini-3.1-flash-tts-preview` TTS by default; override with
`BOT_GEMINI_STT_MODEL` / `BOT_GEMINI_TTS_MODEL` / `BOT_SILERO_VAD_MODEL`).

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
| Agent A | `phone-bot` | Silero VAD (local) + Gemini STT + Gemini TTS |
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

Two real `phone-bot` processes, one of which can call the other:

```bash
uv run python scripts/blackbox_livekit_two_bots.py \
  --caller phone-bot-alice --callee phone-bot-bob
```

The caller's env file gets `BOT_LIVEKIT_DIRECTORY=<callee identity>`; the callee's does not. That
one line is the whole difference between the roles — a capability, not an instruction. Whether the
caller dials, and whether the callee answers, stay judgment. The verdict prints `dialed=` for the
caller and `rang=` for the callee, and both sides must be audible to pass.

## Unit tests (no LiveKit)

From repo root:

```bash
uv run python -m pytest tests/examples/test_phone_bot.py
```
