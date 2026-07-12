# phone_bot example

stateforward.bot phone bot against a real LiveKit SFU: core `Phone` + LiveKit `PhoneService`,
with Gemini cognition (Interactions-ready Processing) and Gemini TTS/STT for a full stack.

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
- Optional for full `can_talk`: `BOT_GEMINI_API_KEY` in `.env`

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

When you join the room, LiveKit `participant_connected` maps to a phone **incoming call**
(`call_id=livekit:<identity>`). The phone **rings** as `world.sound` (`kind=ring`,
`source` = phone id) into **Listening** input — not raw `phone.ringing` into cognition.
After answer, room media can flow `ServiceAudioReceived` → speaker → `world.sound` →
Listening. Local speaker uplink is published to the LiveKit track (remote delivery is
suppressed to avoid echo).

## What “connected” means

| Summary field | Meaning |
|---|---|
| `livekit_room_audio_attempted` | `--connect-livekit` and URL+token present |
| `livekit_room_audio_connected` | Local audio track SID assigned after room join |
| `can_talk` | Room connected **and** Gemini API key configured for cognition + speech |

Room join alone does not require Gemini; full agent speech does.

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | Local LiveKit SFU (`--dev`) |
| `.env.example` | Env template |
| `scripts/mint_livekit_token.py` | JWT mint (no extra deps) |
| `scripts/phone-bot.sh` | Thin wrapper around `uv run phone-bot` |
| `src/phone_bot_example/` | Example app |

## Unit tests (no LiveKit)

From repo root:

```bash
uv run python -m pytest tests/examples/test_phone_bot.py
```
