# phone_bot example

A stateforward.mosfet bot holding a telephone on a real LiveKit SFU: the core `Phone` on a LiveKit
`PhoneService` line, **local Silero VAD**, **local pyannote voice identity**, **off-device Gemini** STT/TTS, and
one model to think with, OpenAI's reasoning model (`gpt-5.6-terra` by default).

Its bot file, `src/phone_bot_example/__main__.py`, reads the owner's `Settings` from the environment and has one
`bot.define` that constructs every part inline: its cognition, its phone (the earpiece at the robot's ear, the
close-talk mouthpiece at its mouth, the LiveKit line its service), its ear (**Listening**), its mouth
(**Speaking**) and **Communication** with its **Conversation**. `bot.run` runs it until it is told to stop, with
the person standing in front of it as a shared actor and an observer that logs what it does.

## Local run

```bash
cd examples/phone_bot
docker compose up -d                 # a local LiveKit SFU (livekit-server --dev), if none is running
cp .env.example .env                 # then set BOT_OPENAI_API_KEY and BOT_GEMINI_API_KEY
set -a; . ./.env; set +a
uv run python -m phone_bot_example --ready-file /tmp/phone-bot.ready
```

The settings are read from the process environment; the process options (`--ready-file`, `--data-dir`,
`--secrets-dir`, the restart policy, each also a `MOSFET_*` variable) are `bot.run`'s. A missing or malformed
setting exits `1`, naming the variable, before anything starts. The ready file exists exactly while the bot runs.
`SIGTERM` or `Ctrl+C` hangs up and stops it, exit `0`. Stdout stays empty; logs go to stderr, at INFO, or DEBUG
with `BOT_VERBOSE=1`.

### Prerequisites

- Docker, or `livekit-server --dev`, for the local SFU
- `uv` + Python 3.13
- `BOT_OPENAI_API_KEY` (or `OPENAI_API_KEY`) and `BOT_GEMINI_API_KEY` (or `GEMINI_API_KEY` / `GOOGLE_API_KEY`)
- `DECISION_BASE_URL`, the decision server the bot's intuition decides on, which speaks the decisions wire
  format (anygame's decide, for example `http://127.0.0.1:8102`), with `DECISION_API_KEY` only for a server that
  takes a key, and the optional `DECISION_MAX_CONTEXT_TOKENS`, the decision model's context, which each decision
  request is fitted to (2048 for decide)
- macOS `say` and `afconvert`, for saying something to the bot in its room

### The line

`BOT_LIVEKIT_URL` puts the phone on a line. Its token is `BOT_LIVEKIT_TOKEN`, or one minted for the line's
identity and room from `BOT_LIVEKIT_API_KEY` and `BOT_LIVEKIT_API_SECRET` (the dev SFU's are `devkey` and
`secret`). Without a URL the phone has no line, and the bot can still be talked to in its room.

### Second participant

Mint a token for yourself and open LiveKit Meet with it, on **this machine** (a local SFU is not reachable from
other devices as-is):

```bash
uv run mint-livekit-token --identity human --room bot-phone-bot
# https://meet.livekit.io/custom?liveKitUrl=ws://127.0.0.1:7880&token=<that token>
```

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
uv run python -m phone_bot_example
# ...
Call Bob at 555-0142.              # type it, press enter
```

Every line you type is **said out loud** in the robot's room. A person (`person.Person`, a shared actor of the
run) stands a metre in front of it, with a mouth — the same `audio.Speaker` the robot uses for its own voice —
renders your words locally with macOS `say`, and broadcasts them through `Environment.broadcast` at 60 dB from
where they are standing. The robot hears them the way it hears anything: environment sound → Listening (VAD +
pyannote voice embeddings, identified as `voice:<id>` source refs by a voice registry) → the speech seed takes the
turn into its Conversation, whose turns Gemini decodes → cognition when the turn closes. Nothing is dispatched at
the body. The end of input (or no terminal) is an empty room: whoever was standing there walked away.

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

Any other `vocal.speech.SpeechEncoder` drops in for the person's voice — the Moonshine `SpeechEncoder` in
`src/providers/moonshine` is the same contract.

## Speech

VAD is local Silero (`mlx-community/silero-vad` via `mosfet-provider-mlx-audio`). Voice identity is local pyannote
speaker embeddings (`pyannote/wespeaker-voxceleb-resnet34-LM` via `mosfet-provider-pyannote`), identified by
an in-process voice `identification.Registry`, so Listening speech products carry `voice:<id>` `source_ids`
into Conversation. STT/TTS stay off-device Gemini (`gemini-3.5-flash` STT + `gemini-3.1-flash-tts-preview` TTS by
default; override with `BOT_GEMINI_STT_MODEL` / `BOT_GEMINI_TTS_MODEL` / `BOT_GEMINI_TTS_VOICE` /
`BOT_SILERO_VAD_MODEL` / `BOT_PYANNOTE_VOICE_IDENTITY_MODEL`). Live pyannote weights use the Hugging Face/pyannote
auth already required by that provider package. One rate decides the robot's voice: `BOT_GEMINI_OUTPUT_SAMPLE_RATE_HZ`
(24 kHz) is the TTS encoder's, Speaking's label and the LiveKit uplink's.

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | Local LiveKit SFU (`--dev`) |
| `.env.example` | Env template |
| `src/phone_bot_example/__main__.py` | The bot file: one `bot.define`, run with `bot.run` |
| `src/phone_bot_example/__init__.py` | `Settings`, `GeminiVoiceDecoder`, the token mint, the run's observer `log` |
| `src/phone_bot_example/person.py` | The person in the room |
| `scripts/mint_livekit_token.py` | JWT mint (no extra deps) |
| `scripts/phone-bot.sh` | Thin wrapper around `uv run python -m phone_bot_example` |

## Blackbox dual agent (LiveKit only)

Two agents meet **only** on the local SFU — no shared Environment and no direct `environment.sound` wiring.
Both sides use **off-device Gemini** for speech so dual talk does not load local MLX STT/TTS:

| Side | Process | Speech |
|---|---|---|
| Agent A | `python -m phone_bot_example` | Silero VAD + pyannote identity (local) + Gemini STT + Gemini TTS |
| Agent B | blackbox / `caller_agent` | Gemini TTS (default; `--tts say` for offline) |

```bash
# terminal 1
set -a; . ./.env; set +a
BOT_VERBOSE=1 uv run python -m phone_bot_example 2> /tmp/phone-bot-live.log

# terminal 2 — agent B publishes Gemini TTS audio via LiveKit RTC
uv run python scripts/blackbox_livekit_dual_agent.py \
  --line "Hello agent A, can you hear me through LiveKit?"
```

Or one-shot (spawns agent A with this example's `.env`):

```bash
uv run python scripts/blackbox_livekit_dual_agent.py --start-phone-bot \
  --line "Hello agent A, can you hear me through LiveKit?"
```

Requires `BOT_GEMINI_API_KEY` in `.env` (plus the OpenAI key for agent A's replies).
Verdict uses LiveKit media plus optional scrapes of `/tmp/phone-bot-live.log`
(DecodingSpeech). Silero VAD loads are expected; local Whisper/Qwen STT/TTS loads fail the harness.

## Blackbox two bots (caller and callee)

Two real phone bots, two lines on one exchange. Identities are the line numbers (normalized digits); no dial plan
is required:

```bash
uv run python scripts/blackbox_livekit_two_bots.py \
  --caller-number 555-0141 --callee-number 555-0142
```

Somebody walks up to the caller and says `Call Bob at 555-0142.` out loud; nobody says anything to
the callee. The harness writes the sentence to the caller's stdin once its ready file exists, and the bot
hears it as sound. Both join the same room as their number identities and can dial each other by
number. Whether the caller dials, and whether the callee answers, stay theirs to decide. The
verdict prints `dialed=` for the caller, `rang=` for the callee and `decoded=` for whoever turned
audio into words, and both sides must be audible to pass. A run where the caller never dials is
reported, not failed.

Each run leaves two OpenTelemetry JSONL files per bot in the record directory:
`<number>-otel-logs.jsonl` (generator requests) and `<number>-otel-spans.jsonl` (spans). The span
file is how a single sound is followed from the LiveKit frame it arrived on, through chunk
assembly, voice detection and speech decoding, to the phone delivering it — or to the reason it
was dropped instead. Disable both with `BOT_OTEL_DISABLED=1`.

The dual-agent script does not set those paths, so its phone bot writes `otel-logs.jsonl` and
`otel-spans.jsonl` at the example root instead.

## Tests (no LiveKit)

From the repository root, in this example's environment:

```bash
uv run --locked --project examples/phone_bot python -m pytest tests/examples/test_phone_bot.py -m "not live" -q
```

Each test writes its own small `bot.define` of this shape on a scripted model (`testing.Scripted`), with fakes at
the provider seams only. They pin what the bot hears and is offered — its phone, whether or not anybody spoke to
it — never what it chooses.
