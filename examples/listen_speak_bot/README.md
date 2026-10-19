# listen_speak_bot example

A bot with no devices hears one recording and may answer it out loud. Its bot file,
`src/listen_speak_bot_example/__main__.py`, has one `bot.define` that constructs every part inline: the bot's
traits, its cognition, its ear (**Listening**), its mouth (**Speaking**) and **Communication**, which owns its
**Conversation**. Then `hear` plays the recording to it once, and the run is reported.

Listening and Speaking are the body's input and output. They are not cognition's tools: cognition may select
the semantic `communication.respond` action, and Communication says it through Speaking. Whether it does is the
bot's own choice. The run reports `status: ok` when it said something and `status: incomplete` when it chose to
say nothing.

No phone or LiveKit. The heard audio is rendered with macOS `say` / `afconvert`. The bot thinks with one model,
**Gemini** (`gemini-3.5-flash` by default).

The bot has a self. Its owner (this example) gives it **traits** (`curious`, `warm`, `playful`, or each `--trait`
given); it writes a **persona** of itself on top of them, which only its reflection may revise, and it sees both
on every turn. Its memory is kept in `assets/memory.sqlite3`, so its persona and its voice are the same on every
run; delete that file to make a new bot.

## Local run

From the repository root:

```bash
uv run --project examples/listen_speak_bot python -m listen_speak_bot_example
```

Or from this directory:

```bash
cd examples/listen_speak_bot
cp .env.example .env   # then set BOT_GEMINI_API_KEY
uv run python -m listen_speak_bot_example
```

The settings are read from the process environment, and the env file's assignments win: `--env PATH`, or this
example's `.env` when it exists. A missing key exits `1`, naming the variable, before anything is rendered or
started; a bad option exits `2`.

That command will:

1. Render *Hey I'm Gabe how are you* with `say` into `assets/hey_gabe.wav`
2. Define the bot with **no devices**: input `Listening`, output `Speaking`, and `Communication`
3. Start it, and play the WAV and then half a second of silence to it as `environment.sound`
4. MLX Audio Silero VAD + the real PyAnnote speaker classifier → a voice registry → a `voice:<id>` source ref
   with the speech product; the speech seed takes it into a Conversation, whose turns MLX Whisper decodes
5. Give the turn to its cognition on Gemini, with its traits and persona shown every turn
6. Wait until that turn is decided, and anything it chose to say has been said; then stop the bot
7. Report `status: ok` and the reply WAV when it said something; otherwise `status: incomplete`

What happened is recorded by `Run`, whose observers (`bot.observe`) watch the bot and never act on it: the
conversation turns that reached its cognition, the turns it decided, and the responses it selected. What it said
is recorded where it is rendered: the example's encoders tell the run each reply they rendered in full. The
summary carries counts and outcomes only, never what was heard, said or recognized. The run waits at most 120
seconds; a run that does not settle exits `1`.

The PyAnnote adapter loads `pyannote/wespeaker-voxceleb-resnet34-LM` lazily and embeds each heard voice; a
voice `identification.Registry` (WeSpeaker's 0.50/0.05/0.65 operating point over an in-process store) turns
each embedding into an opaque voice id, and the speech product carries `voice:<id>` as its source. The
embedding never leaves Listening, and nothing fabricates human-readable identities. The bot's own voice is
enrolled from what it says, and the bot learns the id hearing gives it, so its echo is stamped `voice:<that id>`
and never admitted as a turn. The command exits nonzero for the explicit incomplete status: that is the bot's
choice, not a claim that a part failed.

Options:

```bash
uv run python -m listen_speak_bot_example --play
uv run python -m listen_speak_bot_example --json
uv run python -m listen_speak_bot_example --env path/to/provider.env
uv run python -m listen_speak_bot_example --gemini-model gemini-3.5-flash
uv run python -m listen_speak_bot_example --assets-dir /tmp/listen-speak
uv run python -m listen_speak_bot_example --trait curious --trait "patient with children"
uv run python -m listen_speak_bot_example --verbose
```

### Prerequisites

- macOS with `say` and `afconvert`
- `uv` + Python 3.13
- `BOT_GEMINI_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY`
- `TYPESAFE_API_KEY` for hosted TypeSafe, the decision model the bot's intuition decides on; or
  `TYPESAFE_BASE_URL`, a local system_one server (for example `http://127.0.0.1:8102`), which takes no key
  and is never sent `TYPESAFE_API_KEY`

Optional: `BOT_GEMINI_MODEL` (or `GEMINI_MODEL`, default `gemini-3.5-flash`) and `BOT_SILERO_VAD_MODEL`
(default `mlx-community/silero-vad`; `BOT_VAD_MODEL` and `SILERO_VAD_MODEL` are also accepted).
Set `BOT_STT_MODEL` (or `BOT_WHISPER_MODEL`, `STT_MODEL`, or `WHISPER_MODEL`) to override the
MLX Audio Whisper model; when unset, the provider's default is used.
Set `BOT_PYANNOTE_VOICE_IDENTITY_MODEL` (or `PYANNOTE_VOICE_IDENTITY_MODEL`) to override the
PyAnnote speaker-embedding model; when unset, `pyannote/wespeaker-voxceleb-resnet34-LM` is used.
Configure any required Hugging Face/pyannote model access in the provider runtime; credentials are
never stored in this example's defaults.

Lab wiring (each optional): `BOT_SPEECH_SERVER_STREAM_URL` (for example `ws://192.168.2.2:8001`) hears
through a speech server's streaming transcription instead of Silero VAD, the server's end of turn making each text
speech product; `BOT_SPEECH_SERVER_URL` (for example `http://192.168.2.2:8003`) embeds voices there instead of
PyAnnote; `BOT_SPEECH_SERVER_TTS_URL` (for example `http://192.168.2.2:8002`) speaks through the lab TTS
engine instead of `say`.

On the lab TTS engine the bot speaks in the voice it is born with (`bot.voice`). The first time it starts, its
model conceives one voice from its traits alone, the engine designs it (`speech_server.Designer`,
`POST /v1/voices`), and the bot's memory keeps it; Speaking builds its encoder once, in that voice (`LabVoice`:
`openai_compat.SpeechEncoder` with the lab's `LAB_EXPRESSIVENESS` and `openai_compat.ANGLE_TAGS`, the voice's
description written before any direction). Every start after, the bot recalls that voice and checks the engine
still holds it (`speech_server.Catalog`, `GET /v1/voices`); a voice the engine has lost leaves the bot unvoiced
rather than speaking in another. Speech asked for before the voice is given waits for it. How the bot says each
part of a reply (a laugh before its words, a whisper, an emotion) is its own choice, reported per part on
Speaking's output.

### Lab listen check (manual, not CI)

`scripts/listen_check.py` renders one two-part reply per sound the lab declares (`laugh`, `giggle`, `chuckle`,
`gasp`, `sigh`), with the sound before and after its words, plus a whispered one, each to its own WAV under
`assets/listen_check/`, through exactly the encoder the bot speaks with. Listen to them; nothing judges the audio.

```bash
uv run --project examples/listen_speak_bot python examples/listen_speak_bot/scripts/listen_check.py \
  --url http://192.168.2.2:8002 --voice warm
```

## Tests

From the repository root, in this example's environment (the root environment does not install the example, so
the test module skips there with a reason):

```bash
uv run --locked --project examples/listen_speak_bot python -m pytest tests/examples/test_listen_speak_bot.py -m "not live" -q
```

Each test writes its own small `bot.define` of this shape on a scripted model (`testing.Scripted`), with fakes at
the provider seams only. They pin what the bot hears and is offered, and that what it selects is said; never what
it chooses.

## What runs

| Stage | Implementation |
|---|---|
| Heard audio | macOS `say` → WAV |
| Input ability | `Listening` (MLX Audio Silero VAD + PyAnnote voice embeddings identified as `voice:<id>` refs) |
| Cognition | One Gemini model (`gemini-3.5-flash`), intuition on TypeSafe (`typesafe.Processor`, a decision model), autonomy with the speech seed, associations, consolidation, routines |
| Conversation STT | MLX Audio Whisper, after identity-correlated admission |
| Output route | `communication.respond` → Communication → Speaking |
| Reply audio | `assets/reply.wav`, when the bot says something |
| Self | Owner-given traits and a self-written persona; on the lab TTS, the voice it is born with |

## Layout

| Path | Role |
|---|---|
| `pyproject.toml` | Example package (`bot-example-listen-speak-bot`) |
| `.env.example` | Gemini key, model and lab template |
| `src/listen_speak_bot_example/__main__.py` | The bot file: one `bot.define`, run once with `hear` |
| `src/listen_speak_bot_example/__init__.py` | `Settings`, the encoders, `Run` and `hear` |
| `scripts/listen_check.py` | Manual lab listen check |
| `assets/` | Generated WAVs and the bot's `memory.sqlite3` (gitignored) |
