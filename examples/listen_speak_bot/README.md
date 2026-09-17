# listen_speak_bot example

Device-free stateforward.mosfet integration status: **Listening** is attached to an acquired
**Communication** ability, which owns nested **Conversation** and routes selected responses to
**Speaking**; the demo is incomplete only when cognition selects no response.

Listening and Speaking are body composition ports. Communication is an acquired ability and owns
the nested Conversation actor. They are not direct cognition tools; cognition selects the semantic
`communication.respond` action and Communication routes that action to Speaking.

No phone or LiveKit. Heard audio uses macOS `say` / `afconvert`. Cognition uses
**Mercury 2** intuition (OpenAI-compatible, Inception) and **Gemini** reasoning /
reflection (`gemini-3.5-flash` by default).

## Local run

From the repository root:

```bash
uv run --project examples/listen_speak_bot listen-speak-bot
```

Or from this directory:

```bash
cd examples/listen_speak_bot
cp .env.example .env   # then set BOT_MERCURY_API_KEY + BOT_GEMINI_API_KEY
uv run listen-speak-bot
```

Repo-root `.env` is also loaded (example `.env` wins on conflicts).

That command will:

1. Render *Hey I'm Gabe how are you* with `say` into `assets/hey_gabe.wav`
2. Activate a bot with **no devices**, input=`Listening`, and an internal `Speaking` port
3. Dispatch the WAV as `environment.sound`
4. MLX Audio Silero VAD + the real PyAnnote speaker classifier → provider-neutral source embedding
   with the speech product; Conversation retains the MLX Audio Whisper STT path
5. Configure Autonomy with Communication's seeded speech-admission behavior
6. Run Mercury 2 intuition (then Gemini reasoning if needed) over the available cognition inputs
7. Report `status: ok` and the reply WAV when cognition selects `communication.respond`; otherwise
   report `status: incomplete` because no response was selected

The PyAnnote adapter loads `pyannote/wespeaker-voxceleb-resnet34-LM` lazily and emits embeddings as
Conversation `source_ids`; it does not fabricate human-readable identities. The example therefore
has the identity capability needed for Conversation admission, but this source-only run does not
claim a reply until run-time model access selects `communication.respond`. The CLI exits nonzero for
the explicit incomplete status; this is a cognition-selection result, not a claim that PyAnnote
lacks an identity path.

Optional flags:

```bash
uv run listen-speak-bot --play
uv run listen-speak-bot --json
uv run listen-speak-bot --env path/to/provider.env
uv run listen-speak-bot --gemini-model gemini-3.5-flash
uv run listen-speak-bot --assets-dir /tmp/listen-speak
```

Also available as `listen-speak-bot-example` and `python -m listen_speak_bot_example`.

### Prerequisites

- macOS with `say` and `afconvert`
- `uv` + Python 3.13
- `BOT_MERCURY_API_KEY` (or `MERCURY_API_KEY` / `INCEPTION_API_KEY`) for intuition
- `BOT_GEMINI_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` for reasoning/reflection

Optional: `BOT_MERCURY_MODEL` (default `mercury-2`), `BOT_MERCURY_BASE_URL`
(default `https://api.inceptionlabs.ai/v1`), and `BOT_SILERO_VAD_MODEL`
(default `mlx-community/silero-vad`). `BOT_VAD_MODEL` and `SILERO_VAD_MODEL` are also accepted.
Set `BOT_STT_MODEL` (or `BOT_WHISPER_MODEL`, `STT_MODEL`, or `WHISPER_MODEL`) to override the
MLX Audio Whisper model; when unset, the provider's default is used.
Set `BOT_PYANNOTE_VOICE_IDENTITY_MODEL` (or `PYANNOTE_VOICE_IDENTITY_MODEL`) to override the
PyAnnote speaker-embedding model; when unset, `pyannote/wespeaker-voxceleb-resnet34-LM` is used.
Configure any required Hugging Face/pyannote model access in the provider runtime; credentials are
never stored in this example's defaults.

Keys defined in `--env` / default `.env` override the process environment (so a local
example credential file wins over a stub key in the shell).

## What runs

| Stage | Implementation |
|---|---|
| Heard audio | macOS `say` → WAV |
| Input ability | `Listening` (MLX Audio Silero VAD + PyAnnote source embeddings) |
| Cognition | Communication autonomy + Mercury 2 intuition + Gemini reasoning (`gemini-3.5-flash`) |
| Conversation STT | MLX Audio Whisper, after identity-correlated admission |
| Output route | `communication.respond` → Communication → injected Speaking |
| Reply audio | WAV when cognition selects `communication.respond` |

## Layout

| Path | Role |
|---|---|
| `pyproject.toml` | Example package (`bot-example-listen-speak-bot`) |
| `.env.example` | Mercury + Gemini key / model template |
| `src/listen_speak_bot_example/` | Example app |
| `assets/` | Generated WAVs (gitignored) |
