# listen_speak_bot example

Device-free stateforward.bot demo: **Listening** → mixed cognition → **Speaking**.

No phone or LiveKit. Heard/reply audio uses macOS `say` / `afconvert`. Judgment uses
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
2. Activate a bot with **no devices**, input=`Listening`, output=`Speaking`
3. Dispatch the WAV as `world.sound`
4. Offline VAD + fixed STT → transcript stimulus into cognition
5. Mercury 2 intuition (then Gemini reasoning if needed) selects `bot.ability.speaking.input`
6. Encode the reply with `say` into `assets/reply.wav` and print a summary

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
(default `https://api.inceptionlabs.ai/v1`).

Keys defined in `--env` / default `.env` override the process environment (so a local
example credential file wins over a stub key in the shell).

## What runs

| Stage | Implementation |
|---|---|
| Heard audio | macOS `say` → WAV |
| Input ability | `Listening` (`AlwaysVoiceDetector` + fixed transcript STT) |
| Cognition | Mercury 2 intuition + Gemini reasoning (`gemini-3.5-flash`) |
| Output ability | `Speaking` with `SayEncoder` (no `Speaker` device — avoids self-hearing) |
| Reply audio | `assets/reply.wav` |

## Layout

| Path | Role |
|---|---|
| `pyproject.toml` | Example package (`bot-example-listen-speak-bot`) |
| `.env.example` | Mercury + Gemini key / model template |
| `src/listen_speak_bot_example/` | Example app |
| `assets/` | Generated WAVs (gitignored) |
