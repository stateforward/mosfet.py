# stateforward.mosfet Examples

Examples live outside `src` so they stay out of the distributable stateforward.mosfet package.
Each single-file example is a `uv` script with its own inline metadata, which keeps
example-only dependencies out of the root project and provider packages.

`builder_bot` is the first example: one bot, for one person, reached through a computer, that builds and tests
software on a sandbox bench of its own. Its bot file, `builder_bot_example/__main__.py`, defines it with one
`bot.define` that constructs every part inline: a `computer.Computer` served by a `websocket.Server`, a
`sandbox.Sandbox` on a rootless Docker or Podman socket, and the full cognition on one local OpenAI-compatible
model. The person sits at the computer's far side,
`builder-terminal`. Nothing is scripted: what the bot shows, runs, or leaves alone is its own choice. See
[`builder_bot/README.md`](builder_bot/README.md) for the bench image, the settings and the security posture:

```bash
uv sync --project examples/builder_bot
examples/builder_bot/.venv/bin/python -m builder_bot_example --secrets-dir examples/builder_bot/secrets
examples/builder_bot/.venv/bin/builder-terminal --secret-file examples/builder_bot/secrets/computer_token_secret
```

`minimal_bot` is the smallest bot: one device-free bot with a cognition on any OpenAI-compatible endpoint, run
with `python -m minimal_bot_example`. Its own project installs `mosfet` and the bot, so the command runs exactly
as documented; see [`minimal_bot/README.md`](minimal_bot/README.md):

```bash
uv sync --project examples/minimal_bot
OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_API_KEY=... \
  examples/minimal_bot/.venv/bin/python -m minimal_bot_example
```

The other examples run as programs from the repository root:

```bash
uv run --project examples/listen_speak_bot listen-speak-bot
uv run --project examples/phone_bot phone-bot-example
```

`listen_speak_bot` is its own example package (macOS only, device-free). It uses
`say` and `afconvert` to render *Hey I'm Gabe how are you* into a WAV, injects
that audio as `environment.sound` into a bot with no devices, attaches real **Listening**
(MLX Audio VAD + Whisper STT) and an acquired **Communication** ability, then runs **Mercury 2**
intuition (OpenAI-compatible) / Gemini **reasoning** (`gemini-3.5-flash`) and reports the cognition run.
The real Listening product carries a provider-neutral source identity into Communication, whose
native seeded behavior admits the typed speech into Conversation. Cognition can select
`communication.respond`, which routes through the injected Speaking port and produces a reply WAV.
The bot knows the traits the example gives it and the persona it writes of itself, and on the lab TTS engine
(`BOT_SPEECH_SERVER_TTS_URL`) it speaks in the voice it is given there at inception.
The run remains `status: incomplete` when cognition does not select a response. Cognition requires
`BOT_MERCURY_API_KEY`
(intuition) and `BOT_GEMINI_API_KEY` / `GEMINI_API_KEY` (reasoning). No LiveKit or
phone device. `--play` plays only the generated heard WAV in this slice.

Run its tests from the repository root via the example project env (the root
env does not install the example package, so the test module skips there
with a reason instead of failing collection):

```bash
uv run --project examples/listen_speak_bot python -m pytest tests/examples/test_listen_speak_bot.py -q -m 'not live'
```

Listening and Speaking remain body composition ports, not direct cognition tools. Communication
offers the semantic `communication.respond` action and owns the temporary route to Speaking.

`phone_bot` is its own example package because it depends on real provider
packages: LiveKit for the phone device and room audio, OpenAI-compatible
cognition, Gemini STT/TTS, and SQLite memory codecs. Its package metadata
declares `mosfet-provider-livekit`, `mosfet-provider-openai-compat`,
`mosfet-provider-gemini`, and `mosfet-provider-sqlite-memory`. It can read a
local provider env file passed with `--env`, constructs one phone-capable agent,
and reports which provider credentials were loaded without printing secret
values. The JSON summary separates constructed provider components from
`provider_chain_live_proof`, which is only true after an env-backed LiveKit
connection, remote audio turn, provider-backed voice response, and room-audio
publish all complete in one run.

Run it with a local provider env:

```bash
uv run --project examples/phone_bot phone-bot-example --env path/to/provider.env --json
```

Passing `--connect-livekit` explicitly attempts the configured LiveKit room
join; the default command only constructs the bot and reports configuration
state. The example reports `can_talk` when the LiveKit room is connected and
cognition + Gemini speech credentials are loaded. Unit tests cover package
contracts and configuration summaries; they are not live room proof.

Remote LiveKit PCM frames are batched into utterance-sized chunks at the LiveKit
provider boundary before Listening. A LiveKit SIP call-control gateway remains
out of scope for this example.

`phone_bot/learning_say_lesson.py` is part of the `phone_bot` project (same
`pyproject.toml`/`uv.lock`, same `--project examples/phone_bot`): it is a live
Learning proof in three phases (experience a real ring, hear a spoken lesson,
answer the next ring) that needs the phone device plus OpenAI cognition and
macOS `say`, so it lives with the project that already declares those
dependencies instead of piggybacking from `examples/`. Run it from the
repository root:

```bash
uv run --project examples/phone_bot examples/phone_bot/learning_say_lesson.py
```

It reads credentials from the repo `.env` or `examples/phone_bot/.env`
(`BOT_OPENAI_API_KEY` / `OPENAI_API_KEY`, optional `BOT_REFLECTION_MODEL`);
`.env` files stay untracked and `recordings/` plus per-example `.venv/` are
ignored (see `examples/phone_bot/.gitignore`).

Executable examples use this shebang:

```python
#!/usr/bin/env -S uv run --script
```

Each example should declare stateforward.mosfet as a local editable dependency:

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stateforward.mosfet",
# ]
# [tool.uv.sources]
# stateforward-mosfet = { path = "..", editable = true }
# ///
```

Add example-specific dependencies in that same block. For provider examples, depend
on the provider package explicitly instead of pulling provider SDKs into core stateforward.mosfet:

```python
# dependencies = [
#   "stateforward.mosfet",
#   "mosfet-provider-livekit",
#   "mosfet-provider-elevenlabs",
# ]
# [tool.uv.sources]
# stateforward-mosfet = { path = "..", editable = true }
# mosfet-provider-livekit = { path = "../src/providers/livekit", editable = true }
# mosfet-provider-elevenlabs = { path = "../src/providers/elevenlabs", editable = true }
```

For larger examples, create a directory under `examples/` with its own
`pyproject.toml`. Keep those projects out of the root workspace unless the example
should run as part of normal workspace verification.
