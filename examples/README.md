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

`willy` is one bot for the whole house, that learns who people are: a voice nobody has named is a fact it is
shown, and whom a voice belongs to, what to call someone and whether to ask are its own choices. Its bot file,
`willy_example/__main__.py`, has one `bot.define`: a `computer.Computer` per room, each set on an
`environment.Object` and served by one `websocket.Server`, an ear and a mouth on the lab's streaming speech
server, a voice it is given when it is born, and `people.People` under the owner's household file. Every model
runs on the local lab server. See [`willy/README.md`](willy/README.md) for the settings, the household and where
safety sits:

```bash
uv sync --project examples/willy
set -a; . examples/willy/.env; set +a
examples/willy/.venv/bin/python -m willy_example --secrets-dir examples/willy/secrets
```

`minimal_bot` is the smallest bot: one device-free bot with a cognition on any OpenAI-compatible endpoint, run
with `python -m minimal_bot_example`. Its own project installs `mosfet` and the bot, so the command runs exactly
as documented; see [`minimal_bot/README.md`](minimal_bot/README.md):

```bash
uv sync --project examples/minimal_bot
OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_API_KEY=... TYPESAFE_API_KEY=... \
  examples/minimal_bot/.venv/bin/python -m minimal_bot_example
```

The other examples run as programs from the repository root:

```bash
uv run --project examples/listen_speak_bot python -m listen_speak_bot_example
set -a; . examples/phone_bot/.env; set +a
uv run --project examples/phone_bot python -m phone_bot_example
```

`listen_speak_bot` is its own example package (macOS only, device-free). Its bot file,
`listen_speak_bot_example/__main__.py`, has one `bot.define`: a bot with no devices, whose input is a real
**Listening** (MLX Audio Silero VAD and PyAnnote voice embeddings, or the lab's streaming speech server), whose
output is **Speaking**, and which has **Communication** with its **Conversation**, on one Gemini model. It renders
*Hey I'm Gabe how are you* with `say` and `afconvert`, and `hear` plays it to the bot once as `environment.sound`.
The Listening product carries a provider-neutral source identity, which the speech seed admits into
Conversation. The bot may select `communication.respond`, which Communication says through Speaking into a reply
WAV, or say nothing; the run reports `status: ok` or `status: incomplete`, and its observers never act on it.
The bot knows the traits the example gives it and the persona it writes of itself, and on the lab TTS engine
(`BOT_SPEECH_SERVER_TTS_URL`) it speaks in the voice it is born with. It needs `BOT_GEMINI_API_KEY` (or
`GEMINI_API_KEY` / `GOOGLE_API_KEY`). No LiveKit or phone device.

Run its tests from the repository root via the example project env (the root
env does not install the example package, so the test module skips there
with a reason instead of failing collection):

```bash
uv run --project examples/listen_speak_bot python -m pytest tests/examples/test_listen_speak_bot.py -q -m 'not live'
```

`phone_bot` is its own example package because it depends on real provider packages: LiveKit for the phone's
line, Gemini STT/TTS, MLX Audio's Silero VAD and pyannote voice identity, and an OpenAI-compatible model. Its bot
file, `phone_bot_example/__main__.py`, reads its `Settings` from the environment and has one `bot.define`: the core
`phone.Phone` on a LiveKit `PhoneService` line whose identity is the line's number, an ear and a mouth, and
Communication with its Conversation, on one model. `bot.run` holds the line until it is stopped, with a person
standing in front of the bot who says out loud each line typed at the terminal. Answering, declining, dialing and
hanging up are the bot's own; nothing in the example reads what is said to it. See
[`phone_bot/README.md`](phone_bot/README.md) for the local SFU, the second participant and the blackbox harnesses:

```bash
cd examples/phone_bot && docker compose up -d && cp .env.example .env   # then set the two keys
set -a; . ./.env; set +a
uv run python -m phone_bot_example --ready-file /tmp/phone-bot.ready
```

Unit tests cover its settings, its bot file, the person in its room and a phone bot's perception and capacity;
they are not live room proof.

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
