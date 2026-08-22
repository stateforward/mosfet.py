# stateforward.bot Examples

Examples live outside `src` so they stay out of the distributable stateforward.bot package.
Each single-file example is a `uv` script with its own inline metadata, which keeps
example-only dependencies out of the root project and provider packages.

Run an example from the repository root:

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
The run remains `status: incomplete` when cognition does not select a response. Cognition requires
`BOT_MERCURY_API_KEY`
(intuition) and `BOT_GEMINI_API_KEY` / `GEMINI_API_KEY` (reasoning). No LiveKit or
phone device. `--play` plays only the generated heard WAV in this slice.

Listening and Speaking remain body composition ports, not direct cognition tools. Communication
offers the semantic `communication.respond` action and owns the temporary route to Speaking.

`phone_bot` is its own example package because it depends on real provider
packages: LiveKit for the phone device and room audio, OpenAI-compatible
cognition, Gemini STT/TTS, and SQLite memory codecs. Its package metadata
declares `bot-provider-livekit`, `bot-provider-openai-compat`,
`bot-provider-gemini`, and `bot-provider-sqlite-memory`. It can read a
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

Executable examples use this shebang:

```python
#!/usr/bin/env -S uv run --script
```

Each example should declare stateforward.bot as a local editable dependency:

```python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "stateforward.bot",
# ]
# [tool.uv.sources]
# stateforward-bot = { path = "..", editable = true }
# ///
```

Add example-specific dependencies in that same block. For provider examples, depend
on the provider package explicitly instead of pulling provider SDKs into core stateforward.bot:

```python
# dependencies = [
#   "stateforward.bot",
#   "bot-provider-livekit",
#   "bot-provider-elevenlabs",
# ]
# [tool.uv.sources]
# stateforward-bot = { path = "..", editable = true }
# bot-provider-livekit = { path = "../src/providers/livekit", editable = true }
# bot-provider-elevenlabs = { path = "../src/providers/elevenlabs", editable = true }
```

For larger examples, create a directory under `examples/` with its own
`pyproject.toml`. Keep those projects out of the root workspace unless the example
should run as part of normal workspace verification.
