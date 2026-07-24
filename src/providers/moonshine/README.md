# bot-provider-moonshine

On-device [Moonshine Voice](https://github.com/moonshine-ai/moonshine) provider package for
`stateforward.bot`. This package owns the `moonshine-voice` dependency and adapts Moonshine
speech-to-text / text-to-speech into the same ability contracts used by other voice providers
(`SpeechDecoder` for Listening speech decoding, `Encoder[bytes, bytes]` for Speaking TTS).

The adapters are intentional thin I/O boundaries: Listening / Speaking HSM topology stays in
core `bot` (`bot.abilities.hearing.speech`, `bot.abilities.speaking`). This package does not
own bot focus, cognition, or phone lifecycle.

## Install / test

From the workspace root:

```sh
uv sync --package bot-provider-moonshine --group dev
uv run --package bot-provider-moonshine --group dev python -m pytest src/providers/moonshine/tests
```

Default tests inject fakes so CI does not download Moonshine models. Live smoke (downloads models
on first use):

```sh
# STT: WAV or raw int16 LE mono PCM
BOT_MOONSHINE_SMOKE_WAV=/path/to/sample.wav uv run --package bot-provider-moonshine python - <<'PY'
import asyncio
import os
import pathlib
from bot.providers.moonshine import SpeechDecoder

async def main() -> None:
    audio = pathlib.Path(os.environ["BOT_MOONSHINE_SMOKE_WAV"]).read_bytes()
    print((await SpeechDecoder().decode(audio)).decode("utf-8"))

asyncio.run(main())
PY

# TTS
uv run --package bot-provider-moonshine python - <<'PY'
import asyncio
from bot.providers.moonshine import SpeechEncoder

async def main() -> None:
    wav = await SpeechEncoder(language="en-us").encode(b"Hello from Moonshine.")
    print("wav_bytes", len(wav))

asyncio.run(main())
PY
```

## Usage

```python
from bot.abilities.hearing.speech import SpeechDecoding
from bot.abilities.speaking import Speaking
from bot.providers.moonshine import SpeechDecoder, SpeechEncoder

listening_speech = SpeechDecoding(decoder=SpeechDecoder(language="en"))
speaking = Speaking(
    encoder=SpeechEncoder(language="en-us", voice="kokoro_af_heart"),
    media_type="audio/wav",
)
```

### Defaults

| Adapter | Moonshine API | Default |
|---------|---------------|---------|
| `SpeechDecoder` | `Transcriber.transcribe_without_streaming` | language `en` via `get_model_for_language` |
| `SpeechEncoder` | `TextToSpeech.synthesize` | language `en-us`, format `wav` |

Models are downloaded and cached by Moonshine on first use when `download=True` (TTS) /
`get_model_for_language` (STT).
