# mosfet-provider-moonshine

On-device [Moonshine Voice](https://github.com/moonshine-ai/moonshine) provider package for
`stateforward.mosfet`. This package owns the `moonshine-voice` dependency and adapts Moonshine
speech-to-text / text-to-speech into the same ability contracts used by other voice providers
(`SpeechDecoder` for Listening speech decoding, a streaming `vocal.speech.SpeechEncoder` for Speaking).

The adapters are intentional thin I/O boundaries: Listening / Speaking HSM topology stays in
core `mosfet` (`mosfet.abilities.hearing.speech`, `mosfet.abilities.speaking`). This package does not
own bot focus, cognition, or phone lifecycle.

## Install / test

From the workspace root:

```sh
uv sync --package mosfet-provider-moonshine --group dev
uv run --package mosfet-provider-moonshine --group dev python -m pytest src/providers/moonshine/tests
```

Default tests inject fakes so CI does not download Moonshine models. Live smoke (downloads models
on first use):

```sh
# STT: WAV or raw int16 LE mono PCM
BOT_MOONSHINE_SMOKE_WAV=/path/to/sample.wav uv run --package mosfet-provider-moonshine python - <<'PY'
import asyncio
import os
import pathlib
from mosfet.providers.moonshine import SpeechDecoder

async def main() -> None:
    audio = pathlib.Path(os.environ["BOT_MOONSHINE_SMOKE_WAV"]).read_bytes()
    print((await SpeechDecoder().decode(audio)).decode("utf-8"))

asyncio.run(main())
PY

# TTS
uv run --package mosfet-provider-moonshine python - <<'PY'
import asyncio
from mosfet.abilities import vocal
from mosfet.providers.moonshine import SpeechEncoder

async def main() -> None:
    text = vocal.speech.Text()
    text.write("Hello from Moonshine.")
    text.close()
    chunks = [chunk async for chunk in SpeechEncoder(language="en-us").encode(text)]
    print("pcm_bytes", sum(len(chunk.pcm) for chunk in chunks))

asyncio.run(main())
PY
```

## Usage

```python
from mosfet.abilities import speaking
from mosfet.abilities.hearing.speech import SpeechDecoding
from mosfet.devices import audio
from mosfet.providers.moonshine import SpeechDecoder, SpeechEncoder

listening_speech = SpeechDecoding(decoder=SpeechDecoder(language="en"))
voice = speaking.Speaking(
    encoder=SpeechEncoder(language="en-us", voice="kokoro_af_heart", sample_rate_hz=24000),
    speaker=audio.Speaker(),
    voice_id="tts:kokoro-af-heart",
)
```

### Defaults

| Adapter | Moonshine API | Default |
|---------|---------------|---------|
| `SpeechDecoder` | `Transcriber.transcribe_without_streaming` | language `en` via `get_model_for_language` |
| `SpeechEncoder` | `TextToSpeech.synthesize` | language `en-us`, mono s16le PCM at `sample_rate_hz` (24000); reads the text to its end, one chunk aligned to the whole text |

Models are downloaded and cached by Moonshine on first use when `download=True` (TTS) /
`get_model_for_language` (STT).
