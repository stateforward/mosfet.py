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
    text.write(vocal.speech.Piece(part=0, words="Hello from Moonshine."))
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
from mosfet.providers.moonshine import SpeechDecoder, SpeechEncoder

listening_speech = SpeechDecoding(decoder=SpeechDecoder(language="en"))
# With no speaker, Speaking's mouth is a fresh audio.Speaker(), the bot's own voice.
voice = speaking.Speaking(
    encoder=SpeechEncoder(language="en-us", voice="kokoro_af_heart", sample_rate_hz=24000),
)
```

### Voices

`Catalog(entries)` is the `vocal.voice.Catalog` an on-device bot is given its voice from at
inception. Moonshine has no listing to ask, so composition declares the voices it loads as
canonical `voice.Entry`s of one source, each keyed by Moonshine's own voice name; Speaking's
encoder factory then builds the encoder with that key as its `voice`. Composition writes what
each voice is: a Kokoro voice names its language and gender in the two letters after its
prefix (`kokoro_af_heart` is American English, feminine), so it becomes the entry's
`languages` and `descriptors.gender`. The filters `voices()` takes are hints; every declared
voice is listed, in order. A catalog of no voices, of two sources or with a voice twice is
refused (`ValueError`). The bot is born only as one of the voices its inception is `allowed`, by key.

```python
from mosfet.abilities import speaking, vocal
from mosfet.providers import moonshine

voices = moonshine.Catalog(
    (
        vocal.voice.Entry(
            ref=vocal.voice.Ref(source="moonshine", key="kokoro_af_heart"),
            name="kokoro_af_heart",
            languages=("en-US",),
            descriptors=vocal.voice.Descriptors(gender="feminine"),
        ),
    )
)
mouth = speaking.Speaking(
    encoder=lambda key, description: moonshine.SpeechEncoder(language="en-us", voice=key),
)
inception = vocal.voice.Inception(
    traits=traits,
    languages=("en",),
    speaking=mouth,
    processor=inception_model,
    catalog=voices,
    memory=bot_memory,
    allowed=("kokoro_af_heart",),
)
```

### Defaults

| Adapter | Moonshine API | Default |
|---------|---------------|---------|
| `SpeechDecoder` | `Transcriber.transcribe_without_streaming` | language `en` via `get_model_for_language` |
| `SpeechEncoder` | `TextToSpeech.synthesize` | language `en-us`, mono s16le PCM at `sample_rate_hz` (24000); applies no expression, so reads the words of every part to their end as one run, one chunk aligned to all of them |

Models are downloaded and cached by Moonshine on first use when `download=True` (TTS) /
`get_model_for_language` (STT).
