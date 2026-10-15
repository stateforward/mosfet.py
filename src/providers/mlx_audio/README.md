# mosfet-provider-mlx-audio

MLX Audio voice provider package for `bot`. This package owns the `mlx-audio`
dependency for local Apple Silicon text-to-speech, speech-to-text, voice
activity detection, and speaker diarization.

Run its tests from the workspace root:

```sh
uv run --package mosfet-provider-mlx-audio --group dev python -m pytest src/providers/mlx_audio/tests
```

These tests use injected fake MLX Audio runtimes so they can run without
downloading models. A live speech-to-text smoke test requires macOS on Apple
Silicon, the Darwin-only `mlx-audio` dependency, and first-run access to
download the configured local models. Live MLX Audio smoke is not part of the
default pytest suite; run it explicitly with a local audio file:

```sh
BOT_MLX_AUDIO_SMOKE_WAV=/path/to/sample.wav uv run --package mosfet-provider-mlx-audio python - <<'PY'
import asyncio
import os
import pathlib

from mosfet.providers.mlx_audio import SpeechDecoder


async def main() -> None:
    audio = pathlib.Path(os.environ["BOT_MLX_AUDIO_SMOKE_WAV"]).read_bytes()
    transcript = await SpeechDecoder().decode(audio)
    print(transcript.decode("utf-8"))


asyncio.run(main())
PY
```

The package exposes voice conversation adapters, a streaming speech encoder for
`speaking.Speaking`, a decoder for the existing hearing speech ability, and
voice detection and diarization adapters for the existing hearing voice
abilities:

```python
from mosfet.abilities.hearing.speech import SpeechDecoding
from mosfet.abilities.hearing.voice import VoiceDetection, VoiceDiarization
from mosfet.providers.mlx_audio import (
    VoiceDecoder,
    VoiceEncoder,
    SpeechDecoder,
    SpeechEncoder,
    VoiceActivityClassifier,
    VoiceDiarizer,
)

voice_decoder = VoiceDecoder()
voice_encoder = VoiceEncoder(speech_encoder=SpeechEncoder())
speech_encoder = SpeechEncoder(sample_rate_hz=24000)  # inject into speaking.Speaking(encoder=...)
transcription = SpeechDecoding(decoder=SpeechDecoder())
detection = VoiceDetection(classifier=VoiceActivityClassifier())
diarization = VoiceDiarization(classifier=VoiceDiarizer())
```

`SpeechEncoder` reads the utterance's text to its end (the model's `generate` takes
the whole text; joining the pieces in order is its only buffering), then streams
each chunk the model generates as mono s16le PCM. Alignment is emulated over the
whole text: every chunk carries `""` but the last, which carries the whole text,
so one generated chunk is held until the next arrives. It
declares its `sample_rate_hz` up front and refuses a model that generates at
another rate (`invalid_response`). Closing the stream between chunks stops the
generation; a chunk already being generated finishes first, because MLX native
work cannot be interrupted.
