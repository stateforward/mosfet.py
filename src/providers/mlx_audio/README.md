# bot-provider-mlx-audio

MLX Audio voice provider package for `bot`. This package owns the `mlx-audio`
dependency for local Apple Silicon text-to-speech, speech-to-text, voice
activity detection, and speaker diarization.

Run its tests from the workspace root:

```sh
uv run --package bot-provider-mlx-audio --group dev python -m pytest src/providers/mlx_audio/tests
```

These tests use injected fake MLX Audio runtimes so they can run without
downloading models. A live speech-to-text smoke test requires macOS on Apple
Silicon, the Darwin-only `mlx-audio` dependency, and first-run access to
download the configured local models. Live MLX Audio smoke is not part of the
default pytest suite; run it explicitly with a local audio file:

```sh
BOT_MLX_AUDIO_SMOKE_WAV=/path/to/sample.wav uv run --package bot-provider-mlx-audio python - <<'PY'
import asyncio
import os
import pathlib

from bot.providers.mlx_audio import SpeechDecoder


async def main() -> None:
    audio = pathlib.Path(os.environ["BOT_MLX_AUDIO_SMOKE_WAV"]).read_bytes()
    transcript = await SpeechDecoder().decode(audio)
    print(transcript.decode("utf-8"))


asyncio.run(main())
PY
```

The package exposes voice conversation adapters, an encoder for the existing
vocal speech ability, a decoder for the existing hearing speech ability, and
voice detection and diarization adapters for the existing hearing voice
abilities:

```python
from bot.abilities.vocal.speech import SpeechEncoding
from bot.abilities.hearing.speech import SpeechDecoding
from bot.abilities.hearing.voice import VoiceDetection, VoiceDiarization
from bot.providers.mlx_audio import (
    VoiceDecoder,
    VoiceEncoder,
    SpeechDecoder,
    SpeechEncoder,
    VoiceActivityClassifier,
    VoiceDiarizer,
)

voice_decoder = VoiceDecoder()
voice_encoder = VoiceEncoder()
speech = SpeechEncoding(encoder=SpeechEncoder())
transcription = SpeechDecoding(decoder=SpeechDecoder())
detection = VoiceDetection(classifier=VoiceActivityClassifier())
diarization = VoiceDiarization(classifier=VoiceDiarizer())
```
