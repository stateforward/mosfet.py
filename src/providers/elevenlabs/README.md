# mosfet-provider-elevenlabs

ElevenLabs speech provider package for `bot`. This package owns the ElevenLabs SDK dependency.

Run its tests from the workspace root:

```sh
uv run --package mosfet-provider-elevenlabs --group dev python -m pytest src/providers/elevenlabs/tests
```

The package exposes the provider import path:

```python
from bot.providers.elevenlabs import SpeechEncoder
```
