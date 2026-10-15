# mosfet-provider-elevenlabs

ElevenLabs speech provider package for `bot`. This package owns its one ElevenLabs dependency, `websockets`, which
the speech encoder speaks the stream-input protocol over directly.

Run its tests from the workspace root:

```sh
uv run --package mosfet-provider-elevenlabs --group dev python -m pytest src/providers/elevenlabs/tests
```

The package exposes a streaming `vocal.speech.SpeechEncoder` for `speaking.Speaking`:

```python
from mosfet.providers.elevenlabs import SpeechEncoder

encoder = SpeechEncoder("voice-id", api_key="...", sample_rate_hz=24000)
```

It speaks through ElevenLabs' stream-input WebSocket
(`/v1/text-to-speech/{voice_id}/stream-input?model_id=...&output_format=pcm_<sample_rate_hz>`, over `websockets`).
Each piece of the utterance's text is sent as it is written, and the end of the text sends the end-of-input
message; ElevenLabs' server decides when it has enough text to generate (its `chunk_length_schedule`), so speech
starts while the reply is still being written. Nothing in the provider splits, joins or rewrites the text.

Audio arrives as base64 mono s16le PCM and is yielded as it arrives, re-cut to whole samples. Alignment is native:
each chunk's `text` is as many characters of the text read as the server aligned to it, in order, so the aligned
texts join to exactly the text read. The latest chunk is always held until the next one arrives (or the stream
ends), so a trailing remainder the server never aligned rides on the last chunk.

Closing the stream early closes the socket, which cancels the encoding at ElevenLabs. The constructor's voice id
is the voice the text is spoken in. Blank text is `invalid_input`; a socket that closes or cannot
connect is `unavailable`; a refusal from the server is `encode_failed`; PCM that ends mid-sample or a message
outside the protocol is `invalid_response`. All failures raise `encoding.Error`. Tests inject
`connect=` with a fake socket on the wire.
