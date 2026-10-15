# mosfet-provider-speech-server

Clients for a speech server's own wire protocols, so a bot can hear through served speech models next to its local
ones, through the same `hearing.voice` ability contracts. The OpenAI-compatible speech endpoints
(`/v1/audio/speech`, `/v1/audio/transcriptions`) are served by `mosfet-provider-openai-compat`.

| Client | Serves | Endpoint |
| --- | --- | --- |
| `speech_server.Diarizer` | `hearing.voice.VoiceDiarizer` | `POST /v1/diarize?sample_rate=16000` (for example Sortformer) |
| `speech_server.VoiceEmbedder` | the voice-identification `Classifier` (`voice.identification`) | `POST /v1/embed` (for example WeSpeaker) |

Both are plain collaborators, not state machines: an ability calls them directly. Each request carries raw s16le
mono 16 kHz; other audio is converted first by `speech_server.resample.to_mono` (channels averaged, linear
interpolation, standard library only), in a worker thread.

- `Diarizer` clips each server segment from the *original* audio, at its own rate and channels, snapped to whole
  frames and clipped to the audio's end; the server's speaker index is dropped. An answer with no usable segment is
  `invalid_response`.
- `VoiceEmbedder` posts one request per segment, in order, and returns `VoiceEmbedding(embedding, model)` with the
  server's model name, which identification needs to keep one vector space per model. The server refuses audio under
  `MIN_SEGMENT_SECONDS` (0.3 s), so a shorter segment is never sent: it has no voiceprint (`None` in its place) and
  stays unidentified while the rest of the turn is identified (W3-13). An input of only short segments is not an error.

```python
from mosfet.abilities import identification, listening
from mosfet.providers import speech_server

ears = listening.Listening(
    voice_activity_classifier=vad,
    voice_diarizer=speech_server.Diarizer(base_url=config.diarize_url),
    voice_classifier=speech_server.VoiceEmbedder(base_url=config.embed_url),
    registry=identification.Registry(
        identifier=identification.Matcher(
            store=identification.InProcessStore(max_ids=1000),
            policy=identification.MatchPolicy(threshold=0.50, margin=0.05, refine_threshold=0.65),
        ),
        kind="voice",
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=1000,
    ),
)
```

The embedder's vectors stay inside Listening: the registry turns each into a voice id, and heard speech carries
`voice:<id>` refs. With a file memory database, `sqlite_memory.IdentityStore(engine=...)` replaces the in-process
store and shares the memory engine.

Inject an `httpx.AsyncClient` as `client=` to share connections; otherwise each request opens and closes its own.
`timeout_seconds` (3 s) bounds each request.

Failures raise `speech_server.RequestError` with a `kind` from the existing failure vocabulary: a 4xx answer is
`invalid_input`, a 5xx answer or an unreachable server `unavailable`, a deadline `timeout`, and an answer that is not a
bounded JSON object of the documented shape `invalid_response`. Messages never carry server text, audio or vectors.

`bot.provider.speech_server.request.duration{operation, bot.outcome}` times every request.

## Tests

```sh
uv run --package mosfet-provider-speech-server --group dev python -m pytest src/providers/speech_server/tests -m "not live"
```

`tests/test_live.py` checks served endpoints when `BOT_SPEECH_SERVER_URL` (or the per-endpoint
`BOT_SPEECH_SERVER_EMBED_URL` / `BOT_SPEECH_SERVER_DIARIZE_URL`) is set (`-m live`).
