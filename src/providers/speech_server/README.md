# mosfet-provider-speech-server

Clients for a speech server's own wire protocols, so a bot can hear through served speech models next to its local
ones, through the same `hearing.voice` ability contracts, and be given a voice the server holds or designs. The OpenAI-compatible speech endpoints
(`/v1/audio/speech`, `/v1/audio/transcriptions`) are served by `mosfet-provider-openai-compat`.

| Client | Serves | Endpoint |
| --- | --- | --- |
| `speech_server.Diarizer` | `hearing.voice.VoiceDiarizer` | `POST /v1/diarize?sample_rate=16000` (for example Sortformer) |
| `speech_server.VoiceEmbedder` | the voice-identification `Classifier` (`voice.identification`) | `POST /v1/embed` (for example WeSpeaker) |
| `speech_server.AffectClassifier` | `hearing.voice.AffectClassifier` | `POST /v1/emotion?sample_rate=…` (for example emotion2vec+) |
| `speech_server.Catalog` | `vocal.voice.Catalog` | `GET /v1/voices` (for example anygame's lab TTS engine) |
| `speech_server.Designer` | `vocal.voice.Designer` | `POST /v1/voices` (for example anygame's lab TTS engine) |
| `speech_server.Transcription` | a streaming session (`hearing.speech.stream`) for `listening.Listening(transcription=)` | WebSocket `/v1/audio/stream` (for example Nemotron with turn detection) |

`Diarizer`, `VoiceEmbedder` and `AffectClassifier` are plain collaborators, not state machines: an ability composes
them. Each request carries raw s16le mono 16 kHz (the emotion endpoint also takes 24 and 48 kHz as they are); other
audio is converted first by `speech_server.resample.to_mono` (channels averaged, linear interpolation, standard
library only), in a worker thread.

- `Diarizer` clips each server segment from the *original* audio, at its own rate and channels, snapped to whole
  frames and clipped to the audio's end; the server's speaker index is dropped. An answer with no usable segment is
  `invalid_response`.
- `VoiceEmbedder` posts one request per segment, in order, and returns `VoiceEmbedding(embedding, model)` with the
  server's model name, which identification needs to keep one vector space per model. The server refuses audio under
  `MIN_SEGMENT_SECONDS` (0.3 s), so a shorter segment is never sent: it has no voiceprint (`None` in its place) and
  stays unidentified while the rest of the turn is identified (W3-13). An input of only short segments is not an error.
- `AffectClassifier` posts one voice segment, mono at its own rate when that is 16, 24 or 48 kHz and resampled to
  16 kHz otherwise, with `sample_rate` naming the rate sent, and returns `voice.Affect(label, scores, valence,
  confidence)`; the server's other fields (`snr_db`, `model`, timings) are dropped. A segment under
  `MIN_SEGMENT_SECONDS` is refused as `invalid_input` without a request. Composed as
  `listening.Listening(affect_classifier=…)`, it gives each text speech product the tone of its longest voice
  (`SpeechData.affect`).

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

## Voices

`speech_server.Catalog` and `speech_server.Designer` are the voice source `vocal.voice.Inception` gives a bot its
voice from, on the lab TTS engine's `/v1/voices` protocol. Both are plain collaborators. `source` is the name
composition gives this server; every ref they answer carries it, and the two must share it.

```python
from mosfet.abilities import vocal
from mosfet.providers import speech_server

inception = vocal.voice.Inception(
    traits=traits,
    languages=("en",),
    speaking=mouth,
    processor=inception_model,
    catalog=speech_server.Catalog(config.tts_url, source="lab", languages=("en",)),
    designer=speech_server.Designer(config.tts_url, source="lab"),
    memory=bot_memory,
)
```

- `Catalog.voices()` lists every voice the server holds (the filters are hints the server has no query for). Each
  becomes a `voice.Entry` named by its id, with the server's description and the `languages` composition declared
  for the engine (the server describes no languages or descriptors). A designed voice is labelled
  `{"custom": "true"}` and carries its preview URL, `/v1/voices/{id}/preview`.
- `Designer.create()` asks for exactly one voice (`{description, count: 1}`); the server keeps its reference clip,
  so the voice is permanent from the answer, which becomes `voice.Designed` with the request's description and the
  preview URL. `max_description` is the server's 300 characters; a longer description is refused as
  `invalid_input` without a request. The request's descriptors, languages and sample text are not part of this
  protocol.

Failures raise `vocal.voice.Error`: 404 `unsupported_input` (an engine that holds or designs no voices), 422
`invalid_input`, 503 or an unreachable server `unavailable`, 507 (voice storage full) `overloaded`, a deadline
`timeout`, an answer that is not the documented shape `invalid_response`; any other listing refusal is
`invalid_input` (4xx) or `unavailable` (5xx), and any other design refusal `design_failed`. `timeout_seconds` is 3 s
for a listing and 60 s for a design, which renders the voice. `bot.provider.speech_server.voices.duration{operation,
bot.outcome}` times every request (`operation` is `list` or `design`).

## Streaming transcription

`speech_server.Transcription(url="ws://host:8001")` is one streaming speech-to-text session: a state machine that
its controller attaches to and that hears one sound source. Listening makes one per source from a factory:

```python
ears = listening.Listening(
    transcription=lambda: speech_server.Transcription(url=config.stream_url),
    voice_classifier=speech_server.VoiceEmbedder(base_url=config.embed_url),
    registry=voice_registry,
)
```

While attached it stays connected to `/v1/audio/stream?sample_rate=16000&language=en&chunk=0.56&turns=1`,
reconnecting after any loss with a backoff from 1 s doubling to 30 s. It sends audio resampled to s16le mono
16 kHz in 50 ms frames (held, at most 2 s of it, until the server says `ready`), `{"type":"agent","speaking":…}`
whenever the bot starts or stops producing, and `{"type":"end"}` on detach. Each server frame becomes one
`hearing.speech.stream` output to the controller: `ready`, `speech_start`/`speech_end`, `partial`, `final`
(transcript), `turn_end` (the server's end-of-turn perception) and `interrupt` (overlap); `eot`, `resumed` and
`rejoin` are dropped and counted. Close 1013 is `overloaded`, 1003 `invalid_input`, a frame outside the protocol
`invalid_response`, a server that does not connect or say `ready` within `connect_timeout` (5 s) `timeout`, and
anything else (an `error` frame, a dropped socket) `unavailable`; each is a `stream.FailedEvent`, then the backoff.

Every time the controller hears is in seconds of the audio this session has received, as the contract says. The
server's own clock restarts at zero on each connection and never hears audio that was let go, so the session places
each connection's clock at the first audio sent on it and moves it on past audio let go before `ready`, past a full
send queue, or while detached.

`bot.hearing.speech.stream.session.active`, `.event.count{kind}`, `.failed.count{kind}` and
`.dropped.count{reason}` count sessions, outputs, failures and audio not sent; `bot.hearing.speech.stream.connect`
spans each connection.

## Requests

Inject an `httpx.AsyncClient` as `client=` to share connections; otherwise each request opens and closes its own.
`timeout_seconds` (3 s) bounds each request.

Failures raise `speech_server.RequestError` with a `kind` from the existing failure vocabulary: a 4xx answer is
`invalid_input`, a 5xx answer or an unreachable server `unavailable`, a deadline `timeout`, and an answer that is not a
bounded JSON object of the documented shape `invalid_response`. Messages never carry server text, audio or vectors.

`bot.provider.speech_server.request.duration{operation, bot.outcome}` times every request (`operation` is `diarize`,
`embed` or `emotion`).

## Tests

```sh
uv run --package mosfet-provider-speech-server --group dev python -m pytest src/providers/speech_server/tests -m "not live"
```

`tests/test_live.py` checks served endpoints when `BOT_SPEECH_SERVER_URL` (or the per-endpoint
`BOT_SPEECH_SERVER_EMBED_URL` / `BOT_SPEECH_SERVER_DIARIZE_URL` / `BOT_SPEECH_SERVER_EMOTION_URL`) is set, and a
streaming session when `BOT_SPEECH_SERVER_STREAM_URL` is set (`-m live`).
