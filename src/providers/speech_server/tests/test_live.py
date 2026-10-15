"""Live check of ``speech_server.VoiceEmbedder`` and ``Diarizer`` against served speech-server endpoints.

``BOT_SPEECH_SERVER_URL`` names the server; ``BOT_SPEECH_SERVER_EMBED_URL`` and ``BOT_SPEECH_SERVER_DIARIZE_URL``
override it per endpoint when they are served on different ports. Diarization needs real speech, so it runs only
with ``BOT_SPEECH_SERVER_WAV`` (a 16-bit PCM WAV file)::

    BOT_SPEECH_SERVER_EMBED_URL=http://lab.local:8003 \\
    BOT_SPEECH_SERVER_DIARIZE_URL=http://lab.local:8005 \\
    BOT_SPEECH_SERVER_WAV=/path/to/two-speakers.wav \\
      uv run --package mosfet-provider-speech-server --group dev python -m pytest \\
      src/providers/speech_server/tests/test_live.py -m live -v -s

What the servers answered is printed, never pinned.
"""

from __future__ import annotations

import array
import asyncio
import math
import os
import pathlib
import sys
import wave

import pytest

from mosfet.abilities.hearing import voice
from mosfet.providers import speech_server

pytestmark = pytest.mark.live


def _url(name: str) -> str:
    url = os.environ.get(name) or os.environ.get("BOT_SPEECH_SERVER_URL")
    if not url:
        pytest.skip(f"{name} / BOT_SPEECH_SERVER_URL not set")
    return url


def test_served_voice_embedder_returns_a_named_finite_vector() -> None:
    rate = 16_000
    tone = array.array("h", (round(8_000 * math.sin(2 * math.pi * 220 * index / rate)) for index in range(rate)))
    if sys.byteorder == "big":
        tone.byteswap()
    segment = voice.VoiceSegment(
        audio=tone.tobytes(),
        media_type="audio/pcm",
        sample_rate_hz=rate,
        channels=1,
        start_seconds=0.0,
        end_seconds=1.0,
    )
    embedder = speech_server.VoiceEmbedder(base_url=_url("BOT_SPEECH_SERVER_EMBED_URL"))

    [embedding] = asyncio.run(embedder.classify(voice.identification.InputData(segments=(segment,)))).embeddings

    assert embedding is not None
    print(f"model={embedding.model} dimensions={len(embedding.embedding)}")
    assert embedding.model
    assert all(math.isfinite(value) for value in embedding.embedding)


def test_served_diarizer_returns_ordered_segments_of_the_input() -> None:
    url = _url("BOT_SPEECH_SERVER_DIARIZE_URL")
    path = os.environ.get("BOT_SPEECH_SERVER_WAV")
    if not path:
        pytest.skip("BOT_SPEECH_SERVER_WAV not set")
    with wave.open(str(pathlib.Path(path)), "rb") as source:
        assert source.getsampwidth() == 2
        data = voice.diarization.InputData(
            audio=source.readframes(source.getnframes()),
            media_type="audio/pcm",
            sample_rate_hz=source.getframerate(),
            channels=source.getnchannels(),
        )

    output = asyncio.run(speech_server.Diarizer(base_url=url).classify(data))

    for segment in output.segments:
        print(f"{segment.start_seconds:.2f}-{segment.end_seconds:.2f}")
    assert [segment.start_seconds for segment in output.segments] == sorted(
        segment.start_seconds for segment in output.segments
    )


def test_served_stream_session_says_ready() -> None:
    """``BOT_SPEECH_SERVER_STREAM_URL`` (``ws://host:8001``): a session connects and the server says ready."""

    import hsm

    from mosfet.abilities.hearing.speech import stream
    from mosfet.define import define
    from mosfet.protocols import attachment
    from mosfet.start import started

    url = os.environ.get("BOT_SPEECH_SERVER_STREAM_URL")
    if not url:
        pytest.skip("BOT_SPEECH_SERVER_STREAM_URL not set")

    class Listener(hsm.Instance):
        ready: asyncio.Event

        def __init__(self) -> None:
            super().__init__()
            self.ready = asyncio.Event()

        @staticmethod
        def _hear(ctx: hsm.Context, instance: "Listener", event: hsm.Event[object]) -> None:
            del ctx
            print(f"session said {event.name}")
            if event.name == stream.ReadyEvent.name:
                instance.ready.set()

        model: hsm.Model = define(
            "Listener",
            hsm.initial(hsm.target("/Listener/hearing")),
            hsm.state("hearing", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_hear))),
        )

    async def run() -> None:
        ctx = hsm.Context()
        session = speech_server.Transcription(url=url)
        _ = await started(ctx, session, session.model)
        listener = Listener()
        _ = await started(ctx, listener, listener.model)
        _ = await session.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=listener)))
        try:
            _ = await asyncio.wait_for(listener.ready.wait(), timeout=10.0)
        finally:
            _ = await session.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=listener)))
            await hsm.stop(session)

    asyncio.run(run())
