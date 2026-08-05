from __future__ import annotations

from bot.abilities.hearing import voice
from bot.devices import phone

import asyncio
import collections.abc
import dataclasses
import io
import pathlib
import struct
import wave

import pytest

from bot.providers.mlx_audio import VoiceDetectionError, VoiceDetector
from bot.providers.mlx_audio._mlx import VoiceActivityEvent

SPEECH_WAV = (pathlib.Path(__file__).parent / "assets" / "speech.wav").read_bytes()
"""Real human speech, same encoding as the ring clip (see assets/SOURCES.md)."""


@dataclasses.dataclass
class FakeStreamingSession:
    """Streaming session that replays scripted boundaries against a running audio clock.

    Mirrors the real session's two load-bearing properties: ``audio_ms`` is measured from the
    start of the *session* rather than the current chunk, and ``in_speech`` persists between
    chunks.
    """

    events: collections.abc.Iterator[tuple[VoiceActivityEvent, ...]]
    speaking: bool = False
    chunks: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)

    def in_speech(self) -> bool:
        return self.speaking

    def process_pcm(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> tuple[VoiceActivityEvent, ...]:
        self.chunks.append((len(pcm), sample_rate_hz, channels))
        emitted = next(self.events, ())
        for event in emitted:
            self.speaking = event.started
        return emitted


class FailingStreamingSession:
    def in_speech(self) -> bool:
        return False

    def process_pcm(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int,
        channels: int,
    ) -> tuple[VoiceActivityEvent, ...]:
        del pcm, sample_rate_hz, channels
        raise RuntimeError("mlx unavailable")


def silence_pcm(milliseconds: int, *, sample_rate_hz: int = 16_000) -> bytes:
    samples = sample_rate_hz * milliseconds // 1000
    return struct.pack(f"<{samples}h", *([0] * samples))


def wav_container(pcm: bytes, *, sample_rate_hz: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(pcm)
    return buffer.getvalue()


def classify(detector: VoiceDetector, audio: bytes) -> voice.detection.ApplyData:
    return asyncio.run(detector.classify(audio))


def test_voice_detector_reports_a_talkspurt_that_straddles_a_chunk_boundary() -> None:
    """A talkspurt split across chunks must stay audible in every chunk it covers.

    This is the defect the streaming API exists to remove. Driven per chunk by the offline
    file API, a chunk cut 241 ms after speech began was judged on its own and came back as
    no-voice, because 241 ms is under the model's minimum speech duration; the utterance was
    then dropped before it ever reached the conversation. A streaming session carries speech
    state across the seam, so the second chunk reports voice for its whole length even though
    no speech *starts* inside it -- there is no onset in that chunk to find.
    """

    session = FakeStreamingSession(
        events=iter(
            (
                # Chunk 1: 1200 ms, speech opens 959 ms in and is still open at the cut.
                (VoiceActivityEvent(started=True, audio_ms=959),),
                # Chunk 2: wholly inside the same talkspurt, so it crosses no boundary at all.
                (),
            )
        )
    )
    detector = VoiceDetector(session=session)

    first = classify(detector, silence_pcm(1200))
    second = classify(detector, silence_pcm(1200))

    assert first.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.959, end_seconds=1.2),)
    assert second.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.2),), (
        "a chunk wholly inside a talkspurt must report voice for its whole length"
    )


def test_voice_detector_closes_a_span_on_a_later_chunk_using_the_session_clock() -> None:
    session = FakeStreamingSession(
        events=iter(
            (
                (VoiceActivityEvent(started=True, audio_ms=600),),
                (VoiceActivityEvent(started=False, audio_ms=1500),),
            )
        )
    )
    detector = VoiceDetector(session=session)

    first = classify(detector, silence_pcm(1000))
    second = classify(detector, silence_pcm(1000))

    # 600 ms is inside chunk 1; 1500 ms is 500 ms into chunk 2 on the session clock.
    assert first.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.6, end_seconds=1.0),)
    assert second.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.5),)


def test_voice_detector_reports_no_voice_when_the_session_stays_silent() -> None:
    session = FakeStreamingSession(events=iter(((), ())))
    detector = VoiceDetector(session=session)

    assert classify(detector, silence_pcm(1000)).segments == ()
    assert classify(detector, silence_pcm(1000)).segments == ()


def test_voice_detector_opens_and_closes_within_one_chunk() -> None:
    session = FakeStreamingSession(
        events=iter(
            (
                (
                    VoiceActivityEvent(started=True, audio_ms=200),
                    VoiceActivityEvent(started=False, audio_ms=700),
                ),
            )
        )
    )
    detector = VoiceDetector(session=session)

    assert classify(detector, silence_pcm(1000)).segments == (
        voice.detection.VoiceDetectionSegment(start_seconds=0.2, end_seconds=0.7),
    )


def test_voice_detector_feeds_raw_pcm_at_its_configured_rate() -> None:
    session = FakeStreamingSession(events=iter(((),)))
    detector = VoiceDetector(session=session, sample_rate_hz=48_000, channels=1)

    _ = classify(detector, silence_pcm(100, sample_rate_hz=48_000))

    assert session.chunks == [(2 * 48_000 * 100 // 1000, 48_000, 1)]


def test_voice_detector_prefers_the_rate_declared_by_a_wav_container() -> None:
    """A container is self-describing, so it overrides the configured raw-PCM rate."""

    session = FakeStreamingSession(events=iter(((),)))
    detector = VoiceDetector(session=session, sample_rate_hz=48_000)
    pcm = silence_pcm(100, sample_rate_hz=8_000)

    _ = classify(detector, wav_container(pcm, sample_rate_hz=8_000))

    assert session.chunks == [(len(pcm), 8_000, 1)]


def test_voice_detector_loads_one_session_and_reuses_it_across_chunks() -> None:
    sessions: list[FakeStreamingSession] = []

    def load_session(model_id: str) -> FakeStreamingSession:
        assert model_id == "test-model"
        session = FakeStreamingSession(events=iter(((), ())))
        sessions.append(session)
        return session

    detector = VoiceDetector(model_id="test-model", load_session=load_session)

    _ = classify(detector, silence_pcm(100))
    _ = classify(detector, silence_pcm(100))

    # A session per chunk would reset the stream and reintroduce exactly the boundary blindness
    # this detector exists to avoid.
    assert len(sessions) == 1
    assert len(sessions[0].chunks) == 2


def test_voice_detector_rejects_pcm_that_is_not_aligned_to_its_channel_count() -> None:
    detector = VoiceDetector(session=FakeStreamingSession(events=iter(((),))), channels=2)

    with pytest.raises(VoiceDetectionError):
        _ = classify(detector, b"\x00\x00\x00")


def test_voice_detector_wraps_session_failures() -> None:
    detector = VoiceDetector(session=FailingStreamingSession())

    with pytest.raises(VoiceDetectionError) as error:
        _ = classify(detector, silence_pcm(100))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_voice_detector_is_awaitable() -> None:
    detector = VoiceDetector(session=FakeStreamingSession(events=iter(((),))))

    output = detector.classify(silence_pcm(100))

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == voice.detection.ApplyData(segments=())


@pytest.mark.parametrize(("sample_rate_hz", "channels"), [(0, 1), (-1, 1), (16_000, 0)])
def test_voice_detector_rejects_non_positive_audio_shape(sample_rate_hz: int, channels: int) -> None:
    with pytest.raises(ValueError):
        _ = VoiceDetector(sample_rate_hz=sample_rate_hz, channels=channels)


def test_voice_detector_satisfies_the_core_detector_contract() -> None:
    assert isinstance(VoiceDetector(session=FakeStreamingSession(events=iter(()))), voice.VoiceDetector)


@pytest.mark.live
def test_real_detector_hears_no_voice_in_the_ring_but_hears_speech() -> None:
    """Pin what a real Silero VAD perceives in the shipped ring, against a paired positive control.

    The ring must come back as *not* voice. Voice routes hearing to speech-to-text; the ring has to
    stay on the sound-classification route for the phone to be recognised as ringing at all. Nothing
    else in the suite exercises the real model, so a model or weights change that started hearing
    voice in a ringtone would silently cost the phone its ring perception.

    The speech assertion is what makes the ring assertion mean anything, and it is not optional.
    Empty ``segments`` is equally what this detector returns when the weights fail to load, when
    the model returns nothing, or when it degrades to always-empty -- the ring assertion passes
    trivially in every one of those cases. Putting real speech against a detector proves it
    loaded, ran, and emitted begin/end spans, which is what turns "no segments" into "correctly
    heard no voice." Neither half is worth keeping without the other.

    Each half gets its own detector: one session is one continuous stream, so replaying a second
    clip through the first detector would splice unrelated audio into the same talkspurt.
    """

    assert asyncio.run(VoiceDetector().classify(phone.RING_SOUND_WAV)).segments == ()
    speech = asyncio.run(VoiceDetector().classify(SPEECH_WAV))

    assert speech.segments
    assert speech.segments[0].end_seconds > speech.segments[0].start_seconds


@pytest.mark.live
def test_real_detector_hears_the_same_speech_however_the_stream_is_chunked() -> None:
    """Chunking must not change what is heard -- the property the offline API could not give.

    Feeding identical audio as one clip and as many small clips has to reach the same verdict.
    Under the previous per-clip file API this was false by construction, and that difference is
    what silenced the two-bot call.
    """

    with wave.open(io.BytesIO(SPEECH_WAV), "rb") as stream:
        rate = stream.getframerate()
        pcm = stream.readframes(stream.getnframes())

    whole = asyncio.run(VoiceDetector(sample_rate_hz=rate).classify(pcm))

    chunked = VoiceDetector(sample_rate_hz=rate)
    chunk_bytes = 2 * (rate // 10)  # 100 ms chunks
    heard = [
        segment
        for offset in range(0, len(pcm), chunk_bytes)
        for segment in classify(chunked, pcm[offset : offset + chunk_bytes]).segments
    ]

    assert whole.segments
    assert heard, "speech split into 100 ms chunks must still be heard"
