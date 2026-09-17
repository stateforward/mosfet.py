from __future__ import annotations

from mosfet.abilities.hearing import voice
from mosfet.devices import phone

import asyncio
import collections.abc
import dataclasses
import io
import pathlib
import struct
import threading
import wave

import pytest

from mosfet.providers.mlx_audio import VoiceDetectionError, VoiceActivityClassifier
from mosfet.providers.mlx_audio._mlx import VoiceActivityEvent

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


def classify(classifier: VoiceActivityClassifier, audio: bytes) -> voice.detection.ApplyData:
    return asyncio.run(classifier.classify(audio))


def test_voice_activity_classifier_reports_a_talkspurt_that_straddles_a_chunk_boundary() -> None:
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
    classifier = VoiceActivityClassifier(session=session)

    first = classify(classifier, silence_pcm(1200))
    second = classify(classifier, silence_pcm(1200))

    assert first.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.959, end_seconds=1.2),)
    assert second.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=1.2),), (
        "a chunk wholly inside a talkspurt must report voice for its whole length"
    )


def test_voice_activity_classifier_closes_a_span_on_a_later_chunk_using_the_session_clock() -> None:
    session = FakeStreamingSession(
        events=iter(
            (
                (VoiceActivityEvent(started=True, audio_ms=600),),
                (VoiceActivityEvent(started=False, audio_ms=1500),),
            )
        )
    )
    classifier = VoiceActivityClassifier(session=session)

    first = classify(classifier, silence_pcm(1000))
    second = classify(classifier, silence_pcm(1000))

    # 600 ms is inside chunk 1; 1500 ms is 500 ms into chunk 2 on the session clock.
    assert first.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.6, end_seconds=1.0),)
    assert second.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.5),)


def test_voice_activity_classifier_reports_no_voice_when_the_session_stays_silent() -> None:
    session = FakeStreamingSession(events=iter(((), ())))
    classifier = VoiceActivityClassifier(session=session)

    assert classify(classifier, silence_pcm(1000)).segments == ()
    assert classify(classifier, silence_pcm(1000)).segments == ()


def test_voice_activity_classifier_opens_and_closes_within_one_chunk() -> None:
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
    classifier = VoiceActivityClassifier(session=session)

    assert classify(classifier, silence_pcm(1000)).segments == (
        voice.detection.VoiceDetectionSegment(start_seconds=0.2, end_seconds=0.7),
    )


def test_voice_activity_classifier_feeds_raw_pcm_at_its_configured_rate() -> None:
    session = FakeStreamingSession(events=iter(((),)))
    classifier = VoiceActivityClassifier(session=session, sample_rate_hz=48_000, channels=1)

    _ = classify(classifier, silence_pcm(100, sample_rate_hz=48_000))

    assert session.chunks == [(2 * 48_000 * 100 // 1000, 48_000, 1)]


def test_voice_activity_classifier_prefers_the_rate_declared_by_a_wav_container() -> None:
    """A container is self-describing, so it overrides the configured raw-PCM rate."""

    sessions: list[FakeStreamingSession] = []

    def load_session(model_id: str) -> FakeStreamingSession:
        del model_id
        session = FakeStreamingSession(events=iter(((),)))
        sessions.append(session)
        return session

    classifier = VoiceActivityClassifier(load_session=load_session, sample_rate_hz=48_000)
    pcm = silence_pcm(100, sample_rate_hz=8_000)

    _ = classify(classifier, wav_container(pcm, sample_rate_hz=8_000))

    assert sessions[0].chunks == [(len(pcm), 8_000, 1)]


def test_voice_activity_classifier_judges_a_container_apart_from_the_live_stream() -> None:
    """A self-contained clip uses an isolated session and cannot inherit live-stream speech state."""

    stream_session = FakeStreamingSession(
        # The stream opens a talkspurt and stays inside it.
        events=iter(((VoiceActivityEvent(started=True, audio_ms=100),), ()))
    )
    clip_sessions: list[FakeStreamingSession] = []

    def load_session(model_id: str) -> FakeStreamingSession:
        del model_id
        session = FakeStreamingSession(events=iter(((),)))
        clip_sessions.append(session)
        return session

    classifier = VoiceActivityClassifier(session=stream_session, load_session=load_session, sample_rate_hz=16_000)

    assert classify(classifier, silence_pcm(1000)).segments, "stream is mid-talkspurt"
    ring = classify(classifier, wav_container(silence_pcm(1500), sample_rate_hz=16_000))

    assert ring.segments == (), "a container must be judged on its own, not on the room's speech"
    # The container went to its own session and never touched the stream's.
    assert len(clip_sessions) == 1
    assert len(stream_session.chunks) == 1


def test_voice_activity_classifier_keeps_the_stream_clock_across_an_interleaved_container() -> None:
    """A clip passing through must not advance or disturb the stream it interrupts."""

    stream_session = FakeStreamingSession(
        events=iter(
            (
                (),
                (VoiceActivityEvent(started=True, audio_ms=1200),),
            )
        )
    )
    classifier = VoiceActivityClassifier(
        session=stream_session,
        load_session=lambda model_id: FakeStreamingSession(events=iter(((),))),
        sample_rate_hz=16_000,
    )

    _ = classify(classifier, silence_pcm(1000))
    _ = classify(classifier, wav_container(silence_pcm(5000), sample_rate_hz=16_000))
    resumed = classify(classifier, silence_pcm(1000))

    # The clip's 5000 ms must not have moved the stream clock: 1200 ms is still 200 ms into
    # the second stream chunk. If the clip had advanced it, this span would be clamped to 0.
    assert resumed.segments == (voice.detection.VoiceDetectionSegment(start_seconds=0.2, end_seconds=1.0),)


def test_voice_activity_classifier_loads_one_session_and_reuses_it_across_chunks() -> None:
    sessions: list[FakeStreamingSession] = []

    def load_session(model_id: str) -> FakeStreamingSession:
        assert model_id == "test-model"
        session = FakeStreamingSession(events=iter(((), ())))
        sessions.append(session)
        return session

    classifier = VoiceActivityClassifier(model_id="test-model", load_session=load_session)

    _ = classify(classifier, silence_pcm(100))
    _ = classify(classifier, silence_pcm(100))

    # A session per chunk would reset the stream and reintroduce exactly the boundary blindness
    # this classifier exists to avoid.
    assert len(sessions) == 1
    assert len(sessions[0].chunks) == 2


def test_voice_activity_classifier_runs_concurrent_calls_on_one_owned_worker() -> None:
    first_started = threading.Event()
    release = threading.Event()
    workers: list[threading.Thread] = []

    class RecordingSession:
        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            del pcm, sample_rate_hz, channels
            workers.append(threading.current_thread())
            first_started.set()
            assert release.wait(timeout=1.0)
            return ()

        def in_speech(self) -> bool:
            return False

    classifier = VoiceActivityClassifier(session=RecordingSession())

    async def run() -> None:
        first = asyncio.create_task(classifier.classify(silence_pcm(100)))
        assert await asyncio.to_thread(first_started.wait, 1.0)
        second = asyncio.create_task(classifier.classify(silence_pcm(100)))
        await asyncio.sleep(0)
        release.set()
        assert await first == await second == voice.detection.ApplyData(segments=())
        await classifier.aclose()

    asyncio.run(run())
    assert len(workers) == 2
    assert workers[0] is workers[1]


def test_voice_activity_classifier_shutdown_cancels_queued_work() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStreamingSession:
        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            del pcm, sample_rate_hz, channels
            started.set()
            assert release.wait(timeout=1.0)
            return ()

        def in_speech(self) -> bool:
            return False

    classifier = VoiceActivityClassifier(session=BlockingStreamingSession())

    async def run() -> None:
        first = asyncio.create_task(classifier.classify(silence_pcm(100)))
        _ = await asyncio.to_thread(started.wait, 1.0)
        second = asyncio.create_task(classifier.classify(silence_pcm(100)))
        await asyncio.sleep(0)
        classifier.shutdown()
        _ = second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        assert await first == voice.detection.ApplyData(segments=())

        with pytest.raises(RuntimeError, match="closed"):
            _ = await classifier.classify(silence_pcm(100))

    asyncio.run(run())


def test_voice_activity_classifier_rejects_work_beyond_one_active_and_one_queued_call() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStreamingSession:
        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            del pcm, sample_rate_hz, channels
            started.set()
            assert release.wait(timeout=1.0)
            return ()

        def in_speech(self) -> bool:
            return False

    classifier = VoiceActivityClassifier(session=BlockingStreamingSession())

    async def run() -> None:
        active = asyncio.create_task(classifier.classify(silence_pcm(100)))
        assert await asyncio.to_thread(started.wait, 1.0)
        queued = asyncio.create_task(classifier.classify(silence_pcm(100)))
        await asyncio.sleep(0)
        with pytest.raises(VoiceDetectionError, match="capacity"):
            _ = await classifier.classify(silence_pcm(100))
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await active == voice.detection.ApplyData(segments=())
        await classifier.aclose()

    asyncio.run(run())


def test_voice_activity_classifier_queued_cancellation_prevents_stream_advance() -> None:
    first_started = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingStreamingSession:
        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            nonlocal calls
            del pcm, sample_rate_hz, channels
            calls += 1
            first_started.set()
            assert release.wait(timeout=1.0)
            return ()

        def in_speech(self) -> bool:
            return False

    classifier = VoiceActivityClassifier(session=BlockingStreamingSession())

    async def run() -> None:
        active = asyncio.create_task(classifier.classify(silence_pcm(100)))
        assert await asyncio.to_thread(first_started.wait, 1.0)
        queued = asyncio.create_task(classifier.classify(silence_pcm(100)))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await active == voice.detection.ApplyData(segments=())
        await classifier.aclose()

    asyncio.run(run())
    assert calls == 1


def test_voice_activity_classifier_active_cancellation_still_advances_owned_stream() -> None:
    first_started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = 0

    class BlockingStreamingSession:
        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            nonlocal calls
            del pcm, sample_rate_hz, channels
            calls += 1
            first_started.set()
            assert release.wait(timeout=1.0)
            finished.set()
            return ()

        def in_speech(self) -> bool:
            return False

    classifier = VoiceActivityClassifier(session=BlockingStreamingSession())

    async def run() -> None:
        active = asyncio.create_task(classifier.classify(silence_pcm(100)))
        assert await asyncio.to_thread(first_started.wait, 1.0)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert not finished.is_set()
        queued = asyncio.create_task(classifier.classify(silence_pcm(100)))
        await asyncio.sleep(0)
        with pytest.raises(VoiceDetectionError, match="capacity"):
            _ = await classifier.classify(silence_pcm(100))
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        assert await asyncio.to_thread(finished.wait, 1.0)
        assert await classifier.classify(silence_pcm(100)) == voice.detection.ApplyData(segments=())
        await classifier.aclose()

    asyncio.run(run())
    assert calls == 2


def test_voice_activity_classifier_close_is_bounded_when_native_work_never_returns() -> None:
    started = threading.Event()

    class PermanentlyBlockedSession:
        worker: threading.Thread | None = None

        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            del pcm, sample_rate_hz, channels
            self.worker = threading.current_thread()
            started.set()
            threading.Event().wait()
            raise AssertionError("unreachable")

        def in_speech(self) -> bool:
            return False

    session = PermanentlyBlockedSession()
    classifier = VoiceActivityClassifier(session=session)

    async def start() -> asyncio.Task[voice.detection.ApplyData]:
        work = asyncio.create_task(classifier.classify(silence_pcm(100)))
        assert await asyncio.to_thread(started.wait, 1.0)
        return work

    work = asyncio.run(start())
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (classifier.close(), closed.set()))
    closer.start()
    assert closed.wait(timeout=0.5), "close must have a finite terminal bound"
    closer.join(timeout=0.1)
    assert session.worker is not None
    assert session.worker.daemon
    work.cancel()


def test_voice_activity_classifier_aclose_is_bounded_when_native_work_never_returns() -> None:
    started = threading.Event()

    class PermanentlyBlockedSession:
        worker: threading.Thread | None = None

        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            del pcm, sample_rate_hz, channels
            self.worker = threading.current_thread()
            started.set()
            threading.Event().wait()
            raise AssertionError("unreachable")

        def in_speech(self) -> bool:
            return False

    session = PermanentlyBlockedSession()
    classifier = VoiceActivityClassifier(session=session)

    async def run() -> None:
        work = asyncio.create_task(classifier.classify(silence_pcm(100)))
        assert await asyncio.to_thread(started.wait, 1.0)
        await asyncio.wait_for(classifier.aclose(), timeout=0.5)
        assert session.worker is not None
        assert session.worker.daemon
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work

    asyncio.run(run())


def test_voice_activity_classifier_repeated_lifecycles_stop_cooperative_workers() -> None:
    workers: list[threading.Thread] = []

    class RecordingSession:
        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            del pcm, sample_rate_hz, channels
            workers.append(threading.current_thread())
            return ()

        def in_speech(self) -> bool:
            return False

    async def run() -> None:
        for _ in range(5):
            classifier = VoiceActivityClassifier(session=RecordingSession())
            assert await classifier.classify(silence_pcm(100)) == voice.detection.ApplyData(segments=())
            await classifier.aclose()

    asyncio.run(run())
    assert len(workers) == 5
    assert all(not worker.is_alive() for worker in workers)


def test_voice_activity_classifier_aclose_waits_for_active_work() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingStreamingSession:
        worker: threading.Thread | None = None

        def process_pcm(
            self,
            pcm: bytes,
            *,
            sample_rate_hz: int,
            channels: int,
        ) -> tuple[VoiceActivityEvent, ...]:
            del pcm, sample_rate_hz, channels
            self.worker = threading.current_thread()
            started.set()
            assert release.wait(timeout=1.0)
            return ()

        def in_speech(self) -> bool:
            return False

    session = BlockingStreamingSession()
    classifier = VoiceActivityClassifier(session=session)

    async def run() -> None:
        work = asyncio.create_task(classifier.classify(silence_pcm(100)))
        assert await asyncio.to_thread(started.wait, 1.0)
        closing = asyncio.create_task(classifier.aclose())
        await asyncio.sleep(0)
        try:
            assert not closing.done()
        finally:
            release.set()
        await closing
        assert await work == voice.detection.ApplyData(segments=())

        worker = session.worker
        assert worker is not None
        assert not worker.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            _ = await classifier.classify(silence_pcm(100))

    asyncio.run(run())


def test_voice_activity_classifier_rejects_pcm_that_is_not_aligned_to_its_channel_count() -> None:
    classifier = VoiceActivityClassifier(session=FakeStreamingSession(events=iter(((),))), channels=2)

    with pytest.raises(VoiceDetectionError):
        _ = classify(classifier, b"\x00\x00\x00")


def test_voice_activity_classifier_wraps_session_failures() -> None:
    classifier = VoiceActivityClassifier(session=FailingStreamingSession())

    with pytest.raises(VoiceDetectionError) as error:
        _ = classify(classifier, silence_pcm(100))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_voice_activity_classifier_is_awaitable() -> None:
    classifier = VoiceActivityClassifier(session=FakeStreamingSession(events=iter(((),))))

    output = classifier.classify(silence_pcm(100))

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == voice.detection.ApplyData(segments=())


@pytest.mark.parametrize(("sample_rate_hz", "channels"), [(0, 1), (-1, 1), (16_000, 0)])
def test_voice_activity_classifier_rejects_non_positive_audio_shape(sample_rate_hz: int, channels: int) -> None:
    with pytest.raises(ValueError):
        _ = VoiceActivityClassifier(sample_rate_hz=sample_rate_hz, channels=channels)


def test_voice_activity_classifier_satisfies_the_core_classifier_contract() -> None:
    assert isinstance(
        VoiceActivityClassifier(session=FakeStreamingSession(events=iter(()))), voice.VoiceActivityClassifier
    )


@pytest.mark.live
def test_real_classifier_hears_no_voice_in_the_ring_but_hears_speech() -> None:
    """Pin what a real Silero VAD perceives in the shipped ring, against a paired positive control.

    The ring must come back as *not* voice. Voice routes hearing to speech-to-text; the ring has to
    stay on the sound-classification route for the phone to be recognised as ringing at all. Nothing
    else in the suite exercises the real model, so a model or weights change that started hearing
    voice in a ringtone would silently cost the phone its ring perception.

    The speech assertion is what makes the ring assertion mean anything, and it is not optional.
    Empty ``segments`` is equally what this classifier returns when the weights fail to load, when
    the model returns nothing, or when it degrades to always-empty -- the ring assertion passes
    trivially in every one of those cases. Putting real speech against a classifier proves it
    loaded, ran, and emitted begin/end spans, which is what turns "no segments" into "correctly
    heard no voice." Neither half is worth keeping without the other.

    Each half gets its own classifier: one session is one continuous stream, so replaying a second
    clip through the first classifier would splice unrelated audio into the same talkspurt.
    """

    assert asyncio.run(VoiceActivityClassifier().classify(phone.RING_SOUND_WAV)).segments == ()
    speech = asyncio.run(VoiceActivityClassifier().classify(SPEECH_WAV))

    assert speech.segments
    assert speech.segments[0].end_seconds > speech.segments[0].start_seconds


@pytest.mark.live
def test_real_classifier_hears_the_same_speech_however_the_stream_is_chunked() -> None:
    """Chunking must not change what is heard -- the property the offline API could not give.

    Feeding identical audio as one clip and as many small clips has to reach the same verdict.
    Under the previous per-clip file API this was false by construction, and that difference is
    what silenced the two-bot call.
    """

    with wave.open(io.BytesIO(SPEECH_WAV), "rb") as stream:
        rate = stream.getframerate()
        pcm = stream.readframes(stream.getnframes())

    whole = asyncio.run(VoiceActivityClassifier(sample_rate_hz=rate).classify(pcm))

    chunked = VoiceActivityClassifier(sample_rate_hz=rate)
    chunk_bytes = 2 * (rate // 10)  # 100 ms chunks
    heard = [
        segment
        for offset in range(0, len(pcm), chunk_bytes)
        for segment in classify(chunked, pcm[offset : offset + chunk_bytes]).segments
    ]

    assert whole.segments
    assert heard, "speech split into 100 ms chunks must still be heard"


@pytest.mark.live
def test_real_classifier_still_hears_no_voice_in_a_ring_that_lands_mid_talkspurt() -> None:
    """The live shape of the ring regression, against the real model.

    One ear, two sources: the room stream and the phone's own ring sample. The ring has to read
    as not-voice even when it arrives while someone in the room is mid-sentence. When both went
    through one continuous session this returned a voice span covering the ring, the ring was
    routed to speech instead of sound classification, and the bot never answered its phone.
    """

    with wave.open(io.BytesIO(SPEECH_WAV), "rb") as stream:
        rate = stream.getframerate()
        speech = stream.readframes(stream.getnframes())

    classifier = VoiceActivityClassifier(sample_rate_hz=rate)

    assert classify(classifier, speech).segments, "the room stream is mid-talkspurt"
    assert classify(classifier, phone.RING_SOUND_WAV).segments == ()
