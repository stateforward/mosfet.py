from __future__ import annotations

from mosfet.abilities.hearing import voice

import asyncio
import functools
import io
import threading
import typing
import wave

from mosfet.telemetry import span

from ._mlx import (
    StreamingVoiceDetectionSession,
    StreamingVoiceDetectionSessionLoader,
    VoiceActivityEvent,
    load_streaming_voice_detection_session,
)
from .speech_decoder import BoundedWorker

_SCOPE = "mosfet.providers.mlx_audio"
_COMPONENT = "mlx_audio.voice_detection"


class VoiceDetectionError(RuntimeError):
    """Raised when MLX Audio voice detection fails."""


def _is_wav_container(audio: bytes) -> bool:
    return len(audio) >= 12 and audio.startswith(b"RIFF") and audio[8:12] == b"WAVE"


def _unwrap_wav(audio: bytes) -> tuple[bytes, int, int]:
    """Read PCM, sample rate and channel count out of a RIFF/WAVE container."""

    try:
        with wave.open(io.BytesIO(audio), "rb") as stream:
            if stream.getsampwidth() != 2:
                message = "MLX Audio voice detection requires signed 16-bit PCM audio."
                raise VoiceDetectionError(message)
            return stream.readframes(stream.getnframes()), stream.getframerate(), stream.getnchannels()
    except wave.Error as error:
        message = "MLX Audio voice detection could not read the WAV container."
        raise VoiceDetectionError(message) from error


class VoiceActivityClassifier(voice.VoiceActivityClassifier):
    """Voice-activity classifier backed by MLX Audio's streaming Silero VAD.

    The model is a streaming one: it consumes fixed 16 kHz frames and reports where speech
    begins and ends against its own continuous clock. Headerless PCM is fed through one
    long-lived session, so a talkspurt that straddles a chunk boundary stays open across the
    seam instead of being judged as an isolated fragment. That is the whole reason to prefer
    the streaming API: how the caller happens to slice the stream stops being able to change
    what is heard.

    A self-describing container is **not** part of that stream. It is a complete recording --
    a phone's ring sample, a stored clip -- that merely happens to arrive through the same ear,
    and it gets its own throwaway session. Splicing one into the live stream would carry speech
    state across audio that never adjoined it: a ring landing while the room is mid-talkspurt
    would inherit that speech and be heard as a voice, which costs the phone its ring. One
    session means one continuous stream, and unrelated audio is not that stream.

    ``classify`` answers per chunk with chunk-local spans, which is what the core
    ``VoiceDetection`` ability turns into sticky Start/End presence boundaries.

    The classifier accepts at most one active and one queued chunk. Further calls fail with
    ``VoiceDetectionError`` instead of retaining unbounded audio. Cancelling a queued call removes
    it before native execution; cancelling an active call cancels only the caller's wait because
    MLX native work cannot be interrupted, and the stream advances when that work completes.
    ``close`` and ``aclose`` wait up to 100 ms for cooperative work, then leave a permanently
    blocked native call isolated to a daemon worker that cannot hold process shutdown open.

    ``sample_rate_hz`` and ``channels`` describe the raw PCM stream. A container declares its
    own and overrides both.
    """

    model_id: str
    sample_rate_hz: int
    channels: int
    load_session: StreamingVoiceDetectionSessionLoader

    _stream_session: StreamingVoiceDetectionSession | None
    _worker: BoundedWorker | None
    _worker_lock: threading.Lock
    _closed: bool
    _elapsed_ms: float
    _in_speech: bool

    def __init__(
        self,
        *,
        model_id: str = "mlx-community/silero-vad",
        sample_rate_hz: int = 16_000,
        channels: int = 1,
        session: StreamingVoiceDetectionSession | None = None,
        load_session: StreamingVoiceDetectionSessionLoader = load_streaming_voice_detection_session,
    ) -> None:
        if sample_rate_hz < 1:
            raise ValueError("sample_rate_hz must be a positive integer.")
        if channels < 1:
            raise ValueError("channels must be a positive integer.")
        self.model_id = model_id
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.load_session = load_session
        self._stream_session = session
        self._worker = None
        self._worker_lock = threading.Lock()
        self._closed = False
        self._elapsed_ms = 0.0
        self._in_speech = False

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        # One worker thread for the life of the classifier, not a pool. MLX binds its compute
        # stream to the thread that created it, so a session driven from a second thread fails
        # with "no Stream(gpu) in current thread". The single worker also serializes chunks, and
        # order matters here: the session's clock only makes sense if chunks arrive in sequence.
        #
        # That same long life is why the trace context is bound here, at hand-over, and not where
        # the worker is created: the worker outlives every chunk, so a context captured at
        # bring-up would file each chunk's span under whatever was happening when the classifier
        # first woke up. `run_in_executor` does not copy the caller's context the way
        # `asyncio.to_thread` does, so without this bind each chunk starts its own trace.
        loop = asyncio.get_running_loop()
        with self._worker_lock:
            if self._closed:
                raise RuntimeError("MLX Audio voice activity classifier is closed.")
            worker = self._resolve_worker_unlocked()
            result = worker.try_submit(functools.partial(span.bind(self._classify_blocking), input))
        if result is None:
            with span.operation(
                "bot.provider.mlx_audio.voice_detection.admission",
                scope=_SCOPE,
                component=_COMPONENT,
                stage="admission",
            ):
                raise VoiceDetectionError("MLX Audio voice activity classifier capacity is exhausted.")
        return await asyncio.wrap_future(result, loop=loop)

    def _resolve_worker_unlocked(self) -> BoundedWorker:
        worker = self._worker
        if worker is None:
            worker = BoundedWorker(name="mlx-voice-detection")
            self._worker = worker
        return worker

    def shutdown(self) -> None:
        """Reject new work and cancel queued calls without waiting for active classification."""

        worker = self._begin_shutdown()
        if worker is not None:
            worker.close(wait=False)

    def close(self) -> None:
        """Reject work, cancel queued calls, and wait at most 100 ms for active work."""

        worker = self._begin_shutdown()
        if worker is None:
            return
        with span.operation(
            "bot.provider.mlx_audio.voice_detection.close",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="close",
        ):
            worker.close(wait=True)

    async def aclose(self) -> None:
        """Perform bounded terminal cleanup, deferring cancellation until it finishes."""

        worker = self._begin_shutdown()
        if worker is None:
            return
        with span.operation(
            "bot.provider.mlx_audio.voice_detection.close",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="close",
        ):
            shutdown = asyncio.create_task(
                asyncio.to_thread(worker.close, wait=True),
                name="mlx-audio-voice-detection-close",
            )
            try:
                await asyncio.shield(shutdown)
            except asyncio.CancelledError:
                await shutdown
                raise

    def _begin_shutdown(self) -> BoundedWorker | None:
        with self._worker_lock:
            self._closed = True
            return self._worker

    def _classify_blocking(self, audio: bytes) -> voice.detection.ApplyData:
        # Runs on the classifier's single worker thread, so the stream state below is touched by
        # one thread at a time without further locking.
        #
        # `bot.audio.session` is the load-bearing dimension: a container gets a throwaway session
        # and is judged alone, a raw chunk advances the one continuous stream. Which of the two a
        # piece of audio got is the difference between a ring being heard as a ring and being
        # swallowed by an open talkspurt. Segment counts only — never the audio, never spans of
        # what was said.
        with span.operation(
            "bot.provider.mlx_audio.voice_detection.classify",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="classify",
        ) as active:
            if _is_wav_container(audio):
                active.set_attribute("bot.audio.session", "clip")
                pcm, sample_rate_hz, channels = _unwrap_wav(audio)
                clip = self._classify_clip(pcm, sample_rate_hz=sample_rate_hz, channels=channels)
                active.set_attribute("bot.voice.segments.count", len(clip.segments))
                return clip
            active.set_attribute("bot.audio.session", "stream")
            chunk = self._classify_stream_chunk(audio)
            active.set_attribute("bot.voice.segments.count", len(chunk.segments))
            active.set_attribute("bot.voice.in_speech", self._in_speech)
            return chunk

    def _classify_clip(self, pcm: bytes, *, sample_rate_hz: int, channels: int) -> voice.detection.ApplyData:
        """Judge one self-contained recording on its own, leaving the live stream untouched."""

        chunk_ms = _chunk_ms(pcm, sample_rate_hz=sample_rate_hz, channels=channels)
        session = self.load_session(self.model_id)
        events = _process(session, pcm, sample_rate_hz=sample_rate_hz, channels=channels)
        return voice.detection.ApplyData(
            segments=_segments_from_events(
                events,
                chunk_start_ms=0.0,
                chunk_ms=chunk_ms,
                open_at_chunk_start=False,
            )
        )

    def _classify_stream_chunk(self, pcm: bytes) -> voice.detection.ApplyData:
        """Advance the one continuous stream by this chunk, carrying speech across the seam."""

        chunk_ms = _chunk_ms(pcm, sample_rate_hz=self.sample_rate_hz, channels=self.channels)
        session = self._stream_session
        if session is None:
            session = self.load_session(self.model_id)
            self._stream_session = session
        chunk_start_ms = self._elapsed_ms
        events = _process(session, pcm, sample_rate_hz=self.sample_rate_hz, channels=self.channels)
        open_at_chunk_start = self._in_speech
        self._elapsed_ms = chunk_start_ms + chunk_ms
        self._in_speech = session.in_speech()
        return voice.detection.ApplyData(
            segments=_segments_from_events(
                events,
                chunk_start_ms=chunk_start_ms,
                chunk_ms=chunk_ms,
                open_at_chunk_start=open_at_chunk_start,
            )
        )


def _chunk_ms(pcm: bytes, *, sample_rate_hz: int, channels: int) -> float:
    frame_width = 2 * channels
    if len(pcm) % frame_width:
        message = "MLX Audio voice detection requires PCM aligned to the channel count."
        raise VoiceDetectionError(message)
    return 1000.0 * (len(pcm) // frame_width) / sample_rate_hz


def _process(
    session: StreamingVoiceDetectionSession,
    pcm: bytes,
    *,
    sample_rate_hz: int,
    channels: int,
) -> tuple[VoiceActivityEvent, ...]:
    try:
        return session.process_pcm(pcm, sample_rate_hz=sample_rate_hz, channels=channels)
    except Exception as error:
        message = "MLX Audio voice detection failed."
        raise VoiceDetectionError(message) from error


def _segments_from_events(
    events: tuple[VoiceActivityEvent, ...],
    *,
    chunk_start_ms: float,
    chunk_ms: float,
    open_at_chunk_start: bool,
) -> tuple[voice.detection.VoiceDetectionSegment, ...]:
    """Fold session-absolute boundaries into spans local to this chunk.

    A chunk that lies wholly inside a talkspurt crosses no boundary at all and must still come
    back as voice for its whole length -- that is the case the offline per-chunk call could
    never report, because in isolation such a chunk has no speech onset to find.

    Boundary times are clamped into the chunk. The model's clock advances a frame at a time and
    only counts whole 512-sample frames, so it can trail the true audio position by up to one
    32 ms frame; clamping keeps that skew from producing a span outside the chunk it describes.
    """

    def local_ms(audio_ms: int) -> float:
        return min(max(audio_ms - chunk_start_ms, 0.0), chunk_ms)

    segments: list[voice.detection.VoiceDetectionSegment] = []
    open_ms: float | None = 0.0 if open_at_chunk_start else None
    for event in events:
        position = local_ms(event.audio_ms)
        if event.started:
            if open_ms is None:
                open_ms = position
            continue
        if open_ms is not None:
            if position > open_ms:
                segments.append(
                    voice.detection.VoiceDetectionSegment(
                        start_seconds=open_ms / 1000.0,
                        end_seconds=position / 1000.0,
                    )
                )
            open_ms = None
    if open_ms is not None and chunk_ms > open_ms:
        segments.append(
            voice.detection.VoiceDetectionSegment(
                start_seconds=open_ms / 1000.0,
                end_seconds=chunk_ms / 1000.0,
            )
        )
    return tuple(segments)


__all__ = [
    "VoiceDetectionError",
    "VoiceActivityClassifier",
]
