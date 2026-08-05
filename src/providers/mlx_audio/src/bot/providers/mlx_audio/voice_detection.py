from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import concurrent.futures
import io
import typing
import wave

from ._mlx import (
    StreamingVoiceDetectionSession,
    StreamingVoiceDetectionSessionLoader,
    VoiceActivityEvent,
    load_streaming_voice_detection_session,
)


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


class VoiceDetector(voice.VoiceDetector):
    """Voice detector backed by MLX Audio's streaming Silero VAD.

    The model is a streaming one: it consumes fixed 16 kHz frames and reports where speech
    begins and ends against its own continuous clock. This detector holds one session for its
    lifetime and feeds every chunk through it, so a talkspurt that straddles a chunk boundary
    stays open across the seam instead of being judged as an isolated fragment. That is the
    whole reason to prefer the streaming API: with it, how the caller happens to slice the
    audio stops being able to change what is heard.

    ``classify`` still answers per chunk with chunk-local spans, which is what the core
    ``VoiceDetection`` ability turns into sticky Start/End presence boundaries.

    ``sample_rate_hz`` and ``channels`` describe the raw PCM this detector is fed. A WAV
    container is self-describing and overrides both.
    """

    model_id: str
    sample_rate_hz: int
    channels: int
    load_session: StreamingVoiceDetectionSessionLoader

    _session: StreamingVoiceDetectionSession | None
    _worker: concurrent.futures.ThreadPoolExecutor | None
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
        self._session = session
        self._worker = None
        self._elapsed_ms = 0.0
        self._in_speech = False

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.ApplyData:
        # One worker thread for the life of the detector, not a pool. MLX binds its compute
        # stream to the thread that created it, so a session driven from a second thread fails
        # with "no Stream(gpu) in current thread". The single worker also serializes chunks, and
        # order matters here: the session's clock only makes sense if chunks arrive in sequence.
        return await asyncio.get_running_loop().run_in_executor(self._resolve_worker(), self._classify_blocking, input)

    def _resolve_worker(self) -> concurrent.futures.ThreadPoolExecutor:
        worker = self._worker
        if worker is None:
            worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-voice-detection")
            self._worker = worker
        return worker

    def _classify_blocking(self, audio: bytes) -> voice.detection.ApplyData:
        if _is_wav_container(audio):
            pcm, sample_rate_hz, channels = _unwrap_wav(audio)
        else:
            pcm, sample_rate_hz, channels = audio, self.sample_rate_hz, self.channels
        frame_width = 2 * channels
        if len(pcm) % frame_width:
            message = "MLX Audio voice detection requires PCM aligned to the channel count."
            raise VoiceDetectionError(message)
        chunk_ms = 1000.0 * (len(pcm) // frame_width) / sample_rate_hz

        # Runs on the detector's single worker thread, so the session state below is touched by
        # one thread at a time without further locking.
        session = self._resolve_session()
        chunk_start_ms = self._elapsed_ms
        try:
            events = session.process_pcm(pcm, sample_rate_hz=sample_rate_hz, channels=channels)
        except Exception as error:
            message = "MLX Audio voice detection failed."
            raise VoiceDetectionError(message) from error
        open_at_chunk_start = self._in_speech
        self._elapsed_ms = chunk_start_ms + chunk_ms
        self._in_speech = session.in_speech()

        segments = _segments_from_events(
            events,
            chunk_start_ms=chunk_start_ms,
            chunk_ms=chunk_ms,
            open_at_chunk_start=open_at_chunk_start,
        )
        return voice.detection.ApplyData(segments=segments)

    def _resolve_session(self) -> StreamingVoiceDetectionSession:
        session = self._session
        if session is None:
            session = self.load_session(self.model_id)
            self._session = session
        return session


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
    "VoiceDetector",
]
