from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import dataclasses
import io
import math
import typing
import wave

from ._audio_file import temporary_audio_file
from ._mlx import (
    VoiceDiarizationModel,
    VoiceDiarizationModelLoader,
    coerce_iterable,
    get_member,
    load_voice_diarization_model,
    optional_float,
    required_float,
)


class VoiceDiarizationError(RuntimeError):
    """Raised when MLX Audio voice diarization fails."""


VoiceDiarizationAudioClipper = typing.Callable[[voice.diarization.InputData, float, float], bytes]


def _wav_container(input: voice.diarization.InputData) -> bytes:
    """Wrap signed 16-bit PCM in the WAV container expected by MLX Audio."""

    frame_bytes = 2 * input.channels
    if len(input.audio) % frame_bytes:
        raise ValueError("MLX Audio diarization input is not aligned to complete PCM frames.")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(input.channels)
        wav.setsampwidth(2)
        wav.setframerate(input.sample_rate_hz)
        wav.writeframes(input.audio)
    return output.getvalue()


def _clip_audio(input: voice.diarization.InputData, start_seconds: float, end_seconds: float) -> bytes:
    """Clip raw signed 16-bit PCM to a validated half-open diarization span."""

    frame_bytes = 2 * input.channels
    if len(input.audio) % frame_bytes:
        raise ValueError("MLX Audio diarization input is not aligned to complete PCM frames.")
    duration_seconds = len(input.audio) / float(frame_bytes * input.sample_rate_hz)
    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or start_seconds < 0.0
        or end_seconds <= start_seconds
        or end_seconds > duration_seconds
    ):
        raise ValueError("MLX Audio diarization span is outside the source audio duration.")
    start_frame = math.floor(start_seconds * input.sample_rate_hz)
    end_frame = math.ceil(end_seconds * input.sample_rate_hz)
    frames = input.audio[start_frame * frame_bytes : end_frame * frame_bytes]
    if not frames:
        raise ValueError("MLX Audio diarization segment has no audio frames.")
    return frames


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceDiarizer(voice.VoiceDiarizer):
    """Voice diarizer backed by MLX Audio Sortformer diarization.

    The diarizer accepts raw PCM with explicit format metadata, stages it as a
    temporary WAV file, and converts MLX Audio segment records into the provider-neutral
    diarization output schema.
    """

    model_id: str = "mlx-community/diar_sortformer_4spk-v1-fp32"
    threshold: float = 0.5
    verbose: bool = False
    audio_file_suffix: str = ".wav"
    model: VoiceDiarizationModel | None = None
    load_model: VoiceDiarizationModelLoader = load_voice_diarization_model
    clip_audio: VoiceDiarizationAudioClipper = _clip_audio

    @typing.override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        return await asyncio.to_thread(self._classify_blocking, input)

    def _classify_blocking(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        try:
            model = self.model if self.model is not None else self.load_model(self.model_id)
            with temporary_audio_file(_wav_container(input), suffix=self.audio_file_suffix) as audio_path:
                result = model.generate(audio_path, threshold=self.threshold, verbose=self.verbose)
            segments = tuple(
                sorted(
                    (
                        _segment_from_mlx(segment, input=input, clip_audio=self.clip_audio)
                        for segment in _segments_from_result(result)
                    ),
                    key=lambda segment: segment.start_seconds,
                )
            )
            return voice.diarization.OutputData(segments=segments)
        except Exception as error:
            message = "MLX Audio voice diarization failed."
            raise VoiceDiarizationError(message) from error


def _segments_from_result(result: object) -> tuple[object, ...]:
    return coerce_iterable(get_member(result, "segments"), description="diarization segments")


def _segment_from_mlx(
    segment: object,
    *,
    input: voice.diarization.InputData,
    clip_audio: VoiceDiarizationAudioClipper,
) -> voice.VoiceDiarizationSegment:
    start_seconds = required_float(get_member(segment, "start_seconds", "start", "start_time"), field_name="start")
    end_seconds = required_float(get_member(segment, "end_seconds", "end", "end_time"), field_name="end")
    return voice.VoiceDiarizationSegment(
        audio=clip_audio(input, start_seconds, end_seconds),
        media_type=input.media_type,
        sample_rate_hz=input.sample_rate_hz,
        channels=input.channels,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        confidence=optional_float(get_member(segment, "confidence", "probability", "score")),
    )


__all__ = [
    "VoiceDiarizationError",
    "VoiceDiarizer",
]
