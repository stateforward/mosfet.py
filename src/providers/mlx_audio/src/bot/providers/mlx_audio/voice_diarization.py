from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import dataclasses
import typing

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceDiarizer(voice.VoiceDiarizer):
    """Voice diarizer backed by MLX Audio Sortformer diarization.

    The diarizer accepts encoded audio bytes, writes them to a temporary audio
    file, and converts MLX Audio segment records into bot's provider-neutral
    diarization output schema.
    """

    model_id: str = "mlx-community/diar_sortformer_4spk-v1-fp32"
    threshold: float = 0.5
    verbose: bool = False
    audio_file_suffix: str = ".wav"
    model: VoiceDiarizationModel | None = None
    load_model: VoiceDiarizationModelLoader = load_voice_diarization_model

    @typing.override
    async def classify(self, input: bytes) -> voice.diarization.OutputData:
        return await asyncio.to_thread(self._classify_blocking, input)

    def _classify_blocking(self, audio: bytes) -> voice.diarization.OutputData:
        model = self.model if self.model is not None else self.load_model(self.model_id)
        try:
            with temporary_audio_file(audio, suffix=self.audio_file_suffix) as audio_path:
                result = model.generate(audio_path, threshold=self.threshold, verbose=self.verbose)
            segments = tuple(_segment_from_mlx(segment) for segment in _segments_from_result(result))
            return voice.diarization.OutputData(segments=segments)
        except Exception as error:
            message = "MLX Audio voice diarization failed."
            raise VoiceDiarizationError(message) from error


def _segments_from_result(result: object) -> tuple[object, ...]:
    return coerce_iterable(get_member(result, "segments"), description="diarization segments")


def _segment_from_mlx(segment: object) -> voice.VoiceDiarizationSegment:
    return voice.VoiceDiarizationSegment(
        speaker_label=_speaker_label(get_member(segment, "speaker_label", "speaker", "speaker_id")),
        start_seconds=required_float(get_member(segment, "start_seconds", "start", "start_time"), field_name="start"),
        end_seconds=required_float(get_member(segment, "end_seconds", "end", "end_time"), field_name="end"),
        confidence=optional_float(get_member(segment, "confidence", "probability", "score")),
    )


def _speaker_label(value: object) -> str:
    if isinstance(value, bool) or value is None:
        message = "MLX Audio segment is missing a speaker label."
        raise TypeError(message)
    if isinstance(value, int):
        return f"speaker_{value}"
    label = str(value)
    if not label:
        message = "MLX Audio segment has an empty speaker label."
        raise ValueError(message)
    return label


__all__ = [
    "VoiceDiarizationError",
    "VoiceDiarizer",
]
