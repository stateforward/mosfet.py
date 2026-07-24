from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import dataclasses
import typing

from ._audio_file import temporary_audio_file
from ._mlx import (
    VoiceDetectionModel,
    VoiceDetectionModelLoader,
    coerce_iterable,
    get_member,
    load_voice_detection_model,
    optional_float,
)

# Process-local warm cache: frozen VoiceDetector cannot store a loaded model after first use.
_VOICE_DETECTION_MODEL_CACHE: dict[str, VoiceDetectionModel] = {}


class VoiceDetectionError(RuntimeError):
    """Raised when MLX Audio voice detection fails."""

@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceDetector(voice.VoiceDetector):
    """Voice detector backed by MLX Audio Silero VAD.

    The detector accepts encoded audio bytes, writes them to a temporary audio
    file, and passes that path to MLX Audio's file-oriented VAD API.
    """

    model_id: str = "mlx-community/silero-vad"
    audio_file_suffix: str = ".wav"
    model: VoiceDetectionModel | None = None
    load_model: VoiceDetectionModelLoader = load_voice_detection_model

    @typing.override
    async def classify(self, input: bytes) -> voice.detection.OutputData:
        return await asyncio.to_thread(self._classify_blocking, input)

    def _classify_blocking(self, audio: bytes) -> voice.detection.OutputData:
        model = self._resolve_model()
        try:
            with temporary_audio_file(audio, suffix=self.audio_file_suffix) as audio_path:
                timestamps = coerce_iterable(
                    model.get_speech_timestamps(audio_path, return_seconds=True),
                    description="speech timestamps",
                )
        except Exception as error:
            message = "MLX Audio voice detection failed."
            raise VoiceDetectionError(message) from error

        return voice.detection.OutputData(is_voice=bool(timestamps), confidence=_confidence_from_timestamps(timestamps))

    def _resolve_model(self) -> VoiceDetectionModel:
        if self.model is not None:
            return self.model
        # Only warm-cache the default loader so injected load_model seams stay testable.
        if self.load_model is not load_voice_detection_model:
            return self.load_model(self.model_id)
        cached = _VOICE_DETECTION_MODEL_CACHE.get(self.model_id)
        if cached is not None:
            return cached
        loaded = self.load_model(self.model_id)
        _VOICE_DETECTION_MODEL_CACHE[self.model_id] = loaded
        return loaded

def _confidence_from_timestamps(timestamps: tuple[object, ...]) -> float | None:
    confidence_values: list[float] = []
    for timestamp in timestamps:
        confidence = optional_float(get_member(timestamp, "confidence", "probability", "score"))
        if confidence is not None:
            confidence_values.append(confidence)
    if not confidence_values:
        return None
    return max(confidence_values)

__all__ = [
    "VoiceDetectionError",
    "VoiceDetector",
]
