from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import base64
import collections.abc
import dataclasses
import importlib
import json
import typing

from ._mlx import (
    VoiceIdentificationModel,
    VoiceIdentificationModelLoader,
    load_voice_identification_model,
)

class VoiceIdentificationError(RuntimeError):
    """Raised when MLX Audio voice identification fails."""

VoiceIdentificationAudioDecoder = collections.abc.Callable[[bytes, int], object]

def _decode_audio(audio: bytes, sample_rate: int) -> object:
    miniaudio = importlib.import_module("miniaudio")
    sample_format_type = typing.cast(object, getattr(miniaudio, "SampleFormat"))
    sample_format = typing.cast(object, getattr(sample_format_type, "FLOAT32"))
    decode = typing.cast(collections.abc.Callable[..., object], getattr(miniaudio, "decode"))
    decoded = decode(audio, output_format=sample_format, nchannels=1, sample_rate=sample_rate)
    samples = typing.cast(object, getattr(decoded, "samples"))

    mlx_core = importlib.import_module("mlx.core")
    array = typing.cast(collections.abc.Callable[[object], object], getattr(mlx_core, "array"))
    return array(samples)

@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceIdentifier(voice.VoiceIdentifier):
    """Voice identifier backed by an MLX Audio speaker-embedding model.

    The identifier accepts diarized segment audio, decodes each segment to an
    MLX-compatible waveform, extracts a speaker embedding, and serializes that
    embedding as an opaque provider-neutral voice signature.
    """

    model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    sample_rate: int = 24000
    signature_prefix: str = "mlx-audio:speaker-embedding:"
    model: VoiceIdentificationModel | None = None
    load_model: VoiceIdentificationModelLoader = load_voice_identification_model
    decode_audio: VoiceIdentificationAudioDecoder = _decode_audio

    @typing.override
    async def classify(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        return await asyncio.to_thread(self._classify_blocking, input)

    def _classify_blocking(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        model = self.model if self.model is not None else self.load_model(self.model_id)
        try:
            embeddings_by_speaker: dict[str, list[object]] = {}
            for segment in input.segments:
                speaker_label, embedding = self._embedding_for_segment(model, segment)
                embeddings_by_speaker.setdefault(speaker_label, []).append(embedding)

            signatures = tuple(
                voice.VoiceSignature(
                    speaker_label=speaker_label,
                    signature=_serialize_embeddings(
                        embeddings,
                        model_id=self.model_id,
                        prefix=self.signature_prefix,
                    ),
                    confidence=None,
                )
                for speaker_label, embeddings in embeddings_by_speaker.items()
            )
            return voice.identification.OutputData(signatures=signatures)
        except Exception as error:
            message = "MLX Audio voice identification failed."
            raise VoiceIdentificationError(message) from error

    def _embedding_for_segment(
        self,
        model: VoiceIdentificationModel,
        segment: voice.VoiceIdentificationSegment,
    ) -> tuple[str, object]:
        audio = self.decode_audio(segment.audio, self.sample_rate)
        embedding = model.extract_speaker_embedding(audio, sr=self.sample_rate)
        return segment.diarization.speaker_label, embedding

def _serialize_embeddings(embeddings: collections.abc.Sequence[object], *, model_id: str, prefix: str) -> str:
    payload = {
        "format": "mlx-audio-speaker-embedding-v1",
        "model_id": model_id,
        "embeddings": [_jsonable_embedding(embedding) for embedding in embeddings],
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{prefix}{encoded.decode('ascii')}"

def _jsonable_embedding(value: object) -> object:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable_embedding(tolist())
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, bytes | bytearray):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): _jsonable_embedding(item) for key, item in mapping.items()}
    if isinstance(value, collections.abc.Sequence):
        return [_jsonable_embedding(item) for item in value]

    message = "MLX Audio speaker embedding is not JSON serializable."
    raise TypeError(message)

__all__ = [
    "VoiceIdentificationError",
    "VoiceIdentifier",
]
