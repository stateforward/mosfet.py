from __future__ import annotations

from bot.abilities.hearing import speech

import asyncio
import collections.abc
import dataclasses
import inspect
import typing

from ._audio_file import temporary_audio_file
from ._mlx import SpeechDecodingModel, SpeechDecodingModelLoader, load_speech_decoding_model

# Process-local warm cache: frozen SpeechDecoder cannot store a loaded model after first use.
_SPEECH_DECODING_MODEL_CACHE: dict[str, SpeechDecodingModel] = {}


class SpeechDecodingError(RuntimeError):
    """Raised when MLX Audio speech decoding fails."""

def _empty_generate_kwargs() -> collections.abc.Mapping[str, object]:
    return {}

@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechDecoder(speech.SpeechDecoder):
    """Speech decoder backed by MLX Audio speech-to-text models.

    The decoder accepts encoded audio bytes, writes them to a temporary audio
    file, and returns the local STT transcript as UTF-8 bytes to match stateforward.bot's
    existing speech-decoding contract.
    """

    model_id: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    verbose: bool = False
    audio_file_suffix: str = ".wav"
    generate_kwargs: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_generate_kwargs)
    model: SpeechDecodingModel | None = None
    load_model: SpeechDecodingModelLoader = load_speech_decoding_model

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        return await asyncio.to_thread(self._decode_blocking, input)

    def _decode_blocking(self, audio: bytes) -> bytes:
        try:
            model = self._resolve_model()
            with temporary_audio_file(audio, suffix=self.audio_file_suffix) as audio_path:
                result = model.generate(
                    str(audio_path),
                    **_accepted_generate_kwargs(
                        model,
                        {
                            "language": self.language,
                            "verbose": self.verbose,
                            **self.generate_kwargs,
                        },
                    ),
                )
            return _transcription_text(result).encode("utf-8")
        except Exception as error:
            message = "MLX Audio speech decoding failed."
            raise SpeechDecodingError(message) from error

    def _resolve_model(self) -> SpeechDecodingModel:
        if self.model is not None:
            return self.model
        # Only warm-cache the default loader so injected load_model seams stay testable.
        if self.load_model is not load_speech_decoding_model:
            return self.load_model(self.model_id)
        cached = _SPEECH_DECODING_MODEL_CACHE.get(self.model_id)
        if cached is not None:
            return cached
        loaded = self.load_model(self.model_id)
        _SPEECH_DECODING_MODEL_CACHE[self.model_id] = loaded
        return loaded

def _accepted_generate_kwargs(
    model: SpeechDecodingModel,
    kwargs: collections.abc.Mapping[str, object],
) -> dict[str, object]:
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)

    return {name: value for name, value in kwargs.items() if name in signature.parameters}

def _transcription_text(result: object) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, collections.abc.Mapping):
        result_mapping = typing.cast(collections.abc.Mapping[object, object], result)
        text = result_mapping.get("text")
        if isinstance(text, str):
            return text
        message = "MLX Audio speech decoding result is missing transcript text."
        raise TypeError(message)

    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text

    message = "MLX Audio speech decoding result is missing transcript text."
    raise TypeError(message)

__all__ = [
    "SpeechDecoder",
    "SpeechDecodingError",
]
