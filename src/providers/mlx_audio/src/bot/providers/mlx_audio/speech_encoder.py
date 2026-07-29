from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import importlib
import typing

import bot.abilities

from ._mlx import (
    SpeechEncodingModel,
    SpeechAudioWriter,
    SpeechEncodingModelLoader,
    get_member,
    load_speech_encoding_model,
    write_speech_audio,
)


class SpeechEncodingError(RuntimeError):
    """Raised when MLX Audio speech encoding fails."""


@dataclasses.dataclass(frozen=True)
class _SpeechAudioChunk:
    audio: object
    sample_rate: int


def _empty_generate_kwargs() -> collections.abc.Mapping[str, object]:
    return {}


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechEncoder(bot.abilities.Encoder[bytes, bytes]):
    """bot.abilities.Encoder that converts UTF-8 text bytes into MLX Audio speech audio bytes."""

    model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    voice: str | None = None
    speed: float = 1.0
    lang_code: str = "auto"
    audio_format: str = "wav"
    max_tokens: int | None = None
    temperature: float | None = None
    verbose: bool = False
    stream: bool = False
    generate_kwargs: collections.abc.Mapping[str, object] = dataclasses.field(default_factory=_empty_generate_kwargs)
    model: SpeechEncodingModel | None = None
    load_model: SpeechEncodingModelLoader = load_speech_encoding_model
    write_audio: SpeechAudioWriter = write_speech_audio

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        text = input.decode("utf-8")
        return await asyncio.to_thread(self._encode_blocking, text)

    def _encode_blocking(self, text: str) -> bytes:
        model = self.model if self.model is not None else self.load_model(self.model_id)
        try:
            chunks = tuple(_chunk_from_result(result) for result in model.generate(text=text, **self._model_kwargs()))
            if not chunks:
                message = "MLX Audio did not return speech audio."
                raise ValueError(message)
            chunk = _join_chunks(chunks)
            return self.write_audio(chunk.audio, chunk.sample_rate, self.audio_format)
        except Exception as error:
            message = "MLX Audio speech encoding failed."
            raise SpeechEncodingError(message) from error

    def _model_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "speed": self.speed,
            "lang_code": self.lang_code,
            "verbose": self.verbose,
            "stream": self.stream,
        }
        if self.voice is not None:
            kwargs["voice"] = self.voice
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        kwargs.update(self.generate_kwargs)
        return kwargs


def _chunk_from_result(result: object) -> _SpeechAudioChunk:
    audio = get_member(result, "audio")
    if audio is None:
        message = "MLX Audio speech result is missing audio."
        raise TypeError(message)

    sample_rate = get_member(result, "sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        message = "MLX Audio speech result is missing an integer sample rate."
        raise TypeError(message)

    return _SpeechAudioChunk(audio=audio, sample_rate=sample_rate)


def _join_chunks(chunks: tuple[_SpeechAudioChunk, ...]) -> _SpeechAudioChunk:
    first = chunks[0]
    for chunk in chunks[1:]:
        if chunk.sample_rate != first.sample_rate:
            message = "MLX Audio speech chunks used different sample rates."
            raise ValueError(message)
    if len(chunks) == 1:
        return first
    return _SpeechAudioChunk(
        audio=_concatenate_audio(tuple(chunk.audio for chunk in chunks)),
        sample_rate=first.sample_rate,
    )


def _concatenate_audio(chunks: tuple[object, ...]) -> object:
    try:
        mlx_core = importlib.import_module("mlx.core")
        concatenate = typing.cast(collections.abc.Callable[..., object], getattr(mlx_core, "concatenate"))
        return concatenate(chunks, axis=0)
    except Exception:
        numpy = importlib.import_module("numpy")
        asarray = typing.cast(collections.abc.Callable[[object], object], getattr(numpy, "asarray"))
        concatenate = typing.cast(collections.abc.Callable[..., object], getattr(numpy, "concatenate"))
        return concatenate(tuple(asarray(chunk) for chunk in chunks), axis=0)


__all__ = [
    "SpeechEncoder",
    "SpeechEncodingError",
]
