from __future__ import annotations

import collections.abc
import io
import importlib
import typing


class SpeechGenerationResult(typing.Protocol):
    audio: object
    sample_rate: int


class SpeechEncodingModel(typing.Protocol):
    def generate(self, text: str, **kwargs: object) -> collections.abc.Iterable[SpeechGenerationResult]:
        """Return generated speech audio records for the supplied text."""
        ...


class SpeechDecodingModel(typing.Protocol):
    def generate(self, audio: str, **kwargs: object) -> object:
        """Return generated speech transcription data for the supplied audio file."""
        ...


class VoiceDetectionModel(typing.Protocol):
    def get_speech_timestamps(self, audio: str, *, return_seconds: bool) -> object:
        """Return speech timestamp records for the supplied audio file."""
        ...


class VoiceDiarizationModel(typing.Protocol):
    def generate(self, audio: str, *, threshold: float, verbose: bool) -> object:
        """Return speaker diarization output for the supplied audio file."""
        ...


SpeechEncodingModelLoader = collections.abc.Callable[[str], SpeechEncodingModel]
SpeechDecodingModelLoader = collections.abc.Callable[[str], SpeechDecodingModel]
SpeechAudioWriter = collections.abc.Callable[[object, int, str], bytes]
VoiceDetectionModelLoader = collections.abc.Callable[[str], VoiceDetectionModel]
VoiceDiarizationModelLoader = collections.abc.Callable[[str], VoiceDiarizationModel]


def load_speech_encoding_model(model_id: str) -> SpeechEncodingModel:
    module = importlib.import_module("mlx_audio.tts.utils")
    load = typing.cast(collections.abc.Callable[[str], SpeechEncodingModel], getattr(module, "load_model"))
    return load(model_id)


def write_speech_audio(audio: object, sample_rate: int, audio_format: str) -> bytes:
    module = importlib.import_module("mlx_audio.audio_io")
    write = typing.cast(collections.abc.Callable[..., None], getattr(module, "write"))
    output = io.BytesIO()
    write(output, audio, sample_rate, format=audio_format)
    return output.getvalue()


def load_speech_decoding_model(model_id: str) -> SpeechDecodingModel:
    module = importlib.import_module("mlx_audio.stt.utils")
    load = typing.cast(collections.abc.Callable[[str], SpeechDecodingModel], getattr(module, "load_model"))
    return load(model_id)


def load_voice_detection_model(model_id: str) -> VoiceDetectionModel:
    module = importlib.import_module("mlx_audio.vad")
    load = typing.cast(collections.abc.Callable[[str], VoiceDetectionModel], getattr(module, "load"))
    return load(model_id)


def load_voice_diarization_model(model_id: str) -> VoiceDiarizationModel:
    module = importlib.import_module("mlx_audio.vad")
    load = typing.cast(collections.abc.Callable[[str], VoiceDiarizationModel], getattr(module, "load"))
    return load(model_id)


def get_member(value: object, *names: str) -> object:
    if isinstance(value, collections.abc.Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None

    for name in names:
        member = typing.cast(object, getattr(value, name, None))
        if member is not None:
            return member
    return None


def coerce_iterable(value: object, *, description: str) -> tuple[object, ...]:
    if isinstance(value, str | bytes | bytearray):
        message = f"MLX Audio returned {description} as scalar data."
        raise TypeError(message)
    if not isinstance(value, collections.abc.Iterable):
        message = f"MLX Audio returned {description} that is not iterable."
        raise TypeError(message)
    return tuple(value)


def optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def required_float(value: object, *, field_name: str) -> float:
    parsed = optional_float(value)
    if parsed is None:
        message = f"MLX Audio segment is missing numeric {field_name}."
        raise TypeError(message)
    return parsed
