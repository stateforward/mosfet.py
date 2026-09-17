"""Lazy Moonshine Voice runtime loading and test seams."""

from __future__ import annotations

import collections.abc
import importlib
import typing


class MoonshineTranscriber(typing.Protocol):
    def transcribe_without_streaming(
        self,
        audio_data: list[float],
        sample_rate: int = 16000,
        flags: int = 0,
    ) -> object: ...


class MoonshineTextToSpeech(typing.Protocol):
    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        volume: float | None = None,
        options: collections.abc.Mapping[str, object] | None = None,
    ) -> tuple[list[float], int]: ...


MoonshineTranscriberLoader: typing.TypeAlias = collections.abc.Callable[..., MoonshineTranscriber]
MoonshineTextToSpeechLoader: typing.TypeAlias = collections.abc.Callable[..., MoonshineTextToSpeech]


def load_transcriber(
    *,
    language: str = "en",
    model_path: str | None = None,
    model_arch: object | None = None,
    update_interval: float = 0.5,
) -> MoonshineTranscriber:
    """Load a Moonshine ``Transcriber`` for batch ``transcribe_without_streaming`` calls."""

    moonshine = importlib.import_module("moonshine_voice")
    if model_path is None or model_arch is None:
        get_model = typing.cast(
            collections.abc.Callable[..., tuple[str, object]],
            getattr(moonshine, "get_model_for_language"),
        )
        resolved_path, resolved_arch = get_model(language)
        model_path = model_path or resolved_path
        model_arch = model_arch if model_arch is not None else resolved_arch
    construct = typing.cast(
        collections.abc.Callable[..., MoonshineTranscriber],
        getattr(moonshine, "Transcriber"),
    )
    return construct(
        model_path=model_path,
        model_arch=model_arch,
        update_interval=update_interval,
    )


def load_text_to_speech(
    *,
    language: str = "en-us",
    voice: str | None = None,
    download: bool = True,
) -> MoonshineTextToSpeech:
    """Load a Moonshine ``TextToSpeech`` synthesizer."""

    moonshine = importlib.import_module("moonshine_voice")
    construct = typing.cast(
        collections.abc.Callable[..., MoonshineTextToSpeech],
        getattr(moonshine, "TextToSpeech"),
    )
    kwargs: dict[str, object] = {"download": download}
    if voice is not None:
        kwargs["voice"] = voice
    return construct(language, **kwargs)


def transcript_text(transcript: object) -> str:
    """Join completed Moonshine transcript lines into plain text."""

    lines = getattr(transcript, "lines", None)
    if not isinstance(lines, collections.abc.Sequence) or isinstance(lines, str | bytes | bytearray):
        text = getattr(transcript, "text", None)
        if isinstance(text, str):
            return text.strip()
        return str(transcript).strip()
    parts: list[str] = []
    for line in lines:
        value = getattr(line, "text", None)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(parts).strip()


__all__ = [
    "MoonshineTextToSpeech",
    "MoonshineTextToSpeechLoader",
    "MoonshineTranscriber",
    "MoonshineTranscriberLoader",
    "load_text_to_speech",
    "load_transcriber",
    "transcript_text",
]
