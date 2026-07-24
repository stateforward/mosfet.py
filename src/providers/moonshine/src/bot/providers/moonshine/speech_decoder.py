"""Moonshine on-device speech-to-text adapter for hearing speech decoding."""

from __future__ import annotations

from bot.abilities.hearing import speech

import asyncio
import dataclasses
import typing

from ._audio import audio_bytes_to_float_pcm
from ._runtime import (
    MoonshineTranscriber,
    MoonshineTranscriberLoader,
    load_transcriber,
    transcript_text,
)


class SpeechDecodingError(RuntimeError):
    """Raised when Moonshine speech decoding fails."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechDecoder(speech.SpeechDecoder):
    """``SpeechDecoder`` backed by Moonshine Voice batch transcription.

    Accepts WAV or raw int16 LE mono PCM bytes (LiveKit-style frames should be
    WAV-wrapped first, as with other providers). Returns UTF-8 transcript text
    bytes for the hearing speech-decoding contract used by Listening HSM stages.
    """

    language: str = "en"
    model_path: str | None = None
    model_arch: object | None = None
    sample_rate_hz: int = 16_000
    update_interval: float = 0.5
    transcriber: MoonshineTranscriber | None = None
    load_transcriber: MoonshineTranscriberLoader = load_transcriber

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        return await asyncio.to_thread(self._decode_blocking, input)

    def _decode_blocking(self, audio: bytes) -> bytes:
        try:
            samples, sample_rate = audio_bytes_to_float_pcm(
                audio,
                default_sample_rate_hz=self.sample_rate_hz,
            )
            if not samples:
                raise SpeechDecodingError("Moonshine speech decoding requires non-empty audio samples.")
            transcriber = self.transcriber
            if transcriber is None:
                transcriber = self.load_transcriber(
                    language=self.language,
                    model_path=self.model_path,
                    model_arch=self.model_arch,
                    update_interval=self.update_interval,
                )
            transcript = transcriber.transcribe_without_streaming(samples, sample_rate)
            text = transcript_text(transcript)
            if not text:
                raise SpeechDecodingError("Moonshine transcription returned empty text.")
            return text.encode("utf-8")
        except SpeechDecodingError:
            raise
        except Exception as error:
            message = "Moonshine speech decoding failed."
            raise SpeechDecodingError(message) from error


__all__ = [
    "SpeechDecoder",
    "SpeechDecodingError",
]
