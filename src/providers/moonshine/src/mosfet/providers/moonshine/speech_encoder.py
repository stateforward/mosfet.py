"""Moonshine on-device text-to-speech adapter for speaking / vocal encoding."""

from __future__ import annotations

import asyncio
import dataclasses
import typing

from mosfet import abilities

from ._audio import float_pcm_to_int16_le, float_pcm_to_wav
from ._runtime import MoonshineTextToSpeech, MoonshineTextToSpeechLoader, load_text_to_speech


class SpeechEncodingError(RuntimeError):
    """Raised when Moonshine speech encoding fails."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechEncoder(abilities.Encoder[bytes, bytes]):
    """``Encoder`` that converts UTF-8 text into Moonshine-synthesized audio bytes.

    Uses ``TextToSpeech.synthesize`` (float PCM) and returns WAV by default so
    Speaking / speaker paths can elevate playout without a cloud TTS key.
    """

    language: str = "en-us"
    voice: str | None = None
    speed: float | None = None
    audio_format: str = "wav"
    download: bool = True
    tts: MoonshineTextToSpeech | None = None
    load_tts: MoonshineTextToSpeechLoader = load_text_to_speech

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        text = input.decode("utf-8")
        if not text.strip():
            raise SpeechEncodingError("Moonshine speech encoding requires non-empty text.")
        return await asyncio.to_thread(self._encode_blocking, text)

    def _encode_blocking(self, text: str) -> bytes:
        try:
            synthesizer = self.tts
            if synthesizer is None:
                synthesizer = self.load_tts(
                    language=self.language,
                    voice=self.voice,
                    download=self.download,
                )
            samples, sample_rate = synthesizer.synthesize(text, speed=self.speed)
            if not samples:
                raise SpeechEncodingError("Moonshine did not return speech samples.")
            if sample_rate < 1:
                raise SpeechEncodingError("Moonshine returned an invalid sample rate.")
            if self.audio_format == "wav":
                return float_pcm_to_wav(samples, sample_rate)
            if self.audio_format == "pcm":
                return float_pcm_to_int16_le(samples)
            message = f"Unsupported Moonshine audio_format {self.audio_format!r}; use 'wav' or 'pcm'."
            raise SpeechEncodingError(message)
        except SpeechEncodingError:
            raise
        except Exception as error:
            message = "Moonshine speech encoding failed."
            raise SpeechEncodingError(message) from error


__all__ = [
    "SpeechEncoder",
    "SpeechEncodingError",
]
