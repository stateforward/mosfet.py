"""Moonshine on-device text-to-speech as a ``vocal.speech.SpeechEncoder``."""

from __future__ import annotations

import asyncio
import collections.abc
import typing

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech

from ._audio import float_pcm_to_int16_le
from ._runtime import MoonshineTextToSpeech, MoonshineTextToSpeechLoader, load_text_to_speech


class SpeechEncoder(speech.SpeechEncoder):
    """Moonshine ``TextToSpeech.synthesize`` as mono s16le PCM at ``sample_rate_hz``.

    The model is not incremental and takes the whole text, and it applies no expression (``expressiveness`` is the
    default, so every part has the same direction and all of them form one run): the reply is read to its end
    (joining the words of its pieces in order is the only buffering) and said as one chunk once the model has
    spoken it, aligned to all the words. Audio at
    a rate other than the declared ``sample_rate_hz`` is refused as ``invalid_response``, because raw PCM carries
    no header to correct it. ``voice`` selects the voice the model is loaded with; an injected ``tts`` is used as
    is.
    """

    def __init__(
        self,
        *,
        language: str = "en-us",
        voice: str | None = None,
        speed: float | None = None,
        sample_rate_hz: int = 24_000,
        download: bool = True,
        tts: MoonshineTextToSpeech | None = None,
        load_tts: MoonshineTextToSpeechLoader = load_text_to_speech,
    ) -> None:
        self.format: speech.Format = speech.Format(sample_rate_hz=sample_rate_hz, channels=1)
        self._language: str = language
        self._voice: str | None = voice
        self._speed: float | None = speed
        self._download: bool = download
        self._tts: MoonshineTextToSpeech | None = tts
        self._load_tts: MoonshineTextToSpeechLoader = load_tts

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[speech.Piece]
    ) -> collections.abc.AsyncIterator[speech.Audio]:
        said = "".join([piece.words async for piece in input])
        if not said.strip():
            raise encoding.Error("invalid_input", "Moonshine speech encoding requires non-blank text.")
        pcm = await asyncio.to_thread(self._generate, said)
        yield speech.Audio(pcm=pcm, text=said)

    def _generate(self, text: str) -> bytes:
        try:
            tts = self._tts
            if tts is None:
                tts = self._load_tts(language=self._language, voice=self._voice, download=self._download)
            samples, sample_rate = tts.synthesize(text, speed=self._speed)
        except Exception as error:
            raise encoding.Error("encode_failed", "Moonshine speech encoding failed.") from error
        if not samples:
            raise encoding.Error("invalid_response", "Moonshine did not return speech samples.")
        if sample_rate != self.format.sample_rate_hz:
            raise encoding.Error(
                "invalid_response",
                (
                    f"Moonshine returned {sample_rate} Hz audio; this encoder declares "
                    + f"{self.format.sample_rate_hz} Hz. Configure sample_rate_hz={sample_rate}."
                ),
            )
        return float_pcm_to_int16_le(samples)


__all__ = ["SpeechEncoder"]
