from __future__ import annotations

import asyncio
import dataclasses
import typing

from bot import abilities

if typing.TYPE_CHECKING:
    from elevenlabs.client import ElevenLabs


class SpeechEncodingError(RuntimeError):
    """Raised when ElevenLabs speech encoding fails."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SpeechEncoder(abilities.Encoder[bytes, bytes]):
    """Encoder that converts UTF-8 text bytes into ElevenLabs speech audio bytes."""

    voice_id: str
    api_key: str | None = None
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    base_url: str | None = None
    timeout_seconds: float = 240.0
    client: ElevenLabs | None = None

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        text = input.decode("utf-8")
        return await asyncio.to_thread(self._encode_blocking, text)

    def _encode_blocking(self, text: str) -> bytes:
        client = self.client if self.client is not None else self._client()
        try:
            audio = client.text_to_speech.convert(
                self.voice_id,
                text=text,
                model_id=self.model_id,
                output_format=self.output_format,
            )
            return b"".join(audio)
        except Exception as error:
            message = "ElevenLabs speech encoding failed."
            raise SpeechEncodingError(message) from error

    def _client(self) -> ElevenLabs:
        from elevenlabs.client import ElevenLabs as ElevenLabsClient

        return ElevenLabsClient(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds)


def __getattr__(name: str) -> object:
    if name == "ElevenLabs":
        from elevenlabs.client import ElevenLabs as ElevenLabsClient

        return ElevenLabsClient
    raise AttributeError(name)


__all__ = [
    "ElevenLabs",
    "SpeechEncoder",
    "SpeechEncodingError",
]
