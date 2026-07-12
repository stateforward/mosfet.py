from __future__ import annotations

from bot import abilities
from bot.abilities import conversation
from bot.abilities.hearing import speech

import dataclasses
import typing

from .speech_decoder import SpeechDecoder
from .speech_encoder import SpeechEncoder

@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceDecoder(conversation.voice.VoiceDecoder):
    """Voice conversation decoder backed by the MLX Audio speech decoder."""

    speech_decoder: speech.SpeechDecoder = dataclasses.field(default_factory=SpeechDecoder)

    @typing.override
    async def decode(self, input: abilities.AudioStimulus) -> str:
        decoded = await self.speech_decoder.decode(input.content)
        return decoded.decode("utf-8")

@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceEncoder(conversation.voice.VoiceEncoder):
    """Voice conversation encoder backed by the MLX Audio speech encoder."""

    speech_encoder: abilities.Encoder[bytes, bytes] = dataclasses.field(default_factory=SpeechEncoder)

    @typing.override
    async def encode(self, input: abilities.EncodeData) -> str | bytes:
        return await self.speech_encoder.encode(_response_text(input).encode("utf-8"))

def _response_text(input: abilities.EncodeData) -> str:
    if input.result.reason:
        return input.result.reason
    if input.memory_context:
        return "\n".join(input.memory_context)
    return input.decoded_text

__all__ = ["VoiceDecoder", "VoiceEncoder"]
