from __future__ import annotations

import bot.abilities
from bot.abilities.communication import conversation
from bot.abilities.hearing import speech

import dataclasses
import io
import typing
import wave

from .speech_decoder import SpeechDecoder
from .speech_encoder import SpeechEncoder


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceDecoder(conversation.voice.VoiceDecoder):
    """Voice conversation decoder backed by the MLX Audio speech decoder."""

    speech_decoder: speech.SpeechDecoder = dataclasses.field(default_factory=SpeechDecoder)

    @typing.override
    async def decode(self, input: conversation.turn_detector.AudioStimulus) -> str:
        audio = input.content
        if not audio.startswith(b"RIFF"):
            if input.sample_rate_hz is None:
                raise ValueError("MLX Audio raw PCM decoding requires a sample rate.")
            output = io.BytesIO()
            with wave.open(output, "wb") as stream:
                stream.setnchannels(input.channels or 1)
                stream.setsampwidth(2)
                stream.setframerate(input.sample_rate_hz)
                stream.writeframes(audio)
            audio = output.getvalue()
        decoded = await self.speech_decoder.decode(audio)
        return decoded.decode("utf-8")


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceEncoder(conversation.voice.VoiceEncoder):
    """Voice conversation encoder backed by the MLX Audio speech encoder."""

    speech_encoder: bot.abilities.Encoder[bytes, bytes] = dataclasses.field(default_factory=SpeechEncoder)

    @typing.override
    async def encode(self, input: bot.abilities.EncodeData) -> str | bytes:
        return await self.speech_encoder.encode(_response_text(input).encode("utf-8"))


def _response_text(input: bot.abilities.EncodeData) -> str:
    # Cognition result is a tuple of event selections; speak the first stated reason.
    reason = next((selection.reason for selection in input.result if selection.reason), None)
    if reason:
        return reason
    if input.memory_context:
        return "\n".join(input.memory_context)
    return input.text


__all__ = ["VoiceDecoder", "VoiceEncoder"]
