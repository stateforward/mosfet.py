from __future__ import annotations

import mosfet.abilities
from mosfet.abilities import vocal
from mosfet.abilities.communication import conversation
from mosfet.abilities.hearing import speech

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
    """Voice conversation encoder backed by an MLX Audio speech encoder: the reply as one buffer of s16le PCM."""

    speech_encoder: vocal.speech.SpeechEncoder = dataclasses.field(default_factory=SpeechEncoder)

    @typing.override
    async def encode(self, input: mosfet.abilities.EncodeData) -> str | bytes:
        # A whole string is the text written once and closed.
        text = vocal.speech.Text()
        text.write(vocal.speech.Piece(part=0, words=_response_text(input)))
        text.close()
        return b"".join([chunk.pcm async for chunk in self.speech_encoder.encode(text)])


def _response_text(input: mosfet.abilities.EncodeData) -> str:
    # Cognition result is a tuple of event selections; speak the first stated reason.
    reason = next((selection.reason for selection in input.result if selection.reason), None)
    if reason:
        return reason
    if input.memory_context:
        return "\n".join(input.memory_context)
    return input.text


__all__ = ["VoiceDecoder", "VoiceEncoder"]
