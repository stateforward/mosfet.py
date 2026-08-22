from __future__ import annotations

import bot.abilities
from bot.abilities import cognition
from bot.abilities.communication import conversation
from bot.abilities.communication.conversation import turn_detector
from bot.abilities.hearing import speech

import asyncio
import dataclasses
import io
import typing
import wave

from bot.providers.mlx_audio import VoiceDecoder, VoiceEncoder


@dataclasses.dataclass(frozen=True)
class FixedSpeechDecoder(speech.SpeechDecoder):
    output: bytes

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        with wave.open(io.BytesIO(input), "rb") as stream:
            assert stream.getframerate() == 16_000
            assert stream.getnchannels() == 1
            assert stream.readframes(1)
        return self.output


@dataclasses.dataclass
class RecordingSpeechEncoder(bot.abilities.Encoder[bytes, bytes]):
    calls: list[bytes] = dataclasses.field(default_factory=list)

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return b"encoded voice response"


def voice_message() -> conversation.TurnData:
    return conversation.TurnData(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset({"bot"}),
        content=b"encoded speech",
        content_type="audio/raw",
    )


def turn_detector_output() -> turn_detector.ParticipantContribution:
    return turn_detector.ParticipantContribution(
        conversation_ref="support-call",
        participant_ref="caller",
        perception=turn_detector.Perception(
            source_participant_ref="caller",
            modality="audio",
            speech=b"encoded speech",
        ),
    )


def test_voice_decoder_uses_mlx_speech_decoder() -> None:
    decoder = VoiceDecoder(speech_decoder=FixedSpeechDecoder(output=b"hello caller"))

    output = asyncio.run(
        decoder.decode(
            turn_detector.AudioStimulus(
                source_participant_ref="caller",
                content=b"encoded speech",
                sample_rate_hz=16_000,
                channels=1,
            )
        )
    )

    assert output == "hello caller"


def test_voice_encoder_uses_mlx_speech_encoder_result_reason() -> None:
    speech_encoder = RecordingSpeechEncoder()
    encoder = VoiceEncoder(speech_encoder=speech_encoder)

    output = asyncio.run(
        encoder.encode(
            conversation.EncodeData(
                message=voice_message(),
                text="hello caller",
                participation=turn_detector_output(),
                result=(
                    cognition.types.EventData(
                        event="phone.answer_call",
                        reason="I can help with that.",
                    ),
                ),
            )
        )
    )

    assert output == b"encoded voice response"
    assert speech_encoder.calls == [b"I can help with that."]


def test_voice_encoder_falls_back_to_text() -> None:
    speech_encoder = RecordingSpeechEncoder()
    encoder = VoiceEncoder(speech_encoder=speech_encoder)

    _ = asyncio.run(
        encoder.encode(
            conversation.EncodeData(
                message=voice_message(),
                text="hello caller",
                participation=turn_detector_output(),
                result=(),
            )
        )
    )

    assert speech_encoder.calls == [b"hello caller"]
