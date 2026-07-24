from __future__ import annotations

from bot import abilities
from bot.abilities import cognition
from bot.abilities import conversation
from bot.abilities import participating
from bot.abilities.hearing import speech

import asyncio
import dataclasses
import typing

from bot.providers.mlx_audio import VoiceDecoder, VoiceEncoder


@dataclasses.dataclass(frozen=True)
class FixedSpeechDecoder(speech.SpeechDecoder):
    output: bytes

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        assert input == b"encoded speech"
        return self.output


@dataclasses.dataclass
class RecordingSpeechEncoder(abilities.Encoder[bytes, bytes]):
    calls: list[bytes] = dataclasses.field(default_factory=list)

    @typing.override
    async def encode(self, input: bytes) -> bytes:
        self.calls.append(input)
        return b"encoded voice response"


def voice_message() -> conversation.VoiceMessage:
    return conversation.VoiceMessage(
        conversation_ref="support-call",
        self_participant_ref="bot",
        participants=(
            participating.ParticipantSnapshot(
                ref="bot",
                kind="bot",
                state=participating.ParticipantStateSnapshot(
                    presence="present", attention="available", turn="listening"
                ),
            ),
        ),
        content=participating.AudioStimulus(source_participant_ref="caller", content=b"encoded speech"),
    )


def participating_output() -> participating.OutputData:
    return participating.OutputData(
        participant_ref="bot",
        contribution=participating.ParticipantContribution(
            conversation_ref="support-call",
            participant_ref="caller",
            perception=participating.Perception(
                source_participant_ref="caller",
                modality="audio",
                speech=b"encoded speech",
            ),
        ),
    )


def test_voice_decoder_uses_mlx_speech_decoder() -> None:
    decoder = VoiceDecoder(speech_decoder=FixedSpeechDecoder(output=b"hello caller"))

    output = asyncio.run(
        decoder.decode(participating.AudioStimulus(source_participant_ref="caller", content=b"encoded speech"))
    )

    assert output == "hello caller"

def test_voice_encoder_uses_mlx_speech_encoder_result_reason() -> None:
    speech_encoder = RecordingSpeechEncoder()
    encoder = VoiceEncoder(speech_encoder=speech_encoder)

    output = asyncio.run(
        encoder.encode(
            conversation.EncodeData(
                message=voice_message(),
                decoded_text="hello caller",
                participation=participating_output(),
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

def test_voice_encoder_falls_back_to_decoded_text() -> None:
    speech_encoder = RecordingSpeechEncoder()
    encoder = VoiceEncoder(speech_encoder=speech_encoder)

    _ = asyncio.run(
        encoder.encode(
            conversation.EncodeData(
                message=voice_message(),
                decoded_text="hello caller",
                participation=participating_output(),
                result=(),
            )
        )
    )

    assert speech_encoder.calls == [b"hello caller"]
