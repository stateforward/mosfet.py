from __future__ import annotations

from mosfet.abilities import cognition
from mosfet.abilities import vocal
from mosfet.abilities.communication import conversation
from mosfet.abilities.communication.conversation import turn_detector
from mosfet.abilities.hearing import speech

import asyncio
import collections.abc
import dataclasses
import io
import typing
import wave

from mosfet.providers.mlx_audio import VoiceDecoder, VoiceEncoder


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


class RecordingSpeechEncoder(vocal.speech.SpeechEncoder):
    def __init__(self) -> None:
        self.format: vocal.speech.Format = vocal.speech.Format(sample_rate_hz=24_000, channels=1)
        self.calls: list[str] = []

    @typing.override
    async def encode(
        self, input: collections.abc.AsyncIterable[str]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        said = "".join([piece async for piece in input])
        self.calls.append(said)
        yield vocal.speech.Audio(pcm=b"\x01\x00", text="")
        yield vocal.speech.Audio(pcm=b"\x02\x00", text=said)


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


def test_voice_encoder_speaks_the_result_reason_as_one_pcm_buffer() -> None:
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

    assert output == b"\x01\x00\x02\x00"
    assert speech_encoder.calls == ["I can help with that."]


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

    assert speech_encoder.calls == ["hello caller"]
