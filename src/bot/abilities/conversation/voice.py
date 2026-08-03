from .. import decoding
from .. import encoding
from .. import cognition
from .. import turn_detector

import abc
import typing as typ

import pydantic

from .conversation import ConversationInputData


class VoiceDecoder(decoding.Decoder[turn_detector.AudioStimulus, str], abc.ABC):
    """Decoder for a voice conversation turn.

    Implementations receive audio for the current open turn (clip or assembled) and return readable text.
    """


class EncodeData(pydantic.BaseModel):
    """Public input contract for encoding a completed voice conversation turn."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "InputData for a voice encoder after host-owned decoding, participation, cognition, and memory "
                "have completed. This is the public voice response contract."
            ),
            "examples": [
                {
                    "source_ids": ["caller"],
                    "target_ids": ["bot"],
                    "content": "aGVsbG8=",
                    "content_type": "audio/raw",
                    "decoded_text": "hello",
                    "participation": {
                        "conversation_ref": "support-call",
                        "participant_ref": "caller",
                        "perception": {
                            "source_participant_ref": "caller",
                            "modality": "audio",
                            "speech": "aGVsbG8=",
                            "confidence": 0.91,
                        },
                    },
                    "result": [],
                    "memory_context": [],
                }
            ],
        },
    )

    message: ConversationInputData = pydantic.Field(
        description="Original voice message accepted for this turn.",
    )
    decoded_text: str = pydantic.Field(
        min_length=1,
        description="Readable text decoded from the accepted audio stimulus.",
        examples=["hello"],
    )
    participation: turn_detector.ParticipantContribution = pydantic.Field(
        description="Participant contribution produced from the decoded voice stimulus.",
    )
    result: cognition.types.OutputData = pydantic.Field(
        description="Cognition result selected for the voice contribution.",
        examples=[[]],
    )
    memory_context: tuple[str, ...] = pydantic.Field(
        default=(),
        description=(
            "Plain response context derived from memory output for voice rendering. This intentionally does not "
            "expose raw memory lifecycle results."
        ),
    )


class VoiceEncoder(encoding.Encoder[EncodeData, str | bytes], abc.ABC):
    """Encoder for a completed voice conversation turn."""


__all__ = ["ConversationInputData", "VoiceDecoder", "VoiceEncoder", "EncodeData"]
