from .. import ability
from .. import cognition
from .. import decoding
from .. import encoding
from .. import participating

import abc
import typing as typ

import hsm
import pydantic

from ..participating import ParticipationStimulus
from .conversation import (
    Conversation,
    Message,
    VoiceMessage,
    define_conversation_model,
)


def _has_voice_message(
    ctx: hsm.Context,
    instance: Conversation[typ.Any, typ.Any],
    event: hsm.Event[typ.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, Message) and isinstance(event.data.content, participating.AudioStimulus)


class VoiceDecoder(decoding.Decoder[participating.AudioStimulus, str], abc.ABC):
    """Decoder for a voice conversation turn.

    Implementations receive the accepted audio stimulus for the current voice turn and return decoded readable text
    used by participation and host-owned cognition, memory, and response encoding.
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
                    "message": {
                        "conversation_ref": "support-call",
                        "self_participant_ref": "bot",
                        "participants": [
                            {
                                "ref": "bot",
                                "kind": "bot",
                                "state": {"presence": "present", "attention": "available", "turn": "listening"},
                            },
                            {
                                "ref": "caller",
                                "kind": "human",
                                "state": {"presence": "present", "attention": "available", "turn": "holding"},
                            },
                        ],
                        "content": {"kind": "audio", "source_participant_ref": "caller", "content": "aGVsbG8="},
                    },
                    "decoded_text": "hello",
                    "participation": {
                        "participant_ref": "bot",
                        "contribution": {
                            "conversation_ref": "support-call",
                            "participant_ref": "caller",
                            "perception": {
                                "source_participant_ref": "caller",
                                "modality": "audio",
                                "speech": "aGVsbG8=",
                                "confidence": 0.91,
                            },
                        },
                        "reason": "DecodedData voice stimulus was accepted.",
                    },
                    "result": [],
                    "memory_context": [],
                }
            ],
        },
    )

    message: VoiceMessage = pydantic.Field(
        description="Original voice message accepted for this turn.",
    )
    decoded_text: str = pydantic.Field(
        min_length=1,
        description="Readable text decoded from the accepted audio stimulus.",
        examples=["hello"],
    )
    participation: participating.participating.OutputData = pydantic.Field(
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
    """Encoder for a completed voice conversation turn.

    Implementations receive the public voice encoding input and produce the channel response bytes or text to publish.
    Hosts own encoding after cognition and memory complete.
    """


class VoiceConversation(Conversation[VoiceMessage, typ.Any]):
    """Thin voice conversation coordinator (decode + participate).

    Encoder is retained as a host-facing collaborator, not a conversation HSM child.
    """

    input_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = VoiceMessage
    _encoding: encoding.Encoding[EncodeData, str | bytes]
    input_event: typ.ClassVar[hsm.Event[VoiceMessage]] = ability.ability_input_event(
        "bot.ability.conversation.voice.input",
        VoiceMessage,
    )

    def __init__(
        self,
        *,
        participating: participating.Participating,
        decoder: VoiceDecoder | None,
        encoder: VoiceEncoder | None,
    ) -> None:
        if decoder is None:
            raise ValueError("VoiceConversation requires decoder.")
        if encoder is None:
            raise ValueError("VoiceConversation requires encoder.")
        stimulus_decoding = decoding.Decoding(decoder=typ.cast(decoding.Decoder[ParticipationStimulus, str], decoder))
        self._encoding = encoding.Encoding(encoder=encoder)
        super().__init__(
            decoding=stimulus_decoding,
            participating=participating,
        )

    @property
    def encoding(self) -> encoding.Encoding[EncodeData, str | bytes]:
        """Host-facing voice encoding collaborator (not a conversation HSM child)."""

        return self._encoding

    @typ.override
    def _conversation_kind(self) -> str:
        return "voice"

    submodel: typ.ClassVar[hsm.Model | None] = define_conversation_model(
        "VoiceConversation",
        input_event=input_event,
        input_guard=_has_voice_message,
    )


__all__ = ["VoiceConversation", "VoiceMessage", "VoiceDecoder", "VoiceEncoder", "EncodeData"]
