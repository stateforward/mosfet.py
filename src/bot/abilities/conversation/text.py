from .. import ability
from .. import decoding
from .. import encoding
from .. import language
from .. import participating

import typing as typ

import hsm

from .conversation import (
    Conversation,
    Message,
    TextMessage,
    define_conversation_model,
)


def _has_text_message(
    ctx: hsm.Context,
    instance: Conversation[typ.Any, typ.Any],
    event: hsm.Event[typ.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, Message) and isinstance(event.data.content, participating.TextStimulus)


class TextConversation(Conversation[TextMessage, typ.Any]):
    """Thin text conversation coordinator (decode + participate).

    Hosts that need typed replies own TextGeneration / Encoding via host_turn composition.
    """

    input_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = TextMessage
    _typing: language.TextGeneration | None
    _encoding: encoding.Encoding[str, str | bytes] | None
    input_event: typ.ClassVar[hsm.Event[TextMessage]] = ability.ability_input_event(
        "bot.ability.conversation.text.input",
        TextMessage,
    )

    def __init__(
        self,
        *,
        participating: participating.Participating,
        decoding: decoding.Decoding[participating.ParticipationStimulus, str] | None,
        typing: language.TextGeneration | None = None,
        encoding: encoding.Encoding[str, str | bytes] | None = None,
    ) -> None:
        super().__init__(
            decoding=decoding,
            participating=participating,
        )
        self._typing = typing
        self._encoding = encoding

    @property
    def typing(self) -> language.TextGeneration | None:
        """Optional host-facing text generation collaborator (not a conversation HSM child)."""

        return self._typing

    @property
    def encoding(self) -> encoding.Encoding[str, str | bytes] | None:
        """Optional host-facing text encoding collaborator (not a conversation HSM child)."""

        return self._encoding

    @typ.override
    def _conversation_kind(self) -> str:
        return "text"

    submodel: typ.ClassVar[hsm.Model | None] = define_conversation_model(
        "TextConversation",
        input_event=input_event,
        input_guard=_has_text_message,
    )


__all__ = ["TextConversation", "TextMessage"]
