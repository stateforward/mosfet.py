"""Bot abilities package.

Prefer domain-level package imports so the namespace is present:

    from bot import abilities
    abilities.Encoder
    abilities.Ability

    from bot.abilities import cognition, processing
    cognition.Cognition
    processing.InputData

Import ``InputData`` / ``OutputData`` / ``FailureData`` from the defining submodule
when names collide across abilities. This package re-exports non-colliding public
symbols, ability classes, and domain subpackages/modules.
"""

import importlib
import typing

from . import (
    ability,
    classifying,
    cognition,
    communication,
    decoding,
    encoding,
    generative,
    hearing,
    identity,
    language,
    learning,
    listening,
    memory,
    processing,
    reading,
    speaking,
    vision,
    vocal,
)
from .ability import (
    FailedEvent,
    InputEvent,
    OutputEvent,
    Ability,
    FailureData,
    TInput,
    TOutput,
)

if typing.TYPE_CHECKING:
    from .classifying import Classifier, Classifying
    from .communication import Communication
    from .communication.conversation import (
        TurnData,
        Conversation,
        DecisionInputFactory,
        ParticipatedTurn,
        Messages,
        Message,
        MessageProvenance,
        AppendData,
        AppendEvent,
        Stage,
        VoiceDecoder,
        VoiceEncoder,
        EncodeData,
        agent_conversation_decision_input,
        contribute_conversation_turn,
        contribute_conversation_input,
        append_conversation_message,
        define_conversation_model,
        run_host_text_respond_turn,
        run_host_voice_respond_turn,
    )
    from .decoding import Decoder, Decoding
    from .encoding import Encoder, Encoding
    from .generative import Generative, Generator
    from .identity import Identity, NameRecognizer, SpeechNameRecognizer
    from .listening import Listening, ListeningStage
    from .speaking import Speaking
    from .reading import Reading, ReadingOutputKind, ReadingStage
    from .learning import Learning

_LAZY_EXPORT_MODULES = {
    "Classifying": ".classifying",
    "Classifier": ".classifying",
    "Communication": ".communication",
    "TurnData": ".communication.conversation",
    "Conversation": ".communication.conversation",
    "DecisionInputFactory": ".communication.conversation",
    "ParticipatedTurn": ".communication.conversation",
    "Messages": ".communication.conversation",
    "Message": ".communication.conversation",
    "MessageProvenance": ".communication.conversation",
    "AppendData": ".communication.conversation",
    "AppendEvent": ".communication.conversation",
    "Stage": ".communication.conversation",
    "VoiceDecoder": ".communication.conversation",
    "VoiceEncoder": ".communication.conversation",
    "EncodeData": ".communication.conversation",
    "agent_conversation_decision_input": ".communication.conversation",
    "contribute_conversation_turn": ".communication.conversation",
    "contribute_conversation_input": ".communication.conversation",
    "append_conversation_message": ".communication.conversation",
    "define_conversation_model": ".communication.conversation",
    "run_host_text_respond_turn": ".communication.conversation",
    "run_host_voice_respond_turn": ".communication.conversation",
    "Decoding": ".decoding",
    "Decoder": ".decoding",
    "Encoding": ".encoding",
    "Encoder": ".encoding",
    "Generative": ".generative",
    "Generator": ".generative",
    "Identity": ".identity",
    "NameRecognizer": ".identity",
    "SpeechNameRecognizer": ".identity",
    "Listening": ".listening",
    "ListeningStage": ".listening",
    "Speaking": ".speaking",
    "Reading": ".reading",
    "ReadingOutputKind": ".reading",
    "ReadingStage": ".reading",
    "Learning": ".learning",
}

__all__ = [
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "Ability",
    "FailureData",
    "Classifier",
    "Classifying",
    "Communication",
    "Conversation",
    "DecisionInputFactory",
    "Decoding",
    "Decoder",
    "ParticipatedTurn",
    "Encoder",
    "Encoding",
    "EncodeData",
    "Generative",
    "Generator",
    "Identity",
    "Learning",
    "Listening",
    "ListeningStage",
    "Speaking",
    "NameRecognizer",
    "Reading",
    "ReadingOutputKind",
    "ReadingStage",
    "Messages",
    "Message",
    "MessageProvenance",
    "AppendData",
    "AppendEvent",
    "SpeechNameRecognizer",
    "Stage",
    "TInput",
    "TOutput",
    "VoiceDecoder",
    "VoiceEncoder",
    "TurnData",
    "ability",
    "agent_conversation_decision_input",
    "contribute_conversation_turn",
    "classifying",
    "cognition",
    "contribute_conversation_input",
    "append_conversation_message",
    "communication",
    "decoding",
    "define_conversation_model",
    "encoding",
    "generative",
    "hearing",
    "identity",
    "language",
    "learning",
    "listening",
    "memory",
    "processing",
    "reading",
    "run_host_text_respond_turn",
    "run_host_voice_respond_turn",
    "speaking",
    "vision",
    "vocal",
]


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = typing.cast(object, getattr(module, name))
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
