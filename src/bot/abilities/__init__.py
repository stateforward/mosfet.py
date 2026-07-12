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
    conversation,
    decoding,
    encoding,
    generative,
    hearing,
    language,
    listening,
    memory,
    participating,
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
    ability_input_event,
    ability_output_event,
)

if typing.TYPE_CHECKING:
    from .classifying import Classifier, Classifying
    from .conversation import (
        AnyMessage,
        Conversation,
        DecisionInputFactory,
        Message,
        ParticipatedTurn,
        Response,
        Stage,
        TextConversation,
        TextMessage,
        VoiceConversation,
        VoiceDecoder,
        VoiceEncoder,
        EncodeData,
        VoiceMessage,
        agent_conversation_decision_input,
        contribute_conversation_turn,
        define_conversation_model,
        run_host_text_respond_turn,
        run_host_voice_respond_turn,
    )
    from .decoding import Decoder, Decoding
    from .encoding import Encoder, Encoding
    from .generative import Generative, Generator
    from .listening import Listening, ListeningStage
    from .speaking import Speaking
    from .participating import (
        AudioPerception,
        AudioStimulus,
        EventStimulus,
        ImageStimulus,
        ParticipantChannelSnapshot,
        ParticipantContribution,
        ParticipantSnapshot,
        ParticipantStateSnapshot,
        Participating,
        ParticipationStimulus,
        Perception,
        ReadablePerception,
        TextStimulus,
    )
    from .reading import Reading, ReadingOutputKind, ReadingStage

_LAZY_EXPORT_MODULES = {
    "Classifying": ".classifying",
    "Classifier": ".classifying",
    "AnyMessage": ".conversation",
    "Conversation": ".conversation",
    "DecisionInputFactory": ".conversation",
    "Message": ".conversation",
    "ParticipatedTurn": ".conversation",
    "Response": ".conversation",
    "Stage": ".conversation",
    "TextConversation": ".conversation",
    "TextMessage": ".conversation",
    "VoiceConversation": ".conversation",
    "VoiceMessage": ".conversation",
    "VoiceDecoder": ".conversation",
    "VoiceEncoder": ".conversation",
    "EncodeData": ".conversation",
    "agent_conversation_decision_input": ".conversation",
    "contribute_conversation_turn": ".conversation",
    "define_conversation_model": ".conversation",
    "run_host_text_respond_turn": ".conversation",
    "run_host_voice_respond_turn": ".conversation",
    "Decoding": ".decoding",
    "Decoder": ".decoding",
    "Encoding": ".encoding",
    "Encoder": ".encoding",
    "Generative": ".generative",
    "Generator": ".generative",
    "Listening": ".listening",
    "ListeningStage": ".listening",
    "Speaking": ".speaking",
    "AudioPerception": ".participating",
    "AudioStimulus": ".participating",
    "EventStimulus": ".participating",
    "ImageStimulus": ".participating",
    "ParticipantChannelSnapshot": ".participating",
    "ParticipantContribution": ".participating",
    "ParticipantSnapshot": ".participating",
    "ParticipantStateSnapshot": ".participating",
    "Participating": ".participating",
    "ParticipationStimulus": ".participating",
    "Perception": ".participating",
    "ReadablePerception": ".participating",
    "TextStimulus": ".participating",
    "Reading": ".reading",
    "ReadingOutputKind": ".reading",
    "ReadingStage": ".reading",
}

__all__ = [
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "Ability",
    "FailureData",
    "AnyMessage",
    "AudioPerception",
    "AudioStimulus",
    "Classifier",
    "Classifying",
    "Conversation",
    "DecisionInputFactory",
    "Decoding",
    "Decoder",
    "ParticipatedTurn",
    "EventStimulus",
    "Encoder",
    "Encoding",
    "EncodeData",
    "Generative",
    "Generator",
    "ImageStimulus",
    "Listening",
    "ListeningStage",
    "Speaking",
    "Message",
    "ParticipantChannelSnapshot",
    "ParticipantContribution",
    "ParticipantSnapshot",
    "ParticipantStateSnapshot",
    "Participating",
    "ParticipationStimulus",
    "Perception",
    "ReadablePerception",
    "Reading",
    "ReadingOutputKind",
    "ReadingStage",
    "Response",
    "Stage",
    "TInput",
    "TOutput",
    "TextConversation",
    "TextMessage",
    "TextStimulus",
    "VoiceConversation",
    "VoiceDecoder",
    "VoiceEncoder",
    "VoiceMessage",
    "ability",
    "ability_input_event",
    "ability_output_event",
    "agent_conversation_decision_input",
    "classifying",
    "cognition",
    "contribute_conversation_turn",
    "conversation",
    "decoding",
    "define_conversation_model",
    "encoding",
    "generative",
    "hearing",
    "language",
    "listening",
    "memory",
    "participating",
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
