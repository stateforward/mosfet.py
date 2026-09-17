"""Learning ability: decode a lesson, ground runtime input in evidence, author a behavior.

Learning does not consult peer ability packages for event contracts. Live-input knowledge
grounds in two evidence sources: remembered turns in memory and the bot's event register
(recent admitted stimuli). If neither can name a stimulus_name, Learning fails closed.

Starlark authoring is composed through :class:`~mosfet.abilities.cognition.reflection.revision.Revision`
(create/change inventory) — Revision is authoring composition, not an input-discovery path.
"""

from .learning import (
    GENERATE_INSTRUCTIONS,
    DecodedData,
    RuntimeInputData,
    GenerateData,
    GenerateEvent,
    InputData,
    InputEvent,
    Learning,
    OutputData,
    OutputEvent,
    RememberedSelection,
    RememberedTurn,
    SelectInput,
)

__all__ = [
    "GENERATE_INSTRUCTIONS",
    "DecodedData",
    "RuntimeInputData",
    "GenerateData",
    "GenerateEvent",
    "InputData",
    "InputEvent",
    "Learning",
    "OutputData",
    "OutputEvent",
    "RememberedSelection",
    "RememberedTurn",
    "SelectInput",
]
