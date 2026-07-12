from . import memory
from .. import ability

import typing

import hsm


class ShortTermMemory(memory.Memory):
    """Short-lived memory used for active context and recent retention."""

    default_scope: typing.ClassVar[str] = "short_term"
    input_event: typing.ClassVar[hsm.Event[memory.InputData]] = ability.ability_input_event(
        "bot.ability.memory.short_term.input",
        memory.InputData,
        description="SQL transaction against short-term bot memory.",
    )
    output_event: typing.ClassVar[hsm.Event[memory.OutputData]] = ability.ability_output_event(
        "bot.ability.memory.short_term.output",
        memory.OutputData,
    )
    submodel: typing.ClassVar[hsm.Model | None] = memory.memory_model(
        name="ShortTermMemory",
        input_event=input_event,
    )
