from . import memory
from .. import ability

import typing

import hsm


class LongTermMemory(memory.Memory):
    """Durable memory for longer retention windows."""

    default_scope: typing.ClassVar[str] = "long_term"
    input_event: typing.ClassVar[hsm.Event[memory.InputData]] = ability.ability_input_event(
        "bot.ability.memory.long_term.input",
        memory.InputData,
        description="SQL transaction against long-term bot memory.",
    )
    output_event: typing.ClassVar[hsm.Event[memory.OutputData]] = ability.ability_output_event(
        "bot.ability.memory.long_term.output",
        memory.OutputData,
    )
    submodel: typing.ClassVar[hsm.Model | None] = memory.memory_model(
        name="LongTermMemory",
        input_event=input_event,
    )
