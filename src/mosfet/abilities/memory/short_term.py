from . import memory

import typing

import hsm


class ShortTermMemory(memory.Memory):
    """Short-lived memory used for active context and recent retention."""

    default_scope: typing.ClassVar[str] = "short_term"
    input_event: typing.ClassVar[hsm.Event[memory.InputData]] = hsm.Event[memory.InputData](
        name="bot.ability.memory.short_term.input",
        schema=memory.InputData,
    )
    output_event: typing.ClassVar[hsm.Event[memory.OutputData]] = hsm.Event[memory.OutputData](
        name="bot.ability.memory.short_term.output",
        schema=memory.OutputData,
    )
    submodel: typing.ClassVar[hsm.Model | None] = memory.memory_model(
        name="ShortTermMemory",
        input_event=input_event,
    )
