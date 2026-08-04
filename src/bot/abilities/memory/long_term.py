from . import memory

import typing

import hsm


class LongTermMemory(memory.Memory):
    """Durable memory for longer retention windows."""

    default_scope: typing.ClassVar[str] = "long_term"
    input_event: typing.ClassVar[hsm.Event[memory.InputData]] = hsm.Event[memory.InputData](
    name="bot.ability.memory.long_term.input",
    schema=memory.InputData,

    )
    output_event: typing.ClassVar[hsm.Event[memory.OutputData]] = hsm.Event[memory.OutputData](
    name="bot.ability.memory.long_term.output",
    schema=memory.OutputData,

    )
    submodel: typing.ClassVar[hsm.Model | None] = memory.memory_model(
        name="LongTermMemory",
        input_event=input_event,
    )
