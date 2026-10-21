from .. import ability
from ..ability import Effort
from .. import processing

import typing

import hsm

import pydantic

from . import deliberation
from . import reasoning
from . import types

# No static system prose, as reasoning: the per-turn world and self blocks, then what reasoning holds above
# intuition's ceiling (``<held>``), are the system channel. Tool schemas and the stimulus carry the rest.
DEFAULT_INSTRUCTIONS = ""
# Intuition's ceiling: it may recall, know and forget (memory is M), and holds nothing rated above.
_DEFAULT_MAX_EFFORT: typing.Final = Effort.M


class InputData(deliberation.InputData):
    """Typed Cognition request for one intuition stage."""


class Intuition(deliberation.Deliberation):
    """Fast cognitive ability, the tier between reflex and reasoning: a chat model that answers without thinking first.

    It shares reasoning's chart (``deliberation.Deliberation``) and its processor's input and answer, at a
    lower depth: its default effort ceiling is ``Effort.M``, so memory is offered to it, and it is offered the
    deliberative handoff (``bot.ability.reasoning.input``) with what reasoning holds above its ceiling named in
    ``<held>``. Selecting the handoff hands the turn on to reasoning after its other selections run (what it
    says first rides the turn's ``acted``); anything else it selects settles the turn. It thinks on a
    ``processing.Processor`` that writes words, never a ``processing.DecisionProcessor``: that is a
    ``TypeError``.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.intuition.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[types.CompletionData]] = hsm.Event[types.CompletionData](
        name="bot.ability.intuition.output",
        schema=types.CompletionData,
    )
    instructions: typing.ClassVar[str] = DEFAULT_INSTRUCTIONS
    _events: typing.ClassVar[deliberation.Events] = deliberation.Events.named("intuition")
    _spans: typing.ClassVar[deliberation.Spans] = deliberation.Spans(
        name="bot.intuition.stage",
        component="cognition.intuition",
        initialize="intuition_initialize",
        reason="intuition_reason",
        held="bot.intuition.held.count",
    )
    _request: typing.ClassVar[type[deliberation.InputData]] = InputData
    _processor_input: typing.ClassVar[type[pydantic.BaseModel]] = reasoning.ProcessorInput
    _decision: typing.ClassVar[type[pydantic.BaseModel]] = reasoning.OutputData
    _offers_handoff: typing.ClassVar[bool] = True

    submodel: typing.ClassVar[hsm.Model | None] = deliberation.Deliberation.chart("Intuition", input_event, _events)

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        processor: processing.Processor,
        instructions: str | None = None,
        max_effort: Effort | None = _DEFAULT_MAX_EFFORT,
    ) -> None:
        # Precondition (D-R-52): intuition writes words, which a decision model cannot. The checker cannot see a
        # decision processor given where a processor is taken; this holds for every caller.
        if isinstance(processor, processing.DecisionProcessor):
            raise TypeError(
                "Intuition thinks on a chat model's processing.Processor, not a processing.DecisionProcessor "
                + f"({type(processor).__name__})."
            )
        deliberation.Deliberation.__init__(
            self, id=id, clock=clock, processor=processor, instructions=instructions, max_effort=max_effort
        )


InputEvent = Intuition.input_event
OutputEvent = Intuition.output_event

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "InputData",
    "InputEvent",
    "Intuition",
    "OutputEvent",
]
