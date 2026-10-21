from .. import ability
from ..ability import Effort
from .. import processing

import typing

import hsm
from mosfet import event

import pydantic

from mosfet.behavior import BreakData, ChangeData, CreateData
from . import deliberation
from . import types

# No static system prose: deliberative turns use the same per-turn world XML on the system
# channel as the decision layer (plus the deliberative user payload). Behavior create/change/break stay
# on offered schemas and processor input fields, not a standing assistant policy.
DEFAULT_INSTRUCTIONS = ""


class CallData(pydantic.BaseModel):
    """Model-facing handoff of the current turn to deliberate (System-2) reasoning.

    The Cognition host offers it on its call surface to intuition; reasoning never offers it to itself,
    and the decision layer is never offered it. The host reuses the live processing frame (stimulus, tools, actors). Do not restate the
    stimulus, schemas, or instructions here — leave ``data`` empty / omit fields.
    """

    # Schema marker: selecting this event hands the turn to deliberate processing. Shared so
    # earlier stages can detect handoff without importing this module (see processing.is_deliberative_handoff_schema).
    __deliberative_handoff__: typing.ClassVar[bool] = True

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Hand this turn to deliberate reasoning: it thinks before it answers, takes longer, and "
                "holds the capacities listed in <held> as well as the events offered here. Other events "
                "selected with it run first. The host supplies the deliberative frame; do not include "
                "stimulus text, tools, or schemas in data. Use an empty object as data."
            ),
            "examples": [
                {},
            ],
        },
    )


class InputData(deliberation.InputData):
    """Typed Cognition request for one deliberate reasoning stage."""


class ProcessorInput(pydantic.BaseModel):
    """Deliberative reasoning input: the host decision input this turn is reasoned over."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Deliberative reasoning input. Preserves the original host decision input; what memory "
                "surfaced for this turn rides the processing input as recalled context."
            ),
        },
    )

    host_input: processing.InputData = pydantic.Field(
        description="Original host processing input being evaluated.",
    )


class OutputData(pydantic.BaseModel):
    """Deliberate cognitive selection plus optional behavior create/change/break."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "description": (
                "Deliberate cognitive decision produced by reasoning. result is applied by the host; "
                "set at most one of create, change, or break to record a behavior inventory event with the episode."
            ),
            "examples": [
                {
                    "result": [
                        {
                            "target": "bot",
                            "event": "bot.focus_device",
                            "data": {"device": "device-a"},
                            "reason": "That device is where this turn is happening.",
                        }
                    ],
                    "confidence": 0.77,
                    "reason": "Deliberation settled which device this turn is about.",
                },
                # This branch exists to show result and create together. It deliberately selects
                # nothing utterable: an example the bot can say aloud is a crib rather than
                # documentation, and a bot unsure what to say says the example.
                {
                    "result": [
                        {
                            "target": "bot",
                            "event": "bot.focus_device",
                            "data": {"device": "device-b"},
                            "reason": "The ring came from that device.",
                        }
                    ],
                    "create": {
                        "event": "bot.behavior.create",
                        "name": "FocusTheRingingDevice",
                        "triggers": ["environment.sound"],
                        "reason": "Same focus-on-ring pattern across recalled episodes.",
                    },
                },
            ],
        },
    )

    result: types.OutputData = pydantic.Field(
        description="Selected events from deliberate reasoning (empty tuple = none).",
        examples=[
            [
                {
                    "target": "bot",
                    "event": "bot.focus_device",
                    "data": {"device": "device-a"},
                    "reason": "That device is where this turn is happening.",
                }
            ]
        ],
    )
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional confidence for the deliberate decision, normalized from 0.0 to 1.0.",
        examples=[0.77],
    )
    reason: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional concise rationale for the deliberate decision.",
        examples=["Deliberation settled which device this turn is about."],
    )
    create: CreateData | None = pydantic.Field(
        default=None,
        description="When set, bot.behavior.create payload retained with the episode.",
    )
    change: ChangeData | None = pydantic.Field(
        default=None,
        description="When set, bot.behavior.change payload retained with the episode.",
    )
    break_: BreakData | None = pydantic.Field(
        default=None,
        alias="break",
        description="When set, bot.behavior.break payload retained with the episode.",
    )

    @pydantic.model_validator(mode="after")
    def validate_single_behavior_event(self) -> typing.Self:
        selected = [value for value in (self.create, self.change, self.break_) if value is not None]
        if len(selected) > 1:
            raise ValueError("Set at most one of create, change, or break.")
        return self

    def behavior_data(self) -> CreateData | ChangeData | BreakData | None:
        if self.create is not None:
            return self.create
        if self.change is not None:
            return self.change
        if self.break_ is not None:
            return self.break_
        return None


ReasoningProcessor: typing.TypeAlias = processing.Processor


class Reasoning(deliberation.Deliberation):
    """Slow cognitive ability, the deliberate tier: it thinks before it answers, and takes what earlier tiers hand on.

    It shares intuition's chart (``deliberation.Deliberation``): initializing → idle → applying → idle, with a
    bounded selection repair. It has no effort ceiling by default, and never offers the deliberative handoff to
    itself: it is the handoff's recipient. What memory surfaced for the turn arrives on the host frame as
    ``recalled`` and is passed on to the processor unchanged; Reasoning neither reads nor writes memory itself.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = (
        InputData,
        CallData,
    )
    # CallData remains the model-facing schema; Cognition dispatches the full InputData frame
    # only after the stages before it leave the turn unhandled.
    input_event: typing.ClassVar[hsm.Event[CallData | InputData]] = hsm.Event[CallData | InputData](
        name=types.DELIBERATIVE_HANDOFF_EVENT_NAME,
        kind=event.EventKind,
        schema=CallData,
    )
    output_event: typing.ClassVar[hsm.Event[types.CompletionData]] = hsm.Event[types.CompletionData](
        name="bot.ability.reasoning.output",
        schema=types.CompletionData,
    )
    instructions: typing.ClassVar[str] = DEFAULT_INSTRUCTIONS
    _events: typing.ClassVar[deliberation.Events] = deliberation.Events.named("reasoning")
    _spans: typing.ClassVar[deliberation.Spans] = deliberation.Spans(
        name="bot.reasoning.stage",
        component="cognition.reasoning",
        initialize="reasoning_initialize",
        reason="reasoning_reason",
        held="bot.reasoning.held.count",
    )
    _request: typing.ClassVar[type[deliberation.InputData]] = InputData
    _processor_input: typing.ClassVar[type[pydantic.BaseModel]] = ProcessorInput
    _decision: typing.ClassVar[type[pydantic.BaseModel]] = OutputData
    _offers_handoff: typing.ClassVar[bool] = False

    submodel: typing.ClassVar[hsm.Model | None] = deliberation.Deliberation.chart("Reasoning", input_event, _events)

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        processor: ReasoningProcessor,
        instructions: str | None = None,
        max_effort: Effort | None = None,
    ) -> None:
        deliberation.Deliberation.__init__(
            self, id=id, clock=clock, processor=processor, instructions=instructions, max_effort=max_effort
        )


InputEvent = Reasoning.input_event
OutputEvent = Reasoning.output_event

__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "CallData",
    "InputEvent",
    "OutputEvent",
    "Reasoning",
    "InputData",
    "OutputData",
    "ReasoningProcessor",
]
