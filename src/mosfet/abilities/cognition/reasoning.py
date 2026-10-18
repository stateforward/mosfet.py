from .. import ability
from ..ability import Effort
from .. import processing

import dataclasses
import typing
import uuid

import hsm
import mosfet
from mosfet import event

import pydantic

from mosfet import telemetry
from mosfet.telemetry import span

from mosfet.behavior import BreakData, ChangeData, CreateData
from . import input as cognition_input
from . import types

# No static system prose: deliberative turns use the same per-turn world XML on the system
# channel as intuition (plus the deliberative user payload). Behavior create/change/break stay
# on offered schemas and processor input fields, not a standing assistant policy.
DEFAULT_INSTRUCTIONS = ""


class _InitializingCompleteData(pydantic.BaseModel):
    """Typed empty signal that deliberative reasoning finished initialization."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Empty initialization-complete signal for deliberative reasoning; carries no payload.",
            "examples": [{}],
        },
    )


_InitializingCompleteEvent = hsm.Event[_InitializingCompleteData](
    name="bot.ability.reasoning.initializing.complete",
    kind=hsm.CompletionEventKind,
    schema=_InitializingCompleteData,
)


class CallData(pydantic.BaseModel):
    """Model-facing handoff of the current turn to deliberate (System-2) reasoning.

    The Cognition host offers it to intuition when reasoning holds capacities above intuition's
    ceiling. The host reuses the live processing frame (stimulus, tools, actors). Do not restate the
    stimulus, schemas, or instructions here — leave ``data`` empty / omit fields.
    """

    # Schema marker: selecting this event hands the turn to deliberate processing. Shared so
    # intuition can detect handoff without importing this module (see processing.is_deliberative_handoff_schema).
    __deliberative_handoff__: typing.ClassVar[bool] = True

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Hand this turn to deliberate reasoning, which holds the capacities listed in "
                "<reasoning_only> as well as the events offered here. Other events selected with it "
                "run first. The host supplies the deliberative frame; do not include stimulus text, "
                "tools, or schemas in data. Use an empty object as data."
            ),
            "examples": [
                {},
            ],
        },
    )


class InputData(pydantic.BaseModel):
    """Typed Cognition request for one deliberate reasoning stage."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    processing_input: processing.InputData


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
                            "event": "bot.clear_focus",
                            "reason": "Nothing is live on that device any more.",
                        }
                    ],
                    "create": {
                        "event": "bot.behavior.create",
                        "name": "ReleaseFocusWhenDeviceGoesQuiet",
                        "triggers": ["environment.sound"],
                        "reason": "Same release-focus pattern across recalled episodes.",
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


class _ReasoningCapability(pydantic.BaseModel):
    """JSON-safe authority minted for exactly one live reasoning operation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    actor_id: str
    token: str
    reply_to: str | None = None


class _ReasonedEventData(pydantic.BaseModel):
    """Private completion: the processor produced a decision for this turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    capability: _ReasoningCapability
    host_input: processing.InputData
    reasoned: OutputData


_ReasonedEvent = hsm.Event[_ReasonedEventData](
    name="bot.ability.reasoning.reasoned",
    kind=hsm.CompletionEventKind,
    schema=_ReasonedEventData,
)


class _ReasoningStageFailedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    failure: ability.FailureData
    turn: types.TurnData
    capability: _ReasoningCapability


_ReasoningStageFailedEvent = hsm.Event[_ReasoningStageFailedData](
    name="bot.ability.reasoning.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=_ReasoningStageFailedData,
)


class _SelectionRejectedData(pydantic.BaseModel):
    """Private retry context: the model's selection for this turn was rejected before delivery.

    ``message`` is the redacted diagnostic the model repairs from; the raw exception never crosses the HSM
    boundary, because Pydantic may include the rejected value in it.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    request: InputData = pydantic.Field(description="The reasoning request being repaired.")
    capability: _ReasoningCapability
    message: str = pydantic.Field(min_length=1, description="Redacted rejection diagnostic for model repair.")
    normalized: str = pydantic.Field(
        min_length=1, description="Stable rejection key used to detect the same rejection twice running."
    )
    previous_normalized: str | None = pydantic.Field(
        default=None, min_length=1, description="The key of the rejection just before this one, when any."
    )
    attempt: int = pydantic.Field(
        ge=0,
        le=types.MAX_SELECTION_REPAIRS,
        description="Zero-based bounded invocation attempt that produced this rejection.",
    )


_SelectionRejectedEvent = hsm.Event[_SelectionRejectedData](
    name="bot.ability.reasoning.selection.rejected",
    kind=hsm.ErrorEventKind,
    schema=_SelectionRejectedData,
)


ReasoningProcessor: typing.TypeAlias = processing.Processor


def _public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return dict(metadata)


def _selections_from_output(output: types.OutputData) -> processing.Events:
    return tuple(
        processing.SelectedEvent(
            event=item.event,
            target=item.target,
            data=item.data,
            reason=item.reason,
        )
        for item in output
    )


class Reasoning(processing.Processing):
    """Slow cognitive ability: inject a ``processing.Processor`` and apply it to the deliberate frame.

    Chart phases: initializing → idle → applying → idle. A rejected selection (one the processor refuses, or
    one dispatch refuses) passes through ``routing_rejection``: the model is asked again with the diagnostic,
    at most ``types.MAX_SELECTION_REPAIRS`` times, and the same rejection twice running fails the turn.
    Turn payload rides private completion events (HSM-COMPLETION-001).
    What memory surfaced for the turn arrives on the host frame as ``recalled`` and is passed on
    to the processor unchanged; Reasoning neither reads nor writes memory itself.
    The deliberative ``Processor`` is not an HSM child; Reasoning calls ``process`` in-activity.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = (
        InputData,
        CallData,
    )
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.CompletionData
    # CallData remains the model-facing schema; Cognition dispatches the full InputData frame
    # only after autonomy and intuition leave the turn unhandled.
    input_event: typing.ClassVar[hsm.Event[CallData | InputData]] = hsm.Event[CallData | InputData](
        name=types.DELIBERATIVE_HANDOFF_EVENT_NAME,
        kind=event.EventKind,
        schema=CallData,
    )
    output_event: typing.ClassVar[hsm.Event[types.CompletionData]] = hsm.Event[types.CompletionData](
        name="bot.ability.reasoning.output",
        schema=types.CompletionData,
    )
    failed_event: typing.ClassVar[hsm.Event[types.FailureData]] = hsm.Event[types.FailureData](
        name=ability.FailedEvent.name,
        kind=hsm.ErrorEventKind,
        schema=types.FailureData,
    )
    instructions: typing.ClassVar[str] = DEFAULT_INSTRUCTIONS
    _processor: processing.Processor
    _instructions: str
    _max_effort: Effort | None

    @staticmethod
    async def _initialize_activity(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.reasoning.stage",
            scope="bot.abilities.cognition",
            component="cognition.reasoning",
            stage="reasoning_initialize",
            context=telemetry.event_context(event),
        ):
            del event
            _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(_InitializingCompleteData()))

    @staticmethod
    def _input_from_event(event: hsm.Event[typing.Any]) -> InputData:
        """Resolve the typed deliberative frame supplied by its cognition host."""

        data = event.data
        if isinstance(data, InputData):
            return data
        raise TypeError(f"Reasoning input must be InputData, got {type(data)!r}.")

    @staticmethod
    def _has_reasoning_input(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _matches_operation(instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        capability = (
            data.capability
            if isinstance(
                data,
                _ReasonedEventData | _ReasoningStageFailedData | _SelectionRejectedData,
            )
            else None
        )
        if not isinstance(capability, _ReasoningCapability):
            return False
        return (
            event.id == capability.operation_id
            and event.source == capability.actor_id
            and event.target == hsm.id(instance)
            and processing.matches_operation(instance, capability.operation_id, capability.actor_id)
        )

    @staticmethod
    def _has_reasoned(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _ReasonedEventData) and Reasoning._matches_operation(instance, event)

    @staticmethod
    def _has_stage_failure(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _ReasoningStageFailedData) and Reasoning._matches_operation(instance, event)

    @staticmethod
    def _has_selection_rejection(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _SelectionRejectedData) and Reasoning._matches_operation(instance, event)

    @staticmethod
    def _is_same_selection_rejection(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _SelectionRejectedData) and data.previous_normalized == data.normalized

    @staticmethod
    def _can_repair_selection(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _SelectionRejectedData) and data.attempt < types.MAX_SELECTION_REPAIRS

    @staticmethod
    async def _start_operation(
        instance: "Reasoning",
        operation_id: str,
        *,
        reply_to: str | None,
    ) -> _ReasoningCapability:
        operation = processing.active_operation(instance, operation_id)
        if operation is None:
            operation = await processing.start_operation(instance, operation_id)
        return _ReasoningCapability(
            operation_id=operation_id,
            actor_id=hsm.id(operation),
            token=uuid.uuid4().hex,
            reply_to=reply_to,
        )

    @staticmethod
    def _cancel(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        processing.Processing._emit_cancelled(ctx, instance, event)

    @staticmethod
    def _stage_event(
        event_type: hsm.Event[typing.Any],
        data: object,
        *,
        capability: _ReasoningCapability,
        metadata: dict[str, object],
        target: "Reasoning",
    ) -> hsm.Event[typing.Any]:
        return dataclasses.replace(
            event_type.with_data(data),
            id=capability.operation_id,
            source=capability.actor_id,
            target=hsm.id(target),
            metadata=dict(metadata),
        )

    @staticmethod
    def _dispatch_failure(
        ctx: hsm.Context,
        instance: "Reasoning",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        failure: types.FailureData,
        target: str | None,
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=operation_id,
            metadata=_public_metadata(metadata),
            source=hsm.id(instance),
            target=target,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _dispatch_output(
        ctx: hsm.Context,
        instance: "Reasoning",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        output: types.CompletionData,
        target: str | None,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=_public_metadata(metadata),
            source=hsm.id(instance),
            target=target,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
        if operation_id is not None:
            processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _fail_from_stage(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ReasoningStageFailedData)
        Reasoning._dispatch_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=types.FailureData(message=data.failure.message, turn=data.turn),
            target=data.capability.reply_to,
        )

    @staticmethod
    def _fail_selection_rejection(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _SelectionRejectedData)
        Reasoning._dispatch_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=types.FailureData(message=data.message, turn=data.request.turn),
            target=data.capability.reply_to,
        )

    @staticmethod
    async def _reason_activity(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        with span.operation(
            "bot.reasoning.stage",
            scope="bot.abilities.cognition",
            component="cognition.reasoning",
            stage="reasoning_reason",
            context=telemetry.event_context(event),
        ) as active:
            # A repair re-enters applying with the rejection, which carries the request and the live capability.
            retry = event.data if isinstance(event.data, _SelectionRejectedData) else None
            attempt = retry.attempt + 1 if retry is not None else 0
            previous_normalized = retry.normalized if retry is not None else None
            if retry is not None:
                capability = retry.capability
                operation_id = capability.operation_id
            else:
                operation_id = event.id if event.id else uuid.uuid4().hex
                capability = await Reasoning._start_operation(
                    instance,
                    operation_id,
                    reply_to=event.source if event.target == hsm.id(instance) and event.source else None,
                )
            active.set_attribute("bot.selection.attempt", attempt)
            metadata = dict(event.metadata)

            def dispatch_stage(event_type: hsm.Event[typing.Any], data: object) -> None:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    Reasoning._stage_event(
                        event_type,
                        data,
                        capability=capability,
                        metadata=metadata,
                        target=instance,
                    ),
                )

            def reject(error: processing.SelectionRejectionError) -> None:
                active.set_attribute("bot.selection.rejected", True)
                normalized = types.selection_rejection_key(error.normalized)
                dispatch_stage(
                    _SelectionRejectedEvent,
                    _SelectionRejectedData(
                        request=request,
                        capability=capability,
                        message=types.selection_rejection_message(error.normalized),
                        normalized=normalized,
                        previous_normalized=previous_normalized,
                        attempt=attempt,
                    ),
                )

            request = retry.request if retry is not None else Reasoning._input_from_event(event)
            try:
                # The host's frame narrowed to this stage's effort ceiling. Narrowing reads live
                # actor snapshots, so a frame that cannot be read fails the turn.
                # Reasoning is the handoff's recipient: it never offers the handoff to itself.
                host_input = cognition_input.without_handoff(
                    cognition_input.within_effort(request.processing_input, instance._max_effort)
                )
                # Keep host tools/actors, the host's live instructions (device-state block), and what
                # memory surfaced for this turn; nest the reasoning payload as the process input.
                child_input = processing.Processing._input_for_processor(
                    instance,
                    processing.InputData(
                        input=ProcessorInput(host_input=host_input),
                        schemas=host_input.schemas,
                        actors=host_input.actors,
                        instructions=host_input.instructions,
                        recalled=host_input.recalled,
                    ),
                )
                if retry is not None:
                    child_input = types.with_selection_feedback(child_input, retry.message)
                # Reasoning's selections are dispatched as returned, so what it is still writing may be drafted to
                # its targets as it streams; the drafts carry this operation's id, as the selection will.
                drafts = (
                    processing.Processing._draft_sink(
                        ctx, instance, host_input, operation_id=operation_id, metadata=_public_metadata(metadata)
                    )
                    if host_input.actors
                    else None
                )
                raw = await instance._processor.process(child_input, drafts=drafts)
                reasoned = Reasoning._coerce_reasoned(raw)
            except processing.SelectionRejectionError as error:
                # The processor refused the selection (it did not decode, or fails its schema): the model's to repair.
                reject(error)
                return
            except Exception as error:
                dispatch_stage(
                    _ReasoningStageFailedEvent,
                    _ReasoningStageFailedData(
                        failure=ability.FailureData(message=str(error)),
                        turn=request.turn,
                        capability=capability,
                    ),
                )
                return
            if host_input.actors and reasoned.result:
                try:
                    await types.dispatch_selected_events(
                        ctx,
                        host_input,
                        _selections_from_output(reasoned.result),
                        operation_id=operation_id,
                        source=instance,
                        focus_candidates=request.turn.input.focus_candidates,
                        focused_device=request.turn.input.focus,
                        metadata=_public_metadata(metadata),
                        dispatch_trust=processing.DispatchTrust.MODEL,
                    )
                except processing.SelectionRejectionError as error:
                    # Every selection is validated before any is delivered, so a rejection delivered nothing.
                    reject(error)
                    return
                except Exception as error:
                    dispatch_stage(
                        _ReasoningStageFailedEvent,
                        _ReasoningStageFailedData(
                            failure=ability.FailureData(message=str(error)),
                            turn=request.turn,
                            capability=capability,
                        ),
                    )
                    return
            dispatch_stage(
                _ReasonedEvent,
                _ReasonedEventData(
                    turn=request.turn,
                    capability=capability,
                    host_input=host_input,
                    reasoned=reasoned,
                ),
            )

    @staticmethod
    def _coerce_reasoned(value: object) -> OutputData:
        if isinstance(value, OutputData):
            return value
        if isinstance(value, processing.Result):
            result = typing.cast(processing.Result[object], value)
            if not result.is_handled:
                return OutputData(result=())
            return Reasoning._coerce_reasoned(result.output)
        selections = processing.coerce_event_selections(value)
        if selections is None:
            raise TypeError("Reasoning produced output that does not match its output schema.")
        result = types.OUTPUT_SCHEMA_CONTRACT.validate_python(
            tuple(
                {
                    "event": item.event,
                    "target": item.target,
                    "data": item.data,
                    "reason": item.reason,
                }
                for item in selections
            )
        )
        return OutputData(result=result)

    @staticmethod
    def _complete_from_reasoned(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ReasonedEventData)
        Reasoning._dispatch_output(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            output=types.CompletionData(turn=data.turn, output=data.reasoned.result),
            target=data.capability.reply_to,
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Reasoning",
        hsm.initial(hsm.target("/Reasoning/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event),
            hsm.activity(_initialize_activity),
            hsm.transition(
                hsm.on(_InitializingCompleteEvent),
                hsm.target("/Reasoning/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(_cancel),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_reasoning_input),
                hsm.target("/Reasoning/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_reason_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(_cancel),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.on(_ReasonedEvent),
                hsm.guard(_has_reasoned),
                hsm.effect(_complete_from_reasoned),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.on(_ReasoningStageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.on(_SelectionRejectedEvent),
                hsm.guard(_has_selection_rejection),
                hsm.target("/Reasoning/routing_rejection"),
            ),
        ),
        hsm.choice(
            "routing_rejection",
            hsm.transition(
                hsm.guard(_is_same_selection_rejection),
                hsm.effect(_fail_selection_rejection),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.guard(_can_repair_selection),
                hsm.target("/Reasoning/applying"),
            ),
            hsm.transition(
                hsm.effect(_fail_selection_rejection),
                hsm.target("/Reasoning/idle"),
            ),
        ),
    )

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        processor: ReasoningProcessor,
        instructions: str | None = None,
        max_effort: Effort | None = None,
    ) -> None:
        resolved = type(self).instructions if instructions is None else instructions
        if instructions is not None and not instructions.strip():
            raise ValueError("instructions must not be blank when provided.")
        # Empty default is intentional: system channel is only the per-turn world XML when present.
        # Composition host: leaf Processor is transport; any static policy is ability instructions.
        super(processing.Processing, self).__init__(id=id, clock=clock)
        self._instructions = resolved.strip() if resolved else ""
        self._processor = processor
        self._max_effort = max_effort


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
