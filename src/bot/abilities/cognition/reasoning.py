import bot
from .. import ability
from .. import memory
from .. import processing

import dataclasses
import typing
import uuid

import hsm
from bot import event_schema

import pydantic

from bot.telemetry import observer

from bot.behavior import BreakData, ChangeData, CreateData
from . import directives
from . import episodes
from . import types

DEFAULT_INSTRUCTIONS = (
    "Return the best typed result for the input; use prior_episodes when present; "
    "treat standing_directives as things you were told and still owe, and decide for yourself "
    "whether this moment is one to act on them; "
    "set create/change/break only for clear repeated behavior patterns; else omit them."
)
_InitializingCompleteEvent = hsm.Event[object](
    name="bot.ability.reasoning.initializing.complete",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)


class CallData(pydantic.BaseModel):
    """Model-facing request to run deliberate (System-2) reasoning on the current turn.

    The host reuses the live processing frame (stimulus, tools, actors). Do not restate the
    stimulus, schemas, or instructions here — leave ``data`` empty / omit fields.
    Prefer selecting this together with ``bot.ability.speaking.input`` when the user is waiting
    on speech while deliberation runs.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Invoke deliberate reasoning for the current turn. The host supplies the deliberative "
                "frame; do not include stimulus text, tools, or schemas in data. Use an empty object "
                "as data. Often multi-selected with speaking.input for a short spoken bridge."
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
    """Deliberative reasoning input: host input plus memory-recalled prior episodes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Deliberative reasoning input. Preserves the original host decision input and any prior "
                "similar cognitive episodes recalled from the memory collaborator before the processor runs."
            ),
        },
    )

    host_input: processing.InputData = pydantic.Field(
        description="Original host processing input being evaluated.",
    )
    prior_episodes: tuple[episodes.CognitiveEpisode, ...] = pydantic.Field(
        default=(),
        description=(
            "Prior or similar cognition episodes recalled from memory for this turn. Empty when no "
            "memory collaborator is configured or nothing matched."
        ),
    )
    standing_directives: tuple[directives.Directive, ...] = pydantic.Field(
        default=(),
        description=(
            "Standing instructions the bot was told and has not been released from, recalled from "
            "memory for this turn. Empty when no memory collaborator is configured or nothing "
            "matched. Whether this turn is the one to act on them is your decision."
        ),
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


class _RecalledEventData(pydantic.BaseModel):
    """Private completion: memory recall finished; ready to dispatch the processor."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    capability: _ReasoningCapability
    host_input: processing.InputData
    prior_episodes: tuple[episodes.CognitiveEpisode, ...] = ()
    standing_directives: tuple[directives.Directive, ...] = ()
    memory_consulted: bool = False


class _ReasonedEventData(pydantic.BaseModel):
    """Private completion: processor produced a decision; may still need memory retain."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    capability: _ReasoningCapability
    host_input: processing.InputData
    reasoned: OutputData
    memory_consulted: bool = False


class _RetainedEventData(pydantic.BaseModel):
    """Private completion: episode (and behavior learning) written to memory when required."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    turn: types.TurnData
    capability: _ReasoningCapability
    result: types.OutputData
    memory_consulted: bool = False
    memory_written: bool = False


_RecalledEvent = hsm.Event[_RecalledEventData](
    name="bot.ability.reasoning.recalled",
    kind=hsm.CompletionEventKind,
    schema=_RecalledEventData,
)
_ReasonedEvent = hsm.Event[_ReasonedEventData](
    name="bot.ability.reasoning.reasoned",
    kind=hsm.CompletionEventKind,
    schema=_ReasonedEventData,
)
_RetainedEvent = hsm.Event[_RetainedEventData](
    name="bot.ability.reasoning.retained",
    kind=hsm.CompletionEventKind,
    schema=_RetainedEventData,
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


def _context_ref_from_input(input: processing.InputData) -> str | None:
    """Best-effort context partition for episode recall/retain (device/focus-like identity)."""

    stimulus = input.input
    if isinstance(stimulus, bot.InputEventData) and stimulus.target_device:
        return stimulus.target_device
    return None


def _episode_from_reasoning(
    input: processing.InputData,
    reasoned: OutputData,
) -> episodes.CognitiveEpisode:
    stimulus = input.input
    if not isinstance(stimulus, (bot.InputEventData, hsm.Event)):
        raise TypeError(f"Reasoning cannot retain unsupported stimulus type {type(stimulus)!r}.")
    return episodes.CognitiveEpisode(
        focus=_context_ref_from_input(input),
        stimulus_name=episodes.stimulus_name(stimulus),
        output=reasoned.result,
        behavior=reasoned.behavior_data(),
    )


def _should_retain_episode(reasoned: OutputData) -> bool:
    """Retain when behavior learnings; still store the episode for future similarity on any success."""

    del reasoned
    return True


class Reasoning(processing.Processing):
    """Slow cognitive ability: inject a ``processing.Processor``, recall, apply, retain.

    Chart phases: initializing → idle → recalling → applying → retaining → idle.
    Turn payload rides private completion events (HSM-COMPLETION-001).
    Memory is an injected sibling collaborator, not owned cognition topology.
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
        name="bot.ability.reasoning.input",
        kind=event_schema.EventKind,
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
    _memory: memory.Memory | None

    @staticmethod
    async def _initialize_activity(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(None))

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
                _RecalledEventData | _ReasonedEventData | _RetainedEventData | _ReasoningStageFailedData,
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
    def _has_recalled(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _RecalledEventData) and Reasoning._matches_operation(instance, event)

    @staticmethod
    def _has_reasoned(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _ReasonedEventData) and Reasoning._matches_operation(instance, event)

    @staticmethod
    def _has_retained(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _RetainedEventData) and Reasoning._matches_operation(instance, event)

    @staticmethod
    def _has_stage_failure(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _ReasoningStageFailedData) and Reasoning._matches_operation(instance, event)

    @staticmethod
    async def _start_operation(instance: "Reasoning", operation_id: str) -> _ReasoningCapability:
        operation = processing.active_operation(instance, operation_id)
        if operation is None:
            operation = await processing.start_operation(instance, operation_id)
        return _ReasoningCapability(
            operation_id=operation_id,
            actor_id=hsm.id(operation),
            token=uuid.uuid4().hex,
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
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=operation_id,
            metadata=_public_metadata(metadata),
            source=hsm.id(instance),
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
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=_public_metadata(metadata),
            source=hsm.id(instance),
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
        )

    @staticmethod
    async def _recall_activity(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        operation_id = event.id if event.id else uuid.uuid4().hex
        capability = await Reasoning._start_operation(instance, operation_id)
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

        try:
            request = Reasoning._input_from_event(event)
            input = request.processing_input
        except TypeError as error:
            fallback_turn = types.TurnData(
                input=typing.cast(InputData, event.data).turn.input,
                operation_id=operation_id,
                generation=typing.cast(InputData, event.data).turn.generation,
            )
            dispatch_stage(
                _ReasoningStageFailedEvent,
                _ReasoningStageFailedData(
                    failure=ability.FailureData(message=str(error)),
                    turn=fallback_turn,
                    capability=capability,
                ),
            )
            return
        store = instance._memory
        if store is None:
            dispatch_stage(
                _RecalledEvent,
                _RecalledEventData(
                    turn=request.turn,
                    capability=capability,
                    host_input=input,
                    prior_episodes=(),
                    standing_directives=(),
                    memory_consulted=False,
                ),
            )
            return
        try:
            context_ref = _context_ref_from_input(input)
            # One apply, one transaction, two statements: prior turns and standing directives
            # are recalled together so a turn never reasons from half a memory.
            select_input = memory.InputData(
                statements=(
                    *episodes.episode_select_input(context_ref=context_ref).statements,
                    *directives.directive_select_input(context_ref=context_ref).statements,
                )
            )
            recalled = store.execute(select_input)
            prior = episodes.episodes_from_output(recalled, statement_index=0)
            standing = directives.directives_from_output(recalled, statement_index=1)
        except Exception as error:
            dispatch_stage(
                _ReasoningStageFailedEvent,
                _ReasoningStageFailedData(
                    failure=ability.FailureData(message=f"Reasoning memory recall failed: {error}"),
                    turn=request.turn,
                    capability=capability,
                ),
            )
            return
        dispatch_stage(
            _RecalledEvent,
            _RecalledEventData(
                turn=request.turn,
                capability=capability,
                host_input=input,
                prior_episodes=prior,
                standing_directives=standing,
                memory_consulted=True,
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
    async def _reason_activity(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RecalledEventData)
        capability = data.capability
        operation_id = capability.operation_id
        metadata = {
            **dict(event.metadata),
            "bot.reasoning.memory_consulted": data.memory_consulted,
        }
        reasoning_input = ProcessorInput(
            host_input=data.host_input,
            prior_episodes=data.prior_episodes,
            standing_directives=data.standing_directives,
        )
        # Keep host tools/actors; nest reasoning payload as the process input.
        child_input = processing.Processing._input_for_processor(
            instance,
            processing.InputData(
                input=reasoning_input,
                schemas=data.host_input.schemas,
                actors=data.host_input.actors,
            ),
        )
        try:
            raw = await instance._processor.process(child_input)
            reasoned = Reasoning._coerce_reasoned(raw)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reasoning._stage_event(
                    _ReasoningStageFailedEvent,
                    _ReasoningStageFailedData(
                        failure=ability.FailureData(message=str(error)),
                        turn=data.turn,
                        capability=capability,
                    ),
                    capability=capability,
                    metadata=metadata,
                    target=instance,
                ),
            )
            return
        if data.host_input.actors and reasoned.result:
            try:
                await types.dispatch_selected_events(
                    ctx,
                    data.host_input,
                    _selections_from_output(reasoned.result),
                    operation_id=operation_id,
                    source=instance,
                    focus_candidates=data.turn.input.focus_candidates,
                    focused_device=data.turn.input.focus,
                    metadata=_public_metadata(metadata),
                )
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    Reasoning._stage_event(
                        _ReasoningStageFailedEvent,
                        _ReasoningStageFailedData(
                            failure=ability.FailureData(message=str(error)),
                            turn=data.turn,
                            capability=capability,
                        ),
                        capability=capability,
                        metadata=metadata,
                        target=instance,
                    ),
                )
                return
        _ = hsm.dispatch(
            ctx,
            instance,
            Reasoning._stage_event(
                _ReasonedEvent,
                _ReasonedEventData(
                    turn=data.turn,
                    capability=capability,
                    host_input=data.host_input,
                    reasoned=reasoned,
                    memory_consulted=data.memory_consulted,
                ),
                capability=capability,
                metadata=metadata,
                target=instance,
            ),
        )

    @staticmethod
    def _reasoned_needs_retain(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _ReasonedEventData) or not Reasoning._matches_operation(instance, event):
            return False
        return instance._memory is not None and _should_retain_episode(data.reasoned)

    @staticmethod
    def _reasoned_skips_retain(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        return Reasoning._has_reasoned(ctx, instance, event) and not Reasoning._reasoned_needs_retain(
            ctx, instance, event
        )

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
        )

    @staticmethod
    def _complete_from_retained(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RetainedEventData)
        Reasoning._dispatch_output(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            output=types.CompletionData(turn=data.turn, output=data.result),
        )

    @staticmethod
    async def _retain_activity(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _ReasonedEventData)
        capability = data.capability
        metadata = dict(event.metadata)
        store = instance._memory
        if store is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reasoning._stage_event(
                    _RetainedEvent,
                    _RetainedEventData(
                        turn=data.turn,
                        capability=capability,
                        result=data.reasoned.result,
                        memory_consulted=data.memory_consulted,
                        memory_written=False,
                    ),
                    capability=capability,
                    metadata=metadata,
                    target=instance,
                ),
            )
            return
        try:
            episode = _episode_from_reasoning(data.host_input, data.reasoned)
            insert_input = episodes.episode_insert_input(
                episode,
                context_ref=_context_ref_from_input(data.host_input),
                scope=getattr(type(store), "default_scope", "short_term"),
            )
            _ = store.execute(insert_input)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reasoning._stage_event(
                    _ReasoningStageFailedEvent,
                    _ReasoningStageFailedData(
                        failure=ability.FailureData(message=f"Reasoning memory retain failed: {error}"),
                        turn=data.turn,
                        capability=capability,
                    ),
                    capability=capability,
                    metadata=metadata,
                    target=instance,
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            Reasoning._stage_event(
                _RetainedEvent,
                _RetainedEventData(
                    turn=data.turn,
                    capability=capability,
                    result=data.reasoned.result,
                    memory_consulted=data.memory_consulted,
                    memory_written=True,
                ),
                capability=capability,
                metadata=metadata,
                target=instance,
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
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
                hsm.target("/Reasoning/recalling"),
            ),
        ),
        hsm.state(
            "recalling",
            hsm.defer(input_event),
            hsm.activity(_recall_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(_cancel),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.on(_RecalledEvent),
                hsm.guard(_has_recalled),
                hsm.target("/Reasoning/applying"),
            ),
            hsm.transition(
                hsm.on(_ReasoningStageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reasoning/idle"),
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
                hsm.guard(_reasoned_needs_retain),
                hsm.target("/Reasoning/retaining"),
            ),
            hsm.transition(
                hsm.on(_ReasonedEvent),
                hsm.guard(_reasoned_skips_retain),
                hsm.effect(_complete_from_reasoned),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.on(_ReasoningStageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reasoning/idle"),
            ),
        ),
        hsm.state(
            "retaining",
            hsm.defer(input_event),
            hsm.activity(_retain_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(_cancel),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.on(_RetainedEvent),
                hsm.guard(_has_retained),
                hsm.effect(_complete_from_retained),
                hsm.target("/Reasoning/idle"),
            ),
            hsm.transition(
                hsm.on(_ReasoningStageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reasoning/idle"),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        processor: ReasoningProcessor,
        instructions: str | None = None,
        memory: memory.Memory | None = None,
    ) -> None:
        resolved = type(self).instructions if instructions is None else instructions
        if not resolved.strip():
            raise ValueError("instructions must not be blank.")
        # Composition host: leaf Processor is transport; system policy is ability instructions.
        ability.Ability.__init__(self)
        self._instructions = resolved.strip()
        self._processor = processor
        self._memory = memory


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
