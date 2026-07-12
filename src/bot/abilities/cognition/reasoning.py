import bot
from .. import ability
from .. import memory
from .. import processing

import dataclasses
import typing
import uuid

import hsm
import pydantic

from bot.telemetry import observer

from bot.habit import BreakData, ChangeData, CreateData
from . import episodes
from . import types
_REASONING_INPUT_METADATA_KEY = "bot.reasoning.input"
DEFAULT_INSTRUCTIONS = (
    "Return the best typed result for the input; use prior_episodes when present; "
    "set create/change/break only for clear repeated habit patterns; else omit them."
)
_InitializingCompleteEvent = hsm.Event[object](
    name="bot.ability.reasoning.initializing.complete",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)


class InputData(pydantic.BaseModel):
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


class OutputData(pydantic.BaseModel):
    """Deliberate cognitive selection plus optional habit create/change/break."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        json_schema_extra={
            "description": (
                "Deliberate cognitive decision produced by reasoning. result is applied by the host; "
                "set at most one of create, change, or break to record a habit inventory event with the episode."
            ),
            "examples": [
                {
                    "result": [
                        {
                            "target": "phone",
                            "event": "phone.answer_call",
                            "reason": "Incoming call is urgent.",
                        }
                    ],
                    "confidence": 0.77,
                    "reason": "The phone interrupt outranks the current browser task.",
                },
                {
                    "result": [
                        {
                            "target": "phone",
                            "event": "phone.answer_call",
                            "reason": "Ring pattern matches prior episodes.",
                        }
                    ],
                    "create": {
                        "event": "bot.habit.create",
                        "name": "AnswerIncomingRing",
                        "triggers": ["world.sound"],
                        "reason": "Same ring→answer pattern across recalled episodes.",
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
                    "target": "phone",
                    "event": "phone.answer_call",
                    "reason": "Incoming call is urgent.",
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
        examples=["The phone interrupt outranks the current browser task."],
    )
    create: CreateData | None = pydantic.Field(
        default=None,
        description="When set, bot.habit.create payload retained with the episode.",
    )
    change: ChangeData | None = pydantic.Field(
        default=None,
        description="When set, bot.habit.change payload retained with the episode.",
    )
    break_: BreakData | None = pydantic.Field(
        default=None,
        alias="break",
        description="When set, bot.habit.break payload retained with the episode.",
    )

    @pydantic.model_validator(mode="after")
    def validate_single_habit_event(self) -> typing.Self:
        selected = [value for value in (self.create, self.change, self.break_) if value is not None]
        if len(selected) > 1:
            raise ValueError("Set at most one of create, change, or break.")
        return self

    def habit_data(self) -> CreateData | ChangeData | BreakData | None:
        if self.create is not None:
            return self.create
        if self.change is not None:
            return self.change
        if self.break_ is not None:
            return self.break_
        return None


class _RecalledEventData(pydantic.BaseModel):
    """Private completion: memory recall finished; ready to dispatch the processor."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    host_input: processing.InputData
    prior_episodes: tuple[episodes.CognitiveEpisode, ...] = ()
    memory_consulted: bool = False


class _ReasonedEventData(pydantic.BaseModel):
    """Private completion: processor produced a decision; may still need memory retain."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    host_input: processing.InputData
    reasoned: OutputData
    memory_consulted: bool = False


class _RetainedEventData(pydantic.BaseModel):
    """Private completion: episode (and habit learning) written to memory when required."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

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
_ReasoningStageFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.reasoning.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)

ReasoningProcessor: typing.TypeAlias = processing.Processor


def _public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metadata.items() if key != _REASONING_INPUT_METADATA_KEY}


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
    assert isinstance(stimulus, bot.InputEventData)
    return episodes.CognitiveEpisode(
        focus=_context_ref_from_input(input),
        stimulus_name=episodes.stimulus_name(stimulus),
        output=reasoned.result,
        habit=reasoned.habit_data(),
    )


def _should_retain_episode(reasoned: OutputData) -> bool:
    """Retain when habit learnings; still store the episode for future similarity on any success."""

    del reasoned
    return True


class Reasoning(processing.Processing):
    """Slow cognitive ability: inject a ``processing.Processor``, recall, apply, retain.

    Chart phases: initializing → idle → recalling → applying → retaining → idle.
    Turn payload rides private completion events (HSM-COMPLETION-001).
    Memory is an injected sibling collaborator, not owned cognition topology.
    The deliberative ``Processor`` is not an HSM child; Reasoning calls ``process`` in-activity.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = processing.InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.OutputData
    # CallEventKind so intuition (and other model-facing processors) can multi-select reasoning
    # as a normal actor tool — same pattern as speaking.input / device CallEvents.
    input_event: typing.ClassVar[hsm.Event[processing.InputData]] = hsm.Event[processing.InputData](
        name="bot.ability.reasoning.input",
        kind=hsm.CallEventKind,
        schema=processing.InputData,
    )
    output_event: typing.ClassVar[hsm.Event[types.OutputData]] = hsm.Event[types.OutputData](
        name="bot.ability.reasoning.output",
        schema=types.OUTPUT_SCHEMA_CONTRACT,
    )
    instructions: typing.ClassVar[str] = DEFAULT_INSTRUCTIONS
    _processor: processing.Processor
    _instructions: str
    _memory: memory.Memory | None

    @staticmethod
    async def _attach_memory_activity(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        store = instance._memory
        if store is not None:
            memory_owner = ability.Ability.current_owner(store)
            if memory_owner is not None and memory_owner is not instance:
                raise ValueError(f"{type(store).__name__} is already owned by {type(memory_owner).__name__}.")
            _ = await store.attach(owner=instance, ctx=ctx)
        _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(None))

    @staticmethod
    def _detach_memory_on_detach(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        if event.name != ability.DetachEvent.name:
            return
        store = instance._memory
        if store is not None and ability.Ability.current_owner(store) is instance:
            _ = store.detach(ctx=ctx)

    @staticmethod
    def _has_reasoning_input(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, processing.InputData)

    @staticmethod
    def _has_recalled(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _RecalledEventData)

    @staticmethod
    def _has_reasoned(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _ReasonedEventData)

    @staticmethod
    def _has_retained(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _RetainedEventData)

    @staticmethod
    def _has_stage_failure(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _dispatch_failure(
        ctx: hsm.Context,
        instance: "Reasoning",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        failure: ability.FailureData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=operation_id,
            metadata=_public_metadata(metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _dispatch_output(
        ctx: hsm.Context,
        instance: "Reasoning",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        output: types.OutputData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=_public_metadata(metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _fail_from_stage(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        Reasoning._dispatch_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=data,
        )

    @staticmethod
    async def _recall_activity(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, processing.InputData)
        input = typing.cast(processing.InputData, data)
        operation_id = event.id or None
        metadata = dict(event.metadata)
        store = instance._memory
        if store is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _RecalledEvent.with_data(
                        _RecalledEventData(host_input=input, prior_episodes=(), memory_consulted=False)
                    ),
                    id=operation_id,
                    metadata=metadata,
                ),
            )
            return
        try:
            select_input = episodes.episode_select_input(context_ref=_context_ref_from_input(input))
            recalled = store.execute(select_input)
            prior = episodes.episodes_from_output(recalled)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ReasoningStageFailedEvent.with_data(
                        ability.FailureData(message=f"Reasoning memory recall failed: {error}")
                    ),
                    id=operation_id,
                    metadata=metadata,
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _RecalledEvent.with_data(_RecalledEventData(host_input=input, prior_episodes=prior, memory_consulted=True)),
                id=operation_id,
                metadata=metadata,
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
    async def _apply_activity(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RecalledEventData)
        operation_id = event.id if event.id else uuid.uuid4().hex
        metadata = dict(event.metadata)
        metadata[_REASONING_INPUT_METADATA_KEY] = data.host_input
        metadata["bot.reasoning.memory_consulted"] = data.memory_consulted
        reasoning_input = InputData(
            host_input=data.host_input,
            prior_episodes=data.prior_episodes,
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
                dataclasses.replace(
                    _ReasoningStageFailedEvent.with_data(ability.FailureData(message=str(error))),
                    id=operation_id,
                    metadata=_public_metadata(metadata),
                ),
            )
            return
        if data.host_input.actors and reasoned.result:
            try:
                processing.dispatch_selected_events(
                    ctx,
                    data.host_input,
                    _selections_from_output(reasoned.result),
                    metadata=_public_metadata(metadata),
                )
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ReasoningStageFailedEvent.with_data(ability.FailureData(message=str(error))),
                        id=operation_id,
                        metadata=_public_metadata(metadata),
                    ),
                )
                return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ReasonedEvent.with_data(
                    _ReasonedEventData(
                        host_input=data.host_input,
                        reasoned=reasoned,
                        memory_consulted=data.memory_consulted,
                    )
                ),
                id=operation_id,
                metadata=_public_metadata(metadata),
            ),
        )

    @staticmethod
    def _reasoned_needs_retain(ctx: hsm.Context, instance: "Reasoning", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _ReasonedEventData):
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
            output=data.reasoned.result,
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
            output=data.result,
        )

    @staticmethod
    async def _retain_activity(
        ctx: hsm.Context,
        instance: "Reasoning",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _ReasonedEventData)
        operation_id = event.id or None
        metadata = dict(event.metadata)
        store = instance._memory
        if store is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _RetainedEvent.with_data(
                        _RetainedEventData(
                            result=data.reasoned.result,
                            memory_consulted=data.memory_consulted,
                            memory_written=False,
                        )
                    ),
                    id=operation_id,
                    metadata=metadata,
                ),
            )
            return
        episode = _episode_from_reasoning(data.host_input, data.reasoned)
        try:
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
                dataclasses.replace(
                    _ReasoningStageFailedEvent.with_data(
                        ability.FailureData(message=f"Reasoning memory retain failed: {error}")
                    ),
                    id=operation_id,
                    metadata=metadata,
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _RetainedEvent.with_data(
                    _RetainedEventData(
                        result=data.reasoned.result,
                        memory_consulted=data.memory_consulted,
                        memory_written=True,
                    )
                ),
                id=operation_id,
                metadata=metadata,
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Reasoning",
        hsm.initial(hsm.target("/Reasoning/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event),
            hsm.activity(_attach_memory_activity),
            hsm.exit(_detach_memory_on_detach),
            hsm.transition(
                hsm.on(_InitializingCompleteEvent),
                hsm.target("/Reasoning/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.exit(_detach_memory_on_detach),
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
            hsm.exit(_detach_memory_on_detach),
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
            hsm.activity(_apply_activity),
            hsm.exit(_detach_memory_on_detach),
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
            hsm.exit(_detach_memory_on_detach),
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
    "InputEvent",
    "OutputEvent",
    "Reasoning",
    "InputData",
    "OutputData",
    "ReasoningProcessor",
]
