import bot
from .. import ability
from .. import processing

import dataclasses
import typing
import uuid

import hsm
import pydantic

from bot.telemetry import observer

from .dispatch import build_processing_input
from .input import InputData, is_input
from . import autonomy
from . import intuition
from . import reasoning
from . import reflection
from .types import OUTPUT_SCHEMA_CONTRACT, EventData, OutputData, is_output

# Private event-chain key: carries this turn's cognition input with child requests.
# Not a metric/span attribute; not stored on the cognition instance (HSM-COMPLETION-001).
_COGNITION_INPUT_METADATA_KEY = "bot.cognition.input"
_AUTONOMY_ID_SUFFIX = ":autonomy"
_INTUITION_ID_SUFFIX = ":intuition"
_REASONING_ID_SUFFIX = ":reasoning"


class _ProcessingCompletedEventData(pydantic.BaseModel):
    """Private completion payload carrying applied output and the original cognition turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    output: OutputData = pydantic.Field(
        description="Typed cognition output selected for this turn.",
    )
    cognition_input: InputData = pydantic.Field(
        description="Original cognition input for this turn (for reflection).",
    )
    operation_id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Private apply-operation identifier from the input event.",
    )
    emit_terminal: bool = pydantic.Field(
        default=True,
        description="When false, skip ability terminal output.",
    )


class _UseFailedEventData(pydantic.BaseModel):
    """Private failure payload carrying failure details and operation identity."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    failure: ability.FailureData = pydantic.Field(
        description="FailureData signal produced while the cognition decision was active.",
    )
    operation_id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Private apply-operation identifier from the input event.",
    )


InputEvent = ability.ability_input_event(
    "bot.ability.cognition.input",
    InputData,
)
OutputEvent = hsm.Event[OutputData](
    name="bot.ability.cognition.output",
    schema=OUTPUT_SCHEMA_CONTRACT,
)
_ProcessingCompletedEvent = hsm.Event[_ProcessingCompletedEventData](
    name="bot.ability.cognition.processing.completed",
    kind=hsm.CompletionEventKind,
    schema=_ProcessingCompletedEventData,
)
_ApplyFailedEvent = hsm.Event[_UseFailedEventData](
    name="bot.ability.cognition.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=_UseFailedEventData,
)
_InitializingCompleteEvent = hsm.Event[object](
    name="bot.ability.cognition.initializing.complete",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)


def _failure(operation_id: str | None, message: str) -> _UseFailedEventData:
    return _UseFailedEventData(
        failure=ability.FailureData(message=message),
        operation_id=operation_id,
    )


def _coerce_output(output: object) -> OutputData:
    if isinstance(output, processing.Result):
        result = typing.cast(processing.Result[object], output)
        if not result.is_handled:
            return ()
        return _coerce_output(result.output)
    if output is None:
        return ()
    if is_output(output):
        return output
    if isinstance(output, EventData):
        return (output,)
    selections = processing.coerce_event_selections(output)
    if selections is not None:
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python(
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
        return typing.cast(OutputData, validated)
    if isinstance(output, list | tuple):
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python(tuple(output))
        return typing.cast(OutputData, validated)
    if isinstance(output, dict) and "event" in output:
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python((output,))
        return typing.cast(OutputData, validated)
    raise TypeError("Cognition processing produced output that does not match its output schema.")


def _owner(instance: "Cognition") -> typing.Any | None:
    """Body owner used for focus/clear dispatch (duck-typed; no Bot import)."""

    return ability.Ability.current_owner(instance)


def _public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Metadata safe to forward on host apply / terminals (no turn payload)."""

    return {key: value for key, value in metadata.items() if key != _COGNITION_INPUT_METADATA_KEY}


def _cognition_input_from_event(event: hsm.Event[typing.Any]) -> InputData | None:
    value = event.metadata.get(_COGNITION_INPUT_METADATA_KEY)
    if is_input(value):
        return value
    return None


def _parent_operation_id(event: hsm.Event[typing.Any]) -> str | None:
    child_id = event.id if event.id else None
    if child_id is None:
        return None
    for suffix in (_AUTONOMY_ID_SUFFIX, _INTUITION_ID_SUFFIX, _REASONING_ID_SUFFIX):
        if child_id.endswith(suffix):
            parent = child_id[: -len(suffix)]
            return parent or None
    return child_id


class Cognition(ability.Ability[InputData, OutputData]):
    """Judgment ability: autonomy → intuition → reasoning; optional reflection.

    Chart states are lifecycle phases only. The active turn rides the event chain
    (child request metadata → child terminal → private completion), not instance fields
    (HSM-COMPLETION-001).
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent
    _autonomy: autonomy.Autonomy | None
    _intuition: intuition.Intuition
    _reasoning: reasoning.Reasoning
    _reflection: reflection.Reflection | None

    @staticmethod
    def _autonomy_operation_id(event: hsm.Event[typing.Any]) -> str:
        operation_id = event.id if event.id else uuid.uuid4().hex
        return f"{operation_id}{_AUTONOMY_ID_SUFFIX}"

    @staticmethod
    def _intuition_operation_id(event: hsm.Event[typing.Any]) -> str:
        operation_id = event.id if event.id else uuid.uuid4().hex
        return f"{operation_id}{_INTUITION_ID_SUFFIX}"

    @staticmethod
    def _reasoning_operation_id(operation_id: str | None) -> str:
        base = operation_id if operation_id else uuid.uuid4().hex
        return f"{base}{_REASONING_ID_SUFFIX}"

    @staticmethod
    def _reflection_operation_id(operation_id: str | None) -> str:
        base = operation_id if operation_id else uuid.uuid4().hex
        return f"{base}:reflection"

    @staticmethod
    def _cognition_children(instance: "Cognition") -> list[ability.Ability[typing.Any, typing.Any]]:
        children: list[ability.Ability[typing.Any, typing.Any]] = []
        if instance._autonomy is not None:
            children.append(instance._autonomy)
        children.extend((instance._intuition, instance._reasoning))
        if instance._reflection is not None:
            children.append(instance._reflection)
        return children

    @staticmethod
    async def _attach_children_activity(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        for child in Cognition._cognition_children(instance):
            owner = ability.Ability.current_owner(child)
            if owner is not None and owner is not instance:
                raise ValueError(f"{type(child).__name__} is already owned by {type(owner).__name__}.")
            _ = await child.attach(owner=instance, ctx=ctx)
        _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(None))

    @staticmethod
    def _detach_children_on_detach(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if event.name != ability.DetachEvent.name:
            return
        for child in Cognition._cognition_children(instance):
            if ability.Ability.current_owner(child) is instance:
                _ = child.detach(ctx=ctx)

    @staticmethod
    def _build_processing_input(
        instance: "Cognition",
        cognition_input: InputData,
    ) -> processing.InputData:
        # Reasoning is a normal actor/tool for intuition multi-select — not a hardcoded edge.
        return build_processing_input(
            cognition_input,
            owner=_owner(instance),
            extra_actors={"reasoning": instance._reasoning},
        )

    @staticmethod
    def _start_autonomy(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert is_input(data)
        autonomy_ability = instance._autonomy
        assert autonomy_ability is not None
        child_metadata = dict(event.metadata)
        child_metadata[_COGNITION_INPUT_METADATA_KEY] = data
        input_event = dataclasses.replace(
            autonomy_ability.input_event.with_data_and_id(
                data,
                Cognition._autonomy_operation_id(event),
            ),
            metadata=child_metadata,
        )
        _ = hsm.dispatch(ctx, autonomy_ability, input_event)

    @staticmethod
    def _begin_turn(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Start autonomy when injected; otherwise open with intuition."""

        if instance._autonomy is not None:
            Cognition._start_autonomy(ctx, instance, event)
            return
        Cognition._start_intuition_from_input(ctx, instance, event)

    @staticmethod
    def _start_intuition_from_input(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert is_input(data)
        operation_id = event.id if event.id else None
        try:
            input = Cognition._build_processing_input(instance, data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=_public_metadata(dict(event.metadata)),
                ),
            )
            return
        child_metadata = dict(event.metadata)
        child_metadata[_COGNITION_INPUT_METADATA_KEY] = data
        input_event = dataclasses.replace(
            instance._intuition.input_event.with_data_and_id(
                input,
                Cognition._intuition_operation_id(event),
            ),
            metadata=child_metadata,
        )
        _ = hsm.dispatch(ctx, instance._intuition, input_event)

    @staticmethod
    def _matches_autonomy_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._autonomy
        if child is None:
            return False
        child_id = event.id if event.id else None
        return (
            event.name == child.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(child)
            and child_id is not None
            and child_id.endswith(_AUTONOMY_ID_SUFFIX)
            and _cognition_input_from_event(event) is not None
        )

    @staticmethod
    def _matches_autonomy_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._autonomy
        if child is None:
            return False
        child_id = event.id if event.id else None
        return (
            event.name == child.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(child)
            and child_id is not None
            and child_id.endswith(_AUTONOMY_ID_SUFFIX)
        )

    @staticmethod
    def _autonomy_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Cognition._matches_autonomy_output(ctx, instance, event) and event.data is not None

    @staticmethod
    def _autonomy_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Cognition._matches_autonomy_output(ctx, instance, event) and event.data is None

    @staticmethod
    def _complete_from_autonomy(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        cognition_input = _cognition_input_from_event(event)
        operation_id = _parent_operation_id(event)
        public_metadata = _public_metadata(dict(event.metadata))
        if cognition_input is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        _failure(operation_id, "Cognition autonomy output is missing turn correlation.")
                    ),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        try:
            output = _coerce_output(event.data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        completion = _ProcessingCompletedEventData(
            output=output,
            cognition_input=cognition_input,
            operation_id=operation_id,
            emit_terminal=True,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ProcessingCompletedEvent.with_data(completion),
                id=operation_id,
                metadata=public_metadata,
            ),
        )

    @staticmethod
    def _start_intuition(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Start intuition from an autonomy unhandled terminal (input rides metadata)."""

        cognition_input = _cognition_input_from_event(event)
        operation_id = _parent_operation_id(event)
        public_metadata = _public_metadata(dict(event.metadata))
        if cognition_input is None:
            # Idle path still passes the input event body.
            if is_input(event.data):
                Cognition._start_intuition_from_input(ctx, instance, event)
                return
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        _failure(operation_id, "Cognition intuition start is missing turn correlation.")
                    ),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        try:
            input = Cognition._build_processing_input(instance, cognition_input)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        child_metadata = dict(event.metadata)
        child_metadata[_COGNITION_INPUT_METADATA_KEY] = cognition_input
        base = operation_id if operation_id else uuid.uuid4().hex
        input_event = dataclasses.replace(
            instance._intuition.input_event.with_data_and_id(
                input,
                f"{base}{_INTUITION_ID_SUFFIX}",
            ),
            metadata=child_metadata,
        )
        _ = hsm.dispatch(ctx, instance._intuition, input_event)

    @staticmethod
    def _matches_intuition_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._intuition
        child_id = event.id if event.id else None
        return (
            event.name == child.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(child)
            and child_id is not None
            and child_id.endswith(_INTUITION_ID_SUFFIX)
            and _cognition_input_from_event(event) is not None
        )

    @staticmethod
    def _matches_intuition_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._intuition
        child_id = event.id if event.id else None
        return (
            event.name == child.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(child)
            and child_id is not None
            and child_id.endswith(_INTUITION_ID_SUFFIX)
        )

    @staticmethod
    def _intuition_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Cognition._matches_intuition_output(ctx, instance, event) and event.data is not None

    @staticmethod
    def _intuition_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        # Host cascade only: unhandled means Cognition starts reasoning. Handled multi-select
        # (including reasoning.input as a normal actor) is intuition's job via dispatch.
        return Cognition._matches_intuition_output(ctx, instance, event) and event.data is None

    @staticmethod
    def _complete_from_intuition(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        cognition_input = _cognition_input_from_event(event)
        operation_id = _parent_operation_id(event)
        public_metadata = _public_metadata(dict(event.metadata))
        if cognition_input is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        _failure(operation_id, "Cognition intuition output is missing turn correlation.")
                    ),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        try:
            output = _coerce_output(event.data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        completion = _ProcessingCompletedEventData(
            output=output,
            cognition_input=cognition_input,
            operation_id=operation_id,
            emit_terminal=True,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ProcessingCompletedEvent.with_data(completion),
                id=operation_id,
                metadata=public_metadata,
            ),
        )

    @staticmethod
    def _start_reasoning(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        cognition_input = _cognition_input_from_event(event)
        operation_id = _parent_operation_id(event)
        public_metadata = _public_metadata(dict(event.metadata))
        if cognition_input is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        _failure(operation_id, "Cognition reasoning start is missing turn correlation.")
                    ),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        try:
            input = Cognition._build_processing_input(instance, cognition_input)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        child_metadata = dict(event.metadata)
        child_metadata[_COGNITION_INPUT_METADATA_KEY] = cognition_input
        input_event = dataclasses.replace(
            instance._reasoning.input_event.with_data_and_id(
                input,
                Cognition._reasoning_operation_id(operation_id),
            ),
            metadata=child_metadata,
        )
        _ = hsm.dispatch(ctx, instance._reasoning, input_event)

    @staticmethod
    def _matches_reasoning_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reasoning
        child_id = event.id if event.id else None
        return (
            event.name == child.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(child)
            and child_id is not None
            and child_id.endswith(_REASONING_ID_SUFFIX)
            and _cognition_input_from_event(event) is not None
        )

    @staticmethod
    def _matches_reasoning_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reasoning
        child_id = event.id if event.id else None
        return (
            event.name == child.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(child)
            and child_id is not None
            and child_id.endswith(_REASONING_ID_SUFFIX)
        )

    @staticmethod
    def _complete_from_reasoning(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        cognition_input = _cognition_input_from_event(event)
        operation_id = _parent_operation_id(event)
        public_metadata = _public_metadata(dict(event.metadata))
        if cognition_input is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        _failure(operation_id, "Cognition reasoning output is missing turn correlation.")
                    ),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        try:
            output = _coerce_output(event.data)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(_failure(operation_id, str(error))),
                    id=operation_id,
                    metadata=public_metadata,
                ),
            )
            return
        completion = _ProcessingCompletedEventData(
            output=output,
            cognition_input=cognition_input,
            operation_id=operation_id,
            emit_terminal=True,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ProcessingCompletedEvent.with_data(completion),
                id=operation_id,
                metadata=public_metadata,
            ),
        )

    @staticmethod
    def _fail_child(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        operation_id = _parent_operation_id(event)
        failure = (
            event.data
            if isinstance(event.data, ability.FailureData)
            else ability.FailureData(message="Cognition ability failed.")
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyFailedEvent.with_data(_UseFailedEventData(failure=failure, operation_id=operation_id)),
                id=operation_id,
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _dispatch_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        completion = event.data
        assert isinstance(completion, _ProcessingCompletedEventData)
        if not completion.emit_terminal:
            return
        terminal = dataclasses.replace(
            instance.output_event.with_data(completion.output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_reflection(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Fire-and-forget habit reflection from the completion event only."""

        reflection_ability = instance._reflection
        if reflection_ability is None:
            return
        completion = event.data
        assert isinstance(completion, _ProcessingCompletedEventData)
        turn = reflection.InputData(
            cognition_input=completion.cognition_input,
            cognition_output=completion.output,
        )
        input = processing.InputData(input=turn)
        input_event = dataclasses.replace(
            reflection_ability.input_event.with_data_and_id(
                input,
                Cognition._reflection_operation_id(completion.operation_id),
            ),
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, reflection_ability, input_event)

    @staticmethod
    def _dispatch_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _UseFailedEventData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data.failure),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _has_autonomy(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._autonomy is not None

    @staticmethod
    def _has_input(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        data = event.data
        if not is_input(data):
            return False
        return isinstance(data.stimulus, bot.InputEventData | hsm.Event)

    @staticmethod
    def _has_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _ProcessingCompletedEventData)

    @staticmethod
    def _has_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _UseFailedEventData)

    @staticmethod
    def _has_matching_apply_operation(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        operation_id = event.id if event.id else None
        data = event.data
        if isinstance(data, (_ProcessingCompletedEventData, _UseFailedEventData)):
            return data.operation_id == operation_id
        return operation_id is None

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Cognition",
        hsm.initial(hsm.target("/Cognition/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event),
            hsm.activity(_attach_children_activity),
            hsm.exit(_detach_children_on_detach),
            hsm.transition(
                hsm.on(_InitializingCompleteEvent),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.exit(_detach_children_on_detach),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.effect(_begin_turn),
                hsm.target("/Cognition/routing"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
            ),
        ),
        hsm.choice(
            "routing",
            hsm.transition(
                hsm.guard(_has_autonomy),
                hsm.target("/Cognition/autonomizing"),
            ),
            hsm.transition(
                hsm.target("/Cognition/intuiting"),
            ),
        ),
        hsm.state(
            "autonomizing",
            hsm.defer(input_event),
            hsm.exit(_detach_children_on_detach),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_autonomy_is_handled),
                hsm.effect(_complete_from_autonomy),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_autonomy_is_unhandled),
                hsm.effect(_start_intuition),
                hsm.target("/Cognition/intuiting"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_autonomy_failure),
                hsm.effect(_fail_child),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "intuiting",
            hsm.defer(input_event),
            hsm.exit(_detach_children_on_detach),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_intuition_is_handled),
                hsm.effect(_complete_from_intuition),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_intuition_is_unhandled),
                hsm.effect(_start_reasoning),
                hsm.target("/Cognition/reasoning"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_intuition_failure),
                hsm.effect(_fail_child),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "reasoning",
            hsm.defer(input_event),
            hsm.exit(_detach_children_on_detach),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_reasoning_output),
                hsm.effect(_complete_from_reasoning),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_reasoning_failure),
                hsm.effect(_fail_child),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "completing",
            hsm.defer(input_event),
            hsm.exit(_detach_children_on_detach),
            hsm.transition(
                hsm.on(_ProcessingCompletedEvent),
                hsm.guard(_has_matching_apply_operation),
                hsm.guard(_has_output),
                hsm.effect(_dispatch_output, _dispatch_reflection),
                hsm.target("/Cognition/idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_matching_apply_operation),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        intuition: intuition.Intuition,
        reasoning: reasoning.Reasoning,
        autonomy: autonomy.Autonomy | None = None,
        reflection: reflection.Reflection | None = None,
    ) -> None:
        super().__init__()
        self._autonomy = autonomy
        self._intuition = intuition
        self._reasoning = reasoning
        self._reflection = reflection


__all__ = [
    "InputEvent",
    "OutputEvent",
    "Cognition",
    "InputData",
    "OutputData",
]
