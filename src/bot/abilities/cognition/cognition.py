from .. import ability
from .. import processing

import dataclasses
import datetime
import collections.abc
import typing
import uuid

import hsm

from bot.protocols import attachment
import pydantic

from bot.telemetry import observer

from . import dispatch
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
_REFLECTION_ID_SUFFIX = ":reflection"
_CHILD_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_CANCEL_TEARDOWN_TIMEOUT = datetime.timedelta(seconds=5)


class CancelData(pydantic.BaseModel):
    """Request cancellation of one active cognition turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(
        min_length=1,
        description="Bot turn identifier whose active cognition work must be cancelled.",
        examples=["turn-123"],
    )
    token: str = pydantic.Field(
        min_length=1,
        description="Opaque cancellation capability created by the Bot operation actor.",
        examples=["8d72b83f17654f1788c012ab132b4afd"],
    )


class CancelledData(pydantic.BaseModel):
    """Confirmation that one cognition turn no longer owns active child work."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(
        min_length=1,
        description="Bot turn identifier whose cognition work reached the idle boundary.",
        examples=["turn-123"],
    )
    token: str = pydantic.Field(
        min_length=1,
        description="Exact cancellation capability that reached the idle boundary.",
        examples=["8d72b83f17654f1788c012ab132b4afd"],
    )


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
CancelEvent = hsm.Event[CancelData](
    name="bot.ability.cognition.cancel",
    schema=CancelData,
)
CancelledEvent = hsm.Event[CancelledData](
    name="bot.ability.cognition.cancelled",
    kind=hsm.CompletionEventKind,
    schema=CancelledData,
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
        return validated
    if isinstance(output, list | tuple):
        values = typing.cast(list[object] | tuple[object, ...], output)
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python(tuple(values))
        return validated
    if isinstance(output, dict) and "event" in output:
        validated = OUTPUT_SCHEMA_CONTRACT.validate_python((output,))
        return validated
    raise TypeError("Cognition processing produced output that does not match its output schema.")


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


def _active_bot_operation(event: hsm.Event[typing.Any]) -> tuple[str, str] | None:
    """Validate the immutable Bot cancellation capability without inspecting peer state."""

    operation = event.metadata.get("bot.processing.operation")
    actor = event.metadata.get("bot.processing.actor")
    if not isinstance(operation, pydantic.BaseModel) or not isinstance(actor, hsm.Instance):
        return None
    values = operation.model_dump(mode="json")
    request_id = values.get("request_id")
    token = values.get("token")
    actor_id = values.get("actor_id")
    if (
        not isinstance(request_id, str)
        or not isinstance(token, str)
        or not isinstance(actor_id, str)
        or actor_id != token
        or hsm.id(actor) != actor_id
    ):
        return None
    return request_id, token


def _has_bot_operation_metadata(event: hsm.Event[typing.Any]) -> bool:
    """Return whether this event claims the Bot's private operation-capability path."""

    return "bot.processing.operation" in event.metadata or "bot.processing.actor" in event.metadata


def _normalized_child_event(event: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    data = event.data
    assert isinstance(data, dispatch.TerminalData)
    operation = data.operation
    payload: object = data.failure if data.failure is not None else data.output
    return hsm.Event[object](
        name=data.terminal_name,
        data=payload,
        id=operation.request_id,
        source=operation.child_id,
        target=operation.actor_id,
        metadata=dict(event.metadata),
        schema=pydantic.TypeAdapter(object),
    )


def _matches_child_terminal(
    event: hsm.Event[typing.Any],
    *,
    owner: hsm.Instance,
    child: ability.Ability[typing.Any, typing.Any],
    suffix: str,
    phase: str,
    terminal_name: str,
) -> bool:
    data = event.data
    if not isinstance(data, dispatch.TerminalData):
        return False
    operation = data.operation
    expected_outcomes = {"output"} if terminal_name == child.output_event.name else {"failure", "timed_out"}
    return (
        dispatch.matches_active_operation(owner, event)
        and event.name == dispatch.TerminalEvent.name
        and event.source == operation.actor_id
        and event.target == hsm.id(owner)
        and operation.owner_id == hsm.id(owner)
        and operation.child_id == hsm.id(child)
        and operation.phase == phase
        and operation.request_id == f"{operation.operation_id}{suffix}"
        and data.terminal_name == terminal_name
        and data.outcome in expected_outcomes
        and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == operation
    )


def _matches_teardown_failure(
    event: hsm.Event[typing.Any],
    *,
    owner: hsm.Instance,
    child: ability.Ability[typing.Any, typing.Any],
    suffix: str,
    phase: str,
) -> bool:
    data = event.data
    if not isinstance(data, dispatch.TerminalData):
        return False
    operation = data.operation
    return (
        data.outcome == "cancel_timeout"
        and data.failure is not None
        and dispatch.matches_active_operation(owner, event)
        and event.source == operation.actor_id
        and event.target == hsm.id(owner)
        and operation.owner_id == hsm.id(owner)
        and operation.child_id == hsm.id(child)
        and operation.phase == phase
        and operation.request_id == f"{operation.operation_id}{suffix}"
        and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == operation
    )


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
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _autonomy: autonomy.Autonomy | None
    _intuition: intuition.Intuition
    _reasoning: reasoning.Reasoning
    _reflection: reflection.Reflection | None
    _attachment_group: attachment.Group

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
        return f"{base}{_REFLECTION_ID_SUFFIX}"

    @staticmethod
    def _build_processing_input(
        instance: "Cognition",
        cognition_input: InputData,
    ) -> processing.InputData:
        # Reasoning is a normal actor/tool for intuition multi-select — not a hardcoded edge.
        return dispatch.build_processing_input(
            cognition_input,
            extra_actors={"reasoning": instance._reasoning},
        )

    @staticmethod
    def _child_metadata(event: hsm.Event[typing.Any], cognition_input: InputData) -> dict[str, object]:
        metadata = dict(event.metadata)
        metadata[_COGNITION_INPUT_METADATA_KEY] = cognition_input
        bot_operation = _active_bot_operation(event)
        if bot_operation is not None:
            metadata[dispatch.CANCEL_TOKEN_METADATA_KEY] = bot_operation[1]
        return metadata

    @staticmethod
    async def _start_autonomy(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert is_input(data)
        autonomy_ability = instance._autonomy
        assert autonomy_ability is not None
        child_metadata = Cognition._child_metadata(event, data)
        input_event = dataclasses.replace(
            autonomy_ability.input_event.with_data_and_id(
                data,
                Cognition._autonomy_operation_id(event),
            ),
            metadata=child_metadata,
        )
        await dispatch.Operation.begin(
            owner=instance,
            child=autonomy_ability,
            request=input_event,
            operation_id=event.id if event.id else input_event.id.removesuffix(_AUTONOMY_ID_SUFFIX),
            phase="autonomy",
            timeout=_CHILD_OPERATION_TIMEOUT,
        )

    @staticmethod
    async def _start_intuition_from_input(
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
        child_metadata = Cognition._child_metadata(event, data)
        input_event = dataclasses.replace(
            instance._intuition.input_event.with_data_and_id(
                input,
                Cognition._intuition_operation_id(event),
            ),
            metadata=child_metadata,
        )
        await dispatch.Operation.begin(
            owner=instance,
            child=instance._intuition,
            request=input_event,
            operation_id=operation_id if operation_id else input_event.id.removesuffix(_INTUITION_ID_SUFFIX),
            phase="intuition",
            timeout=_CHILD_OPERATION_TIMEOUT,
        )

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
        return _matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_AUTONOMY_ID_SUFFIX,
            phase="autonomy",
            terminal_name=child.output_event.name,
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
        return _matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_AUTONOMY_ID_SUFFIX,
            phase="autonomy",
            terminal_name=child.failed_event.name,
        )

    @staticmethod
    def _autonomy_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_autonomy_output(ctx, instance, event) and _normalized_child_event(event).data is not None
        )

    @staticmethod
    def _autonomy_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Cognition._matches_autonomy_output(ctx, instance, event) and _normalized_child_event(event).data is None

    @staticmethod
    def _complete_from_autonomy(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        event = _normalized_child_event(event)
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
    async def _start_intuition(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Start intuition from an autonomy unhandled terminal (input rides metadata)."""

        event = _normalized_child_event(event)
        cognition_input = _cognition_input_from_event(event)
        operation_id = _parent_operation_id(event)
        public_metadata = _public_metadata(dict(event.metadata))
        if cognition_input is None:
            # Idle path still passes the input event body.
            if is_input(event.data):
                await Cognition._start_intuition_from_input(ctx, instance, event)
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
        await dispatch.Operation.begin(
            owner=instance,
            child=instance._intuition,
            request=input_event,
            operation_id=operation_id if operation_id else input_event.id.removesuffix(_INTUITION_ID_SUFFIX),
            phase="intuition",
            timeout=_CHILD_OPERATION_TIMEOUT,
        )

    @staticmethod
    async def _start_intuition_activity(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if isinstance(event.data, dispatch.TerminalData):
            await Cognition._start_intuition(ctx, instance, event)
            return
        await Cognition._start_intuition_from_input(ctx, instance, event)

    @staticmethod
    def _matches_intuition_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._intuition
        return _matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_INTUITION_ID_SUFFIX,
            phase="intuition",
            terminal_name=child.output_event.name,
        )

    @staticmethod
    def _matches_intuition_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._intuition
        return _matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_INTUITION_ID_SUFFIX,
            phase="intuition",
            terminal_name=child.failed_event.name,
        )

    @staticmethod
    def _intuition_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_intuition_output(ctx, instance, event)
            and _normalized_child_event(event).data is not None
        )

    @staticmethod
    def _intuition_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        # Host cascade only: unhandled means Cognition starts reasoning. Handled multi-select
        # (including reasoning.input as a normal actor) is intuition's job via dispatch.
        return Cognition._matches_intuition_output(ctx, instance, event) and _normalized_child_event(event).data is None

    @staticmethod
    def _complete_from_intuition(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        event = _normalized_child_event(event)
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
    async def _start_reasoning(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        event = _normalized_child_event(event)
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
        from . import reasoning as reasoning_ability

        child_metadata = dict(event.metadata)
        child_metadata[_COGNITION_INPUT_METADATA_KEY] = cognition_input
        # CallData is model/runtime invoke; host frame rides metadata for Reasoning.
        child_metadata[reasoning_ability.HOST_INPUT_METADATA_KEY] = input
        input_event = dataclasses.replace(
            instance._reasoning.input_event.with_data_and_id(
                reasoning_ability.CallData(),
                Cognition._reasoning_operation_id(operation_id),
            ),
            metadata=child_metadata,
        )
        await dispatch.Operation.begin(
            owner=instance,
            child=instance._reasoning,
            request=input_event,
            operation_id=operation_id if operation_id else input_event.id.removesuffix(_REASONING_ID_SUFFIX),
            phase="reasoning",
            timeout=_CHILD_OPERATION_TIMEOUT,
        )

    @staticmethod
    def _matches_reasoning_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reasoning
        return _matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_REASONING_ID_SUFFIX,
            phase="reasoning",
            terminal_name=child.output_event.name,
        )

    @staticmethod
    def _matches_reasoning_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reasoning
        return _matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_REASONING_ID_SUFFIX,
            phase="reasoning",
            terminal_name=child.failed_event.name,
        )

    @staticmethod
    def _matches_child_teardown_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, dispatch.TerminalData):
            return False
        operation = data.operation
        candidates: list[tuple[ability.Ability[typing.Any, typing.Any], str, str]] = [
            (instance._intuition, _INTUITION_ID_SUFFIX, "intuition"),
            (instance._reasoning, _REASONING_ID_SUFFIX, "reasoning"),
        ]
        if instance._autonomy is not None:
            candidates.append((instance._autonomy, _AUTONOMY_ID_SUFFIX, "autonomy"))
        if instance._reflection is not None:
            candidates.append((instance._reflection, _REFLECTION_ID_SUFFIX, "reflection"))
        return any(
            operation.child_id == hsm.id(child)
            and _matches_teardown_failure(
                event,
                owner=instance,
                child=child,
                suffix=suffix,
                phase=phase,
            )
            for child, suffix, phase in candidates
        )

    @staticmethod
    def _complete_from_reasoning(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        event = _normalized_child_event(event)
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
        event = _normalized_child_event(event)
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
    def _fail_cancel_teardown(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, dispatch.TerminalData)
        failure = data.failure or ability.FailureData(message="Cognition child cancellation timed out.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=data.operation.operation_id,
            metadata=_public_metadata(dict(event.metadata)),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

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
    async def _dispatch_reflection(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Start one supervised Reflection operation from the completion event."""

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
        try:
            await dispatch.Operation.begin(
                owner=instance,
                child=reflection_ability,
                request=input_event,
                operation_id=(completion.operation_id or input_event.id.removesuffix(_REFLECTION_ID_SUFFIX)),
                phase="reflection",
                timeout=_CHILD_OPERATION_TIMEOUT,
            )
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        _failure(completion.operation_id, f"Cognition could not start Reflection: {error}")
                    ),
                    id=completion.operation_id,
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _has_reflection(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, event
        return instance._reflection is not None

    @staticmethod
    def _matches_reflection_terminal(
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        terminal_name: str,
    ) -> bool:
        child = instance._reflection
        return child is not None and _matches_child_terminal(
            event,
            owner=instance,
            child=child,
            suffix=_REFLECTION_ID_SUFFIX,
            phase="reflection",
            terminal_name=terminal_name,
        )

    @staticmethod
    def _matches_reflection_output(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reflection
        return child is not None and Cognition._matches_reflection_terminal(instance, event, child.output_event.name)

    @staticmethod
    def _matches_reflection_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        child = instance._reflection
        return child is not None and Cognition._matches_reflection_terminal(instance, event, child.failed_event.name)

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
    def _forward_child_terminal(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        dispatch.forward_terminal(ctx, instance, event)

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
        return True

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

    @staticmethod
    def _is_cancel_request(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not (
            isinstance(data, CancelData)
            and event.id == data.operation_id
            and event.target == hsm.id(instance)
            and bool(instance._attachments)
            and event.source == hsm.id(instance._attachments[0])
        ):
            return False
        operation = _active_bot_operation(event)
        if _has_bot_operation_metadata(event):
            return operation == (data.operation_id, data.token)
        return True

    @staticmethod
    async def _resolve_child_cancel(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        phase: str,
    ) -> None:
        data = event.data
        assert isinstance(data, CancelData)
        await dispatch.CancelResolution.begin(
            owner=instance,
            operation_id=data.operation_id,
            token=data.token,
            phase=phase,
            metadata=event.metadata,
            teardown_timeout=_CANCEL_TEARDOWN_TIMEOUT,
        )

    @staticmethod
    async def _resolve_autonomy_cancel(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        await Cognition._resolve_child_cancel(ctx, instance, event, "autonomy")

    @staticmethod
    async def _resolve_intuition_cancel(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        await Cognition._resolve_child_cancel(ctx, instance, event, "intuition")

    @staticmethod
    async def _resolve_reasoning_cancel(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        await Cognition._resolve_child_cancel(ctx, instance, event, "reasoning")

    @staticmethod
    def _matches_cancel_resolved(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, dispatch.CancelResolvedData):
            return False
        operation = data.operation
        instances = instance.context().value(hsm.Keys.Instances)
        actor = instances.get(operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
        return (
            dispatch.matches_active_resolution(instance, event)
            and isinstance(actor, dispatch.Operation)
            and hsm.id(actor) == operation.actor_id
            and data.request.owner_id == hsm.id(instance)
            and data.request.operation_id == operation.operation_id
            and data.request.token == operation.token
            and data.request.phase == operation.phase
            and event.id == operation.operation_id
            and event.source == operation.actor_id
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_cancel_unresolved(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, dispatch.ResolveCancelData)
            and dispatch.matches_active_resolution(instance, event)
            and event.name == dispatch.CancelUnresolvedEvent.name
            and event.source == data.resolver_id
            and event.target == hsm.id(instance)
            and event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY) == data
        )

    @staticmethod
    def _matches_child_cancelled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        cancel = event.metadata.get(dispatch.CANCEL_METADATA_KEY)
        request = event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY)
        if (
            not isinstance(data, dispatch.TerminalData)
            or not isinstance(cancel, dispatch.CancelData)
            or not isinstance(request, dispatch.ResolveCancelData)
        ):
            return False
        operation = data.operation
        return (
            data.outcome == "cancelled"
            and data.terminal_name == processing.CancelledEvent.name
            and dispatch.matches_active_operation(instance, event)
            and dispatch.matches_active_resolution(instance, event)
            and cancel.owner_id == hsm.id(instance)
            and cancel.child_id == operation.child_id
            and cancel.request_id == operation.request_id
            and cancel.operation_id == operation.operation_id
            and cancel.token == operation.token
            and cancel.resolver_id == request.resolver_id
            and request.owner_id == hsm.id(instance)
            and request.operation_id == operation.operation_id
            and request.token == operation.token
            and request.phase == operation.phase
            and event.id == cancel.operation_id
        )

    @staticmethod
    def _matches_cancel_teardown_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        data = event.data
        cancel = event.metadata.get(dispatch.CANCEL_METADATA_KEY)
        request = event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY)
        if (
            not Cognition._matches_child_teardown_failure(ctx, instance, event)
            or not isinstance(data, dispatch.TerminalData)
            or not isinstance(cancel, dispatch.CancelData)
            or not isinstance(request, dispatch.ResolveCancelData)
        ):
            return False
        operation = data.operation
        return (
            dispatch.matches_active_resolution(instance, event)
            and cancel.owner_id == hsm.id(instance)
            and cancel.child_id == operation.child_id
            and cancel.request_id == operation.request_id
            and cancel.operation_id == operation.operation_id
            and cancel.token == operation.token
            and cancel.resolver_id == request.resolver_id
            and request.owner_id == hsm.id(instance)
            and request.operation_id == operation.operation_id
            and request.token == operation.token
            and request.phase == operation.phase
        )

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        if isinstance(event.data, CancelData):
            operation_id = event.data.operation_id
            token = event.data.token
        elif isinstance(event.data, dispatch.ResolveCancelData):
            operation_id = event.data.operation_id
            token = event.data.token
        elif isinstance(event.data, dispatch.TerminalData):
            request = event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY)
            assert isinstance(request, dispatch.ResolveCancelData)
            operation_id = request.operation_id
            token = request.token
        else:
            data = event.data
            assert isinstance(data, processing.CancelledData)
            operation = _active_bot_operation(event)
            if operation is not None:
                operation_id, token = operation
            else:
                operation_id = _parent_operation_id(event)
                assert operation_id is not None
                token = data.token
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                CancelledEvent.with_data(CancelledData(operation_id=operation_id, token=token)),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Cognition",
        hsm.initial(hsm.target("/Cognition/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
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
            hsm.activity(_start_autonomy),
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.target("/Cognition/resolving_autonomy_cancel"),
            ),
            hsm.transition(
                hsm.on(autonomy.OutputEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_autonomy_is_handled),
                hsm.effect(_complete_from_autonomy, dispatch.retire_operation),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_autonomy_is_unhandled),
                hsm.effect(dispatch.retire_operation),
                hsm.target("/Cognition/intuiting"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_child_teardown_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
                hsm.target("/Cognition/degraded"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_autonomy_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
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
            hsm.activity(_start_intuition_activity),
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.target("/Cognition/resolving_intuition_cancel"),
            ),
            hsm.transition(
                hsm.on(intuition.OutputEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_intuition_is_handled),
                hsm.effect(_complete_from_intuition, dispatch.retire_operation),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_intuition_is_unhandled),
                hsm.effect(dispatch.retire_operation),
                hsm.target("/Cognition/reasoning"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_child_teardown_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
                hsm.target("/Cognition/degraded"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_intuition_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
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
            hsm.activity(_start_reasoning),
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.target("/Cognition/resolving_reasoning_cancel"),
            ),
            hsm.transition(
                hsm.on(reasoning.OutputEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_reasoning_output),
                hsm.effect(_complete_from_reasoning, dispatch.retire_operation),
                hsm.target("/Cognition/completing"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_child_teardown_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
                hsm.target("/Cognition/degraded"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_reasoning_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
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
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
                hsm.target("/Cognition/idle"),
            ),
            hsm.transition(
                hsm.on(_ProcessingCompletedEvent),
                hsm.guard(_has_matching_apply_operation),
                hsm.guard(_has_output),
                hsm.effect(_dispatch_output),
                hsm.target("/Cognition/post_completion"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_matching_apply_operation),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.choice(
            "post_completion",
            hsm.transition(
                hsm.guard(_has_reflection),
                hsm.target("/Cognition/reflecting"),
            ),
            hsm.transition(hsm.target("/Cognition/idle")),
        ),
        hsm.state(
            "reflecting",
            hsm.defer(input_event),
            hsm.activity(_dispatch_reflection),
            hsm.transition(
                hsm.on(reflection.OutputEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_reflection_output),
                hsm.effect(dispatch.retire_operation),
                hsm.target("/Cognition/idle"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_child_teardown_failure),
                hsm.effect(dispatch.retire_operation),
                hsm.target("/Cognition/degraded"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_reflection_failure),
                hsm.effect(dispatch.retire_operation),
                hsm.target("/Cognition/degraded"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/Cognition/degraded"),
            ),
        ),
        hsm.state(
            "resolving_autonomy_cancel",
            hsm.defer(input_event),
            hsm.activity(_resolve_autonomy_cancel),
            hsm.transition(
                hsm.on(dispatch.CancelResolvedEvent),
                hsm.guard(_matches_cancel_resolved),
                hsm.effect(dispatch.complete_resolution, dispatch.cancel_resolved_operation),
                hsm.target("/Cognition/cancelling"),
            ),
            hsm.transition(
                hsm.on(dispatch.CancelUnresolvedEvent),
                hsm.guard(_matches_cancel_unresolved),
                hsm.effect(dispatch.retire_resolution, _emit_cancelled),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "resolving_intuition_cancel",
            hsm.defer(input_event),
            hsm.activity(_resolve_intuition_cancel),
            hsm.transition(
                hsm.on(dispatch.CancelResolvedEvent),
                hsm.guard(_matches_cancel_resolved),
                hsm.effect(dispatch.complete_resolution, dispatch.cancel_resolved_operation),
                hsm.target("/Cognition/cancelling"),
            ),
            hsm.transition(
                hsm.on(dispatch.CancelUnresolvedEvent),
                hsm.guard(_matches_cancel_unresolved),
                hsm.effect(dispatch.retire_resolution, _emit_cancelled),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "resolving_reasoning_cancel",
            hsm.defer(input_event),
            hsm.activity(_resolve_reasoning_cancel),
            hsm.transition(
                hsm.on(dispatch.CancelResolvedEvent),
                hsm.guard(_matches_cancel_resolved),
                hsm.effect(dispatch.complete_resolution, dispatch.cancel_resolved_operation),
                hsm.target("/Cognition/cancelling"),
            ),
            hsm.transition(
                hsm.on(dispatch.CancelUnresolvedEvent),
                hsm.guard(_matches_cancel_unresolved),
                hsm.effect(dispatch.retire_resolution, _emit_cancelled),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state(
            "cancelling",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(dispatch.CancelTeardownTimedOutEvent),
                hsm.guard(dispatch.matches_teardown_timeout),
                hsm.effect(dispatch.force_cancel_timeout),
            ),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.effect(_forward_child_terminal),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_child_cancelled),
                hsm.effect(dispatch.retire_resolution, dispatch.retire_operation, _emit_cancelled),
                hsm.target("/Cognition/idle"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_cancel_teardown_failure),
                hsm.effect(dispatch.retire_resolution, dispatch.retire_operation, _fail_cancel_teardown),
                hsm.target("/Cognition/degraded"),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Cognition/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Cognition/idle"),
            ),
        ),
        hsm.state("degraded"),
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
        children: list[hsm.Instance] = []
        if autonomy is not None:
            children.append(autonomy)
        children.extend((intuition, reasoning))
        if reflection is not None:
            children.append(reflection)
        self._attachment_group = attachment.Group(*children)


__all__ = [
    "InputEvent",
    "OutputEvent",
    "Cognition",
    "InputData",
    "OutputData",
]
