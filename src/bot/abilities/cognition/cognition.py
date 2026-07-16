from .. import ability
from .. import processing

import dataclasses
import datetime
import collections.abc
import typing
import uuid

import hsm
import bot

from bot.protocols import attachment
import pydantic

from bot.telemetry import observer

from . import operations
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

    private_keys = {
        _COGNITION_INPUT_METADATA_KEY,
        operations.OPERATION_METADATA_KEY,
        operations.CANCEL_METADATA_KEY,
        operations.RESOLVE_CANCEL_METADATA_KEY,
        operations.CANCEL_TOKEN_METADATA_KEY,
    }
    return {key: value for key, value in metadata.items() if key not in private_keys}


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
    assert isinstance(data, operations.TerminalData)
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
    if not isinstance(data, operations.TerminalData):
        return False
    operation = data.operation
    expected_outcomes = {"output"} if terminal_name == child.output_event.name else {"failure", "timed_out"}
    return (
        operations.matches_active_operation(owner, event)
        and event.name == operations.TerminalEvent.name
        and event.source == operation.actor_id
        and event.target == hsm.id(owner)
        and operation.owner_id == hsm.id(owner)
        and operation.child_id == hsm.id(child)
        and operation.phase == phase
        and operation.request_id == f"{operation.operation_id}{suffix}"
        and data.terminal_name == terminal_name
        and data.outcome in expected_outcomes
        and event.metadata.get(operations.OPERATION_METADATA_KEY) == operation
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
    if not isinstance(data, operations.TerminalData):
        return False
    operation = data.operation
    return (
        data.outcome == "cancel_timeout"
        and data.failure is not None
        and operations.matches_active_operation(owner, event)
        and event.source == operation.actor_id
        and event.target == hsm.id(owner)
        and operation.owner_id == hsm.id(owner)
        and operation.child_id == hsm.id(child)
        and operation.phase == phase
        and operation.request_id == f"{operation.operation_id}{suffix}"
        and event.metadata.get(operations.OPERATION_METADATA_KEY) == operation
    )


class Cognition(ability.Ability[InputData, OutputData]):
    """Judgment ability: autonomy → intuition → reasoning, then reflection.

    Chart states are lifecycle phases only. The active turn rides the child request
    and terminal event chain, not instance fields (HSM-COMPLETION-001).
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[OutputData]] = OutputEvent
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _autonomy: autonomy.Autonomy | None
    _intuition: intuition.Intuition
    _reasoning: reasoning.Reasoning
    _reflection: reflection.Reflection
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
        return operations.build_processing_input(
            cognition_input,
            extra_actors={"reasoning": instance._reasoning},
        )

    @staticmethod
    def _child_metadata(event: hsm.Event[typing.Any], cognition_input: InputData) -> dict[str, object]:
        metadata = dict(event.metadata)
        metadata[_COGNITION_INPUT_METADATA_KEY] = cognition_input
        bot_operation = _active_bot_operation(event)
        if bot_operation is not None:
            metadata[operations.CANCEL_TOKEN_METADATA_KEY] = bot_operation[1]
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
        await operations.Operation.begin(
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
        await operations.Operation.begin(
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
    def _complete_processing(
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
                        _failure(operation_id, "Cognition output is missing turn correlation.")
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
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=public_metadata,
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
        reflection_ability = instance._reflection
        turn = reflection.InputData(cognition_input=cognition_input, cognition_output=output)
        reflection_input = processing.InputData(input=turn)
        reflection_event = dataclasses.replace(
            reflection_ability.input_event.with_data_and_id(
                reflection_input,
                Cognition._reflection_operation_id(operation_id),
            ),
            source=hsm.id(instance),
            target=hsm.id(reflection_ability),
            metadata=public_metadata,
        )
        _ = hsm.dispatch(ctx, reflection_ability, reflection_event)

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
        await operations.Operation.begin(
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
        if isinstance(event.data, operations.TerminalData):
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
        # (including reasoning.input as a normal actor) is intuition's job via operation.
        return Cognition._matches_intuition_output(ctx, instance, event) and _normalized_child_event(event).data is None

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
        await operations.Operation.begin(
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
    def _processing_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._autonomy_is_handled(ctx, instance, event)
            or Cognition._intuition_is_handled(ctx, instance, event)
            or Cognition._matches_reasoning_output(ctx, instance, event)
        )

    @staticmethod
    def _matches_processing_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_autonomy_failure(ctx, instance, event)
            or Cognition._matches_intuition_failure(ctx, instance, event)
            or Cognition._matches_reasoning_failure(ctx, instance, event)
        )

    @staticmethod
    def _matches_child_teardown_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, operations.TerminalData):
            return False
        operation = data.operation
        candidates: list[tuple[ability.Ability[typing.Any, typing.Any], str, str]] = [
            (instance._intuition, _INTUITION_ID_SUFFIX, "intuition"),
            (instance._reasoning, _REASONING_ID_SUFFIX, "reasoning"),
        ]
        if instance._autonomy is not None:
            candidates.append((instance._autonomy, _AUTONOMY_ID_SUFFIX, "autonomy"))
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
        assert isinstance(data, operations.TerminalData)
        failure = data.failure or ability.FailureData(message="Cognition child cancellation timed out.")
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=data.operation.operation_id,
            metadata=_public_metadata(dict(event.metadata)),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _request_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not instance._attachments:
            return
        if event.name == ability.Ability._composite_attachment_terminal_event.name:
            reason: bot.RebootReason = "cognition_detach_rollback_failed"
        elif isinstance(event.data, operations.TerminalData) and event.data.outcome == "cancel_timeout":
            reason = "cognition_cancel_teardown_failed"
        else:
            reason = "cognition_child_teardown_failed"
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                bot.RebootEvent.with_data(bot.RebootEventData(reason=reason)),
                id=event.id or uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

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
        operations.forward_terminal(ctx, instance, event)

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
    def _has_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _UseFailedEventData)

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
    ) -> None:
        data = event.data
        assert isinstance(data, CancelData)
        await operations.CancelResolution.begin(
            owner=instance,
            operation_id=data.operation_id,
            token=data.token,
            metadata=event.metadata,
            teardown_timeout=_CANCEL_TEARDOWN_TIMEOUT,
        )

    @staticmethod
    def _matches_cancel_resolved(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, operations.CancelResolvedData):
            return False
        operation = data.operation
        instances = instance.context().value(hsm.Keys.Instances)
        actor = instances.get(operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
        return (
            operations.matches_active_resolution(instance, event)
            and isinstance(actor, operations.Operation)
            and hsm.id(actor) == operation.actor_id
            and data.request.owner_id == hsm.id(instance)
            and data.request.operation_id == operation.operation_id
            and data.request.token == operation.token
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
            isinstance(data, operations.ResolveCancelData)
            and operations.matches_active_resolution(instance, event)
            and event.name == operations.CancelUnresolvedEvent.name
            and event.source == data.resolver_id
            and event.target == hsm.id(instance)
            and event.metadata.get(operations.RESOLVE_CANCEL_METADATA_KEY) == data
        )

    @staticmethod
    def _matches_child_cancelled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        cancel = event.metadata.get(operations.CANCEL_METADATA_KEY)
        request = event.metadata.get(operations.RESOLVE_CANCEL_METADATA_KEY)
        if (
            not isinstance(data, operations.TerminalData)
            or not isinstance(cancel, operations.CancelData)
            or not isinstance(request, operations.ResolveCancelData)
        ):
            return False
        operation = data.operation
        return (
            data.outcome == "cancelled"
            and data.terminal_name == processing.CancelledEvent.name
            and operations.matches_active_operation(instance, event)
            and operations.matches_active_resolution(instance, event)
            and cancel.owner_id == hsm.id(instance)
            and cancel.child_id == operation.child_id
            and cancel.request_id == operation.request_id
            and cancel.operation_id == operation.operation_id
            and cancel.token == operation.token
            and cancel.resolver_id == request.resolver_id
            and request.owner_id == hsm.id(instance)
            and request.operation_id == operation.operation_id
            and request.token == operation.token
            and event.id == cancel.operation_id
        )

    @staticmethod
    def _matches_cancel_teardown_failure(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        data = event.data
        cancel = event.metadata.get(operations.CANCEL_METADATA_KEY)
        request = event.metadata.get(operations.RESOLVE_CANCEL_METADATA_KEY)
        if (
            not Cognition._matches_child_teardown_failure(ctx, instance, event)
            or not isinstance(data, operations.TerminalData)
            or not isinstance(cancel, operations.CancelData)
            or not isinstance(request, operations.ResolveCancelData)
        ):
            return False
        operation = data.operation
        return (
            operations.matches_active_resolution(instance, event)
            and cancel.owner_id == hsm.id(instance)
            and cancel.child_id == operation.child_id
            and cancel.request_id == operation.request_id
            and cancel.operation_id == operation.operation_id
            and cancel.token == operation.token
            and cancel.resolver_id == request.resolver_id
            and request.owner_id == hsm.id(instance)
            and request.operation_id == operation.operation_id
            and request.token == operation.token
        )

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        if isinstance(event.data, CancelData):
            operation_id = event.data.operation_id
            token = event.data.token
        elif isinstance(event.data, operations.ResolveCancelData):
            operation_id = event.data.operation_id
            token = event.data.token
        elif isinstance(event.data, operations.TerminalData):
            request = event.metadata.get(operations.RESOLVE_CANCEL_METADATA_KEY)
            assert isinstance(request, operations.ResolveCancelData)
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
                hsm.guard(_has_autonomy),
                hsm.target("/Cognition/processing/autonomy"),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.target("/Cognition/processing/intuition"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
            ),
        ),
        hsm.state(
            "processing",
            hsm.initial(hsm.target("/Cognition/processing/intuition")),
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.target("/Cognition/processing/cancelling"),
            ),
            hsm.transition(
                hsm.on(autonomy.OutputEvent, intuition.OutputEvent, reasoning.OutputEvent),
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
                hsm.on(operations.TerminalEvent),
                hsm.guard(_matches_child_teardown_failure),
                hsm.effect(_fail_child, operations.retire_operation, _request_reboot),
                hsm.target("/Cognition/rebooting"),
            ),
            hsm.transition(
                hsm.on(operations.TerminalEvent),
                hsm.guard(_processing_is_handled),
                hsm.effect(_complete_processing, operations.retire_operation),
                hsm.target("/Cognition/idle"),
            ),
            hsm.transition(
                hsm.on(operations.TerminalEvent),
                hsm.guard(_matches_processing_failure),
                hsm.effect(_fail_child, operations.retire_operation),
                hsm.target("/Cognition/idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_dispatch_failure),
                hsm.target("/Cognition/idle"),
            ),
            hsm.state(
                "autonomy",
                hsm.activity(_start_autonomy),
                hsm.transition(
                    hsm.on(operations.TerminalEvent),
                    hsm.guard(_autonomy_is_unhandled),
                    hsm.effect(operations.retire_operation),
                    hsm.target("/Cognition/processing/intuition"),
                ),
            ),
            hsm.state(
                "intuition",
                hsm.activity(_start_intuition_activity),
                hsm.transition(
                    hsm.on(operations.TerminalEvent),
                    hsm.guard(_intuition_is_unhandled),
                    hsm.effect(operations.retire_operation),
                    hsm.target("/Cognition/processing/reasoning"),
                ),
            ),
            hsm.state(
                "reasoning",
                hsm.activity(_start_reasoning),
            ),
            hsm.state(
                "cancelling",
                hsm.activity(_resolve_child_cancel),
                hsm.transition(
                    hsm.on(operations.CancelResolvedEvent),
                    hsm.guard(_matches_cancel_resolved),
                    hsm.effect(operations.complete_resolution, operations.cancel_resolved_operation),
                ),
                hsm.transition(
                    hsm.on(operations.CancelUnresolvedEvent),
                    hsm.guard(_matches_cancel_unresolved),
                    hsm.effect(operations.retire_resolution, _emit_cancelled),
                    hsm.target("/Cognition/idle"),
                ),
                hsm.transition(
                    hsm.on(operations.CancelTeardownTimedOutEvent),
                    hsm.guard(operations.matches_teardown_timeout),
                    hsm.effect(operations.force_cancel_timeout),
                ),
                hsm.transition(
                    hsm.on(operations.TerminalEvent),
                    hsm.guard(_matches_child_cancelled),
                    hsm.effect(operations.retire_resolution, operations.retire_operation, _emit_cancelled),
                    hsm.target("/Cognition/idle"),
                ),
                hsm.transition(
                    hsm.on(operations.TerminalEvent),
                    hsm.guard(_matches_cancel_teardown_failure),
                    hsm.effect(
                        operations.retire_resolution,
                        operations.retire_operation,
                        _fail_cancel_teardown,
                        _request_reboot,
                    ),
                    hsm.target("/Cognition/rebooting"),
                ),
            ),
        ),
        hsm.state(
            "rebooting",
            hsm.defer(input_event),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal, _request_reboot),
                hsm.target("/Cognition/rebooting"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
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
        reflection: reflection.Reflection,
        autonomy: autonomy.Autonomy | None = None,
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
        children.append(reflection)
        self._attachment_group = attachment.Group(*children)


__all__ = [
    "InputEvent",
    "OutputEvent",
    "Cognition",
    "InputData",
    "OutputData",
]
