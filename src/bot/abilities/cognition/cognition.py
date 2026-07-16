from .. import ability
from .. import processing

import dataclasses
import datetime
import typing
import uuid

import hsm
import bot

from bot.protocols import attachment
import pydantic

from bot.telemetry import observer

from .input import InputData, is_input
from . import autonomy
from . import input
from . import intuition
from . import reasoning
from . import reflection
from .types import OUTPUT_SCHEMA_CONTRACT, EventData, OutputData, is_output

# Private event-chain key: carries this turn's cognition input with child requests.
# Not a metric/span attribute; not stored on the cognition instance (HSM-COMPLETION-001).
_COGNITION_INPUT_METADATA_KEY = "bot.cognition.input"
_CHILD_CAPABILITY_METADATA_KEY = "bot.cognition.child"
_CHILD_CANCEL_CAPABILITY_METADATA_KEY = "bot.cognition.child.cancel"
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


class _ChildCapability(pydantic.BaseModel):
    """Immutable correlation authority for one directly dispatched child request."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    owner_id: str
    child_id: str
    generation: int


class _ChildCancelCapability(pydantic.BaseModel):
    """Exact active-child cancellation authority carried to its acknowledgement."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    token: str
    owner_id: str
    child_id: str
    request_id: str


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
        _CHILD_CAPABILITY_METADATA_KEY,
        _CHILD_CANCEL_CAPABILITY_METADATA_KEY,
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


def _matches_child_terminal(
    event: hsm.Event[typing.Any],
    *,
    owner: hsm.Instance,
    child: ability.Ability[typing.Any, typing.Any],
    suffix: str,
    terminal_name: str,
    generation: int,
) -> bool:
    capability = event.metadata.get(_CHILD_CAPABILITY_METADATA_KEY)
    return (
        isinstance(capability, _ChildCapability)
        and event.name == terminal_name
        and event.source == hsm.id(child)
        and event.target == hsm.id(owner)
        and capability.owner_id == hsm.id(owner)
        and capability.child_id == hsm.id(child)
        and capability.generation == generation
        and event.id == f"{capability.operation_id}{suffix}"
        and bool(owner.state())
        and owner.state().endswith(f"/{suffix.removeprefix(':')}")
        and _cognition_input_from_event(event) is not None
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
    _child_generation: int

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
        return input.build_processing_input(
            cognition_input,
            extra_actors={"reasoning": instance._reasoning},
        )

    @staticmethod
    def _child_metadata(
        event: hsm.Event[typing.Any],
        cognition_input: InputData,
        owner: "Cognition",
        child: ability.Ability[typing.Any, typing.Any],
    ) -> dict[str, object]:
        owner._child_generation += 1
        metadata = dict(event.metadata)
        metadata[_COGNITION_INPUT_METADATA_KEY] = cognition_input
        operation_id = _parent_operation_id(event) or event.id or uuid.uuid4().hex
        metadata[_CHILD_CAPABILITY_METADATA_KEY] = _ChildCapability(
            operation_id=operation_id,
            owner_id=hsm.id(owner),
            child_id=hsm.id(child),
            generation=owner._child_generation,
        )
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
        child_metadata = Cognition._child_metadata(event, data, instance, autonomy_ability)
        input_event = dataclasses.replace(
            autonomy_ability.input_event.with_data_and_id(
                data,
                Cognition._autonomy_operation_id(event),
            ),
            metadata=child_metadata,
        )
        await hsm.dispatch(ctx, autonomy_ability, input_event)

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
        child_metadata = Cognition._child_metadata(event, data, instance, instance._intuition)
        input_event = dataclasses.replace(
            instance._intuition.input_event.with_data_and_id(
                input,
                Cognition._intuition_operation_id(event),
            ),
            metadata=child_metadata,
        )
        await hsm.dispatch(ctx, instance._intuition, input_event)

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
            terminal_name=child.output_event.name,
            generation=instance._child_generation,
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
            terminal_name=child.failed_event.name,
            generation=instance._child_generation,
        )

    @staticmethod
    def _autonomy_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_autonomy_output(ctx, instance, event) and event.data is not None
        )

    @staticmethod
    def _autonomy_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Cognition._matches_autonomy_output(ctx, instance, event) and event.data is None

    @staticmethod
    def _complete_processing(
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
        child_metadata = Cognition._child_metadata(
            event,
            cognition_input,
            instance,
            instance._intuition,
        )
        base = operation_id if operation_id else uuid.uuid4().hex
        input_event = dataclasses.replace(
            instance._intuition.input_event.with_data_and_id(
                input,
                f"{base}{_INTUITION_ID_SUFFIX}",
            ),
            metadata=child_metadata,
        )
        await hsm.dispatch(ctx, instance._intuition, input_event)

    @staticmethod
    async def _start_intuition_activity(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if Cognition._autonomy_is_unhandled(ctx, instance, event):
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
            terminal_name=child.output_event.name,
            generation=instance._child_generation,
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
            terminal_name=child.failed_event.name,
            generation=instance._child_generation,
        )

    @staticmethod
    def _intuition_is_handled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Cognition._matches_intuition_output(ctx, instance, event)
            and event.data is not None
        )

    @staticmethod
    def _intuition_is_unhandled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        # Host cascade only: unhandled means Cognition starts reasoning. Handled multi-select
        # (including reasoning.input as a normal actor) is intuition's job via direct dispatch.
        return Cognition._matches_intuition_output(ctx, instance, event) and event.data is None

    @staticmethod
    async def _start_reasoning(
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
        from . import reasoning as reasoning_ability

        child_metadata = Cognition._child_metadata(
            event,
            cognition_input,
            instance,
            instance._reasoning,
        )
        # CallData is model/runtime invoke; host frame rides metadata for Reasoning.
        child_metadata[reasoning_ability.HOST_INPUT_METADATA_KEY] = input
        input_event = dataclasses.replace(
            instance._reasoning.input_event.with_data_and_id(
                reasoning_ability.CallData(),
                Cognition._reasoning_operation_id(operation_id),
            ),
            metadata=child_metadata,
        )
        await hsm.dispatch(ctx, instance._reasoning, input_event)

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
            terminal_name=child.output_event.name,
            generation=instance._child_generation,
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
            terminal_name=child.failed_event.name,
            generation=instance._child_generation,
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
    def _request_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not instance._attachments:
            return
        if event.name == ability.Ability._composite_attachment_terminal_event.name:
            reason: bot.RebootReason = "cognition_detach_rollback_failed"
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
    def _dispatch_cancel_to_child(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
        child: ability.Ability[typing.Any, typing.Any],
        suffix: str,
    ) -> None:
        data = event.data
        assert isinstance(data, CancelData)
        child_operation_id = f"{data.operation_id}{suffix}"
        capability = _ChildCancelCapability(
            operation_id=data.operation_id,
            token=data.token,
            owner_id=hsm.id(instance),
            child_id=hsm.id(child),
            request_id=child_operation_id,
        )
        _ = hsm.dispatch(
            ctx,
            child,
            dataclasses.replace(
                processing.CancelEvent.with_data(
                    processing.CancelData(operation_id=child_operation_id, token=data.token)
                ),
                id=child_operation_id,
                source=hsm.id(instance),
                target=hsm.id(child),
                metadata={**event.metadata, _CHILD_CANCEL_CAPABILITY_METADATA_KEY: capability},
            ),
        )

    @staticmethod
    def _cancel_autonomy(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        child = instance._autonomy
        assert child is not None
        Cognition._dispatch_cancel_to_child(ctx, instance, event, child, _AUTONOMY_ID_SUFFIX)

    @staticmethod
    def _cancel_intuition(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        Cognition._dispatch_cancel_to_child(ctx, instance, event, instance._intuition, _INTUITION_ID_SUFFIX)

    @staticmethod
    def _cancel_reasoning(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        Cognition._dispatch_cancel_to_child(ctx, instance, event, instance._reasoning, _REASONING_ID_SUFFIX)

    @staticmethod
    def _matches_child_cancelled(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        capability = event.metadata.get(_CHILD_CANCEL_CAPABILITY_METADATA_KEY)
        if (
            not isinstance(data, processing.CancelledData)
            or not isinstance(capability, _ChildCancelCapability)
            or event.id != data.operation_id
            or event.target != hsm.id(instance)
        ):
            return False
        return (
            capability.owner_id == hsm.id(instance)
            and capability.child_id == event.source
            and capability.request_id == event.id
            and capability.request_id == data.operation_id
            and capability.operation_id == _parent_operation_id(event)
            and capability.token == data.token
        )

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Cognition", event: hsm.Event[typing.Any]) -> None:
        if isinstance(event.data, CancelData):
            operation_id = event.data.operation_id
            token = event.data.token
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

    @staticmethod
    def _child_timeout_delay(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _CHILD_OPERATION_TIMEOUT

    @staticmethod
    def _cancel_timeout_delay(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _CANCEL_TEARDOWN_TIMEOUT

    @staticmethod
    def _request_cancel_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                bot.RebootEvent.with_data(bot.RebootEventData(reason="cognition_cancel_teardown_failed")),
                id=event.id or uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _is_reflection_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, bot.RebootEventData)
            and event.source == hsm.id(instance._reflection)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _forward_reflection_reboot(
        ctx: hsm.Context,
        instance: "Cognition",
        event: hsm.Event[typing.Any],
    ) -> None:
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                event,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=_public_metadata(dict(event.metadata)),
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
                hsm.on(bot.RebootEvent),
                hsm.guard(_is_reflection_reboot),
                hsm.effect(_forward_reflection_reboot),
                hsm.target("/Cognition/rebooting"),
            ),
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
                hsm.on(bot.RebootEvent),
                hsm.guard(_is_reflection_reboot),
                hsm.effect(_forward_reflection_reboot),
                hsm.target("/Cognition/rebooting"),
            ),
            hsm.transition(
                hsm.on(autonomy.OutputEvent, intuition.OutputEvent, reasoning.OutputEvent),
                hsm.guard(_processing_is_handled),
                hsm.effect(_complete_processing),
                hsm.target("/Cognition/idle"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_matches_processing_failure),
                hsm.effect(_fail_child),
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
                    hsm.on(CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_cancel_autonomy),
                    hsm.target("/Cognition/processing/cancelling"),
                ),
                hsm.transition(
                    hsm.on(autonomy.OutputEvent),
                    hsm.guard(_autonomy_is_unhandled),
                    hsm.target("/Cognition/processing/intuition"),
                ),
                hsm.transition(
                    hsm.after(_child_timeout_delay),
                    hsm.effect(_request_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
            ),
            hsm.state(
                "intuition",
                hsm.activity(_start_intuition_activity),
                hsm.transition(
                    hsm.on(CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_cancel_intuition),
                    hsm.target("/Cognition/processing/cancelling"),
                ),
                hsm.transition(
                    hsm.on(intuition.OutputEvent),
                    hsm.guard(_intuition_is_unhandled),
                    hsm.target("/Cognition/processing/reasoning"),
                ),
                hsm.transition(
                    hsm.after(_child_timeout_delay),
                    hsm.effect(_request_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
            ),
            hsm.state(
                "reasoning",
                hsm.activity(_start_reasoning),
                hsm.transition(
                    hsm.on(CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_cancel_reasoning),
                    hsm.target("/Cognition/processing/cancelling"),
                ),
                hsm.transition(
                    hsm.after(_child_timeout_delay),
                    hsm.effect(_request_reboot),
                    hsm.target("/Cognition/rebooting"),
                ),
            ),
            hsm.state(
                "cancelling",
                hsm.transition(
                    hsm.on(processing.CancelledEvent),
                    hsm.guard(_matches_child_cancelled),
                    hsm.effect(_emit_cancelled),
                    hsm.target("/Cognition/idle"),
                ),
                hsm.transition(
                    hsm.after(_cancel_timeout_delay),
                    hsm.effect(_request_cancel_reboot),
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
        self._child_generation = 0
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
