"""Own correlated child operations and cognition action dispatch."""

from __future__ import annotations

from .. import processing
from .. import ability

import dataclasses
import datetime
import collections.abc
import asyncio
import typing
import uuid

import hsm
import pydantic

from .input import InputData

_ACTION_SOURCE_METADATA_KEY = "bot.cognition.action_source"
OPERATION_METADATA_KEY = "bot.cognition.child_operation"
CANCEL_METADATA_KEY = "bot.cognition.child_operation.cancel"
RESOLVE_CANCEL_METADATA_KEY = "bot.cognition.child_operation.cancel.resolve"
_CANCEL_RESOLUTION_TIMEOUT = datetime.timedelta(milliseconds=10)
CANCEL_TOKEN_METADATA_KEY = "bot.cognition.child_operation.cancel.token"


class OperationData(pydantic.BaseModel):
    """Immutable authority for one child request owned by a cognitive host."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    token: str = pydantic.Field(min_length=1)
    owner_id: str = pydantic.Field(min_length=1)
    child_id: str = pydantic.Field(min_length=1)
    request_id: str = pydantic.Field(min_length=1)
    phase: str = pydantic.Field(min_length=1)
    actor_id: str = pydantic.Field(min_length=1)


class TerminalData(pydantic.BaseModel):
    """Portable normalized child outcome emitted only by its operation actor."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation: OperationData
    outcome: typing.Literal["output", "failure", "cancelled", "timed_out", "cancel_timeout"]
    terminal_name: str = pydantic.Field(min_length=1)
    output: pydantic.JsonValue | None = None
    failure: ability.FailureData | None = None


TerminalEvent = hsm.Event[TerminalData](
    name="bot.ability.cognition.child_operation.terminal",
    schema=TerminalData,
)

_StartEvent = hsm.Event[OperationData](
    name="bot.ability.cognition.child_operation.start",
    schema=OperationData,
)


class CancelData(pydantic.BaseModel):
    """Exact host request for an operation mediator to cancel its child request."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    owner_id: str = pydantic.Field(min_length=1)
    child_id: str = pydantic.Field(min_length=1)
    request_id: str = pydantic.Field(min_length=1)
    operation_id: str = pydantic.Field(min_length=1)
    token: str = pydantic.Field(min_length=1)
    resolver_id: str = pydantic.Field(min_length=1)


CancelEvent = hsm.Event[CancelData](
    name="bot.ability.cognition.child_operation.cancel",
    schema=CancelData,
)


class ResolveCancelData(pydantic.BaseModel):
    """Host request to resolve the exact active child mediator."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    owner_id: str = pydantic.Field(min_length=1)
    operation_id: str = pydantic.Field(min_length=1)
    token: str = pydantic.Field(min_length=1)
    phase: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Exact child phase when a composed owner can have more than one mediator for one operation.",
    )
    resolver_id: str = pydantic.Field(min_length=1)


ResolveCancelEvent = hsm.Event[ResolveCancelData](
    name="bot.ability.cognition.child_operation.cancel.resolve",
    schema=ResolveCancelData,
)


class CancelResolvedData(pydantic.BaseModel):
    """Exact mediator capability resolved for a host cancellation request."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation: OperationData
    request: ResolveCancelData


CancelResolvedEvent = hsm.Event[CancelResolvedData](
    name="bot.ability.cognition.child_operation.cancel.resolved",
    schema=CancelResolvedData,
)

CancelUnresolvedEvent = hsm.Event[ResolveCancelData](
    name="bot.ability.cognition.child_operation.cancel.unresolved",
    schema=ResolveCancelData,
)

_ResolutionStartEvent = hsm.Event[ResolveCancelData](
    name="bot.ability.cognition.child_operation.cancel.resolution.start",
    schema=ResolveCancelData,
)
_ResolutionCompletedEvent = hsm.Event[ResolveCancelData](
    name="bot.ability.cognition.child_operation.cancel.resolution.completed",
    schema=ResolveCancelData,
)
_ResolutionAcceptedEvent = hsm.Event[CancelResolvedData](
    name="bot.ability.cognition.child_operation.cancel.resolution.accepted",
    schema=CancelResolvedData,
)
CancelTeardownTimedOutEvent = hsm.Event[CancelResolvedData](
    name="bot.ability.cognition.child_operation.cancel.teardown_timed_out",
    schema=CancelResolvedData,
)
ForceCancelTimeoutEvent = hsm.Event[CancelResolvedData](
    name="bot.ability.cognition.child_operation.cancel.force_timeout",
    schema=CancelResolvedData,
)


class CancelResolution(hsm.Instance):
    """Bounded authority proving whether an exact child mediator resolved."""

    @staticmethod
    def _model(
        owner: hsm.Instance,
        request_ref: list[ResolveCancelData],
        resolved_ref: list[CancelResolvedData],
        metadata: collections.abc.Mapping[str, object],
        teardown_timeout: datetime.timedelta,
    ) -> hsm.Model:
        def request() -> ResolveCancelData:
            return request_ref[0]

        def matches_request(
            ctx: hsm.Context,
            instance: CancelResolution,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                event.data == request()
                and event.id == request().operation_id
                and event.source == request().owner_id
                and event.target == hsm.id(instance)
            )

        def matches_accepted(
            ctx: hsm.Context,
            instance: CancelResolution,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            data = event.data
            return (
                isinstance(data, CancelResolvedData)
                and data.request == request()
                and data.operation.owner_id == request().owner_id
                and data.operation.operation_id == request().operation_id
                and data.operation.token == request().token
                and (request().phase is None or data.operation.phase == request().phase)
                and event.id == request().operation_id
                and event.source == request().owner_id
                and event.target == hsm.id(instance)
            )

        def retain_resolved(
            ctx: hsm.Context,
            instance: CancelResolution,
            event: hsm.Event[typing.Any],
        ) -> None:
            del ctx, instance
            data = event.data
            assert isinstance(data, CancelResolvedData)
            resolved_ref.append(data)

        def delay(
            ctx: hsm.Context,
            instance: CancelResolution,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return _CANCEL_RESOLUTION_TIMEOUT

        def teardown_delay(
            ctx: hsm.Context,
            instance: CancelResolution,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return teardown_timeout

        def unresolved(
            ctx: hsm.Context,
            instance: CancelResolution,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            data = request()
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    CancelUnresolvedEvent.with_data(data),
                    id=data.operation_id,
                    source=hsm.id(instance),
                    target=data.owner_id,
                    metadata={**metadata, RESOLVE_CANCEL_METADATA_KEY: data},
                ),
            )

        def teardown_timed_out(
            ctx: hsm.Context,
            instance: CancelResolution,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            data = resolved_ref[0]
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    CancelTeardownTimedOutEvent.with_data(data),
                    id=data.operation.operation_id,
                    source=hsm.id(instance),
                    target=data.operation.owner_id,
                    metadata={
                        **metadata,
                        OPERATION_METADATA_KEY: data.operation,
                        RESOLVE_CANCEL_METADATA_KEY: data.request,
                    },
                ),
            )

        return hsm.define(
            "CognitiveCancelResolution",
            hsm.initial(hsm.target("idle")),
            hsm.state(
                "idle",
                hsm.transition(
                    hsm.on(_ResolutionStartEvent),
                    hsm.guard(matches_request),
                    hsm.target("/CognitiveCancelResolution/waiting"),
                ),
            ),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(_ResolutionCompletedEvent),
                    hsm.guard(matches_request),
                    hsm.target("/CognitiveCancelResolution/done"),
                ),
                hsm.transition(
                    hsm.on(_ResolutionAcceptedEvent),
                    hsm.guard(matches_accepted),
                    hsm.effect(retain_resolved),
                    hsm.target("/CognitiveCancelResolution/monitoring_teardown"),
                ),
                hsm.transition(
                    hsm.after(delay),
                    hsm.effect(unresolved),
                    hsm.target("/CognitiveCancelResolution/done"),
                ),
            ),
            hsm.state(
                "monitoring_teardown",
                hsm.transition(
                    hsm.on(_ResolutionCompletedEvent),
                    hsm.guard(matches_request),
                    hsm.target("/CognitiveCancelResolution/done"),
                ),
                hsm.transition(
                    hsm.after(teardown_delay),
                    hsm.effect(teardown_timed_out),
                    hsm.target("/CognitiveCancelResolution/done"),
                ),
            ),
            hsm.final("done"),
        )

    @classmethod
    async def begin(
        cls,
        *,
        owner: hsm.Instance,
        operation_id: str,
        token: str,
        metadata: collections.abc.Mapping[str, object],
        teardown_timeout: datetime.timedelta,
        phase: str | None = None,
    ) -> ResolveCancelData:
        actor = cls()
        request_ref: list[ResolveCancelData] = []
        resolved_ref: list[CancelResolvedData] = []
        private = hsm.Context(parent=owner.context(), values={hsm.Keys.Instances: {}})
        started = await hsm.started(
            private,
            actor,
            cls._model(owner, request_ref, resolved_ref, metadata, teardown_timeout),
        )
        request = ResolveCancelData(
            owner_id=hsm.id(owner),
            operation_id=operation_id,
            token=token,
            phase=phase,
            resolver_id=hsm.id(started),
        )
        request_ref.append(request)
        instances = owner.context().value(hsm.Keys.Instances)
        try:
            if isinstance(instances, collections.abc.MutableMapping):
                instances[request.resolver_id] = started
            _ = await hsm.dispatch(
                owner.context(),
                started,
                dataclasses.replace(
                    _ResolutionStartEvent.with_data(request),
                    id=operation_id,
                    source=hsm.id(owner),
                    target=request.resolver_id,
                ),
            )
            resolve_active_operation(owner.context(), owner, request=request, metadata=metadata)
        except asyncio.CancelledError:
            if isinstance(instances, collections.abc.MutableMapping):
                _ = instances.pop(request.resolver_id, None)
            await started.stop(started.context())
            raise
        return request


def _json_output(value: object) -> pydantic.JsonValue:
    if isinstance(value, processing.OutputData):
        value = value.events if value.handled else None
    if isinstance(value, processing.Result):
        result = typing.cast(processing.Result[object], value)
        value = result.output if result.is_handled else None
    dumped = pydantic.TypeAdapter(object).dump_python(value, mode="json")
    return pydantic.TypeAdapter(pydantic.JsonValue).validate_python(dumped)


class Operation(hsm.Instance):
    """One-shot HSM authority that validates and normalizes one child terminal."""

    @staticmethod
    def _model(
        *,
        owner: hsm.Instance,
        child: ability.Ability[typing.Any, typing.Any],
        request: hsm.Event[typing.Any],
        operation_ref: list[OperationData],
        timeout: datetime.timedelta,
    ) -> hsm.Model:
        def operation() -> OperationData:
            return operation_ref[0]

        def timeout_delay(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return timeout

        def matches_start(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            data = event.data
            return (
                isinstance(data, OperationData)
                and data == operation()
                and event.source == data.owner_id
                and event.target == hsm.id(instance)
            )

        def dispatch_request(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            current = operation()
            metadata = {
                **request.metadata,
                OPERATION_METADATA_KEY: current,
            }
            _ = hsm.dispatch(
                ctx,
                child,
                dataclasses.replace(
                    request,
                    source=current.owner_id,
                    target=current.child_id,
                    metadata=metadata,
                ),
            )

        def matches_terminal(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            current = operation()
            return (
                event.name in {child.output_event.name, child.failed_event.name}
                and event.id == current.request_id
                and event.source == current.child_id
                and event.target == hsm.id(instance)
                and event.metadata.get(OPERATION_METADATA_KEY) == current
            )

        def matches_cancelled(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            current = operation()
            data = event.data
            return (
                isinstance(data, processing.CancelledData)
                and data.operation_id == current.request_id
                and data.token == current.token
                and event.id == current.request_id
                and event.source == current.child_id
                and event.target == hsm.id(instance)
                and event.metadata.get(OPERATION_METADATA_KEY) == current
            )

        def matches_host_cancel(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            current = operation()
            data = event.data
            request = event.metadata.get(RESOLVE_CANCEL_METADATA_KEY)
            return (
                isinstance(data, CancelData)
                and isinstance(request, ResolveCancelData)
                and data.owner_id == current.owner_id
                and data.child_id == current.child_id
                and data.request_id == current.request_id
                and data.operation_id == current.operation_id
                and data.token == current.token
                and data.resolver_id == request.resolver_id
                and request.owner_id == current.owner_id
                and request.operation_id == current.operation_id
                and request.token == current.token
                and (request.phase is None or request.phase == current.phase)
                and event.id == current.operation_id
                and event.source == current.owner_id
                and event.target == hsm.id(instance)
                and event.metadata.get(CANCEL_METADATA_KEY) == data
            )

        def matches_force_cancel_timeout(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            current = operation()
            data = event.data
            if not isinstance(data, CancelResolvedData):
                return False
            request_data = data.request
            return (
                data.operation == current
                and request_data.owner_id == current.owner_id
                and request_data.operation_id == current.operation_id
                and request_data.token == current.token
                and (request_data.phase is None or request_data.phase == current.phase)
                and event.id == current.operation_id
                and event.source == current.owner_id
                and event.target == hsm.id(instance)
                and event.metadata.get(OPERATION_METADATA_KEY) == current
                and event.metadata.get(RESOLVE_CANCEL_METADATA_KEY) == request_data
            )

        def matches_cancel_resolution(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            current = operation()
            data = event.data
            return (
                isinstance(data, ResolveCancelData)
                and data.owner_id == current.owner_id
                and data.operation_id == current.operation_id
                and data.token == current.token
                and (data.phase is None or data.phase == current.phase)
                and event.id == data.operation_id
                and event.source == current.owner_id
                and event.target == hsm.id(instance)
                and event.metadata.get(RESOLVE_CANCEL_METADATA_KEY) == data
            )

        def publish_cancel_resolution(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            data = event.data
            assert isinstance(data, ResolveCancelData)
            current = operation()
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    CancelResolvedEvent.with_data(CancelResolvedData(operation=current, request=data)),
                    id=current.operation_id,
                    source=hsm.id(instance),
                    target=current.owner_id,
                    metadata={
                        **event.metadata,
                        OPERATION_METADATA_KEY: current,
                    },
                ),
            )

        def publish(
            ctx: hsm.Context,
            instance: Operation,
            data: TerminalData,
            metadata: dict[str, object],
        ) -> None:
            current = operation()
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    TerminalEvent.with_data(data),
                    id=current.operation_id,
                    source=hsm.id(instance),
                    target=current.owner_id,
                    metadata=metadata,
                ),
            )

        def emit_terminal(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            current = operation()
            if event.name == child.failed_event.name:
                failure = (
                    event.data
                    if isinstance(event.data, ability.FailureData)
                    else ability.FailureData(message=f"Cognitive {current.phase} child failed.")
                )
                data = TerminalData(
                    operation=current,
                    outcome="failure",
                    terminal_name=event.name,
                    failure=failure,
                )
            else:
                try:
                    output = _json_output(event.data)
                except (TypeError, ValueError) as error:
                    data = TerminalData(
                        operation=current,
                        outcome="failure",
                        terminal_name=child.failed_event.name,
                        failure=ability.FailureData(message=f"Child output is not JSON-safe: {error}"),
                    )
                else:
                    data = TerminalData(
                        operation=current,
                        outcome="output",
                        terminal_name=event.name,
                        output=output,
                    )
            publish(ctx, instance, data, dict(event.metadata))

        def request_cancel(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            current = operation()
            _ = hsm.dispatch(
                ctx,
                child,
                dataclasses.replace(
                    processing.CancelEvent.with_data(
                        processing.CancelData(
                            operation_id=current.request_id,
                            token=current.token,
                        )
                    ),
                    id=current.request_id,
                    source=current.owner_id,
                    target=current.child_id,
                    metadata={
                        **request.metadata,
                        **event.metadata,
                        OPERATION_METADATA_KEY: current,
                    },
                ),
            )

        def emit_cancelled(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            current = operation()
            publish(
                ctx,
                instance,
                TerminalData(
                    operation=current,
                    outcome="cancelled",
                    terminal_name=processing.CancelledEvent.name,
                ),
                dict(event.metadata),
            )

        def emit_timed_out(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            current = operation()
            publish(
                ctx,
                instance,
                TerminalData(
                    operation=current,
                    outcome="timed_out",
                    terminal_name=child.failed_event.name,
                    failure=ability.FailureData(
                        message=f"Cognitive {current.phase} operation timed out; cancellation completed."
                    ),
                ),
                dict(event.metadata),
            )

        def emit_cancel_timeout(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            current = operation()
            publish(
                ctx,
                instance,
                TerminalData(
                    operation=current,
                    outcome="cancel_timeout",
                    terminal_name=child.failed_event.name,
                    failure=ability.FailureData(
                        message=f"Cognitive {current.phase} cancellation timed out; child teardown is unproven."
                    ),
                ),
                {**request.metadata, OPERATION_METADATA_KEY: current},
            )

        def emit_forced_cancel_timeout(
            ctx: hsm.Context,
            instance: Operation,
            event: hsm.Event[typing.Any],
        ) -> None:
            current = operation()
            data = event.data
            assert isinstance(data, CancelResolvedData)
            cancel = CancelData(
                owner_id=current.owner_id,
                child_id=current.child_id,
                request_id=current.request_id,
                operation_id=current.operation_id,
                token=current.token,
                resolver_id=data.request.resolver_id,
            )
            publish(
                ctx,
                instance,
                TerminalData(
                    operation=current,
                    outcome="cancel_timeout",
                    terminal_name=child.failed_event.name,
                    failure=ability.FailureData(
                        message=f"Cognitive {current.phase} cancellation timed out; child teardown is unproven."
                    ),
                ),
                {
                    **event.metadata,
                    CANCEL_METADATA_KEY: cancel,
                    RESOLVE_CANCEL_METADATA_KEY: data.request,
                    OPERATION_METADATA_KEY: current,
                },
            )

        return hsm.define(
            "CognitiveChildOperation",
            hsm.initial(hsm.target("idle")),
            hsm.state(
                "idle",
                hsm.transition(
                    hsm.on(_StartEvent),
                    hsm.guard(matches_start),
                    hsm.effect(dispatch_request),
                    hsm.target("/CognitiveChildOperation/waiting"),
                ),
            ),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(ResolveCancelEvent),
                    hsm.guard(matches_cancel_resolution),
                    hsm.effect(publish_cancel_resolution),
                ),
                hsm.transition(
                    hsm.on(CancelEvent),
                    hsm.guard(matches_host_cancel),
                    hsm.effect(request_cancel),
                    hsm.target("/CognitiveChildOperation/host_cancelling"),
                ),
                hsm.transition(
                    hsm.on(child.output_event),
                    hsm.guard(matches_terminal),
                    hsm.effect(emit_terminal),
                    hsm.target("/CognitiveChildOperation/done"),
                ),
                hsm.transition(
                    hsm.on(child.failed_event),
                    hsm.guard(matches_terminal),
                    hsm.effect(emit_terminal),
                    hsm.target("/CognitiveChildOperation/done"),
                ),
                hsm.transition(
                    hsm.after(timeout_delay),
                    hsm.effect(request_cancel),
                    hsm.target("/CognitiveChildOperation/cancelling"),
                ),
            ),
            hsm.state(
                "cancelling",
                hsm.transition(
                    hsm.on(processing.CancelledEvent),
                    hsm.guard(matches_cancelled),
                    hsm.effect(emit_timed_out),
                    hsm.target("/CognitiveChildOperation/done"),
                ),
                hsm.transition(
                    hsm.after(timeout_delay),
                    hsm.effect(emit_cancel_timeout),
                    hsm.target("/CognitiveChildOperation/done"),
                ),
            ),
            hsm.state(
                "host_cancelling",
                hsm.transition(
                    hsm.on(ForceCancelTimeoutEvent),
                    hsm.guard(matches_force_cancel_timeout),
                    hsm.effect(emit_forced_cancel_timeout),
                    hsm.target("/CognitiveChildOperation/done"),
                ),
                hsm.transition(
                    hsm.on(processing.CancelledEvent),
                    hsm.guard(matches_cancelled),
                    hsm.effect(emit_cancelled),
                    hsm.target("/CognitiveChildOperation/done"),
                ),
            ),
            hsm.final("done"),
        )

    @classmethod
    async def begin(
        cls,
        *,
        owner: hsm.Instance,
        child: ability.Ability[typing.Any, typing.Any],
        request: hsm.Event[typing.Any],
        operation_id: str,
        phase: str,
        timeout: datetime.timedelta,
    ) -> OperationData:
        actor = cls()
        operation_ref: list[OperationData] = []
        private = hsm.Context(
            parent=owner.context(),
            values={hsm.Keys.Instances: {}},
        )
        started = await hsm.started(
            private,
            actor,
            cls._model(
                owner=owner,
                child=child,
                request=request,
                operation_ref=operation_ref,
                timeout=timeout,
            ),
        )
        parent_operation = request.metadata.get(OPERATION_METADATA_KEY)
        parent_token = request.metadata.get(CANCEL_TOKEN_METADATA_KEY)
        operation = OperationData(
            operation_id=operation_id,
            token=(
                parent_token
                if isinstance(parent_token, str) and parent_token
                else parent_operation.token
                if isinstance(parent_operation, OperationData)
                else uuid.uuid4().hex
            ),
            owner_id=hsm.id(owner),
            child_id=hsm.id(child),
            request_id=request.id,
            phase=phase,
            actor_id=hsm.id(started),
        )
        operation_ref.append(operation)
        instances = owner.context().value(hsm.Keys.Instances)
        try:
            if isinstance(instances, collections.abc.MutableMapping):
                instances[operation.actor_id] = started
            _ = await hsm.dispatch(
                owner.context(),
                started,
                dataclasses.replace(
                    _StartEvent.with_data(operation),
                    id=operation.request_id,
                    source=operation.owner_id,
                    target=operation.actor_id,
                ),
            )
        except asyncio.CancelledError:
            if isinstance(instances, collections.abc.MutableMapping):
                _ = instances.pop(operation.actor_id, None)
            await started.stop(started.context())
            raise
        return operation


def matches_active_operation(owner: hsm.Instance, event: hsm.Event[typing.Any]) -> bool:
    """Return whether a normalized terminal names an actor still owned by this host."""

    data = event.data
    if not isinstance(data, TerminalData):
        return False
    operation = data.operation
    instances = owner.context().value(hsm.Keys.Instances)
    actor = instances.get(operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
    return (
        isinstance(actor, Operation)
        and hsm.id(actor) == operation.actor_id
        and event.name == TerminalEvent.name
        and event.id == operation.operation_id
        and event.source == operation.actor_id
        and event.target == hsm.id(owner)
        and operation.owner_id == hsm.id(owner)
        and event.metadata.get(OPERATION_METADATA_KEY) == operation
    )


def retire_operation(
    ctx: hsm.Context,
    owner: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    """Retire an exact operation only after its owning host consumed the terminal."""

    del ctx
    if not matches_active_operation(owner, event):
        return
    data = event.data
    assert isinstance(data, TerminalData)
    instances = owner.context().value(hsm.Keys.Instances)
    if isinstance(instances, collections.abc.MutableMapping):
        _ = instances.pop(data.operation.actor_id, None)


def resolve_active_operation(
    ctx: hsm.Context,
    owner: hsm.Instance,
    *,
    request: ResolveCancelData,
    metadata: collections.abc.Mapping[str, object],
) -> None:
    """Ask registered mediators to disclose the exact capability for one active phase."""

    instances = owner.context().value(hsm.Keys.Instances)
    actors = tuple(instances.values()) if isinstance(instances, collections.abc.Mapping) else ()
    for actor in actors:
        if not isinstance(actor, Operation):
            continue
        _ = hsm.dispatch(
            ctx,
            actor,
            dataclasses.replace(
                ResolveCancelEvent.with_data(request),
                id=request.operation_id,
                source=hsm.id(owner),
                target=hsm.id(actor),
                metadata={**metadata, RESOLVE_CANCEL_METADATA_KEY: request},
            ),
        )


def matches_active_resolution(owner: hsm.Instance, event: hsm.Event[typing.Any]) -> bool:
    """Return whether an exact resolution actor still owns this outcome."""

    if isinstance(event.data, CancelResolvedData):
        request = event.data.request
    elif isinstance(event.data, ResolveCancelData):
        request = event.data
    elif isinstance(event.data, TerminalData):
        request = event.metadata.get(RESOLVE_CANCEL_METADATA_KEY)
    else:
        return False
    if not isinstance(request, ResolveCancelData):
        return False
    instances = owner.context().value(hsm.Keys.Instances)
    actor = instances.get(request.resolver_id) if isinstance(instances, collections.abc.Mapping) else None
    return (
        isinstance(actor, CancelResolution)
        and hsm.id(actor) == request.resolver_id
        and request.owner_id == hsm.id(owner)
        and event.id == request.operation_id
        and event.target == hsm.id(owner)
    )


def retire_resolution(
    ctx: hsm.Context,
    owner: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    """Complete and retire the exact bounded resolution actor."""

    if not matches_active_resolution(owner, event):
        return
    if isinstance(event.data, CancelResolvedData):
        request = event.data.request
    elif isinstance(event.data, ResolveCancelData):
        request = event.data
    else:
        request = event.metadata.get(RESOLVE_CANCEL_METADATA_KEY)
    assert isinstance(request, ResolveCancelData)
    instances = owner.context().value(hsm.Keys.Instances)
    actor = instances.get(request.resolver_id) if isinstance(instances, collections.abc.Mapping) else None
    if isinstance(actor, CancelResolution):
        _ = hsm.dispatch(
            ctx,
            actor,
            dataclasses.replace(
                _ResolutionCompletedEvent.with_data(request),
                id=request.operation_id,
                source=hsm.id(owner),
                target=request.resolver_id,
            ),
        )
    if isinstance(instances, collections.abc.MutableMapping):
        _ = instances.pop(request.resolver_id, None)


def complete_resolution(
    ctx: hsm.Context,
    owner: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    """Move a successful resolver from discovery into bounded teardown monitoring."""

    if not isinstance(event.data, CancelResolvedData) or not matches_active_resolution(owner, event):
        return
    request = event.data.request
    instances = owner.context().value(hsm.Keys.Instances)
    actor = instances.get(request.resolver_id) if isinstance(instances, collections.abc.Mapping) else None
    if isinstance(actor, CancelResolution):
        _ = hsm.dispatch(
            ctx,
            actor,
            dataclasses.replace(
                _ResolutionAcceptedEvent.with_data(event.data),
                id=request.operation_id,
                source=hsm.id(owner),
                target=request.resolver_id,
            ),
        )


def matches_teardown_timeout(
    ctx: hsm.Context,
    owner: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> bool:
    """Validate a teardown timeout from the retained exact resolver actor."""

    del ctx
    data = event.data
    if not isinstance(data, CancelResolvedData) or not matches_active_resolution(owner, event):
        return False
    operation = data.operation
    request = data.request
    instances = owner.context().value(hsm.Keys.Instances)
    operation_actor = instances.get(operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
    return (
        isinstance(operation_actor, Operation)
        and hsm.id(operation_actor) == operation.actor_id
        and event.name == CancelTeardownTimedOutEvent.name
        and event.source == request.resolver_id
        and event.target == operation.owner_id
        and request.owner_id == operation.owner_id
        and request.operation_id == operation.operation_id
        and request.token == operation.token
        and (request.phase is None or request.phase == operation.phase)
        and event.metadata.get(OPERATION_METADATA_KEY) == operation
        and event.metadata.get(RESOLVE_CANCEL_METADATA_KEY) == request
    )


def force_cancel_timeout(
    ctx: hsm.Context,
    owner: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    """Forward an exact resolver-owned teardown timeout to its operation mediator."""

    if not matches_teardown_timeout(ctx, owner, event):
        return
    data = event.data
    assert isinstance(data, CancelResolvedData)
    instances = owner.context().value(hsm.Keys.Instances)
    actor = instances.get(data.operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
    if not isinstance(actor, Operation):
        return
    _ = hsm.dispatch(
        ctx,
        actor,
        dataclasses.replace(
            ForceCancelTimeoutEvent.with_data(data),
            id=data.operation.operation_id,
            source=hsm.id(owner),
            target=data.operation.actor_id,
            metadata=dict(event.metadata),
        ),
    )


def cancel_resolved_operation(
    ctx: hsm.Context,
    owner: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    """Cancel the exact mediator capability returned by its resolution event."""

    data = event.data
    if not isinstance(data, CancelResolvedData):
        return
    operation = data.operation
    instances = owner.context().value(hsm.Keys.Instances)
    actor = instances.get(operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
    if not isinstance(actor, Operation) or hsm.id(actor) != operation.actor_id:
        return
    cancel = CancelData(
        owner_id=operation.owner_id,
        child_id=operation.child_id,
        request_id=operation.request_id,
        operation_id=operation.operation_id,
        token=operation.token,
        resolver_id=data.request.resolver_id,
    )
    _ = hsm.dispatch(
        ctx,
        actor,
        dataclasses.replace(
            CancelEvent.with_data(cancel),
            id=operation.operation_id,
            source=hsm.id(owner),
            target=operation.actor_id,
            metadata={
                **event.metadata,
                CANCEL_METADATA_KEY: cancel,
                OPERATION_METADATA_KEY: operation,
            },
        ),
    )


def forward_terminal(
    ctx: hsm.Context,
    owner: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    """Forward a raw child terminal to the exact operation actor without progressing the host."""

    operation = event.metadata.get(OPERATION_METADATA_KEY)
    if not isinstance(operation, OperationData):
        return
    instances = owner.context().value(hsm.Keys.Instances)
    actor = instances.get(operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
    if (
        not isinstance(actor, Operation)
        or operation.owner_id != hsm.id(owner)
        or operation.child_id != event.source
        or operation.request_id != event.id
        or operation.actor_id != hsm.id(actor)
        or event.target != hsm.id(owner)
    ):
        return
    _ = hsm.dispatch(ctx, actor, dataclasses.replace(event, target=hsm.id(actor)))


async def dispatch_selected_events(
    ctx: hsm.Context,
    input: processing.InputData,
    selections: processing.Events,
    *,
    operation_id: str,
    source: hsm.Instance,
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> None:
    """Validate body-action constraints, then dispatch through generic Processing."""

    import bot
    from bot import device

    event_metadata = dict(metadata or {})
    event_metadata[_ACTION_SOURCE_METADATA_KEY] = source
    configured_candidates = tuple(
        name for name, actor in input.actors.items() if name != "bot" and isinstance(actor, device.Device)
    )
    candidates = configured_candidates
    restricted = event_metadata.get("bot.focus_candidates")
    if isinstance(restricted, collections.abc.Sequence) and not isinstance(restricted, str | bytes | bytearray):
        allowed = {item for item in restricted if isinstance(item, str) and item}
        candidates = tuple(item for item in candidates if item in allowed)

    for selection in selections:
        if selection.event == bot.FocusDeviceEvent.name:
            if selection.target is not None and selection.target != "bot":
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
            data = bot.FocusDeviceEventData.model_validate(selection.data or {})
            bot_actor = input.actors.get("bot")
            if data.device not in candidates and (configured_candidates or isinstance(bot_actor, bot.Bot)):
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
        elif selection.event == bot.ClearFocusEvent.name and selection.target not in (None, "bot"):
            raise RuntimeError("Processing selected clear_focus for a non-bot target.")

    await processing.dispatch_selected_events(
        ctx,
        input,
        selections,
        operation_id=operation_id,
        source=source,
        metadata=event_metadata,
    )


def events_from_instance(instance: hsm.Instance) -> tuple[processing.Event[typing.Any], ...]:
    """Enabled CallEventKind events on ``instance`` (same discovery Processing uses)."""

    return processing.enabled_call_events(instance)


def _schemas_from_actors(
    actors: collections.abc.Mapping[str, hsm.Instance],
) -> tuple[processing.Event[typing.Any], ...]:
    """Collect selectable CallEventKind schemas from actors (plus bot body focus events)."""

    schemas: list[processing.Event[typing.Any]] = []
    seen: set[str] = set()
    for instance in actors.values():
        for event in processing.enabled_call_events(instance):
            if event.name in seen:
                continue
            seen.add(event.name)
            schemas.append(event)
    if "bot" in actors:
        from bot import ClearFocusEvent, FocusDeviceEvent

        for event in (FocusDeviceEvent, ClearFocusEvent):
            if event.name in seen:
                continue
            seen.add(event.name)
            schemas.append(event)
    return tuple(schemas)


def build_processing_input(
    cognition_input: InputData,
    *,
    extra_actors: collections.abc.Mapping[str, hsm.Instance] | None = None,
) -> processing.InputData:
    """Build a processing input: stimulus + schemas + named actors.

    ``extra_actors`` lets a host put sibling abilities (e.g. reasoning) on the actor map
    so intuition can multi-select and dispatch them like any other ability.
    """

    actors: dict[str, hsm.Instance] = dict(cognition_input.actors)
    if extra_actors:
        actors.update(dict(extra_actors))
    return processing.InputData(
        input=cognition_input.stimulus,
        schemas=_schemas_from_actors(actors),
        actors=actors,
    )


__all__ = [
    "build_processing_input",
    "dispatch_selected_events",
    "events_from_instance",
]
