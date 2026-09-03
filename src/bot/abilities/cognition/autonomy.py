"""Autonomy: practiced automatic behavior invocation before deliberative abilities.

Outside deliberative ``Processing``. On attach, loads learned Starlark behaviors from
memory and retains injected trusted native seeds. Each turn matches both by stimulus
trigger and runs the chosen Ability/HSM through one attach, input, terminal, and detach
lifecycle. Only materialization and input adaptation differ between the two sources.
"""

from __future__ import annotations

from .. import ability
from .. import memory
from .. import processing

import collections.abc
import asyncio
import dataclasses
import datetime
import typing

import hsm
import bot

from bot.protocols import attachment
import pydantic

from bot import behavior
from bot.behavior.instance import Instance
from bot.behavior import runtime
from bot.behavior import seed
from bot.behavior import storage as behavior_storage
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span

from . import episodes
from . import input
from . import types

_AUTONOMY_ID_MARKER = ":autonomy:"
_BEHAVIOR_SILENCE_TIMEOUT = datetime.timedelta(seconds=1)
_BEHAVIOR_ATTACH_TIMEOUT = datetime.timedelta(seconds=runtime.CALLBACK_WARMUP_SECONDS + 1)
_BEHAVIOR_DETACH_TIMEOUT = datetime.timedelta(seconds=1)


class _InitializingCompleteData(pydantic.BaseModel):
    """Typed empty signal that autonomy finished initialization."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Empty initialization-complete signal for autonomy; carries no payload.",
            "examples": [{}],
        },
    )


_InitializingCompleteEvent = hsm.Event[_InitializingCompleteData](
    name="bot.ability.autonomy.initializing.complete",
    kind=hsm.CompletionEventKind,
    schema=_InitializingCompleteData,
)
_BehaviorsLoadedEvent = hsm.Event[memory.OutputData](
    name="bot.ability.autonomy.behaviors.loaded",
    kind=hsm.CompletionEventKind,
    schema=memory.OutputData,
)
_BehaviorsLoadFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.autonomy.behaviors.load.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)

BehaviorCandidate: typing.TypeAlias = Instance | seed.Seed
CandidateOrigin: typing.TypeAlias = typing.Literal["native", "learned"]
_CandidateOutcome: typing.TypeAlias = typing.Literal[
    "handled",
    "unhandled",
    "failed",
    "silent",
    "cancelled",
    "attach_timeout",
    "detach_failed",
    "detach_timeout",
]


class _InitializeData(pydantic.BaseModel):
    """Private immutable initialization capability prepared inside RTC."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    store: memory.Memory | None


class _MatchData(pydantic.BaseModel):
    """Private immutable behavior inventory snapshot for one match operation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    inventory: tuple[BehaviorCandidate, ...]
    reply_to: str | None = None


_InitializeEvent = hsm.Event[_InitializeData](
    name="bot.ability.autonomy.initialize",
    schema=_InitializeData,
)
_MatchEvent = hsm.Event[_MatchData](
    name="bot.ability.autonomy.match",
    schema=_MatchData,
)


class _MatchedEventData(pydantic.BaseModel):
    """Private completion: candidate behavior instances for this stimulus."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    candidates: tuple[BehaviorCandidate, ...]
    reply_to: str | None = None


_MatchedEmptyEvent = hsm.Event[_MatchedEventData](
    name="bot.ability.autonomy.matched.empty",
    kind=hsm.CompletionEventKind,
    schema=_MatchedEventData,
)
_MatchedCandidatesEvent = hsm.Event[_MatchedEventData](
    name="bot.ability.autonomy.matched.candidates",
    kind=hsm.CompletionEventKind,
    schema=_MatchedEventData,
)


class _StartCandidateEventData(pydantic.BaseModel):
    """Private: start (or advance to) candidate behavior at index."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    candidates: tuple[BehaviorCandidate, ...]
    index: int
    reply_to: str | None = None


_StartCandidateEvent = hsm.Event[_StartCandidateEventData](
    name="bot.ability.autonomy.start_candidate",
    kind=hsm.CompletionEventKind,
    schema=_StartCandidateEventData,
)


class _ApplyCompletedEventData(pydantic.BaseModel):
    """Private completion: handled selections or unhandled turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    output: processing.Events | types.OutputData | None
    turn: types.TurnData
    operation_id: str | None = None
    reply_to: str | None = None


_ApplyCompletedEvent = hsm.Event[_ApplyCompletedEventData](
    name="bot.ability.autonomy.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=_ApplyCompletedEventData,
)
_ApplyFailedEvent = hsm.Event[types.FailureData](
    name="bot.ability.autonomy.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=types.FailureData,
)


class _CandidateCapability(pydantic.BaseModel):
    """Immutable authority for exactly one candidate run.

    ``origin`` is carried through typed candidate events so the shared dispatch engine can
    apply the native trust capability only to code-supplied HSMs. Learned Starlark output
    remains under the model validation policy even though both origins use this same
    CandidateRun attach/input/terminal/detach topology.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    generation: str
    index: int
    token: str
    origin: CandidateOrigin
    reply_to: str | None = None


class _CandidateResultData(pydantic.BaseModel):
    """Typed candidate outcome emitted only after correlated child teardown."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    capability: _CandidateCapability
    turn: types.TurnData
    candidates: tuple[BehaviorCandidate, ...]
    outcome: _CandidateOutcome
    output: processing.Events | None = None
    message: str | None = None
    cancel_operation_id: str | None = None
    cancel_parent_operation_id: str | None = None
    cancel_request_id: str | None = None
    cancel_token: str | None = None


class _CandidateCancelData(pydantic.BaseModel):
    """Typed request to terminate the active candidate actor."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str | None
    parent_operation_id: str | None = None
    request_id: str
    token: str
    reason: str


class _CandidateMaterializedData(pydantic.BaseModel):
    """Typed result of candidate materialization before an actor is started."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    capability: _CandidateCapability
    turn: types.TurnData
    candidates: tuple[BehaviorCandidate, ...]
    index: int
    candidate: BehaviorCandidate
    program: ability.Ability[typing.Any, typing.Any]
    input_adapter: collections.abc.Callable[[input.InputData], object]
    metadata: dict[str, object]


class _CandidateStartFailureData(pydantic.BaseModel):
    """Typed candidate startup failure used by learned/native topology routes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    capability: _CandidateCapability
    turn: types.TurnData
    candidates: tuple[BehaviorCandidate, ...]
    index: int
    candidate: BehaviorCandidate
    message: str


class _CandidateUsageData(pydantic.BaseModel):
    """Typed persistence request/result for one learned candidate usage update."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    result: _CandidateResultData
    behavior: Instance
    outcome: typing.Literal["used", "failed"]
    store: memory.Memory | None


_CandidateMaterializeEvent = hsm.Event[_StartCandidateEventData](
    name="bot.ability.autonomy.candidate.start.materialize",
    kind=hsm.CompletionEventKind,
    schema=_StartCandidateEventData,
)
_CandidateExhaustedEvent = hsm.Event[_StartCandidateEventData](
    name="bot.ability.autonomy.candidate.start.exhausted",
    kind=hsm.CompletionEventKind,
    schema=_StartCandidateEventData,
)
_CandidateMaterializedEvent = hsm.Event[_CandidateMaterializedData](
    name="bot.ability.autonomy.candidate.start.materialized",
    kind=hsm.CompletionEventKind,
    schema=_CandidateMaterializedData,
)
_CandidateMaterializationFailedEvent = hsm.Event[_CandidateStartFailureData](
    name="bot.ability.autonomy.candidate.start.materialization_failed",
    kind=hsm.ErrorEventKind,
    schema=_CandidateStartFailureData,
)
_CandidateActorStartFailedEvent = hsm.Event[_CandidateStartFailureData](
    name="bot.ability.autonomy.candidate.start.actor_failed",
    kind=hsm.ErrorEventKind,
    schema=_CandidateStartFailureData,
)

_CandidateHandledEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.handled",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateAdvanceEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.advance",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateFailedEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.failed",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateSilentEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.silent",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateAttachTimeoutEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.attach_timeout",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateCancelledEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.cancelled",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateDetachFailedEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.detach_failed",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateDetachTimeoutEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.detach_timeout",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateCancelEvent = hsm.Event[_CandidateCancelData](
    name="bot.ability.autonomy.candidate.cancel",
    schema=_CandidateCancelData,
)
_CandidateDispatchSucceededEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.dispatch.succeeded",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateDispatchFailedEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.dispatch.failed",
    kind=hsm.ErrorEventKind,
    schema=_CandidateResultData,
)
_BehaviorUsagePersistedEvent = hsm.Event[_CandidateUsageData](
    name="bot.ability.autonomy.behavior_usage.persisted",
    kind=hsm.CompletionEventKind,
    schema=_CandidateUsageData,
)
_BehaviorUsagePersistenceRequestedEvent = hsm.Event[_CandidateUsageData](
    name="bot.ability.autonomy.behavior_usage.persistence_requested",
    schema=_CandidateUsageData,
)
_BehaviorUsagePersistenceFailedEvent = hsm.Event[_CandidateUsageData](
    name="bot.ability.autonomy.behavior_usage.persistence_failed",
    kind=hsm.ErrorEventKind,
    schema=_CandidateUsageData,
)
_CandidateAttachRequestFailedEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.attach.request_failed",
    kind=hsm.ErrorEventKind,
    schema=_CandidateResultData,
)
_CandidateDetachRequestFailedEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.detach.request_failed",
    kind=hsm.ErrorEventKind,
    schema=_CandidateResultData,
)


class _BehaviorOperationData(pydantic.BaseModel):
    """Typed child-owned operation lifecycle message correlated to CandidateRun."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    operation_id: str
    result: _CandidateResultData | None = None
    message: str | None = None


_BehaviorOperationStartedEvent = hsm.Event[_BehaviorOperationData](
    name="bot.ability.autonomy.behavior_operation.started",
    kind=hsm.CompletionEventKind,
    schema=_BehaviorOperationData,
)
_BehaviorOperationStartFailedEvent = hsm.Event[_BehaviorOperationData](
    name="bot.ability.autonomy.behavior_operation.start_failed",
    kind=hsm.ErrorEventKind,
    schema=_BehaviorOperationData,
)
_BehaviorOperationFinishEvent = hsm.Event[_BehaviorOperationData](
    name="bot.ability.autonomy.behavior_operation.finish",
    schema=_BehaviorOperationData,
)
_BehaviorOperationFinishedEvent = hsm.Event[_BehaviorOperationData](
    name="bot.ability.autonomy.behavior_operation.finished",
    kind=hsm.CompletionEventKind,
    schema=_BehaviorOperationData,
)


class _BehaviorOperation(processing.Operation):
    """Behavior-owned operation actor controlled only through typed lifecycle events."""

    @classmethod
    def model_for(
        cls,
        *,
        behavior: ability.Ability[typing.Any, typing.Any],
        owner: "_CandidateRun",
        operation_id: str,
        metadata: dict[str, object],
    ) -> hsm.Model:
        async def start_activity(
            ctx: hsm.Context,
            instance: _BehaviorOperation,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            terminal = _BehaviorOperationStartedEvent.with_data(_BehaviorOperationData(operation_id=operation_id))
            await hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    terminal,
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(owner),
                    metadata=dict(metadata),
                ),
            )

        def is_finish_request(
            ctx: hsm.Context,
            instance: _BehaviorOperation,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            data = event.data
            return (
                isinstance(data, _BehaviorOperationData)
                and data.operation_id == operation_id
                and data.result is not None
                and event.id == operation_id
                and event.source == hsm.id(owner)
                and event.target == hsm.id(instance)
            )

        def finish_operation(
            ctx: hsm.Context,
            instance: _BehaviorOperation,
            event: hsm.Event[typing.Any],
        ) -> None:
            data = event.data
            assert isinstance(data, _BehaviorOperationData)
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    _BehaviorOperationFinishedEvent.with_data(data),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(owner),
                    metadata=dict(metadata),
                ),
            )

        return bot.define(
            "AutonomyBehaviorOperation",
            hsm.initial(hsm.target("active")),
            hsm.state(
                "active",
                hsm.activity(start_activity),
                hsm.transition(
                    hsm.on(_BehaviorOperationFinishEvent),
                    hsm.guard(is_finish_request),
                    hsm.effect(finish_operation),
                    hsm.target("/AutonomyBehaviorOperation/done"),
                ),
            ),
            hsm.final("done"),
        )


class _CandidateRun(hsm.Instance):
    """Operation actor that owns one Behavior through attach, run, and bounded detach."""

    @classmethod
    def model_for(
        cls,
        *,
        owner: "Autonomy",
        behavior: ability.Ability[typing.Any, typing.Any],
        capability: _CandidateCapability,
        turn: types.TurnData,
        candidates: tuple[BehaviorCandidate, ...],
        input_adapter: collections.abc.Callable[[input.InputData], object],
        metadata: dict[str, object],
    ) -> hsm.Model:
        child_id = _child_operation_id(capability.operation_id, capability.index)
        cognition_input = turn.input
        event_id, event_source, event_target = behavior_event_fields(cognition_input)
        behavior_operation_id = event_id or child_id
        operation_manager = _BehaviorOperation()

        def candidate_result(
            outcome: _CandidateOutcome,
            *,
            output: processing.Events | None = None,
            message: str | None = None,
            cancel_operation_id: str | None = None,
            cancel_parent_operation_id: str | None = None,
            cancel_request_id: str | None = None,
            cancel_token: str | None = None,
        ) -> _CandidateResultData:
            return _CandidateResultData(
                capability=capability,
                turn=turn,
                candidates=candidates,
                outcome=outcome,
                output=output,
                message=message,
                cancel_operation_id=cancel_operation_id,
                cancel_parent_operation_id=cancel_parent_operation_id,
                cancel_request_id=cancel_request_id,
                cancel_token=cancel_token,
            )

        def silence_delay(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return _BEHAVIOR_SILENCE_TIMEOUT

        def detach_delay(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return _BEHAVIOR_DETACH_TIMEOUT

        def attach_delay(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return _BEHAVIOR_ATTACH_TIMEOUT

        async def attach_activity(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            request = dataclasses.replace(
                attachment.AttachEvent.with_data(attachment.AttachData(actor=instance, reply_to=instance)),
                id=child_id,
                source=hsm.id(instance),
                metadata=dict(metadata),
            )
            try:
                await behavior.attach(instance.context(), request)
            except Exception as error:
                result = candidate_result("failed", message=f"Autonomy behavior attach failed: {error}")
                await hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _CandidateAttachRequestFailedEvent.with_data(result),
                        id=capability.operation_id,
                        source=hsm.id(instance),
                        target=hsm.id(instance),
                        metadata=dict(metadata),
                    ),
                )

        def is_attach_terminal(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            data = event.data
            return (
                isinstance(data, (attachment.AttachCompleteData, attachment.FailedData))
                and data.actor is instance
                and event.id == child_id
                and event.source == hsm.id(behavior)
                and event.target == hsm.id(instance)
            )

        async def start_behavior_operation(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del ctx, event
            _ = await bot.started(
                behavior.context(),
                operation_manager,
                _BehaviorOperation.model_for(
                    behavior=behavior,
                    owner=instance,
                    operation_id=behavior_operation_id,
                    metadata=metadata,
                ),
            )

        def is_operation_terminal(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            data = event.data
            return (
                isinstance(data, _BehaviorOperationData)
                and data.operation_id == behavior_operation_id
                and event.id == behavior_operation_id
                and event.source == hsm.id(operation_manager)
                and event.target == hsm.id(instance)
            )

        async def dispatch_behavior_input(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del ctx, event
            payload = input_adapter(cognition_input)
            # The typed input starts the child-owned operation. Its output/failure terminal
            # closes that operation; CandidateRun only correlates those messages.
            input_event = behavior.input_event.with_data(payload)
            input_event = dataclasses.replace(
                input_event,
                id=behavior_operation_id,
                source=event_source or hsm.id(instance),
                target=event_target or hsm.id(behavior),
                metadata=dict(metadata),
            )
            await hsm.dispatch(instance.context(), behavior, input_event)

        def is_correlated_behavior_output(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            # Each CandidateRun owns one freshly materialized Behavior and one input. Source
            # plus directed delivery therefore provide operation correlation even when the
            # behavior runtime stamps its own callback-operation id on the terminal.
            return event.source == hsm.id(behavior) and event.target == hsm.id(instance)

        def is_behavior_handled(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx, instance
            return not isinstance(event.data, ability.FailureData) and _coerce_behavior_output(event.data) is not None

        def is_behavior_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                isinstance(event.data, ability.FailureData)
                and event.source == hsm.id(behavior)
                and event.target == hsm.id(instance)
            )

        def is_cancel(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            data = event.data
            return (
                isinstance(data, _CandidateCancelData)
                and event.id == data.request_id
                and event.source == hsm.id(owner)
                and event.target == hsm.id(instance)
                and data.operation_id in {None, capability.operation_id}
            )

        def is_attach_request_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                isinstance(event.data, _CandidateResultData)
                and event.id == capability.operation_id
                and event.source == hsm.id(instance)
                and event.target == hsm.id(instance)
                and event.data.capability == capability
            )

        def is_detach_terminal(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                isinstance(event.data, attachment.DetachedData | attachment.FailedData)
                and event.data.actor is instance
                and event.id == child_id
                and event.source == hsm.id(behavior)
                and event.target == hsm.id(instance)
            )

        def is_pending_result(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                isinstance(event.data, _CandidateResultData)
                and event.id == capability.operation_id
                and event.source == hsm.id(instance)
                and event.target == hsm.id(instance)
                and event.data.capability == capability
            )

        def is_pending_outcome(
            outcome: _CandidateOutcome,
        ) -> collections.abc.Callable[[hsm.Context, _CandidateRun, hsm.Event[typing.Any]], bool]:
            def guard(
                ctx: hsm.Context,
                instance: _CandidateRun,
                event: hsm.Event[typing.Any],
            ) -> bool:
                data = event.data
                return (
                    is_pending_result(ctx, instance, event)
                    and isinstance(data, _CandidateResultData)
                    and data.outcome == outcome
                )

            return guard

        def is_detach_request_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                isinstance(event.data, _CandidateResultData)
                and event.id == capability.operation_id
                and event.source == hsm.id(instance)
                and event.target == hsm.id(instance)
                and event.data.capability == capability
            )

        async def run_detach(
            ctx: hsm.Context,
            instance: _CandidateRun,
            result: _CandidateResultData,
            result_event: hsm.Event[_CandidateResultData],
        ) -> None:
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    result_event.with_data(result),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(metadata),
                ),
            )
            request = dataclasses.replace(
                attachment.DetachEvent.with_data(
                    attachment.DetachData(
                        actor=instance,
                        reply_to=instance,
                        timeout=_BEHAVIOR_DETACH_TIMEOUT,
                    )
                ),
                id=child_id,
                source=hsm.id(instance),
                target=hsm.id(behavior),
                metadata=dict(metadata),
            )
            try:
                await behavior.detach(instance.context(), request)
            except Exception:
                await hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _CandidateDetachRequestFailedEvent.with_data(result),
                        id=capability.operation_id,
                        source=hsm.id(instance),
                        target=hsm.id(instance),
                        metadata=dict(metadata),
                    ),
                )
                return

        async def request_operation_finish(
            ctx: hsm.Context,
            instance: _CandidateRun,
            result: _CandidateResultData,
        ) -> None:
            await hsm.dispatch(
                ctx,
                operation_manager,
                dataclasses.replace(
                    _BehaviorOperationFinishEvent.with_data(
                        _BehaviorOperationData(operation_id=behavior_operation_id, result=result)
                    ),
                    id=behavior_operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(operation_manager),
                    metadata=dict(metadata),
                ),
            )

        async def stop_handled(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            await request_operation_finish(
                ctx,
                instance,
                candidate_result("handled", output=_coerce_behavior_output(event.data)),
            )

        async def stop_failed(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            await request_operation_finish(
                ctx,
                instance,
                candidate_result("failed", message=getattr(event.data, "message", "Autonomy behavior failed.")),
            )

        async def stop_unhandled(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            await request_operation_finish(ctx, instance, candidate_result("unhandled"))

        async def stop_cancelled(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            data = event.data
            assert isinstance(data, _CandidateCancelData)
            await request_operation_finish(
                ctx,
                instance,
                candidate_result(
                    "cancelled",
                    message=data.reason,
                    cancel_operation_id=data.operation_id,
                    cancel_parent_operation_id=data.parent_operation_id,
                    cancel_request_id=data.request_id,
                    cancel_token=data.token,
                ),
            )

        async def stop_silent(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            await request_operation_finish(ctx, instance, candidate_result("silent"))

        async def detach_after_operation(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            data = event.data
            assert isinstance(data, _BehaviorOperationData)
            assert data.result is not None
            result_events: dict[_CandidateOutcome, hsm.Event[_CandidateResultData]] = {
                "handled": _CandidateHandledEvent,
                "unhandled": _CandidateAdvanceEvent,
                "failed": _CandidateFailedEvent,
                "silent": _CandidateSilentEvent,
                "cancelled": _CandidateCancelledEvent,
                "attach_timeout": _CandidateAttachTimeoutEvent,
                "detach_failed": _CandidateDetachFailedEvent,
                "detach_timeout": _CandidateDetachTimeoutEvent,
            }
            await run_detach(ctx, instance, data.result, result_events[data.result.outcome])

        async def detach_handled(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            await run_detach(
                ctx,
                instance,
                candidate_result("handled", output=_coerce_behavior_output(event.data)),
                _CandidateHandledEvent,
            )

        async def detach_unhandled(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            await run_detach(ctx, instance, candidate_result("unhandled"), _CandidateAdvanceEvent)

        async def detach_failed(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            await run_detach(
                ctx,
                instance,
                candidate_result("failed", message=getattr(event.data, "message", "Autonomy behavior failed.")),
                _CandidateFailedEvent,
            )

        async def detach_cancelled(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            data = event.data
            assert isinstance(data, _CandidateCancelData)
            await run_detach(
                ctx,
                instance,
                candidate_result(
                    "cancelled",
                    message=data.reason,
                    cancel_operation_id=data.operation_id,
                    cancel_parent_operation_id=data.parent_operation_id,
                    cancel_request_id=data.request_id,
                    cancel_token=data.token,
                ),
                _CandidateCancelledEvent,
            )

        async def detach_silent(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            await run_detach(ctx, instance, candidate_result("silent"), _CandidateSilentEvent)

        async def detach_attach_timeout(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            await run_detach(
                ctx,
                instance,
                candidate_result(
                    "attach_timeout",
                    message=(
                        "Autonomy behavior attach timed out after "
                        f"{_BEHAVIOR_ATTACH_TIMEOUT.total_seconds():g} seconds."
                    ),
                ),
                _CandidateAttachTimeoutEvent,
            )

        def forward_owner(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
            outcome_event: hsm.Event[_CandidateResultData],
            result: _CandidateResultData | None = None,
        ) -> None:
            value = result if result is not None else event.data
            assert isinstance(value, _CandidateResultData)
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    outcome_event.with_data(value),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(owner),
                    metadata=dict(metadata),
                ),
            )

        def forward_handled(ctx: hsm.Context, instance: _CandidateRun, event: hsm.Event[typing.Any]) -> None:
            forward_owner(ctx, instance, event, _CandidateHandledEvent)

        def forward_unhandled(ctx: hsm.Context, instance: _CandidateRun, event: hsm.Event[typing.Any]) -> None:
            forward_owner(ctx, instance, event, _CandidateAdvanceEvent)

        def forward_failed(ctx: hsm.Context, instance: _CandidateRun, event: hsm.Event[typing.Any]) -> None:
            forward_owner(ctx, instance, event, _CandidateFailedEvent)

        def forward_silent(ctx: hsm.Context, instance: _CandidateRun, event: hsm.Event[typing.Any]) -> None:
            forward_owner(ctx, instance, event, _CandidateSilentEvent)

        def forward_cancelled(ctx: hsm.Context, instance: _CandidateRun, event: hsm.Event[typing.Any]) -> None:
            forward_owner(ctx, instance, event, _CandidateCancelledEvent)

        def forward_attach_timeout(ctx: hsm.Context, instance: _CandidateRun, event: hsm.Event[typing.Any]) -> None:
            forward_owner(ctx, instance, event, _CandidateAttachTimeoutEvent)

        def forward_detach_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = event.data
            assert isinstance(result, _CandidateResultData)
            forward_owner(
                ctx,
                instance,
                event,
                _CandidateDetachFailedEvent,
                result.model_copy(update={"outcome": "detach_failed", "message": "Autonomy behavior detach failed."}),
            )

        def forward_detach_timeout(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = event.data
            assert isinstance(result, _CandidateResultData)
            forward_owner(
                ctx,
                instance,
                event,
                _CandidateDetachTimeoutEvent,
                result.model_copy(
                    update={
                        "outcome": "detach_timeout",
                        "message": (
                            "Autonomy behavior detach timed out after "
                            f"{_BEHAVIOR_DETACH_TIMEOUT.total_seconds():g} seconds."
                        ),
                    }
                ),
            )

        def forward_attach_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = event.data
            assert isinstance(result, _CandidateResultData)
            forward_owner(ctx, instance, event, _CandidateFailedEvent, result)

        def forward_attach_terminal_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = candidate_result(
                "failed",
                message=getattr(event.data, "message", "Autonomy behavior attach failed."),
            )
            forward_owner(ctx, instance, event, _CandidateFailedEvent, result)

        return bot.define(
            "AutonomyCandidateRun",
            hsm.initial(hsm.target("attaching")),
            hsm.state(
                "attaching",
                hsm.activity(attach_activity),
                hsm.defer(_CandidateCancelEvent),
                hsm.transition(
                    hsm.on(attachment.AttachCompleteEvent),
                    hsm.guard(is_attach_terminal),
                    hsm.target("/AutonomyCandidateRun/starting_operation"),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachFailedEvent),
                    hsm.guard(is_attach_terminal),
                    hsm.effect(forward_attach_terminal_failure),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
                hsm.transition(
                    hsm.on(_CandidateAttachRequestFailedEvent),
                    hsm.guard(is_attach_request_failure),
                    hsm.effect(forward_attach_failure),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
                hsm.transition(
                    hsm.after(attach_delay),
                    hsm.target("/AutonomyCandidateRun/detaching/attach_timeout"),
                ),
            ),
            hsm.state(
                "starting_operation",
                hsm.activity(start_behavior_operation),
                hsm.transition(
                    hsm.on(_CandidateCancelEvent),
                    hsm.guard(is_cancel),
                    hsm.target("/AutonomyCandidateRun/stopping/cancelled"),
                ),
                hsm.transition(
                    hsm.on(_BehaviorOperationStartedEvent),
                    hsm.guard(is_operation_terminal),
                    hsm.target("/AutonomyCandidateRun/running"),
                ),
                hsm.transition(
                    hsm.on(_BehaviorOperationStartFailedEvent),
                    hsm.guard(is_operation_terminal),
                    hsm.target("/AutonomyCandidateRun/detaching/failed"),
                ),
            ),
            hsm.state(
                "running",
                hsm.activity(dispatch_behavior_input),
                hsm.transition(
                    hsm.on(behavior.output_event),
                    hsm.guard(is_correlated_behavior_output),
                    hsm.target("/AutonomyCandidateRun/routing_output"),
                ),
                hsm.transition(
                    hsm.on(behavior.failed_event),
                    hsm.guard(is_behavior_failure),
                    hsm.target("/AutonomyCandidateRun/stopping/failed"),
                ),
                hsm.transition(
                    hsm.on(_CandidateCancelEvent),
                    hsm.guard(is_cancel),
                    hsm.target("/AutonomyCandidateRun/stopping/cancelled"),
                ),
                hsm.transition(
                    hsm.after(silence_delay),
                    hsm.target("/AutonomyCandidateRun/stopping/silent"),
                ),
            ),
            hsm.choice(
                "routing_output",
                hsm.transition(
                    hsm.guard(is_behavior_handled),
                    hsm.target("/AutonomyCandidateRun/stopping/handled"),
                ),
                hsm.transition(hsm.target("/AutonomyCandidateRun/stopping/unhandled")),
            ),
            hsm.state(
                "stopping",
                hsm.initial(hsm.target("handled")),
                hsm.transition(
                    hsm.on(_BehaviorOperationFinishedEvent),
                    hsm.guard(is_operation_terminal),
                    hsm.target("/AutonomyCandidateRun/detaching/from_operation"),
                ),
                hsm.state("handled", hsm.activity(stop_handled)),
                hsm.state("unhandled", hsm.activity(stop_unhandled)),
                hsm.state("failed", hsm.activity(stop_failed)),
                hsm.state("cancelled", hsm.activity(stop_cancelled)),
                hsm.state("silent", hsm.activity(stop_silent)),
            ),
            hsm.state(
                "detaching",
                hsm.initial(hsm.target("handled")),
                hsm.defer(
                    _CandidateHandledEvent,
                    _CandidateAdvanceEvent,
                    _CandidateFailedEvent,
                    _CandidateSilentEvent,
                    _CandidateCancelledEvent,
                    _CandidateAttachTimeoutEvent,
                ),
                hsm.transition(
                    hsm.on(attachment.DetachedEvent),
                    hsm.guard(is_detach_terminal),
                    hsm.target("/AutonomyCandidateRun/reporting"),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachFailedEvent),
                    hsm.guard(is_detach_terminal),
                    hsm.target("/AutonomyCandidateRun/reporting_failure"),
                ),
                hsm.transition(
                    hsm.on(_CandidateDetachRequestFailedEvent),
                    hsm.guard(is_detach_request_failure),
                    hsm.target("/AutonomyCandidateRun/reporting_failure"),
                ),
                hsm.transition(
                    hsm.after(detach_delay),
                    hsm.target("/AutonomyCandidateRun/reporting_timeout"),
                ),
                hsm.state("handled", hsm.activity(detach_handled)),
                hsm.state("unhandled", hsm.activity(detach_unhandled)),
                hsm.state("failed", hsm.activity(detach_failed)),
                hsm.state("cancelled", hsm.activity(detach_cancelled)),
                hsm.state("silent", hsm.activity(detach_silent)),
                hsm.state("attach_timeout", hsm.activity(detach_attach_timeout)),
                hsm.state("from_operation", hsm.activity(detach_after_operation)),
            ),
            hsm.state(
                "reporting",
                hsm.transition(
                    hsm.on(_CandidateHandledEvent),
                    hsm.target("/AutonomyCandidateRun/routing_result"),
                ),
                hsm.transition(
                    hsm.on(_CandidateAdvanceEvent),
                    hsm.target("/AutonomyCandidateRun/routing_result"),
                ),
                hsm.transition(
                    hsm.on(_CandidateFailedEvent),
                    hsm.target("/AutonomyCandidateRun/routing_result"),
                ),
                hsm.transition(
                    hsm.on(_CandidateSilentEvent),
                    hsm.target("/AutonomyCandidateRun/routing_result"),
                ),
                hsm.transition(
                    hsm.on(_CandidateCancelledEvent),
                    hsm.target("/AutonomyCandidateRun/routing_result"),
                ),
                hsm.transition(
                    hsm.on(_CandidateAttachTimeoutEvent),
                    hsm.target("/AutonomyCandidateRun/routing_result"),
                ),
            ),
            hsm.choice(
                "routing_result",
                hsm.transition(
                    hsm.guard(is_pending_outcome("handled")),
                    hsm.effect(forward_handled),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
                hsm.transition(
                    hsm.guard(is_pending_outcome("unhandled")),
                    hsm.effect(forward_unhandled),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
                hsm.transition(
                    hsm.guard(is_pending_outcome("failed")),
                    hsm.effect(forward_failed),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
                hsm.transition(
                    hsm.guard(is_pending_outcome("silent")),
                    hsm.effect(forward_silent),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
                hsm.transition(
                    hsm.guard(is_pending_outcome("cancelled")),
                    hsm.effect(forward_cancelled),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
                hsm.transition(
                    hsm.effect(forward_attach_timeout),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
            ),
            hsm.state(
                "reporting_failure",
                hsm.transition(
                    hsm.on(
                        _CandidateHandledEvent,
                        _CandidateAdvanceEvent,
                        _CandidateFailedEvent,
                        _CandidateSilentEvent,
                        _CandidateCancelledEvent,
                        _CandidateAttachTimeoutEvent,
                    ),
                    hsm.guard(is_pending_result),
                    hsm.effect(forward_detach_failure),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
            ),
            hsm.state(
                "reporting_timeout",
                hsm.transition(
                    hsm.on(
                        _CandidateHandledEvent,
                        _CandidateAdvanceEvent,
                        _CandidateFailedEvent,
                        _CandidateSilentEvent,
                        _CandidateCancelledEvent,
                        _CandidateAttachTimeoutEvent,
                    ),
                    hsm.guard(is_pending_result),
                    hsm.effect(forward_detach_timeout),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
            ),
            hsm.final("done"),
        )


def _public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return dict(metadata)


def _child_operation_id(parent_operation_id: str, index: int) -> str:
    return f"{parent_operation_id}{_AUTONOMY_ID_MARKER}{index}"


def _behavior_select_input() -> memory.InputData:
    """Memory ability input: SELECT ACTIVE behaviors + triggers (skip DRAFT/BROKEN)."""

    clauses = behavior_storage.select_active_behaviors_clauses()
    return memory.InputData(statements=memory.compile_statements(*clauses))


def _behaviors_from_memory_output(output: memory.OutputData) -> tuple[Instance, ...]:
    if len(output.results) < 2:
        return ()
    behavior_rows = tuple(row.as_mapping() for row in output.results[0].rows)
    trigger_rows = tuple(row.as_mapping() for row in output.results[1].rows)
    return behavior_storage.instances_from_behavior_results(behavior_rows, trigger_rows)


def _behavior_at_index(
    candidates: tuple[BehaviorCandidate, ...] | None,
    index: int | None,
) -> BehaviorCandidate | None:
    if candidates is None or index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _replace_behavior_in_tuple(
    behaviors: tuple[BehaviorCandidate, ...], updated: Instance
) -> tuple[BehaviorCandidate, ...]:
    return tuple(updated if isinstance(item, Instance) and item.name == updated.name else item for item in behaviors)


def _event_data_payload(event: object) -> object:
    """Build the JSON-like behavior input *data* payload from a turn event or model."""

    if isinstance(event, hsm.Event):
        return _json_like(event.data)
    if isinstance(event, pydantic.BaseModel):
        return _json_like(event)
    return _json_like(event)


def behavior_event_fields(
    cognition_input: input.InputData,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(id, source, target)`` from the turn's ``hsm.Event`` when present.

    Part of building the behavior input **event** (with :func:`behavior_input_payload` for
    ``data``). Starlark sees one ``event`` with name/data/id/source/target/kind —
    not a separate "envelope" object. Metadata is never included.
    """

    event = cognition_input.stimulus
    if not isinstance(event, hsm.Event):
        return None, None, None
    event_id = event.id if event.id else None
    source = event.source if event.source else None
    target = event.target if event.target else None
    return event_id, source, target


def behavior_input_payload(cognition_input: input.InputData) -> object:
    """Turn event **data** plus live focus context (becomes Starlark ``event["data"]``).

    Id/source/target come from :func:`behavior_event_fields` and are set on the same
    behavior input event. Public so Reflection can dry-run the same shape.
    """

    payload_value = _event_data_payload(cognition_input.stimulus)
    payload: dict[str, object] = (
        dict(typing.cast(dict[str, object], payload_value))
        if isinstance(payload_value, dict)
        else {"event": payload_value}
    )
    if cognition_input.focus is not None:
        _ = payload.setdefault("focus", cognition_input.focus)
    if cognition_input.focus_candidates:
        _ = payload.setdefault("focus_candidates", list(cognition_input.focus_candidates))
    return payload


def _json_like(value: object) -> object:
    """Expose the canonical JSON-compatible event/Data tree to Starlark."""

    from bot.event import event_json_value

    return event_json_value(value)


def _matches_trigger(behavior: BehaviorCandidate, stimulus: str | None) -> bool:
    if not behavior.triggers:
        return False
    if stimulus is None:
        return False
    return stimulus in behavior.triggers


def _coerce_behavior_output(data: object) -> processing.Events | None:
    """Coerce behavior terminal payload to cognition selections; None means unhandled by this behavior."""

    if data is None:
        return None
    if isinstance(data, processing.Result):
        result = typing.cast(processing.Result[object], data)
        if not result.is_handled:
            return None
        return _coerce_behavior_output(result.output)
    if isinstance(data, processing.OutputData):
        return data.events if data.handled else None
    if isinstance(data, processing.SelectedEvent):
        return (data,)
    selections = processing.coerce_event_selections(data)
    if selections is None:
        return None
    return selections


def _candidate_dispatch_trust(origin: CandidateOrigin) -> processing.DispatchTrust:
    """Map typed candidate origin to the shared selected-event trust policy."""

    if origin == "native":
        return processing.DispatchTrust.TRUSTED_BEHAVIOR
    return processing.DispatchTrust.MODEL


class Autonomy(ability.Ability[types.TurnData, types.CompletionData]):
    """Practiced automatic behavior use: load on attach, then match → run a candidate.

    Chart states are lifecycle phases only. Turn product rides the event chain
    (match / start-candidate payloads and child request metadata → child terminal)
    (HSM-COMPLETION-001). ``_behaviors`` holds installed behaviors loaded at attach;
    Candidate identity remains on the scoped operation actor and typed capability events.
    ``seeded_behaviors`` are trusted native HSM descriptors injected by the owner; learned
    behaviors are Starlark inventory records loaded from ``memory``. Both produce an
    Ability and use the same candidate lifecycle. Native input adapters receive the typed
    cognition input, while learned input is projected through the canonical JSON-safe event
    tree. Native selected payloads may retain typed producer-owned data; learned selections
    use the model trust policy and cannot forge producer-stamped fields.

    The owner supplies seed descriptors and retains ownership of the injected
    dependencies. Autonomy owns each materialized Ability until its bounded detach completes
    (compiled behavior attach allows the callback warmup budget plus one second; detach is
    bounded to one second). Seed factories and adapters are synchronous and must be safe for
    concurrent candidate runs. Each factory must return a fresh Ability; Seed is an immutable
    descriptor and does not track or enforce returned identity.
    Memory load, factory/type, input, dispatch, attach, and detach failures reach
    ``failed_event`` as typed ``types.FailureData`` unless an unbuildable learned inventory
    candidate is skipped so a later learned candidate can be tried.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.TurnData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.CompletionData
    input_event: typing.ClassVar[hsm.Event[types.TurnData]] = hsm.Event[types.TurnData](
        name="bot.ability.autonomy.input",
        schema=types.TurnData,
    )
    output_event: typing.ClassVar[hsm.Event[types.CompletionData]] = hsm.Event[types.CompletionData](
        name="bot.ability.autonomy.output",
        schema=types.CompletionData,
    )
    failed_event: typing.ClassVar[hsm.Event[types.FailureData]] = hsm.Event[types.FailureData](
        name=ability.FailedEvent.name,
        kind=hsm.ErrorEventKind,
        schema=types.FailureData,
    )
    _memory: memory.Memory | None
    _behaviors: tuple[Instance, ...]
    _seeded_behaviors: tuple[seed.Seed, ...]

    @staticmethod
    def _updated_behavior_usage(
        behavior: Instance,
        *,
        outcome: typing.Literal["used", "failed"],
    ) -> Instance:
        return behavior_storage.mark_used(behavior) if outcome == "used" else behavior_storage.mark_failed(behavior)

    @staticmethod
    async def _persist_usage_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
        *,
        outcome: typing.Literal["used", "failed"],
    ) -> None:
        """Persist one usage update and publish typed success or degradation."""

        data = event.data
        assert isinstance(data, _CandidateUsageData)
        assert data.outcome == outcome
        updated = Autonomy._updated_behavior_usage(data.behavior, outcome=outcome)
        result = data.result
        if outcome == "failed":
            result = result.model_copy(update={"candidates": _replace_behavior_in_tuple(result.candidates, updated)})
        persisted_event = _BehaviorUsagePersistedEvent
        store = data.store
        if store is not None:
            try:
                clauses = behavior_storage.replace_behavior_clauses(updated)
                _ = store.execute(memory.InputData(statements=memory.compile_statements(*clauses)))
            except Exception:
                telemetry.span.record_current_failure("behavior_usage_persistence")
                persisted_event = _BehaviorUsagePersistenceFailedEvent
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                persisted_event.with_data(
                    _CandidateUsageData(
                        result=result,
                        behavior=updated,
                        outcome=outcome,
                        store=store,
                    )
                ),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _prepare_usage_persistence(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
        *,
        outcome: typing.Literal["used", "failed"],
    ) -> None:
        """Snapshot the learned candidate and Memory capability inside RTC."""

        result = event.data
        assert isinstance(result, _CandidateResultData)
        behavior = _behavior_at_index(result.candidates, result.capability.index)
        assert isinstance(behavior, Instance)
        request = _CandidateUsageData(
            result=result,
            behavior=behavior,
            outcome=outcome,
            store=instance._memory,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _BehaviorUsagePersistenceRequestedEvent.with_data(request),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _prepare_used_persistence(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        Autonomy._prepare_usage_persistence(ctx, instance, event, outcome="used")

    @staticmethod
    def _prepare_failed_persistence(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        Autonomy._prepare_usage_persistence(ctx, instance, event, outcome="failed")

    @staticmethod
    def _apply_usage_update(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Apply an event-carried inventory update inside the owner's RTC step."""

        del ctx
        data = event.data
        assert isinstance(data, _CandidateUsageData)
        instance._behaviors = tuple(
            data.behavior if item.name == data.behavior.name else item for item in instance._behaviors
        )

    @staticmethod
    async def _persist_used_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        await Autonomy._persist_usage_activity(ctx, instance, event, outcome="used")

    @staticmethod
    async def _persist_failed_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        await Autonomy._persist_usage_activity(ctx, instance, event, outcome="failed")

    @staticmethod
    async def _initialize_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Load installed behaviors through the injected Memory capability."""

        data = event.data
        assert isinstance(data, _InitializeData)
        store = data.store
        if store is None:
            _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(_InitializingCompleteData()))
            return
        try:
            output = store.execute(_behavior_select_input())
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                _BehaviorsLoadFailedEvent.with_data(ability.FailureData(message=str(error))),
            )
            return
        _ = hsm.dispatch(ctx, instance, _BehaviorsLoadedEvent.with_data(output))

    @staticmethod
    def _prepare_initialization(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Snapshot the injected Memory capability during the owner's RTC step."""

        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _InitializeEvent.with_data(_InitializeData(store=instance._memory)),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _on_load_behaviors_output(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        output = event.data
        assert isinstance(output, memory.OutputData)
        instance._behaviors = _behaviors_from_memory_output(output)
        _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(_InitializingCompleteData()))

    @staticmethod
    def _on_load_behaviors_failure(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, instance
        failure = (
            event.data
            if isinstance(event.data, ability.FailureData)
            else ability.FailureData(message="Autonomy behavior load failed.")
        )
        raise RuntimeError(f"Autonomy behavior load failed: {failure.message}")

    @staticmethod
    def _detach_on_detach(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        # Shared exit fires on every exit; cascade only for typed DetachEvent payloads.
        if not isinstance(event.data, attachment.DetachData):
            return
        Autonomy._cancel_active_candidate(ctx, instance, reason="Autonomy detached.")
        instance._behaviors = ()

    @staticmethod
    def _cancel_active_candidate(
        ctx: hsm.Context,
        instance: "Autonomy",
        *,
        reason: str,
        operation_id: str | None = None,
        parent_operation_id: str | None = None,
        request_id: str = "detach",
        token: str = "detach",
    ) -> None:
        del ctx
        request = dataclasses.replace(
            _CandidateCancelEvent.with_data(
                _CandidateCancelData(
                    operation_id=operation_id,
                    parent_operation_id=parent_operation_id,
                    request_id=request_id,
                    token=token,
                    reason=reason,
                )
            ),
            id=request_id,
            source=hsm.id(instance),
        )
        _ = hsm.dispatch_all(instance.context(), request)

    @staticmethod
    def _has_autonomy_input(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, types.TurnData)

    @staticmethod
    def _has_matched(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _MatchedEventData)

    @staticmethod
    def _has_start_candidate(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _StartCandidateEventData)

    @staticmethod
    def _start_failure_is_learned(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _CandidateStartFailureData) and data.capability.origin == "learned"

    @staticmethod
    def _has_apply_completed(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _ApplyCompletedEventData)

    @staticmethod
    def _has_apply_failure(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, types.FailureData)

    @staticmethod
    async def _match_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.autonomy.match",
            scope="bot.abilities.cognition",
            component="cognition.autonomy",
            stage="behavior_match",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            assert isinstance(data, _MatchData)
            if processing.active_operation(instance, event.id) is None:
                _ = await processing.start_operation(instance, event.id)
            cognition_input = data.turn.input
            stimulus = episodes.stimulus_name(cognition_input.stimulus)
            candidates = tuple(item for item in data.inventory if _matches_trigger(item, stimulus))
            # Zero candidates from a non-empty inventory is a trigger mismatch, not a missing
            # behavior — the two look identical downstream, where the turn is simply unhandled.
            active.set_attribute("bot.stimulus.name", stimulus or "")
            active.set_attribute("bot.behavior.installed.count", len(data.inventory))
            active.set_attribute("bot.behavior.candidate.count", len(candidates))
            matched = _MatchedEventData(
                turn=data.turn,
                candidates=candidates,
                reply_to=data.reply_to,
            )
            matched_event = _MatchedCandidatesEvent if candidates else _MatchedEmptyEvent
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    matched_event.with_data(matched),
                    id=event.id,
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _prepare_match(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        """Snapshot the complete behavior inventory inside the owner's RTC step."""

        turn = event.data
        assert isinstance(turn, types.TurnData)
        inventory: tuple[BehaviorCandidate, ...] = (*instance._seeded_behaviors, *instance._behaviors)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _MatchEvent.with_data(
                    _MatchData(
                        turn=turn,
                        inventory=inventory,
                        reply_to=event.source if event.target == hsm.id(instance) and event.source else None,
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_terminal(
        ctx: hsm.Context,
        instance: "Autonomy",
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
    def _dispatch_failure_terminal(
        ctx: hsm.Context,
        instance: "Autonomy",
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
    def _complete_apply(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ApplyCompletedEventData)
        Autonomy._dispatch_terminal(
            ctx,
            instance,
            operation_id=data.operation_id if data.operation_id is not None else (event.id or None),
            metadata=dict(event.metadata),
            output=types.CompletionData(turn=data.turn, output=data.output),
            target=data.reply_to,
        )

    @staticmethod
    def _fail_apply(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        assert isinstance(failure, types.FailureData)
        Autonomy._dispatch_failure_terminal(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=failure,
            target=event.source if event.source and event.source != hsm.id(instance) else None,
        )

    @staticmethod
    def _complete_empty_match(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        matched = event.data
        assert isinstance(matched, _MatchedEventData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyCompletedEvent.with_data(
                    _ApplyCompletedEventData(
                        output=None,
                        turn=matched.turn,
                        operation_id=event.id or matched.turn.operation_id,
                        reply_to=matched.reply_to,
                    )
                ),
                id=event.id,
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    def _start_matched_candidates(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        matched = event.data
        assert isinstance(matched, _MatchedEventData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _StartCandidateEvent.with_data(
                    _StartCandidateEventData(
                        turn=matched.turn,
                        candidates=matched.candidates,
                        index=0,
                        reply_to=matched.reply_to,
                    )
                ),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _complete_exhausted(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _StartCandidateEventData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyCompletedEvent.with_data(
                    _ApplyCompletedEventData(
                        output=None,
                        turn=data.turn,
                        operation_id=event.id or data.turn.operation_id,
                        reply_to=data.reply_to,
                    )
                ),
                id=event.id,
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    @staticmethod
    async def _check_candidate_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _StartCandidateEventData)
        if data.index >= len(data.candidates):
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CandidateExhaustedEvent.with_data(data),
                    id=event.id,
                    metadata=dict(event.metadata),
                ),
            )
            return

        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CandidateMaterializeEvent.with_data(data),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _materialize_candidate_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _StartCandidateEventData)
        candidate = _behavior_at_index(data.candidates, data.index)
        assert candidate is not None
        input_adapter: collections.abc.Callable[[input.InputData], object]
        origin: CandidateOrigin = "learned" if isinstance(candidate, Instance) else "native"
        capability = _CandidateCapability(
            operation_id=event.id or data.turn.operation_id,
            generation=data.turn.generation,
            index=data.index,
            token=(
                f"{hsm.id(instance)}:candidate:{event.id or data.turn.operation_id}:{data.index}:{data.turn.generation}"
            ),
            origin=origin,
            reply_to=data.reply_to,
        )
        try:
            if isinstance(candidate, Instance):
                program = behavior.build(candidate.source)
                input_adapter = behavior_input_payload
            else:
                program = candidate.factory()
                input_adapter = candidate.input_adapter
            if not isinstance(program, ability.Ability):
                raise TypeError("Autonomy behavior factory must return an Ability.")
        except Exception as error:
            failure = _CandidateStartFailureData(
                capability=capability,
                turn=data.turn,
                candidates=data.candidates,
                index=data.index,
                candidate=candidate,
                message=str(error),
            )
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CandidateMaterializationFailedEvent.with_data(failure),
                    id=event.id,
                    metadata=dict(event.metadata),
                ),
            )
            return

        materialized = _CandidateMaterializedData(
            capability=capability,
            turn=data.turn,
            candidates=data.candidates,
            index=data.index,
            candidate=candidate,
            program=program,
            input_adapter=input_adapter,
            metadata=dict(event.metadata),
        )
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CandidateMaterializedEvent.with_data(materialized),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _start_candidate_actor_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _CandidateMaterializedData)
        coordinator = _CandidateRun()
        try:
            started = await bot.started(
                instance.context(),
                coordinator,
                _CandidateRun.model_for(
                    owner=instance,
                    behavior=data.program,
                    capability=data.capability,
                    turn=data.turn,
                    candidates=data.candidates,
                    input_adapter=data.input_adapter,
                    metadata=data.metadata,
                ),
                hsm.Config(id=data.capability.token),
            )
        except Exception as error:
            failure = _CandidateStartFailureData(
                capability=data.capability,
                turn=data.turn,
                candidates=data.candidates,
                index=data.index,
                candidate=data.candidate,
                message=str(error),
            )
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CandidateActorStartFailedEvent.with_data(failure),
                    id=event.id,
                    metadata=dict(data.metadata),
                ),
            )
            return
        # Keep the child strongly owned by this state's activity until a typed outcome exits
        # the state. The HSM runtime cancels the activity context on that transition.
        assert started is coordinator
        await asyncio.wrap_future(ctx.done())

    @staticmethod
    async def _emit_learned_start_failure(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _CandidateStartFailureData)
        result = _CandidateResultData(
            capability=data.capability,
            turn=data.turn,
            candidates=data.candidates,
            outcome="failed",
            message=data.message,
        )
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CandidateFailedEvent.with_data(result),
                id=event.id,
                source=data.capability.token,
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _emit_native_start_failure(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _CandidateStartFailureData)
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyFailedEvent.with_data(
                    types.FailureData(
                        message=f"Autonomy native seed {data.candidate.name!r} failed: {data.message}",
                        turn=data.turn,
                    )
                ),
                id=event.id,
                source=data.capability.reply_to or hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _matches_candidate_result(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _CandidateResultData)
            and event.id == data.capability.operation_id
            and event.target == hsm.id(instance)
            and event.source == data.capability.token
        )

    @staticmethod
    def _consume_candidate_result(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, instance, event

    @staticmethod
    async def _dispatch_candidate_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        parent_id = result.capability.operation_id
        cognition_input = result.turn.input
        public_metadata = _public_metadata(dict(event.metadata))
        try:
            output = result.output
            if output is None:
                raise TypeError("Autonomy behavior output is unhandled.")
            processing_input = input.build_processing_input(
                cognition_input,
                authority=instance._attachments[0] if instance._attachments else instance,
            )
            # Deliver only when the turn already provides host actors (bot/devices). A
            # cognition-only actor map (e.g. authority injected as "cognition") is not a
            # body/device delivery surface — still complete with the selection product.
            if cognition_input.actors and output:
                selections = processing.coerce_event_selections(output)
                if selections is None:
                    raise TypeError("Autonomy behavior output does not match event selections.")
                await types.dispatch_selected_events(
                    ctx,
                    processing_input,
                    selections,
                    operation_id=parent_id,
                    source=instance,
                    focus_candidates=result.turn.input.focus_candidates,
                    focused_device=result.turn.input.focus,
                    metadata=public_metadata,
                    dispatch_trust=_candidate_dispatch_trust(result.capability.origin),
                )
        except Exception as error:
            failed = result.model_copy(update={"outcome": "failed", "message": str(error)})
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CandidateDispatchFailedEvent.with_data(failed),
                    id=event.id,
                    source=result.capability.token,
                    target=hsm.id(instance),
                    metadata=public_metadata,
                ),
            )
            return
        await hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CandidateDispatchSucceededEvent.with_data(result),
                id=event.id,
                source=result.capability.token,
                target=hsm.id(instance),
                metadata=public_metadata,
            ),
        )

    @staticmethod
    def _advance_candidate(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _StartCandidateEvent.with_data(
                    _StartCandidateEventData(
                        turn=result.turn,
                        candidates=result.candidates,
                        index=result.capability.index + 1,
                        reply_to=result.capability.reply_to,
                    )
                ),
                id=result.capability.operation_id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _candidate_is_learned(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _CandidateResultData) and data.capability.origin == "learned"

    @staticmethod
    def _start_failed_candidate_persistence(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, instance, event

    @staticmethod
    def _complete_used_after_persistence(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _CandidateUsageData)
        result = data.result
        output = result.output
        if output is None:
            raise TypeError("Autonomy behavior output is unhandled.")
        completion_output: types.OutputData = types.OUTPUT_SCHEMA_CONTRACT.validate_python(
            tuple(
                {
                    "event": item.event,
                    "target": item.target,
                    "data": item.data,
                    "reason": item.reason,
                }
                for item in output
            )
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyCompletedEvent.with_data(
                    _ApplyCompletedEventData(
                        output=completion_output,
                        turn=result.turn,
                        operation_id=result.capability.operation_id,
                        reply_to=result.capability.reply_to,
                    )
                ),
                id=result.capability.operation_id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _complete_native_dispatch(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyCompletedEvent.with_data(
                    _ApplyCompletedEventData(
                        output=result.output,
                        turn=result.turn,
                        operation_id=result.capability.operation_id,
                        reply_to=result.capability.reply_to,
                    )
                ),
                id=result.capability.operation_id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _fail_native_dispatch(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyFailedEvent.with_data(
                    types.FailureData(message=result.message or "Autonomy behavior dispatch failed.", turn=result.turn)
                ),
                id=result.capability.operation_id,
                source=result.capability.reply_to or hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _fail_after_dispatch_persistence(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _CandidateUsageData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyFailedEvent.with_data(
                    types.FailureData(message=data.result.message or "Autonomy dispatch failed.", turn=data.result.turn)
                ),
                id=data.result.capability.operation_id,
                source=data.result.capability.reply_to or hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _advance_after_failed_persistence(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _CandidateUsageData)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _StartCandidateEvent.with_data(
                    _StartCandidateEventData(
                        turn=data.result.turn,
                        candidates=data.result.candidates,
                        index=data.result.capability.index + 1,
                        reply_to=data.result.capability.reply_to,
                    )
                ),
                id=data.result.capability.operation_id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _degrade_candidate(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        Autonomy._dispatch_failure_terminal(
            ctx,
            instance,
            operation_id=result.capability.operation_id,
            metadata=dict(event.metadata),
            failure=types.FailureData(
                message=result.message or "Autonomy behavior teardown failed.",
                turn=result.turn,
            ),
            target=result.capability.reply_to,
        )

    @staticmethod
    def _is_cancel_request(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            isinstance(event.data, processing.CancelData)
            and event.id in {event.data.operation_id, f"{event.data.operation_id}:autonomy"}
            and event.target == hsm.id(instance)
            and bool(instance._attachments)
            and event.source == hsm.id(instance._attachments[0])
            and processing.active_operation(instance, event.data.operation_id) is not None
        )

    @staticmethod
    def _request_cancel(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelData)
        Autonomy._cancel_active_candidate(
            ctx,
            instance,
            reason="Autonomy processing cancelled.",
            operation_id=data.operation_id,
            parent_operation_id=data.parent_operation_id,
            request_id=event.id,
            token=data.token,
        )

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelData)
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                processing.CancelledEvent.with_data(
                    processing.CancelledData(
                        operation_id=data.operation_id,
                        token=data.token,
                        parent_operation_id=data.parent_operation_id,
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )
        processing.finish_operation(ctx, instance, data.operation_id)

    @staticmethod
    def _complete_cancel(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _CandidateResultData)
        assert data.cancel_operation_id is not None
        assert data.cancel_request_id is not None
        assert data.cancel_token is not None
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                processing.CancelledEvent.with_data(
                    processing.CancelledData(
                        operation_id=data.cancel_operation_id,
                        token=data.cancel_token,
                        parent_operation_id=data.cancel_parent_operation_id,
                    )
                ),
                id=data.cancel_request_id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=_public_metadata(dict(event.metadata)),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "Autonomy",
        hsm.initial(hsm.target("/Autonomy/initializing")),
        hsm.state(
            "initializing",
            hsm.initial(hsm.target("preparing")),
            hsm.defer(input_event),
            hsm.exit(_detach_on_detach),
            hsm.state(
                "preparing",
                hsm.entry(_prepare_initialization),
                hsm.transition(
                    hsm.on(_InitializeEvent),
                    hsm.target("/Autonomy/initializing/loading"),
                ),
            ),
            hsm.state("loading", hsm.activity(_initialize_activity)),
            hsm.transition(
                hsm.on(_BehaviorsLoadedEvent),
                hsm.effect(_on_load_behaviors_output),
            ),
            hsm.transition(
                hsm.on(_BehaviorsLoadFailedEvent),
                hsm.effect(_on_load_behaviors_failure),
            ),
            hsm.transition(
                hsm.on(_InitializingCompleteEvent),
                hsm.target("/Autonomy/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_autonomy_input),
                hsm.target("/Autonomy/workflow"),
            ),
        ),
        hsm.state(
            "workflow",
            hsm.initial(hsm.target("matching")),
            hsm.defer(input_event, processing.CancelEvent),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(_ApplyCompletedEvent),
                hsm.guard(_has_apply_completed),
                hsm.effect(_complete_apply),
                hsm.target("/Autonomy/idle"),
            ),
            hsm.transition(
                hsm.on(_ApplyFailedEvent),
                hsm.guard(_has_apply_failure),
                hsm.effect(_fail_apply),
                hsm.target("/Autonomy/idle"),
            ),
            hsm.transition(
                hsm.on(_CandidateExhaustedEvent),
                hsm.guard(_has_start_candidate),
                hsm.effect(_complete_exhausted),
            ),
            hsm.transition(
                hsm.on(_StartCandidateEvent),
                hsm.guard(_has_start_candidate),
                hsm.target("/Autonomy/workflow/starting"),
            ),
            hsm.transition(
                hsm.on(_CandidateHandledEvent),
                hsm.guard(_matches_candidate_result),
                hsm.effect(_consume_candidate_result),
                hsm.target("/Autonomy/workflow/dispatching"),
            ),
            hsm.transition(
                hsm.on(_CandidateAdvanceEvent),
                hsm.guard(_matches_candidate_result),
                hsm.effect(_advance_candidate),
            ),
            hsm.transition(
                hsm.on(_CandidateSilentEvent),
                hsm.guard(_matches_candidate_result),
                hsm.effect(_advance_candidate),
            ),
            hsm.transition(
                hsm.on(_CandidateFailedEvent),
                hsm.guard(_matches_candidate_result),
                hsm.target("/Autonomy/workflow/routing_candidate_failed"),
            ),
            hsm.transition(
                hsm.on(_CandidateAttachTimeoutEvent),
                hsm.guard(_matches_candidate_result),
                hsm.effect(_degrade_candidate),
                hsm.target("/Autonomy/workflow/degraded"),
            ),
            hsm.transition(
                hsm.on(_CandidateDetachFailedEvent),
                hsm.guard(_matches_candidate_result),
                hsm.effect(_degrade_candidate),
                hsm.target("/Autonomy/workflow/degraded"),
            ),
            hsm.transition(
                hsm.on(_CandidateDetachTimeoutEvent),
                hsm.guard(_matches_candidate_result),
                hsm.effect(_degrade_candidate),
                hsm.target("/Autonomy/workflow/degraded"),
            ),
            hsm.transition(
                hsm.on(_CandidateCancelledEvent),
                hsm.guard(_matches_candidate_result),
                hsm.effect(_complete_cancel),
                hsm.target("/Autonomy/idle"),
            ),
            hsm.state(
                "matching",
                hsm.initial(hsm.target("preparing")),
                hsm.transition(
                    hsm.on(processing.CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_emit_cancelled),
                    hsm.target("/Autonomy/idle"),
                ),
                hsm.state(
                    "preparing",
                    hsm.entry(_prepare_match),
                    hsm.transition(
                        hsm.on(_MatchEvent),
                        hsm.target("/Autonomy/workflow/matching/classifying"),
                    ),
                ),
                hsm.state("classifying", hsm.activity(_match_activity)),
                hsm.transition(
                    hsm.on(_MatchedEmptyEvent),
                    hsm.guard(_has_matched),
                    hsm.effect(_complete_empty_match),
                    hsm.target("/Autonomy/workflow/running"),
                ),
                hsm.transition(
                    hsm.on(_MatchedCandidatesEvent),
                    hsm.guard(_has_matched),
                    hsm.effect(_start_matched_candidates),
                    hsm.target("/Autonomy/workflow/running"),
                ),
            ),
            hsm.choice(
                "routing_candidate_failed",
                hsm.transition(
                    hsm.guard(_candidate_is_learned),
                    hsm.effect(_start_failed_candidate_persistence),
                    hsm.target("/Autonomy/workflow/persisting_failed"),
                ),
                hsm.transition(
                    hsm.effect(_advance_candidate),
                    hsm.target("/Autonomy/workflow/running"),
                ),
            ),
            hsm.state(
                "running",
                hsm.transition(
                    hsm.on(processing.CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_request_cancel),
                    hsm.target("/Autonomy/workflow/cancelling"),
                ),
            ),
            hsm.state(
                "dispatching",
                hsm.activity(_dispatch_candidate_activity),
                hsm.transition(
                    hsm.on(processing.CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_emit_cancelled),
                    hsm.target("/Autonomy/idle"),
                ),
                hsm.transition(
                    hsm.on(_CandidateDispatchSucceededEvent),
                    hsm.guard(_matches_candidate_result),
                    hsm.target("/Autonomy/workflow/routing_dispatch_succeeded"),
                ),
                hsm.transition(
                    hsm.on(_CandidateDispatchFailedEvent),
                    hsm.guard(_matches_candidate_result),
                    hsm.target("/Autonomy/workflow/routing_dispatch_failed"),
                ),
            ),
            hsm.choice(
                "routing_dispatch_succeeded",
                hsm.transition(
                    hsm.guard(_candidate_is_learned),
                    hsm.target("/Autonomy/workflow/persisting_used"),
                ),
                hsm.transition(
                    hsm.effect(_complete_native_dispatch),
                    hsm.target("/Autonomy/workflow/running"),
                ),
            ),
            hsm.choice(
                "routing_dispatch_failed",
                hsm.transition(
                    hsm.guard(_candidate_is_learned),
                    hsm.target("/Autonomy/workflow/persisting_dispatch_failed"),
                ),
                hsm.transition(
                    hsm.effect(_fail_native_dispatch),
                    hsm.target("/Autonomy/workflow/running"),
                ),
            ),
            hsm.state(
                "starting",
                hsm.initial(hsm.target("checking")),
                hsm.state(
                    "checking",
                    hsm.activity(_check_candidate_activity),
                    hsm.transition(
                        hsm.on(_CandidateMaterializeEvent),
                        hsm.target("/Autonomy/workflow/starting/materializing"),
                    ),
                ),
                hsm.state(
                    "materializing",
                    hsm.activity(_materialize_candidate_activity),
                    hsm.transition(
                        hsm.on(_CandidateMaterializedEvent),
                        hsm.target("/Autonomy/workflow/starting/starting_actor"),
                    ),
                    hsm.transition(
                        hsm.on(_CandidateMaterializationFailedEvent),
                        hsm.target("/Autonomy/workflow/starting/routing_materialization_failed"),
                    ),
                ),
                hsm.choice(
                    "routing_materialization_failed",
                    hsm.transition(
                        hsm.guard(_start_failure_is_learned),
                        hsm.target("/Autonomy/workflow/starting/learned_failed"),
                    ),
                    hsm.transition(hsm.target("/Autonomy/workflow/starting/native_failed")),
                ),
                hsm.state(
                    "starting_actor",
                    hsm.initial(hsm.target("active")),
                    hsm.activity(_start_candidate_actor_activity),
                    hsm.state(
                        "active",
                        hsm.transition(
                            hsm.on(processing.CancelEvent),
                            hsm.guard(_is_cancel_request),
                            hsm.effect(_request_cancel),
                            hsm.target("/Autonomy/workflow/starting/starting_actor/cancelling"),
                        ),
                    ),
                    hsm.state("cancelling"),
                    hsm.transition(
                        hsm.on(_CandidateActorStartFailedEvent),
                        hsm.target("/Autonomy/workflow/starting/routing_actor_start_failed"),
                    ),
                ),
                hsm.choice(
                    "routing_actor_start_failed",
                    hsm.transition(
                        hsm.guard(_start_failure_is_learned),
                        hsm.target("/Autonomy/workflow/starting/learned_failed"),
                    ),
                    hsm.transition(hsm.target("/Autonomy/workflow/starting/native_failed")),
                ),
                hsm.state("learned_failed", hsm.activity(_emit_learned_start_failure)),
                hsm.state("native_failed", hsm.activity(_emit_native_start_failure)),
            ),
            hsm.state(
                "persisting_used",
                hsm.initial(hsm.target("preparing")),
                hsm.state(
                    "preparing",
                    hsm.entry(_prepare_used_persistence),
                    hsm.transition(
                        hsm.on(_BehaviorUsagePersistenceRequestedEvent),
                        hsm.target("/Autonomy/workflow/persisting_used/writing"),
                    ),
                ),
                hsm.state("writing", hsm.activity(_persist_used_activity)),
                hsm.transition(
                    hsm.on(_BehaviorUsagePersistedEvent),
                    hsm.effect(_apply_usage_update, _complete_used_after_persistence),
                ),
                hsm.transition(
                    hsm.on(_BehaviorUsagePersistenceFailedEvent),
                    hsm.effect(_apply_usage_update, _complete_used_after_persistence),
                ),
            ),
            hsm.state(
                "persisting_failed",
                hsm.initial(hsm.target("preparing")),
                hsm.state(
                    "preparing",
                    hsm.entry(_prepare_failed_persistence),
                    hsm.transition(
                        hsm.on(_BehaviorUsagePersistenceRequestedEvent),
                        hsm.target("/Autonomy/workflow/persisting_failed/writing"),
                    ),
                ),
                hsm.state("writing", hsm.activity(_persist_failed_activity)),
                hsm.transition(
                    hsm.on(_BehaviorUsagePersistedEvent),
                    hsm.effect(_apply_usage_update, _advance_after_failed_persistence),
                ),
                hsm.transition(
                    hsm.on(_BehaviorUsagePersistenceFailedEvent),
                    hsm.effect(_apply_usage_update, _advance_after_failed_persistence),
                ),
            ),
            hsm.state(
                "persisting_dispatch_failed",
                hsm.initial(hsm.target("preparing")),
                hsm.state(
                    "preparing",
                    hsm.entry(_prepare_failed_persistence),
                    hsm.transition(
                        hsm.on(_BehaviorUsagePersistenceRequestedEvent),
                        hsm.target("/Autonomy/workflow/persisting_dispatch_failed/writing"),
                    ),
                ),
                hsm.state("writing", hsm.activity(_persist_failed_activity)),
                hsm.transition(
                    hsm.on(_BehaviorUsagePersistedEvent),
                    hsm.effect(_apply_usage_update, _fail_after_dispatch_persistence),
                ),
                hsm.transition(
                    hsm.on(_BehaviorUsagePersistenceFailedEvent),
                    hsm.effect(_apply_usage_update, _fail_after_dispatch_persistence),
                ),
            ),
            hsm.state(
                "cancelling",
                hsm.transition(
                    hsm.on(_CandidateCancelledEvent),
                    hsm.guard(_matches_candidate_result),
                    hsm.effect(_complete_cancel),
                    hsm.target("/Autonomy/idle"),
                ),
                hsm.transition(
                    hsm.on(_CandidateAttachTimeoutEvent),
                    hsm.guard(_matches_candidate_result),
                    hsm.effect(_degrade_candidate),
                    hsm.target("/Autonomy/workflow/degraded"),
                ),
                hsm.transition(
                    hsm.on(_CandidateDetachFailedEvent),
                    hsm.guard(_matches_candidate_result),
                    hsm.effect(_degrade_candidate),
                    hsm.target("/Autonomy/workflow/degraded"),
                ),
                hsm.transition(
                    hsm.on(_CandidateDetachTimeoutEvent),
                    hsm.guard(_matches_candidate_result),
                    hsm.effect(_degrade_candidate),
                    hsm.target("/Autonomy/workflow/degraded"),
                ),
            ),
            hsm.state(
                "degraded",
                hsm.transition(
                    hsm.on(processing.CancelEvent),
                    hsm.guard(_is_cancel_request),
                    hsm.effect(_emit_cancelled),
                ),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        memory: memory.Memory | None = None,
        seeded_behaviors: tuple[seed.Seed, ...] = (),
    ) -> None:
        """Construct Autonomy with optional Memory inventory and native seed descriptors.

        ``memory`` is the owner-injected Memory capability used to load learned Starlark
        inventory on attach; it may be ``None`` for seed-only operation. ``seeded_behaviors``
        is an immutable tuple of trusted ``Seed`` descriptors. Autonomy does not construct
        or start their machines here: each matching turn must receive a fresh Ability from
        the descriptor's factory, which Autonomy owns through attach/input/terminal/detach.
        Seed does not track returned identities; factories and input adapters must be
        synchronous, re-entrant, and safe for concurrent candidate runs. A native factory,
        type, or adapter failure is surfaced
        through ``failed_event``; a learned Starlark build failure is recorded and falls
        through to the next learned inventory candidate. Attach allows the compiled callback
        warmup budget plus one second for lifecycle settlement; detach is bounded to one second.
        """

        super().__init__()
        self._memory = memory
        self._behaviors = ()
        self._seeded_behaviors = tuple(seeded_behaviors)


InputEvent = Autonomy.input_event
OutputEvent = Autonomy.output_event

__all__ = [
    "InputEvent",
    "OutputEvent",
    "Autonomy",
    "behavior_input_payload",
    "behavior_event_fields",
]
