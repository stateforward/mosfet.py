"""Autonomy: practiced automatic habit invocation before deliberative abilities.

Outside deliberative ``Processing``. On attach, loads installed habits from memory.
Each turn matches those habits by stimulus trigger, runs compiled ``Behavior``,
dispatches selected events when a habit handles, or leaves the input unhandled so
Cognition continues to Intuition.
"""

from __future__ import annotations

from .. import ability
from .. import memory
from .. import processing

import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm

from bot.protocols import attachment
import pydantic
from sqlalchemy.sql import Executable

from bot import habit
from bot.habit.instance import Instance
from bot.habit import storage as habit_storage
from bot.telemetry import observer

from . import episodes
from . import input
from . import types

_AUTONOMY_ID_MARKER = ":autonomy:"
_HABIT_SILENCE_TIMEOUT = datetime.timedelta(seconds=1)
_HABIT_ATTACH_TIMEOUT = datetime.timedelta(seconds=1)
_HABIT_DETACH_TIMEOUT = datetime.timedelta(seconds=1)
_InitializingCompleteEvent = hsm.Event[object](
    name="bot.ability.autonomy.initializing.complete",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)
_HabitsLoadedEvent = hsm.Event[memory.OutputData](
    name="bot.ability.autonomy.habits.loaded",
    kind=hsm.CompletionEventKind,
    schema=memory.OutputData,
)
_HabitsLoadFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.autonomy.habits.load.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class _MatchedEventData(pydantic.BaseModel):
    """Private completion: candidate habit instances for this stimulus."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    candidates: tuple[Instance, ...]


_MatchedEvent = hsm.Event[_MatchedEventData](
    name="bot.ability.autonomy.matched",
    kind=hsm.CompletionEventKind,
    schema=_MatchedEventData,
)


class _StartCandidateEventData(pydantic.BaseModel):
    """Private: start (or advance to) candidate habit at index."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: types.TurnData
    candidates: tuple[Instance, ...]
    index: int


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

    output: types.OutputData | None
    turn: types.TurnData
    operation_id: str | None = None


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
    """Immutable JSON-safe authority for exactly one candidate run."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    generation: str
    index: int
    token: str


class _CandidateResultData(pydantic.BaseModel):
    """Typed candidate outcome emitted only after correlated child teardown."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    capability: _CandidateCapability
    turn: types.TurnData
    candidates: tuple[Instance, ...]
    outcome: typing.Literal[
        "handled",
        "unhandled",
        "failed",
        "silent",
        "cancelled",
        "attach_timeout",
        "detach_failed",
        "detach_timeout",
    ]
    output: types.OutputData | None = None
    message: str | None = None
    cancel_operation_id: str | None = None
    cancel_parent_operation_id: str | None = None
    cancel_request_id: str | None = None
    cancel_token: str | None = None


class _CandidateCancelData(pydantic.BaseModel):
    """Typed request to terminate the active candidate actor."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    parent_operation_id: str | None = None
    request_id: str
    token: str
    reason: str


_CandidateResultEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.result",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidatePendingResultEvent = hsm.Event[_CandidateResultData](
    name="bot.ability.autonomy.candidate.result.pending",
    kind=hsm.CompletionEventKind,
    schema=_CandidateResultData,
)
_CandidateStartedEvent = hsm.Event[_CandidateCapability](
    name="bot.ability.autonomy.candidate.started",
    kind=hsm.CompletionEventKind,
    schema=_CandidateCapability,
)
_CandidateCancelEvent = hsm.Event[_CandidateCancelData](
    name="bot.ability.autonomy.candidate.cancel",
    schema=_CandidateCancelData,
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
        candidates: tuple[Instance, ...],
        metadata: dict[str, object],
    ) -> hsm.Model:
        child_id = _child_operation_id(capability.operation_id, capability.index)
        cognition_input = turn.input

        def candidate_result(
            outcome: typing.Literal[
                "handled",
                "unhandled",
                "failed",
                "silent",
                "cancelled",
                "attach_timeout",
                "detach_failed",
                "detach_timeout",
            ],
            *,
            output: types.OutputData | None = None,
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
            return _HABIT_SILENCE_TIMEOUT

        def detach_delay(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return _HABIT_DETACH_TIMEOUT

        def attach_delay(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> datetime.timedelta:
            del ctx, instance, event
            return _HABIT_ATTACH_TIMEOUT

        def attach_behavior(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del ctx, event
            _ = behavior.attach(
                instance.context(),
                dataclasses.replace(
                    attachment.AttachEvent.with_data(attachment.AttachData(actor=instance, reply_to=instance)),
                    id=child_id,
                    source=hsm.id(instance),
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
                and event.source == hsm.id(behavior)
                and event.target == hsm.id(instance)
            )

        async def dispatch_behavior_input(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del ctx, event
            payload = habit_input_payload(cognition_input)
            if processing.active_operation(behavior, child_id) is None:
                _ = await processing.start_operation(behavior, child_id)
            await hsm.dispatch(
                instance.context(),
                behavior,
                dataclasses.replace(
                    behavior.input_event.with_data_and_id(payload, child_id),
                    source=hsm.id(instance),
                    target=hsm.id(behavior),
                    metadata=dict(metadata),
                ),
            )

        def is_behavior_output(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                event.name == behavior.output_event.name
                and event.id == child_id
                and event.source == hsm.id(behavior)
                and event.target == hsm.id(instance)
            )

        def is_behavior_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                event.name == behavior.failed_event.name
                and event.id == child_id
                and event.source == hsm.id(behavior)
                and event.target == hsm.id(instance)
            )

        def is_cancel(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            return (
                isinstance(event.data, _CandidateCancelData)
                and event.source == hsm.id(owner)
                and event.target == hsm.id(instance)
            )

        def result_for(instance: _CandidateRun, event: hsm.Event[typing.Any]) -> _CandidateResultData:
            if is_behavior_output(hsm.Context(), instance, event):
                output = _coerce_habit_output(event.data)
                return candidate_result(
                    "handled" if output is not None else "unhandled",
                    output=output,
                )
            if is_behavior_failure(hsm.Context(), instance, event):
                return candidate_result(
                    "failed",
                    message=getattr(event.data, "message", "Autonomy habit failed."),
                )
            if isinstance(event.data, _CandidateCancelData):
                return candidate_result(
                    "cancelled",
                    message=event.data.reason,
                    cancel_operation_id=event.data.operation_id,
                    cancel_parent_operation_id=event.data.parent_operation_id,
                    cancel_request_id=event.data.request_id,
                    cancel_token=event.data.token,
                )
            return candidate_result("silent")

        def begin_attach_timeout_detach(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            del event
            result = candidate_result(
                "attach_timeout",
                message=(f"Autonomy habit attach timed out after {_HABIT_ATTACH_TIMEOUT.total_seconds():g} seconds."),
            )
            _ = behavior.detach(
                instance.context(),
                dataclasses.replace(
                    attachment.DetachEvent.with_data(
                        attachment.DetachData(
                            actor=instance,
                            reply_to=instance,
                            timeout=_HABIT_DETACH_TIMEOUT,
                        )
                    ),
                    id=child_id,
                    source=hsm.id(instance),
                    target=hsm.id(behavior),
                    metadata=dict(metadata),
                ),
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CandidatePendingResultEvent.with_data(result),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(metadata),
                ),
            )

        def begin_detach(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = result_for(instance, event)
            processing.finish_operation(ctx, behavior, child_id)
            _ = behavior.detach(
                instance.context(),
                dataclasses.replace(
                    attachment.DetachEvent.with_data(
                        attachment.DetachData(
                            actor=instance,
                            reply_to=instance,
                            timeout=_HABIT_DETACH_TIMEOUT,
                        )
                    ),
                    id=child_id,
                    source=hsm.id(instance),
                    target=hsm.id(behavior),
                    metadata=dict(metadata),
                ),
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CandidatePendingResultEvent.with_data(result),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(metadata),
                ),
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
            )

        def forward_pending_result(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = event.data
            assert isinstance(result, _CandidateResultData)
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    _CandidateResultEvent.with_data(result),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(owner),
                    metadata=dict(metadata),
                ),
            )

        def forward_detach_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = event.data
            assert isinstance(result, _CandidateResultData)
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    _CandidateResultEvent.with_data(
                        result.model_copy(
                            update={
                                "outcome": "detach_failed",
                                "message": "Autonomy habit detach failed.",
                            }
                        )
                    ),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(owner),
                    metadata=dict(metadata),
                ),
            )

        def forward_detach_timeout(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = event.data
            assert isinstance(result, _CandidateResultData)
            _ = hsm.dispatch(
                ctx,
                owner,
                dataclasses.replace(
                    _CandidateResultEvent.with_data(
                        result.model_copy(
                            update={
                                "outcome": "detach_timeout",
                                "message": (
                                    "Autonomy habit detach timed out after "
                                    f"{_HABIT_DETACH_TIMEOUT.total_seconds():g} seconds."
                                ),
                            }
                        )
                    ),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(owner),
                    metadata=dict(metadata),
                ),
            )

        def forward_attach_failure(
            ctx: hsm.Context,
            instance: _CandidateRun,
            event: hsm.Event[typing.Any],
        ) -> None:
            result = candidate_result(
                "failed",
                message=getattr(event.data, "message", "Autonomy habit attach failed."),
            )
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _CandidatePendingResultEvent.with_data(result),
                    id=capability.operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(metadata),
                ),
            )

        return hsm.define(
            "AutonomyCandidateRun",
            hsm.initial(hsm.target("attaching")),
            hsm.state(
                "attaching",
                hsm.entry(attach_behavior),
                hsm.transition(
                    hsm.on(_CandidateCancelEvent),
                    hsm.guard(is_cancel),
                    hsm.effect(begin_detach),
                    hsm.target("/AutonomyCandidateRun/detaching"),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachCompleteEvent),
                    hsm.guard(is_attach_terminal),
                    hsm.target("/AutonomyCandidateRun/running"),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachFailedEvent),
                    hsm.guard(is_attach_terminal),
                    hsm.effect(forward_attach_failure),
                    hsm.target("/AutonomyCandidateRun/reporting"),
                ),
                hsm.transition(
                    hsm.after(attach_delay),
                    hsm.effect(begin_attach_timeout_detach),
                    hsm.target("/AutonomyCandidateRun/detaching"),
                ),
            ),
            hsm.state(
                "running",
                hsm.activity(dispatch_behavior_input),
                hsm.transition(
                    hsm.on(behavior.output_event),
                    hsm.guard(is_behavior_output),
                    hsm.effect(begin_detach),
                    hsm.target("/AutonomyCandidateRun/detaching"),
                ),
                hsm.transition(
                    hsm.on(behavior.failed_event),
                    hsm.guard(is_behavior_failure),
                    hsm.effect(begin_detach),
                    hsm.target("/AutonomyCandidateRun/detaching"),
                ),
                hsm.transition(
                    hsm.on(_CandidateCancelEvent),
                    hsm.guard(is_cancel),
                    hsm.effect(begin_detach),
                    hsm.target("/AutonomyCandidateRun/detaching"),
                ),
                hsm.transition(
                    hsm.after(silence_delay),
                    hsm.effect(begin_detach),
                    hsm.target("/AutonomyCandidateRun/detaching"),
                ),
            ),
            hsm.state(
                "detaching",
                hsm.defer(_CandidatePendingResultEvent),
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
                    hsm.after(detach_delay),
                    hsm.target("/AutonomyCandidateRun/reporting_timeout"),
                ),
            ),
            hsm.state(
                "reporting",
                hsm.transition(
                    hsm.on(_CandidatePendingResultEvent),
                    hsm.guard(is_pending_result),
                    hsm.effect(forward_pending_result),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
            ),
            hsm.state(
                "reporting_failure",
                hsm.transition(
                    hsm.on(_CandidatePendingResultEvent),
                    hsm.guard(is_pending_result),
                    hsm.effect(forward_detach_failure),
                    hsm.target("/AutonomyCandidateRun/done"),
                ),
            ),
            hsm.state(
                "reporting_timeout",
                hsm.transition(
                    hsm.on(_CandidatePendingResultEvent),
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


def _habit_select_input() -> memory.InputData:
    """Memory ability input: SELECT ACTIVE habits + triggers (skip DRAFT/BROKEN)."""

    clauses = typing.cast(tuple[Executable, ...], habit_storage.select_active_habits_clauses())
    return memory.InputData(statements=memory.compile_statements(*clauses))


def _habits_from_memory_output(output: memory.OutputData) -> tuple[Instance, ...]:
    if len(output.results) < 2:
        return ()
    habit_rows = tuple(row.as_mapping() for row in output.results[0].rows)
    trigger_rows = tuple(row.as_mapping() for row in output.results[1].rows)
    return habit_storage.instances_from_habit_results(habit_rows, trigger_rows)


def _habit_at_index(
    candidates: tuple[Instance, ...] | None,
    index: int | None,
) -> Instance | None:
    if candidates is None or index is None or index < 0 or index >= len(candidates):
        return None
    return candidates[index]


def _replace_habit_in_tuple(habits: tuple[Instance, ...], updated: Instance) -> tuple[Instance, ...]:
    return tuple(updated if item.name == updated.name else item for item in habits)


def _stimulus_payload(stimulus: object) -> object:
    """Build the JSON-like habit input payload from typed stimulus data."""

    if isinstance(stimulus, hsm.Event):
        payload = _json_like(stimulus.data)
        return payload
    if isinstance(stimulus, pydantic.BaseModel):
        return stimulus.model_dump(mode="json")
    return _json_like(stimulus)


def habit_input_payload(cognition_input: input.InputData) -> object:
    """Stimulus payload plus live focus context for habit guards and effects.

    Public so Reflection (and other cognitive abilities) can dry-run the same habit input shape
    Autonomy uses at runtime without reaching into Autonomy private helpers.
    """

    payload = _stimulus_payload(cognition_input.stimulus)
    if not isinstance(payload, dict):
        payload = {"stimulus": payload}
    else:
        payload = dict(typing.cast(dict[str, object], payload))
    if cognition_input.focus is not None:
        payload.setdefault("focus", cognition_input.focus)
    if cognition_input.focus_candidates:
        payload.setdefault("focus_candidates", list(cognition_input.focus_candidates))
    return payload


def _json_like(value: object) -> object:
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_like(dataclasses.asdict(value))
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): _json_like(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        sequence = typing.cast(collections.abc.Sequence[object], value)
        return [_json_like(item) for item in sequence]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _matches_trigger(habit: Instance, stimulus: str | None) -> bool:
    if not habit.triggers:
        return False
    if stimulus is None:
        return False
    return stimulus in habit.triggers


def _coerce_habit_output(data: object) -> types.OutputData | None:
    """Coerce habit terminal payload to cognition selections; None means unhandled by this habit."""

    if data is None:
        return None
    if isinstance(data, processing.Result):
        result = typing.cast(processing.Result[object], data)
        if not result.is_handled:
            return None
        return _coerce_habit_output(result.output)
    if types.is_output(data):
        return data
    selections = processing.coerce_event_selections(data)
    if selections is None:
        return None
    if not selections:
        return ()
    return types.OUTPUT_SCHEMA_CONTRACT.validate_python(
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


class Autonomy(ability.Ability[types.TurnData, types.CompletionData]):
    """Practiced automatic habit use: load on attach, then match → run Behavior.

    Chart states are lifecycle phases only. Turn product rides the event chain
    (match / start-candidate payloads and child request metadata → child terminal)
    (HSM-COMPLETION-001). ``_habits`` holds installed habits loaded at attach;
    Candidate identity remains on the scoped operation actor and typed capability events.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.TurnData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = types.CompletionData
    input_event: typing.ClassVar[hsm.Event[types.TurnData]] = ability.ability_input_event(
        "bot.ability.autonomy.input",
        types.TurnData,
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
    _habits: tuple[Instance, ...]

    @staticmethod
    def _persist_habit_usage(
        instance: "Autonomy",
        habit: Instance,
        *,
        outcome: typing.Literal["used", "failed"],
    ) -> Instance:
        """Write used/failed practice telemetry for one habit; update in-memory inventory.

        Returns the updated inventory Instance.
        """

        updated = habit_storage.mark_used(habit) if outcome == "used" else habit_storage.mark_failed(habit)
        store = instance._memory
        if store is not None:
            try:
                clauses = typing.cast(tuple[Executable, ...], habit_storage.replace_habit_clauses(updated))
                _ = store.execute(memory.InputData(statements=memory.compile_statements(*clauses)))
            except Exception:
                # Practice telemetry must not fail the turn; inventory may lag until next attach load.
                pass
        instance._habits = _replace_habit_in_tuple(instance._habits, updated)
        return updated

    @staticmethod
    async def _initialize_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Load installed habits through the injected Memory capability."""

        del event
        store = instance._memory
        if store is None:
            instance._habits = ()
            _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(None))
            return
        try:
            output = store.execute(_habit_select_input())
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                _HabitsLoadFailedEvent.with_data(ability.FailureData(message=str(error))),
            )
            return
        _ = hsm.dispatch(ctx, instance, _HabitsLoadedEvent.with_data(output))

    @staticmethod
    def _on_load_habits_output(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        output = event.data
        assert isinstance(output, memory.OutputData)
        instance._habits = _habits_from_memory_output(output)
        _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(None))

    @staticmethod
    def _on_load_habits_failure(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, instance
        failure = (
            event.data
            if isinstance(event.data, ability.FailureData)
            else ability.FailureData(message="Autonomy habit load failed.")
        )
        raise RuntimeError(f"Autonomy habit load failed: {failure.message}")

    @staticmethod
    def _detach_on_detach(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        if event.name != attachment.DetachEvent.name:
            return
        Autonomy._cancel_active_candidate(ctx, instance, reason="Autonomy detached.")
        instance._habits = ()

    @staticmethod
    def _cancel_active_candidate(
        ctx: hsm.Context,
        instance: "Autonomy",
        *,
        reason: str,
        operation_id: str = "detach",
        parent_operation_id: str | None = None,
        request_id: str = "detach",
        token: str = "detach",
    ) -> None:
        instances = instance.context().value(hsm.Keys.Instances)
        if not isinstance(instances, collections.abc.Mapping):
            return
        for actor_id, actor in tuple(instances.items()):
            if isinstance(actor_id, str) and isinstance(actor, _CandidateRun):
                _ = hsm.dispatch(
                    ctx,
                    actor,
                    dataclasses.replace(
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
                        target=actor_id,
                    ),
                )

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
        data = event.data
        assert isinstance(data, types.TurnData)
        if processing.active_operation(instance, event.id) is None:
            await processing.start_operation(instance, event.id)
        cognition_input = data.input
        stimulus = episodes.stimulus_name(cognition_input.stimulus)
        candidates = tuple(item for item in instance._habits if _matches_trigger(item, stimulus))
        matched = _MatchedEventData(
            turn=data,
            candidates=candidates,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _MatchedEvent.with_data(matched),
                id=event.id,
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
    def _dispatch_failure_terminal(
        ctx: hsm.Context,
        instance: "Autonomy",
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
    def _complete_apply(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ApplyCompletedEventData)
        Autonomy._dispatch_terminal(
            ctx,
            instance,
            operation_id=data.operation_id if data.operation_id is not None else (event.id or None),
            metadata=dict(event.metadata),
            output=types.CompletionData(turn=data.turn, output=data.output),
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
        )

    @staticmethod
    def _route_matched(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        matched = event.data
        assert isinstance(matched, _MatchedEventData)
        if not matched.candidates:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyCompletedEvent.with_data(
                        _ApplyCompletedEventData(
                            output=None,
                            turn=matched.turn,
                            operation_id=event.id or matched.turn.operation_id,
                        )
                    ),
                    id=event.id,
                    metadata=_public_metadata(dict(event.metadata)),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _StartCandidateEvent.with_data(
                    _StartCandidateEventData(
                        turn=matched.turn,
                        candidates=matched.candidates,
                        index=0,
                    )
                ),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _start_candidate_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _StartCandidateEventData)
        if data.index >= len(data.candidates):
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyCompletedEvent.with_data(
                        _ApplyCompletedEventData(
                            output=None,
                            turn=data.turn,
                            operation_id=event.id or data.turn.operation_id,
                        )
                    ),
                    id=event.id,
                    metadata=_public_metadata(dict(event.metadata)),
                ),
            )
            return
        compiled = data.candidates[data.index]
        try:
            behavior = habit.build(compiled.source)
        except Exception:
            # Unbuildable inventory is a runtime failure for this candidate; skip to next.
            updated = Autonomy._persist_habit_usage(instance, compiled, outcome="failed")
            updated_candidates = _replace_habit_in_tuple(data.candidates, updated)
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _StartCandidateEvent.with_data(
                        _StartCandidateEventData(
                            turn=data.turn,
                            candidates=updated_candidates,
                            index=data.index + 1,
                        )
                    ),
                    id=event.id,
                    metadata=dict(event.metadata),
                ),
            )
            return
        child_metadata = dict(event.metadata)
        capability = _CandidateCapability(
            operation_id=event.id or data.turn.operation_id,
            generation=data.turn.generation,
            index=data.index,
            token=uuid.uuid4().hex,
        )
        coordinator = _CandidateRun()
        private = hsm.Context(
            parent=instance.context(),
            values={hsm.Keys.Instances: {}},
        )
        started = await hsm.started(
            private,
            coordinator,
            _CandidateRun.model_for(
                owner=instance,
                behavior=behavior,
                capability=capability,
                turn=data.turn,
                candidates=data.candidates,
                metadata=child_metadata,
            ),
            hsm.Config(id=capability.token),
        )
        instances = instance.context().value(hsm.Keys.Instances)
        if isinstance(instances, collections.abc.MutableMapping):
            instances[hsm.id(started)] = started
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CandidateStartedEvent.with_data(capability),
                id=event.id,
                source=hsm.id(started),
                target=hsm.id(instance),
                metadata=dict(child_metadata),
            ),
        )

    @staticmethod
    def _matches_candidate_started(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        instances = instance.context().value(hsm.Keys.Instances)
        source = instances.get(event.source) if isinstance(instances, collections.abc.Mapping) else None
        return (
            isinstance(event.data, _CandidateCapability)
            and event.id == event.data.operation_id
            and event.target == hsm.id(instance)
            and event.source == event.data.token
            and isinstance(source, _CandidateRun)
        )

    @staticmethod
    def _matches_candidate_result(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        instances = instance.context().value(hsm.Keys.Instances)
        source = instances.get(event.source) if isinstance(instances, collections.abc.Mapping) else None
        return (
            isinstance(event.data, _CandidateResultData)
            and event.id == event.data.capability.operation_id
            and event.target == hsm.id(instance)
            and event.source == event.data.capability.token
            and isinstance(source, _CandidateRun)
        )

    @staticmethod
    def _retire_candidate(instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        instances = instance.context().value(hsm.Keys.Instances)
        if isinstance(instances, collections.abc.MutableMapping):
            _ = instances.pop(event.source, None)

    @staticmethod
    def _candidate_is_handled(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        return (
            Autonomy._matches_candidate_result(ctx, instance, event)
            and isinstance(data, _CandidateResultData)
            and data.outcome == "handled"
        )

    @staticmethod
    def _candidate_advances(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        return (
            Autonomy._matches_candidate_result(ctx, instance, event)
            and isinstance(data, _CandidateResultData)
            and data.outcome
            in {
                "unhandled",
                "failed",
                "silent",
            }
        )

    @staticmethod
    def _candidate_is_degraded(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        return (
            Autonomy._matches_candidate_result(ctx, instance, event)
            and isinstance(data, _CandidateResultData)
            and data.outcome
            in {
                "detach_failed",
                "detach_timeout",
                "attach_timeout",
            }
        )

    @staticmethod
    def _candidate_is_cancelled(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        return (
            Autonomy._matches_candidate_result(ctx, instance, event)
            and isinstance(data, _CandidateResultData)
            and data.outcome == "cancelled"
            and data.cancel_operation_id is not None
            and data.cancel_request_id is not None
            and data.cancel_token is not None
        )

    @staticmethod
    def _consume_candidate_result(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        Autonomy._retire_candidate(instance, event)

    @staticmethod
    async def _dispatch_candidate_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        parent_id = result.capability.operation_id
        index = result.capability.index
        cognition_input = result.turn.input
        candidates = result.candidates
        compiled = _habit_at_index(candidates, index)
        public_metadata = _public_metadata(dict(event.metadata))
        try:
            output = result.output
            if output is None:
                raise TypeError("Autonomy habit output is unhandled.")
            processing_input = input.build_processing_input(
                cognition_input,
                authority=instance._attachments[0] if instance._attachments else instance,
            )
            if processing_input.actors and output:
                selections = processing.coerce_event_selections(output)
                if selections is None:
                    raise TypeError("Autonomy habit output does not match event selections.")
                await types.dispatch_selected_events(
                    ctx,
                    processing_input,
                    selections,
                    operation_id=parent_id,
                    source=instance,
                    focus_candidates=result.turn.input.focus_candidates,
                    metadata=public_metadata,
                )
        except Exception as error:
            if compiled is not None:
                _ = Autonomy._persist_habit_usage(instance, compiled, outcome="failed")
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(types.FailureData(message=str(error), turn=result.turn)),
                    id=parent_id,
                    metadata=public_metadata,
                ),
            )
            return
        if compiled is not None:
            _ = Autonomy._persist_habit_usage(instance, compiled, outcome="used")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyCompletedEvent.with_data(
                    _ApplyCompletedEventData(
                        output=output,
                        turn=result.turn,
                        operation_id=parent_id,
                    )
                ),
                id=parent_id,
                metadata=public_metadata,
            ),
        )

    @staticmethod
    def _advance_candidate(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        Autonomy._retire_candidate(instance, event)
        parent_id = result.capability.operation_id
        index = result.capability.index
        candidates = result.candidates
        failed_habit = _habit_at_index(candidates, index) if result.outcome == "failed" else None
        if failed_habit is not None:
            updated = Autonomy._persist_habit_usage(instance, failed_habit, outcome="failed")
            candidates = _replace_habit_in_tuple(candidates or (), updated)
        next_index = index + 1
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _StartCandidateEvent.with_data(
                    _StartCandidateEventData(
                        turn=result.turn,
                        candidates=candidates,
                        index=next_index,
                    )
                ),
                id=parent_id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _degrade_candidate(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        result = event.data
        assert isinstance(result, _CandidateResultData)
        Autonomy._retire_candidate(instance, event)
        Autonomy._dispatch_failure_terminal(
            ctx,
            instance,
            operation_id=result.capability.operation_id,
            metadata=dict(event.metadata),
            failure=types.FailureData(
                message=result.message or "Autonomy habit teardown failed.",
                turn=result.turn,
            ),
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
        Autonomy._retire_candidate(instance, event)
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

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Autonomy",
        hsm.initial(hsm.target("/Autonomy/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event),
            hsm.activity(_initialize_activity),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(_HabitsLoadedEvent),
                hsm.effect(_on_load_habits_output),
            ),
            hsm.transition(
                hsm.on(_HabitsLoadFailedEvent),
                hsm.effect(_on_load_habits_failure),
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
                hsm.target("/Autonomy/matching"),
            ),
        ),
        hsm.state(
            "matching",
            hsm.defer(input_event),
            hsm.activity(_match_activity),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
                hsm.target("/Autonomy/idle"),
            ),
            hsm.transition(
                hsm.on(_MatchedEvent),
                hsm.guard(_has_matched),
                hsm.effect(_route_matched),
                hsm.target("/Autonomy/running"),
            ),
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
        ),
        hsm.state(
            "running",
            hsm.defer(input_event),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_request_cancel),
                hsm.target("/Autonomy/cancelling"),
            ),
            hsm.transition(
                hsm.on(_StartCandidateEvent),
                hsm.guard(_has_start_candidate),
                hsm.target("/Autonomy/starting"),
            ),
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
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_is_handled),
                hsm.effect(_consume_candidate_result),
                hsm.target("/Autonomy/dispatching"),
            ),
            hsm.transition(
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_advances),
                hsm.effect(_advance_candidate),
            ),
            hsm.transition(
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_is_degraded),
                hsm.effect(_degrade_candidate),
                hsm.target("/Autonomy/degraded"),
            ),
        ),
        hsm.state(
            "dispatching",
            hsm.defer(input_event),
            hsm.activity(_dispatch_candidate_activity),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
                hsm.target("/Autonomy/idle"),
            ),
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
        ),
        hsm.state(
            "starting",
            hsm.defer(input_event),
            hsm.activity(_start_candidate_activity),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_request_cancel),
                hsm.target("/Autonomy/cancelling"),
            ),
            hsm.transition(
                hsm.on(_CandidateStartedEvent),
                hsm.guard(_matches_candidate_started),
                hsm.target("/Autonomy/running"),
            ),
            hsm.transition(
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_advances),
                hsm.effect(_advance_candidate),
                hsm.target("/Autonomy/running"),
            ),
            hsm.transition(
                hsm.on(_StartCandidateEvent),
                hsm.guard(_has_start_candidate),
                # Rebuild next candidate without leaving the start path.
                hsm.target("/Autonomy/starting"),
            ),
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
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_is_handled),
                hsm.effect(_consume_candidate_result),
                hsm.target("/Autonomy/dispatching"),
            ),
            hsm.transition(
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_is_degraded),
                hsm.effect(_degrade_candidate),
                hsm.target("/Autonomy/degraded"),
            ),
        ),
        hsm.state(
            "cancelling",
            hsm.defer(input_event),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_is_cancelled),
                hsm.effect(_complete_cancel),
                hsm.target("/Autonomy/idle"),
            ),
            hsm.transition(
                hsm.on(_CandidateResultEvent),
                hsm.guard(_candidate_is_degraded),
                hsm.effect(_degrade_candidate),
                hsm.target("/Autonomy/degraded"),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.defer(input_event),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        memory: memory.Memory | None = None,
    ) -> None:
        super().__init__()
        self._memory = memory
        self._habits = ()


InputEvent = Autonomy.input_event
OutputEvent = Autonomy.output_event

__all__ = [
    "InputEvent",
    "OutputEvent",
    "Autonomy",
    "habit_input_payload",
]
