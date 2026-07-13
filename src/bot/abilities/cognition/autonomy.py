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
import pydantic

from bot import habit
from bot.habit.instance import Instance
from bot.habit import storage as habit_storage
from bot.telemetry import observer

from . import dispatch
from . import episodes
from . import input
from . import types

_AUTONOMY_INPUT_METADATA_KEY = "bot.autonomy.input"
_AUTONOMY_CANDIDATES_METADATA_KEY = "bot.autonomy.candidates"
_AUTONOMY_INDEX_METADATA_KEY = "bot.autonomy.candidate_index"
_AUTONOMY_ID_MARKER = ":autonomy:"
_LOAD_HABITS_ID_SUFFIX = ":load_habits"
_LOAD_HABITS_OPERATION_ID = f"autonomy{_LOAD_HABITS_ID_SUFFIX}"
_InitializingCompleteEvent = hsm.Event[object](
    name="bot.ability.autonomy.initializing.complete",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)


class _MatchedEventData(pydantic.BaseModel):
    """Private completion: candidate habit instances for this stimulus."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    cognition_input: input.InputData
    candidates: tuple[Instance, ...]
    operation_id: str


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

    cognition_input: input.InputData
    candidates: tuple[Instance, ...]
    index: int
    operation_id: str


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
    operation_id: str | None = None


_ApplyCompletedEvent = hsm.Event[_ApplyCompletedEventData](
    name="bot.ability.autonomy.apply.completed",
    kind=hsm.CompletionEventKind,
    schema=_ApplyCompletedEventData,
)
_ApplyFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.autonomy.apply.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


def _public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            _AUTONOMY_INPUT_METADATA_KEY,
            _AUTONOMY_CANDIDATES_METADATA_KEY,
            _AUTONOMY_INDEX_METADATA_KEY,
        }
    }


def _cognition_input_from_event(event: hsm.Event[typing.Any]) -> input.InputData | None:
    value = event.metadata.get(_AUTONOMY_INPUT_METADATA_KEY)
    if input.is_input(value):
        return value
    return None


def _candidates_from_event(event: hsm.Event[typing.Any]) -> tuple[Instance, ...] | None:
    value = event.metadata.get(_AUTONOMY_CANDIDATES_METADATA_KEY)
    if isinstance(value, tuple) and all(isinstance(item, Instance) for item in value):
        return typing.cast(tuple[Instance, ...], value)
    return None


def _index_from_event(event: hsm.Event[typing.Any]) -> int | None:
    value = event.metadata.get(_AUTONOMY_INDEX_METADATA_KEY)
    if isinstance(value, int):
        return value
    return None


def _parse_child_id(event_id: str | None) -> tuple[str | None, int | None]:
    """Parse ``{parent}:autonomy:{index}`` into parent id and candidate index."""

    if event_id is None or not event_id:
        return None, None
    parent, separator, index_text = event_id.rpartition(_AUTONOMY_ID_MARKER)
    if not separator or not parent:
        return None, None
    if not index_text.isdigit():
        return None, None
    return parent, int(index_text)


def _child_operation_id(parent_operation_id: str, index: int) -> str:
    return f"{parent_operation_id}{_AUTONOMY_ID_MARKER}{index}"


def _habit_select_input() -> memory.InputData:
    """Memory ability input: SELECT ACTIVE habits + triggers (skip DRAFT/BROKEN)."""

    return memory.InputData(
        statements=memory.compile_statements(*habit_storage.select_active_habits_clauses())
    )


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
    """JSON-like payload for habit input, including call_id from stimulus metadata when present."""

    if isinstance(stimulus, hsm.Event):
        payload = _json_like(stimulus.data)
        call_id = stimulus.metadata.get("bot.phone.call_id")
        if isinstance(call_id, str) and call_id and isinstance(payload, dict):
            payload = dict(typing.cast(dict[str, object], payload))
            payload.setdefault("call_id", call_id)
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


def _owner(instance: "Autonomy") -> typing.Any | None:
    return ability.Ability.current_owner(instance)


def _zero_timeout(
    ctx: hsm.Context,
    instance: "Autonomy",
    event: hsm.Event[typing.Any],
) -> datetime.timedelta:
    """Immediate after-timeout used to leave starting / detect silent habits."""

    del ctx, instance, event
    return datetime.timedelta(0)


class Autonomy(ability.Ability[input.InputData, types.OutputData | None]):
    """Practiced automatic habit use: load on attach, then match → run Behavior.

    Chart states are lifecycle phases only. Turn product rides the event chain
    (match / start-candidate payloads and child request metadata → child terminal)
    (HSM-COMPLETION-001). ``_habits`` holds installed habits loaded at attach;
    ``_active_behavior`` is only the owned child machine slot for the active run.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = input.InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    input_event: typing.ClassVar[hsm.Event[input.InputData]] = ability.ability_input_event(
        "bot.ability.autonomy.input",
        input.InputData,
    )
    output_event: typing.ClassVar[hsm.Event[types.OutputData | None]] = hsm.Event[types.OutputData | None](
        name="bot.ability.autonomy.output",
        schema=types.OPTIONAL_OUTPUT_SCHEMA_CONTRACT,
    )
    _memory: memory.Memory | None
    _habits: tuple[Instance, ...]
    _active_behavior: ability.Ability[typing.Any, typing.Any] | None

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

        updated = (
            habit_storage.mark_used(habit)
            if outcome == "used"
            else habit_storage.mark_failed(habit)
        )
        store = instance._memory
        if store is not None:
            try:
                _ = store.execute(
                    memory.InputData(
                        statements=memory.compile_statements(*habit_storage.replace_habit_clauses(updated))
                    )
                )
            except Exception:
                # Practice telemetry must not fail the turn; inventory may lag until next attach load.
                pass
        instance._habits = _replace_habit_in_tuple(instance._habits, updated)
        return updated

    @staticmethod
    async def _attach_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Attach memory (if any) and request habit load through the Memory ability."""

        del event
        store = instance._memory
        if store is None:
            instance._habits = ()
            _ = hsm.dispatch(ctx, instance, _InitializingCompleteEvent.with_data(None))
            return
        owner = ability.Ability.current_owner(store)
        if owner is not None and owner is not instance:
            raise ValueError(f"{type(store).__name__} is already owned by {type(owner).__name__}.")
        _ = await store.attach(owner=instance, ctx=ctx)
        load_event = dataclasses.replace(
            store.input_event.with_data_and_id(
                _habit_select_input(),
                _LOAD_HABITS_OPERATION_ID,
            ),
            metadata={},
        )
        _ = hsm.dispatch(ctx, store, load_event)

    @staticmethod
    def _matches_load_habits_output(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        store = instance._memory
        if store is None:
            return False
        child_id = event.id if event.id else None
        return (
            event.name == store.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(store)
            and child_id is not None
            and child_id.endswith(_LOAD_HABITS_ID_SUFFIX)
            and isinstance(event.data, memory.OutputData)
        )

    @staticmethod
    def _matches_load_habits_failure(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        store = instance._memory
        if store is None:
            return False
        child_id = event.id if event.id else None
        return (
            event.name == store.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(store)
            and child_id is not None
            and child_id.endswith(_LOAD_HABITS_ID_SUFFIX)
        )

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
        if event.name != ability.DetachEvent.name:
            return
        Autonomy._detach_active(ctx, instance)
        instance._habits = ()
        store = instance._memory
        if store is not None and ability.Ability.current_owner(store) is instance:
            _ = store.detach(ctx=ctx)

    @staticmethod
    def _detach_active(ctx: hsm.Context, instance: "Autonomy") -> None:
        active = instance._active_behavior
        if active is None:
            return
        if ability.Ability.current_owner(active) is instance:
            _ = active.detach(ctx=ctx)
        instance._active_behavior = None

    @staticmethod
    def _has_autonomy_input(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return input.is_input(event.data)

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
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    async def _match_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert input.is_input(data)
        operation_id = event.id if event.id else uuid.uuid4().hex
        stimulus = episodes.stimulus_name(data.stimulus)
        candidates = tuple(item for item in instance._habits if _matches_trigger(item, stimulus))
        matched = _MatchedEventData(
            cognition_input=data,
            candidates=candidates,
            operation_id=operation_id,
        )
        child_metadata = dict(event.metadata)
        child_metadata[_AUTONOMY_INPUT_METADATA_KEY] = data
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _MatchedEvent.with_data(matched),
                id=operation_id,
                metadata=child_metadata,
            ),
        )

    @staticmethod
    def _dispatch_terminal(
        ctx: hsm.Context,
        instance: "Autonomy",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        output: types.OutputData | None,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=operation_id,
            metadata=_public_metadata(metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _dispatch_failure_terminal(
        ctx: hsm.Context,
        instance: "Autonomy",
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
    def _complete_apply(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ApplyCompletedEventData)
        Autonomy._detach_active(ctx, instance)
        Autonomy._dispatch_terminal(
            ctx,
            instance,
            operation_id=data.operation_id if data.operation_id is not None else (event.id or None),
            metadata=dict(event.metadata),
            output=data.output,
        )

    @staticmethod
    def _fail_apply(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        Autonomy._detach_active(ctx, instance)
        failure = (
            event.data
            if isinstance(event.data, ability.FailureData)
            else ability.FailureData(message="Autonomy failed.")
        )
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
                        _ApplyCompletedEventData(output=None, operation_id=matched.operation_id)
                    ),
                    id=matched.operation_id,
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
                        cognition_input=matched.cognition_input,
                        candidates=matched.candidates,
                        index=0,
                        operation_id=matched.operation_id,
                    )
                ),
                id=matched.operation_id,
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
        Autonomy._detach_active(ctx, instance)
        if data.index >= len(data.candidates):
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyCompletedEvent.with_data(
                        _ApplyCompletedEventData(output=None, operation_id=data.operation_id)
                    ),
                    id=data.operation_id,
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
                            cognition_input=data.cognition_input,
                            candidates=updated_candidates,
                            index=data.index + 1,
                            operation_id=data.operation_id,
                        )
                    ),
                    id=data.operation_id,
                    metadata={
                        **dict(event.metadata),
                        _AUTONOMY_CANDIDATES_METADATA_KEY: updated_candidates,
                    },
                ),
            )
            return
        instance._active_behavior = behavior
        _ = await behavior.attach(owner=instance, ctx=ctx)
        child_metadata = dict(event.metadata)
        # Preserve stimulus provenance (e.g. bot.phone.call_id) for habit callbacks.
        stimulus = data.cognition_input.stimulus
        if isinstance(stimulus, hsm.Event):
            child_metadata = {**dict(stimulus.metadata), **child_metadata}
        # Habit guards may check focus candidates (same key Bot uses).
        if data.cognition_input.focus_candidates:
            child_metadata.setdefault(
                "bot.focus_candidates",
                data.cognition_input.focus_candidates,
            )
        if data.cognition_input.focus is not None:
            child_metadata.setdefault("bot.bot.focus", data.cognition_input.focus)
        child_metadata[_AUTONOMY_INPUT_METADATA_KEY] = data.cognition_input
        child_metadata[_AUTONOMY_CANDIDATES_METADATA_KEY] = data.candidates
        child_metadata[_AUTONOMY_INDEX_METADATA_KEY] = data.index
        payload = habit_input_payload(data.cognition_input)
        input_event = dataclasses.replace(
            behavior.input_event.with_data_and_id(
                payload,
                _child_operation_id(data.operation_id, data.index),
            ),
            metadata=child_metadata,
        )
        _ = hsm.dispatch(ctx, behavior, input_event)

    @staticmethod
    def _matches_behavior_output(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        active = instance._active_behavior
        if active is None:
            return False
        parent_id, index = _parse_child_id(event.id if event.id else None)
        if parent_id is None or index is None:
            return False
        return (
            event.name == active.output_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(active)
            and _cognition_input_from_event(event) is not None
        )

    @staticmethod
    def _matches_behavior_failure(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        active = instance._active_behavior
        if active is None:
            return False
        parent_id, index = _parse_child_id(event.id if event.id else None)
        if parent_id is None or index is None:
            return False
        return (
            event.name == active.failed_event.name
            and event.target == hsm.id(instance)
            and event.source == hsm.id(active)
        )

    @staticmethod
    def _behavior_output_is_handled(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        return (
            Autonomy._matches_behavior_output(ctx, instance, event)
            and _coerce_habit_output(event.data) is not None
        )

    @staticmethod
    def _behavior_output_is_unhandled(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> bool:
        return (
            Autonomy._matches_behavior_output(ctx, instance, event)
            and _coerce_habit_output(event.data) is None
        )

    @staticmethod
    async def _dispatch_behavior_activity(
        ctx: hsm.Context,
        instance: "Autonomy",
        event: hsm.Event[typing.Any],
    ) -> None:
        parent_id, index = _parse_child_id(event.id if event.id else None)
        cognition_input = _cognition_input_from_event(event)
        candidates = _candidates_from_event(event)
        if index is None:
            index = _index_from_event(event)
        compiled = _habit_at_index(candidates, index)
        public_metadata = _public_metadata(dict(event.metadata))
        Autonomy._detach_active(ctx, instance)
        if cognition_input is None or parent_id is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        ability.FailureData(message="Autonomy habit output is missing turn correlation.")
                    ),
                    id=parent_id,
                    metadata=public_metadata,
                ),
            )
            return
        try:
            output = _coerce_habit_output(event.data)
            if output is None:
                raise TypeError("Autonomy habit output is unhandled.")
            input = dispatch.build_processing_input(cognition_input, owner=_owner(instance))
            if input.actors and output:
                selections = processing.coerce_event_selections(output)
                if selections is None:
                    raise TypeError("Autonomy habit output does not match event selections.")
                await dispatch.dispatch_selected_events(
                    ctx,
                    input,
                    selections,
                    operation_id=parent_id,
                    source=instance,
                    metadata=public_metadata,
                )
        except Exception as error:
            if compiled is not None:
                Autonomy._persist_habit_usage(instance, compiled, outcome="failed")
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(ability.FailureData(message=str(error))),
                    id=parent_id,
                    metadata=public_metadata,
                ),
            )
            return
        if compiled is not None:
            Autonomy._persist_habit_usage(instance, compiled, outcome="used")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ApplyCompletedEvent.with_data(
                    _ApplyCompletedEventData(output=output, operation_id=parent_id)
                ),
                id=parent_id,
                metadata=public_metadata,
            ),
        )

    @staticmethod
    def _advance_candidate(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        parent_id, index = _parse_child_id(event.id if event.id else None)
        cognition_input = _cognition_input_from_event(event)
        candidates = _candidates_from_event(event)
        meta_index = _index_from_event(event)
        if index is None:
            index = meta_index
        active = instance._active_behavior
        is_behavior_failure = (
            active is not None
            and event.name == active.failed_event.name
            and event.source == hsm.id(active)
        )
        failed_habit = _habit_at_index(candidates, index) if is_behavior_failure else None
        Autonomy._detach_active(ctx, instance)
        if failed_habit is not None:
            updated = Autonomy._persist_habit_usage(instance, failed_habit, outcome="failed")
            candidates = _replace_habit_in_tuple(candidates or (), updated)
        if parent_id is None or index is None or cognition_input is None or candidates is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyFailedEvent.with_data(
                        ability.FailureData(message="Autonomy advance is missing turn correlation.")
                    ),
                    id=parent_id,
                    metadata=_public_metadata(dict(event.metadata)),
                ),
            )
            return
        next_index = index + 1
        if next_index >= len(candidates):
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _ApplyCompletedEvent.with_data(
                        _ApplyCompletedEventData(output=None, operation_id=parent_id)
                    ),
                    id=parent_id,
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
                        cognition_input=cognition_input,
                        candidates=candidates,
                        index=next_index,
                        operation_id=parent_id,
                    )
                ),
                id=parent_id,
                metadata={
                    **dict(event.metadata),
                    _AUTONOMY_CANDIDATES_METADATA_KEY: candidates,
                },
            ),
        )

    @staticmethod
    def _silence_advance(ctx: hsm.Context, instance: "Autonomy", event: hsm.Event[typing.Any]) -> None:
        """Habit input produced no terminal: advance to next candidate or unhandled."""

        del event
        active = instance._active_behavior
        if active is None:
            return
        # Without the last child event we cannot recover candidate metadata from silence alone.
        # Complete unhandled; habits should always terminal (output or failure).
        Autonomy._detach_active(ctx, instance)
        _ = hsm.dispatch(
            ctx,
            instance,
            _ApplyCompletedEvent.with_data(_ApplyCompletedEventData(output=None, operation_id=None)),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Autonomy",
        hsm.initial(hsm.target("/Autonomy/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event),
            hsm.activity(_attach_activity),
            hsm.exit(_detach_on_detach),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_load_habits_output),
                hsm.effect(_on_load_habits_output),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_load_habits_failure),
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
                hsm.on(hsm.AnyEvent),
                hsm.guard(_behavior_output_is_handled),
                hsm.target("/Autonomy/dispatching"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_behavior_output_is_unhandled),
                hsm.effect(_advance_candidate),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_behavior_failure),
                hsm.effect(_advance_candidate),
            ),
            hsm.transition(
                hsm.after(_zero_timeout),
                hsm.effect(_silence_advance),
            ),
        ),
        hsm.state(
            "dispatching",
            hsm.defer(input_event),
            hsm.activity(_dispatch_behavior_activity),
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
        ),
        hsm.state(
            "starting",
            hsm.defer(input_event),
            hsm.activity(_start_candidate_activity),
            hsm.exit(_detach_on_detach),
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
                hsm.on(hsm.AnyEvent),
                hsm.guard(_behavior_output_is_handled),
                hsm.target("/Autonomy/dispatching"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_behavior_output_is_unhandled),
                hsm.effect(_advance_candidate),
                hsm.target("/Autonomy/running"),
            ),
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.guard(_matches_behavior_failure),
                hsm.effect(_advance_candidate),
                hsm.target("/Autonomy/running"),
            ),
            # Activity finished dispatching habit input; wait for terminal in running.
            hsm.transition(
                hsm.after(_zero_timeout),
                hsm.target("/Autonomy/running"),
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
        self._active_behavior = None


InputEvent = Autonomy.input_event
OutputEvent = Autonomy.output_event

__all__ = [
    "InputEvent",
    "OutputEvent",
    "Autonomy",
    "habit_input_payload",
]
