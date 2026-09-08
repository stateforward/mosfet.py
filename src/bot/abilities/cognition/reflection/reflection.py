"""Post-output reflection: recall, select behavior inventory, break, and store.

HSM phases (children attached; never call child ``_process`` / ``_apply``):
1. recall priors
2. **processing** — dispatch select processor (create | change | break | empty)
3. **revising** — delegate create/change authoring and validation to ``Revision``
4. **break** sets status=BROKEN (does not delete)
5. store cognitive episode; complete with no host product

Inventory policy:
- ``bot.behavior.create`` delegates DRAFT seeding and authoring to ``Revision``.
- ``bot.behavior.change`` delegates existing-row loading and authoring to ``Revision``.
- Changing always upserts; status=ACTIVE only when checks pass (usage counters preserved), else
  status=DRAFT with status_reason=validation.
- ``bot.behavior.break`` sets status=BROKEN with status_reason / status_updated_at; row stays for later change.
- Autonomy loads only status=ACTIVE behaviors; it writes ``used_*`` / ``failed_*`` practice telemetry.

Revision owns the HSM-visible author, validate, retry, and inventory persistence policy.
"""

from __future__ import annotations

from ... import ability
from ... import processing
from ... import memory

import dataclasses
import datetime
import collections.abc
import typing
import uuid

import hsm
import bot
from sqlalchemy.sql import ClauseElement

from bot.protocols import attachment
import pydantic
from pydantic.json_schema import SkipJsonSchema
from bot.behavior import (
    BreakData,
    BreakEvent,
    ChangeData,
    ChangeEvent,
    CreateData,
    CreateEvent,
    event_for_data,
)
from bot.behavior import storage as behavior_storage
from bot.behavior.instance import Instance
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span

from .. import episodes
from .. import input
from .. import types
from . import revision

CognitiveEpisode = episodes.CognitiveEpisode
stimulus_name = episodes.stimulus_name

SELECT_INSTRUCTIONS = (
    "Select at most one offered behavior inventory event (bot.behavior.create, change, or break) "
    "when a clear repeated pattern across this turn and prior_episodes should improve future behavior. "
    "Otherwise return an empty selection. Use only offered events; data must match the event schema. "
    "behaviors lists installed inventory with status (ACTIVE|DRAFT|BROKEN), status_reason, status_updated_at, "
    "and usage: used_count / last_used_at (handled Autonomy runs) and failed_count / last_failed_at. "
    "Prefer change over break when used_count is low (not enough practice evidence). "
    "Prefer break when failures or harm outweigh practice value; include reason. "
    "create delegates a DRAFT behavior revision; change revises an existing behavior; "
    "break sets status=BROKEN (keeps inventory for later change; Autonomy will not run it). "
    "Select intent may omit source — do not invent source here."
)

CHANGE_INSTRUCTIONS = revision.CHANGE_INSTRUCTIONS

INSTRUCTIONS = SELECT_INSTRUCTIONS

_SELECT_ID_SUFFIX = ":reflection:select"
_CHILD_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_CANCEL_TEARDOWN_TIMEOUT = datetime.timedelta(seconds=5)


def _one_shot_timeout() -> datetime.timedelta:
    return (
        _CHILD_OPERATION_TIMEOUT if _CHILD_OPERATION_TIMEOUT > datetime.timedelta(0) else datetime.timedelta(seconds=30)
    )


class InputData(pydantic.BaseModel):
    """Post-output reflection input: the cognition turn just handled by the host."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Reflection input after the host handled a cognition result. Reflection recalls prior "
                "episodes, may select create/change/break, runs a change pass for create/change, marks "
                "break, and stores this turn."
            ),
        },
    )

    cognition_input: SkipJsonSchema[input.InputData] = pydantic.Field(
        description="Original input payload given to cognition for this turn.",
    )
    cognition_output: types.OutputData = pydantic.Field(
        description="Typed cognition output selected for this turn and already handled by the host.",
    )


class SelectInput(pydantic.BaseModel):
    """Context for the inventory selection step."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Select step input. Choose one offered behavior inventory event or an empty selection. "
                "behaviors carries installed inventory including used/failed practice telemetry."
            ),
        },
    )

    cognition_input: SkipJsonSchema[input.InputData] = pydantic.Field(
        description="Original input payload given to cognition for this turn.",
    )
    cognition_output: types.OutputData = pydantic.Field(
        description="Typed cognition output selected for this turn and already handled by the host.",
    )
    prior_episodes: tuple[CognitiveEpisode, ...] = pydantic.Field(
        default=(),
        description="Prior cognition episodes recalled from memory for similar-turn behavior work.",
    )
    behaviors: tuple[Instance, ...] = pydantic.Field(
        default=(),
        description=(
            "Installed behavior inventory (any status) with status, status_reason, status_updated_at, "
            "used_count, last_used_at, failed_count, and last_failed_at."
        ),
    )
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


ChangeWriteInput = revision.ChangeWriteInput


ProcessorInput = SelectInput


ProcessorFactory = revision.ProcessorFactory


class _RecalledEventData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: InputData
    prior_episodes: tuple[CognitiveEpisode, ...] = ()
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


class _AppliedEventData(pydantic.BaseModel):
    """Behavior inventory side effects done; ready to store the episode."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: InputData
    behavior: CreateData | ChangeData | BreakData | None = None
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


class _StoredEventData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


class _SelectedEventData(pydantic.BaseModel):
    """Authorized select result carried through the active Reflection state."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: InputData
    prior_episodes: tuple[CognitiveEpisode, ...] = ()
    selection: types.EventData
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


_RecalledEvent = hsm.Event[_RecalledEventData](
    name="bot.ability.reflection.recalled",
    kind=hsm.CompletionEventKind,
    schema=_RecalledEventData,
)
_AppliedEvent = hsm.Event[_AppliedEventData](
    name="bot.ability.reflection.applied",
    kind=hsm.CompletionEventKind,
    schema=_AppliedEventData,
)
_StoredEvent = hsm.Event[_StoredEventData](
    name="bot.ability.reflection.stored",
    kind=hsm.CompletionEventKind,
    schema=_StoredEventData,
)
_SelectedEvent = hsm.Event[_SelectedEventData](
    name="bot.ability.reflection.selected",
    kind=hsm.CompletionEventKind,
    schema=_SelectedEventData,
)


class _StageFailedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    failure: ability.FailureData
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)


_StageFailedEvent = hsm.Event[_StageFailedData](
    name="bot.ability.reflection.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=_StageFailedData,
)
_SelectHopTimedOutEvent = hsm.Event[object](
    name="bot.ability.reflection.select.hop.timed_out",
    kind=hsm.ErrorEventKind,
    schema=object,
)


def behavior_events() -> tuple[processing.Event[typing.Any], ...]:
    """Behavior inventory HSM events offered on the select input."""

    return (CreateEvent, ChangeEvent, BreakEvent)


def episode_from_turn(
    cognition_turn: InputData,
    *,
    behavior: CreateData | ChangeData | BreakData | None = None,
) -> CognitiveEpisode:
    """Build a recallable episode from the reflection turn just handled."""

    return CognitiveEpisode(
        focus=cognition_turn.cognition_input.focus,
        focus_candidates=cognition_turn.cognition_input.focus_candidates,
        stimulus_name=stimulus_name(cognition_turn.cognition_input.stimulus),
        output=cognition_turn.cognition_output,
        behavior=behavior,
    )


def _coerce_output_data(value: object) -> types.OutputData:
    selections = processing.coerce_event_selections(value)
    if selections is None:
        raise TypeError("Reflection select step must return an events array.")
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


def event_for_data_from_selection(item: types.EventData) -> hsm.Event[typing.Any]:
    """Build a behavior inventory HSM event from one select-step selection."""

    if item.data is None:
        raw: dict[str, object] = {}
    elif isinstance(item.data, collections.abc.Mapping):
        raw = dict(typing.cast("collections.abc.Mapping[str, object]", item.data))
    else:
        raise TypeError(f"Reflection inventory selection data must be a mapping, got {type(item.data).__name__}.")
    if item.event == CreateEvent.name:
        return event_for_data(CreateData.model_validate(raw))
    if item.event == ChangeEvent.name:
        return event_for_data(ChangeData.model_validate(raw))
    if item.event == BreakEvent.name:
        return event_for_data(BreakData.model_validate(raw))
    raise TypeError(f"Reflection select returned unsupported event: {item.event}.")


def _compile_behavior_statements(clauses: tuple[ClauseElement, ...]) -> tuple[memory.Statement, ...]:
    return memory.compile_statements(*clauses)


def _load_behavior(store: memory.Memory, *, name: str) -> Instance | None:
    out = store.execute(
        memory.InputData(
            statements=_compile_behavior_statements(behavior_storage.select_behavior_by_name_clauses(name))
        )
    )
    if len(out.results) < 2:
        return None
    behavior_rows = tuple(row.as_mapping() for row in out.results[0].rows)
    trigger_rows = tuple(row.as_mapping() for row in out.results[1].rows)
    loaded = behavior_storage.instances_from_behavior_results(behavior_rows, trigger_rows)
    return loaded[0] if loaded else None


def _load_all_behaviors(store: memory.Memory) -> tuple[Instance, ...]:
    """Load full behavior inventory (any status) for Reflection select context."""

    out = store.execute(
        memory.InputData(statements=_compile_behavior_statements(behavior_storage.select_all_behaviors_clauses()))
    )
    if len(out.results) < 2:
        return ()
    behavior_rows = tuple(row.as_mapping() for row in out.results[0].rows)
    trigger_rows = tuple(row.as_mapping() for row in out.results[1].rows)
    return behavior_storage.instances_from_behavior_results(behavior_rows, trigger_rows)


def _store_behavior(
    store: memory.Memory,
    *,
    behavior: Instance,
    context_ref: str | None,
) -> None:
    """Upsert behavior inventory row."""

    del context_ref
    _ = store.execute(
        memory.InputData(statements=_compile_behavior_statements(behavior_storage.replace_behavior_clauses(behavior)))
    )


def _store_episode(
    store: memory.Memory,
    episode: CognitiveEpisode,
    *,
    context_ref: str | None,
) -> None:
    insert_input = episodes.episode_insert_input(
        episode,
        context_ref=context_ref,
        scope=getattr(type(store), "default_scope", "short_term"),
    )
    _ = store.execute(insert_input)


def _apply_break(data: BreakData, *, store: memory.Memory, context_ref: str | None) -> BreakData:
    """Set status=BROKEN in inventory; do not delete (change can revive it later)."""

    existing = _load_behavior(store, name=data.name)
    if existing is None:
        raise ValueError(f"Reflection break selected unknown behavior: {data.name}.")
    reason = data.reason.strip() if data.reason and data.reason.strip() else None
    retired = behavior_storage.mark_broken(existing, reason=reason)
    _store_behavior(store, behavior=retired, context_ref=context_ref)
    return data


class Reflection(processing.Processing):
    _attachment_group: attachment.Group | None
    """Post-output ability: recall → select → revise or break → store."""

    instructions: typing.ClassVar[str] = SELECT_INSTRUCTIONS
    select_instructions: typing.ClassVar[str] = SELECT_INSTRUCTIONS
    change_instructions: typing.ClassVar[str] = CHANGE_INSTRUCTIONS
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = type(None)
    input_event: typing.ClassVar[hsm.Event[InputData]] = hsm.Event[InputData](
        name="bot.ability.reflection.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[None]] = hsm.Event[type(None)](
        name="bot.ability.reflection.output",
        schema=type(None),
    )
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _select_processing: processing.Processing
    _revision: revision.Revision
    _memory: memory.Memory
    _events: tuple[processing.Event[typing.Any], ...]

    @staticmethod
    def _child_id(instance: "Reflection", suffix: str) -> str:
        return f"{hsm.id(instance)}{suffix}"

    @staticmethod
    def _has_reflection_input(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _operation(event: hsm.Event[typing.Any]) -> tuple[str, str] | None:
        data = event.data
        if isinstance(data, processing.CompletionData):
            nested = data.input.input
            if isinstance(nested, SelectInput):
                return nested.operation_id, nested.generation
        if isinstance(data, revision.OutputData | revision.FailureData):
            return data.input.parent_operation_id, data.input.parent_generation
        if isinstance(
            data,
            _RecalledEventData | _AppliedEventData | _StoredEventData | _SelectedEventData | _StageFailedData,
        ):
            return data.operation_id, data.generation
        return None

    @staticmethod
    def _matches_operation(instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        return processing.matches_private_terminal(instance, event, Reflection._operation(event))

    @staticmethod
    def _private_event(
        instance: "Reflection",
        source_event: hsm.Event[typing.Any],
        event_type: hsm.Event[typing.Any],
        data: object,
    ) -> hsm.Event[typing.Any]:
        operation = Reflection._operation(source_event)
        assert operation is not None
        operation_id, generation = operation
        if event_type.name == _StageFailedEvent.name and isinstance(data, ability.FailureData):
            data = _StageFailedData(
                failure=data,
                operation_id=operation_id,
                generation=generation,
            )
        return dataclasses.replace(
            event_type.with_data(data),
            id=operation_id,
            source=hsm.id(instance),
            target=hsm.id(instance),
            metadata=dict(source_event.metadata),
        )

    @staticmethod
    def _is_reflection_cancel_request(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        return processing.Processing._is_cancel_request(ctx, instance, event)

    @staticmethod
    def _cancel_select(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelData)
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._select_processing,
            event,
            request_id=Reflection._child_id(instance, _SELECT_ID_SUFFIX),
            parent_operation_id=data.operation_id,
            token=data.token,
        )

    @staticmethod
    def _cancel_revision(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelData)
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._revision,
            event,
            request_id=data.operation_id,
            parent_operation_id=data.operation_id,
            token=data.token,
        )

    @staticmethod
    def _cancel_select_timeout(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        request_id = Reflection._child_id(instance, _SELECT_ID_SUFFIX)
        processing.finish_operation(ctx, instance, request_id)
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._select_processing,
            event,
            request_id=request_id,
            parent_operation_id=processing.active_operation_id(instance) or event.id or request_id,
            token=uuid.uuid4().hex,
        )

    @staticmethod
    def _is_select_hop_timeout(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return event.source == hsm.id(instance) and event.target == hsm.id(instance)

    @staticmethod
    def _cancel_revision_timeout(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        operation_id = processing.active_operation_id(instance)
        if operation_id is None:
            return
        processing.dispatch_child_cancel(
            ctx,
            instance,
            instance._revision,
            event,
            request_id=operation_id,
            parent_operation_id=operation_id,
            token=uuid.uuid4().hex,
        )

    @staticmethod
    def _child_timeout_delay(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        if _CHILD_OPERATION_TIMEOUT <= datetime.timedelta(0):
            return datetime.timedelta(milliseconds=1)
        return _CHILD_OPERATION_TIMEOUT

    @staticmethod
    def _matches_cancelled(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, processing.CancelledData):
            return False
        select_id = Reflection._child_id(instance, _SELECT_ID_SUFFIX)
        select_matches = event.source == hsm.id(instance._select_processing) and data.operation_id == select_id
        revision_matches = (
            event.source == hsm.id(instance._revision)
            and data.parent_operation_id is not None
            and data.operation_id == data.parent_operation_id
            and processing.active_operation(instance, data.parent_operation_id) is not None
        )
        return (
            (select_matches or revision_matches) and event.id == data.operation_id and event.target == hsm.id(instance)
        )

    @staticmethod
    def _cancel_timeout_delay(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _CANCEL_TEARDOWN_TIMEOUT

    @staticmethod
    def _emit_reflection_cancelled(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelledData)
        operation_id = data.parent_operation_id or data.operation_id
        token = data.token
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                processing.CancelledEvent.with_data(
                    processing.CancelledData(
                        operation_id=operation_id,
                        token=token,
                    )
                ),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )
        processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _has_recalled(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _RecalledEventData) and Reflection._matches_operation(instance, event)

    @staticmethod
    def _has_reflection_applied(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _AppliedEventData) and Reflection._matches_operation(instance, event)

    @staticmethod
    def _has_stored(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _StoredEventData) and Reflection._matches_operation(instance, event)

    @staticmethod
    def _has_stage_failure(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _StageFailedData) and Reflection._matches_operation(instance, event)

    @staticmethod
    def _fail_from_stage(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _StageFailedData)
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=data.failure,
        )

    @staticmethod
    def _fail_child(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.FailureData)
        child_input = data.input.input
        assert isinstance(child_input, SelectInput)
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=child_input.operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message=data.message),
        )

    @staticmethod
    def _fail_cancel_timeout(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        operation_id = processing.active_operation_id(instance)
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message="Reflection child cancellation timed out."),
        )

    @staticmethod
    def _request_reboot(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        # Child teardown / revision-reboot / cancel-timeout path (not detach terminal).
        processing.request_reboot(ctx, instance, event, reason="cognition_child_teardown_failed")

    @staticmethod
    def _request_detach_rollback_reboot(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        processing.request_reboot(ctx, instance, event, reason="cognition_detach_rollback_failed")

    @staticmethod
    def _is_revision_reboot(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            isinstance(event.data, bot.RebootEventData)
            and event.source == hsm.id(instance._revision)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    async def _recall_activity(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.reflection.stage",
            scope="bot.abilities.cognition",
            component="cognition.reflection",
            stage="reflection_recall",
            context=telemetry.event_context(event),
        ):
            turn = event.data
            assert isinstance(turn, InputData)
            operation_id = event.id if event.id else uuid.uuid4().hex
            if processing.active_operation(instance, operation_id) is None:
                _ = await processing.start_operation(instance, operation_id)
            active = processing.active_operation(instance, operation_id)
            assert active is not None
            generation = hsm.id(active)
            try:
                select_input = episodes.episode_select_input(context_ref=turn.cognition_input.focus)
                recalled = instance._memory.execute(select_input)
                prior = episodes.episodes_from_output(recalled)
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _StageFailedEvent.with_data(
                            _StageFailedData(
                                failure=ability.FailureData(message=f"Reflection memory recall failed: {error}"),
                                operation_id=operation_id,
                                generation=generation,
                            )
                        ),
                        id=operation_id,
                        source=hsm.id(instance),
                        target=hsm.id(instance),
                        metadata=dict(event.metadata),
                    ),
                )
                return
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _RecalledEvent.with_data(
                        _RecalledEventData(
                            turn=turn,
                            prior_episodes=prior,
                            operation_id=operation_id,
                            generation=generation,
                        )
                    ),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    async def _dispatch_select(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RecalledEventData)
        turn = data.turn
        operation_id = data.operation_id
        try:
            behaviors = _load_all_behaviors(instance._memory)
        except Exception:
            behaviors = ()
        select_input = processing.InputData(
            input=SelectInput(
                cognition_input=turn.cognition_input,
                cognition_output=turn.cognition_output,
                prior_episodes=data.prior_episodes,
                behaviors=behaviors,
                operation_id=operation_id,
                generation=data.generation,
            ),
            schemas=behavior_events(),
            actors={},
        )
        hop_id = Reflection._child_id(instance, _SELECT_ID_SUFFIX)
        input_event = dataclasses.replace(
            instance._select_processing.input_event.with_data_and_id(
                select_input,
                hop_id,
            ),
            metadata=dict(event.metadata),
        )
        if processing.active_operation(instance, hop_id) is None:
            _ = await processing.start_operation(instance, hop_id)
        try:
            terminal = await ability.run_terminal_operation(
                instance.context(),
                child=instance._select_processing,
                request=input_event,
                terminals=(
                    instance._select_processing.output_event,
                    instance._select_processing.failed_event,
                ),
                timeout=_one_shot_timeout(),
                on_terminal=lambda _terminal: processing.finish_operation(ctx, instance, hop_id),
            )
        except TimeoutError:
            processing.finish_operation(ctx, instance, hop_id)
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _SelectHopTimedOutEvent.with_data_and_id(None, hop_id),
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )
            return
        processing.finish_operation(ctx, instance, hop_id)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                terminal,
                source=hsm.id(instance._select_processing),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    def _matches_select_output(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._select_processing
        completion = event.data
        select_input = completion.input.input if isinstance(completion, processing.CompletionData) else None
        if not isinstance(select_input, SelectInput):
            return False
        return processing.matches_child_terminal(
            instance,
            child,
            event,
            request_id=Reflection._child_id(instance, _SELECT_ID_SUFFIX),
            operation_id=select_input.operation_id,
            generation=select_input.generation,
        )

    @staticmethod
    def _matches_select_failure(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._select_processing
        failure = event.data
        select_input = failure.input.input if isinstance(failure, processing.FailureData) else None
        if not isinstance(select_input, SelectInput):
            return False
        return processing.matches_child_terminal(
            instance,
            child,
            event,
            request_id=Reflection._child_id(instance, _SELECT_ID_SUFFIX),
            operation_id=select_input.operation_id,
            generation=select_input.generation,
        )

    @staticmethod
    def _matches_revision_output(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, revision.OutputData):
            return False
        return processing.matches_child_terminal(
            instance,
            instance._revision,
            event,
            request_id=data.input.parent_operation_id,
            operation_id=data.input.parent_operation_id,
            generation=data.input.parent_generation,
        )

    @staticmethod
    def _matches_revision_failure(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, revision.FailureData):
            return False
        return processing.matches_child_terminal(
            instance,
            instance._revision,
            event,
            request_id=data.input.parent_operation_id,
            operation_id=data.input.parent_operation_id,
            generation=data.input.parent_generation,
        )

    @staticmethod
    def _apply_revision(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, revision.OutputData)
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                event,
                _AppliedEvent,
                _AppliedEventData(
                    turn=InputData(
                        cognition_input=data.input.cognition_input,
                        cognition_output=data.input.cognition_output,
                    ),
                    behavior=data.applied,
                    operation_id=data.input.parent_operation_id,
                    generation=data.input.parent_generation,
                ),
            ),
        )

    @staticmethod
    def _fail_revision(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, revision.FailureData)
        processing.dispatch_terminal_failure(
            ctx,
            instance,
            operation_id=data.input.parent_operation_id,
            metadata=dict(event.metadata),
            failure=ability.FailureData(message=data.message),
        )

    @staticmethod
    def _on_select_output(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        """Route select terminal: empty → store; create/change/break → seed/write/break."""

        completion = event.data
        assert isinstance(completion, processing.CompletionData)
        select_input = completion.input.input
        assert isinstance(select_input, SelectInput)
        turn = InputData(
            cognition_input=select_input.cognition_input,
            cognition_output=select_input.cognition_output,
        )
        operation_id = select_input.operation_id
        try:
            selection = _coerce_output_data(completion.output)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=str(error)),
                ),
            )
            return
        if not selection:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _AppliedEvent,
                    _AppliedEventData(
                        turn=turn,
                        behavior=None,
                        operation_id=operation_id,
                        generation=select_input.generation,
                    ),
                ),
            )
            return
        if len(selection) != 1:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message="Reflection select must return at most one behavior event."),
                ),
            )
            return
        chosen = selection[0]
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                event,
                _SelectedEvent,
                _SelectedEventData(
                    turn=turn,
                    prior_episodes=select_input.prior_episodes,
                    selection=chosen,
                    operation_id=operation_id,
                    generation=select_input.generation,
                ),
            ),
        )

    @staticmethod
    def _matches_select_empty(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        return Reflection._matches_select_output(ctx, instance, event)

    @staticmethod
    def _selected_is(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
        name: str,
    ) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _SelectedEventData)
            and Reflection._matches_operation(instance, event)
            and event.id == data.operation_id
            and data.selection.event == name
        )

    @staticmethod
    def _selected_is_create(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        return Reflection._selected_is(ctx, instance, event, CreateEvent.name)

    @staticmethod
    def _selected_is_change(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        return Reflection._selected_is(ctx, instance, event, ChangeEvent.name)

    @staticmethod
    def _selected_is_break(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        return Reflection._selected_is(ctx, instance, event, BreakEvent.name)

    @staticmethod
    def _dispatch_revision(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        selected = event.data
        assert isinstance(selected, _SelectedEventData)
        try:
            raw = typing.cast(object, selected.selection.data)
            if selected.selection.event == CreateEvent.name:
                intent: CreateData | ChangeData = (
                    raw if isinstance(raw, CreateData) else CreateData.model_validate(raw if raw is not None else {})
                )
            else:
                intent = (
                    raw if isinstance(raw, ChangeData) else ChangeData.model_validate(raw if raw is not None else {})
                )
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=str(error)),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance._revision,
            dataclasses.replace(
                instance._revision.input_event.with_data_and_id(
                    revision.InputData(
                        cognition_input=selected.turn.cognition_input,
                        cognition_output=selected.turn.cognition_output,
                        prior_episodes=selected.prior_episodes,
                        intent=intent,
                        parent_operation_id=selected.operation_id,
                        parent_generation=selected.generation,
                    ),
                    selected.operation_id,
                ),
                source=hsm.id(instance),
                target=hsm.id(instance._revision),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _on_break_event(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        """BreakEvent → set inventory status=BROKEN (no write phase)."""

        selected = event.data
        assert isinstance(selected, _SelectedEventData)
        turn = selected.turn
        try:
            raw = typing.cast(object, selected.selection.data)
            data = raw if isinstance(raw, BreakData) else BreakData.model_validate(raw if raw is not None else {})
            applied = _apply_break(data, store=instance._memory, context_ref=turn.cognition_input.focus)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=f"Reflection break failed: {error}"),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                event,
                _AppliedEvent,
                _AppliedEventData(
                    turn=turn,
                    behavior=applied,
                    operation_id=selected.operation_id,
                    generation=selected.generation,
                ),
            ),
        )

    @staticmethod
    async def _store_activity(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> None:
        with span.operation(
            "bot.reflection.stage",
            scope="bot.abilities.cognition",
            component="cognition.reflection",
            stage="reflection_store",
            context=telemetry.event_context(event),
        ):
            data = event.data
            assert isinstance(data, _AppliedEventData)
            try:
                episode = episode_from_turn(data.turn, behavior=data.behavior)
                _store_episode(instance._memory, episode, context_ref=data.turn.cognition_input.focus)
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    Reflection._private_event(
                        instance,
                        event,
                        _StageFailedEvent,
                        ability.FailureData(message=f"Reflection episode store failed: {error}"),
                    ),
                )
                return
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StoredEvent,
                    _StoredEventData(
                        operation_id=data.operation_id,
                        generation=data.generation,
                    ),
                ),
            )

    @staticmethod
    def _complete_from_stored(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _StoredEventData)
        processing.dispatch_terminal_output(
            ctx,
            instance,
            operation_id=data.operation_id,
            metadata=dict(event.metadata),
            output=None,
        )

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "Reflection",
        hsm.initial(hsm.target("/Reflection/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Reflection/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_reflection_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
            ),
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_reflection_input),
                hsm.target("/Reflection/recalling"),
            ),
        ),
        hsm.state(
            "recalling",
            hsm.defer(input_event),
            hsm.activity(_recall_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_reflection_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.on(_RecalledEvent),
                hsm.guard(_has_recalled),
                hsm.target("/Reflection/processing"),
            ),
            hsm.transition(
                hsm.on(_StageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reflection/idle"),
            ),
        ),
        hsm.state(
            "processing",
            hsm.defer(input_event),
            hsm.activity(_dispatch_select),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_reflection_cancel_request),
                hsm.effect(_cancel_select),
                hsm.target("/Reflection/cancelling"),
            ),
            hsm.transition(
                hsm.on(processing.OutputEvent),
                hsm.guard(_matches_select_empty),
                hsm.effect(_on_select_output),
            ),
            hsm.transition(
                hsm.on(_SelectedEvent),
                hsm.guard(_selected_is_create),
                hsm.effect(_dispatch_revision),
                hsm.target("/Reflection/revising"),
            ),
            hsm.transition(
                hsm.on(_SelectedEvent),
                hsm.guard(_selected_is_change),
                hsm.effect(_dispatch_revision),
                hsm.target("/Reflection/revising"),
            ),
            hsm.transition(
                hsm.on(_SelectedEvent),
                hsm.guard(_selected_is_break),
                hsm.effect(_on_break_event),
            ),
            hsm.transition(
                hsm.on(_AppliedEvent),
                hsm.guard(_has_reflection_applied),
                hsm.target("/Reflection/storing"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_matches_select_failure),
                hsm.effect(_fail_child),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.on(_StageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.on(_SelectHopTimedOutEvent),
                hsm.guard(_is_select_hop_timeout),
                hsm.effect(_cancel_select_timeout),
                hsm.target("/Reflection/timing_out"),
            ),
        ),
        hsm.state(
            "revising",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(bot.RebootEvent),
                hsm.guard(_is_revision_reboot),
                hsm.effect(_request_reboot),
                hsm.target("/Reflection/rebooting"),
            ),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_reflection_cancel_request),
                hsm.effect(_cancel_revision),
                hsm.target("/Reflection/cancelling"),
            ),
            hsm.transition(
                hsm.on(revision.OutputEvent),
                hsm.guard(_matches_revision_output),
                hsm.effect(_apply_revision),
            ),
            hsm.transition(
                hsm.on(_AppliedEvent),
                hsm.guard(_has_reflection_applied),
                hsm.target("/Reflection/storing"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_matches_revision_failure),
                hsm.effect(_fail_revision),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.after(_child_timeout_delay),
                hsm.effect(_cancel_revision_timeout),
                hsm.target("/Reflection/timing_out"),
            ),
        ),
        hsm.state(
            "storing",
            hsm.defer(input_event),
            hsm.activity(_store_activity),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_reflection_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.on(_StoredEvent),
                hsm.guard(_has_stored),
                hsm.effect(_complete_from_stored),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.on(_StageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reflection/idle"),
            ),
        ),
        hsm.state(
            "cancelling",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(bot.RebootEvent),
                hsm.guard(_is_revision_reboot),
                hsm.effect(_request_reboot),
                hsm.target("/Reflection/rebooting"),
            ),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.guard(_matches_cancelled),
                hsm.effect(_emit_reflection_cancelled),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.after(_cancel_timeout_delay),
                hsm.effect(_fail_cancel_timeout, _request_reboot),
                hsm.target("/Reflection/rebooting"),
            ),
        ),
        hsm.state(
            "timing_out",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(bot.RebootEvent),
                hsm.guard(_is_revision_reboot),
                hsm.effect(_request_reboot),
                hsm.target("/Reflection/rebooting"),
            ),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.guard(_matches_cancelled),
                hsm.effect(_fail_cancel_timeout),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.after(_cancel_timeout_delay),
                hsm.effect(_fail_cancel_timeout, _request_reboot),
                hsm.target("/Reflection/rebooting"),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(
                    ability.Ability._deliver_composite_attachment_terminal,
                    _request_detach_rollback_reboot,
                ),
                hsm.target("/Reflection/rebooting"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Reflection/idle"),
            ),
        ),
        hsm.state("rebooting", hsm.defer(input_event)),
        hsm.observe(observer),
    )

    def __init__(
        self,
        *,
        processor: processing.Processor | ProcessorFactory,
        memory: memory.Memory,
    ) -> None:
        super(processing.Processing, self).__init__()
        leaf = processor() if not isinstance(processor, processing.Processor) else processor
        # Shared transport; each phase stamps its own system policy on apply.
        self._select_processing = processing.Processing(
            processor=leaf,
            instructions=type(self).select_instructions,
        )
        self._revision = revision.Revision(
            processor=leaf,
            memory=memory,
        )
        self._memory = memory
        self._attachment_group = attachment.Group(
            self._select_processing,
            self._revision,
            self._memory,
        )
        self._events = behavior_events()


InputEvent = Reflection.input_event
OutputEvent = Reflection.output_event

__all__ = [
    "CHANGE_INSTRUCTIONS",
    "CognitiveEpisode",
    "ChangeWriteInput",
    "INSTRUCTIONS",
    "InputData",
    "InputEvent",
    "OutputEvent",
    "ProcessorFactory",
    "ProcessorInput",
    "SELECT_INSTRUCTIONS",
    "SelectInput",
    "Reflection",
    "episode_from_turn",
    "behavior_events",
    "stimulus_name",
]
