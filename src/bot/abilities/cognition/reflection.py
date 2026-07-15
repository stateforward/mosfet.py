"""Post-output reflection: select habit inventory, then change or break.

HSM phases (children attached; never call child ``_process`` / ``_apply``):
1. recall priors
2. **processing** — dispatch select processor (create | change | break | empty)
3. **changing** — author / fix Starlark for a stored habit (empty stub or existing)
4. **break** sets status=BROKEN (does not delete)
5. store cognitive episode; complete with no host product

Inventory policy:
- ``bot.habit.create`` seeds a DRAFT empty inventory row, then enters **changing**.
- ``bot.habit.change`` loads an existing row (any status) and enters **changing**.
- Changing always upserts; status=ACTIVE only when checks pass (usage counters preserved), else
  status=DRAFT with status_reason=validation.
- ``bot.habit.break`` sets status=BROKEN with status_reason / status_updated_at; row stays for later change.
- Autonomy loads only status=ACTIVE habits; it writes ``used_*`` / ``failed_*`` practice telemetry.

Change fix policy is HSM-visible: same diagnostic messages after a fix attempt → abandon;
changed messages → retry change.
"""

from __future__ import annotations

from .. import ability
from .. import processing
from .. import memory

import dataclasses
import datetime
import collections.abc
import typing
import uuid

import hsm
from sqlalchemy.sql import Executable

from bot.protocols import attachment
import pydantic
from pydantic.json_schema import SkipJsonSchema
from bot.habit import (
    BreakData,
    BreakEvent,
    ChangeData,
    ChangeEvent,
    CreateData,
    CreateEvent,
    event_for_data,
)
from bot.habit import diagnostic as habit_diagnostic
from bot.habit import storage as habit_storage
from bot.habit.instance import (
    STATUS_REASON_VALIDATION,
    Instance,
)
from bot.habit.source import STARLARK_API
from bot.telemetry import observer

from . import episodes
from . import dispatch
from . import input
from . import types

CognitiveEpisode = episodes.CognitiveEpisode
stimulus_name = episodes.stimulus_name

SELECT_INSTRUCTIONS = (
    "Select at most one offered habit inventory event (bot.habit.create, change, or break) "
    "when a clear repeated pattern across this turn and prior_episodes should improve future behavior. "
    "Otherwise return an empty selection. Use only offered events; data must match the event schema. "
    "habits lists installed inventory with status (ACTIVE|DRAFT|BROKEN), status_reason, status_updated_at, "
    "and usage: used_count / last_used_at (handled Autonomy runs) and failed_count / last_failed_at. "
    "Prefer change over break when used_count is low (not enough practice evidence). "
    "Prefer break when failures or harm outweigh practice value; include reason. "
    "create seeds a DRAFT empty habit then enters changing; change revises an existing habit; "
    "break sets status=BROKEN (keeps inventory for later change; Autonomy will not run it). "
    "Select intent may omit source — do not invent source here."
)

CHANGE_INSTRUCTIONS = (
    "You are in the changing phase: author or revise Starlark for the stored habit. "
    "existing_habit is the current inventory row (source may be empty for a new create stub; "
    "status may be DRAFT or BROKEN for unfinished or retired habits). "
    "Select the offered bot.habit.change event once with the same name and required `source`. "
    "If existing_habit.source is empty, write a full new program from this turn, prior_episodes, "
    "and intent — invent event contracts, guards, and effects from the observed pattern only. "
    "If source is non-empty, rewrite or patch it so it better fits this turn and prior_episodes. "
    "Keep the model name stable unless the intent clearly renames the habit.\n"
    "Habit terminal output must be a cognition event selection object (keys: event, target?, data?, reason?) "
    "or a list of such objects, matching the shape of prior episode / this-turn outputs. "
    "output_event JSON schema root must be type object. "
    "Derive input fields, guards, and selection field names from the observed pattern. "
    "At runtime, fill selection data from the live event (event['data'] / event['metadata']) — "
    "never hardcode identifier values copied from episode examples. "
    "Set triggers to the stimulus names that should propose the habit. "
    "Effects must hsm.dispatch(output_event, selection) and must not return a value. "
    "Starlark only: no Python docstrings, type annotations, or imports; callbacks are def name(event): ...\n"
    "If diagnostics is present, prior source failed validation: revise `source` to clear every error "
    "(use code, stage, message, and help). failed_source is the rejected program when provided. "
    "Repeating the same diagnostic message after a fix ends change; change the source so messages clear.\n\n"
    f"{STARLARK_API}"
)

INSTRUCTIONS = SELECT_INSTRUCTIONS

_REFLECTION_TURN_METADATA_KEY = "bot.reflection.turn"
_REFLECTION_PRIOR_METADATA_KEY = "bot.reflection.prior_episodes"
_REFLECTION_ABILITIES_METADATA_KEY = "bot.reflection.abilities"
_REFLECTION_SKILLS_METADATA_KEY = "bot.reflection.skills"
_REFLECTION_OPERATION_ID_METADATA_KEY = "bot.reflection.operation_id"
_REFLECTION_FIX_ATTEMPTS_METADATA_KEY = "bot.reflection.fix_attempts"
_REFLECTION_LAST_DIAGNOSTIC_MESSAGES_KEY = "bot.reflection.last_diagnostic_messages"
_REFLECTION_CREATE_INTENT_METADATA_KEY = "bot.reflection.create_intent"
_REFLECTION_CHANGE_INTENT_METADATA_KEY = "bot.reflection.change_intent"
_REFLECTION_EXISTING_HABIT_METADATA_KEY = "bot.reflection.existing_habit"
_REFLECTION_CAPABILITY_METADATA_KEY = "bot.reflection.capability"
_REFLECTION_CANCEL_CAPABILITY_METADATA_KEY = "bot.reflection.cancel_capability"
_SELECT_ID_SUFFIX = ":reflection:select"
_CHANGE_ID_SUFFIX = ":reflection:change"
_CHILD_OPERATION_TIMEOUT = datetime.timedelta(seconds=30)
_CANCEL_TEARDOWN_TIMEOUT = datetime.timedelta(seconds=5)
_MAX_REFLECTION_FIX_ATTEMPTS = 2


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
                "Select step input. Choose one offered habit inventory event or an empty selection. "
                "habits carries installed inventory including used/failed practice telemetry."
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
        description="Prior cognition episodes recalled from memory for similar-turn habit work.",
    )
    habits: tuple[Instance, ...] = pydantic.Field(
        default=(),
        description=(
            "Installed habit inventory (any status) with status, status_reason, status_updated_at, "
            "used_count, last_used_at, failed_count, and last_failed_at."
        ),
    )


class ChangeWriteInput(pydantic.BaseModel):
    """Changing-phase input: turn context, change intent, and the stored existing habit."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Author or revise executable habit Starlark. existing_habit may be an empty DRAFT "
                "create stub or an ACTIVE/DRAFT/BROKEN installed habit. Return ChangeData with source. "
                "When diagnostics is set, revise failed_source."
            ),
        },
    )

    cognition_input: SkipJsonSchema[input.InputData] = pydantic.Field(
        description="Original input payload given to cognition for this turn.",
    )
    cognition_output: types.OutputData = pydantic.Field(
        description="Typed cognition output for this turn.",
    )
    prior_episodes: tuple[CognitiveEpisode, ...] = pydantic.Field(
        default=(),
        description="Prior cognition episodes recalled for this reflection.",
    )
    intent: ChangeData = pydantic.Field(
        description="Change payload naming the habit to author or revise.",
    )
    existing_habit: Instance = pydantic.Field(
        description=(
            "Current inventory habit: empty DRAFT stub (create) or ACTIVE/DRAFT/BROKEN row (including source)."
        ),
    )
    diagnostics: habit_diagnostic.Report | None = pydantic.Field(
        default=None,
        description="Structured validation failures from a prior change attempt, if any.",
    )
    failed_source: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Rejected Starlark source from the prior change attempt, when diagnostics is set.",
    )


ProcessorInput = SelectInput


class ProcessorFactory(typing.Protocol):
    """Build a leaf ``processing.Processor`` for Reflection phases (transport only).

    System policy is owned by each phase's ``Processing(instructions=...)`` wrapper, not
    the factory.
    """

    def __call__(self) -> processing.Processor: ...


class _RecalledEventData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    host_input: processing.InputData
    prior_episodes: tuple[CognitiveEpisode, ...] = ()


class _AppliedEventData(pydantic.BaseModel):
    """Habit inventory side effects done; ready to store the episode."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: InputData
    habit: CreateData | ChangeData | BreakData | None = None


class _StoredEventData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


class _ChangeWriteCheckedData(pydantic.BaseModel):
    """Change-write validation result for HSM accept / retry / abandon guards."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: InputData
    prior_episodes: tuple[CognitiveEpisode, ...] = ()
    written: ChangeData
    habit_instance: Instance | None = None
    report: habit_diagnostic.Report = pydantic.Field(default_factory=habit_diagnostic.Report)
    previous_messages: tuple[str, ...] | None = None
    operation_id: str
    existing: Instance | None = None


class _SelectedEventData(pydantic.BaseModel):
    """Authorized select result normalized by the operation actor."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: InputData
    prior_episodes: tuple[CognitiveEpisode, ...] = ()
    selection: types.EventData
    operation_id: str = pydantic.Field(min_length=1)


class _ChangeRequestedEventData(pydantic.BaseModel):
    """Typed request for one operation-scoped change-processing attempt."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    turn: InputData
    prior_episodes: tuple[CognitiveEpisode, ...]
    intent: ChangeData
    existing: Instance
    operation_id: str = pydantic.Field(min_length=1)
    diagnostics: habit_diagnostic.Report | None = None
    failed_source: str | None = None
    create_intent: CreateData | None = None


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
_ChangeWriteCheckedEvent = hsm.Event[_ChangeWriteCheckedData](
    name="bot.ability.reflection.change_write.checked",
    kind=hsm.CompletionEventKind,
    schema=_ChangeWriteCheckedData,
)
_SelectedEvent = hsm.Event[_SelectedEventData](
    name="bot.ability.reflection.selected",
    kind=hsm.CompletionEventKind,
    schema=_SelectedEventData,
)
_ChangeRequestedEvent = hsm.Event[_ChangeRequestedEventData](
    name="bot.ability.reflection.change.requested",
    schema=_ChangeRequestedEventData,
)
_ChangeStartedEvent = hsm.Event[dispatch.OperationData](
    name="bot.ability.reflection.change.started",
    kind=hsm.CompletionEventKind,
    schema=dispatch.OperationData,
)
_StageFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.reflection.stage.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class _ReflectionCapability(pydantic.BaseModel):
    """Immutable authority for one complete Reflection turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    actor_id: str
    token: str


class _ReflectionCancelCapability(pydantic.BaseModel):
    """Immutable host cancellation authority retained through mediator teardown."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str
    token: str
    phase: typing.Literal["reflection-select", "reflection-change"]
    parent_operation: dispatch.OperationData | None = None
    turn_actor_id: str | None = None


class _ReflectionOperation(hsm.Instance):
    """Scoped identity actor retained for the full Reflection turn."""

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "ReflectionOperation",
        hsm.initial(hsm.target("/ReflectionOperation/active")),
        hsm.state("active"),
    )


def habit_events() -> tuple[processing.Event[typing.Any], ...]:
    """Habit inventory HSM events offered on the select input."""

    return (CreateEvent, ChangeEvent, BreakEvent)


def episode_from_turn(
    cognition_turn: InputData,
    *,
    habit: CreateData | ChangeData | BreakData | None = None,
) -> CognitiveEpisode:
    """Build a recallable episode from the reflection turn just handled."""

    return CognitiveEpisode(
        focus=cognition_turn.cognition_input.focus,
        focus_candidates=cognition_turn.cognition_input.focus_candidates,
        stimulus_name=stimulus_name(cognition_turn.cognition_input.stimulus),
        output=cognition_turn.cognition_output,
        habit=habit,
    )


def _public_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            _REFLECTION_TURN_METADATA_KEY,
            _REFLECTION_PRIOR_METADATA_KEY,
            _REFLECTION_ABILITIES_METADATA_KEY,
            _REFLECTION_SKILLS_METADATA_KEY,
            _REFLECTION_OPERATION_ID_METADATA_KEY,
            dispatch.OPERATION_METADATA_KEY,
        }
    }


def _operation_id_from_metadata(metadata: dict[str, object]) -> str | None:
    value = metadata.get(_REFLECTION_OPERATION_ID_METADATA_KEY)
    return value if isinstance(value, str) and value else None


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
    """Build a habit inventory HSM event from one select-step selection."""

    raw = dict(item.data or {})
    if item.event == CreateEvent.name:
        return event_for_data(CreateData.model_validate(raw))
    if item.event == ChangeEvent.name:
        return event_for_data(ChangeData.model_validate(raw))
    if item.event == BreakEvent.name:
        return event_for_data(BreakData.model_validate(raw))
    raise TypeError(f"Reflection select returned unsupported event: {item.event}.")


def _change_data_from_output(value: object) -> ChangeData:
    if isinstance(value, ChangeData):
        return value
    selections = processing.coerce_event_selections(value)
    if selections is not None:
        if len(selections) != 1 or selections[0].event != ChangeEvent.name:
            raise TypeError("Reflection write must return a single change event.")
        return ChangeData.model_validate(selections[0].data or {})
    return ChangeData.model_validate(value)


def _habit_sample_input(cognition_input: input.InputData) -> tuple[object, dict[str, object]]:
    """Build dry-run payload/metadata matching Autonomy's live habit input."""

    from . import autonomy

    payload = autonomy.habit_input_payload(cognition_input)
    metadata: dict[str, object] = {}
    stimulus = cognition_input.stimulus
    if isinstance(stimulus, hsm.Event):
        metadata.update(dict(stimulus.metadata))
    if cognition_input.focus_candidates:
        _ = metadata.setdefault("bot.focus_candidates", cognition_input.focus_candidates)
    if cognition_input.focus is not None:
        _ = metadata.setdefault("bot.bot.focus", cognition_input.focus)
    return payload, metadata


def _habit_check_from_write(
    *,
    name: str,
    source: str | None,
    triggers: tuple[str, ...],
    description: str | None,
    cognition_input: input.InputData | None = None,
) -> habit_diagnostic.Checked[Instance]:
    """Validate write payload into an installable Instance (or a Report)."""

    from bot.habit.instance import check
    from bot.habit.verify import verify_apply

    if source is None or not source.strip():
        report = habit_diagnostic.report_of(
            habit_diagnostic.diagnostic(
                code=habit_diagnostic.E0001_EMPTY,
                message="Executable habit write requires non-empty starlark source.",
                stage=habit_diagnostic.Stage.INVENTORY,
                help=habit_diagnostic.help_for(habit_diagnostic.E0001_EMPTY),
            )
        )
        return habit_diagnostic.Checked[Instance](value=None, report=report)
    if cognition_input is None:
        return check(
            source,
            name=name,
            triggers=triggers if triggers else None,
            description=description,
            require_build=True,
        )
    sample_input, sample_metadata = _habit_sample_input(cognition_input)
    return verify_apply(
        source,
        name=name,
        triggers=triggers if triggers else None,
        description=description,
        input_data=sample_input,
        metadata=sample_metadata,
    )


def _fix_attempts_from_metadata(metadata: dict[str, object]) -> int:
    value = metadata.get(_REFLECTION_FIX_ATTEMPTS_METADATA_KEY, 0)
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def _diagnostic_messages(report: habit_diagnostic.Report) -> tuple[str, ...]:
    return tuple(item.message for item in report.errors)


def _last_diagnostic_messages(metadata: dict[str, object]) -> tuple[str, ...] | None:
    value = metadata.get(_REFLECTION_LAST_DIAGNOSTIC_MESSAGES_KEY)
    if isinstance(value, (tuple, list)):
        values = typing.cast(tuple[object, ...] | list[object], value)
        messages = tuple(item for item in values if isinstance(item, str))
        if len(messages) == len(values):
            return messages
    return None


def _compile_habit_statements(clauses: tuple[object, ...]) -> tuple[memory.Statement, ...]:
    return memory.compile_statements(*(typing.cast(Executable, clause) for clause in clauses))


def _load_habit(store: memory.Memory, *, name: str) -> Instance | None:
    out = store.execute(
        memory.InputData(statements=_compile_habit_statements(habit_storage.select_habit_by_name_clauses(name)))
    )
    if len(out.results) < 2:
        return None
    habit_rows = tuple(row.as_mapping() for row in out.results[0].rows)
    trigger_rows = tuple(row.as_mapping() for row in out.results[1].rows)
    loaded = habit_storage.instances_from_habit_results(habit_rows, trigger_rows)
    return loaded[0] if loaded else None


def _load_all_habits(store: memory.Memory) -> tuple[Instance, ...]:
    """Load full habit inventory (any status) for Reflection select context."""

    out = store.execute(
        memory.InputData(statements=_compile_habit_statements(habit_storage.select_all_habits_clauses()))
    )
    if len(out.results) < 2:
        return ()
    habit_rows = tuple(row.as_mapping() for row in out.results[0].rows)
    trigger_rows = tuple(row.as_mapping() for row in out.results[1].rows)
    return habit_storage.instances_from_habit_results(habit_rows, trigger_rows)


def _store_habit(
    store: memory.Memory,
    *,
    habit: Instance,
    context_ref: str | None,
) -> None:
    """Upsert habit inventory row."""

    del context_ref
    _ = store.execute(
        memory.InputData(statements=_compile_habit_statements(habit_storage.replace_habit_clauses(habit)))
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


def _draft_stub_from_create(intent: CreateData) -> Instance:
    """Empty DRAFT inventory row created when select chooses create."""

    return habit_storage.mark_draft(
        Instance(
            name=intent.name,
            source="",
            triggers=intent.triggers,
            description=intent.description or "",
        )
    )


def _instance_for_inventory(
    *,
    name: str,
    source: str | None,
    triggers: tuple[str, ...],
    description: str | None,
    habit_instance: Instance | None,
    existing: Instance | None = None,
) -> Instance | None:
    """Build inventory Instance: ACTIVE when checks passed, else DRAFT after validation failure.

    Usage counters are preserved from ``existing`` (same habit name across change).
    """

    if habit_instance is not None:
        active = habit_storage.mark_active(habit_instance)
        return habit_storage.preserve_usage(active, existing)
    if source is None or not source.strip():
        return None
    draft = habit_storage.mark_draft(
        Instance(
            name=name,
            source=source.strip(),
            triggers=triggers,
            description=description or "",
        ),
        reason=STATUS_REASON_VALIDATION,
    )
    return habit_storage.preserve_usage(draft, existing)


def _store_change_result(
    data: ChangeData,
    habit: Instance,
    *,
    store: memory.Memory,
    context_ref: str | None,
) -> ChangeData:
    _store_habit(store, habit=habit, context_ref=context_ref)
    return data.model_copy(
        update={
            "name": habit.name,
            "triggers": habit.triggers,
            "description": habit.description or None,
            "source": habit.source,
        }
    )


def _applied_habit_from_change(
    written: ChangeData,
    *,
    metadata: dict[str, object],
) -> CreateData | ChangeData:
    """Episode habit: CreateData when select was create, else ChangeData."""

    create_intent = metadata.get(_REFLECTION_CREATE_INTENT_METADATA_KEY)
    if isinstance(create_intent, CreateData):
        return CreateData(
            name=written.name,
            triggers=written.triggers,
            description=written.description,
            reason=written.reason or create_intent.reason,
            source=written.source,
        )
    return written


def _apply_break(data: BreakData, *, store: memory.Memory, context_ref: str | None) -> BreakData:
    """Set status=BROKEN in inventory; do not delete (change can revive it later)."""

    existing = _load_habit(store, name=data.name)
    if existing is None:
        raise ValueError(f"Reflection break selected unknown habit: {data.name}.")
    reason = data.reason.strip() if data.reason and data.reason.strip() else None
    retired = habit_storage.mark_broken(existing, reason=reason)
    _store_habit(store, habit=retired, context_ref=context_ref)
    return data


class Reflection(processing.Processing):
    """Post-output ability: HSM select → seed-or-load → changing | break-mark."""

    instructions: typing.ClassVar[str] = SELECT_INSTRUCTIONS
    select_instructions: typing.ClassVar[str] = SELECT_INSTRUCTIONS
    change_instructions: typing.ClassVar[str] = CHANGE_INSTRUCTIONS
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = type(None)
    input_event: typing.ClassVar[hsm.Event[processing.InputData]] = ability.ability_input_event(
        "bot.ability.reflection.input",
        processing.InputData,
        description="Post-output reflection input after cognition handled a turn.",
    )
    output_event: typing.ClassVar[hsm.Event[None]] = ability.ability_output_event(
        "bot.ability.reflection.output",
        type(None),
        description="Reflection completes with no product; habit work is applied as side effects only.",
    )
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _select_processing: processing.Processing
    _change_processing: processing.Processing
    _memory: memory.Memory
    _events: tuple[processing.Event[typing.Any], ...]

    @staticmethod
    def _child_id(event: hsm.Event[typing.Any], suffix: str) -> str:
        base = event.id if event.id else uuid.uuid4().hex
        return f"{base}{suffix}"

    @staticmethod
    def _parent_id_from_child(event: hsm.Event[typing.Any], suffix: str) -> str | None:
        child_id = event.id if event.id else None
        if child_id is None:
            return None
        index = child_id.find(suffix)
        if index < 0:
            return None
        parent = child_id[:index]
        return parent or None

    @staticmethod
    def _turn_from_metadata(metadata: dict[str, object]) -> InputData | None:
        turn = metadata.get(_REFLECTION_TURN_METADATA_KEY)
        return turn if isinstance(turn, InputData) else None

    @staticmethod
    def _prior_from_metadata(metadata: dict[str, object]) -> tuple[CognitiveEpisode, ...]:
        prior = metadata.get(_REFLECTION_PRIOR_METADATA_KEY)
        if isinstance(prior, tuple):
            return typing.cast(tuple[CognitiveEpisode, ...], prior)
        return ()

    @staticmethod
    def _has_reflection_input(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, processing.InputData)

    @staticmethod
    async def _start_operation(instance: "Reflection", operation_id: str) -> _ReflectionCapability:
        actor = _ReflectionOperation()
        private = hsm.Context(parent=instance.context(), values={hsm.Keys.Instances: {}})
        started = await hsm.started(private, actor, actor.model)
        capability = _ReflectionCapability(
            operation_id=operation_id,
            actor_id=hsm.id(started),
            token=uuid.uuid4().hex,
        )
        instances = instance.context().value(hsm.Keys.Instances)
        if isinstance(instances, collections.abc.MutableMapping):
            instances[capability.actor_id] = started
        return capability

    @staticmethod
    def _matches_operation(instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        capability = event.metadata.get(_REFLECTION_CAPABILITY_METADATA_KEY)
        if not isinstance(capability, _ReflectionCapability):
            return False
        instances = instance.context().value(hsm.Keys.Instances)
        actor = instances.get(capability.actor_id) if isinstance(instances, collections.abc.Mapping) else None
        return (
            isinstance(actor, _ReflectionOperation)
            and event.id == capability.operation_id
            and event.source == capability.actor_id
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _private_event(
        instance: "Reflection",
        source_event: hsm.Event[typing.Any],
        event_type: hsm.Event[typing.Any],
        data: object,
    ) -> hsm.Event[typing.Any]:
        capability = source_event.metadata.get(_REFLECTION_CAPABILITY_METADATA_KEY)
        assert isinstance(capability, _ReflectionCapability)
        return dataclasses.replace(
            event_type.with_data(data),
            id=capability.operation_id,
            source=capability.actor_id,
            target=hsm.id(instance),
            metadata=dict(source_event.metadata),
        )

    @staticmethod
    def _finish_operation(instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        capability = event.metadata.get(_REFLECTION_CAPABILITY_METADATA_KEY)
        cancel_capability = event.metadata.get(_REFLECTION_CANCEL_CAPABILITY_METADATA_KEY)
        actor_id = (
            capability.actor_id
            if isinstance(capability, _ReflectionCapability)
            else cancel_capability.turn_actor_id
            if isinstance(cancel_capability, _ReflectionCancelCapability)
            else None
        )
        instances = instance.context().value(hsm.Keys.Instances)
        if actor_id is not None and isinstance(instances, collections.abc.MutableMapping):
            _ = instances.pop(actor_id, None)

    @staticmethod
    def _is_reflection_cancel_request(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        return processing.Processing._is_cancel_request(ctx, instance, event)

    @staticmethod
    async def _resolve_select_cancel(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelData)
        instances = instance.context().value(hsm.Keys.Instances)
        turn_actor_id = (
            next(
                (hsm.id(actor) for actor in instances.values() if isinstance(actor, _ReflectionOperation)),
                None,
            )
            if isinstance(instances, collections.abc.Mapping)
            else None
        )
        capability = _ReflectionCancelCapability(
            operation_id=data.operation_id,
            token=data.token,
            phase="reflection-select",
            parent_operation=(
                event.metadata.get(dispatch.OPERATION_METADATA_KEY)
                if isinstance(event.metadata.get(dispatch.OPERATION_METADATA_KEY), dispatch.OperationData)
                else None
            ),
            turn_actor_id=turn_actor_id,
        )
        await dispatch.CancelResolution.begin(
            owner=instance,
            operation_id=data.operation_id,
            token=data.token,
            phase=capability.phase,
            metadata={
                **event.metadata,
                _REFLECTION_CANCEL_CAPABILITY_METADATA_KEY: capability,
            },
            teardown_timeout=_CANCEL_TEARDOWN_TIMEOUT,
        )

    @staticmethod
    async def _resolve_change_cancel(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelData)
        instances = instance.context().value(hsm.Keys.Instances)
        turn_actor_id = (
            next(
                (hsm.id(actor) for actor in instances.values() if isinstance(actor, _ReflectionOperation)),
                None,
            )
            if isinstance(instances, collections.abc.Mapping)
            else None
        )
        capability = _ReflectionCancelCapability(
            operation_id=data.operation_id,
            token=data.token,
            phase="reflection-change",
            parent_operation=(
                event.metadata.get(dispatch.OPERATION_METADATA_KEY)
                if isinstance(event.metadata.get(dispatch.OPERATION_METADATA_KEY), dispatch.OperationData)
                else None
            ),
            turn_actor_id=turn_actor_id,
        )
        await dispatch.CancelResolution.begin(
            owner=instance,
            operation_id=data.operation_id,
            token=data.token,
            phase=capability.phase,
            metadata={
                **event.metadata,
                _REFLECTION_CANCEL_CAPABILITY_METADATA_KEY: capability,
            },
            teardown_timeout=_CANCEL_TEARDOWN_TIMEOUT,
        )

    @staticmethod
    def _matches_cancel_unresolved(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        capability = event.metadata.get(_REFLECTION_CANCEL_CAPABILITY_METADATA_KEY)
        return (
            isinstance(data, dispatch.ResolveCancelData)
            and isinstance(capability, _ReflectionCancelCapability)
            and dispatch.matches_active_resolution(instance, event)
            and data.owner_id == hsm.id(instance)
            and data.operation_id == capability.operation_id
            and data.token == capability.token
            and data.phase == capability.phase
            and event.id == data.operation_id
            and event.source == data.resolver_id
            and event.target == hsm.id(instance)
            and event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY) == data
        )

    @staticmethod
    def _matches_cancelled(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        capability = event.metadata.get(_REFLECTION_CANCEL_CAPABILITY_METADATA_KEY)
        cancel = event.metadata.get(dispatch.CANCEL_METADATA_KEY)
        request = event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY)
        if (
            not isinstance(data, dispatch.TerminalData)
            or not isinstance(capability, _ReflectionCancelCapability)
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
            and operation.owner_id == hsm.id(instance)
            and operation.operation_id == capability.operation_id
            and operation.token == capability.token
            and operation.phase == capability.phase
            and cancel.owner_id == operation.owner_id
            and cancel.child_id == operation.child_id
            and cancel.request_id == operation.request_id
            and cancel.operation_id == operation.operation_id
            and cancel.token == operation.token
            and cancel.resolver_id == request.resolver_id
            and request.owner_id == operation.owner_id
            and request.operation_id == operation.operation_id
            and request.token == operation.token
            and request.phase == operation.phase
            and event.id == capability.operation_id
            and event.source == operation.actor_id
            and event.target == hsm.id(instance)
            and event.metadata.get(dispatch.CANCEL_METADATA_KEY) == cancel
            and event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY) == request
        )

    @staticmethod
    def _matches_cancel_teardown_failure(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        capability = event.metadata.get(_REFLECTION_CANCEL_CAPABILITY_METADATA_KEY)
        cancel = event.metadata.get(dispatch.CANCEL_METADATA_KEY)
        request = event.metadata.get(dispatch.RESOLVE_CANCEL_METADATA_KEY)
        if (
            not isinstance(data, dispatch.TerminalData)
            or not isinstance(capability, _ReflectionCancelCapability)
            or not isinstance(cancel, dispatch.CancelData)
            or not isinstance(request, dispatch.ResolveCancelData)
        ):
            return False
        operation = data.operation
        child = instance._select_processing if capability.phase == "reflection-select" else instance._change_processing
        return (
            data.outcome == "cancel_timeout"
            and data.failure is not None
            and data.terminal_name == child.failed_event.name
            and dispatch.matches_active_operation(instance, event)
            and dispatch.matches_active_resolution(instance, event)
            and operation.owner_id == hsm.id(instance)
            and operation.child_id == hsm.id(child)
            and operation.operation_id == capability.operation_id
            and operation.token == capability.token
            and operation.phase == capability.phase
            and cancel.owner_id == operation.owner_id
            and cancel.child_id == operation.child_id
            and cancel.request_id == operation.request_id
            and cancel.operation_id == operation.operation_id
            and cancel.token == operation.token
            and cancel.resolver_id == request.resolver_id
            and request.owner_id == operation.owner_id
            and request.operation_id == operation.operation_id
            and request.token == operation.token
            and request.phase == operation.phase
            and event.id == capability.operation_id
            and event.source == operation.actor_id
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _matches_cancel_resolved(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        capability = event.metadata.get(_REFLECTION_CANCEL_CAPABILITY_METADATA_KEY)
        if not isinstance(data, dispatch.CancelResolvedData) or not isinstance(capability, _ReflectionCancelCapability):
            return False
        operation = data.operation
        instances = instance.context().value(hsm.Keys.Instances)
        actor = instances.get(operation.actor_id) if isinstance(instances, collections.abc.Mapping) else None
        return (
            dispatch.matches_active_resolution(instance, event)
            and isinstance(actor, dispatch.Operation)
            and hsm.id(actor) == operation.actor_id
            and data.request.owner_id == hsm.id(instance)
            and data.request.operation_id == capability.operation_id
            and data.request.token == capability.token
            and data.request.phase == capability.phase
            and operation.owner_id == hsm.id(instance)
            and operation.operation_id == capability.operation_id
            and operation.token == capability.token
            and operation.phase == capability.phase
            and event.id == capability.operation_id
            and event.source == operation.actor_id
            and event.target == hsm.id(instance)
            and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == operation
        )

    @staticmethod
    def _emit_reflection_cancelled(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        Reflection._finish_operation(instance, event)
        capability = event.metadata.get(_REFLECTION_CANCEL_CAPABILITY_METADATA_KEY)
        if isinstance(capability, _ReflectionCancelCapability):
            operation_id = capability.operation_id
            token = capability.token
        else:
            data = event.data
            assert isinstance(data, processing.CancelData)
            operation_id = data.operation_id
            token = data.token
        owner = instance._attachments[0]
        metadata = dict(event.metadata)
        if isinstance(capability, _ReflectionCancelCapability):
            if capability.parent_operation is None:
                _ = metadata.pop(dispatch.OPERATION_METADATA_KEY, None)
            else:
                metadata[dispatch.OPERATION_METADATA_KEY] = capability.parent_operation
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
                metadata=metadata,
            ),
        )

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
        return isinstance(event.data, ability.FailureData) and Reflection._matches_operation(instance, event)

    @staticmethod
    def _dispatch_failure(
        ctx: hsm.Context,
        instance: "Reflection",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
        failure: ability.FailureData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(failure),
            id=operation_id,
            metadata={
                key: value
                for key, value in _public_metadata(metadata).items()
                if key != _REFLECTION_CAPABILITY_METADATA_KEY
            },
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _dispatch_output(
        ctx: hsm.Context,
        instance: "Reflection",
        *,
        operation_id: str | None,
        metadata: dict[str, object],
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(None),
            id=operation_id,
            metadata={
                key: value
                for key, value in _public_metadata(metadata).items()
                if key != _REFLECTION_CAPABILITY_METADATA_KEY
            },
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _fail_from_stage(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        Reflection._finish_operation(instance, event)
        Reflection._dispatch_failure(
            ctx,
            instance,
            operation_id=event.id or None,
            metadata=dict(event.metadata),
            failure=data,
        )

    @staticmethod
    def _fail_child(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, dispatch.TerminalData)
        Reflection._finish_operation(instance, event)
        failure = data.failure or ability.FailureData(message="Reflection child failed.")
        Reflection._dispatch_failure(
            ctx,
            instance,
            operation_id=data.operation.operation_id,
            metadata=dict(event.metadata),
            failure=failure,
        )

    @staticmethod
    async def _recall_activity(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, processing.InputData)
        input = data
        operation_id = event.id if event.id else uuid.uuid4().hex
        capability = await Reflection._start_operation(instance, operation_id)
        metadata = dict(event.metadata)
        metadata[_REFLECTION_CAPABILITY_METADATA_KEY] = capability
        correlated = dataclasses.replace(event, id=operation_id, metadata=metadata)
        try:
            turn = typing.cast(InputData, input.input)
            select_input = episodes.episode_select_input(context_ref=turn.cognition_input.focus)
            recalled = instance._memory.execute(select_input)
            prior = episodes.episodes_from_output(recalled)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    correlated,
                    _StageFailedEvent,
                    ability.FailureData(message=f"Reflection memory recall failed: {error}"),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                correlated,
                _RecalledEvent,
                _RecalledEventData(host_input=input, prior_episodes=prior),
            ),
        )

    @staticmethod
    async def _dispatch_select(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RecalledEventData)
        turn = typing.cast(InputData, data.host_input.input)
        source_metadata = dict(event.metadata)
        parent_operation = source_metadata.get(dispatch.OPERATION_METADATA_KEY)
        if isinstance(parent_operation, dispatch.OperationData):
            source_metadata[dispatch.CANCEL_TOKEN_METADATA_KEY] = parent_operation.token
        child_metadata = _public_metadata(source_metadata)
        child_metadata[_REFLECTION_TURN_METADATA_KEY] = turn
        child_metadata[_REFLECTION_PRIOR_METADATA_KEY] = data.prior_episodes
        child_metadata[_REFLECTION_OPERATION_ID_METADATA_KEY] = event.id if event.id else uuid.uuid4().hex
        try:
            habits = _load_all_habits(instance._memory)
        except Exception:
            habits = ()
        select_input = processing.InputData(
            input=SelectInput(
                cognition_input=turn.cognition_input,
                cognition_output=turn.cognition_output,
                prior_episodes=data.prior_episodes,
                habits=habits,
            ),
            schemas=habit_events(),
            actors={},
        )
        input_event = dataclasses.replace(
            instance._select_processing.input_event.with_data_and_id(
                select_input,
                Reflection._child_id(event, _SELECT_ID_SUFFIX),
            ),
            metadata=child_metadata,
        )
        operation_id = _operation_id_from_metadata(child_metadata)
        assert operation_id is not None
        await dispatch.Operation.begin(
            owner=instance,
            child=instance._select_processing,
            request=input_event,
            operation_id=operation_id,
            phase="reflection-select",
            timeout=_CHILD_OPERATION_TIMEOUT,
        )

    @staticmethod
    def _forward_child_terminal(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> None:
        dispatch.forward_terminal(ctx, instance, event)

    @staticmethod
    def _matches_select_output(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._select_processing
        data = event.data
        if not isinstance(data, dispatch.TerminalData):
            return False
        operation = data.operation
        return (
            dispatch.matches_active_operation(instance, event)
            and event.name == dispatch.TerminalEvent.name
            and event.target == hsm.id(instance)
            and event.source == operation.actor_id
            and operation.owner_id == hsm.id(instance)
            and operation.child_id == hsm.id(child)
            and operation.phase == "reflection-select"
            and operation.request_id == f"{operation.operation_id}{_SELECT_ID_SUFFIX}"
            and data.terminal_name == child.output_event.name
            and data.outcome == "output"
            and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == operation
        )

    @staticmethod
    def _matches_select_failure(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        if not isinstance(data, dispatch.TerminalData):
            return False
        operation = data.operation
        return (
            dispatch.matches_active_operation(instance, event)
            and event.target == hsm.id(instance)
            and event.source == operation.actor_id
            and operation.owner_id == hsm.id(instance)
            and operation.child_id == hsm.id(instance._select_processing)
            and operation.phase == "reflection-select"
            and operation.request_id == f"{operation.operation_id}{_SELECT_ID_SUFFIX}"
            and data.terminal_name == instance._select_processing.failed_event.name
            and data.outcome in {"failure", "timed_out"}
            and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == operation
        )

    @staticmethod
    def _matches_change_output(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        child = instance._change_processing
        data = event.data
        if not isinstance(data, dispatch.TerminalData):
            return False
        operation = data.operation
        return (
            dispatch.matches_active_operation(instance, event)
            and event.name == dispatch.TerminalEvent.name
            and event.target == hsm.id(instance)
            and event.source == operation.actor_id
            and operation.owner_id == hsm.id(instance)
            and operation.child_id == hsm.id(child)
            and operation.phase == "reflection-change"
            and operation.request_id.startswith(f"{operation.operation_id}{_CHANGE_ID_SUFFIX}:fix")
            and data.terminal_name == child.output_event.name
            and data.outcome == "output"
            and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == operation
        )

    @staticmethod
    def _matches_change_failure(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        if not isinstance(data, dispatch.TerminalData):
            return False
        operation = data.operation
        return (
            dispatch.matches_active_operation(instance, event)
            and event.target == hsm.id(instance)
            and event.source == operation.actor_id
            and operation.owner_id == hsm.id(instance)
            and operation.child_id == hsm.id(instance._change_processing)
            and operation.phase == "reflection-change"
            and operation.request_id.startswith(f"{operation.operation_id}{_CHANGE_ID_SUFFIX}:fix")
            and data.terminal_name == instance._change_processing.failed_event.name
            and data.outcome in {"failure", "timed_out"}
            and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == operation
        )

    @staticmethod
    def _on_select_output(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        """Route select terminal: empty → store; create/change/break → seed/write/break."""

        normalized = event.data
        assert isinstance(normalized, dispatch.TerminalData)
        metadata = dict(event.metadata)
        turn = Reflection._turn_from_metadata(metadata)
        operation_id = normalized.operation.operation_id
        if turn is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message="Reflection select output is missing turn correlation."),
                ),
            )
            return
        try:
            selection = _coerce_output_data(normalized.output)
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
                    _AppliedEventData(turn=turn, habit=None),
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
                    ability.FailureData(message="Reflection select must return at most one habit event."),
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
                    prior_episodes=Reflection._prior_from_metadata(metadata),
                    selection=chosen,
                    operation_id=operation_id,
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
    async def _start_change(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _ChangeRequestedEventData)
        child_metadata = dict(event.metadata)
        child_metadata[_REFLECTION_TURN_METADATA_KEY] = data.turn
        child_metadata[_REFLECTION_PRIOR_METADATA_KEY] = data.prior_episodes
        child_metadata[_REFLECTION_OPERATION_ID_METADATA_KEY] = data.operation_id
        child_metadata[_REFLECTION_CHANGE_INTENT_METADATA_KEY] = data.intent
        child_metadata[_REFLECTION_EXISTING_HABIT_METADATA_KEY] = data.existing
        if data.create_intent is not None:
            child_metadata[_REFLECTION_CREATE_INTENT_METADATA_KEY] = data.create_intent
        if _REFLECTION_FIX_ATTEMPTS_METADATA_KEY not in child_metadata:
            child_metadata[_REFLECTION_FIX_ATTEMPTS_METADATA_KEY] = 0
        write_input = processing.InputData(
            input=ChangeWriteInput(
                cognition_input=data.turn.cognition_input,
                cognition_output=data.turn.cognition_output,
                prior_episodes=data.prior_episodes,
                intent=data.intent,
                existing_habit=data.existing,
                diagnostics=data.diagnostics,
                failed_source=data.failed_source,
            ),
            schemas=(ChangeEvent,),
            actors={},
        )
        fix_attempts = _fix_attempts_from_metadata(child_metadata)
        input_event = dataclasses.replace(
            instance._change_processing.input_event.with_data_and_id(
                write_input,
                f"{data.operation_id}{_CHANGE_ID_SUFFIX}:fix{fix_attempts}",
            ),
            metadata=child_metadata,
        )
        operation = await dispatch.Operation.begin(
            owner=instance,
            child=instance._change_processing,
            request=input_event,
            operation_id=data.operation_id,
            phase="reflection-change",
            timeout=_CHILD_OPERATION_TIMEOUT,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                dataclasses.replace(
                    event,
                    metadata={**child_metadata, dispatch.OPERATION_METADATA_KEY: operation},
                ),
                _ChangeStartedEvent,
                operation,
            ),
        )

    @staticmethod
    def _queue_change(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
        data: _ChangeRequestedEventData,
        metadata: dict[str, object],
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                dataclasses.replace(event, metadata=metadata),
                _ChangeRequestedEvent,
                data,
            ),
        )

    @staticmethod
    def _has_change_requested(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _ChangeRequestedEventData) and Reflection._matches_operation(instance, event)

    @staticmethod
    def _has_change_started(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, dispatch.OperationData)
            and Reflection._matches_operation(instance, event)
            and event.metadata.get(dispatch.OPERATION_METADATA_KEY) == data
            and data.owner_id == hsm.id(instance)
            and data.child_id == hsm.id(instance._change_processing)
            and data.phase == "reflection-change"
        )

    @staticmethod
    def _on_create_event(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        """CreateEvent → seed DRAFT empty habit (if new), then enter changing."""

        selected = event.data
        assert isinstance(selected, _SelectedEventData)
        metadata = dict(event.metadata)
        turn = selected.turn
        prior = selected.prior_episodes
        try:
            raw = typing.cast(object, selected.selection.data)
            create_intent = (
                raw if isinstance(raw, CreateData) else CreateData.model_validate(raw if raw is not None else {})
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
        existing = _load_habit(instance._memory, name=create_intent.name)
        if existing is None:
            existing = _draft_stub_from_create(create_intent)
            _store_habit(instance._memory, habit=existing, context_ref=turn.cognition_input.focus)
        change_intent = ChangeData(
            name=create_intent.name,
            triggers=create_intent.triggers,
            description=create_intent.description,
            reason=create_intent.reason,
            source=None,
        )
        operation_id = selected.operation_id
        Reflection._queue_change(
            ctx,
            instance,
            event,
            _ChangeRequestedEventData(
                turn=turn,
                prior_episodes=prior,
                intent=change_intent,
                existing=existing,
                operation_id=operation_id,
                create_intent=create_intent,
            ),
            metadata,
        )

    @staticmethod
    def _on_change_event(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        """ChangeEvent → load existing (any status) and enter changing."""

        selected = event.data
        assert isinstance(selected, _SelectedEventData)
        metadata = dict(event.metadata)
        turn = selected.turn
        prior = selected.prior_episodes
        try:
            raw = typing.cast(object, selected.selection.data)
            intent = raw if isinstance(raw, ChangeData) else ChangeData.model_validate(raw if raw is not None else {})
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
        existing = _load_habit(instance._memory, name=intent.name)
        if existing is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=f"Reflection change selected unknown habit: {intent.name}."),
                ),
            )
            return
        operation_id = selected.operation_id
        Reflection._queue_change(
            ctx,
            instance,
            event,
            _ChangeRequestedEventData(
                turn=turn,
                prior_episodes=prior,
                intent=intent,
                existing=existing,
                operation_id=operation_id,
            ),
            metadata,
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
                _AppliedEventData(turn=turn, habit=applied),
            ),
        )

    @staticmethod
    def _change_write_accepted(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _ChangeWriteCheckedData)
            and data.habit_instance is not None
            and Reflection._matches_operation(instance, event)
        )

    @staticmethod
    def _change_write_retryable(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if (
            not isinstance(data, _ChangeWriteCheckedData)
            or data.habit_instance is not None
            or not Reflection._matches_operation(instance, event)
        ):
            return False
        messages = _diagnostic_messages(data.report)
        if not messages:
            return False
        return _fix_attempts_from_metadata(dict(event.metadata)) < _MAX_REFLECTION_FIX_ATTEMPTS and (
            data.previous_messages is None or data.previous_messages != messages
        )

    @staticmethod
    def _change_write_abandoned(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if (
            not isinstance(data, _ChangeWriteCheckedData)
            or data.habit_instance is not None
            or not Reflection._matches_operation(instance, event)
        ):
            return False
        messages = _diagnostic_messages(data.report)
        if not messages:
            return True
        return _fix_attempts_from_metadata(dict(event.metadata)) >= _MAX_REFLECTION_FIX_ATTEMPTS or (
            data.previous_messages is not None and data.previous_messages == messages
        )

    @staticmethod
    def _emit_change_write_checked(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        """Parse/validate change product; emit ``_ChangeWriteCheckedEvent`` only."""

        normalized = event.data
        assert isinstance(normalized, dispatch.TerminalData)
        metadata = dict(event.metadata)
        turn = Reflection._turn_from_metadata(metadata)
        prior = Reflection._prior_from_metadata(metadata)
        operation_id = normalized.operation.operation_id
        if turn is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message="Reflection change output is missing turn correlation."),
                ),
            )
            return
        try:
            written = _change_data_from_output(normalized.output)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=f"Reflection change failed: {error}"),
                ),
            )
            return
        existing = metadata.get(_REFLECTION_EXISTING_HABIT_METADATA_KEY)
        if not isinstance(existing, Instance):
            existing = _load_habit(instance._memory, name=written.name)
        checked = _habit_check_from_write(
            name=written.name,
            source=written.source,
            triggers=written.triggers,
            description=written.description,
            cognition_input=turn.cognition_input,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                event,
                _ChangeWriteCheckedEvent,
                _ChangeWriteCheckedData(
                    turn=turn,
                    prior_episodes=prior,
                    written=written,
                    habit_instance=checked.value if checked.ok else None,
                    report=checked.report,
                    previous_messages=_last_diagnostic_messages(metadata),
                    operation_id=operation_id,
                    existing=existing if isinstance(existing, Instance) else None,
                ),
            ),
        )

    @staticmethod
    def _persist_change_draft(
        instance: "Reflection",
        data: _ChangeWriteCheckedData,
    ) -> Instance | None:
        """Upsert latest change source as DRAFT when validation failed."""

        draft = _instance_for_inventory(
            name=data.written.name,
            source=data.written.source,
            triggers=data.written.triggers or (data.existing.triggers if data.existing else ()),
            description=data.written.description or (data.existing.description if data.existing else None),
            habit_instance=None,
            existing=data.existing,
        )
        if draft is None:
            return data.existing
        _store_habit(instance._memory, habit=draft, context_ref=data.turn.cognition_input.focus)
        return draft

    @staticmethod
    def _accept_change_write(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ChangeWriteCheckedData)
        assert data.habit_instance is not None
        stored = _instance_for_inventory(
            name=data.written.name,
            source=data.written.source,
            triggers=data.written.triggers,
            description=data.written.description,
            habit_instance=data.habit_instance,
            existing=data.existing,
        )
        assert stored is not None
        written = _store_change_result(
            data.written,
            stored,
            store=instance._memory,
            context_ref=data.turn.cognition_input.focus,
        )
        metadata = dict(event.metadata)
        applied = _applied_habit_from_change(written, metadata=metadata)
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                event,
                _AppliedEvent,
                _AppliedEventData(turn=data.turn, habit=applied),
            ),
        )

    @staticmethod
    def _retry_change_write(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ChangeWriteCheckedData)
        metadata = dict(event.metadata)
        intent = metadata.get(_REFLECTION_CHANGE_INTENT_METADATA_KEY)
        if not isinstance(intent, ChangeData):
            intent = data.written
        existing = Reflection._persist_change_draft(instance, data)
        if existing is None:
            existing = _load_habit(instance._memory, name=data.written.name)
        if existing is None:
            _ = hsm.dispatch(
                ctx,
                instance,
                Reflection._private_event(
                    instance,
                    event,
                    _StageFailedEvent,
                    ability.FailureData(message=f"Reflection change fix missing existing habit: {data.written.name}."),
                ),
            )
            return
        fix_metadata = dict(metadata)
        fix_metadata[_REFLECTION_FIX_ATTEMPTS_METADATA_KEY] = _fix_attempts_from_metadata(metadata) + 1
        fix_metadata[_REFLECTION_LAST_DIAGNOSTIC_MESSAGES_KEY] = _diagnostic_messages(data.report)
        create_intent = metadata.get(_REFLECTION_CREATE_INTENT_METADATA_KEY)
        Reflection._queue_change(
            ctx,
            instance,
            event,
            _ChangeRequestedEventData(
                turn=data.turn,
                prior_episodes=data.prior_episodes,
                intent=intent,
                existing=existing,
                operation_id=data.operation_id,
                diagnostics=data.report,
                failed_source=data.written.source,
                create_intent=create_intent if isinstance(create_intent, CreateData) else None,
            ),
            fix_metadata,
        )

    @staticmethod
    def _abandon_change_write(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _ChangeWriteCheckedData)
        _ = Reflection._persist_change_draft(instance, data)
        _ = hsm.dispatch(
            ctx,
            instance,
            Reflection._private_event(
                instance,
                event,
                _StageFailedEvent,
                ability.FailureData(
                    message=(
                        "Reflection change abandoned: same diagnostic message after fix or retry budget exhausted:\n"
                        f"{data.report.render()}"
                    )
                ),
            ),
        )

    @staticmethod
    async def _store_activity(
        ctx: hsm.Context,
        instance: "Reflection",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _AppliedEventData)
        try:
            episode = episode_from_turn(data.turn, habit=data.habit)
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
                _StoredEventData(),
            ),
        )

    @staticmethod
    def _complete_from_stored(ctx: hsm.Context, instance: "Reflection", event: hsm.Event[typing.Any]) -> None:
        metadata = dict(event.metadata)
        Reflection._finish_operation(instance, event)
        Reflection._dispatch_output(
            ctx,
            instance,
            operation_id=_operation_id_from_metadata(metadata) or (event.id if event.id else None),
            metadata=metadata,
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
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
                hsm.target("/Reflection/resolving_select_cancel"),
            ),
            hsm.transition(
                hsm.on(processing.OutputEvent),
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
                hsm.guard(_matches_select_empty),
                hsm.effect(_on_select_output, dispatch.retire_operation),
            ),
            hsm.transition(
                hsm.on(_SelectedEvent),
                hsm.guard(_selected_is_create),
                hsm.effect(_on_create_event),
            ),
            hsm.transition(
                hsm.on(_SelectedEvent),
                hsm.guard(_selected_is_change),
                hsm.effect(_on_change_event),
            ),
            hsm.transition(
                hsm.on(_SelectedEvent),
                hsm.guard(_selected_is_break),
                hsm.effect(_on_break_event),
            ),
            hsm.transition(
                hsm.on(_ChangeRequestedEvent),
                hsm.guard(_has_change_requested),
                hsm.target("/Reflection/starting_change"),
            ),
            hsm.transition(
                hsm.on(_AppliedEvent),
                hsm.guard(_has_reflection_applied),
                hsm.target("/Reflection/storing"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_select_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
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
            "starting_change",
            hsm.defer(input_event),
            hsm.activity(_start_change),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_reflection_cancel_request),
                hsm.target("/Reflection/resolving_change_cancel"),
            ),
            hsm.transition(
                hsm.on(_ChangeStartedEvent),
                hsm.guard(_has_change_started),
                hsm.target("/Reflection/changing"),
            ),
            hsm.transition(
                hsm.on(_StageFailedEvent),
                hsm.guard(_has_stage_failure),
                hsm.effect(_fail_from_stage),
                hsm.target("/Reflection/idle"),
            ),
        ),
        hsm.state(
            "changing",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(_is_reflection_cancel_request),
                hsm.target("/Reflection/resolving_change_cancel"),
            ),
            hsm.transition(
                hsm.on(processing.OutputEvent),
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
                hsm.guard(_matches_change_output),
                hsm.effect(_emit_change_write_checked, dispatch.retire_operation),
            ),
            hsm.transition(
                hsm.on(_ChangeWriteCheckedEvent),
                hsm.guard(_change_write_accepted),
                hsm.effect(_accept_change_write),
            ),
            hsm.transition(
                hsm.on(_ChangeWriteCheckedEvent),
                hsm.guard(_change_write_retryable),
                hsm.effect(_retry_change_write),
            ),
            hsm.transition(
                hsm.on(_ChangeRequestedEvent),
                hsm.guard(_has_change_requested),
                hsm.target("/Reflection/starting_change"),
            ),
            hsm.transition(
                hsm.on(_ChangeWriteCheckedEvent),
                hsm.guard(_change_write_abandoned),
                hsm.effect(_abandon_change_write),
            ),
            hsm.transition(
                hsm.on(_AppliedEvent),
                hsm.guard(_has_reflection_applied),
                hsm.target("/Reflection/storing"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_change_failure),
                hsm.effect(_fail_child, dispatch.retire_operation),
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
            "resolving_select_cancel",
            hsm.defer(input_event),
            hsm.activity(_resolve_select_cancel),
            hsm.transition(
                hsm.on(dispatch.CancelResolvedEvent),
                hsm.guard(_matches_cancel_resolved),
                hsm.effect(dispatch.complete_resolution, dispatch.cancel_resolved_operation),
                hsm.target("/Reflection/cancelling"),
            ),
            hsm.transition(
                hsm.on(dispatch.CancelUnresolvedEvent),
                hsm.guard(_matches_cancel_unresolved),
                hsm.effect(dispatch.retire_resolution, _emit_reflection_cancelled),
                hsm.target("/Reflection/idle"),
            ),
        ),
        hsm.state(
            "resolving_change_cancel",
            hsm.defer(input_event),
            hsm.activity(_resolve_change_cancel),
            hsm.transition(
                hsm.on(dispatch.CancelResolvedEvent),
                hsm.guard(_matches_cancel_resolved),
                hsm.effect(dispatch.complete_resolution, dispatch.cancel_resolved_operation),
                hsm.target("/Reflection/cancelling"),
            ),
            hsm.transition(
                hsm.on(dispatch.CancelUnresolvedEvent),
                hsm.guard(_matches_cancel_unresolved),
                hsm.effect(dispatch.retire_resolution, _emit_reflection_cancelled),
                hsm.target("/Reflection/idle"),
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
                hsm.guard(_matches_cancelled),
                hsm.effect(dispatch.retire_resolution, dispatch.retire_operation, _emit_reflection_cancelled),
                hsm.target("/Reflection/idle"),
            ),
            hsm.transition(
                hsm.on(dispatch.TerminalEvent),
                hsm.guard(_matches_cancel_teardown_failure),
                hsm.effect(dispatch.retire_resolution, dispatch.retire_operation, _fail_child),
                hsm.target("/Reflection/degraded"),
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
                hsm.target("/Reflection/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Reflection/idle"),
            ),
        ),
        hsm.state("degraded"),
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
        self._change_processing = processing.Processing(
            processor=leaf,
            instructions=type(self).change_instructions,
        )
        self._memory = memory
        self._attachment_group: attachment.Group = attachment.Group(
            self._select_processing,
            self._change_processing,
            self._memory,
        )
        self._events = habit_events()


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
    "habit_events",
    "stimulus_name",
]
