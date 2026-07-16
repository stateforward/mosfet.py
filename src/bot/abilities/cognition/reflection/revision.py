"""Typed habit revision actor owned by post-output reflection."""

from __future__ import annotations

from ... import ability
from ... import memory
from ... import processing

import dataclasses
import datetime
import typing
import uuid

import bot
import hsm
from pydantic.json_schema import SkipJsonSchema
import pydantic
from sqlalchemy.sql import Executable

from bot.habit import ChangeData
from bot.habit import ChangeEvent
from bot.habit import CreateData
from bot.habit import diagnostic as habit_diagnostic
from bot.habit import storage as habit_storage
from bot.habit.instance import STATUS_REASON_VALIDATION
from bot.habit.instance import Instance
from bot.habit.source import STARLARK_API
from bot.protocols import attachment
from bot.telemetry import observer

from .. import episodes
from .. import input
from .. import types

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
    "At runtime, fill selection data from the live event data — "
    "never hardcode identifier values copied from episode examples. "
    "Set triggers to the stimulus names that should propose the habit. "
    "Effects must hsm.dispatch(output_event, selection) and must not return a value. "
    "Starlark only: no Python docstrings, type annotations, or imports; callbacks are def name(event): ...\n"
    "If diagnostics is present, prior source failed validation: revise `source` to clear every error "
    "(use code, stage, message, and help). failed_source is the rejected program when provided. "
    "Repeating the same diagnostic message after a fix ends change; change the source so messages clear.\n\n"
    f"{STARLARK_API}"
)

_CHANGE_ID_SUFFIX = ":change"
_CANCEL_TEARDOWN_TIMEOUT = datetime.timedelta(seconds=5)
_MAX_FIX_ATTEMPTS = 2


class ProcessorFactory(typing.Protocol):
    """Build the provider-neutral processor used to author habit source."""

    def __call__(self) -> processing.Processor: ...


class InputData(pydantic.BaseModel):
    """One create or change intent with immutable parent-turn correlation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "One create or change request delegated by Reflection with exact parent-turn correlation."
        },
    )

    cognition_input: SkipJsonSchema[input.InputData] = pydantic.Field(
        description="Original cognition input whose handled turn triggered reflection."
    )
    cognition_output: types.OutputData = pydantic.Field(
        description="Already-handled cognition output associated with the reflected turn."
    )
    prior_episodes: tuple[episodes.CognitiveEpisode, ...] = pydantic.Field(
        default=(), description="Prior similar episodes available as authoring evidence."
    )
    intent: CreateData | ChangeData = pydantic.Field(
        description="Selected create or change intent that Revision must author and validate."
    )
    parent_operation_id: str = pydantic.Field(
        min_length=1,
        description="Reflection operation ID that must receive this revision terminal.",
        examples=["turn-42"],
    )
    parent_generation: str = pydantic.Field(
        min_length=1,
        description="Live Reflection operation actor ID used to reject stale terminals.",
        examples=["processing-operation-actor"],
    )


class OutputData(pydantic.BaseModel):
    """Applied habit revision and the turn needed by Reflection to store its episode."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
    )

    input: InputData = pydantic.Field(description="Immutable revision input and parent-turn correlation.")
    applied: CreateData | ChangeData = pydantic.Field(
        description="Validated create or change payload persisted to habit inventory."
    )


class FailureData(pydantic.BaseModel):
    """Revision failure with the immutable input that identifies its parent turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
    )

    input: InputData = pydantic.Field(description="Immutable revision input and parent-turn correlation.")
    message: str = pydantic.Field(
        min_length=1,
        description="Actionable revision failure suitable for the parent ability terminal.",
        examples=["Reflection change abandoned: retry budget exhausted."],
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
    prior_episodes: tuple[episodes.CognitiveEpisode, ...] = pydantic.Field(
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
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)
    attempt: int = pydantic.Field(ge=0, le=_MAX_FIX_ATTEMPTS)
    create_intent: CreateData | None = None
    previous_messages: tuple[str, ...] | None = None


class _RequestedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    write: ChangeWriteInput
    generation: str = pydantic.Field(min_length=1)


class _CheckedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    write: ChangeWriteInput
    generation: str = pydantic.Field(min_length=1)
    written: ChangeData
    habit_instance: Instance | None = None
    report: habit_diagnostic.Report = pydantic.Field(default_factory=habit_diagnostic.Report)
    existing: Instance | None = None


class _StartedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)
    attempt: int = pydantic.Field(ge=0, le=_MAX_FIX_ATTEMPTS)


class _FailedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input: InputData
    operation_id: str = pydantic.Field(min_length=1)
    generation: str = pydantic.Field(min_length=1)
    message: str = pydantic.Field(min_length=1)


class _CancelRequestedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    child_operation_id: str = pydantic.Field(min_length=1)
    parent_operation_id: str = pydantic.Field(min_length=1)
    token: str = pydantic.Field(min_length=1)


_RequestedEvent = hsm.Event[_RequestedData](
    name="bot.ability.reflection.revision.requested",
    schema=_RequestedData,
)
_CheckedEvent = hsm.Event[_CheckedData](
    name="bot.ability.reflection.revision.checked",
    kind=hsm.CompletionEventKind,
    schema=_CheckedData,
)
_StartedEvent = hsm.Event[_StartedData](
    name="bot.ability.reflection.revision.started",
    kind=hsm.CompletionEventKind,
    schema=_StartedData,
)
_FailedEvent = hsm.Event[_FailedData](
    name="bot.ability.reflection.revision.failed",
    kind=hsm.ErrorEventKind,
    schema=_FailedData,
)
_CancelRequestedEvent = hsm.Event[_CancelRequestedData](
    name="bot.ability.reflection.revision.cancel.requested",
    schema=_CancelRequestedData,
)
_CancelStartedEvent = hsm.Event[_CancelRequestedData](
    name="bot.ability.reflection.revision.cancel.started",
    kind=hsm.CompletionEventKind,
    schema=_CancelRequestedData,
)


def _compile_statements(clauses: tuple[object, ...]) -> tuple[memory.Statement, ...]:
    return memory.compile_statements(*(typing.cast(Executable, clause) for clause in clauses))


def _load_habit(store: memory.Memory, name: str) -> Instance | None:
    output = store.execute(
        memory.InputData(statements=_compile_statements(habit_storage.select_habit_by_name_clauses(name)))
    )
    if len(output.results) < 2:
        return None
    habits = habit_storage.instances_from_habit_results(
        tuple(row.as_mapping() for row in output.results[0].rows),
        tuple(row.as_mapping() for row in output.results[1].rows),
    )
    return habits[0] if habits else None


def _store_habit(store: memory.Memory, habit: Instance) -> None:
    _ = store.execute(memory.InputData(statements=_compile_statements(habit_storage.replace_habit_clauses(habit))))


def _draft(intent: CreateData) -> Instance:
    return habit_storage.mark_draft(
        Instance(
            name=intent.name,
            source="",
            triggers=intent.triggers,
            description=intent.description or "",
        )
    )


def _coerce_change(value: object) -> ChangeData:
    if isinstance(value, ChangeData):
        return value
    selections = processing.coerce_event_selections(value)
    if selections is not None:
        if len(selections) != 1 or selections[0].event != ChangeEvent.name:
            raise TypeError("Revision must return exactly one bot.habit.change event.")
        return ChangeData.model_validate(selections[0].data or {})
    return ChangeData.model_validate(value)


def _check(write: ChangeWriteInput, written: ChangeData) -> habit_diagnostic.Checked[Instance]:
    from bot.habit.verify import verify_apply
    from .. import autonomy

    if written.source is None or not written.source.strip():
        report = habit_diagnostic.report_of(
            habit_diagnostic.diagnostic(
                code=habit_diagnostic.E0001_EMPTY,
                message="Executable habit write requires non-empty starlark source.",
                stage=habit_diagnostic.Stage.INVENTORY,
                help=habit_diagnostic.help_for(habit_diagnostic.E0001_EMPTY),
            )
        )
        return habit_diagnostic.Checked[Instance](value=None, report=report)
    return verify_apply(
        written.source,
        name=written.name,
        triggers=written.triggers if written.triggers else None,
        description=written.description,
        input_data=autonomy.habit_input_payload(write.cognition_input),
        metadata={},
    )


def _inventory_instance(
    written: ChangeData,
    checked: Instance | None,
    existing: Instance | None,
) -> Instance | None:
    if checked is not None:
        return habit_storage.preserve_usage(habit_storage.mark_active(checked), existing)
    if written.source is None or not written.source.strip():
        return None
    triggers = written.triggers or (existing.triggers if existing is not None else ())
    description = written.description or (existing.description if existing is not None else "")
    draft = habit_storage.mark_draft(
        Instance(
            name=written.name,
            source=written.source.strip(),
            triggers=triggers,
            description=description,
        ),
        reason=STATUS_REASON_VALIDATION,
    )
    return habit_storage.preserve_usage(draft, existing)


def _revision_input(write: ChangeWriteInput) -> InputData:
    return InputData(
        cognition_input=write.cognition_input,
        cognition_output=write.cognition_output,
        prior_episodes=write.prior_episodes,
        intent=write.create_intent if write.create_intent is not None else write.intent,
        parent_operation_id=write.operation_id,
        parent_generation=write.generation,
    )


class Revision(processing.Processing):
    """Create or change one habit through typed author, validate, retry, and persist states."""

    instructions: typing.ClassVar[str] = CHANGE_INSTRUCTIONS
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = OutputData
    input_event: typing.ClassVar[hsm.Event[InputData]] = ability.ability_input_event(
        "bot.ability.reflection.revision.input",
        InputData,
        description="Create or revise one installed habit from a reflected cognition turn.",
    )
    output_event: typing.ClassVar[hsm.Event[OutputData]] = ability.ability_output_event(
        "bot.ability.reflection.revision.output",
        OutputData,
        description="A validated habit revision persisted to inventory.",
    )
    failed_event: typing.ClassVar[hsm.Event[FailureData]] = hsm.Event[FailureData](
        name=ability.FailedEvent.name,
        kind=hsm.ErrorEventKind,
        schema=FailureData,
    )
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _change_processing: processing.Processing
    _memory: memory.Memory

    @staticmethod
    def _operation(event: hsm.Event[typing.Any]) -> tuple[str, str] | None:
        data = event.data
        if isinstance(data, _RequestedData):
            return data.write.operation_id, data.generation
        if isinstance(data, _CheckedData):
            return data.write.operation_id, data.generation
        if isinstance(data, _StartedData):
            return data.operation_id, data.generation
        if isinstance(data, _FailedData):
            return data.operation_id, data.generation
        return None

    @staticmethod
    def _matches_operation(instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        operation = Revision._operation(event)
        return operation is not None and processing.matches_operation(instance, *operation)

    @staticmethod
    def _matches_private_operation(instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        operation = Revision._operation(event)
        return (
            operation is not None
            and processing.matches_operation(instance, *operation)
            and event.id == operation[0]
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _child_id(operation_id: str, attempt: int, generation: str) -> str:
        return f"{operation_id}{_CHANGE_ID_SUFFIX}:fix{attempt}:{generation}"

    @staticmethod
    def _active_child_id(instance: "Revision") -> str | None:
        operation_id = processing.active_operation_id(instance)
        state = instance.state()
        if operation_id is None or "/authoring/attempt_" not in state:
            return None
        operation = processing.active_operation(instance, operation_id)
        if operation is None:
            return None
        attempt = int(state.rsplit("_", 1)[1])
        return Revision._child_id(operation_id, attempt, hsm.id(operation))

    @staticmethod
    def _private_event(
        instance: "Revision",
        source: hsm.Event[typing.Any],
        event_type: hsm.Event[typing.Any],
        data: object,
        operation: tuple[str, str] | None = None,
    ) -> hsm.Event[typing.Any]:
        operation = operation if operation is not None else Revision._operation(source)
        assert operation is not None
        operation_id, _ = operation
        return dataclasses.replace(
            event_type.with_data(data),
            id=operation_id,
            source=hsm.id(instance),
            target=hsm.id(instance),
            metadata=dict(source.metadata),
        )

    @staticmethod
    def _has_revision_input(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            isinstance(event.data, InputData)
            and bool(event.id)
            and event.id == event.data.parent_operation_id
            and event.target in ("", hsm.id(instance))
            and bool(instance._attachments)
            and event.source in ("", hsm.id(instance._attachments[0]))
        )

    @staticmethod
    async def _prepare(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, InputData)
        operation_id = data.parent_operation_id
        operation = await processing.start_operation(instance, operation_id)
        try:
            if isinstance(data.intent, CreateData):
                create_intent = data.intent
                existing = _load_habit(instance._memory, create_intent.name)
                if existing is None:
                    existing = _draft(create_intent)
                    _store_habit(instance._memory, existing)
                intent = ChangeData(
                    name=create_intent.name,
                    triggers=create_intent.triggers,
                    description=create_intent.description,
                    reason=create_intent.reason,
                )
            else:
                create_intent = None
                intent = data.intent
                existing = _load_habit(instance._memory, intent.name)
                if existing is None:
                    raise ValueError(f"Revision selected unknown habit: {intent.name}.")
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _FailedEvent.with_data(
                        _FailedData(
                            input=data,
                            operation_id=operation_id,
                            generation=hsm.id(operation),
                            message=str(error),
                        )
                    ),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )
            return
        write = ChangeWriteInput(
            cognition_input=data.cognition_input,
            cognition_output=data.cognition_output,
            prior_episodes=data.prior_episodes,
            intent=intent,
            existing_habit=existing,
            operation_id=operation_id,
            generation=data.parent_generation,
            attempt=0,
            create_intent=create_intent,
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _RequestedEvent.with_data(_RequestedData(write=write, generation=hsm.id(operation))),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _has_requested(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _RequestedData) and Revision._matches_private_operation(instance, event)

    @staticmethod
    async def _dispatch_change(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _RequestedData)
        write = data.write
        request = processing.InputData(
            input=write,
            schemas=(ChangeEvent,),
            actors={},
        )
        await hsm.dispatch(
            ctx,
            instance._change_processing,
            dataclasses.replace(
                instance._change_processing.input_event.with_data_and_id(
                    request,
                    Revision._child_id(write.operation_id, write.attempt, data.generation),
                ),
                metadata=dict(event.metadata),
            ),
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            Revision._private_event(
                instance,
                event,
                _StartedEvent,
                _StartedData(
                    operation_id=write.operation_id,
                    generation=data.generation,
                    attempt=write.attempt,
                ),
            ),
        )

    @staticmethod
    def _has_started(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _StartedData) and Revision._matches_private_operation(instance, event)

    @staticmethod
    def _attempt_is(
        ctx: hsm.Context,
        instance: "Revision",
        event: hsm.Event[typing.Any],
        attempt: int,
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, _StartedData) and event.data.attempt == attempt

    @staticmethod
    def _attempt_is_0(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        return Revision._attempt_is(ctx, instance, event, 0)

    @staticmethod
    def _attempt_is_1(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        return Revision._attempt_is(ctx, instance, event, 1)

    @staticmethod
    def _matches_change_output(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        write = data.input.input if isinstance(data, processing.CompletionData) else None
        return (
            isinstance(write, ChangeWriteInput)
            and processing.active_operation(instance, write.operation_id) is not None
            and event.source == hsm.id(instance._change_processing)
            and event.target == hsm.id(instance)
            and event.id == Revision._active_child_id(instance)
        )

    @staticmethod
    def _matches_change_failure(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        write = data.input.input if isinstance(data, processing.FailureData) else None
        return (
            isinstance(write, ChangeWriteInput)
            and processing.active_operation(instance, write.operation_id) is not None
            and event.source == hsm.id(instance._change_processing)
            and event.target == hsm.id(instance)
            and event.id == Revision._active_child_id(instance)
        )

    @staticmethod
    def _check_change(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        completion = event.data
        assert isinstance(completion, processing.CompletionData)
        write = completion.input.input
        assert isinstance(write, ChangeWriteInput)
        operation = processing.active_operation(instance, write.operation_id)
        if operation is None:
            return
        generation = hsm.id(operation)
        try:
            written = _coerce_change(completion.output)
            existing = _load_habit(instance._memory, write.existing_habit.name)
            checked = _check(write, written)
            inventory = _inventory_instance(written, checked.value, existing)
            data = _CheckedData(
                write=write,
                generation=generation,
                written=written,
                habit_instance=inventory,
                report=checked.report,
                existing=existing,
            )
            terminal = Revision._private_event(
                instance,
                event,
                _CheckedEvent,
                data,
                operation=(write.operation_id, generation),
            )
        except Exception as error:
            terminal = Revision._private_event(
                instance,
                event,
                _FailedEvent,
                _FailedData(
                    input=_revision_input(write),
                    operation_id=write.operation_id,
                    generation=generation,
                    message=str(error),
                ),
                operation=(write.operation_id, generation),
            )
        _ = hsm.dispatch(ctx, instance, terminal)

    @staticmethod
    def _accepted(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return (
            isinstance(event.data, _CheckedData)
            and Revision._matches_private_operation(instance, event)
            and event.data.report.ok
            and event.data.habit_instance is not None
        )

    @staticmethod
    def _retryable(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if not isinstance(event.data, _CheckedData) or not Revision._matches_private_operation(instance, event):
            return False
        data = event.data
        messages = tuple(item.message for item in data.report.errors)
        return (
            not data.report.ok and data.write.attempt < _MAX_FIX_ATTEMPTS and messages != data.write.previous_messages
        )

    @staticmethod
    def _abandoned(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if not isinstance(event.data, _CheckedData) or not Revision._matches_private_operation(instance, event):
            return False
        data = event.data
        messages = tuple(item.message for item in data.report.errors)
        return not data.report.ok and (
            data.write.attempt >= _MAX_FIX_ATTEMPTS or messages == data.write.previous_messages
        )

    @staticmethod
    def _persist_draft(instance: "Revision", data: _CheckedData) -> Instance:
        habit = data.habit_instance
        if habit is None:
            habit = habit_storage.mark_draft(data.write.existing_habit, reason=STATUS_REASON_VALIDATION)
        _store_habit(instance._memory, habit)
        return habit

    @staticmethod
    def _accept(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _CheckedData)
        assert data.habit_instance is not None
        _store_habit(instance._memory, data.habit_instance)
        written = data.written.model_copy(
            update={
                "name": data.habit_instance.name,
                "triggers": data.habit_instance.triggers,
                "description": data.habit_instance.description or None,
                "source": data.habit_instance.source,
            }
        )
        applied: CreateData | ChangeData
        if data.write.create_intent is not None:
            applied = CreateData(
                name=written.name,
                triggers=written.triggers,
                description=written.description,
                reason=written.reason or data.write.create_intent.reason,
                source=written.source,
            )
        else:
            applied = written
        terminal = dataclasses.replace(
            instance.output_event.with_data(OutputData(input=_revision_input(data.write), applied=applied)),
            id=data.write.operation_id,
            source=hsm.id(instance),
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
        processing.finish_operation(ctx, instance, data.write.operation_id)

    @staticmethod
    def _retry(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _CheckedData)
        existing = Revision._persist_draft(instance, data)
        messages = tuple(item.message for item in data.report.errors)
        write = data.write.model_copy(
            update={
                "existing_habit": existing,
                "diagnostics": data.report,
                "failed_source": data.written.source,
                "attempt": data.write.attempt + 1,
                "previous_messages": messages,
            }
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            Revision._private_event(
                instance,
                event,
                _RequestedEvent,
                _RequestedData(write=write, generation=data.generation),
            ),
        )

    @staticmethod
    def _fail(
        ctx: hsm.Context,
        instance: "Revision",
        input_data: InputData,
        operation_id: str,
        message: str,
        metadata: dict[str, object],
    ) -> None:
        terminal = dataclasses.replace(
            instance.failed_event.with_data(FailureData(input=input_data, message=message)),
            id=operation_id,
            source=hsm.id(instance),
            metadata=metadata,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
        processing.finish_operation(ctx, instance, operation_id)

    @staticmethod
    def _abandon(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _CheckedData)
        _ = Revision._persist_draft(instance, data)
        messages = tuple(item.message for item in data.report.errors)
        reason = (
            "same diagnostic message after fix"
            if data.write.previous_messages is not None and messages == data.write.previous_messages
            else "retry budget exhausted"
        )
        Revision._fail(
            ctx,
            instance,
            _revision_input(data.write),
            data.write.operation_id,
            f"Reflection change abandoned: {reason}:\n{data.report.render()}",
            dict(event.metadata),
        )

    @staticmethod
    def _fail_internal(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _FailedData)
        Revision._fail(ctx, instance, data.input, data.operation_id, data.message, dict(event.metadata))

    @staticmethod
    def _has_internal_failure(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _FailedData) and Revision._matches_private_operation(instance, event)

    @staticmethod
    def _fail_child(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.FailureData)
        write = data.input.input
        assert isinstance(write, ChangeWriteInput)
        Revision._fail(ctx, instance, _revision_input(write), write.operation_id, data.message, dict(event.metadata))

    @staticmethod
    def _queue_cancel(
        ctx: hsm.Context,
        instance: "Revision",
        event: hsm.Event[typing.Any],
    ) -> None:
        operation_id = processing.active_operation_id(instance)
        if operation_id is None:
            return
        child_id = Revision._active_child_id(instance)
        if child_id is None:
            return
        cancel = event.data if isinstance(event.data, processing.CancelData) else None
        token = uuid.uuid4().hex if cancel is None else cancel.token
        parent_operation_id = operation_id if cancel is None else cancel.parent_operation_id or cancel.operation_id
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CancelRequestedEvent.with_data(
                    _CancelRequestedData(
                        operation_id=operation_id,
                        child_operation_id=child_id,
                        parent_operation_id=parent_operation_id,
                        token=token,
                    )
                ),
                id=operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _queue_requested_cancel(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        Revision._queue_cancel(ctx, instance, event)

    @staticmethod
    def _has_cancel_requested(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _CancelRequestedData)
            and event.id == data.operation_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and processing.active_operation(instance, data.operation_id) is not None
        )

    @staticmethod
    async def _start_cancel(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _CancelRequestedData)
        cancellation_id = processing.cancellation_operation_id(
            data.operation_id,
            data.child_operation_id,
            data.token,
            hsm.id(instance._change_processing),
        )
        _ = await processing.start_operation(instance, cancellation_id)
        await hsm.dispatch(
            ctx,
            instance._change_processing,
            dataclasses.replace(
                processing.CancelEvent.with_data(
                    processing.CancelData(
                        operation_id=data.child_operation_id,
                        token=data.token,
                        parent_operation_id=data.parent_operation_id,
                    )
                ),
                id=data.child_operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance._change_processing),
                metadata=dict(event.metadata),
            ),
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _CancelStartedEvent.with_data(data),
                id=data.operation_id,
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _has_cancel_started(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        return Revision._has_cancel_requested(ctx, instance, event)

    @staticmethod
    def _matches_cancelled(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        revision_operation_id = data.parent_operation_id if isinstance(data, processing.CancelledData) else None
        cancellation_id = (
            processing.cancellation_operation_id(
                revision_operation_id,
                event.id,
                data.token,
                hsm.id(instance._change_processing),
            )
            if isinstance(data, processing.CancelledData) and revision_operation_id is not None
            else ""
        )
        return (
            isinstance(data, processing.CancelledData)
            and event.source == hsm.id(instance._change_processing)
            and event.target == hsm.id(instance)
            and event.id == data.operation_id
            and revision_operation_id is not None
            and processing.active_operation(instance, revision_operation_id) is not None
            and processing.active_operation(instance, cancellation_id) is not None
        )

    @staticmethod
    def _cancelled_is_requested(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> bool:
        return Revision._matches_cancelled(ctx, instance, event)

    @staticmethod
    def _finish_cancellation(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> str:
        data = event.data
        assert isinstance(data, processing.CancelledData)
        revision_operation_id = data.parent_operation_id
        assert revision_operation_id is not None
        cancellation_id = processing.cancellation_operation_id(
            revision_operation_id,
            event.id,
            data.token,
            hsm.id(instance._change_processing),
        )
        processing.finish_operation(ctx, instance, cancellation_id)
        return revision_operation_id

    @staticmethod
    def _emit_revision_cancelled(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, processing.CancelledData)
        owner = instance._attachments[0]
        revision_operation_id = Revision._finish_cancellation(ctx, instance, event)
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                processing.CancelledEvent.with_data(
                    processing.CancelledData(
                        operation_id=revision_operation_id,
                        token=data.token,
                        parent_operation_id=data.parent_operation_id,
                    )
                ),
                id=revision_operation_id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )
        processing.finish_operation(ctx, instance, revision_operation_id)

    @staticmethod
    def _cancel_timeout_delay(
        ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _CANCEL_TEARDOWN_TIMEOUT

    @staticmethod
    def _request_reboot(ctx: hsm.Context, instance: "Revision", event: hsm.Event[typing.Any]) -> None:
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                bot.RebootEvent.with_data(bot.RebootEventData(reason="cognition_child_teardown_failed")),
                id=event.id or uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Revision",
        hsm.initial(hsm.target("/Revision/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Revision/idle"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
            ),
            hsm.transition(hsm.on(input_event), hsm.guard(_has_revision_input), hsm.target("/Revision/preparing")),
        ),
        hsm.state(
            "preparing",
            hsm.defer(input_event),
            hsm.activity(_prepare),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(processing.Processing._emit_cancelled),
                hsm.target("/Revision/idle"),
            ),
            hsm.transition(hsm.on(_RequestedEvent), hsm.guard(_has_requested), hsm.target("/Revision/starting")),
            hsm.transition(
                hsm.on(_FailedEvent),
                hsm.guard(_has_internal_failure),
                hsm.effect(_fail_internal),
                hsm.target("/Revision/idle"),
            ),
        ),
        hsm.state(
            "starting",
            hsm.defer(input_event, processing.CancelEvent),
            hsm.activity(_dispatch_change),
            hsm.transition(
                hsm.on(_StartedEvent),
                hsm.guard(_has_started),
                hsm.target("/Revision/attempt"),
            ),
            hsm.transition(
                hsm.on(_FailedEvent),
                hsm.guard(_has_internal_failure),
                hsm.effect(_fail_internal),
                hsm.target("/Revision/idle"),
            ),
        ),
        hsm.choice(
            "attempt",
            hsm.transition(hsm.guard(_attempt_is_0), hsm.target("/Revision/authoring/attempt_0")),
            hsm.transition(hsm.guard(_attempt_is_1), hsm.target("/Revision/authoring/attempt_1")),
            hsm.transition(hsm.target("/Revision/authoring/attempt_2")),
        ),
        hsm.state(
            "authoring",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(processing.CancelEvent),
                hsm.guard(processing.Processing._is_cancel_request),
                hsm.effect(_queue_requested_cancel),
            ),
            hsm.transition(
                hsm.on(processing.OutputEvent),
                hsm.guard(_matches_change_output),
                hsm.effect(_check_change),
            ),
            hsm.transition(
                hsm.on(_CheckedEvent), hsm.guard(_accepted), hsm.effect(_accept), hsm.target("/Revision/idle")
            ),
            hsm.transition(hsm.on(_CheckedEvent), hsm.guard(_retryable), hsm.effect(_retry)),
            hsm.transition(
                hsm.on(_CheckedEvent), hsm.guard(_abandoned), hsm.effect(_abandon), hsm.target("/Revision/idle")
            ),
            hsm.transition(hsm.on(_RequestedEvent), hsm.guard(_has_requested), hsm.target("/Revision/starting")),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_matches_change_failure),
                hsm.effect(_fail_child),
                hsm.target("/Revision/idle"),
            ),
            hsm.transition(
                hsm.on(_CancelRequestedEvent),
                hsm.guard(_has_cancel_requested),
                hsm.target("/Revision/starting_cancel"),
            ),
            hsm.state("attempt_0"),
            hsm.state("attempt_1"),
            hsm.state("attempt_2"),
        ),
        hsm.state(
            "starting_cancel",
            hsm.defer(input_event, processing.CancelEvent, processing.CancelledEvent),
            hsm.activity(_start_cancel),
            hsm.transition(
                hsm.on(_CancelStartedEvent),
                hsm.guard(_has_cancel_started),
                hsm.target("/Revision/cancelling"),
            ),
        ),
        hsm.state(
            "cancelling",
            hsm.defer(input_event),
            hsm.transition(
                hsm.on(processing.CancelledEvent),
                hsm.guard(_cancelled_is_requested),
                hsm.effect(_emit_revision_cancelled),
                hsm.target("/Revision/idle"),
            ),
            hsm.transition(
                hsm.after(_cancel_timeout_delay),
                hsm.effect(_request_reboot),
                hsm.target("/Revision/rebooting"),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(input_event, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal, _request_reboot),
                hsm.target("/Revision/rebooting"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Revision/idle"),
            ),
        ),
        hsm.state("rebooting", hsm.defer(input_event)),
        hsm.observe(observer),
    )

    def __init__(self, *, processor: processing.Processor | ProcessorFactory, memory: memory.Memory) -> None:
        super(processing.Processing, self).__init__()
        leaf = processor() if not isinstance(processor, processing.Processor) else processor
        self._change_processing = processing.Processing(processor=leaf, instructions=type(self).instructions)
        self._memory = memory
        self._attachment_group: attachment.Group = attachment.Group(self._change_processing)


InputEvent = Revision.input_event
OutputEvent = Revision.output_event

__all__ = [
    "CHANGE_INSTRUCTIONS",
    "ChangeWriteInput",
    "FailureData",
    "InputData",
    "InputEvent",
    "OutputData",
    "OutputEvent",
    "ProcessorFactory",
    "Revision",
]
