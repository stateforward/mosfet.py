from bot import abilities
import bot
from bot import behavior as behavior_events
from bot.abilities import cognition
from bot.abilities import memory
from bot.abilities import processing
from bot.abilities.cognition import cognition as cognition_module
from bot.abilities.cognition import reflection as reflection_module
from bot.behavior import instance as behavior_instance
from bot.protocols import attachment

import asyncio
import collections.abc
import dataclasses
import datetime
import inspect
import sqlite3
import typing

import hsm
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
import pydantic
import pytest

from bot.event_schema import validate_event_data
from tests.type_helpers import model_view, object_dict
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination

CONTEXT_KEY = "tests.cognitive.context"


def test_cognition_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(cognition_module)


def test_cognition_types_never_use_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(cognition.types)


# Minimal valid Starlark HSM behavior used by Reflection create/change tests.
_ANSWER_RING_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.input",
    schema = {
        "type": "object",
        "additionalProperties": True,
    },
    description = "Live behavior input for the observed turn stimulus.",
)
output_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.output",
    schema = {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "target": {"type": "string"},
            "data": {"type": "object"},
            "reason": {"type": "string"},
        },
        "required": ["event"],
        "additionalProperties": True,
    },
    description = "Cognition event selection.",
)
triggers = ["environment.sound"]
description = "Select focus_device for the observed pattern."

def always(event):
    return True

def select_focus(event):
    hsm.dispatch(output_event, {
        "event": "bot.focus_device",
        "data": {"device": "phone"},
        "reason": "pattern",
    })

behavior = hsm.define(
    "AnswerIncomingRing",
    hsm.initial(hsm.target("/AnswerIncomingRing/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("always"),
            hsm.effect("select_focus"),
        ),
    ),
)
""".strip()


def _insert_content(
    *,
    content: str,
    scope: str = "memory",
    context_ref: str | None = None,
    subject_ref: str | None = None,
    kind: str | None = "task",
    query_tags: str | None = None,
    content_format: str = "text/plain",
    memory_id: str | None = None,
) -> memory.Statement:
    import uuid
    from sqlalchemy import insert

    table = memory.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=scope,
        context_ref=context_ref,
        subject_ref=subject_ref,
        kind=kind,
        sensitivity="standard",
        retention="retain",
        content=content,
        content_format=content_format,
        query_tags=query_tags,
    )
    return memory.compile_statement(clause)


def _select_by_query_tags(*, query_tags: str, context_ref: str | None = None, limit: int = 50) -> memory.Statement:
    from sqlalchemy import or_, select

    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == query_tags)
    if context_ref is not None:
        clause = clause.where(or_(table.c.context_ref.is_(None), table.c.context_ref == context_ref))
    clause = clause.order_by(table.c.created_at).limit(limit)
    return memory.compile_statement(clause)


# Typed accessors for the protected internals these tests deliberately exercise
# (same convention as tests/hsm_instance_state.py).
def cognition_intuition(ability: cognition.Cognition) -> cognition.Intuition:
    return typing.cast(cognition.Intuition, object.__getattribute__(ability, "_intuition"))


def cognition_reasoning(ability: cognition.Cognition) -> cognition.Reasoning:
    return typing.cast(cognition.Reasoning, object.__getattribute__(ability, "_reasoning"))


def cognition_autonomy(ability: cognition.Cognition) -> cognition.Autonomy | None:
    return typing.cast(cognition.Autonomy | None, object.__getattribute__(ability, "_autonomy"))


def cognition_reflection(ability: cognition.Cognition) -> cognition.Reflection:
    return typing.cast(cognition.Reflection, object.__getattribute__(ability, "_reflection"))


def cognition_attachment_group(ability: cognition.Cognition) -> attachment.Group:
    return typing.cast(attachment.Group, object.__getattribute__(ability, "_attachment_group"))


def reflection_select_processing(ability: cognition.Reflection) -> processing.Processing:
    return typing.cast(processing.Processing, object.__getattribute__(ability, "_select_processing"))


_CognitionGuard = collections.abc.Callable[[hsm.Context, cognition.Cognition, hsm.Event[typing.Any]], bool]
_ReflectionGuard = collections.abc.Callable[[hsm.Context, cognition.Reflection, hsm.Event[typing.Any]], bool]

cognition_matches_intuition_output = typing.cast(
    _CognitionGuard, object.__getattribute__(cognition.Cognition, "_matches_intuition_output")
)
cognition_matches_autonomy_output = typing.cast(
    _CognitionGuard, object.__getattribute__(cognition.Cognition, "_matches_autonomy_output")
)
cognition_matches_child_cancelled = typing.cast(
    _CognitionGuard, object.__getattribute__(cognition.Cognition, "_matches_child_cancelled")
)
reflection_matches_select_output = typing.cast(
    _ReflectionGuard, object.__getattribute__(reflection_module.Reflection, "_matches_select_output")
)


def _processor_events(output: object) -> processing.Events:
    """Processing honors Result declines and invalid payloads at runtime; Processor declares Events only."""

    return typing.cast(processing.Events, output)


def empty_output() -> cognition.types.OutputData:
    """Empty selection product — intuition treats as unhandled and cascades to reasoning."""

    return ()


def ignore_output(reason: str = "") -> cognition.types.OutputData:
    """Handled deliberate pass (cognition.ignore); not an empty cascade."""

    return (
        cognition.types.EventData(
            event=cognition.types.IgnoreEvent.name,
            reason=reason or None,
        ),
    )


def no_output(reason: str = "") -> cognition.types.OutputData:
    """Handled no-op product for stage fixtures (ignore). Use empty_output() for cascade."""

    return ignore_output(reason)


def focus_output(device: str, reason: str) -> cognition.types.OutputData:
    return (
        cognition.types.EventData(
            event=bot.FocusDeviceEvent.name,
            data=bot.FocusDeviceEventData(device=device).model_dump(),
            reason=reason,
        ),
    )


def _as_events(
    output: cognition.types.OutputData | cognition.types.EventData | processing.Events | processing.SelectedEvent,
    *,
    confidence: int | None = None,
) -> processing.Events:
    if output is None:
        return ()
    if isinstance(output, processing.SelectedEvent):
        if confidence is None or output.confidence is not None:
            return (output,)
        return (dataclasses.replace(output, confidence=confidence),)
    if isinstance(output, cognition.types.EventData):
        return (
            processing.SelectedEvent(
                event=output.event,
                target=output.target,
                data=output.data,
                reason=output.reason,
                confidence=confidence,
            ),
        )
    if not output:
        return ()
    items: list[processing.SelectedEvent] = []
    for item in output:
        if isinstance(item, processing.SelectedEvent):
            if confidence is not None and item.confidence is None:
                items.append(dataclasses.replace(item, confidence=confidence))
            else:
                items.append(item)
            continue
        if isinstance(item, cognition.types.EventData):
            items.append(
                processing.SelectedEvent(
                    event=item.event,
                    target=item.target,
                    data=item.data,
                    reason=item.reason,
                    confidence=confidence,
                )
            )
            continue
        raise TypeError(f"unsupported event selection item: {type(item)!r}")
    return tuple(items)


def focus_event_offer() -> hsm.Event[object]:
    return bot.FocusDeviceEvent


def _accept_focus_event(
    ctx: hsm.Context,
    instance: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    del ctx, instance, event


class _BotActor(hsm.Instance):
    """Minimal modeled recipient for cognition selections in unit tests."""

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "BotActor",
        hsm.initial(hsm.target("/BotActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(bot.FocusDeviceEvent), hsm.effect(_accept_focus_event)),
        ),
    )

    @typing.override
    async def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx, event


class RecordingIntuitionProcessor(processing.Processor):
    """Return events (with optional patched confidence) or unhandled OutputData."""

    calls: list[processing.InputData]
    output: processing.Events | cognition.intuition.OutputData

    def __init__(
        self,
        output: cognition.types.OutputData
        | cognition.intuition.OutputData
        | processing.Events
        | processing.SelectedEvent,
        *,
        confidence: int | None = 99,
    ) -> None:
        self.calls = []
        if isinstance(output, cognition.intuition.OutputData):
            if output.result is None:
                self.output = output
            else:
                self.output = _as_events(output.result, confidence=confidence)
        elif isinstance(output, processing.SelectedEvent | tuple):
            self.output = _as_events(output, confidence=confidence)
        else:
            self.output = _as_events(output, confidence=confidence)

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        if isinstance(self.output, cognition.intuition.OutputData):
            if self.output.result is None:
                return _processor_events(processing.Result[processing.Events].unhandled())
            return _as_events(self.output.result)
        return self.output


class DelayedIntuitionProcessor(processing.Processor):
    calls: list[processing.InputData]
    cancelled: bool
    release_first: asyncio.Event
    output: processing.Events | cognition.intuition.OutputData

    def __init__(
        self,
        output: cognition.types.OutputData | cognition.intuition.OutputData,
        *,
        confidence: int | None = 99,
    ) -> None:
        self.calls = []
        self.cancelled = False
        self.release_first = asyncio.Event()
        if isinstance(output, cognition.intuition.OutputData):
            if output.result is None:
                self.output = output
            else:
                self.output = _as_events(output.result, confidence=confidence)
        else:
            self.output = _as_events(output, confidence=confidence)

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        if len(self.calls) == 1:
            try:
                _ = await self.release_first.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if isinstance(self.output, cognition.intuition.OutputData):
            if self.output.result is None:
                return _processor_events(processing.Result[processing.Events].unhandled())
            return _as_events(self.output.result)
        return self.output


class RecordingReasoningProcessor(processing.Processor):
    calls: list[processing.InputData]
    output: cognition.reasoning.OutputData

    def __init__(self, output: cognition.types.OutputData | cognition.reasoning.OutputData) -> None:
        self.calls = []
        self.output = (
            output
            if isinstance(output, cognition.reasoning.OutputData)
            else cognition.reasoning.OutputData(result=output)
        )

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        if isinstance(self.output, cognition.reasoning.OutputData):
            return _as_events(self.output.result)
        return _as_events(self.output)


class DelayedReasoningProcessor(processing.Processor):
    calls: list[processing.InputData]
    release_first: asyncio.Event
    cancelled: bool

    def __init__(self) -> None:
        self.calls = []
        self.release_first = asyncio.Event()
        self.cancelled = False

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        if len(self.calls) == 1:
            try:
                _ = await self.release_first.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        label = f"reasoned {len(self.calls)}"
        return _as_events(ignore_output(label), confidence=99)


class FailingReasoningProcessor(processing.Processor):
    calls: list[processing.InputData]

    def __init__(self) -> None:
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        raise RuntimeError("reasoning failed")


class InvalidReasoningProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return _processor_events({"not": "valid-output"})


class RecordingOutputOperation(processing.Processor):
    calls: list[processing.InputData]
    output: processing.Events | processing.Result[processing.Events] | None

    def __init__(
        self,
        output: cognition.types.OutputData
        | cognition.types.EventData
        | processing.Events
        | processing.Result[processing.Events]
        | None,
    ) -> None:
        self.calls = []
        if isinstance(output, processing.Result) or output is None:
            self.output = output
        elif isinstance(output, tuple) and (not output or isinstance(output[0], processing.SelectedEvent)):
            self.output = typing.cast(processing.Events, output)
        else:
            self.output = _as_events(output)

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return _processor_events(self.output)


def _events_from_output(selection: cognition.types.OutputData) -> processing.Events:
    return tuple(
        processing.SelectedEvent(
            event=item.event,
            target=item.target,
            data=item.data,
            reason=item.reason,
        )
        for item in selection
    )


class FixedSelectionProcessor(processing.Processor):
    selection: cognition.types.OutputData
    calls: list[processing.InputData]

    def __init__(self, selection: cognition.types.OutputData | None = None) -> None:
        # Do not use `or`: empty_output() is falsy but means "no behavior selected".
        self.selection = empty_output() if selection is None else selection
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        return _events_from_output(self.selection)


class FixedWriteProcessor(processing.Processor):
    """Single Reflection write-phase processor (create and change both use change events)."""

    written: list[behavior_events.ChangeData]
    calls: list[processing.InputData]

    def __init__(
        self,
        written: behavior_events.ChangeData | list[behavior_events.ChangeData],
    ) -> None:
        self.written = [written] if isinstance(written, behavior_events.ChangeData) else list(written)
        self.calls = []

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        self.calls.append(input)
        if not self.written:
            raise RuntimeError("FixedWriteProcessor has no remaining write payloads.")
        payload = self.written.pop(0)
        return (
            processing.SelectedEvent(
                event=behavior_events.ChangeEvent.name,
                data=payload.model_dump(mode="json"),
            ),
        )


class FixedProcessor(processing.Processor):
    """Test reflection transport: routes select vs change by stamped input instructions."""

    selection: cognition.types.OutputData
    write: behavior_events.ChangeData | list[behavior_events.ChangeData]
    step: FixedSelectionProcessor
    write_step: FixedWriteProcessor
    builds: list[str]

    def __init__(
        self,
        selection: cognition.types.OutputData | None = None,
        *,
        write: behavior_events.ChangeData | list[behavior_events.ChangeData] | None = None,
        # Compat kwargs used by older tests / call sites.
        change_write: behavior_events.ChangeData | list[behavior_events.ChangeData] | None = None,
        create_write: behavior_events.CreateData | list[behavior_events.CreateData] | None = None,
    ) -> None:
        # Do not use `or`: empty_output() is falsy but means "no behavior selected".
        self.selection = empty_output() if selection is None else selection
        if write is not None:
            self.write = write
        elif change_write is not None:
            self.write = change_write
        elif create_write is not None:
            # Create select still authors via the write machine as ChangeData.
            creates = [create_write] if isinstance(create_write, behavior_events.CreateData) else list(create_write)
            self.write = [
                behavior_events.ChangeData(
                    name=item.name,
                    triggers=item.triggers,
                    description=item.description,
                    reason=item.reason,
                    source=item.source,
                )
                for item in creates
            ]
        elif self.selection and self.selection[0].event in {
            behavior_events.CreateEvent.name,
            behavior_events.ChangeEvent.name,
        }:
            raw = {**(self.selection[0].data or {}), "event": behavior_events.ChangeEvent.name}
            self.write = behavior_events.ChangeData.model_validate(raw)
        else:
            self.write = behavior_events.ChangeData(name="unused-write", reason="phase processor unused")
        self.step = FixedSelectionProcessor(self.selection)
        self.write_step = FixedWriteProcessor(self.write)
        self.builds = []

    @property
    def change_step(self) -> FixedWriteProcessor:
        return self.write_step

    @property
    def create_step(self) -> FixedWriteProcessor:
        return self.write_step

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        instructions = input.instructions or ""
        self.builds.append(instructions)
        if instructions == cognition.reflection.CHANGE_INSTRUCTIONS or "changing phase" in instructions:
            return await self.write_step.process(input)
        return await self.step.process(input)


def behavior_create_selection(
    *,
    name: str = "AnswerIncomingRing",
    triggers: tuple[str, ...] = (),
    reason: str = "stored pattern",
) -> cognition.types.OutputData:
    return (
        cognition.types.EventData(
            event=behavior_events.CreateEvent.name,
            data={
                "event": behavior_events.CreateEvent.name,
                "name": name,
                "triggers": list(triggers),
                "reason": reason,
            },
            reason=reason,
        ),
    )


class RecordingReflectionAbility(cognition.Reflection):
    """Reflection with fixed select/write processors for structural tests."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        selection: cognition.types.OutputData | None = None,
    ) -> None:
        super().__init__(
            processor=FixedProcessor(selection or behavior_create_selection()),
            memory=memory.Memory(connection=connection),
        )


class RecordingCognition(cognition.Cognition):
    outputs: list[cognition.types.OutputData]
    failures: list[abilities.FailureData]

    def __init__(
        self,
        *,
        intuition: cognition.Intuition | None = None,
        reasoning: cognition.Reasoning | None = None,
        autonomy: cognition.Autonomy | None = None,
        intuition_processor: processing.Processor | None = None,
        reasoning_processor: processing.Processor | None = None,
        reflection: cognition.Reflection | None = None,
        processing: cognition.Cognition | processing.Processing | None = None,
    ) -> None:
        # `processing=` accepted only for transitional call sites that pass make_cognition().
        if isinstance(processing, cognition.Cognition):
            intuition = processing._intuition
            reasoning = processing._reasoning
            autonomy = processing._autonomy if autonomy is None else autonomy
            reflection = processing._reflection if reflection is None else reflection
        if intuition is None or reasoning is None:
            default_i, default_r = cognition_abilities(
                intuition_processor=intuition_processor,
                reasoning_processor=reasoning_processor,
            )
            intuition = intuition or default_i
            reasoning = reasoning or default_r
        reflection = reflection or cognition.Reflection(
            processor=FixedProcessor(no_output("reflection")),
            memory=memory.Memory(),
        )
        super().__init__(
            intuition=intuition,
            reasoning=reasoning,
            autonomy=autonomy,
            reflection=reflection,
        )
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == self.output_event.name:
            self.outputs.append(typing.cast(cognition.types.OutputData, event.data))
        if event.name == self.failed_event.name:
            failure = event.data
            assert isinstance(failure, abilities.FailureData)
            self.failures.append(failure)
        return super().dispatch(ctx, event)


def cognition_input() -> cognition.InputData:
    """Live body context for Cognition ability input (no bot focus/devices)."""

    return cognition.InputData(
        stimulus=bot.InputEventData(target_device="phone", priority=0),
        abilities=(),
        actors={"bot": _BotActor()},
        focus=None,
        focus_candidates=("phone",),
    )


async def started_cognition_input(ctx: hsm.Context) -> cognition.InputData:
    data = cognition_input()
    actor = data.actors["bot"]
    assert isinstance(actor, _BotActor)
    assert actor.model is not None
    _ = await hsm.started(ctx, actor, actor.model)
    return data


def deliberative_input(
    *,
    schemas: tuple[hsm.Event[object], ...] | None = None,
) -> processing.InputData:
    """Deliberative input for nested processors (intuition/reasoning/reflection)."""

    return processing.InputData(
        input=bot.InputEventData(target_device="phone", priority=0),
        schemas=schemas or (),
        actors={},
    )


def cognition_turn(
    *,
    operation_id: str = "test-turn",
    generation: str = "operation-token",
) -> cognition.types.TurnData:
    return cognition.types.TurnData(
        input=cognition_input(),
        operation_id=operation_id,
        generation=generation,
    )


def intuition_input(
    input: processing.InputData | None = None,
    *,
    operation_id: str = "test-turn",
    generation: str = "operation-token",
) -> cognition.intuition.InputData:
    return cognition.intuition.InputData(
        turn=cognition_turn(operation_id=operation_id, generation=generation),
        processing_input=input or deliberative_input(),
    )


def reasoning_input(
    input: processing.InputData | None = None,
    *,
    operation_id: str = "test-turn",
    generation: str = "operation-token",
) -> cognition.reasoning.InputData:
    return cognition.reasoning.InputData(
        turn=cognition_turn(operation_id=operation_id, generation=generation),
        processing_input=input or deliberative_input(),
    )


def cognition_abilities(
    *,
    intuition_processor: processing.Processor | None = None,
    reasoning_processor: processing.Processor | None = None,
) -> tuple[cognition.Intuition, cognition.Reasoning]:
    intuition = cognition.Intuition(
        processor=intuition_processor
        or RecordingIntuitionProcessor(cognition.intuition.OutputData(result=no_output("fast"))),
    )
    reasoning = cognition.Reasoning(
        processor=reasoning_processor or RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
    )
    return intuition, reasoning


def make_cognition(
    *,
    intuition_processor: processing.Processor | None = None,
    reasoning_processor: processing.Processor | None = None,
    reflection: cognition.Reflection | None = None,
    autonomy: cognition.Autonomy | None = None,
) -> cognition.Cognition:
    intuition, reasoning = cognition_abilities(
        intuition_processor=intuition_processor,
        reasoning_processor=reasoning_processor,
    )
    return cognition.Cognition(
        intuition=intuition,
        reasoning=reasoning,
        autonomy=autonomy,
        reflection=reflection
        or cognition.Reflection(
            processor=FixedProcessor(no_output("reflection")),
            memory=memory.Memory(),
        ),
    )


def test_cognition_rejects_child_terminal_from_wrong_source() -> None:
    async def run() -> bool:
        ability = make_cognition()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, ability)
        child = cognition_intuition(ability)
        turn = cognition_turn(operation_id="forged-turn", generation="forged-operation-token")
        forged = dataclasses.replace(
            child.output_event.with_data(cognition.types.CompletionData(turn=turn, output=None)),
            id="forged-turn:intuition",
            source="forged-child",
            target=hsm.id(ability),
        )
        return cognition_matches_intuition_output(ctx, ability, forged)

    assert not asyncio.run(run())


def test_reflection_rejects_child_terminal_from_wrong_source() -> None:
    async def run() -> bool:
        ability = cognition.Reflection(
            processor=RecordingIntuitionProcessor(no_output("unused")), memory=memory.Memory()
        )
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, ability)
        child = reflection_select_processing(ability)
        forged = dataclasses.replace(
            child.output_event.with_data(None),
            id="forged-reflection:select",
            source="forged-child",
            target=hsm.id(ability),
            metadata={},
        )
        return reflection_matches_select_output(ctx, ability, forged)

    assert not asyncio.run(run())


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)


class CognitionAttachmentOwner(hsm.Instance):
    lifecycle: list[hsm.Event[typing.Any]]
    watched: hsm.Instance | None

    @staticmethod
    def _record(
        ctx: hsm.Context,
        instance: "CognitionAttachmentOwner",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        instance.lifecycle.append(event)

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "CognitionAttachmentOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.AttachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(cognition.CancelledEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(cognition.Cognition.failed_event), hsm.effect(_record)),
            hsm.transition(hsm.on(bot.RebootEvent), hsm.effect(_record)),
        ),
    )

    def __init__(self, watched: hsm.Instance | None = None) -> None:
        super().__init__()
        self.lifecycle = []
        self.watched = watched


async def start_cognition_ability_for_test(
    ability: abilities.Ability[typing.Any, typing.Any],
    ctx: hsm.Context | None = None,
) -> hsm.Context:
    context = shared_hsm_context(ctx)
    await start_abilities_for_test(context, ability)
    return context


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


@typing.runtime_checkable
class CognitionModelInternals(typing.Protocol):
    deferred_map: collections.abc.Mapping[str, collections.abc.Mapping[str, str]]


def cognition_model_internals(model: object) -> CognitionModelInternals:
    assert isinstance(model, CognitionModelInternals)
    return model


def test_cognitive_abilities_are_concrete_processing_abilities() -> None:
    intuition_processor = RecordingIntuitionProcessor(cognition.intuition.OutputData(result=no_output("fast")))
    reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "deliberate"))
    intuition = cognition.Intuition(processor=intuition_processor)
    reasoning = cognition.Reasoning(processor=reasoning_processor)
    connection = sqlite3.connect(":memory:")
    try:
        reflection = RecordingReflectionAbility(connection)

        assert isinstance(intuition, abilities.Ability)
        assert isinstance(intuition, processing.Processing)
        assert isinstance(reasoning, abilities.Ability)
        assert isinstance(reasoning, processing.Processing)
        assert isinstance(reflection, abilities.Ability)
        assert isinstance(reflection, processing.Processing)
    finally:
        connection.close()


def test_cognition_package_has_no_generic_operations_layer() -> None:
    """Cognitive states own child coordination; no mediator module is public."""

    assert "operations" not in cognition.__all__


def test_cognition_builds_one_group_for_required_children_and_waits_for_aggregate_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[tuple[hsm.Instance, ...]], int, str]:
        groups: list[tuple[hsm.Instance, ...]] = []
        requests: list[hsm.Event[attachment.AttachData]] = []
        group_init = attachment.Group.__init__

        def record_group(group: attachment.Group, *members: hsm.Instance) -> None:
            groups.append(members)
            group_init(group, *members)

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del group, ctx
            requests.append(event)

        monkeypatch.setattr(attachment.Group, "__init__", record_group)
        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        intuition, reasoning = cognition_abilities()
        minimal = cognition.Cognition(
            intuition=intuition,
            reasoning=reasoning,
            reflection=cognition.Reflection(
                processor=FixedProcessor(no_output("reflect")),
                memory=memory.Memory(),
            ),
        )
        connection = sqlite3.connect(":memory:")
        try:
            maximal = cognition.Cognition(
                autonomy=cognition.Autonomy(),
                intuition=cognition.Intuition(processor=RecordingIntuitionProcessor(no_output("fast"))),
                reasoning=cognition.Reasoning(processor=RecordingReasoningProcessor(no_output("slow"))),
                reflection=cognition.Reflection(
                    processor=FixedProcessor(no_output("reflect")),
                    memory=memory.Memory(connection=connection),
                ),
            )
        finally:
            connection.close()
        owner = hsm.Instance()
        owner_model = hsm.define(
            "CognitionAttachmentOwner",
            hsm.initial(hsm.target("ready")),
            hsm.state("ready"),
        )
        ctx = hsm.Context()
        _ = await hsm.started(ctx, owner, owner_model)
        await minimal.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "cognition-attach",
            ),
        )
        await wait_until(lambda: bool(requests) or minimal.state().endswith("/idle"))
        state = minimal.state()
        await minimal.stop(minimal.context())
        del maximal
        cognition_groups = [
            members
            for members in groups
            if members and isinstance(members[0], cognition.Autonomy | cognition.Intuition)
        ]
        return cognition_groups, len(requests), state

    groups, request_count, state = asyncio.run(run())

    assert len(groups) == 2
    assert len(groups[0]) == 3
    assert isinstance(groups[0][0], cognition.Intuition)
    assert isinstance(groups[0][1], cognition.Reasoning)
    assert isinstance(groups[0][2], cognition.Reflection)
    assert len(groups[1]) == 4
    assert isinstance(groups[1][0], cognition.Autonomy)
    assert isinstance(groups[1][1], cognition.Intuition)
    assert isinstance(groups[1][2], cognition.Reasoning)
    assert isinstance(groups[1][3], cognition.Reflection)
    assert request_count == 1
    assert state == "/CognitionLifecycle/attached/behavior/initializing"


def test_cognition_reports_aggregate_attachment_failure_and_accepts_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int, str]:
        requests: list[tuple[attachment.Group, hsm.Event[attachment.AttachData]]] = []

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del ctx
            requests.append((group, event))

        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        ctx = hsm.Context()
        ability = make_cognition()
        owner = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "cognition-attach-failed",
            ),
        )
        await wait_until(lambda: len(requests) == 1)
        group, request = requests[0]
        request_data = request.data
        assert request_data is not None
        reply = request_data.reply_to
        assert reply is not None
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(
                attachment.AttachFailedEvent.with_data(
                    attachment.FailedData(
                        actor=ability,
                        kind=attachment.FailureKind.INITIALIZATION,
                        message="cognition child failed",
                    )
                ),
                id=request.id,
                source=hsm.id(group),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await wait_until(lambda: ability.state().endswith("/detached"))
        await ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "cognition-attach-retry",
            ),
        )
        await wait_until(lambda: len(requests) == 2)
        lifecycle = owner.lifecycle
        state = ability.state()
        await ability.stop(ability.context())
        return lifecycle, len(requests), state

    lifecycle, request_count, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachFailedEvent.name]
    assert lifecycle[0].id == "cognition-attach-failed"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.INITIALIZATION
    assert request_count == 2
    assert state == "/CognitionLifecycle/attached/behavior/initializing"


def test_cognition_detaches_once_through_group_and_can_reattach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[attachment.DetachData]], list[hsm.Event[typing.Any]], str]:
        requests: list[hsm.Event[attachment.DetachData]] = []
        group_detach = attachment.Group.detach

        async def record_detach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            if group is cognition_attachment_group(ability):
                requests.append(event)
            await group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "detach", record_detach)
        ctx = hsm.Context()
        owner = CognitionAttachmentOwner()
        ability = make_cognition()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "cognition-first-attach",
            ),
        )
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        await ability.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "cognition-detach",
            ),
        )
        await wait_until(lambda: ability.state().endswith("/detached"))
        await ability.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "cognition-second-attach",
            ),
        )
        await wait_until(lambda: ability.state().endswith("/idle"))
        lifecycle = owner.lifecycle
        state = ability.state()
        await ability.stop(ability.context())
        return requests, lifecycle, state

    requests, lifecycle, state = asyncio.run(run())

    assert len(requests) == 1
    assert requests[0].id == "cognition-detach"
    assert [event.name for event in lifecycle] == [
        attachment.DetachedEvent.name,
        attachment.AttachCompleteEvent.name,
    ]
    assert [event.id for event in lifecycle] == ["cognition-detach", "cognition-second-attach"]
    assert state == "/CognitionLifecycle/attached/behavior/idle"


def test_cognition_detach_cancels_active_intuition_through_group() -> None:
    async def run() -> tuple[bool, list[hsm.Event[typing.Any]], str]:
        processor = DelayedIntuitionProcessor(no_output("held"))
        ability = make_cognition(intuition_processor=processor)
        ctx = hsm.Context()
        owner = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data(await started_cognition_input(ctx)))
        await wait_until(lambda: bool(processor.calls))
        assert ability.state().endswith("/intuition")
        await ability.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "cognition-active-detach",
            ),
        )
        await wait_until(lambda: ability.state().endswith("/detached"))
        await wait_until(lambda: processor.cancelled)
        lifecycle = owner.lifecycle
        state = ability.state()
        await ability.stop(ability.context())
        return processor.cancelled, lifecycle, state

    cancelled, lifecycle, state = asyncio.run(run())

    assert cancelled
    assert [event.name for event in lifecycle] == [attachment.DetachedEvent.name]
    assert lifecycle[0].id == "cognition-active-detach"
    assert state == "/CognitionLifecycle/detached"


@pytest.mark.parametrize("child", ["intuition", "reasoning"])
def test_cognition_cancel_waits_for_correlated_active_child(child: str) -> None:
    async def run() -> tuple[bool, list[hsm.Event[typing.Any]], str]:
        intuition_processor = DelayedIntuitionProcessor(no_output("held"))
        reasoning_processor = DelayedReasoningProcessor()
        if child == "reasoning":
            ability = make_cognition(
                intuition_processor=RecordingIntuitionProcessor(cognition.intuition.OutputData(reason="escalate")),
                reasoning_processor=reasoning_processor,
            )
        else:
            ability = make_cognition(
                intuition_processor=intuition_processor,
                reasoning_processor=reasoning_processor,
            )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner(ability)
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()

        operation_id = f"cancel-{child}"
        token = f"token-{child}"
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                ability.input_event.with_data_and_id(await started_cognition_input(ctx), operation_id),
            ),
        )
        expected_state = f"/{child}"
        await wait_until(lambda: ability.state().endswith(expected_state))
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                cognition.CancelEvent.with_data(cognition.CancelData(operation_id=operation_id, token=token)),
                id=operation_id,
                source=hsm.id(owner),
                target=hsm.id(ability),
            ),
        )
        await wait_until(lambda: any(event.name == cognition.CancelledEvent.name for event in owner.lifecycle))

        cancelled = intuition_processor.cancelled if child == "intuition" else reasoning_processor.cancelled
        return cancelled, owner.lifecycle, ability.state()

    cancelled, lifecycle, state = asyncio.run(run())

    terminals = [event for event in lifecycle if event.name == cognition.CancelledEvent.name]
    assert cancelled
    assert len(terminals) == 1
    assert terminals[0].id == f"cancel-{child}"
    assert isinstance(terminals[0].data, cognition.CancelledData)
    assert terminals[0].data.operation_id == f"cancel-{child}"
    assert terminals[0].data.token == f"token-{child}"
    assert state.endswith("/idle")


@pytest.mark.parametrize("phase", ["autonomy", "intuition", "reasoning"])
def test_cognition_direct_cancellation_acknowledges_active_child(
    phase: str,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, tuple[str, ...]]:
        intuition_processor: processing.Processor
        if phase == "reasoning":
            intuition_processor = RecordingIntuitionProcessor(cognition.intuition.OutputData(reason="escalate"))
        else:
            intuition_processor = DelayedIntuitionProcessor(no_output("held"))
        intuition, reasoning = cognition_abilities(
            intuition_processor=intuition_processor,
            reasoning_processor=DelayedReasoningProcessor(),
        )
        ability = cognition.Cognition(
            intuition=intuition,
            reasoning=reasoning,
            reflection=cognition.Reflection(
                processor=FixedProcessor(no_output("reflection")),
                memory=memory.Memory(),
            ),
            autonomy=cognition.Autonomy() if phase == "autonomy" else None,
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        operation_id = f"pre-registration-{phase}"
        _ = await hsm.dispatch(
            ctx,
            ability,
            ability.input_event.with_data_and_id(await started_cognition_input(ctx), operation_id),
        )
        await wait_until(lambda: ability.state().endswith(f"/{phase}"))
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                cognition.CancelEvent.with_data(
                    cognition.CancelData(operation_id=operation_id, token=f"token-{phase}")
                ),
                id=operation_id,
                source=hsm.id(owner),
                target=hsm.id(ability),
            ),
        )
        await asyncio.sleep(0.02)
        await wait_until(lambda: any(event.name == cognition.CancelledEvent.name for event in owner.lifecycle))
        instances = ability.context().value(hsm.Keys.Instances)
        operation_keys = (
            tuple(
                key
                for key in instances
                if isinstance(key, str) and key.startswith(f"processing.operation:{hsm.id(ability)}:")
            )
            if isinstance(instances, collections.abc.Mapping)
            else ()
        )
        return list(owner.lifecycle), ability.state(), operation_keys

    lifecycle, state, operation_keys = asyncio.run(run())

    terminals = [event for event in lifecycle if event.name == cognition.CancelledEvent.name]
    assert len(terminals) == 1
    assert state.endswith("/idle")
    assert operation_keys == ()


def test_cognition_stubborn_child_cancel_timeout_requests_reboot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubbornIntuition(cognition.Intuition):
        submodel = hsm.define(
            "StubbornIntuition",
            hsm.initial(hsm.target("/StubbornIntuition/waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(cognition.Intuition.input_event),
                    hsm.effect(lambda ctx, instance, event: None),
                ),
            ),
        )

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, tuple[str, ...]]:
        monkeypatch.setattr(cognition_module, "_CANCEL_TEARDOWN_TIMEOUT", datetime.timedelta(milliseconds=10))
        stubborn = StubbornIntuition(processor=RecordingIntuitionProcessor(no_output("unused")))
        _, reasoning = cognition_abilities()
        ability = cognition.Cognition(
            intuition=stubborn,
            reasoning=reasoning,
            reflection=cognition.Reflection(
                processor=FixedProcessor(no_output("reflection")),
                memory=memory.Memory(),
            ),
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        operation_id = "stubborn-cognition"
        token = "stubborn-cognition-token"
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                ability.input_event.with_data_and_id(await started_cognition_input(ctx), operation_id),
            ),
        )
        await wait_until(lambda: ability.state().endswith("/intuition"))
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                cognition.CancelEvent.with_data(cognition.CancelData(operation_id=operation_id, token=token)),
                id=operation_id,
                source=hsm.id(owner),
                target=hsm.id(ability),
            ),
        )
        await asyncio.sleep(0.03)
        await wait_until(lambda: any(event.name == bot.RebootEvent.name for event in owner.lifecycle))
        await asyncio.sleep(0.01)
        instances = ability.context().value(hsm.Keys.Instances)
        operation_keys = (
            tuple(
                key
                for key in instances
                if isinstance(key, str) and key.startswith(f"processing.operation:{hsm.id(ability)}:")
            )
            if isinstance(instances, collections.abc.Mapping)
            else ()
        )
        return list(owner.lifecycle), ability.state(), operation_keys

    lifecycle, state, operation_keys = asyncio.run(run())

    reboots = [event for event in lifecycle if event.name == bot.RebootEvent.name]
    assert len(reboots) == 1
    assert reboots[0].data == bot.RebootEventData(reason="cognition_cancel_teardown_failed")
    assert state.endswith("/rebooting")
    assert operation_keys == ()


def test_cognition_child_timeout_reboots_from_active_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubbornIntuition(cognition.Intuition):
        submodel = hsm.define(
            "TimedOutIntuition",
            hsm.initial(hsm.target("/TimedOutIntuition/waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(cognition.Intuition.input_event),
                    hsm.effect(lambda ctx, instance, event: None),
                ),
            ),
        )

    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        monkeypatch.setattr(cognition_module, "_CHILD_OPERATION_TIMEOUT", datetime.timedelta(milliseconds=10))
        stubborn = StubbornIntuition(processor=RecordingIntuitionProcessor(no_output("unused")))
        _, reasoning = cognition_abilities()
        ability = cognition.Cognition(
            intuition=stubborn,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            ability,
            ability.input_event.with_data_and_id(await started_cognition_input(ctx), "timed-out-turn"),
        )
        await asyncio.sleep(0.03)
        await wait_until(lambda: any(event.name == bot.RebootEvent.name for event in owner.lifecycle))
        return list(owner.lifecycle), ability.state()

    lifecycle, state = asyncio.run(run())

    reboots = [event for event in lifecycle if event.name == bot.RebootEvent.name]
    assert len(reboots) == 1
    assert reboots[0].data == bot.RebootEventData(reason="cognition_child_teardown_failed")
    assert state.endswith("/rebooting")


def test_cognition_forwards_reflection_reboot_to_bot_owner() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ability = make_cognition()
        owner = CognitionAttachmentOwner()
        ctx = hsm.Context()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        request = dataclasses.replace(
            bot.RebootEvent.with_data(bot.RebootEventData(reason="cognition_child_teardown_failed")),
            id="reflection-reboot",
            source=hsm.id(cognition_reflection(ability)),
            target=hsm.id(ability),
        )

        _ = await hsm.dispatch(ctx, ability, request)
        await wait_until(lambda: bool(owner.lifecycle))
        return list(owner.lifecycle), ability.state()

    lifecycle, state = asyncio.run(run())

    assert len(lifecycle) == 1
    assert lifecycle[0].name == bot.RebootEvent.name
    assert lifecycle[0].id == "reflection-reboot"
    assert lifecycle[0].data == bot.RebootEventData(reason="cognition_child_teardown_failed")
    assert state.endswith("/rebooting")


def test_cancelled_turn_cannot_cancel_the_next_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    class TwoTurnHangingProcessor(processing.Processor):
        calls: list[processing.InputData]
        cancelled_calls: list[int]
        releases: tuple[asyncio.Event, asyncio.Event]

        def __init__(self) -> None:
            self.calls = []
            self.cancelled_calls = []
            self.releases = (asyncio.Event(), asyncio.Event())

        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            index = len(self.calls)
            self.calls.append(input)
            try:
                await self.releases[index].wait()
            except asyncio.CancelledError:
                self.cancelled_calls.append(index)
                raise
            return ()

    async def run() -> tuple[list[int], str]:
        monkeypatch.setattr(cognition_module, "_CHILD_OPERATION_TIMEOUT", datetime.timedelta(milliseconds=30))
        processor = TwoTurnHangingProcessor()
        ability = make_cognition(intuition_processor=processor)
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()

        first_id = "cancelled-first-turn"
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                ability.input_event.with_data_and_id(await started_cognition_input(ctx), first_id),
            ),
        )
        await wait_until(lambda: len(processor.calls) == 1)
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                cognition.CancelEvent.with_data(cognition.CancelData(operation_id=first_id, token="first-token")),
                id=first_id,
                source=hsm.id(owner),
                target=hsm.id(ability),
            ),
        )
        await wait_until(lambda: ability.state().endswith("/idle"))
        monkeypatch.setattr(cognition_module, "_CHILD_OPERATION_TIMEOUT", datetime.timedelta(seconds=1))

        _ = await hsm.dispatch(
            ctx,
            ability,
            ability.input_event.with_data_and_id(await started_cognition_input(ctx), "second-turn"),
        )
        await wait_until(lambda: len(processor.calls) == 2)
        await asyncio.sleep(0.05)
        result = list(processor.cancelled_calls), ability.state()
        processor.releases[1].set()
        return result

    cancelled_calls, state = asyncio.run(run())

    assert cancelled_calls == [0]
    assert state.endswith("/intuition")


def test_cognition_ignores_stale_public_child_terminal() -> None:
    async def run() -> tuple[list[cognition.types.OutputData], str]:
        processor = DelayedIntuitionProcessor(no_output("held"))
        intuition, reasoning = cognition_abilities(intuition_processor=processor)
        ability = RecordingCognition(intuition=intuition, reasoning=reasoning)
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        current_input = await started_cognition_input(ctx)
        _ = await hsm.dispatch(
            ctx,
            ability,
            ability.input_event.with_data_and_id(current_input, "current-turn"),
        )
        await wait_until(lambda: ability.state().endswith("/intuition"))
        stale = dataclasses.replace(
            intuition.output_event.with_data(_as_events(focus_output("phone", "stale"))),
            id="old-turn:intuition",
            source=hsm.id(intuition),
            target=hsm.id(ability),
            metadata={"bot.cognition.input": current_input},
        )
        _ = await hsm.dispatch(ctx, ability, stale)
        await asyncio.sleep(0)
        outputs = list(ability.outputs)
        state = ability.state()
        processor.release_first.set()
        await wait_until(lambda: ability.state().endswith("/idle"))
        return outputs, state

    outputs, state = asyncio.run(run())

    assert outputs == []
    assert state.endswith("/intuition")


def test_cognition_rejects_terminal_from_inactive_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    ability = make_cognition(autonomy=cognition.Autonomy())
    autonomy_ability = cognition_autonomy(ability)
    assert autonomy_ability is not None
    monkeypatch.setattr(ability, "state", lambda: "/Cognition/processing/intuition")
    monkeypatch.setattr(
        hsm,
        "id",
        lambda actor: "cognition" if actor is ability else "autonomy" if actor is autonomy_ability else "other",
    )
    turn = cognition_turn(
        operation_id="stale-turn",
        generation="stale-operation-token",
    )
    stale = dataclasses.replace(
        autonomy_ability.output_event.with_data(cognition.types.CompletionData(turn=turn, output=no_output("stale"))),
        id="stale-turn:autonomy",
        source="autonomy",
        target="cognition",
    )

    assert not cognition_matches_autonomy_output(hsm.Context(), ability, stale)


def test_cognition_rejects_valid_terminal_from_prior_turn_in_same_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ability = make_cognition()
    intuition_ability = cognition_intuition(ability)
    monkeypatch.setattr(ability, "state", lambda: "/Cognition/processing/intuition")
    monkeypatch.setattr(hsm, "id", lambda actor: "cognition" if actor is ability else "intuition")
    turn = cognition_turn(operation_id="prior-turn", generation="prior-operation-token")
    stale = dataclasses.replace(
        intuition_ability.output_event.with_data(cognition.types.CompletionData(turn=turn, output=no_output("stale"))),
        id="prior-turn:intuition",
        source="intuition",
        target="cognition",
    )

    assert not cognition_matches_intuition_output(hsm.Context(), ability, stale)


def test_cognition_cancel_ack_requires_exact_parent_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    ability = make_cognition()
    monkeypatch.setattr(hsm, "id", lambda actor: "cognition" if actor is ability else "intuition")
    request_id = "cancel-turn:intuition"
    wrong_parent = dataclasses.replace(
        processing.CancelledEvent.with_data(
            processing.CancelledData(
                operation_id=request_id,
                token="expected-token",
                parent_operation_id="another-turn",
            )
        ),
        id=request_id,
        source="intuition",
        target="cognition",
    )

    assert not cognition_matches_child_cancelled(hsm.Context(), ability, wrong_parent)


def test_cognition_cancel_ack_requires_exact_token_and_child(monkeypatch: pytest.MonkeyPatch) -> None:
    ability = make_cognition()
    identities = {
        id(ability): "cognition",
        id(cognition_autonomy(ability)): "autonomy",
        id(cognition_intuition(ability)): "intuition",
        id(cognition_reasoning(ability)): "reasoning",
    }
    monkeypatch.setattr(hsm, "id", lambda actor: identities.get(id(actor), "unknown"))
    parent_id = "cancel-turn"
    child_id = f"{parent_id}:intuition"
    token = "exact-token"
    cancellation_id = processing.cancellation_operation_id(parent_id, child_id, token, "intuition")
    live_operation = typing.cast(processing.Operation, object())
    monkeypatch.setattr(
        processing,
        "active_operation",
        lambda owner, operation_id: live_operation if owner is ability and operation_id == cancellation_id else None,
    )

    def acknowledgement(*, source: str, acknowledged_token: str) -> hsm.Event[typing.Any]:
        return dataclasses.replace(
            processing.CancelledEvent.with_data(
                processing.CancelledData(
                    operation_id=child_id,
                    token=acknowledged_token,
                    parent_operation_id=parent_id,
                )
            ),
            id=child_id,
            source=source,
            target="cognition",
        )

    assert cognition_matches_child_cancelled(
        hsm.Context(), ability, acknowledgement(source="intuition", acknowledged_token=token)
    )
    assert not cognition_matches_child_cancelled(
        hsm.Context(), ability, acknowledgement(source="intuition", acknowledged_token="wrong-token")
    )
    assert not cognition_matches_child_cancelled(
        hsm.Context(), ability, acknowledgement(source="reasoning", acknowledged_token=token)
    )


def test_autonomy_cancellation_requires_attachment_owner_and_preserves_token() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        ability = cognition.Autonomy()
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        intruder = CognitionAttachmentOwner()
        _ = await hsm.started(ctx, owner, owner.model)
        _ = await hsm.started(ctx, intruder, intruder.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        forged = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id="autonomy-operation", token="forged-token")
            ),
            id="autonomy-operation:autonomy",
            source=hsm.id(intruder),
            target=hsm.id(ability),
        )
        _ = await hsm.dispatch(ctx, ability, forged)
        await asyncio.sleep(0)
        assert not owner.lifecycle
        accepted = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id="autonomy-operation", token="exact-token")
            ),
            id="autonomy-operation:autonomy",
            source=hsm.id(owner),
            target=hsm.id(ability),
        )
        _ = await hsm.dispatch(ctx, ability, accepted)
        await wait_until(lambda: bool(owner.lifecycle))
        return owner.lifecycle

    terminals = asyncio.run(run())
    assert terminals == []


def test_intuition_and_reasoning_own_instructions_on_ability() -> None:
    """Instructions live on the ability ClassVar/constructor and stamp onto processor input at apply."""

    async def run() -> tuple[str | None, str | None, str | None]:
        default_processor = RecordingIntuitionProcessor(no_output("ready"))
        custom_processor = RecordingReasoningProcessor(no_output("ready"))
        subclass_processor = RecordingIntuitionProcessor(no_output("ready"))

        class CustomIntuition(cognition.Intuition):
            instructions = "subclass intuition prompt"

        default_intuition = cognition.Intuition(processor=default_processor)
        custom_reasoning = cognition.Reasoning(processor=custom_processor, instructions="reason carefully")
        subclassed = CustomIntuition(processor=subclass_processor)

        assert type(default_intuition).instructions == cognition.intuition.DEFAULT_INSTRUCTIONS
        assert type(custom_reasoning).instructions == cognition.reasoning.DEFAULT_INSTRUCTIONS
        assert type(subclassed).instructions == "subclass intuition prompt"

        default_ctx = await start_cognition_ability_for_test(default_intuition)
        _ = await dispatch_ability_for_test(default_intuition, default_ctx, intuition_input())
        custom_ctx = await start_cognition_ability_for_test(custom_reasoning)
        _ = await dispatch_ability_for_test(custom_reasoning, custom_ctx, reasoning_input())
        subclass_ctx = await start_cognition_ability_for_test(subclassed)
        _ = await dispatch_ability_for_test(subclassed, subclass_ctx, intuition_input())

        default_stamped = default_processor.calls[0].instructions if default_processor.calls else None
        custom_stamped = custom_processor.calls[0].instructions if custom_processor.calls else None
        subclass_stamped = subclass_processor.calls[0].instructions if subclass_processor.calls else None
        return default_stamped, custom_stamped, subclass_stamped

    default_stamped, custom_stamped, subclass_stamped = asyncio.run(run())

    # Default intuition stamps no static prose; only per-turn world XML (when present) is system text.
    assert default_stamped is None or default_stamped == cognition.intuition.DEFAULT_INSTRUCTIONS
    assert custom_stamped == "reason carefully"
    assert subclass_stamped == "subclass intuition prompt"


def test_intuition_output_may_leave_input_unhandled() -> None:
    output = cognition.intuition.OutputData(result=no_output("fast"))
    unhandled = cognition.intuition.OutputData(reason="uncertain")

    assert output.result == no_output("fast")
    assert unhandled.result is None
    assert unhandled.reason == "uncertain"
    assert "confidence" not in cognition.intuition.OutputData.model_fields


def test_reasoning_and_reflection_payloads_carry_typed_decisions() -> None:
    request = reasoning_input()
    reasoning_output = cognition.reasoning.OutputData(result=focus_output("phone", "deliberate"))
    reflection_input = cognition.reflection.InputData(
        cognition_input=cognition_input(),
        cognition_output=reasoning_output.result,
    )
    selection = behavior_create_selection(reason="repeated pattern")
    ops = cognition.reflection.behavior_events()

    assert request.processing_input == deliberative_input()
    assert request.turn.input.stimulus == cognition_input().stimulus
    assert request.turn.input.focus_candidates == ("phone",)
    assert reasoning_output.result == focus_output("phone", "deliberate")
    assert reflection_input.cognition_output == focus_output("phone", "deliberate")
    assert cognition.Reflection.instructions
    assert selection[0].event == behavior_events.CreateEvent.name
    assert {event.name for event in ops} == {
        behavior_events.CreateEvent.name,
        behavior_events.ChangeEvent.name,
        behavior_events.BreakEvent.name,
    }


def test_reflection_behavior_events_require_name() -> None:
    created = behavior_events.CreateData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        reason="repeated pattern",
    )
    assert created.name == "AnswerIncomingRing"
    assert behavior_events.event_for_data(created).name == behavior_events.CreateEvent.name
    with pytest.raises(ValueError):
        _ = behavior_events.ChangeData(name="", reason="revise")


def test_cognition_dispatches_reflection_after_processing_completes() -> None:
    async def run() -> list[processing.InputData]:
        processor = FixedProcessor(no_output("noop"))
        reflection = cognition.Reflection(processor=processor, memory=memory.Memory())
        ability = RecordingCognition(reflection=reflection)
        ctx = await start_cognition_ability_for_test(ability)
        _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(processor.step.calls))
        return processor.step.calls

    calls = asyncio.run(run())

    assert len(calls) == 1
    processor_input = calls[0].input
    assert isinstance(processor_input, cognition.reflection.ProcessorInput)
    assert processor_input.cognition_input.focus is None
    # Default intuition handles with deliberate ignore; reflection still sees that result.
    assert processor_input.cognition_output == ignore_output("fast")
    assert processor_input.prior_episodes == ()
    assert {event.name for event in calls[0].schemas} == {
        behavior_events.CreateEvent.name,
        behavior_events.ChangeEvent.name,
        behavior_events.BreakEvent.name,
    }


def test_cognition_returns_idle_without_waiting_for_reflection() -> None:
    class HangingReflectionProcessor(processing.Processor):
        calls: list[processing.InputData]
        release: asyncio.Event

        def __init__(self) -> None:
            self.calls = []
            self.release = asyncio.Event()

        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            self.calls.append(input)
            await self.release.wait()
            return ()

    async def run() -> tuple[int, str, int]:
        processor = HangingReflectionProcessor()
        reflection = cognition.Reflection(processor=processor, memory=memory.Memory())
        ability = RecordingCognition(reflection=reflection)
        ctx = await start_cognition_ability_for_test(ability)
        input = await started_cognition_input(ctx)

        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(input, "reflection-turn-1"))
        await wait_until(lambda: len(ability.outputs) == 1 and len(processor.calls) == 1)
        first_state = ability.state()
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(input, "reflection-turn-2"))
        await wait_until(lambda: len(ability.outputs) == 2)
        processor.release.set()
        await wait_until(lambda: len(processor.calls) == 2)
        return len(ability.outputs), first_state, len(processor.calls)

    output_count, first_state, reflection_count = asyncio.run(run())

    assert first_state.endswith("/idle")
    assert output_count == 2
    assert reflection_count == 2


def test_reflection_failure_does_not_block_the_next_cognition_turn() -> None:
    class FailingReflectionProcessor(processing.Processor):
        calls: int

        def __init__(self) -> None:
            self.calls = 0

        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            del input
            self.calls += 1
            raise RuntimeError("reflection failed")

    async def run() -> tuple[int, str, int]:
        processor = FailingReflectionProcessor()
        reflection = cognition.Reflection(processor=processor, memory=memory.Memory())
        ability = RecordingCognition(reflection=reflection)
        ctx = await start_cognition_ability_for_test(ability)
        input = await started_cognition_input(ctx)

        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(input, "failed-reflection-1"))
        await wait_until(lambda: len(ability.outputs) == 1 and processor.calls == 1)
        await wait_until(lambda: reflection.state().endswith("/idle"))
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(input, "failed-reflection-2"))
        await wait_until(lambda: len(ability.outputs) == 2)
        return len(ability.outputs), ability.state(), processor.calls

    output_count, state, reflection_calls = asyncio.run(run())

    assert output_count == 2
    assert state.endswith("/idle")
    assert reflection_calls >= 1


def test_reflection_episode_from_turn_captures_stimulus_and_output() -> None:
    turn = cognition.reflection.InputData(
        cognition_input=cognition_input(),
        cognition_output=focus_output("phone", "deliberate"),
    )
    episode = cognition.reflection.episode_from_turn(turn)

    assert episode.focus is None
    assert episode.stimulus_name == bot.InputEvent.name
    assert episode.output == focus_output("phone", "deliberate")


def test_cognitive_outputs_reject_defer_until_resume_is_modeled() -> None:
    deferred = {"kind": "defer", "reason": "wait"}

    with pytest.raises(ValueError):
        _ = cognition.intuition.OutputData.model_validate({"result": deferred})
    with pytest.raises(ValueError):
        _ = cognition.reasoning.OutputData.model_validate({"result": deferred})
    with pytest.raises(ValueError):
        _ = cognition.reflection.InputData.model_validate(
            {
                "cognition_input": {"focus": "phone"},
                "cognition_output": deferred,
            }
        )


def test_cognitive_operation_output_rejects_legacy_payload_wrapper() -> None:
    with pytest.raises(ValueError):
        _ = cognition.types.EventData.model_validate(
            {
                "target": "phone",
                "event": "phone.answer_call",
                "payload": {"call_id": "call-123"},
            }
        )


def test_cognitive_rejects_output_for_unoffered_event() -> None:
    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            intuition_processor=RecordingOutputOperation(
                cognition.types.EventData(event="nope.unavailable", reason="unoffered"),
            ),
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="unavailable event"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/idle"
    assert failures
    assert failures[0].message == "Processing selected unavailable event: nope.unavailable."


def test_cognitive_rejects_output_data_that_does_not_match_offered_event_schema() -> None:
    output = cognition.types.EventData(
        event=bot.FocusDeviceEvent.name,
        data={"device": ""},
        reason="invalid focus target",
    )

    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            intuition_processor=RecordingOutputOperation(output),
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="invalid event data|validation"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/idle"
    assert failures
    assert "device" in failures[0].message.lower() or "string" in failures[0].message.lower()


def test_cognitive_rejects_schema_only_operation_data_that_does_not_match_offered_schema() -> None:
    output = cognition.types.EventData(
        event=bot.FocusDeviceEvent.name,
        data={"device": ""},
        reason="invalid focus target",
    )

    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            intuition_processor=RecordingOutputOperation(output),
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="invalid event data|validation"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/idle"
    assert failures
    assert "device" in failures[0].message.lower() or "string" in failures[0].message.lower()


def test_cognitive_ability_events_use_concrete_pydantic_schemas() -> None:
    assert cognition.Cognition.input_event is cognition.cognition.InputEvent
    assert cognition.Cognition.output_event is cognition.cognition.OutputEvent
    assert cognition.Intuition.input_event is cognition.intuition.InputEvent
    assert cognition.Intuition.output_event is cognition.intuition.OutputEvent
    assert cognition.Reasoning.input_event is cognition.reasoning.InputEvent
    assert cognition.Reasoning.output_event is cognition.reasoning.OutputEvent
    assert cognition.Reflection.input_event is cognition.reflection.InputEvent
    assert cognition.Reflection.output_event is cognition.reflection.OutputEvent

    assert object_dict(cognition.Cognition.input_event.schema) == cognition.InputData.model_json_schema()
    assert (
        object_dict(cognition.Cognition.output_event.schema)
        == pydantic.TypeAdapter(cognition.types.OutputData).json_schema()
    )
    assert object_dict(cognition.Intuition.input_event.schema) == cognition.intuition.InputData.model_json_schema()
    assert object_dict(cognition.Intuition.output_event.schema) == cognition.types.CompletionData.model_json_schema()
    # Reasoning CallEvent remains the model-facing schema; typed host frames ride event data.
    assert object_dict(cognition.Reasoning.input_event.schema) == cognition.reasoning.CallData.model_json_schema()
    assert object_dict(cognition.Reasoning.output_event.schema) == cognition.types.CompletionData.model_json_schema()
    reflection_input_schema = object_dict(cognition.Reflection.input_event.schema)
    reflection_input_properties = reflection_input_schema.get("properties", {})
    assert isinstance(reflection_input_properties, dict)
    assert "cognition_output" in reflection_input_properties
    assert "cognition_input" in cognition.reflection.InputData.model_fields
    reflection_output_schema = object_dict(cognition.Reflection.output_event.schema)
    # Reflection has no host-facing product (null completion); bare type(None) schema.
    assert reflection_output_schema.get("type") == "null" or "null" in str(reflection_output_schema)
    assert "CognitiveEpisode" in cognition.reflection.ProcessorInput.model_json_schema().get("$defs", {})
    assert cognition.Reflection.instructions == cognition.reflection.INSTRUCTIONS
    assert len(cognition.reflection.behavior_events()) == 3


def test_cognitive_output_events_validate_through_typed_schema_contracts() -> None:
    operation_data = {
        "target": "phone",
        "event": bot.FocusDeviceEvent.name,
        "data": {"device": "phone"},
    }

    validated = typing.cast(
        collections.abc.Sequence[object],
        validate_event_data(cognition.cognition.OutputEvent, [operation_data]),
    )
    assert isinstance(validated[0], cognition.types.EventData)
    completion = cognition.types.CompletionData(
        turn=cognition_turn(),
        output=(cognition.types.EventData.model_validate(operation_data),),
    )
    assert validate_event_data(cognition.intuition.OutputEvent, completion) == completion
    with pytest.raises(pydantic.ValidationError):
        _ = validate_event_data(cognition.cognition.OutputEvent, {"kind": "missing"})


def test_cognitive_model_tracks_processing_lifecycle() -> None:
    model = require_model(cognition.Cognition.model)
    view = model_view(model)
    internals = cognition_model_internals(model)

    assert view.qualified_name == "/CognitionLifecycle"
    assert view.initial == "/CognitionLifecycle/.initial"
    assert "/CognitionLifecycle/detached" in view.members
    assert "/CognitionLifecycle/attaching" not in view.members
    assert "/CognitionLifecycle/attached" in view.members
    assert "/CognitionLifecycle/attached/behavior/initializing" in view.members
    assert "/CognitionLifecycle/attached/behavior/idle" in view.members
    processing_path = "/CognitionLifecycle/attached/behavior/processing"
    assert processing_path in view.members
    assert f"{processing_path}/autonomy" in view.members
    assert f"{processing_path}/intuition" in view.members
    assert f"{processing_path}/reasoning" in view.members
    assert f"{processing_path}/cancelling" in view.members
    assert "/CognitionLifecycle/attached/behavior/rebooting" in view.members
    assert "/CognitionLifecycle/attached/behavior/detaching" in view.members
    for obsolete in (
        "routing",
        "autonomizing",
        "intuiting",
        "completing",
        "post_completion",
        "reflecting",
        "resolving_autonomy_cancel",
        "resolving_intuition_cancel",
        "resolving_reasoning_cancel",
        "degraded",
    ):
        assert f"/CognitionLifecycle/attached/behavior/{obsolete}" not in view.members
    initializing_events = view.transition_map["/CognitionLifecycle/attached/behavior/initializing"]
    assert "bot.ability.attachment.terminal" in initializing_events
    assert "bot.ability.cognition.initializing.complete" not in initializing_events
    assert "bot.ability.cognition.input" in view.transition_map["/CognitionLifecycle/attached/behavior/idle"]
    assert "bot.ability.autonomy.output" in view.transition_map[processing_path]
    assert "bot.ability.intuition.output" in view.transition_map[processing_path]
    assert "bot.ability.reasoning.output" in view.transition_map[processing_path]
    assert "*" not in view.transition_map[processing_path]
    assert internals.deferred_map[processing_path]["bot.ability.cognition.input"] == processing_path


def test_cognition_requires_reflection_dependency() -> None:
    parameter = inspect.signature(cognition.Cognition).parameters["reflection"]

    assert parameter.default is inspect.Parameter.empty


def test_cognitive_uses_injected_processing_plan() -> None:
    ability = make_cognition()

    assert not hasattr(ability, "processing")
    assert not hasattr(ability, "reflection")
    assert not hasattr(ability, "intuition")
    assert not hasattr(ability, "reasoning")


def test_cognitive_keeps_injected_processing_plan_private() -> None:
    ability = make_cognition()

    assert not hasattr(ability, "processing")
    assert not hasattr(ability, "use")


def test_cognitive_apply_dispatches_ability_output_event() -> None:
    async def run() -> list[cognition.types.OutputData]:
        ability = RecordingCognition()
        ctx = await start_cognition_ability_for_test(ability)

        _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs

    outputs = asyncio.run(run())

    assert outputs == [no_output("fast")]


def test_cognitive_dispatch_returns_modeled_output_event_result() -> None:
    async def run() -> tuple[cognition.types.OutputData, list[cognition.types.OutputData], str]:
        ability = RecordingCognition()
        ctx = await start_cognition_ability_for_test(ability)

        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, ability.outputs, ability.state()

    result, outputs, state = asyncio.run(run())

    assert result == no_output("fast")
    assert outputs == [no_output("fast")]
    assert state == "/RecordingCognitionLifecycle/attached/behavior/idle"


def test_cognitive_completes_when_trace_metadata_present() -> None:
    async def run() -> tuple[cognition.types.OutputData, list[cognition.types.OutputData]]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        ability = RecordingCognition(
            intuition_processor=RecordingIntuitionProcessor(
                cognition.intuition.OutputData(result=no_output("metadata"))
            )
        )
        ctx = await start_cognition_ability_for_test(ability)
        operation_id = "metadata-operation"
        # Use ability.apply path via dispatch so metadata can ride the event chain.
        from tests.bot.abilities.support import shared_hsm_context
        from tests.hsm_instance_state import ability_terminal_owner

        shared = shared_hsm_context(ctx)
        owner = ability_terminal_owner(ability)
        assert owner is not None
        result_for = getattr(owner, "result_for", None)
        assert callable(result_for)
        result_future = result_for(operation_id)
        assert isinstance(result_future, asyncio.Future)
        _ = await hsm.dispatch(
            shared,
            ability,
            dataclasses.replace(
                ability.input_event.with_data_and_id(cognition_input(), operation_id),
                metadata=metadata,
            ),
        )
        result = await asyncio.wait_for(result_future, timeout=2.0)
        return typing.cast(cognition.types.OutputData, result), ability.outputs

    result, outputs = asyncio.run(run())

    assert result == no_output("metadata")
    assert outputs == [no_output("metadata")]


def test_cognition_intuition_handles_without_reasoning() -> None:
    """Intuition processor handling short-circuits reasoning."""

    async def run() -> tuple[
        cognition.types.OutputData,
        list[processing.InputData],
        list[processing.InputData],
        list[processing.InputData],
    ]:
        intuition = RecordingOutputOperation(focus_output("phone", "reflex"))
        reasoning = FailingReasoningProcessor()
        reflection_processor = FixedProcessor(no_output("observed"))
        ability = RecordingCognition(
            intuition=cognition.Intuition(processor=intuition),
            reasoning=cognition.Reasoning(processor=reasoning),
            reflection=cognition.Reflection(processor=reflection_processor, memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)

        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: bool(reflection_processor.step.calls))
        return result, intuition.calls, reasoning.calls, reflection_processor.step.calls

    result, intuition_calls, reasoning_calls, reflection_calls = asyncio.run(run())

    assert result == focus_output("phone", "reflex")
    assert len(intuition_calls) == 1
    assert reasoning_calls == []
    assert len(reflection_calls) == 1
    reflection_input = reflection_calls[0].input
    assert isinstance(reflection_input, cognition.reflection.ProcessorInput)
    assert reflection_input.cognition_output == result


def test_cognition_continues_to_reasoning_when_intuition_does_not_handle() -> None:
    async def run() -> tuple[
        cognition.types.OutputData,
        list[processing.InputData],
        list[processing.InputData],
        list[processing.InputData],
    ]:
        intuition_processor = RecordingIntuitionProcessor(cognition.intuition.OutputData(reason="ambiguous interrupt"))
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "reasoned focus"))
        )
        reflection_processor = FixedProcessor(no_output("observed"))
        ability = RecordingCognition(
            intuition_processor=intuition_processor,
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=reflection_processor, memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)

        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: bool(reflection_processor.step.calls))
        return result, intuition_processor.calls, reasoning_processor.calls, reflection_processor.step.calls

    result, intuition_calls, reasoning_calls, reflection_calls = asyncio.run(run())

    assert result == focus_output("phone", "reasoned focus")
    assert len(intuition_calls) == 1
    assert len(reasoning_calls) == 1
    assert bot.FocusDeviceEvent.name in {event.name for event in reasoning_calls[0].schemas}
    assert len(reflection_calls) == 1
    reflection_input = reflection_calls[0].input
    assert isinstance(reflection_input, cognition.reflection.ProcessorInput)
    assert reflection_input.cognition_output == result


def test_intuition_low_confidence_escalates_to_reasoning_after_environment_actions() -> None:
    """Self-tuning confidence: low reported confidence → environment dispatch then System 2 cascade."""

    async def run() -> tuple[cognition.types.OutputData, int, float]:
        # Pre-seed tuner so we are past warmup with a high baseline.
        tuner = cognition.intuition.ConfidenceTuner(
            mean=85.0,
            var=100.0,
            n=20,
            floor=35,
            warmup=8,
            k_sigma=1.0,
        )
        actions = focus_output("phone", "tentative")
        intuition_processor = RecordingIntuitionProcessor(actions, confidence=20)
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "deliberate"))
        )
        intuition = cognition.Intuition(processor=intuition_processor, confidence_tuner=tuner)
        ability = RecordingCognition(
            intuition=intuition,
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls), tuner.threshold()

    result, reasoning_calls, threshold = asyncio.run(run())

    assert reasoning_calls == 1
    assert result == focus_output("phone", "deliberate")
    assert 20 < threshold  # low confidence was below adaptive threshold


def test_intuition_reads_confidence_from_patched_event_selections() -> None:
    """Tool-call path: confidence on each event payload is unpatched and drives the tuner."""

    class _PatchedEventsProcessor(processing.Processor):
        @typing.override
        async def process(self, input: processing.InputData) -> processing.Events:
            del input
            return (
                processing.SelectedEvent(
                    event="bot.focus_device",
                    data={"device": "phone", "confidence": 22},
                ),
            )

    async def run() -> tuple[cognition.types.OutputData, int, float, int]:
        tuner = cognition.intuition.ConfidenceTuner(
            mean=85.0,
            var=100.0,
            n=20,
            floor=35,
            warmup=8,
            k_sigma=1.0,
        )
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "deliberate"))
        )
        intuition = cognition.Intuition(processor=_PatchedEventsProcessor(), confidence_tuner=tuner)
        ability = RecordingCognition(
            intuition=intuition,
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls), tuner.threshold(), tuner.n

    result, reasoning_calls, threshold, n = asyncio.run(run())
    assert n >= 21  # observed patched confidence
    assert reasoning_calls == 1
    assert result == focus_output("phone", "deliberate")
    assert 22 < threshold


def test_intuition_empty_dispatch_cascades_to_reasoning() -> None:
    """events: [] / empty product is unhandled → System 2 (not a deliberate pass)."""

    async def run() -> tuple[cognition.types.OutputData, int]:
        intuition_processor = RecordingIntuitionProcessor(empty_output())
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "from-reasoning"))
        )
        ability = RecordingCognition(
            intuition=cognition.Intuition(processor=intuition_processor),
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls)

    result, reasoning_calls = asyncio.run(run())
    assert reasoning_calls == 1
    assert result == focus_output("phone", "from-reasoning")


def test_intuition_high_confidence_skips_reasoning_cascade() -> None:
    """High confidence relative to tuner baseline completes without System 2 cascade."""

    async def run() -> tuple[cognition.types.OutputData, int]:
        tuner = cognition.intuition.ConfidenceTuner(
            mean=50.0,
            var=100.0,
            n=20,
            floor=35,
            warmup=8,
            k_sigma=1.0,
        )
        actions = focus_output("phone", "sure focus")
        intuition_processor = RecordingIntuitionProcessor(
            cognition.intuition.OutputData(result=actions, reason="clear")
        )
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "should not run"))
        )
        intuition = cognition.Intuition(processor=intuition_processor, confidence_tuner=tuner)
        ability = RecordingCognition(
            intuition=intuition,
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls)

    result, reasoning_calls = asyncio.run(run())

    assert result == focus_output("phone", "sure focus")
    assert reasoning_calls == 0


def test_cognition_keeps_reasoning_off_intuition_actor_map() -> None:
    """Cognition owns the intuition-to-reasoning edge."""

    async def run() -> set[str]:
        # Handled ignore (not empty cascade) so we only inspect intuition's offered map.
        processor = RecordingIntuitionProcessor(cognition.intuition.OutputData(result=ignore_output("map-only")))
        ability = RecordingCognition(
            intuition_processor=processor,
            reasoning_processor=RecordingReasoningProcessor(ignore_output("should not run")),
        )
        ctx = await start_cognition_ability_for_test(ability)
        _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        assert processor.calls
        return set(processor.calls[0].actors) | {event.name for event in processor.calls[0].schemas}

    offered = asyncio.run(run())

    assert "reasoning" not in offered
    assert cognition.Reasoning.input_event.name not in offered


def test_intuition_reasoning_selection_cascades_through_cognition() -> None:
    """A reasoning selection becomes unhandled so Cognition owns and awaits the next stage."""

    async def run() -> tuple[cognition.types.OutputData, int, set[str]]:
        reasoning_input_name = cognition.Reasoning.input_event.name
        act_and_reason = (
            *focus_output("phone", "acknowledge"),
            cognition.types.EventData(
                event=reasoning_input_name,
                target="reasoning",
                reason="need deliberate follow-through",
            ),
        )
        intuition_processor = RecordingIntuitionProcessor(cognition.intuition.OutputData(result=act_and_reason))
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "deliberate"))
        )
        ability = RecordingCognition(
            intuition_processor=intuition_processor,
            reasoning_processor=reasoning_processor,
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls), {item.event for item in result}

    result, reasoning_calls, events = asyncio.run(run())

    assert cognition.Reasoning.input_event.name not in events
    assert bot.FocusDeviceEvent.name in events
    assert reasoning_calls == 1


def test_intuition_processor_receives_input() -> None:
    async def run() -> tuple[processing.CompletionData, list[processing.InputData]]:
        processor = RecordingIntuitionProcessor(cognition.intuition.OutputData(result=ignore_output("seen")))
        intuition = cognition.Intuition(processor=processor)
        ctx = await start_cognition_ability_for_test(intuition)

        result = await dispatch_ability_for_test(intuition, ctx, intuition_input())
        return result, processor.calls

    result, calls = asyncio.run(run())

    assert result.output == ignore_output("seen")
    assert len(calls) == 1


def test_reasoning_processor_receives_input() -> None:
    async def run() -> tuple[processing.CompletionData, list[processing.InputData]]:
        processor = RecordingReasoningProcessor(no_output("reasoned"))
        reasoning = cognition.Reasoning(processor=processor)
        ctx = await start_cognition_ability_for_test(reasoning)

        result = await dispatch_ability_for_test(reasoning, ctx, reasoning_input())
        return result, processor.calls

    result, calls = asyncio.run(run())

    assert result.output == no_output("reasoned")
    assert len(calls) == 1


def test_cognitive_routes_processing_failure_to_failed_event() -> None:
    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            intuition_processor=RecordingIntuitionProcessor(
                cognition.intuition.OutputData(reason="ambiguous interrupt")
            ),
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="reasoning failed"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/idle"
    assert failures
    assert "reasoning failed" in failures[-1].message


def test_cognitive_routes_invalid_output_to_failed_event() -> None:
    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            intuition_processor=RecordingIntuitionProcessor(
                cognition.intuition.OutputData(reason="ambiguous interrupt")
            ),
            reasoning_processor=InvalidReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="array of events|output schema"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/idle"
    assert failures
    assert "array of events" in failures[-1].message or "output schema" in failures[-1].message


def test_cognitive_defers_repeated_input_while_processing() -> None:
    async def run() -> tuple[
        list[cognition.types.OutputData],
        list[processing.InputData],
    ]:
        intuition_processor = DelayedIntuitionProcessor(cognition.intuition.OutputData(result=no_output("intuition")))
        ability = RecordingCognition(intuition_processor=intuition_processor)
        ctx = shared_hsm_context()
        ctx = await start_cognition_ability_for_test(ability, ctx)

        async def apply_input() -> None:
            # Cognition is Processing[InputData, OutputData] (FrameData-wrapped TInput) but dispatches raw InputData.
            await typing.cast(typing.Any, ability).apply(cognition_input(), ctx=ctx)

        first = asyncio.create_task(apply_input())
        await wait_until(lambda: len(intuition_processor.calls) == 1)
        second = asyncio.create_task(apply_input())
        await asyncio.sleep(0)

        assert len(intuition_processor.calls) == 1
        intuition_processor.release_first.set()
        _ = await asyncio.gather(first, second)
        await wait_until(lambda: len(ability.outputs) == 2)
        return ability.outputs, intuition_processor.calls

    outputs, intuition_calls = asyncio.run(run())

    assert outputs == [no_output("intuition"), no_output("intuition")]
    assert len(intuition_calls) == 2


def test_intuition_no_operations_returns_to_idle_and_accepts_next_input() -> None:
    async def run() -> tuple[str, processing.CompletionData, processing.CompletionData]:
        intuition = cognition.Intuition(
            processor=RecordingIntuitionProcessor(cognition.intuition.OutputData(result=None, reason="no ops"))
        )
        ctx = shared_hsm_context()
        ctx = await start_cognition_ability_for_test(intuition, ctx)

        first = await dispatch_ability_for_test(intuition, ctx, intuition_input(operation_id="first"))
        second = await dispatch_ability_for_test(intuition, ctx, intuition_input(operation_id="second"))

        return intuition.state(), first, second

    state, first, second = asyncio.run(run())

    assert state == "/IntuitionLifecycle/attached/behavior/idle"
    assert first.output is None
    assert second.output is None


def test_intuition_defers_repeated_input_while_dispatching() -> None:
    async def run() -> tuple[str, list[processing.CompletionData], list[processing.InputData]]:
        processor = DelayedIntuitionProcessor(cognition.intuition.OutputData(result=no_output("intuition")))
        intuition = cognition.Intuition(processor=processor)
        ctx = shared_hsm_context()
        ctx = await start_cognition_ability_for_test(intuition, ctx)

        first = asyncio.create_task(dispatch_ability_for_test(intuition, ctx, intuition_input(operation_id="first")))
        await wait_until(lambda: len(processor.calls) == 1)
        second = asyncio.create_task(dispatch_ability_for_test(intuition, ctx, intuition_input(operation_id="second")))
        await asyncio.sleep(0)

        assert len(processor.calls) == 1
        processor.release_first.set()
        outputs = list(await asyncio.gather(first, second))
        return intuition.state(), outputs, processor.calls

    state, outputs, calls = asyncio.run(run())

    assert state == "/IntuitionLifecycle/attached/behavior/idle"
    assert [output.output for output in outputs] == [no_output("intuition"), no_output("intuition")]
    assert len(calls) == 2


def test_reasoning_defers_repeated_input_while_applying() -> None:
    async def run() -> tuple[str, list[processing.CompletionData], list[processing.InputData]]:
        processor = DelayedReasoningProcessor()
        reasoning = cognition.Reasoning(processor=processor)
        ctx = shared_hsm_context()
        ctx = await start_cognition_ability_for_test(reasoning, ctx)

        first = asyncio.create_task(dispatch_ability_for_test(reasoning, ctx, reasoning_input(operation_id="first")))
        await wait_until(lambda: len(processor.calls) == 1)
        second = asyncio.create_task(dispatch_ability_for_test(reasoning, ctx, reasoning_input(operation_id="second")))
        await asyncio.sleep(0)

        assert len(processor.calls) == 1
        processor.release_first.set()
        outputs = list(await asyncio.gather(first, second))
        return reasoning.state(), outputs, processor.calls

    state, outputs, calls = asyncio.run(run())

    assert state == "/ReasoningLifecycle/attached/behavior/idle"
    assert [output.output for output in outputs] == [ignore_output("reasoned 1"), ignore_output("reasoned 2")]
    assert len(calls) == 2


def test_reasoning_recalls_prior_episodes_and_retains_behavior_episode() -> None:
    """Reasoning SELECTs prior episodes then INSERTs the new episode via SQL transactions."""

    prior = cognition.episodes.CognitiveEpisode(
        focus="phone",
        stimulus_name="environment.sound",
        output=focus_output("phone", "prior answer"),
        behavior=behavior_events.CreateData(
            name="AnswerIncomingRing",
            triggers=("environment.sound",),
            reason="seed",
        ),
    )

    async def run() -> tuple[
        processing.CompletionData,
        list[processing.InputData],
        tuple[cognition.episodes.CognitiveEpisode, ...],
    ]:
        store = memory.Memory()
        seed = cognition.episodes.episode_insert_input(prior, context_ref="phone")
        _ = store.execute(seed)

        processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(
                result=focus_output("phone", "deliberate with memory"),
                create=behavior_events.CreateData(
                    name="AnswerIncomingRing",
                    triggers=("environment.sound",),
                    reason="matches prior episode",
                ),
            )
        )
        reasoning = cognition.Reasoning(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reasoning)
        result = await dispatch_ability_for_test(reasoning, ctx, reasoning_input())

        select = cognition.episodes.episode_select_input(context_ref="phone")
        recalled = store.execute(select)
        stored = cognition.episodes.episodes_from_output(recalled)
        return result, processor.calls, stored

    result, calls, stored = asyncio.run(run())

    assert result.output == focus_output("phone", "deliberate with memory")
    assert len(calls) == 1
    reasoning_processor_input = calls[0].input
    assert isinstance(reasoning_processor_input, cognition.reasoning.ProcessorInput)
    assert reasoning_processor_input.prior_episodes == (prior,)
    assert len(stored) >= 2
    assert prior in stored
    latest = next(
        episode for episode in reversed(stored) if episode.output == focus_output("phone", "deliberate with memory")
    )
    # Behavior inventory create/change/break is Reflection's job; Reasoning retains the episode product.
    assert latest.focus == "phone"


def test_reflection_creates_validates_and_stores_behavior() -> None:
    """Select create → seed broken empty stub → write → store executable behavior + episode."""

    intent = behavior_events.CreateData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        description="Answer when a labeled ring arrives.",
        reason="select intent",
    )
    written = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        description="Answer when a labeled ring arrives (written).",
        reason="repeated ring→answer",
        source=_ANSWER_RING_BEHAVIOR_SOURCE.replace(
            "Select focus_device for the observed pattern.",
            "Answer when a labeled ring arrives (written).",
        ),
    )
    selection: cognition.types.OutputData = (
        cognition.types.EventData(
            event=behavior_events.CreateEvent.name,
            data=intent.model_dump(mode="json"),
            reason="clear repeated pattern",
        ),
    )

    async def run() -> tuple[
        processing.CompletionData | None,
        tuple[cognition.episodes.CognitiveEpisode, ...],
        tuple[str, ...],
        list[str],
        list[processing.InputData],
    ]:
        store = memory.Memory()
        processor = FixedProcessor(selection, write=written)
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        input = cognition.reflection.InputData(
            cognition_input=cognition_input(),
            cognition_output=focus_output("phone", "answered"),
        )
        result = await dispatch_ability_for_test(reflection, ctx, input)

        select = cognition.episodes.episode_select_input(context_ref=None)
        recalled = store.execute(select)
        episodes = cognition.episodes.episodes_from_output(recalled)

        from bot.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = store.execute(behavior_select)
        behavior_sources = _behavior_sources_from_output(behavior_out)
        return result, episodes, behavior_sources, processor.builds, processor.write_step.calls

    result, episodes, behavior_sources, builds, write_calls = asyncio.run(run())

    assert result is None
    assert builds == [
        cognition.Reflection.select_instructions,
        cognition.Reflection.change_instructions,
    ]
    assert len(write_calls) == 1
    write_input = write_calls[0].input
    assert isinstance(write_input, cognition.reflection.ChangeWriteInput)
    assert write_input.existing_behavior.name == "AnswerIncomingRing"
    assert write_input.existing_behavior.source == ""
    assert write_input.existing_behavior.status == "DRAFT"
    assert write_input.intent.name == intent.name
    assert len(episodes) == 1
    applied = episodes[0].behavior
    assert isinstance(applied, behavior_events.CreateData)
    assert applied.name == "AnswerIncomingRing"
    assert applied.source is not None
    assert "hsm.define" in applied.source
    assert "AnswerIncomingRing" in applied.source
    assert len(behavior_sources) == 1
    assert "AnswerIncomingRing" in behavior_sources[0]
    assert "hsm.define" in behavior_sources[0]
    assert "written" in behavior_sources[0]
    # Stored inventory must compile into executable behavior.
    behavior = behavior_events.start(applied.source)
    compiled = behavior_events.build(behavior.source)
    assert compiled.input_event.name == "bot.behavior.answer_incoming_ring.input"


def test_reflection_create_stops_when_fix_yields_same_diagnostics() -> None:
    """Create seeds empty stub; two identical bad writes abandon and leave broken draft."""

    intent = behavior_events.CreateData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        reason="select intent",
    )
    bad_source = """
input_event = hsm.event(name="bot.behavior.answer_incoming_ring.input", schema={"type": "object"})
output_event = hsm.event(name="bot.behavior.answer_incoming_ring.output", schema={"type": "object"})
def focus_phone(event):
    hsm.dispatch(output_event, {"device": "phone"})
behavior = hsm.define(
    "AnswerIncomingRing",
    hsm.initial(
        hsm.transition(hsm.on(input_event), hsm.effect("focus_phone")),
    ),
)
"""
    bad = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        reason="invalid topo",
        source=bad_source,
    )
    selection: cognition.types.OutputData = (
        cognition.types.EventData(
            event=behavior_events.CreateEvent.name,
            data=intent.model_dump(mode="json"),
            reason="pattern",
        ),
    )

    async def run() -> tuple[str, list[processing.InputData], behavior_instance.Status]:
        store = memory.Memory()
        processor = FixedProcessor(selection, write=[bad, bad])
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        with pytest.raises(RuntimeError, match="same diagnostic message after fix") as error:
            _ = await dispatch_ability_for_test(
                reflection,
                ctx,
                cognition.reflection.InputData(
                    cognition_input=cognition_input(),
                    cognition_output=focus_output("phone", "answered"),
                ),
            )
        from bot.behavior import storage as behavior_storage

        behavior_out = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses()))
        )
        behaviors = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in behavior_out.results[0].rows),
            tuple(row.as_mapping() for row in behavior_out.results[1].rows),
        )
        return str(error.value), processor.write_step.calls, behaviors[0].status if behaviors else "ACTIVE"

    message, write_calls, draft_status = asyncio.run(run())

    assert len(write_calls) == 2
    first_write = write_calls[0].input
    second_write = write_calls[1].input
    assert isinstance(first_write, cognition.reflection.ChangeWriteInput)
    assert isinstance(second_write, cognition.reflection.ChangeWriteInput)
    assert first_write.diagnostics is None
    assert first_write.existing_behavior.source == ""
    assert first_write.existing_behavior.status == "DRAFT"
    assert second_write.diagnostics is not None
    assert draft_status == "DRAFT"
    assert "same diagnostic message after fix" in message


def test_reflection_change_retries_while_diagnostics_change_without_fixed_budget() -> None:
    """Changing diagnostics keep repairing; there is no attempt-count budget (only same-error stop)."""

    intent = behavior_events.CreateData(name="BoundedRepair", reason="exercise unbounded fixes")
    selection: cognition.types.OutputData = (
        cognition.types.EventData(
            event=behavior_events.CreateEvent.name,
            data=intent.model_dump(mode="json"),
        ),
    )
    invalid_topology = behavior_events.ChangeData(
        name="BoundedRepair",
        reason="invalid topology",
        source='behavior = hsm.define("BoundedRepair", hsm.initial())',
    )
    invalid_syntax = behavior_events.ChangeData(
        name="BoundedRepair",
        reason="invalid syntax",
        source='behavior = hsm.define("BoundedRepair"',
    )
    # Former max was 3 writes; alternating errors past that must still reach a fix.
    good = behavior_events.ChangeData(
        name="BoundedRepair",
        reason="fixed after alternating diagnostics",
        source=_ANSWER_RING_BEHAVIOR_SOURCE.replace("AnswerIncomingRing", "BoundedRepair").replace(
            "answer_incoming_ring", "bounded_repair"
        ),
    )

    async def run() -> tuple[int, str]:
        store = memory.Memory()
        processor = FixedProcessor(
            selection,
            write=[
                invalid_topology,
                invalid_syntax,
                invalid_topology,
                invalid_syntax,
                good,
            ],
        )
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        _ = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=(),
            ),
        )
        from bot.behavior import storage as behavior_storage

        behavior_out = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses()))
        )
        behaviors = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in behavior_out.results[0].rows),
            tuple(row.as_mapping() for row in behavior_out.results[1].rows),
        )
        status = behaviors[0].status if behaviors else "missing"
        return len(processor.write_step.calls), status

    write_count, status = asyncio.run(run())
    assert write_count == 5
    assert status == "ACTIVE"


def test_reflection_create_fix_pass_rewrites_invalid_source() -> None:
    """Create stub + bad write → diagnostics retry → good write becomes ACTIVE."""

    intent = behavior_events.CreateData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        reason="select intent",
    )
    bad_source = """
input_event = hsm.event(name="bot.behavior.answer_incoming_ring.input", schema={"type": "object"})
output_event = hsm.event(name="bot.behavior.answer_incoming_ring.output", schema={"type": "object"})
def focus_phone(event):
    hsm.dispatch(output_event, {"device": "phone"})
behavior = hsm.define(
    "AnswerIncomingRing",
    hsm.initial(
        hsm.transition(hsm.on(input_event), hsm.effect("focus_phone")),
    ),
)
"""
    bad = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        reason="invalid topo first",
        source=bad_source,
    )
    good = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        description="Fixed after diagnostics.",
        reason="fix pass",
        source=_ANSWER_RING_BEHAVIOR_SOURCE,
    )
    selection: cognition.types.OutputData = (
        cognition.types.EventData(
            event=behavior_events.CreateEvent.name,
            data=intent.model_dump(mode="json"),
            reason="pattern",
        ),
    )

    async def run() -> tuple[
        processing.CompletionData | None, list[processing.InputData], tuple[str, ...], behavior_instance.Status
    ]:
        store = memory.Memory()
        processor = FixedProcessor(selection, write=[bad, good])
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        result = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "answered"),
            ),
        )
        from bot.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = store.execute(behavior_select)
        sources = _behavior_sources_from_output(behavior_out)
        behaviors = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in behavior_out.results[0].rows),
            tuple(row.as_mapping() for row in behavior_out.results[1].rows),
        )
        return result, processor.write_step.calls, sources, behaviors[0].status if behaviors else "DRAFT"

    result, write_calls, sources, status = asyncio.run(run())

    assert result is None
    assert len(write_calls) == 2
    first_write = write_calls[0].input
    second_write = write_calls[1].input
    assert isinstance(first_write, cognition.reflection.ChangeWriteInput)
    assert isinstance(second_write, cognition.reflection.ChangeWriteInput)
    assert first_write.diagnostics is None
    assert first_write.existing_behavior.source == ""
    assert second_write.diagnostics is not None
    assert not second_write.diagnostics.ok
    assert second_write.failed_source is not None
    assert "AnswerIncomingRing" in second_write.failed_source
    assert any(item.code == "E0007" for item in second_write.diagnostics.errors)
    assert len(sources) == 1
    assert "hsm.define" in sources[0]
    assert "Fixed after diagnostics" in sources[0] or "AnswerIncomingRing" in sources[0]
    assert status == "ACTIVE"


def _behavior_sources_from_output(output: memory.OutputData) -> tuple[str, ...]:
    """Source column values from select_all_behaviors_clauses result (statement 0)."""

    if not output.results:
        return ()
    sources: list[str] = []
    for row in output.results[0].rows:
        mapping = row.as_mapping()
        source = mapping.get("source")
        if isinstance(source, str):
            sources.append(source)
    return tuple(sources)


async def _seed_behavior_record(
    store: memory.Memory,
    *,
    name: str = "AnswerIncomingRing",
    triggers: tuple[str, ...] = (),
    description: str = "",
    source: str | None = None,
) -> None:
    """Insert executable behavior into bot_behavior / bot_behavior_trigger tables."""

    from bot.behavior import storage as behavior_storage

    behavior_source = source if source is not None else _ANSWER_RING_BEHAVIOR_SOURCE
    if description:
        behavior = behavior_events.start(
            behavior_source,
            name=name,
            triggers=triggers if triggers else None,
            description=description,
        )
    else:
        behavior = behavior_events.start(
            behavior_source,
            name=name,
            triggers=triggers if triggers else None,
        )
    _ = store.execute(
        memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(behavior)))
    )


def test_reflection_change_loads_existing_and_writes_update() -> None:
    """Select change → changing → replace installed behavior."""

    change_intent = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        reason="select change",
    )
    revised_source = _ANSWER_RING_BEHAVIOR_SOURCE.replace(
        'triggers = ["environment.sound"]',
        'triggers = ["environment.sound", "phone.ringing"]',
    ).replace(
        "Select focus_device for the observed pattern.",
        "Also clear focus before answer.",
    )
    written = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        triggers=("environment.sound", "phone.ringing"),
        description="Also clear focus before answer.",
        reason="revised from episodes",
        source=revised_source,
    )
    selection: cognition.types.OutputData = (
        cognition.types.EventData(
            event=behavior_events.ChangeEvent.name,
            data=change_intent.model_dump(mode="json"),
            reason="revise behavior",
        ),
    )

    async def run() -> tuple[
        processing.CompletionData | None,
        tuple[str, ...],
        list[processing.InputData],
        behavior_events.CreateData | behavior_events.ChangeData | behavior_events.BreakData | None,
    ]:
        store = memory.Memory()
        await _seed_behavior_record(
            store,
            name="AnswerIncomingRing",
            triggers=("environment.sound",),
            description="Original description.",
        )
        processor = FixedProcessor(selection, change_write=written)
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        result = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "revised"),
            ),
        )
        from bot.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = store.execute(behavior_select)
        sources = _behavior_sources_from_output(behavior_out)
        select = cognition.episodes.episode_select_input(context_ref=None)
        episodes = cognition.episodes.episodes_from_output(store.execute(select))
        latest_behavior = episodes[-1].behavior if episodes else None
        return result, sources, processor.change_step.calls, latest_behavior

    result, sources, change_calls, latest_behavior = asyncio.run(run())

    assert result is None
    assert len(change_calls) == 1
    change_input = change_calls[0].input
    assert isinstance(change_input, cognition.reflection.ChangeWriteInput)
    assert change_input.existing_behavior.name == "AnswerIncomingRing"
    assert change_input.existing_behavior.description == "Original description."
    assert change_input.existing_behavior.source
    assert change_input.intent == change_intent
    assert isinstance(latest_behavior, behavior_events.ChangeData)
    assert latest_behavior.name == written.name
    assert latest_behavior.triggers == written.triggers
    assert latest_behavior.description == written.description
    assert latest_behavior.source is not None
    assert "phone.ringing" in latest_behavior.source
    assert len(sources) == 1
    assert "clear focus" in sources[0]
    assert "hsm.define" in sources[0]


def test_reflection_break_marks_behavior_broken_without_write_step() -> None:
    """Select break → mark inventory broken (keep row); no create/change write processor."""

    break_data = behavior_events.BreakData(name="AnswerIncomingRing", reason="harmful")
    selection: cognition.types.OutputData = (
        cognition.types.EventData(
            event=behavior_events.BreakEvent.name,
            data=break_data.model_dump(mode="json"),
            reason="break it",
        ),
    )

    async def run() -> tuple[
        processing.CompletionData | None,
        tuple[str, ...],
        behavior_instance.Status,
        list[str],
        behavior_events.CreateData | behavior_events.ChangeData | behavior_events.BreakData | None,
        int,
    ]:
        store = memory.Memory()
        await _seed_behavior_record(
            store,
            name="AnswerIncomingRing",
            triggers=("environment.sound",),
            description="To be broken.",
        )
        processor = FixedProcessor(selection)
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        result = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "break"),
            ),
        )
        from bot.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = store.execute(behavior_select)
        sources = _behavior_sources_from_output(behavior_out)
        behaviors = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in behavior_out.results[0].rows),
            tuple(row.as_mapping() for row in behavior_out.results[1].rows),
        )
        active_out = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.select_active_behaviors_clauses()))
        )
        active = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in active_out.results[0].rows),
            tuple(row.as_mapping() for row in active_out.results[1].rows),
        )
        select = cognition.episodes.episode_select_input(context_ref=None)
        episodes = cognition.episodes.episodes_from_output(store.execute(select))
        return (
            result,
            sources,
            behaviors[0].status if behaviors else "ACTIVE",
            processor.builds,
            episodes[-1].behavior,
            len(active),
        )

    result, sources, status, builds, latest_behavior, active_count = asyncio.run(run())

    assert result is None
    assert len(sources) == 1
    assert "hsm.define" in sources[0]
    assert status == "BROKEN"
    assert active_count == 0
    # Break is applied after select; change-phase Processing is not used.
    assert builds == [cognition.Reflection.select_instructions]
    assert latest_behavior == break_data


def test_reflection_no_selection_still_stores_episode() -> None:
    async def run() -> tuple[processing.CompletionData | None, tuple[cognition.episodes.CognitiveEpisode, ...]]:
        store = memory.Memory()
        processor = FixedProcessor(empty_output())
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        input = cognition.reflection.InputData(
            cognition_input=cognition_input(),
            cognition_output=ignore_output("nothing to learn"),
        )
        result = await dispatch_ability_for_test(reflection, ctx, input)
        select = cognition.episodes.episode_select_input(context_ref=None)
        recalled = store.execute(select)
        episodes = cognition.episodes.episodes_from_output(recalled)
        return result, episodes

    result, episodes = asyncio.run(run())

    assert result is None
    assert len(episodes) == 1
    assert episodes[0].behavior is None


_FOCUS_RING_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.focus_on_ring.input",
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "audio": {"type": "string"},
        },
    },
)
output_event = hsm.event(
    name = "bot.behavior.focus_on_ring.output",
    schema = {
        "type": "object",
        "properties": {
            "event": {"type": "string"},
            "data": {"type": "object"},
            "reason": {"type": "string"},
        },
        "required": ["event"],
    },
)
triggers = ["environment.sound"]
description = "Focus the phone when a ring sound arrives."

def is_ring(event):
    data = event["data"] or {}
    return data.get("kind") == "phone.ringing"

def focus_phone(event):
    hsm.dispatch(output_event, {
        "event": "bot.focus_device",
        "data": {"device": "phone"},
        "reason": "ring behavior",
    })

behavior = hsm.define(
    "FocusOnRing",
    hsm.initial(hsm.target("/FocusOnRing/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("is_ring"),
            hsm.effect("focus_phone"),
        ),
    ),
)
""".strip()


def _ring_stimulus() -> hsm.Event[object]:
    from bot.environment import SoundData, SoundEvent

    return SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))


def test_autonomy_handles_matching_behavior_without_intuition_processor() -> None:
    """Stimulus → behavior Behavior → EventData selection without deliberative Processing."""

    async def run() -> tuple[list[cognition.types.OutputData], list[processing.InputData], list[processing.InputData]]:
        store = memory.Memory()
        await _seed_behavior_record(
            store,
            name="FocusOnRing",
            triggers=("environment.sound",),
            source=_FOCUS_RING_BEHAVIOR_SOURCE,
        )
        autonomy = cognition.Autonomy(memory=store)
        intuition_processor = RecordingIntuitionProcessor(cognition.intuition.OutputData(reason="should not run"))
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "should not run"))
        reflection_processor = FixedProcessor(no_output("observed"))
        ability = RecordingCognition(
            autonomy=autonomy,
            intuition_processor=intuition_processor,
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=reflection_processor, memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)
        turn = cognition.InputData(
            stimulus=_ring_stimulus(),
            abilities=(),
            actors={"bot": _BotActor()},
            focus=None,
            focus_candidates=("phone",),
        )
        bot_actor = turn.actors["bot"]
        assert isinstance(bot_actor, _BotActor)
        assert bot_actor.model is not None
        _ = await hsm.started(ctx, bot_actor, bot_actor.model)
        _ = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: bool(ability.outputs))
        await wait_until(lambda: bool(reflection_processor.step.calls))
        return ability.outputs, intuition_processor.calls, reflection_processor.step.calls

    outputs, intuition_calls, reflection_calls = asyncio.run(run())

    assert len(outputs) == 1
    assert outputs[0] == (
        cognition.types.EventData(
            event=bot.FocusDeviceEvent.name,
            data={"device": "phone"},
            reason="ring behavior",
        ),
    )
    assert intuition_calls == []
    assert len(reflection_calls) == 1
    reflection_input = reflection_calls[0].input
    assert isinstance(reflection_input, cognition.reflection.ProcessorInput)
    assert reflection_input.cognition_output == outputs[0]


def test_autonomy_unhandled_falls_through_to_intuition() -> None:
    async def run() -> tuple[list[cognition.types.OutputData], list[processing.InputData]]:
        store = memory.Memory()
        await _seed_behavior_record(
            store,
            name="FocusOnRing",
            triggers=("environment.sound",),
            source=_FOCUS_RING_BEHAVIOR_SOURCE,
        )
        autonomy = cognition.Autonomy(memory=store)
        intuition_processor = RecordingIntuitionProcessor(no_output("intuition after autonomy"))
        ability = RecordingCognition(
            autonomy=autonomy,
            intuition_processor=intuition_processor,
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)
        # bot.input stimulus does not match environment.sound trigger.
        _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs, intuition_processor.calls

    outputs, intuition_calls = asyncio.run(run())

    assert outputs == [no_output("intuition after autonomy")]
    assert len(intuition_calls) == 1


def _speech_event_stimulus() -> hsm.Event[object]:
    """Labeled Listening speech product used as cognition stimulus (no priors)."""

    from bot.abilities import listening
    from bot.abilities.hearing import voice

    speech = listening.SpeechData(
        content=bytes([0, 1]) * 160,
        voice_detection=voice.detection.ApplyData(
            segments=(
                voice.detection.VoiceDetectionSegment(
                    start_seconds=0.0,
                    end_seconds=0.02,
                    confidence=0.9,
                ),
            )
        ),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({(0.12, -0.08, 0.31)}),
    )
    return listening.SpeechEvent.with_data(speech)


def _behavior_wires_speech_event_to_conversation(item: object) -> bool:
    """True when an installed behavior is the SpeechEvent → Conversation wire."""

    from bot.abilities import listening
    from bot.behavior import instance as behavior_instance

    if not isinstance(item, behavior_instance.Instance):
        return False
    triggers = tuple(item.triggers or ())
    if listening.SpeechEvent.name not in triggers:
        return False
    source = item.source or ""
    # Seed selects Communication.input (routes to Conversation).
    from bot.abilities import communication

    return communication.InputEvent.name in source


def test_autonomy_seeded_speech_event_selects_conversation_input() -> None:
    """Seeded Communication behavior turns SpeechEvent into communication.input selection."""

    from bot.abilities import listening
    from bot.abilities import communication
    from bot.abilities.communication import conversation
    from bot.abilities.communication.conversation import turn_detector
    from bot.abilities.communication import behaviors
    from bot.abilities.hearing import voice

    async def run() -> tuple[list[cognition.types.OutputData], tuple[str, ...]]:
        store = memory.Memory()
        installed = behaviors.install_seed_behaviors(store)
        autonomy = cognition.Autonomy(memory=store)
        ability = RecordingCognition(
            autonomy=autonomy,
            intuition_processor=RecordingIntuitionProcessor(no_output("should not run")),
            reasoning_processor=FailingReasoningProcessor(),
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)
        bot_actor = _BotActor()
        assert bot_actor.model is not None
        _ = await hsm.started(ctx, bot_actor, bot_actor.model)
        conversation_actor = conversation.Conversation(
            turn_detector=turn_detector.TurnDetector(participant_ref="bot", conversation_ref="ambient")
        )
        communication_actor = communication.Communication(active_conversation=conversation_actor)
        assert conversation_actor.model is not None
        assert communication_actor.model is not None
        _ = await hsm.started(ctx, conversation_actor, conversation_actor.model)
        _ = await hsm.started(ctx, communication_actor, communication_actor.model)
        from bot.protocols import attachment

        _ = await communication_actor.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=ability)),
        )
        await wait_until(lambda: "/behavior/active" in (communication_actor.state() or ""))
        await wait_until(lambda: "/behavior/inactive" in (conversation_actor.state() or ""))

        speech = listening.SpeechData(
            content=bytes([0, 1]) * 160,
            voice_detection=voice.detection.ApplyData(
                segments=(voice.detection.VoiceDetectionSegment(start_seconds=0.0, end_seconds=0.02, confidence=0.9),)
            ),
            sample_rate_hz=16_000,
            channels=1,
            media_type="audio/pcm",
            source_ids=frozenset({(0.12, -0.08, 0.31)}),
        )
        turn = cognition.InputData(
            stimulus=listening.SpeechEvent.with_data(speech),
            abilities=(),
            actors={
                "bot": bot_actor,
                "communication": communication_actor,
                "conversation": conversation_actor,
            },
            focus=None,
            focus_candidates=(),
        )
        _ = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs, installed[0].triggers

    outputs, triggers = asyncio.run(run())
    assert triggers == (listening.SpeechEvent.name,)
    assert len(outputs) == 1
    assert outputs[0] == (
        cognition.types.EventData(
            event=communication.InputEvent.name,
            target=None,
            data={
                "source_ids": [[0.12, -0.08, 0.31]],
                "target_ids": [],
                "content": "AAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAEAAQABAAE=",
                "content_type": "audio/pcm",
                "sample_rate_hz": 16000,
                "channels": 1,
            },
            reason="seeded speech admit via communication",
        ),
    )


def test_cognition_without_priors_time_to_wire_speech_event_to_conversation() -> None:
    """Use the real phone_bot e2e path; fixture processors cannot invent behaviors."""

    import pytest

    pytest.skip(
        "fixture processors short-circuit learning; run "
        "tests/examples/test_phone_bot.py::test_phone_bot_e2e_cognition_wires_speech_event_to_conversation"
    )
