from mosfet import abilities
import mosfet
from mosfet import behavior as behavior_events
from mosfet import event
from mosfet import media
from mosfet.abilities import ability as ability_module
from mosfet.abilities import cognition
from mosfet.abilities import communication
from mosfet.abilities import identity
from mosfet.abilities import listening
from mosfet.abilities import memory
from mosfet.abilities import persona
from mosfet.abilities import processing
from mosfet.abilities import sandbox
from mosfet.abilities import speaking
from mosfet.abilities import vocal
from mosfet.abilities.cognition import cognition as cognition_module
from mosfet.abilities.language import text
from mosfet.abilities.hearing import speech, voice
from mosfet.abilities.cognition import reflection as reflection_module
from mosfet.behavior import instance as behavior_instance
from mosfet.devices import audio
from mosfet.devices import computer
from mosfet.environment import Environment
from mosfet.protocols import attachment
from mosfet.telemetry import span

import asyncio
import collections.abc
import dataclasses
import datetime
import inspect
import itertools
import pathlib
import sqlite3
import typing
import xml.etree.ElementTree

import hsm
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy.engine import Engine
from tests.bot.abilities.memory.memory_fixtures import execute_on
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.hsm_instance_state import start_ability_tree
from tests import metric_reader
import pydantic
import pytest

from mosfet.event import validate_event_data
from tests.type_helpers import model_view, object_dict
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination

CONTEXT_KEY = "tests.cognitive.context"
# Revision validates each Starlark candidate in an isolated process with a three-second budget;
# the host wait scales with the test's expected validation passes plus fixed pipeline overhead.
_STARLARK_VALIDATION_TIMEOUT_SECONDS = 3.0
_REVISION_PIPELINE_OVERHEAD_SECONDS = 10.0
# Learned-source materialization currently takes about eight seconds before attachment.
_LEARNED_BEHAVIOR_HOST_TIMEOUT_SECONDS = 15.0


def _revision_host_timeout(*, validation_passes: int) -> float:
    return validation_passes * _STARLARK_VALIDATION_TIMEOUT_SECONDS + _REVISION_PIPELINE_OVERHEAD_SECONDS


class _NoopSpeechEncoder(vocal.speech.SpeechEncoder):
    def __init__(self) -> None:
        self.format = vocal.speech.Format(sample_rate_hz=16_000, channels=1)

    async def encode(
        self, input: collections.abc.AsyncIterable[vocal.speech.Piece]
    ) -> collections.abc.AsyncIterator[vocal.speech.Audio]:
        yield vocal.speech.Audio(pcm=b"\x00\x00", text="".join([piece.words async for piece in input]))


def test_cognition_never_uses_metadata_for_coordination() -> None:
    assert_metadata_is_not_coordination(cognition_module)


def test_cognition_child_phase_is_topology_owned_not_activity_state() -> None:
    """Child stage correlation must not depend on activity-mutated machine fields."""

    source = inspect.getsource(cognition_module.Cognition)
    assert "_active_child_suffix" not in source


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
def cognition_decision(ability: cognition.Cognition) -> cognition.Decision:
    return typing.cast(cognition.Decision, object.__getattribute__(ability, "_decision"))


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

cognition_matches_decision_output = typing.cast(
    _CognitionGuard, object.__getattribute__(cognition.Cognition, "_matches_decision_output")
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
    """Empty selection product — decision treats as unhandled and cascades to reasoning."""

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
            event=mosfet.FocusDeviceEvent.name,
            data=mosfet.FocusDeviceEventData(device=device).model_dump(),
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
    return mosfet.FocusDeviceEvent


def _accept_focus_event(
    ctx: hsm.Context,
    instance: hsm.Instance,
    event: hsm.Event[typing.Any],
) -> None:
    del ctx, instance, event


class _BotActor(hsm.Instance):
    """Minimal modeled recipient for cognition selections in unit tests."""

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "BotActor",
        hsm.initial(hsm.target("/BotActor/active")),
        hsm.state(
            "active",
            hsm.transition(hsm.on(mosfet.FocusDeviceEvent), hsm.effect(_accept_focus_event)),
        ),
    )

    @typing.override
    async def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> bool:
        # Drops every event: nothing is delivered.
        del ctx, event
        return False


class RecordingDecisionProcessor(processing.DecisionProcessor):
    """Return events (with optional patched confidence) or unhandled OutputData."""

    calls: list[processing.InputData]
    output: processing.Events | cognition.decision.OutputData

    def __init__(
        self,
        output: cognition.types.OutputData
        | cognition.decision.OutputData
        | processing.Events
        | processing.SelectedEvent,
        *,
        confidence: int | None = 99,
    ) -> None:
        self.calls = []
        if isinstance(output, cognition.decision.OutputData):
            if output.result is None:
                self.output = output
            else:
                self.output = _as_events(output.result, confidence=confidence)
        elif isinstance(output, processing.SelectedEvent | tuple):
            self.output = _as_events(output, confidence=confidence)
        else:
            self.output = _as_events(output, confidence=confidence)

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        if isinstance(self.output, cognition.decision.OutputData):
            if self.output.result is None:
                # The pass, as a decision provider answers it: the envelope, with the tier it hands the turn to.
                return typing.cast(processing.Events, typing.cast(object, self.output))
            return _as_events(self.output.result)
        return self.output


class DelayedDecisionProcessor(processing.DecisionProcessor):
    calls: list[processing.InputData]
    cancelled: bool
    release_first: asyncio.Event
    output: processing.Events | cognition.decision.OutputData

    def __init__(
        self,
        output: cognition.types.OutputData | cognition.decision.OutputData,
        *,
        confidence: int | None = 99,
    ) -> None:
        self.calls = []
        self.cancelled = False
        self.release_first = asyncio.Event()
        if isinstance(output, cognition.decision.OutputData):
            if output.result is None:
                self.output = output
            else:
                self.output = _as_events(output.result, confidence=confidence)
        else:
            self.output = _as_events(output, confidence=confidence)

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        if len(self.calls) == 1:
            try:
                _ = await self.release_first.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if isinstance(self.output, cognition.decision.OutputData):
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        raise RuntimeError("reasoning failed")


class InvalidReasoningProcessor(processing.Processor):
    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input
        return _processor_events({"not": "valid-output"})


class RecordingOutputOperation(processing.DecisionProcessor):
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
            selected_data = self.selection[0].data
            raw = {
                **(dict(selected_data) if isinstance(selected_data, collections.abc.Mapping) else {}),
                "event": behavior_events.ChangeEvent.name,
            }
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
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
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
            memory=memory.Memory(engine=memory.store.open_sqlite_engine(connection=connection)),
        )


class RecordingCognition(cognition.Cognition):
    outputs: list[cognition.types.OutputData]
    failures: list[abilities.FailureData]

    def __init__(
        self,
        *,
        decision: cognition.Decision | None = None,
        reasoning: cognition.Reasoning | None = None,
        autonomy: cognition.Autonomy | None = None,
        decision_processor: processing.DecisionProcessor | None = None,
        reasoning_processor: processing.Processor | None = None,
        reflection: cognition.Reflection | None = None,
        store: memory.Memory | None = None,
        associative: memory.Associative | None = None,
        stm_memory: memory.StmMemory | None = None,
        intuition: cognition.Intuition | None = None,
    ) -> None:
        if decision is None or reasoning is None:
            default_i, default_r = cognition_abilities(
                decision_processor=decision_processor,
                reasoning_processor=reasoning_processor,
            )
            decision = decision or default_i
            reasoning = reasoning or default_r
        reflection = reflection or cognition.Reflection(
            processor=FixedProcessor(no_output("reflection")),
            memory=store if store is not None else memory.Memory(),
        )
        super().__init__(
            decision=decision,
            intuition=intuition,
            reasoning=reasoning,
            autonomy=autonomy,
            reflection=reflection,
            memory=store,
            associative=associative,
            stm_memory=stm_memory,
        )
        self.outputs = []
        self.failures = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
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
        stimulus=mosfet.InputEventData(target_device="phone", priority=0),
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
    _ = await mosfet.started(ctx, actor, actor.model)
    return data


def deliberative_input(
    *,
    schemas: tuple[hsm.Event[object], ...] | None = None,
) -> processing.InputData:
    """Deliberative input for nested processors (decision/reasoning/reflection)."""

    return processing.InputData(
        input=mosfet.InputEventData(target_device="phone", priority=0),
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


def decision_input(
    input: processing.InputData | None = None,
    *,
    operation_id: str = "test-turn",
    generation: str = "operation-token",
) -> cognition.decision.InputData:
    return cognition.decision.InputData(
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
    decision_processor: processing.DecisionProcessor | None = None,
    reasoning_processor: processing.Processor | None = None,
) -> tuple[cognition.Decision, cognition.Reasoning]:
    decision = cognition.Decision(
        processor=decision_processor
        or RecordingDecisionProcessor(cognition.decision.OutputData(result=no_output("fast"))),
    )
    reasoning = cognition.Reasoning(
        processor=reasoning_processor or RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
    )
    return decision, reasoning


def make_cognition(
    *,
    decision_processor: processing.DecisionProcessor | None = None,
    reasoning_processor: processing.Processor | None = None,
    reflection: cognition.Reflection | None = None,
    autonomy: cognition.Autonomy | None = None,
) -> cognition.Cognition:
    decision, reasoning = cognition_abilities(
        decision_processor=decision_processor,
        reasoning_processor=reasoning_processor,
    )
    return cognition.Cognition(
        decision=decision,
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
        child = cognition_decision(ability)
        turn = cognition_turn(operation_id="forged-turn", generation="forged-operation-token")
        forged = dataclasses.replace(
            child.output_event.with_data(cognition.types.CompletionData(turn=turn, output=None)),
            id="forged-turn:decision",
            source="forged-child",
            target=hsm.id(ability),
        )
        return cognition_matches_decision_output(ctx, ability, forged)

    assert not asyncio.run(run())


def test_reflection_rejects_child_terminal_from_wrong_source() -> None:
    async def run() -> bool:
        ability = cognition.Reflection(
            processor=RecordingDecisionProcessor(no_output("unused")), memory=memory.Memory()
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

    model: typing.ClassVar[hsm.Model] = mosfet.define(
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
            hsm.transition(hsm.on(mosfet.RebootEvent), hsm.effect(_record)),
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
    decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=no_output("fast")))
    reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "deliberate"))
    decision = cognition.Decision(processor=decision_processor)
    reasoning = cognition.Reasoning(processor=reasoning_processor)
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        reflection = RecordingReflectionAbility(connection)

        assert isinstance(decision, abilities.Ability)
        assert isinstance(decision, processing.Processing)
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
        decision, reasoning = cognition_abilities()
        minimal = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(
                processor=FixedProcessor(no_output("reflect")),
                memory=memory.Memory(),
            ),
        )
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            maximal = cognition.Cognition(
                autonomy=cognition.Autonomy(),
                decision=cognition.Decision(processor=RecordingDecisionProcessor(no_output("fast"))),
                reasoning=cognition.Reasoning(processor=RecordingReasoningProcessor(no_output("slow"))),
                reflection=cognition.Reflection(
                    processor=FixedProcessor(no_output("reflect")),
                    memory=memory.Memory(engine=memory.store.open_sqlite_engine(connection=connection)),
                ),
            )
        finally:
            connection.close()
        owner = hsm.Instance()
        owner_model = mosfet.define(
            "CognitionAttachmentOwner",
            hsm.initial(hsm.target("ready")),
            hsm.state("ready"),
        )
        ctx = hsm.Context()
        _ = await mosfet.started(ctx, owner, owner_model)
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
            members for members in groups if members and isinstance(members[0], cognition.Autonomy | cognition.Decision)
        ]
        return cognition_groups, len(requests), state

    groups, request_count, state = asyncio.run(run())

    assert len(groups) == 2
    assert len(groups[0]) == 3
    assert isinstance(groups[0][0], cognition.Decision)
    assert isinstance(groups[0][1], cognition.Reasoning)
    assert isinstance(groups[0][2], cognition.Reflection)
    assert len(groups[1]) == 4
    assert isinstance(groups[1][0], cognition.Autonomy)
    assert isinstance(groups[1][1], cognition.Decision)
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
        _ = await mosfet.started(ctx, owner, owner.model)
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
        _ = await mosfet.started(ctx, owner, owner.model)
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
    assert state == "/CognitionLifecycle/attached/behavior/ready/idle"


def test_cognition_detach_cancels_active_decision_through_group() -> None:
    async def run() -> tuple[bool, list[hsm.Event[typing.Any]], str]:
        processor = DelayedDecisionProcessor(no_output("held"))
        ability = make_cognition(decision_processor=processor)
        ctx = hsm.Context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data(await started_cognition_input(ctx)))
        await wait_until(lambda: bool(processor.calls))
        assert ability.state().endswith("/decision")
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


@pytest.mark.parametrize("child", ["decision", "reasoning"])
def test_cognition_cancel_waits_for_correlated_active_child(child: str) -> None:
    async def run() -> tuple[bool, list[hsm.Event[typing.Any]], str]:
        decision_processor = DelayedDecisionProcessor(no_output("held"))
        reasoning_processor = DelayedReasoningProcessor()
        if child == "reasoning":
            ability = make_cognition(
                decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="escalate")),
                reasoning_processor=reasoning_processor,
            )
        else:
            ability = make_cognition(
                decision_processor=decision_processor,
                reasoning_processor=reasoning_processor,
            )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner(ability)
        _ = await mosfet.started(ctx, owner, owner.model)
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

        cancelled = decision_processor.cancelled if child == "decision" else reasoning_processor.cancelled
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


@pytest.mark.parametrize("phase", ["autonomy", "decision", "reasoning"])
def test_cognition_direct_cancellation_acknowledges_active_child(
    phase: str,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, tuple[str, ...]]:
        decision_processor: processing.DecisionProcessor
        if phase == "reasoning":
            decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(reason="escalate"))
        else:
            decision_processor = DelayedDecisionProcessor(no_output("held"))
        decision, reasoning = cognition_abilities(
            decision_processor=decision_processor,
            reasoning_processor=DelayedReasoningProcessor(),
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(
                processor=FixedProcessor(no_output("reflection")),
                memory=memory.Memory(),
            ),
            autonomy=cognition.Autonomy() if phase == "autonomy" else None,
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
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
    class StubbornDecision(cognition.Decision):
        @staticmethod
        async def _hold(ctx: hsm.Context, instance: "StubbornDecision", event: hsm.Event[typing.Any]) -> None:
            del ctx
            operation_id = event.id if event.id else hsm.id(instance)
            if processing.active_operation(instance, operation_id) is None:
                _ = await processing.start_operation(instance, operation_id)
            await asyncio.Event().wait()

        submodel = mosfet.define(
            "StubbornDecision",
            hsm.initial(hsm.target("/StubbornDecision/waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(cognition.Decision.input_event),
                    hsm.target("/StubbornDecision/holding"),
                ),
            ),
            hsm.state(
                "holding",
                hsm.activity(_hold.__get__(None, object)),
            ),
        )

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, tuple[str, ...]]:
        monkeypatch.setattr(cognition_module, "_CANCEL_TEARDOWN_TIMEOUT", datetime.timedelta(milliseconds=10))
        stubborn = StubbornDecision(processor=RecordingDecisionProcessor(no_output("unused")))
        _, reasoning = cognition_abilities()
        ability = cognition.Cognition(
            decision=stubborn,
            reasoning=reasoning,
            reflection=cognition.Reflection(
                processor=FixedProcessor(no_output("reflection")),
                memory=memory.Memory(),
            ),
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
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
        await wait_until(lambda: ability.state().endswith("/decision"))
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
        await wait_until(lambda: any(event.name == mosfet.RebootEvent.name for event in owner.lifecycle))
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

    reboots = [event for event in lifecycle if event.name == mosfet.RebootEvent.name]
    assert len(reboots) == 1
    assert reboots[0].data == mosfet.RebootEventData(reason="cognition_cancel_teardown_failed")
    assert state.endswith("/rebooting")
    assert operation_keys == ()


class _SlowFirstStepProcessor(processing.DecisionProcessor):
    """A local model whose first step outlasts any step timeout, then answers every later one at once."""

    calls: int
    cancelled: int
    _hangs: bool

    def __init__(self, *, hangs: bool) -> None:
        self.calls = 0
        self.cancelled = 0
        self._hangs = hangs

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        self.calls += 1
        if self._hangs and self.calls == 1:
            try:
                _ = await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return ()


class _StepTimeoutBody(hsm.Instance):
    """Body stand-in: records how Cognition settles its turns and what it hands the body to perceive."""

    told: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.told = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "_StepTimeoutBody", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.told.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "StepTimeoutBody",
        hsm.initial(hsm.target("ready")),
        hsm.state(
            "ready",
            hsm.transition(hsm.on(cognition.InputEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(cognition.OutputEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(cognition.Cognition.failed_event), hsm.effect(_record)),
            hsm.transition(hsm.on(mosfet.RebootEvent), hsm.effect(_record)),
        ),
    )


@pytest.mark.parametrize("stage", ["decision", "reasoning"])
def test_a_step_that_outlasts_the_step_timeout_fails_its_turn_as_a_typed_failure_the_bot_perceives(
    stage: typing.Literal["decision", "reasoning"],
) -> None:
    """D-R-3 (live: a 31.8 s decision rebooted the bot and destroyed its bench): a slow step fails
    only its turn. The step is stopped, the turn is remembered and reported as a typed failure, the
    bot perceives that failure as its next turn, and the same children serve the turn after it."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, str, _SlowFirstStepProcessor, dict[str, memory.Episode]]:
        slow = _SlowFirstStepProcessor(hangs=True)
        store = memory.Memory()
        decision, reasoning = cognition_abilities(
            decision_processor=slow if stage == "decision" else _SlowFirstStepProcessor(hangs=False),
            reasoning_processor=slow if stage == "reasoning" else None,
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
            child_operation_timeout=datetime.timedelta(milliseconds=500),
        )
        ctx = shared_hsm_context()
        body = _StepTimeoutBody()
        _ = await mosfet.started(ctx, body, body.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=body)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        turn = (await started_cognition_input(ctx)).model_copy(update={"participants": (_ADA,)})
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "slow-turn"))
        await wait_until(lambda: any(event.name == cognition.InputEvent.name for event in body.told))
        await wait_until(lambda: ability.state().endswith("/idle"))
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "next-turn"))
        await wait_until(lambda: any(event.name == cognition.OutputEvent.name for event in body.told))
        episodes = {episode.turn_id: episode for episode in await _remembered(ctx, store)}
        return list(body.told), ability.state(), hsm.id(ability), slow, episodes

    told, state, cognition_id, slow, episodes = asyncio.run(run())

    assert [event for event in told if event.name == mosfet.RebootEvent.name] == []
    assert state.endswith("/idle")
    (failed,) = [event for event in told if event.name == cognition.Cognition.failed_event.name]
    assert failed.id == "slow-turn"
    timed_out = failed.data
    assert isinstance(timed_out, cognition.StepTimedOutData)
    # The decision layer is reflex's (D-R-52).
    named = "reflex" if stage == "decision" else stage
    assert (timed_out.stage, timed_out.timeout_seconds, timed_out.turn_id, timed_out.stimulus) == (
        named,
        0.5,
        "slow-turn",
        mosfet.InputEvent.name,
    )
    assert timed_out.message == f"Cognition's {named} step took longer than 0.5 seconds, so the turn failed."
    (perceived,) = [event for event in told if event.name == cognition.InputEvent.name]
    assert perceived.source == cognition_id
    assert isinstance(perceived.data, cognition.InputData)
    assert perceived.data.participants == (_ADA,)
    stimulus = perceived.data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == cognition.StepTimedOutEvent.name
    assert stimulus.data == timed_out
    assert (slow.calls, slow.cancelled) == (2, 1)
    assert [event.id for event in told if event.name == cognition.OutputEvent.name] == ["next-turn"]
    assert episodes["slow-turn"].outcome is memory.Outcome.FAILED
    _assert_keeps_what_was_said(episodes["slow-turn"])
    assert episodes["next-turn"].outcome is memory.Outcome.COMPLETED


class _FailingStepProcessor(processing.DecisionProcessor):
    """A model step that fails outright, every time, the way a malformed dispatch failed reasoning live."""

    calls: int

    def __init__(self) -> None:
        self.calls = 0

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input, drafts
        self.calls += 1
        raise RuntimeError("the step failed")


def test_a_step_that_fails_is_a_typed_failure_the_bot_perceives_but_failing_on_it_is_not_handed_back() -> None:
    """D-R-13 (live round 2, E3): after a reasoning step failed (not timed out), no turn ran until the
    person typed again, and the screen said "writing code…" indefinitely. A failed step is now handed
    back as a typed failure the bot perceives as its next turn, like a timed-out one; a step that fails
    while the bot perceives that failure is not handed back again. (A failing decision hands its turn
    to reasoning instead, D-R-36.)"""

    stage = "reasoning"

    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        failing = _FailingStepProcessor()
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="needs thought")),
            reasoning_processor=failing,
        )
        store = memory.Memory()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
        )
        ctx = shared_hsm_context()
        body = _StepTimeoutBody()
        _ = await mosfet.started(ctx, body, body.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=body)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        turn = (await started_cognition_input(ctx)).model_copy(update={"participants": (_ADA,)})
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "failing-turn"))
        await wait_until(lambda: any(event.name == cognition.InputEvent.name for event in body.told))
        await wait_until(lambda: ability.state().endswith("/idle"))
        (perceived,) = [event for event in body.told if event.name == cognition.InputEvent.name]
        assert isinstance(perceived.data, cognition.InputData)
        # The bot takes up the failure as a turn of its own, and that step fails too.
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(perceived.data, "perceiving-turn"))
        await wait_until(
            lambda: len([event for event in body.told if event.name == cognition.Cognition.failed_event.name]) == 2
        )
        await wait_until(lambda: ability.state().endswith("/idle"))
        for _ in range(50):
            await asyncio.sleep(0)
        return list(body.told), hsm.id(ability)

    told, cognition_id = asyncio.run(run())

    assert [event.id for event in told if event.name == cognition.Cognition.failed_event.name] == [
        "failing-turn",
        "perceiving-turn",
    ]
    (perceived,) = [event for event in told if event.name == cognition.InputEvent.name]
    assert perceived.source == cognition_id
    assert isinstance(perceived.data, cognition.InputData)
    assert perceived.data.participants == (_ADA,)
    stimulus = perceived.data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == cognition.StepFailedEvent.name
    failed = stimulus.data
    assert isinstance(failed, cognition.StepFailedData)
    assert (failed.stage, failed.turn_id, failed.stimulus) == (stage, "failing-turn", mosfet.InputEvent.name)
    assert failed.message == f"Cognition's {stage} step failed, so the turn failed: the step failed"


class _UnreadableOnce(hsm.Instance):
    """A live actor whose state, once armed, cannot be read the next time, so only decision's frame fails."""

    effort: typing.ClassVar[abilities.ability.Effort] = abilities.ability.Effort.XS
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "UnreadableOnce",
        hsm.initial(hsm.target("/UnreadableOnce/active")),
        hsm.state("active"),
    )
    armed: bool = False

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        if self.armed:
            self.armed = False
            raise RuntimeError("actor state unavailable")
        return super().take_snapshot()


@pytest.mark.parametrize("failing", ["decider", "frame"])
def test_a_failing_decision_hands_its_turn_to_reasoning_in_the_same_turn(
    failing: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-R-36 (companion run 7, D-2): decide refused an over-long request, the turn failed at decision, and reasoning
    never saw what was said. An decision whose decider fails, or whose frame cannot be built, now hands the same
    turn to reasoning: nothing is handed back, and the turn is remembered as reasoning settles it."""

    def cascaded(reader: metric_reader.Reader) -> float:
        return sum(
            point.value
            for point in reader.points("bot.cognition.step.failure.count")
            if point.attributes.get("stage") == "reflex"
            and point.attributes.get("kind") == "failed"
            and point.attributes.get("outcome") == "cascaded"
        )

    async def run() -> tuple[list[hsm.Event[typing.Any]], cognition.InputData, list[processing.InputData], str]:
        reasoning_processor = RecordingReasoningProcessor(no_output("reasoned"))
        decision, reasoning = cognition_abilities(
            decision_processor=_FailingStepProcessor() if failing == "decider" else None,
            reasoning_processor=reasoning_processor,
        )
        store = memory.Memory()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
        )
        ctx = shared_hsm_context()
        body = _StepTimeoutBody()
        _ = await mosfet.started(ctx, body, body.model)
        _ = await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=body)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        turn = (await started_cognition_input(ctx)).model_copy(update={"participants": (_ADA,)})
        if failing == "frame":
            actor = _UnreadableOnce()
            assert actor.model is not None
            _ = await mosfet.started(ctx, actor, actor.model)
            actor.armed = True
            turn = turn.model_copy(update={"actors": {**turn.actors, "broken": actor}})
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "cascading-turn"))
        await wait_until(lambda: any(event.name == cognition.OutputEvent.name for event in body.told))
        await wait_until(lambda: ability.state().endswith("/idle"))
        for _ in range(50):
            await asyncio.sleep(0)
        (kept,) = await _remembered(ctx, store)
        return list(body.told), turn, reasoning_processor.calls, kept.outcome.value

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = cascaded(reader)
        told, turn, reasoning_calls, outcome = asyncio.run(run())
        after = cascaded(reader)

    (frame,) = reasoning_calls
    assert isinstance(frame.input, cognition.reasoning.ProcessorInput)
    assert frame.input.host_input.input == turn.stimulus
    assert [event.name for event in told] == [cognition.OutputEvent.name]
    (output,) = told
    assert (output.id, output.data) == ("cascading-turn", no_output("reasoned"))
    assert outcome == memory.Outcome.COMPLETED.value
    assert after - before == 1.0


def test_a_step_is_given_a_local_models_time_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-R-3: one step of a local model with thinking on takes 30 s and more (live: 31.8 s), so a
    cognition built without a step timeout waits at least 120 s for each of its steps."""

    hops: list[tuple[str, datetime.timedelta]] = []
    run_terminal_operation = ability_module.run_terminal_operation

    async def recorded(
        ctx: hsm.Context,
        *,
        child: hsm.Instance,
        request: hsm.Event[typing.Any] | collections.abc.Callable[[hsm.Instance], hsm.Event[typing.Any]],
        terminals: tuple[hsm.Event[typing.Any], ...],
        timeout: datetime.timedelta,
        on_terminal: collections.abc.Callable[[hsm.Event[typing.Any]], None] | None = None,
    ) -> hsm.Event[typing.Any]:
        if isinstance(child, cognition.Decision | cognition.Reasoning):
            hops.append((type(child).__name__, timeout))
        return await run_terminal_operation(
            ctx, child=child, request=request, terminals=terminals, timeout=timeout, on_terminal=on_terminal
        )

    monkeypatch.setattr(ability_module, "run_terminal_operation", recorded)

    async def run() -> None:
        # Decision leaves the turn to reasoning, so both steps are taken.
        ability = make_cognition(decision_processor=_SlowFirstStepProcessor(hangs=False))
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        _ = await hsm.dispatch(
            ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn")
        )
        await wait_until(lambda: len(hops) >= 2 and ability.state().endswith("/idle"))

    asyncio.run(run())

    assert [step for step, _ in hops] == ["Decision", "Reasoning"]
    assert all(timeout >= datetime.timedelta(seconds=120) for _, timeout in hops)


def test_each_tier_is_bounded_by_its_own_timeout_or_else_the_step_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """D-R-52: ``bot.timeouts(reflex=, intuition=, reasoning=)`` bound each tier's steps; one left unset is
    bounded by the step timeout."""

    hops: list[tuple[str, datetime.timedelta]] = []
    run_terminal_operation = ability_module.run_terminal_operation

    async def recorded(
        ctx: hsm.Context,
        *,
        child: hsm.Instance,
        request: hsm.Event[typing.Any] | collections.abc.Callable[[hsm.Instance], hsm.Event[typing.Any]],
        terminals: tuple[hsm.Event[typing.Any], ...],
        timeout: datetime.timedelta,
        on_terminal: collections.abc.Callable[[hsm.Event[typing.Any]], None] | None = None,
    ) -> hsm.Event[typing.Any]:
        if isinstance(child, cognition.Decision | cognition.Intuition | cognition.Reasoning):
            hops.append((type(child).__name__, timeout))
        return await run_terminal_operation(
            ctx, child=child, request=request, terminals=terminals, timeout=timeout, on_terminal=on_terminal
        )

    monkeypatch.setattr(ability_module, "run_terminal_operation", recorded)
    handoff = cognition.types.EventData(target="cognition", event=cognition.Reasoning.input_event.name, data={})

    async def run() -> None:
        ability = cognition.Cognition(
            decision=cognition.Decision(
                processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=None))
            ),
            intuition=cognition.Intuition(processor=RecordingReasoningProcessor((handoff,))),
            reasoning=cognition.Reasoning(processor=RecordingReasoningProcessor(no_output("reasoned"))),
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
            child_operation_timeout=datetime.timedelta(seconds=150),
            intuition_timeout=datetime.timedelta(seconds=20),
            reasoning_timeout=datetime.timedelta(seconds=180),
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        _ = await hsm.dispatch(
            ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn")
        )
        await wait_until(lambda: len(hops) >= 3 and ability.state().endswith("/idle"))

    asyncio.run(run())

    assert hops == [
        ("Decision", datetime.timedelta(seconds=150)),
        ("Intuition", datetime.timedelta(seconds=20)),
        ("Reasoning", datetime.timedelta(seconds=180)),
    ]


def test_cognition_settled_child_terminal_is_not_abandoned_by_parent_deadline() -> None:
    """An instant child hop completes through the one-shot; a settled hop is not rebooted."""

    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ability = RecordingCognition(decision_processor=RecordingOutputOperation(no_output("settled")))
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            ability,
            ability.input_event.with_data_and_id(await started_cognition_input(ctx), "settled-turn"),
        )
        await wait_until(lambda: ability.state().endswith("/idle") or ability.state().endswith("/rebooting"))
        return list(owner.lifecycle), ability.state()

    lifecycle, state = asyncio.run(run())

    reboots = [event for event in lifecycle if event.name == mosfet.RebootEvent.name]
    assert reboots == []
    assert state.endswith("/idle")


def test_cognition_forwards_reflection_reboot_to_bot_owner() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ability = make_cognition()
        owner = CognitionAttachmentOwner()
        ctx = hsm.Context()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        request = dataclasses.replace(
            mosfet.RebootEvent.with_data(mosfet.RebootEventData(reason="cognition_child_teardown_failed")),
            id="reflection-reboot",
            source=hsm.id(cognition_reflection(ability)),
            target=hsm.id(ability),
        )

        _ = await hsm.dispatch(ctx, ability, request)
        await wait_until(lambda: bool(owner.lifecycle))
        return list(owner.lifecycle), ability.state()

    lifecycle, state = asyncio.run(run())

    assert len(lifecycle) == 1
    assert lifecycle[0].name == mosfet.RebootEvent.name
    assert lifecycle[0].id == "reflection-reboot"
    assert lifecycle[0].data == mosfet.RebootEventData(reason="cognition_child_teardown_failed")
    assert state.endswith("/rebooting")


def test_cancelled_turn_cannot_cancel_the_next_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    class TwoTurnHangingProcessor(processing.DecisionProcessor):
        calls: list[processing.InputData]
        cancelled_calls: list[int]
        releases: tuple[asyncio.Event, asyncio.Event]

        def __init__(self) -> None:
            self.calls = []
            self.cancelled_calls = []
            self.releases = (asyncio.Event(), asyncio.Event())

        @typing.override
        async def process(
            self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
        ) -> processing.Events:
            index = len(self.calls)
            self.calls.append(input)
            try:
                await self.releases[index].wait()
            except asyncio.CancelledError:
                self.cancelled_calls.append(index)
                raise
            return ()

    async def run() -> tuple[list[int], str]:
        monkeypatch.setattr(cognition_module.types, "STEP_TIMEOUT", datetime.timedelta(milliseconds=30))
        processor = TwoTurnHangingProcessor()
        ability = make_cognition(decision_processor=processor)
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
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
        monkeypatch.setattr(cognition_module.types, "STEP_TIMEOUT", datetime.timedelta(seconds=1))

        _ = await hsm.dispatch(
            ctx,
            ability,
            ability.input_event.with_data_and_id(await started_cognition_input(ctx), "second-turn"),
        )
        await wait_until(lambda: len(processor.calls) == 2)
        result = list(processor.cancelled_calls), ability.state()
        processor.releases[1].set()
        return result

    cancelled_calls, state = asyncio.run(run())

    assert cancelled_calls == [0]
    assert state.endswith("/decision")


def test_cognition_ignores_stale_public_child_terminal() -> None:
    async def run() -> tuple[list[cognition.types.OutputData], str]:
        processor = DelayedDecisionProcessor(no_output("held"))
        decision, reasoning = cognition_abilities(decision_processor=processor)
        ability = RecordingCognition(decision=decision, reasoning=reasoning)
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        current_input = await started_cognition_input(ctx)
        _ = await hsm.dispatch(
            ctx,
            ability,
            ability.input_event.with_data_and_id(current_input, "current-turn"),
        )
        await wait_until(lambda: ability.state().endswith("/decision"))
        stale = dataclasses.replace(
            decision.output_event.with_data(_as_events(focus_output("phone", "stale"))),
            id="old-turn:decision",
            source=hsm.id(decision),
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
    assert state.endswith("/decision")


def test_cognition_rejects_terminal_from_inactive_sibling(monkeypatch: pytest.MonkeyPatch) -> None:
    ability = make_cognition(autonomy=cognition.Autonomy())
    autonomy_ability = cognition_autonomy(ability)
    assert autonomy_ability is not None
    monkeypatch.setattr(ability, "state", lambda: "/Cognition/processing/decision")
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
    decision_ability = cognition_decision(ability)
    monkeypatch.setattr(ability, "state", lambda: "/Cognition/processing/decision")
    monkeypatch.setattr(hsm, "id", lambda actor: "cognition" if actor is ability else "decision")
    turn = cognition_turn(operation_id="prior-turn", generation="prior-operation-token")
    stale = dataclasses.replace(
        decision_ability.output_event.with_data(cognition.types.CompletionData(turn=turn, output=no_output("stale"))),
        id="prior-turn:decision",
        source="decision",
        target="cognition",
    )

    assert not cognition_matches_decision_output(hsm.Context(), ability, stale)


def test_cognition_cancel_ack_requires_exact_parent_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    ability = make_cognition()
    monkeypatch.setattr(hsm, "id", lambda actor: "cognition" if actor is ability else "decision")
    request_id = "cancel-turn:decision"
    wrong_parent = dataclasses.replace(
        processing.CancelledEvent.with_data(
            processing.CancelledData(
                operation_id=request_id,
                token="expected-token",
                parent_operation_id="another-turn",
            )
        ),
        id=request_id,
        source="decision",
        target="cognition",
    )

    assert not cognition_matches_child_cancelled(hsm.Context(), ability, wrong_parent)


def test_cognition_cancel_ack_requires_exact_token_and_child(monkeypatch: pytest.MonkeyPatch) -> None:
    ability = make_cognition()
    identities = {
        id(ability): "cognition",
        id(cognition_autonomy(ability)): "autonomy",
        id(cognition_decision(ability)): "decision",
        id(cognition_reasoning(ability)): "reasoning",
    }
    monkeypatch.setattr(hsm, "id", lambda actor: identities.get(id(actor), "unknown"))
    parent_id = "cancel-turn"
    child_id = f"{parent_id}:decision"
    token = "exact-token"
    cancellation_id = processing.cancellation_operation_id(parent_id, child_id, token, "decision")
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
        hsm.Context(), ability, acknowledgement(source="decision", acknowledged_token=token)
    )
    assert not cognition_matches_child_cancelled(
        hsm.Context(), ability, acknowledgement(source="decision", acknowledged_token="wrong-token")
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
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await mosfet.started(ctx, intruder, intruder.model)
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


def test_decision_and_reasoning_own_instructions_on_ability() -> None:
    """Instructions live on the ability ClassVar/constructor and stamp onto processor input at apply."""

    async def run() -> tuple[str | None, str | None, str | None]:
        default_processor = RecordingDecisionProcessor(no_output("ready"))
        custom_processor = RecordingReasoningProcessor(no_output("ready"))
        subclass_processor = RecordingDecisionProcessor(no_output("ready"))

        class CustomDecision(cognition.Decision):
            instructions = "subclass decision prompt"

        default_decision = cognition.Decision(processor=default_processor)
        custom_reasoning = cognition.Reasoning(processor=custom_processor, instructions="reason carefully")
        subclassed = CustomDecision(processor=subclass_processor)

        assert type(default_decision).instructions == cognition.decision.DEFAULT_INSTRUCTIONS
        assert type(custom_reasoning).instructions == cognition.reasoning.DEFAULT_INSTRUCTIONS
        assert type(subclassed).instructions == "subclass decision prompt"

        default_ctx = await start_cognition_ability_for_test(default_decision)
        _ = await dispatch_ability_for_test(default_decision, default_ctx, decision_input())
        custom_ctx = await start_cognition_ability_for_test(custom_reasoning)
        _ = await dispatch_ability_for_test(custom_reasoning, custom_ctx, reasoning_input())
        subclass_ctx = await start_cognition_ability_for_test(subclassed)
        _ = await dispatch_ability_for_test(subclassed, subclass_ctx, decision_input())

        default_stamped = default_processor.calls[0].instructions if default_processor.calls else None
        custom_stamped = custom_processor.calls[0].instructions if custom_processor.calls else None
        subclass_stamped = subclass_processor.calls[0].instructions if subclass_processor.calls else None
        return default_stamped, custom_stamped, subclass_stamped

    default_stamped, custom_stamped, subclass_stamped = asyncio.run(run())

    # Default decision stamps no static prose; only per-turn world XML (when present) is system text.
    assert default_stamped is None or default_stamped == cognition.decision.DEFAULT_INSTRUCTIONS
    assert custom_stamped == "reason carefully"
    assert subclass_stamped == "subclass decision prompt"


def test_decision_output_may_leave_input_unhandled() -> None:
    output = cognition.decision.OutputData(result=no_output("fast"))
    unhandled = cognition.decision.OutputData(reason="uncertain")

    assert output.result == no_output("fast")
    assert unhandled.result is None
    assert unhandled.reason == "uncertain"
    assert "confidence" not in cognition.decision.OutputData.model_fields


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
    # Default decision handles with deliberate ignore; reflection still sees that result.
    assert processor_input.cognition_output == ignore_output("fast")
    assert processor_input.associations is None
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
        async def process(
            self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
        ) -> processing.Events:
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
        async def process(
            self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
        ) -> processing.Events:
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


def test_cognitive_outputs_reject_defer_until_resume_is_modeled() -> None:
    deferred = {"kind": "defer", "reason": "wait"}

    with pytest.raises(ValueError):
        _ = cognition.decision.OutputData.model_validate({"result": deferred})
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
        # Decision leaves the turn to reasoning, whose selection is rejected (a rejected decision would cascade).
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="needs thought")),
            reasoning_processor=RecordingOutputOperation(
                cognition.types.EventData(event="nope.unavailable", reason="unoffered"),
            ),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="unavailable event"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/ready/idle"
    assert failures
    assert failures[0].message == "Selection rejected: Processing selected unavailable event: nope.unavailable."


def test_cognitive_rejects_output_data_that_does_not_match_offered_event_schema() -> None:
    output = cognition.types.EventData(
        event=mosfet.FocusDeviceEvent.name,
        data={"device": ""},
        reason="invalid focus target",
    )

    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            decision_processor=RecordingOutputOperation(output),
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/ready/idle"
    assert failures
    expected_message = (
        "Selection rejected: Processing selected invalid focus selection data for event: bot.focus_device: "
        "bot.focus_device|device:string_too_short:String should have at least 1 character"
    )
    assert failures[0].message == expected_message
    assert "input_value" not in failures[0].message
    assert "secret" not in failures[0].message.lower()


def test_cognitive_rejects_schema_only_operation_data_that_does_not_match_offered_schema() -> None:
    output = cognition.types.EventData(
        event=mosfet.FocusDeviceEvent.name,
        data={"device": ""},
        reason="invalid focus target",
    )

    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            decision_processor=RecordingOutputOperation(output),
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/ready/idle"
    assert failures
    expected_message = (
        "Selection rejected: Processing selected invalid focus selection data for event: bot.focus_device: "
        "bot.focus_device|device:string_too_short:String should have at least 1 character"
    )
    assert failures[0].message == expected_message
    assert "input_value" not in failures[0].message
    assert "secret" not in failures[0].message.lower()


def test_cognitive_ability_events_use_concrete_pydantic_schemas() -> None:
    assert cognition.Cognition.input_event is cognition.cognition.InputEvent
    assert cognition.Cognition.output_event is cognition.cognition.OutputEvent
    assert cognition.Decision.input_event is cognition.decision.InputEvent
    assert cognition.Decision.output_event is cognition.decision.OutputEvent
    assert cognition.Reasoning.input_event is cognition.reasoning.InputEvent
    assert cognition.Reasoning.output_event is cognition.reasoning.OutputEvent
    assert cognition.Reflection.input_event is cognition.reflection.InputEvent
    assert cognition.Reflection.output_event is cognition.reflection.OutputEvent

    assert object_dict(cognition.Cognition.input_event.schema) == cognition.InputData.model_json_schema()
    assert (
        object_dict(cognition.Cognition.output_event.schema)
        == pydantic.TypeAdapter(cognition.types.OutputData).json_schema()
    )
    assert object_dict(cognition.Decision.input_event.schema) == cognition.decision.InputData.model_json_schema()
    assert object_dict(cognition.Decision.output_event.schema) == cognition.types.CompletionData.model_json_schema()
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
    assert "AssociationsData" in cognition.reflection.ProcessorInput.model_json_schema().get("$defs", {})
    assert cognition.Reflection.instructions == cognition.reflection.INSTRUCTIONS
    assert len(cognition.reflection.behavior_events()) == 3


def test_cognitive_output_events_validate_through_typed_schema_contracts() -> None:
    operation_data = {
        "target": "phone",
        "event": mosfet.FocusDeviceEvent.name,
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
    assert validate_event_data(cognition.decision.OutputEvent, completion) == completion
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
    assert "/CognitionLifecycle/attached/behavior/ready/idle" in view.members
    processing_path = "/CognitionLifecycle/attached/behavior/processing"
    assert processing_path in view.members
    assert f"{processing_path}/reflex" in view.members
    assert f"{processing_path}/reflex/behaviors" in view.members
    assert f"{processing_path}/reflex/decision" in view.members
    assert f"{processing_path}/associating" in view.members
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
        "resolving_decision_cancel",
        "resolving_reasoning_cancel",
        "degraded",
    ):
        assert f"/CognitionLifecycle/attached/behavior/{obsolete}" not in view.members
    initializing_events = view.transition_map["/CognitionLifecycle/attached/behavior/initializing"]
    assert "bot.ability.attachment.terminal" in initializing_events
    assert "bot.ability.cognition.initializing.complete" not in initializing_events
    assert "bot.ability.cognition.input" in view.transition_map["/CognitionLifecycle/attached/behavior/ready/idle"]
    assert "bot.ability.cognition.input" in view.transition_map["/CognitionLifecycle/attached/behavior/ready/resting"]
    assert "bot.ability.autonomy.output" in view.transition_map[f"{processing_path}/reflex/behaviors"]
    assert "bot.ability.decision.output" in view.transition_map[f"{processing_path}/reflex/decision"]
    assert "bot.ability.reasoning.output" in view.transition_map[f"{processing_path}/reasoning"]
    assert "*" not in view.transition_map[processing_path]
    assert internals.deferred_map[processing_path]["bot.ability.cognition.input"] == processing_path


def test_cognition_requires_reflection_dependency() -> None:
    parameter = inspect.signature(cognition.Cognition).parameters["reflection"]

    assert parameter.default is inspect.Parameter.empty


def test_cognitive_uses_injected_processing_plan() -> None:
    ability = make_cognition()

    assert not hasattr(ability, "processing")
    assert not hasattr(ability, "reflection")
    assert not hasattr(ability, "decision")
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
    assert state == "/RecordingCognitionLifecycle/attached/behavior/ready/idle"


def test_cognitive_completes_when_trace_metadata_present() -> None:
    async def run() -> tuple[cognition.types.OutputData, list[cognition.types.OutputData]]:
        metadata = {"traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"}
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=no_output("metadata")))
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


def test_cognition_decision_handles_without_reasoning() -> None:
    """Decision processor handling short-circuits reasoning."""

    async def run() -> tuple[
        cognition.types.OutputData,
        list[processing.InputData],
        list[processing.InputData],
        list[processing.InputData],
    ]:
        decision = RecordingOutputOperation(focus_output("phone", "reflex"))
        reasoning = FailingReasoningProcessor()
        reflection_processor = FixedProcessor(no_output("observed"))
        ability = RecordingCognition(
            decision=cognition.Decision(processor=decision),
            reasoning=cognition.Reasoning(processor=reasoning),
            reflection=cognition.Reflection(processor=reflection_processor, memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)

        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: bool(reflection_processor.step.calls))
        return result, decision.calls, reasoning.calls, reflection_processor.step.calls

    result, decision_calls, reasoning_calls, reflection_calls = asyncio.run(run())

    assert result == focus_output("phone", "reflex")
    assert len(decision_calls) == 1
    assert reasoning_calls == []
    assert len(reflection_calls) == 1
    reflection_input = reflection_calls[0].input
    assert isinstance(reflection_input, cognition.reflection.ProcessorInput)
    assert reflection_input.cognition_output == result


def test_cognition_continues_to_reasoning_when_decision_does_not_handle() -> None:
    async def run() -> tuple[
        cognition.types.OutputData,
        list[processing.InputData],
        list[processing.InputData],
        list[processing.InputData],
    ]:
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(reason="ambiguous interrupt"))
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "reasoned focus"))
        )
        reflection_processor = FixedProcessor(no_output("observed"))
        ability = RecordingCognition(
            decision_processor=decision_processor,
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=reflection_processor, memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)

        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: bool(reflection_processor.step.calls))
        return result, decision_processor.calls, reasoning_processor.calls, reflection_processor.step.calls

    result, decision_calls, reasoning_calls, reflection_calls = asyncio.run(run())

    assert result == focus_output("phone", "reasoned focus")
    assert len(decision_calls) == 1
    assert len(reasoning_calls) == 1
    assert mosfet.FocusDeviceEvent.name in {event.name for event in reasoning_calls[0].schemas}
    assert len(reflection_calls) == 1
    reflection_input = reflection_calls[0].input
    assert isinstance(reflection_input, cognition.reflection.ProcessorInput)
    assert reflection_input.cognition_output == result


def test_decision_low_confidence_escalates_to_reasoning_after_environment_actions() -> None:
    """Self-tuning confidence: low reported confidence → environment dispatch then System 2 cascade.

    D-R-52: what the decision layer dispatched before passing the turn on rides its ``acted``: reasoning is told
    it as a fact of the turn, and the turn's output puts it first.
    """

    async def run() -> tuple[cognition.types.OutputData, list[processing.InputData], float]:
        # Pre-seed tuner so we are past warmup with a high baseline.
        tuner = cognition.decision.ConfidenceTuner(
            mean=85.0,
            var=100.0,
            n=20,
            floor=35,
            warmup=8,
            k_sigma=1.0,
        )
        actions = focus_output("phone", "tentative")
        decision_processor = RecordingDecisionProcessor(actions, confidence=20)
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "deliberate"))
        )
        decision = cognition.Decision(processor=decision_processor, confidence_tuner=tuner)
        ability = RecordingCognition(
            decision=decision,
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, reasoning_processor.calls, tuner.threshold()

    result, reasoning_calls, threshold = asyncio.run(run())

    (frame,) = reasoning_calls
    assert "tentative" in (frame.situation or "")
    assert result == (*focus_output("phone", "tentative"), *focus_output("phone", "deliberate"))
    assert 20 < threshold  # low confidence was below adaptive threshold


def test_decision_reads_confidence_from_patched_event_selections() -> None:
    """Tool-call path: confidence on each event payload is unpatched and drives the tuner."""

    class _PatchedEventsProcessor(processing.DecisionProcessor):
        @typing.override
        async def process(
            self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
        ) -> processing.Events:
            del input
            return (
                processing.SelectedEvent(
                    event="bot.focus_device",
                    data={"device": "phone", "confidence": 22},
                ),
            )

    async def run() -> tuple[cognition.types.OutputData, int, float, int]:
        tuner = cognition.decision.ConfidenceTuner(
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
        decision = cognition.Decision(processor=_PatchedEventsProcessor(), confidence_tuner=tuner)
        ability = RecordingCognition(
            decision=decision,
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls), tuner.threshold(), tuner.n

    result, reasoning_calls, threshold, n = asyncio.run(run())
    assert n >= 21  # observed patched confidence
    assert reasoning_calls == 1
    # The focus the decision layer dispatched before passing the turn on comes first (D-R-52).
    assert [item.reason for item in result] == [None, "deliberate"]
    assert 22 < threshold


def test_decision_empty_dispatch_cascades_to_reasoning() -> None:
    """events: [] / empty product is unhandled → System 2 (not a deliberate pass)."""

    async def run() -> tuple[cognition.types.OutputData, int]:
        decision_processor = RecordingDecisionProcessor(empty_output())
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "from-reasoning"))
        )
        ability = RecordingCognition(
            decision=cognition.Decision(processor=decision_processor),
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls)

    result, reasoning_calls = asyncio.run(run())
    assert reasoning_calls == 1
    assert result == focus_output("phone", "from-reasoning")


def test_decision_high_confidence_skips_reasoning_cascade() -> None:
    """High confidence relative to tuner baseline completes without System 2 cascade."""

    async def run() -> tuple[cognition.types.OutputData, int]:
        tuner = cognition.decision.ConfidenceTuner(
            mean=50.0,
            var=100.0,
            n=20,
            floor=35,
            warmup=8,
            k_sigma=1.0,
        )
        actions = focus_output("phone", "sure focus")
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=actions, reason="clear"))
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "should not run"))
        )
        decision = cognition.Decision(processor=decision_processor, confidence_tuner=tuner)
        ability = RecordingCognition(
            decision=decision,
            reasoning=cognition.Reasoning(processor=reasoning_processor),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls)

    result, reasoning_calls = asyncio.run(run())

    assert result == focus_output("phone", "sure focus")
    assert reasoning_calls == 0


def test_cognition_keeps_reasoning_off_decision_actor_map() -> None:
    """Cognition owns the decision-to-reasoning edge."""

    async def run() -> set[str]:
        # Handled ignore (not empty cascade) so we only inspect decision's offered map.
        processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=ignore_output("map-only")))
        ability = RecordingCognition(
            decision_processor=processor,
            reasoning_processor=RecordingReasoningProcessor(ignore_output("should not run")),
        )
        ctx = await start_cognition_ability_for_test(ability)
        _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        assert processor.calls
        return set(processor.calls[0].actors) | {event.name for event in processor.calls[0].schemas}

    offered = asyncio.run(run())

    assert "reasoning" not in offered
    assert cognition.Reasoning.input_event.name not in offered


def test_decision_reasoning_selection_cascades_through_cognition() -> None:
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
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=act_and_reason))
        reasoning_processor = RecordingReasoningProcessor(
            cognition.reasoning.OutputData(result=focus_output("phone", "deliberate"))
        )
        ability = RecordingCognition(
            decision_processor=decision_processor,
            reasoning_processor=reasoning_processor,
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, len(reasoning_processor.calls), {item.event for item in result}

    result, reasoning_calls, events = asyncio.run(run())

    assert cognition.Reasoning.input_event.name not in events
    assert mosfet.FocusDeviceEvent.name in events
    assert reasoning_calls == 1


class _ShowData(pydantic.BaseModel):
    text: str = pydantic.Field(description="What the desk shows.")


_ShowEvent = hsm.Event[_ShowData](name="tests.cognition.desk.show", kind=event.EventKind, schema=_ShowData)


class _DeskActor(hsm.Instance):
    """A display within decision's ceiling that keeps what it was asked to show."""

    shown: list[str]

    @staticmethod
    def _show(ctx: hsm.Context, instance: "_DeskActor", event: hsm.Event[typing.Any]) -> None:
        del ctx
        assert isinstance(event.data, _ShowData)
        instance.shown.append(event.data.text)

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "CognitionDeskActor",
        hsm.initial(hsm.target("/CognitionDeskActor/active")),
        hsm.state("active", hsm.transition(hsm.on(_ShowEvent), hsm.effect(_show))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.shown = []


class _BenchActor(hsm.Instance):
    """An M-effort bench enabling the real sandbox run command, above decision's ceiling (S)."""

    effort: typing.ClassVar[ability_module.Effort] = ability_module.Effort.M
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "CognitionBenchActor",
        hsm.initial(hsm.target("/CognitionBenchActor/active")),
        hsm.state("active", hsm.transition(hsm.on(sandbox.RunEvent), hsm.effect(_accept_focus_event))),
    )


class _MenuBoundDecisionProcessor(processing.DecisionProcessor):
    """Selects events by name the way a provider's dispatch tool does: one not on the turn's menu is refused."""

    calls: list[processing.InputData]
    selections: tuple[tuple[str, dict[str, object]], ...]

    def __init__(self, *selections: tuple[str, dict[str, object]]) -> None:
        self.calls = []
        self.selections = selections

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.calls.append(input)
        return processing.events_from_dispatch_args(
            {"events": [{"event": name, "data": {**data, "confidence": 95}} for name, data in self.selections]},
            patch=input.patch,
            offered=input.schemas,
            targets_by_event=input.actor_events,
        )


def test_the_decision_layer_is_never_offered_the_handoff_and_its_pass_hands_the_turn_on() -> None:
    """D-R-52: the decision layer's pass is its hand-on, so it is never offered the deliberative handoff.

    It is still told, in ``<held>``, what a later tier holds above its ceiling, reached by passing the turn on;
    its pass reaches reasoning, which holds the bench and never offers the handoff into itself.
    """

    async def run() -> tuple[processing.InputData, list[processing.InputData], list[abilities.FailureData]]:
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=None, reason="pass"))
        reasoning_processor = RecordingReasoningProcessor(ignore_output("deliberated"))
        ability = RecordingCognition(decision_processor=decision_processor, reasoning_processor=reasoning_processor)
        ctx = await start_cognition_ability_for_test(ability)
        turn_input = await started_cognition_input(ctx)
        desk = _DeskActor()
        bench = _BenchActor()
        for actor in (desk, bench):
            assert actor.model is not None
            _ = await mosfet.started(ctx, actor, actor.model)
        turn_input = turn_input.model_copy(update={"actors": {**turn_input.actors, "desk": desk, "sandbox": bench}})
        _ = await dispatch_ability_for_test(ability, ctx, turn_input)
        await wait_until(lambda: ability.state().endswith("/idle"))
        (frame,) = decision_processor.calls
        return frame, reasoning_processor.calls, ability.failures

    frame, reasoning_calls, failures = asyncio.run(run())

    handoff = cognition.Reasoning.input_event.name
    offered = {schema.name for schema in frame.schemas}
    assert handoff not in offered
    assert sandbox.RunEvent.name not in offered
    assert '<held reached_by="passing this turn on to a later tier of thinking">' in (frame.instructions or "")
    assert failures == []
    (deliberate,) = reasoning_calls
    reasoning_offers = {schema.name for schema in deliberate.schemas}
    # Reasoning holds the bench, and never offers the handoff into itself.
    assert sandbox.RunEvent.name in reasoning_offers
    assert handoff not in reasoning_offers


def test_the_decision_layer_is_offered_no_handoff_when_nothing_is_held() -> None:
    async def run() -> processing.InputData:
        decision_processor = _MenuBoundDecisionProcessor((cognition.types.IgnoreEvent.name, {}))
        ability = RecordingCognition(decision_processor=decision_processor)
        ctx = await start_cognition_ability_for_test(ability)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        (frame,) = decision_processor.calls
        return frame

    frame = asyncio.run(run())

    assert cognition.Reasoning.input_event.name not in {schema.name for schema in frame.schemas}
    assert "<held" not in (frame.instructions or "")


def _intuited(
    intuition_answer: cognition.types.OutputData,
    *,
    intuition_processor: processing.Processor | None = None,
    decided: cognition.decision.OutputData | None = None,
) -> tuple[
    cognition.types.OutputData,
    list[processing.InputData],
    list[processing.InputData],
    list[str],
    list[abilities.FailureData],
]:
    """A turn the decision layer passes, taken up by intuition answering ``intuition_answer`` at a desk."""

    async def run() -> tuple[
        cognition.types.OutputData,
        list[processing.InputData],
        list[processing.InputData],
        list[str],
        list[abilities.FailureData],
    ]:
        intuition_processor_ = intuition_processor or RecordingReasoningProcessor(intuition_answer)
        reasoning_processor = RecordingReasoningProcessor(ignore_output("deliberated"))
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(decided or cognition.decision.OutputData(result=None)),
            intuition=cognition.Intuition(processor=intuition_processor_),
            reasoning_processor=reasoning_processor,
        )
        ctx = await start_cognition_ability_for_test(ability)
        turn_input = await started_cognition_input(ctx)
        desk = _DeskActor()
        assert desk.model is not None
        _ = await mosfet.started(ctx, desk, desk.model)
        turn_input = turn_input.model_copy(update={"actors": {**turn_input.actors, "desk": desk}})
        result = await dispatch_ability_for_test(ability, ctx, turn_input)
        await wait_until(lambda: ability.state().endswith("/idle"))
        calls = (
            typing.cast(RecordingReasoningProcessor, intuition_processor_).calls if intuition_processor is None else []
        )
        return result, calls, reasoning_processor.calls, desk.shown, ability.failures

    return asyncio.run(run())


def _show(text: str) -> cognition.types.OutputData:
    return (cognition.types.EventData(target="desk", event=_ShowEvent.name, data={"text": text}),)


def test_a_decision_model_choosing_reasoning_lands_the_turn_in_reasoning_with_recall_and_skips_intuition() -> None:
    """R-23: the fallback reflex chooses the tier its pass hands the turn to. Choosing reasoning, the turn is
    associated first (recall runs before either tier) and reasoning takes it; intuition is never asked."""

    async def run() -> tuple[
        cognition.types.OutputData, list[processing.InputData], list[processing.InputData], list[processing.InputData]
    ]:
        store = memory.Memory()
        decision_processor = RecordingDecisionProcessor(
            cognition.decision.OutputData(result=None, hand_to="reasoning", reason="it needs thought")
        )
        intuition_processor = RecordingReasoningProcessor(no_output("intuited"))
        reasoning_processor = RecordingReasoningProcessor(no_output("reasoned"))
        ability = RecordingCognition(
            decision_processor=decision_processor,
            intuition=cognition.Intuition(processor=intuition_processor),
            reasoning_processor=reasoning_processor,
            store=store,
            associative=memory.Associative(memory=store),
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        await _remember_by_message(
            ctx,
            store,
            memory.EpisodeData(
                turn_id="earlier",
                parts=(text.TextPart(text="Ada asked for a call at noon."),),
                participants=(_ADA,),
                outcome=memory.Outcome.COMPLETED,
            ),
        )
        turn = (await started_cognition_input(ctx)).model_copy(update={"participants": (_ADA,)})
        result = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: ability.state().endswith("/idle"))
        return result, decision_processor.calls, intuition_processor.calls, reasoning_processor.calls

    result, decision_calls, intuition_calls, reasoning_calls = asyncio.run(run())

    assert result == no_output("reasoned")
    (decided,) = decision_calls
    assert decided.hand_ons == ("intuition", "reasoning")
    assert intuition_calls == []
    (reasoned,) = reasoning_calls
    assert isinstance(reasoned.recalled, cognition.RecalledData)
    assert isinstance(reasoned.recalled.associations, memory.AssociationsData)
    assert [item.text for item in reasoned.recalled.associations.items] == ["Ada asked for a call at noon."]


@pytest.mark.parametrize("chosen", ["intuition", None], ids=["chose-intuition", "no-choice-keeps-intuition"])
def test_a_pass_to_intuition_or_without_a_choice_lands_the_turn_in_intuition(
    chosen: typing.Literal["intuition"] | None,
) -> None:
    """R-23: choosing intuition, or a hand-on that names no tier (an empty product, low confidence), keeps intuition."""

    result, intuition_calls, reasoning_calls, _, _ = _intuited(
        _show("Hello."), decided=cognition.decision.OutputData(result=None, hand_to=chosen)
    )

    assert len(intuition_calls) == 1
    assert reasoning_calls == []
    assert result == _show("Hello.")


def test_the_decision_layer_is_offered_reasoning_alone_when_no_intuition_is_composed() -> None:
    async def run() -> list[processing.InputData]:
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=None))
        ability = RecordingCognition(decision_processor=decision_processor)
        ctx = await start_cognition_ability_for_test(ability)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return decision_processor.calls

    (decided,) = asyncio.run(run())

    assert decided.hand_ons == ("reasoning",)


def test_intuition_takes_up_what_reflex_hands_on_and_settles_the_turn_in_words() -> None:
    """D-R-52: reflex → intuition → reasoning; intuition selecting anything but the handoff settles the turn."""

    result, intuition_calls, reasoning_calls, shown, failures = _intuited(_show("Hello."))

    assert failures == []
    assert shown == ["Hello."]
    assert result == _show("Hello.")
    (frame,) = intuition_calls
    assert cognition.Reasoning.input_event.name in {schema.name for schema in frame.schemas}
    assert reasoning_calls == []


def test_intuition_selecting_the_handoff_says_its_words_first_and_reasoning_knows_it_did() -> None:
    """D-R-52: the handoff runs intuition's other selections first; they ride the turn's ``acted`` to reasoning,
    and the turn's output puts them first."""

    handoff = cognition.types.EventData(target="cognition", event=cognition.Reasoning.input_event.name, data={})
    result, _, reasoning_calls, shown, failures = _intuited((*_show("Let me think."), handoff))

    assert failures == []
    assert shown == ["Let me think."]
    (frame,) = reasoning_calls
    assert "Let me think." in (frame.situation or "")
    assert cognition.Reasoning.input_event.name not in {schema.name for schema in frame.schemas}
    assert result == (*_show("Let me think."), *ignore_output("deliberated"))


def test_a_failing_intuition_hands_its_turn_to_reasoning_in_the_same_turn(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-R-52: an intuition that fails cascades to reasoning, counted as ``stage="intuition"`` cascaded."""

    def cascaded(reader: metric_reader.Reader) -> float:
        return sum(
            point.value
            for point in reader.points("bot.cognition.step.failure.count")
            if point.attributes.get("stage") == "intuition" and point.attributes.get("outcome") == "cascaded"
        )

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = cascaded(reader)
        result, _, reasoning_calls, _, failures = _intuited((), intuition_processor=FailingReasoningProcessor())
        after = cascaded(reader)

    # Only intuition's own failure reached cognition; the turn did not fail.
    assert [type(failure) for failure in failures] == [cognition.types.FailureData]
    assert len(reasoning_calls) == 1
    assert result == ignore_output("deliberated")
    assert after - before == 1.0


def test_decision_processor_receives_input() -> None:
    async def run() -> tuple[processing.CompletionData, list[processing.InputData]]:
        processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=ignore_output("seen")))
        decision = cognition.Decision(processor=processor)
        ctx = await start_cognition_ability_for_test(decision)

        result = await dispatch_ability_for_test(decision, ctx, decision_input())
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
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="ambiguous interrupt")),
            reasoning_processor=FailingReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="reasoning failed"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/ready/idle"
    assert failures
    assert "reasoning failed" in failures[-1].message


def test_cognitive_routes_invalid_output_to_failed_event() -> None:
    async def run() -> tuple[str, list[abilities.FailureData]]:
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="ambiguous interrupt")),
            reasoning_processor=InvalidReasoningProcessor(),
        )
        ctx = await start_cognition_ability_for_test(ability)

        with pytest.raises(RuntimeError, match="array of events|output schema"):
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.failures))
        return ability.state(), ability.failures

    state, failures = asyncio.run(run())

    assert state == "/RecordingCognitionLifecycle/attached/behavior/ready/idle"
    assert failures
    assert "array of events" in failures[-1].message or "output schema" in failures[-1].message


def test_cognitive_defers_repeated_input_while_processing() -> None:
    async def run() -> tuple[
        list[cognition.types.OutputData],
        list[processing.InputData],
    ]:
        decision_processor = DelayedDecisionProcessor(cognition.decision.OutputData(result=no_output("decision")))
        ability = RecordingCognition(decision_processor=decision_processor)
        ctx = shared_hsm_context()
        ctx = await start_cognition_ability_for_test(ability, ctx)

        async def apply_input() -> None:
            # Cognition is Processing[InputData, OutputData] (FrameData-wrapped TInput) but dispatches raw InputData.
            await typing.cast(typing.Any, ability).apply(cognition_input(), ctx=ctx)

        first = asyncio.create_task(apply_input())
        await wait_until(lambda: len(decision_processor.calls) == 1)
        second = asyncio.create_task(apply_input())
        await asyncio.sleep(0)

        assert len(decision_processor.calls) == 1
        decision_processor.release_first.set()
        _ = await asyncio.gather(first, second)
        await wait_until(lambda: len(ability.outputs) == 2)
        return ability.outputs, decision_processor.calls

    outputs, decision_calls = asyncio.run(run())

    assert outputs == [no_output("decision"), no_output("decision")]
    assert len(decision_calls) == 2


def test_decision_no_operations_returns_to_idle_and_accepts_next_input() -> None:
    async def run() -> tuple[str, processing.CompletionData, processing.CompletionData]:
        decision = cognition.Decision(
            processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=None, reason="no ops"))
        )
        ctx = shared_hsm_context()
        ctx = await start_cognition_ability_for_test(decision, ctx)

        first = await dispatch_ability_for_test(decision, ctx, decision_input(operation_id="first"))
        second = await dispatch_ability_for_test(decision, ctx, decision_input(operation_id="second"))

        return decision.state(), first, second

    state, first, second = asyncio.run(run())

    assert state == "/DecisionLifecycle/attached/behavior/idle"
    assert first.output is None
    assert second.output is None


def test_decision_defers_repeated_input_while_dispatching() -> None:
    async def run() -> tuple[str, list[processing.CompletionData], list[processing.InputData]]:
        processor = DelayedDecisionProcessor(cognition.decision.OutputData(result=no_output("decision")))
        decision = cognition.Decision(processor=processor)
        ctx = shared_hsm_context()
        ctx = await start_cognition_ability_for_test(decision, ctx)

        first = asyncio.create_task(dispatch_ability_for_test(decision, ctx, decision_input(operation_id="first")))
        await wait_until(lambda: len(processor.calls) == 1)
        second = asyncio.create_task(dispatch_ability_for_test(decision, ctx, decision_input(operation_id="second")))
        await asyncio.sleep(0)

        assert len(processor.calls) == 1
        processor.release_first.set()
        outputs = list(await asyncio.gather(first, second))
        return decision.state(), outputs, processor.calls

    state, outputs, calls = asyncio.run(run())

    assert state == "/DecisionLifecycle/attached/behavior/idle"
    assert [output.output for output in outputs] == [no_output("decision"), no_output("decision")]
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
        tuple[str, ...],
        list[str],
        list[processing.InputData],
    ]:
        store = memory.Memory()
        processor = FixedProcessor(selection, write=written)
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        # Cognition attaches the bot's memory in production; Reflection is only its peer.
        await start_abilities_for_test(ctx, store)
        input = cognition.reflection.InputData(
            cognition_input=cognition_input(),
            cognition_output=focus_output("phone", "answered"),
        )
        result = await dispatch_ability_for_test(
            reflection, ctx, input, timeout=_revision_host_timeout(validation_passes=1)
        )

        from mosfet.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = await memory.transact(ctx, store, behavior_select)
        behavior_sources = _behavior_sources_from_output(behavior_out)
        return result, behavior_sources, processor.builds, processor.write_step.calls

    result, behavior_sources, builds, write_calls = asyncio.run(run())

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
    assert len(behavior_sources) == 1
    assert "AnswerIncomingRing" in behavior_sources[0]
    assert "hsm.define" in behavior_sources[0]
    assert "written" in behavior_sources[0]
    # Stored inventory must compile into executable behavior.
    behavior = behavior_events.start(behavior_sources[0])
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
        # Cognition attaches the bot's memory in production; Reflection is only its peer.
        await start_abilities_for_test(ctx, store)
        with pytest.raises(RuntimeError, match="same diagnostic message after fix") as error:
            _ = await dispatch_ability_for_test(
                reflection,
                ctx,
                cognition.reflection.InputData(
                    cognition_input=cognition_input(),
                    cognition_output=focus_output("phone", "answered"),
                ),
                timeout=_revision_host_timeout(validation_passes=2),
            )
        from mosfet.behavior import storage as behavior_storage

        behavior_out = await memory.transact(
            ctx,
            store,
            memory.InputData(statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())),
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
        # Cognition attaches the bot's memory in production; Reflection is only its peer.
        await start_abilities_for_test(ctx, store)
        _ = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=(),
            ),
            timeout=_revision_host_timeout(validation_passes=5),
        )
        from mosfet.behavior import storage as behavior_storage

        behavior_out = await memory.transact(
            ctx,
            store,
            memory.InputData(statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())),
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
        # Cognition attaches the bot's memory in production; Reflection is only its peer.
        await start_abilities_for_test(ctx, store)
        result = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "answered"),
            ),
            timeout=_revision_host_timeout(validation_passes=2),
        )
        from mosfet.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = await memory.transact(ctx, store, behavior_select)
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
    store: memory.Memory | Engine,
    *,
    name: str = "AnswerIncomingRing",
    triggers: tuple[str, ...] = (),
    description: str = "",
    source: str | None = None,
    ctx: hsm.Context | None = None,
) -> None:
    """Insert executable behavior into bot_behavior / bot_behavior_trigger tables.

    With ``ctx`` the row is written by message to an attached store (the way a peer would);
    without it the store's database engine is seeded before anything attaches the store.
    """

    from mosfet.behavior import storage as behavior_storage

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
    seed = memory.InputData(statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(behavior)))
    if isinstance(store, memory.Memory):
        assert ctx is not None, "an attached store is reached by message"
        _ = await memory.transact(ctx, store, seed)
        return
    _ = execute_on(store, seed)


def test_reflection_change_loads_existing_and_writes_update() -> None:
    """Select change → changing → replace installed behavior.

    The revision retargets the behavior onto the turn's own stimulus, the trigger the
    bot's evidence can vouch for, so the authored triggers persist unchanged.
    """

    change_intent = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        reason="select change",
    )
    observed_trigger = mosfet.InputEvent.name
    revised_source = _ANSWER_RING_BEHAVIOR_SOURCE.replace(
        'triggers = ["environment.sound"]',
        f'triggers = ["{observed_trigger}"]',
    ).replace(
        "Select focus_device for the observed pattern.",
        "Also look at the phone before answer.",
    )
    written = behavior_events.ChangeData(
        name="AnswerIncomingRing",
        triggers=(observed_trigger,),
        description="Also look at the phone before answer.",
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
    ]:
        engine = memory.store.open_sqlite_engine()
        store = memory.Memory(engine=engine)
        await _seed_behavior_record(
            engine,
            name="AnswerIncomingRing",
            triggers=("environment.sound",),
            description="Original description.",
        )
        processor = FixedProcessor(selection, change_write=written)
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        # Cognition attaches the bot's memory in production; Reflection is only its peer.
        await start_abilities_for_test(ctx, store)
        result = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "revised"),
            ),
            timeout=_revision_host_timeout(validation_passes=1),
        )
        from mosfet.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = await memory.transact(ctx, store, behavior_select)
        sources = _behavior_sources_from_output(behavior_out)
        return result, sources, processor.change_step.calls

    result, sources, change_calls = asyncio.run(run())

    assert result is None
    assert len(change_calls) == 1
    change_input = change_calls[0].input
    assert isinstance(change_input, cognition.reflection.ChangeWriteInput)
    assert change_input.existing_behavior.name == "AnswerIncomingRing"
    assert change_input.existing_behavior.description == "Original description."
    assert change_input.existing_behavior.source
    assert change_input.intent == change_intent
    assert len(sources) == 1
    assert f'triggers = ["{observed_trigger}"]' in sources[0]
    assert "look at the phone before answer" in sources[0]
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
        int,
    ]:
        # Break reads and rewrites the inventory row by message: the store refuses direct calls.
        store = memory.Memory()
        processor = FixedProcessor(selection)
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        # Cognition attaches the bot's memory in production; Reflection is only its peer.
        await start_abilities_for_test(ctx, store)
        await wait_until(lambda: reflection.state().endswith("/idle"))
        await _seed_behavior_record(
            store,
            name="AnswerIncomingRing",
            triggers=("environment.sound",),
            description="To be broken.",
            ctx=ctx,
        )
        result = await dispatch_ability_for_test(
            reflection,
            ctx,
            cognition.reflection.InputData(
                cognition_input=cognition_input(),
                cognition_output=focus_output("phone", "break"),
            ),
        )
        from mosfet.behavior import storage as behavior_storage

        behavior_select = memory.InputData(
            statements=memory.compile_statements(*behavior_storage.select_all_behaviors_clauses())
        )
        behavior_out = await memory.transact(ctx, store, behavior_select)
        sources = _behavior_sources_from_output(behavior_out)
        behaviors = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in behavior_out.results[0].rows),
            tuple(row.as_mapping() for row in behavior_out.results[1].rows),
        )
        active_out = await memory.transact(
            ctx,
            store,
            memory.InputData(
                statements=memory.compile_statements(*behavior_storage.select_active_behaviors_clauses(lifetime="turn"))
            ),
        )
        active = behavior_storage.instances_from_behavior_results(
            tuple(row.as_mapping() for row in active_out.results[0].rows),
            tuple(row.as_mapping() for row in active_out.results[1].rows),
        )
        return (
            result,
            sources,
            behaviors[0].status if behaviors else "ACTIVE",
            processor.builds,
            len(active),
        )

    result, sources, status, builds, active_count = asyncio.run(run())

    assert result is None
    assert len(sources) == 1
    assert "hsm.define" in sources[0]
    assert status == "BROKEN"
    assert active_count == 0
    # Break is applied after select; change-phase Processing is not used.
    assert builds == [cognition.Reflection.select_instructions]


def test_reflection_no_selection_completes_without_writing_an_episode() -> None:
    """Cognition remembers every turn; Reflection never writes one of its own."""

    async def run() -> tuple[processing.CompletionData | None, memory.EpisodesData]:
        store = memory.Memory()
        processor = FixedProcessor(empty_output())
        reflection = cognition.Reflection(processor=processor, memory=store)
        ctx = await start_cognition_ability_for_test(reflection)
        await start_abilities_for_test(ctx, store)
        input = cognition.reflection.InputData(
            cognition_input=cognition_input(),
            cognition_output=ignore_output("nothing to learn"),
        )
        result = await dispatch_ability_for_test(reflection, ctx, input)
        recent = await abilities.run_terminal_operation(
            ctx,
            child=store,
            request=memory.RecentEvent.with_data_and_id(memory.RecentData(limit=8), "recent"),
            terminals=(memory.RecentOutputEvent, memory.FailedEvent),
            timeout=datetime.timedelta(seconds=3),
        )
        assert isinstance(recent.data, memory.EpisodesData)
        return result, recent.data

    result, recent = asyncio.run(run())

    assert result is None
    assert recent.episodes == ()


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
    from mosfet.environment import SoundData, SoundEvent

    return SoundEvent.with_data(SoundData(audio=b"ring", kind="phone.ringing"))


def test_autonomy_handles_matching_behavior_without_decision_processor() -> None:
    """Stimulus → behavior Behavior → EventData selection without deliberative Processing."""

    async def run() -> tuple[list[cognition.types.OutputData], list[processing.InputData], list[processing.InputData]]:
        engine = memory.store.open_sqlite_engine()
        store = memory.Memory(engine=engine)
        await _seed_behavior_record(
            engine,
            name="FocusOnRing",
            triggers=("environment.sound",),
            source=_FOCUS_RING_BEHAVIOR_SOURCE,
        )
        autonomy = cognition.Autonomy(memory=store)
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(reason="should not run"))
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "should not run"))
        reflection_processor = FixedProcessor(no_output("observed"))
        ability = RecordingCognition(
            autonomy=autonomy,
            decision_processor=decision_processor,
            reasoning_processor=reasoning_processor,
            # One memory per bot: Cognition attaches it, Autonomy and Reflection reach it by message.
            reflection=cognition.Reflection(processor=reflection_processor, memory=store),
            store=store,
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
        _ = await mosfet.started(ctx, bot_actor, bot_actor.model)
        _ = await dispatch_ability_for_test(
            ability,
            ctx,
            turn,
            timeout=_LEARNED_BEHAVIOR_HOST_TIMEOUT_SECONDS,
        )
        await wait_until(lambda: bool(ability.outputs))
        await wait_until(lambda: bool(reflection_processor.step.calls))
        return ability.outputs, decision_processor.calls, reflection_processor.step.calls

    outputs, decision_calls, reflection_calls = asyncio.run(run())

    assert len(outputs) == 1
    assert outputs[0] == (
        cognition.types.EventData(
            event=mosfet.FocusDeviceEvent.name,
            data={"device": "phone"},
            reason="ring behavior",
        ),
    )
    assert decision_calls == []
    assert len(reflection_calls) == 1
    reflection_input = reflection_calls[0].input
    assert isinstance(reflection_input, cognition.reflection.ProcessorInput)
    assert reflection_input.cognition_output == outputs[0]


def test_autonomy_unhandled_falls_through_to_decision() -> None:
    async def run() -> tuple[list[cognition.types.OutputData], list[processing.InputData]]:
        engine = memory.store.open_sqlite_engine()
        store = memory.Memory(engine=engine)
        await _seed_behavior_record(
            engine,
            name="FocusOnRing",
            triggers=("environment.sound",),
            source=_FOCUS_RING_BEHAVIOR_SOURCE,
        )
        autonomy = cognition.Autonomy(memory=store)
        decision_processor = RecordingDecisionProcessor(no_output("decision after autonomy"))
        ability = RecordingCognition(
            autonomy=autonomy,
            decision_processor=decision_processor,
            reasoning_processor=FailingReasoningProcessor(),
            store=store,
        )
        ctx = await start_cognition_ability_for_test(ability)
        # bot.input stimulus does not match environment.sound trigger.
        _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
        await wait_until(lambda: bool(ability.outputs))
        return ability.outputs, decision_processor.calls

    outputs, decision_calls = asyncio.run(run())

    assert outputs == [no_output("decision after autonomy")]
    assert len(decision_calls) == 1


def test_a_cognition_without_reflex_takes_every_turn_up_with_reasoning() -> None:
    """D-R-52: every tier but reasoning is optional; without reflex, the turn's input goes on to reasoning."""

    async def run() -> tuple[cognition.types.OutputData, list[processing.InputData]]:
        reasoning_processor = RecordingReasoningProcessor(no_output("reasoned"))
        ability = cognition.Cognition(
            reasoning=cognition.Reasoning(processor=reasoning_processor),
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        return result, reasoning_processor.calls

    result, reasoning_calls = asyncio.run(run())

    assert result == no_output("reasoned")
    assert len(reasoning_calls) == 1


def test_reflex_is_one_stage_whose_layers_and_spans_say_which_settled_or_handed_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-R-52 OTEL: ``bot.cognition.stage`` names reflex once per layer (``bot.reflex.layer``), then the stages
    after it; autonomy's and the decision layer's own spans are ``bot.reflex.*``."""

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)

    async def run() -> None:
        store = memory.Memory()
        ability = RecordingCognition(
            autonomy=cognition.Autonomy(memory=store),
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=None)),
            reasoning_processor=RecordingReasoningProcessor(no_output("reasoned")),
            store=store,
            associative=memory.Associative(memory=store),
        )
        ctx = await start_cognition_ability_for_test(ability)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: ability.state().endswith("/idle"))

    asyncio.run(run())

    finished = exporter.get_finished_spans()
    stages = [
        (str((recorded.attributes or {}).get("bot.stage")), (recorded.attributes or {}).get("bot.reflex.layer"))
        for recorded in sorted(finished, key=lambda recorded: recorded.start_time or 0)
        if recorded.name == "bot.cognition.stage"
    ]
    assert stages == [("reflex", "behaviors"), ("reflex", "decision"), ("associating", None), ("reasoning", None)]
    names = {recorded.name for recorded in finished}
    assert {"bot.reflex.match", "bot.reflex.invoke", "bot.reflex.classify"} <= names
    assert not {name for name in names if name.startswith(("bot.autonomy.", "bot.intuition.", "bot.decision."))}


def _speech_event_stimulus() -> hsm.Event[object]:
    """Labeled Listening speech product used as cognition stimulus (no priors)."""

    from mosfet.abilities import listening
    from mosfet.abilities.hearing import voice

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
        source_ids=frozenset({"voice:3f2a9c"}),
    )
    return listening.SpeechEvent.with_data(speech)


def test_autonomy_native_seed_preserves_acoustic_speech_for_conversation() -> None:
    """A trusted native seed carries typed acoustic data through Autonomy unchanged."""

    from mosfet.abilities import listening
    from mosfet.abilities import communication
    from mosfet.abilities.communication import conversation
    from mosfet.abilities.communication.conversation import turn_detector
    from mosfet.abilities.communication import behaviors
    from mosfet.abilities.hearing import voice

    async def run() -> tuple[
        list[cognition.types.OutputData],
        tuple[str, ...],
        list[abilities.FailureData],
        list[processing.InputData],
        list[processing.InputData],
    ]:
        autonomy = cognition.Autonomy(seeded_behaviors=(behaviors.speech_heard_seed(),))
        decision_processor = RecordingDecisionProcessor(no_output("should not run"))
        reasoning_processor = FailingReasoningProcessor()
        ability = RecordingCognition(
            autonomy=autonomy,
            decision_processor=decision_processor,
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)
        bot_actor = _BotActor()
        assert bot_actor.model is not None
        _ = await mosfet.started(ctx, bot_actor, bot_actor.model)
        conversation_actor = conversation.Conversation(
            turn_detector=turn_detector.TurnDetector(participant_ref="bot", conversation_ref="ambient")
        )
        communication_actor = communication.Communication(
            active_conversation=conversation_actor,
            speaking=abilities.Speaking(
                encoder=_NoopSpeechEncoder(),
                speaker=audio.Speaker(),
                conversation=(conversation_actor,),
            ),
        )
        assert conversation_actor.model is not None
        assert communication_actor.model is not None
        _ = await mosfet.started(ctx, conversation_actor, conversation_actor.model)
        _ = await mosfet.started(ctx, communication_actor, communication_actor.model)
        from mosfet.protocols import attachment

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
            source_ids=frozenset({"voice:3f2a9c"}),
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
        return (
            ability.outputs,
            (listening.SpeechEvent.name,),
            ability.failures,
            decision_processor.calls,
            reasoning_processor.calls,
        )

    outputs, triggers, failures, decision_calls, reasoning_calls = asyncio.run(run())
    assert triggers == (listening.SpeechEvent.name,)
    assert not failures
    assert not decision_calls
    assert not reasoning_calls
    assert len(outputs) == 1
    selected = outputs[0]
    assert len(selected) == 1
    assert selected[0].event == communication.InputEvent.name
    assert isinstance(selected[0].data, conversation.TurnData)
    assert selected[0].data.content == bytes([0, 1]) * 160
    assert selected[0].data.content_type == "audio/pcm"
    assert selected[0].data.sample_rate_hz == 16_000
    assert selected[0].data.channels == 1
    assert selected[0].data.source_ids == frozenset({"voice:3f2a9c"})
    assert selected[0].data.parent is not None
    assert selected[0].data.parent.event == listening.SpeechEvent.name
    assert '"content"' not in selected[0].model_dump_json()


def test_cognition_without_priors_time_to_wire_speech_event_to_conversation() -> None:
    """Use the real phone_bot e2e path; fixture processors cannot invent behaviors."""

    import pytest

    pytest.skip(
        "fixture processors short-circuit learning; run "
        "tests/examples/test_phone_bot.py::test_phone_bot_e2e_cognition_wires_speech_event_to_conversation"
    )


_ROUTINE_START = datetime.datetime(2026, 10, 5, 12, 0, tzinfo=datetime.UTC)


def _routine_install_input() -> memory.InputData:
    from tests.bot.behavior.support import routine_source

    checked = behavior_events.check(routine_source(schedule="hsm.every(seconds = 300)"))
    assert checked.value is not None, checked.report.render()
    return memory.InputData(
        statements=memory.compile_statements(
            *behavior_events.storage.replace_behavior_clauses(checked.value),
        )
    )


async def _until_routines(condition: collections.abc.Callable[[], bool], *, timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise TimeoutError("routine condition not reached")
        await asyncio.sleep(0.01)


# Routines keeps time on the routines' clock too, and its own bounds (the attach and the memory
# reply) are shorter than any schedule may be, so a wait past the shortest schedule is a routine's.
_SCHEDULE_FLOOR = behavior_instance.ROUTINE_MIN_EVERY - datetime.timedelta(seconds=1)


def _open_routine_waits(timers: object) -> list[datetime.timedelta]:
    from tests.bot.behavior.support import ManualTimers

    assert isinstance(timers, ManualTimers)
    return [duration for duration, waiter in timers.pending if not waiter.done() and duration > _SCHEDULE_FLOOR]


def test_cognition_reconciles_routines_after_reflection_settles() -> None:
    from tests.bot.behavior.support import ManualTimers

    async def run() -> tuple[list[datetime.timedelta], list[datetime.timedelta]]:
        store = memory.Memory()
        timers = ManualTimers()
        decision, reasoning = cognition_abilities()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(empty_output()), memory=store),
            routines=cognition.Routines(memory=store, now=lambda: _ROUTINE_START, clock=timers),
            memory=store,
        )
        ctx = await start_cognition_ability_for_test(ability)
        try:
            await _until_routines(lambda: ability.state().endswith("/idle"))
            before = _open_routine_waits(timers)
            # Reflection is the inventory writer; whatever it wrote, Routines re-reads after it settles.
            _ = await memory.transact(ctx, store, _routine_install_input())
            _ = await dispatch_ability_for_test(ability, ctx, cognition_input())
            await _until_routines(lambda: bool(_open_routine_waits(timers)))
            return before, _open_routine_waits(timers)
        finally:
            await ability.stop(ctx)

    before, after = asyncio.run(run())

    assert before == []
    assert after == [datetime.timedelta(seconds=300)]


class _RoutineTickBody(hsm.Instance):
    """Body stand-in: records the cognition input handoffs it is given."""

    handoffs: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.handoffs = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "_RoutineTickBody", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.handoffs.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "RoutineTickBody",
        hsm.initial(hsm.target("ready")),
        hsm.state("ready", hsm.transition(hsm.on(cognition.InputEvent), hsm.effect(_record))),
    )


def test_cognition_hands_routine_ticks_to_the_body_as_observed_input(tmp_path: pathlib.Path) -> None:
    """A routine a restart left behind rehydrates once Cognition is up, then ticks into the body.

    The previous process wrote the inventory; this one reaches memory by message only, and
    Cognition asks Routines to reconcile once its children (memory among them) are attached.
    """

    from tests.bot.behavior.support import ManualTimers

    database = str(tmp_path / "bot.db")
    # The database the previous process left behind, written without any memory machine.
    with memory.store.open_sqlite_engine(database=database).begin() as connection:
        for statement in _routine_install_input().statements:
            _ = connection.exec_driver_sql(statement.sql, statement.parameters)

    async def run() -> tuple[hsm.Event[typing.Any], str]:
        store = memory.Memory(database=database)
        timers = ManualTimers()
        decision, reasoning = cognition_abilities()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(empty_output()), memory=store),
            routines=cognition.Routines(memory=store, now=lambda: _ROUTINE_START, clock=timers),
            memory=store,
        )
        body = _RoutineTickBody()
        ctx = shared_hsm_context()
        _ = await mosfet.started(ctx, body, body.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=body), "attach"))
        try:
            await _until_routines(lambda: bool(_open_routine_waits(timers)))
            _ = await timers.fire(_ROUTINE_START, longer_than=_SCHEDULE_FLOOR)
            await _until_routines(lambda: bool(body.handoffs))
            return body.handoffs[0], hsm.id(ability)
        finally:
            await ability.stop(ctx)

    handoff, cognition_id = asyncio.run(run())

    assert handoff.name == cognition.InputEvent.name
    assert handoff.source == cognition_id
    assert handoff.id
    data = handoff.data
    assert isinstance(data, cognition.InputData)
    stimulus = data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == behavior_events.TickEvent.name
    tick = stimulus.data
    assert isinstance(tick, behavior_events.TickData)
    assert tick.name == "MorningBriefing"
    assert tick.output[0]["event"] == "bot.speaking.say"


# --- remembering every turn and associating it --------------------------------------------------

_ADA = mosfet.Participant(ref="+15555550100", role=mosfet.ParticipantRole.SOURCE)


async def _remember_by_message(ctx: hsm.Context, store: memory.Memory, episode: memory.EpisodeData) -> None:
    reply = await abilities.run_terminal_operation(
        ctx,
        child=store,
        request=memory.RememberEvent.with_data_and_id(episode, f"seed-{episode.turn_id}"),
        terminals=(memory.RememberedEvent, memory.FailedEvent),
        timeout=datetime.timedelta(seconds=3),
    )
    assert isinstance(reply.data, memory.RememberedData), reply.data


def _assert_keeps_what_was_said(kept: memory.Episode) -> None:
    """A turn that did not complete is still kept with its stimulus and who took part."""

    assert kept.stimulus == mosfet.InputEvent.name
    assert kept.text
    assert kept.participants == (_ADA,)
    assert kept.focus_candidates == ("phone",)
    assert kept.acts == ()


async def _remembered(ctx: hsm.Context, store: memory.Memory) -> tuple[memory.Episode, ...]:
    reply = await abilities.run_terminal_operation(
        ctx,
        child=store,
        request=memory.RecentEvent.with_data_and_id(memory.RecentData(limit=16), "recent"),
        terminals=(memory.RecentOutputEvent, memory.FailedEvent),
        timeout=datetime.timedelta(seconds=3),
    )
    assert isinstance(reply.data, memory.EpisodesData), reply.data
    return reply.data.episodes


def test_cognition_requires_memory_for_associative() -> None:
    decision, reasoning = cognition_abilities()
    reflection = cognition.Reflection(processor=FixedProcessor(no_output("unused")), memory=memory.Memory())
    with pytest.raises(ValueError, match="associative requires memory"):
        _ = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=reflection,
            associative=memory.Associative(memory=memory.Memory()),
        )
    members = require_model(cognition.Cognition.model).members
    assert "/CognitionLifecycle/attached/behavior/processing/associating" in members
    assert "/CognitionLifecycle/attached/behavior/processing/cancelling/associating" in members


def test_cognition_associates_after_reflex_and_remembers_the_completed_turn() -> None:
    """R-21: recall comes after reflex. What memory surfaced reaches reasoning (as recalled) and reflection,
    never the decision layer, which chooses without it; the turn is kept."""

    async def run() -> tuple[
        cognition.types.OutputData,
        list[processing.InputData],
        list[processing.InputData],
        list[processing.InputData],
        tuple[memory.Episode, ...],
    ]:
        store = memory.Memory()
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over"))
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        reflection_processor = FixedProcessor(no_output("observed"))
        ability = RecordingCognition(
            decision_processor=decision_processor,
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=reflection_processor, memory=store),
            store=store,
            associative=memory.Associative(memory=store),
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        await _remember_by_message(
            ctx,
            store,
            memory.EpisodeData(
                turn_id="earlier",
                parts=(text.TextPart(text="Ada asked for a call at noon."),),
                participants=(_ADA,),
                outcome=memory.Outcome.COMPLETED,
            ),
        )
        turn = (await started_cognition_input(ctx)).model_copy(update={"participants": (_ADA,)})
        result = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: bool(reflection_processor.step.calls))
        episodes = await _remembered(ctx, store)
        return (
            result,
            decision_processor.calls,
            reasoning_processor.calls,
            reflection_processor.step.calls,
            episodes,
        )

    result, decision_calls, reasoning_calls, reflection_calls, episodes = asyncio.run(run())

    assert result == focus_output("phone", "reasoned focus")
    (decision_frame,) = decision_calls
    assert decision_frame.recalled is None
    (reasoning_frame,) = reasoning_calls
    assert isinstance(reasoning_frame.recalled, cognition.RecalledData)
    assert reasoning_frame.recalled.recent == ()
    surfaced = reasoning_frame.recalled.associations
    assert isinstance(surfaced, memory.AssociationsData)
    assert [item.text for item in surfaced.items] == ["Ada asked for a call at noon."]
    assert surfaced.items[0].involves == (_ADA.ref,)
    select = reflection_calls[0].input
    assert isinstance(select, cognition.reflection.ProcessorInput)
    assert select.associations == surfaced
    # The turn itself is remembered once, stamped where Cognition owns it.
    assert [episode.turn_id for episode in episodes][0] == "earlier"
    (kept,) = episodes[1:]
    assert kept.outcome is memory.Outcome.COMPLETED
    assert kept.stimulus == mosfet.InputEvent.name
    assert kept.participants == (_ADA,)
    assert kept.focus_candidates == ("phone",)
    assert [act.event for act in kept.acts] == [mosfet.FocusDeviceEvent.name]
    assert kept.acts[0].reason == "reasoned focus"
    assert kept.text


def test_cognition_remembers_what_it_did_alongside_what_it_perceived_and_recalls_both() -> None:
    """A completed turn's episode is what was perceived, then cognition's own output; a later turn recalls both."""

    async def run() -> tuple[tuple[memory.Episode, ...], list[processing.InputData]]:
        store = memory.Memory()
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "the phone is ringing"))
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=None)),
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=store),
            store=store,
            associative=memory.Associative(memory=store),
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: ability.state().endswith("/idle"))
        # Memory serves in order, so the turn's remember (sent first) is kept before this read.
        episodes = await _remembered(ctx, store)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return episodes, reasoning_processor.calls

    episodes, reasoning_calls = asyncio.run(run())

    (kept,) = episodes
    perceived = kept.text.index("<events:input_event")
    did = kept.text.index(f'stimulus:event="{cognition.OutputEvent.name}"')
    assert perceived < did
    assert mosfet.FocusDeviceEvent.name in kept.text[did:]
    assert "the phone is ringing" in kept.text[did:]
    assert [act.event for act in kept.acts] == [mosfet.FocusDeviceEvent.name]
    _, later = reasoning_calls
    assert isinstance(later.recalled, cognition.RecalledData)
    assert isinstance(later.recalled.associations, memory.AssociationsData)
    (recalled,) = later.recalled.associations.items
    # A ring carries no words: its recollection names what was perceived, then what the bot did.
    assert kept.heard == ()
    assert recalled.text == f"perceived: {mosfet.InputEvent.name}\ndid: {mosfet.FocusDeviceEvent.name}"


def test_a_recalled_episode_reads_as_when_who_what_was_heard_and_what_the_bot_did_never_as_markup() -> None:
    """An association is a compact recollection: the words heard and the acts named, never the turn's projection.

    The episode keeps its full projection (what it is embedded and recalled in full from); the association a
    later turn is shown says when, who, what was heard, and what the bot did, as plain words, so the model never
    reads its own earlier events as escaped markup it might echo back.
    """

    words = 'Hey anyone home? I just popped in to say "hi" & <wave>'

    async def run() -> tuple[tuple[memory.Episode, ...], list[processing.InputData]]:
        store = memory.Memory()
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "answer Ada"))
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=None)),
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=store),
            store=store,
            associative=memory.Associative(memory=store),
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        base = await started_cognition_input(ctx)
        turn = communication.InputEvent.with_data(
            communication.conversation.TurnData(
                content=words, content_type="text/plain", source_ids=frozenset({"voice:ada"}), target_ids=frozenset()
            )
        )
        exchange = dataclasses.replace(
            communication.conversation.OutputEvent.with_data(
                communication.conversation.Exchange(parent=mosfet.StimulusData.from_event(turn))
            ),
            id="said-1",
        )
        _ = await dispatch_ability_for_test(
            ability, ctx, base.model_copy(update={"stimulus": exchange, "participants": (_ADA,)})
        )
        await wait_until(lambda: ability.state().endswith("/idle"))
        episodes = await _remembered(ctx, store)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return episodes, reasoning_processor.calls

    episodes, reasoning_calls = asyncio.run(run())

    (kept,) = episodes
    assert kept.heard == (words,)
    assert "communication:exchange" in kept.text
    _, later = reasoning_calls
    assert isinstance(later.recalled, cognition.RecalledData)
    assert isinstance(later.recalled.associations, memory.AssociationsData)
    (recalled,) = later.recalled.associations.items
    assert recalled.text == f"heard: {words}\ndid: {mosfet.FocusDeviceEvent.name}"
    rendered = "".join(
        part.text
        for part in cognition.event.model_facing_content(later.recalled, capacity=media.ImageCapacity())
        if isinstance(part, text.TextPart)
    )
    association = rendered[rendered.index("<memory:association ") : rendered.index("</memory:association>")]
    for markup in ("&quot;", "&lt;communication", "cognition:event", "stimulus:", "said-1"):
        assert markup not in association
    assert 'heard: Hey anyone home? I just popped in to say "hi" &amp; &lt;wave&gt;' in association
    assert f"did: {mosfet.FocusDeviceEvent.name}" in association
    assert _ADA.ref in association
    assert f'at="{kept.created_at.isoformat().replace("+00:00", "Z")}"' in association


def test_an_association_that_fails_leaves_the_turn_without_associations_not_failed() -> None:
    async def run() -> tuple[cognition.types.OutputData, list[processing.InputData], str]:
        store = memory.Memory()
        decision_processor = RecordingDecisionProcessor(focus_output("phone", "fast focus"))
        ability = RecordingCognition(
            decision_processor=decision_processor,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=store),
            store=store,
            # Its memory is never attached or started: every association fails, typed.
            associative=memory.Associative(memory=memory.Memory()),
        )
        ctx = await start_cognition_ability_for_test(ability)
        result = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return result, decision_processor.calls, ability.state()

    result, decision_calls, state = asyncio.run(run())

    assert result == focus_output("phone", "fast focus")
    assert len(decision_calls) == 1
    assert decision_calls[0].recalled is None
    assert state.endswith("/idle")


_MAYA = mosfet.Participant(ref="voice:maya", role=mosfet.ParticipantRole.SOURCE)
_OWN = mosfet.Participant(ref="voice:willy", role=mosfet.ParticipantRole.SOURCE, own=True)


def _heard_now(turn_id: str) -> hsm.Event[mosfet.InputEventData]:
    """A stimulus stamped with the id its Conversation wrote its heard turn under."""

    return dataclasses.replace(
        mosfet.InputEvent.with_data(mosfet.InputEventData(target_device="phone", participants=(_MAYA,))), id=turn_id
    )


def test_recent_turns_reach_reasoning_oldest_first_ending_with_the_turn_being_answered() -> None:
    """R-20 C3: what was heard and said up to this turn reaches reasoning from short-term memory.

    The register's turns ride the turn beside what memory associates, render one compact line each
    (when, which way, who by their ref, the words), oldest first, and the turn that is this turn's own
    stimulus is among them where it was heard, so the conversation reads as one timeline whose newest
    line is what was just said; the stimulus itself still closes the prompt.
    """

    async def run() -> tuple[list[processing.InputData], list[processing.InputData]]:
        register = memory.StmMemory()
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over"))
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        ability = RecordingCognition(
            decision_processor=decision_processor,
            reasoning_processor=reasoning_processor,
            stm_memory=register,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await start_abilities_for_test(ctx, register)
        await wait_until(lambda: ability.state().endswith("/idle"))
        turns: tuple[tuple[str, memory.stm.TurnDirection, mosfet.Participant, str], ...] = (
            ("earlier", "heard", _MAYA, "Hi Willy, I'm Maya."),
            ("reply", "said", _OWN, "Nice to meet you, Maya."),
            ("now", "heard", _MAYA, "What did I just tell you?"),
        )
        for turn_id, direction, speaker, words in turns:
            _ = await hsm.dispatch(
                ctx,
                register,
                memory.stm.TurnEvent.with_data(
                    memory.stm.TurnRecordData.of(
                        id=turn_id, direction=direction, speakers=(speaker,), words=words, conversation="talk"
                    )
                ),
            )
        turn = (await started_cognition_input(ctx)).model_copy(
            update={"stimulus": _heard_now("now"), "participants": (_MAYA,)}
        )
        _ = await dispatch_ability_for_test(ability, ctx, turn)
        return decision_processor.calls, reasoning_processor.calls

    decision_calls, reasoning_calls = asyncio.run(run())

    (frame,) = reasoning_calls
    recalled = frame.recalled
    assert isinstance(recalled, cognition.RecalledData)
    assert [(turn.id, turn.direction) for turn in recalled.recent] == [
        ("earlier", "heard"),
        ("reply", "said"),
        ("now", "heard"),
    ]
    assert recalled.associations is None
    # Reflex chooses without recall (D-R-52): the recent turns are read after it, for the tiers that answer.
    (decided,) = decision_calls
    assert decided.recalled is None
    prompt = "".join(
        part.text for part in frame.model_facing_content(media.ImageCapacity()) if isinstance(part, text.TextPart)
    )
    heard = prompt.index('direction="heard" who="voice:maya">Hi Willy, I\'m Maya.</memory:turn>')
    said = prompt.index('direction="said" who="voice:willy">Nice to meet you, Maya.</memory:turn>')
    now = prompt.index('direction="heard" who="voice:maya">What did I just tell you?</memory:turn>')
    # One timeline, its newest line what was just said: the very next line closes the recent turns.
    assert heard < said < now
    assert prompt[now:].split("\n")[1].strip() == "</recent>"
    assert 'id="earlier"' not in prompt and "conversation=" not in prompt and "cut=" not in prompt
    # R-20 C5: recalled, then the situation (the world block), then the stimulus; the system message is not repeated.
    assert frame.situation is not None and frame.situation in prompt
    assert now < prompt.index(frame.situation) < prompt.index('stimulus:id="now"')
    assert "instructions=" not in prompt


def test_recent_turns_reach_intuition_as_they_reach_reasoning() -> None:
    """R-20 C3, D-R-52: the recent turns are read after reflex, so intuition, the tier that answers in words now,
    is given them as reasoning is, ending with the turn being answered."""

    async def run() -> list[processing.InputData]:
        register = memory.StmMemory()
        intuition_processor = RecordingReasoningProcessor(no_output("intuited"))
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=None)),
            intuition=cognition.Intuition(processor=intuition_processor),
            reasoning_processor=RecordingReasoningProcessor(no_output("reasoned")),
            stm_memory=register,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await start_abilities_for_test(ctx, register)
        await wait_until(lambda: ability.state().endswith("/idle"))
        for turn_id, direction, speaker, words in (
            ("earlier", "heard", _MAYA, "Hi Willy, I'm Maya."),
            ("reply", "said", _OWN, "Nice to meet you, Maya."),
            ("now", "heard", _MAYA, "What did I just tell you?"),
        ):
            _ = await hsm.dispatch(
                ctx,
                register,
                memory.stm.TurnEvent.with_data(
                    memory.stm.TurnRecordData.of(
                        id=turn_id,
                        direction=typing.cast(memory.stm.TurnDirection, direction),
                        speakers=(speaker,),
                        words=words,
                        conversation="talk",
                    )
                ),
            )
        turn = (await started_cognition_input(ctx)).model_copy(
            update={"stimulus": _heard_now("now"), "participants": (_MAYA,)}
        )
        _ = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: ability.state().endswith("/idle"))
        return intuition_processor.calls

    (frame,) = asyncio.run(run())

    assert isinstance(frame.recalled, cognition.RecalledData)
    assert [(turn.id, turn.direction) for turn in frame.recalled.recent] == [
        ("earlier", "heard"),
        ("reply", "said"),
        ("now", "heard"),
    ]


def _speech_now() -> hsm.Event[listening.SpeechData]:
    """What the ear heard from a voice nobody identified: Listening wrote its heard turn under this id."""

    speech = listening.SpeechData(
        content="What did I just tell you?",
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        media_type="audio/pcm",
    )
    return dataclasses.replace(listening.SpeechEvent.with_data(speech), id="now")


def _exchange_now() -> hsm.Event[communication.conversation.Exchange]:
    """The same speech admitted through Conversation: the exchange's own ids differ, its causal parents carry it."""

    turn = communication.conversation.TurnData(
        source_ids=frozenset({_MAYA.ref}),
        target_ids=frozenset(),
        content="What did I just tell you?",
        content_type="text/plain",
        parent=mosfet.StimulusData.from_event(_speech_now()),
    )
    inbound = dataclasses.replace(communication.conversation.InputEvent.with_data(turn), id="conversation-input")
    exchange = communication.conversation.Exchange(parent=mosfet.StimulusData.from_event(inbound))
    return dataclasses.replace(communication.conversation.OutputEvent.with_data(exchange), id="exchange")


@pytest.mark.parametrize("stimulus", [_speech_now, _exchange_now], ids=["raw-speech", "conversation-exchange"])
def test_the_turn_listening_heard_for_this_stimulus_is_the_newest_recent_turn_after_earlier_ones(
    stimulus: collections.abc.Callable[[], hsm.Event[typing.Any]],
) -> None:
    """R-20: Listening writes every utterance it hears under its speech event's id, identified or not.

    Whether that speech reaches cognition raw (a voice nobody identified) or through Conversation (whose
    exchange carries it as a causal parent), this turn's own heard turn is the newest of the recent turns,
    after the earlier ones, so what was just said reads as the latest line of the conversation.
    """

    async def run() -> list[processing.InputData]:
        register = memory.StmMemory()
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
            stm_memory=register,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await start_abilities_for_test(ctx, register)
        await wait_until(lambda: ability.state().endswith("/idle"))
        turns: tuple[tuple[str, memory.stm.TurnDirection, tuple[mosfet.Participant, ...], str], ...] = (
            ("earlier", "heard", (), "Hi Willy, I'm Maya."),
            ("reply", "said", (_OWN,), "Nice to meet you, Maya."),
            ("now", "heard", (), "What did I just tell you?"),
        )
        for turn_id, direction, speakers, words in turns:
            _ = await hsm.dispatch(
                ctx,
                register,
                memory.stm.TurnEvent.with_data(
                    memory.stm.TurnRecordData.of(
                        id=turn_id, direction=direction, speakers=speakers, words=words, conversation="ear"
                    )
                ),
            )
        turn = (await started_cognition_input(ctx)).model_copy(update={"stimulus": stimulus()})
        _ = await dispatch_ability_for_test(ability, ctx, turn)
        return reasoning_processor.calls

    (frame,) = asyncio.run(run())

    assert isinstance(frame.recalled, cognition.RecalledData)
    assert [(turn.id, turn.direction) for turn in frame.recalled.recent] == [
        ("earlier", "heard"),
        ("reply", "said"),
        ("now", "heard"),
    ]


@pytest.mark.parametrize("stimulus", [_speech_now, _exchange_now], ids=["raw-speech", "conversation-exchange"])
def test_an_episode_of_the_turn_being_answered_is_not_recalled_as_an_association_while_earlier_ones_are(
    stimulus: collections.abc.Callable[[], hsm.Event[typing.Any]],
) -> None:
    """R-20 C3: the turn being answered is not shown again as a memory of an earlier one.

    Speech admitted through Conversation is two turns: cognition first takes the heard speech (and
    remembers that turn under the speech's id), then answers the exchange that carries the speech as a
    causal parent. Associating for the exchange must not surface the speech turn's episode, nor one
    kept under the stimulus's own id, as if it were an earlier memory; an earlier episode still surfaces.
    """

    async def run() -> list[processing.InputData]:
        store = memory.Memory()
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=store),
            store=store,
            associative=memory.Associative(memory=store),
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        given = stimulus()
        for turn_id, words in (
            ("earlier", "Hi Willy, I'm Maya."),
            ("now", "What did I just tell you?"),
            (given.id, "What did I just tell you?"),
        ):
            await _remember_by_message(
                ctx,
                store,
                memory.EpisodeData(
                    turn_id=turn_id, parts=(text.TextPart(text=words),), outcome=memory.Outcome.COMPLETED
                ),
            )
        turn = (await started_cognition_input(ctx)).model_copy(update={"stimulus": given})
        _ = await dispatch_ability_for_test(ability, ctx, turn)
        return reasoning_processor.calls

    (frame,) = asyncio.run(run())

    assert isinstance(frame.recalled, cognition.RecalledData)
    surfaced = frame.recalled.associations
    assert isinstance(surfaced, memory.AssociationsData)
    assert [item.text for item in surfaced.items] == ["Hi Willy, I'm Maya."]
    assert surfaced.omitted == 0


def test_cognition_remembers_a_failed_turn_with_its_stimulus_and_no_acts() -> None:
    async def run() -> tuple[list[abilities.FailureData], tuple[memory.Episode, ...]]:
        store = memory.Memory()
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="escalate")),
            reasoning_processor=FailingReasoningProcessor(),
            store=store,
        )
        ctx = await start_cognition_ability_for_test(ability)
        _ = await hsm.dispatch(
            ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "failing-turn")
        )
        await wait_until(lambda: bool(ability.failures))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return ability.failures, await _remembered(ctx, store)

    failures, episodes = asyncio.run(run())

    # The recorder sees the child's failure arrive and Cognition's own failed terminal.
    assert {failure.message for failure in failures} == {"reasoning failed"}
    (kept,) = episodes
    assert kept.turn_id == "failing-turn"
    assert kept.outcome is memory.Outcome.FAILED
    assert kept.stimulus == mosfet.InputEvent.name
    assert kept.acts == ()


class _RefusingProcessor(processing.Processor):
    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        raise RuntimeError("the bench refused the write")


def test_cognition_remembers_a_failed_turn_with_its_failure_after_what_it_perceived() -> None:
    """Live check round 1 (D3): after a turn failed, the next turn recalled only its input, not that it failed.

    The bot went on to promise work it could not reach. A failed turn's episode is now what was perceived,
    then the turn's typed failure (cognition's failed event), so recall covers what happened to what it tried.
    """

    async def run() -> tuple[memory.Episode, ...]:
        store = memory.Memory()
        # Decision leaves the turn to reasoning, whose step fails (a failing decision would cascade, D-R-36).
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="needs thought")),
            reasoning_processor=_RefusingProcessor(),
            store=store,
        )
        ctx = await start_cognition_ability_for_test(ability)
        _ = await hsm.dispatch(
            ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "refused-turn")
        )
        await wait_until(lambda: bool(ability.failures))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return await _remembered(ctx, store)

    (kept,) = asyncio.run(run())

    assert kept.turn_id == "refused-turn"
    assert kept.outcome is memory.Outcome.FAILED
    perceived = kept.text.index("<events:input_event")
    failed = kept.text.index("<ability:failure", perceived)
    failure = kept.text[failed : kept.text.index("/>", failed)]
    assert f'stimulus:event="{cognition.Cognition.failed_event.name}"' in failure
    assert 'message="the bench refused the write"' in failure


class _UnreadableActor(hsm.Instance):
    """A live actor whose state cannot be read, so no frame that offers it can be built."""

    effort: typing.ClassVar[abilities.ability.Effort] = abilities.ability.Effort.XS
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "UnreadableActor",
        hsm.initial(hsm.target("/UnreadableActor/active")),
        hsm.state("active"),
    )

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        raise RuntimeError("actor state unavailable")


class _BreaksAfterDecision(hsm.Instance):
    """Readable while decision decides; unreadable once decision escalates, so the reasoning frame fails."""

    effort: typing.ClassVar[abilities.ability.Effort] = abilities.ability.Effort.XS
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "BreaksAfterDecision",
        hsm.initial(hsm.target("/BreaksAfterDecision/active")),
        hsm.state("active"),
    )
    broken: bool = False

    @typing.override
    def take_snapshot(self) -> hsm.Snapshot:
        if self.broken:
            raise RuntimeError("actor state unavailable")
        return super().take_snapshot()


class _EscalatingThenBreaking(RecordingDecisionProcessor):
    def __init__(self, actor: _BreaksAfterDecision) -> None:
        super().__init__(cognition.decision.OutputData(reason="escalate"))
        self.actor = actor

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        output = await super().process(input)
        self.actor.broken = True
        return output


async def _remembered_after_unframed_turn(
    ability: RecordingCognition, store: memory.Memory, actor: _UnreadableActor | _BreaksAfterDecision
) -> tuple[list[abilities.FailureData], tuple[memory.Episode, ...]]:
    """Run one turn whose frame offers ``actor``; return the failures and what memory kept."""

    ctx = await start_cognition_ability_for_test(ability)
    assert actor.model is not None
    _ = await mosfet.started(ctx, actor, actor.model)
    turn = await started_cognition_input(ctx)
    turn = turn.model_copy(update={"actors": {**turn.actors, "broken": actor}})
    _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "unframed-turn"))
    await wait_until(lambda: bool(ability.failures))
    await wait_until(lambda: ability.state().endswith("/idle"))
    return ability.failures, await _remembered(ctx, store)


def test_cognition_remembers_a_failed_turn_whose_decision_frame_cannot_be_built() -> None:
    """The turn cascades to reasoning (D-R-36), whose frame offers the same unreadable actor, so it fails there,
    once, and is remembered once."""

    async def run() -> tuple[list[abilities.FailureData], tuple[memory.Episode, ...]]:
        store = memory.Memory()
        ability = RecordingCognition(store=store)
        return await _remembered_after_unframed_turn(ability, store, _UnreadableActor())

    failures, episodes = asyncio.run(run())

    assert [failure.message for failure in failures] == ["actor state unavailable"]
    (kept,) = episodes
    assert kept.turn_id == "unframed-turn"
    assert kept.outcome is memory.Outcome.FAILED
    assert kept.stimulus == mosfet.InputEvent.name


def test_cognition_remembers_a_failed_turn_whose_reasoning_frame_cannot_be_built() -> None:
    async def run() -> tuple[list[abilities.FailureData], tuple[memory.Episode, ...]]:
        store = memory.Memory()
        actor = _BreaksAfterDecision()
        ability = RecordingCognition(decision_processor=_EscalatingThenBreaking(actor), store=store)
        return await _remembered_after_unframed_turn(ability, store, actor)

    failures, episodes = asyncio.run(run())

    assert [failure.message for failure in failures] == ["actor state unavailable"]
    (kept,) = episodes
    assert kept.turn_id == "unframed-turn"
    assert kept.outcome is memory.Outcome.FAILED
    assert kept.stimulus == mosfet.InputEvent.name


class _HoldingDecision(cognition.Decision):
    """Decision that takes a turn and never answers or acknowledges a cancel."""

    @staticmethod
    async def _hold(ctx: hsm.Context, instance: "_HoldingDecision", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation_id = event.id if event.id else hsm.id(instance)
        if processing.active_operation(instance, operation_id) is None:
            _ = await processing.start_operation(instance, operation_id)
        await asyncio.Event().wait()

    submodel = mosfet.define(
        "HoldingDecision",
        hsm.initial(hsm.target("/HoldingDecision/waiting")),
        hsm.state(
            "waiting",
            hsm.transition(hsm.on(cognition.Decision.input_event), hsm.target("/HoldingDecision/holding")),
        ),
        hsm.state("holding", hsm.activity(_hold.__get__(None, object))),
    )


async def _rebooted_turn(
    *,
    cancel_teardown_timeout: datetime.timedelta,
) -> tuple[list[hsm.Event[typing.Any]], tuple[memory.Episode, ...]]:
    store = memory.Memory()
    _, reasoning = cognition_abilities()
    ability = cognition.Cognition(
        decision=_HoldingDecision(processor=RecordingDecisionProcessor(no_output("unused"))),
        reasoning=reasoning,
        reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
        memory=store,
        cancel_teardown_timeout=cancel_teardown_timeout,
    )
    ctx = shared_hsm_context()
    owner = CognitionAttachmentOwner()
    _ = await mosfet.started(ctx, owner, owner.model)
    await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
    await wait_until(lambda: ability.state().endswith("/idle"))
    owner.lifecycle.clear()
    turn = (await started_cognition_input(ctx)).model_copy(update={"participants": (_ADA,)})
    _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "rebooted-turn"))
    await wait_until(lambda: ability.state().endswith("/decision"))
    _ = await hsm.dispatch(
        ctx,
        ability,
        dataclasses.replace(
            cognition.CancelEvent.with_data(cognition.CancelData(operation_id="rebooted-turn", token="token")),
            id="rebooted-turn",
            source=hsm.id(owner),
            target=hsm.id(ability),
        ),
    )
    async with asyncio.timeout(5):
        while not any(event.name == mosfet.RebootEvent.name for event in owner.lifecycle):
            await asyncio.sleep(0.01)
    return list(owner.lifecycle), await _remembered(ctx, store)


def test_cognition_remembers_a_cancelled_turn_whose_teardown_reboots_as_cancelled() -> None:
    lifecycle, episodes = asyncio.run(_rebooted_turn(cancel_teardown_timeout=datetime.timedelta(milliseconds=50)))

    assert [event.data for event in lifecycle if event.name == mosfet.RebootEvent.name] == [
        mosfet.RebootEventData(reason="cognition_cancel_teardown_failed")
    ]
    (kept,) = episodes
    assert kept.turn_id == "rebooted-turn"
    assert kept.outcome is memory.Outcome.CANCELLED
    _assert_keeps_what_was_said(kept)


def test_cancelling_an_association_acknowledges_at_once_and_remembers_the_cancelled_turn() -> None:
    """Associative has no cancel event: the cancelled association is acknowledged like autonomy's."""

    class GatedEmbedder(abilities.embedding.Embedder):
        def __init__(self, gate: asyncio.Event) -> None:
            space = abilities.embedding.Space(model="test/embedder", dimensions=2)
            self.capacity = abilities.embedding.Capacity(space=space, max_items=8, max_input_tokens=2048)
            self.attachments = ()
            self.gate = gate

        async def embed(
            self, input: abilities.embedding.InputData, *, ctx: hsm.Context
        ) -> abilities.embedding.OutputData:
            del ctx
            _ = await self.gate.wait()
            return abilities.embedding.OutputData(
                space=self.capacity.space,
                role=input.role,
                vectors=tuple(abilities.embedding.Vector(values=(1.0, 0.0)) for _ in input.items),
            )

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, tuple[memory.Episode, ...]]:
        gate = asyncio.Event()
        store = memory.Memory(
            embedder=GatedEmbedder(gate),
            embed_timeout=datetime.timedelta(seconds=5),
            operation_timeout=datetime.timedelta(seconds=10),
        )
        # The decision layer passes, so the turn goes on to associating after reflex.
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=None))
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
            associative=memory.Associative(memory=store),
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        turn = (await started_cognition_input(ctx)).model_copy(update={"participants": (_ADA,)})
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "held-turn"))
        await wait_until(lambda: ability.state().endswith("/associating"))
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                cognition.CancelEvent.with_data(cognition.CancelData(operation_id="held-turn", token="token")),
                id="held-turn",
                source=hsm.id(owner),
                target=hsm.id(ability),
            ),
        )
        await wait_until(lambda: any(event.name == cognition.CancelledEvent.name for event in owner.lifecycle))
        state = ability.state()
        gate.set()
        episodes = await _remembered(ctx, store)
        return list(owner.lifecycle), state, episodes

    lifecycle, state, episodes = asyncio.run(run())

    assert [event.name for event in lifecycle] == [cognition.CancelledEvent.name]
    assert state.endswith("/idle")
    (kept,) = episodes
    assert kept.turn_id == "held-turn"
    assert kept.outcome is memory.Outcome.CANCELLED
    _assert_keeps_what_was_said(kept)


def test_a_hung_embedder_does_not_reboot_the_next_turn() -> None:
    """A remembered turn waits on its embed outside Memory's busy state, so the next turn's reads
    (autonomy's inventory refresh, reflection) are answered in their own SQL time and no hop times out."""

    class HungEmbedder(abilities.embedding.Embedder):
        capacity: abilities.embedding.Capacity
        attachments: tuple[hsm.Instance, ...]
        calls: list[abilities.embedding.InputData]

        def __init__(self) -> None:
            space = abilities.embedding.Space(model="test/embedder", dimensions=2)
            self.capacity = abilities.embedding.Capacity(space=space, max_items=8, max_input_tokens=2048)
            self.attachments = ()
            self.calls = []

        @typing.override
        async def embed(
            self, input: abilities.embedding.InputData, *, ctx: hsm.Context
        ) -> abilities.embedding.OutputData:
            del ctx
            self.calls.append(input)
            _ = await asyncio.Event().wait()
            raise AssertionError("a hung embedder never answers")

    async def run() -> tuple[list[hsm.Event[typing.Any]], str, list[str]]:
        embedder = HungEmbedder()
        store = memory.Memory(
            embedder=embedder,
            embed_timeout=datetime.timedelta(seconds=5),
            operation_timeout=datetime.timedelta(seconds=10),
        )
        decision, reasoning = cognition_abilities()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            autonomy=cognition.Autonomy(memory=store),
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
            child_operation_timeout=datetime.timedelta(milliseconds=500),
        )
        ctx = shared_hsm_context()
        owner = CognitionAttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "first")
        )
        async with asyncio.timeout(3):
            while not embedder.calls:
                await asyncio.sleep(0.01)
        _ = await hsm.dispatch(
            ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "next")
        )
        await asyncio.sleep(1.5)
        lifecycle, state = list(owner.lifecycle), ability.state()
        return lifecycle, state, [episode.turn_id for episode in await _remembered(ctx, store)]

    lifecycle, state, remembered = asyncio.run(run())

    assert [event.name for event in lifecycle] == []
    assert state.endswith("/idle")
    assert remembered == ["first", "next"]


# --- deliberate recall --------------------------------------------------------------------------


def _recall_output(recall: memory.RecallData) -> cognition.types.OutputData:
    return (
        cognition.types.EventData(
            event=memory.RecallEvent.name,
            data=recall.model_dump(exclude_none=True),
            reason="think about it",
        ),
    )


async def _recalling_cognition(
    associative: memory.Associative, store: memory.Memory
) -> tuple[
    cognition.Cognition,
    hsm.Context,
    _RoutineTickBody,
    RecordingDecisionProcessor,
    RecordingReasoningProcessor,
]:
    decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over"))
    reasoning_processor = RecordingReasoningProcessor(no_output("placeholder"))
    decision, reasoning = cognition_abilities(
        decision_processor=decision_processor, reasoning_processor=reasoning_processor
    )
    ability = cognition.Cognition(
        decision=decision,
        reasoning=reasoning,
        reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
        memory=store,
        associative=associative,
    )
    body = _RoutineTickBody()
    ctx = shared_hsm_context()
    _ = await mosfet.started(ctx, body, body.model)
    await ability.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=body), "attach"))
    await wait_until(lambda: ability.state().endswith("/idle"))
    return ability, ctx, body, decision_processor, reasoning_processor


def test_reasoning_may_recall_a_memory_and_the_recollection_becomes_an_observation() -> None:
    """Recall is offered to reasoning (effort M) and not to decision; what comes to mind is a new turn."""

    async def run() -> tuple[
        list[processing.InputData], list[processing.InputData], hsm.Event[typing.Any], str, str, str, str
    ]:
        store = memory.Memory()
        associative = memory.Associative(memory=store)
        ability, ctx, body, decision_processor, reasoning_processor = await _recalling_cognition(associative, store)
        try:
            await _remember_by_message(
                ctx,
                store,
                memory.EpisodeData(
                    turn_id="earlier",
                    parts=(text.TextPart(text="Ada asked for a call at noon."),),
                    participants=(_ADA,),
                    outcome=memory.Outcome.COMPLETED,
                ),
            )
            (earlier,) = await _remembered(ctx, store)
            reasoning_processor.output = cognition.reasoning.OutputData(
                result=_recall_output(memory.RecallData(node=earlier.node_id))
            )
            _ = await hsm.dispatch(
                ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn-1")
            )
            await _until_routines(lambda: bool(body.handoffs), timeout=5.0)
            return (
                decision_processor.calls,
                reasoning_processor.calls,
                body.handoffs[0],
                earlier.node_id,
                hsm.id(ability),
                hsm.id(associative),
                # Read while it runs: stopping cognition stops the faculties it started.
                hsm.id(reasoning_processor.calls[0].actors["memory"]),
            )
        finally:
            await ability.stop(ctx)

    decision_calls, reasoning_calls, handoff, node_id, cognition_id, associative_id, offered_memory_id = asyncio.run(
        run()
    )

    assert memory.RecallEvent.name not in {event.name for event in decision_calls[0].schemas}
    (reasoning_frame,) = reasoning_calls
    assert memory.RecallEvent.name in {event.name for event in reasoning_frame.schemas}
    assert offered_memory_id == associative_id
    assert handoff.name == cognition.InputEvent.name
    assert handoff.source == cognition_id
    data = handoff.data
    assert isinstance(data, cognition.InputData)
    assert data.participants == ()
    stimulus = data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == memory.RecollectedEvent.name
    assert stimulus.source == associative_id
    recollection = stimulus.data
    assert isinstance(recollection, memory.RecollectionData)
    assert [node.node_id for node in recollection.nodes] == [node_id]
    assert recollection.nodes[0].text == "Ada asked for a call at noon."


def test_a_recall_that_fails_is_observed_too_never_silently_dropped() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str]:
        store = memory.Memory()
        # Its memory is never attached or started: every recall fails, typed.
        associative = memory.Associative(memory=memory.Memory())
        ability, ctx, body, _, reasoning_processor = await _recalling_cognition(associative, store)
        try:
            reasoning_processor.output = cognition.reasoning.OutputData(
                result=_recall_output(memory.RecallData(about="the noon call"))
            )
            _ = await hsm.dispatch(
                ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn-1")
            )
            await _until_routines(lambda: bool(body.handoffs), timeout=5.0)
            return body.handoffs[0], hsm.id(associative)
        finally:
            await ability.stop(ctx)

    handoff, associative_id = asyncio.run(run())

    data = handoff.data
    assert isinstance(data, cognition.InputData)
    stimulus = data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == abilities.FailedEvent.name
    assert stimulus.source == associative_id
    failure = stimulus.data
    assert isinstance(failure, abilities.FailureData)
    assert failure.message.startswith("unavailable")


def test_reasoning_may_know_a_fact_about_a_person_and_what_memory_says_of_it_becomes_an_observation() -> None:
    """R-12: knowing who someone is is memory's capacity, offered to reasoning; its outcome is a new turn."""

    async def run() -> tuple[list[processing.InputData], list[processing.InputData], hsm.Event[typing.Any], str]:
        store = memory.Memory()
        associative = memory.Associative(memory=store)
        ability, ctx, body, decision_processor, reasoning_processor = await _recalling_cognition(associative, store)
        try:
            await _remember_by_message(
                ctx,
                store,
                memory.EpisodeData(
                    turn_id="heard",
                    parts=(text.TextPart(text="Hi, I'm Ada."),),
                    participants=(_ADA,),
                    outcome=memory.Outcome.COMPLETED,
                ),
            )
            know = memory.KnowData(about=_ADA.ref, fact=memory.NameFact(name="Ada"))
            reasoning_processor.output = cognition.reasoning.OutputData(
                result=(
                    cognition.types.EventData(
                        event=memory.KnowEvent.name, data=know.model_dump(exclude_none=True), reason="she said so"
                    ),
                )
            )
            _ = await hsm.dispatch(
                ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn-1")
            )
            await _until_routines(lambda: bool(body.handoffs), timeout=5.0)
            return decision_processor.calls, reasoning_processor.calls, body.handoffs[0], hsm.id(associative)
        finally:
            await ability.stop(ctx)

    decision_calls, reasoning_calls, handoff, associative_id = asyncio.run(run())

    offered = {event.name for event in reasoning_calls[0].schemas}
    assert {memory.KnowEvent.name, memory.ForgetEvent.name} <= offered
    assert not {memory.KnowEvent.name, memory.ForgetEvent.name} & {event.name for event in decision_calls[0].schemas}
    data = handoff.data
    assert isinstance(data, cognition.InputData)
    stimulus = data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == memory.KnownEvent.name
    assert stimulus.source == associative_id
    known = stimulus.data
    assert isinstance(known, memory.KnownData)
    assert (known.op, known.person.name, known.person.refs) == ("know", "Ada", (_ADA.ref,))


# --- rest ---------------------------------------------------------------------------------------


class _LeaveAsIs(processing.Processor):
    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input
        return ()


class _RecordingConsolidation(memory.Consolidation):
    """Consolidation that records the rest and wake notices Cognition sends it (test-only observation)."""

    notices: list[str]

    def __init__(self, *, memory: memory.Memory) -> None:
        super().__init__(memory=memory, processor=_LeaveAsIs())
        self.notices = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        if event.name in (memory.RestEvent.name, memory.WakeEvent.name):
            self.notices.append(event.name.rsplit(".", 1)[-1])
        return super().dispatch(ctx, event)


async def _until_rest(condition: collections.abc.Callable[[], bool], *, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


async def _resting_cognition(
    rest_after: datetime.timedelta,
) -> tuple[cognition.Cognition, _RecordingConsolidation, hsm.Context, _RoutineTickBody]:
    store = memory.Memory()
    consolidation = _RecordingConsolidation(memory=store)
    decision, reasoning = cognition_abilities(
        decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=no_output("fast")))
    )
    ability = cognition.Cognition(
        decision=decision,
        reasoning=reasoning,
        reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
        memory=store,
        consolidation=consolidation,
        rest_after=rest_after,
    )
    body = _RoutineTickBody()
    ctx = shared_hsm_context()
    _ = await mosfet.started(ctx, body, body.model)
    await ability.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=body), "attach"))
    await _until_rest(lambda: ability.state().endswith("/ready/awake"))
    return ability, consolidation, ctx, body


def test_cognition_rest_needs_memory_and_a_positive_rest_after() -> None:
    store = memory.Memory()
    decision, reasoning = cognition_abilities()
    reflection = cognition.Reflection(processor=FixedProcessor(no_output("unused")), memory=store)
    with pytest.raises(ValueError, match="consolidation requires memory"):
        _ = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=reflection,
            consolidation=memory.Consolidation(memory=store, processor=_LeaveAsIs()),
        )
    with pytest.raises(ValueError, match="rest_after must be positive"):
        _ = cognition.Cognition(
            decision=decision, reasoning=reasoning, reflection=reflection, rest_after=datetime.timedelta(0)
        )
    members = require_model(cognition.Cognition.model).members
    for state in ("ready", "ready/idle", "ready/awake", "ready/resting"):
        assert f"/CognitionLifecycle/attached/behavior/{state}" in members


def test_cognition_rests_after_inactivity_and_any_input_wakes_it() -> None:
    async def run() -> tuple[list[str], list[str], list[str], str]:
        ability, consolidation, ctx, _ = await _resting_cognition(datetime.timedelta(milliseconds=50))
        try:
            await _until_rest(lambda: consolidation.notices == ["rest"])
            resting = ability.state()
            _ = await hsm.dispatch(
                ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn-1")
            )
            await _until_rest(lambda: consolidation.notices[-1:] == ["wake"])
            woken = list(consolidation.notices)
            # Idle again, so it rests again; the next turn wakes it the same way.
            await _until_rest(lambda: consolidation.notices[-1:] == ["rest"])
            _ = await hsm.dispatch(
                ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn-2")
            )
            await _until_rest(lambda: consolidation.notices[2:4] == ["rest", "wake"])
            by_turn = list(consolidation.notices[:4])
            return woken, by_turn, list(consolidation.notices), resting
        finally:
            await ability.stop(ctx)

    woken, by_turn, notices, resting = asyncio.run(run())

    assert resting.endswith("/ready/resting")
    assert woken == ["rest", "wake"]
    assert by_turn == ["rest", "wake", "rest", "wake"]
    # Rest and wake always alternate: every rest is woken once.
    assert all(a != b for a, b in itertools.pairwise(notices))


def test_a_turn_while_awake_restarts_the_inactivity_timer() -> None:
    async def run() -> tuple[list[str], list[str]]:
        ability, consolidation, ctx, _ = await _resting_cognition(datetime.timedelta(milliseconds=500))
        try:
            await asyncio.sleep(0.35)
            # The turn runs to its terminal and returns through ready/settling to a fresh awake.
            _ = await hsm.dispatch(
                ctx, ability, ability.input_event.with_data_and_id(await started_cognition_input(ctx), "turn-1")
            )
            await _until_rest(lambda: ability.state().endswith("/ready/awake"))
            await asyncio.sleep(0.35)
            # 0.7 s since attach: without the restart it would have rested at 0.5 s.
            before = list(consolidation.notices)
            await _until_rest(lambda: consolidation.notices == ["rest"])
            return before, list(consolidation.notices)
        finally:
            await ability.stop(ctx)

    before, after = asyncio.run(run())

    assert before == []
    assert after == ["rest"]


_TRAITS = persona.Traits(traits=("curious", "patient with children"))


def _self_block(frame: processing.InputData) -> xml.etree.ElementTree.Element | None:
    """The self block a frame shows the model, its stable instructions, or None."""

    instructions = frame.instructions or ""
    start = instructions.find("<persona>")
    return None if start < 0 else xml.etree.ElementTree.fromstring(instructions[start:])


def _description(frame: processing.InputData) -> str | None:
    block = _self_block(frame)
    assert block is not None
    description = block.find("description")
    assert description is not None and description.get("written_by") == "you"
    return description.text


class _TellingPersona(persona.Persona):
    """Persona that keeps what it tells its owner, so a test can wait until it was told."""

    told: list[persona.OutputData]

    def __init__(self) -> None:
        super().__init__(traits=_TRAITS)
        self.told = []

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        terminal = event.data
        if isinstance(terminal, hsm.Event) and isinstance(terminal.data, persona.OutputData):
            self.told.append(terminal.data)
        return super().dispatch(ctx, event)


async def _until_persona(condition: collections.abc.Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.001)
    # Persona's telling is one more hop to Cognition after persona settles.
    for _ in range(50):
        await asyncio.sleep(0)


def test_cognition_attaches_persona_and_shows_what_it_told_on_every_turn() -> None:
    """Every frame shows the traits and the self-written description persona last told, by event.

    The revision is dispatched by the test itself, standing in for reflection: this pins the
    capacity to know oneself, never a choice to revise.
    """

    async def run() -> tuple[list[processing.InputData], list[processing.InputData], str]:
        self_ = persona.Persona(traits=_TRAITS)
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over"))
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        decision, reasoning = cognition_abilities(
            decision_processor=decision_processor, reasoning_processor=reasoning_processor
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
            persona=self_,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await _until_persona(lambda: ability.state().endswith("/idle") and self_.state().endswith("/knowing"))
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        _ = await hsm.dispatch(
            ctx,
            self_,
            persona.ReviseEvent.with_data(persona.ReviseData(text="I explain things slowly.", reason="A long talk.")),
        )
        await _until_persona(lambda: self_.state().endswith("/knowing"))
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        # Cognition attached it: persona is a child, in its own attached behavior.
        attached = self_.state()
        await ability.stop(ctx)
        return decision_processor.calls, reasoning_processor.calls, attached

    decision_calls, reasoning_calls, attached = asyncio.run(run())

    assert attached == "/PersonaLifecycle/attached/behavior/knowing"
    assert len(decision_calls) == len(reasoning_calls) == 2
    for frame in (*decision_calls, *reasoning_calls):
        block = _self_block(frame)
        assert block is not None
        traits = block.find("traits")
        assert traits is not None and traits.get("given_by") == "owner"
        assert [trait.text for trait in traits.findall("trait")] == list(_TRAITS.traits)
    # Before any revision the bot knows it has written nothing; after one, it knows what it wrote.
    assert [_description(frame) for frame in (decision_calls[0], reasoning_calls[0])] == [None, None]
    assert [_description(frame) for frame in (decision_calls[1], reasoning_calls[1])] == [
        "I explain things slowly.",
        "I explain things slowly.",
    ]


def test_cognition_without_persona_shows_no_self_block() -> None:
    async def run() -> list[processing.InputData]:
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        ability = make_cognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
        )
        ctx = await start_cognition_ability_for_test(ability)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await ability.stop(ctx)
        return reasoning_processor.calls

    (frame,) = asyncio.run(run())

    assert _self_block(frame) is None


class _EchoDecoder(speech.SpeechDecoder):
    """Stands in for STT: the bytes are the words."""

    @typing.override
    async def decode(self, input: bytes) -> bytes:
        return input


def _name_line(frame: processing.InputData) -> str | None:
    """The bot's own name as the self block shows it: the name, ``none yet``, or None when it says nothing."""

    block = _self_block(frame)
    if block is None:
        return None
    named = block.find("name")
    if named is not None:
        assert named.get("taken_by") == "you"
        return named.text
    return "none yet" if block.find("no_name_yet") is not None else None


def test_cognition_shows_the_name_its_identity_tells_it_none_yet_before_one_is_taken_and_again_after_a_restart() -> (
    None
):
    """The bot knows its own name (P-7): Identity tells Cognition, by event, on waking and after an adoption.

    The adoption is dispatched by the test itself, standing in for the bot's selection (the selection's source is
    the bot's cognition): this pins the capacity to know its name, never a choice to take one.
    """

    async def life(store: memory.Memory, *, adopt: str | None) -> list[processing.InputData]:
        ears = identity.Identity(recognizer=identity.SpeechNameRecognizer(decoder=_EchoDecoder()), memory=store)
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
        )
        mind = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
            identity=ears,
        )
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, ears)
        ctx = await start_cognition_ability_for_test(mind, ctx)
        await wait_until(lambda: mind.state().endswith("/idle") and ears.state().endswith("/knowing/listening"))
        for _ in range(50):
            await asyncio.sleep(0)
        _ = await dispatch_ability_for_test(mind, ctx, await started_cognition_input(ctx))
        if adopt is not None:
            _ = await hsm.dispatch(
                ctx,
                ears,
                dataclasses.replace(identity.AdoptEvent.with_data(identity.AdoptData(name=adopt)), source=hsm.id(mind)),
            )
            await wait_until(lambda: ears.state().endswith("/knowing/listening"))
            for _ in range(50):
                await asyncio.sleep(0)
            _ = await dispatch_ability_for_test(mind, ctx, await started_cognition_input(ctx))
        await mind.stop(ctx)
        await hsm.stop(ears, ctx)
        return reasoning_processor.calls

    async def run() -> tuple[list[processing.InputData], list[processing.InputData]]:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            engine = memory.store.open_sqlite_engine(connection=connection)
            first = await life(memory.Memory(engine=engine), adopt="Bob")
            # A new body over the same database: the name was written down when it was taken.
            again = await life(memory.Memory(engine=engine), adopt=None)
            return first, again
        finally:
            connection.close()

    first, again = asyncio.run(run())

    assert [_name_line(frame) for frame in first] == ["none yet", "Bob"]
    assert [_name_line(frame) for frame in again] == ["Bob"]
    # Stable between turns: only a rename changes the self block.
    assert len({frame.instructions for frame in again}) == 1


def test_cognition_without_identity_shows_no_name_at_all() -> None:
    """A bot with no Identity cannot know a name either way, so its self block claims nothing about one."""

    async def run() -> list[processing.InputData]:
        self_ = persona.Persona(traits=_TRAITS)
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
            persona=self_,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await _until_persona(lambda: ability.state().endswith("/idle") and self_.state().endswith("/knowing"))
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await ability.stop(ctx)
        return reasoning_processor.calls

    (frame,) = asyncio.run(run())

    assert _self_block(frame) is not None
    assert _name_line(frame) is None


def test_cognition_keeps_only_what_its_own_persona_told() -> None:
    """A persona output from any other source is not the bot's self-knowledge."""

    async def run() -> list[processing.InputData]:
        self_ = persona.Persona(traits=_TRAITS)
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
            persona=self_,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await _until_persona(lambda: ability.state().endswith("/idle") and self_.state().endswith("/knowing"))
        forged = persona.OutputData(
            traits=("obedient",),
            description=persona.Description(text="I do as I am told.", reason="forged", traits=("obedient",)),
        )
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(persona.OutputEvent.with_data(forged), source="someone-else", target=hsm.id(ability)),
        )
        for _ in range(50):
            await asyncio.sleep(0)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await ability.stop(ctx)
        return reasoning_processor.calls

    (frame,) = asyncio.run(run())

    block = _self_block(frame)
    assert block is not None
    assert [trait.text for trait in block.iter("trait")] == list(_TRAITS.traits)
    assert _description(frame) is None


def test_a_revision_reflection_selects_is_known_on_the_next_turn() -> None:
    """Reflection sees what the bot knows of itself; what it revises comes back by event to every frame.

    The fixed reflection processor stands in for the model's own selection: this pins the path a
    revision takes, never that a bot revises.
    """

    revise = (
        cognition.types.EventData(
            event=persona.ReviseEvent.name,
            data={"text": "I explain things slowly.", "reason": "A long talk with a child."},
            reason="I noticed it.",
        ),
    )

    async def run() -> tuple[list[processing.InputData], list[processing.InputData]]:
        self_ = _TellingPersona()
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        reflection_processor = FixedProcessor(revise)
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=reflection_processor, memory=memory.Memory(), persona=self_),
            persona=self_,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await _until_persona(lambda: ability.state().endswith("/idle") and self_.state().endswith("/knowing"))
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await _until_persona(lambda: len(self_.told) == 2 and ability.state().endswith("/idle"))
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await ability.stop(ctx)
        return reasoning_processor.calls, reflection_processor.step.calls

    reasoning_calls, reflection_calls = asyncio.run(run())

    select = reflection_calls[0].input
    assert isinstance(select, cognition.reflection.SelectInput)
    assert select.persona == persona.OutputData(traits=_TRAITS.traits)
    assert persona.ReviseEvent in reflection_calls[0].schemas
    assert [_description(frame) for frame in reasoning_calls] == [None, "I explain things slowly."]


# --- the voice given at inception ---------------------------------------------------------------

_WARM = vocal.voice.Entry(
    ref=vocal.voice.Ref(source="lab", key="warm"),
    name="warm",
    description="A warm, unhurried voice.",
    languages=("en",),
)
_VOICED = vocal.speech.Expressiveness(cues=frozenset({"instructions", "emotion"}))


class _Catalog(vocal.voice.Catalog):
    """A voice source that holds the given voices and designs none."""

    def __init__(self, *entries: vocal.voice.Entry) -> None:
        self.source: str = "lab"
        self.entries: tuple[vocal.voice.Entry, ...] = entries

    @typing.override
    async def voices(
        self, *, languages: tuple[vocal.speech.Locale, ...] = (), descriptors: vocal.voice.Descriptors | None = None
    ) -> tuple[vocal.voice.Entry, ...]:
        del languages, descriptors
        return self.entries


class _Shortlisting(processing.Processor):
    """The makers' inception model, scripted to shortlist the warm voice."""

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        del input
        conceived = {"conception": {"kind": "shortlisted", "ref": {"source": "lab", "key": "warm"}}}
        return (processing.SelectedEvent(event=vocal.voice.ConceiveEvent.name, data=conceived),)


class _VoiceExpressively(_NoopSpeechEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.expressiveness: vocal.speech.Expressiveness = _VOICED


async def _inception(ctx: hsm.Context, catalog: _Catalog) -> vocal.voice.Inception:
    """An inception whose Speaking (built with an encoder factory) is started in ``ctx``."""

    mouth = speaking.Speaking(encoder=lambda key, description: _VoiceExpressively(), speaker=audio.Speaker())
    await start_ability_tree(ctx, mouth)
    return vocal.voice.Inception(
        traits=_TRAITS,
        languages=("en",),
        speaking=mouth,
        processor=_Shortlisting(),
        catalog=catalog,
        allowed=tuple(entry.ref.key for entry in catalog.entries),
        retry_after=datetime.timedelta(seconds=30),
    )


def _voice(frame: processing.InputData) -> xml.etree.ElementTree.Element | None:
    block = _self_block(frame)
    assert block is not None
    return block.find("voice")


def test_cognition_attaches_inception_and_shows_the_voice_the_bot_was_given_on_every_turn() -> None:
    async def run() -> tuple[list[processing.InputData], str]:
        # Speaking's mouth is a device: it is started in an environment.
        ctx = Environment(shared_hsm_context())
        inception = await _inception(ctx, _Catalog(_WARM))
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
            voice=inception,
        )
        await start_cognition_ability_for_test(ability, ctx)
        await _until_persona(lambda: ability.state().endswith("/idle") and inception.state().endswith("/made"))
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        attached = inception.state()
        await ability.stop(ctx)
        return reasoning_processor.calls, attached

    (frame,), attached = asyncio.run(run())

    assert attached == "/InceptionLifecycle/attached/behavior/made"
    voiced = _voice(frame)
    assert voiced is not None
    assert (voiced.get("given_at"), voiced.get("languages"), voiced.text) == (
        "inception",
        "en",
        "A warm, unhurried voice.",
    )
    expressiveness = voiced.find("expressiveness")
    assert expressiveness is not None
    cues = expressiveness.find("cues")
    assert cues is not None and cues.text == "emotion instructions"
    # Without a persona the self block still shows the voice, and nothing else.
    block = _self_block(frame)
    assert block is not None and [child.tag for child in block] == ["voice"]


def test_cognition_keeps_only_the_voice_its_own_inception_told() -> None:
    async def run() -> list[processing.InputData]:
        # Speaking's mouth is a device: it is started in an environment.
        ctx = Environment(shared_hsm_context())
        # A catalog that holds no voice: this inception never makes one.
        inception = await _inception(ctx, _Catalog())
        reasoning_processor = RecordingReasoningProcessor(focus_output("phone", "reasoned focus"))
        decision, reasoning = cognition_abilities(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(reason="think it over")),
            reasoning_processor=reasoning_processor,
        )
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
            voice=inception,
        )
        await start_cognition_ability_for_test(ability, ctx)
        await _until_persona(lambda: ability.state().endswith("/idle"))
        forged = vocal.voice.VoiceData(
            ref=vocal.voice.Ref(source="lab", key="forged"),
            origin="catalog",
            description="Somebody else's voice.",
            expressiveness=vocal.speech.Expressiveness(),
        )
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                vocal.voice.OutputEvent.with_data(forged), source="someone-else", target=hsm.id(ability)
            ),
        )
        for _ in range(50):
            await asyncio.sleep(0)
        _ = await dispatch_ability_for_test(ability, ctx, await started_cognition_input(ctx))
        await ability.stop(ctx)
        return reasoning_processor.calls

    (frame,) = asyncio.run(run())

    assert _self_block(frame) is None


def test_what_inception_fails_at_is_observed_by_the_body_never_silently_dropped() -> None:
    async def run() -> tuple[hsm.Event[typing.Any], str]:
        # Speaking's mouth is a device: it is started in an environment.
        ctx = Environment(shared_hsm_context())
        inception = await _inception(ctx, _Catalog())
        decision, reasoning = cognition_abilities()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=memory.Memory()),
            voice=inception,
        )
        body = _RoutineTickBody()
        _ = await mosfet.started(ctx, body, body.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=body), "attach"))
        try:
            await _until_routines(lambda: bool(body.handoffs), timeout=5.0)
            return body.handoffs[0], hsm.id(inception)
        finally:
            await ability.stop(ctx)

    handoff, inception_id = asyncio.run(run())

    data = handoff.data
    assert isinstance(data, cognition.InputData)
    stimulus = data.stimulus
    assert isinstance(stimulus, hsm.Event)
    assert stimulus.name == abilities.FailedEvent.name
    assert stimulus.source == inception_id
    failure = stimulus.data
    assert isinstance(failure, abilities.FailureData)
    assert "catalog holds no voices" in failure.message


# --- a batch turn: what queued while the bot was busy, perceived together ------------------------

_BOB = mosfet.Participant(ref="person-7", role=mosfet.ParticipantRole.SOURCE)


def _typed_percept(line: str, *, person: mosfet.Participant) -> mosfet.PerceptData:
    """A line typed on the desk computer while the bot was busy, as the body holds it."""

    return mosfet.PerceptData(
        stimulus=mosfet.InputEventData(
            target_device="phone",
            observation=mosfet.StimulusData.from_event(
                computer.TextEnteredEvent.with_data(
                    computer.TextEnteredData(device_id="desk", person_id=person.ref, participants=(person,), text=line)
                )
            ),
            participants=(person,),
        ),
        source="desk-1",
        participants=(person,),
    )


def _failed_percept(
    turn_id: str,
    *,
    timed_out: bool = False,
    stage: typing.Literal["reflex", "intuition", "reasoning"] = "reflex",
) -> mosfet.PerceptData:
    """A failed step cognition handed back, as the body holds it."""

    observed = (
        cognition.StepTimedOutEvent.with_data(
            cognition.StepTimedOutData(
                message=f"Cognition's {stage} step took longer than 0.5 seconds, so the turn failed.",
                stage=stage,
                timeout_seconds=0.5,
                turn_id=turn_id,
                stimulus=mosfet.InputEvent.name,
            )
        )
        if timed_out
        else cognition.StepFailedEvent.with_data(
            cognition.StepFailedData(
                message=f"Cognition's {stage} step failed, so the turn failed: the step failed",
                stage=stage,
                turn_id=turn_id,
                stimulus=mosfet.InputEvent.name,
            )
        )
    )
    return mosfet.PerceptData(stimulus=dataclasses.replace(observed, source="cognition-1"), source="cognition-1")


def _batch_turn(base: cognition.InputData, *percepts: mosfet.PerceptData) -> cognition.InputData:
    """The cognition input a body hands for a batch: the batch, and the union of who took part, in order."""

    union: list[mosfet.Participant] = []
    for percept in percepts:
        union.extend(participant for participant in percept.participants if participant not in union)
    batch = dataclasses.replace(
        mosfet.PerceivedEvent.with_data(mosfet.PerceivedData(percepts=percepts)), source="bot-1"
    )
    return base.model_copy(update={"stimulus": batch, "participants": tuple(union)})


def test_a_batch_turn_is_remembered_as_one_episode_of_everything_it_perceived() -> None:
    async def run() -> tuple[cognition.types.OutputData, tuple[memory.Episode, ...]]:
        store = memory.Memory()
        ability = RecordingCognition(
            decision_processor=RecordingDecisionProcessor(cognition.decision.OutputData(result=no_output("both"))),
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=store),
            store=store,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        turn = _batch_turn(
            await started_cognition_input(ctx),
            _typed_percept("first line", person=_ADA),
            _typed_percept("second line", person=_BOB),
            _typed_percept("third line", person=_ADA),
        )
        result = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: ability.state().endswith("/idle"))
        return result, await _remembered(ctx, store)

    result, episodes = asyncio.run(run())

    assert result == no_output("both")
    (kept,) = episodes
    assert kept.stimulus == "bot.perceived"
    assert kept.participants == (_ADA, _BOB)
    assert kept.outcome is memory.Outcome.COMPLETED
    first, second, third = (kept.text.index(f'text="{line}"') for line in ("first line", "second line", "third line"))
    assert first < second < third
    assert 'omitted="size"' not in kept.text


@pytest.mark.parametrize(
    ("percepts", "handed_back"),
    [
        ((_failed_percept("failed-1"), _typed_percept("are you there?", person=_ADA)), True),
        ((_failed_percept("failed-1"), _failed_percept("failed-2", timed_out=True)), False),
    ],
    ids=["mixed", "only-failed-steps"],
)
def test_a_failed_batch_turn_is_handed_back_once_unless_it_perceived_only_failed_steps(
    percepts: tuple[mosfet.PerceptData, ...],
    handed_back: bool,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D-W12-6 (Q6): a batch whose turn fails is handed back once, so the next turn perceives that failure
    with whatever else queued; a batch that perceived nothing but failed steps is not handed back, so a
    failing step cannot wake the bot more often than new stimuli arrive. A batch holding an decision
    failure is decided by reasoning (D-R-30), so here it is reasoning's step that fails."""

    def outcomes(reader: metric_reader.Reader) -> dict[str, float]:
        return {
            str(point.attributes["outcome"]): point.value
            for point in reader.points("bot.cognition.step.failure.count")
            if point.attributes.get("stage") == "reasoning" and point.attributes.get("kind") == "failed"
        }

    async def run() -> list[hsm.Event[typing.Any]]:
        decision, reasoning = cognition_abilities(
            decision_processor=_FailingStepProcessor(), reasoning_processor=_FailingStepProcessor()
        )
        store = memory.Memory()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
        )
        ctx = shared_hsm_context()
        body = _StepTimeoutBody()
        _ = await mosfet.started(ctx, body, body.model)
        _ = await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=body)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        turn = _batch_turn(await started_cognition_input(ctx), *percepts)
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "batch-turn"))
        await wait_until(lambda: any(event.name == cognition.Cognition.failed_event.name for event in body.told))
        await wait_until(lambda: ability.state().endswith("/idle"))
        for _ in range(50):
            await asyncio.sleep(0)
        return list(body.told)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = outcomes(reader)
        told = asyncio.run(run())
        after = outcomes(reader)

    handed = [event for event in told if event.name == cognition.InputEvent.name]
    outcome = "handed_back" if handed_back else "bounded"
    assert after.get(outcome, 0.0) - before.get(outcome, 0.0) == 1.0
    assert len(handed) == (1 if handed_back else 0)
    if handed_back:
        (perceived,) = handed
        assert isinstance(perceived.data, cognition.InputData)
        assert perceived.data.participants == (_ADA,)
        failure = perceived.data.stimulus
        assert isinstance(failure, hsm.Event)
        assert isinstance(failure.data, cognition.StepFailedData)
        assert (failure.data.stage, failure.data.turn_id, failure.data.stimulus) == (
            "reasoning",
            "batch-turn",
            "bot.perceived",
        )


def test_a_batch_turn_whose_step_outlasts_its_timeout_names_bot_perceived_as_its_stimulus() -> None:
    async def run() -> list[hsm.Event[typing.Any]]:
        decision, reasoning = cognition_abilities(decision_processor=_SlowFirstStepProcessor(hangs=True))
        store = memory.Memory()
        ability = cognition.Cognition(
            decision=decision,
            reasoning=reasoning,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("reflection")), memory=store),
            memory=store,
            child_operation_timeout=datetime.timedelta(milliseconds=200),
        )
        ctx = shared_hsm_context()
        body = _StepTimeoutBody()
        _ = await mosfet.started(ctx, body, body.model)
        _ = await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=body)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        turn = _batch_turn(
            await started_cognition_input(ctx),
            _typed_percept("first line", person=_ADA),
            _typed_percept("second line", person=_BOB),
        )
        _ = await hsm.dispatch(ctx, ability, ability.input_event.with_data_and_id(turn, "slow-batch"))
        await wait_until(lambda: any(event.name == cognition.InputEvent.name for event in body.told))
        return list(body.told)

    told = asyncio.run(run())

    (failed,) = [event for event in told if event.name == cognition.Cognition.failed_event.name]
    assert isinstance(failed.data, cognition.StepTimedOutData)
    assert (failed.data.turn_id, failed.data.stimulus) == ("slow-batch", "bot.perceived")
    (perceived,) = [event for event in told if event.name == cognition.InputEvent.name]
    assert isinstance(perceived.data, cognition.InputData)
    assert perceived.data.participants == (_ADA, _BOB)


@pytest.mark.parametrize(
    ("perceived", "decider"),
    [
        ("reflex-failed", "reasoning"),
        ("reflex-timed-out", "reasoning"),
        ("batch-holding-a-reflex-failure", "reasoning"),
        ("reasoning-failed", "decision"),
    ],
)
def test_a_turn_perceiving_a_handed_back_reflex_failure_passes_the_decision_layer(perceived: str, decider: str) -> None:
    """D-R-30, D-R-52: a handed-back reflex failure went back to the decision layer, which failed on it again,
    and the bound stopped it, so a later stage never saw it. A turn that perceives one (as its stimulus or as
    any percept of its batch) starts after reflex; one that perceives reasoning's failure does not."""

    async def run() -> tuple[
        cognition.types.OutputData, cognition.InputData, list[processing.InputData], list[processing.InputData]
    ]:
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=no_output("fast")))
        reasoning_processor = RecordingReasoningProcessor(no_output("reasoned"))
        ability = RecordingCognition(decision_processor=decision_processor, reasoning_processor=reasoning_processor)
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        base = await started_cognition_input(ctx)
        turn = (
            _batch_turn(base, _typed_percept("are you there?", person=_ADA), _failed_percept("failed-1"))
            if perceived == "batch-holding-a-reflex-failure"
            else base.model_copy(
                update={
                    "stimulus": _failed_percept(
                        "failed-1",
                        timed_out=perceived == "reflex-timed-out",
                        stage="reasoning" if perceived == "reasoning-failed" else "reflex",
                    ).stimulus
                }
            )
        )
        result = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: ability.state().endswith("/idle"))
        return result, turn, decision_processor.calls, reasoning_processor.calls

    result, turn, decision_calls, reasoning_calls = asyncio.run(run())

    if decider == "reasoning":
        assert result == no_output("reasoned")
        assert decision_calls == []
        (frame,) = reasoning_calls
        assert isinstance(frame.input, cognition.reasoning.ProcessorInput)
        assert frame.input.host_input.input == turn.stimulus
    else:
        assert result == no_output("fast")
        assert reasoning_calls == []
        (frame,) = decision_calls
        assert frame.input == turn.stimulus


@pytest.mark.parametrize(("stage", "decider"), [("reflex", "intuition"), ("intuition", "reasoning")])
def test_a_turn_perceiving_a_handed_back_failure_starts_at_the_first_stage_after_it(stage: str, decider: str) -> None:
    """D-R-52: a turn perceiving a handed-back failure of stage S starts at the first stage after S."""

    async def run() -> tuple[list[processing.InputData], list[processing.InputData], list[processing.InputData]]:
        decision_processor = RecordingDecisionProcessor(cognition.decision.OutputData(result=no_output("fast")))
        intuition_processor = RecordingReasoningProcessor(no_output("intuited"))
        reasoning_processor = RecordingReasoningProcessor(no_output("reasoned"))
        ability = RecordingCognition(
            decision_processor=decision_processor,
            intuition=cognition.Intuition(processor=intuition_processor),
            reasoning_processor=reasoning_processor,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        base = await started_cognition_input(ctx)
        failed = _failed_percept("failed-1", timed_out=True, stage=typing.cast(typing.Any, stage))
        _ = await dispatch_ability_for_test(ability, ctx, base.model_copy(update={"stimulus": failed.stimulus}))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return decision_processor.calls, intuition_processor.calls, reasoning_processor.calls

    decision_calls, intuition_calls, reasoning_calls = asyncio.run(run())

    assert decision_calls == []
    assert (len(intuition_calls), len(reasoning_calls)) == ((1, 0) if decider == "intuition" else (0, 1))


def _record_admitted(ctx: hsm.Context, instance: "_Communication", event: hsm.Event[typing.Any]) -> None:
    del ctx
    instance.admitted.append(event)


class _Communication(hsm.Instance):
    """Minimal communication ability: records the speech a behavior admits to it."""

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "CognitionTestCommunication",
        hsm.initial(hsm.target("/CognitionTestCommunication/ready")),
        hsm.state("ready", hsm.transition(hsm.on(communication.InputEvent), hsm.effect(_record_admitted))),
    )
    admitted: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.admitted = []


def test_speech_and_the_exchange_conversation_returns_for_it_are_remembered_as_one_episode() -> None:
    """R-20: speech admitted through Conversation is one experience, so it is one episode.

    The speech turn (kept under the speech's id, the way Listening stamps it) is handled by the seeded
    SpeechHeard behavior, which forwards the speech to Communication as its causal parent. The exchange
    Conversation returns carries that forward, and through it the speech, as its causal parents; the
    exchange turn extends the speech turn's episode instead of becoming a second one: what was heard, the
    behavior's act, then the reply.
    """

    from mosfet.abilities import listening
    from mosfet.abilities.communication import behaviors

    async def run() -> tuple[tuple[memory.Episode, ...], list[hsm.Event[typing.Any]]]:
        store = memory.Memory()
        ability = RecordingCognition(
            autonomy=cognition.Autonomy(seeded_behaviors=(behaviors.speech_heard_seed(),)),
            decision_processor=RecordingDecisionProcessor(focus_output("phone", "answer Ada")),
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=store),
            store=store,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        talk = _Communication()
        _ = await mosfet.started(ctx, talk, talk.model)
        base = await started_cognition_input(ctx)
        base = base.model_copy(update={"actors": {**base.actors, "communication": talk}, "participants": (_ADA,)})
        speech = dataclasses.replace(_speech_event_stimulus(), id="heard")
        _ = await dispatch_ability_for_test(
            ability, ctx, base.model_copy(update={"stimulus": speech}), operation_id="heard"
        )
        await wait_until(lambda: bool(talk.admitted))
        await wait_until(lambda: ability.state().endswith("/idle"))
        (admitted,) = talk.admitted
        exchange = dataclasses.replace(
            communication.conversation.OutputEvent.with_data(
                communication.conversation.Exchange(parent=mosfet.StimulusData.from_event(admitted))
            ),
            id="heard:autonomy",
        )
        _ = await dispatch_ability_for_test(
            ability, ctx, base.model_copy(update={"stimulus": exchange}), operation_id="heard:autonomy"
        )
        await wait_until(lambda: ability.state().endswith("/idle"))
        return await _remembered(ctx, store), talk.admitted

    episodes, admitted = asyncio.run(run())

    assert [event.name for event in admitted] == [communication.InputEvent.name]
    (kept,) = episodes
    assert kept.turn_id == "heard"
    assert kept.continued == ("heard:autonomy",)
    assert kept.stimulus == listening.SpeechEvent.name
    assert [act.event for act in kept.acts] == [communication.InputEvent.name, mosfet.FocusDeviceEvent.name]
    assert kept.acts[1].reason == "answer Ada"
    heard = kept.text.index(listening.SpeechEvent.name)
    replied = kept.text.index("answer Ada")
    assert heard < replied
    assert "voice:3f2a9c" in kept.text[:replied]


def test_a_partly_handled_batch_carries_its_behaviors_acts_first_into_the_turn_and_its_episode() -> None:
    """D-W12-9: what behaviors already did on their percepts is the turn's, ahead of deliberation's.

    Speech and a typed line queue together. SpeechHeard admits the speech; the line is decision's.
    Decision knows, as a fact of the turn, that a behavior already acted on the speech, and the
    turn's output and its one episode carry that act first, then decision's.
    """

    from mosfet.abilities.communication import behaviors

    async def run() -> tuple[
        list[cognition.types.OutputData],
        list[processing.InputData],
        tuple[memory.Episode, ...],
        list[hsm.Event[typing.Any]],
    ]:
        store = memory.Memory()
        decision_processor = RecordingDecisionProcessor(focus_output("phone", "answer the line"))
        ability = RecordingCognition(
            autonomy=cognition.Autonomy(seeded_behaviors=(behaviors.speech_heard_seed(),)),
            decision_processor=decision_processor,
            reflection=cognition.Reflection(processor=FixedProcessor(no_output("observed")), memory=store),
            store=store,
        )
        ctx = await start_cognition_ability_for_test(ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        talk = _Communication()
        _ = await mosfet.started(ctx, talk, talk.model)
        base = await started_cognition_input(ctx)
        turn = _batch_turn(
            base.model_copy(update={"actors": {**base.actors, "communication": talk}}),
            mosfet.PerceptData(stimulus=_speech_event_stimulus(), source="listening-1"),
            _typed_percept("are you there?", person=_ADA),
        )
        _ = await dispatch_ability_for_test(ability, ctx, turn)
        await wait_until(lambda: bool(ability.outputs))
        await wait_until(lambda: ability.state().endswith("/idle"))
        return ability.outputs, decision_processor.calls, await _remembered(ctx, store), talk.admitted

    outputs, decision_calls, episodes, admitted = asyncio.run(run())

    assert [event.name for event in admitted] == [communication.InputEvent.name]
    (output,) = outputs
    assert [selection.event for selection in output] == [communication.InputEvent.name, mosfet.FocusDeviceEvent.name]
    assert output[1].reason == "answer the line"
    (call,) = decision_calls
    assert call.situation is not None
    # The fact follows the world block in the turn's situation: which percepts behaviors handled, and what they did.
    acted = call.situation[call.situation.index("<cognition:acted>") :]
    assert "<cognition:acted>" not in (call.instructions or "")
    assert acted.endswith("</cognition:acted>")
    assert '<percepts count="1">\n    <item>0</item>\n  </percepts>' in acted
    assert f'event="{communication.InputEvent.name}"' in acted
    assert "answer the line" not in acted
    (kept,) = episodes
    assert kept.stimulus == "bot.perceived"
    assert [act.event for act in kept.acts] == [communication.InputEvent.name, mosfet.FocusDeviceEvent.name]
