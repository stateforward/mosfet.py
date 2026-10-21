"""Learning ability: decode lesson, synthesize runtime input, revise behavior source."""

from __future__ import annotations

import ast
import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import sqlite3
import typing

import hsm
import pytest

import mosfet
from mosfet import abilities, behavior
from mosfet.abilities import decoding
from mosfet.abilities import learning
from mosfet.abilities import memory
from mosfet.abilities import processing
from mosfet.abilities.cognition import reflection
from mosfet.abilities.cognition.reflection import revision
from mosfet.abilities.learning import Learning
from mosfet.abilities.learning import learning as learning_impl
from mosfet.protocols import attachment
from tests.bot.abilities.memory.memory_fixtures import register_recent, register_record
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test

# One Learning revision performs one isolated Starlark validation (a cold validation worker can
# take several seconds on a loaded host, and it blocks the loop while it runs); the host wait also
# covers decode, recall and register reads by message, selection, terminal propagation, and teardown.
_LEARNING_REVISION_HOST_TIMEOUT_SECONDS = 30.0


ANSWER_RING_BEHAVIOR_SOURCE = """
input_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.input",
    schema = {"type": "object", "additionalProperties": True},
    description = "Live behavior input.",
)
output_event = hsm.event(
    name = "bot.behavior.answer_incoming_ring.output",
    schema = {
        "type": "object",
        "properties": {"event": {"type": "string"}},
        "required": ["event"],
        "additionalProperties": True,
    },
    description = "Cognition event selection.",
)
triggers = ["environment.sound"]
description = "Answer an incoming ring."

def select_focus(event):
    hsm.dispatch(output_event, {"event": "bot.focus_device", "data": {"device": "phone"}})

behavior = hsm.define(
    "AnswerIncomingRing",
    hsm.initial(hsm.target("/AnswerIncomingRing/idle")),
    hsm.state(
        "idle",
        hsm.transition(hsm.on(input_event), hsm.effect("select_focus")),
    ),
)
""".strip()


class TextDecoder(decoding.Decoder[learning.InputData, learning.DecodedData]):
    @typing.override
    async def decode(self, input: learning.InputData) -> learning.DecodedData:
        if isinstance(input.content, bytes):
            text = input.content.decode("utf-8")
        else:
            text = input.content
        return learning.DecodedData(text=text, kind="instruction")


class LearningTestProcessor(processing.Processor):
    """Routes generate-select vs revision write by instructions stamp."""

    selection: processing.Events
    write: behavior.ChangeData

    def __init__(self, *, selection: processing.Events, write: behavior.ChangeData) -> None:
        self.selection = selection
        self.write = write

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        instructions = input.instructions or ""
        if instructions == reflection.CHANGE_INSTRUCTIONS or "changing phase" in instructions:
            return (
                processing.SelectedEvent(
                    event=behavior.ChangeEvent.name,
                    data=self.write.model_dump(mode="python"),
                ),
            )
        return self.selection


async def _attached(*peers: memory.MemoryStore) -> None:
    """Stand in for the owners that attach the bot's memory and register; Learning is their peer."""

    ctx = shared_hsm_context()
    for peer in peers:
        if not peer.state():
            await start_abilities_for_test(ctx, peer)


def _generate_selection(
    *,
    name: str = "AnswerIncomingRing",
    lesson: str = "When the phone rings, answer it.",
) -> processing.Events:
    return (
        processing.SelectedEvent(
            event=learning.GenerateEvent.name,
            data={
                "event": learning.GenerateEvent.name,
                "inventory_event": "bot.behavior.create",
                "name": name,
                "reason": lesson,
                "lesson_kind": "instruction",
                "runtime_input": {
                    "stimulus_name": "environment.sound",
                    "payload": {"kind": "phone.ringing", "call_id": "incoming-call"},
                    "focus": "phone",
                    "focus_candidates": ["phone"],
                    "expected_event": "bot.focus_device",
                    "expected_data": {"device": "phone"},
                    "expected_reason": "answer ring",
                },
            },
            reason=lesson,
        ),
    )


def test_learning_callbacks_do_not_use_assert_for_control_flow() -> None:
    """`python -O` strips asserts; a stripped type guard becomes silent undefined behavior.

    Learning's HSM callbacks must narrow with real control flow (typed failure or a
    guard-paired early return), never with `assert`.
    """

    source = pathlib.Path(typing.cast(str, learning_impl.__file__)).read_text(encoding="utf-8")
    asserts = [node.lineno for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assert)]
    assert asserts == [], f"assert used for control flow at learning.py lines {asserts}"


def test_learning_revision_progresses_from_typed_child_terminals() -> None:
    source = pathlib.Path(typing.cast(str, learning_impl.__file__)).read_text(encoding="utf-8")

    assert "Ability.await_child_terminal" not in source


def test_learning_event_names_and_types() -> None:
    assert Learning.input_event.name == "bot.ability.learning.input"
    assert Learning.output_event.name == "bot.ability.learning.output"
    assert Learning.input_data_type is learning.InputData
    assert Learning.output_data_type is learning.OutputData
    assert learning.GenerateEvent.name == "bot.ability.learning.generate"


def test_learning_is_reasonings_alone_above_intuitions_ceiling() -> None:
    """D-R-52: authoring behaviors takes deliberation; intuition's ceiling is M, so learning is rated L."""

    assert Learning.effort is abilities.ability.Effort.L


def test_select_input_grounds_in_remembered_episodes() -> None:
    """prior_turns are memory.Episode: the model sees stimulus, text, images, participants, and acts."""

    episode = memory.Episode(
        node_id="5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f",
        stimulus="environment.sound",
        text="phone ringing",
        acts=(memory.Act(event="phone.answer_call", target="phone"),),
        outcome=memory.Outcome.COMPLETED,
        turn_id="turn-1",
        created_at=datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.UTC),
    )
    select = learning.SelectInput(
        turn=learning.InputData(content="When the phone rings, answer it."),
        decoded=learning.DecodedData(text="When the phone rings, answer it."),
        prior_turns=(episode,),
        operation_id="op",
        generation="gen",
    )

    assert select.prior_turns == (episode,)
    episode_schema = learning.SelectInput.model_json_schema()["$defs"]["Episode"]["properties"]
    assert {"stimulus", "acts", "text", "images", "participants", "outcome", "created_at"} <= set(episode_schema)
    assert "stimulus_name" not in episode_schema
    assert learning.SelectInput.__model_facing_budgeted_fields__ == {"prior_turns": "tail"}
    assert not hasattr(learning, "RememberedTurn")
    assert not hasattr(learning, "RememberedSelection")


def test_learning_prefers_memory_episodes_for_runtime_input() -> None:
    """When cognitive episodes exist, Learning grounds runtime_input in them over free invention."""

    async def run() -> learning.OutputData:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        # Model invents a wrong expected event; memory prefer-step should restore episode output.
        selection = (
            processing.SelectedEvent(
                event=learning.GenerateEvent.name,
                data={
                    "event": learning.GenerateEvent.name,
                    "inventory_event": "bot.behavior.create",
                    "name": "AnswerIncomingRing",
                    "reason": "When the phone rings, answer it.",
                    "lesson_kind": "instruction",
                    "runtime_input": {
                        "stimulus_name": "environment.sound",
                        "payload": {"kind": "phone.ringing"},
                        "focus": "phone",
                        "focus_candidates": ["phone"],
                        "expected_event": "bot.focus_device",
                        "expected_data": {"device": "phone"},
                    },
                },
                reason="When the phone rings, answer it.",
            ),
        )
        processor = LearningTestProcessor(
            selection=selection,
            write=behavior.ChangeData(
                name="AnswerIncomingRing",
                triggers=("environment.sound",),
                reason="When the phone rings, answer it.",
                source=ANSWER_RING_BEHAVIOR_SOURCE,
            ),
        )
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
        )
        await _attached(store)
        return await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it.", media_type="text/plain"),
            timeout=_LEARNING_REVISION_HOST_TIMEOUT_SECONDS,
        )

    output = asyncio.run(run())
    assert output.runtime_input.stimulus_name == "environment.sound"
    assert output.runtime_input.focus == "phone"
    # Model omitted expected answer_call; memory episode supplies it.
    assert output.runtime_input.expected_event == "phone.answer_call"
    assert output.runtime_input.expected_target == "phone"
    assert output.runtime_input.expected_data == {"call_id": "from-memory"}
    # Memory grounds identity and the expected selection; it never synthesizes the live payload,
    # so the model's live-shaped payload is preserved verbatim.
    assert output.runtime_input.payload == {"kind": "phone.ringing"}
    assert output.behavior.name == "AnswerIncomingRing"


async def _seed_register_knock(register: memory.StmMemory) -> None:
    await _attached(register)
    event = hsm.Event[object](name="environment.sound", schema=object).with_data(
        {"kind": "phone.ringing", "call_id": "incoming-call"}
    )
    await register_record(
        shared_hsm_context(), register, dataclasses.replace(event, id="turn-1", source="phone", target="bot")
    )


def _selection_with_invented_payload() -> processing.Events:
    """Generate selection whose payload is invented (not observed) and whose expectation is wrong.

    Grounding must replace the invented payload from real evidence and discard the invented
    expectation when no remembered turn vouches for it.
    """

    return (
        processing.SelectedEvent(
            event=learning.GenerateEvent.name,
            data={
                "event": learning.GenerateEvent.name,
                "inventory_event": "bot.behavior.create",
                "name": "AnswerIncomingRing",
                "reason": "When the phone rings, answer it.",
                "lesson_kind": "instruction",
                "runtime_input": {
                    "stimulus_name": "environment.sound",
                    "payload": {"kind": "invented"},
                    "focus": "phone",
                    "focus_candidates": ["phone"],
                    "expected_event": "bot.focus_device",
                    "expected_data": {"device": "phone"},
                },
            },
            reason="When the phone rings, answer it.",
        ),
    )


def _ring_write() -> behavior.ChangeData:
    return behavior.ChangeData(
        name="AnswerIncomingRing",
        triggers=("environment.sound",),
        reason="When the phone rings, answer it.",
        source=ANSWER_RING_BEHAVIOR_SOURCE,
    )


async def _grounded_register_entry(register: memory.StmMemory) -> memory.ObservedEvent:
    matches = [
        entry
        for entry in await register_recent(shared_hsm_context(), register, "environment.sound", limit=10)
        if entry.payload.get("kind") == "phone.ringing"
    ]
    assert matches, "register lost the grounded entry"
    return matches[0]


def test_learning_grounds_payload_in_event_register_when_memory_empty() -> None:
    """No remembered turns, but the register observed the stimulus: payload comes from perception.

    Register-only grounding vouches for the live payload but never for a selection, so the
    invented expected_* fields are dropped rather than authored into standing behavior.
    """

    async def run() -> tuple[learning.OutputData, memory.ObservedEvent]:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        register = memory.StmMemory()
        await _seed_register_knock(register)
        processor = LearningTestProcessor(
            selection=_selection_with_invented_payload(),
            write=_ring_write(),
        )
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
            stm_memory=register,
        )
        await _attached(store, register)
        output = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it."),
            timeout=_LEARNING_REVISION_HOST_TIMEOUT_SECONDS,
        )
        return output, await _grounded_register_entry(register)

    output, grounded = asyncio.run(run())
    assert output.runtime_input.stimulus_name == "environment.sound"
    # The invented payload is replaced by what the register actually observed on that stimulus.
    assert output.runtime_input.payload == {"kind": "phone.ringing", "call_id": "incoming-call"}
    # Focus survives from the lesson turn; selections do not (only remembered output vouches).
    assert output.runtime_input.focus == "phone"
    assert output.runtime_input.expected_event is None
    assert output.runtime_input.expected_data is None
    # Reconsolidation: the entry that carried the grounding carries an access stamp now.
    assert grounded.last_accessed_at is not None
    assert output.behavior.name == "AnswerIncomingRing"


def test_learning_register_payload_backs_memory_grounded_stimulus() -> None:
    """Memory vouches for stimulus identity and selection; the register vouches for the payload."""

    async def run() -> tuple[learning.OutputData, memory.ObservedEvent]:
        # Memory and the register are peers reached by message: the store refuses direct calls.
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        register = memory.StmMemory()
        await _seed_register_knock(register)
        processor = LearningTestProcessor(
            selection=_selection_with_invented_payload(),
            write=_ring_write(),
        )
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
            stm_memory=register,
        )
        await _attached(store, register)
        output = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it."),
            timeout=_LEARNING_REVISION_HOST_TIMEOUT_SECONDS,
        )
        return output, await _grounded_register_entry(register)

    output, grounded = asyncio.run(run())
    assert output.runtime_input.payload == {"kind": "phone.ringing", "call_id": "incoming-call"}
    # Expected selection still comes from the remembered episode, not from perception.
    assert output.runtime_input.expected_event == "phone.answer_call"
    assert output.runtime_input.expected_target == "phone"
    assert output.runtime_input.expected_data == {"call_id": "from-memory"}
    # The register entry that grounded the payload is reinforced even though memory also spoke.
    assert grounded.last_accessed_at is not None
    assert output.behavior.name == "AnswerIncomingRing"


def test_learning_grounds_in_the_latest_remembered_turn_beyond_the_recall_window() -> None:
    """A long-lived bot grounds a lesson in what it did lately, not in its oldest turns.

    Memory reads episodes back in remember order, so the last of these is the latest.
    """

    async def run() -> learning.OutputData:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(
            store, *(_ring_episode(call_id=f"call-{index:02d}", turn_id=f"turn-{index:02d}") for index in range(60))
        )
        processor = LearningTestProcessor(
            selection=_generate_selection(),
            write=_ring_write(),
        )
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
        )
        await _attached(store)
        return await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it."),
            timeout=_LEARNING_REVISION_HOST_TIMEOUT_SECONDS,
        )

    output = asyncio.run(run())
    assert output.runtime_input.expected_data == {"call_id": "call-59"}


def _ring_episode(*, call_id: str = "from-memory", turn_id: str = "turn-ring") -> memory.EpisodeData:
    return memory.EpisodeData(
        turn_id=turn_id,
        stimulus="environment.sound",
        focus="phone",
        focus_candidates=("phone",),
        acts=(
            memory.Act(
                event="phone.answer_call",
                target="phone",
                data={"call_id": call_id},
                reason="answered from prior turn",
            ),
        ),
        outcome=memory.Outcome.COMPLETED,
    )


async def _remember(store: memory.Memory, *episodes: memory.EpisodeData) -> None:
    """Seed remembered turns the way Cognition does: by message to the attached memory."""

    await _attached(store)
    for index, episode in enumerate(episodes):
        reply = await abilities.run_terminal_operation(
            shared_hsm_context(),
            child=store,
            request=memory.RememberEvent.with_data_and_id(episode, f"seed-{index}"),
            terminals=(memory.RememberedEvent, memory.FailedEvent),
            timeout=datetime.timedelta(seconds=5),
        )
        assert isinstance(reply.data, memory.RememberedData), reply.data


def test_learning_fails_honestly_without_memory_stimulus() -> None:
    """No groundable episodes → refuse to invent runtime_input."""

    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        processor = LearningTestProcessor(
            selection=_generate_selection(),
            write=behavior.ChangeData(name="Unused", source=ANSWER_RING_BEHAVIOR_SOURCE),
        )
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
        )
        await _attached(store)
        _ = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it."),
            timeout=5.0,
        )

    try:
        asyncio.run(run())
        raise AssertionError("expected failure")
    except RuntimeError as error:
        message = str(error).lower()
        assert "cannot determine the input stimulus" in message or "no remembered turns" in message


def test_learning_decodes_grounds_memory_and_generates_behavior() -> None:
    async def run() -> learning.OutputData:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        processor = LearningTestProcessor(
            selection=_generate_selection(),
            write=behavior.ChangeData(
                name="AnswerIncomingRing",
                triggers=("environment.sound",),
                reason="When the phone rings, answer it.",
                source=ANSWER_RING_BEHAVIOR_SOURCE,
            ),
        )
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
        )
        await _attached(store)
        return await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it.", media_type="text/plain"),
            timeout=_LEARNING_REVISION_HOST_TIMEOUT_SECONDS,
        )

    output = asyncio.run(run())
    assert isinstance(output, learning.OutputData)
    assert output.decoded.text == "When the phone rings, answer it."
    assert output.decoded.kind == "instruction"
    assert output.runtime_input.stimulus_name == "environment.sound"
    assert output.runtime_input.focus == "phone"
    assert output.runtime_input.expected_event == "phone.answer_call"
    assert output.behavior.name == "AnswerIncomingRing"
    assert output.behavior.triggers == ("environment.sound",)
    assert output.behavior.source is not None
    assert "AnswerIncomingRing" in output.behavior.source


def test_learning_empty_generate_selection_fails() -> None:
    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        processor = LearningTestProcessor(
            selection=(),
            write=behavior.ChangeData(name="Unused", source=ANSWER_RING_BEHAVIOR_SOURCE),
        )
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
        )
        await _attached(store)
        _ = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="noise"),
            timeout=5.0,
        )

    try:
        asyncio.run(run())
        raise AssertionError("expected failure")
    except RuntimeError as error:
        assert "empty selection" in str(error)


def test_learning_rejects_lesson_text_as_runtime_input_payload() -> None:
    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        lesson = "When the phone rings, answer it immediately without delay."
        selection = (
            processing.SelectedEvent(
                event=learning.GenerateEvent.name,
                data={
                    "event": learning.GenerateEvent.name,
                    "inventory_event": "bot.behavior.create",
                    "name": "BadRuntimeInput",
                    "reason": lesson,
                    "runtime_input": {
                        "stimulus_name": "environment.sound",
                        # Lesson smuggled as runtime payload — must fail.
                        "payload": {"text": lesson},
                    },
                },
            ),
        )
        processor = LearningTestProcessor(
            selection=selection,
            write=behavior.ChangeData(name="BadRuntimeInput", source=ANSWER_RING_BEHAVIOR_SOURCE),
        )
        ability = Learning(
            decoder=TextDecoder(),
            processor=processor,
            memory=store,
        )
        await _attached(store)
        _ = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content=lesson),
            timeout=5.0,
        )

    try:
        asyncio.run(run())
        raise AssertionError("expected failure")
    except RuntimeError as error:
        # After grounding, lesson-text payload is replaced by memory payload — if model
        # still only had lesson text, grounding uses memory payload and continues.
        # Force failure by empty groundable episodes instead is separate; here ensure
        # _coerce_generate still rejects pre-grounding when payload is pure lesson.
        assert "lesson text" in str(error).lower() or "runtime_input.payload" in str(error).lower()


class AttachmentOwner(hsm.Instance):
    """Records the owner-facing terminals Learning dispatches upward."""

    lifecycle: list[hsm.Event[typing.Any]]

    @staticmethod
    def _record(ctx: hsm.Context, instance: "AttachmentOwner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.lifecycle.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "LearningAttachmentOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.AttachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(Learning.output_event), hsm.effect(_record)),
            hsm.transition(hsm.on(Learning.failed_event), hsm.effect(_record)),
            hsm.transition(hsm.on(mosfet.RebootEvent), hsm.effect(_record)),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0)


def test_learning_reboot_reason_is_learning_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck teardown reboots with a learning-domain reason, not a cognition one."""

    class StubbornProcessing(processing.Processing):
        submodel = mosfet.define(
            "StubbornLearningProcessing",
            hsm.initial(hsm.target("/StubbornLearningProcessing/waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(processing.Processing.input_event),
                    hsm.effect(lambda ctx, instance, event: None),
                ),
            ),
        )

    class StubbornLearning(Learning):
        def replace_processing(self, stubborn: processing.Processing) -> None:
            self._select_processing = stubborn
            self._attachment_group = attachment.Group(stubborn, self._revision)

    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        monkeypatch.setattr(learning_impl, "_CANCEL_TEARDOWN_TIMEOUT", datetime.timedelta(0))
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        processor = LearningTestProcessor(
            selection=_generate_selection(),
            write=behavior.ChangeData(name="Unused", source=ANSWER_RING_BEHAVIOR_SOURCE),
        )
        ability = StubbornLearning(decoder=TextDecoder(), processor=processor, memory=store)
        await _attached(store)
        stubborn = StubbornProcessing(processor=processor)
        ability.replace_processing(stubborn)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await ability.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: ability.state().endswith("/idle"))
        owner.lifecycle.clear()
        operation_id = "stubborn-learning"
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                ability.input_event.with_data_and_id(learning.InputData(content="When it rings, answer."), operation_id)
            ),
        )
        await wait_until(lambda: ability.state().endswith("/selecting"))
        _ = await hsm.dispatch(
            ctx,
            ability,
            dataclasses.replace(
                processing.CancelEvent.with_data(
                    processing.CancelData(operation_id=operation_id, token="stubborn-learning-token")
                ),
                id=operation_id,
                source=hsm.id(owner),
                target=hsm.id(ability),
            ),
        )
        await wait_until(lambda: ability.state().endswith("/rebooting"))
        await wait_until(lambda: any(event.name == mosfet.RebootEvent.name for event in owner.lifecycle))
        result = list(owner.lifecycle), ability.state()
        await ability.stop(ability.context())
        connection.close()
        return result

    lifecycle, state = asyncio.run(run())

    reboots = [event for event in lifecycle if event.name == mosfet.RebootEvent.name]
    assert len(reboots) == 1
    assert reboots[0].data == mosfet.RebootEventData(reason="learning_child_teardown_failed")
    assert state.endswith("/rebooting")


def test_runtime_input_schema_rejects_lesson_prose_payload() -> None:
    """The 'payload is not lesson text' constraint is the schema's, not a caller-side heuristic."""

    lesson = "When the phone rings, answer it immediately without any delay whatsoever."
    with pytest.raises(ValueError, match="lesson"):
        _ = learning.RuntimeInputData(stimulus_name="environment.sound", payload={"note": lesson})
    # A live-shaped payload of the same shape but short values stays valid.
    accepted = learning.RuntimeInputData(stimulus_name="environment.sound", payload={"note": "ringing"})
    assert accepted.payload == {"note": "ringing"}


def test_runtime_input_schema_requires_non_empty_payload() -> None:
    with pytest.raises(ValueError):
        _ = learning.RuntimeInputData(stimulus_name="environment.sound", payload={})


def test_learning_does_not_promote_domain_keys_into_runtime_payload() -> None:
    """Memory grounds turn identity and the expected selection — never a synthesized payload.

    Promoting selection fields (call_id / device / kind) into the dry-run payload would bake
    phone-domain knowledge into a generic ability, and the empty case must not invent a sentinel.
    """

    async def run() -> learning.OutputData:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        selection = (
            processing.SelectedEvent(
                event=learning.GenerateEvent.name,
                data={
                    "event": learning.GenerateEvent.name,
                    "inventory_event": "bot.behavior.create",
                    "name": "AnswerIncomingRing",
                    "reason": "When the phone rings, answer it.",
                    "runtime_input": {
                        "stimulus_name": "environment.sound",
                        "payload": {"kind": "phone.ringing"},
                    },
                },
            ),
        )
        processor = LearningTestProcessor(
            selection=selection,
            write=behavior.ChangeData(
                name="AnswerIncomingRing",
                triggers=("environment.sound",),
                reason="When the phone rings, answer it.",
                source=ANSWER_RING_BEHAVIOR_SOURCE,
            ),
        )
        ability = Learning(decoder=TextDecoder(), processor=processor, memory=store)
        await _attached(store)
        return await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it."),
            timeout=_LEARNING_REVISION_HOST_TIMEOUT_SECONDS,
        )

    output = asyncio.run(run())
    # Memory is authoritative for identity and expected selection.
    assert output.runtime_input.stimulus_name == "environment.sound"
    assert output.runtime_input.expected_event == "phone.answer_call"
    assert output.runtime_input.expected_data == {"call_id": "from-memory"}
    # The model's live-shaped payload survives verbatim: no promoted keys, no sentinel.
    assert output.runtime_input.payload == {"kind": "phone.ringing"}


def test_learning_module_has_no_domain_key_heuristics() -> None:
    """Guard against reintroducing hardcoded key lists / sentinels in a generic ability.

    Only executable bodies are inspected: schema ``examples`` are documentation the framework
    requires, while the same literals inside logic would be product policy in the wrong layer.
    """

    tree = ast.parse(pathlib.Path(typing.cast(str, learning_impl.__file__)).read_text(encoding="utf-8"))
    literals = {
        node.value
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    banned = {"call_id", "device", "lesson", "memory_turn", "text", "content"} & literals
    assert banned == set(), f"hardcoded domain-key policy in a generic ability: {sorted(banned)}"


def test_learning_recall_failure_is_not_reported_as_missing_memory() -> None:
    """A recall error must surface as a recall failure, not as the fail-closed no-stimulus refusal."""

    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        # The database goes away under the attached memory: the recall request is answered with
        # memory.FailedEvent (store_failed), not with an empty list of episodes.
        connection.close()
        processor = LearningTestProcessor(
            selection=_generate_selection(),
            write=behavior.ChangeData(name="Unused", source=ANSWER_RING_BEHAVIOR_SOURCE),
        )
        ability = Learning(decoder=TextDecoder(), processor=processor, memory=store)
        await _attached(store)
        _ = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it."),
            timeout=5.0,
        )

    try:
        asyncio.run(run())
        raise AssertionError("expected failure")
    except RuntimeError as error:
        message = str(error)
        assert "store_failed" in message, message
        assert "recall failed" in message.lower(), message
        # The operator must not be pointed at "store or recall real turns first".
        assert "no remembered turns" not in message.lower(), message


def test_learning_no_stimulus_message_distinguishes_empty_from_ungroundable() -> None:
    """Turns exist but none name a stimulus — say that, not 'no remembered turns'."""

    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(
            store,
            memory.EpisodeData(
                turn_id="turn-unnamed", focus="phone", focus_candidates=("phone",), outcome=memory.Outcome.COMPLETED
            ),
        )
        processor = LearningTestProcessor(
            selection=_generate_selection(),
            write=behavior.ChangeData(name="Unused", source=ANSWER_RING_BEHAVIOR_SOURCE),
        )
        ability = Learning(decoder=TextDecoder(), processor=processor, memory=store)
        await _attached(store)
        _ = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it."),
            timeout=5.0,
        )

    try:
        asyncio.run(run())
        raise AssertionError("expected failure")
    except RuntimeError as error:
        message = str(error)
        assert "cannot determine the input stimulus" in message, message
        assert "1 remembered turn" in message, message


class _FailingDecoder(decoding.Decoder[learning.InputData, learning.DecodedData]):
    @typing.override
    async def decode(self, input: learning.InputData) -> learning.DecodedData:
        del input
        raise ValueError("decode blew up")


def test_learning_decode_failure_surfaces() -> None:
    async def run() -> None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        processor = LearningTestProcessor(
            selection=_generate_selection(),
            write=behavior.ChangeData(name="Unused", source=ANSWER_RING_BEHAVIOR_SOURCE),
        )
        ability = Learning(
            decoder=_FailingDecoder(),
            processor=processor,
            memory=store,
        )
        await _attached(store)
        _ = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="noise"),
            timeout=5.0,
        )

    try:
        asyncio.run(run())
        raise AssertionError("expected failure")
    except RuntimeError as error:
        assert "decode failed" in str(error).lower()


class _CapturingLearningProcessor(LearningTestProcessor):
    """Records the authoring input Revision hands the model."""

    captured: list[object]

    def __init__(self, *, selection: processing.Events, write: behavior.ChangeData) -> None:
        super().__init__(selection=selection, write=write)
        self.captured = []

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        self.captured.append(input.input)
        return await super().process(input)


def test_learning_hands_revision_the_lesson_as_typed_data_not_a_synthetic_selection() -> None:
    """The decoded lesson must reach Revision as typed data, not a fake selection.

    Learning used to fabricate a ``bot.ability.learning.lesson`` entry inside
    ``cognition_output`` -- a typed list of real cognition selections -- carrying the
    lesson prose in ``reason``, then recover it by string-matching that name. That is
    scratch smuggled through a typed field and correlated by literal. The lesson is
    authoring evidence and must travel as its own typed field.
    """

    async def run() -> _CapturingLearningProcessor:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        await _remember(store, _ring_episode())
        processor = _CapturingLearningProcessor(
            selection=_generate_selection(),
            write=behavior.ChangeData(
                name="AnswerIncomingRing",
                triggers=("environment.sound",),
                reason="When the phone rings, answer it.",
                source=ANSWER_RING_BEHAVIOR_SOURCE,
            ),
        )
        ability = Learning(decoder=TextDecoder(), processor=processor, memory=store)
        await _attached(store)
        _ = await dispatch_ability_for_test(
            ability,
            None,
            learning.InputData(content="When the phone rings, answer it.", media_type="text/plain"),
            timeout=_LEARNING_REVISION_HOST_TIMEOUT_SECONDS,
        )
        return processor

    processor = asyncio.run(run())
    authoring = [item for item in processor.captured if isinstance(item, revision.ChangeWriteInput)]
    assert authoring, "expected Revision to request an authoring turn"
    write = authoring[0]

    selection_names = [item.event for item in write.cognition_output]
    assert not any(name.startswith("bot.ability.learning.lesson") for name in selection_names), (
        f"lesson must not be smuggled as a synthetic selection; got {selection_names!r}"
    )
    assert all(name.startswith("phone.") or name.startswith("bot.focus") for name in selection_names), (
        f"cognition_output must carry only real turn selections; got {selection_names!r}"
    )

    assert write.instruction is not None, "Revision must receive the decoded lesson as typed data"
    assert write.instruction.text == "When the phone rings, answer it."
    assert write.instruction.kind == "instruction"


def test_learning_input_is_offered_as_a_model_callable_tool() -> None:
    """An attached Learning offers its input event, so cognition can select being taught.

    Without this the bot can only be taught by a host dispatching Learning directly; the
    model never sees learning as an option on a live turn.
    """

    async def run() -> tuple[tuple[str, ...], str | None]:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        store = memory.Memory(engine=engine)
        ability = Learning(
            decoder=TextDecoder(),
            processor=LearningTestProcessor(
                selection=_generate_selection(),
                write=behavior.ChangeData(name="Unused", source=ANSWER_RING_BEHAVIOR_SOURCE),
            ),
            memory=store,
        )
        await _attached(store)
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, ability)
        await wait_until(lambda: ability.state().endswith("/idle"))
        offered = tuple(event.name for event in processing.enabled_call_events(ability))
        state = ability.state()
        connection.close()
        return offered, state

    offered, state = asyncio.run(run())
    assert Learning.input_event.kind == processing.EventKind, "Learning input must be model-offerable"
    assert "bot.ability.learning.input" in offered, (
        f"attached Learning in {state!r} did not offer its input event; offered={offered!r}"
    )
