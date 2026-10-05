import mosfet
from mosfet import behavior
from mosfet.abilities import cognition
from mosfet.abilities import memory
from mosfet.abilities import processing
from mosfet.abilities.cognition import reflection as reflection_module
from mosfet.abilities.cognition.reflection import reflection as reflection_impl
from mosfet.abilities.cognition.reflection import revision
from mosfet.protocols import attachment

import asyncio
import collections.abc
import dataclasses
import datetime
import pathlib
import sqlite3
import typing
import uuid

import hsm
import pytest
from tests.type_helpers import model_view
from tests.bot.abilities.cognition.metadata_contract import assert_metadata_is_not_coordination


def test_reflection_never_uses_metadata_for_coordination() -> None:
    for module in (reflection_impl, revision):
        assert_metadata_is_not_coordination(module)


def test_reflection_revision_progresses_from_typed_child_terminals() -> None:
    source = pathlib.Path(typing.cast(str, reflection_impl.__file__)).read_text(encoding="utf-8")

    assert "Ability.await_child_terminal" not in source


class EmptyProcessor(processing.Processor):
    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return ()


class HangingProcessor(processing.Processor):
    calls: int
    cancelled: bool
    first_output: processing.Events | None

    def __init__(self, first_output: processing.Events | None = None) -> None:
        self.calls = 0
        self.cancelled = False
        self.first_output = first_output

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        self.calls += 1
        if self.calls == 1 and self.first_output is not None:
            return self.first_output
        try:
            _ = await asyncio.Event().wait()
        finally:
            self.cancelled = True
        raise AssertionError("unreachable")


class AttachmentOwner(hsm.Instance):
    lifecycle: list[hsm.Event[typing.Any]]

    @staticmethod
    def _record(
        ctx: hsm.Context,
        instance: "AttachmentOwner",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        instance.lifecycle.append(event)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "ReflectionAttachmentOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.AttachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(processing.CancelledEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(cognition.Reflection.output_event), hsm.effect(_record)),
            hsm.transition(hsm.on(cognition.Reflection.failed_event), hsm.effect(_record)),
            hsm.transition(hsm.on(mosfet.RebootEvent), hsm.effect(_record)),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1.0
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0)


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


def reflection_ability() -> tuple[cognition.Reflection, sqlite3.Connection]:
    return reflection_with_processor(EmptyProcessor())


def reflection_with_processor(
    processor: processing.Processor,
) -> tuple[cognition.Reflection, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return cognition.Reflection(processor=processor, memory=memory.Memory(connection=connection)), connection


def reflection_input() -> cognition.reflection.InputData:
    return cognition.reflection.InputData(
        cognition_input=cognition.InputData(
            stimulus=mosfet.InputEventData(target_device="phone", priority=0),
            abilities=(),
            actors={},
            focus=None,
            focus_candidates=("phone",),
        ),
        cognition_output=(),
    )


def test_reflection_ignores_forged_selected_event_without_turn_capability() -> None:
    async def run() -> tuple[str, object]:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        store = memory.Memory(connection=connection)
        reflection = cognition.Reflection(processor=HangingProcessor(), memory=store)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        input = reflection_input()
        _ = await hsm.dispatch(ctx, reflection, reflection.input_event.with_data_and_id(input, "forged-turn"))
        await wait_until(lambda: reflection.state().endswith("/processing"))
        turn = input
        forged = dataclasses.replace(
            getattr(reflection_impl, "_SelectedEvent").with_data(
                getattr(reflection_impl, "_SelectedEventData")(
                    turn=turn,
                    selection=cognition.types.EventData(
                        event=behavior.CreateEvent.name,
                        data=behavior.CreateData(name="ForgedBehavior", triggers=(mosfet.InputEvent.name,)).model_dump(
                            mode="json"
                        ),
                    ),
                    operation_id="forged-turn",
                    generation="forged-operation-token",
                )
            ),
            id="forged-turn",
            source=hsm.id(reflection),
            target=hsm.id(reflection),
        )
        _ = await hsm.dispatch(ctx, reflection, forged)
        await asyncio.sleep(0)
        stored = getattr(reflection_impl, "_load_behavior")(store, name="ForgedBehavior")
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return state, stored

    state, stored = asyncio.run(run())

    assert state.endswith("/processing")
    assert stored is None


def test_reflection_waits_for_direct_child_cancel_before_acknowledging() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, bool]:
        processor = HangingProcessor()
        reflection, connection = reflection_with_processor(processor)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            reflection,
            dataclasses.replace(
                reflection.input_event.with_data_and_id(reflection_input(), "cancel-reflection"),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/processing"))
        cancel = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id="cancel-reflection", token="reflection-token")
            ),
            id="cancel-reflection",
            source=hsm.id(owner),
            target=hsm.id(reflection),
        )
        _ = await hsm.dispatch(ctx, reflection, cancel)
        await wait_until(lambda: bool(owner.lifecycle))
        lifecycle, state, cancelled = owner.lifecycle, reflection.state(), processor.cancelled
        await reflection.stop(reflection.context())
        connection.close()
        return lifecycle, state, cancelled

    lifecycle, state, cancelled = asyncio.run(run())

    assert cancelled
    assert len(lifecycle) == 1
    assert lifecycle[0].data == processing.CancelledData(
        operation_id="cancel-reflection",
        token="reflection-token",
    )
    assert state.endswith("/idle")


def test_reflection_change_cancellation_handles_delimiter_in_parent_operation_id() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int]:
        processor = HangingProcessor(
            first_output=(
                processing.SelectedEvent(
                    event=behavior.CreateEvent.name,
                    data=behavior.CreateData(
                        name="CancellationBehavior",
                        triggers=(mosfet.InputEvent.name,),
                    ).model_dump(mode="json"),
                ),
            )
        )
        reflection, connection = reflection_with_processor(processor)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            reflection,
            dataclasses.replace(
                reflection.input_event.with_data_and_id(reflection_input(), "cancel:change:operation"),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/revising") and processor.calls == 2)
        cancel = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id="cancel:change:operation", token="change-token")
            ),
            id="cancel:change:operation",
            source=hsm.id(owner),
            target=hsm.id(reflection),
        )
        _ = await hsm.dispatch(ctx, reflection, cancel)
        await wait_until(lambda: bool(owner.lifecycle))
        lifecycle = list(owner.lifecycle)
        calls = processor.calls
        await reflection.stop(reflection.context())
        connection.close()
        return lifecycle, calls

    lifecycle, calls = asyncio.run(run())

    assert len(lifecycle) == 1
    assert lifecycle[0].data == processing.CancelledData(operation_id="cancel:change:operation", token="change-token")
    assert calls == 2


def test_reflection_change_cancellation_prevents_late_completion() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int, str]:
        processor = HangingProcessor(
            first_output=(
                processing.SelectedEvent(
                    event=behavior.CreateEvent.name,
                    data=behavior.CreateData(
                        name="StartingCancellationBehavior",
                        triggers=(mosfet.InputEvent.name,),
                    ).model_dump(mode="json"),
                ),
            )
        )
        reflection, connection = reflection_with_processor(processor)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            reflection,
            reflection.input_event.with_data_and_id(reflection_input(), "cancel-starting-change"),
        )
        await wait_until(lambda: reflection.state().endswith("/revising") and processor.calls == 2)
        cancel = dataclasses.replace(
            processing.CancelEvent.with_data(
                processing.CancelData(operation_id="cancel-starting-change", token="starting-token")
            ),
            id="cancel-starting-change",
            source=hsm.id(owner),
            target=hsm.id(reflection),
        )
        _ = await hsm.dispatch(ctx, reflection, cancel)
        await wait_until(lambda: bool(owner.lifecycle))
        result = list(owner.lifecycle), processor.calls, reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return result

    lifecycle, calls, state = asyncio.run(run())

    assert len(lifecycle) == 1
    assert lifecycle[0].data == processing.CancelledData(
        operation_id="cancel-starting-change",
        token="starting-token",
    )
    assert calls == 2
    assert state.endswith("/idle")


def test_reflection_stubborn_child_cancel_timeout_requests_reboot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubbornProcessing(processing.Processing):
        submodel = mosfet.define(
            "StubbornReflectionProcessing",
            hsm.initial(hsm.target("/StubbornReflectionProcessing/waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(processing.Processing.input_event),
                    hsm.effect(lambda ctx, instance, event: None),
                ),
            ),
        )

    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        monkeypatch.setattr(
            reflection_impl,
            "_CANCEL_TEARDOWN_TIMEOUT",
            datetime.timedelta(0),
        )
        reflection, connection = reflection_ability()
        stubborn = StubbornProcessing(processor=EmptyProcessor())
        setattr(reflection, "_select_processing", stubborn)
        setattr(
            reflection,
            "_attachment_group",
            attachment.Group(
                stubborn,
                typing.cast(revision.Revision, getattr(reflection, "_revision")),
                typing.cast(memory.Memory, getattr(reflection, "_memory")),
            ),
        )
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        operation_id = "stubborn-reflection"
        token = "stubborn-reflection-token"
        _ = await hsm.dispatch(
            ctx,
            reflection,
            dataclasses.replace(
                reflection.input_event.with_data_and_id(reflection_input(), operation_id),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/processing"))
        _ = await hsm.dispatch(
            ctx,
            reflection,
            dataclasses.replace(
                processing.CancelEvent.with_data(processing.CancelData(operation_id=operation_id, token=token)),
                id=operation_id,
                source=hsm.id(owner),
                target=hsm.id(reflection),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/rebooting"))
        await wait_until(lambda: any(event.name == mosfet.RebootEvent.name for event in owner.lifecycle))
        result = list(owner.lifecycle), reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return result

    lifecycle, state = asyncio.run(run())

    failures = [event for event in lifecycle if event.name == cognition.Reflection.failed_event.name]
    reboots = [event for event in lifecycle if event.name == mosfet.RebootEvent.name]
    assert len(failures) == 1
    assert len(reboots) == 1
    assert reboots[0].data == mosfet.RebootEventData(reason="cognition_child_teardown_failed")
    assert state.endswith("/rebooting")


def test_reflection_cancel_guard_rejects_wrong_operation_and_source() -> None:
    async def run() -> tuple[bool, bool]:
        reflection, connection = reflection_ability()
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        request_id = getattr(reflection_module.Reflection, "_child_id")(
            reflection, getattr(reflection_impl, "_SELECT_ID_SUFFIX")
        )
        wrong_operation = dataclasses.replace(
            processing.CancelledEvent.with_data(
                processing.CancelledData(operation_id="wrong-request", token="guard-token")
            ),
            id="wrong-request",
            source=hsm.id(getattr(reflection, "_select_processing")),
            target=hsm.id(reflection),
        )
        wrong_source = dataclasses.replace(
            processing.CancelledEvent.with_data(processing.CancelledData(operation_id=request_id, token="guard-token")),
            id=request_id,
            source="forged-child",
            target=hsm.id(reflection),
        )
        operation_matches = getattr(reflection_module.Reflection, "_matches_cancelled")(
            ctx, reflection, wrong_operation
        )
        source_matches = getattr(reflection_module.Reflection, "_matches_cancelled")(ctx, reflection, wrong_source)
        await reflection.stop(reflection.context())
        connection.close()
        return operation_matches, source_matches

    assert asyncio.run(run()) == (False, False)


def test_reflection_revision_cancel_guard_requires_typed_parent_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reflection, connection = reflection_ability()
    try:
        monkeypatch.setattr(
            hsm,
            "id",
            lambda actor: "reflection"
            if actor is reflection
            else "revision"
            if actor is getattr(reflection, "_revision")
            else "select",
        )
        monkeypatch.setattr(processing, "active_operation", lambda owner, operation_id: object())
        missing_parent = dataclasses.replace(
            processing.CancelledEvent.with_data(
                processing.CancelledData(operation_id="cancel:change:operation", token="typed-token")
            ),
            id="cancel:change:operation",
            source="revision",
            target="reflection",
        )
        wrong_parent = dataclasses.replace(
            processing.CancelledEvent.with_data(
                processing.CancelledData(
                    operation_id="cancel:change:operation",
                    token="typed-token",
                    parent_operation_id="other-operation",
                )
            ),
            id="cancel:change:operation",
            source="revision",
            target="reflection",
        )
        correlated = dataclasses.replace(
            processing.CancelledEvent.with_data(
                processing.CancelledData(
                    operation_id="cancel:change:operation",
                    token="typed-token",
                    parent_operation_id="cancel:change:operation",
                )
            ),
            id="cancel:change:operation",
            source="revision",
            target="reflection",
        )

        assert not getattr(cognition.Reflection, "_matches_cancelled")(hsm.Context(), reflection, missing_parent)
        assert not getattr(cognition.Reflection, "_matches_cancelled")(hsm.Context(), reflection, wrong_parent)
        assert getattr(cognition.Reflection, "_matches_cancelled")(hsm.Context(), reflection, correlated)
    finally:
        connection.close()


def test_reflection_select_timeout_cancels_child_and_fails_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, bool]:
        monkeypatch.setattr(reflection_impl, "_CHILD_OPERATION_TIMEOUT", datetime.timedelta(milliseconds=100))
        processor = HangingProcessor()
        reflection, connection = reflection_with_processor(processor)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            reflection,
            reflection.input_event.with_data_and_id(reflection_input(), "timed-reflection"),
        )
        await wait_until(lambda: any(event.name == cognition.Reflection.failed_event.name for event in owner.lifecycle))
        result = list(owner.lifecycle), reflection.state(), processor.cancelled
        await reflection.stop(reflection.context())
        connection.close()
        return result

    lifecycle, state, cancelled = asyncio.run(run())

    assert cancelled
    assert len(lifecycle) == 1
    assert lifecycle[0].name == cognition.Reflection.failed_event.name
    assert state.endswith("/idle")


def test_reflection_change_timeout_cancels_exact_attempt_and_fails_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, bool]:
        monkeypatch.setattr(reflection_impl, "_CHILD_OPERATION_TIMEOUT", datetime.timedelta(0))
        processor = HangingProcessor(
            first_output=(
                processing.SelectedEvent(
                    event=behavior.CreateEvent.name,
                    data=behavior.CreateData(
                        name="TimedChangeBehavior",
                        triggers=(mosfet.InputEvent.name,),
                    ).model_dump(mode="json"),
                ),
            )
        )
        reflection, connection = reflection_with_processor(processor)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            reflection,
            reflection.input_event.with_data_and_id(reflection_input(), "timed-change"),
        )
        await wait_until(lambda: any(event.name == cognition.Reflection.failed_event.name for event in owner.lifecycle))
        result = list(owner.lifecycle), reflection.state(), processor.cancelled
        await reflection.stop(reflection.context())
        connection.close()
        return result

    lifecycle, state, cancelled = asyncio.run(run())

    assert cancelled
    assert len(lifecycle) == 1
    assert lifecycle[0].name == cognition.Reflection.failed_event.name
    assert state.endswith("/idle")


def test_reflection_rejects_valid_select_terminal_from_prior_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reflection, connection = reflection_ability()
    try:
        monkeypatch.setattr(reflection, "state", lambda: "/Reflection/processing")
        monkeypatch.setattr(
            hsm,
            "id",
            lambda actor: "reflection" if actor is reflection else "select-processing",
        )
        turn = reflection_input()
        select_input = reflection_module.SelectInput(
            cognition_input=turn.cognition_input,
            cognition_output=turn.cognition_output,
            operation_id="prior-turn",
            generation="prior-operation-token",
        )
        stale = dataclasses.replace(
            getattr(reflection, "_select_processing").output_event.with_data(
                processing.CompletionData(
                    input=processing.InputData(input=select_input),
                    output=processing.OutputData(),
                )
            ),
            id="reflection:reflection:select",
            source="select-processing",
            target="reflection",
        )

        assert not getattr(cognition.Reflection, "_matches_select_output")(hsm.Context(), reflection, stale)
    finally:
        connection.close()


def test_reflection_builds_one_attachment_group_for_fixed_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups: list[tuple[hsm.Instance, ...]] = []
    group_init = attachment.Group.__init__

    def record_group(group: attachment.Group, *members: hsm.Instance) -> None:
        groups.append(members)
        group_init(group, *members)

    monkeypatch.setattr(attachment.Group, "__init__", record_group)
    reflection, connection = reflection_ability()

    assert isinstance(reflection, cognition.Reflection)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert isinstance(groups[0][0], processing.Processing)
    assert len(groups[1]) == 3
    assert isinstance(groups[1][0], processing.Processing)
    assert isinstance(groups[1][1], revision.Revision)
    assert isinstance(groups[1][2], memory.Memory)
    connection.close()


def test_reflection_rejects_forged_behavior_mutation_during_select() -> None:
    async def run() -> tuple[int, str]:
        processor = HangingProcessor()
        reflection, connection = reflection_with_processor(processor)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        intruder = AttachmentOwner()
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await mosfet.started(ctx, intruder, intruder.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        turn = reflection_input()
        _ = await hsm.dispatch(
            ctx,
            reflection,
            reflection.input_event.with_data_and_id(turn, "current-reflection"),
        )
        await wait_until(lambda: reflection.state().endswith("/processing") and processor.calls == 1)
        forged = dataclasses.replace(
            behavior.CreateEvent.with_data(behavior.CreateData(name="ForgedBehavior", reason="untrusted mutation")),
            id="current-reflection:reflection:select",
            source=hsm.id(intruder),
            target=hsm.id(reflection),
        )
        _ = await hsm.dispatch(ctx, reflection, forged)
        await asyncio.sleep(0)
        calls = processor.calls
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return calls, state

    calls, state = asyncio.run(run())

    assert calls == 1
    assert state.endswith("/processing")


def test_reflection_waits_for_aggregate_attachment_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], str, bool]:
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
        owner = AttachmentOwner()
        reflection, connection = reflection_ability()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reflection-attach",
            ),
        )
        await wait_until(lambda: bool(requests))
        assert owner.lifecycle == []
        assert reflection.state() == "/ReflectionLifecycle/attached/behavior/initializing"
        group, request = requests[0]
        group_is_private = group.context().value(hsm.Keys.Instances) is not reflection.context().value(
            hsm.Keys.Instances
        )
        request_data = request.data
        assert request_data is not None
        reply = request_data.reply_to
        assert reply is not None
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(
                attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=reflection, created=True)),
                id=request.id,
                source=hsm.id(group),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/idle"))
        lifecycle = owner.lifecycle
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return lifecycle, state, group_is_private

    lifecycle, state, group_is_private = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachCompleteEvent.name]
    assert lifecycle[0].id == "reflection-attach"
    assert state == "/ReflectionLifecycle/attached/behavior/idle"
    assert group_is_private


def test_reflection_defers_detach_during_initialization_then_detaches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[attachment.DetachData]], list[hsm.Event[typing.Any]], str]:
        attach_requests: list[tuple[attachment.Group, hsm.Event[attachment.AttachData]]] = []
        detach_requests: list[hsm.Event[attachment.DetachData]] = []
        group_detach = attachment.Group.detach

        async def hold_attach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> None:
            del ctx
            attach_requests.append((group, event))

        async def record_detach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            detach_requests.append(event)
            await group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "attach", hold_attach)
        monkeypatch.setattr(attachment.Group, "detach", record_detach)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reflection, connection = reflection_ability()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reflection-initializing-attach",
            ),
        )
        await wait_until(lambda: bool(attach_requests))
        await reflection.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "reflection-initializing-detach",
            ),
        )
        await asyncio.sleep(0)
        assert detach_requests == []
        group, request = attach_requests[0]
        request_data = request.data
        assert request_data is not None
        reply = request_data.reply_to
        assert reply is not None
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(
                attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=reflection, created=True)),
                id=request.id,
                source=hsm.id(group),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/detached"))
        lifecycle = owner.lifecycle
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return detach_requests, lifecycle, state

    detach_requests, lifecycle, state = asyncio.run(run())

    assert len(detach_requests) == 1
    assert detach_requests[0].id == "reflection-initializing-detach"
    assert [event.name for event in lifecycle] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert [event.id for event in lifecycle] == [
        "reflection-initializing-attach",
        "reflection-initializing-detach",
    ]
    assert state == "/ReflectionLifecycle/detached"


@pytest.mark.parametrize(
    ("expected_state", "first_output"),
    [
        ("processing", None),
        (
            "revising",
            (
                processing.SelectedEvent(
                    event=behavior.CreateEvent.name,
                    data={
                        "event": behavior.CreateEvent.name,
                        "name": "RuntimeDetachBehavior",
                        "reason": "exercise changing detach",
                    },
                ),
            ),
        ),
    ],
)
def test_reflection_detaches_once_and_cancels_active_processing(
    monkeypatch: pytest.MonkeyPatch,
    expected_state: str,
    first_output: processing.Events | None,
) -> None:
    async def run() -> tuple[list[hsm.Event[attachment.DetachData]], bool, list[hsm.Event[typing.Any]], str]:
        requests: list[hsm.Event[attachment.DetachData]] = []
        group_detach = attachment.Group.detach
        processor = HangingProcessor(first_output)

        async def record_detach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            requests.append(event)
            await group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "detach", record_detach)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reflection, connection = reflection_with_processor(processor)
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        _ = await hsm.dispatch(
            ctx,
            reflection,
            reflection.input_event.with_data_and_id(reflection_input(), uuid.uuid4().hex),
        )
        await wait_until(lambda: reflection.state().endswith(f"/{expected_state}"))
        await reflection.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                f"reflection-{expected_state}-detach",
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/detached") and processor.cancelled)
        lifecycle = owner.lifecycle
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return requests, processor.cancelled, lifecycle, state

    requests, cancelled, lifecycle, state = asyncio.run(run())

    assert len(requests) == 2
    # Outer composite group detach carries the request id; nested group fanout
    # correlates members with reply envelope ids (not the parent request id).
    assert requests[0].id == f"reflection-{expected_state}-detach"
    assert requests[1].id
    assert cancelled
    assert [event.name for event in lifecycle] == [attachment.DetachedEvent.name]
    assert lifecycle[0].id == f"reflection-{expected_state}-detach"
    assert state == "/ReflectionLifecycle/detached"


@pytest.mark.parametrize(
    ("expected_state", "held_terminal"),
    [
        ("recalling", "bot.ability.reflection.recalled"),
        ("storing", "bot.ability.reflection.stored"),
    ],
)
def test_reflection_detaches_once_from_synchronous_activity_state(
    monkeypatch: pytest.MonkeyPatch,
    expected_state: str,
    held_terminal: str,
) -> None:
    async def run() -> tuple[list[hsm.Event[attachment.DetachData]], list[hsm.Event[typing.Any]], str]:
        requests: list[hsm.Event[attachment.DetachData]] = []
        group_detach = attachment.Group.detach

        async def record_detach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            requests.append(event)
            await group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "detach", record_detach)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reflection, connection = reflection_ability()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        dispatch = hsm.dispatch

        def hold_activity_terminal(
            dispatch_ctx: hsm.Context | None,
            target: hsm.Dispatchable | None,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[bool]:
            if target is reflection and event.name == held_terminal:
                # Held back, not delivered to the machine.
                held: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
                held.set_result(False)
                return held
            return dispatch(dispatch_ctx, target, event)

        monkeypatch.setattr(hsm, "dispatch", hold_activity_terminal)
        _ = await hsm.dispatch(
            ctx,
            reflection,
            reflection.input_event.with_data_and_id(reflection_input(), uuid.uuid4().hex),
        )
        await wait_until(lambda: reflection.state().endswith(f"/{expected_state}"))
        await reflection.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                f"reflection-{expected_state}-detach",
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/detached"))
        lifecycle = owner.lifecycle
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return requests, lifecycle, state

    requests, lifecycle, state = asyncio.run(run())

    assert len(requests) == 2
    assert requests[0].id == f"reflection-{expected_state}-detach"
    assert requests[1].id
    assert [event.name for event in lifecycle] == [attachment.DetachedEvent.name]
    assert lifecycle[0].id == f"reflection-{expected_state}-detach"
    assert state == "/ReflectionLifecycle/detached"


def test_reflection_reports_aggregate_attachment_failure_and_accepts_retry(
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
        owner = AttachmentOwner()
        reflection, connection = reflection_ability()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reflection-attach-failed",
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
                        actor=reflection,
                        kind=attachment.FailureKind.INITIALIZATION,
                        message="reflection child failed",
                    )
                ),
                id=request.id,
                source=hsm.id(group),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/detached"))
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reflection-attach-retry",
            ),
        )
        await wait_until(lambda: len(requests) == 2)
        lifecycle = owner.lifecycle
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return lifecycle, len(requests), state

    lifecycle, request_count, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [attachment.AttachFailedEvent.name]
    assert lifecycle[0].id == "reflection-attach-failed"
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.INITIALIZATION
    assert request_count == 2
    assert state == "/ReflectionLifecycle/attached/behavior/initializing"


def test_reflection_detaches_once_through_group_and_can_reattach(
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
            requests.append(event)
            await group_detach(group, ctx, event)

        monkeypatch.setattr(attachment.Group, "detach", record_detach)
        ctx = hsm.Context()
        owner = AttachmentOwner()
        reflection, connection = reflection_ability()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reflection-first-attach",
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        await reflection.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "reflection-detach",
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/detached"))
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reflection-second-attach",
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/idle"))
        lifecycle = owner.lifecycle
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return requests, lifecycle, state

    requests, lifecycle, state = asyncio.run(run())

    assert len(requests) == 2
    assert requests[0].id == "reflection-detach"
    assert requests[1].id
    assert [event.name for event in lifecycle] == [
        attachment.DetachedEvent.name,
        attachment.AttachCompleteEvent.name,
    ]
    assert [event.id for event in lifecycle] == ["reflection-detach", "reflection-second-attach"]
    assert state == "/ReflectionLifecycle/attached/behavior/idle"


def test_reflection_reports_detach_failure_and_accepts_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], int, str]:
        requests: list[tuple[attachment.Group, hsm.Event[attachment.DetachData]]] = []
        group_detach = attachment.Group.detach
        hold = True

        async def hold_detach(
            group: attachment.Group,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> None:
            if hold:
                requests.append((group, event))
                return
            await group_detach(group, ctx, event)

        ctx = hsm.Context()
        owner = AttachmentOwner()
        reflection, connection = reflection_ability()
        _ = await mosfet.started(ctx, owner, owner.model)
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        monkeypatch.setattr(attachment.Group, "detach", hold_detach)
        await reflection.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "reflection-detach-failed",
            ),
        )
        await wait_until(lambda: bool(requests))
        group, request = requests[0]
        request_data = request.data
        assert request_data is not None
        reply = request_data.reply_to
        assert reply is not None
        await hsm.dispatch(
            ctx,
            reply,
            dataclasses.replace(
                attachment.DetachFailedEvent.with_data(
                    attachment.FailedData(
                        actor=reflection,
                        kind=attachment.FailureKind.TIMEOUT,
                        message="reflection detach timed out",
                    )
                ),
                id=request.id,
                source=hsm.id(group),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/idle"))
        hold = False
        await reflection.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(
                attachment.DetachData(actor=owner),
                "reflection-detach-retry",
            ),
        )
        await wait_until(lambda: reflection.state().endswith("/detached"))
        lifecycle = owner.lifecycle
        state = reflection.state()
        await reflection.stop(reflection.context())
        connection.close()
        return lifecycle, len(requests), state

    lifecycle, request_count, state = asyncio.run(run())

    assert [event.name for event in lifecycle] == [
        attachment.DetachFailedEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert [event.id for event in lifecycle] == ["reflection-detach-failed", "reflection-detach-retry"]
    assert isinstance(lifecycle[0].data, attachment.FailedData)
    assert lifecycle[0].data.kind is attachment.FailureKind.TIMEOUT
    assert request_count == 1
    assert state == "/ReflectionLifecycle/detached"


def test_reflection_model_uses_group_lifecycle_states_without_synthetic_readiness() -> None:
    model = model_view(require_model(cognition.Reflection.model))
    assert model.qualified_name == "/ReflectionLifecycle"
    assert "/ReflectionLifecycle/attaching" not in model.members
    assert "/ReflectionLifecycle/attached/behavior/initializing" in model.members
    assert "/ReflectionLifecycle/attached/behavior/idle" in model.members
    assert "/ReflectionLifecycle/attached/behavior/detaching" in model.members
    assert "/ReflectionLifecycle/attached/behavior/rebooting" in model.members
    assert "bot.ability.reflection.initializing.complete" not in {
        name
        for names in model.transition_map.values()
        for name in typing.cast(collections.abc.Iterable[str], typing.cast(object, names))
    }


def test_reflection_select_offers_routines_without_forcing_them() -> None:
    from mosfet.abilities.cognition.reflection import reflection as reflection_module

    instructions = reflection_module.SELECT_INSTRUCTIONS
    # A recurring/time-based need or an explicit user request are grounds; the choice stays the model's.
    assert "recurring or time-based need" in instructions
    assert "user explicitly asking" in instructions
    assert "Otherwise return an empty selection" in instructions
    assert "lifetime" in instructions
