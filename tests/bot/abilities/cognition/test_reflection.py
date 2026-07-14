import bot
from bot import habit
from bot.abilities import cognition
from bot.abilities import memory
from bot.abilities import processing
from bot.protocols import attachment

import asyncio
import collections.abc
import dataclasses
import sqlite3
import typing

import hsm
import pytest
from tests.type_helpers import model_view


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

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "ReflectionAttachmentOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.AttachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachFailedEvent), hsm.effect(_record)),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.lifecycle = []


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    for _ in range(100):
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


def reflection_input() -> processing.InputData:
    return processing.InputData(
        input=cognition.reflection.InputData(
            cognition_input=cognition.InputData(
                stimulus=bot.InputEventData(target_device="phone", priority=0),
                abilities=(),
                actors={},
                focus=None,
                focus_candidates=("phone",),
            ),
            cognition_output=(),
        )
    )


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
    assert len(groups) == 1
    assert len(groups[0]) == 3
    assert isinstance(groups[0][0], processing.Processing)
    assert isinstance(groups[0][1], processing.Processing)
    assert isinstance(groups[0][2], memory.Memory)
    connection.close()


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
        _ = await hsm.started(ctx, owner, owner.model)
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
        reply = request.data.reply_to
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
        _ = await hsm.started(ctx, owner, owner.model)
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
        reply = request.data.reply_to
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
            "changing",
            (
                processing.SelectedEvent(
                    event=habit.CreateEvent.name,
                    data={
                        "event": habit.CreateEvent.name,
                        "name": "RuntimeDetachHabit",
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
        _ = await hsm.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        await reflection.apply(reflection_input(), ctx=ctx)
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

    assert len(requests) == 1
    assert requests[0].id == f"reflection-{expected_state}-detach"
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
        _ = await hsm.started(ctx, owner, owner.model)
        await reflection.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        await wait_until(lambda: reflection.state().endswith("/idle"))
        owner.lifecycle.clear()
        dispatch = hsm.dispatch

        def hold_activity_terminal(
            dispatch_ctx: hsm.Context | None,
            target: hsm.Dispatchable | None,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if target is reflection and event.name == held_terminal:
                held = asyncio.get_running_loop().create_future()
                held.set_result(None)
                return held
            return dispatch(dispatch_ctx, target, event)

        monkeypatch.setattr(hsm, "dispatch", hold_activity_terminal)
        await reflection.apply(reflection_input(), ctx=ctx)
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

    assert len(requests) == 1
    assert requests[0].id == f"reflection-{expected_state}-detach"
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
        _ = await hsm.started(ctx, owner, owner.model)
        await reflection.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=owner),
                "reflection-attach-failed",
            ),
        )
        await wait_until(lambda: len(requests) == 1)
        group, request = requests[0]
        reply = request.data.reply_to
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
        _ = await hsm.started(ctx, owner, owner.model)
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

    assert len(requests) == 1
    assert requests[0].id == "reflection-detach"
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
        _ = await hsm.started(ctx, owner, owner.model)
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
        reply = request.data.reply_to
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
    transition_events = {name for names in model.transition_map.values() for name in names}

    assert model.qualified_name == "/ReflectionLifecycle"
    assert "/ReflectionLifecycle/attaching" not in model.members
    assert "/ReflectionLifecycle/attached/behavior/initializing" in model.members
    assert "/ReflectionLifecycle/attached/behavior/idle" in model.members
    assert "/ReflectionLifecycle/attached/behavior/detaching" in model.members
    assert "/ReflectionLifecycle/attached/behavior/degraded" in model.members
    assert "bot.ability.reflection.initializing.complete" not in transition_events
