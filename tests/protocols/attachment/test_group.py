"""Tests for the event-driven attachment group."""

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm
import bot
import pytest

from bot.protocols import attachment


BroadcastEvent = hsm.Event[None](name="attachment.group.test.broadcast")


class BroadcastRecorder(hsm.Instance, attachment.Attachment):
    count: int
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str

    def __init__(self) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self.count = 0

    @typing.override
    def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.AttachData],
    ) -> collections.abc.Awaitable[None]:
        return hsm.dispatch(ctx, self, event)

    @typing.override
    def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.DetachData],
    ) -> collections.abc.Awaitable[None]:
        return hsm.dispatch(ctx, self, event)

    @staticmethod
    def _record(ctx: hsm.Context, instance: "BroadcastRecorder", event: hsm.Event[None]) -> None:
        del ctx, event
        instance.count += 1

    model: typing.ClassVar[hsm.Model] = bot.define(
        "AttachmentGroupBroadcastRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(BroadcastEvent), hsm.effect(_record)),
        ),
    )


class LifecycleRecorder(hsm.Instance):
    events: list[hsm.Event[typing.Any]]
    recorded: asyncio.Event

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.recorded = asyncio.Event()

    @staticmethod
    def _record(ctx: hsm.Context, instance: "LifecycleRecorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)
        _ = instance.recorded.set()

    model: typing.ClassVar[hsm.Model] = bot.define(
        "AttachmentGroupLifecycleRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.AttachFailedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(attachment.DetachFailedEvent), hsm.effect(_record)),
        ),
    )


class TestAttachment(hsm.Instance, attachment.Attachment):
    __test__: typing.ClassVar[bool] = False
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
    attach_calls: list[hsm.Event[attachment.AttachData]]
    detach_calls: list[hsm.Event[attachment.DetachData]]
    attach_contexts: list[hsm.Context]
    detach_contexts: list[hsm.Context]
    fail_attach: bool
    fail_detach: bool
    raise_attach: bool
    raise_detach: bool
    respond_attach: bool
    respond_detach: bool
    created: bool
    attach_started: asyncio.Event
    detach_started: asyncio.Event
    attach_finished: asyncio.Event
    detach_finished: asyncio.Event
    attach_release: asyncio.Event | None
    detach_release: asyncio.Event | None
    attach_failure_message: str
    detach_failure_message: str

    model: typing.ClassVar[hsm.Model] = bot.define(
        "TestAttachment",
        hsm.initial(hsm.target("ready")),
        hsm.state("ready"),
    )

    def __init__(
        self,
        *,
        fail_attach: bool = False,
        fail_detach: bool = False,
        raise_attach: bool = False,
        raise_detach: bool = False,
        respond_attach: bool = True,
        respond_detach: bool = True,
        created: bool = True,
        attach_release: asyncio.Event | None = None,
        detach_release: asyncio.Event | None = None,
        attach_failure_message: str = "test attachment failed",
        detach_failure_message: str = "test detachment failed",
    ) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self.attach_calls = []
        self.detach_calls = []
        self.attach_contexts = []
        self.detach_contexts = []
        self.fail_attach = fail_attach
        self.fail_detach = fail_detach
        self.raise_attach = raise_attach
        self.raise_detach = raise_detach
        self.respond_attach = respond_attach
        self.respond_detach = respond_detach
        self.created = created
        self.attach_started = asyncio.Event()
        self.detach_started = asyncio.Event()
        self.attach_finished = asyncio.Event()
        self.detach_finished = asyncio.Event()
        self.attach_release = attach_release
        self.detach_release = detach_release
        self.attach_failure_message = attach_failure_message
        self.detach_failure_message = detach_failure_message

    @typing.override
    async def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.AttachData],
    ) -> None:
        self.attach_calls.append(event)
        self.attach_contexts.append(ctx)
        _ = self.attach_started.set()
        if self.attach_release is not None:
            _ = await self.attach_release.wait()
        if self.raise_attach:
            _ = self.attach_finished.set()
            raise RuntimeError("test attach exception")
        if not self.respond_attach:
            _ = self.attach_finished.set()
            return
        data = event.data
        assert isinstance(data, attachment.AttachData)
        target = data.actor if data.reply_to is None else data.reply_to
        if self.fail_attach:
            result = attachment.AttachFailedEvent.with_data(
                attachment.FailedData(
                    actor=data.actor,
                    kind=attachment.FailureKind.INITIALIZATION,
                    message=self.attach_failure_message,
                )
            )
        else:
            result = attachment.AttachCompleteEvent.with_data(
                attachment.AttachCompleteData(actor=data.actor, created=self.created, reply_to=data.reply_to)
            )
        await hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                result,
                id=event.id,
                source=hsm.id(self),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            ),
        )
        _ = self.attach_finished.set()

    @typing.override
    async def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.DetachData],
    ) -> None:
        self.detach_calls.append(event)
        self.detach_contexts.append(ctx)
        _ = self.detach_started.set()
        if self.detach_release is not None:
            _ = await self.detach_release.wait()
        if self.raise_detach:
            _ = self.detach_finished.set()
            raise RuntimeError("test detach exception")
        if not self.respond_detach:
            _ = self.detach_finished.set()
            return
        data = event.data
        assert isinstance(data, attachment.DetachData)
        target = data.actor if data.reply_to is None else data.reply_to
        if self.fail_detach:
            result = attachment.DetachFailedEvent.with_data(
                attachment.FailedData(
                    actor=data.actor,
                    kind=attachment.FailureKind.ROLLBACK,
                    message=self.detach_failure_message,
                )
            )
        else:
            result = attachment.DetachedEvent.with_data(attachment.DetachedData(actor=data.actor, removed=True))
        await hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                result,
                id=event.id,
                source=hsm.id(self),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            ),
        )
        _ = self.detach_finished.set()


def test_group_implements_attachment_and_dispatchable_protocols() -> None:
    group = attachment.Group()
    attachment_contract: attachment.Attachment = group
    dispatchable_contract: hsm.Dispatchable = group

    assert isinstance(attachment_contract, attachment.Attachment)
    assert isinstance(dispatchable_contract, hsm.Dispatchable)


def test_group_is_not_an_hsm_group() -> None:
    assert not issubclass(attachment.Group, hsm.Group)


def test_group_members_start_detached() -> None:
    async def run() -> str:
        ctx = hsm.Context()
        group = attachment.Group(TestAttachment(), TestAttachment())
        _ = await bot.started(ctx, group, group.model)
        return group.state()

    assert asyncio.run(run()) == "/AttachmentGroup/detached"


def test_group_uses_the_durable_request_context_for_member_lifecycle() -> None:
    async def run() -> tuple[bool, bool]:
        ctx = hsm.Context()
        group_ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(group_ctx, group, group.model)

        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)

        return member.attach_contexts == [ctx], member.detach_contexts == [ctx]

    assert asyncio.run(run()) == (True, True)


def test_group_manages_attachment_lifecycle() -> None:
    async def run() -> tuple[str, LifecycleRecorder, tuple[TestAttachment, TestAttachment]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        members = (TestAttachment(), TestAttachment())
        group = attachment.Group(*members)
        _ = await bot.started(ctx, actor, actor.model)
        for member in members:
            _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=actor, timeout=datetime.timedelta(seconds=7)),
                "group-attach",
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=actor), "group-detach"),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor, members

    state, actor, members = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in actor.events] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert [event.id for event in actor.events] == ["group-attach", "group-detach"]
    assert [len(member.attach_calls) for member in members] == [1, 1]
    assert [len(member.detach_calls) for member in members] == [1, 1]
    assert [member.attach_calls[0].data.timeout for member in members if member.attach_calls[0].data is not None] == [
        datetime.timedelta(seconds=7),
        datetime.timedelta(seconds=7),
    ]


def test_group_attaches_members_concurrently_and_waits_for_every_outcome() -> None:
    async def run() -> tuple[bool, str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        first = TestAttachment(attach_release=first_release)
        second = TestAttachment(attach_release=second_release)
        group = attachment.Group(first, second)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, second, second.model)
        _ = await bot.started(ctx, group, group.model)

        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(
            asyncio.gather(first.attach_started.wait(), second.attach_started.wait()),
            timeout=1,
        )
        both_started = first.attach_started.is_set() and second.attach_started.is_set()
        waiting_state = group.state()
        _ = first_release.set()
        _ = await asyncio.wait_for(first.attach_finished.wait(), timeout=1)
        state_after_first = group.state()
        _ = second_release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return both_started, waiting_state, state_after_first, actor.events

    both_started, waiting_state, state_after_first, recorded = asyncio.run(run())

    assert both_started
    assert waiting_state == "/AttachmentGroup/attaching"
    assert state_after_first == "/AttachmentGroup/attaching"
    assert [event.name for event in recorded] == [attachment.AttachCompleteEvent.name]


def test_group_detaches_members_concurrently_and_waits_for_every_outcome() -> None:
    async def run() -> tuple[bool, str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first = TestAttachment()
        second = TestAttachment()
        group = attachment.Group(first, second)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, second, second.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()

        first_release = asyncio.Event()
        second_release = asyncio.Event()
        first.detach_release = first_release
        second.detach_release = second_release
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(
            asyncio.gather(first.detach_started.wait(), second.detach_started.wait()),
            timeout=1,
        )
        both_started = first.detach_started.is_set() and second.detach_started.is_set()
        waiting_state = group.state()
        _ = first_release.set()
        _ = await asyncio.wait_for(first.detach_finished.wait(), timeout=1)
        state_after_first = group.state()
        _ = second_release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return both_started, waiting_state, state_after_first, actor.events

    both_started, waiting_state, state_after_first, recorded = asyncio.run(run())

    assert both_started
    assert waiting_state == "/AttachmentGroup/detaching"
    assert state_after_first == "/AttachmentGroup/detaching"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]


def test_group_waits_for_all_attach_outcomes_before_rolling_back() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]], int]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        release = asyncio.Event()
        failing = TestAttachment(fail_attach=True)
        delayed = TestAttachment(attach_release=release)
        group = attachment.Group(failing, delayed)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, failing, failing.model)
        _ = await bot.started(ctx, delayed, delayed.model)
        _ = await bot.started(ctx, group, group.model)

        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(
            asyncio.gather(failing.attach_finished.wait(), delayed.attach_started.wait()),
            timeout=1,
        )
        waiting_state = group.state()
        assert actor.events == []
        _ = release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return waiting_state, actor.events, len(delayed.detach_calls)

    waiting_state, recorded, rollback_calls = asyncio.run(run())

    assert waiting_state == "/AttachmentGroup/attaching"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert rollback_calls == 1


def test_group_selects_attach_failure_by_member_order_not_arrival_order() -> None:
    async def run() -> attachment.FailedData:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        release = asyncio.Event()
        first = TestAttachment(
            fail_attach=True,
            attach_release=release,
            attach_failure_message="first member failed",
        )
        second = TestAttachment(fail_attach=True, attach_failure_message="second member failed")
        group = attachment.Group(first, second)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, second, second.model)
        _ = await bot.started(ctx, group, group.model)

        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(second.attach_finished.wait(), timeout=1)
        assert actor.events == []
        _ = release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        data = actor.events[0].data
        assert isinstance(data, attachment.FailedData)
        return data

    failure = asyncio.run(run())

    assert failure.message == "first member failed"


def test_group_selects_detach_failure_by_member_order_not_arrival_order() -> None:
    async def run() -> attachment.FailedData:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        release = asyncio.Event()
        first = TestAttachment(
            fail_detach=True,
            detach_release=release,
            detach_failure_message="first member failed",
        )
        second = TestAttachment(fail_detach=True, detach_failure_message="second member failed")
        group = attachment.Group(first, second)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, second, second.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()

        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(second.detach_finished.wait(), timeout=1)
        assert len(actor.events) == 1
        _ = release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        data = actor.events[1].data
        assert isinstance(data, attachment.FailedData)
        return data

    failure = asyncio.run(run())

    assert failure.message == "first member failed"


def test_group_rolls_back_created_members_concurrently() -> None:
    async def run() -> tuple[bool, str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first_release = asyncio.Event()
        second_release = asyncio.Event()
        first = TestAttachment(detach_release=first_release)
        second = TestAttachment(detach_release=second_release)
        failing = TestAttachment(fail_attach=True)
        group = attachment.Group(first, second, failing)
        _ = await bot.started(ctx, actor, actor.model)
        for member in (first, second, failing):
            _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)

        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(
            asyncio.gather(first.detach_started.wait(), second.detach_started.wait()),
            timeout=1,
        )
        both_started = first.detach_started.is_set() and second.detach_started.is_set()
        waiting_state = group.state()
        _ = first_release.set()
        _ = await asyncio.wait_for(first.detach_finished.wait(), timeout=1)
        state_after_first = group.state()
        _ = second_release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return both_started, waiting_state, state_after_first, actor.events

    both_started, waiting_state, state_after_first, recorded = asyncio.run(run())

    assert both_started
    assert waiting_state == "/AttachmentGroup/rolling_back"
    assert state_after_first == "/AttachmentGroup/rolling_back"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]


def test_group_selects_rollback_failure_by_member_order_not_arrival_order() -> None:
    async def run() -> attachment.FailedData:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        release = asyncio.Event()
        first = TestAttachment(
            fail_detach=True,
            detach_release=release,
            detach_failure_message="first rollback failed",
        )
        second = TestAttachment(fail_detach=True, detach_failure_message="second rollback failed")
        failing = TestAttachment(fail_attach=True)
        group = attachment.Group(first, second, failing)
        _ = await bot.started(ctx, actor, actor.model)
        for member in (first, second, failing):
            _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)

        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(second.detach_finished.wait(), timeout=1)
        assert actor.events == []
        _ = release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        data = actor.events[0].data
        assert isinstance(data, attachment.FailedData)
        return data

    failure = asyncio.run(run())

    assert failure.kind is attachment.FailureKind.ROLLBACK
    assert "first rollback failed" in failure.message


def test_group_rolls_back_created_members_when_attach_fails() -> None:
    async def run() -> tuple[str, LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first = TestAttachment()
        failing = TestAttachment(fail_attach=True)
        group = attachment.Group(first, failing)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, failing, failing.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "group-failure"),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor, first, failing

    state, actor, first, failing = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in actor.events] == [attachment.AttachFailedEvent.name]
    assert actor.events[0].id == "group-failure"
    assert len(first.attach_calls) == 1
    assert len(first.detach_calls) == 1
    assert len(failing.attach_calls) == 1


def test_group_rolls_back_created_members_when_first_member_attach_fails() -> None:
    async def run() -> tuple[str, LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        failing = TestAttachment(fail_attach=True)
        unattempted = TestAttachment()
        group = attachment.Group(failing, unattempted)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, failing, failing.model)
        _ = await bot.started(ctx, unattempted, unattempted.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor, failing, unattempted

    state, actor, failing, unattempted = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in actor.events] == [attachment.AttachFailedEvent.name]
    assert len(failing.attach_calls) == 1
    assert failing.detach_calls == []
    assert len(unattempted.attach_calls) == 1
    assert len(unattempted.detach_calls) == 1


def test_group_does_not_roll_back_preexisting_members_after_reattach_failure() -> None:
    async def run() -> tuple[str, LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        existing = TestAttachment()
        failing = TestAttachment()
        group = attachment.Group(existing, failing)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, existing, existing.model)
        _ = await bot.started(ctx, failing, failing.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()

        existing.created = False
        failing.fail_attach = True
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor, existing, failing

    state, actor, existing, failing = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert [event.name for event in actor.events] == [
        attachment.AttachCompleteEvent.name,
        attachment.AttachFailedEvent.name,
    ]
    assert existing.detach_calls == []
    assert len(existing.attach_calls) == 2
    assert len(failing.attach_calls) == 2


def test_group_composes_nested_group_lifecycles() -> None:
    async def run() -> tuple[list[hsm.Event[typing.Any]], tuple[TestAttachment, TestAttachment]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        leaves = (TestAttachment(), TestAttachment())
        nested = attachment.Group(*leaves)
        group = attachment.Group(nested)
        _ = await bot.started(ctx, actor, actor.model)
        for leaf in leaves:
            _ = await bot.started(ctx, leaf, leaf.model)
        _ = await bot.started(ctx, nested, nested.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return actor.events, leaves

    recorded, leaves = asyncio.run(run())

    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert [len(member.attach_calls) for member in leaves] == [1, 1]
    assert [len(member.detach_calls) for member in leaves] == [1, 1]


def test_group_concurrently_waits_for_nested_and_direct_members() -> None:
    async def run() -> tuple[bool, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        leaf_release = asyncio.Event()
        direct_release = asyncio.Event()
        leaf = TestAttachment(attach_release=leaf_release)
        direct = TestAttachment(attach_release=direct_release)
        nested = attachment.Group(leaf)
        group = attachment.Group(nested, direct)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, leaf, leaf.model)
        _ = await bot.started(ctx, direct, direct.model)
        _ = await bot.started(ctx, nested, nested.model)
        _ = await bot.started(ctx, group, group.model)

        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(
            asyncio.gather(leaf.attach_started.wait(), direct.attach_started.wait()),
            timeout=1,
        )
        both_started = leaf.attach_started.is_set() and direct.attach_started.is_set()
        _ = direct_release.set()
        _ = await asyncio.wait_for(direct.attach_finished.wait(), timeout=1)
        state_after_direct = group.state()
        _ = leaf_release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return both_started, state_after_direct, actor.events

    both_started, state_after_direct, recorded = asyncio.run(run())

    assert both_started
    assert state_after_direct == "/AttachmentGroup/attaching"
    assert [event.name for event in recorded] == [attachment.AttachCompleteEvent.name]


def test_group_attempts_every_detach_before_reporting_failure() -> None:
    async def run() -> tuple[LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first = TestAttachment()
        failing = TestAttachment(fail_detach=True)
        group = attachment.Group(first, failing)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, failing, failing.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return actor, first, failing

    actor, first, failing = asyncio.run(run())

    assert [event.name for event in actor.events] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
    ]
    assert len(first.detach_calls) == 1
    assert len(failing.detach_calls) == 1


def test_group_times_out_detach_and_accepts_a_retry() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]], int]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(respond_detach=False)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()

        await group.detach(
            ctx,
            attachment.DetachEvent.with_data(
                attachment.DetachData(actor=actor, timeout=datetime.timedelta(milliseconds=1))
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        member.respond_detach = True
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor.events, len(member.detach_calls)

    state, recorded, detach_calls = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert isinstance(recorded[1].data, attachment.FailedData)
    assert recorded[1].data.kind is attachment.FailureKind.TIMEOUT
    assert detach_calls == 2


def test_nested_group_propagates_detach_timeout_and_recovers_for_retry() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]], datetime.timedelta]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        leaf = TestAttachment(respond_detach=False)
        nested = attachment.Group(leaf)
        group = attachment.Group(nested)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, leaf, leaf.model)
        _ = await bot.started(ctx, nested, nested.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()

        timeout = datetime.timedelta(milliseconds=1)
        await group.detach(
            ctx,
            attachment.DetachEvent.with_data(attachment.DetachData(actor=actor, timeout=timeout)),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        leaf.respond_detach = True
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        first_request = leaf.detach_calls[0].data
        assert isinstance(first_request, attachment.DetachData)
        return group.state(), actor.events, first_request.timeout

    state, recorded, propagated_timeout = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert isinstance(recorded[1].data, attachment.FailedData)
    assert recorded[1].data.kind is attachment.FailureKind.TIMEOUT
    assert propagated_timeout == datetime.timedelta(milliseconds=1)


def test_group_reattaches_removed_members_before_reporting_detach_failure() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]], int]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        removed = TestAttachment()
        failing = TestAttachment(fail_detach=True)
        group = attachment.Group(removed, failing)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, removed, removed.model)
        _ = await bot.started(ctx, failing, failing.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        removed.created = True

        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor.events, len(removed.attach_calls)

    state, recorded, removed_attach_calls = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
    ]
    assert removed_attach_calls == 2


def test_group_attempts_every_rollback_after_detach_failure() -> None:
    async def run() -> tuple[str, str, LifecycleRecorder, TestAttachment, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first = TestAttachment()
        rollback_failing = TestAttachment(fail_detach=True)
        attach_failing = TestAttachment(fail_attach=True)
        members = (first, rollback_failing, attach_failing)
        group = attachment.Group(*members)
        _ = await bot.started(ctx, actor, actor.model)
        for member in members:
            _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        failed_state = group.state()
        rollback_failing.fail_detach = False
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return failed_state, group.state(), actor, first, rollback_failing, attach_failing

    failed_state, recovered_state, actor, first, rollback_failing, attach_failing = asyncio.run(run())

    assert failed_state == "/AttachmentGroup/attached"
    assert recovered_state == "/AttachmentGroup/detached"
    assert [event.name for event in actor.events] == [
        attachment.AttachFailedEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert isinstance(actor.events[0].data, attachment.FailedData)
    assert actor.events[0].data.kind is attachment.FailureKind.ROLLBACK
    assert len(first.detach_calls) == 2
    assert len(rollback_failing.detach_calls) == 2
    assert len(attach_failing.detach_calls) == 1


def test_group_converts_member_attach_exception_to_failure() -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(raise_attach=True)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        request = member.attach_calls[0]
        assert isinstance(request.data, attachment.AttachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)
        return group.state(), reply.state(), actor.events

    state, reply_state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert reply_state == "/AttachmentGroupReply/done"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.DISPATCH


def test_group_converts_member_start_exception_to_correlated_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, group, group.model)
        started = hsm.started

        async def fail_member_start[T: hsm.Instance](
            start_ctx: hsm.Context | None,
            instance: T,
            model: hsm.Model,
            config: hsm.Config | None = None,
        ) -> T:
            if instance is member:
                raise RuntimeError("member start failed")
            return await started(start_ctx, instance, model, config)

        monkeypatch.setattr(hsm, "started", fail_member_start)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=actor),
                "member-start-failed",
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert recorded[0].id == "member-start-failed"
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.INITIALIZATION


def test_group_converts_reply_start_exception_to_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        started = hsm.started

        async def fail_reply_start[T: hsm.Instance](
            start_ctx: hsm.Context | None,
            instance: T,
            model: hsm.Model,
            config: hsm.Config | None = None,
        ) -> T:
            if model.qualified_name == "/AttachmentGroupReply":
                raise RuntimeError("reply start failed")
            return await started(start_ctx, instance, model, config)

        monkeypatch.setattr(hsm, "started", fail_reply_start)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.DISPATCH
    assert "reply start failed" in recorded[0].data.message


def test_group_attach_cancellation_during_reply_start_fails_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, str, int, list[hsm.Event[typing.Any]]]:
        lifetime = hsm.Context()
        group_lifetime, cancel = lifetime.with_cancel()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(lifetime, actor, actor.model)
        _ = await bot.started(lifetime, member, member.model)
        _ = await bot.started(group_lifetime, group, group.model)
        reply_started = asyncio.Event()
        replies: list[hsm.Instance] = []
        started = hsm.started

        async def block_reply_start[T: hsm.Instance](
            start_ctx: hsm.Context | None,
            instance: T,
            model: hsm.Model,
            config: hsm.Config | None = None,
        ) -> T:
            result = await started(start_ctx, instance, model, config)
            if model.qualified_name == "/AttachmentGroupReply":
                _ = replies.append(result)
                _ = reply_started.set()
                _ = await asyncio.Event().wait()
            return result

        monkeypatch.setattr(hsm, "started", block_reply_start)
        await group.attach(lifetime, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(reply_started.wait(), timeout=1)
        cancel()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), replies[0].state(), len(member.attach_calls), actor.events

    group_state, reply_state, attach_calls, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/detached"
    # hsm 1.3.2+: cancelled reply is stopped; state() is empty (was model root previously).
    assert reply_state in {"", "/AttachmentGroupReply"}
    assert attach_calls == 0
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.DISPATCH


def test_group_detach_cancellation_during_reply_start_fails_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, str, int, list[hsm.Event[typing.Any]]]:
        lifetime = hsm.Context()
        group_lifetime, cancel = lifetime.with_cancel()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(lifetime, actor, actor.model)
        _ = await bot.started(lifetime, member, member.model)
        _ = await bot.started(group_lifetime, group, group.model)
        await group.attach(lifetime, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        reply_started = asyncio.Event()
        replies: list[hsm.Instance] = []
        started = hsm.started

        async def block_reply_start[T: hsm.Instance](
            start_ctx: hsm.Context | None,
            instance: T,
            model: hsm.Model,
            config: hsm.Config | None = None,
        ) -> T:
            result = await started(start_ctx, instance, model, config)
            if model.qualified_name == "/AttachmentGroupReply":
                _ = replies.append(result)
                _ = reply_started.set()
                _ = await asyncio.Event().wait()
            return result

        monkeypatch.setattr(hsm, "started", block_reply_start)
        await group.detach(lifetime, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(reply_started.wait(), timeout=1)
        cancel()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), replies[0].state(), len(member.detach_calls), actor.events

    group_state, reply_state, detach_calls, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/attached"
    # hsm 1.3.2+: cancelled reply is stopped; state() is empty (was model root previously).
    assert reply_state in {"", "/AttachmentGroupReply"}
    assert detach_calls == 0
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
    ]
    assert isinstance(recorded[1].data, attachment.FailedData)
    assert recorded[1].data.kind is attachment.FailureKind.DISPATCH


def test_group_converts_member_detach_exception_to_failure() -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(raise_detach=True)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        request = member.detach_calls[0]
        assert isinstance(request.data, attachment.DetachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)
        return group.state(), reply.state(), actor.events

    state, reply_state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert reply_state == "/AttachmentGroupReply/done"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
    ]
    assert isinstance(recorded[1].data, attachment.FailedData)
    assert recorded[1].data.kind is attachment.FailureKind.DISPATCH


def test_group_converts_rollback_detach_exception_to_failure() -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        rollback_failing = TestAttachment(raise_detach=True)
        attach_failing = TestAttachment(fail_attach=True)
        group = attachment.Group(rollback_failing, attach_failing)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, rollback_failing, rollback_failing.model)
        _ = await bot.started(ctx, attach_failing, attach_failing.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        request = rollback_failing.detach_calls[0]
        assert isinstance(request.data, attachment.DetachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)
        return group.state(), reply.state(), actor.events

    state, reply_state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert reply_state == "/AttachmentGroupReply/done"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.ROLLBACK


def test_group_ignores_foreign_terminal_event() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(respond_attach=False)
        foreign = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, foreign, foreign.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(member.attach_started.wait(), timeout=1)
        request = member.attach_calls[0]
        assert isinstance(request.data, attachment.AttachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)
        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=actor, created=True)),
                id=request.id,
                source=hsm.id(foreign),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attaching"
    assert recorded == []


def test_group_reply_accepts_only_correlated_member_terminal_event() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(respond_attach=False)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "attach"))
        _ = await asyncio.wait_for(member.attach_started.wait(), timeout=1)
        request = member.attach_calls[0]
        assert isinstance(request.data, attachment.AttachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)
        terminal = attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=actor, created=True))
        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                terminal,
                id="wrong",
                source=hsm.id(member),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                terminal,
                id=request.id,
                source=hsm.id(member),
                target=hsm.id(group),
                metadata=dict(request.metadata),
            ),
        )
        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                terminal,
                id=request.id,
                source=hsm.id(member),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert [event.id for event in recorded] == ["attach"]


def test_group_attach_reply_ignores_terminal_for_another_actor() -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        other_actor = LifecycleRecorder()
        member = TestAttachment(respond_attach=False)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, other_actor, other_actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(member.attach_started.wait(), timeout=1)
        request = member.attach_calls[0]
        assert isinstance(request.data, attachment.AttachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)

        wrong_actor = dataclasses.replace(
            attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=other_actor, created=True)),
            id=request.id,
            source=hsm.id(member),
            target=hsm.id(reply),
            metadata=dict(request.metadata),
        )
        await hsm.Instance.dispatch(reply, ctx, wrong_actor)
        group_state = group.state()
        reply_state = reply.state()
        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                wrong_actor,
                data=attachment.AttachCompleteData(actor=actor, created=True),
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group_state, reply_state, actor.events

    group_state, reply_state, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/attaching"
    assert reply_state == "/AttachmentGroupReply/waiting"
    assert [event.name for event in recorded] == [attachment.AttachCompleteEvent.name]


def test_group_attach_reply_ignores_detach_terminal_event() -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(respond_attach=False)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(member.attach_started.wait(), timeout=1)
        request = member.attach_calls[0]
        assert isinstance(request.data, attachment.AttachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)

        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                attachment.DetachedEvent.with_data(attachment.DetachedData(actor=actor, removed=True)),
                id=request.id,
                source=hsm.id(member),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        wrong_kind_group_state = group.state()
        wrong_kind_reply_state = reply.state()

        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=actor, created=True)),
                id=request.id,
                source=hsm.id(member),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return wrong_kind_group_state, wrong_kind_reply_state, actor.events

    group_state, reply_state, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/attaching"
    assert reply_state == "/AttachmentGroupReply/waiting"
    assert [event.name for event in recorded] == [attachment.AttachCompleteEvent.name]


def test_group_detach_reply_ignores_attach_terminal_event() -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        detach_release = asyncio.Event()
        member = TestAttachment(detach_release=detach_release)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(member.detach_started.wait(), timeout=1)
        request = member.detach_calls[0]
        assert isinstance(request.data, attachment.DetachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)

        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=actor, created=True)),
                id=request.id,
                source=hsm.id(member),
                target=hsm.id(reply),
                metadata=dict(request.metadata),
            ),
        )
        wrong_kind_group_state = group.state()
        wrong_kind_reply_state = reply.state()

        _ = detach_release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return wrong_kind_group_state, wrong_kind_reply_state, actor.events

    group_state, reply_state, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/detaching"
    assert reply_state == "/AttachmentGroupReply/waiting"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]


def test_group_detach_reply_ignores_terminal_for_another_actor() -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        other_actor = LifecycleRecorder()
        member = TestAttachment(respond_detach=False)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, other_actor, other_actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(member.detach_started.wait(), timeout=1)
        request = member.detach_calls[0]
        assert isinstance(request.data, attachment.DetachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)

        wrong_actor = dataclasses.replace(
            attachment.DetachedEvent.with_data(attachment.DetachedData(actor=other_actor, removed=True)),
            id=request.id,
            source=hsm.id(member),
            target=hsm.id(reply),
            metadata=dict(request.metadata),
        )
        await hsm.Instance.dispatch(reply, ctx, wrong_actor)
        group_state = group.state()
        reply_state = reply.state()
        await hsm.Instance.dispatch(
            reply,
            ctx,
            dataclasses.replace(
                wrong_actor,
                data=attachment.DetachedData(actor=actor, removed=True),
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group_state, reply_state, actor.events

    group_state, reply_state, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/detaching"
    assert reply_state == "/AttachmentGroupReply/waiting"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]


@pytest.mark.parametrize("blocked", [False, True], ids=["returned", "blocked"])
def test_group_cancellation_fails_operation_and_finishes_started_reply(blocked: bool) -> None:
    async def run() -> tuple[str, tuple[str, str], list[hsm.Event[typing.Any]]]:
        lifetime = hsm.Context()
        group_lifetime, cancel = lifetime.with_cancel()
        actor = LifecycleRecorder()
        members = tuple(
            TestAttachment(attach_release=asyncio.Event()) if blocked else TestAttachment(respond_attach=False)
            for _ in range(2)
        )
        group = attachment.Group(*members)
        _ = await bot.started(lifetime, actor, actor.model)
        for member in members:
            _ = await bot.started(lifetime, member, member.model)
        _ = await bot.started(group_lifetime, group, group.model)
        await group.attach(
            lifetime,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)),
        )
        _ = await asyncio.wait_for(
            asyncio.gather(
                *(member.attach_started.wait() if blocked else member.attach_finished.wait() for member in members)
            ),
            timeout=1,
        )
        replies: list[hsm.Instance] = []
        for member in members:
            request = member.attach_calls[0]
            assert isinstance(request.data, attachment.AttachData)
            reply = request.data.reply_to
            assert isinstance(reply, hsm.Instance)
            replies.append(reply)

        cancel()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), (replies[0].state(), replies[1].state()), actor.events

    group_state, reply_states, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/detached"
    assert reply_states == ("/AttachmentGroupReply/done", "/AttachmentGroupReply/done")
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.DISPATCH


@pytest.mark.parametrize("blocked", [False, True], ids=["returned", "blocked"])
def test_group_detach_cancellation_fails_operation_and_finishes_started_reply(blocked: bool) -> None:
    async def run() -> tuple[str, str, list[hsm.Event[typing.Any]]]:
        lifetime = hsm.Context()
        group_lifetime, cancel = lifetime.with_cancel()
        actor = LifecycleRecorder()
        member = TestAttachment(detach_release=asyncio.Event()) if blocked else TestAttachment(respond_detach=False)
        group = attachment.Group(member)
        _ = await bot.started(lifetime, actor, actor.model)
        _ = await bot.started(lifetime, member, member.model)
        _ = await bot.started(group_lifetime, group, group.model)
        await group.attach(lifetime, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()
        await group.detach(lifetime, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        _ = await asyncio.wait_for(
            member.detach_started.wait() if blocked else member.detach_finished.wait(),
            timeout=1,
        )
        request = member.detach_calls[0]
        assert isinstance(request.data, attachment.DetachData)
        reply = request.data.reply_to
        assert isinstance(reply, hsm.Instance)

        cancel()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), reply.state(), actor.events

    group_state, reply_state, recorded = asyncio.run(run())

    assert group_state == "/AttachmentGroup/attached"
    assert reply_state == "/AttachmentGroupReply/done"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
    ]
    assert isinstance(recorded[1].data, attachment.FailedData)
    assert recorded[1].data.kind is attachment.FailureKind.DISPATCH


def test_group_ignores_stale_public_terminal_event_during_new_operation() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "first-attach"),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        actor.recorded.clear()

        member.respond_attach = False
        member.attach_started.clear()
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "first-attach"),
        )
        _ = await asyncio.wait_for(member.attach_started.wait(), timeout=1)
        first_request = member.attach_calls[0]
        assert isinstance(first_request.data, attachment.AttachData)
        second_request = member.attach_calls[1]
        assert isinstance(second_request.data, attachment.AttachData)
        assert first_request.id != second_request.id
        assert first_request.metadata == second_request.metadata == {}
        active_reply = second_request.data.reply_to
        assert isinstance(active_reply, hsm.Instance)
        await hsm.Instance.dispatch(
            active_reply,
            ctx,
            dataclasses.replace(
                attachment.AttachCompleteEvent.with_data(attachment.AttachCompleteData(actor=actor, created=True)),
                id=first_request.id,
                source=hsm.id(member),
                target=hsm.id(active_reply),
                metadata=dict(first_request.metadata),
            ),
        )
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attaching"
    assert [event.id for event in recorded] == ["first-attach"]


def test_group_times_out_when_member_does_not_reply() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(respond_attach=False)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data(
                attachment.AttachData(actor=actor, timeout=datetime.timedelta(milliseconds=1))
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.TIMEOUT


def test_group_member_metadata_cannot_control_completion_correlation() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        release = asyncio.Event()
        member = TestAttachment(attach_release=release)
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(
            ctx,
            dataclasses.replace(
                attachment.AttachEvent.with_data(
                    attachment.AttachData(actor=actor, timeout=datetime.timedelta(milliseconds=20))
                ),
                metadata={"traceparent": "test-trace"},
            ),
        )
        _ = await asyncio.wait_for(member.attach_started.wait(), timeout=1)
        request = member.attach_calls[0]
        request.metadata.clear()
        request.metadata["attachment.group.operation"] = "poison"
        request.metadata["attachment.group.member.index"] = -1
        _ = release.set()
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert [event.name for event in recorded] == [attachment.AttachCompleteEvent.name]
    assert recorded[0].metadata == {"traceparent": "test-trace"}


def test_group_internal_coordination_does_not_leak_into_event_metadata() -> None:
    async def run() -> tuple[dict[str, object], dict[str, object]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await bot.started(ctx, actor, actor.model)
        _ = await bot.started(ctx, member, member.model)
        _ = await bot.started(ctx, group, group.model)
        await group.attach(
            ctx,
            dataclasses.replace(
                attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)),
                metadata={"traceparent": "test-trace"},
            ),
        )
        _ = await asyncio.wait_for(actor.recorded.wait(), timeout=1)
        return member.attach_calls[0].metadata, actor.events[0].metadata

    member_metadata, terminal_metadata = asyncio.run(run())

    assert member_metadata == {"traceparent": "test-trace"}
    assert terminal_metadata == {"traceparent": "test-trace"}


def test_group_rejects_duplicate_members() -> None:
    member = TestAttachment()

    try:
        _ = attachment.Group(member, member)
    except ValueError as error:
        assert str(error) == "Attachment Group members and descendants must be unique."
    else:
        raise AssertionError("expected duplicate Group membership to fail")


def test_group_rejects_duplicate_nested_descendants() -> None:
    leaf = TestAttachment()
    nested = attachment.Group(leaf)

    try:
        _ = attachment.Group(leaf, nested)
    except ValueError as error:
        assert str(error) == "Attachment Group members and descendants must be unique."
    else:
        raise AssertionError("expected duplicate nested Group membership to fail")


def test_group_dispatches_recursively_to_all_attachments() -> None:
    async def run() -> tuple[int, int]:
        ctx = hsm.Context()
        first = BroadcastRecorder()
        second = BroadcastRecorder()
        nested = attachment.Group(second)
        group = attachment.Group(first, nested)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, second, second.model)
        _ = await bot.started(ctx, nested, nested.model)
        _ = await bot.started(ctx, group, group.model)

        await group.dispatch(ctx, BroadcastEvent)
        return first.count, second.count

    assert asyncio.run(run()) == (1, 1)


def test_group_fans_out_when_name_matches_model_but_payload_is_not_coordination() -> None:
    """Coordination vs fan-out is typed-payload, not ``event.name in model.events``.

    An event that forges a group model event name but carries a non-coordination payload
    must still fan out to members. Name-only admission would have swallowed it on the
    coordinator without rewriting to members.
    """

    class AnyEventRecorder(hsm.Instance, attachment.Attachment):
        received: list[hsm.Event[typing.Any]]
        _attachments: list[hsm.Instance]
        _attachment_timeout: datetime.timedelta
        _attachment_request_id: str

        def __init__(self) -> None:
            super().__init__()
            self._attachments = []
            self._attachment_timeout = datetime.timedelta(seconds=30)
            self._attachment_request_id = ""
            self.received = []

        @typing.override
        def attach(
            self,
            ctx: hsm.Context,
            event: hsm.Event[attachment.AttachData],
        ) -> collections.abc.Awaitable[None]:
            return hsm.dispatch(ctx, self, event)

        @typing.override
        def detach(
            self,
            ctx: hsm.Context,
            event: hsm.Event[attachment.DetachData],
        ) -> collections.abc.Awaitable[None]:
            return hsm.dispatch(ctx, self, event)

        @typing.override
        def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
            self.received.append(event)
            return super().dispatch(ctx, event)

        model: typing.ClassVar[hsm.Model] = bot.define(
            "AttachmentGroupAnyEventRecorder",
            hsm.initial(hsm.target("recording")),
            hsm.state("recording"),
        )

    async def run() -> list[str]:
        ctx = hsm.Context()
        first = AnyEventRecorder()
        group = attachment.Group(first)
        _ = await bot.started(ctx, first, first.model)
        _ = await bot.started(ctx, group, group.model)

        forged = dataclasses.replace(BroadcastEvent, name="attachment.group.member.attach.complete")
        await group.dispatch(ctx, forged)
        await asyncio.sleep(0)
        return [event.name for event in first.received]

    assert asyncio.run(run()) == ["attachment.group.member.attach.complete"]
