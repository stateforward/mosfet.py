"""Tests for the event-driven attachment group."""

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm

from bot.protocols import attachment


BroadcastEvent = hsm.Event[None](name="attachment.group.test.broadcast")


class BroadcastRecorder(hsm.Instance, attachment.Attachment):
    count: int
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta

    def __init__(self) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
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

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "AttachmentGroupBroadcastRecorder",
        hsm.initial(hsm.target("recording")),
        hsm.state(
            "recording",
            hsm.transition(hsm.on(BroadcastEvent), hsm.effect(_record)),
        ),
    )


class LifecycleRecorder(hsm.Instance):
    events: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    @staticmethod
    def _record(ctx: hsm.Context, instance: "LifecycleRecorder", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append(event)

    model: typing.ClassVar[hsm.Model] = hsm.define(
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
    attach_calls: list[hsm.Event[attachment.AttachData]]
    detach_calls: list[hsm.Event[attachment.DetachData]]
    fail_attach: bool
    fail_detach: bool
    raise_attach: bool
    raise_detach: bool
    respond_attach: bool
    created: bool

    model: typing.ClassVar[hsm.Model] = hsm.define(
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
        created: bool = True,
    ) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self.attach_calls = []
        self.detach_calls = []
        self.fail_attach = fail_attach
        self.fail_detach = fail_detach
        self.raise_attach = raise_attach
        self.raise_detach = raise_detach
        self.respond_attach = respond_attach
        self.created = created

    @typing.override
    def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.AttachData],
    ) -> collections.abc.Awaitable[None]:
        self.attach_calls.append(event)
        if self.raise_attach:
            raise RuntimeError("test attach exception")
        if not self.respond_attach:
            return asyncio.create_task(asyncio.sleep(0))
        data = event.data
        assert isinstance(data, attachment.AttachData)
        target = data.actor if data.reply_to is None else data.reply_to
        if self.fail_attach:
            result = attachment.AttachFailedEvent.with_data(
                attachment.FailedData(
                    actor=data.actor,
                    kind=attachment.FailureKind.INITIALIZATION,
                    message="test attachment failed",
                )
            )
        else:
            result = attachment.AttachCompleteEvent.with_data(
                attachment.AttachCompleteData(actor=data.actor, created=self.created, reply_to=data.reply_to)
            )
        return hsm.dispatch(
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

    @typing.override
    def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.DetachData],
    ) -> collections.abc.Awaitable[None]:
        self.detach_calls.append(event)
        if self.raise_detach:
            raise RuntimeError("test detach exception")
        data = event.data
        assert isinstance(data, attachment.DetachData)
        target = data.actor if data.reply_to is None else data.reply_to
        if self.fail_detach:
            result = attachment.DetachFailedEvent.with_data(
                attachment.FailedData(
                    actor=data.actor,
                    kind=attachment.FailureKind.ROLLBACK,
                    message="test detachment failed",
                )
            )
        else:
            result = attachment.DetachedEvent.with_data(attachment.DetachedData(actor=data.actor, removed=True))
        return hsm.dispatch(
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
        _ = await hsm.started(ctx, group, group.model)
        return group.state()

    assert asyncio.run(run()) == "/AttachmentGroup/detached"


def test_group_manages_attachment_lifecycle() -> None:
    async def run() -> tuple[str, LifecycleRecorder, tuple[TestAttachment, TestAttachment]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        members = (TestAttachment(), TestAttachment())
        group = attachment.Group(*members)
        _ = await hsm.started(ctx, actor, actor.model)
        for member in members:
            _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=actor, timeout=datetime.timedelta(seconds=7)),
                "group-attach",
            ),
        )
        await asyncio.sleep(0.01)
        await group.detach(
            ctx,
            attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=actor), "group-detach"),
        )
        await asyncio.sleep(0.01)
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


def test_group_rolls_back_completed_members_when_attach_fails() -> None:
    async def run() -> tuple[str, LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first = TestAttachment()
        failing = TestAttachment(fail_attach=True)
        group = attachment.Group(first, failing)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, first, first.model)
        _ = await hsm.started(ctx, failing, failing.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "group-failure"),
        )
        await asyncio.sleep(0.01)
        return group.state(), actor, first, failing

    state, actor, first, failing = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in actor.events] == [attachment.AttachFailedEvent.name]
    assert actor.events[0].id == "group-failure"
    assert len(first.attach_calls) == 1
    assert len(first.detach_calls) == 1
    assert len(failing.attach_calls) == 1


def test_group_reports_first_member_attach_failure_without_rollback() -> None:
    async def run() -> tuple[str, LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        failing = TestAttachment(fail_attach=True)
        unattempted = TestAttachment()
        group = attachment.Group(failing, unattempted)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, failing, failing.model)
        _ = await hsm.started(ctx, unattempted, unattempted.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)
        return group.state(), actor, failing, unattempted

    state, actor, failing, unattempted = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in actor.events] == [attachment.AttachFailedEvent.name]
    assert len(failing.attach_calls) == 1
    assert failing.detach_calls == []
    assert unattempted.attach_calls == []


def test_group_does_not_roll_back_preexisting_members_after_reattach_failure() -> None:
    async def run() -> tuple[str, LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        existing = TestAttachment()
        failing = TestAttachment()
        group = attachment.Group(existing, failing)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, existing, existing.model)
        _ = await hsm.started(ctx, failing, failing.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)

        existing.created = False
        failing.fail_attach = True
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)
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
        _ = await hsm.started(ctx, actor, actor.model)
        for leaf in leaves:
            _ = await hsm.started(ctx, leaf, leaf.model)
        _ = await hsm.started(ctx, nested, nested.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        await asyncio.sleep(0.01)
        return actor.events, leaves

    recorded, leaves = asyncio.run(run())

    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]
    assert [len(member.attach_calls) for member in leaves] == [1, 1]
    assert [len(member.detach_calls) for member in leaves] == [1, 1]


def test_group_attempts_every_detach_before_reporting_failure() -> None:
    async def run() -> tuple[LifecycleRecorder, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first = TestAttachment()
        failing = TestAttachment(fail_detach=True)
        group = attachment.Group(first, failing)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, first, first.model)
        _ = await hsm.started(ctx, failing, failing.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        await asyncio.sleep(0.01)
        return actor, first, failing

    actor, first, failing = asyncio.run(run())

    assert [event.name for event in actor.events] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
    ]
    assert len(first.detach_calls) == 1
    assert len(failing.detach_calls) == 1


def test_group_attempts_every_rollback_after_detach_failure() -> None:
    async def run() -> tuple[str, str, LifecycleRecorder, TestAttachment, TestAttachment, TestAttachment]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        first = TestAttachment()
        rollback_failing = TestAttachment(fail_detach=True)
        attach_failing = TestAttachment(fail_attach=True)
        members = (first, rollback_failing, attach_failing)
        group = attachment.Group(*members)
        _ = await hsm.started(ctx, actor, actor.model)
        for member in members:
            _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.02)
        failed_state = group.state()
        rollback_failing.fail_detach = False
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        await asyncio.sleep(0.02)
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
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(raise_attach=True)
        group = attachment.Group(member)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.DISPATCH


def test_group_converts_member_detach_exception_to_failure() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment(raise_detach=True)
        group = attachment.Group(member)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        await asyncio.sleep(0.01)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachFailedEvent.name,
    ]
    assert isinstance(recorded[1].data, attachment.FailedData)
    assert recorded[1].data.kind is attachment.FailureKind.DISPATCH


def test_group_converts_rollback_detach_exception_to_failure() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        rollback_failing = TestAttachment(raise_detach=True)
        attach_failing = TestAttachment(fail_attach=True)
        group = attachment.Group(rollback_failing, attach_failing)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, rollback_failing, rollback_failing.model)
        _ = await hsm.started(ctx, attach_failing, attach_failing.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0.01)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
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
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, foreign, foreign.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await asyncio.sleep(0)
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
        await asyncio.sleep(0)
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
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "attach"))
        await asyncio.sleep(0)
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
        await asyncio.sleep(0)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/attached"
    assert [event.id for event in recorded] == ["attach"]


def test_group_ignores_stale_public_terminal_event_during_new_operation() -> None:
    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = LifecycleRecorder()
        member = TestAttachment()
        group = attachment.Group(member)
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "first-attach"),
        )
        await asyncio.sleep(0.01)

        member.respond_attach = False
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=actor), "first-attach"),
        )
        await asyncio.sleep(0.01)
        first_request = member.attach_calls[0]
        assert isinstance(first_request.data, attachment.AttachData)
        second_request = member.attach_calls[1]
        assert isinstance(second_request.data, attachment.AttachData)
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
        await asyncio.sleep(0)
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
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, member, member.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(
            ctx,
            attachment.AttachEvent.with_data(
                attachment.AttachData(actor=actor, timeout=datetime.timedelta(milliseconds=1))
            ),
        )
        await asyncio.sleep(0.02)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [attachment.AttachFailedEvent.name]
    assert isinstance(recorded[0].data, attachment.FailedData)
    assert recorded[0].data.kind is attachment.FailureKind.TIMEOUT


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
        _ = await hsm.started(ctx, first, first.model)
        _ = await hsm.started(ctx, second, second.model)
        _ = await hsm.started(ctx, nested, nested.model)
        _ = await hsm.started(ctx, group, group.model)

        await group.dispatch(ctx, BroadcastEvent)
        return first.count, second.count

    assert asyncio.run(run()) == (1, 1)
