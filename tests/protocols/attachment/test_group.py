"""Tests for the event-driven attachment group."""

import asyncio
import typing

import hsm

from bot.protocols import attachment


BroadcastEvent = hsm.Event[None](name="attachment.group.test.broadcast")


class BroadcastRecorder(hsm.Instance):
    count: int

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

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


def test_group_implements_attachment_and_dispatchable_protocols() -> None:
    group = attachment.Group()
    attachment_contract: attachment.Attachment = group
    dispatchable_contract: hsm.Dispatchable = group

    assert isinstance(attachment_contract, attachment.Attachment)
    assert isinstance(dispatchable_contract, hsm.Dispatchable)


def test_group_is_not_an_hsm_group() -> None:
    assert not issubclass(attachment.Group, hsm.Group)


def test_group_preserves_initial_session_membership() -> None:
    async def run() -> str:
        ctx = hsm.Context()
        group = attachment.Group(hsm.Instance(), hsm.Instance())
        _ = await hsm.started(ctx, group, group.model)
        return group.state()

    assert asyncio.run(run()) == "/AttachmentGroup/attached"


def test_group_manages_attachment_lifecycle() -> None:
    class Recorder(hsm.Instance):
        events: list[hsm.Event[typing.Any]]

        def __init__(self) -> None:
            super().__init__()
            self.events = []

        @staticmethod
        def _record(ctx: hsm.Context, instance: "Recorder", event: hsm.Event[typing.Any]) -> None:
            del ctx
            instance.events.append(event)

        model: typing.ClassVar[hsm.Model] = hsm.define(
            "AttachmentGroupRecorder",
            hsm.initial(hsm.target("recording")),
            hsm.state(
                "recording",
                hsm.transition(hsm.on(attachment.AttachCompleteEvent), hsm.effect(_record)),
                hsm.transition(hsm.on(attachment.DetachedEvent), hsm.effect(_record)),
            ),
        )

    async def run() -> tuple[str, list[hsm.Event[typing.Any]]]:
        ctx = hsm.Context()
        actor = Recorder()
        group = attachment.Group()
        _ = await hsm.started(ctx, actor, actor.model)
        _ = await hsm.started(ctx, group, group.model)
        await group.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=actor)))
        await group.detach(ctx, attachment.DetachEvent.with_data(attachment.DetachData(actor=actor)))
        await asyncio.sleep(0)
        return group.state(), actor.events

    state, recorded = asyncio.run(run())

    assert state == "/AttachmentGroup/detached"
    assert [event.name for event in recorded] == [
        attachment.AttachCompleteEvent.name,
        attachment.DetachedEvent.name,
    ]


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
