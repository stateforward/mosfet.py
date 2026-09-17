"""Tests for the shared attachment relationship behavior."""

import asyncio
import collections.abc
import datetime
import typing

import hsm
import mosfet

from mosfet.protocols import attachment


class Requester(hsm.Instance):
    """A caller that accepts only the outcome of the request it actually made.

    This is HSM-CORRELATION-001 applied literally: both outcome transitions are guarded on the
    id of the outstanding request, so an outcome that carries some other id — or none — is not
    this operation's and never settles it.
    """

    outcomes: list[hsm.Event[typing.Any]]
    settled: asyncio.Event
    request_id: str

    def __init__(self) -> None:
        super().__init__()
        self.outcomes = []
        self.settled = asyncio.Event()
        self.request_id = ""

    @staticmethod
    def _is_this_request(ctx: hsm.Context, instance: "Requester", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return bool(instance.request_id) and event.id == instance.request_id

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Requester", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.outcomes.append(event)
        instance.settled.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "AttachmentRequester",
        hsm.initial(hsm.target("waiting")),
        hsm.state(
            "waiting",
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.guard(_is_this_request),
                hsm.effect(_record),
                hsm.target("../settled"),
            ),
            hsm.transition(
                hsm.on(attachment.AttachFailedEvent),
                hsm.guard(_is_this_request),
                hsm.effect(_record),
                hsm.target("../settled"),
            ),
        ),
        hsm.state("settled"),
    )


class UnfinishedAttachment(hsm.Instance, attachment.Attachment):
    """An attachable component whose attaching work never finishes.

    Derived machines place the protocol's guards and effects in their own topology, so this one
    omits ``_queue_attach_complete`` and leaves ``attaching`` with only its modeled deadline to
    leave by. That makes the timeout the deterministic outcome rather than a race against a
    completion the component was always going to queue for itself.
    """

    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str

    def __init__(self) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""

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

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "UnfinishedAttachment",
        hsm.initial(hsm.target("detached")),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    attachment.Attachment._remember_attachment_request,
                ),
                hsm.target("../attaching"),
            ),
        ),
        hsm.state(
            "attaching",
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.effect(attachment.Attachment._deliver_attach_complete),
                hsm.target("../attached"),
            ),
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(attachment.Attachment._timeout_attachment),
                hsm.target("../detached"),
            ),
        ),
        hsm.state("attached"),
    )


def test_attach_timeout_failure_correlates_to_its_request() -> None:
    """A caller correlating attach outcomes on ``event.id`` must see the timeout.

    Success has always carried the request id, so a correlating caller settles on it. If the
    timeout failure does not carry the same id, the caller's guard rejects it and the operation
    it was waiting on never ends — a hang caused by following the correlation rule.
    """

    async def run() -> tuple[list[hsm.Event[typing.Any]], str]:
        ctx = hsm.Context()
        requester = Requester()
        component = UnfinishedAttachment()
        _ = await mosfet.started(ctx, requester, requester.model)
        _ = await mosfet.started(ctx, component, component.model)
        requester.request_id = "attachment-request-1"
        await component.attach(
            ctx,
            attachment.AttachEvent.with_data_and_id(
                attachment.AttachData(actor=requester, timeout=datetime.timedelta(milliseconds=1)),
                requester.request_id,
            ),
        )
        _ = await asyncio.wait_for(requester.settled.wait(), timeout=1)
        return requester.outcomes, requester.state()

    outcomes, state = asyncio.run(run())

    assert [outcome.name for outcome in outcomes] == [attachment.AttachFailedEvent.name]
    failure = outcomes[0]
    assert failure.id == "attachment-request-1"
    assert isinstance(failure.data, attachment.FailedData)
    assert failure.data.kind is attachment.FailureKind.TIMEOUT
    assert state == "/AttachmentRequester/settled"
