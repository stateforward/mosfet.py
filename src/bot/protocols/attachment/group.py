"""Event-driven attachment group."""

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm

from . import events
from .attachment import Attachment


class Group(hsm.Instance, Attachment, hsm.Dispatchable):
    """Manage attachments and broadcast events to every attached actor."""

    _attachment_limit: typing.ClassVar[int | None] = None

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "AttachmentGroup",
        hsm.initial(hsm.target("routing_initial")),
        hsm.choice(
            "routing_initial",
            hsm.transition(hsm.guard(Attachment._has_attachments), hsm.target("/AttachmentGroup/attached")),
            hsm.transition(hsm.target("/AttachmentGroup/detached")),
        ),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(events.AttachEvent),
                hsm.guard(Attachment._can_attach),
                hsm.effect(
                    Attachment._attach,
                    Attachment._remember_attachment_timeout,
                    Attachment._queue_attach_complete,
                ),
                hsm.target("/AttachmentGroup/attaching"),
            ),
            hsm.transition(
                hsm.on(events.AttachEvent),
                hsm.guard(Attachment._is_attach_request),
                hsm.effect(Attachment._dispatch_attach_failed),
            ),
            hsm.transition(
                hsm.on(events.DetachEvent),
                hsm.guard(Attachment._is_detach_request),
                hsm.effect(Attachment._dispatch_detach_complete_absent),
            ),
        ),
        hsm.state(
            "attaching",
            hsm.transition(
                hsm.on(events.AttachCompleteEvent),
                hsm.effect(Attachment._deliver_attach_complete),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(
                hsm.after(Attachment._attachment_timeout_delay),
                hsm.effect(Attachment._timeout_attachment),
                hsm.target("/AttachmentGroup/detached"),
            ),
        ),
        hsm.state(
            "attached",
            hsm.transition(
                hsm.on(events.AttachEvent),
                hsm.guard(Attachment._is_attached),
                hsm.effect(Attachment._dispatch_attach_complete_existing),
            ),
            hsm.transition(
                hsm.on(events.AttachEvent),
                hsm.guard(Attachment._is_attach_request),
                hsm.effect(Attachment._dispatch_attach_failed),
            ),
            hsm.transition(
                hsm.on(events.DetachEvent),
                hsm.guard(Attachment._detach_would_leave_attachments),
                hsm.effect(Attachment._detach, Attachment._dispatch_detach_complete_removed),
            ),
            hsm.transition(
                hsm.on(events.DetachEvent),
                hsm.guard(Attachment._detach_last_attachment),
                hsm.effect(Attachment._detach, Attachment._dispatch_detach_complete_removed),
                hsm.target("/AttachmentGroup/detached"),
            ),
            hsm.transition(
                hsm.on(events.DetachEvent),
                hsm.guard(Attachment._is_detach_request),
                hsm.effect(Attachment._dispatch_detach_failed),
            ),
        ),
    )

    def __init__(self, *attachments: hsm.Instance) -> None:
        super().__init__()
        self._attachments: list[hsm.Instance] = list(attachments)
        self._attachment_timeout: datetime.timedelta = datetime.timedelta(seconds=30)

    @typing.override
    def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[events.AttachData],
    ) -> collections.abc.Awaitable[None]:
        return hsm.Instance.dispatch(self, ctx, event)

    @typing.override
    def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[events.DetachData],
    ) -> collections.abc.Awaitable[None]:
        return hsm.Instance.dispatch(self, ctx, event)

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name in self.model.events:
            return hsm.Instance.dispatch(self, ctx, event)

        async def dispatch_all() -> None:
            source = event.source or hsm.id(self)
            _ = await asyncio.gather(
                *(
                    hsm.dispatch(
                        ctx,
                        attachment,
                        dataclasses.replace(
                            event,
                            source=source,
                            target=hsm.id(attachment),
                            metadata=dict(event.metadata),
                        ),
                    )
                    for attachment in self._attachments
                )
            )

        return asyncio.create_task(dispatch_all())


__all__ = ["Group"]
