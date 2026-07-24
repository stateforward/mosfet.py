"""Attachable component protocol and shared relationship behavior."""

import collections.abc
import dataclasses
import datetime
import typing

import hsm

from bot import lifecycle

from . import events


@typing.runtime_checkable
class Attachment(typing.Protocol):
    """Component that accepts correlated attach and detach requests.

    Derived machines place these guards and effects in their own topology when
    attachment is one region of a larger lifecycle.
    """

    _attachment_limit: typing.ClassVar[int | None] = None
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta

    def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[events.AttachData],
    ) -> collections.abc.Awaitable[None]: ...

    def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[events.DetachData],
    ) -> collections.abc.Awaitable[None]: ...

    def _attachment_index(self, actor: hsm.Instance) -> int | None:
        actor_id = Attachment._actor_id(actor)
        for index, attached in enumerate(self._attachments):
            if attached is actor:
                return index
            attached_id = Attachment._actor_id(attached)
            if actor_id and attached_id == actor_id:
                return index
        return None

    @staticmethod
    def _actor_id(actor: hsm.Instance) -> str:
        if lifecycle.is_started(actor):
            return hsm.id(actor)
        identifier = getattr(actor, "id", None)
        return identifier if isinstance(identifier, str) else ""

    @staticmethod
    def _is_attached(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> bool:
        del ctx
        assert isinstance(instance, Attachment)
        data = event.data
        return isinstance(data, (events.AttachData, events.DetachData)) and instance._attachment_index(data.actor) is not None

    @staticmethod
    def _has_attachments(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        assert isinstance(instance, Attachment)
        return bool(instance._attachments)

    @staticmethod
    def _can_attach(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> bool:
        del ctx
        assert isinstance(instance, Attachment)
        data = event.data
        if not isinstance(data, events.AttachData) or data.actor is instance:
            return False
        if instance._attachment_index(data.actor) is not None:
            return False
        limit = instance._attachment_limit
        return limit is None or len(instance._attachments) < limit

    @staticmethod
    def _is_attach_request(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, events.AttachData)

    @staticmethod
    def _is_detach_request(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, events.DetachData)

    @staticmethod
    def _detach_would_leave_attachments(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        assert isinstance(instance, Attachment)
        data = event.data
        return (
            isinstance(data, events.DetachData)
            and instance._attachment_index(data.actor) is not None
            and len(instance._attachments) > 1
        )

    @staticmethod
    def _detach_last_attachment(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        assert isinstance(instance, Attachment)
        data = event.data
        return (
            isinstance(data, events.DetachData)
            and instance._attachment_index(data.actor) is not None
            and len(instance._attachments) == 1
        )

    @staticmethod
    def _attach(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del ctx
        assert isinstance(instance, Attachment)
        data = event.data
        assert isinstance(data, events.AttachData)
        instance._attachments.append(data.actor)

    @staticmethod
    def _remember_attachment_timeout(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        assert isinstance(instance, Attachment)
        data = event.data
        assert isinstance(data, events.AttachData)
        instance._attachment_timeout = data.timeout

    @staticmethod
    def _attachment_timeout_delay(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, event
        assert isinstance(instance, Attachment)
        return instance._attachment_timeout

    @staticmethod
    def _timeout_attachment(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        assert isinstance(instance, Attachment)
        actor = instance._attachments.pop()
        _ = hsm.dispatch(
            ctx,
            actor,
            dataclasses.replace(
                events.AttachFailedEvent.with_data(
                    events.FailedData(
                        actor=actor,
                        kind=events.FailureKind.TIMEOUT,
                        message=(
                            f"{type(instance).__name__} attachment timed out after "
                            f"{instance._attachment_timeout.total_seconds():g} seconds."
                        ),
                    )
                ),
                source=hsm.id(instance),
                target=Attachment._actor_id(actor),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _detach(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del ctx
        assert isinstance(instance, Attachment)
        data = event.data
        assert isinstance(data, events.DetachData)
        index = instance._attachment_index(data.actor)
        assert index is not None
        del instance._attachments[index]

    @staticmethod
    def _queue_attach_complete(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, events.AttachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                events.AttachCompleteEvent.with_data(
                    events.AttachCompleteData(actor=data.actor, created=True, reply_to=data.reply_to)
                ),
                id=event.id,
                source=hsm.id(instance),
                target=Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _deliver_attach_complete(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del instance
        data = event.data
        assert isinstance(data, events.AttachCompleteData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(ctx, target, event)

    @staticmethod
    def _dispatch_attach_complete_created(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, events.AttachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                events.AttachCompleteEvent.with_data(events.AttachCompleteData(actor=data.actor, created=True)),
                id=event.id,
                source=hsm.id(instance),
                target=Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_attach_complete_existing(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, events.AttachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                events.AttachCompleteEvent.with_data(events.AttachCompleteData(actor=data.actor, created=False)),
                id=event.id,
                source=hsm.id(instance),
                target=Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_attach_failed(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, events.AttachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                events.AttachFailedEvent.with_data(
                    events.FailedData(
                        actor=data.actor,
                        kind=events.FailureKind.CONFLICT,
                        message=f"{type(instance).__name__} cannot accept this attachment.",
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detach_complete_removed(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, events.DetachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                events.DetachedEvent.with_data(events.DetachedData(actor=data.actor, removed=True)),
                id=event.id,
                source=hsm.id(instance),
                target=Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detach_complete_absent(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, events.DetachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                events.DetachedEvent.with_data(events.DetachedData(actor=data.actor, removed=False)),
                id=event.id,
                source=hsm.id(instance),
                target=Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detach_failed(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, events.DetachData)
        target = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                events.DetachFailedEvent.with_data(
                    events.FailedData(
                        actor=data.actor,
                        kind=events.FailureKind.CONFLICT,
                        message=f"{type(instance).__name__} is attached to another actor.",
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=Attachment._actor_id(target),
                metadata=dict(event.metadata),
            ),
        )

__all__ = ["Attachment"]
