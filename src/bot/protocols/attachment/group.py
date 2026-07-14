"""Event-driven attachment group."""

import asyncio
import collections.abc
import dataclasses
import datetime
import typing

import hsm

from . import events
from .attachment import Attachment


_OPERATION_METADATA_KEY = "attachment.group.operation"
_MemberAttachCompleteEvent = hsm.Event[events.AttachCompleteData](
    name="attachment.group.member.attach.complete",
    kind=hsm.CompletionEventKind,
    schema=events.AttachCompleteData,
)
_MemberAttachFailedEvent = hsm.Event[events.FailedData](
    name="attachment.group.member.attach.failed",
    kind=hsm.ErrorEventKind,
    schema=events.FailedData,
)
_MemberDetachedEvent = hsm.Event[events.DetachedData](
    name="attachment.group.member.detached",
    kind=hsm.CompletionEventKind,
    schema=events.DetachedData,
)
_MemberDetachFailedEvent = hsm.Event[events.FailedData](
    name="attachment.group.member.detach.failed",
    kind=hsm.ErrorEventKind,
    schema=events.FailedData,
)
_FORWARDED_EVENTS = {
    events.AttachCompleteEvent.name: _MemberAttachCompleteEvent,
    events.AttachFailedEvent.name: _MemberAttachFailedEvent,
    events.DetachedEvent.name: _MemberDetachedEvent,
    events.DetachFailedEvent.name: _MemberDetachFailedEvent,
}


@dataclasses.dataclass(frozen=True, slots=True)
class _Operation:
    coordinator: hsm.Instance
    actor: hsm.Instance
    reply_to: hsm.Instance
    request_id: str
    expected_source: str
    metadata: dict[str, object]
    timeout: datetime.timedelta
    current: int
    remaining: tuple[int, ...]
    created_members: tuple[int, ...] = ()
    changed: bool = False
    failure: events.FailedData | None = None
    fallback_attached: bool = False


def _operation(event: hsm.Event[typing.Any]) -> _Operation:
    operation = event.metadata.get(_OPERATION_METADATA_KEY)
    assert isinstance(operation, _Operation)
    return operation


class _Reply(hsm.Instance):
    """One-shot boundary that converts public member outcomes into private Group events."""

    @staticmethod
    def _forward(ctx: hsm.Context, instance: "_Reply", event: hsm.Event[typing.Any]) -> None:
        del instance
        operation = _operation(event)
        forwarded = _FORWARDED_EVENTS[event.name]
        _ = hsm.dispatch(
            ctx,
            operation.coordinator,
            dataclasses.replace(
                forwarded.with_data(event.data),
                id=event.id,
                source=event.source,
                target=hsm.id(operation.coordinator),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _timeout(ctx: hsm.Context, instance: "_Reply", event: hsm.Event[typing.Any]) -> None:
        del instance
        operation = _operation(event)
        _ = hsm.dispatch(
            ctx,
            operation.coordinator,
            dataclasses.replace(
                _MemberAttachFailedEvent.with_data(
                    events.FailedData(
                        actor=operation.actor,
                        kind=events.FailureKind.TIMEOUT,
                        message=(
                            f"Attachment Group member attach timed out after "
                            f"{operation.timeout.total_seconds():g} seconds."
                        ),
                    )
                ),
                id=operation.request_id,
                source=operation.expected_source,
                target=hsm.id(operation.coordinator),
                metadata={**operation.metadata, _OPERATION_METADATA_KEY: operation},
            ),
        )

    @classmethod
    def _define_model(cls, operation: _Operation, *, with_timeout: bool) -> hsm.Model:
        def is_expected(ctx: hsm.Context, instance: _Reply, event: hsm.Event[typing.Any]) -> bool:
            del ctx
            return (
                event.metadata.get(_OPERATION_METADATA_KEY) is operation
                and event.id == operation.request_id
                and event.source == operation.expected_source
                and event.target == hsm.id(instance)
            )

        waiting: list[hsm.Element] = [
            hsm.transition(
                hsm.on(events.AttachCompleteEvent),
                hsm.guard(is_expected),
                hsm.effect(cls._forward),
                hsm.target("/AttachmentGroupReply/done"),
            ),
            hsm.transition(
                hsm.on(events.AttachFailedEvent),
                hsm.guard(is_expected),
                hsm.effect(cls._forward),
                hsm.target("/AttachmentGroupReply/done"),
            ),
            hsm.transition(
                hsm.on(events.DetachedEvent),
                hsm.guard(is_expected),
                hsm.effect(cls._forward),
                hsm.target("/AttachmentGroupReply/done"),
            ),
            hsm.transition(
                hsm.on(events.DetachFailedEvent),
                hsm.guard(is_expected),
                hsm.effect(cls._forward),
                hsm.target("/AttachmentGroupReply/done"),
            ),
        ]
        if with_timeout:

            def timeout_delay(
                ctx: hsm.Context,
                instance: _Reply,
                event: hsm.Event[typing.Any],
            ) -> datetime.timedelta:
                del ctx, instance
                event.metadata[_OPERATION_METADATA_KEY] = operation
                return operation.timeout

            waiting.append(
                hsm.transition(
                    hsm.after(timeout_delay),
                    hsm.effect(cls._timeout),
                    hsm.target("/AttachmentGroupReply/done"),
                )
            )
        return hsm.define(
            "AttachmentGroupReply",
            hsm.initial(hsm.target("waiting")),
            hsm.state("waiting", *waiting),
            hsm.final("done"),
        )

    @classmethod
    async def started(cls, ctx: hsm.Context, operation: _Operation, *, with_timeout: bool) -> "_Reply":
        reply = cls()
        return await hsm.started(ctx, reply, cls._define_model(operation, with_timeout=with_timeout))


class Group(hsm.Instance, Attachment, hsm.Dispatchable):
    """Coordinate attachment lifecycle and event dispatch for fixed members."""

    _attachment_limit: typing.ClassVar[int | None] = None

    def __init__(self, *attachments: hsm.Instance) -> None:
        super().__init__()
        descendants: set[int] = set()

        def visit(member: hsm.Instance, path: set[int]) -> None:
            identifier = id(member)
            if identifier in path:
                raise ValueError("Attachment Group membership cannot contain cycles.")
            if identifier in descendants:
                raise ValueError("Attachment Group members and descendants must be unique.")
            descendants.add(identifier)
            if isinstance(member, Group):
                nested_path = {*path, identifier}
                for child in member._attachments:
                    visit(child, nested_path)

        for member in attachments:
            if member is self or not isinstance(member, Attachment):
                raise TypeError("Attachment Group members must implement attachment.Attachment.")
            visit(member, {id(self)})
        self._attachments: list[hsm.Instance] = list(attachments)
        self._attachment_timeout: datetime.timedelta = datetime.timedelta(seconds=30)

    @staticmethod
    async def _attach_member_activity(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        member = instance._attachments[operation.current]
        assert isinstance(member, Attachment)
        try:
            reply = await _Reply.started(instance.context(), operation, with_timeout=True)
            await member.attach(
                ctx,
                dataclasses.replace(
                    events.AttachEvent.with_data(
                        events.AttachData(actor=operation.actor, reply_to=reply, timeout=operation.timeout)
                    ),
                    id=operation.request_id,
                    source=hsm.id(instance),
                    target=hsm.id(member),
                    metadata={**operation.metadata, _OPERATION_METADATA_KEY: operation},
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _ = hsm.Instance.dispatch(
                instance,
                ctx,
                dataclasses.replace(
                    _MemberAttachFailedEvent.with_data(
                        events.FailedData(
                            actor=operation.actor,
                            kind=events.FailureKind.DISPATCH,
                            message=f"{type(instance).__name__} member attach failed: {error}",
                        )
                    ),
                    id=operation.request_id,
                    source=hsm.id(member),
                    target=hsm.id(instance),
                    metadata={**operation.metadata, _OPERATION_METADATA_KEY: operation},
                ),
            )

    @staticmethod
    async def _detach_member_activity(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        member = instance._attachments[operation.current]
        assert isinstance(member, Attachment)
        try:
            reply = await _Reply.started(instance.context(), operation, with_timeout=False)
            await member.detach(
                ctx,
                dataclasses.replace(
                    events.DetachEvent.with_data(events.DetachData(actor=operation.actor, reply_to=reply)),
                    id=operation.request_id,
                    source=hsm.id(instance),
                    target=hsm.id(member),
                    metadata={**operation.metadata, _OPERATION_METADATA_KEY: operation},
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _ = hsm.Instance.dispatch(
                instance,
                ctx,
                dataclasses.replace(
                    _MemberDetachFailedEvent.with_data(
                        events.FailedData(
                            actor=operation.actor,
                            kind=events.FailureKind.DISPATCH,
                            message=f"{type(instance).__name__} member detach failed: {error}",
                        )
                    ),
                    id=operation.request_id,
                    source=hsm.id(member),
                    target=hsm.id(instance),
                    metadata={**operation.metadata, _OPERATION_METADATA_KEY: operation},
                ),
            )

    @staticmethod
    def _has_members(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._attachments)

    @staticmethod
    def _is_correlated(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        operation = event.metadata.get(_OPERATION_METADATA_KEY)
        if not isinstance(operation, _Operation):
            return False
        if operation.current < 0 or operation.current >= len(instance._attachments):
            return False
        member = instance._attachments[operation.current]
        return event.id == operation.request_id and event.source == hsm.id(member) and event.target == hsm.id(instance)

    @staticmethod
    def _is_correlated_with_next(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        return Group._is_correlated(ctx, instance, event) and bool(_operation(event).remaining)

    @staticmethod
    def _is_correlated_with_completed(
        ctx: hsm.Context,
        instance: "Group",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Group._is_correlated(ctx, instance, event) and bool(_operation(event).created_members)

    @staticmethod
    def _fallback_is_attached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return _operation(event).fallback_attached

    @staticmethod
    def _is_correlated_detach_success(
        ctx: hsm.Context,
        instance: "Group",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Group._is_correlated(ctx, instance, event) and _operation(event).failure is None

    @staticmethod
    def _begin_attach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, events.AttachData)
        reply_to = data.actor if data.reply_to is None else data.reply_to
        operation = _Operation(
            coordinator=instance,
            actor=data.actor,
            reply_to=reply_to,
            request_id=event.id,
            expected_source=hsm.id(instance._attachments[0]),
            metadata=dict(event.metadata),
            timeout=data.timeout,
            current=0,
            remaining=tuple(range(1, len(instance._attachments))),
            fallback_attached=instance.state() == "/AttachmentGroup/attached",
        )
        event.metadata[_OPERATION_METADATA_KEY] = operation

    @staticmethod
    def _attach_next(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        data = event.data
        assert isinstance(data, events.AttachCompleteData)
        next_index = operation.remaining[0]
        operation = dataclasses.replace(
            operation,
            current=next_index,
            remaining=operation.remaining[1:],
            expected_source=hsm.id(instance._attachments[next_index]),
            created_members=(
                (*operation.created_members, operation.current) if data.created else operation.created_members
            ),
            changed=operation.changed or data.created,
        )
        event.metadata[_OPERATION_METADATA_KEY] = operation

    @staticmethod
    def _begin_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        data = event.data
        assert isinstance(data, events.FailedData)
        completed = operation.created_members
        current = completed[-1]
        operation = dataclasses.replace(
            operation,
            current=current,
            remaining=completed[:-1],
            expected_source=hsm.id(instance._attachments[current]),
            failure=data,
        )
        event.metadata[_OPERATION_METADATA_KEY] = operation

    @staticmethod
    def _rollback_next(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        current = operation.remaining[-1]
        operation = dataclasses.replace(
            operation,
            current=current,
            remaining=operation.remaining[:-1],
            expected_source=hsm.id(instance._attachments[current]),
        )
        event.metadata[_OPERATION_METADATA_KEY] = operation

    @staticmethod
    def _record_rollback_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        data = event.data
        assert isinstance(data, events.FailedData)
        event.metadata[_OPERATION_METADATA_KEY] = dataclasses.replace(
            operation,
            failure=events.FailedData(
                actor=operation.actor,
                kind=events.FailureKind.ROLLBACK,
                message=f"{type(instance).__name__} rollback failed: {data.message}",
            ),
            fallback_attached=True,
        )

    @staticmethod
    def _begin_detach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, events.DetachData)
        reply_to = data.actor if data.reply_to is None else data.reply_to
        current = len(instance._attachments) - 1
        operation = _Operation(
            coordinator=instance,
            actor=data.actor,
            reply_to=reply_to,
            request_id=event.id,
            expected_source=hsm.id(instance._attachments[current]),
            metadata=dict(event.metadata),
            timeout=instance._attachment_timeout,
            current=current,
            remaining=tuple(range(current)),
            fallback_attached=True,
        )
        event.metadata[_OPERATION_METADATA_KEY] = operation

    @staticmethod
    def _detach_next(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        data = event.data
        failure = operation.failure
        changed = operation.changed
        if isinstance(data, events.DetachedData):
            changed = changed or data.removed
        elif isinstance(data, events.FailedData) and failure is None:
            failure = data
        current = operation.remaining[-1]
        operation = dataclasses.replace(
            operation,
            current=current,
            remaining=operation.remaining[:-1],
            expected_source=hsm.id(instance._attachments[current]),
            changed=changed,
            failure=failure,
        )
        event.metadata[_OPERATION_METADATA_KEY] = operation

    @staticmethod
    def _remember_detach_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        operation = _operation(event)
        data = event.data
        assert isinstance(data, events.FailedData)
        if operation.failure is None:
            event.metadata[_OPERATION_METADATA_KEY] = dataclasses.replace(operation, failure=data)

    @staticmethod
    def _dispatch_attach_complete(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        data = event.data
        created = operation.changed
        if isinstance(data, events.AttachCompleteData):
            created = created or data.created
        _ = hsm.dispatch(
            ctx,
            operation.reply_to,
            dataclasses.replace(
                events.AttachCompleteEvent.with_data(events.AttachCompleteData(actor=operation.actor, created=created)),
                id=operation.request_id,
                source=hsm.id(instance),
                target=hsm.id(operation.reply_to),
                metadata=dict(operation.metadata),
            ),
        )

    @staticmethod
    def _dispatch_empty_attach_complete(
        ctx: hsm.Context,
        instance: "Group",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, events.AttachData)
        reply_to = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            reply_to,
            dataclasses.replace(
                events.AttachCompleteEvent.with_data(events.AttachCompleteData(actor=data.actor, created=True)),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_attach_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        data = operation.failure if operation.failure is not None else event.data
        assert isinstance(data, events.FailedData)
        _ = hsm.dispatch(
            ctx,
            operation.reply_to,
            dataclasses.replace(
                events.AttachFailedEvent.with_data(data),
                id=operation.request_id,
                source=hsm.id(instance),
                target=hsm.id(operation.reply_to),
                metadata=dict(operation.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        data = event.data
        removed = operation.changed
        if isinstance(data, events.DetachedData):
            removed = removed or data.removed
        _ = hsm.dispatch(
            ctx,
            operation.reply_to,
            dataclasses.replace(
                events.DetachedEvent.with_data(events.DetachedData(actor=operation.actor, removed=removed)),
                id=operation.request_id,
                source=hsm.id(instance),
                target=hsm.id(operation.reply_to),
                metadata=dict(operation.metadata),
            ),
        )

    @staticmethod
    def _dispatch_empty_detached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, events.DetachData)
        reply_to = data.actor if data.reply_to is None else data.reply_to
        _ = hsm.dispatch(
            ctx,
            reply_to,
            dataclasses.replace(
                events.DetachedEvent.with_data(events.DetachedData(actor=data.actor, removed=True)),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detach_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        data = operation.failure if operation.failure is not None else event.data
        assert isinstance(data, events.FailedData)
        _ = hsm.dispatch(
            ctx,
            operation.reply_to,
            dataclasses.replace(
                events.DetachFailedEvent.with_data(data),
                id=operation.request_id,
                source=hsm.id(instance),
                target=hsm.id(operation.reply_to),
                metadata=dict(operation.metadata),
            ),
        )

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "AttachmentGroup",
        hsm.initial(hsm.target("detached")),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(events.AttachEvent),
                hsm.guard(_has_members),
                hsm.effect(_begin_attach),
                hsm.target("/AttachmentGroup/attaching"),
            ),
            hsm.transition(
                hsm.on(events.AttachEvent),
                hsm.effect(_dispatch_empty_attach_complete),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(hsm.on(events.DetachEvent), hsm.effect(_dispatch_empty_detached)),
        ),
        hsm.state(
            "attaching",
            hsm.activity(_attach_member_activity.__get__(None, object)),
            hsm.transition(
                hsm.on(_MemberAttachCompleteEvent),
                hsm.guard(_is_correlated_with_next),
                hsm.effect(_attach_next),
                hsm.target("/AttachmentGroup/attaching"),
            ),
            hsm.transition(
                hsm.on(_MemberAttachCompleteEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_dispatch_attach_complete),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(
                hsm.on(_MemberAttachFailedEvent),
                hsm.guard(_is_correlated_with_completed),
                hsm.effect(_begin_rollback),
                hsm.target("/AttachmentGroup/rolling_back"),
            ),
            hsm.transition(
                hsm.on(_MemberAttachFailedEvent),
                hsm.guard(_is_correlated),
                hsm.target("/AttachmentGroup/routing_attach_failure"),
            ),
        ),
        hsm.state(
            "rolling_back",
            hsm.activity(_detach_member_activity.__get__(None, object)),
            hsm.transition(
                hsm.on(_MemberDetachedEvent),
                hsm.guard(_is_correlated_with_next),
                hsm.effect(_rollback_next),
                hsm.target("/AttachmentGroup/rolling_back"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachedEvent),
                hsm.guard(_is_correlated),
                hsm.target("/AttachmentGroup/routing_attach_failure"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachFailedEvent),
                hsm.guard(_is_correlated_with_next),
                hsm.effect(_record_rollback_failure, _rollback_next),
                hsm.target("/AttachmentGroup/rolling_back"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachFailedEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_record_rollback_failure),
                hsm.target("/AttachmentGroup/routing_attach_failure"),
            ),
        ),
        hsm.choice(
            "routing_attach_failure",
            hsm.transition(
                hsm.guard(_fallback_is_attached),
                hsm.effect(_dispatch_attach_failure),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(
                hsm.effect(_dispatch_attach_failure),
                hsm.target("/AttachmentGroup/detached"),
            ),
        ),
        hsm.state(
            "attached",
            hsm.transition(
                hsm.on(events.AttachEvent),
                hsm.guard(_has_members),
                hsm.effect(_begin_attach),
                hsm.target("/AttachmentGroup/attaching"),
            ),
            hsm.transition(hsm.on(events.AttachEvent), hsm.effect(_dispatch_empty_attach_complete)),
            hsm.transition(
                hsm.on(events.DetachEvent),
                hsm.guard(_has_members),
                hsm.effect(_begin_detach),
                hsm.target("/AttachmentGroup/detaching"),
            ),
            hsm.transition(
                hsm.on(events.DetachEvent),
                hsm.effect(_dispatch_empty_detached),
                hsm.target("/AttachmentGroup/detached"),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.activity(_detach_member_activity.__get__(None, object)),
            hsm.transition(
                hsm.on(_MemberDetachedEvent),
                hsm.guard(_is_correlated_with_next),
                hsm.effect(_detach_next),
                hsm.target("/AttachmentGroup/detaching"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachedEvent),
                hsm.guard(_is_correlated_detach_success),
                hsm.effect(_dispatch_detached),
                hsm.target("/AttachmentGroup/detached"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachedEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_dispatch_detach_failure),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachFailedEvent),
                hsm.guard(_is_correlated_with_next),
                hsm.effect(_detach_next),
                hsm.target("/AttachmentGroup/detaching"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachFailedEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_remember_detach_failure, _dispatch_detach_failure),
                hsm.target("/AttachmentGroup/attached"),
            ),
        ),
    )

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
                        member,
                        dataclasses.replace(
                            event,
                            source=source,
                            target=hsm.id(member),
                            metadata=dict(event.metadata),
                        ),
                    )
                    for member in self._attachments
                )
            )

        return asyncio.create_task(dispatch_all())


__all__ = ["Group"]
