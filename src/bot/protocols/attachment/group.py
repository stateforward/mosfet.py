"""Event-driven attachment group."""

import asyncio
import collections.abc
import dataclasses
import datetime
import enum
import typing
import weakref

import hsm

from . import events
from .attachment import Attachment


_OPERATION_METADATA_KEY = "attachment.group.operation"
_MEMBER_INDEX_METADATA_KEY = "attachment.group.member.index"
_REQUEST_CONTEXT_METADATA_KEY = "attachment.group.request.context"
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


type _MemberResult = events.AttachCompleteData | events.DetachedData | events.FailedData


class _Modeled(typing.Protocol):
    model: hsm.Model | None


class _OperationKind(enum.StrEnum):
    ATTACH = "attach"
    DETACH = "detach"


@dataclasses.dataclass(slots=True)
class _Operation:
    coordinator: hsm.Instance
    actor: hsm.Instance
    reply_to: hsm.Instance
    request_id: str
    context: hsm.Context
    expected_members: tuple[hsm.Instance, ...]
    metadata: dict[str, object]
    timeout: datetime.timedelta
    kind: _OperationKind
    members: tuple[int, ...]
    results: dict[int, _MemberResult] = dataclasses.field(default_factory=dict)
    attempted_members: set[int] = dataclasses.field(default_factory=set)
    created_members: tuple[int, ...] = ()
    failure: events.FailedData | None = None
    fallback_attached: bool = False


def _operation(event: hsm.Event[typing.Any]) -> _Operation:
    operation = event.metadata.get(_OPERATION_METADATA_KEY)
    assert isinstance(operation, _Operation)
    return operation


def _member_index(event: hsm.Event[typing.Any]) -> int:
    index = event.metadata.get(_MEMBER_INDEX_METADATA_KEY)
    assert isinstance(index, int)
    return index


def _first_failure(operation: _Operation) -> events.FailedData | None:
    for index in sorted(operation.results):
        result = operation.results[index]
        if isinstance(result, events.FailedData):
            return result
    return None


class _Reply(hsm.Instance):
    """Accept one correlated member outcome for one aggregate operation."""

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
        index = _member_index(event)
        terminal = _MemberAttachFailedEvent if operation.kind is _OperationKind.ATTACH else _MemberDetachFailedEvent
        operation_name = "attach" if operation.kind is _OperationKind.ATTACH else "detach"
        _ = hsm.dispatch(
            ctx,
            operation.coordinator,
            dataclasses.replace(
                terminal.with_data(
                    events.FailedData(
                        actor=operation.actor,
                        kind=events.FailureKind.TIMEOUT,
                        message=(
                            f"Attachment Group member {operation_name} timed out after "
                            f"{operation.timeout.total_seconds():g} seconds."
                        ),
                    )
                ),
                id=operation.request_id,
                source=hsm.id(operation.expected_members[index]),
                target=hsm.id(operation.coordinator),
                metadata={
                    **operation.metadata,
                    _OPERATION_METADATA_KEY: operation,
                    _MEMBER_INDEX_METADATA_KEY: index,
                },
            ),
        )

    @classmethod
    def _define_model(cls, operation: _Operation, index: int, *, with_timeout: bool) -> hsm.Model:
        def is_expected(ctx: hsm.Context, instance: _Reply, event: hsm.Event[typing.Any]) -> bool:
            del ctx
            data = event.data
            return (
                event.metadata.get(_OPERATION_METADATA_KEY) is operation
                and event.metadata.get(_MEMBER_INDEX_METADATA_KEY) == index
                and event.id == operation.request_id
                and event.source == hsm.id(operation.expected_members[index])
                and event.target == hsm.id(instance)
                and isinstance(data, (events.AttachCompleteData, events.DetachedData, events.FailedData))
                and data.actor is operation.actor
            )

        if operation.kind is _OperationKind.ATTACH:
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
            ]
        else:
            waiting = [
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
                event.metadata.update(
                    {
                        _OPERATION_METADATA_KEY: operation,
                        _MEMBER_INDEX_METADATA_KEY: index,
                    }
                )
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
    async def started(
        cls,
        ctx: hsm.Context,
        operation: _Operation,
        index: int,
        *,
        with_timeout: bool,
    ) -> "_Reply":
        reply = cls()
        reply_ctx = hsm.Context(
            parent=ctx,
            values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()},
        )
        try:
            return await hsm.started(reply_ctx, reply, cls._define_model(operation, index, with_timeout=with_timeout))
        except asyncio.CancelledError:
            await hsm.stop(reply, hsm.Context())
            raise


class Group(hsm.Instance, Attachment, hsm.Dispatchable):
    """Coordinate concurrent attachment lifecycle barriers for fixed members."""

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
            if typing.cast(_Modeled, typing.cast(object, member)).model is None:
                raise TypeError("Attachment Group members must define an HSM lifecycle model.")
            visit(member, {id(self)})
        self._attachments: list[hsm.Instance] = list(attachments)
        self._attachment_timeout: datetime.timedelta = datetime.timedelta(seconds=30)

    @staticmethod
    async def _attach_members_activity(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        replies: dict[int, _Reply] = {}

        def fail_replies() -> None:
            lifetime = instance.context()
            failure = events.FailedData(
                actor=operation.actor,
                kind=events.FailureKind.DISPATCH,
                message=f"{type(instance).__name__} member attach was canceled.",
            )
            for index, reply in replies.items():
                member = instance._attachments[index]
                _ = hsm.Instance.dispatch(
                    reply,
                    lifetime,
                    dataclasses.replace(
                        events.AttachFailedEvent.with_data(failure),
                        id=operation.request_id,
                        source=hsm.id(member),
                        target=hsm.id(reply),
                        metadata={
                            **operation.metadata,
                            _OPERATION_METADATA_KEY: operation,
                            _MEMBER_INDEX_METADATA_KEY: index,
                        },
                    ),
                )

        async def attach_member(index: int) -> None:
            member = instance._attachments[index]
            assert isinstance(member, Attachment)
            reply: _Reply | None = None
            try:
                reply = await _Reply.started(operation.context, operation, index, with_timeout=True)
                replies[index] = reply
                await member.attach(
                    operation.context,
                    dataclasses.replace(
                        events.AttachEvent.with_data(
                            events.AttachData(actor=operation.actor, reply_to=reply, timeout=operation.timeout)
                        ),
                        id=operation.request_id,
                        source=hsm.id(instance),
                        target=hsm.id(member),
                        metadata={
                            **operation.metadata,
                            _OPERATION_METADATA_KEY: operation,
                            _MEMBER_INDEX_METADATA_KEY: index,
                        },
                    ),
                )
            except asyncio.CancelledError:
                if ctx.is_done() and reply is None:
                    _ = hsm.Instance.dispatch(
                        instance,
                        instance.context(),
                        dataclasses.replace(
                            _MemberAttachFailedEvent.with_data(
                                events.FailedData(
                                    actor=operation.actor,
                                    kind=events.FailureKind.DISPATCH,
                                    message=f"{type(instance).__name__} member reply start was canceled.",
                                )
                            ),
                            id=operation.request_id,
                            source=hsm.id(member),
                            target=hsm.id(instance),
                            metadata={
                                **operation.metadata,
                                _OPERATION_METADATA_KEY: operation,
                                _MEMBER_INDEX_METADATA_KEY: index,
                            },
                        ),
                    )
                    return
                raise
            except Exception as error:
                failure = events.FailedData(
                    actor=operation.actor,
                    kind=events.FailureKind.DISPATCH,
                    message=f"{type(instance).__name__} member attach failed: {error}",
                )
                if reply is not None:
                    _ = hsm.Instance.dispatch(
                        reply,
                        ctx,
                        dataclasses.replace(
                            events.AttachFailedEvent.with_data(failure),
                            id=operation.request_id,
                            source=hsm.id(member),
                            target=hsm.id(reply),
                            metadata={
                                **operation.metadata,
                                _OPERATION_METADATA_KEY: operation,
                                _MEMBER_INDEX_METADATA_KEY: index,
                            },
                        ),
                    )
                    return
                _ = hsm.Instance.dispatch(
                    instance,
                    ctx,
                    dataclasses.replace(
                        _MemberAttachFailedEvent.with_data(failure),
                        id=operation.request_id,
                        source=hsm.id(member),
                        target=hsm.id(instance),
                        metadata={
                            **operation.metadata,
                            _OPERATION_METADATA_KEY: operation,
                            _MEMBER_INDEX_METADATA_KEY: index,
                        },
                    ),
                )

        fanout = asyncio.gather(*(attach_member(index) for index in operation.members))
        context_done = asyncio.wrap_future(ctx.done())
        try:
            done, _ = await asyncio.wait((fanout, context_done), return_when=asyncio.FIRST_COMPLETED)
            if context_done in done:
                _ = fanout.cancel()
                try:
                    await fanout
                except asyncio.CancelledError:
                    pass
                fail_replies()
                return
            await fanout
            await context_done
            fail_replies()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if ctx.is_done() or task is None or task.cancelling() == 0:
                fail_replies()
            raise
        finally:
            _ = context_done.cancel()
            if not fanout.done():
                _ = fanout.cancel()
            try:
                await fanout
            except asyncio.CancelledError:
                pass

    @staticmethod
    async def _detach_members_activity(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        replies: dict[int, _Reply] = {}

        def fail_replies() -> None:
            lifetime = instance.context()
            failure = events.FailedData(
                actor=operation.actor,
                kind=events.FailureKind.DISPATCH,
                message=f"{type(instance).__name__} member detach was canceled.",
            )
            for index, reply in replies.items():
                member = instance._attachments[index]
                _ = hsm.Instance.dispatch(
                    reply,
                    lifetime,
                    dataclasses.replace(
                        events.DetachFailedEvent.with_data(failure),
                        id=operation.request_id,
                        source=hsm.id(member),
                        target=hsm.id(reply),
                        metadata={
                            **operation.metadata,
                            _OPERATION_METADATA_KEY: operation,
                            _MEMBER_INDEX_METADATA_KEY: index,
                        },
                    ),
                )

        async def detach_member(index: int) -> None:
            member = instance._attachments[index]
            assert isinstance(member, Attachment)
            reply: _Reply | None = None
            try:
                reply = await _Reply.started(operation.context, operation, index, with_timeout=True)
                replies[index] = reply
                operation.attempted_members.add(index)
                await member.detach(
                    operation.context,
                    dataclasses.replace(
                        events.DetachEvent.with_data(
                            events.DetachData(
                                actor=operation.actor,
                                reply_to=reply,
                                timeout=operation.timeout,
                            )
                        ),
                        id=operation.request_id,
                        source=hsm.id(instance),
                        target=hsm.id(member),
                        metadata={
                            **operation.metadata,
                            _OPERATION_METADATA_KEY: operation,
                            _MEMBER_INDEX_METADATA_KEY: index,
                        },
                    ),
                )
            except asyncio.CancelledError:
                if ctx.is_done() and reply is None:
                    _ = hsm.Instance.dispatch(
                        instance,
                        instance.context(),
                        dataclasses.replace(
                            _MemberDetachFailedEvent.with_data(
                                events.FailedData(
                                    actor=operation.actor,
                                    kind=events.FailureKind.DISPATCH,
                                    message=f"{type(instance).__name__} member reply start was canceled.",
                                )
                            ),
                            id=operation.request_id,
                            source=hsm.id(member),
                            target=hsm.id(instance),
                            metadata={
                                **operation.metadata,
                                _OPERATION_METADATA_KEY: operation,
                                _MEMBER_INDEX_METADATA_KEY: index,
                            },
                        ),
                    )
                    return
                raise
            except Exception as error:
                failure = events.FailedData(
                    actor=operation.actor,
                    kind=events.FailureKind.DISPATCH,
                    message=f"{type(instance).__name__} member detach failed: {error}",
                )
                if reply is not None:
                    _ = hsm.Instance.dispatch(
                        reply,
                        ctx,
                        dataclasses.replace(
                            events.DetachFailedEvent.with_data(failure),
                            id=operation.request_id,
                            source=hsm.id(member),
                            target=hsm.id(reply),
                            metadata={
                                **operation.metadata,
                                _OPERATION_METADATA_KEY: operation,
                                _MEMBER_INDEX_METADATA_KEY: index,
                            },
                        ),
                    )
                    return
                _ = hsm.Instance.dispatch(
                    instance,
                    ctx,
                    dataclasses.replace(
                        _MemberDetachFailedEvent.with_data(failure),
                        id=operation.request_id,
                        source=hsm.id(member),
                        target=hsm.id(instance),
                        metadata={
                            **operation.metadata,
                            _OPERATION_METADATA_KEY: operation,
                            _MEMBER_INDEX_METADATA_KEY: index,
                        },
                    ),
                )

        fanout = asyncio.gather(*(detach_member(index) for index in operation.members))
        context_done = asyncio.wrap_future(ctx.done())
        try:
            done, _ = await asyncio.wait((fanout, context_done), return_when=asyncio.FIRST_COMPLETED)
            if context_done in done:
                _ = fanout.cancel()
                try:
                    await fanout
                except asyncio.CancelledError:
                    pass
                fail_replies()
                return
            await fanout
            await context_done
            fail_replies()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if ctx.is_done() or task is None or task.cancelling() == 0:
                fail_replies()
            raise
        finally:
            _ = context_done.cancel()
            if not fanout.done():
                _ = fanout.cancel()
            try:
                await fanout
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _has_members(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._attachments)

    @staticmethod
    def _is_correlated(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        operation = event.metadata.get(_OPERATION_METADATA_KEY)
        index = event.metadata.get(_MEMBER_INDEX_METADATA_KEY)
        data = event.data
        return (
            isinstance(operation, _Operation)
            and isinstance(index, int)
            and index in operation.members
            and index not in operation.results
            and 0 <= index < len(instance._attachments)
            and event.id == operation.request_id
            and event.source == hsm.id(operation.expected_members[index])
            and event.target == hsm.id(instance)
            and isinstance(data, (events.AttachCompleteData, events.DetachedData, events.FailedData))
            and data.actor is operation.actor
        )

    @staticmethod
    def _is_last_result(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        operation = event.metadata.get(_OPERATION_METADATA_KEY)
        return (
            Group._is_correlated(ctx, instance, event)
            and isinstance(operation, _Operation)
            and len(operation.results) + 1 == len(operation.members)
        )

    @staticmethod
    def _operation_failed(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return _operation(event).failure is not None

    @staticmethod
    def _attach_needs_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        operation = _operation(event)
        return operation.failure is not None and bool(operation.created_members)

    @staticmethod
    def _fallback_is_attached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return _operation(event).fallback_attached

    @staticmethod
    def _begin_attach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, events.AttachData)
        request_context = event.metadata.pop(_REQUEST_CONTEXT_METADATA_KEY)
        assert isinstance(request_context, hsm.Context)
        reply_to = data.actor if data.reply_to is None else data.reply_to
        event.metadata[_OPERATION_METADATA_KEY] = _Operation(
            coordinator=instance,
            actor=data.actor,
            reply_to=reply_to,
            request_id=event.id,
            context=request_context,
            expected_members=tuple(instance._attachments),
            metadata=dict(event.metadata),
            timeout=data.timeout,
            kind=_OperationKind.ATTACH,
            members=tuple(range(len(instance._attachments))),
            fallback_attached=instance.state() == "/AttachmentGroup/attached",
        )

    @staticmethod
    def _begin_detach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, events.DetachData)
        request_context = event.metadata.pop(_REQUEST_CONTEXT_METADATA_KEY)
        assert isinstance(request_context, hsm.Context)
        reply_to = data.actor if data.reply_to is None else data.reply_to
        event.metadata[_OPERATION_METADATA_KEY] = _Operation(
            coordinator=instance,
            actor=data.actor,
            reply_to=reply_to,
            request_id=event.id,
            context=request_context,
            expected_members=tuple(instance._attachments),
            metadata=dict(event.metadata),
            timeout=data.timeout,
            kind=_OperationKind.DETACH,
            members=tuple(range(len(instance._attachments))),
            fallback_attached=True,
        )

    @staticmethod
    def _record_result(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = event.data
        assert isinstance(data, (events.AttachCompleteData, events.DetachedData, events.FailedData))
        _operation(event).results[_member_index(event)] = data

    @staticmethod
    def _finalize_attach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        operation = _operation(event)
        operation.failure = _first_failure(operation)
        created_members: list[int] = []
        for index in operation.members:
            result = operation.results[index]
            if isinstance(result, events.AttachCompleteData) and result.created:
                created_members.append(index)
        operation.created_members = tuple(created_members)

    @staticmethod
    def _begin_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        operation = _operation(event)
        operation.kind = _OperationKind.DETACH
        operation.members = operation.created_members
        operation.results.clear()

    @staticmethod
    def _finalize_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        failure = _first_failure(operation)
        if failure is not None:
            operation.failure = events.FailedData(
                actor=operation.actor,
                kind=events.FailureKind.ROLLBACK,
                message=f"{type(instance).__name__} rollback failed: {failure.message}",
            )
            operation.fallback_attached = True

    @staticmethod
    def _finalize_detach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        operation = _operation(event)
        operation.failure = _first_failure(operation)
        operation.created_members = (
            tuple(index for index in operation.members if index in operation.attempted_members)
            if operation.failure is not None
            else ()
        )

    @staticmethod
    def _detach_needs_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        operation = _operation(event)
        return operation.failure is not None and bool(operation.created_members)

    @staticmethod
    def _begin_detach_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        operation.kind = _OperationKind.ATTACH
        operation.members = operation.created_members
        operation.results.clear()
        operation.timeout = instance._attachment_timeout

    @staticmethod
    def _finalize_detach_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        operation = _operation(event)
        failure = _first_failure(operation)
        if failure is not None:
            operation.failure = events.FailedData(
                actor=operation.actor,
                kind=events.FailureKind.ROLLBACK,
                message=f"{type(instance).__name__} detach recovery failed: {failure.message}",
            )

    @staticmethod
    def _dispatch_attach_complete(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        created = any(
            isinstance(result, events.AttachCompleteData) and result.created for result in operation.results.values()
        )
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
                metadata={key: value for key, value in event.metadata.items() if key != _REQUEST_CONTEXT_METADATA_KEY},
            ),
        )

    @staticmethod
    def _dispatch_attach_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        assert operation.failure is not None
        _ = hsm.dispatch(
            ctx,
            operation.reply_to,
            dataclasses.replace(
                events.AttachFailedEvent.with_data(operation.failure),
                id=operation.request_id,
                source=hsm.id(instance),
                target=hsm.id(operation.reply_to),
                metadata=dict(operation.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        removed = any(
            isinstance(result, events.DetachedData) and result.removed for result in operation.results.values()
        )
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
                metadata={key: value for key, value in event.metadata.items() if key != _REQUEST_CONTEXT_METADATA_KEY},
            ),
        )

    @staticmethod
    def _dispatch_detach_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        operation = _operation(event)
        assert operation.failure is not None
        _ = hsm.dispatch(
            ctx,
            operation.reply_to,
            dataclasses.replace(
                events.DetachFailedEvent.with_data(operation.failure),
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
            hsm.activity(_attach_members_activity.__get__(None, object)),
            hsm.transition(
                hsm.on(_MemberAttachCompleteEvent, _MemberAttachFailedEvent),
                hsm.guard(_is_last_result),
                hsm.effect(_record_result, _finalize_attach),
                hsm.target("/AttachmentGroup/routing_attach_results"),
            ),
            hsm.transition(
                hsm.on(_MemberAttachCompleteEvent, _MemberAttachFailedEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_record_result),
            ),
        ),
        hsm.choice(
            "routing_attach_results",
            hsm.transition(
                hsm.guard(_attach_needs_rollback),
                hsm.effect(_begin_rollback),
                hsm.target("/AttachmentGroup/rolling_back"),
            ),
            hsm.transition(
                hsm.guard(_operation_failed),
                hsm.target("/AttachmentGroup/routing_attach_failure"),
            ),
            hsm.transition(
                hsm.effect(_dispatch_attach_complete),
                hsm.target("/AttachmentGroup/attached"),
            ),
        ),
        hsm.state(
            "rolling_back",
            hsm.activity(_detach_members_activity.__get__(None, object)),
            hsm.transition(
                hsm.on(_MemberDetachedEvent, _MemberDetachFailedEvent),
                hsm.guard(_is_last_result),
                hsm.effect(_record_result, _finalize_rollback),
                hsm.target("/AttachmentGroup/routing_attach_failure"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachedEvent, _MemberDetachFailedEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_record_result),
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
            hsm.defer(events.AttachEvent),
            hsm.activity(_detach_members_activity.__get__(None, object)),
            hsm.transition(
                hsm.on(_MemberDetachedEvent, _MemberDetachFailedEvent),
                hsm.guard(_is_last_result),
                hsm.effect(_record_result, _finalize_detach),
                hsm.target("/AttachmentGroup/routing_detach_results"),
            ),
            hsm.transition(
                hsm.on(_MemberDetachedEvent, _MemberDetachFailedEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_record_result),
            ),
        ),
        hsm.choice(
            "routing_detach_results",
            hsm.transition(
                hsm.guard(_detach_needs_rollback),
                hsm.effect(_begin_detach_rollback),
                hsm.target("/AttachmentGroup/rolling_forward"),
            ),
            hsm.transition(
                hsm.guard(_operation_failed),
                hsm.effect(_dispatch_detach_failure),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(
                hsm.effect(_dispatch_detached),
                hsm.target("/AttachmentGroup/detached"),
            ),
        ),
        hsm.state(
            "rolling_forward",
            hsm.defer(events.AttachEvent),
            hsm.activity(_attach_members_activity.__get__(None, object)),
            hsm.transition(
                hsm.on(_MemberAttachCompleteEvent, _MemberAttachFailedEvent),
                hsm.guard(_is_last_result),
                hsm.effect(_record_result, _finalize_detach_rollback, _dispatch_detach_failure),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(
                hsm.on(_MemberAttachCompleteEvent, _MemberAttachFailedEvent),
                hsm.guard(_is_correlated),
                hsm.effect(_record_result),
            ),
        ),
    )

    @typing.override
    def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[events.AttachData],
    ) -> collections.abc.Awaitable[None]:
        async def start_members_and_dispatch() -> None:
            for member in self._attachments:
                model = typing.cast(_Modeled, typing.cast(object, member)).model
                try:
                    _ = hsm.id(member)
                    continue
                except hsm.ErrorValidatingModel:
                    pass
                assert model is not None
                try:
                    _ = await hsm.started(ctx, member, model)
                except Exception as error:
                    data = event.data
                    assert isinstance(data, events.AttachData)
                    reply_to = data.actor if data.reply_to is None else data.reply_to
                    await hsm.dispatch(
                        ctx,
                        reply_to,
                        dataclasses.replace(
                            events.AttachFailedEvent.with_data(
                                events.FailedData(
                                    actor=data.actor,
                                    kind=events.FailureKind.INITIALIZATION,
                                    message=f"Attachment Group member start failed: {error}",
                                )
                            ),
                            id=event.id,
                            source=hsm.id(self),
                            target=hsm.id(reply_to),
                            metadata=dict(event.metadata),
                        ),
                    )
                    return
            await hsm.Instance.dispatch(
                self,
                ctx,
                dataclasses.replace(
                    event,
                    metadata={**event.metadata, _REQUEST_CONTEXT_METADATA_KEY: ctx},
                ),
            )

        task = asyncio.Task(
            start_members_and_dispatch(),
            loop=asyncio.get_running_loop(),
            eager_start=True,
        )
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    @typing.override
    def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[events.DetachData],
    ) -> collections.abc.Awaitable[None]:
        return hsm.Instance.dispatch(
            self,
            ctx,
            dataclasses.replace(
                event,
                metadata={**event.metadata, _REQUEST_CONTEXT_METADATA_KEY: ctx},
            ),
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        if event.name == events.AttachEvent.name and isinstance(event.data, events.AttachData):
            return self.attach(ctx, event)
        if event.name == events.DetachEvent.name and isinstance(event.data, events.DetachData):
            return self.detach(ctx, event)
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
