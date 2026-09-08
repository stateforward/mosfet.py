"""Event-driven attachment group."""

import asyncio
import collections.abc
import dataclasses
import datetime
import enum
import typing
import weakref

import hsm
import bot
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot import lifecycle
from bot import scope

from . import events
from .attachment import Attachment


type _MemberResult = events.AttachCompleteData | events.DetachedData | events.FailedData


class _Modeled(typing.Protocol):
    model: hsm.Model | None


class _OperationKind(enum.StrEnum):
    ATTACH = "attach"
    DETACH = "detach"


class _OperationData(pydantic.BaseModel):
    """Immutable correlation + phase descriptor for one group barrier operation.

    Frozen: never mutated in place. Phase transitions (attach -> rollback, detach ->
    recovery) produce a new snapshot via ``model_copy`` stored on the owning Group
    instance; member outcome events carry the phase-correct snapshot for correlation.
    Result accumulation (per-member outcomes, failures, created sets) lives on the
    owning Group instance (owned state, read/written only by Group's own HSM
    callbacks), never in this event-carried object.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    coordinator: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]]
    actor: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]]
    reply_to: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]]
    request_id: str
    context: SkipJsonSchema[pydantic.SkipValidation[hsm.Context]]
    expected_members: SkipJsonSchema[pydantic.SkipValidation[tuple[hsm.Instance, ...]]]
    timeout: datetime.timedelta
    kind: _OperationKind
    members: tuple[int, ...]
    fallback_attached: bool = False


class _MemberOutcomeData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    operation: SkipJsonSchema[pydantic.SkipValidation[_OperationData]]
    index: int = pydantic.Field(ge=0)


class _MemberAttachCompleteData(_MemberOutcomeData):
    result: events.AttachCompleteData


class _MemberAttachFailedData(_MemberOutcomeData):
    result: events.FailedData


class _MemberDetachedData(_MemberOutcomeData):
    result: events.DetachedData


class _MemberDetachFailedData(_MemberOutcomeData):
    result: events.FailedData


_StartAttachEvent = hsm.Event[_OperationData](name="attachment.group.attach.start", schema=_OperationData)
_StartDetachEvent = hsm.Event[_OperationData](name="attachment.group.detach.start", schema=_OperationData)
_MemberAttachCompleteEvent = hsm.Event[_MemberAttachCompleteData](
    name="attachment.group.member.attach.complete",
    kind=hsm.CompletionEventKind,
    schema=_MemberAttachCompleteData,
)
_MemberAttachFailedEvent = hsm.Event[_MemberAttachFailedData](
    name="attachment.group.member.attach.failed",
    kind=hsm.ErrorEventKind,
    schema=_MemberAttachFailedData,
)
_MemberDetachedEvent = hsm.Event[_MemberDetachedData](
    name="attachment.group.member.detached",
    kind=hsm.CompletionEventKind,
    schema=_MemberDetachedData,
)
_MemberDetachFailedEvent = hsm.Event[_MemberDetachFailedData](
    name="attachment.group.member.detach.failed",
    kind=hsm.ErrorEventKind,
    schema=_MemberDetachFailedData,
)


def _operation(event: hsm.Event[typing.Any]) -> _OperationData:
    data = event.data
    if isinstance(
        data,
        (_MemberAttachCompleteData, _MemberAttachFailedData, _MemberDetachedData, _MemberDetachFailedData),
    ):
        return data.operation
    assert isinstance(data, _OperationData)
    return data


def _first_failure(results: collections.abc.Mapping[int, _MemberResult]) -> events.FailedData | None:
    for index in sorted(results):
        result = results[index]
        if isinstance(result, events.FailedData):
            return result
    return None


class _Reply(hsm.Instance):
    """Accept one correlated member outcome for one aggregate operation."""

    @staticmethod
    def _forward(
        ctx: hsm.Context,
        instance: "_Reply",
        event: hsm.Event[typing.Any],
        operation: _OperationData,
        index: int,
        metadata: dict[str, object],
    ) -> None:
        del instance
        result = event.data
        # Topology already selected the terminal trigger; classify by payload (+ operation kind
        # for shared FailedData) — no event.name discrimination.
        if isinstance(result, events.AttachCompleteData):
            forwarded = _MemberAttachCompleteEvent.with_data(
                _MemberAttachCompleteData(operation=operation, index=index, result=result)
            )
        elif isinstance(result, events.DetachedData):
            forwarded = _MemberDetachedEvent.with_data(
                _MemberDetachedData(operation=operation, index=index, result=result)
            )
        else:
            assert isinstance(result, events.FailedData)
            if operation.kind is _OperationKind.ATTACH:
                forwarded = _MemberAttachFailedEvent.with_data(
                    _MemberAttachFailedData(operation=operation, index=index, result=result)
                )
            else:
                forwarded = _MemberDetachFailedEvent.with_data(
                    _MemberDetachFailedData(operation=operation, index=index, result=result)
                )
        _ = hsm.dispatch(
            ctx,
            operation.coordinator,
            dataclasses.replace(
                forwarded,
                id=operation.request_id,
                source=event.source,
                target=hsm.id(operation.coordinator),
                metadata=dict(metadata),
            ),
        )

    @staticmethod
    def _timeout(
        ctx: hsm.Context,
        instance: "_Reply",
        event: hsm.Event[typing.Any],
        operation: _OperationData,
        index: int,
        metadata: dict[str, object],
    ) -> None:
        del instance, event
        operation_name = "attach" if operation.kind is _OperationKind.ATTACH else "detach"
        failure = events.FailedData(
            actor=operation.actor,
            kind=events.FailureKind.TIMEOUT,
            message=(
                f"Attachment Group member {operation_name} timed out after "
                f"{operation.timeout.total_seconds():g} seconds."
            ),
        )
        if operation.kind is _OperationKind.ATTACH:
            terminal = _MemberAttachFailedEvent.with_data(
                _MemberAttachFailedData(operation=operation, index=index, result=failure)
            )
        else:
            terminal = _MemberDetachFailedEvent.with_data(
                _MemberDetachFailedData(operation=operation, index=index, result=failure)
            )
        _ = hsm.dispatch(
            ctx,
            operation.coordinator,
            dataclasses.replace(
                terminal,
                id=operation.request_id,
                source=hsm.id(operation.expected_members[index]),
                target=hsm.id(operation.coordinator),
                metadata=dict(metadata),
            ),
        )

    @classmethod
    def _define_model(
        cls,
        operation: _OperationData,
        index: int,
        metadata: dict[str, object],
        *,
        with_timeout: bool,
    ) -> hsm.Model:
        def is_expected(ctx: hsm.Context, instance: _Reply, event: hsm.Event[typing.Any]) -> bool:
            del ctx
            data = event.data
            return (
                event.id == hsm.id(instance)
                and event.source == hsm.id(operation.expected_members[index])
                and event.target == hsm.id(instance)
                and isinstance(data, (events.AttachCompleteData, events.DetachedData, events.FailedData))
                and data.actor is operation.actor
            )

        def forward(ctx: hsm.Context, instance: _Reply, event: hsm.Event[typing.Any]) -> None:
            cls._forward(ctx, instance, event, operation, index, metadata)

        def timeout(ctx: hsm.Context, instance: _Reply, event: hsm.Event[typing.Any]) -> None:
            cls._timeout(ctx, instance, event, operation, index, metadata)

        if operation.kind is _OperationKind.ATTACH:
            waiting: list[hsm.Element] = [
                hsm.transition(
                    hsm.on(events.AttachCompleteEvent),
                    hsm.guard(is_expected),
                    hsm.effect(forward),
                    hsm.target("/AttachmentGroupReply/done"),
                ),
                hsm.transition(
                    hsm.on(events.AttachFailedEvent),
                    hsm.guard(is_expected),
                    hsm.effect(forward),
                    hsm.target("/AttachmentGroupReply/done"),
                ),
            ]
        else:
            waiting = [
                hsm.transition(
                    hsm.on(events.DetachedEvent),
                    hsm.guard(is_expected),
                    hsm.effect(forward),
                    hsm.target("/AttachmentGroupReply/done"),
                ),
                hsm.transition(
                    hsm.on(events.DetachFailedEvent),
                    hsm.guard(is_expected),
                    hsm.effect(forward),
                    hsm.target("/AttachmentGroupReply/done"),
                ),
            ]
        if with_timeout:

            def timeout_delay(
                ctx: hsm.Context,
                instance: _Reply,
                event: hsm.Event[typing.Any],
            ) -> datetime.timedelta:
                del ctx, instance, event
                return operation.timeout

            waiting.append(
                hsm.transition(
                    hsm.after(timeout_delay),
                    hsm.effect(timeout),
                    hsm.target("/AttachmentGroupReply/done"),
                )
            )
        return bot.define(
            "AttachmentGroupReply",
            hsm.initial(hsm.target("waiting")),
            hsm.state("waiting", *waiting),
            hsm.final("done"),
        )

    @classmethod
    async def started(
        cls,
        ctx: hsm.Context,
        operation: _OperationData,
        index: int,
        metadata: dict[str, object],
        *,
        with_timeout: bool,
    ) -> "_Reply":
        reply = cls()
        values = scope.mark_private({hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()})
        reply_ctx = hsm.Context(parent=ctx, values=values)
        try:
            return await bot.started(
                reply_ctx,
                reply,
                cls._define_model(operation, index, metadata, with_timeout=with_timeout),
            )
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
        self._attachment_request_id: str = ""
        # Machine-owned attach hold (HSM-CONTEXT-001): not instance.state() for fallback routing.
        self._held_attached: bool = False
        # Owned barrier accumulation for the active operation phase. Events carry an
        # immutable _OperationData snapshot for correlation; per-member outcomes and
        # derived failure/created sets accumulate here (owning-class reads/writes only).
        self._op: _OperationData | None = None
        self._op_results: dict[int, _MemberResult] = {}
        self._op_attempted: set[int] = set()
        self._op_failure: events.FailedData | None = None
        self._op_created: tuple[int, ...] = ()
        self._op_fallback: bool = False

    @staticmethod
    async def _attach_members_activity(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        phase = instance._op
        assert phase is not None
        replies: dict[int, _Reply] = {}

        def fail_replies() -> None:
            lifetime = instance.context()
            failure = events.FailedData(
                actor=phase.actor,
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
                        id=hsm.id(reply),
                        source=hsm.id(member),
                        target=hsm.id(reply),
                        metadata=dict(event.metadata),
                    ),
                )

        async def attach_member(index: int) -> None:
            member = instance._attachments[index]
            assert isinstance(member, Attachment)
            reply: _Reply | None = None
            try:
                reply = await _Reply.started(
                    phase.context,
                    phase,
                    index,
                    dict(event.metadata),
                    with_timeout=True,
                )
                replies[index] = reply
                await member.attach(
                    phase.context,
                    dataclasses.replace(
                        events.AttachEvent.with_data(
                            events.AttachData(actor=phase.actor, reply_to=reply, timeout=phase.timeout)
                        ),
                        id=hsm.id(reply),
                        source=hsm.id(instance),
                        target=Attachment._actor_id(member),
                        metadata=dict(event.metadata),
                    ),
                )
            except asyncio.CancelledError:
                if ctx.is_done() and reply is None:
                    _ = hsm.Instance.dispatch(
                        instance,
                        instance.context(),
                        dataclasses.replace(
                            _MemberAttachFailedEvent.with_data(
                                _MemberAttachFailedData(
                                    operation=phase,
                                    index=index,
                                    result=events.FailedData(
                                        actor=phase.actor,
                                        kind=events.FailureKind.DISPATCH,
                                        message=f"{type(instance).__name__} member reply start was canceled.",
                                    ),
                                )
                            ),
                            id=phase.request_id,
                            source=hsm.id(member),
                            target=hsm.id(instance),
                            metadata=dict(event.metadata),
                        ),
                    )
                    return
                raise
            except Exception:
                failure = events.FailedData(
                    actor=phase.actor,
                    kind=events.FailureKind.DISPATCH,
                    message=f"{type(instance).__name__} member attach failed.",
                )
                # Prefer Attachment._actor_id: hsm.id can fail mid-stop (hsm 1.3.2+).
                member_id = Attachment._actor_id(member)
                if reply is not None:
                    _ = hsm.Instance.dispatch(
                        reply,
                        ctx,
                        dataclasses.replace(
                            events.AttachFailedEvent.with_data(failure),
                            id=hsm.id(reply),
                            source=member_id,
                            target=hsm.id(reply),
                            metadata=dict(event.metadata),
                        ),
                    )
                    return
                _ = hsm.Instance.dispatch(
                    instance,
                    ctx,
                    dataclasses.replace(
                        _MemberAttachFailedEvent.with_data(
                            _MemberAttachFailedData(operation=phase, index=index, result=failure)
                        ),
                        id=phase.request_id,
                        source=member_id,
                        target=hsm.id(instance),
                        metadata=dict(event.metadata),
                    ),
                )

        fanout = asyncio.gather(*(attach_member(index) for index in phase.members))
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
        phase = instance._op
        assert phase is not None
        replies: dict[int, _Reply] = {}

        def fail_replies() -> None:
            lifetime = instance.context()
            failure = events.FailedData(
                actor=phase.actor,
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
                        id=hsm.id(reply),
                        source=hsm.id(member),
                        target=hsm.id(reply),
                        metadata=dict(event.metadata),
                    ),
                )

        async def detach_member(index: int) -> None:
            member = instance._attachments[index]
            assert isinstance(member, Attachment)
            reply: _Reply | None = None
            try:
                reply = await _Reply.started(
                    phase.context,
                    phase,
                    index,
                    dict(event.metadata),
                    with_timeout=True,
                )
                replies[index] = reply
                instance._op_attempted.add(index)
                await member.detach(
                    phase.context,
                    dataclasses.replace(
                        events.DetachEvent.with_data(
                            events.DetachData(
                                actor=phase.actor,
                                reply_to=reply,
                                timeout=phase.timeout,
                            )
                        ),
                        id=hsm.id(reply),
                        source=hsm.id(instance),
                        target=hsm.id(member),
                        metadata=dict(event.metadata),
                    ),
                )
            except asyncio.CancelledError:
                if ctx.is_done() and reply is None:
                    _ = hsm.Instance.dispatch(
                        instance,
                        instance.context(),
                        dataclasses.replace(
                            _MemberDetachFailedEvent.with_data(
                                _MemberDetachFailedData(
                                    operation=phase,
                                    index=index,
                                    result=events.FailedData(
                                        actor=phase.actor,
                                        kind=events.FailureKind.DISPATCH,
                                        message=f"{type(instance).__name__} member reply start was canceled.",
                                    ),
                                )
                            ),
                            id=phase.request_id,
                            source=hsm.id(member),
                            target=hsm.id(instance),
                            metadata=dict(event.metadata),
                        ),
                    )
                    return
                raise
            except Exception:
                failure = events.FailedData(
                    actor=phase.actor,
                    kind=events.FailureKind.DISPATCH,
                    message=f"{type(instance).__name__} member detach failed.",
                )
                if reply is not None:
                    _ = hsm.Instance.dispatch(
                        reply,
                        ctx,
                        dataclasses.replace(
                            events.DetachFailedEvent.with_data(failure),
                            id=hsm.id(reply),
                            source=hsm.id(member),
                            target=hsm.id(reply),
                            metadata=dict(event.metadata),
                        ),
                    )
                    return
                _ = hsm.Instance.dispatch(
                    instance,
                    ctx,
                    dataclasses.replace(
                        _MemberDetachFailedEvent.with_data(
                            _MemberDetachFailedData(operation=phase, index=index, result=failure)
                        ),
                        id=phase.request_id,
                        source=hsm.id(member),
                        target=hsm.id(instance),
                        metadata=dict(event.metadata),
                    ),
                )

        fanout = asyncio.gather(*(detach_member(index) for index in phase.members))
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
        data = event.data
        if not isinstance(
            data,
            (_MemberAttachCompleteData, _MemberAttachFailedData, _MemberDetachedData, _MemberDetachFailedData),
        ):
            return False
        snapshot = data.operation
        phase = instance._op
        if phase is None:
            return False
        index = data.index
        result = data.result
        # Phase match: ignore stale member outcomes from a previous phase (same request,
        # different kind/members after a rollback transition). Accumulation lives on the
        # instance; the event snapshot is correlation only.
        return (
            snapshot.request_id == phase.request_id
            and snapshot.kind == phase.kind
            and snapshot.members == phase.members
            and index in phase.members
            and index not in instance._op_results
            and 0 <= index < len(instance._attachments)
            and event.id == phase.request_id
            and event.source == hsm.id(phase.expected_members[index])
            and event.target == hsm.id(instance)
            and hsm.id(result.actor) == hsm.id(phase.actor)
        )

    @staticmethod
    def _is_last_result(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        phase = instance._op
        return (
            Group._is_correlated(ctx, instance, event)
            and isinstance(
                data,
                (_MemberAttachCompleteData, _MemberAttachFailedData, _MemberDetachedData, _MemberDetachFailedData),
            )
            and phase is not None
            and len(instance._op_results) + 1 == len(phase.members)
        )

    @staticmethod
    def _operation_failed(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._op_failure is not None

    @staticmethod
    def _attach_needs_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._op_failure is not None and bool(instance._op_created)

    @staticmethod
    def _mark_held_attached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._held_attached = True

    @staticmethod
    def _clear_held_attached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._held_attached = False

    @staticmethod
    def _fallback_is_attached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._op_fallback

    @staticmethod
    def _begin_operation(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        snapshot = _operation(event)
        # New barrier phase: store the immutable snapshot and reset accumulation.
        # The snapshot already carries request_id == event.id from attach()/detach().
        instance._op = snapshot.model_copy(update={"request_id": event.id or snapshot.request_id})
        instance._op_results = {}
        instance._op_attempted = set()
        instance._op_failure = None
        instance._op_created = ()
        instance._op_fallback = snapshot.fallback_attached

    @staticmethod
    def _record_result(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(
            data,
            (_MemberAttachCompleteData, _MemberAttachFailedData, _MemberDetachedData, _MemberDetachFailedData),
        )
        instance._op_results[data.index] = data.result

    @staticmethod
    def _finalize_attach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        phase = instance._op
        assert phase is not None
        instance._op_failure = _first_failure(instance._op_results)
        created_members: list[int] = []
        for index in phase.members:
            result = instance._op_results[index]
            if isinstance(result, events.AttachCompleteData) and result.created:
                created_members.append(index)
        instance._op_created = tuple(created_members)

    @staticmethod
    def _begin_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        phase = instance._op
        assert phase is not None
        # New detach phase over the members this attach created; fresh accumulation.
        # dataclasses.replace would copy the event, but the phase descriptor itself is
        # immutable — produce the next snapshot via model_copy and store it owned.
        instance._op = phase.model_copy(update={"kind": _OperationKind.DETACH, "members": instance._op_created})
        instance._op_results = {}

    @staticmethod
    def _finalize_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        phase = instance._op
        assert phase is not None
        failure = _first_failure(instance._op_results)
        if failure is not None:
            instance._op_failure = events.FailedData(
                actor=phase.actor,
                kind=events.FailureKind.ROLLBACK,
                message=f"{type(instance).__name__} rollback failed.",
            )
            instance._op_fallback = True

    @staticmethod
    def _finalize_detach(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        phase = instance._op
        assert phase is not None
        instance._op_failure = _first_failure(instance._op_results)
        instance._op_created = (
            tuple(index for index in phase.members if index in instance._op_attempted)
            if instance._op_failure is not None
            else ()
        )

    @staticmethod
    def _detach_needs_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._op_failure is not None and bool(instance._op_created)

    @staticmethod
    def _begin_detach_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        phase = instance._op
        assert phase is not None
        instance._op = phase.model_copy(
            update={
                "kind": _OperationKind.ATTACH,
                "members": instance._op_created,
                "timeout": instance._attachment_timeout,
            }
        )
        instance._op_results = {}

    @staticmethod
    def _finalize_detach_rollback(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        phase = instance._op
        assert phase is not None
        failure = _first_failure(instance._op_results)
        if failure is not None:
            instance._op_failure = events.FailedData(
                actor=phase.actor,
                kind=events.FailureKind.ROLLBACK,
                message=f"{type(instance).__name__} detach recovery failed.",
            )

    @staticmethod
    def _dispatch_attach_complete(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        phase = instance._op
        assert phase is not None
        created = any(
            isinstance(result, events.AttachCompleteData) and result.created for result in instance._op_results.values()
        )
        _ = hsm.dispatch(
            ctx,
            phase.reply_to,
            dataclasses.replace(
                events.AttachCompleteEvent.with_data(events.AttachCompleteData(actor=phase.actor, created=created)),
                id=phase.request_id,
                source=hsm.id(instance),
                target=hsm.id(phase.reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_empty_attach_complete(
        ctx: hsm.Context,
        instance: "Group",
        event: hsm.Event[typing.Any],
    ) -> None:
        phase = instance._op
        assert phase is not None
        _ = hsm.dispatch(
            ctx,
            phase.reply_to,
            dataclasses.replace(
                events.AttachCompleteEvent.with_data(events.AttachCompleteData(actor=phase.actor, created=True)),
                id=phase.request_id,
                source=hsm.id(instance),
                target=hsm.id(phase.reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_attach_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        phase = instance._op
        assert phase is not None
        assert instance._op_failure is not None
        _ = hsm.dispatch(
            ctx,
            phase.reply_to,
            dataclasses.replace(
                events.AttachFailedEvent.with_data(instance._op_failure),
                id=phase.request_id,
                source=hsm.id(instance),
                target=hsm.id(phase.reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        phase = instance._op
        assert phase is not None
        removed = any(
            isinstance(result, events.DetachedData) and result.removed for result in instance._op_results.values()
        )
        _ = hsm.dispatch(
            ctx,
            phase.reply_to,
            dataclasses.replace(
                events.DetachedEvent.with_data(events.DetachedData(actor=phase.actor, removed=removed)),
                id=phase.request_id,
                source=hsm.id(instance),
                target=hsm.id(phase.reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_empty_detached(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        phase = instance._op
        assert phase is not None
        _ = hsm.dispatch(
            ctx,
            phase.reply_to,
            dataclasses.replace(
                events.DetachedEvent.with_data(events.DetachedData(actor=phase.actor, removed=True)),
                id=phase.request_id,
                source=hsm.id(instance),
                target=hsm.id(phase.reply_to),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _dispatch_detach_failure(ctx: hsm.Context, instance: "Group", event: hsm.Event[typing.Any]) -> None:
        phase = instance._op
        assert phase is not None
        assert instance._op_failure is not None
        _ = hsm.dispatch(
            ctx,
            phase.reply_to,
            dataclasses.replace(
                events.DetachFailedEvent.with_data(instance._op_failure),
                id=phase.request_id,
                source=hsm.id(instance),
                target=hsm.id(phase.reply_to),
                metadata=dict(event.metadata),
            ),
        )

    model: typing.ClassVar[hsm.Model] = bot.define(
        "AttachmentGroup",
        hsm.initial(hsm.target("detached")),
        hsm.state(
            "detached",
            hsm.entry(_clear_held_attached),
            hsm.transition(
                hsm.on(_StartAttachEvent),
                hsm.guard(_has_members),
                hsm.effect(_begin_operation),
                hsm.target("/AttachmentGroup/attaching"),
            ),
            hsm.transition(
                hsm.on(_StartAttachEvent),
                hsm.effect(_begin_operation, _dispatch_empty_attach_complete),
                hsm.target("/AttachmentGroup/attached"),
            ),
            hsm.transition(
                hsm.on(_StartDetachEvent),
                hsm.effect(_begin_operation, _dispatch_empty_detached),
            ),
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
            hsm.entry(_mark_held_attached),
            hsm.transition(
                hsm.on(_StartAttachEvent),
                hsm.guard(_has_members),
                hsm.effect(_begin_operation),
                hsm.target("/AttachmentGroup/attaching"),
            ),
            hsm.transition(
                hsm.on(_StartAttachEvent),
                hsm.effect(_begin_operation, _dispatch_empty_attach_complete),
            ),
            hsm.transition(
                hsm.on(_StartDetachEvent),
                hsm.guard(_has_members),
                hsm.effect(_begin_operation),
                hsm.target("/AttachmentGroup/detaching"),
            ),
            hsm.transition(
                hsm.on(_StartDetachEvent),
                hsm.effect(_begin_operation, _dispatch_empty_detached),
                hsm.target("/AttachmentGroup/detached"),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(_StartAttachEvent),
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
            hsm.defer(_StartAttachEvent),
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
        data = event.data
        assert isinstance(data, events.AttachData)

        async def start_members_and_dispatch() -> None:
            for member in self._attachments:
                model = typing.cast(_Modeled, typing.cast(object, member)).model
                if lifecycle.is_started(member):
                    continue
                assert model is not None
                try:
                    # Members must outlive the request that attaches them: parent them under the
                    # group's owning context, not the caller's transient ``ctx`` (HSM-CONTEXT-001).
                    _ = await bot.started(self.context(), member, model)
                except Exception:
                    reply_to = data.actor if data.reply_to is None else data.reply_to
                    await hsm.dispatch(
                        ctx,
                        reply_to,
                        dataclasses.replace(
                            events.AttachFailedEvent.with_data(
                                events.FailedData(
                                    actor=data.actor,
                                    kind=events.FailureKind.INITIALIZATION,
                                    message="Attachment Group member start failed.",
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
                    _StartAttachEvent.with_data(
                        _OperationData(
                            coordinator=self,
                            actor=data.actor,
                            reply_to=data.actor if data.reply_to is None else data.reply_to,
                            request_id=event.id,
                            context=ctx,
                            expected_members=tuple(self._attachments),
                            timeout=data.timeout,
                            kind=_OperationKind.ATTACH,
                            members=tuple(range(len(self._attachments))),
                            fallback_attached=self._held_attached,
                        )
                    ),
                    id=event.id,
                    source=event.source,
                    target=hsm.id(self),
                    metadata=dict(event.metadata),
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
        data = event.data
        assert isinstance(data, events.DetachData)
        operation = _OperationData(
            coordinator=self,
            actor=data.actor,
            reply_to=data.actor if data.reply_to is None else data.reply_to,
            request_id=event.id,
            context=ctx,
            expected_members=tuple(self._attachments),
            timeout=data.timeout,
            kind=_OperationKind.DETACH,
            members=tuple(range(len(self._attachments))),
            fallback_attached=True,
        )
        return hsm.Instance.dispatch(
            self,
            ctx,
            dataclasses.replace(
                _StartDetachEvent.with_data(operation),
                id=event.id,
                source=event.source,
                target=hsm.id(self),
                metadata=dict(event.metadata),
            ),
        )

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[None]:
        """Attach/detach by typed payload; group HSM vs member fan-out for the rest.

        AttachData/DetachData select lifecycle methods (not event.name). Group
        coordination payloads (operation/member outcomes) use Instance.dispatch; other
        events fan out to members. No ``event.name`` admission door.
        """

        data = event.data
        if isinstance(data, events.AttachData):
            return self.attach(ctx, event)
        if isinstance(data, events.DetachData):
            return self.detach(ctx, event)
        # Coordination payloads stay on the group machine (typed, not event.name).
        if isinstance(
            data,
            (
                _OperationData,
                _MemberAttachCompleteData,
                _MemberAttachFailedData,
                _MemberDetachedData,
                _MemberDetachFailedData,
            ),
        ):
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
