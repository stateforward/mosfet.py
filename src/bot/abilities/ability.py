import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid
import weakref

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot.protocols import attachment
from bot.telemetry import observer


TInput = typing.TypeVar("TInput")
TOutput = typing.TypeVar("TOutput")
_DataType = type[object] | tuple[type[object], ...] | None


class _CompositeAttachmentTerminalData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    request: SkipJsonSchema[pydantic.SkipValidation[attachment.AttachData | attachment.DetachData]] = pydantic.Field(
        exclude=True
    )
    terminal: SkipJsonSchema[pydantic.SkipValidation[hsm.Event[typing.Any]]] = pydantic.Field(exclude=True)
    operation: SkipJsonSchema[pydantic.SkipValidation[object]] = pydantic.Field(exclude=True)
    reply: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(exclude=True)


_CompositeAttachmentTerminalEvent = hsm.Event[_CompositeAttachmentTerminalData](
    name="bot.ability.attachment.terminal",
    schema=_CompositeAttachmentTerminalData,
)


@dataclasses.dataclass(slots=True)
class _CompositeAttachmentOperation:
    owner: "Ability[typing.Any, typing.Any]"
    source: hsm.Instance
    request: attachment.AttachData | attachment.DetachData
    request_id: str
    metadata: dict[str, object]
    reply: hsm.Instance | None = None
    terminal: hsm.Event[typing.Any] | None = None
    consumed: bool = False


class _CompositeAttachmentReply(hsm.Instance):
    @classmethod
    def model_for(cls, operation: _CompositeAttachmentOperation) -> hsm.Model:
        expected = (
            (attachment.AttachCompleteEvent, attachment.AttachFailedEvent)
            if isinstance(operation.request, attachment.AttachData)
            else (attachment.DetachedEvent, attachment.DetachFailedEvent)
        )

        def correlated(
            ctx: hsm.Context,
            instance: _CompositeAttachmentReply,
            event: hsm.Event[typing.Any],
        ) -> bool:
            del ctx
            data = event.data
            # Reply actor is one-shot and closure-bound to ``operation``; correlate by envelope.
            return (
                event.id == operation.request_id
                and event.source == hsm.id(operation.source)
                and event.target == hsm.id(instance)
                and isinstance(data, (attachment.AttachCompleteData, attachment.DetachedData, attachment.FailedData))
                and data.actor is operation.owner
            )

        def forward(
            ctx: hsm.Context,
            instance: _CompositeAttachmentReply,
            event: hsm.Event[typing.Any],
        ) -> None:
            operation.terminal = event
            _ = hsm.dispatch(
                ctx,
                operation.owner,
                dataclasses.replace(
                    _CompositeAttachmentTerminalEvent.with_data(
                        _CompositeAttachmentTerminalData(
                            request=operation.request,
                            terminal=event,
                            operation=operation,
                            reply=instance,
                        )
                    ),
                    id=operation.request_id,
                    source=hsm.id(instance),
                    target=hsm.id(operation.owner),
                    metadata=dict(operation.metadata),
                ),
            )

        return hsm.define(
            "AbilityAttachmentReply",
            hsm.initial(hsm.target("waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(*expected),
                    hsm.guard(correlated),
                    hsm.effect(forward),
                    hsm.target("/AbilityAttachmentReply/done"),
                ),
            ),
            hsm.final("done"),
        )


class FailureData(pydantic.BaseModel):
    """FailureData signal produced when an ability operation cannot complete."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": "FailureData signal produced when an ability operation cannot complete.",
            "examples": [{"message": "Provider request failed."}],
        },
    )

    message: str = pydantic.Field(
        description="Human-readable failure message for the ability operation.",
        examples=["Provider request failed."],
    )


def _is_base_model_type(data_type: object) -> typing.TypeGuard[type[pydantic.BaseModel]]:
    if not isinstance(data_type, type):
        return False
    try:
        return issubclass(data_type, pydantic.BaseModel)
    except TypeError:
        return False


def _model_with_json_schema_extra(
    data_type: type[pydantic.BaseModel],
    *,
    description: str | None,
    examples: collections.abc.Sequence[object] | None,
) -> type[pydantic.BaseModel]:
    """Return a BaseModel subclass with merged root json_schema_extra (still a model, not TypeAdapter)."""

    existing_extra = data_type.model_config.get("json_schema_extra")
    extra: dict[str, pydantic.JsonValue] = {}
    if isinstance(existing_extra, dict):
        extra = dict(existing_extra)
    if description is not None:
        extra["description"] = description
    if examples is not None:
        extra["examples"] = typing.cast(pydantic.JsonValue, list(examples))
    # Preserve relevant base config; create_model needs an explicit ConfigDict for extra merge.
    config = pydantic.ConfigDict(
        frozen=bool(data_type.model_config.get("frozen", False)),
        extra=data_type.model_config.get("extra", "ignore"),  # type: ignore[arg-type]
        arbitrary_types_allowed=bool(data_type.model_config.get("arbitrary_types_allowed", False)),
        json_schema_extra=extra,
    )
    return pydantic.create_model(
        f"{data_type.__name__}EventSchema",
        __base__=data_type,
        __config__=config,
    )


def _schema_for_data_type(
    data_type: type[object],
    *,
    description: str | None,
    examples: collections.abc.Sequence[object] | None,
) -> object:
    """Resolve an event payload schema.

    Prefer concrete ``BaseModel`` types (required fields / Field descriptions stay on the model).
    Do not wrap models in ``TypeAdapter`` — that breaks create_model patching and hides
    domain required lists. Non-model types may still use TypeAdapter when metadata is needed.
    """

    if _is_base_model_type(data_type):
        if description is None and examples is None:
            return data_type
        return _model_with_json_schema_extra(
            data_type,
            description=description,
            examples=examples,
        )
    if description is None and examples is None:
        return data_type
    # Non-BaseModel payloads (e.g. bytes) still need TypeAdapter to attach description/examples.
    field = _schema_metadata_field(description=description, examples=examples)
    schema_type: object = typing.Annotated[data_type, field]
    return typing.cast(pydantic.TypeAdapter[object], pydantic.TypeAdapter(schema_type))


def _schema_metadata_field(
    *,
    description: str | None,
    examples: collections.abc.Sequence[object] | None,
) -> object:
    if description is None:
        return typing.cast(object, pydantic.Field(examples=list(examples or ())))
    if examples is None:
        return typing.cast(object, pydantic.Field(description=description))
    return typing.cast(object, pydantic.Field(description=description, examples=list(examples)))


InputEvent = hsm.Event[typing.Any](
    name="bot.ability.input",
    schema=_schema_for_data_type(
        object,
        description=(
            "InputData event data accepted by a generic ability. Concrete abilities should replace this with "
            "a specific input event schema."
        ),
        examples=["Summarize this note."],
    ),
)
OutputEvent = hsm.Event[typing.Any](
    name="bot.ability.output",
    schema=_schema_for_data_type(
        object,
        description=(
            "OutputData event data dispatched by a generic ability. Concrete abilities should replace this with "
            "a specific output event schema."
        ),
        examples=["Summary text."],
    ),
)
FailedEvent = hsm.Event[FailureData](
    name="bot.ability.failed",
    schema=FailureData,
)
TerminalOutputEvent = hsm.Event[hsm.Event[typing.Any]](
    name="bot.ability.terminal.output",
    kind=hsm.CompletionEventKind,
    schema=hsm.Event[typing.Any],
)
TerminalErrorEvent = hsm.Event[hsm.Event[typing.Any]](
    name="bot.ability.terminal.error",
    kind=hsm.ErrorEventKind,
    schema=hsm.Event[typing.Any],
)


def ability_input_event(
    name: str,
    data_type: type[TInput],
    *,
    description: str | None = None,
    examples: collections.abc.Sequence[object] | None = None,
) -> hsm.Event[TInput]:
    """Build an ability input event with a concrete Pydantic payload schema."""

    schema = _schema_for_data_type(
        data_type,
        description=description,
        examples=examples,
    )
    return hsm.Event[TInput](
        name=name,
        schema=schema,
    )


def ability_output_event(
    name: str,
    data_type: type[TOutput],
    *,
    description: str | None = None,
    examples: collections.abc.Sequence[object] | None = None,
) -> hsm.Event[TOutput]:
    """Build an ability output event with a concrete Pydantic payload schema."""

    schema = _schema_for_data_type(
        data_type,
        description=description,
        examples=examples,
    )
    return hsm.Event[TOutput](
        name=name,
        schema=schema,
    )


class Ability(hsm.Instance, attachment.Attachment, typing.Generic[TInput, TOutput]):
    """Specific event-driven operational capacity.

    Answers: What operations can the system perform?
    """

    _attachment_limit: typing.ClassVar[int | None] = 1
    _attachments: list[hsm.Instance]
    _attachment_group: attachment.Group
    _attachment_timeout: datetime.timedelta
    _composite_attachment_lifecycle: typing.ClassVar[bool] = False
    _composite_attachment_terminal_event: typing.ClassVar[hsm.Event[_CompositeAttachmentTerminalData]] = (
        _CompositeAttachmentTerminalEvent
    )
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = OutputEvent
    failed_event: typing.ClassVar[hsm.Event[typing.Any]] = FailedEvent
    # Cancel contract: abilities that support cancellation declare their cancel request and
    # cancelled-confirmation events (Processing and Cognition declare their own; None means no
    # declared contract). Payloads must expose ``operation_id`` and ``token`` fields so owners
    # can build the request from the declared event's schema. Machines that trigger on these
    # events model static triggers; an owner can only consume cancelled events named in its
    # own model.
    cancel_event: typing.ClassVar[hsm.Event[typing.Any] | None] = None
    cancelled_event: typing.ClassVar[hsm.Event[typing.Any] | None] = None
    input_data_type: typing.ClassVar[_DataType] = None
    output_data_type: typing.ClassVar[_DataType] = None
    submodel: typing.ClassVar[hsm.Model | None] = None
    # Host-boundary waiters keyed by operation id (not event.metadata). Single-flight hosts register before dispatch.
    _terminal_waiters: dict[str, asyncio.Future[hsm.Event[typing.Any]]]

    def register_terminal_waiter(
        self,
        operation_id: str,
        waiter: asyncio.Future[hsm.Event[typing.Any]],
    ) -> None:
        """Register a host Future completed when this ability emits a terminal for ``operation_id``."""

        if not operation_id:
            raise ValueError("operation_id is required.")
        self._terminal_waiters[operation_id] = waiter

    def clear_terminal_waiter(self, operation_id: str) -> None:
        """Drop a host terminal waiter if it is still registered."""

        _ = self._terminal_waiters.pop(operation_id, None)

    @staticmethod
    async def await_child_terminal(
        ctx: hsm.Context,
        *,
        owner: hsm.Instance,
        child: "Ability[typing.Any, typing.Any]",
        operation_id: str,
        input: object,
        metadata: collections.abc.Mapping[str, object],
    ) -> hsm.Event[typing.Any]:
        """Dispatch one child apply and await its terminal by envelope id (HSM-CORRELATION-001).

        Registers a waiter on ``child``, dispatches the child's input event with
        ``source=owner`` / ``target=child`` / ``id=operation_id``, and returns the
        terminal event. Cancels the waiter cleanly when the owning activity exits.
        """

        if not operation_id:
            raise ValueError("operation_id is required.")
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[hsm.Event[typing.Any]] = loop.create_future()
        child.register_terminal_waiter(operation_id, waiter)
        try:
            child_event = dataclasses.replace(
                child.input_event.with_data_and_id(input, operation_id),
                source=hsm.id(owner),
                target=hsm.id(child),
                metadata=dict(metadata),
            )
            await hsm.dispatch(ctx, child, child_event)
            return await waiter
        except asyncio.CancelledError:
            child.clear_terminal_waiter(operation_id)
            if not waiter.done():
                _ = waiter.cancel()
            raise
        finally:
            child.clear_terminal_waiter(operation_id)

    @staticmethod
    def _has_terminal_event(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx, instance
        return isinstance(event.data, hsm.Event)

    @staticmethod
    def _forward_terminal_event(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        terminal = event.data
        assert isinstance(terminal, hsm.Event)
        operation_id = terminal.id if terminal.id else ""
        waiter = instance._terminal_waiters.pop(operation_id, None) if operation_id else None
        if isinstance(waiter, asyncio.Future) and not waiter.done():
            waiter.set_result(terminal)
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                terminal,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(terminal.metadata),
            ),
        )

    @staticmethod
    async def _attach_composite_group(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        request = event.data
        assert isinstance(request, attachment.AttachData)
        reply: hsm.Instance | None = None
        source: hsm.Instance = instance._attachment_group
        try:
            private_scope = hsm.Context(
                parent=instance.context(),
                values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()},
            )
            # hsm 1.3.2+: id fails after stop, so this is a valid start gate again.
            try:
                _ = hsm.id(instance._attachment_group)
            except hsm.ErrorValidatingModel:
                try:
                    _ = await hsm.started(
                        private_scope, instance._attachment_group, instance._attachment_group.model
                    )
                except Exception:
                    source = instance
                    raise
            reply = await instance._start_composite_attachment_reply(
                source,
                request,
                event,
            )
            await instance._attachment_group.attach(
                private_scope,
                dataclasses.replace(
                    attachment.AttachEvent.with_data(
                        attachment.AttachData(actor=instance, reply_to=reply, timeout=request.timeout)
                    ),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._attachment_group),
                    metadata=dict(event.metadata),
                ),
            )
        except Exception as error:
            failure = attachment.FailedData(
                actor=instance,
                kind=attachment.FailureKind.DISPATCH,
                message=f"{type(instance).__name__} attachment Group start failed: {error}",
            )
            if reply is None:
                instance._dispatch_composite_attachment_failure(
                    ctx,
                    source,
                    request,
                    event,
                    failure,
                )
                return
            _ = hsm.dispatch(
                ctx,
                reply,
                dataclasses.replace(
                    attachment.AttachFailedEvent.with_data(failure),
                    id=event.id,
                    source=hsm.id(source),
                    target=hsm.id(reply),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    async def _detach_composite_group(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        request = event.data
        assert isinstance(request, attachment.DetachData)
        reply: hsm.Instance | None = None
        try:
            private_scope = hsm.Context(
                parent=instance.context(),
                values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()},
            )
            reply = await instance._start_composite_attachment_reply(
                instance._attachment_group,
                request,
                event,
            )
            await instance._attachment_group.detach(
                private_scope,
                dataclasses.replace(
                    attachment.DetachEvent.with_data(
                        attachment.DetachData(actor=instance, reply_to=reply, timeout=request.timeout)
                    ),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(instance._attachment_group),
                    metadata=dict(event.metadata),
                ),
            )
        except Exception as error:
            failure = attachment.FailedData(
                actor=instance,
                kind=attachment.FailureKind.DISPATCH,
                message=f"{type(instance).__name__} attachment Group detach failed: {error}",
            )
            if reply is None:
                instance._dispatch_composite_attachment_failure(
                    ctx,
                    instance._attachment_group,
                    request,
                    event,
                    failure,
                )
                return
            _ = hsm.dispatch(
                ctx,
                reply,
                dataclasses.replace(
                    attachment.DetachFailedEvent.with_data(failure),
                    id=event.id,
                    source=hsm.id(instance._attachment_group),
                    target=hsm.id(reply),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _deliver_composite_attachment_terminal(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, _CompositeAttachmentTerminalData)
        operation = data.operation
        assert isinstance(operation, _CompositeAttachmentOperation)
        operation.consumed = True
        request = data.request
        terminal = data.terminal
        target = request.actor if request.reply_to is None else request.reply_to
        if terminal.name == attachment.AttachCompleteEvent.name:
            assert isinstance(request, attachment.AttachData)
            public = attachment.AttachCompleteEvent.with_data(
                attachment.AttachCompleteData(actor=request.actor, created=True)
            )
        elif terminal.name == attachment.DetachedEvent.name:
            assert isinstance(request, attachment.DetachData)
            index = instance._attachment_index(request.actor)
            assert index is not None
            del instance._attachments[index]
            public = attachment.DetachedEvent.with_data(attachment.DetachedData(actor=request.actor, removed=True))
        else:
            failure = terminal.data
            assert isinstance(failure, attachment.FailedData)
            if isinstance(request, attachment.AttachData):
                index = instance._attachment_index(request.actor)
                assert index is not None
                del instance._attachments[index]
                public = attachment.AttachFailedEvent.with_data(
                    attachment.FailedData(actor=request.actor, kind=failure.kind, message=failure.message)
                )
            else:
                public = attachment.DetachFailedEvent.with_data(
                    attachment.FailedData(actor=request.actor, kind=failure.kind, message=failure.message)
                )
        _ = hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(
                public,
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            ),
        )

    async def _start_composite_attachment_reply(
        self,
        source: hsm.Instance,
        request: attachment.AttachData | attachment.DetachData,
        event: hsm.Event[typing.Any],
    ) -> hsm.Instance:
        operation = _CompositeAttachmentOperation(
            owner=self,
            source=source,
            request=request,
            request_id=event.id,
            metadata=dict(event.metadata),
        )
        private_scope = hsm.Context(
            parent=self.context(),
            values={hsm.Keys.Instances: weakref.WeakValueDictionary[str, hsm.Instance]()},
        )
        reply = await hsm.started(
            private_scope,
            _CompositeAttachmentReply(),
            _CompositeAttachmentReply.model_for(operation),
        )
        operation.reply = reply
        return reply

    def _dispatch_composite_attachment_failure(
        self,
        ctx: hsm.Context,
        source: hsm.Instance,
        request: attachment.AttachData | attachment.DetachData,
        event: hsm.Event[typing.Any],
        failure: attachment.FailedData,
    ) -> None:
        operation = _CompositeAttachmentOperation(
            owner=self,
            source=source,
            request=request,
            request_id=event.id,
            metadata=dict(event.metadata),
            reply=self,
        )
        terminal_type = (
            attachment.AttachFailedEvent if isinstance(request, attachment.AttachData) else attachment.DetachFailedEvent
        )
        terminal = dataclasses.replace(
            terminal_type.with_data(failure),
            id=event.id,
            source=hsm.id(source),
            target=hsm.id(self),
            metadata=dict(event.metadata),
        )
        operation.terminal = terminal
        _ = hsm.dispatch(
            ctx,
            self,
            dataclasses.replace(
                _CompositeAttachmentTerminalEvent.with_data(
                    _CompositeAttachmentTerminalData(
                        request=request,
                        terminal=terminal,
                        operation=operation,
                        reply=self,
                    )
                ),
                id=event.id,
                source=hsm.id(self),
                target=hsm.id(self),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _is_composite_attach_complete(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Ability._is_valid_composite_attachment_terminal(ctx, instance, event) and (
            isinstance(event.data, _CompositeAttachmentTerminalData)
            and event.data.terminal.name == attachment.AttachCompleteEvent.name
        )

    @staticmethod
    def _is_composite_detach_failed(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Ability._is_valid_composite_attachment_terminal(ctx, instance, event) and (
            isinstance(event.data, _CompositeAttachmentTerminalData)
            and event.data.terminal.name == attachment.DetachFailedEvent.name
        )

    @staticmethod
    def _is_composite_rollback_failure(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return (
            Ability._is_valid_composite_attachment_terminal(ctx, instance, event)
            and isinstance(event.data, _CompositeAttachmentTerminalData)
            and event.data.terminal.name == attachment.DetachFailedEvent.name
            and isinstance(event.data.terminal.data, attachment.FailedData)
            and event.data.terminal.data.kind is attachment.FailureKind.ROLLBACK
        )

    @staticmethod
    def _is_valid_composite_attachment_terminal(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _CompositeAttachmentTerminalData):
            return False
        operation = data.operation
        return (
            isinstance(operation, _CompositeAttachmentOperation)
            and operation.owner is instance
            and not operation.consumed
            and operation.reply is data.reply
            and operation.terminal is data.terminal
            and data.request is operation.request
            and event.id == operation.request_id
            and event.source == hsm.id(data.reply)
            and event.target == hsm.id(instance)
            and data.terminal.id == operation.request_id
            and data.terminal.source == hsm.id(operation.source)
            and data.terminal.target == hsm.id(data.reply)
        )

    @staticmethod
    def _is_composite_detached_terminal(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Ability._is_valid_composite_attachment_terminal(ctx, instance, event) and (
            isinstance(event.data, _CompositeAttachmentTerminalData)
            and event.data.terminal.name in {attachment.AttachFailedEvent.name, attachment.DetachedEvent.name}
        )

    model: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Ability",
        hsm.initial(hsm.target("/Ability/detached")),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    attachment.Attachment._remember_attachment_timeout,
                    attachment.Attachment._queue_attach_complete,
                ),
                hsm.target("/Ability/attaching"),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attach_request),
                hsm.effect(attachment.Attachment._dispatch_attach_failed),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_detach_request),
                hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
            ),
        ),
        hsm.state(
            "attaching",
            hsm.transition(
                hsm.on(attachment.AttachCompleteEvent),
                hsm.effect(attachment.Attachment._deliver_attach_complete),
                hsm.target("/Ability/attached"),
            ),
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(attachment.Attachment._timeout_attachment),
                hsm.target("/Ability/detached"),
            ),
        ),
        hsm.state(
            "attached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attached),
                hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._is_attach_request),
                hsm.effect(attachment.Attachment._dispatch_attach_failed),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._detach_last_attachment),
                hsm.effect(attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed),
                hsm.target("/Ability/detached"),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_detach_request),
                hsm.effect(attachment.Attachment._dispatch_detach_failed),
            ),
            hsm.transition(
                hsm.on(TerminalOutputEvent),
                hsm.guard(_has_terminal_event),
                hsm.effect(_forward_terminal_event),
            ),
            hsm.transition(
                hsm.on(TerminalErrorEvent),
                hsm.guard(_has_terminal_event),
                hsm.effect(_forward_terminal_event),
            ),
        ),
        hsm.observe(observer),
    )

    @staticmethod
    def _define_model(
        name: str,
        submodel: hsm.Model,
        *,
        composite_attachment_lifecycle: bool = False,
    ) -> hsm.Model:
        root = f"/{name}Lifecycle"
        attached = f"{root}/attached"
        if composite_attachment_lifecycle:
            attach_effect = hsm.effect(
                attachment.Attachment._attach,
                attachment.Attachment._remember_attachment_timeout,
            )
            attaching: tuple[hsm.Element, ...] = ()
            detach_elements = (hsm.target(f"{attached}/behavior/detaching"),)
            composite_transitions = (
                hsm.transition(
                    hsm.on(_CompositeAttachmentTerminalEvent),
                    hsm.guard(Ability._is_composite_detached_terminal),
                    hsm.effect(Ability._deliver_composite_attachment_terminal),
                    hsm.target(f"{root}/detached"),
                ),
            )
        else:
            attach_effect = hsm.effect(
                attachment.Attachment._attach,
                attachment.Attachment._remember_attachment_timeout,
                attachment.Attachment._queue_attach_complete,
            )
            attaching = (
                hsm.state(
                    "attaching",
                    hsm.transition(
                        hsm.on(attachment.AttachCompleteEvent),
                        hsm.effect(attachment.Attachment._deliver_attach_complete),
                        hsm.target(attached),
                    ),
                    hsm.transition(
                        hsm.after(attachment.Attachment._attachment_timeout_delay),
                        hsm.effect(attachment.Attachment._timeout_attachment),
                        hsm.target(f"{root}/detached"),
                    ),
                ),
            )
            detach_elements = (
                hsm.effect(
                    attachment.Attachment._detach,
                    attachment.Attachment._dispatch_detach_complete_removed,
                ),
                hsm.target(f"{root}/detached"),
            )
            composite_transitions = ()

        return hsm.define(
            f"{name}Lifecycle",
            hsm.initial(hsm.target(f"{root}/detached")),
            hsm.state(
                "detached",
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._can_attach),
                    attach_effect,
                    hsm.target(attached if composite_attachment_lifecycle else f"{root}/attaching"),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attach_request),
                    hsm.effect(attachment.Attachment._dispatch_attach_failed),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._is_detach_request),
                    hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
                ),
            ),
            *attaching,
            hsm.state(
                "attached",
                hsm.initial(hsm.target(f"{attached}/behavior")),
                hsm.submachine_state("behavior", submodel),
                *composite_transitions,
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attached),
                    hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
                ),
                hsm.transition(
                    hsm.on(attachment.AttachEvent),
                    hsm.guard(attachment.Attachment._is_attach_request),
                    hsm.effect(attachment.Attachment._dispatch_attach_failed),
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._detach_last_attachment),
                    *detach_elements,
                ),
                hsm.transition(
                    hsm.on(attachment.DetachEvent),
                    hsm.guard(attachment.Attachment._is_detach_request),
                    hsm.effect(attachment.Attachment._dispatch_detach_failed),
                ),
                hsm.transition(
                    hsm.on(TerminalOutputEvent),
                    hsm.guard(Ability._has_terminal_event),
                    hsm.effect(Ability._forward_terminal_event),
                ),
                hsm.transition(
                    hsm.on(TerminalErrorEvent),
                    hsm.guard(Ability._has_terminal_event),
                    hsm.effect(Ability._forward_terminal_event),
                ),
            ),
            hsm.observe(observer),
        )

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        declared_submodel = cls.__dict__.get("submodel")
        if declared_submodel is None:
            declared_submodel = cls.submodel
        assert declared_submodel is not None, f"{cls.__name__} must define submodel."
        submodel = typing.cast(hsm.Model, declared_submodel)
        cls.submodel = submodel
        cls.model = Ability._define_model(
            cls.__name__,
            submodel,
            composite_attachment_lifecycle=cls._composite_attachment_lifecycle,
        )

    def __init__(self) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._terminal_waiters = {}

    @property
    def attachment_owner(self) -> hsm.Instance | None:
        """First attachment owner (Bot body or host), if any.

        Public surface for host_turn fail-closed checks and owner-targeted delivery without
        reading private ``_attachments`` from outside Ability.
        """

        if not self._attachments:
            return None
        return self._attachments[0]

    @typing.override
    def attach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.AttachData],
    ) -> collections.abc.Awaitable[None]:
        """Start this ability lifecycle and dispatch its attach request."""

        model = self.model

        async def start_and_dispatch() -> None:
            if model is None:
                return
            # hsm 1.3.2+: id fails after stop — restart when not started.
            try:
                _ = hsm.id(self)
            except hsm.ErrorValidatingModel:
                _ = await hsm.started(ctx, self, model)
            _ = await hsm.dispatch(ctx, self, event)

        task = asyncio.Task(
            start_and_dispatch(),
            loop=asyncio.get_running_loop(),
            eager_start=True,
        )
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    @typing.override
    def detach(
        self,
        ctx: hsm.Context,
        event: hsm.Event[attachment.DetachData],
    ) -> collections.abc.Awaitable[None]:
        """Dispatch this ability's detach request."""

        async def detach_or_fail() -> None:
            try:
                _ = hsm.id(self)
            except hsm.ErrorValidatingModel:
                data = event.data
                reply_to = data.reply_to if data.reply_to is not None else data.actor
                if reply_to is not None:
                    try:
                        reply_id = hsm.id(reply_to)
                    except hsm.ErrorValidatingModel:
                        reply_id = ""
                    await hsm.dispatch(
                        ctx,
                        reply_to,
                        dataclasses.replace(
                            attachment.DetachFailedEvent.with_data(
                                attachment.FailedData(
                                    actor=data.actor,
                                    kind=attachment.FailureKind.DISPATCH,
                                    message=f"{type(self).__name__} is stopped or not started; detach refused.",
                                )
                            ),
                            id=event.id,
                            source=event.source or "",
                            target=reply_id,
                            metadata=dict(event.metadata),
                        ),
                    )
                return
            await hsm.dispatch(ctx, self, event)

        task = asyncio.Task(
            detach_or_fail(),
            loop=asyncio.get_running_loop(),
            eager_start=True,
        )
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop this ability and any nested composite attachment group if started."""

        group = getattr(self, "_attachment_group", None)
        await hsm.Instance.stop(self, ctx)
        if not isinstance(group, attachment.Group):
            return
        # Group is started lazily on attach; do not hsm.stop an unstarted machine.
        try:
            _ = hsm.id(group)
        except hsm.ErrorValidatingModel:
            return
        await hsm.stop(group, ctx)

    def apply(self, input: TInput, *, ctx: hsm.Context | None = None) -> collections.abc.Awaitable[None]:
        """Dispatch this ability's input event."""

        return hsm.dispatch(
            self.context() if ctx is None else ctx,
            self,
            self.input_event.with_data_and_id(input, uuid.uuid4().hex),
        )


__all__ = [
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "TerminalErrorEvent",
    "TerminalOutputEvent",
    "Ability",
    "FailureData",
    "TInput",
    "TOutput",
    "ability_input_event",
    "ability_output_event",
]
