import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import bot
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot import lifecycle
from bot import scope
from bot.protocols import attachment
from bot.telemetry import observer


TInput = typing.TypeVar("TInput")
TOutput = typing.TypeVar("TOutput")
_DataType = type[object] | tuple[type[object], ...] | None


def _private_instance_scope(parent: hsm.Context) -> hsm.Context:
    """Create an explicitly addressed scope for private one-shot attachment actors.

    The registry is strong (compare bot._private_scope's weak registry): attachment groups
    must stay alive in the map for the whole attach/detach operation.
    """

    values: dict[typing.Hashable, object] = {hsm.Keys.Instances: {}}
    return hsm.Context(parent=parent, values=scope.mark_private(values))


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

        return bot.define(
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


InputEvent = hsm.Event[object](
    name="bot.ability.input",
    schema=object,
)
# Intentional generic-envelope exception to the typed-payload rule: base ``Ability`` is
# generic over ``TInput`` / ``TOutput``, so no single Pydantic schema can describe its
# input/output. Concrete subclasses narrow ``input_event`` / ``output_event`` with real
# schemas, and runtime payload validation happens through ``input_data_type`` /
# ``output_data_type`` — never by sniffing these envelopes. The same exception covers the
# generic input/output/apply-completed events in decoding, classifying, generative, and
# encoding, which subclass this envelope the same way.
OutputEvent = hsm.Event[object](
    name="bot.ability.output",
    schema=object,
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
# Self-dispatched by an ability that cannot recover on its own. Carries the built reboot event so
# the reason stays typed in the domain that owns it; Ability lifecycle forwards it to the owner.
RebootRequestEvent = hsm.Event[hsm.Event[typing.Any]](
    name="bot.ability.reboot.request",
    schema=hsm.Event[typing.Any],
)


class Ability(hsm.Instance, attachment.Attachment, typing.Generic[TInput, TOutput]):
    """Specific event-driven operational capacity.

    Answers: What operations can the system perform?

    Terminal emission:
        When this ability emits a terminal output or failure event, delivery
        is target-first and at most once. A set (truthy) ``target`` is
        delivered once to that address and is never also delivered to the
        attachment owner. An absent or empty ``target`` is delivered once to
        the first attachment owner, or dropped when no owner is attached.
        The emitted event's ``source`` is rewritten to this child; a
        caller-supplied source is not preserved.
    """

    _attachment_limit: typing.ClassVar[int | None] = 1
    _attachments: list[hsm.Instance]
    _attachment_group: attachment.Group | None = None
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
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

    @staticmethod
    def _carries_event(
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
        """Deliver one child terminal under the Ability terminal emission postcondition."""

        terminal = event.data
        assert isinstance(terminal, hsm.Event)
        child_id = hsm.id(instance)
        routed = dataclasses.replace(
            terminal,
            source=child_id,
            metadata=dict(terminal.metadata),
        )
        if routed.target:
            _ = hsm.dispatch_to(ctx, routed, routed.target)
            return
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                routed,
                target=hsm.id(owner),
            ),
        )

    @staticmethod
    def _forward_reboot_to_owner(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Forward a self-requested reboot to this ability's attachment owner.

        No owner attached means nothing can act on the request, so nothing is sent.
        """

        reboot = event.data
        assert isinstance(reboot, hsm.Event)
        if not instance._attachments:
            return
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                reboot,
                id=reboot.id or event.id or uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
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
        group = instance._attachment_group
        assert isinstance(group, attachment.Group)
        reply: hsm.Instance | None = None
        source: hsm.Instance = group
        try:
            private_scope = _private_instance_scope(instance.context())
            # Start the group when it is not live yet, then attach members.
            if not lifecycle.is_started(group):
                try:
                    _ = await bot.started(private_scope, group, group.model)
                except Exception:
                    source = instance
                    raise
            reply = await instance._start_composite_attachment_reply(
                source,
                request,
                event,
            )
            await group.attach(
                private_scope,
                dataclasses.replace(
                    attachment.AttachEvent.with_data(
                        attachment.AttachData(actor=instance, reply_to=reply, timeout=request.timeout)
                    ),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(group),
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
        group = instance._attachment_group
        assert isinstance(group, attachment.Group)
        reply: hsm.Instance | None = None
        try:
            private_scope = _private_instance_scope(instance.context())
            reply = await instance._start_composite_attachment_reply(
                group,
                request,
                event,
            )
            await group.detach(
                private_scope,
                dataclasses.replace(
                    attachment.DetachEvent.with_data(
                        attachment.DetachData(actor=instance, reply_to=reply, timeout=request.timeout)
                    ),
                    id=event.id,
                    source=hsm.id(instance),
                    target=hsm.id(group),
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
                    group,
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
                    source=hsm.id(group),
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
        terminal_data = terminal.data
        target = request.actor if request.reply_to is None else request.reply_to
        # Classify by typed payload + request type (no event.name routing, HSM-DELIVERY-001).
        if isinstance(request, attachment.AttachData) and isinstance(terminal_data, attachment.AttachCompleteData):
            public = attachment.AttachCompleteEvent.with_data(
                attachment.AttachCompleteData(actor=request.actor, created=True)
            )
        elif isinstance(request, attachment.DetachData) and isinstance(terminal_data, attachment.DetachedData):
            index = instance._attachment_index(request.actor)
            assert index is not None
            del instance._attachments[index]
            public = attachment.DetachedEvent.with_data(attachment.DetachedData(actor=request.actor, removed=True))
        elif isinstance(terminal_data, attachment.FailedData):
            if isinstance(request, attachment.AttachData):
                index = instance._attachment_index(request.actor)
                assert index is not None
                del instance._attachments[index]
                public = attachment.AttachFailedEvent.with_data(
                    attachment.FailedData(actor=request.actor, kind=terminal_data.kind, message=terminal_data.message)
                )
            else:
                assert isinstance(request, attachment.DetachData)
                public = attachment.DetachFailedEvent.with_data(
                    attachment.FailedData(actor=request.actor, kind=terminal_data.kind, message=terminal_data.message)
                )
        else:
            raise AssertionError(
                f"unsupported composite attachment terminal: {type(request).__name__} + {type(terminal_data).__name__}"
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
        private_scope = _private_instance_scope(self.context())
        reply = await bot.started(
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
            and isinstance(event.data.request, attachment.AttachData)
            and isinstance(event.data.terminal.data, attachment.AttachCompleteData)
        )

    @staticmethod
    def _is_composite_detach_failed(
        ctx: hsm.Context,
        instance: "Ability[typing.Any, typing.Any]",
        event: hsm.Event[typing.Any],
    ) -> bool:
        return Ability._is_valid_composite_attachment_terminal(ctx, instance, event) and (
            isinstance(event.data, _CompositeAttachmentTerminalData)
            and isinstance(event.data.request, attachment.DetachData)
            and isinstance(event.data.terminal.data, attachment.FailedData)
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
            and isinstance(event.data.request, attachment.DetachData)
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
        if not Ability._is_valid_composite_attachment_terminal(ctx, instance, event):
            return False
        if not isinstance(event.data, _CompositeAttachmentTerminalData):
            return False
        request = event.data.request
        terminal_data = event.data.terminal.data
        return (isinstance(request, attachment.AttachData) and isinstance(terminal_data, attachment.FailedData)) or (
            isinstance(request, attachment.DetachData) and isinstance(terminal_data, attachment.DetachedData)
        )

    model: typing.ClassVar[hsm.Model | None] = bot.define(
        "Ability",
        hsm.initial(hsm.target("/Ability/detached")),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(
                    attachment.Attachment._attach,
                    attachment.Attachment._remember_attachment_request,
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
                hsm.guard(_carries_event),
                hsm.effect(_forward_terminal_event),
            ),
            hsm.transition(
                hsm.on(TerminalErrorEvent),
                hsm.guard(_carries_event),
                hsm.effect(_forward_terminal_event),
            ),
            hsm.transition(
                hsm.on(RebootRequestEvent),
                hsm.guard(_carries_event),
                hsm.effect(_forward_reboot_to_owner),
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
        initial_state: typing.Literal["detached", "attached"] = "detached",
    ) -> hsm.Model:
        root = f"/{name}Lifecycle"
        attached = f"{root}/attached"
        if composite_attachment_lifecycle:
            attach_effect = hsm.effect(
                attachment.Attachment._attach,
                attachment.Attachment._remember_attachment_request,
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
                attachment.Attachment._remember_attachment_request,
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

        return bot.define(
            f"{name}Lifecycle",
            hsm.initial(hsm.target(f"{root}/{initial_state}")),
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
                    hsm.guard(Ability._carries_event),
                    hsm.effect(Ability._forward_terminal_event),
                ),
                hsm.transition(
                    hsm.on(TerminalErrorEvent),
                    hsm.guard(Ability._carries_event),
                    hsm.effect(Ability._forward_terminal_event),
                ),
                hsm.transition(
                    hsm.on(RebootRequestEvent),
                    hsm.guard(Ability._carries_event),
                    hsm.effect(Ability._forward_reboot_to_owner),
                ),
            ),
            hsm.observe(observer),
        )

    @classmethod
    def define_lifecycle_model(
        cls,
        name: str,
        submodel: hsm.Model,
        *,
        composite_attachment_lifecycle: bool | None = None,
    ) -> hsm.Model:
        """Build the public lifecycle wrapper for an ability behavior model."""

        if composite_attachment_lifecycle is None:
            composite_attachment_lifecycle = cls._composite_attachment_lifecycle
        return Ability._define_model(
            name,
            submodel,
            composite_attachment_lifecycle=composite_attachment_lifecycle,
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
        self._attachment_request_id = ""

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
            # Restart when not live.
            if not lifecycle.is_started(self):
                _ = await bot.started(ctx, self, model)
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
            if not lifecycle.is_started(self):
                data = event.data
                assert isinstance(data, attachment.DetachData)
                reply_to = data.reply_to if data.reply_to is not None else data.actor
                if reply_to is not None:
                    reply_id = hsm.id(reply_to) if lifecycle.is_started(reply_to) else ""
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

        group = self._attachment_group
        await hsm.Instance.stop(self, ctx)
        if not isinstance(group, attachment.Group):
            return
        # Group is started lazily on attach; do not hsm.stop an unstarted machine.
        if not lifecycle.is_started(group):
            return
        await hsm.stop(group, ctx)

    def apply(self, input: TInput, *, ctx: hsm.Context | None = None) -> collections.abc.Awaitable[None]:
        """Dispatch this ability's input event."""

        return hsm.dispatch(
            self.context() if ctx is None else ctx,
            self,
            self.input_event.with_data_and_id(input, uuid.uuid4().hex),
        )


class _TerminalOperation(hsm.Instance):
    """One-shot reply address; all operation data is closure-bound in its model."""


async def run_terminal_operation(
    ctx: hsm.Context,
    *,
    child: Ability[typing.Any, typing.Any],
    request: hsm.Event[typing.Any],
    terminals: tuple[hsm.Event[typing.Any], ...],
    timeout: datetime.timedelta,
    on_terminal: collections.abc.Callable[[hsm.Event[typing.Any]], None] | None = None,
) -> hsm.Event[typing.Any]:
    """Dispatch one request and await its correlated terminal through a one-shot HSM.

    ``request`` remains the canonical domain event. This function only addresses its envelope:
    the operation actor becomes ``source``, ``child`` becomes ``target``, and ``request.id`` is
    preserved for correlation. The child must return one of ``terminals`` with the same id,
    ``source=hsm.id(child)``, and ``target`` set to the operation actor.

    The caller must supply a started ``child`` with an HSM addressing scope. The operation actor
    shares that address map so the child can deliver its terminal directly, while its lifetime is
    parented to ``ctx`` rather than the child's lifetime.

    Child terminal emission follows the Ability terminal postcondition: a set
    target is delivered once to that address and never also to the owner; an
    absent or empty target is delivered once to the attachment owner or
    dropped; source is rewritten to the emitting child.

    The result awaitable is invocation-local and captured by the operation model; it is never
    stored on an HSM instance or in a shared registry. Caller cancellation stops only this reply
    actor. A child whose domain supports cancellation must still be sent its declared typed
    cancellation event by the owning caller; work already dispatched may otherwise continue and
    emit a late terminal, which cannot settle this or any later operation.

    Raises:
        ValueError: If the request id, terminal set, or timeout is invalid.
        RuntimeError: If ``child`` is not started in an addressable HSM scope.
        TimeoutError: If no correlated terminal arrives before the modeled HSM timeout.
    """

    operation_id = request.id
    if not operation_id:
        raise ValueError("Terminal operation request.id is required.")
    if not terminals:
        raise ValueError("Terminal operation requires at least one terminal event.")
    if timeout < datetime.timedelta(0):
        raise ValueError("Terminal operation timeout cannot be negative.")

    instances = child.context().value(hsm.Keys.Instances)
    if not isinstance(instances, collections.abc.MutableMapping):
        raise RuntimeError("Terminal operation child must be started in an addressable HSM scope.")
    child_id = hsm.id(child)
    operation_scope = hsm.Context(
        parent=ctx,
        values={hsm.Keys.Instances: instances},
    )
    result: asyncio.Future[hsm.Event[typing.Any]] = asyncio.get_running_loop().create_future()

    def correlated(
        operation_ctx: hsm.Context,
        instance: _TerminalOperation,
        event: hsm.Event[typing.Any],
    ) -> bool:
        del operation_ctx
        return event.id == operation_id and event.source == child_id and event.target == hsm.id(instance)

    def settle(
        operation_ctx: hsm.Context,
        instance: _TerminalOperation,
        event: hsm.Event[typing.Any],
    ) -> None:
        del operation_ctx, instance
        if on_terminal is not None:
            on_terminal(event)
        if not result.done():
            result.set_result(event)

    def timeout_delay(
        operation_ctx: hsm.Context,
        instance: _TerminalOperation,
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del operation_ctx, instance, event
        return timeout

    def expire(
        operation_ctx: hsm.Context,
        instance: _TerminalOperation,
        event: hsm.Event[typing.Any],
    ) -> None:
        del operation_ctx, instance, event
        if not result.done():
            result.set_exception(TimeoutError("Ability terminal operation timed out."))

    operation_model = bot.define(
        "AbilityTerminalOperation",
        hsm.initial(hsm.target("waiting")),
        hsm.state(
            "waiting",
            hsm.transition(
                hsm.on(*terminals),
                hsm.guard(correlated),
                hsm.effect(settle),
                hsm.target("/AbilityTerminalOperation/completed"),
            ),
            hsm.transition(
                hsm.after(timeout_delay),
                hsm.effect(expire),
                hsm.target("/AbilityTerminalOperation/timed_out"),
            ),
        ),
        hsm.final("completed"),
        hsm.final("timed_out"),
        hsm.observe(observer),
    )
    operation = _TerminalOperation()
    try:
        _ = await bot.started(operation_scope, operation, operation_model)
        await hsm.dispatch(
            ctx,
            child,
            dataclasses.replace(
                request,
                source=hsm.id(operation),
                target=child_id,
                metadata=dict(request.metadata),
            ),
        )
        return await result
    finally:
        await asyncio.shield(hsm.stop(operation, hsm.Context()))


__all__ = [
    "FailedEvent",
    "InputEvent",
    "OutputEvent",
    "RebootRequestEvent",
    "TerminalErrorEvent",
    "TerminalOutputEvent",
    "Ability",
    "FailureData",
    "TInput",
    "TOutput",
    "run_terminal_operation",
]
