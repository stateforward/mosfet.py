"""Communication: ability to communicate by holding Conversations.

Event-driven: activate the engaged Conversation; admit speech and route it to
``_active_conversation``; and route a selected response to the injected Speaking
ability. Durable recall is Memory.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot import event
import bot
from bot.abilities import ability
from bot.abilities import cognition
from bot.abilities import processing
from bot.abilities import speaking
from . import conversation
from bot.protocols import attachment
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span

Conversation = conversation.Conversation
TurnData = conversation.TurnData

_RESPONSE_TIMEOUT = datetime.timedelta(seconds=6)


class ActivateData(pydantic.BaseModel):
    """Select which Conversation is active for attach and routing.

    The instance is kept in the local engagement set if not already present.
    Activation is refused while Communication is engaged (behavior active).
    Future: hosts load a Conversation via Memory, then activate it here.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    conversation: SkipJsonSchema[Conversation] = pydantic.Field(
        description="Conversation instance to make active.",
    )


ActivateEvent = hsm.Event[ActivateData](
    name="bot.ability.communication.activate",
    kind=event.EventKind,
    schema=ActivateData,
)
# Speech/text admit into Communication; routed to the active Conversation.
InputEvent = hsm.Event[TurnData](
    name="bot.ability.communication.input",
    kind=hsm.EventKind,
    schema=TurnData,
)


class RespondData(pydantic.BaseModel):
    """Semantic response selected by cognition for the current communication.

    Communication currently delivers this action through its injected Speaking ability. The
    selected payload is deliberately text-only: cognition chooses to respond, while Speaking
    owns synthesis and playout.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A response to communicate aloud. Communication routes the text through its injected "
                "Speaking ability; cognition does not select Speaking directly."
            ),
            "examples": [{"text": "I am doing well, thank you."}],
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description=(
            "The complete non-blank text to communicate. It is spoken verbatim by the configured "
            "Speaking ability after this action is selected."
        ),
        examples=["I am doing well, thank you."],
    )

    @pydantic.field_validator("text")
    @classmethod
    def require_nonblank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Communication response text must be non-blank.")
        return text


RespondEvent = hsm.Event[RespondData](
    name="bot.ability.communication.respond",
    kind=event.EventKind,
    schema=RespondData,
)

FailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.communication.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class _PreparedResponseData(pydantic.BaseModel):
    """A normalized response operation ready for Speaking delivery."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    response: RespondData


_PreparedResponseEvent = hsm.Event[_PreparedResponseData](
    name="bot.ability.communication.response.prepared",
    schema=_PreparedResponseData,
)
_ResponseDispatchedEvent = hsm.Event[_PreparedResponseData](
    name="bot.ability.communication.response.dispatched",
    schema=_PreparedResponseData,
)
_ResponseHopTimedOutEvent = hsm.Event[object](
    name="bot.ability.communication.response.hop.timed_out",
    kind=hsm.ErrorEventKind,
    schema=object,
)


def _has_conversation_product(
    ctx: hsm.Context,
    instance: "Communication",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    if isinstance(event.data, cognition.InputData):
        return True
    if isinstance(event.data, conversation.Messages):
        return bool(event.data.messages) and event.data.messages[-1].direction == "inbound"
    return False


def _has_conversation_failure(
    ctx: hsm.Context,
    instance: "Communication",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, conversation.FailureData)


def _has_activate(
    ctx: hsm.Context,
    instance: "Communication",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, ActivateData)


def _has_communication_input(
    ctx: hsm.Context,
    instance: "Communication",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, TurnData)


def _has_respond(
    ctx: hsm.Context,
    instance: "Communication",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, RespondData)


def _is_speaking_ability(value: object) -> bool:
    return isinstance(value, speaking.Speaking)


def _normalize_catalog(
    *,
    active_conversation: Conversation | None,
    conversations: collections.abc.Sequence[Conversation] | None,
) -> tuple[list[Conversation], Conversation]:
    """Build engagement set + required active conversation for initial attach wiring."""

    catalog: list[Conversation] = []
    if conversations is not None:
        for item in conversations:
            if item not in catalog:
                catalog.append(item)
    if active_conversation is not None and active_conversation not in catalog:
        catalog.insert(0, active_conversation)

    chosen = active_conversation
    if chosen is None and catalog:
        chosen = catalog[0]
    if chosen is None:
        raise ValueError("Communication requires active_conversation= or a non-empty conversations= catalog.")
    if chosen not in catalog:
        catalog.insert(0, chosen)
    return catalog, chosen


class Communication(ability.Ability[TurnData, object]):
    """Route communication ingress to the active Conversation; forward its products.

    Topology (behavior under Ability lifecycle ``attached``):

    - ``inactive``: attaching the active Conversation composite (or after End).
    - ``active``: route :data:`InputEvent` to ``_active_conversation`` and accept
      :data:`RespondEvent` for the configured Speaking route; forward terminals.
    - ``responding``: await the injected Speaking ability's typed completion or failure.

    Construction is keyword-only (``active_conversation=``, ``conversations=``, ``speaking=``).
    ``speaking`` is a required composition dependency and is never discovered through the actor
    graph.
    """

    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    input_event: typing.ClassVar[hsm.Event[TurnData]] = InputEvent
    _active_conversation: Conversation
    _conversations: list[Conversation]
    _speaking: speaking.Speaking
    failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = FailedEvent

    @staticmethod
    def _attachment_owner_id(instance: "Communication") -> str:
        """Address of the attached owner (outer requester) surfaces respond terminals to it."""

        if not instance._attachments:
            return ""
        return hsm.id(instance._attachments[0])

    @staticmethod
    def _has_response_completed(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, speaking.OutputData)
            and bool(event.id)
            and event.source == hsm.id(instance._speaking)
            and event.target == hsm.id(instance)
            and (
                processing.active_operation(instance, event.id) is not None
                or processing.active_operation_id(instance) is None
            )
        )

    @staticmethod
    def _has_response_failure(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, ability.FailureData)
            and bool(event.id)
            and event.source == hsm.id(instance._speaking)
            and event.target == hsm.id(instance)
            and (
                processing.active_operation(instance, event.id) is not None
                or processing.active_operation_id(instance) is None
            )
        )

    def __init__(
        self,
        *,
        active_conversation: Conversation | None = None,
        conversations: collections.abc.Sequence[Conversation] | None = None,
        speaking: speaking.Speaking,
    ) -> None:
        super().__init__()
        if not _is_speaking_ability(speaking):
            raise TypeError("Communication speaking must be a Speaking ability.")
        catalog, chosen = _normalize_catalog(
            active_conversation=active_conversation,
            conversations=conversations,
        )
        self._conversations = catalog
        self._active_conversation = chosen
        self._speaking = speaking
        self._attachment_group = attachment.Group(chosen)

    @staticmethod
    def _activate(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ActivateData)
        if data.conversation not in instance._conversations:
            instance._conversations.append(data.conversation)
        if data.conversation is instance._active_conversation:
            return
        instance._active_conversation = data.conversation
        instance._attachment_group = attachment.Group(data.conversation)

    @staticmethod
    def _route_to_active_conversation(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Route admit/input to the active Conversation (lookup/swap later)."""

        with span.operation(
            "bot.communication.route",
            scope="bot.abilities.communication",
            component="communication",
            stage="conversation_route",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            assert isinstance(data, TurnData)
            target = instance._active_conversation
            parent = bot.StimulusData[TurnData](
                event=event.name,
                data=data,
                id=event.id,
                source=event.source,
                target=event.target,
            )
            # Speaker identity is what a conversation turns into participants; whether the product
            # arrived carrying any is the difference between a routed turn and a dropped one.
            active.set_attribute("bot.identity.source.count", len(data.source_ids))
            active.set_attribute("bot.identity.target.count", len(data.target_ids))
            active.set_attribute("bot.content.type", data.content_type or "")
            routed = dataclasses.replace(
                conversation.RoutedInputEvent.with_data(conversation.RoutedInputData(parent=parent)),
                id=event.id or None,
                source=hsm.id(instance),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            )
            _ = hsm.dispatch(ctx, target, routed)

    @staticmethod
    async def _prepare_response(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Mint one unique response operation and carry it into the routing state."""

        data = event.data
        assert isinstance(data, RespondData)
        operation_id = event.id or uuid.uuid4().hex
        _ = await processing.start_operation(instance, operation_id)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _PreparedResponseEvent.with_data_and_id(_PreparedResponseData(response=data), operation_id),
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _route_response_to_speaking(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Move one prepared response into ``responding`` where the speaking hop is awaited."""

        data = event.data
        assert isinstance(data, _PreparedResponseData)
        operation_id = event.id
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _ResponseDispatchedEvent.with_data_and_id(data, operation_id),
                source=hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _await_speaking_terminal(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Wait on the speaking hop through the settled one-shot terminal operation.

        The operation actor becomes the request source, so Speaking echoes its correlated
        terminal back to the reply actor, which settles it here. The settled terminal is then
        re-dispatched into the same typed continuations (source=speaking, target=owner) that
        ``responding`` already exposes, so _has_response_completed / _has_response_failure ->
        _forward_response_* run unchanged. Nothing runs after the state-transition-causing
        re-dispatch, so the activity is never cancelled mid-work. The modeled ``responding``
        deadline (hsm.after(_response_timeout_delay)) remains the hop timeout, abandoning only
        a hop that has not yet settled its terminal.
        """

        data = event.data
        assert isinstance(data, _PreparedResponseData)
        operation_id = event.id
        hop_id = operation_id
        with span.operation(
            "bot.communication.response.dispatch",
            scope="bot.abilities.communication",
            component="communication",
            stage="response_dispatch",
            context=telemetry.event_context(event),
        ):
            if hop_id and processing.active_operation(instance, hop_id) is None:
                _ = await processing.start_operation(instance, hop_id)
            try:
                terminal = await ability.run_terminal_operation(
                    instance.context(),
                    child=instance._speaking,
                    request=dataclasses.replace(
                        instance._speaking.input_event.with_data_and_id(
                            speaking.InputData(text=data.response.text), operation_id
                        ),
                        metadata=dict(event.metadata),
                    ),
                    terminals=(
                        instance._speaking.output_event,
                        instance._speaking.failed_event,
                    ),
                    timeout=_RESPONSE_TIMEOUT,
                    on_terminal=lambda _terminal: (
                        processing.finish_operation(ctx, instance, hop_id) if hop_id else None
                    ),
                )
            except TimeoutError:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    dataclasses.replace(
                        _ResponseHopTimedOutEvent.with_data_and_id(None, hop_id),
                        source=hsm.id(instance),
                        target=hsm.id(instance),
                        metadata=dict(event.metadata),
                    ),
                )
                return
        if hop_id:
            processing.finish_operation(ctx, instance, hop_id)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                terminal,
                source=hsm.id(instance._speaking),
                target=hsm.id(instance),
            ),
        )

    @staticmethod
    def _has_active_response(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return (
            isinstance(event.data, _PreparedResponseData)
            and bool(event.id)
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and processing.active_operation(instance, event.id) is not None
        )

    @staticmethod
    def _response_timeout_delay(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, instance, event
        return _RESPONSE_TIMEOUT

    @staticmethod
    def _speaking_deadline_still_pending(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> bool:
        """Whether the speaking hop is still pending on this machine's token."""

        del ctx, event
        hop_id = processing.active_operation_id(instance)
        return hop_id is not None and processing.active_operation(instance, hop_id) is not None

    @staticmethod
    def _is_response_hop_timeout(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> bool:
        del ctx
        return event.source == hsm.id(instance) and event.target == hsm.id(instance)

    @staticmethod
    def _dispatch_response_timeout(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        del event
        operation_id = processing.active_operation_id(instance)
        if operation_id is None:
            return
        processing.finish_operation(ctx, instance, operation_id)
        failure = ability.FailureData(message="Speaking response timed out.")
        with span.operation(
            "bot.communication.response.terminal",
            scope="bot.abilities.communication",
            component="communication",
            stage="response_terminal",
        ) as active:
            span.record_failure(active, "response_timeout")
            terminal = dataclasses.replace(
                instance.failed_event.with_data_and_id(failure, operation_id),
                source=hsm.id(instance),
                target=Communication._attachment_owner_id(instance),
            )
            _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _forward_response_output(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, speaking.OutputData)
        processing.finish_operation(ctx, instance, event.id)
        with span.operation(
            "bot.communication.response.terminal",
            scope="bot.abilities.communication",
            component="communication",
            stage="response_terminal",
            context=telemetry.event_context(event),
        ):
            terminal = dataclasses.replace(
                instance._speaking.output_event.with_data(data),
                id=event.id or None,
                source=hsm.id(instance._speaking),
                target=Communication._attachment_owner_id(instance),
                metadata=dict(event.metadata),
            )
            _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _forward_response_failure(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        processing.finish_operation(ctx, instance, event.id)
        with span.operation(
            "bot.communication.response.terminal",
            scope="bot.abilities.communication",
            component="communication",
            stage="response_terminal",
            context=telemetry.event_context(event),
        ) as active:
            span.record_failure(active, "speaking_failed")
            terminal = dataclasses.replace(
                instance.failed_event.with_data(data),
                id=event.id or None,
                source=hsm.id(instance),
                target=Communication._attachment_owner_id(instance),
                metadata=dict(event.metadata),
            )
            _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _forward_conversation_product(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Re-emit Conversation contribution products through this ability's terminal.

        Contribute the conversation product as a cognition.InputEvent handoff (text products)
        or a bare Messages terminal to the attached owner (outer requester) so the ability
        lifecycle surfaces it to the owner object directly. The inner event carries no target
        so ``_forward_terminal_event`` dispatches to the owner object rather than resolving the
        owner by id through an Instances map (Communication runs under a private scope). With
        no attached owner the contribution is dropped.
        """

        routed = dataclasses.replace(event, target=None)
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(routed))

    @staticmethod
    def _forward_conversation_failure(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        routed = dataclasses.replace(event, target=None)
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(routed))

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "Communication",
        hsm.initial(hsm.target("/Communication/inactive")),
        hsm.state(
            "inactive",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, InputEvent),
            hsm.transition(
                hsm.on(ActivateEvent),
                hsm.guard(_has_activate),
                hsm.effect(_activate),
            ),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Communication/active"),
            ),
        ),
        hsm.state(
            "active",
            # Admit / route speech (and other conversation-shaped input) to the active talk.
            hsm.transition(
                hsm.on(InputEvent),
                hsm.guard(_has_communication_input),
                hsm.effect(_route_to_active_conversation),
            ),
            hsm.transition(
                hsm.on(RespondEvent),
                hsm.guard(_has_respond),
                hsm.target("/Communication/preparing_response"),
            ),
            # Conversation contribution → body cognition (Listening-style terminal re-emit).
            hsm.transition(
                hsm.on(cognition.InputEvent),
                hsm.guard(_has_conversation_product),
                hsm.effect(_forward_conversation_product),
            ),
            hsm.transition(
                hsm.on(conversation.OutputEvent),
                hsm.guard(_has_conversation_product),
                hsm.effect(_forward_conversation_product),
            ),
            hsm.transition(
                hsm.on(conversation.FailedEvent),
                hsm.guard(_has_conversation_failure),
                hsm.effect(_forward_conversation_failure),
            ),
        ),
        hsm.state(
            "preparing_response",
            hsm.defer(InputEvent, RespondEvent),
            hsm.activity(_prepare_response),
            hsm.transition(
                hsm.on(_PreparedResponseEvent),
                hsm.guard(_has_active_response),
                hsm.target("/Communication/dispatching_response"),
            ),
        ),
        hsm.state(
            "dispatching_response",
            hsm.defer(InputEvent, RespondEvent),
            hsm.activity(_route_response_to_speaking),
            hsm.transition(
                hsm.on(_ResponseDispatchedEvent),
                hsm.guard(_has_active_response),
                hsm.target("/Communication/responding"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_has_response_failure),
                hsm.effect(_forward_response_failure),
                hsm.target("/Communication/active"),
            ),
            hsm.transition(
                hsm.on(speaking.OutputEvent),
                hsm.guard(_has_response_completed),
                hsm.effect(_forward_response_output),
                hsm.target("/Communication/active"),
            ),
        ),
        hsm.state(
            "responding",
            hsm.defer(InputEvent, RespondEvent),
            hsm.activity(_await_speaking_terminal),
            hsm.transition(
                hsm.on(speaking.OutputEvent),
                hsm.guard(_has_response_completed),
                hsm.effect(_forward_response_output),
                hsm.target("/Communication/active"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_has_response_failure),
                hsm.effect(_forward_response_failure),
                hsm.target("/Communication/active"),
            ),
            hsm.transition(
                hsm.after(_response_timeout_delay),
                hsm.guard(_speaking_deadline_still_pending),
                hsm.effect(_dispatch_response_timeout),
                hsm.target("/Communication/active"),
            ),
            hsm.transition(
                hsm.on(_ResponseHopTimedOutEvent),
                hsm.guard(_is_response_hop_timeout),
                hsm.effect(_dispatch_response_timeout),
                hsm.target("/Communication/active"),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Communication/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Communication/active"),
            ),
        ),
        hsm.state("degraded"),
        hsm.observe(observer),
    )


__all__ = [
    "ActivateData",
    "ActivateEvent",
    "Communication",
    "InputEvent",
    "RespondData",
    "RespondEvent",
    "FailedEvent",
]
