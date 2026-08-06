"""Communication: ability to communicate by holding Conversations.

Event-driven: activate the engaged Conversation; admit speech and route it to
``_active_conversation`` (future: lookup then route). Durable recall is Memory.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot import event_schema
from bot import events
from bot.abilities import ability
from bot.abilities import cognition
from .conversation import conversation as conversation_module
from bot.protocols import attachment
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span

Conversation = conversation_module.Conversation
ConversationInputData = conversation_module.ConversationInputData


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
    kind=event_schema.EventKind,
    schema=ActivateData,
)
# Speech/text admit into Communication; routed to the active Conversation.
InputEvent = hsm.Event[ConversationInputData](
    name="bot.ability.communication.input",
    kind=event_schema.EventKind,
    schema=ConversationInputData,
)


def _has_conversation_product(
    ctx: hsm.Context,
    instance: "Communication",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, cognition.InputData) or isinstance(event.data, conversation_module.Response)


def _has_conversation_failure(
    ctx: hsm.Context,
    instance: "Communication",
    event: hsm.Event[typing.Any],
) -> bool:
    del ctx, instance
    return isinstance(event.data, conversation_module.FailureData)


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
    return isinstance(event.data, ConversationInputData)


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


class Communication(ability.Ability[ConversationInputData, object]):
    """Route communication ingress to the active Conversation; forward its products.

    Topology (behavior under Ability lifecycle ``attached``):

    - ``inactive``: attaching the active Conversation composite (or after End).
    - ``active``: route :data:`InputEvent` to ``_active_conversation``; forward terminals.

    Construction is keyword-only (``active_conversation=``, ``conversations=``).
    """

    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    input_event: typing.ClassVar[hsm.Event[ConversationInputData]] = InputEvent
    _active_conversation: Conversation
    _conversations: list[Conversation]
    _attachment_group: attachment.Group

    def __init__(
        self,
        *,
        active_conversation: Conversation | None = None,
        conversations: collections.abc.Sequence[Conversation] | None = None,
    ) -> None:
        super().__init__()
        catalog, chosen = _normalize_catalog(
            active_conversation=active_conversation,
            conversations=conversations,
        )
        self._conversations = catalog
        self._active_conversation = chosen
        self._attachment_group = attachment.Group(chosen)

    def nested_actors(self) -> collections.abc.Mapping[str, hsm.Instance]:
        """Host composition: flatten active Conversation into Bot dispatch actors."""

        return {"conversation": self._active_conversation}

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
            assert isinstance(data, ConversationInputData)
            target = instance._active_conversation
            parent = events.StimulusData[ConversationInputData](
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
                conversation_module.RoutedInputEvent.with_data(
                    conversation_module.RoutedInputData(input=data, parent=parent)
                ),
                id=event.id or None,
                source=hsm.id(instance),
                target=hsm.id(target),
                metadata=dict(event.metadata),
            )
            _ = hsm.dispatch(ctx, target, routed)

    @staticmethod
    def _forward_conversation_product(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Re-emit Conversation contribution products through this ability's terminal."""

        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(event))

    @staticmethod
    def _forward_conversation_failure(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(event))

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
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
            # Conversation contribution → body cognition (Listening-style terminal re-emit).
            hsm.transition(
                hsm.on(cognition.InputEvent),
                hsm.guard(_has_conversation_product),
                hsm.effect(_forward_conversation_product),
            ),
            hsm.transition(
                hsm.on(conversation_module.OutputEvent),
                hsm.guard(_has_conversation_product),
                hsm.effect(_forward_conversation_product),
            ),
            hsm.transition(
                hsm.on(conversation_module.FailedEvent),
                hsm.guard(_has_conversation_failure),
                hsm.effect(_forward_conversation_failure),
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
]
