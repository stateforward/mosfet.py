"""Communication: ability to communicate by holding Conversations.

Event-driven: activate the engaged Conversation; admit speech and route it to
``_active_conversation``; and route a selected response to the injected Speaking
ability. Durable recall is Memory.
"""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import datetime
import typing
import uuid

import hsm
import pydantic
from opentelemetry import metrics
from pydantic.json_schema import SkipJsonSchema

from mosfet import event
import mosfet
from mosfet.abilities import ability
from mosfet.abilities import cognition
from mosfet.abilities import processing
from mosfet.abilities import speaking
from mosfet.abilities import vocal
from . import conversation
from mosfet.protocols import attachment
from mosfet import telemetry
from mosfet.telemetry import span

Conversation = conversation.Conversation
TurnData = conversation.TurnData

_RESPONSE_IDLE_TIMEOUT = datetime.timedelta(seconds=15)
"""Default longest silence from Speaking on a response; it exceeds Speaking's own longest silence inside one."""

_CLOSED_RESPONSES = 16
"""How many finished response ids Communication remembers, so a late copy of one can never reopen it."""

_HELD_RESPONSES = 16
"""How many other responses Communication holds while one is being spoken; past it, a new one is refused."""

_METER = metrics.get_meter("mosfet.abilities.communication")
_LATE = _METER.create_counter(
    "bot.communication.response.late.count",
    unit="{event}",
    description="Responses or drafts that arrived for a response id Communication had already finished; ignored.",
)


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


DraftEvent = hsm.Event[processing.DraftData](
    name="bot.ability.communication.draft",
    kind=hsm.EventKind,
    schema=processing.DraftData,
)
"""Text newly written into a ``respond`` the model is still writing (an append-only delta), under the selection's id.
Not model-callable: Processing dispatches it while the selection streams, and the final ``respond`` under the same id
follows."""


class RespondData(pydantic.BaseModel):
    """Semantic response selected by cognition for the current communication.

    Communication currently delivers this action through its injected Speaking ability. The
    selected payload is deliberately text-only: cognition chooses to respond, while Speaking
    owns speech encoding and playout. ``text`` is marked for drafting: while a model is still writing
    it, Processing hands it here as :data:`DraftEvent`, so speech can start before the response
    is complete. The marker is not part of the schema the model sees.
    """

    __draft__: typing.ClassVar[processing.Draft] = processing.Draft(field="text", event=DraftEvent)

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


class StopData(pydantic.BaseModel):
    """Stop the response being spoken now."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Stop the response being spoken now. What has already been heard stays said; the rest of the "
                "response is not said. Offered only while a response is being spoken, and it takes no arguments."
            ),
            "examples": [{}],
        },
    )


StopEvent = hsm.Event[StopData](
    name="bot.ability.communication.stop",
    kind=event.EventKind,
    schema=StopData,
)
"""Stop the response being spoken; offered only while Communication is ``responding``."""

FailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.communication.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
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
    _attachment_group: attachment.Group | None
    """Route communication ingress to the active Conversation; forward its products.

    Topology (behavior under Ability lifecycle ``attached``):

    - ``inactive``: attaching the active Conversation composite (or after End).
    - ``active``: route :data:`InputEvent` to ``_active_conversation`` and accept
      :data:`RespondEvent` for the configured Speaking route; forward terminals.
    - ``responding``: the response's text has gone to the injected Speaking ability as
      ``speaking.input`` (or, while the model is still writing it, ``speaking.draft``, each
      append-only delta forwarded unchanged) under the response id; drafts and the final ``respond``
      for that id keep being forwarded, and Communication forwards Speaking's terminal for that id to
      its owner. The ``respond`` for another response is held (bounded) until this one ends, then
      handled in order; its drafts are not held, because the ``respond`` carries all of their text. ``waiting`` re-enters on every ``speaking.progress`` for the response, so a long
      spoken reply is never cut off; ``response_idle_timeout`` of silence from Speaking ends it as
      failed and stops it (``speaking.stop`` for the response id), so Speaking does not go on saying a
      response whose failure was already reported. :data:`StopEvent` is enabled only here, so it is offered only while a response is being
      spoken; it reaches Speaking as ``speaking.stop`` for the response id, and Speaking's terminal
      (outcome ``stopped``) ends the response as any other. Leaving ``responding`` by any path retires the
      response id into a bounded ring of finished ids, and a ``respond`` for a finished id is ignored and
      counted.

    Construction is keyword-only (``active_conversation=``, ``conversations=``, ``speaking=``,
    ``response_idle_timeout=``). ``speaking`` is a required composition dependency and is never
    discovered through the actor graph.
    """

    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    input_event: typing.ClassVar[hsm.Event[TurnData]] = InputEvent
    _active_conversation: Conversation
    _conversations: list[Conversation]
    _speaking: speaking.Speaking
    _response_idle_timeout: datetime.timedelta
    # Durable: the id of the response being spoken ("" outside ``responding``), and the ids of the last
    # responses that finished, so a late copy of one can never reopen it.
    _response_id: str
    _closed_responses: collections.deque[str]
    # Durable: the respond of each other response that arrived while one was being spoken, in arrival order;
    # released one at a time as Communication becomes free.
    _held_responses: collections.OrderedDict[str, hsm.Event[typing.Any]]
    failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = FailedEvent

    @staticmethod
    def _attachment_owner_id(instance: "Communication") -> str:
        """Address of the attached owner (outer requester) surfaces respond terminals to it."""

        if not instance._attachments:
            return ""
        return hsm.id(instance._attachments[0])

    def __init__(
        self,
        *,
        active_conversation: Conversation | None = None,
        conversations: collections.abc.Sequence[Conversation] | None = None,
        speaking: speaking.Speaking,
        response_idle_timeout: datetime.timedelta = _RESPONSE_IDLE_TIMEOUT,
    ) -> None:
        """Build Communication.

        Args:
            active_conversation: The Conversation made active first.
            conversations: Conversations Communication may activate.
            speaking: The Speaking ability responses are said through.
            response_idle_timeout: The longest Speaking may go without a sign of progress on a response before
                Communication gives up on it. It must exceed Speaking's own longest silence inside an utterance,
                including how long it waits for more drafted text (Speaking's ``draft_timeout``).

        Raises:
            TypeError: ``speaking`` is not a Speaking ability.
            ValueError: No conversation is given, or ``response_idle_timeout`` is not positive.
        """

        super().__init__()
        if not _is_speaking_ability(speaking):
            raise TypeError("Communication speaking must be a Speaking ability.")
        if response_idle_timeout <= datetime.timedelta(0):
            raise ValueError(f"Communication response_idle_timeout must be positive, got {response_idle_timeout}.")
        catalog, chosen = _normalize_catalog(
            active_conversation=active_conversation,
            conversations=conversations,
        )
        self._conversations = catalog
        self._active_conversation = chosen
        self._speaking = speaking
        self._response_idle_timeout = response_idle_timeout
        self._response_id = ""
        self._closed_responses = collections.deque(maxlen=_CLOSED_RESPONSES)
        self._held_responses = collections.OrderedDict()
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
            parent = mosfet.StimulusData[TurnData](
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

    # ----- responding: Communication forwards the response to Speaking and follows its progress ---------------

    @staticmethod
    def _is_response_text(event: hsm.Event[typing.Any]) -> bool:
        """A respond, or a draft of one with some newly written text: the two ways a response's text arrives."""

        data = event.data
        if isinstance(data, RespondData):
            return True
        return isinstance(data, processing.DraftData) and data.event == RespondEvent.name and bool(data.text)

    @staticmethod
    def _is_new_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if not Communication._is_response_text(event):
            return False
        if isinstance(event.data, processing.DraftData) and not event.id:
            # A draft is only ever correlated with its final respond by id.
            return False
        return not event.id or event.id not in instance._closed_responses

    @staticmethod
    def _is_late_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return Communication._is_response_text(event) and bool(event.id) and event.id in instance._closed_responses

    @staticmethod
    def _is_current_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return Communication._is_response_text(event) and bool(event.id) and event.id == instance._response_id

    @staticmethod
    def _is_other_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        return Communication._is_new_response(ctx, instance, event) and event.id != instance._response_id

    @staticmethod
    def _count_late_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _LATE.add(1)

    @staticmethod
    def _forward_text(
        ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any], response_id: str
    ) -> None:
        """Hand the response's text to Speaking under the response id: a draft as a draft, the final as the input."""

        data = event.data
        target = instance._speaking
        forwarded: hsm.Event[typing.Any]
        if isinstance(data, RespondData):
            forwarded = speaking.InputEvent.with_data(speaking.InputData(parts=(vocal.speech.Part(words=data.text),)))
        else:
            assert isinstance(data, processing.DraftData)
            # The delta goes on unchanged: what was newly written into the one part, at the offset it was written at.
            forwarded = speaking.DraftEvent.with_data(speaking.DraftData(part=0, words=data.text, offset=data.offset))
        with span.operation(
            "bot.communication.response.dispatch",
            scope="bot.abilities.communication",
            component="communication",
            stage="response_dispatch",
            context=telemetry.event_context(event),
        ):
            _ = hsm.dispatch(
                ctx,
                target,
                dataclasses.replace(
                    forwarded,
                    id=response_id,
                    source=hsm.id(instance),
                    target=hsm.id(target),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _open_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> None:
        """Hold the response's id for as long as it is being spoken, and hand the text written so far to Speaking."""

        response_id = event.id or uuid.uuid4().hex
        instance._response_id = response_id
        _ = instance._held_responses.pop(response_id, None)
        Communication._forward_text(ctx, instance, event, response_id)

    @staticmethod
    def _continue_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> None:
        """More of the response being spoken, or its final text: on to Speaking under the same id."""

        Communication._forward_text(ctx, instance, event, instance._response_id)

    @staticmethod
    def _hold_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> None:
        """Another response waits its turn: its respond is kept. Its drafts are not: the respond carries all of their
        text, and nothing of that response can be said before its turn."""

        if not isinstance(event.data, RespondData):
            return
        if event.id in instance._held_responses:
            instance._held_responses[event.id] = event
            return
        if len(instance._held_responses) >= _HELD_RESPONSES:
            failure = ability.FailureData(message="Communication is holding too many responses; this one is refused.")
            with span.operation(
                "bot.communication.response.terminal",
                scope="bot.abilities.communication",
                component="communication",
                stage="response_terminal",
            ) as active:
                span.record_failure(active, "overloaded")
                terminal = dataclasses.replace(
                    instance.failed_event.with_data_and_id(failure, event.id),
                    source=hsm.id(instance),
                    target=Communication._attachment_owner_id(instance),
                )
                _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
            return
        instance._held_responses[event.id] = event

    @staticmethod
    def _release_held_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> None:
        """Free again: the first held response is handled now, as if it had just arrived."""

        del event
        if not instance._held_responses:
            return
        _, held = instance._held_responses.popitem(last=False)
        _ = hsm.dispatch(ctx, instance, held)

    @staticmethod
    def _has_stop(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, StopData)

    @staticmethod
    def _stop_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> None:
        """Ask Speaking to stop the response being spoken, by the response id; its terminal follows as usual."""

        target = instance._speaking
        with span.operation(
            "bot.communication.response.stop",
            scope="bot.abilities.communication",
            component="communication",
            stage="response_stop",
            context=telemetry.event_context(event),
        ):
            _ = hsm.dispatch(
                ctx,
                target,
                dataclasses.replace(
                    speaking.StopEvent.with_data(speaking.StopData()),
                    id=instance._response_id,
                    source=hsm.id(instance),
                    target=hsm.id(target),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _close_response(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> None:
        """Every way out of ``responding`` retires the response id, so nothing late can match or reopen it."""

        del ctx, event
        if instance._response_id:
            instance._closed_responses.append(instance._response_id)
        instance._response_id = ""

    @staticmethod
    def _is_from_speaking(instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        """Post-delivery correlation: Speaking's reply for the response being spoken, addressed here."""

        return (
            bool(instance._response_id)
            and event.id == instance._response_id
            and event.source == hsm.id(instance._speaking)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _is_response_output(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, speaking.OutputData) and Communication._is_from_speaking(instance, event)

    @staticmethod
    def _is_response_failure(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, ability.FailureData) and Communication._is_from_speaking(instance, event)

    @staticmethod
    def _is_response_progress(ctx: hsm.Context, instance: "Communication", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, speaking.ProgressData) and Communication._is_from_speaking(instance, event)

    @staticmethod
    def _response_idle_delay(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> datetime.timedelta:
        del ctx, event
        return instance._response_idle_timeout

    @staticmethod
    def _dispatch_response_timeout(
        ctx: hsm.Context,
        instance: "Communication",
        event: hsm.Event[typing.Any],
    ) -> None:
        """Give up on the response: Speaking is told to stop it, and the owner is told it failed."""

        Communication._stop_response(ctx, instance, event)
        failure = ability.FailureData(message="Speaking response timed out.")
        with span.operation(
            "bot.communication.response.terminal",
            scope="bot.abilities.communication",
            component="communication",
            stage="response_terminal",
        ) as active:
            span.record_failure(active, "timeout")
            terminal = dataclasses.replace(
                instance.failed_event.with_data_and_id(failure, instance._response_id),
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

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
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
            hsm.entry(_release_held_response),
            # Admit / route speech (and other conversation-shaped input) to the active talk.
            hsm.transition(
                hsm.on(InputEvent),
                hsm.guard(_has_communication_input),
                hsm.effect(_route_to_active_conversation),
            ),
            hsm.transition(
                hsm.on(RespondEvent, DraftEvent),
                hsm.guard(_is_new_response),
                hsm.effect(_open_response),
                hsm.target("/Communication/responding"),
            ),
            hsm.transition(
                hsm.on(RespondEvent, DraftEvent),
                hsm.guard(_is_late_response),
                hsm.effect(_count_late_response),
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
            "responding",
            hsm.exit(_close_response),
            # Heard speech waits until this response is said or ends. Other responses are held explicitly rather
            # than deferred: this response's own drafts and final text must always be taken here.
            hsm.defer(InputEvent),
            hsm.initial(hsm.target("waiting")),
            hsm.transition(
                hsm.on(RespondEvent, DraftEvent),
                hsm.guard(_is_other_response),
                hsm.effect(_hold_response),
            ),
            hsm.transition(
                hsm.on(RespondEvent, DraftEvent),
                hsm.guard(_is_late_response),
                hsm.effect(_count_late_response),
            ),
            hsm.transition(
                hsm.on(speaking.OutputEvent),
                hsm.guard(_is_response_output),
                hsm.effect(_forward_response_output),
                hsm.target("/Communication/active"),
            ),
            hsm.transition(
                hsm.on(ability.FailedEvent),
                hsm.guard(_is_response_failure),
                hsm.effect(_forward_response_failure),
                hsm.target("/Communication/active"),
            ),
            hsm.state(
                "waiting",
                # Re-entering restarts the silence deadline: the response is still being written or said.
                hsm.transition(
                    hsm.on(RespondEvent, DraftEvent),
                    hsm.guard(_is_current_response),
                    hsm.effect(_continue_response),
                    hsm.target("/Communication/responding/waiting"),
                ),
                hsm.transition(
                    hsm.on(speaking.ProgressEvent),
                    hsm.guard(_is_response_progress),
                    hsm.target("/Communication/responding/waiting"),
                ),
                hsm.transition(
                    hsm.on(StopEvent),
                    hsm.guard(_has_stop),
                    hsm.effect(_stop_response),
                    hsm.target("/Communication/responding/waiting"),
                ),
                hsm.transition(
                    hsm.after(_response_idle_delay),
                    hsm.effect(_dispatch_response_timeout),
                    hsm.target("/Communication/responding/expired"),
                ),
            ),
            # Giving up passes through here, inside ``responding``, so the effect above still has the response
            # id it gives up on: leaving ``responding`` retires it.
            hsm.choice("expired", hsm.transition(hsm.target("/Communication/active"))),
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
    )


__all__ = [
    "ActivateData",
    "ActivateEvent",
    "Communication",
    "DraftEvent",
    "InputEvent",
    "RespondData",
    "RespondEvent",
    "FailedEvent",
    "StopData",
    "StopEvent",
]
