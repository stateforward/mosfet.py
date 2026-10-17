"""``Associative``: what memory surfaces for a turn, and what comes to mind when the bot recalls one.

Cognition sends ``AssociateEvent`` (``CueData``: the memory rendering of the stimulus and who is
present) and gets one reply: ``AssociatedEvent`` (``AssociationsData``, best first) or
``ability.FailedEvent``. Associative asks its peer ``Memory`` for candidates by message
(``memory.SearchEvent``: nearest by vector, recent, involving someone present, one link away),
scores each with the injected ``Weights``, and admits them best first while their text fits
``budget_tokens``. It never embeds and never reads Memory's state; Memory is only an address.

Scoring is pure and deterministic::

    score = w.similarity * cosine + w.recency * 0.5 ** (age / w.recency_half_life)
            + w.present * [involves someone present] + w.linked * link_weight

ties broken by newer first, then node id. A candidate with neither text nor images carries
nothing to associate and is skipped.

``RecallEvent`` (``RecallData``) is the bot's deliberate recall, a model-callable tool offered
while Associative is idle. Its effort is ``M``: a deliberate stage (Reasoning) is offered it, the
reflex stage (Intuition) is not. Recall by ``node`` reads that memory in full; recall ``about``
something searches with those words, ranks the candidates with the same ``Weights``, and reads the
best ``limit`` in full. Either way Memory reads them with image bytes and reinforces them, and
the owner Associative is attached to (Cognition, in a bot) is told with ``RecollectedEvent``
(``RecollectionData``) or a typed ``ability.FailedEvent``: an observation of the bot's own
thought, not an answer to the stage that selected it. Image bytes ride the newest memories, at
most ``media.MAX_TURN_IMAGES`` in all.

Every addressed request gets exactly one reply: its own (an association, a recollection, or a
typed failure), ``timeout`` after ``timeout``, or ``unavailable`` when Associative is detached,
detaching, or degraded (requests that wait behind the one in flight are answered by
``detaching``; requests after detach by the base lifecycle).

Telemetry (scope ``bot.abilities.memory``; every attribute value comes from code):

* ``bot.memory.association.count`` — ``bot.outcome`` ∈ {ok, failed, timeout}: 3 series.
* ``bot.memory.association.items`` / ``bot.memory.association.omitted`` — histograms, no attributes.
* ``bot.memory.recall.count`` — ``bot.memory.recall.mode`` ∈ {about, node} × ``bot.outcome`` ∈
  {ok, failed}: 4 series.
* ``bot.ability.request.dropped.count`` — requests answered while detaching or degraded.

Spans ``bot.memory.associate`` and ``bot.memory.recall`` (component ``memory.associative``) carry
only bounded counts.
"""

import collections.abc
import dataclasses
import datetime
import math
import time
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics, trace

from mosfet import event, media
from mosfet.protocols import attachment
from mosfet.telemetry import span

from .. import ability
from ..ability import Effort
from ..language import text
from . import memory, payload

_DEFAULT_TIMEOUT: typing.Final = datetime.timedelta(seconds=10)

_METER = metrics.get_meter("bot.abilities.memory")
_ASSOCIATIONS = _METER.create_counter(
    "bot.memory.association.count",
    unit="{association}",
    description="Association requests answered, by outcome.",
)
_ITEMS = _METER.create_histogram(
    "bot.memory.association.items",
    unit="{item}",
    description="Memories one association admitted.",
)
_OMITTED = _METER.create_histogram(
    "bot.memory.association.omitted",
    unit="{item}",
    description="Memories one association surfaced but left out to fit its token budget.",
)
_RECALLS = _METER.create_counter(
    "bot.memory.recall.count",
    unit="{recall}",
    description="Deliberate recalls answered, by how the memory was named and outcome.",
)

AssociateEvent = hsm.Event[payload.CueData](name="bot.ability.memory.associate", schema=payload.CueData)
"""Associate from a cue; answered with ``AssociatedEvent`` or ``ability.FailedEvent``."""
AssociatedEvent = hsm.Event[payload.AssociationsData](
    name="bot.ability.memory.associated", schema=payload.AssociationsData
)
RecallEvent = hsm.Event[payload.RecallData](
    name="bot.ability.memory.recall",
    kind=event.EventKind,
    schema=payload.RecallData,
)
"""Think about a memory deliberately (model-callable); the owner is told with ``RecollectedEvent`` or a failure."""
RecollectedEvent = hsm.Event[payload.RecollectionData](
    name="bot.ability.memory.recollected", schema=payload.RecollectionData
)
"""What came to mind on a deliberate recall, sent to the owner as an observation."""

_Recall = typing.Literal["about", "node"]


class _CompletedData(pydantic.BaseModel):
    """The reply an association produced, correlated to its request by the event id."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    reply: hsm.Event[typing.Any]


class _SettledData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


_CompletedEvent = hsm.Event[_CompletedData](
    name="bot.ability.memory.associative.completed", kind=hsm.CompletionEventKind, schema=_CompletedData
)
_SettledEvent = hsm.Event[_SettledData](
    name="bot.ability.memory.associative.settled", kind=hsm.CompletionEventKind, schema=_SettledData
)


@dataclasses.dataclass(frozen=True)
class _InFlight:
    """The request being served (durable machine data while ``busy``)."""

    id: str
    requester: str
    metadata: dict[str, typing.Any]
    started: float
    # How a deliberate recall named its memory; None for an association.
    recall: _Recall | None = None
    answered: bool = False


def _released(image: media.Image | media.ReleasedImage) -> media.ReleasedImage:
    return image if isinstance(image, media.ReleasedImage) else media.ReleasedImage.of(image, reason="memory")


def _score(candidate: payload.Candidate, *, weights: payload.Weights, now: datetime.datetime) -> float:
    """One candidate's association score under ``weights`` at ``now`` (pure)."""

    age = max(0.0, (now - candidate.node.created_at).total_seconds())
    recency = math.pow(0.5, age / weights.recency_half_life.total_seconds())
    return (
        weights.similarity * (candidate.similarity or 0.0)
        + weights.recency * recency
        + weights.present * (1.0 if candidate.present else 0.0)
        + weights.linked * (candidate.link_weight or 0.0)
    )


def _ranked(found: payload.FoundData, *, weights: payload.Weights, now: datetime.datetime) -> list[payload.Candidate]:
    """Candidates that carry something, best first: by score, then newer, then node id."""

    scored = sorted(
        (
            (_score(candidate, weights=weights, now=now), candidate)
            for candidate in found.candidates
            if candidate.node.text.strip() or (candidate.node.episode is not None and candidate.node.episode.images)
        ),
        key=lambda entry: (-entry[0], -entry[1].node.created_at.timestamp(), entry[1].node.node_id),
    )
    return [candidate for _, candidate in scored]


def _association(candidate: payload.Candidate) -> payload.Association:
    """One candidate as an association: its images as memory placeholders, who it involves."""

    node = candidate.node
    episode = node.episode
    return payload.Association(
        ref=node.node_id,
        kind=node.kind,
        text=node.text,
        images=() if episode is None else tuple(_released(image) for image in episode.images),
        at=node.created_at,
        lanes=candidate.lanes,
        involves=()
        if episode is None
        else tuple(dict.fromkeys(participant.ref for participant in episode.participants)),
    )


def _associations(
    found: payload.FoundData,
    *,
    weights: payload.Weights,
    budget_tokens: int,
    count_tokens: collections.abc.Callable[[str], int],
    now: datetime.datetime,
) -> payload.AssociationsData:
    """Rank candidates by score and admit them, best first, while their text fits ``budget_tokens``."""

    items: list[payload.Association] = []
    used = 0
    omitted = 0
    for candidate in _ranked(found, weights=weights, now=now):
        tokens = count_tokens(candidate.node.text)
        if len(items) >= payload.MAX_ASSOCIATIONS or used + tokens > budget_tokens:
            omitted += 1
            continue
        used += tokens
        items.append(_association(candidate))
    return payload.AssociationsData(items=tuple(items), omitted=omitted)


def _recollection(about: str | None, fetched: payload.FetchedData) -> payload.RecollectionData:
    """What came to mind: the memories read in full, image bytes on the newest ones within the turn bound."""

    remaining = media.MAX_TURN_IMAGES
    bounded: dict[str, payload.Node] = {}
    for node in sorted(fetched.nodes, key=lambda read: read.created_at, reverse=True):
        episode = node.episode
        if episode is None or node.node_id in bounded:
            continue
        images: list[media.Image | media.ReleasedImage] = []
        for image in episode.images:
            if isinstance(image, media.Image) and remaining > 0:
                images.append(image)
                remaining -= 1
            else:
                images.append(_released(image))
        bounded[node.node_id] = node.model_copy(
            update={"episode": episode.model_copy(update={"images": tuple(images)})}
        )
    return payload.RecollectionData(
        about=about,
        nodes=tuple(bounded.get(node.node_id, node) for node in fetched.nodes),
        neighbors=tuple(_association(candidate) for candidate in fetched.neighbors[: payload.MAX_ASSOCIATIONS]),
    )


def _recall_mode(request: hsm.Event[typing.Any]) -> _Recall | None:
    """How a request names a memory to recall; None when its payload is not a recall."""

    data = request.data
    if not isinstance(data, payload.RecallData):
        return None
    return "node" if data.node is not None else "about"


def _count(recall: _Recall | None, outcome: typing.Literal["ok", "failed", "timeout"]) -> None:
    if recall is None:
        _ASSOCIATIONS.add(1, {"bot.outcome": outcome})
    else:
        _RECALLS.add(1, {"bot.memory.recall.mode": recall, "bot.outcome": "ok" if outcome == "ok" else "failed"})


class Associative(ability.Ability[payload.CueData, payload.AssociationsData]):
    """Surface what memory associates with a turn, and bring a memory to mind when the bot recalls it.

    ``memory`` is a peer ``Memory`` reached only by message. ``budget_tokens`` bounds the text
    the associations carry, counted by ``count_tokens``. ``weights`` and ``search`` (how many
    candidates each lane offers) are capacities of the composition, shared by associations and
    recalls. ``timeout`` bounds one association or recall end to end.

    Raises:
        ValueError: if ``budget_tokens`` or ``timeout`` is not positive.
    """

    # Recall (deliberate thought about a memory) is deliberate-tier work: never offered on the reflex arc.
    effort: typing.ClassVar[Effort] = Effort.M
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.CueData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.AssociationsData
    input_event: typing.ClassVar[hsm.Event[payload.CueData]] = AssociateEvent
    output_event: typing.ClassVar[hsm.Event[payload.AssociationsData]] = AssociatedEvent
    failed_event: typing.ClassVar[hsm.Event[typing.Any]] = ability.FailedEvent
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = (AssociateEvent, RecallEvent)
    # The composite form gives the submodel its own detaching state, so a request waiting behind
    # the one in flight is answered there instead of being discarded on detach.
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _attachment_group: attachment.Group | None
    _memory: memory.Memory
    _budget_tokens: int
    _count_tokens: collections.abc.Callable[[str], int]
    _weights: payload.Weights
    _search: payload.SearchData
    _now: collections.abc.Callable[[], datetime.datetime]
    _timeout: datetime.timedelta
    # Durable machine data: the request being served.
    _in_flight: _InFlight | None

    @classmethod
    @typing.override
    def detached_failure(cls, event: hsm.Event[typing.Any]) -> pydantic.BaseModel:
        del event
        return ability.FailureData(message=f"unavailable: {cls.__name__} is not attached.")

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        memory: memory.Memory,
        budget_tokens: int = 1024,
        count_tokens: collections.abc.Callable[[str], int] = payload.approximate_tokens,
        weights: payload.Weights = payload.Weights(),
        search: payload.SearchData = payload.SearchData(cue=()),
        now: collections.abc.Callable[[], datetime.datetime] | None = None,
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
    ) -> None:
        if budget_tokens <= 0:
            raise ValueError("Associative budget_tokens must be positive.")
        if timeout <= datetime.timedelta(0):
            raise ValueError("Associative timeout must be positive.")
        super().__init__(id=id, clock=clock)
        self._attachment_group = attachment.Group()
        self._memory = memory
        self._budget_tokens = budget_tokens
        self._count_tokens = count_tokens
        self._weights = weights
        self._search = search
        self._now = now if now is not None else (lambda: datetime.datetime.now(datetime.UTC))
        self._timeout = timeout
        self._in_flight = None

    # --- replies -------------------------------------------------------------------------------

    @staticmethod
    def _requester(instance: "Associative", event: hsm.Event[typing.Any]) -> str:
        """Where a reply goes, by payload: a recall's outcome to the owner (""); any other to its requester."""

        if isinstance(event.data, payload.RecallData):
            return ""
        return event.source if event.source and event.source != hsm.id(instance) else ""

    @staticmethod
    def _send(
        ctx: hsm.Context,
        instance: "Associative",
        *,
        request_id: str,
        requester: str,
        metadata: dict[str, typing.Any],
        reply: hsm.Event[typing.Any],
    ) -> None:
        """Deliver one reply: to the requester's address, or to the owner when the request had none."""

        terminal = dataclasses.replace(
            reply, id=request_id or None, source=hsm.id(instance), target=requester, metadata=dict(metadata)
        )
        if requester:
            _ = hsm.dispatch_to(ctx, terminal, requester)
            return
        carrier = (
            ability.TerminalErrorEvent if isinstance(reply.data, ability.FailureData) else ability.TerminalOutputEvent
        )
        _ = hsm.dispatch(ctx, instance, carrier.with_data(terminal))

    @staticmethod
    def _failure(message: str) -> hsm.Event[ability.FailureData]:
        return ability.FailedEvent.with_data(ability.FailureData(message=message))

    @staticmethod
    def _reply_now(
        ctx: hsm.Context, instance: "Associative", request: hsm.Event[typing.Any], reply: hsm.Event[typing.Any]
    ) -> None:
        Associative._send(
            ctx,
            instance,
            request_id=request.id,
            requester=Associative._requester(instance, request),
            metadata=dict(request.metadata),
            reply=reply,
        )
        _count(_recall_mode(request), "failed")

    @staticmethod
    def _reply_unavailable_detaching(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        Associative._reply_now(ctx, instance, event, Associative._failure("unavailable: Associative is detaching."))
        ability.Ability._count_dropped(instance, reason="detaching", replied=True)

    @staticmethod
    def _reply_unavailable_degraded(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        Associative._reply_now(ctx, instance, event, Associative._failure("unavailable: Associative is degraded."))
        ability.Ability._count_dropped(instance, reason="degraded", replied=True)

    @staticmethod
    def _reply_invalid(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        Associative._reply_now(
            ctx,
            instance,
            event,
            Associative._failure(f"invalid_input: {event.name} carried {type(event.data).__name__}."),
        )

    # --- busy ----------------------------------------------------------------------------------

    @staticmethod
    def _carries_cue(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, payload.CueData)

    @staticmethod
    def _carries_recall(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, payload.RecallData)

    @staticmethod
    def _begin(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._in_flight = _InFlight(
            id=event.id,
            requester=Associative._requester(instance, event),
            metadata=dict(event.metadata),
            started=time.monotonic(),  # telemetry only; hsm.after owns the deadline
            recall=_recall_mode(event),
        )

    @staticmethod
    def _answer(
        ctx: hsm.Context,
        instance: "Associative",
        reply: hsm.Event[typing.Any],
        outcome: typing.Literal["ok", "failed", "timeout"],
    ) -> None:
        """Send the in-flight request's one reply and record it answered."""

        in_flight = instance._in_flight
        assert in_flight is not None and not in_flight.answered
        instance._in_flight = dataclasses.replace(in_flight, answered=True)
        Associative._send(
            ctx,
            instance,
            request_id=in_flight.id,
            requester=in_flight.requester,
            metadata=in_flight.metadata,
            reply=reply,
        )
        _count(in_flight.recall, outcome)
        if isinstance(reply.data, payload.AssociationsData):
            _ITEMS.record(len(reply.data.items))
            _OMITTED.record(reply.data.omitted)

    @staticmethod
    def _settle(ctx: hsm.Context, instance: "Associative") -> None:
        in_flight = instance._in_flight
        assert in_flight is not None
        _ = hsm.dispatch(ctx, instance, _SettledEvent.with_data_and_id(_SettledData(), in_flight.id))

    @staticmethod
    def _is_current(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        in_flight = instance._in_flight
        return in_flight is not None and event.id == in_flight.id and not in_flight.answered

    @staticmethod
    def _is_unanswered(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._in_flight is not None and not instance._in_flight.answered

    @staticmethod
    def _is_settled(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        in_flight = instance._in_flight
        return in_flight is not None and event.id == in_flight.id and in_flight.answered

    @staticmethod
    def _complete(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _CompletedData)
        Associative._answer(
            ctx, instance, data.reply, "failed" if isinstance(data.reply.data, ability.FailureData) else "ok"
        )
        Associative._settle(ctx, instance)

    @staticmethod
    def _timeout_delay(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._timeout

    @staticmethod
    def _time_out(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        del event
        seconds = instance._timeout.total_seconds()
        in_flight = instance._in_flight
        assert in_flight is not None
        served = "association" if in_flight.recall is None else "recollection"
        Associative._answer(
            ctx, instance, Associative._failure(f"timeout: no {served} within {seconds:g}s."), "timeout"
        )
        Associative._settle(ctx, instance)

    @staticmethod
    def _abandon(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        """Leaving busy: a request still unanswered (detach or stop) is answered unavailable."""

        del event
        if instance._in_flight is not None and not instance._in_flight.answered:
            Associative._answer(
                ctx, instance, Associative._failure("unavailable: Associative stopped serving the request."), "failed"
            )
        instance._in_flight = None

    @staticmethod
    async def _ask(
        ctx: hsm.Context,
        instance: "Associative",
        active: trace.Span,
        request: hsm.Event[typing.Any],
        answer: hsm.Event[typing.Any],
    ) -> object:
        """Send ``request`` to Memory by message; return its ``answer`` data, or the typed failure."""

        try:
            terminal = await ability.run_terminal_operation(
                ctx,
                child=instance._memory,
                request=request.with_data_and_id(request.data, uuid.uuid4().hex),
                terminals=(answer, memory.FailedEvent),
                timeout=instance._timeout,
            )
        except TimeoutError:
            return ability.FailureData(message=f"timeout: memory did not answer {request.name}.")
        except RuntimeError as error:
            return ability.FailureData(message=f"unavailable: memory cannot be reached: {error}")
        data = terminal.data
        if isinstance(data, payload.FailureData):
            span.record_failure(active, data.kind.value)
            return ability.FailureData(message=f"{data.kind.value}: {data.message}")
        return data

    @staticmethod
    async def _associate(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        """Ask Memory for candidates by message, score them, and dispatch the correlated reply."""

        data = event.data
        assert isinstance(data, payload.CueData)
        with span.operation(
            "bot.memory.associate", scope="bot.abilities.memory", component="memory.associative", stage="associate"
        ) as active:
            reply: hsm.Event[typing.Any]
            found = await Associative._ask(
                ctx,
                instance,
                active,
                memory.SearchEvent.with_data(
                    instance._search.model_copy(update={"cue": data.cue, "participants": data.participants})
                ),
                memory.FoundEvent,
            )
            if isinstance(found, payload.FoundData):
                surfaced = _associations(
                    found,
                    weights=instance._weights,
                    budget_tokens=instance._budget_tokens,
                    count_tokens=instance._count_tokens,
                    now=instance._now(),
                )
                active.set_attribute("bot.memory.candidates", len(found.candidates))
                active.set_attribute("bot.memory.items", len(surfaced.items))
                reply = AssociatedEvent.with_data(surfaced)
            else:
                assert isinstance(found, ability.FailureData)
                reply = ability.FailedEvent.with_data(found)
            _ = instance.dispatch(ctx, _CompletedEvent.with_data_and_id(_CompletedData(reply=reply), event.id))

    @staticmethod
    async def _recall(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        """Bring the named memory, or the best ones about something, to mind in full; dispatch the reply."""

        data = event.data
        assert isinstance(data, payload.RecallData)
        with span.operation(
            "bot.memory.recall", scope="bot.abilities.memory", component="memory.associative", stage="recall"
        ) as active:
            reply: hsm.Event[typing.Any] | None = None
            node_ids: tuple[str, ...] = ()
            if data.node is not None:
                node_ids = (data.node,)
            else:
                assert data.about is not None
                found = await Associative._ask(
                    ctx,
                    instance,
                    active,
                    memory.SearchEvent.with_data(
                        instance._search.model_copy(
                            update={"cue": (text.TextPart(text=data.about),), "participants": ()}
                        )
                    ),
                    memory.FoundEvent,
                )
                if isinstance(found, payload.FoundData):
                    active.set_attribute("bot.memory.candidates", len(found.candidates))
                    ranked = _ranked(found, weights=instance._weights, now=instance._now())
                    node_ids = tuple(candidate.node.node_id for candidate in ranked[: data.limit])
                    if not node_ids:
                        reply = RecollectedEvent.with_data(payload.RecollectionData(about=data.about))
                else:
                    assert isinstance(found, ability.FailureData)
                    reply = ability.FailedEvent.with_data(found)
            if reply is None:
                fetched = await Associative._ask(
                    ctx,
                    instance,
                    active,
                    memory.FetchEvent.with_data(payload.FetchData(node_ids=node_ids, media=True, reinforce=True)),
                    memory.FetchedEvent,
                )
                if isinstance(fetched, payload.FetchedData):
                    active.set_attribute("bot.memory.items", len(fetched.nodes))
                    reply = RecollectedEvent.with_data(_recollection(data.about, fetched))
                else:
                    assert isinstance(fetched, ability.FailureData)
                    reply = ability.FailedEvent.with_data(fetched)
            _ = instance.dispatch(ctx, _CompletedEvent.with_data_and_id(_CompletedData(reply=reply), event.id))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Associative",
        hsm.initial(hsm.target("/Associative/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, AssociateEvent, RecallEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Associative/idle"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                # Unreachable with the empty group; kept so the composite lifecycle's targets exist.
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Associative/degraded"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(AssociateEvent), hsm.guard(_carries_cue), hsm.target("/Associative/busy/associating")
            ),
            # Offered to a deliberate frame from here: recall is model-callable only while idle.
            hsm.transition(hsm.on(RecallEvent), hsm.guard(_carries_recall), hsm.target("/Associative/busy/recalling")),
            hsm.transition(hsm.on(AssociateEvent, RecallEvent), hsm.effect(_reply_invalid)),
        ),
        hsm.state(
            "busy",
            hsm.entry(_begin),
            hsm.exit(_abandon),
            hsm.defer(AssociateEvent, RecallEvent),
            hsm.transition(hsm.after(_timeout_delay), hsm.guard(_is_unanswered), hsm.effect(_time_out)),
            hsm.transition(hsm.on(_CompletedEvent), hsm.guard(_is_current), hsm.effect(_complete)),
            hsm.transition(hsm.on(_SettledEvent), hsm.guard(_is_settled), hsm.target("/Associative/idle")),
            hsm.state("associating", hsm.activity(_associate)),
            hsm.state("recalling", hsm.activity(_recall)),
        ),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(hsm.on(AssociateEvent, RecallEvent), hsm.effect(_reply_unavailable_detaching)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Associative/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Associative/idle"),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.transition(hsm.on(AssociateEvent, RecallEvent), hsm.effect(_reply_unavailable_degraded)),
        ),
    )


__all__ = [
    "AssociateEvent",
    "AssociatedEvent",
    "Associative",
    "RecallEvent",
    "RecollectedEvent",
]
