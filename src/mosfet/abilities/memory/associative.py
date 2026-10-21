"""``Associative``: what memory surfaces for a turn, and what comes to mind when the bot recalls one.

Cognition sends ``AssociateEvent`` (``CueData``: the memory rendering of the stimulus and who is
present) and gets one reply: ``AssociatedEvent`` (``AssociationsData``, best first) or
``ability.FailedEvent``. Associative asks its peer ``Memory`` for candidates by message
(``memory.SearchEvent``: nearest by vector, recent, involving someone present, one link away),
scores each with the injected ``Weights``, and admits them best first while their text fits
``budget_tokens``. It never embeds and never reads Memory's state; Memory is only an address.
The association also says who is here, as Memory resolved it in the search: each participant of
the cue and the person it is (``known`` false for someone nobody has named), those
persons, and the bot itself (``you``, with its own refs). Those facts are what the bot is offered;
whether to ask who someone is, or to do anything at all, stays the bot's choice.

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

Knowing people is remembering (R-12, D-R-41). ``KnowEvent`` (``KnowData``: a fact ``about`` a ref
or a person: ``same_as`` someone, ``name``, or ``not_same_as`` someone) and ``ForgetEvent``
(``ForgetData``: a person) are the bot's model-callable capacity to remember who people are, offered
while Associative is idle, at effort ``M`` like recall. Choosing either is the bot's: a voice nobody
has named is a fact the associations show (``known`` false), never a reason Associative acts. Memory
keeps the person index from those facts (``memory.PersonEvent``) and refuses what would make one ref
two persons' or change the bot itself (``conflict``), or names nobody it holds (``not_found``). The
owner is told what memory now holds with ``KnownEvent`` (``KnownData``) or a typed refusal
(``memory.FailureData`` on ``ability.FailedEvent``): an observation the bot perceives on a new turn.

The voice and face prints follow what memory holds. ``registries`` are the perceiving abilities'
registries, at most one of each kind, reached by message (``identification.LinkEvent`` /
``identification.ForgetEvent``); a ref of a kind with none is memory's only. A print is kept through
retention exactly while its person is named (D-W8-18, D-R-39):

* a fact that leaves ``about``'s person named marks every print of theirs. When a registry refuses
  the mark of a ref the fact moved for good (the id is the bot's own, holds no print, or is no id),
  memory's move is taken back (the ref is not the same as them) and the bot is told the refusal. A
  mark a registry cannot set yet, or refuses for a ref the fact did not move, is answered failed
  with memory's fact standing; knowing it again sets the mark.
* ``not_same_as`` clears the mark of the ref it gave back to nobody;
* ``forget`` erases each print of the person from its registry: erasing is part of forgetting.

Every addressed request gets exactly one reply: its own (an association, a recollection, what
memory now knows, or a typed failure), ``timeout`` when an association or a recall outlives
``timeout`` or a step of a fact (one request to memory or to one registry) does, or ``unavailable`` when
Associative is detached, detaching, or degraded (requests that wait behind the one in flight are
answered by ``detaching``; requests after detach by the base lifecycle).

Telemetry (scope ``bot.abilities.memory``; every attribute value comes from code):

* ``bot.memory.association.count`` — ``bot.outcome`` ∈ {ok, failed, timeout}: 3 series.
* ``bot.memory.association.items`` / ``bot.memory.association.omitted`` — histograms, no attributes.
* ``bot.memory.recall.count`` — ``bot.memory.recall.mode`` ∈ {about, node} × ``bot.outcome`` ∈
  {ok, failed}: 4 series.
* ``bot.memory.know.count`` — ``bot.memory.know.op`` ∈ {know, forget} × ``bot.outcome`` ∈ {ok,
  failed} × ``bot.failure.kind`` (``memory.FailureKind`` or ``none``; ok is always ``none``): at
  most 2 × (1 + 7) = 16 series.
* ``bot.ability.request.dropped.count`` — requests answered while detaching or degraded.

Spans ``bot.memory.associate``, ``bot.memory.recall``, ``bot.memory.know`` and ``bot.memory.forget``
(component ``memory.associative``) carry only bounded counts and failure kinds. A person id, a name,
a ref, a note or a print is never an attribute, a log field or a metric value.
"""

import collections.abc
import dataclasses
import datetime
import logging
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

from .. import ability, identification
from ..ability import Effort
from ..language import text
from . import memory, payload

_LOG = logging.getLogger(__name__)

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
_KNOWN = _METER.create_counter(
    "bot.memory.know.count",
    unit="{fact}",
    description="Facts about people the bot knew or forgot, by what it did, outcome and failure kind.",
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
KnowEvent = hsm.Event[payload.KnowData](name="bot.ability.memory.know", kind=event.EventKind, schema=payload.KnowData)
"""Remember a fact about a person (model-callable); the owner is told with ``KnownEvent`` or a refusal."""
ForgetEvent = hsm.Event[payload.ForgetData](
    name="bot.ability.memory.forget", kind=event.EventKind, schema=payload.ForgetData
)
"""Forget a person and erase their prints (model-callable); the owner is told with ``KnownEvent`` or a refusal."""
KnownEvent = hsm.Event[payload.KnownData](name="bot.ability.memory.known", schema=payload.KnownData)
"""What memory now holds about the person, sent to the owner as an observation."""

_Recall = typing.Literal["about", "node"]
_Served = typing.Literal["about", "node", "know", "forget"]
_REQUESTS: typing.Final[tuple[hsm.Event[typing.Any], ...]] = (AssociateEvent, RecallEvent, KnowEvent, ForgetEvent)
_KINDS: typing.Final[dict[str, payload.FailureKind]] = {kind.value: kind for kind in payload.FailureKind}
# A registry refuses these marks for good: the id is the bot's own, holds no print, or is no id.
_PERMANENT: typing.Final = frozenset(
    {payload.FailureKind.CONFLICT, payload.FailureKind.NOT_FOUND, payload.FailureKind.INVALID_INPUT}
)
_ID = pydantic.TypeAdapter[str](identification.Id)


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
    # What the bot asked for: how a deliberate recall named its memory, or a fact; None for an association.
    served: _Served | None = None
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
    return payload.AssociationsData(
        you=found.you, people=found.people, persons=found.persons, items=tuple(items), omitted=omitted
    )


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


def _served(request: hsm.Event[typing.Any]) -> _Served | None:
    """What a request asks the bot's memory for, by payload; None for an association (or no payload of ours)."""

    data = request.data
    if isinstance(data, payload.RecallData):
        return "node" if data.node is not None else "about"
    if isinstance(data, payload.KnowData):
        return "know"
    if isinstance(data, payload.ForgetData):
        return "forget"
    return None


def _count(served: _Served | None, outcome: typing.Literal["ok", "failed", "timeout"], reply: object = None) -> None:
    if served is None:
        _ASSOCIATIONS.add(1, {"bot.outcome": outcome})
    elif served in ("know", "forget"):
        kind = reply.kind.value if isinstance(reply, payload.FailureData) else "unavailable"
        _KNOWN.add(
            1,
            {
                "bot.memory.know.op": served,
                "bot.outcome": "ok" if outcome == "ok" else "failed",
                "bot.failure.kind": "none" if outcome == "ok" else kind,
            },
        )
    else:
        _RECALLS.add(1, {"bot.memory.recall.mode": served, "bot.outcome": "ok" if outcome == "ok" else "failed"})


def _failure(served: _Served | None, kind: payload.FailureKind, message: str) -> hsm.Event[ability.FailureData]:
    """A typed failure reply: memory's kind for a fact, the ability's message for an association or recall."""

    if served in ("know", "forget"):
        return ability.FailedEvent.with_data(payload.FailureData(kind=kind, message=message))
    return ability.FailedEvent.with_data(ability.FailureData(message=f"{kind.value}: {message}"))


class Associative(ability.Ability[payload.CueData, payload.AssociationsData]):
    """Surface what memory associates with a turn, bring a memory to mind, and remember who people are.

    ``memory`` is a peer ``Memory`` reached only by message. ``registries`` are the voice and face
    registries the perceiving abilities own, at most one of each kind, whose prints follow what the
    bot knows. ``budget_tokens`` bounds the text the associations carry, counted by ``count_tokens``.
    ``weights`` and ``search`` (how many candidates each lane offers) are capacities of the
    composition, shared by associations and recalls. ``timeout`` bounds one association or recall
    end to end, and each step of a fact on its own (one request to memory or to one registry), never
    the fact as a whole: a fact takes as many steps as its person has prints.

    Raises:
        ValueError: if ``budget_tokens`` or ``timeout`` is not positive, or two registries serve one kind.
    """

    # Recall (deliberate thought about a memory) is deliberate-tier work: never offered on the reflex arc.
    effort: typing.ClassVar[Effort] = Effort.M
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.CueData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = payload.AssociationsData
    input_event: typing.ClassVar[hsm.Event[payload.CueData]] = AssociateEvent
    output_event: typing.ClassVar[hsm.Event[payload.AssociationsData]] = AssociatedEvent
    failed_event: typing.ClassVar[hsm.Event[typing.Any]] = ability.FailedEvent
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = _REQUESTS
    # The composite form gives the submodel its own detaching state, so a request waiting behind
    # the one in flight is answered there instead of being discarded on detach.
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _attachment_group: attachment.Group | None
    _memory: memory.Memory
    _registries: dict[str, identification.Registry]
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
        message = f"{cls.__name__} is not attached."
        if _served(event) in ("know", "forget"):
            return payload.FailureData(kind=payload.FailureKind.UNAVAILABLE, message=message)
        return ability.FailureData(message=f"unavailable: {message}")

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        memory: memory.Memory,
        registries: collections.abc.Sequence[identification.Registry] = (),
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
        by_kind: dict[str, identification.Registry] = {}
        for registry in registries:
            kind = registry.kind()
            if kind in by_kind:
                raise ValueError(f"Associative takes one {kind} registry; two were given.")
            by_kind[kind] = registry
        super().__init__(id=id, clock=clock)
        self._attachment_group = attachment.Group()
        self._memory = memory
        self._registries = by_kind
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
        """Where a reply goes, by payload: what the bot asked for to the owner (""); an association to its requester."""

        if _served(event) is not None:
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
        _count(_served(request), "failed", reply.data)

    @staticmethod
    def _reply_unavailable_detaching(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        unavailable = _failure(_served(event), payload.FailureKind.UNAVAILABLE, "Associative is detaching.")
        Associative._reply_now(ctx, instance, event, unavailable)
        ability.Ability._count_dropped(instance, reason="detaching", replied=True)

    @staticmethod
    def _reply_unavailable_degraded(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        unavailable = _failure(_served(event), payload.FailureKind.UNAVAILABLE, "Associative is degraded.")
        Associative._reply_now(ctx, instance, event, unavailable)
        ability.Ability._count_dropped(instance, reason="degraded", replied=True)

    @staticmethod
    def _reply_invalid(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        invalid = _failure(
            _served(event),
            payload.FailureKind.INVALID_INPUT,
            f"{event.name} carried {type(event.data).__name__}.",
        )
        Associative._reply_now(ctx, instance, event, invalid)

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
    def _carries_fact(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, payload.KnowData)

    @staticmethod
    def _carries_person(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, payload.ForgetData)

    @staticmethod
    def _begin(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._in_flight = _InFlight(
            id=event.id,
            requester=Associative._requester(instance, event),
            metadata=dict(event.metadata),
            started=time.monotonic(),  # telemetry only; hsm.after owns the deadline
            served=_served(event),
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
        _count(in_flight.served, outcome, reply.data)
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
        served = "association" if in_flight.served is None else "recollection"
        late = _failure(in_flight.served, payload.FailureKind.TIMEOUT, f"no {served} within {seconds:g}s.")
        Associative._answer(ctx, instance, late, "timeout")
        Associative._settle(ctx, instance)

    @staticmethod
    def _abandon(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        """Leaving busy: a request still unanswered (detach or stop) is answered unavailable."""

        del event
        in_flight = instance._in_flight
        if in_flight is not None and not in_flight.answered:
            stopped = _failure(
                in_flight.served, payload.FailureKind.UNAVAILABLE, "Associative stopped serving the request."
            )
            Associative._answer(ctx, instance, stopped, "failed")
        instance._in_flight = None

    @staticmethod
    async def _request(
        ctx: hsm.Context,
        instance: "Associative",
        child: hsm.Instance,
        request: hsm.Event[typing.Any],
        terminals: tuple[hsm.Event[typing.Any], hsm.Event[typing.Any]],
    ) -> object:
        """Send ``request`` to a peer (Memory or a registry) by message: its reply data, or a ``payload.FailureData``."""

        try:
            terminal = await ability.run_terminal_operation(
                ctx,
                child=child,
                request=request.with_data_and_id(request.data, uuid.uuid4().hex),
                terminals=terminals,
                timeout=instance._timeout,
            )
        except TimeoutError:
            return payload.FailureData(
                kind=payload.FailureKind.TIMEOUT, message=f"{type(child).__name__} did not answer {request.name}."
            )
        except RuntimeError as error:
            return payload.FailureData(
                kind=payload.FailureKind.UNAVAILABLE, message=f"{type(child).__name__} cannot be reached: {error}"
            )
        reply = terminal.data
        if isinstance(reply, payload.FailureData):
            return reply
        if isinstance(reply, identification.FailureData):
            kind = _KINDS.get(reply.kind.value, payload.FailureKind.STORE_FAILED)
            return payload.FailureData(kind=kind, message=reply.message)
        if isinstance(reply, ability.FailureData):
            return payload.FailureData(kind=payload.FailureKind.UNAVAILABLE, message=reply.message)
        return reply

    @staticmethod
    async def _ask(
        ctx: hsm.Context,
        instance: "Associative",
        active: trace.Span,
        request: hsm.Event[typing.Any],
        answer: hsm.Event[typing.Any],
    ) -> object:
        """Send ``request`` to Memory by message; return its ``answer`` data, or the typed failure."""

        reply = await Associative._request(ctx, instance, instance._memory, request, (answer, memory.FailedEvent))
        if isinstance(reply, payload.FailureData):
            span.record_failure(active, reply.kind.value)
            return ability.FailureData(message=f"{reply.kind.value}: {reply.message}")
        return reply

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
                # The turn being answered is already shown: its episodes are not earlier memories.
                excluded = frozenset(data.exclude)
                found = found.model_copy(
                    update={
                        "candidates": tuple(
                            candidate
                            for candidate in found.candidates
                            if candidate.node.episode is None or candidate.node.episode.turn_id not in excluded
                        )
                    }
                )
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

    # --- knowing people ---------------------------------------------------------------------------

    @staticmethod
    def _print(instance: "Associative", ref: str) -> tuple[identification.Registry, str] | payload.FailureData | None:
        """The registry and id a biometric ``ref`` names; None for a ref no registry here holds."""

        kind, _, id = ref.partition(":")
        registry = instance._registries.get(kind)
        if registry is None:
            return None
        try:
            return registry, _ID.validate_python(id)
        except pydantic.ValidationError:
            return payload.FailureData(
                kind=payload.FailureKind.INVALID_INPUT, message=f"That is not a valid {kind} ref."
            )

    @staticmethod
    async def _mark(ctx: hsm.Context, instance: "Associative", ref: str, linked: bool) -> payload.FailureData | None:
        """Set (or clear) the link mark of ``ref``'s print: the failure, or None when done or no registry holds it."""

        held = Associative._print(instance, ref)
        if not isinstance(held, tuple):
            return held
        registry, id = held
        reply = await Associative._request(
            ctx,
            instance,
            registry,
            identification.LinkEvent.with_data(identification.LinkData(id=id, linked=linked)),
            (identification.LinkedEvent, identification.FailedEvent),
        )
        return reply if isinstance(reply, payload.FailureData) else None

    @staticmethod
    async def _person(ctx: hsm.Context, instance: "Associative", data: payload.PersonData) -> object:
        """One person statement to memory: its ``PersonDoneData``, or the typed failure."""

        reply = await Associative._request(
            ctx,
            instance,
            instance._memory,
            memory.PersonEvent.with_data(data),
            (memory.PersonDoneEvent, memory.FailedEvent),
        )
        if isinstance(reply, payload.PersonDoneData) and reply.person is None:
            return payload.FailureData(
                kind=payload.FailureKind.STORE_FAILED, message="Memory answered without the person."
            )
        return reply

    @staticmethod
    async def _learn(ctx: hsm.Context, instance: "Associative", data: payload.KnowData) -> hsm.Event[typing.Any]:
        """Remember one fact, then make the prints follow what memory holds."""

        done = await Associative._person(ctx, instance, payload.PersonKnowData(about=data.about, fact=data.fact))
        if not isinstance(done, payload.PersonDoneData):
            assert isinstance(done, payload.FailureData)
            return _failure("know", done.kind, done.message)
        assert done.person is not None
        if done.released is not None:
            kept = await Associative._mark(ctx, instance, done.released, False)
            if kept is not None and kept.kind is not payload.FailureKind.NOT_FOUND:
                # Memory is the authority and has let the ref go; a mark left behind only keeps a print longer.
                _LOG.warning("memory: a link mark could not be cleared after not_same_as (%s)", kept.kind.value)
        if done.person.name is None:
            # Nobody has named them, so no print of theirs is kept.
            return KnownEvent.with_data(payload.KnownData(op="know", person=done.person))
        refused: list[tuple[str, payload.FailureData]] = []
        unkept: payload.FailureData | None = None
        for ref in done.refs:
            failure = await Associative._mark(ctx, instance, ref, True)
            if failure is None:
                continue
            if ref in done.moved and failure.kind in _PERMANENT:
                refused.append((ref, failure))
            elif failure.kind is not payload.FailureKind.NOT_FOUND and unkept is None:
                # A print a registry no longer holds has nothing left to keep.
                unkept = failure
        if refused:
            # The id is the bot's own or holds no print, so nobody may hold the ref: memory's move is taken back.
            for ref, _ in refused:
                undone = await Associative._person(
                    ctx,
                    instance,
                    payload.PersonKnowData(about=ref, fact=payload.NotSameAsFact(other=done.person.id)),
                )
                if isinstance(undone, payload.FailureData):
                    _LOG.warning("memory: a fact a registry refused could not be taken back (%s)", undone.kind.value)
            return _failure(
                "know", refused[0][1].kind, "That ref's print is the bot's own or holds no print; nobody may hold it."
            )
        if unkept is not None:
            return _failure(
                "know", unkept.kind, "Memory knows it, but not every print of theirs could be kept yet; know it again."
            )
        return KnownEvent.with_data(payload.KnownData(op="know", person=done.person))

    @staticmethod
    async def _erase(ctx: hsm.Context, instance: "Associative", data: payload.ForgetData) -> hsm.Event[typing.Any]:
        """Forget a person in memory, then erase each of their prints."""

        done = await Associative._person(ctx, instance, payload.PersonForgetData(person=data.person))
        if not isinstance(done, payload.PersonDoneData):
            assert isinstance(done, payload.FailureData)
            return _failure("forget", done.kind, done.message)
        assert done.person is not None
        unerased: payload.FailureData | None = None
        for ref in done.refs:
            held = Associative._print(instance, ref)
            if not isinstance(held, tuple):
                continue
            registry, id = held
            erased = await Associative._request(
                ctx,
                instance,
                registry,
                identification.ForgetEvent.with_data(identification.ForgetData(id=id)),
                (identification.ForgottenEvent, identification.FailedEvent),
            )
            if isinstance(erased, payload.FailureData) and unerased is None:
                unerased = erased
        if unerased is not None:
            return _failure(
                "forget", unerased.kind, "The person is forgotten, but not every one of their prints could be erased."
            )
        return KnownEvent.with_data(payload.KnownData(op="forget", person=done.person))

    @staticmethod
    async def _know(ctx: hsm.Context, instance: "Associative", event: hsm.Event[typing.Any]) -> None:
        """Serve the fact or the forgetting in flight; dispatch its one correlated reply."""

        data = event.data
        assert isinstance(data, payload.KnowData | payload.ForgetData)
        served = "know" if isinstance(data, payload.KnowData) else "forget"
        with span.operation(
            f"bot.memory.{served}", scope="bot.abilities.memory", component="memory.associative", stage=served
        ) as active:
            reply: hsm.Event[typing.Any]
            try:
                reply = (
                    await Associative._learn(ctx, instance, data)
                    if isinstance(data, payload.KnowData)
                    else await Associative._erase(ctx, instance, data)
                )
            except Exception:
                # The boundary of the request's one answer: an unexpected error is answered, never left hanging.
                _LOG.exception("memory: the %s request failed unexpectedly", served)
                reply = _failure(served, payload.FailureKind.STORE_FAILED, "Memory could not finish the request.")
            if isinstance(reply.data, payload.FailureData):
                span.record_failure(active, reply.data.kind.value)
            _ = instance.dispatch(ctx, _CompletedEvent.with_data_and_id(_CompletedData(reply=reply), event.id))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Associative",
        hsm.initial(hsm.target("/Associative/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, *_REQUESTS),
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
            # Offered to a deliberate frame from here: recall, know and forget are model-callable only while idle.
            hsm.transition(hsm.on(RecallEvent), hsm.guard(_carries_recall), hsm.target("/Associative/busy/recalling")),
            hsm.transition(hsm.on(KnowEvent), hsm.guard(_carries_fact), hsm.target("/Associative/busy/knowing")),
            hsm.transition(hsm.on(ForgetEvent), hsm.guard(_carries_person), hsm.target("/Associative/busy/knowing")),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_reply_invalid)),
        ),
        hsm.state(
            "busy",
            hsm.entry(_begin),
            hsm.exit(_abandon),
            hsm.defer(*_REQUESTS),
            hsm.transition(hsm.on(_CompletedEvent), hsm.guard(_is_current), hsm.effect(_complete)),
            hsm.transition(hsm.on(_SettledEvent), hsm.guard(_is_settled), hsm.target("/Associative/idle")),
            # An association or a recall is bounded end to end; a fact is bounded per step (``_request``),
            # so a fact of many steps that each answer in time is never cut short.
            hsm.state(
                "associating",
                hsm.activity(_associate),
                hsm.transition(hsm.after(_timeout_delay), hsm.guard(_is_unanswered), hsm.effect(_time_out)),
            ),
            hsm.state(
                "recalling",
                hsm.activity(_recall),
                hsm.transition(hsm.after(_timeout_delay), hsm.guard(_is_unanswered), hsm.effect(_time_out)),
            ),
            hsm.state("knowing", hsm.activity(_know)),
        ),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_reply_unavailable_detaching)),
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
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_reply_unavailable_degraded)),
        ),
    )


__all__ = [
    "AssociateEvent",
    "AssociatedEvent",
    "Associative",
    "ForgetEvent",
    "KnowEvent",
    "KnownEvent",
    "RecallEvent",
    "RecollectedEvent",
]
