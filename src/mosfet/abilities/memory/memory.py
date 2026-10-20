"""The bot's one durable ``Memory``: relational transactions plus an episodic graph with vectors.

``Memory`` is a state machine reached by message only. Besides the relational statement
transaction it inherits (``input_event``, used by behavior inventory and identity), it keeps every
cognition turn as an episode node of a graph, links episodes to the persons they involve, and
holds a canonical vector per node in its embedder's space. Requests:

* ``RememberEvent`` (``EpisodeData``) → ``RememberedEvent``
* ``SearchEvent`` (``SearchData``) → ``FoundEvent``: candidates, and who is here (each participant
  resolved to its person, the persons they are, and ``self``)
* ``FetchEvent`` (``FetchData``) → ``FetchedEvent``
* ``RecentEvent`` (``RecentData``) → ``RecentOutputEvent``
* ``BackfillEvent`` (``BackfillData``) → ``BackfilledEvent``
* ``LeaseEvent`` (``LeaseData``) → ``LeasedEvent``
* ``ConsolidateEvent`` (``ConsolidateData``, model-callable) → ``ConsolidatedEvent``
* ``ReleaseEvent`` (``ReleaseData``) → ``ReleasedEvent``
* ``PersonEvent`` (``PersonData``: name, link, unlink, forget, list) → ``PersonDoneEvent``

Memory is the authority on who is who: every participant ref belongs to exactly one person
(``bot_person_ref``), resolved inside the transaction that keeps an episode. A ref first seen gets a
provisional person, and a ref stamped own is the bot's own person, ``self``, which Memory creates
when it opens. A person statement is one transaction; it refuses ``conflict`` (the ref is another
person's, or the statement would change the bot itself or its own refs) or ``not_found``, and changes
nothing then. Nobody declares who anyone is: every person is one the bot perceived and learned.

Every addressed request gets exactly one reply, sent to its ``source`` with its ``id``: the
operation's own reply, or ``FailedEvent`` with a ``FailureKind``. SQL runs one step at a time, in
order; the rest wait. Memory never waits on an embed while it holds that order: a request that
needs vectors (remember, search with a cue, backfill, consolidate with merges) sends its embed to
Memory's own ``Embedding`` child by message and Memory goes back to serving the next request; the
embed's reply, correlated by the embed's id, queues that request's next SQL step like any other
request. Replies can therefore arrive out of request order: a request waiting on its embed is
answered after later requests that need none. All SQL, the database's open and migration
included, runs on one database worker thread, so transactions never interleave and the event loop
never blocks on SQL. Construction opens nothing and starts nothing (DSL-29): ``start`` creates the
worker and ``stop`` shuts it down, and Memory opens its database on the worker in its ``opening``
state, before it attaches its ``Embedding`` child. A database that cannot be opened fails the attach
(``attachment.FailureKind.INITIALIZATION``), and one that does not open within the attach timeout
fails it as ``TIMEOUT``; either way Memory is ``degraded`` and answers every request unavailable.

Embedding failures never lose an episode: a node that cannot be embedded is kept unindexed and the
attempt is recorded (``bot_memory_index_attempt``); backfill retries only transient failures, at
most once per pass, and never retries a node that cannot be embedded in that space.

Telemetry (cardinality, METRICS > SPANS > Logs; every attribute value comes from code):

* ``bot.memory.operation.count`` — operation (10) × outcome {ok, failed} × ``bot.failure.kind``
  (FailureKind or ``none``; ok is always ``none``): at most 10 × (1 + 7) = 80 series.
* ``bot.memory.operation.duration`` — operation × outcome, from accepting a request to answering
  it (its embed wait included): 20 series.
* ``bot.memory.embed.waiting`` — requests waiting on an embed reply, no attributes: 1 series.
* ``bot.memory.remember.indexed.count`` — indexed × unindexed reason × index path: 10 series.
* ``bot.memory.node.count`` — node kind: 3 series.
* ``bot.memory.search.lane.count`` — lane: 4 series.
* ``bot.memory.search.nearest.skipped.count`` — skip reason: 4 series.
* ``bot.memory.person.conflict.count`` — own-stamped refs kept with the other person they belong
  to, no attributes: 1 series.
* ``bot.ability.request.dropped.count`` — requests answered while detaching or degraded.

Spans ``bot.memory.<operation>`` (component ``memory``), one per SQL step, carry only bounded
counts, whether the step resumes after an embed, and a person statement's ``bot.memory.person.op``
(5 values); node ids, unit ids, person ids, participant refs, names, relation labels, text, images,
vectors, and space names are never attributes.
"""

import asyncio
import collections.abc
import concurrent.futures
import dataclasses
import datetime
import logging
import time
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics, trace
from sqlalchemy.engine import Engine

from mosfet import event
from mosfet.protocols import attachment
from mosfet.telemetry import span

from .. import ability, embedding
from ..language import text
from . import graph, index, payload, store
from .index import ExactIndex  # the ``index`` constructor parameter shadows the module there

_LOG = logging.getLogger(__name__)

_DEFAULT_EMBED_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_DEFAULT_OPERATION_TIMEOUT: typing.Final = datetime.timedelta(seconds=60)

Operation: typing.TypeAlias = typing.Literal[
    "transact", "remember", "search", "fetch", "recent", "backfill", "lease", "consolidate", "release", "person"
]
_UnindexedReason: typing.TypeAlias = typing.Literal["none", "no_embedder", "capacity", "too_long", "embed_failed"]

_METER = metrics.get_meter("bot.abilities.memory")
_OPERATIONS = _METER.create_counter(
    "bot.memory.operation.count",
    unit="{operation}",
    description="Memory requests answered, by operation, outcome, and failure kind.",
)
_DURATION = _METER.create_histogram(
    "bot.memory.operation.duration",
    unit="s",
    description="Time from accepting a memory request to answering it, by operation and outcome.",
)
_WAITING = _METER.create_up_down_counter(
    "bot.memory.embed.waiting",
    unit="{request}",
    description="Memory requests waiting on an embed reply before their next SQL step.",
)
_INDEXED = _METER.create_counter(
    "bot.memory.remember.indexed.count",
    unit="{node}",
    description="Nodes Memory tried to index, by whether they were indexed, why not, and which path tried.",
)
_NODES = _METER.create_up_down_counter(
    "bot.memory.node.count",
    unit="{node}",
    description="Memory nodes created minus nodes forgotten in this process, by kind.",
)
_LANES = _METER.create_counter(
    "bot.memory.search.lane.count",
    unit="{candidate}",
    description="Candidates each search lane surfaced.",
)
_SKIPPED = _METER.create_counter(
    "bot.memory.search.nearest.skipped.count",
    unit="{search}",
    description="Searches whose nearest lane did not run, by reason.",
)
_CONFLICTS = _METER.create_counter(
    "bot.memory.person.conflict.count",
    unit="{ref}",
    description="Refs stamped as the bot's own that memory kept with the other person they already belong to.",
)

InputEvent = hsm.Event[store.InputData](name="bot.ability.memory.input", schema=store.InputData)
"""One relational statement transaction; answered with ``OutputEvent`` or ``FailedEvent``."""
OutputEvent = hsm.Event[store.OutputData](name="bot.ability.memory.output", schema=store.OutputData)
FailedEvent = hsm.Event[payload.FailureData](name="bot.ability.memory.failed", schema=payload.FailureData)
"""The one failure reply of every Memory request."""

RememberEvent = hsm.Event[payload.EpisodeData](name="bot.ability.memory.remember", schema=payload.EpisodeData)
RememberedEvent = hsm.Event[payload.RememberedData](name="bot.ability.memory.remembered", schema=payload.RememberedData)
SearchEvent = hsm.Event[payload.SearchData](name="bot.ability.memory.search", schema=payload.SearchData)
FoundEvent = hsm.Event[payload.FoundData](name="bot.ability.memory.found", schema=payload.FoundData)
FetchEvent = hsm.Event[payload.FetchData](name="bot.ability.memory.fetch", schema=payload.FetchData)
FetchedEvent = hsm.Event[payload.FetchedData](name="bot.ability.memory.fetched", schema=payload.FetchedData)
RecentEvent = hsm.Event[payload.RecentData](name="bot.ability.memory.recent", schema=payload.RecentData)
RecentOutputEvent = hsm.Event[payload.EpisodesData](
    name="bot.ability.memory.recent.output", schema=payload.EpisodesData
)
BackfillEvent = hsm.Event[payload.BackfillData](name="bot.ability.memory.backfill", schema=payload.BackfillData)
BackfilledEvent = hsm.Event[payload.BackfilledData](name="bot.ability.memory.backfilled", schema=payload.BackfilledData)
LeaseEvent = hsm.Event[payload.LeaseData](name="bot.ability.memory.lease", schema=payload.LeaseData)
LeasedEvent = hsm.Event[payload.LeasedData](name="bot.ability.memory.leased", schema=payload.LeasedData)
ConsolidateEvent = hsm.Event[payload.ConsolidateData](
    name="bot.ability.memory.consolidate",
    kind=event.EventKind,
    schema=payload.ConsolidateData,
)
"""The consolidation decision a model makes about one leased unit; the only model-callable Memory event."""
ConsolidatedEvent = hsm.Event[payload.ConsolidatedData](
    name="bot.ability.memory.consolidated", schema=payload.ConsolidatedData
)
ReleaseEvent = hsm.Event[payload.ReleaseData](name="bot.ability.memory.release", schema=payload.ReleaseData)
ReleasedEvent = hsm.Event[payload.ReleasedData](name="bot.ability.memory.released", schema=payload.ReleasedData)
PersonEvent = hsm.Event[payload.PersonData](name="bot.ability.memory.person", schema=payload.PersonData)
"""One person statement (name, link, unlink, forget, list); not model-callable."""
PersonDoneEvent = hsm.Event[payload.PersonDoneData](
    name="bot.ability.memory.person.done", schema=payload.PersonDoneData
)

REQUESTS: typing.Final[tuple[hsm.Event[typing.Any], ...]] = (
    InputEvent,
    RememberEvent,
    SearchEvent,
    FetchEvent,
    RecentEvent,
    BackfillEvent,
    LeaseEvent,
    ConsolidateEvent,
    ReleaseEvent,
    PersonEvent,
)
"""Every request Memory answers."""

_OPERATIONS_BY_DATA: typing.Final[
    tuple[tuple[type[pydantic.BaseModel] | tuple[type[pydantic.BaseModel], ...], Operation], ...]
] = (
    (store.InputData, "transact"),
    (payload.EpisodeData, "remember"),
    (payload.SearchData, "search"),
    (payload.FetchData, "fetch"),
    (payload.RecentData, "recent"),
    (payload.BackfillData, "backfill"),
    (payload.LeaseData, "lease"),
    (payload.ConsolidateData, "consolidate"),
    (payload.ReleaseData, "release"),
    (payload.PERSON_DATA, "person"),
)


@dataclasses.dataclass(frozen=True)
class _Remembering:
    """Remember after its embed: index the node the first step wrote."""

    seq: int
    node_id: str


@dataclasses.dataclass(frozen=True)
class _Searching:
    """Search after its cue's embed."""

    data: payload.SearchData


@dataclasses.dataclass(frozen=True)
class _Backfilling:
    """Backfill after its batch's embed: the batch's node seqs in item order, and attempts so far."""

    pass_id: str
    batch: tuple[int, ...]
    attempts: tuple[tuple[int, embedding.FailureKind], ...]


@dataclasses.dataclass(frozen=True)
class _Consolidating:
    """Consolidate after one chunk of merge texts is embedded: its size, the chunks left, vectors so far."""

    data: payload.ConsolidateData
    pending: int
    chunks: tuple[tuple[embedding.Item, ...], ...]
    merged: tuple[tuple[float, ...] | None, ...] = ()


_Resume: typing.TypeAlias = _Remembering | _Searching | _Backfilling | _Consolidating


@dataclasses.dataclass(frozen=True)
class _Embed:
    """An embed a step hands to the Embedding child, and how the request resumes on its reply."""

    role: embedding.Role
    items: tuple[embedding.Item, ...]
    resume: _Resume


class _CompletedData(pydantic.BaseModel):
    """A step's outcome: the reply to send, or the embed to wait on; and what it does to the lease."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    reply: hsm.Event[typing.Any] | None = None
    embed: pydantic.SkipValidation[_Embed | None] = None
    lease: typing.Literal["keep", "set", "clear", "decide"] = "keep"
    leased: graph.Lease | None = None


class _SettledData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


_CompletedEvent = hsm.Event[_CompletedData](
    name="bot.ability.memory.completed", kind=hsm.CompletionEventKind, schema=_CompletedData
)
_SettledEvent = hsm.Event[_SettledData](
    name="bot.ability.memory.settled", kind=hsm.CompletionEventKind, schema=_SettledData
)


@dataclasses.dataclass(frozen=True)
class _InFlight:
    """A request being served: in ``busy`` for one SQL step, or waiting on an embed between steps.

    ``resume`` is None for a request's first step and names the step its embed reply resumes.
    ``done`` marks the step finished: the request answered, or handed to an embed.
    """

    id: str
    requester: str
    operation: Operation
    metadata: dict[str, typing.Any]
    started: float
    resume: _Resume | None = None
    done: bool = False


@dataclasses.dataclass(frozen=True)
class _Embedded:
    """An embedding outcome: vectors in order, or the typed failure."""

    vectors: tuple[tuple[float, ...], ...] = ()
    failure: embedding.FailureData | None = None

    @classmethod
    def of(cls, data: object) -> "_Embedded":
        """Read the Embedding child's reply payload."""

        if isinstance(data, embedding.OutputData):
            return cls(vectors=tuple(vector.values for vector in data.vectors))
        assert isinstance(data, embedding.FailureData), "only embedding replies resume a request"
        return cls(failure=data)


def _carrying(
    request: hsm.Event[typing.Any],
    data_type: type[pydantic.BaseModel] | tuple[type[pydantic.BaseModel], ...],
    operation: Operation,
) -> hsm.Element:
    """Admit ``request`` into busy only when it carries its own schema (typed payload, not its name)."""

    def carries(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, data_type)

    carries.__name__ = carries.__qualname__ = f"carries_{operation}"
    return hsm.transition(hsm.on(request), hsm.guard(carries), hsm.target("/Memory/serving/busy"))


def _operation(data: object) -> Operation | None:
    for data_type, operation in _OPERATIONS_BY_DATA:
        if isinstance(data, data_type):
            return operation
    return None


def _item(capacity: embedding.Capacity, parts: tuple[text.ContentPart, ...]) -> embedding.Item | None:
    """Adapt parts to what the embedder accepts, mechanically; None when nothing acceptable is left.

    Image parts are dropped when the embedder takes no images, or when the item mixes text and
    images and the embedder does not interleave them. Adjacent text parts are joined. Text is
    never trimmed; an item whose images exceed the embedder's per-request budget is not offered.
    """

    kept = list(parts)
    has_text = any(isinstance(part, text.TextPart) and part.text.strip() for part in kept)
    if capacity.images.max_images == 0 or (has_text and not capacity.interleaved):
        kept = [part for part in kept if isinstance(part, text.TextPart)]
    joined: list[text.ContentPart] = []
    for part in kept:
        previous = joined[-1] if joined else None
        if isinstance(part, text.TextPart) and isinstance(previous, text.TextPart):
            joined[-1] = text.TextPart(text=f"{previous.text}\n{part.text}")
        else:
            joined.append(part)
    images = [part.image for part in joined if isinstance(part, text.ImagePart)]
    if not (images or any(isinstance(part, text.TextPart) and part.text.strip() for part in joined)):
        return None
    if len(joined) > embedding.MAX_PARTS or any(verdict is not None for verdict in capacity.images.admit(images)):
        return None
    return embedding.Item(parts=tuple(joined))


def _unindexed_reason(kind: embedding.FailureKind) -> _UnindexedReason:
    if kind is embedding.FailureKind.TOO_LONG:
        return "too_long"
    if kind is embedding.FailureKind.UNSUPPORTED_INPUT:
        return "capacity"
    return "embed_failed"


class Memory(store.MemoryStore):
    """The bot's one durable memory: relational transactions and an episodic graph with vectors.

    ``embedder=None`` is a real capacity, "this bot cannot embed": no vectors are written, the
    nearest search lane is skipped, and backfill does nothing. ``index=None`` uses
    ``ExactIndex``. ``now`` stamps episodes and reviews (UTC; default: the system clock).
    ``embed_timeout`` bounds one embedding request in Memory's ``Embedding`` child, which answers
    every request it is sent (``TIMEOUT`` when it is slow); ``operation_timeout`` bounds one SQL
    step. Neither wait holds another request: a request waiting on its embed is not in a step.

    Construction records the database and allocates nothing else: the worker exists from ``start``
    to ``stop``, and the database is opened when Memory first attaches (see the module docstring).

    Raises:
        ValueError: if a timeout is not positive.
    """

    default_scope: typing.ClassVar[str] = "memory"
    input_event: typing.ClassVar[hsm.Event[store.InputData]] = InputEvent
    output_event: typing.ClassVar[hsm.Event[store.OutputData]] = OutputEvent
    failed_event: typing.ClassVar[hsm.Event[payload.FailureData]] = FailedEvent
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = REQUESTS
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _attachment_group: attachment.Group | None
    _embedding: embedding.Embedding | None
    _capacity: embedding.Capacity | None
    _index: index.Index
    _now: collections.abc.Callable[[], datetime.datetime]
    _operation_timeout: datetime.timedelta
    # The one database worker, created by start and shut down by stop. A restart's new worker first
    # waits for the previous one to drain, so a job left running by a timeout, a detach or a stop can
    # never overlap a later one on the same connection.
    _worker: concurrent.futures.ThreadPoolExecutor | None
    # The jobs submitted to the worker and not yet done, so stop can close the database after the last.
    _jobs: set[concurrent.futures.Future[typing.Any]]
    # Durable machine data: the request in its SQL step, the requests waiting on an embed (by
    # the embed's id; the reply's id selects one), and the unit leased for consolidation.
    _in_flight: _InFlight | None
    _waiting: dict[str, _InFlight]
    _lease: graph.Lease | None

    @classmethod
    @typing.override
    def detached_failure(cls, event: hsm.Event[typing.Any]) -> pydantic.BaseModel:
        del event
        return payload.FailureData(kind=payload.FailureKind.UNAVAILABLE, message=f"{cls.__name__} is not attached.")

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        engine: Engine | collections.abc.Callable[[], Engine] | None = None,
        database: str = ":memory:",
        embedder: embedding.Embedder | None = None,
        index: index.Index | None = None,
        now: collections.abc.Callable[[], datetime.datetime] | None = None,
        embed_timeout: datetime.timedelta = _DEFAULT_EMBED_TIMEOUT,
        operation_timeout: datetime.timedelta = _DEFAULT_OPERATION_TIMEOUT,
    ) -> None:
        if embed_timeout <= datetime.timedelta(0) or operation_timeout <= datetime.timedelta(0):
            raise ValueError("Memory timeouts must be positive.")
        super().__init__(id=id, clock=clock, engine=engine, database=database)
        self._embedding = None if embedder is None else embedding.Embedding(embedder=embedder, timeout=embed_timeout)
        self._capacity = None if embedder is None else embedder.capacity
        self._attachment_group = attachment.Group() if self._embedding is None else attachment.Group(self._embedding)
        self._index = ExactIndex() if index is None else index
        self._now = now if now is not None else (lambda: datetime.datetime.now(datetime.UTC))
        self._operation_timeout = operation_timeout
        self._worker = None
        self._jobs = set()
        self._in_flight = None
        self._waiting = {}
        self._lease = None

    @typing.override
    async def start(self, ctx: hsm.Context, data: object = None) -> typing.Self:
        """Give Memory its database worker, then start it.

        A worker left by an earlier start is drained first, on the new worker, so its last job
        finishes before any job of this start runs.
        """

        previous = self._worker
        self._worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")
        if previous is not None:
            _ = self._worker.submit(previous.shutdown, wait=True)
        return await super().start(ctx, data)

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop, close the database it opened, and let the database worker exit: queued jobs are cancelled, a running one finishes."""

        await super().stop(ctx)
        if self._worker is not None:
            self._worker.shutdown(wait=False, cancel_futures=True)

    @typing.override
    def _close_engine(self, engine: Engine) -> None:
        """Close the database once the worker's last job is done: queued jobs are cancelled, a running one finishes first.

        Stop never waits on that job: the engine is disposed on the worker, right after it.
        """

        worker = self._worker
        if worker is not None:
            worker.shutdown(wait=False, cancel_futures=True)
        # One worker: once the queue is cancelled, at most the job it is running is not done.
        running = next((job for job in self._jobs if not job.done()), None)
        if running is None:
            engine.dispose()
            return
        running.add_done_callback(lambda _: engine.dispose())

    @typing.override
    async def _open_engine(self) -> Engine:
        """Open the database on the worker, so the event loop never blocks on the open or the migration.

        The bot is a person in its own memory from the moment it opens: ``self`` is created here.
        """

        return await Memory._run(self, self._open)

    def _open(self) -> Engine:
        """Open and migrate the database and create ``self`` (database I/O; runs on the worker)."""

        engine = self._connect()
        if graph.ensure_self(engine, at=self._now()):
            _NODES.add(1, {"bot.memory.node.kind": payload.NodeKind.PERSON.value})
        return engine

    # --- replies -------------------------------------------------------------------------------

    @staticmethod
    def _send(
        ctx: hsm.Context,
        instance: "Memory",
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
    def _failure(kind: payload.FailureKind, message: str) -> hsm.Event[payload.FailureData]:
        return FailedEvent.with_data(payload.FailureData(kind=kind, message=message))

    @staticmethod
    def _reply_now(
        ctx: hsm.Context, instance: "Memory", request: hsm.Event[typing.Any], reply: hsm.Event[typing.Any]
    ) -> None:
        """Answer a request that never became the one in flight."""

        Memory._send(
            ctx,
            instance,
            request_id=request.id,
            requester=store.MemoryStore._requester(instance, request),
            metadata=dict(request.metadata),
            reply=reply,
        )
        operation = _operation(request.data)
        if operation is not None and isinstance(reply.data, payload.FailureData):
            _OPERATIONS.add(
                1,
                {"bot.memory.operation": operation, "bot.outcome": "failed", "bot.failure.kind": reply.data.kind.value},
            )

    @staticmethod
    def _reply_unavailable_detaching(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        Memory._reply_now(
            ctx, instance, event, Memory._failure(payload.FailureKind.UNAVAILABLE, "Memory is detaching.")
        )
        ability.Ability._count_dropped(instance, reason="detaching", replied=True)

    @staticmethod
    def _reply_unavailable_degraded(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        Memory._reply_now(ctx, instance, event, Memory._failure(payload.FailureKind.UNAVAILABLE, "Memory is degraded."))
        ability.Ability._count_dropped(instance, reason="degraded", replied=True)

    @staticmethod
    def _reply_invalid(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        Memory._reply_now(
            ctx,
            instance,
            event,
            Memory._failure(
                payload.FailureKind.INVALID_INPUT, f"{event.name} carried {type(event.data).__name__}, not its schema."
            ),
        )

    # --- lifecycle -----------------------------------------------------------------------------

    @staticmethod
    def _log_attached(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        capacity = instance._capacity
        _LOG.info(
            "memory attached: index=%s dimensions=%s embedder=%s",
            type(instance._index).__name__,
            None if capacity is None else capacity.space.dimensions,
            capacity is not None,
        )

    # --- request admission ---------------------------------------------------------------------

    @staticmethod
    def _is_stale_decision(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, payload.ConsolidateData)
            and graph.decision_problem(instance._lease, data) is payload.FailureKind.STALE_LEASE
        )

    @staticmethod
    def _is_invalid_decision(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, payload.ConsolidateData)
            and graph.decision_problem(instance._lease, data) is payload.FailureKind.INVALID_INPUT
        )

    @staticmethod
    def _is_leased_release(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, payload.ReleaseData) and instance._lease is not None and instance._lease.unit == data.unit
        )

    @staticmethod
    def _is_stale_release(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        return isinstance(event.data, payload.ReleaseData) and not Memory._is_leased_release(ctx, instance, event)

    @staticmethod
    def _reply_stale(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        Memory._reply_now(
            ctx, instance, event, Memory._failure(payload.FailureKind.STALE_LEASE, "That unit is no longer leased.")
        )

    @staticmethod
    def _reject_decision(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        """An invalid decision decides the unit: it is kept leased for its release, never decided again."""

        assert instance._lease is not None
        instance._lease = dataclasses.replace(instance._lease, decided=True)
        Memory._reply_now(
            ctx,
            instance,
            event,
            Memory._failure(
                payload.FailureKind.INVALID_INPUT,
                "The decision names a node outside the unit, or forgets a person; nothing was changed.",
            ),
        )

    # --- busy ----------------------------------------------------------------------------------

    @staticmethod
    def _begin(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        """Enter one SQL step: a request's first, or the one its embed reply resumes."""

        del ctx
        if isinstance(event.data, embedding.OutputData | embedding.FailureData):
            instance._in_flight = instance._waiting.pop(event.id)
            _WAITING.add(-1)
            return
        operation = _operation(event.data)
        assert operation is not None, "only typed requests enter busy"
        instance._in_flight = _InFlight(
            id=event.id,
            requester=store.MemoryStore._requester(instance, event),
            operation=operation,
            metadata=dict(event.metadata),
            started=time.monotonic(),  # telemetry only; hsm.after owns the deadline
        )

    @staticmethod
    def _answer(ctx: hsm.Context, instance: "Memory", in_flight: _InFlight, reply: hsm.Event[typing.Any]) -> None:
        """Send a request's one reply and count it."""

        Memory._send(
            ctx,
            instance,
            request_id=in_flight.id,
            requester=in_flight.requester,
            metadata=in_flight.metadata,
            reply=reply,
        )
        failed = isinstance(reply.data, payload.FailureData)
        outcome = "failed" if failed else "ok"
        kind = reply.data.kind.value if isinstance(reply.data, payload.FailureData) else "none"
        _OPERATIONS.add(
            1, {"bot.memory.operation": in_flight.operation, "bot.outcome": outcome, "bot.failure.kind": kind}
        )
        _DURATION.record(
            time.monotonic() - in_flight.started, {"bot.memory.operation": in_flight.operation, "bot.outcome": outcome}
        )

    @staticmethod
    def _finish(instance: "Memory") -> _InFlight:
        """Mark the step in flight finished, so no later path answers it again; return it."""

        in_flight = instance._in_flight
        assert in_flight is not None and not in_flight.done
        instance._in_flight = dataclasses.replace(in_flight, done=True)
        return in_flight

    @staticmethod
    def _settle(ctx: hsm.Context, instance: "Memory") -> None:
        in_flight = instance._in_flight
        assert in_flight is not None
        _ = hsm.dispatch(ctx, instance, _SettledEvent.with_data_and_id(_SettledData(), in_flight.id))

    @staticmethod
    def _is_current(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        in_flight = instance._in_flight
        return in_flight is not None and event.id == in_flight.id and not in_flight.done

    @staticmethod
    def _is_unfinished(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._in_flight is not None and not instance._in_flight.done

    @staticmethod
    def _is_settled(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        in_flight = instance._in_flight
        return in_flight is not None and event.id == in_flight.id and in_flight.done

    @staticmethod
    def _is_awaited(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> bool:
        """The reply of an embed a request is waiting on: correlated by the embed's id and its sender."""

        del ctx
        child = instance._embedding
        return child is not None and event.id in instance._waiting and event.source == hsm.id(child)

    @staticmethod
    def _complete(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        """Finish the step: answer the request, or send its embed and keep it waiting on the reply."""

        data = event.data
        assert isinstance(data, _CompletedData)
        if data.lease == "set":
            instance._lease = data.leased
        elif data.lease == "clear":
            instance._lease = None
        elif data.lease == "decide":
            assert instance._lease is not None, "admission guards check the lease"
            instance._lease = dataclasses.replace(instance._lease, decided=True)
        in_flight = Memory._finish(instance)
        if data.embed is None:
            assert data.reply is not None, "a step ends with a reply or an embed"
            Memory._answer(ctx, instance, in_flight, data.reply)
        else:
            child = instance._embedding
            assert child is not None, "only a Memory with an embedder waits on an embed"
            embed_id = uuid.uuid4().hex
            instance._waiting[embed_id] = dataclasses.replace(in_flight, resume=data.embed.resume)
            _WAITING.add(1)
            # Sent with no target (no context, so hsm stamps none): the Embedding child answers its
            # attachment owner, this Memory. A target would name Memory inside the child's private
            # address scope, where Memory is not registered, and the reply would be lost.
            _ = hsm.dispatch(
                None,
                child,
                dataclasses.replace(
                    embedding.InputEvent.with_data_and_id(
                        embedding.InputData(role=data.embed.role, items=data.embed.items), embed_id
                    ),
                    source=hsm.id(instance),
                    metadata=dict(in_flight.metadata),
                ),
            )
        Memory._settle(ctx, instance)

    @staticmethod
    def _timeout_delay(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._operation_timeout

    @staticmethod
    def _time_out(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        del event
        seconds = instance._operation_timeout.total_seconds()
        Memory._answer(
            ctx,
            instance,
            Memory._finish(instance),
            Memory._failure(payload.FailureKind.TIMEOUT, f"Memory did not finish within {seconds:g}s."),
        )
        Memory._settle(ctx, instance)

    @staticmethod
    def _abandon(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        """Leaving busy: a step still unfinished (detach or stop) is answered UNAVAILABLE."""

        del event
        if instance._in_flight is not None and not instance._in_flight.done:
            Memory._answer(
                ctx,
                instance,
                Memory._finish(instance),
                Memory._failure(payload.FailureKind.UNAVAILABLE, "Memory stopped serving the request."),
            )
        instance._in_flight = None

    @staticmethod
    def _abandon_waiting(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        """Leaving serving (detach or stop): every request waiting on an embed is answered UNAVAILABLE.

        A late embed reply then matches nothing and is ignored.
        """

        del event
        waiting, instance._waiting = instance._waiting, {}
        _WAITING.add(-len(waiting))
        for in_flight in waiting.values():
            Memory._answer(
                ctx,
                instance,
                in_flight,
                Memory._failure(payload.FailureKind.UNAVAILABLE, "Memory stopped serving the request."),
            )

    # --- work ----------------------------------------------------------------------------------

    @staticmethod
    async def _run[**P, T](
        instance: "Memory", job: collections.abc.Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Run one blocking database job on this Memory's worker, in submission order."""

        worker = instance._worker
        if worker is None:
            raise RuntimeError("Memory runs database jobs only while started.")
        bound = span.bind(job)
        submitted = worker.submit(bound, *args, **kwargs)
        instance._jobs.add(submitted)
        submitted.add_done_callback(instance._jobs.discard)
        return await asyncio.wrap_future(submitted)

    @staticmethod
    def _done(
        ctx: hsm.Context,
        instance: "Memory",
        in_flight: _InFlight,
        reply: hsm.Event[typing.Any],
        *,
        lease: typing.Literal["keep", "set", "clear"] = "keep",
        leased: graph.Lease | None = None,
    ) -> None:
        """End the step with the request's reply."""

        _ = instance.dispatch(
            ctx,
            _CompletedEvent.with_data_and_id(_CompletedData(reply=reply, lease=lease, leased=leased), in_flight.id),
        )

    @staticmethod
    def _wait(
        ctx: hsm.Context,
        instance: "Memory",
        in_flight: _InFlight,
        role: embedding.Role,
        items: collections.abc.Sequence[embedding.Item],
        resume: _Resume,
        *,
        lease: typing.Literal["keep", "decide"] = "keep",
    ) -> None:
        """End the step with an embed: the request waits on its reply, which resumes it at ``resume``."""

        _ = instance.dispatch(
            ctx,
            _CompletedEvent.with_data_and_id(
                _CompletedData(embed=_Embed(role=role, items=tuple(items), resume=resume), lease=lease), in_flight.id
            ),
        )

    @staticmethod
    async def _record_failed_index(
        instance: "Memory", seq: int, kind: embedding.FailureKind, reason: _UnindexedReason
    ) -> None:
        """Record a node the remember path could not index: one attempt row, counted, never raised."""

        capacity = instance._capacity
        assert capacity is not None
        if kind is embedding.FailureKind.UNSUPPORTED_INPUT:
            _LOG.warning("memory node is permanently unindexable in this space: %s", kind.value)
        await Memory._run(
            instance,
            graph.record_attempts,
            instance._opened_engine(),
            capacity.space,
            [(seq, kind)],
            pass_id=None,
            at=instance._now(),
        )
        _INDEXED.add(
            1,
            {"bot.memory.indexed": "false", "bot.memory.unindexed.reason": reason, "bot.memory.index.path": "remember"},
        )

    @staticmethod
    async def _serve(ctx: hsm.Context, instance: "Memory", event: hsm.Event[typing.Any]) -> None:
        """Run the SQL step that entered busy and dispatch its correlated completion."""

        in_flight = instance._in_flight
        assert in_flight is not None
        operation = in_flight.operation
        with span.operation(
            f"bot.memory.{operation}", scope="bot.abilities.memory", component="memory", stage=operation
        ) as active:
            active.set_attribute("bot.memory.resumed", in_flight.resume is not None)
            try:
                if in_flight.resume is None:
                    await Memory._work(ctx, instance, in_flight, event.data, active)
                else:
                    await Memory._resume(ctx, instance, in_flight, _Embedded.of(event.data), active)
            except Exception as error:
                # Boundary: a database failure rolled its transaction back; the requester gets STORE_FAILED.
                span.record_failure(active, payload.FailureKind.STORE_FAILED.value)
                if isinstance(event.data, payload.EpisodeData):
                    _LOG.warning("memory did not keep an episode (stimulus=%s): store_failed", event.data.stimulus)
                Memory._done(
                    ctx,
                    instance,
                    in_flight,
                    Memory._failure(payload.FailureKind.STORE_FAILED, f"{type(error).__name__}: {error}"),
                )

    @staticmethod
    async def _work(
        ctx: hsm.Context, instance: "Memory", in_flight: _InFlight, data: object, active: trace.Span
    ) -> None:
        """A request's first step: everything up to its embed, or the whole request when it needs none."""

        engine = instance._opened_engine()
        capacity = instance._capacity
        space = None if capacity is None else capacity.space
        if isinstance(data, store.InputData):
            output = await Memory._run(instance, instance._execute, data)
            Memory._done(ctx, instance, in_flight, OutputEvent.with_data(output))
        elif isinstance(data, payload.EpisodeData):
            kept = await Memory._run(instance, graph.write_episode, engine, data, at=instance._now())
            seq, node_id = kept.seq, kept.node_id
            _NODES.add(1, {"bot.memory.node.kind": payload.NodeKind.EPISODE.value})
            if kept.persons:
                _NODES.add(kept.persons, {"bot.memory.node.kind": payload.NodeKind.PERSON.value})
            if kept.conflicts:
                _CONFLICTS.add(kept.conflicts)
                _LOG.warning(
                    "memory kept %d own-stamped ref(s) with the person they belong to: conflict", kept.conflicts
                )
            # An episode with no parts has nothing to embed: it is not an index attempt, so it
            # records no attempt row, logs nothing and counts nothing, and backfill never selects it.
            item = None if capacity is None or not data.parts else _item(capacity, data.parts)
            if item is not None:
                resume = _Remembering(seq=seq, node_id=node_id)
                Memory._wait(ctx, instance, in_flight, embedding.Role.DOCUMENT, (item,), resume)
                return
            if capacity is None and data.parts:
                _INDEXED.add(
                    1,
                    {
                        "bot.memory.indexed": "false",
                        "bot.memory.unindexed.reason": "no_embedder",
                        "bot.memory.index.path": "remember",
                    },
                )
            elif data.parts:
                await Memory._record_failed_index(instance, seq, embedding.FailureKind.UNSUPPORTED_INPUT, "capacity")
            active.set_attribute("bot.memory.indexed", False)
            Memory._done(
                ctx,
                instance,
                in_flight,
                RememberedEvent.with_data(payload.RememberedData(node_id=node_id, indexed=False)),
            )
        elif isinstance(data, payload.SearchData):
            if data.cue and data.nearest and capacity is not None and (item := _item(capacity, data.cue)) is not None:
                Memory._wait(ctx, instance, in_flight, embedding.Role.QUERY, (item,), _Searching(data=data))
                return
            await Memory._search(ctx, instance, in_flight, data, None, active)
        elif isinstance(data, payload.FetchData):
            fetched = await Memory._run(instance, graph.fetch, engine, data, space=space, at=instance._now())
            Memory._done(ctx, instance, in_flight, FetchedEvent.with_data(fetched))
        elif isinstance(data, payload.RecentData):
            episodes = await Memory._run(instance, graph.recent, engine, data, space=space)
            Memory._done(ctx, instance, in_flight, RecentOutputEvent.with_data(episodes))
        elif isinstance(data, payload.BackfillData):
            await Memory._backfill(ctx, instance, in_flight, data)
        elif isinstance(data, payload.LeaseData):
            unit, leased = await Memory._run(instance, graph.lease, engine, instance._index, data, space=space)
            active.set_attribute("bot.memory.unit.nodes", 0 if unit is None else len(unit.nodes))
            Memory._done(
                ctx,
                instance,
                in_flight,
                LeasedEvent.with_data(payload.LeasedData(unit=unit)),
                lease="set",
                leased=leased,
            )
        elif isinstance(data, payload.ConsolidateData):
            lease_state = instance._lease
            assert lease_state is not None, "admission guards check the lease"
            chunks: tuple[tuple[embedding.Item, ...], ...] = ()
            if capacity is not None:
                items = tuple(embedding.Item(parts=(text.TextPart(text=merge.text),)) for merge in data.merges)
                size = capacity.max_items
                chunks = tuple(items[start : start + size] for start in range(0, len(items), size))
            if chunks:
                # The unit is decided while its merges are embedded: a second decision is stale.
                resume = _Consolidating(data=data, pending=len(chunks[0]), chunks=chunks[1:])
                Memory._wait(ctx, instance, in_flight, embedding.Role.DOCUMENT, chunks[0], resume, lease="decide")
                return
            await Memory._commit(ctx, instance, in_flight, data, lease_state, (None,) * len(data.merges))
        elif isinstance(data, payload.PERSON_DATA):
            active.set_attribute("bot.memory.person.op", data.op)
            outcome, change = await Memory._run(
                instance, graph.person_statement, engine, instance._index, data, at=instance._now()
            )
            if change:
                _NODES.add(change, {"bot.memory.node.kind": payload.NodeKind.PERSON.value})
            if isinstance(outcome, graph.Refused):
                span.record_failure(active, outcome.kind.value)
                Memory._done(ctx, instance, in_flight, Memory._failure(outcome.kind, outcome.message))
                return
            Memory._done(ctx, instance, in_flight, PersonDoneEvent.with_data(outcome))
        else:
            assert isinstance(data, payload.ReleaseData)
            lease_state = instance._lease
            assert lease_state is not None, "admission guards check the lease"
            await Memory._run(instance, graph.release, engine, lease_state, reviewed=data.reviewed, at=instance._now())
            Memory._done(
                ctx,
                instance,
                in_flight,
                ReleasedEvent.with_data(payload.ReleasedData(unit=data.unit)),
                lease="clear",
            )

    @staticmethod
    async def _resume(
        ctx: hsm.Context, instance: "Memory", in_flight: _InFlight, embedded: _Embedded, active: trace.Span
    ) -> None:
        """A request's step after its embed reply: the SQL that needed the vectors."""

        resume = in_flight.resume
        capacity = instance._capacity
        assert capacity is not None, "only a Memory with an embedder waits on an embed"
        if isinstance(resume, _Remembering):
            if embedded.failure is not None:
                kind = embedded.failure.kind
                await Memory._record_failed_index(instance, resume.seq, kind, _unindexed_reason(kind))
            else:
                await Memory._run(
                    instance,
                    graph.write_vectors,
                    instance._opened_engine(),
                    instance._index,
                    capacity.space,
                    [(resume.seq, embedded.vectors[0])],
                )
                _INDEXED.add(
                    1,
                    {
                        "bot.memory.indexed": "true",
                        "bot.memory.unindexed.reason": "none",
                        "bot.memory.index.path": "remember",
                    },
                )
            indexed = embedded.failure is None
            active.set_attribute("bot.memory.indexed", indexed)
            Memory._done(
                ctx,
                instance,
                in_flight,
                RememberedEvent.with_data(payload.RememberedData(node_id=resume.node_id, indexed=indexed)),
            )
        elif isinstance(resume, _Searching):
            await Memory._search(ctx, instance, in_flight, resume.data, embedded, active)
        elif isinstance(resume, _Backfilling):
            await Memory._backfilled(ctx, instance, in_flight, resume, embedded)
        else:
            assert isinstance(resume, _Consolidating)
            vectors = embedded.vectors if embedded.failure is None else (None,) * resume.pending
            merged = (*resume.merged, *vectors)
            if resume.chunks:
                following = _Consolidating(
                    data=resume.data, pending=len(resume.chunks[0]), chunks=resume.chunks[1:], merged=merged
                )
                Memory._wait(ctx, instance, in_flight, embedding.Role.DOCUMENT, resume.chunks[0], following)
                return
            lease_state = instance._lease
            if lease_state is None or lease_state.unit != resume.data.unit or not lease_state.decided:
                # Released (and maybe leased again) while the merges were embedded: nothing is applied.
                Memory._done(
                    ctx,
                    instance,
                    in_flight,
                    Memory._failure(payload.FailureKind.STALE_LEASE, "That unit is no longer leased."),
                )
                return
            await Memory._commit(ctx, instance, in_flight, resume.data, lease_state, merged)

    @staticmethod
    async def _search(
        ctx: hsm.Context,
        instance: "Memory",
        in_flight: _InFlight,
        data: payload.SearchData,
        embedded: _Embedded | None,
        active: trace.Span,
    ) -> None:
        """Search with the cue's vector when it was embedded; ``embedded`` is None when no embed was sent."""

        capacity = instance._capacity
        cue: tuple[float, ...] | None = None
        if data.cue and data.nearest:
            reason: _UnindexedReason | None = None
            if capacity is None:
                reason = "no_embedder"
            elif embedded is None:
                reason = "capacity"
            elif embedded.failure is None:
                cue = embedded.vectors[0]
            else:
                reason = _unindexed_reason(embedded.failure.kind)
            if reason is not None:
                _SKIPPED.add(1, {"bot.memory.skip.reason": reason})
        found, created, conflicts = await Memory._run(
            instance,
            graph.search,
            instance._opened_engine(),
            instance._index,
            data,
            space=None if capacity is None else capacity.space,
            cue=cue,
            at=instance._now(),
        )
        if created:
            _NODES.add(created, {"bot.memory.node.kind": payload.NodeKind.PERSON.value})
        if conflicts:
            _CONFLICTS.add(conflicts)
            _LOG.warning("memory kept %d own-stamped ref(s) with the person they belong to: conflict", conflicts)
        for candidate in found.candidates:
            for lane in candidate.lanes:
                _LANES.add(1, {"bot.memory.lane": lane.value})
        active.set_attribute("bot.memory.candidates", len(found.candidates))
        Memory._done(ctx, instance, in_flight, FoundEvent.with_data(found))

    @staticmethod
    async def _commit(
        ctx: hsm.Context,
        instance: "Memory",
        in_flight: _InFlight,
        data: payload.ConsolidateData,
        lease_state: graph.Lease,
        merged: collections.abc.Sequence[tuple[float, ...] | None],
    ) -> None:
        """Apply one decision in one transaction, with whatever merge vectors were embedded."""

        capacity = instance._capacity
        outcome, changes = await Memory._run(
            instance,
            graph.commit,
            instance._opened_engine(),
            instance._index,
            data,
            lease_state,
            space=None if capacity is None else capacity.space,
            merged_vectors=list(merged),
            at=instance._now(),
        )
        for kind, change in changes.items():
            _NODES.add(change, {"bot.memory.node.kind": kind.value})
        Memory._done(ctx, instance, in_flight, ConsolidatedEvent.with_data(outcome), lease="clear")

    @staticmethod
    async def _backfill(ctx: hsm.Context, instance: "Memory", in_flight: _InFlight, data: payload.BackfillData) -> None:
        """Pick one batch within the embedder's whole capacity and send it to be embedded.

        A node that was tried before is embedded alone, so a repeated failure is attributed to it.
        """

        capacity = instance._capacity
        if capacity is None:
            Memory._done(
                ctx,
                instance,
                in_flight,
                BackfilledEvent.with_data(payload.BackfilledData(indexed=0, unindexable=0, remaining=False)),
            )
            return
        engine = instance._opened_engine()
        candidates = await Memory._run(
            instance, graph.backfill_candidates, engine, capacity.space, pass_id=data.pass_id, limit=data.limit
        )
        parts = await Memory._run(instance, graph.node_parts, engine, [seq for seq, _ in candidates])
        batch: list[tuple[int, embedding.Item]] = []
        attempts: list[tuple[int, embedding.FailureKind]] = []
        for seq, tried in candidates:
            item = _item(capacity, parts.get(seq, ()))
            if item is None:
                attempts.append((seq, embedding.FailureKind.UNSUPPORTED_INPUT))
                continue
            if tried:
                if not batch:
                    batch.append((seq, item))
                break
            images = [
                part.image
                for _, kept in [*batch, (seq, item)]
                for part in kept.parts
                if isinstance(part, text.ImagePart)
            ]
            if len(batch) >= capacity.max_items or any(
                verdict is not None for verdict in capacity.images.admit(images)
            ):
                break
            batch.append((seq, item))
        if batch:
            resume = _Backfilling(pass_id=data.pass_id, batch=tuple(seq for seq, _ in batch), attempts=tuple(attempts))
            Memory._wait(ctx, instance, in_flight, embedding.Role.DOCUMENT, [item for _, item in batch], resume)
            return
        await Memory._backfilled(
            ctx, instance, in_flight, _Backfilling(pass_id=data.pass_id, batch=(), attempts=tuple(attempts)), None
        )

    @staticmethod
    async def _backfilled(
        ctx: hsm.Context,
        instance: "Memory",
        in_flight: _InFlight,
        resume: _Backfilling,
        embedded: _Embedded | None,
    ) -> None:
        """Write the batch's vectors or attribute its failure exactly, then record and answer.

        A failure naming an item marks only that node; a batch failure naming no item is transient
        for the batch (not selected again in this pass) unless the batch held one node.
        """

        capacity = instance._capacity
        assert capacity is not None
        engine = instance._opened_engine()
        space = capacity.space
        attempts = list(resume.attempts)
        indexed = 0
        failure = None if embedded is None else embedded.failure
        if embedded is not None and failure is None:
            await Memory._run(
                instance,
                graph.write_vectors,
                engine,
                instance._index,
                space,
                list(zip(resume.batch, embedded.vectors, strict=True)),
            )
            indexed = len(resume.batch)
        elif failure is not None and failure.item_index is not None and failure.item_index < len(resume.batch):
            attempts.append((resume.batch[failure.item_index], failure.kind))
        elif failure is not None and len(resume.batch) == 1:
            attempts.append((resume.batch[0], failure.kind))
        elif failure is not None:
            transient = embedding.FailureKind.TIMEOUT if failure.kind in graph.PERMANENT_FAILURES else failure.kind
            attempts.extend((seq, transient) for seq in resume.batch)
        await Memory._run(
            instance, graph.record_attempts, engine, space, attempts, pass_id=resume.pass_id, at=instance._now()
        )
        unindexable = sum(kind in graph.PERMANENT_FAILURES for _, kind in attempts)
        for _, kind in attempts:
            if kind is embedding.FailureKind.UNSUPPORTED_INPUT:
                _LOG.warning("memory node is permanently unindexable in this space: %s", kind.value)
            _INDEXED.add(
                1,
                {
                    "bot.memory.indexed": "false",
                    "bot.memory.unindexed.reason": _unindexed_reason(kind),
                    "bot.memory.index.path": "backfill",
                },
            )
        if indexed:
            _INDEXED.add(
                indexed,
                {
                    "bot.memory.indexed": "true",
                    "bot.memory.unindexed.reason": "none",
                    "bot.memory.index.path": "backfill",
                },
            )
        remaining = await Memory._run(instance, graph.has_eligible, engine, space, pass_id=resume.pass_id)
        Memory._done(
            ctx,
            instance,
            in_flight,
            BackfilledEvent.with_data(
                payload.BackfilledData(indexed=indexed, unindexable=unindexable, remaining=remaining)
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Memory",
        hsm.initial(hsm.target("/Memory/opening")),
        hsm.state(
            "opening",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, *REQUESTS),
            hsm.activity(store.MemoryStore._open_activity),
            hsm.transition(hsm.on(store.MemoryStore._opened_event), hsm.target("/Memory/initializing")),
            hsm.transition(hsm.on(store.MemoryStore._open_failed_event), hsm.effect(store.MemoryStore._refuse_attach)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Memory/degraded"),
            ),
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(attachment.Attachment._timeout_attachment, store.MemoryStore._count_open_timeout),
                hsm.target("/Memory/degraded"),
            ),
        ),
        hsm.state(
            "initializing",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, *REQUESTS),
            hsm.activity(store.MemoryStore._attach_opened),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal, _log_attached),
                hsm.target("/Memory/serving"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                # In initializing the only composite operation is the attach, so its "detached"
                # terminal is the attach failure: the owner is told, and Memory fails closed.
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Memory/degraded"),
            ),
        ),
        hsm.state(
            "serving",
            hsm.exit(_abandon_waiting),
            hsm.initial(hsm.target("/Memory/serving/idle")),
            hsm.state(
                "idle",
                hsm.transition(hsm.on(ConsolidateEvent), hsm.guard(_is_stale_decision), hsm.effect(_reply_stale)),
                hsm.transition(hsm.on(ConsolidateEvent), hsm.guard(_is_invalid_decision), hsm.effect(_reject_decision)),
                hsm.transition(hsm.on(ReleaseEvent), hsm.guard(_is_stale_release), hsm.effect(_reply_stale)),
                *(
                    _carrying(request, data_type, operation)
                    for request, (data_type, operation) in zip(REQUESTS, _OPERATIONS_BY_DATA, strict=True)
                ),
                hsm.transition(hsm.on(*REQUESTS), hsm.effect(_reply_invalid)),
                # An embed reply queues the step it resumes; one that matches no waiting request is ignored.
                hsm.transition(
                    hsm.on(embedding.OutputEvent, embedding.FailedEvent),
                    hsm.guard(_is_awaited),
                    hsm.target("/Memory/serving/busy"),
                ),
            ),
            hsm.state(
                "busy",
                hsm.entry(_begin),
                hsm.exit(_abandon),
                hsm.defer(*REQUESTS, embedding.OutputEvent, embedding.FailedEvent),
                hsm.activity(_serve),
                hsm.transition(hsm.after(_timeout_delay), hsm.guard(_is_unfinished), hsm.effect(_time_out)),
                hsm.transition(hsm.on(_CompletedEvent), hsm.guard(_is_current), hsm.effect(_complete)),
                hsm.transition(hsm.on(_SettledEvent), hsm.guard(_is_settled), hsm.target("/Memory/serving/idle")),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(hsm.on(*REQUESTS), hsm.effect(_reply_unavailable_detaching)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Memory/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Memory/serving"),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.transition(hsm.on(*REQUESTS), hsm.effect(_reply_unavailable_degraded)),
        ),
    )


__all__ = [
    "REQUESTS",
    "BackfillEvent",
    "BackfilledEvent",
    "ConsolidateEvent",
    "ConsolidatedEvent",
    "FailedEvent",
    "FetchEvent",
    "FetchedEvent",
    "FoundEvent",
    "InputEvent",
    "LeaseEvent",
    "LeasedEvent",
    "Memory",
    "OutputEvent",
    "PersonDoneEvent",
    "PersonEvent",
    "RecentEvent",
    "RecentOutputEvent",
    "ReleaseEvent",
    "ReleasedEvent",
    "RememberEvent",
    "RememberedEvent",
    "SearchEvent",
]
