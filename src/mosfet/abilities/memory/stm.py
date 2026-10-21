"""Short-term memory register: bounded rolling record of admitted stimuli and of turns.

The register is sensory short-term memory at the event layer, not a durable memory stream.
One row per admitted environment stimulus, written immediately, evicted by capacity in the
same transaction. It grounds actors (e.g. Learning) in the actual signal the bot received:
name, envelope identity, and a bounded scalar payload projection. Raw media, long text, and
nested structures are dropped by the projection, never stored.

Beside it the register keeps turns: one row per message a Conversation committed, what the bot
heard and what it said, in words (``TurnEvent``). A turn's text is bounded at
``MAX_TURN_TEXT_BYTES`` and cut at a whole word, never dropped. Turns have their own capacity
(evicted oldest first in the same transaction) and recency window, both injected, and are read
oldest first with ``RecentTurnsEvent``. They never reach the scalar register Learning reads.

Retention is access-reinforced (reconsolidation discipline): grounding an entry stamps its
access time, and both recall and eviction read the effective clock max(created_at,
last_accessed_at) — how recently the entry *mattered*, not only when it arrived. Capacity
remains a fixed count invariant enforced at every write; the recency window is a read-policy.

Nothing sent to the register is lost silently. A request it cannot serve (while detaching,
degraded, or not attached) is answered with the failed event, ``unavailable: …``; a notification
it cannot apply is counted. Telemetry: ``bot.ability.request.dropped.count`` with
``bot.ability.name`` = ``StmMemory``, ``bot.ability.drop.reason`` ∈ {detached, detaching,
degraded} and ``bot.ability.replied`` ∈ {true, false}: 6 series. ``bot.memory.stm.turn.recorded.count``
counts turns written, by ``bot.turn.direction`` ∈ {heard, said} and ``bot.turn.cut`` ∈ {true, false}:
4 series.
"""

from __future__ import annotations

from . import schema
from . import store
from .. import ability
from mosfet.protocols import attachment

import collections.abc
import dataclasses
import datetime
import json
import logging
import math
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.sql.elements import ColumnElement

# Identifier-like projection bound: stimulus payload fields are identifiers, enum-like
# kinds, and short scalars. Long strings read as prose and are dropped, never truncated.
MAX_PROJECTION_TEXT_LENGTH = 40
_MAX_PROJECTION_FIELDS = 16
ProjectionValue: typing.TypeAlias = str | int | float | bool
ProjectionPayload: typing.TypeAlias = dict[str, ProjectionValue]
_DEFAULT_CAPACITY = 64
_MAX_LIMIT = 256
_DEFAULT_RECENCY_WINDOW = datetime.timedelta(minutes=5)
MAX_TURN_TEXT_BYTES = 2048
"""Most UTF-8 bytes of one turn's text; a longer turn is cut at a whole word."""
MAX_TURN_SPEAKERS = 8
"""Most participants one turn names as its speakers."""
_DEFAULT_TURN_CAPACITY = 32
_DEFAULT_TURN_WINDOW = datetime.timedelta(minutes=10)

_LOG = logging.getLogger(__name__)
_TURNS_RECORDED = metrics.get_meter("bot.abilities.memory").create_counter(
    "bot.memory.stm.turn.recorded.count",
    unit="{turn}",
    description="Turns the short-term register wrote, by direction and whether the text was cut.",
)


def _utc_text(moment: datetime.datetime) -> str:
    """Microsecond UTC ISO text so lexicographic ordering matches chronological ordering."""

    return moment.astimezone(datetime.timezone.utc).isoformat()


def _effective_time(table: typing.Any) -> ColumnElement[typing.Any]:
    """The clock retention reads: last access when reinforced, else admission."""

    return func.coalesce(table.c.last_accessed_at, table.c.created_at)


class RecordData(pydantic.BaseModel):
    """One register entry to record: stimulus identity and bounded payload projection."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Short-term memory record. stimulus_name is the admitted event name; payload "
                "carries only bounded identifier-like scalar projections of the stimulus data."
            ),
            "examples": [
                {
                    "stimulus_name": "environment.sound",
                    "payload": {"kind": "knock", "source": "device-a"},
                    "event_id": "turn-1",
                    "source": "device-a",
                }
            ],
        },
    )

    stimulus_name: str = pydantic.Field(
        min_length=1,
        description="Name of the admitted stimulus event (e.g. environment.sound).",
        examples=["environment.sound"],
    )
    payload: ProjectionPayload = pydantic.Field(
        default_factory=dict,
        description=(
            "Bounded scalar projection of the stimulus data: top-level identifier-like fields "
            "only. Media bytes, long strings, and nested structures are omitted."
        ),
        examples=[{"kind": "knock"}],
    )
    event_id: str | None = pydantic.Field(default=None, min_length=1, description="Envelope id of the stimulus event.")
    source: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Envelope source address of the stimulus event.",
    )
    target: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Envelope target address of the stimulus event.",
    )

    @pydantic.field_validator("payload", mode="before")
    @classmethod
    def _validate_payload_fields(cls, raw_value: object) -> ProjectionPayload:
        if not isinstance(raw_value, dict):
            raise ValueError("payload must be a mapping of scalar values.")
        payload = typing.cast(ProjectionPayload, raw_value)
        for key, value in payload.items():
            _require_projection_field(key, value)
        return payload


def _require_projection_field(key: object, value: object) -> None:
    if not isinstance(key, str) or not key:
        raise ValueError("payload keys must be non-empty strings.")
    if isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload float values must be finite.")
        return
    if isinstance(value, str):
        if len(value) > MAX_PROJECTION_TEXT_LENGTH:
            raise ValueError(f"payload string values must be at most {MAX_PROJECTION_TEXT_LENGTH} characters.")
        return
    raise ValueError(f"payload values must be finite scalars, got {type(value).__name__}.")


class ObservedEvent(pydantic.BaseModel):
    """One register entry read back for grounding, ordered by the effective clock."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Recent stimulus the bot's short-term memory observed. payload mirrors what the "
                "admitted environment stimulus looked like as bounded scalar fields; selections "
                "are never part of it. stm_id is the handle access reinforcement stamps."
            ),
            "examples": [
                {
                    "stm_id": "6b6f1d7a2c5440eab91c9f8e33a74a10",
                    "stimulus_name": "environment.sound",
                    "payload": {"kind": "knock", "source": "device-a"},
                    "event_id": "turn-1",
                    "source": "device-a",
                    "created_at": "2026-07-07T12:00:00.123456+00:00",
                    "last_accessed_at": None,
                }
            ],
        },
    )

    stm_id: str = pydantic.Field(
        min_length=1,
        description="Register entry handle; access reinforcement stamps this id.",
        examples=["6b6f1d7a2c5440eab91c9f8e33a74a10"],
    )
    stimulus_name: str = pydantic.Field(
        min_length=1,
        description="Name of the admitted stimulus event.",
        examples=["environment.sound"],
    )
    payload: ProjectionPayload = pydantic.Field(
        description="Bounded scalar projection of the stimulus data as it was admitted.",
        examples=[{"kind": "knock"}],
    )
    event_id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Envelope id of the stimulus event.",
    )
    source: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Envelope source address of the stimulus event.",
    )
    target: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Envelope target address of the stimulus event.",
    )
    created_at: str = pydantic.Field(
        min_length=1,
        description="Admission time as UTC ISO text with timezone semantics.",
        examples=["2026-07-07T12:00:00.123456+00:00"],
    )
    last_accessed_at: str | None = pydantic.Field(
        default=None,
        description=(
            "Access time stamped when grounding used this entry, as UTC ISO text. Effective "
            "recency is max(created_at, last_accessed_at)."
        ),
    )


def _scalar_projection_value(value: object) -> ProjectionValue | None:
    """Keep only bounded identifier-like scalars; drop everything else (never truncate)."""

    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value if len(value) <= MAX_PROJECTION_TEXT_LENGTH else None
    return None


def _payload_projection(data: object) -> ProjectionPayload:
    """Project stimulus data onto bounded top-level scalar fields.

    Pydantic payloads dump first; mappings walk one level; bare scalars land under
    ``value``. Everything projected away is dropped, not truncated, so a lesson can
    never masquerade as live event data through the register either.
    """

    fields: ProjectionPayload = {}
    if isinstance(data, pydantic.BaseModel):
        dumped = data.model_dump()
        raw_fields: dict[object, object] = typing.cast("dict[object, object]", dumped)
    elif isinstance(data, dict):
        raw_fields = dict(typing.cast("dict[object, object]", data))
    else:
        scalar = _scalar_projection_value(data)
        return {"value": scalar} if scalar is not None else {}
    for key, value in raw_fields.items():
        if len(fields) >= _MAX_PROJECTION_FIELDS:
            break
        if isinstance(key, str) and key and (projected := _scalar_projection_value(value)) is not None:
            fields[key] = projected
    return fields


def event_record(event: hsm.Event[typing.Any]) -> RecordData:
    """Build the register record for one admitted stimulus event (what ``RecordEvent`` carries).

    The producer that admitted the stimulus builds it: name, envelope identity, and the bounded
    scalar projection of the payload. Media, long text, and nested structures never leave here.
    """

    return RecordData(
        stimulus_name=event.name,
        payload=_payload_projection(event.data),
        event_id=event.id or None,
        source=event.source or None,
        target=event.target or None,
    )


def _projection_payload_from_json(payload: object) -> ProjectionPayload:
    """Read back a stored projection, dropping anything that is not a bounded scalar field."""

    fields: ProjectionPayload = {}
    if not isinstance(payload, dict):
        raise ValueError("register row payload_json must be a JSON object.")
    keyed = typing.cast("dict[object, object]", payload)
    for key, value in keyed.items():
        if isinstance(key, str) and key and isinstance(value, str | int | float | bool):
            fields[key] = value
    return fields


def _optional_text(value: store.ParameterValue | object | None) -> str | None:
    return value if isinstance(value, str) else None


class ReinforceData(pydantic.BaseModel):
    """Register entries that just carried evidence, to stamp with an access time."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Short-term memory reinforcement: the stm_ids of entries grounding just relied on. "
                "Stamping them moves their effective clock forward so used signals outlive arrival churn."
            ),
            "examples": [{"stm_ids": ["6b6f1d7a2c5440eab91c9f8e33a74a10"]}],
        },
    )

    stm_ids: tuple[str, ...] = pydantic.Field(
        min_length=1,
        max_length=_MAX_LIMIT,
        description="Register entry handles (ObservedEvent.stm_id) to stamp with the access time.",
        examples=[("6b6f1d7a2c5440eab91c9f8e33a74a10",)],
    )


class RecentData(pydantic.BaseModel):
    """Read the register inside its recency window, newest effective clock first."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Short-term memory read: entries inside the recency window ordered by the effective "
                "clock max(created_at, last_accessed_at), newest first, optionally for one stimulus."
            ),
            "examples": [{"stimulus_name": "environment.sound", "limit": 16}, {"limit": 64}],
        },
    )

    stimulus_name: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Only entries recorded for this stimulus event name; every stimulus when omitted.",
        examples=["environment.sound"],
    )
    limit: int = pydantic.Field(
        default=_DEFAULT_CAPACITY,
        ge=1,
        le=_MAX_LIMIT,
        description="Maximum number of entries to return.",
        examples=[16],
    )


class RecentOutputData(pydantic.BaseModel):
    """The register entries a recent read found."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Short-term memory read result: entries newest effective clock first.",
            "examples": [
                {
                    "entries": [
                        {
                            "stm_id": "6b6f1d7a2c5440eab91c9f8e33a74a10",
                            "stimulus_name": "environment.sound",
                            "payload": {"kind": "knock"},
                            "created_at": "2026-07-07T12:00:00.123456+00:00",
                        }
                    ]
                }
            ],
        },
    )

    entries: tuple[ObservedEvent, ...] = pydantic.Field(
        description="Entries inside the recency window, newest effective clock first; empty when none.",
    )


TurnDirection: typing.TypeAlias = typing.Literal["heard", "said"]


def _cut_at_word(words: str) -> tuple[str, bool]:
    """``words`` within ``MAX_TURN_TEXT_BYTES``, cut at the last whole word, and whether it was cut.

    A text with no word break inside the bound is cut on the last whole character instead.
    """

    encoded = words.encode()
    if len(encoded) <= MAX_TURN_TEXT_BYTES:
        return words, False
    head = encoded[:MAX_TURN_TEXT_BYTES].decode(errors="ignore")
    if not (head[-1:].isspace() or words[len(head)].isspace()):
        # The bound fell inside a word: drop that partial word.
        whole = head.rsplit(maxsplit=1)
        head = whole[0] if len(whole) > 1 else head
    return head.rstrip(), True


class TurnRecordData(pydantic.BaseModel):
    """One committed message to remember as a turn: who spoke, in which direction, and the words.

    The register stamps the time it is written (``Turn.at``). Build one with ``TurnRecordData.of``,
    which cuts long text at a whole word and says so in ``cut``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A turn of a conversation to remember: what the bot heard or said, who said it, and the words, "
                "bounded and cut at a whole word."
            ),
            "examples": [
                {
                    "id": "turn-1",
                    "direction": "heard",
                    "speakers": [{"ref": "voice:3f2a9c", "role": "source"}],
                    "text": "Hi Willy, I'm Maya.",
                    "cut": False,
                    "conversation": "conversation-1",
                }
            ],
        },
    )

    id: str = pydantic.Field(
        min_length=1,
        description=(
            "Id of the committed message: the provenance id of the event that committed it, so a reader can tell "
            "the turn that is its own stimulus from the turns before it."
        ),
        examples=["turn-1"],
    )
    direction: TurnDirection = pydantic.Field(
        description="'heard' for a message someone sent the bot, 'said' for a message the bot sent.",
        examples=["heard", "said"],
    )
    speakers: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=MAX_TURN_SPEAKERS,
        description=(
            "Who said it, as their producers stamped them: the inbound message's source participants for a heard "
            "turn, the participants stamped on the bot's own message (its own voice) for a said turn. Empty when "
            "no producer stamped anyone."
        ),
        examples=[
            [{"ref": "voice:3f2a9c", "role": "source"}],
            [{"ref": "voice:tts:warm", "role": "source", "own": True}],
        ],
    )
    text: str = pydantic.Field(
        min_length=1,
        description=f"The words of the turn, at most {MAX_TURN_TEXT_BYTES} UTF-8 bytes, cut at a whole word.",
        examples=["Hi Willy, I'm Maya."],
    )
    cut: bool = pydantic.Field(
        default=False,
        description="True when the words were longer than the bound and ``text`` holds only their beginning.",
        examples=[False],
    )
    conversation: str = pydantic.Field(
        min_length=1,
        description="Id of the Conversation that committed the message.",
        examples=["conversation-1"],
    )

    @pydantic.field_validator("text")
    @classmethod
    def _bound_text(cls, text: str) -> str:
        if len(text.encode()) > MAX_TURN_TEXT_BYTES:
            raise ValueError(f"turn text must be at most {MAX_TURN_TEXT_BYTES} UTF-8 bytes; build it with of().")
        return text

    @classmethod
    def of(
        cls,
        *,
        id: str,
        direction: TurnDirection,
        speakers: collections.abc.Sequence[mosfet.Participant],
        words: str,
        conversation: str,
    ) -> TurnRecordData:
        """The turn for ``words``, cut at a whole word when they are longer than ``MAX_TURN_TEXT_BYTES``."""

        text, cut = _cut_at_word(words)
        return cls(
            id=id,
            direction=direction,
            speakers=tuple(speakers[:MAX_TURN_SPEAKERS]),
            text=text,
            cut=cut,
            conversation=conversation,
        )


class Turn(TurnRecordData):
    """One remembered turn, read back from the register: the record and when the register wrote it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "A remembered turn: what the bot heard or said, who said it, the words, and when.",
            "examples": [
                {
                    "id": "turn-1",
                    "at": "2026-07-07T12:00:00.123456Z",
                    "direction": "heard",
                    "speakers": [{"ref": "voice:3f2a9c", "role": "source"}],
                    "text": "Hi Willy, I'm Maya.",
                    "cut": False,
                    "conversation": "conversation-1",
                }
            ],
        },
    )

    at: datetime.datetime = pydantic.Field(
        description="When the register wrote the turn, timezone-aware UTC.",
        examples=["2026-07-07T12:00:00.123456Z"],
    )


class RecentTurnsData(pydantic.BaseModel):
    """Read the most recent turns inside the recency window, oldest first."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Short-term memory read of turns: the newest turns inside the recency window, returned oldest "
                "first. within may narrow the register's window, never widen it."
            ),
            "examples": [{"limit": 32}, {"limit": 8, "within": "PT5M"}],
        },
    )

    limit: int = pydantic.Field(
        default=_DEFAULT_TURN_CAPACITY,
        ge=1,
        le=_MAX_LIMIT,
        description="Most turns to return; the newest are kept.",
        examples=[32],
    )
    within: datetime.timedelta | None = pydantic.Field(
        default=None,
        gt=datetime.timedelta(),
        description="Only turns written this recently; the register's own recency window when omitted or longer.",
        examples=["PT5M"],
    )


class RecentTurnsOutputData(pydantic.BaseModel):
    """The turns a recent-turns read found."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Short-term memory read result: the most recent turns, oldest first.",
            "examples": [
                {
                    "turns": [
                        {
                            "id": "turn-1",
                            "at": "2026-07-07T12:00:00.123456Z",
                            "direction": "heard",
                            "speakers": [{"ref": "voice:3f2a9c", "role": "source"}],
                            "text": "Hi Willy, I'm Maya.",
                            "cut": False,
                            "conversation": "conversation-1",
                        }
                    ]
                }
            ],
        },
    )

    turns: tuple[Turn, ...] = pydantic.Field(
        description="Turns inside the recency window, oldest first; empty when none.",
    )


RecordEvent = hsm.Event[RecordData](
    name="bot.ability.memory.stm.record",
    schema=RecordData,
)
"""Notification: write one admitted stimulus projection and evict beyond capacity. No reply."""

ReinforceEvent = hsm.Event[ReinforceData](
    name="bot.ability.memory.stm.reinforce",
    schema=ReinforceData,
)
"""Notification: stamp the access time on entries that just carried evidence. No reply."""

RecentEvent = hsm.Event[RecentData](
    name="bot.ability.memory.stm.recent",
    schema=RecentData,
)
"""Request: read the register; answered with ``RecentOutputEvent`` or the failed event."""

RecentOutputEvent = hsm.Event[RecentOutputData](
    name="bot.ability.memory.stm.recent.output",
    schema=RecentOutputData,
)
"""Reply to ``RecentEvent``, sent to the request's ``source`` with the request's ``id``."""

TurnEvent = hsm.Event[TurnRecordData](
    name="bot.ability.memory.stm.turn",
    schema=TurnRecordData,
)
"""Notification: write one turn and evict turns beyond capacity. No reply."""

RecentTurnsEvent = hsm.Event[RecentTurnsData](
    name="bot.ability.memory.stm.recent_turns",
    schema=RecentTurnsData,
)
"""Request: read the most recent turns; answered with ``RecentTurnsOutputEvent`` or the failed event."""

RecentTurnsOutputEvent = hsm.Event[RecentTurnsOutputData](
    name="bot.ability.memory.stm.recent_turns.output",
    schema=RecentTurnsOutputData,
)
"""Reply to ``RecentTurnsEvent``, sent to the request's ``source`` with the request's ``id``."""

_SettledEvent = hsm.Event[None](
    name="bot.ability.memory.stm.settled",
    kind=hsm.CompletionEventKind,
    schema=type(None),
)

_REQUESTS = (store.MemoryStore.input_event, RecentEvent, RecentTurnsEvent)
"""Requests the register answers."""
_NOTIFICATIONS = (RecordEvent, ReinforceEvent, TurnEvent)
"""Notifications the register applies without a reply."""


class StmMemory(store.MemoryStore):
    """Short-term memory of admitted stimuli and of turns: capacity-bounded, access-reinforced.

    A state machine reached by message only. ``RecordEvent`` writes one stimulus projection
    and evicts beyond capacity inside one transaction; ``ReinforceEvent`` stamps access times
    on entries grounding relied on, so used signals outlive arrival churn; ``TurnEvent`` writes
    one turn and evicts turns beyond ``turn_capacity`` inside one transaction. All three are
    notifications: perception upkeep never blocks its sender, and a failed write is logged
    here. ``RecentEvent`` reads entries inside the recency window ordered by the effective
    clock and is answered with ``RecentOutputEvent`` (or the failed event) at the requester;
    ``RecentTurnsEvent`` reads the newest turns inside ``turn_window``, oldest first, and is
    answered with ``RecentTurnsOutputEvent`` (or the failed event).
    The inherited statement-transaction request stays available for maintenance. Requests
    are applied one at a time; the rest wait. A detach enters the register's own
    ``detaching`` state, so whatever still waits there is answered unavailable or counted.

    Construction opens nothing (DSL-29). The register opens its database in its ``opening`` state,
    the first time it attaches. A database that cannot be opened fails the attach
    (``attachment.FailureKind.INITIALIZATION``), and one that does not open within the attach
    timeout fails it as ``TIMEOUT``; either way the register is ``degraded``.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = store.InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = store.OutputData
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = _REQUESTS
    detached_notifications: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = _NOTIFICATIONS
    # The composite form gives the register its own detaching state, so what waits behind the
    # request being applied is answered there instead of being discarded on detach.
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True

    _attachment_group: attachment.Group | None
    _capacity: int
    _recency_window: datetime.timedelta
    _turn_capacity: int
    _turn_window: datetime.timedelta
    _now: collections.abc.Callable[[], datetime.datetime]

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        engine: typing.Any | None = None,
        connection: typing.Any | None = None,
        database: str = ":memory:",
        capacity: int = _DEFAULT_CAPACITY,
        recency_window: datetime.timedelta | None = None,
        turn_capacity: int = _DEFAULT_TURN_CAPACITY,
        turn_window: datetime.timedelta = _DEFAULT_TURN_WINDOW,
        now: collections.abc.Callable[[], datetime.datetime] | None = None,
    ) -> None:
        super().__init__(id=id, clock=clock, engine=engine, connection=connection, database=database)
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if recency_window is not None and recency_window <= datetime.timedelta():
            raise ValueError("recency_window must be positive.")
        if turn_capacity <= 0:
            raise ValueError("turn_capacity must be positive.")
        if turn_window <= datetime.timedelta():
            raise ValueError("turn_window must be positive.")
        self._capacity = capacity
        self._recency_window = recency_window if recency_window is not None else _DEFAULT_RECENCY_WINDOW
        self._turn_capacity = turn_capacity
        self._turn_window = turn_window
        self._now = now if now is not None else (lambda: datetime.datetime.now(datetime.timezone.utc))
        self._attachment_group = attachment.Group()

    @classmethod
    @typing.override
    def detached_failure(cls, event: hsm.Event[typing.Any]) -> pydantic.BaseModel:
        del event
        return ability.FailureData(message=f"unavailable: {cls.__name__} is not attached.")

    def _record(self, data: RecordData) -> None:
        insert_clause = insert(schema.stm_memory_table).values(
            stm_id=uuid.uuid4().hex,
            stimulus_name=data.stimulus_name,
            payload_json=json.dumps(data.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            event_id=data.event_id,
            source=data.source,
            target=data.target,
            created_at=_utc_text(self._now()),
        )
        evict_clause = delete(schema.stm_memory_table).where(
            schema.stm_memory_table.c.stm_id.not_in(
                select(schema.stm_memory_table.c.stm_id)
                .order_by(_effective_time(schema.stm_memory_table).desc(), schema.stm_memory_table.c.stm_id.desc())
                .limit(self._capacity)
            )
        )
        _ = self._execute(store.InputData(statements=store.compile_statements(insert_clause, evict_clause)))

    def _reinforce(self, data: ReinforceData) -> None:
        access_clause = (
            update(schema.stm_memory_table)
            .where(schema.stm_memory_table.c.stm_id.in_(data.stm_ids))
            .values(last_accessed_at=_utc_text(self._now()))
        )
        _ = self._execute(store.InputData(statements=store.compile_statements(access_clause)))

    def _recent(self, data: RecentData) -> RecentOutputData:
        cutoff = _utc_text(self._now() - self._recency_window)
        effective = _effective_time(schema.stm_memory_table)
        clause = select(
            schema.stm_memory_table.c.stm_id,
            schema.stm_memory_table.c.stimulus_name,
            schema.stm_memory_table.c.payload_json,
            schema.stm_memory_table.c.event_id,
            schema.stm_memory_table.c.source,
            schema.stm_memory_table.c.target,
            schema.stm_memory_table.c.created_at,
            schema.stm_memory_table.c.last_accessed_at,
        ).where(effective >= cutoff)
        if data.stimulus_name is not None:
            clause = clause.where(schema.stm_memory_table.c.stimulus_name == data.stimulus_name)
        clause = clause.order_by(effective.desc(), schema.stm_memory_table.c.stm_id.desc()).limit(data.limit)
        output = self._execute(store.InputData(statements=store.compile_statements(clause)))
        rows = tuple(row.as_mapping() for row in output.results[0].rows)
        return RecentOutputData(entries=tuple(self._observed_from_row(row) for row in rows))

    def _record_turn(self, data: TurnRecordData) -> None:
        table = schema.stm_turn_table
        insert_clause = insert(table).values(
            turn_id=data.id,
            at=_utc_text(self._now()),
            direction=data.direction,
            speakers_json=json.dumps([speaker.model_dump(mode="json") for speaker in data.speakers]),
            text=data.text,
            cut=int(data.cut),
            conversation=data.conversation,
        )
        evict_clause = delete(table).where(
            table.c.seq.not_in(select(table.c.seq).order_by(table.c.seq.desc()).limit(self._turn_capacity))
        )
        _ = self._execute(store.InputData(statements=store.compile_statements(insert_clause, evict_clause)))
        _TURNS_RECORDED.add(1, {"bot.turn.direction": data.direction, "bot.turn.cut": str(data.cut).lower()})

    def _recent_turns(self, data: RecentTurnsData) -> RecentTurnsOutputData:
        table = schema.stm_turn_table
        window = self._turn_window if data.within is None else min(data.within, self._turn_window)
        clause = (
            select(
                table.c.turn_id,
                table.c.at,
                table.c.direction,
                table.c.speakers_json,
                table.c.text,
                table.c.cut,
                table.c.conversation,
            )
            .where(table.c.at >= _utc_text(self._now() - window))
            .order_by(table.c.seq.desc())
            .limit(data.limit)
        )
        output = self._execute(store.InputData(statements=store.compile_statements(clause)))
        newest_first = tuple(row.as_mapping() for row in output.results[0].rows)
        return RecentTurnsOutputData(
            turns=tuple(
                Turn.model_validate(
                    {
                        "id": row["turn_id"],
                        "at": row["at"],
                        "direction": row["direction"],
                        "speakers": json.loads(str(row["speakers_json"])),
                        "text": row["text"],
                        "cut": bool(row["cut"]),
                        "conversation": row["conversation"],
                    }
                )
                for row in reversed(newest_first)
            )
        )

    @staticmethod
    def _observed_from_row(row: dict[str, store.ParameterValue]) -> ObservedEvent:
        stm_id = row.get("stm_id")
        stimulus_name = row.get("stimulus_name")
        payload_json = row.get("payload_json")
        created_at = row.get("created_at")
        if not isinstance(stm_id, str) or not stm_id:
            raise ValueError("register row is missing stm_id.")
        if not isinstance(stimulus_name, str) or not stimulus_name:
            raise ValueError("register row is missing stimulus_name.")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("register row is missing created_at.")
        if not isinstance(payload_json, str):
            raise ValueError("register row is missing payload_json.")
        try:
            payload = typing.cast(object, json.loads(payload_json))
        except ValueError as error:
            raise ValueError(f"register row payload_json is not valid JSON: {error}") from error
        return ObservedEvent(
            stm_id=stm_id,
            stimulus_name=stimulus_name,
            payload=_projection_payload_from_json(payload),
            event_id=_optional_text(row.get("event_id")),
            source=_optional_text(row.get("source")),
            target=_optional_text(row.get("target")),
            created_at=created_at,
            last_accessed_at=_optional_text(row.get("last_accessed_at")),
        )

    @staticmethod
    def _is_request(ctx: hsm.Context, instance: "StmMemory", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(
            event.data, store.InputData | RecordData | ReinforceData | RecentData | TurnRecordData | RecentTurnsData
        )

    @staticmethod
    def _reply(
        ctx: hsm.Context,
        instance: "StmMemory",
        request: hsm.Event[typing.Any],
        reply: hsm.Event[typing.Any],
    ) -> None:
        terminal = dataclasses.replace(
            reply,
            id=request.id or None,
            metadata=dict(request.metadata),
            source=hsm.id(instance),
            target=store.MemoryStore._requester(instance, request),
        )
        carrier = (
            ability.TerminalErrorEvent if isinstance(reply.data, ability.FailureData) else ability.TerminalOutputEvent
        )
        _ = hsm.dispatch(ctx, instance, carrier.with_data(terminal))

    @staticmethod
    def _reply_unavailable(
        ctx: hsm.Context,
        instance: "StmMemory",
        event: hsm.Event[typing.Any],
        reason: typing.Literal["detaching", "degraded"],
    ) -> None:
        failure = ability.FailureData(message=f"unavailable: {type(instance).__name__} is {reason}.")
        StmMemory._reply(ctx, instance, event, instance.failed_event.with_data(failure))
        ability.Ability._count_dropped(instance, reason=reason, replied=True)

    @staticmethod
    def _reply_unavailable_detaching(ctx: hsm.Context, instance: "StmMemory", event: hsm.Event[typing.Any]) -> None:
        StmMemory._reply_unavailable(ctx, instance, event, "detaching")

    @staticmethod
    def _reply_unavailable_degraded(ctx: hsm.Context, instance: "StmMemory", event: hsm.Event[typing.Any]) -> None:
        StmMemory._reply_unavailable(ctx, instance, event, "degraded")

    @staticmethod
    def _drop_detaching(ctx: hsm.Context, instance: "StmMemory", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        ability.Ability._count_dropped(instance, reason="detaching", replied=False)

    @staticmethod
    def _drop_degraded(ctx: hsm.Context, instance: "StmMemory", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        ability.Ability._count_dropped(instance, reason="degraded", replied=False)

    @staticmethod
    async def _apply_activity(ctx: hsm.Context, instance: "StmMemory", event: hsm.Event[typing.Any]) -> None:
        """Apply one request, answer it when it has a reply contract, then settle."""

        data = event.data
        try:
            if isinstance(data, RecordData):
                instance._record(data)
            elif isinstance(data, ReinforceData):
                instance._reinforce(data)
            elif isinstance(data, RecentData):
                StmMemory._reply(ctx, instance, event, RecentOutputEvent.with_data(instance._recent(data)))
            elif isinstance(data, TurnRecordData):
                instance._record_turn(data)
            elif isinstance(data, RecentTurnsData):
                StmMemory._reply(ctx, instance, event, RecentTurnsOutputEvent.with_data(instance._recent_turns(data)))
            else:
                assert isinstance(data, store.InputData)
                StmMemory._reply(ctx, instance, event, instance.output_event.with_data(instance._execute(data)))
        except Exception as error:
            if isinstance(data, RecordData | ReinforceData | TurnRecordData):
                # Notifications have no requester to tell; the register's own log is the record.
                _LOG.warning("short-term memory %s failed: %r", event.name, error)
            else:
                failure = ability.FailureData(message=f"short-term memory {event.name} failed: {error}")
                StmMemory._reply(ctx, instance, event, instance.failed_event.with_data(failure))
        _ = hsm.dispatch(ctx, instance, _SettledEvent.with_data(None))

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "StmMemory",
        hsm.initial(hsm.target("/StmMemory/opening")),
        hsm.state(
            "opening",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, *_REQUESTS, *_NOTIFICATIONS),
            hsm.activity(store.MemoryStore._open_activity),
            hsm.transition(hsm.on(store.MemoryStore._opened_event), hsm.target("/StmMemory/initializing")),
            hsm.transition(hsm.on(store.MemoryStore._open_failed_event), hsm.effect(store.MemoryStore._refuse_attach)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/StmMemory/degraded"),
            ),
            hsm.transition(
                hsm.after(attachment.Attachment._attachment_timeout_delay),
                hsm.effect(attachment.Attachment._timeout_attachment, store.MemoryStore._count_open_timeout),
                hsm.target("/StmMemory/degraded"),
            ),
        ),
        hsm.state(
            "initializing",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent, *_REQUESTS, *_NOTIFICATIONS),
            hsm.activity(store.MemoryStore._attach_opened),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/StmMemory/idle"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                # Unreachable with the empty group; kept so the composite lifecycle's targets exist.
                hsm.guard(ability.Ability._is_composite_detached_terminal),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/StmMemory/degraded"),
            ),
        ),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(*_REQUESTS, *_NOTIFICATIONS),
                hsm.guard(_is_request),
                hsm.target("/StmMemory/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(*_REQUESTS, *_NOTIFICATIONS),
            hsm.activity(_apply_activity),
            hsm.transition(hsm.on(_SettledEvent), hsm.target("/StmMemory/idle")),
        ),
        hsm.state(
            "detaching",
            hsm.defer(attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_reply_unavailable_detaching)),
            hsm.transition(hsm.on(*_NOTIFICATIONS), hsm.effect(_drop_detaching)),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/StmMemory/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/StmMemory/idle"),
            ),
        ),
        hsm.state(
            "degraded",
            hsm.transition(hsm.on(*_REQUESTS), hsm.effect(_reply_unavailable_degraded)),
            hsm.transition(hsm.on(*_NOTIFICATIONS), hsm.effect(_drop_degraded)),
        ),
    )


__all__ = [
    "MAX_PROJECTION_TEXT_LENGTH",
    "MAX_TURN_SPEAKERS",
    "MAX_TURN_TEXT_BYTES",
    "ObservedEvent",
    "RecentData",
    "RecentEvent",
    "RecentOutputData",
    "RecentOutputEvent",
    "RecentTurnsData",
    "RecentTurnsEvent",
    "RecentTurnsOutputData",
    "RecentTurnsOutputEvent",
    "RecordData",
    "RecordEvent",
    "ReinforceData",
    "ReinforceEvent",
    "StmMemory",
    "Turn",
    "TurnDirection",
    "TurnEvent",
    "TurnRecordData",
    "event_record",
]
