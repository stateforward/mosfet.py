"""Short-term memory event register: bounded rolling record of admitted stimuli.

The register is a sensory working memory at the event layer, not a durable memory stream.
One row per admitted environment stimulus, written immediately, evicted by count in the
same transaction. It grounds actors (e.g. Learning) in the actual signal the bot received:
name, envelope identity, and a bounded scalar payload projection. Raw media, long text, and
nested structures are dropped by the projection, never stored.

Capacity is a fixed invariant enforced at every write (insert + eviction in one apply);
staleness is a recall policy (injected recency window) so an idle bot never grounds a
lesson in an ancient observation.
"""

from __future__ import annotations

from . import schema
from . import store

import collections.abc
import datetime
import json
import math
import typing
import uuid

import hsm
import pydantic
from sqlalchemy import delete
from sqlalchemy import insert
from sqlalchemy import select

# Identifier-like projection bound: stimulus payload fields are identifiers, enum-like
# kinds, and short scalars. Long strings read as prose and are dropped, never truncated.
MAX_PROJECTION_TEXT_LENGTH = 40
_MAX_PROJECTION_FIELDS = 16
ProjectionValue: typing.TypeAlias = str | int | float | bool
ProjectionPayload: typing.TypeAlias = dict[str, ProjectionValue]
_DEFAULT_CAPACITY = 64
_MAX_LIMIT = 256
_DEFAULT_RECENCY_WINDOW = datetime.timedelta(minutes=5)


def _utc_text(moment: datetime.datetime) -> str:
    """Microsecond UTC ISO text so lexicographic ordering matches chronological ordering."""

    return moment.astimezone(datetime.timezone.utc).isoformat()


class RecordData(pydantic.BaseModel):
    """One register entry to record: stimulus identity and bounded payload projection."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Short-term register record. stimulus_name is the admitted event name; payload "
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
            raise ValueError(
                f"payload string values must be at most {MAX_PROJECTION_TEXT_LENGTH} characters."
            )
        return
    raise ValueError(f"payload values must be finite scalars, got {type(value).__name__}.")


class ObservedEvent(pydantic.BaseModel):
    """One register row read back for grounding, newest first when queried."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Recent stimulus observed by the bot's event register. payload mirrors what the "
                "admitted environment stimulus looked like as bounded scalar fields; selections "
                "are never part of it."
            ),
            "examples": [
                {
                    "stimulus_name": "environment.sound",
                    "payload": {"kind": "knock", "source": "device-a"},
                    "event_id": "turn-1",
                    "source": "device-a",
                    "created_at": "2026-07-07T12:00:00.123456+00:00",
                }
            ],
        },
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


def _event_record(event: hsm.Event[typing.Any]) -> RecordData:
    """Build the register record for one admitted stimulus event."""

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


class StmEventMemory(store.MemoryStore):
    """Short-term memory of admitted stimuli: fixed-capacity rolling event register.

    Public surface is two direct primitives (same role as ``MemoryStore.execute``):
    ``record`` writes one stimulus projection and evicts beyond capacity inside one
    transaction; ``recent`` reads newest-first entries inside the injected recency
    window. The inherited generic transaction surface stays available for maintenance.
    """

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = store.InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = store.OutputData

    _capacity: int
    _recency_window: datetime.timedelta
    _clock: collections.abc.Callable[[], datetime.datetime]

    def __init__(
        self,
        *,
        engine: typing.Any | None = None,
        connection: typing.Any | None = None,
        database: str = ":memory:",
        capacity: int = _DEFAULT_CAPACITY,
        recency_window: datetime.timedelta | None = None,
        clock: collections.abc.Callable[[], datetime.datetime] | None = None,
    ) -> None:
        super().__init__(engine=engine, connection=connection, database=database)
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if recency_window is not None and recency_window <= datetime.timedelta():
            raise ValueError("recency_window must be positive.")
        self._capacity = capacity
        self._recency_window = recency_window if recency_window is not None else _DEFAULT_RECENCY_WINDOW
        self._clock = clock if clock is not None else (lambda: datetime.datetime.now(datetime.timezone.utc))

    def record(self, event: hsm.Event[typing.Any]) -> RecordData:
        """Write one admitted stimulus projection and evict beyond capacity in one transaction."""

        data = _event_record(event)
        insert_clause = insert(schema.stm_events_table).values(
            stm_id=uuid.uuid4().hex,
            stimulus_name=data.stimulus_name,
            payload_json=json.dumps(data.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            event_id=data.event_id,
            source=data.source,
            target=data.target,
            created_at=_utc_text(self._clock()),
        )
        evict_clause = delete(schema.stm_events_table).where(
            schema.stm_events_table.c.stm_id.not_in(
                select(schema.stm_events_table.c.stm_id)
                .order_by(schema.stm_events_table.c.created_at.desc(), schema.stm_events_table.c.stm_id.desc())
                .limit(self._capacity)
            )
        )
        _ = self.execute(store.InputData(statements=store.compile_statements(insert_clause, evict_clause)))
        return data

    def recent(
        self,
        stimulus_name: str | None = None,
        *,
        limit: int = _DEFAULT_CAPACITY,
    ) -> tuple[ObservedEvent, ...]:
        """Newest-first register entries inside the recency window, optionally per stimulus."""

        if limit < 1 or limit > _MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}.")
        cutoff = _utc_text(self._clock() - self._recency_window)
        clause = (
            select(
                schema.stm_events_table.c.stimulus_name,
                schema.stm_events_table.c.payload_json,
                schema.stm_events_table.c.event_id,
                schema.stm_events_table.c.source,
                schema.stm_events_table.c.target,
                schema.stm_events_table.c.created_at,
            )
            .where(schema.stm_events_table.c.created_at >= cutoff)
            .order_by(schema.stm_events_table.c.created_at.desc(), schema.stm_events_table.c.stm_id.desc())
            .limit(limit)
        )
        if stimulus_name is not None:
            clause = clause.where(schema.stm_events_table.c.stimulus_name == stimulus_name)
        output = self.execute(store.InputData(statements=store.compile_statements(clause)))
        rows = tuple(row.as_mapping() for row in output.results[0].rows)
        return tuple(self._observed_from_row(row) for row in rows)

    @staticmethod
    def _observed_from_row(row: dict[str, store.ParameterValue]) -> ObservedEvent:
        stimulus_name = row.get("stimulus_name")
        payload_json = row.get("payload_json")
        created_at = row.get("created_at")
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
            stimulus_name=stimulus_name,
            payload=_projection_payload_from_json(payload),
            event_id=_optional_text(row.get("event_id")),
            source=_optional_text(row.get("source")),
            target=_optional_text(row.get("target")),
            created_at=created_at,
        )


__all__ = [
    "MAX_PROJECTION_TEXT_LENGTH",
    "ObservedEvent",
    "RecordData",
    "StmEventMemory",
]
