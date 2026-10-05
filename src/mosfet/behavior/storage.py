"""Canonical relational storage for Starlark behaviors (SQLAlchemy Core).

Autonomy loads these tables at attach; Reflection create/change/break writes them.
Autonomy also writes usage counters (used / failed) after behavior outcomes.
Tables register on the shared memory ``metadata``; memory migrations
(``mosfet.abilities.memory.migrations``) bring them up with the rest of bot storage.

Maps to ``behavior.Instance``:

- ``bot_behavior`` — one row per behavior (name PK, source, status, usage telemetry)
- ``bot_behavior_trigger`` — normalized trigger event names for Autonomy matching

DRAFT and BROKEN behaviors stay in inventory so Reflection can change them later.
Autonomy loads only ``status=ACTIVE`` rows.
"""

from __future__ import annotations

import collections.abc
import datetime
import json
import typing

from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy.sql import ClauseElement

from mosfet.abilities.memory.schema import metadata

from .instance import (
    STATUS_ACTIVE,
    STATUS_BROKEN,
    STATUS_DRAFT,
    Instance,
    Status,
)

BEHAVIOR_TABLE = "bot_behavior"
BEHAVIOR_TRIGGER_TABLE = "bot_behavior_trigger"

behavior_table = Table(
    BEHAVIOR_TABLE,
    metadata,
    # Stable PascalCase model name (Instance.name); primary install identity.
    Column("name", Text, primary_key=True),
    # Full Starlark HSM program (input_event, output_event, behavior = hsm.define(...)).
    Column("source", Text, nullable=False),
    Column("description", Text, nullable=False, server_default=""),
    # JSON array of strings (Instance.examples); text for dialect neutrality.
    Column("examples_json", Text, nullable=False, server_default="[]"),
    # ACTIVE | DRAFT | BROKEN — Autonomy runs ACTIVE only.
    Column("status", Text, nullable=False, server_default=STATUS_ACTIVE),
    # Optional note for current status (validation, break reason, …); empty when unset.
    Column("status_reason", Text, nullable=False, server_default=""),
    # UTC ISO-8601 when status last changed; empty when never set.
    Column("status_updated_at", Text, nullable=False, server_default=""),
    # Practice telemetry: handled uses only (canonical "used").
    Column("used_count", Text, nullable=False, server_default="0"),
    Column("last_used_at", Text, nullable=False, server_default=""),
    # Runtime failures while attempting the behavior.
    Column("failed_count", Text, nullable=False, server_default="0"),
    Column("last_failed_at", Text, nullable=False, server_default=""),
    Column("created_at", Text, nullable=False, server_default=func.now()),
    Column("updated_at", Text, nullable=False, server_default=func.now()),
)

behavior_trigger_table = Table(
    BEHAVIOR_TRIGGER_TABLE,
    metadata,
    Column(
        "behavior_name",
        Text,
        ForeignKey(f"{BEHAVIOR_TABLE}.name", ondelete="CASCADE"),
        primary_key=True,
    ),
    # External event name that may activate this behavior (e.g. environment.sound).
    Column("trigger", Text, primary_key=True),
    Index("bot_behavior_trigger_event_idx", "trigger"),
)


def utc_now_iso() -> str:
    """UTC timestamp string for inventory telemetry columns."""

    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _examples_json(examples: tuple[str, ...]) -> str:
    return json.dumps(list(examples), separators=(",", ":"), ensure_ascii=False)


def _parse_examples(raw: object) -> tuple[str, ...]:
    if raw is None or raw == "":
        return ()
    if not isinstance(raw, str):
        return ()
    try:
        value = typing.cast("object", json.loads(raw))
    except json.JSONDecodeError:
        return ()
    if not isinstance(value, list):
        return ()
    return tuple(item for item in typing.cast("list[object]", value) if isinstance(item, str))


def _optional_text(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped or None
    return None


def _parse_count(raw: object) -> int:
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return max(0, raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return 0
        try:
            return max(0, int(stripped))
        except ValueError:
            return 0
    return 0


def _count_text(value: int) -> str:
    return str(max(0, value))


def _optional_text_column(value: str | None) -> str:
    return value if value else ""


def _parse_status(raw: object) -> Status:
    if isinstance(raw, str):
        normalized = raw.strip().upper()
        if normalized in {STATUS_ACTIVE, STATUS_DRAFT, STATUS_BROKEN}:
            return typing.cast(Status, normalized)
        # Legacy boolean column values if any rows still use them.
        if normalized in {"TRUE", "1", "YES"}:
            return STATUS_BROKEN
        if normalized in {"FALSE", "0", "NO", ""}:
            return STATUS_ACTIVE
    if isinstance(raw, bool):
        return STATUS_BROKEN if raw else STATUS_ACTIVE
    return STATUS_ACTIVE


def select_all_behaviors_clauses() -> tuple[ClauseElement, ClauseElement]:
    """Core clauses: all behavior rows (any status), then all triggers."""

    behaviors = select(behavior_table).order_by(behavior_table.c.name)
    triggers = select(behavior_trigger_table).order_by(
        behavior_trigger_table.c.behavior_name,
        behavior_trigger_table.c.trigger,
    )
    return behaviors, triggers


def select_active_behaviors_clauses() -> tuple[ClauseElement, ClauseElement]:
    """Core clauses: ACTIVE behaviors only (Autonomy runtime load)."""

    behaviors = select(behavior_table).where(behavior_table.c.status == STATUS_ACTIVE).order_by(behavior_table.c.name)
    triggers = select(behavior_trigger_table).order_by(
        behavior_trigger_table.c.behavior_name,
        behavior_trigger_table.c.trigger,
    )
    return behaviors, triggers


def select_behavior_by_name_clauses(name: str) -> tuple[ClauseElement, ClauseElement]:
    """Core clauses to load one behavior and its triggers (any status)."""

    behaviors = select(behavior_table).where(behavior_table.c.name == name)
    triggers = select(behavior_trigger_table).where(behavior_trigger_table.c.behavior_name == name)
    return behaviors, triggers


def delete_behavior_clauses(name: str) -> tuple[ClauseElement, ClauseElement]:
    """Core clauses to remove a behavior (triggers first for non-CASCADE dialects)."""

    delete_triggers = delete(behavior_trigger_table).where(behavior_trigger_table.c.behavior_name == name)
    delete_behavior = delete(behavior_table).where(behavior_table.c.name == name)
    return delete_triggers, delete_behavior


def insert_behavior_clauses(instance: Instance) -> tuple[ClauseElement, ...]:
    """Core clauses to insert one behavior row plus its trigger rows."""

    clauses: list[ClauseElement] = [
        insert(behavior_table).values(
            name=instance.name,
            source=instance.source,
            description=instance.description,
            examples_json=_examples_json(instance.examples),
            status=instance.status,
            status_reason=_optional_text_column(instance.status_reason),
            status_updated_at=_optional_text_column(instance.status_updated_at),
            used_count=_count_text(instance.used_count),
            last_used_at=_optional_text_column(instance.last_used_at),
            failed_count=_count_text(instance.failed_count),
            last_failed_at=_optional_text_column(instance.last_failed_at),
        )
    ]
    for trigger in instance.triggers:
        clauses.append(
            insert(behavior_trigger_table).values(
                behavior_name=instance.name,
                trigger=trigger,
            )
        )
    return tuple(clauses)


def replace_behavior_clauses(instance: Instance) -> tuple[ClauseElement, ...]:
    """Core clauses for create/change/break/usage upsert: delete existing, then insert replacement."""

    return (*delete_behavior_clauses(instance.name), *insert_behavior_clauses(instance))


def mark_used(instance: Instance, *, at: str | None = None) -> Instance:
    """Return a copy with used_count / last_used_at advanced (handled Autonomy outcome)."""

    stamp = at if at else utc_now_iso()
    return instance.model_copy(
        update={
            "used_count": instance.used_count + 1,
            "last_used_at": stamp,
        }
    )


def mark_failed(instance: Instance, *, at: str | None = None) -> Instance:
    """Return a copy with failed_count / last_failed_at advanced (runtime failure)."""

    stamp = at if at else utc_now_iso()
    return instance.model_copy(
        update={
            "failed_count": instance.failed_count + 1,
            "last_failed_at": stamp,
        }
    )


def preserve_usage(target: Instance, source: Instance | None) -> Instance:
    """Copy usage counters from ``source`` onto ``target`` (same behavior identity across change)."""

    if source is None:
        return target
    return target.model_copy(
        update={
            "used_count": source.used_count,
            "last_used_at": source.last_used_at,
            "failed_count": source.failed_count,
            "last_failed_at": source.last_failed_at,
        }
    )


def set_status(
    instance: Instance,
    status: Status,
    *,
    reason: str | None = None,
    at: str | None = None,
) -> Instance:
    """Return a copy with status / status_reason / status_updated_at (usage preserved)."""

    stamp = at if at else utc_now_iso()
    cleaned = reason.strip() if isinstance(reason, str) and reason.strip() else None
    return instance.model_copy(
        update={
            "status": status,
            "status_reason": cleaned,
            "status_updated_at": stamp,
        }
    )


def mark_active(instance: Instance, *, at: str | None = None) -> Instance:
    """ACTIVE status; clears status_reason."""

    return set_status(instance, STATUS_ACTIVE, reason=None, at=at)


def mark_draft(instance: Instance, *, reason: str | None = None, at: str | None = None) -> Instance:
    """DRAFT status (create stub or failed validation while authoring)."""

    return set_status(instance, STATUS_DRAFT, reason=reason, at=at)


def mark_broken(instance: Instance, *, reason: str | None = None, at: str | None = None) -> Instance:
    """BROKEN status (explicit retire via bot.behavior.break)."""

    return set_status(instance, STATUS_BROKEN, reason=reason, at=at)


def instances_from_behavior_results(
    behavior_rows: collections.abc.Sequence[collections.abc.Mapping[str, object]],
    trigger_rows: collections.abc.Sequence[collections.abc.Mapping[str, object]],
) -> tuple[Instance, ...]:
    """Build ``Instance`` values from SELECT result row mappings."""

    triggers_by_name: dict[str, list[str]] = {}
    for row in trigger_rows:
        behavior_name = row.get("behavior_name")
        trigger: object | None = row.get("trigger")
        if isinstance(behavior_name, str) and isinstance(trigger, str):
            triggers_by_name.setdefault(behavior_name, []).append(trigger)

    instances: list[Instance] = []
    for row in behavior_rows:
        name = row.get("name")
        source = row.get("source")
        if not isinstance(name, str) or not isinstance(source, str):
            continue
        description = row.get("description")
        # Prefer status column; fall back to legacy broken bool if present without status.
        if "status" in row or row.get("status") is not None:
            status = _parse_status(row.get("status"))
        elif "broken" in row:
            status = _parse_status(row.get("broken"))
        else:
            status = STATUS_ACTIVE
        status_reason = _optional_text(row.get("status_reason"))
        if status_reason is None:
            status_reason = _optional_text(row.get("broken_reason"))
        status_updated_at = _optional_text(row.get("status_updated_at"))
        if status_updated_at is None:
            status_updated_at = _optional_text(row.get("broken_at"))
        instances.append(
            Instance(
                name=name,
                source=source,
                description=description if isinstance(description, str) else "",
                examples=_parse_examples(row.get("examples_json")),
                triggers=tuple(triggers_by_name.get(name, ())),
                status=status,
                status_reason=status_reason,
                status_updated_at=status_updated_at,
                used_count=_parse_count(row.get("used_count")),
                last_used_at=_optional_text(row.get("last_used_at")),
                failed_count=_parse_count(row.get("failed_count")),
                last_failed_at=_optional_text(row.get("last_failed_at")),
            )
        )
    return tuple(instances)


__all__ = [
    "BEHAVIOR_TABLE",
    "BEHAVIOR_TRIGGER_TABLE",
    "delete_behavior_clauses",
    "behavior_table",
    "behavior_trigger_table",
    "insert_behavior_clauses",
    "instances_from_behavior_results",
    "mark_active",
    "mark_broken",
    "mark_draft",
    "mark_failed",
    "mark_used",
    "metadata",
    "preserve_usage",
    "replace_behavior_clauses",
    "select_active_behaviors_clauses",
    "select_all_behaviors_clauses",
    "select_behavior_by_name_clauses",
    "set_status",
    "utc_now_iso",
]
