"""The one write path for behavior inventory rows, and the record each write leaves.

Autonomy (usage), Reflection (break), and Revision (draft, accept) all replace a behavior row
straight in memory rather than through a transition another machine observes. Every such write
goes through `store_behavior`, which reads the row it replaces and reports the change once:

- metric ``bot.behavior.inventory.write.count`` with low-cardinality ``bot.behavior.status.from``,
  ``bot.behavior.status.to``, and ``bot.behavior.write.cause``;
- under ``BOT_OTEL_CAPTURE=full``, a log record in the active trace naming the behavior, the
  status it moved from and to, the status reason, its source revision, and the cause.
"""

from __future__ import annotations

import hashlib
import typing

from opentelemetry import metrics

from mosfet.abilities import memory
from mosfet.behavior import storage as behavior_storage
from mosfet.behavior.instance import Instance
from mosfet.telemetry import capture

_SCOPE = "bot.abilities.cognition.inventory"
_CAPTURE_LOGGER_NAME = "bot.telemetry.behavior.status"
_ABSENT = "absent"

_WRITES = metrics.get_meter(_SCOPE).create_counter(
    "bot.behavior.inventory.write.count",
    unit="{write}",
    description="Behavior inventory rows written to memory, by status transition and cause.",
)

Cause = typing.Literal["usage", "break", "draft", "accept"]


def revision(behavior: Instance) -> str:
    """A short, stable content revision of the behavior's source (empty source reads ``empty``)."""

    if not behavior.source:
        return "empty"
    return hashlib.sha256(behavior.source.encode("utf-8")).hexdigest()[:12]


def _load(store: memory.Memory, name: str) -> Instance | None:
    output = store.execute(
        memory.InputData(statements=memory.compile_statements(*behavior_storage.select_behavior_by_name_clauses(name)))
    )
    if len(output.results) < 2:
        return None
    loaded = behavior_storage.instances_from_behavior_results(
        tuple(row.as_mapping() for row in output.results[0].rows),
        tuple(row.as_mapping() for row in output.results[1].rows),
    )
    return loaded[0] if loaded else None


def store_behavior(store: memory.Memory, behavior: Instance, *, cause: Cause) -> None:
    """Replace ``behavior``'s inventory row and record the status change it makes."""

    previous = _load(store, behavior.name)
    _ = store.execute(
        memory.InputData(statements=memory.compile_statements(*behavior_storage.replace_behavior_clauses(behavior)))
    )
    source = previous.status if previous is not None else _ABSENT
    _WRITES.add(
        1,
        attributes={
            "bot.behavior.status.from": source,
            "bot.behavior.status.to": behavior.status,
            "bot.behavior.write.cause": cause,
        },
    )
    if capture.full_enabled():
        capture.emit(
            _CAPTURE_LOGGER_NAME,
            body={
                "behavior": behavior.name,
                "from": source,
                "to": behavior.status,
                "reason": behavior.status_reason,
                "revision": revision(behavior),
                "previous_revision": revision(previous) if previous is not None else None,
                "cause": cause,
            },
            attributes={"component": "cognition.inventory", "stage": "behavior_status"},
        )


__all__ = ["Cause", "revision", "store_behavior"]
