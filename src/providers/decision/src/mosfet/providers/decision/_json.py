"""JSON-ness adapter: bot-model payloads are pydantic-first; payload candidates are JSON values.

A payload candidate (the value a decision answer's label stands for) is a JSON-compatible scalar,
sequence, mapping, or null nested to any depth. This helper converts the stimulus data candidates
are taken from to that shape and raises a :class:`ValueError` for anything else (bytes, custom
objects).
"""

from __future__ import annotations

import collections.abc
import datetime
import typing

import dataclasses
import pydantic

from mosfet import event

JSONValue = str | int | float | bool | None


def jsonable(value: object) -> JSONValue | list[typing.Any] | dict[str, typing.Any]:
    """Convert stimulus data to the payload candidate shape or raise ValueError.

    Media never belongs in a decision request (serialized base64 once blew the request budget, D-R-50):
    a pydantic model or dataclass is read through core's canonical event/Data JSON tree
    (``mosfet.event.event_json_value``), which leaves raw media out instead of encoding it, and bytes
    anywhere else raise. Text is kept whole, however long (D-R-47).
    """

    if value is None or isinstance(value, int | bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("payload candidate float values must be finite.")
        return value
    if isinstance(value, pydantic.BaseModel) or (dataclasses.is_dataclass(value) and not isinstance(value, type)):
        return jsonable(event.event_json_value(value))
    if isinstance(value, dict):
        mapping = typing.cast("dict[typing.Any, typing.Any]", value)
        return {str(key): jsonable(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = typing.cast("collections.abc.Iterable[typing.Any]", value)
        return [jsonable(item) for item in items]
    if isinstance(value, str):
        return value
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    raise ValueError(
        "Payload candidates accept JSON-compatible scalars, sequences, mappings, pydantic models, "
        f"and ISO datetimes; got {type(value).__name__}."
    )


__all__ = ["jsonable"]
