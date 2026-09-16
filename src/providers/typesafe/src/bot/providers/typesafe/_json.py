"""JSON-ness adapter: bot-model payloads are pydantic-first; system_one state is JSONValue.

``state`` accepts only JSON-compatible scalars, sequences, mappings, and nulls nested to
any depth. This helper converts arbitrary caller objects to that shape and raises a
:class:`ValueError` for anything else (bytes, media, custom objects) — the honest
boundary is the provider's documented state contract.
"""

from __future__ import annotations

import collections.abc
import datetime
import typing

import dataclasses
import pydantic

JSONValue = str | int | float | bool | None


def jsonable(value: object) -> JSONValue | list[typing.Any] | dict[str, typing.Any]:
    """Convert a caller object to the system_one state shape or raise ValueError."""

    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("state float values must be finite.")
        return value
    if isinstance(value, pydantic.BaseModel):
        return jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
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
        "system_one state accepts JSON-compatible scalars, sequences, mappings, pydantic models, "
        f"and ISO datetimes; got {type(value).__name__}."
    )


__all__ = ["jsonable"]
