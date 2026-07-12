"""Build processing inputs from live actors (event dispatch is owned by Processing)."""

from __future__ import annotations

from .. import processing

import collections.abc
import typing

import hsm

from .input import InputData


def events_from_instance(instance: hsm.Instance) -> tuple[processing.Event[typing.Any], ...]:
    """Enabled CallEventKind events on ``instance`` (same discovery Processing uses)."""

    return processing.enabled_call_events(instance)


def _schemas_from_actors(
    actors: collections.abc.Mapping[str, hsm.Instance],
) -> tuple[processing.Event[typing.Any], ...]:
    """Collect selectable CallEventKind schemas from actors (plus bot body focus events)."""

    schemas: list[processing.Event[typing.Any]] = []
    seen: set[str] = set()
    for instance in actors.values():
        for event in processing.enabled_call_events(instance):
            if event.name in seen:
                continue
            seen.add(event.name)
            schemas.append(event)
    if "bot" in actors:
        from bot import ClearFocusEvent, FocusDeviceEvent

        for event in (FocusDeviceEvent, ClearFocusEvent):
            if event.name in seen:
                continue
            seen.add(event.name)
            schemas.append(event)
    return tuple(schemas)


def build_processing_input(
    cognition_input: InputData,
    *,
    owner: hsm.Instance | None = None,
    extra_actors: collections.abc.Mapping[str, hsm.Instance] | None = None,
) -> processing.InputData:
    """Build a processing input: stimulus + schemas + named actors.

    ``extra_actors`` lets a host put sibling abilities (e.g. reasoning) on the actor map
    so intuition can multi-select and dispatch them like any other ability.
    """

    actors: dict[str, hsm.Instance] = dict(cognition_input.actors)
    if extra_actors:
        actors.update(dict(extra_actors))
    if owner is not None:
        actors.setdefault("bot", owner)
    return processing.InputData(
        input=cognition_input.stimulus,
        schemas=_schemas_from_actors(actors),
        actors=actors,
    )


__all__ = [
    "build_processing_input",
    "events_from_instance",
]
