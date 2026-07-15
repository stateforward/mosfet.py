"""Build processing inputs and enforce cognition's body-action constraints."""

from __future__ import annotations

from .. import processing

import collections.abc
import typing

import hsm

from .input import InputData

_ACTION_SOURCE_METADATA_KEY = "bot.cognition.action_source"


async def dispatch_selected_events(
    ctx: hsm.Context,
    input: processing.InputData,
    selections: processing.Events,
    *,
    operation_id: str,
    source: hsm.Instance,
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> None:
    """Validate body-action constraints, then dispatch through generic Processing."""

    import bot
    from bot import device

    event_metadata = dict(metadata or {})
    event_metadata[_ACTION_SOURCE_METADATA_KEY] = source
    candidates = tuple(
        name for name, actor in input.actors.items() if name != "bot" and isinstance(actor, device.Device)
    )
    restricted = event_metadata.get("bot.focus_candidates")
    if isinstance(restricted, collections.abc.Sequence) and not isinstance(restricted, str | bytes | bytearray):
        candidates = tuple(item for item in restricted if isinstance(item, str) and item)

    for selection in selections:
        if selection.event == bot.FocusDeviceEvent.name:
            if selection.target is not None and selection.target != "bot":
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
            data = bot.FocusDeviceEventData.model_validate(selection.data or {})
            bot_actor = input.actors.get("bot")
            if data.device not in candidates and (candidates or isinstance(bot_actor, bot.Bot)):
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
        elif selection.event == bot.ClearFocusEvent.name and selection.target not in (None, "bot"):
            raise RuntimeError("Processing selected clear_focus for a non-bot target.")

    await processing.dispatch_selected_events(
        ctx,
        input,
        selections,
        operation_id=operation_id,
        source=source,
        metadata=event_metadata,
    )


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
    extra_actors: collections.abc.Mapping[str, hsm.Instance] | None = None,
) -> processing.InputData:
    """Build a processing input: stimulus + schemas + named actors.

    ``extra_actors`` lets a host put sibling abilities (e.g. reasoning) on the actor map
    so intuition can multi-select and dispatch them like any other ability.
    """

    actors: dict[str, hsm.Instance] = dict(cognition_input.actors)
    if extra_actors:
        actors.update(dict(extra_actors))
    return processing.InputData(
        input=cognition_input.stimulus,
        schemas=_schemas_from_actors(actors),
        actors=actors,
    )


__all__ = [
    "build_processing_input",
    "dispatch_selected_events",
    "events_from_instance",
]
