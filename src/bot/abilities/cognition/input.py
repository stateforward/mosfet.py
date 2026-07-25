"""InputData event payload for the cognition (`bot.ability.cognition.input`).

Matches the usual ability/cognition naming pattern: ``*InputData`` data for ``*.input`` events.

Bot forwards live robot state — not a prebuilt decision input:

- ``stimulus``: the event that had no body transition
- ``abilities``: ability instances on the bot
- ``actors``: named HSM instances (devices, input/output abilities, …) the cognition may snapshot/dispatch to
- ``focus``: name of the instance the bot is looking at, if any
"""

from __future__ import annotations

import bot
from .. import ability
from .. import processing

import collections.abc
import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema


class InputData(pydantic.BaseModel):
    """Payload of ``bot.ability.cognition.input``: stimulus plus live body references for one turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "InputData payload for the cognition. Bot supplies the stimulus, abilities, named HSM instances, and "
                "focus name; the cognition builds tools from snapshots. Selected events are dispatched by Processing."
            ),
            "examples": [
                {
                    "focus": "phone",
                    "focus_candidates": ["phone"],
                }
            ],
        },
    )

    stimulus: SkipJsonSchema[bot.BotInputData] = pydantic.Field(
        exclude=True,
        repr=False,
        description="Event or bot input payload that triggered the cognition.",
    )
    abilities: SkipJsonSchema[tuple[ability.Ability[typing.Any, typing.Any], ...]] = pydantic.Field(
        default=(),
        exclude=True,
        repr=False,
        description="Ability instances currently attached on the bot body.",
    )
    actors: SkipJsonSchema[collections.abc.Mapping[str, hsm.Instance]] = pydantic.Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description=(
            "Named HSM instances available for snapshot and dispatch (devices plus bot input/output "
            "and other attached abilities, e.g. speaking)."
        ),
    )
    focus: str | None = pydantic.Field(
        default=None,
        description="Instance name the bot is looking at, if any.",
        examples=["phone"],
    )
    focus_candidates: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Instance names that may become focus for this turn (body attention policy).",
        examples=[["phone"], ["phone", "browser"]],
    )


def is_input(value: object) -> typing.TypeGuard[InputData]:
    return isinstance(value, InputData)


def build_processing_input(
    cognition_input: InputData,
    *,
    extra_actors: collections.abc.Mapping[str, hsm.Instance] | None = None,
    authority: hsm.Instance | None = None,
) -> processing.InputData:
    """Build the deliberative input and callable schemas for one cognition turn.

    Offered tools are deduced from each actor's live HSM transition snapshot
    (``enabled_call_events``). Body attention (``bot.focus_device`` /
    ``bot.clear_focus``) appears when the bot actor's active topology enables
    those ``processing.EventKind`` transitions — never via a parallel hard-coded schema
    allowlist. The cognition host itself is included when ``authority`` is the
    live Cognition instance so host model-offerable events such as
    ``bot.ability.cognition.ignore`` appear only when that topology enables them.
    """

    actors: dict[str, hsm.Instance] = dict(cognition_input.actors)
    if extra_actors:
        actors.update(extra_actors)
    if authority is not None and not any(actor is authority for actor in actors.values()):
        # Host call surface (ignore, …) comes from Cognition's snapshot, not a schema allowlist.
        actors = {**actors, "cognition": authority}
    schemas: list[processing.Event[typing.Any]] = []
    seen: set[str] = set()
    for instance in actors.values():
        for event in processing.enabled_call_events(instance):
            if event.name not in seen:
                seen.add(event.name)
                schemas.append(event)
    return processing.InputData(
        input=cognition_input.stimulus,
        schemas=tuple(schemas),
        actors=actors,
        authority=authority,
    )


__all__ = [
    "InputData",
    "build_processing_input",
    "is_input",
]
