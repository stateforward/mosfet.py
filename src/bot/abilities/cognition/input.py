"""InputData event payload for the cognition (`bot.ability.cognition.input`).

Matches the usual ability/cognition naming pattern: ``*InputData`` data for ``*.input`` events.

Bot forwards live robot state — not a prebuilt decision input:

- ``stimulus``: the event that had no body transition
- ``abilities``: innate and acquired ability instances on the bot
- ``actors``: named HSM instances (devices, innate, and acquired abilities) the cognition may snapshot/dispatch to
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

from bot.environment import Environment


class InputData(pydantic.BaseModel):
    """Payload of ``bot.ability.cognition.input``: stimulus plus live body references for one turn."""

    __model_facing_input__: typing.ClassVar[bool] = True

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
                    "focus": "device-a",
                    "focus_candidates": ["device-a"],
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
        description="Innate and acquired ability instances currently available to cognition.",
    )
    actors: SkipJsonSchema[collections.abc.Mapping[str, hsm.Instance]] = pydantic.Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description=(
            "Named HSM instances available for snapshot and dispatch: devices, innate/acquired abilities, "
            "not ability-owned child machines. Body input/output ports are internal composition wiring and are excluded."
        ),
    )
    focus: str | None = pydantic.Field(
        default=None,
        description="Instance name the bot is looking at, if any.",
        examples=["device-a"],
    )
    focus_candidates: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Instance names that may become focus for this turn (body attention policy).",
        examples=[["device-a"], ["device-a", "device-b"]],
    )


def is_input(value: object) -> typing.TypeGuard[InputData]:
    return isinstance(value, InputData)


def _device_state_instructions(actors: collections.abc.Mapping[str, hsm.Instance]) -> str | None:
    """Compose one turn's world block by asking the bot's own environment to render it.

    The environment the bot is in takes the snapshot (see ``Environment.model_snapshot``):
    it owns the world's own identity, its addressing scope, and how a referenced device's
    snapshot folds into the block — cognition invents none of that. This function's whole job
    is finding the live bot actor for this turn and handing it to its own environment as the
    perspective the block is taken from; when there is no live ``bot`` actor this turn, there is
    no perspective to render from and no instructions to invent.
    """

    owner = actors.get("bot")
    if owner is None:
        return None
    return Environment.from_context(owner.context()).model_snapshot(owner)


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

    The per-turn instructions open a live device-state block (same snapshots), which the
    provider sends as the system message; any static Processing policy is stamped ahead of
    it at apply time.
    """

    actors: dict[str, hsm.Instance] = dict(cognition_input.actors)
    if extra_actors:
        actors.update(extra_actors)
    if authority is not None and not any(actor is authority for actor in actors.values()):
        # Host call surface (ignore, …) comes from Cognition's snapshot, not a schema allowlist.
        actors = {**actors, "cognition": authority}
    schemas, actor_events = processing.collect_offered_events(actors)
    return processing.InputData(
        input=cognition_input.stimulus,
        schemas=schemas,
        actors=actors,
        actor_events=actor_events,
        authority=authority,
        instructions=_device_state_instructions(actors),
    )


__all__ = [
    "InputData",
    "build_processing_input",
    "is_input",
]
