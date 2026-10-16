"""InputData event payload for the cognition (`bot.ability.cognition.input`).

Matches the usual ability/cognition naming pattern: ``*InputData`` data for ``*.input`` events.

Bot forwards live robot state — not a prebuilt decision input:

- ``stimulus``: the event that had no body transition
- ``abilities``: innate and acquired ability instances on the bot
- ``actors``: named HSM instances (devices, innate, and acquired abilities) the cognition may snapshot/dispatch to
- ``focus``: name of the instance the bot is looking at, if any
- ``participants``: who took part in the stimulus, as the producer of this input stamped them

What the bot knows of itself (:class:`SelfData`) is not body context: Cognition keeps it, as it was
told by its own children, and hands it to :func:`build_processing_input` for every turn.
"""

from __future__ import annotations

import mosfet
from .. import ability
from .. import processing
from .. import vocal

# SelfData's field is named ``persona``, which would shadow the package in the class body (these
# annotations are resolved there), so the package goes by its role in this module.
from .. import persona as self_knowledge
from ..ability import Effort, effort_within

import collections.abc
import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from mosfet.environment import Environment


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

    stimulus: SkipJsonSchema[mosfet.InputData] = pydantic.Field(
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
            "Named HSM instances available for snapshot and dispatch: devices, innate/acquired abilities, and input "
            "abilities (sense organs cognition may direct through what their live topology enables). Output ports "
            "and ability-owned child machines are composition wiring and are excluded."
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
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=32,
        description=(
            "Who took part in the stimulus, stamped by the producer of this input. Empty when the producer "
            "knows no one (environment sound, routine tick)."
        ),
        examples=[[{"ref": "caller", "role": "source"}, {"ref": "bot", "role": "target"}]],
    )


class SelfData(pydantic.BaseModel):
    """What the bot knows of itself on a turn, shown to the model after the world block (P-7).

    Cognition builds it from what its own children last told it, by event; nothing here reads a
    peer's state. The traits are its owner's, the description is the bot's own and is shown as it
    wrote it: marked self-written, never filtered. The voice was given to it at inception and is
    shown with what it can do with an expression, so the bot knows its capacity before it speaks.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "What the bot knows of itself this turn: the traits its owner gave it, the description of "
                "itself it wrote (or none when it has not written one), and the voice it was given when it was "
                "made, with what that voice can do."
            ),
        },
    )

    persona: self_knowledge.OutputData | None = pydantic.Field(
        default=None,
        description="The bot's owner-given traits and its self-written description; None without a persona.",
    )
    voice: vocal.voice.VoiceData | None = pydantic.Field(
        default=None,
        description="The voice the bot was given at inception and what it can do; None until it has one.",
    )

    def __model_repr__(self) -> str:
        """The self block: ``<persona>`` holding the traits (given by the owner), the description (by you),
        and the voice (given at inception)."""

        known = "" if self.persona is None else self.persona.__model_repr__()
        voiced = "" if self.voice is None else self.voice.__model_repr__()
        return f"<persona>{known}{voiced}</persona>"


def stimulus_name(stimulus: mosfet.InputData) -> str | None:
    """Low-cardinality stimulus label: the event name, or ``bot.input`` for device input."""

    if isinstance(stimulus, mosfet.InputEventData):
        return mosfet.InputEvent.name
    return stimulus.name


def is_input(value: object) -> typing.TypeGuard[InputData]:
    return isinstance(value, InputData)


def _turn_instructions(actors: collections.abc.Mapping[str, hsm.Instance], known: SelfData | None) -> str | None:
    """Compose one turn's world block, then the self block after it.

    The environment the bot is in takes the snapshot (see ``Environment.model_snapshot``):
    it owns the world's own identity, its addressing scope, and how a referenced device's
    snapshot folds into the block — cognition invents none of that. This function finds the live
    bot actor for this turn and hands it to its own environment as the perspective the block is
    taken from, then follows it with what the bot knows of itself (``known``). Both blocks are
    the bot's perspective: when there is no live ``bot`` actor this turn, there is no perspective
    to render from and no instructions to invent.
    """

    owner = actors.get("bot")
    if owner is None:
        return None
    blocks = (
        Environment.from_context(owner.context()).model_snapshot(owner),
        None if known is None else known.__model_repr__(),
    )
    return "\n".join(block for block in blocks if block) or None


def build_processing_input(
    cognition_input: InputData,
    *,
    extra_actors: collections.abc.Mapping[str, hsm.Instance] | None = None,
    authority: hsm.Instance | None = None,
    known: SelfData | None = None,
) -> processing.InputData:
    """Build the deliberative input and callable schemas for one cognition turn.

    Offered tools are deduced from each actor's live HSM transition snapshot
    (``enabled_call_events``). Body attention (``bot.focus_device`` /
    ``bot.clear_focus``) appears when the bot actor's active topology enables
    those ``processing.EventKind`` transitions — never via a parallel hard-coded schema
    allowlist. The cognition host itself is included when ``authority`` is the
    live Cognition instance so host model-offerable events such as
    ``bot.ability.cognition.ignore`` appear only when that topology enables them.

    The per-turn instructions open a live device-state block (same snapshots), followed by the
    self block from ``known`` (what the bot knows of itself), which the provider sends as the
    system message; any static Processing policy is stamped ahead of it at apply time.

    The frame offers every actor; each stage narrows it to its own capacity with
    ``within_effort`` when the frame reaches it.
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
        instructions=_turn_instructions(actors, known),
    )


def within_effort(frame: processing.InputData, ceiling: Effort | None) -> processing.InputData:
    """``frame`` as a stage whose effort ceiling is ``ceiling`` may use it.

    Stage capacity: an actor rated above the ceiling drops from the frame's actors, so from its
    offered events and its dispatch map, which are collected again from the actors left. The
    authority (the host's own call surface) is never dropped, and the world and self blocks stay
    unless the bot actor they are rendered from was dropped. ``None`` is unbounded, and a frame
    that loses no actor is returned as it is.
    """

    if ceiling is None:
        return frame
    actors = {
        key: actor
        for key, actor in frame.actors.items()
        if actor is frame.authority or effort_within(getattr(actor, "effort", Effort.XS), ceiling)
    }
    if len(actors) == len(frame.actors):
        return frame
    schemas, actor_events = processing.collect_offered_events(actors)
    return frame.model_copy(
        update={
            "actors": actors,
            "schemas": schemas,
            "actor_events": actor_events,
            "instructions": frame.instructions if "bot" in actors else None,
        }
    )


__all__ = [
    "InputData",
    "SelfData",
    "build_processing_input",
    "is_input",
    "stimulus_name",
    "within_effort",
]
