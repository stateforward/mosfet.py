"""Typed payloads of the episodic ``Memory``: what is remembered, searched, fetched, and consolidated.

An episode is one cognition turn as its producer stamped it: the memory rendering of the stimulus
(text and image parts) followed by the bot's own output on that turn, who took part, what the bot chose
to do, and how the turn ended. Memory
keeps every episode as a node of a graph (episodes, the persons they involve, and the concepts
consolidation merges them into), with a canonical vector per node in the embedder's space.

A person is one node with many participant refs, and each ref is exactly one person's. A ref seen
for the first time gets a provisional person (no name); a ref its producer stamped ``own`` is the
bot's, whose person is the reserved ``SELF``. Nobody declares who lives here: the bot learns who
everyone is and links, unlinks, names and forgets them (``PersonData``), and code only refuses what
would make a ref two persons' or change the bot itself.

Read models (``Episode``, ``Node``, ``Candidate``) carry images as ``media.ReleasedImage``
placeholders (reason ``"memory"``) unless a read asked for media. Consolidation sees a leased
``UnitData`` whose nodes are named by unit-scoped aliases, never by node ids, and answers with one
``ConsolidateData`` decision: links, merges, and forgets, or nothing at all.
"""

from __future__ import annotations

import datetime
import enum
import operator
import re
import typing

import mosfet
import pydantic

from mosfet import media

from .. import ability
from ..language import text

MAX_EPISODE_PARTS: typing.Final = 17
"""Most content parts one episode or cue keeps: text parts around at most ``media.MAX_TURN_IMAGES`` images."""

MAX_EPISODE_TEXT_BYTES: typing.Final = 262_144
"""Most UTF-8 bytes of text one episode or cue keeps (a boundary size limit, PY-DATA-002)."""

MAX_PARTICIPANTS: typing.Final = 32
"""Most participants one episode or search carries."""

MAX_EXCLUDED_TURNS: typing.Final = 128
"""Most turn ids one cue leaves out, or one episode goes on with: a turn's stimulus, or each percept of its batch,
and their causal parents."""

_RELATION = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _text_bytes(parts: tuple[text.ContentPart, ...]) -> int:
    return sum(len(part.text.encode("utf-8")) for part in parts if isinstance(part, text.TextPart))


def _check_parts(parts: tuple[text.ContentPart, ...]) -> None:
    if sum(isinstance(part, text.ImagePart) for part in parts) > media.MAX_TURN_IMAGES:
        raise ValueError(f"at most {media.MAX_TURN_IMAGES} image parts are kept")
    if _text_bytes(parts) > MAX_EPISODE_TEXT_BYTES:
        raise ValueError(f"text parts exceed {MAX_EPISODE_TEXT_BYTES} UTF-8 bytes")


class NodeKind(enum.StrEnum):
    """What a memory node is."""

    EPISODE = "episode"
    """One cognition turn as its producer stamped it."""
    PERSON = "person"
    """Someone episodes involve, keyed by person id; the participant refs producers stamp each belong to one."""
    CONCEPT = "concept"
    """A statement consolidation merged out of other nodes, in the model's words."""


class Outcome(enum.StrEnum):
    """How the remembered turn ended."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Lane(enum.StrEnum):
    """Which mechanical lane of a search surfaced a candidate."""

    SIMILAR = "similar"
    """Nearest by vector to the cue."""
    RECENT = "recent"
    """Among the latest nodes."""
    PRESENT = "present"
    """An episode that involves someone present now."""
    LINKED = "linked"
    """One link away from another candidate."""


class FailureKind(enum.StrEnum):
    """Why a memory request failed; values match ``telemetry/FAILURE_KINDS.md``."""

    INVALID_INPUT = "invalid_input"
    """The request is malformed or names nodes it may not (a ref outside the unit, a person to forget)."""
    UNAVAILABLE = "unavailable"
    """Memory is not serving: not attached, detaching, or degraded."""
    STORE_FAILED = "store_failed"
    """The SQL transaction failed and rolled back; nothing was changed."""
    STALE_LEASE = "stale_lease"
    """A consolidation decision or release named a unit that is no longer leased."""
    TIMEOUT = "timeout"
    """The request did not complete within Memory's operation timeout."""
    CONFLICT = "conflict"
    """A ref already belongs to another person, or the statement would change the bot itself (``self``) or
    its own refs; nothing was changed."""
    NOT_FOUND = "not_found"
    """The statement names a person or ref memory does not hold."""


class FailureData(ability.FailureData):
    """A memory request failed; ``kind`` says how the requester should respond."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "A memory request failed; kind says how the requester should respond.",
            "examples": [{"message": "unit 3f2a is no longer leased", "kind": "stale_lease"}],
        },
    )

    kind: FailureKind = pydantic.Field(
        description="Closed failure kind; the same value is reported on spans and metrics.",
        examples=["store_failed"],
    )


class Act(pydantic.BaseModel):
    """One thing the bot chose to do on a turn: the event it selected, addressed and with its data."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "event": "bot.ability.speaking.say",
                    "target": "speaking",
                    "data": {"text": "Hello, this is Ada."},
                    "reason": "Answer the caller.",
                }
            ]
        },
    )

    event: str = pydantic.Field(
        min_length=1,
        max_length=200,
        description="Name of the event the bot selected on that turn.",
        examples=["bot.ability.speaking.say"],
    )
    target: str | None = pydantic.Field(
        default=None,
        max_length=200,
        description="The actor the event was addressed to, when the selection named one.",
        examples=["speaking"],
    )
    data: dict[str, pydantic.JsonValue] = pydantic.Field(
        default_factory=dict[str, pydantic.JsonValue],
        description="The selected event's data. Speech the bot produced appears here as the speaking event's data.",
        examples=[{"text": "Hello, this is Ada."}],
    )
    reason: str | None = pydantic.Field(
        default=None,
        max_length=4000,
        description="Why the bot chose it, in its own words, when it said.",
        examples=["Answer the caller."],
    )


class EpisodeData(pydantic.BaseModel):
    """One cognition turn to remember, stamped by the producer that owns the turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "turn_id": "turn-7c1e",
                    "stimulus": "bot.ability.conversation.messages",
                    "parts": [{"kind": "text", "text": "Ada: can you call me back at noon?"}],
                    "participants": [{"ref": "+15555550100", "role": "source"}],
                    "focus": "conversation",
                    "focus_candidates": ["conversation"],
                    "acts": [{"event": "bot.ability.speaking.say", "data": {"text": "Sure, at noon."}}],
                    "outcome": "completed",
                }
            ]
        },
    )

    turn_id: str = pydantic.Field(
        min_length=1,
        max_length=200,
        description="The cognition operation id of the turn (provenance).",
        examples=["turn-7c1e"],
    )
    stimulus: str | None = pydantic.Field(
        default=None,
        max_length=200,
        description="Low-cardinality label of the stimulus: its event name, or 'bot.input' for device input.",
        examples=["bot.ability.conversation.messages"],
    )
    parts: tuple[text.ContentPart, ...] = pydantic.Field(
        default=(),
        max_length=MAX_EPISODE_PARTS,
        description=(
            "The memory rendering of the turn: the stimulus (text parts and at most 8 image parts, in order), then "
            "what the bot output, when it selected anything. Text totals "
            f"at most {MAX_EPISODE_TEXT_BYTES} UTF-8 bytes."
        ),
        examples=[[{"kind": "text", "text": "Ada: can you call me back at noon?"}]],
    )
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PARTICIPANTS,
        description="Who took part in the stimulus, as its producer stamped them.",
        examples=[[{"ref": "+15555550100", "role": "source"}]],
    )
    focus: str | None = pydantic.Field(
        default=None,
        max_length=200,
        description="What cognition attended to on this turn, when it chose a focus.",
        examples=["conversation"],
    )
    focus_candidates: tuple[str, ...] = pydantic.Field(
        default=(),
        max_length=32,
        description="What cognition could have attended to on this turn.",
        examples=[["conversation", "phone"]],
    )
    acts: tuple[Act, ...] = pydantic.Field(
        default=(),
        max_length=32,
        description="What the bot chose to do on this turn, in order; empty when it did nothing.",
    )
    outcome: Outcome = pydantic.Field(
        description="How the turn ended.",
        examples=["completed"],
    )
    continues: tuple[typing.Annotated[str, pydantic.StringConstraints(min_length=1, max_length=200)], ...] = (
        pydantic.Field(
            default=(),
            max_length=MAX_EXCLUDED_TURNS,
            description=(
                "Turn ids of earlier turns this turn goes on with, stamped by the turn's producer: the ids of the "
                "causal parents its stimulus carries (or each percept of its batch carries), nearest first. Speech "
                "admitted through a conversation is first a turn of its own, kept under the speech's id, and then "
                "the exchange that carries that speech as a parent: one experience. When memory keeps an episode "
                "under one of these turn ids (the first such, in this order), this turn extends that episode in "
                "place: its parts follow the episode's, its acts follow the episode's acts, its participants and "
                "focus candidates join the episode's, its focus (when it chose one) and outcome become the "
                "episode's, and the episode is embedded again. Otherwise, or when the extended episode would "
                "exceed what one episode may hold, the turn is kept as an episode of its own under turn_id."
            ),
            examples=[["5e5b95c437364ad5a1763d96a735b673:autonomy", "5e5b95c437364ad5a1763d96a735b673"]],
        )
    )

    @pydantic.model_validator(mode="after")
    def _validate_parts(self) -> typing.Self:
        _check_parts(self.parts)
        return self


class RememberedData(pydantic.BaseModel):
    """An episode was kept."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"node_id": "5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f", "indexed": True}]},
    )

    node_id: str = pydantic.Field(
        min_length=1,
        description="The id of the episode node that now holds the turn.",
        examples=["5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f"],
    )
    indexed: bool = pydantic.Field(
        description=(
            "Whether the episode also has a vector now. False without an embedder, when the episode is outside the "
            "embedder's capacity, or when embedding failed; it is still kept and found by recency, presence, and links."
        ),
        examples=[True],
    )


class Episode(pydantic.BaseModel):
    """A remembered episode as read back."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "node_id": "5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f",
                    "stimulus": "bot.ability.conversation.messages",
                    "text": "Ada: can you call me back at noon?",
                    "images": [],
                    "participants": [{"ref": "+15555550100", "role": "source"}],
                    "focus": "conversation",
                    "focus_candidates": ["conversation"],
                    "acts": [{"event": "bot.ability.speaking.say", "data": {"text": "Sure, at noon."}}],
                    "outcome": "completed",
                    "turn_id": "turn-7c1e",
                    "created_at": "2026-10-14T16:30:00.000000Z",
                    "recalled_at": None,
                    "recall_count": 0,
                    "reviewed": False,
                    "indexed": True,
                }
            ]
        },
    )

    node_id: str = pydantic.Field(description="The episode node's id.", examples=["5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f"])
    stimulus: str | None = pydantic.Field(
        description="Low-cardinality label of the stimulus.", examples=["bot.ability.conversation.messages"]
    )
    text: str = pydantic.Field(
        description="The episode's text parts, joined by newlines.", examples=["Ada: can you call me back at noon?"]
    )
    images: tuple[media.Image | media.ReleasedImage, ...] = pydantic.Field(
        default=(),
        description="The episode's images in order: bytes only when the read asked for media, else placeholders.",
    )
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(), description="Who took part in the stimulus, as stamped when it was remembered."
    )
    focus: str | None = pydantic.Field(default=None, description="What cognition attended to on that turn.")
    focus_candidates: tuple[str, ...] = pydantic.Field(default=(), description="What cognition could have attended to.")
    acts: tuple[Act, ...] = pydantic.Field(default=(), description="What the bot chose to do on that turn.")
    outcome: Outcome = pydantic.Field(description="How the turn ended.", examples=["completed"])
    turn_id: str = pydantic.Field(description="The cognition operation id of the turn.", examples=["turn-7c1e"])
    continued: tuple[str, ...] = pydantic.Field(
        default=(),
        description=(
            "Turn ids of the later turns that went on with this one and extended this episode in place, in the "
            "order they were kept (see EpisodeData.continues); empty when the episode is one turn."
        ),
        examples=[["turn-7c1e:autonomy"]],
    )
    created_at: datetime.datetime = pydantic.Field(description="When the episode was remembered (UTC).")
    recalled_at: datetime.datetime | None = pydantic.Field(
        default=None, description="When the bot last deliberately recalled it (UTC), if ever."
    )
    recall_count: int = pydantic.Field(default=0, ge=0, description="How often the bot deliberately recalled it.")
    reviewed: bool = pydantic.Field(default=False, description="Whether consolidation has reviewed it.")
    indexed: bool = pydantic.Field(default=False, description="Whether it has a vector in the current space.")


class Node(pydantic.BaseModel):
    """One memory node as read back: an episode, a person, or a concept."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "node_id": "9b8a7c6d5e4f3a2b1c0d9e8f7a6b5c4d",
                    "kind": "concept",
                    "text": "Ada prefers calls at noon.",
                    "created_at": "2026-10-15T03:00:00.000000Z",
                    "episode": None,
                    "key": None,
                }
            ]
        },
    )

    node_id: str = pydantic.Field(description="The node's id.", examples=["9b8a7c6d5e4f3a2b1c0d9e8f7a6b5c4d"])
    kind: NodeKind = pydantic.Field(description="What the node is.", examples=["concept"])
    text: str = pydantic.Field(description="The node's text.", examples=["Ada prefers calls at noon."])
    created_at: datetime.datetime = pydantic.Field(description="When the node was created (UTC).")
    episode: Episode | None = pydantic.Field(default=None, description="The episode, for episode nodes.")
    key: str | None = pydantic.Field(
        default=None,
        description="The person id, for person nodes; the turn id the episode began with, for episode nodes.",
        examples=["self"],
    )


SELF: typing.Final = "self"
"""The reserved person id of the bot itself: memory creates it when it opens, and every ref stamped own is its."""

MAX_PERSON_REFS: typing.Final = 16
"""Most refs one person shows."""

MAX_PERSONS: typing.Final = 256
"""Most persons one statement's reply lists."""

PersonId = typing.Annotated[
    str,
    pydantic.StringConstraints(pattern=r"^(self|[0-9a-f]{32})$"),
    pydantic.Field(
        description="A person memory knows: 'self' (the bot itself), or the id memory minted for someone the bot perceived.",
        examples=["self", "3f2a9c0d4b5e6f708192a3b4c5d6e7f8"],
    ),
]

Name = typing.Annotated[
    str,
    pydantic.StringConstraints(min_length=1, max_length=64),
    pydantic.Field(
        description=(
            "What a person is called, exactly as it was given: the name itself, not the sentence it arrived in. "
            "No example on purpose: a name is a complete answer, and a name nobody gave names nobody."
        ),
    ),
]

Ref = typing.Annotated[
    str,
    pydantic.StringConstraints(min_length=1, max_length=200),
    pydantic.Field(
        description=(
            "A participant ref as producers stamp it: 'voice:<id>' or 'face:<id>' for a voice or face, or another "
            "opaque identity such as a phone number. Never a print."
        ),
        examples=["voice:3f2a9c", "face:7b1e04"],
    ),
]


class Person(pydantic.BaseModel):
    """Someone memory knows: their id, the name the bot learned they go by, and their refs."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Someone memory knows. name is what you learned they are called. refs are the voices, faces and "
                "other identities that are theirs, each belonging to exactly one person."
            ),
            "examples": [{"id": "3f2a9c0d4b5e6f708192a3b4c5d6e7f8", "refs": ["voice:3f2a9c"]}],
        },
    )

    id: PersonId
    name: Name | None = pydantic.Field(
        default=None, description="What they are called; None while nobody has named them."
    )
    refs: tuple[Ref, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PERSON_REFS,
        description=f"Their refs, newest link first, at most {MAX_PERSON_REFS}.",
        examples=[["voice:3f2a9c", "face:7b1e04"]],
    )


Who = typing.Annotated[
    str,
    pydantic.StringConstraints(min_length=1, max_length=200),
    pydantic.Field(
        description=(
            "Someone memory knows, written the way the turn or memory showed them: a person id ('self' or the 32 "
            "hex characters memory minted), or a ref such as a voice or face ref from people. Never one made up."
        ),
    ),
]
"""A person id or a participant ref: a value that is a ``PersonId`` names a person, any other a ref."""


class SameAsFact(pydantic.BaseModel):
    """``about`` and ``other`` are one and the same person."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "The same person as other: a voice, face or person nobody has named yet becomes theirs. Two people "
                "you have both named stay two until you say a ref is not the same as one of them."
            ),
        },
    )

    kind: typing.Literal["same_as"] = pydantic.Field(default="same_as", description="This fact: the same person.")
    # No example: the other is someone the turn or memory showed, and an example is nobody's.
    other: Who = pydantic.Field(description="Who about is the same person as: a ref or a person id, as shown.")


class NameFact(pydantic.BaseModel):
    """``about`` is called ``name``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "What the person about is called; it replaces any name you gave them before. To say a voice is "
                "someone you already know, say it is the same as them instead."
            ),
        },
    )

    kind: typing.Literal["name"] = pydantic.Field(default="name", description="This fact: what they are called.")
    name: Name = pydantic.Field(
        description="Their name, exactly as it was given to you: the name itself and nothing around it."
    )


class NotSameAsFact(pydantic.BaseModel):
    """``about`` and ``other`` are not one person."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Not the same person as other: a ref you took for theirs becomes nobody's you know again, and what "
                "you remember from before stays with whoever you thought it was then."
            ),
        },
    )

    kind: typing.Literal["not_same_as"] = pydantic.Field(
        default="not_same_as", description="This fact: not the same person."
    )
    other: Who = pydantic.Field(description="Who about is not: a ref or a person id, as shown.")


Fact = typing.Annotated[SameAsFact | NameFact | NotSameAsFact, pydantic.Field(discriminator="kind")]
"""One fact the bot learned about a person, discriminated by ``kind``."""

_NOTE = pydantic.Field(
    default=None,
    min_length=1,
    max_length=200,
    description="Optional short note on how you know; it is remembered with what you did.",
    examples=["They said so."],
)


class KnowData(pydantic.BaseModel):
    """What the bot learned about a person, to remember."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Remember a fact about someone you perceive or know: that they are the same person as someone, what "
                "they are called, or that they are not the same person as someone. Each voice, face or other ref is "
                "exactly one person's, and your own voice is always yours: a fact that would make a ref two people's, "
                "or change you yourself, is refused. This only remembers; it says nothing to anyone."
            ),
        },
    )

    # No example, deliberately: about is someone the turn or memory showed, and an example is nobody's.
    about: Who = pydantic.Field(description="Who the fact is about: a ref or a person id, as the turn or memory shows.")
    fact: Fact = pydantic.Field(description="What you learned: same_as, name or not_same_as.")
    note: str | None = _NOTE


class ForgetData(pydantic.BaseModel):
    """A person the bot is to forget."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Forget a person: who they are, which voices and faces were theirs, and the voice and face prints "
                "themselves are erased, so you will not recognize them again. What you remember happening stays, "
                "without them in it. You yourself cannot be forgotten."
            ),
        },
    )

    person: PersonId = pydantic.Field(description="The id of the person to forget, as memory showed them.")
    note: str | None = _NOTE


class KnownData(pydantic.BaseModel):
    """What memory now holds after the bot knew or forgot something about a person."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Observation: memory remembers what you said about a person. person is who about is now (for "
                "forget, who they were)."
            ),
            "examples": [
                {"op": "know", "person": {"id": "3f2a9c0d4b5e6f708192a3b4c5d6e7f8", "refs": ["voice:3f2a9c"]}}
            ],
        },
    )

    op: typing.Literal["know", "forget"] = pydantic.Field(
        description="What you did: know or forget.", examples=["know"]
    )
    person: Person = pydantic.Field(description="know: who about is now; forget: the person as they were.")


class PersonKnowData(pydantic.BaseModel):
    """Remember one fact about a person, in one transaction."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"op": "know", "about": "voice:3f2a9c", "fact": {"kind": "name", "name": "Sam"}}]
        },
    )

    op: typing.Literal["know"] = "know"
    about: Who
    fact: Fact


class PersonForgetData(pydantic.BaseModel):
    """Forget a person, never the bot itself: the node, its ref links and its involves links; episodes stay."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"op": "forget", "person": "3f2a9c0d4b5e6f708192a3b4c5d6e7f8"}]},
    )

    op: typing.Literal["forget"] = "forget"
    person: PersonId


class PersonListData(pydantic.BaseModel):
    """List the bot itself and every person the bot has named."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{"op": "list"}]}
    )

    op: typing.Literal["list"] = "list"


PersonData = typing.Annotated[
    PersonKnowData | PersonForgetData | PersonListData,
    pydantic.Field(discriminator="op"),
]
"""One person statement, discriminated by ``op``; Memory answers each in one transaction."""

PERSON_DATA: typing.Final = (
    PersonKnowData,
    PersonForgetData,
    PersonListData,
)
"""Every ``PersonData`` variant, for typed admission."""

PersonOp: typing.TypeAlias = typing.Literal["know", "forget", "list"]


class PersonDoneData(pydantic.BaseModel):
    """What a person statement did."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"op": "forget", "person": {"id": "3f2a9c0d4b5e6f708192a3b4c5d6e7f8"}, "refs": ["voice:3f2a9c"]}
            ]
        },
    )

    op: PersonOp = pydantic.Field(description="The statement answered.", examples=["know"])
    person: Person | None = pydantic.Field(
        default=None,
        description="know: who about is now; forget: the person as it was. None for list.",
    )
    refs: tuple[Ref, ...] = pydantic.Field(
        default=(),
        max_length=256,
        description=(
            "know: every ref the person about is now has, so their prints can be kept once they are named; "
            "forget: every ref the forgotten person had, so their prints can be erased."
        ),
    )
    moved: tuple[Ref, ...] = pydantic.Field(
        default=(),
        max_length=256,
        description="know (same_as): the refs that became the person's, so a link a registry refuses can be undone.",
    )
    released: Ref | None = pydantic.Field(
        default=None,
        description="know (not_same_as): the ref that is nobody's the bot knows now, so its print need not be kept.",
    )
    persons: tuple[Person, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PERSONS,
        description="list: the bot itself and every named person.",
    )


MAX_PRESENT_PERSONS: typing.Final = 17
"""Most persons one turn shows: everyone its participants belong to (at most 16) and the bot itself."""


class Resemblance(pydantic.BaseModel):
    """A person the bot named whose voice or face a new one is like, and how closely."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Someone you know whose voice (or face) this new one is like: the person and how similar the two "
                "are, -1..1. Nothing is linked: whether it is them is yours to infer, ask, or remember."
            ),
            "examples": [{"person": "3f2a9c0d4b5e6f708192a3b4c5d6e7f8", "similarity": 0.42}],
        },
    )

    person: PersonId
    similarity: float = pydantic.Field(
        ge=-1.0, le=1.0, description="How similar the two are, -1..1; higher is more alike.", examples=[0.42]
    )


class Present(pydantic.BaseModel):
    """One participant of the turn, resolved to the person memory knows it as."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Someone taking part in this turn: the ref the producer stamped, the part it plays (source: "
                "speaking, or seen in view; target: addressed), whose it is, and whether that person is known."
            ),
            "examples": [
                {"ref": "voice:3f2a9c", "part": "source", "person": "3f2a9c0d4b5e6f708192a3b4c5d6e7f8", "known": False}
            ],
        },
    )

    ref: Ref
    part: mosfet.ParticipantRole = pydantic.Field(
        description="source: speaking now or seen in view; target: who the turn was addressed to.",
        examples=["source"],
    )
    person: PersonId
    known: bool = pydantic.Field(
        description=(
            "False when nobody has named this person: memory has perceived them but does not know who they are. "
            "True for the bot itself and for anyone named."
        ),
        examples=[False],
    )
    resembles: tuple[Resemblance, ...] = pydantic.Field(
        default=(),
        # Left out when empty, so a participant that resembles nobody reads as it always has.
        exclude_if=operator.not_,
        max_length=mosfet.events.MAX_RESEMBLANCES,
        description=(
            "Only when known is false: people you know whose voice (or face) this one is like, best first. "
            "Perception, never a link."
        ),
    )


class SearchData(pydantic.BaseModel):
    """Find candidate memories for a cue through every mechanical lane at once."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "cue": [{"kind": "text", "text": "Ada asks about the noon call"}],
                    "participants": [{"ref": "+15555550100", "role": "source"}],
                    "nearest": 12,
                    "recent": 6,
                    "present": 6,
                    "linked": 6,
                    "kinds": ["episode", "concept"],
                }
            ]
        },
    )

    cue: tuple[text.ContentPart, ...] = pydantic.Field(
        default=(),
        max_length=MAX_EPISODE_PARTS,
        description="What to search for: the rendered stimulus, or plain words. Empty skips the nearest lane.",
    )
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PARTICIPANTS,
        description=(
            "Who is present now; episodes involving them surface in the present lane, and memory says who each one "
            "is (a ref seen for the first time gets a provisional person)."
        ),
    )
    nearest: int = pydantic.Field(default=12, ge=0, le=64, description="Most candidates from the nearest lane.")
    recent: int = pydantic.Field(default=6, ge=0, le=64, description="Most candidates from the recent lane.")
    present: int = pydantic.Field(default=6, ge=0, le=64, description="Most candidates from the present lane.")
    linked: int = pydantic.Field(default=6, ge=0, le=64, description="Most one-link neighbors of other candidates.")
    kinds: tuple[NodeKind, ...] = pydantic.Field(
        default=(NodeKind.EPISODE, NodeKind.CONCEPT),
        min_length=1,
        description="Which node kinds may be candidates.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_cue(self) -> typing.Self:
        _check_parts(self.cue)
        return self


class Candidate(pydantic.BaseModel):
    """One node a search surfaced, with what surfaced it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    node: Node = pydantic.Field(description="The surfaced node.")
    lanes: tuple[Lane, ...] = pydantic.Field(min_length=1, description="Every lane that surfaced it.")
    similarity: float | None = pydantic.Field(
        default=None, ge=-1.0, le=1.0, description="Cosine similarity to the cue, when the nearest lane found it."
    )
    hops: int = pydantic.Field(default=0, ge=0, le=1, description="0 when found directly, 1 when only linked.")
    link_weight: float | None = pydantic.Field(
        default=None, ge=0.0, le=1.0, description="Weight of the link it was reached through, when linked."
    )
    present: bool = pydantic.Field(default=False, description="Whether it involves someone present now.")


class FoundData(pydantic.BaseModel):
    """What a search surfaced."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[Candidate, ...] = pydantic.Field(
        default=(), max_length=256, description="Every candidate, each once."
    )
    nearest_skipped: bool = pydantic.Field(
        default=False,
        description="True when the nearest lane did not run: no embedder, no cue, or the cue could not be embedded.",
    )
    people: tuple[Present, ...] = pydantic.Field(
        default=(), max_length=MAX_PARTICIPANTS, description="Each participant of the cue, resolved, in stamp order."
    )
    persons: tuple[Person, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PRESENT_PERSONS,
        description="Every person named in people, in order of appearance, and the bot itself.",
    )
    you: Person = pydantic.Field(default=Person(id=SELF), description="The bot itself, with its own refs.")


class FetchData(pydantic.BaseModel):
    """Read nodes in full by id, optionally with image bytes, and optionally mark them recalled."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"node_ids": ["5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f"], "media": True, "reinforce": True}]
        },
    )

    node_ids: tuple[str, ...] = pydantic.Field(min_length=1, max_length=8, description="The nodes to read.")
    media: bool = pydantic.Field(default=False, description="Return image bytes instead of placeholders.")
    reinforce: bool = pydantic.Field(
        default=False, description="Stamp recalled_at and count one recall on each node read, in the same transaction."
    )
    neighbors: int = pydantic.Field(default=6, ge=0, le=64, description="Most one-link neighbors to return.")


class FetchedData(pydantic.BaseModel):
    """The nodes read, in the order asked (missing ids are left out), and their neighbors."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[Node, ...] = pydantic.Field(default=(), description="The nodes that still exist.")
    neighbors: tuple[Candidate, ...] = pydantic.Field(default=(), description="Their one-link neighbors.")


class RecentData(pydantic.BaseModel):
    """Read the latest episodes, oldest first."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{"kinds": ["episode"], "limit": 20}]}
    )

    kinds: tuple[NodeKind, ...] = pydantic.Field(
        default=(NodeKind.EPISODE,),
        min_length=1,
        description="Which node kinds to read; only episodes are read back in order.",
    )
    limit: int = pydantic.Field(ge=1, le=256, description="Most episodes to return.", examples=[20])

    @pydantic.model_validator(mode="after")
    def _validate_kinds(self) -> typing.Self:
        if set(self.kinds) != {NodeKind.EPISODE}:
            raise ValueError("only episodes are read back in order")
        return self


class EpisodesData(pydantic.BaseModel):
    """Episodes, oldest first."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    episodes: tuple[Episode, ...] = pydantic.Field(default=(), description="The episodes, oldest first.")


class BackfillData(pydantic.BaseModel):
    """Index one batch of nodes that have no vector in the current space."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{"limit": 32, "pass_id": "a1b2c3"}]}
    )

    limit: int = pydantic.Field(ge=1, le=64, description="Most nodes to consider in this batch.", examples=[32])
    pass_id: str = pydantic.Field(
        min_length=1,
        max_length=64,
        description="Opaque id of one backfill pass; a node attempted in this pass is not selected again in it.",
        examples=["a1b2c3"],
    )


class BackfilledData(pydantic.BaseModel):
    """What one backfill batch did."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    indexed: int = pydantic.Field(ge=0, description="Nodes that gained a vector.")
    unindexable: int = pydantic.Field(
        ge=0, description="Nodes recorded as never indexable in the current space (too long, unsupported, invalid)."
    )
    remaining: bool = pydantic.Field(description="Whether an eligible node is left for this pass.")


class LeaseData(pydantic.BaseModel):
    """Lease the next unit for consolidation: the oldest unreviewed episodes and their nearest nodes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{"episodes": 8, "neighbors": 4}]}
    )

    episodes: int = pydantic.Field(default=8, ge=1, le=32, description="Most unreviewed episodes in the unit.")
    neighbors: int = pydantic.Field(default=4, ge=0, le=16, description="Most nearest nodes per episode.")


UnitRole: typing.TypeAlias = typing.Literal["episode", "neighbor", "person"]


class UnitNode(pydantic.BaseModel):
    """One node of a leased unit, named by its unit alias."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "ref": "n1",
                    "kind": "episode",
                    "text": "Ada: can you call me back at noon?",
                    "created_at": "2026-10-14T16:30:00.000000Z",
                    "participants": ["n3"],
                    "role": "episode",
                }
            ]
        },
    )

    ref: str = pydantic.Field(
        description="Alias of this node inside the unit ('n1', 'n2', ...). Use it to link, merge, or forget.",
        examples=["n1"],
    )
    kind: NodeKind = pydantic.Field(description="What the node is.", examples=["episode"])
    text: str = pydantic.Field(description="The node's text.", examples=["Ada: can you call me back at noon?"])
    created_at: datetime.datetime = pydantic.Field(description="When the node was created (UTC).")
    participants: tuple[str, ...] = pydantic.Field(
        default=(), description="Aliases of the person nodes this node involves.", examples=[["n3"]]
    )
    role: UnitRole = pydantic.Field(
        description=(
            "'episode': an unreviewed episode this unit is about; 'neighbor': an existing node near one of them; "
            "'person': someone they involve, by name when they have one (persons can be linked but never "
            "forgotten here)."
        ),
        examples=["episode"],
    )


class UnitData(pydantic.BaseModel):
    """A leased consolidation unit: some unreviewed episodes, the nodes nearest them, and who they involve."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "unit": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
                    "nodes": [
                        {
                            "ref": "n1",
                            "kind": "episode",
                            "text": "Ada: can you call me back at noon?",
                            "created_at": "2026-10-14T16:30:00.000000Z",
                            "participants": ["n2"],
                            "role": "episode",
                        },
                        {
                            "ref": "n2",
                            "kind": "person",
                            "text": "Ada",
                            "created_at": "2026-10-14T16:30:00.000000Z",
                            "participants": [],
                            "role": "person",
                        },
                    ],
                    "edges": [["n1", "n2", "involves_source"]],
                }
            ]
        },
    )
    # Ranked: the unit's own episodes come first, then neighbors, then persons. A node too large
    # for the prompt budget is left out of the rendering, never a reason to fail the model call.
    __model_facing_budgeted_fields__: typing.ClassVar[dict[str, typing.Literal["head", "tail"]]] = {"nodes": "head"}

    unit: str = pydantic.Field(
        min_length=1,
        max_length=64,
        description="The lease id; a decision about this unit names it.",
        examples=["0f1e2d3c4b5a69788796a5b4c3d2e1f0"],
    )
    nodes: tuple[UnitNode, ...] = pydantic.Field(description="The unit's nodes, episodes first.")
    edges: tuple[tuple[str, str, str], ...] = pydantic.Field(
        default=(), description="Existing links among the unit's nodes as (source ref, target ref, relation)."
    )


class LeasedData(pydantic.BaseModel):
    """The leased unit, or None when nothing unreviewed is left."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    unit: UnitData | None = pydantic.Field(default=None, description="The unit; None when nothing is left to review.")


def _check_ref(value: str) -> str:
    if not re.fullmatch(r"n[1-9][0-9]{0,3}", value):
        raise ValueError("a unit ref looks like 'n1'")
    return value


UnitRef = typing.Annotated[
    str,
    pydantic.AfterValidator(_check_ref),
    pydantic.Field(description="A node's alias inside the unit, such as 'n1'.", examples=["n1"]),
]


class LinkData(pydantic.BaseModel):
    """Link two nodes of the unit with a relation in your own words."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"source": "n1", "target": "n4", "relation": "mentions", "weight": 0.8}]},
    )

    source: UnitRef
    target: UnitRef
    relation: str = pydantic.Field(
        min_length=1,
        max_length=64,
        description="What the source is to the target, as one lowercase snake_case word or phrase.",
        examples=["mentions"],
    )
    weight: float = pydantic.Field(default=1.0, ge=0.0, le=1.0, description="How strong the link is.", examples=[0.8])

    @pydantic.field_validator("relation")
    @classmethod
    def _validate_relation(cls, value: str) -> str:
        if not _RELATION.fullmatch(value):
            raise ValueError("relation is lowercase snake_case, starting with a letter")
        return value


class MergeData(pydantic.BaseModel):
    """Merge several nodes of the unit into one new statement."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [{"sources": ["n1", "n2"], "text": "Ada prefers calls at noon.", "forget_sources": False}]
        },
    )

    sources: tuple[UnitRef, ...] = pydantic.Field(
        min_length=2, max_length=16, description="The nodes the statement is merged from."
    )
    text: str = pydantic.Field(
        min_length=1,
        max_length=4000,
        description="The merged statement, in your own words.",
        examples=["Ada prefers calls at noon."],
    )
    forget_sources: bool = pydantic.Field(
        default=False, description="Also forget the source nodes (persons are never forgotten)."
    )


class ConsolidateData(pydantic.BaseModel):
    """Your decision about one leased unit of memory: links, merges, forgets, or nothing at all.

    Every list may be empty; an empty decision leaves the unit as it is. Name nodes only by the unit's
    refs. Persons can be linked or merged from but never forgotten.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Your decision about one leased unit of memory: links between its nodes, statements merged from "
                "them, and nodes to forget. Every list may be empty; an empty decision leaves the unit as it is."
            ),
            "examples": [
                {
                    "unit": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
                    "links": [{"source": "n1", "target": "n4", "relation": "mentions"}],
                    "merges": [{"sources": ["n1", "n2"], "text": "Ada prefers calls at noon."}],
                    "forgets": ["n5"],
                    "reason": "n5 repeats n1.",
                }
            ],
        },
    )

    unit: str = pydantic.Field(
        min_length=1,
        max_length=64,
        description="The id of the leased unit this decision is about.",
        examples=["0f1e2d3c4b5a69788796a5b4c3d2e1f0"],
    )
    links: tuple[LinkData, ...] = pydantic.Field(default=(), max_length=64, description="Links to add.")
    merges: tuple[MergeData, ...] = pydantic.Field(default=(), max_length=16, description="Statements to merge.")
    forgets: tuple[UnitRef, ...] = pydantic.Field(
        default=(), max_length=32, description="Episode or neighbor nodes to forget for good."
    )
    reason: str | None = pydantic.Field(default=None, max_length=4000, description="Why, in your own words.")


class ConsolidatedData(pydantic.BaseModel):
    """What one consolidation decision changed, committed in one transaction."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    unit: str = pydantic.Field(description="The unit the decision was about.")
    linked: int = pydantic.Field(ge=0, description="Links written.")
    merged: int = pydantic.Field(ge=0, description="Concept nodes created.")
    forgotten: int = pydantic.Field(ge=0, description="Nodes deleted.")


class ReleaseData(pydantic.BaseModel):
    """Give a leased unit back, marking its episodes reviewed or leaving them for a later unit."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"unit": "0f1e2d3c4b5a69788796a5b4c3d2e1f0", "reviewed": True}]},
    )

    unit: str = pydantic.Field(min_length=1, max_length=64, description="The leased unit to release.")
    reviewed: bool = pydantic.Field(description="Mark the unit's episodes reviewed so they are not leased again.")


class ReleasedData(pydantic.BaseModel):
    """A unit was released."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    unit: str = pydantic.Field(description="The released unit.")


class RestData(pydantic.BaseModel):
    """The bot came to rest: its memory may be consolidated until it wakes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{}]}
    )


class WakeData(pydantic.BaseModel):
    """The bot woke: consolidation stops, and a unit not yet committed is given back unreviewed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{}]}
    )


UnitOutcome: typing.TypeAlias = typing.Literal["committed", "reviewed", "interrupted", "failed", "exhausted"]


class UnitOutcomeData(pydantic.BaseModel):
    """What became of one consolidation unit during a rest."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"unit": "0f1e2d3c4b5a69788796a5b4c3d2e1f0", "outcome": "committed"}]},
    )

    unit: str | None = pydantic.Field(
        default=None,
        max_length=64,
        description="The leased unit; None when no unit was leased (nothing left, or the lease failed).",
        examples=["0f1e2d3c4b5a69788796a5b4c3d2e1f0"],
    )
    outcome: UnitOutcome = pydantic.Field(
        description=(
            "'committed': the decision was applied in one transaction; 'reviewed': the bot left the unit as it is; "
            "'interrupted': the bot woke before the unit was decided; 'failed': no decision could be made or "
            "applied; 'exhausted': nothing unreviewed is left."
        ),
        examples=["committed"],
    )


MAX_ASSOCIATIONS: typing.Final = 64
"""Most associations one turn carries."""


def approximate_tokens(text: str) -> int:
    """Approximate model tokens in ``text``: one per four UTF-8 bytes, rounded up (the default counter)."""

    return -(-len(text.encode("utf-8")) // 4)


class CueData(pydantic.BaseModel):
    """What to associate from: the memory rendering of a stimulus, who is present, and the turns already shown."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "cue": [{"kind": "text", "text": "Ada: can you call me back at noon?"}],
                    "participants": [{"ref": "+15555550100", "role": "source"}],
                    "exclude": ["5e5b95c437364ad5a1763d96a735b673", "5e5b95c437364ad5a1763d96a735b673:autonomy"],
                }
            ]
        },
    )

    cue: tuple[text.ContentPart, ...] = pydantic.Field(
        default=(),
        max_length=MAX_EPISODE_PARTS,
        description="The memory rendering of the stimulus this turn is about. Empty skips the nearest lane.",
        examples=[[{"kind": "text", "text": "Ada: can you call me back at noon?"}]],
    )
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PARTICIPANTS,
        description="Who is present now, as the stimulus producer stamped them.",
        examples=[[{"ref": "+15555550100", "role": "source"}]],
    )
    exclude: tuple[typing.Annotated[str, pydantic.StringConstraints(min_length=1, max_length=200)], ...] = (
        pydantic.Field(
            default=(),
            max_length=MAX_EXCLUDED_TURNS,
            description=(
                "Turn ids (an episode's turn_id) of the turn being answered: its stimulus's id, or each percept's of "
                "its batch, and the ids of the causal parents each carries. Their episodes are what the turn already "
                "shows, not earlier memories, so they never surface and are not counted as omitted. Speech admitted "
                "through a conversation is first a turn of its own, kept under the speech's id, and then the exchange "
                "that carries that speech as a parent."
            ),
            examples=[["5e5b95c437364ad5a1763d96a735b673", "5e5b95c437364ad5a1763d96a735b673:autonomy"]],
        )
    )

    @pydantic.model_validator(mode="after")
    def _validate_cue(self) -> typing.Self:
        _check_parts(self.cue)
        return self


class Association(pydantic.BaseModel):
    """One memory surfaced for the current turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "ref": "5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f",
                    "kind": "episode",
                    "text": "Ada: can you call me back at noon?",
                    "images": [],
                    "at": "2026-10-14T16:30:00.000000Z",
                    "lanes": ["similar", "present"],
                    "involves": ["+15555550100"],
                }
            ]
        },
    )

    ref: str = pydantic.Field(
        min_length=1,
        description="The remembered node's id; name it to think about this memory in full.",
        examples=["5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f"],
    )
    kind: NodeKind = pydantic.Field(description="What the memory is.", examples=["episode"])
    text: str = pydantic.Field(
        description="What the memory says, as it was kept.", examples=["Ada: can you call me back at noon?"]
    )
    images: tuple[media.ReleasedImage, ...] = pydantic.Field(
        default=(),
        description="Images the memory holds, as placeholders without their bytes.",
    )
    at: datetime.datetime = pydantic.Field(description="When the memory was made (UTC).")
    lanes: tuple[Lane, ...] = pydantic.Field(min_length=1, description="What surfaced it.")
    involves: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Participant refs of who the memory involves.",
        examples=[["+15555550100"]],
    )


class AssociationsData(pydantic.BaseModel):
    """What memory surfaced for one turn, best first, and who is here: who is speaking, who is present, who you are.

    ``people`` and ``persons`` render before the memories and are left out only after every memory
    is; ``you`` is never left out. A person who is ``known=False`` is someone the bot has perceived
    but nobody has named: what to do about that is the bot's to decide.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")
    # Who is here comes first; the memories are ranked best first. A prompt that cannot hold them
    # all keeps who is here and the best memories.
    __model_facing_budgeted_fields__: typing.ClassVar[dict[str, typing.Literal["head", "tail"]]] = {
        "people": "head",
        "persons": "head",
        "items": "head",
    }

    you: Person = pydantic.Field(
        default=Person(id=SELF),
        description="You, the bot, as your memory knows you: your refs are your own voices (and faces).",
    )
    people: tuple[Present, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PARTICIPANTS,
        description="Who takes part in this turn, in the order the producers stamped them, each resolved to a person.",
    )
    persons: tuple[Person, ...] = pydantic.Field(
        default=(),
        max_length=MAX_PRESENT_PERSONS,
        description="Every person named in people, and you: their names and refs.",
    )
    items: tuple[Association, ...] = pydantic.Field(
        default=(), max_length=MAX_ASSOCIATIONS, description="The memories surfaced, best first."
    )
    omitted: int = pydantic.Field(
        default=0, ge=0, description="Memories surfaced but left out to fit the association token budget."
    )


_MAX_RECALLED: typing.Final = 8
"""Most memories one deliberate recall brings to mind in full (one fetch)."""


class RecallData(pydantic.BaseModel):
    """Think about a memory deliberately: about something in plain words, or one surfaced memory in full.

    Name exactly one of ``about`` and ``node``. What comes to mind arrives later as its own
    observation (``bot.ability.memory.recollected``): the memories in full, images included, and
    what they are linked to. Recalling a memory makes it more likely to surface again.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"about": "what Ada said about the noon call", "limit": 3},
                {"node": "5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f"},
            ]
        },
    )

    about: str | None = pydantic.Field(
        default=None,
        min_length=1,
        max_length=500,
        description="What to think about, in plain words. Leave unset when naming a node.",
        examples=["what Ada said about the noon call"],
    )
    node: str | None = pydantic.Field(
        default=None,
        min_length=1,
        max_length=64,
        description=(
            "The ref of one memory surfaced this turn, to look at it in full with its images. "
            "Leave unset when thinking about something."
        ),
        examples=["5d0c3f7e9a1b4c2d8e6f0a1b2c3d4e5f"],
    )
    limit: int = pydantic.Field(
        default=5,
        ge=1,
        le=_MAX_RECALLED,
        description="Most memories to bring to mind when thinking about something.",
        examples=[3],
    )

    @pydantic.model_validator(mode="after")
    def _validate_one_subject(self) -> typing.Self:
        if (self.about is None) == (self.node is None):
            raise ValueError("name exactly one of about or node")
        return self


class RecollectionData(pydantic.BaseModel):
    """What came to mind when the bot deliberately recalled a memory.

    ``nodes`` and ``neighbors`` are recalled context: a prompt that cannot hold them all keeps the
    best and counts the rest. Memory keeps only what was thought about; the memories themselves
    are already kept.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")
    __model_facing_budgeted_fields__: typing.ClassVar[dict[str, typing.Literal["head", "tail"]]] = {
        "nodes": "head",
        "neighbors": "head",
    }
    __memory_fields__: typing.ClassVar[frozenset[str]] = frozenset({"about"})

    about: str | None = pydantic.Field(
        default=None, description="What the bot thought about, when it recalled by words.", examples=["the noon call"]
    )
    nodes: tuple[Node, ...] = pydantic.Field(
        default=(),
        max_length=_MAX_RECALLED,
        description=(
            "The memories recalled in full, best first. Image bytes ride the newest memories, at most "
            f"{media.MAX_TURN_IMAGES} in all; older images are placeholders."
        ),
    )
    neighbors: tuple[Association, ...] = pydantic.Field(
        default=(), max_length=MAX_ASSOCIATIONS, description="Memories linked to the recalled ones, by their refs."
    )

    @pydantic.model_validator(mode="after")
    def _validate_images(self) -> typing.Self:
        carried = sum(
            isinstance(image, media.Image)
            for node in self.nodes
            if node.episode is not None
            for image in node.episode.images
        )
        if carried > media.MAX_TURN_IMAGES:
            raise ValueError(f"a recollection carries image bytes for at most {media.MAX_TURN_IMAGES} images")
        return self


class Weights(pydantic.BaseModel):
    """How Associative scores a candidate; a capacity of the composition, never a per-turn choice.

    ``score = similarity * cosine + recency * 0.5 ** (age / recency_half_life) + present * [involves someone
    present] + linked * link_weight``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    similarity: float = pydantic.Field(default=1.0, ge=0.0, description="Weight of cosine similarity to the cue.")
    recency: float = pydantic.Field(default=0.35, ge=0.0, description="Weight of how recent the memory is.")
    recency_half_life: datetime.timedelta = pydantic.Field(
        default=datetime.timedelta(days=2),
        gt=datetime.timedelta(0),
        description="Age at which the recency term halves.",
    )
    present: float = pydantic.Field(default=0.5, ge=0.0, description="Weight of involving someone present now.")
    linked: float = pydantic.Field(default=0.6, ge=0.0, description="Weight of the link a memory was reached through.")


__all__ = [
    "MAX_ASSOCIATIONS",
    "MAX_EPISODE_PARTS",
    "MAX_EPISODE_TEXT_BYTES",
    "MAX_PARTICIPANTS",
    "MAX_PERSONS",
    "MAX_PERSON_REFS",
    "MAX_PRESENT_PERSONS",
    "PERSON_DATA",
    "SELF",
    "Act",
    "Association",
    "AssociationsData",
    "BackfillData",
    "BackfilledData",
    "Candidate",
    "ConsolidateData",
    "ConsolidatedData",
    "CueData",
    "Episode",
    "EpisodeData",
    "EpisodesData",
    "Fact",
    "FailureData",
    "FailureKind",
    "FetchData",
    "FetchedData",
    "ForgetData",
    "FoundData",
    "KnowData",
    "KnownData",
    "Lane",
    "LeaseData",
    "LeasedData",
    "LinkData",
    "MergeData",
    "Name",
    "NameFact",
    "Node",
    "NodeKind",
    "NotSameAsFact",
    "Outcome",
    "Person",
    "PersonData",
    "PersonDoneData",
    "PersonForgetData",
    "PersonId",
    "PersonKnowData",
    "PersonListData",
    "PersonOp",
    "Present",
    "RecallData",
    "RecentData",
    "RecollectionData",
    "Ref",
    "ReleaseData",
    "ReleasedData",
    "RememberedData",
    "Resemblance",
    "RestData",
    "SameAsFact",
    "SearchData",
    "UnitData",
    "UnitNode",
    "UnitOutcome",
    "UnitOutcomeData",
    "UnitRef",
    "UnitRole",
    "WakeData",
    "Weights",
    "Who",
    "approximate_tokens",
]
