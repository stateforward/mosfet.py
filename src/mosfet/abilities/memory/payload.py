"""Typed payloads of the episodic ``Memory``: what is remembered, searched, fetched, and consolidated.

An episode is one cognition turn as its producer stamped it: the memory rendering of the stimulus
(text and image parts), who took part, what the bot chose to do, and how the turn ended. Memory
keeps every episode as a node of a graph (episodes, the persons they involve, and the concepts
consolidation merges them into), with a canonical vector per node in the embedder's space.

Read models (``Episode``, ``Node``, ``Candidate``) carry images as ``media.ReleasedImage``
placeholders (reason ``"memory"``) unless a read asked for media. Consolidation sees a leased
``UnitData`` whose nodes are named by unit-scoped aliases, never by node ids, and answers with one
``ConsolidateData`` decision: links, merges, and forgets, or nothing at all.
"""

from __future__ import annotations

import datetime
import enum
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
    """Someone episodes involve, keyed by the participant ref producers stamp."""
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
            "The memory rendering of the stimulus: text parts and at most 8 image parts, in order. Text totals "
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
        default=None, description="The participant ref, for person nodes.", examples=["+15555550100"]
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
        description="Who is present now; episodes involving them surface in the present lane.",
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
            "'person': someone they involve (persons can be linked but never forgotten)."
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
                            "text": "+15555550100",
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


MAX_ASSOCIATIONS: typing.Final = 64
"""Most associations one turn carries."""


def approximate_tokens(text: str) -> int:
    """Approximate model tokens in ``text``: one per four UTF-8 bytes, rounded up (the default counter)."""

    return -(-len(text.encode("utf-8")) // 4)


class CueData(pydantic.BaseModel):
    """What to associate from: the memory rendering of a stimulus and who is present."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "cue": [{"kind": "text", "text": "Ada: can you call me back at noon?"}],
                    "participants": [{"ref": "+15555550100", "role": "source"}],
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
    """What memory surfaced for one turn, best first."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")
    # Ranked best first: a prompt that cannot hold them all keeps the best.
    __model_facing_budgeted_fields__: typing.ClassVar[dict[str, typing.Literal["head", "tail"]]] = {"items": "head"}

    items: tuple[Association, ...] = pydantic.Field(
        default=(), max_length=MAX_ASSOCIATIONS, description="The memories surfaced, best first."
    )
    omitted: int = pydantic.Field(
        default=0, ge=0, description="Memories surfaced but left out to fit the association token budget."
    )


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
    "FailureData",
    "FailureKind",
    "FetchData",
    "FetchedData",
    "FoundData",
    "Lane",
    "LeaseData",
    "LeasedData",
    "LinkData",
    "MergeData",
    "Node",
    "NodeKind",
    "Outcome",
    "RecentData",
    "ReleaseData",
    "ReleasedData",
    "RememberedData",
    "SearchData",
    "UnitData",
    "UnitNode",
    "UnitRef",
    "UnitRole",
    "Weights",
    "approximate_tokens",
]
