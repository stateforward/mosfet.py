"""Modality-neutral conversation coordination.

Conversation owns the relationship between identity sets.  The input boundary
does not expose a conversation reference: a relationship is inferred from the
source and target sets, and each source identity gets a turn detector in that
relationship.
"""

from __future__ import annotations

from ... import ability
from ... import encoding as encoding_module
from ... import language
from ... import memory as memory_ability
from ...identity import value
from . import memory as conversation_memory
from . import turn_detector

import asyncio
import collections.abc
import dataclasses
import datetime
import typing as typ
import uuid

import hsm
import bot
import pydantic
from pydantic.config import JsonDict, JsonValue
from pydantic.json_schema import SkipJsonSchema

from bot import event
from bot import events
from bot.abilities import cognition
from bot.abilities.listening import interpretation
from bot import telemetry
from bot.telemetry import observer
from bot.telemetry import span

Stage: typ.TypeAlias = typ.Literal["memory", "turn_detector", "voice_routing"]
Content: typ.TypeAlias = object
type MessageContent = str | int | float | bool | None | list[MessageContent] | dict[str, MessageContent]
IdentitySet: typ.TypeAlias = value.IdentitySet
IdentityValue: typ.TypeAlias = value.IdentityValue
TrackRef: typ.TypeAlias = str
TurnDetectorFactory: typ.TypeAlias = typ.Callable[[str, TrackRef], turn_detector.TurnDetector]

_TURN_TIMEOUT_SECONDS = 5.0
# Recall, detector ready, detector turn, and remember are sequential inner stages.
# The contribution waiter must outlive one inner stage so a typed stage failure can settle.
_CONTRIBUTION_TIMEOUT_STAGES = 4


def _default_turn_detector(session_ref: str, track_ref: TrackRef) -> turn_detector.TurnDetector:
    return turn_detector.TurnDetector(
        participant_ref=track_ref,
        conversation_ref=session_ref,
    )


MAX_IDENTITY_SET_SIZE = value.MAX_IDENTITY_SET_SIZE
"""Maximum source or target identities accepted by Conversation input."""
MAX_EMBEDDING_DIMENSION = value.MAX_EMBEDDING_DIMENSION
"""Maximum embedding dimension accepted by Conversation input."""


_INPUT_EXAMPLE: JsonDict = {
    "source_ids": ["caller"],
    "target_ids": [],
    "content": "What is the weather like?",
    "content_type": "text/plain",
    "sample_rate_hz": None,
    "channels": None,
}


def _schema(description: str, example: JsonDict) -> JsonDict:
    examples: list[JsonValue] = [example]
    return {"description": description, "examples": examples}


class TurnData(pydantic.BaseModel):
    """One modality-neutral input entering Conversation.

    ``source_ids`` and ``target_ids`` are sets of opaque identity references.
    The relationship they describe is used only to derive a transient reference;
    ``content_type`` describes the payload and never selects an HSM route.

    Audio content is typed ``bytes``. Selection/JSON hops may carry base64 text; validation
    rehydrates audio/* content to bytes so downstream turn decode stays on typed events.
    """

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
        json_schema_extra=_schema(
            "Conversation input identified by source identities and an optional target identity set; an empty target set represents an ambient input with an unknown addressee.",
            _INPUT_EXAMPLE,
        ),
    )

    __producer_stamped_fields__: typ.ClassVar[frozenset[str]] = frozenset({"parent"})

    parent: SkipJsonSchema[events.StimulusData[interpretation.SpeechData] | None] = pydantic.Field(
        default=None,
        description=(
            "The exact Listening speech event and typed payload admitted by SpeechHeard. "
            "Host-created Conversation inputs may omit this parent."
        ),
    )
    source_ids: IdentitySet = pydantic.Field(
        min_length=1,
        description="One or more opaque identities that produced this content.",
        examples=[["caller"], ["left-speaker", "right-speaker"]],
    )
    target_ids: IdentitySet = pydantic.Field(
        description="Zero or more opaque identities addressed by this content; an empty set means the addressee is unknown or the input is ambient.",
        examples=[[], ["bot"]],
    )
    content: object = pydantic.Field(
        description=(
            "Modality-specific content carried without interpretation by Conversation. "
            "text/* uses str; audio/* uses bytes (base64 on JSON wires); other modalities may use structured objects."
        ),
        examples=["What is the weather like?", "AAECAw=="],
    )
    content_type: str = pydantic.Field(
        min_length=1,
        description="Content media type or modality label; it is data, not an event route.",
        examples=["text/plain", "audio/pcm", "audio/raw", "application/x-sign-language"],
    )
    sample_rate_hz: int | None = pydantic.Field(
        default=None,
        ge=1,
        description=(
            "Sample rate of raw PCM audio content in hertz. Required for correct STT packaging when "
            "content_type is audio/pcm, audio/raw, or similar non-container audio. Omit for text and WAV."
        ),
        examples=[16_000, 48_000],
    )
    channels: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Channel count of raw PCM audio content. Omit for text and self-describing containers.",
        examples=[1],
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def rehydrate_audio_content(cls, raw_value: object) -> object:
        """Restore audio/* content from JSON hops: bytes as-is, or base64 text only."""

        if not isinstance(raw_value, dict):
            return raw_value
        data = dict(typ.cast(dict[str, object], raw_value))
        content_type = data.get("content_type")
        content = data.get("content")
        if isinstance(content_type, str) and content_type.lower().startswith("audio/"):
            data["content"] = event.bytes_from_base64(content)
        return data

    @pydantic.field_validator("source_ids", "target_ids", mode="before")
    @classmethod
    def validate_ids(cls, raw_value: object, info: pydantic.ValidationInfo) -> IdentitySet:
        field_name = info.field_name or "identity set"
        identities = value.normalize_identity_set(
            raw_value,
            field_name=field_name,
            allow_empty=field_name == "target_ids",
        )
        if len(identities) > MAX_IDENTITY_SET_SIZE:
            raise ValueError(
                f"{field_name} contains {len(identities)} identities; the maximum is {MAX_IDENTITY_SET_SIZE}."
            )
        for identity in identities:
            if not isinstance(identity, str):
                _ = value.normalize_embedding(identity, field_name=field_name)
        return identities

    @pydantic.model_validator(mode="after")
    def require_audio_bytes(self) -> typ.Self:
        """Audio modalities must carry bytes after typed validation."""

        if self.content_type.lower().startswith("audio/") and not isinstance(self.content, bytes):
            raise ValueError("audio/* conversation content must be bytes after typed validation")
        return self

    @pydantic.field_serializer("source_ids", "target_ids", when_used="json")
    def serialize_ids(self, identities: IdentitySet) -> list[str | list[float]]:
        return value.identities_json(identities)


class MessageProvenance(pydantic.BaseModel):
    """Typed HSM provenance for one committed model-facing message.

    Provenance identifies the event that committed the item.  It intentionally carries
    no payload or media, so model-facing history cannot become a raw-media side channel.
    """

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "event": "bot.ability.conversation.append",
                    "id": "append-1",
                    "source": "speaking",
                    "target": "conversation",
                }
            ]
        },
    )

    event: str = pydantic.Field(min_length=1, description="Canonical event that committed this message.")
    id: str | None = pydantic.Field(default=None, description="Correlation id of the committing event.")
    source: str | None = pydantic.Field(default=None, description="HSM source identity of the committing event.")
    target: str | None = pydantic.Field(default=None, description="HSM target identity of the committing event.")
    session_ref: str | None = pydantic.Field(
        default=None, min_length=1, description="Conversation relationship reference, when known."
    )
    turn_ref: str | None = pydantic.Field(
        default=None, min_length=1, description="Turn detector reference, when known."
    )


class Message(pydantic.BaseModel):
    """One immutable, ordered, model-safe inbound or outbound message."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "sequence": 0,
                    "direction": "inbound",
                    "source_ids": ["caller"],
                    "target_ids": ["bot"],
                    "content": "Hello",
                    "content_type": "text/plain",
                    "provenance": {
                        "event": "bot.ability.conversation.input",
                        "id": "turn-1",
                        "source": "caller",
                        "target": "conversation",
                    },
                }
            ]
        },
    )

    sequence: int = pydantic.Field(ge=0, description="Zero-based position in committed conversation history.")
    direction: typ.Literal["inbound", "outbound"] = pydantic.Field(
        description="Whether the message came from a participant or the bot."
    )
    source_ids: IdentitySet = pydantic.Field(description="Opaque source identities associated with the message.")
    target_ids: IdentitySet = pydantic.Field(description="Opaque target identities associated with the message.")
    content: MessageContent = pydantic.Field(
        default=None,
        description="Decoded text or JSON-safe structured content; raw media bytes are never accepted here.",
    )

    @pydantic.field_validator("content", mode="before")
    @classmethod
    def reject_media(cls, raw_value: object) -> object:
        def contains_media(value: object) -> bool:
            if isinstance(value, (bytes, bytearray, memoryview)):
                return True
            if isinstance(value, dict):
                return any(contains_media(item) for item in value.values())
            if isinstance(value, list | tuple):
                return any(contains_media(item) for item in value)
            return False

        if contains_media(raw_value):
            raise ValueError("message content cannot contain raw media")
        return raw_value

    content_type: str = pydantic.Field(min_length=1, description="Media type of the model-safe message content.")
    provenance: MessageProvenance = pydantic.Field(description="Typed event provenance for this committed message.")

    @pydantic.field_validator("source_ids", "target_ids", mode="before")
    @classmethod
    def normalize_ids(cls, raw_value: object, info: pydantic.ValidationInfo) -> IdentitySet:
        return value.normalize_identity_set(raw_value, field_name=info.field_name or "identity set", allow_empty=True)

    @pydantic.field_serializer("source_ids", "target_ids", when_used="json")
    def serialize_ids(self, identities: IdentitySet) -> list[str | list[float]]:
        return value.identities_json(identities)


class Messages(pydantic.BaseModel):
    """Cumulative immutable conversation history emitted after each commit."""

    __model_facing_excluded_fields__: typ.ClassVar[frozenset[str]] = frozenset({"memories"})

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {
                            "sequence": 0,
                            "direction": "inbound",
                            "source_ids": ["caller"],
                            "target_ids": ["bot"],
                            "content": "Hello",
                            "content_type": "text/plain",
                            "provenance": {
                                "event": "bot.ability.conversation.input",
                                "id": "turn-1",
                            },
                        }
                    ]
                }
            ]
        },
    )

    __producer_stamped_fields__: typ.ClassVar[frozenset[str]] = frozenset({"parent"})

    parent: events.StimulusData[TurnData] | None = pydantic.Field(
        default=None,
        description="The exact typed inbound event that caused this history emission, when one exists.",
    )
    messages: tuple[Message, ...] = pydantic.Field(
        default=(),
        description="All committed inbound and outbound messages in causal order.",
    )
    memories: SkipJsonSchema[tuple[conversation_memory.Memory, ...]] = pydantic.Field(
        default=(),
        exclude=True,
        description="Host-only recalled context for the latest inbound turn; never model-projected.",
    )


def _message_for_turn(
    data: TurnData,
    *,
    content: MessageContent,
    content_type: str,
    sequence: int,
    provenance: MessageProvenance,
) -> Message:
    return Message(
        sequence=sequence,
        direction="inbound",
        source_ids=data.source_ids,
        target_ids=data.target_ids,
        content=content,
        content_type=content_type,
        provenance=provenance,
    )


class FailureData(pydantic.BaseModel):
    """Failure signal produced when a conversation phase cannot complete."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    stage: Stage = pydantic.Field(description="Conversation phase that failed.")
    message: str = pydantic.Field(min_length=1, description="Human-readable failure message.")


class SnapshotRequest(pydantic.BaseModel):
    """Request for the current inferred relationship snapshot."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request_ref: str = pydantic.Field(min_length=1, description="Caller correlation reference.")


class Snapshot(pydantic.BaseModel):
    """Current transient detector registry observation."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    request_ref: str = pydantic.Field(min_length=1)
    detector_refs: tuple[tuple[str, IdentityValue], ...] = pydantic.Field(
        default=(),
        description="Transient (relationship reference, source identity) detector keys.",
    )


class ParticipatedTurn(pydantic.BaseModel):
    """Completed source contribution exposed to host composition."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    input: pydantic.SkipValidation[TurnData]
    stimulus: turn_detector.ParticipationStimulus
    participation: turn_detector.ParticipantContribution
    participations: tuple[turn_detector.ParticipantContribution, ...] = ()
    session_ref: str = pydantic.Field(min_length=1)
    memories: tuple[conversation_memory.Memory, ...] = ()
    messages: tuple[Message, ...] = ()


def participated_turn_from_messages(
    input_data: TurnData,
    messages: Messages,
) -> ParticipatedTurn:
    """Reconstruct host composition data from the latest inbound history item."""

    inbound = next((item for item in reversed(messages.messages) if item.direction == "inbound"), None)
    if inbound is None or inbound.source_ids != input_data.source_ids or inbound.target_ids != input_data.target_ids:
        raise RuntimeError("Conversation history has no matching inbound message for the input identities.")
    content = inbound.content
    content_type = inbound.content_type
    normalized_type = content_type.lower()
    sources = value.sorted_identities(inbound.source_ids)
    structured = typ.cast(dict[str, object] | None, content) if isinstance(content, dict) else None
    readable = content if isinstance(content, str) else None
    if content is None and structured is None:
        readable = ""
        perception_content: Content | None = ""
    else:
        perception_content = content

    def stimulus_for(source_id: IdentityValue) -> turn_detector.ParticipationStimulus:
        if normalized_type.startswith("text/") and isinstance(content, str):
            return turn_detector.TextStimulus(source_participant_ref=source_id, content=content)
        if content is None and readable == "":
            return turn_detector.TextStimulus(source_participant_ref=source_id, content="")
        return turn_detector.ContentStimulus(
            source_participant_ref=source_id,
            content=perception_content,
            content_type=content_type,
        )

    if not sources:
        raise RuntimeError("Conversation history inbound message contains no source identities.")
    stimuli = tuple(stimulus_for(source_id) for source_id in sources)
    session_ref = inbound.provenance.session_ref or "history"
    contributions = tuple(
        turn_detector.ParticipantContribution(
            conversation_ref=session_ref,
            participant_ref=source_id,
            perception=turn_detector.Perception(
                source_participant_ref=source_id,
                modality=(
                    "text"
                    if normalized_type.startswith("text/") or isinstance(content, str) or content is None
                    else "multimodal"
                ),
                readable=readable,
                structured=structured,
                content=None if isinstance(content, str) or content is None else content,
            ),
        )
        for source_id in sources
    )
    return ParticipatedTurn(
        input=input_data,
        stimulus=stimuli[0],
        participation=contributions[0],
        participations=contributions,
        session_ref=session_ref,
        memories=messages.memories,
        messages=messages.messages,
    )


InputEvent = hsm.Event[TurnData](
    name="bot.ability.conversation.input",
    kind=event.EventKind,
    schema=TurnData,
)


class RoutedInputData(pydantic.BaseModel):
    """Internal typed handoff preserving Communication's original input event envelope."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    parent: events.StimulusData[TurnData]


RoutedInputEvent = hsm.Event[RoutedInputData](
    name="bot.ability.conversation.routed_input",
    schema=RoutedInputData,
)


class AppendData(pydantic.BaseModel):
    """Trusted outbound message record submitted by an effector to Conversation."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    message: Message = pydantic.Field(description="Immutable bot message and its typed effector provenance.")


AppendEvent = hsm.Event[AppendData](
    name="bot.ability.conversation.append",
    schema=AppendData,
)
OutputEvent = hsm.Event[Messages](
    name="bot.ability.conversation.output",
    schema=Messages,
)
FailedEvent = hsm.Event[FailureData](
    name="bot.ability.conversation.failed",
    schema=FailureData,
)
SnapshotRequestEvent = hsm.Event[SnapshotRequest](
    name="bot.ability.conversation.snapshot.request",
    schema=SnapshotRequest,
)
SnapshotOutputEvent = hsm.Event[Snapshot](
    name="bot.ability.conversation.snapshot.output",
    schema=Snapshot,
)


class _InputWorkData(pydantic.BaseModel):
    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    input: pydantic.SkipValidation[TurnData]
    input_parent: events.StimulusData[TurnData] | None = None


class _InputCancelledData(pydantic.BaseModel):
    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)


class _TurnOperationProvenance(pydantic.BaseModel):
    """Typed identity for one Conversation child-operation aggregate."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)
    session_ref: str = pydantic.Field(min_length=1)
    source_ids: IdentitySet = pydantic.Field(min_length=1)
    detector_ids: frozenset[TrackRef] = pydantic.Field(default=frozenset())
    turn_refs: frozenset[str] = pydantic.Field(default=frozenset())
    failure_track_ref: TrackRef | None = pydantic.Field(default=None)

    @pydantic.field_validator("source_ids", mode="before")
    @classmethod
    def normalize_identity_sets(cls, raw_value: object, info: pydantic.ValidationInfo) -> IdentitySet:
        return value.normalize_identity_set(
            raw_value,
            field_name=info.field_name or "identity set",
            allow_empty=False,
        )


class _TurnOperationCompletedData(pydantic.BaseModel):
    """Conversation-owned aggregate of correlated child turn terminals."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    input: pydantic.SkipValidation[TurnData]
    input_parent: events.StimulusData[TurnData] | None = None
    operation_id: str = pydantic.Field(min_length=1)
    provenance: _TurnOperationProvenance
    turns: tuple[turn_detector.TurnCompleteData, ...] = pydantic.Field(min_length=1)
    memories: tuple[conversation_memory.Memory, ...] = ()


class _TurnFailedData(pydantic.BaseModel):
    """Failure correlated to the Conversation operation that produced it."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    input: pydantic.SkipValidation[TurnData]
    input_parent: events.StimulusData[TurnData] | None = None
    operation_id: str = pydantic.Field(min_length=1)
    provenance: _TurnOperationProvenance
    failure: FailureData
    memories: tuple[conversation_memory.Memory, ...] = ()


_InputWorkEvent = hsm.Event[_InputWorkData](
    name="bot.ability.conversation.input.work",
    schema=_InputWorkData,
)
_InputCancelledEvent = hsm.Event[_InputCancelledData](
    name="bot.ability.conversation.input.cancelled",
    schema=_InputCancelledData,
)
_TurnOperationCompletedEvent = hsm.Event[_TurnOperationCompletedData](
    name="bot.ability.conversation.turn.completed",
    kind=hsm.CompletionEventKind,
    schema=_TurnOperationCompletedData,
)
_TurnFailedEvent = hsm.Event[_TurnFailedData](
    name="bot.ability.conversation.turn.failed",
    kind=hsm.ErrorEventKind,
    schema=_TurnFailedData,
)


@dataclasses.dataclass
class _ParticipantProfile:
    track_ref: TrackRef
    source_id: IdentityValue
    centroid: value.Embedding | None = None
    vector_sum: value.Embedding | None = None
    count: int = 0
    detector: turn_detector.TurnDetector | None = None


@dataclasses.dataclass
class _RelationshipProfile:
    session_ref: str
    source_ids: tuple[IdentityValue, ...]
    target_ids: tuple[IdentityValue, ...]
    identity_groups: tuple[tuple[IdentityValue, ...], tuple[IdentityValue, ...]]
    context_ref: str
    participants: list[_ParticipantProfile]


def _new_relationship_ref(context_ref: str) -> str:
    return f"relationship-{context_ref}"


def _new_track_ref(source_id: IdentityValue) -> TrackRef:
    normalized = value.normalize_embedding(source_id)
    return f"track-{value.identity_json_value(normalized)}"


def _stable_identity_basis(
    identities: collections.abc.Iterable[IdentityValue],
) -> tuple[IdentityValue, ...]:
    named = tuple(sorted(identity for identity in identities if isinstance(identity, str)))
    vectors = tuple(
        sorted(
            (value.normalize_embedding(identity) for identity in identities if not isinstance(identity, str)),
            key=value.identity_sort_key,
        )
    )
    return (*named, *vectors)


def _stable_context_ref(data: TurnData) -> str:
    return conversation_memory.relationship_context_ref(
        source_ids=_stable_identity_basis(data.source_ids),
        target_ids=_stable_identity_basis(data.target_ids),
    )


def _identity_collection_score(
    observed: tuple[IdentityValue, ...],
    expected: tuple[IdentityValue, ...],
    *,
    threshold: float,
    margin: float,
) -> float | None:
    """Match one relationship identity group without hashing vector identities."""

    if len(observed) != len(expected):
        return None
    observed_order = tuple(sorted(observed, key=value.identity_sort_key))
    expected_order = tuple(sorted(expected, key=value.identity_sort_key))
    score_matrix: list[list[float | None]] = []
    for observed_identity in observed_order:
        row: list[float | None] = []
        for expected_identity in expected_order:
            if isinstance(observed_identity, str) or isinstance(expected_identity, str):
                row.append(1.0 if observed_identity == expected_identity else None)
                continue
            try:
                score = value.cosine_similarity(observed_identity, expected_identity)
            except (TypeError, ValueError):
                score = None
            row.append(score if score is not None and score >= threshold else None)
        score_matrix.append(row)

    best = _maximum_identity_assignment(score_matrix)
    if best is None:
        return None
    best_score, best_assignment = best
    second_best: float | None = None
    for row_index, column in enumerate(best_assignment):
        alternative = _maximum_identity_assignment(score_matrix, forbidden={(row_index, column)})
        if alternative is not None and (second_best is None or alternative[0] > second_best):
            second_best = alternative[0]
    if second_best is not None and best_score - second_best <= margin:
        return None
    return best_score / len(best_assignment)


def _maximum_identity_assignment(
    scores: list[list[float | None]],
    *,
    forbidden: collections.abc.Set[tuple[int, int]] = frozenset(),
) -> tuple[float, tuple[int, ...]] | None:
    """Return a deterministic maximum-weight perfect assignment in O(n³).

    ``None`` edges are below the similarity threshold and cannot be assigned.
    The optional forbidden edge set is used to find the best assignment other
    than a known optimum without enumerating permutations.
    """

    size = len(scores)
    if size == 0 or any(len(row) != size for row in scores):
        return None
    potentials_row = [0.0] * (size + 1)
    potentials_column = [0.0] * (size + 1)
    matched_column = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row_index in range(1, size + 1):
        matched_column[0] = row_index
        column_zero = 0
        minimum_cost = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column_zero] = True
            row_zero = matched_column[column_zero]
            delta = float("inf")
            column_one = 0
            for column_index in range(1, size + 1):
                if used[column_index] or (row_zero - 1, column_index - 1) in forbidden:
                    continue
                score = scores[row_zero - 1][column_index - 1]
                if score is None:
                    continue
                cost = -score - potentials_row[row_zero] - potentials_column[column_index]
                if cost < minimum_cost[column_index]:
                    minimum_cost[column_index] = cost
                    predecessor[column_index] = column_zero
                if minimum_cost[column_index] < delta:
                    delta = minimum_cost[column_index]
                    column_one = column_index
            if delta == float("inf"):
                return None
            for column_index in range(size + 1):
                if used[column_index]:
                    potentials_row[matched_column[column_index]] += delta
                    potentials_column[column_index] -= delta
                else:
                    minimum_cost[column_index] -= delta
            column_zero = column_one
            if matched_column[column_zero] == 0:
                break
        while True:
            column_one = predecessor[column_zero]
            matched_column[column_zero] = matched_column[column_one]
            column_zero = column_one
            if column_zero == 0:
                break

    assignment = [0] * size
    for column_index in range(1, size + 1):
        row_index = matched_column[column_index]
        if row_index == 0:
            return None
        assignment[row_index - 1] = column_index - 1
    assigned_scores = [scores[row][column] for row, column in enumerate(assignment)]
    if any(score is None for score in assigned_scores):
        return None
    return sum(typ.cast(float, score) for score in assigned_scores), tuple(assignment)


def _operation_id(event: hsm.Event[typ.Any]) -> str:
    if not event.id:
        raise ValueError(f"Conversation event {event.name!r} requires an id.")
    return event.id


def _with_operation(
    event: hsm.Event[typ.Any],
    source: hsm.Event[typ.Any],
    operation_id: str | None = None,
) -> hsm.Event[typ.Any]:
    resolved = operation_id or source.id
    if resolved:
        event = event.with_data_and_id(event.data, resolved)
    return dataclasses.replace(event, metadata=dict(source.metadata))


conversation_event_with_operation = _with_operation


def _model_safe_content(content: object) -> MessageContent:
    """Project completed products to JSON-safe content without carrying raw media."""

    if content is None or isinstance(content, (str, int, float, bool)):
        return content
    if isinstance(content, bytes | bytearray | memoryview):
        return None
    if isinstance(content, dict):
        projected: dict[str, MessageContent] = {}
        for key, item in content.items():
            if not isinstance(key, str):
                continue
            projected[key] = _model_safe_content(item)
        return typ.cast(MessageContent, projected)
    if isinstance(content, list | tuple):
        return [_model_safe_content(item) for item in content]
    return None


def _content_for_detector(
    data: TurnData,
    source_id: IdentityValue,
) -> turn_detector.ParticipationStimulus:
    """Preserve the declared modality while adapting to the detector boundary.

    Raw PCM (``audio/pcm``, ``audio/raw``, …) must stay an ``AudioStimulus`` with rate and
    channels so STT can wrap WAV at the real sample rate. Only matching ``audio/raw`` used to
    work; ``audio/pcm`` from Listening SpeechData fell through to ContentStimulus and lost packaging.
    """

    content_type = data.content_type.lower()
    if content_type == "text/plain" and isinstance(data.content, str):
        return turn_detector.TextStimulus(source_participant_ref=source_id, content=data.content)
    if content_type.startswith("audio/") and isinstance(data.content, bytes):
        return turn_detector.AudioStimulus(
            source_participant_ref=source_id,
            content=data.content,
            sample_rate_hz=data.sample_rate_hz,
            channels=data.channels,
        )
    return turn_detector.ContentStimulus(
        source_participant_ref=source_id,
        content=data.content,
        content_type=data.content_type,
    )


async def _run_memory_operation(
    ctx: hsm.Context,
    *,
    memory: ability.Ability[typ.Any, typ.Any],
    operation_id: str,
    operation: typ.Literal["recall", "remember"],
    memory_input: memory_ability.InputData,
    metadata: collections.abc.Mapping[str, object],
    timeout: float,
) -> memory_ability.OutputData:
    request_id = f"{operation_id}:memory:{operation}"
    try:
        terminal = await ability.run_terminal_operation(
            ctx,
            child=memory,
            request=dataclasses.replace(
                memory.input_event.with_data_and_id(memory_input, request_id),
                metadata=dict(metadata),
            ),
            terminals=(memory.output_event, memory.failed_event),
            timeout=datetime.timedelta(seconds=timeout),
        )
    except TimeoutError as error:
        raise RuntimeError(f"Conversation memory {operation} timed out.") from error

    if isinstance(terminal.data, ability.FailureData):
        raise RuntimeError(terminal.data.message)
    if not isinstance(terminal.data, memory_ability.OutputData):
        raise TypeError(f"Conversation memory {operation} produced an invalid output.")
    return terminal.data


def _messages_from_contribution_terminal(terminal: hsm.Event[typ.Any]) -> Messages:
    """Extract cumulative history from a Conversation contribution terminal."""

    if isinstance(terminal.data, Messages):
        return terminal.data
    if isinstance(terminal.data, cognition.InputData):
        stimulus = terminal.data.stimulus
        if isinstance(stimulus, hsm.Event) and isinstance(stimulus.data, Messages):
            return stimulus.data
    raise RuntimeError("Conversation produced no messages terminal.")


def _terminal_output(
    ctx: hsm.Context,
    instance: "Conversation",
    event: hsm.Event[typ.Any],
    output: Messages,
    *,
    reply_target: str | None,
) -> None:
    history_event = dataclasses.replace(
        _with_operation(instance.output_event.with_data(output), event),
        source=hsm.id(instance),
        target=reply_target,
    )
    latest_inbound = next((item for item in reversed(output.messages) if item.direction == "inbound"), None)
    text_product = (
        latest_inbound is not None
        and isinstance(latest_inbound.content, str)
        and latest_inbound.content_type.lower().startswith("text/")
    )
    if text_product:
        # Text contributions hand body cognition a typed history stimulus. Non-text products
        # remain bare Conversation output terminals, preserving the audio trust boundary.
        handoff = dataclasses.replace(
            _with_operation(
                cognition.InputEvent.with_data(cognition.InputData(stimulus=history_event)),
                event,
            ),
            source=hsm.id(instance),
            target=reply_target,
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(handoff))
        return
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(history_event))


def _terminal_failure(
    ctx: hsm.Context,
    instance: "Conversation",
    event: hsm.Event[typ.Any],
    failure: FailureData,
    *,
    reply_target: str | None,
) -> None:
    terminal = dataclasses.replace(
        _with_operation(instance.failed_event.with_data(failure), event),
        source=hsm.id(instance),
        target=reply_target,
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


class Conversation(ability.Ability[TurnData, Messages]):
    """Infer relationships and own one detector per participant track.

    One Conversation instance is the topology-owned ambient stream boundary:
    targetless inputs in that instance share its ambient relationship. Separate
    rooms require separate Conversation instances because the input has no
    conversation_ref.
    """

    input_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = TurnData
    output_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = Messages
    input_event: typ.ClassVar[hsm.Event[TurnData]] = InputEvent
    output_event: typ.ClassVar[hsm.Event[Messages]] = OutputEvent
    failed_event: typ.ClassVar[hsm.Event[FailureData]] = FailedEvent
    snapshot_request_event: typ.ClassVar[hsm.Event[SnapshotRequest]] = SnapshotRequestEvent
    snapshot_output_event: typ.ClassVar[hsm.Event[Snapshot]] = SnapshotOutputEvent
    _composite_attachment_lifecycle: typ.ClassVar[bool] = False
    submodel: typ.ClassVar[hsm.Model | None] = bot.define(
        "Conversation",
        hsm.initial(hsm.target("/Conversation/inactive")),
        hsm.state("inactive"),
        hsm.state("detaching"),
        hsm.state("degraded"),
        hsm.observe(observer),
    )

    _turn_detector_factory: TurnDetectorFactory
    _detectors: dict[tuple[str, TrackRef], turn_detector.TurnDetector]
    _relationships: list[_RelationshipProfile]
    _similarity_threshold: float
    _similarity_margin: float
    _typing: language.TextGeneration | None
    _encoding: encoding_module.Encoding[typ.Any, str | bytes] | None
    _memory: ability.Ability[typ.Any, typ.Any] | None
    _history: list[Message]
    _history_parent: events.StimulusData[TurnData] | None

    def __init__(
        self,
        *,
        turn_detector: turn_detector.TurnDetector | None = None,
        turn_detector_factory: TurnDetectorFactory | None = None,
        typing: language.TextGeneration | None = None,
        encoding: encoding_module.Encoding[typ.Any, str | bytes] | None = None,
        encoder: encoding_module.Encoder[typ.Any, str | bytes] | None = None,
        memory: ability.Ability[typ.Any, typ.Any] | None = None,
        similarity_threshold: float = 0.85,
        similarity_margin: float = 0.05,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
        if similarity_margin < 0.0 or similarity_margin > 1.0:
            raise ValueError("similarity_margin must be between 0.0 and 1.0.")
        if turn_detector is not None and turn_detector_factory is not None:
            raise ValueError("Conversation accepts turn_detector or turn_detector_factory, not both.")
        if turn_detector is not None:
            template_detector = turn_detector

            def factory(session_ref: str, track_ref: TrackRef) -> turn_detector.TurnDetector:
                return template_detector.clone_for(participant_ref=track_ref, conversation_ref=session_ref)

            self._turn_detector_factory = factory
        elif turn_detector_factory is not None:
            self._turn_detector_factory = turn_detector_factory
        else:
            self._turn_detector_factory = _default_turn_detector
        self._encoding = encoding_module.Encoding(encoder=encoder) if encoder is not None else encoding
        self._typing = typing
        self._memory = memory
        super().__init__()
        self._detectors = {}
        self._relationships = []
        self._similarity_threshold = similarity_threshold
        self._similarity_margin = similarity_margin
        self._history = []
        self._history_parent = None

    @property
    def typing(self) -> language.TextGeneration | None:
        return self._typing

    @property
    def encoding(self) -> encoding_module.Encoding[typ.Any, str | bytes] | None:
        return self._encoding

    @property
    def memory(self) -> ability.Ability[typ.Any, typ.Any] | None:
        return self._memory

    @property
    def detector_refs(self) -> tuple[tuple[str, IdentityValue], ...]:
        refs = [
            (relationship.session_ref, participant.source_id)
            for relationship in self._relationships
            for participant in relationship.participants
            if participant.detector is not None
        ]
        return tuple(sorted(refs, key=lambda item: (item[0], value.identity_sort_key(item[1]))))

    @property
    def session_refs(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                relationship.session_ref
                for relationship in self._relationships
                if any(participant.detector is not None for participant in relationship.participants)
            )
        )

    @property
    def detector_count(self) -> int:
        return len(self._detectors)

    def _relationship_for_input(self, data: TurnData) -> _RelationshipProfile:
        observed_sources = value.sorted_identities(data.source_ids)
        observed_targets = value.sorted_identities(data.target_ids)
        observed_groups = value.identity_groups(data.source_ids, data.target_ids)
        if not observed_targets:
            ambient = next(
                (relationship for relationship in self._relationships if not relationship.target_ids),
                None,
            )
            if ambient is not None:
                return ambient
            relationship = _RelationshipProfile(
                session_ref=_new_relationship_ref(_stable_context_ref(data)),
                source_ids=observed_sources,
                target_ids=observed_targets,
                identity_groups=observed_groups,
                context_ref=_stable_context_ref(data),
                participants=[],
            )
            self._relationships.append(relationship)
            return relationship

        candidates: list[tuple[float, _RelationshipProfile]] = []
        for relationship in self._relationships:
            if not relationship.target_ids and any(
                (
                    isinstance(target_id, str)
                    and isinstance(participant.source_id, str)
                    and target_id == participant.source_id
                )
                or (
                    not isinstance(target_id, str)
                    and not isinstance(participant.source_id, str)
                    and (value.cosine_similarity(target_id, participant.source_id) or 0.0) >= self._similarity_threshold
                )
                for target_id in observed_targets
                for participant in relationship.participants
            ):
                candidates.append((self._similarity_threshold, relationship))
                continue
            direct_source_score = _identity_collection_score(
                observed_sources,
                relationship.source_ids,
                threshold=self._similarity_threshold,
                margin=self._similarity_margin,
            )
            direct_target_score = _identity_collection_score(
                observed_targets,
                relationship.target_ids,
                threshold=self._similarity_threshold,
                margin=self._similarity_margin,
            )
            direct_score = (
                min(direct_source_score, direct_target_score)
                if direct_source_score is not None and direct_target_score is not None
                else None
            )

            reverse_source_score = _identity_collection_score(
                observed_sources,
                relationship.target_ids,
                threshold=self._similarity_threshold,
                margin=self._similarity_margin,
            )
            reverse_target_score = _identity_collection_score(
                observed_targets,
                relationship.source_ids,
                threshold=self._similarity_threshold,
                margin=self._similarity_margin,
            )
            reverse_score = (
                min(reverse_source_score, reverse_target_score)
                if reverse_source_score is not None and reverse_target_score is not None
                else None
            )
            score = (
                direct_score
                if reverse_score is None
                else reverse_score
                if direct_score is None
                else max(direct_score, reverse_score)
            )
            if score is not None:
                candidates.append((score, relationship))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1].session_ref))
            return candidates[0][1]
        relationship = _RelationshipProfile(
            session_ref=_new_relationship_ref(_stable_context_ref(data)),
            source_ids=observed_sources,
            target_ids=observed_targets,
            identity_groups=observed_groups,
            context_ref=_stable_context_ref(data),
            participants=[],
        )
        self._relationships.append(relationship)
        return relationship

    def _participant_for_source(
        self,
        relationship: _RelationshipProfile,
        source_id: IdentityValue,
        used_tracks: frozenset[TrackRef],
    ) -> _ParticipantProfile:
        candidates = [
            participant for participant in relationship.participants if participant.track_ref not in used_tracks
        ]
        if isinstance(source_id, str):
            exact = next(
                (participant for participant in candidates if participant.source_id == source_id),
                None,
            )
            if exact is not None:
                return exact
        else:
            vector_candidates = sorted(
                (participant for participant in candidates if participant.centroid is not None),
                key=lambda participant: participant.track_ref,
            )
            matched_index = value.match_vector(
                source_id,
                (participant.centroid for participant in vector_candidates),
                threshold=self._similarity_threshold,
                margin=self._similarity_margin,
            )
            if matched_index is not None:
                return vector_candidates[matched_index]

        track_ref = source_id if isinstance(source_id, str) else _new_track_ref(source_id)
        participant = _ParticipantProfile(track_ref=track_ref, source_id=source_id)
        relationship.participants.append(participant)
        return participant

    @staticmethod
    def _profile_update(
        profile: _ParticipantProfile,
        source_id: IdentityValue,
    ) -> tuple[IdentityValue, value.Embedding | None, value.Embedding | None, int]:
        if isinstance(source_id, str):
            return source_id, profile.centroid, profile.vector_sum, profile.count
        observation = value.normalize_embedding(source_id)
        if profile.count == 0:
            return source_id, observation, observation, 1
        if profile.vector_sum is None:
            raise RuntimeError("A completed vector profile is missing its centroid.")
        if len(profile.vector_sum) != len(observation):
            raise ValueError(
                f"embedding dimension {len(observation)} does not match profile dimension {len(profile.vector_sum)}."
            )
        vector_sum = tuple(
            accumulated_component + observation_component
            for accumulated_component, observation_component in zip(
                profile.vector_sum,
                observation,
                strict=True,
            )
        )
        centroid = value.normalize_embedding(
            vector_sum,
        )
        return source_id, centroid, vector_sum, profile.count + 1

    @staticmethod
    def _has_input(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, TurnData)

    @staticmethod
    def _has_routed_input(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, RoutedInputData)

    @staticmethod
    def _has_snapshot_request(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, SnapshotRequest)

    @staticmethod
    def _has_append(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, AppendData)

    @staticmethod
    def _append_message(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        data = event.data
        assert isinstance(data, AppendData)
        message = data.message
        if message.direction != "outbound":
            raise ValueError("Conversation append accepts outbound messages only.")
        event_source = event.source or None
        event_target = event.target or None
        duplicate = bool(event.id) and any(
            committed.provenance.id == event.id
            and committed.provenance.source == event_source
            and committed.provenance.target == event_target
            for committed in instance._history
        )
        if not duplicate:
            provenance = message.provenance.model_copy(
                update={
                    "event": event.name,
                    "id": event.id or None,
                    "source": event_source,
                    "target": event_target,
                }
            )
            committed = message.model_copy(update={"sequence": len(instance._history), "provenance": provenance})
            instance._history.append(committed)
        output = Messages(parent=instance._history_parent, messages=tuple(instance._history))
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            source=hsm.id(instance),
            target=event.source,
            metadata=dict(event.metadata),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _has_input_cancelled(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, _InputCancelledData)
            and event.id == data.operation_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
        )

    @staticmethod
    def _queue_input(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        with span.operation(
            "bot.conversation.ingress",
            scope="bot.abilities.communication",
            component="communication.conversation",
            stage="conversation_ingress",
            context=telemetry.event_context(event),
        ) as active:
            data = event.data
            assert isinstance(data, TurnData)
            active.set_attribute("bot.identity.source.count", len(data.source_ids))
            active.set_attribute("bot.identity.target.count", len(data.target_ids))
            active.set_attribute("bot.content.type", data.content_type or "")
            operation_id = _operation_id(event)
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _InputWorkEvent.with_data(
                        _InputWorkData(input=data, input_parent=events.StimulusData.from_event(event))
                    ),
                    id=operation_id,
                    source=event.source or hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )

    @staticmethod
    def _queue_routed_input(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        routed = event.data
        assert isinstance(routed, RoutedInputData)
        operation_id = _operation_id(event)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _InputWorkEvent.with_data(_InputWorkData(input=routed.parent.data, input_parent=routed.parent)),
                id=operation_id,
                source=event.source or hsm.id(instance),
                target=hsm.id(instance),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _run_input(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        work = event.data
        assert isinstance(work, _InputWorkData)
        data = work.input
        operation_id = _operation_id(event)
        relationships_before = tuple(instance._relationships)
        relationship = instance._relationship_for_input(data)
        relationship_ref = relationship.session_ref
        relationship_provisional = not any(candidate is relationship for candidate in relationships_before)
        original_participants = tuple(relationship.participants)
        profile_snapshots = tuple(
            (
                participant,
                participant.source_id,
                participant.centroid,
                participant.vector_sum,
                participant.count,
                participant.detector,
            )
            for participant in original_participants
        )
        operation_detectors: dict[tuple[str, TrackRef], turn_detector.TurnDetector] = {}
        provisional_detector_keys: set[tuple[str, TrackRef]] = set()
        failed_detector_keys: set[tuple[str, TrackRef]] = set()
        active_turn_detector_keys: set[tuple[str, TrackRef]] = set()

        async def rollback(
            *,
            stop_all: bool,
            remove_relationship: bool,
        ) -> tuple[str, ...]:
            keys_to_stop = (
                active_turn_detector_keys | provisional_detector_keys
                if stop_all
                else provisional_detector_keys | failed_detector_keys
            )
            cleanup_failures: list[str] = []
            stopped_keys: set[tuple[str, TrackRef]] = set()
            for detector_key in keys_to_stop:
                detector = operation_detectors.get(detector_key)
                if detector is None:
                    continue
                try:
                    await hsm.stop(detector, instance.context())
                    instance._detectors.pop(detector_key, None)
                    stopped_keys.add(detector_key)
                except BaseException as error:
                    cleanup_failures.append(f"failed to stop detector {detector_key[1]!r}: {error}")
            for participant, source_id, centroid, vector_sum, count, detector in profile_snapshots:
                participant.source_id = source_id
                participant.centroid = centroid
                participant.vector_sum = vector_sum
                participant.count = count
                participant.detector = None if (relationship_ref, participant.track_ref) in stopped_keys else detector
            for participant in relationship.participants:
                if (relationship_ref, participant.track_ref) in stopped_keys:
                    participant.detector = None
            relationship.participants[:] = (
                list(original_participants) if remove_relationship else list(relationship.participants)
            )
            if remove_relationship and relationship_provisional and relationship in instance._relationships:
                instance._relationships.remove(relationship)
            return tuple(cleanup_failures)

        turn_refs: dict[TrackRef, str] = {}
        provenance = _TurnOperationProvenance(
            operation_id=operation_id,
            session_ref=relationship_ref,
            source_ids=data.source_ids,
            turn_refs=frozenset(turn_refs.values()),
        )
        memories: tuple[conversation_memory.Memory, ...] = ()
        stage: Stage = "memory"
        processing_track_ref: TrackRef | None = None
        committed = False
        try:
            if instance._memory is not None:
                memory_output = await _run_memory_operation(
                    ctx,
                    memory=instance._memory,
                    operation_id=operation_id,
                    operation="recall",
                    memory_input=conversation_memory.conversation_memory_recall_input(
                        source_ids=data.source_ids,
                        target_ids=data.target_ids,
                        context_ref=relationship.context_ref,
                    ),
                    metadata=event.metadata,
                    timeout=_TURN_TIMEOUT_SECONDS,
                )
                memories = conversation_memory.conversation_memories_from_output(memory_output)

            stage = "turn_detector"
            detectors: list[tuple[IdentityValue, _ParticipantProfile]] = []
            used_tracks: frozenset[TrackRef] = frozenset()
            for source_id in value.sorted_identities(data.source_ids):
                participant = instance._participant_for_source(relationship, source_id, used_tracks)
                processing_track_ref = participant.track_ref
                detector_key = (relationship_ref, participant.track_ref)
                detector = participant.detector or instance._detectors.get(detector_key)
                if detector is None:
                    detector = instance._turn_detector_factory(relationship_ref, participant.track_ref)
                    if not isinstance(detector, turn_detector.TurnDetector):
                        raise TypeError("turn_detector_factory must return a TurnDetector.")
                    operation_detectors[detector_key] = detector
                    provisional_detector_keys.add(detector_key)
                    # This child must outlive this activity; the parent Conversation
                    # context is its lifetime, never the activity context ``ctx``.
                    await bot.started(instance.context(), detector, detector.owned_model)
                    # The detector is owned by Conversation's context and is
                    # coordinated through one-shot directed operations.
                    ready_operation_id = f"{operation_id}:ready:{participant.track_ref}"
                    try:
                        ready = await ability.run_terminal_operation(
                            ctx,
                            child=detector,
                            request=dataclasses.replace(
                                turn_detector.TurnDetectorReadyRequestEvent.with_data_and_id(
                                    turn_detector.TurnDetectorReadyRequestData(
                                        participant_ref=participant.track_ref,
                                        conversation_ref=relationship_ref,
                                    ),
                                    ready_operation_id,
                                ),
                                metadata=dict(event.metadata),
                            ),
                            terminals=(turn_detector.TurnDetectorReadyEvent, detector.failed_event),
                            timeout=datetime.timedelta(seconds=_TURN_TIMEOUT_SECONDS),
                        )
                    except TimeoutError as error:
                        raise RuntimeError(
                            f"Turn detector readiness timed out after {_TURN_TIMEOUT_SECONDS:g} seconds."
                        ) from error
                    if (
                        ready.name != turn_detector.TurnDetectorReadyEvent.name
                        or not isinstance(ready.data, turn_detector.TurnDetectorReadyData)
                        or ready.data.participant_ref != participant.track_ref
                        or ready.data.conversation_ref != relationship_ref
                    ):
                        raise RuntimeError("Turn detector did not produce a correlated readiness terminal.")
                    instance._detectors[detector_key] = detector
                    participant.detector = detector
                else:
                    operation_detectors[detector_key] = detector
                provenance = provenance.model_copy(
                    update={"detector_ids": provenance.detector_ids | frozenset({participant.track_ref})}
                )
                turn_refs[participant.track_ref] = f"{operation_id}:turn:{participant.track_ref}"
                used_tracks = used_tracks | frozenset({participant.track_ref})
                detectors.append((source_id, participant))
            if not detectors:
                raise ValueError("Conversation input requires at least one source identity.")
            provenance = provenance.model_copy(update={"turn_refs": frozenset(turn_refs.values())})

            terminals: list[turn_detector.TurnCompleteData] = []
            for source_id, participant in detectors:
                processing_track_ref = participant.track_ref
                detector = participant.detector
                if detector is None:
                    raise RuntimeError(f"Participant track {participant.track_ref!r} has no turn detector.")
                detector_key = (relationship_ref, participant.track_ref)
                turn_ref = turn_refs[participant.track_ref]
                content = _content_for_detector(data, source_id)
                start = turn_detector.TurnStartData(
                    conversation_ref=relationship_ref,
                    turn_ref=turn_ref,
                    self_participant_ref=participant.track_ref,
                    source_participant_ref=source_id,
                    content=content,
                )
                active_turn_detector_keys.add(detector_key)
                with span.operation(
                    "bot.conversation.turn_exchange",
                    scope="bot.abilities.communication",
                    component="communication.conversation",
                    stage="turn_detector",
                    context=telemetry.event_context(event),
                ):
                    # One start/end exchange with one detector. A trace that stops here says the
                    # product reached the detector and no turn ever came back.
                    child_operation_id = turn_ref
                    try:
                        await hsm.dispatch(
                            ctx,
                            detector,
                            dataclasses.replace(
                                turn_detector.TurnStartEvent.with_data(start),
                                id=child_operation_id,
                                source=hsm.id(instance),
                                target=hsm.id(detector),
                                metadata=dict(event.metadata),
                            ),
                        )
                        terminal = await ability.run_terminal_operation(
                            ctx,
                            child=detector,
                            request=dataclasses.replace(
                                turn_detector.TurnEndEvent.with_data_and_id(
                                    turn_detector.TurnEndData(
                                        conversation_ref=relationship_ref,
                                        turn_ref=turn_ref,
                                        source_participant_ref=source_id,
                                    ),
                                    child_operation_id,
                                ),
                                metadata=dict(event.metadata),
                            ),
                            terminals=(detector.output_event, detector.failed_event),
                            timeout=datetime.timedelta(seconds=_TURN_TIMEOUT_SECONDS),
                        )
                    except TimeoutError as error:
                        provenance = provenance.model_copy(
                            update={
                                "detector_ids": provenance.detector_ids - frozenset({participant.track_ref}),
                                "failure_track_ref": participant.track_ref,
                            }
                        )
                        raise RuntimeError(
                            f"Turn detector turn timed out after {_TURN_TIMEOUT_SECONDS:g} seconds."
                        ) from error
                    if terminal.name == detector.failed_event.name and isinstance(
                        terminal.data, turn_detector.FailedEventData
                    ):
                        raise RuntimeError(terminal.data.message)
                    if (
                        terminal.name != detector.output_event.name
                        or not isinstance(terminal.data, turn_detector.TurnCompleteData)
                        or terminal.data.conversation_ref != relationship_ref
                        or terminal.data.turn_ref != turn_ref
                        or terminal.data.source_participant_ref != source_id
                        or terminal.data.participant_ref != participant.track_ref
                    ):
                        raise RuntimeError(
                            "Turn detector returned an unrelated completion: "
                            f"id={terminal.id!r} source={terminal.source!r} name={terminal.name!r} "
                            f"expected_name={detector.output_event.name!r} data={terminal.data!r}"
                        )
                    terminals.append(terminal.data)
                    active_turn_detector_keys.discard(detector_key)

            staged_profiles = tuple(
                (participant, instance._profile_update(participant, source_id)) for source_id, participant in detectors
            )

            stage = "memory"
            if instance._memory is not None:
                remembered = conversation_memory.Memory(
                    source_ids=data.source_ids,
                    target_ids=data.target_ids,
                    content=data.content,
                    content_type=data.content_type,
                )
                memory_scope = getattr(type(instance._memory), "default_scope", "short_term")
                if not isinstance(memory_scope, str) or not memory_scope.strip():
                    memory_scope = "short_term"
                _ = await _run_memory_operation(
                    ctx,
                    memory=instance._memory,
                    operation_id=operation_id,
                    operation="remember",
                    memory_input=conversation_memory.conversation_memory_insert_input(
                        remembered,
                        scope=memory_scope.strip(),
                        context_ref=relationship.context_ref,
                    ),
                    metadata=event.metadata,
                    timeout=_TURN_TIMEOUT_SECONDS,
                )

            for participant, (source_id, centroid, vector_sum, count) in staged_profiles:
                participant.source_id = source_id
                participant.centroid = centroid
                participant.vector_sum = vector_sum
                participant.count = count
            committed = True

            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _TurnOperationCompletedEvent.with_data(
                        _TurnOperationCompletedData(
                            input=data,
                            input_parent=work.input_parent,
                            operation_id=operation_id,
                            provenance=provenance,
                            turns=tuple(terminals),
                            memories=memories,
                        )
                    ),
                    id=operation_id,
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                ),
            )
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if not committed and task is not None and task.cancelling() > 0:
                await rollback(stop_all=True, remove_relationship=True)
            raise
        except Exception as error:
            if stage == "turn_detector" and processing_track_ref is not None:
                failed_detector_keys.add((relationship_ref, processing_track_ref))
                provisional_detector_keys.clear()
            cleanup_failures = await rollback(
                stop_all=False,
                remove_relationship=False,
            )
            failure_track_ref = (
                processing_track_ref
                if stage == "turn_detector"
                and processing_track_ref is not None
                and (relationship_ref, processing_track_ref) not in instance._detectors
                else None
            )
            failure_provenance = provenance.model_copy(
                update={
                    "detector_ids": (
                        provenance.detector_ids - frozenset({failure_track_ref})
                        if failure_track_ref is not None
                        else provenance.detector_ids
                    ),
                    "failure_track_ref": failure_track_ref,
                }
            )
            await hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _with_operation(
                        _TurnFailedEvent.with_data(
                            _TurnFailedData(
                                input=data,
                                input_parent=work.input_parent,
                                operation_id=operation_id,
                                provenance=failure_provenance,
                                failure=FailureData(
                                    stage=stage,
                                    message=(
                                        f"{error}; cleanup: {'; '.join(cleanup_failures)}"
                                        if cleanup_failures
                                        else str(error)
                                    ),
                                ),
                                memories=memories,
                            )
                        ),
                        event,
                        operation_id,
                    ),
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                ),
            )

    @staticmethod
    def _has_turn_operation_completed(
        ctx: hsm.Context,
        instance: "Conversation",
        event: hsm.Event[typ.Any],
    ) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _TurnOperationCompletedData):
            return False
        relationship = next(
            (
                candidate
                for candidate in instance._relationships
                if candidate.session_ref == data.provenance.session_ref
            ),
            None,
        )
        if relationship is None:
            return False
        relationship_ref = relationship.session_ref
        if (
            event.id != data.operation_id
            or event.source != hsm.id(instance)
            or event.target != hsm.id(instance)
            or relationship_ref != data.provenance.session_ref
            or data.input.source_ids != data.provenance.source_ids
            or data.provenance.operation_id != data.operation_id
            or not data.provenance.detector_ids
            or not data.provenance.turn_refs
            or len(data.turns) != len(data.input.source_ids)
            or len({turn.source_participant_ref for turn in data.turns}) != len(data.turns)
            or data.provenance.turn_refs != frozenset(turn.turn_ref for turn in data.turns)
            or data.provenance.detector_ids != frozenset(turn.participant_ref for turn in data.turns)
        ):
            return False
        if any(
            not any(turn.source_participant_ref == source_id for source_id in data.input.source_ids)
            for turn in data.turns
        ):
            return False
        for turn in data.turns:
            if (
                relationship_ref,
                turn.participant_ref,
            ) not in instance._detectors or turn.conversation_ref != relationship_ref:
                return False
        return True

    @staticmethod
    def _has_failure(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _TurnFailedData):
            return False
        relationship_ref = data.provenance.session_ref
        provenance = data.provenance
        relationship = next(
            (candidate for candidate in instance._relationships if candidate.session_ref == relationship_ref),
            None,
        )
        if relationship is None:
            return False
        cached_detector_ids = frozenset(
            track_ref
            for track_ref in provenance.detector_ids
            if instance._detectors.get((provenance.session_ref, track_ref)) is not None
        )
        missing_detector_ids = provenance.detector_ids - cached_detector_ids
        detector_ids_match_cache = cached_detector_ids == provenance.detector_ids
        failure_profile_is_missing = (
            provenance.failure_track_ref is not None
            and provenance.failure_track_ref not in provenance.detector_ids
            and (relationship_ref, provenance.failure_track_ref) not in instance._detectors
            and any(
                participant.track_ref == provenance.failure_track_ref and participant.detector is None
                for participant in relationship.participants
            )
        )
        normal_failure = not missing_detector_ids and provenance.failure_track_ref is None
        source_failure = (
            provenance.failure_track_ref is not None
            and failure_profile_is_missing
            and not any(track_ref < provenance.failure_track_ref for track_ref in missing_detector_ids)
        )
        failure_track_correlated = (
            provenance.failure_track_ref is None
            or provenance.failure_track_ref in provenance.detector_ids
            or (data.failure.stage == "turn_detector" and source_failure)
        )
        correlated = (
            event.id == data.operation_id == provenance.operation_id
            and event.source == hsm.id(instance)
            and event.target == hsm.id(instance)
            and relationship_ref == provenance.session_ref
            and data.input.source_ids == provenance.source_ids
            and (
                len(provenance.turn_refs) == len(provenance.detector_ids)
                or (data.failure.stage == "turn_detector" and source_failure)
            )
            and failure_track_correlated
        )
        if not correlated:
            return False
        if data.failure.stage == "memory":
            return True
        return (
            data.failure.stage == "turn_detector"
            and detector_ids_match_cache
            and (normal_failure or source_failure)
            and all((relationship_ref, track_ref) in instance._detectors for track_ref in cached_detector_ids)
        )

    @staticmethod
    def _complete_turn(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        data = event.data
        assert isinstance(data, _TurnOperationCompletedData)
        input_data = data.input
        # Decode rewrites the product: media stays turn-local and only transcript/structured
        # products enter model-facing committed history.
        non_empty = tuple(part for turn in data.turns if (part := turn.text))
        if non_empty:
            product_content: object = " ".join(non_empty)
            product_type = "text/plain"
        elif input_data.content_type.lower().startswith("audio/") or isinstance(input_data.content, bytes):
            product_content = None
            product_type = input_data.content_type
        else:
            product_content = input_data.content
            product_type = input_data.content_type
        parent = data.input_parent
        provenance = MessageProvenance(
            event=parent.event if parent is not None else instance.input_event.name,
            id=parent.id if parent is not None else event.id,
            source=parent.source if parent is not None else event.source,
            target=parent.target if parent is not None else event.target,
            session_ref=data.provenance.session_ref,
            turn_ref=next(iter(data.provenance.turn_refs), None),
        )
        inbound = _message_for_turn(
            input_data,
            content=_model_safe_content(product_content),
            content_type=product_type,
            sequence=len(instance._history),
            provenance=provenance,
        )
        instance._history.append(inbound)
        instance._history_parent = parent
        _terminal_output(
            ctx,
            instance,
            event,
            Messages(parent=parent, messages=tuple(instance._history), memories=data.memories),
            reply_target=(
                parent.source if parent is not None and parent.target == hsm.id(instance) and parent.source else None
            ),
        )

    @staticmethod
    def _fail_turn(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        failure = event.data
        assert isinstance(failure, _TurnFailedData)
        relationship = next(
            (
                candidate
                for candidate in instance._relationships
                if candidate.session_ref == failure.provenance.session_ref
            ),
            None,
        )
        if relationship is not None:
            relationship.participants[:] = [
                participant for participant in relationship.participants if participant.detector is not None
            ]
        parent = failure.input_parent
        _terminal_failure(
            ctx,
            instance,
            event,
            failure.failure,
            reply_target=(
                parent.source if parent is not None and parent.target == hsm.id(instance) and parent.source else None
            ),
        )

    @staticmethod
    def _snapshot(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        request = event.data
        assert isinstance(request, SnapshotRequest)
        _ = hsm.dispatch(
            ctx,
            instance,
            instance.snapshot_output_event.with_data(
                Snapshot(request_ref=request.request_ref, detector_refs=instance.detector_refs)
            ),
        )

    @classmethod
    def define_model(cls, root_name: str) -> hsm.Model:
        root = f"/{root_name}"
        # Queue work while a turn is running; redeliver when active/waiting. InputEvent is
        # *not* deferred: it stays tool-offerable via an explicit active transition (RC-1).
        deferred = (
            _InputWorkEvent,
            RoutedInputEvent,
            _TurnOperationCompletedEvent,
            _TurnFailedEvent,
        )
        return bot.define(
            root_name,
            hsm.initial(hsm.target(f"{root}/inactive")),
            hsm.transition(
                hsm.on(SnapshotRequestEvent),
                hsm.guard(cls._has_snapshot_request),
                hsm.effect(cls._snapshot),
            ),
            hsm.state(
                "inactive",
                hsm.transition(
                    hsm.on(InputEvent),
                    hsm.guard(cls._has_input),
                    hsm.effect(cls._queue_input),
                    hsm.target(f"{root}/active"),
                ),
                hsm.transition(
                    hsm.on(RoutedInputEvent),
                    hsm.guard(cls._has_routed_input),
                    hsm.effect(cls._queue_routed_input),
                    hsm.target(f"{root}/active"),
                ),
                hsm.transition(
                    hsm.on(AppendEvent),
                    hsm.guard(cls._has_append),
                    hsm.effect(cls._append_message),
                ),
            ),
            hsm.state(
                "active",
                hsm.initial(hsm.target(f"{root}/active/waiting")),
                # Continuous ingress while the relationship is open: queue additional input
                # without leaving active. Visible to enabled_call_events for tool menus.
                hsm.transition(
                    hsm.on(InputEvent),
                    hsm.guard(cls._has_input),
                    hsm.effect(cls._queue_input),
                ),
                hsm.transition(
                    hsm.on(RoutedInputEvent),
                    hsm.guard(cls._has_routed_input),
                    hsm.effect(cls._queue_routed_input),
                ),
                hsm.transition(
                    hsm.on(_InputCancelledEvent),
                    hsm.guard(cls._has_input_cancelled),
                    hsm.target(f"{root}/inactive"),
                ),
                hsm.transition(
                    hsm.on(AppendEvent),
                    hsm.guard(cls._has_append),
                    hsm.effect(cls._append_message),
                ),
                hsm.defer(*deferred),
                hsm.state(
                    "waiting",
                    hsm.transition(hsm.on(_InputWorkEvent), hsm.target(f"{root}/active/running")),
                ),
                hsm.state(
                    "running",
                    hsm.activity(cls._run_input),
                    hsm.transition(
                        hsm.on(_TurnOperationCompletedEvent),
                        hsm.guard(cls._has_turn_operation_completed),
                        hsm.effect(cls._complete_turn),
                        hsm.target(f"{root}/active/waiting"),
                    ),
                    hsm.transition(
                        hsm.on(_TurnFailedEvent),
                        hsm.guard(cls._has_failure),
                        hsm.effect(cls._fail_turn),
                        hsm.target(f"{root}/active/waiting"),
                    ),
                ),
            ),
            hsm.state("detaching"),
            hsm.state("degraded"),
            hsm.observe(observer),
        )

    @classmethod
    def define_lifecycle_model(
        cls,
        name: str,
        submodel: hsm.Model,
        *,
        composite_attachment_lifecycle: bool | None = None,
    ) -> hsm.Model:
        return cls._define_model(
            name,
            submodel,
            composite_attachment_lifecycle=cls._composite_attachment_lifecycle
            if composite_attachment_lifecycle is None
            else composite_attachment_lifecycle,
        )


def define_conversation_model(root_name: str, **_: object) -> hsm.Model:
    """Build the Conversation behavior model."""

    return Conversation.define_model(root_name)


async def append_conversation_message(
    conversation: Conversation,
    message: Message,
    *,
    ctx: hsm.Context | None = None,
) -> Messages:
    """Commit one trusted outbound message through Conversation's typed event boundary."""

    context = conversation.context() if ctx is None else ctx
    operation_id = uuid.uuid4().hex
    terminal = await ability.run_terminal_operation(
        context,
        child=conversation,
        request=AppendEvent.with_data_and_id(AppendData(message=message), operation_id),
        terminals=(conversation.output_event, conversation.failed_event),
        timeout=datetime.timedelta(seconds=_TURN_TIMEOUT_SECONDS),
    )
    if not isinstance(terminal.data, Messages):
        raise RuntimeError("Conversation outbound append produced no history output.")
    return terminal.data


async def contribute_conversation_input(
    conversation: Conversation,
    input_data: TurnData,
    *,
    ctx: hsm.Context | None = None,
) -> ParticipatedTurn:
    """Dispatch one Conversation input and await its typed contribution terminal."""

    context = conversation.context() if ctx is None else ctx
    operation_id = uuid.uuid4().hex
    try:
        terminal = await ability.run_terminal_operation(
            context,
            child=conversation,
            request=conversation.input_event.with_data_and_id(input_data, operation_id),
            terminals=(conversation.output_event, cognition.InputEvent, conversation.failed_event),
            timeout=datetime.timedelta(seconds=_TURN_TIMEOUT_SECONDS * _CONTRIBUTION_TIMEOUT_STAGES),
        )
    except asyncio.CancelledError as cancelled:
        cancellation = dataclasses.replace(
            _InputCancelledEvent.with_data(_InputCancelledData(operation_id=operation_id)),
            id=operation_id,
            source=hsm.id(conversation),
            target=hsm.id(conversation),
        )
        try:
            await asyncio.shield(hsm.dispatch(context, conversation, cancellation))
        except Exception as error:
            span.record_current_failure("input_cancel_dispatch_failed")
            raise cancelled from error
        raise
    if terminal.name == conversation.failed_event.name:
        raise RuntimeError(f"Conversation failed during input contribution: {terminal.data!r}")
    return participated_turn_from_messages(input_data, _messages_from_contribution_terminal(terminal))


Conversation.submodel = define_conversation_model("Conversation")
Conversation.model = Conversation.define_lifecycle_model("Conversation", Conversation.submodel)


__all__ = [
    "TurnData",
    "Conversation",
    "FailedEvent",
    "FailureData",
    "InputEvent",
    "OutputEvent",
    "ParticipatedTurn",
    "participated_turn_from_messages",
    "Messages",
    "Message",
    "MessageProvenance",
    "RoutedInputData",
    "RoutedInputEvent",
    "Snapshot",
    "SnapshotOutputEvent",
    "SnapshotRequest",
    "SnapshotRequestEvent",
    "Stage",
    "TrackRef",
    "TurnDetectorFactory",
    "contribute_conversation_input",
    "append_conversation_message",
    "conversation_event_with_operation",
    "define_conversation_model",
]
