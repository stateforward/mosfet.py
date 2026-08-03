"""Modality-neutral conversation coordination.

Conversation owns the relationship between identity sets.  The input boundary
does not expose a conversation reference: a relationship is inferred from the
source and target sets, and each source identity gets a turn detector in that
relationship.
"""

from __future__ import annotations

from .. import ability
from .. import cognition
from .. import encoding as encoding_module
from .. import language
from .. import listening
from .. import memory as memory_ability
from .. import turn_detector
from .. import turn_detector as turn_detector_module
from ..identity import value
from . import memory as conversation_memory

import asyncio
import collections.abc
import dataclasses
import typing as typ
import uuid

import hsm
import pydantic
from pydantic.config import JsonDict, JsonValue

from bot.telemetry import observer

Stage: typ.TypeAlias = typ.Literal["memory", "turn_detector", "voice_routing"]
Content: typ.TypeAlias = object
IdentitySet: typ.TypeAlias = value.IdentitySet
IdentityValue: typ.TypeAlias = value.IdentityValue
TrackRef: typ.TypeAlias = str
TurnDetectorFactory: typ.TypeAlias = typ.Callable[[str, TrackRef], turn_detector.TurnDetector]

_TURN_TIMEOUT_SECONDS = 5.0
MAX_IDENTITY_SET_SIZE = value.MAX_IDENTITY_SET_SIZE
"""Maximum source or target identities accepted by Conversation input."""
MAX_EMBEDDING_DIMENSION = value.MAX_EMBEDDING_DIMENSION
"""Maximum embedding dimension accepted by Conversation input."""


_INPUT_EXAMPLE: JsonDict = {
    "source_ids": ["caller"],
    "target_ids": [],
    "content": "What is the weather like?",
    "content_type": "text/plain",
}


def _schema(description: str, example: JsonDict) -> JsonDict:
    examples: list[JsonValue] = [example]
    return {"description": description, "examples": examples}


class ConversationInputData(pydantic.BaseModel):
    """One modality-neutral input entering Conversation.

    ``source_ids`` and ``target_ids`` are sets of opaque identity references.
    The relationship they describe is used only to derive a transient reference;
    ``content_type`` describes the payload and never selects an HSM route.
    """

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        json_schema_extra=_schema(
            "Conversation input identified by source identities and an optional target identity set; an empty target set represents an ambient input with an unknown addressee.",
            _INPUT_EXAMPLE,
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
    content: Content = pydantic.Field(
        description="Modality-specific content carried without interpretation by Conversation.",
        examples=["What is the weather like?", "AAECAw=="],
    )
    content_type: str = pydantic.Field(
        min_length=1,
        description="Content media type or modality label; it is data, not an event route.",
        examples=["text/plain", "audio/raw", "application/x-sign-language"],
    )

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
                f"{field_name} contains {len(identities)} identities; the maximum is "
                f"{MAX_IDENTITY_SET_SIZE}."
            )
        for identity in identities:
            if not isinstance(identity, str):
                _ = value.normalize_embedding(identity, field_name=field_name)
        return identities

    @pydantic.field_serializer("source_ids", "target_ids", when_used="json")
    def serialize_ids(self, identities: IdentitySet) -> list[str | list[float]]:
        return value.identities_json(identities)


class Response(pydantic.BaseModel):
    """Conversation's contribution terminal.

    ``session_ref`` is an opaque relationship reference for terminal
    correlation and turn-detector provenance.  It is derived from the current
    identity sets for each operation; Conversation never stores a session
    object for it.
    """

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    source_ids: IdentitySet = pydantic.Field(description="Source identities for the completed turn.")
    target_ids: IdentitySet = pydantic.Field(
        description="Zero or more target identities for the completed turn; an empty set means no addressee is known.",
        examples=[[], ["bot"]],
    )
    content: Content | None = pydantic.Field(default=None, description="Optional response payload.")
    content_type: str = pydantic.Field(description="Response content modality.")
    decoded_text: str | None = pydantic.Field(default=None, description="Readable text when available.")
    session_ref: str = pydantic.Field(
        description="Transient relationship reference derived from the current identity sets; not a stored session.",
    )
    memories: tuple[conversation_memory.Memory, ...] = pydantic.Field(
        default=(),
        description="Conversation memories recalled for this relationship before the completed turn.",
    )

    @pydantic.field_validator("source_ids", "target_ids", mode="before")
    @classmethod
    def normalize_ids(cls, raw_value: object, info: pydantic.ValidationInfo) -> IdentitySet:
        field_name = info.field_name or "identity set"
        return value.normalize_identity_set(
            raw_value,
            field_name=field_name,
            allow_empty=field_name == "target_ids",
        )

    @pydantic.field_serializer("source_ids", "target_ids", when_used="json")
    def serialize_ids(self, identities: IdentitySet) -> list[str | list[float]]:
        return value.identities_json(identities)


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

    input: pydantic.SkipValidation[ConversationInputData]
    stimulus: turn_detector.ParticipationStimulus
    decoded_text: str | None = None
    participation: turn_detector.ParticipantContribution
    participations: tuple[turn_detector.ParticipantContribution, ...] = ()
    session_ref: str = pydantic.Field(min_length=1)
    memories: tuple[conversation_memory.Memory, ...] = ()


def participated_turn_from_response(
    input_data: ConversationInputData,
    response: Response,
) -> ParticipatedTurn:
    """Reconstruct host composition data from Conversation's typed response terminal."""

    if response.source_ids != input_data.source_ids or response.target_ids != input_data.target_ids:
        raise RuntimeError("Conversation response identities do not match the input identities.")
    content = response.content
    content_type = response.content_type
    normalized_type = content_type.lower()
    sources = value.sorted_identities(response.source_ids)
    structured = content if isinstance(content, dict) else None
    readable = response.decoded_text

    def stimulus_for(source_id: IdentityValue) -> turn_detector.ParticipationStimulus:
        if normalized_type.startswith("text/") and isinstance(content, str) and content.strip():
            return turn_detector.TextStimulus(source_participant_ref=source_id, content=content)
        if normalized_type.startswith("audio/") and isinstance(content, bytes):
            return turn_detector.AudioStimulus(source_participant_ref=source_id, content=content)
        return turn_detector.ContentStimulus(
            source_participant_ref=source_id,
            content=content,
            content_type=content_type,
        )

    if not sources:
        raise RuntimeError("Conversation response contains no source identities.")
    stimuli = tuple(stimulus_for(source_id) for source_id in sources)
    contributions = tuple(
        turn_detector.ParticipantContribution(
            conversation_ref=response.session_ref,
            participant_ref=source_id,
            perception=turn_detector.Perception(
                source_participant_ref=source_id,
                modality=(
                    "text"
                    if normalized_type.startswith("text/")
                    else "audio"
                    if normalized_type.startswith("audio/")
                    else "multimodal"
                ),
                readable=readable,
                speech=content if normalized_type.startswith("audio/") and isinstance(content, bytes) else None,
                structured=structured,
                content=content,
            ),
        )
        for source_id in sources
    )
    return ParticipatedTurn(
        input=input_data,
        stimulus=stimuli[0],
        decoded_text=readable,
        participation=contributions[0],
        participations=contributions,
        session_ref=response.session_ref,
        memories=response.memories,
    )


InputEvent = hsm.Event[ConversationInputData](
    name="bot.ability.conversation.input",
    schema=ConversationInputData,
)
OutputEvent = hsm.Event[Response](
    name="bot.ability.conversation.output",
    schema=Response,
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

    input: pydantic.SkipValidation[ConversationInputData]


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

    input: pydantic.SkipValidation[ConversationInputData]
    operation_id: str = pydantic.Field(min_length=1)
    provenance: _TurnOperationProvenance
    turns: tuple[turn_detector.TurnCompleteData, ...] = pydantic.Field(min_length=1)
    memories: tuple[conversation_memory.Memory, ...] = ()


class _TurnFailedData(pydantic.BaseModel):
    """Failure correlated to the Conversation operation that produced it."""

    model_config: typ.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    input: pydantic.SkipValidation[ConversationInputData]
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


def _stable_context_ref(data: ConversationInputData) -> str:
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


def _content_for_detector(
    data: ConversationInputData,
    source_id: IdentityValue,
) -> turn_detector.ParticipationStimulus:
    """Preserve the declared modality while adapting to the detector boundary."""

    if data.content_type == "text/plain" and isinstance(data.content, str):
        return turn_detector.TextStimulus(source_participant_ref=source_id, content=data.content)
    if data.content_type == "audio/raw" and isinstance(data.content, bytes):
        return turn_detector.AudioStimulus(source_participant_ref=source_id, content=data.content)
    return turn_detector.ContentStimulus(
        source_participant_ref=source_id,
        content=data.content,
        content_type=data.content_type,
    )


async def _run_memory_operation(
    ctx: hsm.Context,
    *,
    owner: hsm.Instance,
    memory: ability.Ability[typ.Any, typ.Any],
    operation_id: str,
    operation: typ.Literal["recall", "remember"],
    memory_input: memory_ability.InputData,
    metadata: collections.abc.Mapping[str, object],
    timeout: float,
) -> memory_ability.OutputData:
    request_id = f"{operation_id}:memory:{operation}"
    try:
        terminal = await asyncio.wait_for(
            ability.Ability.await_child_terminal(
                ctx,
                owner=owner,
                child=memory,
                operation_id=request_id,
                input=memory_input,
                metadata=metadata,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError as error:
        raise RuntimeError(f"Conversation memory {operation} timed out.") from error

    if terminal.id != request_id or terminal.source != hsm.id(memory):
        raise RuntimeError(
            "Conversation memory returned an unrelated terminal: "
            f"id={terminal.id!r} source={terminal.source!r} "
            f"expected_id={request_id!r} expected_source={hsm.id(memory)!r}"
        )
    if isinstance(terminal.data, ability.FailureData):
        raise RuntimeError(terminal.data.message)
    if not isinstance(terminal.data, memory_ability.OutputData):
        raise TypeError(f"Conversation memory {operation} produced an invalid output.")
    return terminal.data


def _terminal_output(
    ctx: hsm.Context,
    instance: "Conversation",
    event: hsm.Event[typ.Any],
    output: Response,
) -> None:
    terminal = dataclasses.replace(
        _with_operation(instance.output_event.with_data(output), event),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))


def _terminal_failure(
    ctx: hsm.Context,
    instance: "Conversation",
    event: hsm.Event[typ.Any],
    failure: FailureData,
) -> None:
    terminal = dataclasses.replace(
        _with_operation(instance.failed_event.with_data(failure), event),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))


class Conversation(ability.Ability[ConversationInputData, Response]):
    """Infer relationships and own one detector per participant track.

    One Conversation instance is the topology-owned ambient stream boundary:
    targetless inputs in that instance share its ambient relationship. Separate
    rooms require separate Conversation instances because the input has no
    conversation_ref.
    """

    input_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = ConversationInputData
    output_data_type: typ.ClassVar[type[object] | tuple[type[object], ...] | None] = Response
    input_event: typ.ClassVar[hsm.Event[ConversationInputData]] = InputEvent
    output_event: typ.ClassVar[hsm.Event[Response]] = OutputEvent
    failed_event: typ.ClassVar[hsm.Event[FailureData]] = FailedEvent
    snapshot_request_event: typ.ClassVar[hsm.Event[SnapshotRequest]] = SnapshotRequestEvent
    snapshot_output_event: typ.ClassVar[hsm.Event[Snapshot]] = SnapshotOutputEvent
    _composite_attachment_lifecycle: typ.ClassVar[bool] = False
    submodel: typ.ClassVar[hsm.Model | None] = hsm.define(
        "Conversation",
        hsm.initial(hsm.target("/Conversation/silent")),
        hsm.state("silent"),
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

            def factory(session_ref: str, track_ref: TrackRef) -> turn_detector_module.TurnDetector:
                return template_detector.clone_for(participant_ref=track_ref, conversation_ref=session_ref)

            self._turn_detector_factory = factory
        elif turn_detector_factory is not None:
            self._turn_detector_factory = turn_detector_factory
        else:
            self._turn_detector_factory = lambda session_ref, track_ref: turn_detector_module.TurnDetector(
                participant_ref=track_ref,
                conversation_ref=session_ref,
            )
        self._encoding = encoding_module.Encoding(encoder=encoder) if encoder is not None else encoding
        self._typing = typing
        self._memory = memory
        super().__init__()
        self._detectors = {}
        self._relationships = []
        self._similarity_threshold = similarity_threshold
        self._similarity_margin = similarity_margin

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

    def _relationship_for_input(self, data: ConversationInputData) -> _RelationshipProfile:
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
                    and (
                        value.cosine_similarity(target_id, participant.source_id) or 0.0
                    )
                    >= self._similarity_threshold
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
            score = direct_score if reverse_score is None else reverse_score if direct_score is None else max(
                direct_score, reverse_score
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
            participant
            for participant in relationship.participants
            if participant.track_ref not in used_tracks
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
        return isinstance(event.data, ConversationInputData)

    @staticmethod
    def _has_speech_input(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx, instance
        if not isinstance(event.data, cognition.InputData):
            return False
        stimulus = event.data.stimulus
        return (
            isinstance(stimulus, hsm.Event)
            and isinstance(stimulus.data, listening.SpeechData)
            and bool(stimulus.data.source_ids)
        )

    @staticmethod
    def _has_speech_without_identity(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx, instance
        if not isinstance(event.data, cognition.InputData):
            return False
        stimulus = event.data.stimulus
        return (
            isinstance(stimulus, hsm.Event)
            and isinstance(stimulus.data, listening.SpeechData)
            and not stimulus.data.source_ids
        )

    @staticmethod
    def _has_snapshot_request(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, SnapshotRequest)

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
        data = event.data
        assert isinstance(data, ConversationInputData)
        operation_id = _operation_id(event)
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _InputWorkEvent.with_data(_InputWorkData(input=data)),
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
                participant.detector = (
                    None if (relationship_ref, participant.track_ref) in stopped_keys else detector
                )
            for participant in relationship.participants:
                if (relationship_ref, participant.track_ref) in stopped_keys:
                    participant.detector = None
            relationship.participants[:] = (
                list(original_participants)
                if remove_relationship
                else list(relationship.participants)
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
                    owner=instance,
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
                    await hsm.started(instance.context(), detector, detector.owned_model)
                    # The detector is owned by Conversation's context.  It is
                    # intentionally not attached back to Conversation: an
                    # Ability already has one owner, and this child is
                    # coordinated through its typed terminal waiter.
                    ready_operation_id = f"{operation_id}:ready:{participant.track_ref}"
                    ready_waiter: asyncio.Future[hsm.Event[typ.Any]] = asyncio.get_running_loop().create_future()
                    detector.register_terminal_waiter(ready_operation_id, ready_waiter)
                    readiness_error: BaseException | None = None
                    ready: hsm.Event[typ.Any] | None = None
                    try:
                        try:
                            await hsm.dispatch(
                                ctx,
                                detector,
                                dataclasses.replace(
                                    turn_detector.TurnDetectorReadyRequestEvent.with_data(
                                        turn_detector.TurnDetectorReadyRequestData(
                                            participant_ref=participant.track_ref,
                                            conversation_ref=relationship_ref,
                                        )
                                    ),
                                    id=ready_operation_id,
                                    source=hsm.id(instance),
                                    target=hsm.id(detector),
                                    metadata=dict(event.metadata),
                                ),
                            )
                            ready = await asyncio.wait_for(ready_waiter, timeout=_TURN_TIMEOUT_SECONDS)
                            if (
                                ready.name != turn_detector.TurnDetectorReadyEvent.name
                                or not isinstance(ready.data, turn_detector.TurnDetectorReadyData)
                                or ready.id != ready_operation_id
                                or ready.source != hsm.id(detector)
                                or ready.data.participant_ref != participant.track_ref
                                or ready.data.conversation_ref != relationship_ref
                            ):
                                raise RuntimeError("Turn detector did not produce a correlated readiness terminal.")
                        except asyncio.TimeoutError as error:
                            readiness_error = RuntimeError(
                                f"Turn detector readiness timed out after {_TURN_TIMEOUT_SECONDS:g} seconds."
                            )
                            readiness_error.__cause__ = error
                        except BaseException as error:
                            readiness_error = error
                    finally:
                        try:
                            detector.clear_terminal_waiter(ready_operation_id)
                        except BaseException as error:
                            if readiness_error is None:
                                readiness_error = error
                            else:
                                readiness_error.add_note(f"Failed to clear readiness waiter: {error!r}")
                    if readiness_error is not None:
                        raise readiness_error
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
                waiter: asyncio.Future[hsm.Event[typ.Any]] = asyncio.get_running_loop().create_future()
                active_turn_detector_keys.add(detector_key)
                detector.register_terminal_waiter(operation_id, waiter)
                try:
                    await hsm.dispatch(
                        ctx,
                        detector,
                        dataclasses.replace(
                            turn_detector.TurnStartEvent.with_data(start),
                            id=operation_id,
                            source=hsm.id(instance),
                            target=hsm.id(detector),
                            metadata=dict(event.metadata),
                        ),
                    )
                    await hsm.dispatch(
                        ctx,
                        detector,
                        dataclasses.replace(
                            turn_detector.TurnEndEvent.with_data(
                                turn_detector.TurnEndData(
                                    conversation_ref=relationship_ref,
                                    turn_ref=turn_ref,
                                    source_participant_ref=source_id,
                                )
                            ),
                            id=operation_id,
                            source=hsm.id(instance),
                            target=hsm.id(detector),
                            metadata=dict(event.metadata),
                        ),
                    )
                    terminal = await asyncio.wait_for(waiter, timeout=_TURN_TIMEOUT_SECONDS)
                except asyncio.TimeoutError as error:
                    detector.clear_terminal_waiter(operation_id)
                    provenance = provenance.model_copy(
                        update={
                            "detector_ids": provenance.detector_ids - frozenset({participant.track_ref}),
                            "failure_track_ref": participant.track_ref,
                        }
                    )
                    raise RuntimeError(
                        f"Turn detector turn timed out after {_TURN_TIMEOUT_SECONDS:g} seconds."
                    ) from error
                finally:
                    detector.clear_terminal_waiter(operation_id)
                correlated_envelope = terminal.id == operation_id and terminal.source == hsm.id(detector)
                if not correlated_envelope:
                    raise RuntimeError(
                        "Turn detector returned an unrelated terminal: "
                        f"id={terminal.id!r} source={terminal.source!r} name={terminal.name!r} "
                        f"expected_id={operation_id!r} expected_source={hsm.id(detector)!r} "
                        f"data={terminal.data!r}"
                    )
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
                (participant, instance._profile_update(participant, source_id))
                for source_id, participant in detectors
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
                    owner=instance,
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
    def _forward_listening_speech(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        data = event.data
        assert isinstance(data, cognition.InputData)
        stimulus = data.stimulus
        assert isinstance(stimulus, hsm.Event)
        speech = stimulus.data
        assert isinstance(speech, listening.SpeechData)
        if not speech.source_ids:
            raise AssertionError("Conversation speech routing requires Listening to assign source_ids first.")
        input_data = ConversationInputData(
            source_ids=speech.source_ids,
            target_ids=frozenset(),
            content=speech.audio,
            content_type="audio/raw",
        )
        operation_id = stimulus.id or event.id
        if not operation_id:
            raise ValueError("Listening speech forwarding requires an event id.")
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                instance.input_event.with_data_and_id(input_data, operation_id), metadata=dict(event.metadata)
            ),
        )

    @staticmethod
    def _fail_listening_speech_without_identity(
        ctx: hsm.Context,
        instance: "Conversation",
        event: hsm.Event[typ.Any],
    ) -> None:
        _terminal_failure(
            ctx,
            instance,
            event,
            FailureData(
                stage="voice_routing",
                message="Listening speech has no source_ids; configure a voice classifier or diarizer.",
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
            (candidate for candidate in instance._relationships if candidate.session_ref == data.provenance.session_ref),
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
                (relationship_ref, turn.participant_ref) not in instance._detectors
                or turn.conversation_ref != relationship_ref
            ):
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
        _terminal_output(
            ctx,
            instance,
            event,
            Response(
                source_ids=input_data.source_ids,
                target_ids=input_data.target_ids,
                content=input_data.content,
                content_type=input_data.content_type,
                decoded_text=next((turn.text for turn in data.turns if turn.text), None),
                session_ref=data.provenance.session_ref,
                memories=data.memories,
            ),
        )

    @staticmethod
    def _fail_turn(ctx: hsm.Context, instance: "Conversation", event: hsm.Event[typ.Any]) -> None:
        failure = event.data
        assert isinstance(failure, _TurnFailedData)
        relationship = next(
            (candidate for candidate in instance._relationships if candidate.session_ref == failure.provenance.session_ref),
            None,
        )
        if relationship is not None:
            relationship.participants[:] = [
                participant for participant in relationship.participants if participant.detector is not None
            ]
        _terminal_failure(ctx, instance, event, failure.failure)

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
        deferred = (
            InputEvent,
            cognition.InputEvent,
            _TurnOperationCompletedEvent,
            _TurnFailedEvent,
        )
        return hsm.define(
            root_name,
            hsm.initial(hsm.target(f"{root}/silent")),
            hsm.transition(
                hsm.on(SnapshotRequestEvent),
                hsm.guard(cls._has_snapshot_request),
                hsm.effect(cls._snapshot),
            ),
            hsm.state(
                "silent",
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.guard(cls._has_speech_input),
                    hsm.effect(cls._forward_listening_speech),
                ),
                hsm.transition(
                    hsm.on(cognition.InputEvent),
                    hsm.guard(cls._has_speech_without_identity),
                    hsm.effect(cls._fail_listening_speech_without_identity),
                ),
                hsm.transition(
                    hsm.on(InputEvent),
                    hsm.guard(cls._has_input),
                    hsm.effect(cls._queue_input),
                    hsm.target(f"{root}/active"),
                ),
            ),
            hsm.state(
                "active",
                hsm.initial(hsm.target(f"{root}/active/waiting")),
                hsm.transition(
                    hsm.on(_InputCancelledEvent),
                    hsm.guard(cls._has_input_cancelled),
                    hsm.target(f"{root}/silent"),
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
                        hsm.target(f"{root}/silent"),
                    ),
                    hsm.transition(
                        hsm.on(_TurnFailedEvent),
                        hsm.guard(cls._has_failure),
                        hsm.effect(cls._fail_turn),
                        hsm.target(f"{root}/silent"),
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


async def contribute_conversation_input(
    conversation: Conversation,
    input_data: ConversationInputData,
    *,
    ctx: hsm.Context | None = None,
) -> ParticipatedTurn:
    """Dispatch one Conversation input and await its typed contribution terminal."""

    context = conversation.context() if ctx is None else ctx
    operation_id = uuid.uuid4().hex
    result: asyncio.Future[hsm.Event[typ.Any]] = asyncio.get_running_loop().create_future()
    conversation.register_terminal_waiter(operation_id, result)
    try:
        await hsm.dispatch(context, conversation, conversation.input_event.with_data_and_id(input_data, operation_id))
        terminal = await asyncio.wait_for(result, timeout=5.0)
    except asyncio.CancelledError:
        cancellation = dataclasses.replace(
            _InputCancelledEvent.with_data(_InputCancelledData(operation_id=operation_id)),
            id=operation_id,
            source=hsm.id(conversation),
            target=hsm.id(conversation),
        )
        try:
            await asyncio.shield(hsm.dispatch(context, conversation, cancellation))
        except BaseException:
            pass
        raise
    finally:
        conversation.clear_terminal_waiter(operation_id)
    if terminal.name == conversation.failed_event.name:
        raise RuntimeError(f"Conversation failed during input contribution: {terminal.data!r}")
    if terminal.name != conversation.output_event.name or not isinstance(terminal.data, Response):
        raise RuntimeError("Conversation produced no response terminal.")
    return participated_turn_from_response(input_data, terminal.data)


Conversation.submodel = define_conversation_model("Conversation")
Conversation.model = Conversation.define_lifecycle_model("Conversation", Conversation.submodel)


__all__ = [
    "ConversationInputData",
    "Conversation",
    "FailedEvent",
    "FailureData",
    "InputEvent",
    "OutputEvent",
    "ParticipatedTurn",
    "participated_turn_from_response",
    "Response",
    "Snapshot",
    "SnapshotOutputEvent",
    "SnapshotRequest",
    "SnapshotRequestEvent",
    "Stage",
    "TrackRef",
    "TurnDetectorFactory",
    "contribute_conversation_input",
    "conversation_event_with_operation",
    "define_conversation_model",
]
