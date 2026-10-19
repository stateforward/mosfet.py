"""One provider-agnostic identity contract for voices and faces.

A perceiving ability (Listening for voices, Seeing for faces) owns one ``Registry`` per kind. The
registry is the HSM wrapper around an injected ``Identifier``: it serves ``IdentifyEvent``,
``EnrollEvent``, ``ForgetEvent`` and ``LinkEvent`` one at a time, replies to each request's source
under the request's id, and prunes ids that are not linked to a person on its own timer. A link
mark (``LinkEvent``) only says whether an id belongs to some person, so retention keeps its print;
who that person is lives in memory, never here. An own id belongs to the bot itself, so a link mark
on it is a ``CONFLICT``. Unknown prints
are enrolled automatically: identifying a print either matches an existing id or mints a new one,
unless the request marks that print match-only (``IdentifyData.mint``), when an unknown print is
left unidentified and nothing is minted.
An enrollment either names the id it refines (declared) or, for the bot's own print, names none:
the print then refines the nearest own id of its kind or mints a new own id, and never matches an
id that is not own.

Prints are biometric (``telemetry.capture.Biometric``): full capture never writes their vectors,
and no failure message carries a payload. Telemetry (every attribute is a closed set from code;
ids are never attributes):

* ``bot.identification.count`` — ``bot.identification.kind`` {voice, face} × ``bot.outcome``
  {recognized, minted, unmatched, failed, skipped, linked, unlinked} × ``bot.failure.kind``
  (``FailureKind`` or ``none``): one point per print identified or enrolled, one per match-only
  print that matched nobody (``unmatched`` with ``none``), one per link mark set (``linked``) or
  cleared (``unlinked``), one per failed request, one per print an identifier could not identify
  (``failed`` with ``none``), and, from the ear, one per voice segment its classifier gave no
  voiceprint (``skipped`` with ``none``, voice only). An enrolled print's point also carries
  ``bot.identification.mode`` {declared, own_matched, own_minted}: at most 33 series (9 plain, 8
  enrolled, 4 link marks, and 12 failures across the six failure kinds).
* ``bot.identification.pruned`` — ids pruned by retention, by kind: 2 series.
* ``bot.identification.prune.failed`` — prunes that failed or timed out, by kind ×
  ``bot.failure.kind`` (``FailureKind``): at most 8 series.
* ``bot.identification.forgotten`` — ids erased by a ``ForgetEvent``, by kind: 2 series.
* Spans ``bot.identification.{identify,enroll,forget,link,prune}`` record a failure with the same
  ``FailureKind`` string the reply carries; an enrollment's span carries its ``bot.identification.mode``.
"""

import abc
import collections.abc
import dataclasses
import datetime
import enum
import math
import typing
import uuid

import hsm
import mosfet
import pydantic
import pydantic.dataclasses
from opentelemetry import metrics

from mosfet.telemetry import capture, span

from .. import ability

Kind: typing.TypeAlias = typing.Literal["voice", "face"]
"""The ref namespace an identity belongs to."""

Id: typing.TypeAlias = typing.Annotated[str, pydantic.StringConstraints(pattern=r"^[a-z0-9][a-z0-9:._-]{0,127}$")]
"""An identity id: minted ids (own ones included) are ``uuid4().hex``; a producer may declare any other."""

Ref: typing.TypeAlias = typing.Annotated[
    str, pydantic.StringConstraints(pattern=r"^(voice|face):[a-z0-9][a-z0-9:._-]{0,127}$")
]
"""A participant ref for an identity, ``<kind>:<id>``; at most 134 characters."""

_Mode: typing.TypeAlias = typing.Literal["declared", "own_matched", "own_minted"]
"""How an enrollment found its id: the one it declared, the nearest own id, or an own id minted now."""

_MAX_DIMENSIONS: typing.Final = 4096
_MAX_PRINTS: typing.Final = 16

_REF = pydantic.TypeAdapter[str](Ref)
_DEFAULT_PRUNE_EVERY: typing.Final = datetime.timedelta(hours=1)
_DEFAULT_TIMEOUT: typing.Final = datetime.timedelta(seconds=10)
_DEFAULT_PRUNE_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_SCOPE: typing.Final = "bot.abilities.identification"
_METER = metrics.get_meter(_SCOPE)
_COUNT = _METER.create_counter(
    "bot.identification.count",
    unit="{print}",
    description="Prints identified or enrolled, by kind and outcome; failed requests count once.",
)
_PRUNED = _METER.create_counter(
    "bot.identification.pruned",
    unit="{id}",
    description="Identity ids pruned by retention, by kind.",
)
_PRUNE_FAILED = _METER.create_counter(
    "bot.identification.prune.failed",
    unit="{prune}",
    description="Retention prunes that failed or timed out, by kind and failure kind.",
)
_FORGOTTEN = _METER.create_counter(
    "bot.identification.forgotten",
    unit="{id}",
    description="Identity ids erased by a forget request, by kind.",
)


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def ref(kind: Kind, id: Id) -> Ref:
    """The participant ref ``<kind>:<id>`` for an identity.

    Raises:
        pydantic.ValidationError: If ``kind`` or ``id`` is not a valid kind or id.
    """

    return _REF.validate_python(f"{kind}:{id}")


def participant(kind: Kind, id: Id, *, own: bool = False) -> mosfet.Participant:
    """The source participant a producer stamps for an identity it recognized or declared.

    ``own`` is the producer's own knowledge that the identity is the bot's (``Recognition.own``, or
    the voice the bot speaks with); nobody else infers it.
    """

    return mosfet.Participant(ref=ref(kind, id), role=mosfet.ParticipantRole.SOURCE, own=own)


class Print(capture.Biometric):
    """One biometric print (a voiceprint or a face print) in one model's vector space.

    The vector is validated (non-empty, finite) and stored L2-normalized, so stores compare prints
    by dot product. Full capture renders a print as ``<biometric:Print>``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "A biometric print: an embedding vector of a voice or a face in one model's space.",
            "examples": [{"vector": [0.6, 0.8], "model": "pyannote/wespeaker-voxceleb-resnet34-LM"}],
        },
    )

    vector: tuple[pydantic.FiniteFloat, ...] = pydantic.Field(
        min_length=1,
        max_length=_MAX_DIMENSIONS,
        description="The embedding vector; validated finite and non-zero, then L2-normalized.",
        examples=[[0.6, 0.8]],
    )
    model: str = pydantic.Field(
        min_length=1,
        max_length=128,
        description="The model that produced the vector; stores keep one vector space per model.",
        examples=["pyannote/wespeaker-voxceleb-resnet34-LM"],
    )

    @pydantic.field_validator("vector")
    @classmethod
    def _normalized(cls, vector: tuple[float, ...]) -> tuple[float, ...]:
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError("a print vector cannot be all zeros")
        return tuple(value / norm for value in vector)


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
class Recognition:
    """Who a print is: an existing id it matched, or the id minted for it now.

    Shown to a model, a recognition is its id, ``recognized`` and ``own``; the similarity score is not.
    """

    __model_facing_excluded_fields__: typing.ClassVar[frozenset[str]] = frozenset({"similarity"})

    id: Id
    recognized: bool
    """True when the print matched an existing id; false when the id was minted now."""
    similarity: float | None = pydantic.Field(ge=-1.0, le=1.0)
    """Cosine similarity to the matched id (or the best candidate when minted); None when nothing was compared."""
    own: bool
    """True for an id its producer declared as the bot's own (for example the bot's own voice)."""


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
class MatchPolicy:
    """How close a print must be to an existing id; every number is model-specific, so none has a default.

    The best candidate must reach ``threshold`` and beat the runner-up by ``margin``; a match at
    or above ``refine_threshold`` also refines the id's running mean, weighted by at most
    ``refine_weight_cap`` earlier prints.
    """

    threshold: float = pydantic.Field(ge=-1.0, le=1.0)
    margin: float = pydantic.Field(ge=0.0, le=2.0)
    refine_threshold: float = pydantic.Field(ge=-1.0, le=1.0)
    refine_weight_cap: int = pydantic.Field(default=50, ge=1)

    @pydantic.model_validator(mode="after")
    def _ordered(self) -> typing.Self:
        if self.refine_threshold < self.threshold:
            raise ValueError("refine_threshold cannot be below threshold")
        return self


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
class Neighbor:
    """One stored id a store compared a print with."""

    id: Id
    similarity: float = pydantic.Field(ge=-1.0, le=1.0)
    own: bool


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
class Match:
    """The print is this id; its vector stays as it is."""

    id: Id
    similarity: float = pydantic.Field(ge=-1.0, le=1.0)
    own: bool


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
class Refine:
    """The print is this id and close enough to refine the id's running mean."""

    id: Id
    similarity: float = pydantic.Field(ge=-1.0, le=1.0)
    own: bool


@pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
class Mint:
    """The print is nobody known: the store mints a new id; ``similarity`` is the best candidate's, if any."""

    similarity: float | None = pydantic.Field(ge=-1.0, le=1.0)


Decision: typing.TypeAlias = Match | Refine | Mint


def decide(neighbors: collections.abc.Sequence[Neighbor], policy: MatchPolicy) -> Decision:
    """The one matching algorithm every store runs inside its own transaction; pure.

    The best neighbor must reach ``policy.threshold`` and beat the runner-up by ``policy.margin``;
    one at or above ``policy.refine_threshold`` is a ``Refine``. Anything else is a ``Mint``.
    """

    if not neighbors:
        return Mint(similarity=None)
    ranked = sorted(neighbors, key=lambda neighbor: neighbor.similarity, reverse=True)
    best = ranked[0]
    if best.similarity < policy.threshold:
        return Mint(similarity=best.similarity)
    if len(ranked) > 1 and best.similarity - ranked[1].similarity < policy.margin:
        return Mint(similarity=best.similarity)
    if best.similarity >= policy.refine_threshold:
        return Refine(id=best.id, similarity=best.similarity, own=best.own)
    return Match(id=best.id, similarity=best.similarity, own=best.own)


class FailureKind(enum.StrEnum):
    """Why an identification request failed; values match ``telemetry/FAILURE_KINDS.md``."""

    INVALID_INPUT = "invalid_input"
    """The request is malformed, or a print's dimensions differ from its model's recorded space."""
    UNAVAILABLE = "unavailable"
    """The registry is not serving: it is not attached."""
    STORE_FAILED = "store_failed"
    """The store (or remote identifier) failed; a store transaction rolled back."""
    TIMEOUT = "timeout"
    """The request did not complete within the registry's timeout."""
    NOT_FOUND = "not_found"
    """A link mark named an id the registry's kind does not have."""
    CONFLICT = "conflict"
    """A link mark named an own id: it is the bot's own, so it belongs to the bot and to no other person."""


class FailureData(ability.FailureData):
    """An identification request failed; ``kind`` says how the requester should respond.

    The message never carries a payload: no print, vector, or validation error text.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "An identification request failed; kind says how the requester should respond.",
            "examples": [{"message": "the identity store failed", "kind": "store_failed"}],
        },
    )

    kind: FailureKind = pydantic.Field(
        description="Closed failure kind; the same value is reported on spans and metrics.",
        examples=["store_failed"],
    )


class StoreError(Exception):
    """A store (or identifier) call failed with a closed ``FailureKind``; nothing was changed.

    The message is written by the store and never carries a print or a vector.
    """

    kind: FailureKind

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class Identifier(abc.ABC):
    """A provider-agnostic, non-HSM capacity to tell who a print is.

    ``Matcher`` is the store-backed implementation; a remote service implements this directly.
    Failures raise ``StoreError``.
    """

    @abc.abstractmethod
    async def identify(
        self, kind: Kind, prints: tuple[Print, ...], mint: tuple[bool, ...]
    ) -> tuple[Recognition | None, ...]:
        """One result per print, in order; ``None`` when this identifier cannot identify that print.

        ``mint`` has one flag per print: an unknown print whose flag is set is enrolled automatically
        under a minted id, and one whose flag is clear (match only) is ``None``, with nothing minted.

        Raises:
            StoreError: For a request invalid as a whole (``INVALID_INPUT``, for example a dimension
                mismatch with the model's space) or a failed store (``STORE_FAILED``).
        """

    @abc.abstractmethod
    async def enroll(self, kind: Kind, id: Id | None, print: Print, own: bool) -> Recognition:
        """Upsert ``id`` with ``print``: insert a new id, or refine an existing one.

        Refining advances ``seen_at`` and the capped ``seen_count`` weight; ``own`` is sticky. An
        existing id answers ``recognized=True``. With ``id=None`` (only with ``own``), the print is
        matched among the own ids of ``kind`` alone under the ``MatchPolicy``: a ``Match`` or
        ``Refine`` refines that id and answers ``recognized=True``; a ``Mint`` inserts a new own id
        (``uuid4().hex``) and answers ``recognized=False`` with the best own candidate's similarity.

        Raises:
            StoreError: ``INVALID_INPUT`` when ``print.model`` differs from the id's model, its
                dimensions differ from the model's space, or ``id`` is ``None`` without ``own``;
                ``STORE_FAILED`` for a failed store.
        """

    @abc.abstractmethod
    async def forget(self, kind: Kind, id: Id) -> bool:
        """Erase ``id`` and its print; whether it existed.

        Raises:
            StoreError: ``STORE_FAILED`` for a failed store.
        """

    @abc.abstractmethod
    async def link(self, kind: Kind, id: Id, linked: bool) -> None:
        """Mark whether ``id`` belongs to a person; a linked id is never pruned.

        Setting a mark that is already set changes nothing. The mark never changes the print,
        ``seen_at`` or ``seen_count``.

        Raises:
            StoreError: ``NOT_FOUND`` when ``kind`` has no ``id``; ``CONFLICT`` when ``id`` is own,
                since an own id is the bot's; ``STORE_FAILED`` for a failed store.
        """

    @abc.abstractmethod
    async def prune(self, kind: Kind, *, unseen_before: datetime.datetime, max_ids: int) -> int:
        """Remove unlinked, non-own ids last seen before ``unseen_before``, then the least recently seen
        of them until at most ``max_ids`` remain; how many were removed.

        Raises:
            StoreError: ``STORE_FAILED`` for a failed store.
        """


class IdentifyData(pydantic.BaseModel):
    """Who are these prints? Each is matched to an existing id or, unless it is match-only, enrolled under a new one."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Prints to identify; each is matched to a known id or, unless it is match-only, enrolled under a "
                "new one."
            ),
            "examples": [
                {"prints": [{"vector": [0.6, 0.8], "model": "pyannote/wespeaker-voxceleb-resnet34-LM"}]},
                {
                    "prints": [{"vector": [0.6, 0.8], "model": "pyannote/wespeaker-voxceleb-resnet34-LM"}],
                    "mint": [False],
                },
            ],
        },
    )

    prints: tuple[Print, ...] = pydantic.Field(
        min_length=1,
        max_length=_MAX_PRINTS,
        description="The prints to identify, all of the registry's kind.",
    )
    mint: tuple[bool, ...] | None = pydantic.Field(
        default=None,
        description=(
            "One flag per print, in order: true enrolls an unknown print under a new id, false matches only and "
            "leaves an unknown print unidentified. Null enrolls every unknown print."
        ),
        examples=[None, [False], [True, False]],
    )

    @pydantic.model_validator(mode="after")
    def _one_flag_per_print(self) -> typing.Self:
        if self.mint is not None and len(self.mint) != len(self.prints):
            raise ValueError("mint must have one flag per print")
        return self


class IdentifiedData(pydantic.BaseModel):
    """The answer to an ``IdentifyEvent``: one recognition per print, in order."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "One recognition per identified print, in request order.",
            "examples": [
                {
                    "kind": "voice",
                    "recognitions": [{"id": "3f2a9c", "recognized": True, "similarity": 0.82, "own": False}],
                }
            ],
        },
    )

    kind: Kind = pydantic.Field(
        description="The registry's kind; a consumer handed the wrong registry sees the mismatch here.",
        examples=["voice"],
    )
    recognitions: tuple[Recognition | None, ...] = pydantic.Field(
        max_length=_MAX_PRINTS,
        description="One per print in request order; null when the identifier could not identify that print.",
    )


class EnrollData(pydantic.BaseModel):
    """Enroll (or refine) an id with a print: a declared id, or, for the bot's own print, no id at all.

    Without an id the print refines the nearest own id of the registry's kind, or mints a new own
    id; it never matches an id that is not own. The reply names the id either way.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Enroll or refine an id with a print. A null id enrolls the bot's own print: it refines the "
                "nearest own id or mints a new own id, and never matches an id that is not own."
            ),
            "examples": [
                {
                    "id": None,
                    "print": {"vector": [0.6, 0.8], "model": "pyannote/wespeaker-voxceleb-resnet34-LM"},
                    "own": True,
                },
                {
                    "id": "3f2a9c",
                    "print": {"vector": [0.6, 0.8], "model": "pyannote/wespeaker-voxceleb-resnet34-LM"},
                    "own": False,
                },
            ],
        },
    )

    id: Id | None = pydantic.Field(
        description=(
            "The declared id to enroll or refine; null only with own, to match the nearest own id or mint one."
        ),
        examples=[None, "3f2a9c"],
    )
    print: Print = pydantic.Field(description="The print to enroll the id with.")
    own: bool = pydantic.Field(
        description="True when the producer declares the print as the bot's own; sticky once set.",
        examples=[True],
    )

    @pydantic.model_validator(mode="after")
    def _own_when_undeclared(self) -> typing.Self:
        if self.id is None and not self.own:
            raise ValueError("an enrollment without an id must be the bot's own")
        return self


class ForgetData(pydantic.BaseModel):
    """Erase an id: its print and its identity row; memory keeps the opaque ref."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Erase an identity id and its biometric print.",
            "examples": [{"id": "3f2a9c"}],
        },
    )

    id: Id = pydantic.Field(description="The id to erase.", examples=["3f2a9c"])


class ForgottenData(pydantic.BaseModel):
    """The answer to a ``ForgetEvent``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Whether the id existed and was erased.",
            "examples": [{"id": "3f2a9c", "forgotten": True}],
        },
    )

    id: Id = pydantic.Field(description="The id the request named.", examples=["3f2a9c"])
    forgotten: bool = pydantic.Field(description="True when the id existed and was erased.", examples=[True])


class LinkData(pydantic.BaseModel):
    """Mark whether an id belongs to a person, so retention keeps (or may again prune) its print.

    The mark carries no person: memory holds which person an id's ref belongs to. An own id is the
    bot's, so a mark on it is refused as a ``CONFLICT``.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Mark whether an identity id belongs to a person; a linked id is never pruned.",
            "examples": [{"id": "3f2a9c", "linked": True}, {"id": "3f2a9c", "linked": False}],
        },
    )

    id: Id = pydantic.Field(description="The id to mark; it must exist and must not be own.", examples=["3f2a9c"])
    linked: bool = pydantic.Field(
        description="True when the id now belongs to a person; false when it no longer does.", examples=[True]
    )


class LinkedData(pydantic.BaseModel):
    """The answer to a ``LinkEvent``: the id's mark as it now stands."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "The id's link mark as it now stands.",
            "examples": [{"id": "3f2a9c", "linked": True}],
        },
    )

    id: Id = pydantic.Field(description="The id the request named.", examples=["3f2a9c"])
    linked: bool = pydantic.Field(description="Whether the id is now marked as a person's.", examples=[True])


IdentifyEvent = hsm.Event[IdentifyData](name="bot.ability.identification.identify", schema=IdentifyData)
IdentifiedEvent = hsm.Event[IdentifiedData](name="bot.ability.identification.identified", schema=IdentifiedData)
EnrollEvent = hsm.Event[EnrollData](name="bot.ability.identification.enroll", schema=EnrollData)
EnrolledEvent = hsm.Event[Recognition](name="bot.ability.identification.enrolled", schema=Recognition)
ForgetEvent = hsm.Event[ForgetData](name="bot.ability.identification.forget", schema=ForgetData)
ForgottenEvent = hsm.Event[ForgottenData](name="bot.ability.identification.forgotten", schema=ForgottenData)
LinkEvent = hsm.Event[LinkData](name="bot.ability.identification.link", schema=LinkData)
LinkedEvent = hsm.Event[LinkedData](name="bot.ability.identification.linked", schema=LinkedData)
FailedEvent = hsm.Event[FailureData](name="bot.ability.identification.failed", schema=FailureData)

# The requests a ``Registry`` answers, each with one reply or one ``FailedEvent`` under its id.
_REQUESTS: typing.Final[tuple[hsm.Event[typing.Any], ...]] = (IdentifyEvent, EnrollEvent, ForgetEvent, LinkEvent)

_Operation: typing.TypeAlias = typing.Literal["identify", "enroll", "forget", "link"]
_Reply: typing.TypeAlias = IdentifiedData | Recognition | ForgottenData | LinkedData


class _DoneData(pydantic.BaseModel):
    """The in-flight request's result, carried back to the registry by its own activity."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    reply: _Reply
    mode: _Mode | None = None
    # An identify request's flags, one per print: an unknown match-only print is unmatched, not failed.
    mint: tuple[bool, ...] = ()


class _PrunedData(pydantic.BaseModel):
    """One prune's outcome; ``due`` is the deadline it ran for, so a late one correlates to nothing."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    due: datetime.datetime
    removed: int = 0
    failure: FailureKind | None = None


_DoneEvent = hsm.Event[_DoneData](
    name="bot.ability.identification.done", kind=hsm.CompletionEventKind, schema=_DoneData
)
_FailedEvent = hsm.Event[FailureData](
    name="bot.ability.identification.request.failed", kind=hsm.ErrorEventKind, schema=FailureData
)
_PrunedEvent = hsm.Event[_PrunedData](
    name="bot.ability.identification.pruned", kind=hsm.CompletionEventKind, schema=_PrunedData
)
_PruneFailedEvent = hsm.Event[_PrunedData](
    name="bot.ability.identification.prune.failed", kind=hsm.ErrorEventKind, schema=_PrunedData
)

_REQUEST_DATA: typing.Final[dict[_Operation, type[pydantic.BaseModel]]] = {
    "identify": IdentifyData,
    "enroll": EnrollData,
    "forget": ForgetData,
    "link": LinkData,
}


@dataclasses.dataclass(frozen=True, slots=True)
class _InFlight:
    """The address of the one request in flight: never its prints or its result."""

    id: str
    requester: str
    operation: _Operation
    metadata: dict[str, typing.Any]


class Registry(ability.Ability[IdentifyData, IdentifiedData]):
    """The HSM wrapper around one ``Identifier`` for one ``kind``, owned by the perceiving ability.

    Requests (``IdentifyEvent``, ``EnrollEvent``, ``ForgetEvent``, ``LinkEvent``) are served one at
    a time and answered with one reply, or one ``FailedEvent``, sent to the request's source under
    its id. A request that does not finish within ``timeout`` fails with ``TIMEOUT``. Every
    ``prune_every`` the registry prunes ids of its kind not linked to a person and last seen
    more than ``max_unlinked_age`` ago, then the least recently seen until at most ``max_ids``
    remain; the deadline is durable, so steady request traffic never starves it, and one prune is
    bounded by ``prune_timeout``. A detached registry answers every request ``UNAVAILABLE``, the
    one in flight when it detaches included.

    A store call abandoned by a timeout still commits (a thread cannot be cancelled): a late mint
    is an orphan id nobody was told about, which retention prunes.
    """

    input_event: typing.ClassVar[hsm.Event[IdentifyData]] = IdentifyEvent
    output_event: typing.ClassVar[hsm.Event[IdentifiedData]] = IdentifiedEvent
    failed_event: typing.ClassVar[hsm.Event[FailureData]] = FailedEvent
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = IdentifyData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = IdentifiedData
    detached_requests: typing.ClassVar[tuple[hsm.Event[typing.Any], ...]] = _REQUESTS

    _identifier: Identifier
    _kind: Kind
    _max_unlinked_age: datetime.timedelta
    _max_ids: int
    _prune_every: datetime.timedelta
    _timeout: datetime.timedelta
    _prune_timeout: datetime.timedelta
    _now: collections.abc.Callable[[], datetime.datetime]
    # Durable machine data: the retention deadline (re-armed when a prune ends) and the address of
    # the one request in flight (cleared on every reply), so a busy timeout can still reply.
    _next_prune_at: datetime.datetime
    _in_flight: _InFlight | None

    @classmethod
    @typing.override
    def detached_failure(cls, event: hsm.Event[typing.Any]) -> pydantic.BaseModel:
        del event
        return FailureData(kind=FailureKind.UNAVAILABLE, message="identification registry is detached")

    def __init__(
        self,
        *,
        id: ability.AbilityId | None = None,
        clock: hsm.Clock | None = None,
        identifier: Identifier,
        kind: Kind,
        max_unlinked_age: datetime.timedelta,
        max_ids: int,
        prune_every: datetime.timedelta = _DEFAULT_PRUNE_EVERY,
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
        prune_timeout: datetime.timedelta = _DEFAULT_PRUNE_TIMEOUT,
        now: collections.abc.Callable[[], datetime.datetime] = _utc_now,
    ) -> None:
        """Serve ``identifier`` for ``kind``; ``now`` is the timezone-aware UTC wall clock retention reads.

        Raises:
            ValueError: If a duration is not positive or ``max_ids`` is negative.
        """

        durations = (max_unlinked_age, prune_every, timeout, prune_timeout)
        if any(duration <= datetime.timedelta(0) for duration in durations):
            raise ValueError("Registry durations must be positive.")
        if max_ids < 0:
            raise ValueError("Registry max_ids cannot be negative.")
        super().__init__(id=id, clock=clock)
        self._identifier = identifier
        self._kind = kind
        self._max_unlinked_age = max_unlinked_age
        self._max_ids = max_ids
        self._prune_every = prune_every
        self._timeout = timeout
        self._prune_timeout = prune_timeout
        self._now = now
        self._next_prune_at = now() + prune_every
        self._in_flight = None

    def kind(self) -> Kind:
        """The kind of identity this registry serves, fixed when it was built.

        Like :meth:`identity`, it is a construction-time fact a peer may ask for (``people.People`` knows
        which registry a ``voice:`` or ``face:`` ref belongs to by it); it reads no machine state, and
        everything the registry does is still reached only by its request events.
        """

        return self._kind

    # --- replies -------------------------------------------------------------------------------

    @staticmethod
    def _send(ctx: hsm.Context, instance: "Registry", in_flight: _InFlight, reply: hsm.Event[typing.Any]) -> None:
        """Deliver one reply to the requester's address, or to the owner when the request had none."""

        terminal = dataclasses.replace(
            reply,
            id=in_flight.id,
            source=hsm.id(instance),
            target=in_flight.requester,
            metadata=dict(in_flight.metadata),
        )
        if in_flight.requester:
            _ = hsm.dispatch_to(ctx, terminal, in_flight.requester)
            return
        carrier = (
            ability.TerminalErrorEvent if isinstance(reply.data, ability.FailureData) else ability.TerminalOutputEvent
        )
        _ = hsm.dispatch(ctx, instance, carrier.with_data(terminal))

    @staticmethod
    def _fail(ctx: hsm.Context, instance: "Registry", failure: FailureData) -> None:
        """Answer the request in flight with ``failure`` and clear it."""

        in_flight = instance._in_flight
        assert in_flight is not None
        instance._in_flight = None
        Registry._send(ctx, instance, in_flight, FailedEvent.with_data(failure))
        _COUNT.add(
            1,
            {
                "bot.identification.kind": instance._kind,
                "bot.outcome": "failed",
                "bot.failure.kind": failure.kind.value,
            },
        )

    # --- idle ----------------------------------------------------------------------------------

    @staticmethod
    def _arm(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        """Entering service: the first prune is due one ``prune_every`` from now."""

        del ctx, event
        instance._next_prune_at = instance._now() + instance._prune_every

    @staticmethod
    def _remaining(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        """Only the time left to the durable deadline, so re-entering idle never restarts it."""

        del ctx, event
        return max(datetime.timedelta(0), instance._next_prune_at - instance._now())

    @staticmethod
    def _admitting(
        operation: _Operation,
    ) -> collections.abc.Callable[[hsm.Context, "Registry", hsm.Event[typing.Any]], None]:
        """The effect that makes a request of ``operation`` the one in flight (its trigger selects it)."""

        def admit(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
            del ctx
            instance._in_flight = _InFlight(
                # Replies correlate by id (HSM-CORRELATION-001), so a request without one gets one.
                id=event.id or uuid.uuid4().hex,
                requester=event.source if event.source and event.source != hsm.id(instance) else "",
                operation=operation,
                metadata=dict(event.metadata),
            )

        return admit

    # --- busy ----------------------------------------------------------------------------------

    @staticmethod
    async def _serve(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        """Make the one identifier call for the request in flight and report it to this registry."""

        in_flight = instance._in_flight
        assert in_flight is not None
        operation = in_flight.operation
        kind = instance._kind
        with span.operation(
            f"bot.identification.{operation}",
            scope=_SCOPE,
            component="identification",
            stage=operation,
            attributes={"bot.identification.kind": kind},
        ) as active:
            try:
                # The payload's own validation error would echo the vector; only its kind leaves here.
                data = _REQUEST_DATA[operation].model_validate(event.data)
            except pydantic.ValidationError:
                failure = FailureData(kind=FailureKind.INVALID_INPUT, message=f"the {operation} request is invalid")
            else:
                try:
                    mode: _Mode | None = None
                    mint: tuple[bool, ...] = ()
                    if isinstance(data, IdentifyData):
                        mint = (True,) * len(data.prints) if data.mint is None else data.mint
                        recognitions = await instance._identifier.identify(kind, data.prints, mint)
                        if len(recognitions) != len(data.prints):
                            raise StoreError(FailureKind.STORE_FAILED, "the identifier answered the wrong count")
                        reply: _Reply = IdentifiedData(kind=kind, recognitions=recognitions)
                    elif isinstance(data, EnrollData):
                        reply = await instance._identifier.enroll(kind, data.id, data.print, data.own)
                        if data.id is not None:
                            mode = "declared"
                        else:
                            mode = "own_matched" if reply.recognized else "own_minted"
                        active.set_attribute("bot.identification.mode", mode)
                    elif isinstance(data, ForgetData):
                        forgotten = await instance._identifier.forget(kind, data.id)
                        reply = ForgottenData(id=data.id, forgotten=forgotten)
                    else:
                        assert isinstance(data, LinkData)
                        await instance._identifier.link(kind, data.id, data.linked)
                        reply = LinkedData(id=data.id, linked=data.linked)
                except StoreError as error:
                    failure = FailureData(kind=error.kind, message=str(error))
                except Exception as error:
                    # Boundary: an identifier's own error may quote its input, so only its type is kept.
                    failure = FailureData(
                        kind=FailureKind.STORE_FAILED, message=f"the identifier failed: {type(error).__name__}"
                    )
                else:
                    _ = instance.dispatch(
                        ctx, _DoneEvent.with_data_and_id(_DoneData(reply=reply, mode=mode, mint=mint), in_flight.id)
                    )
                    return
            span.record_failure(active, failure.kind.value)
            _ = instance.dispatch(ctx, _FailedEvent.with_data_and_id(failure, in_flight.id))

    @staticmethod
    def _is_current(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return instance._in_flight is not None and event.id == instance._in_flight.id

    @staticmethod
    def _reply_done(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _DoneData)
        in_flight = instance._in_flight
        assert in_flight is not None
        instance._in_flight = None
        reply = data.reply
        kind = instance._kind
        if isinstance(reply, IdentifiedData):
            Registry._send(ctx, instance, in_flight, IdentifiedEvent.with_data(reply))
            recognitions: tuple[Recognition | None, ...] = reply.recognitions
        elif isinstance(reply, Recognition):
            Registry._send(ctx, instance, in_flight, EnrolledEvent.with_data(reply))
            recognitions = (reply,)
        elif isinstance(reply, ForgottenData):
            Registry._send(ctx, instance, in_flight, ForgottenEvent.with_data(reply))
            recognitions = ()
            if reply.forgotten:
                _FORGOTTEN.add(1, {"bot.identification.kind": kind})
        else:
            Registry._send(ctx, instance, in_flight, LinkedEvent.with_data(reply))
            recognitions = ()
            _COUNT.add(
                1,
                {
                    "bot.identification.kind": kind,
                    "bot.outcome": "linked" if reply.linked else "unlinked",
                    "bot.failure.kind": "none",
                },
            )
        for index, recognition in enumerate(recognitions):
            if recognition is None:
                outcome = "failed" if index >= len(data.mint) or data.mint[index] else "unmatched"
            else:
                outcome = "recognized" if recognition.recognized else "minted"
            attributes = {"bot.identification.kind": kind, "bot.outcome": outcome, "bot.failure.kind": "none"}
            if data.mode is not None:
                attributes["bot.identification.mode"] = data.mode
            _COUNT.add(1, attributes)

    @staticmethod
    def _reply_failed(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        failure = event.data
        assert isinstance(failure, FailureData)
        Registry._fail(ctx, instance, failure)

    @staticmethod
    def _timeout_delay(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._timeout

    @staticmethod
    def _reply_timeout(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        del event
        failure = FailureData(
            kind=FailureKind.TIMEOUT,
            message=f"identification did not complete within {instance._timeout.total_seconds():g}s",
        )
        Registry._fail(ctx, instance, failure)

    @staticmethod
    def _is_prune_due(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._now() >= instance._next_prune_at

    # --- pruning -------------------------------------------------------------------------------

    @staticmethod
    async def _prune(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        """Run one retention prune and report it to this registry, correlated by its deadline."""

        del event
        due = instance._next_prune_at
        with span.operation(
            "bot.identification.prune",
            scope=_SCOPE,
            component="identification",
            stage="prune",
            attributes={"bot.identification.kind": instance._kind},
        ) as active:
            try:
                removed = await instance._identifier.prune(
                    instance._kind,
                    unseen_before=instance._now() - instance._max_unlinked_age,
                    max_ids=instance._max_ids,
                )
            except StoreError as error:
                failure = error.kind
            except Exception:
                # Boundary: any other identifier failure is a failed prune; requests are unaffected.
                failure = FailureKind.STORE_FAILED
            else:
                _ = instance.dispatch(ctx, _PrunedEvent.with_data(_PrunedData(due=due, removed=removed)))
                return
            span.record_failure(active, failure.value)
            _ = instance.dispatch(ctx, _PruneFailedEvent.with_data(_PrunedData(due=due, failure=failure)))

    @staticmethod
    def _is_this_prune(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        return isinstance(event.data, _PrunedData) and event.data.due == instance._next_prune_at

    @staticmethod
    def _count_pruned(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _PrunedData)
        if data.failure is None:
            _PRUNED.add(data.removed, {"bot.identification.kind": instance._kind})
            return
        _PRUNE_FAILED.add(1, {"bot.identification.kind": instance._kind, "bot.failure.kind": data.failure.value})

    @staticmethod
    def _prune_timeout_delay(
        ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._prune_timeout

    @staticmethod
    def _count_prune_timeout(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _PRUNE_FAILED.add(1, {"bot.identification.kind": instance._kind, "bot.failure.kind": FailureKind.TIMEOUT.value})

    @staticmethod
    def _rearm(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        """Leaving pruning, however it ended: the next prune is one ``prune_every`` from now."""

        del ctx, event
        instance._next_prune_at = instance._now() + instance._prune_every

    @staticmethod
    def _abandon(ctx: hsm.Context, instance: "Registry", event: hsm.Event[typing.Any]) -> None:
        """Leaving service (detach or stop): a request still in flight is answered ``UNAVAILABLE``.

        Its activity is already cancelled, so nothing else would answer it before its timeout.
        """

        if instance._in_flight is None:
            return
        failure = Registry.detached_failure(event)
        assert isinstance(failure, FailureData)
        Registry._fail(ctx, instance, failure)

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "IdentificationRegistry",
        hsm.initial(hsm.target("/IdentificationRegistry/serving")),
        hsm.state(
            "serving",
            hsm.initial(hsm.target("/IdentificationRegistry/serving/idle"), hsm.effect(_arm)),
            hsm.exit(_abandon),
            hsm.state(
                "idle",
                hsm.transition(
                    hsm.on(IdentifyEvent),
                    hsm.effect(_admitting("identify")),
                    hsm.target("/IdentificationRegistry/serving/busy"),
                ),
                hsm.transition(
                    hsm.on(EnrollEvent),
                    hsm.effect(_admitting("enroll")),
                    hsm.target("/IdentificationRegistry/serving/busy"),
                ),
                hsm.transition(
                    hsm.on(ForgetEvent),
                    hsm.effect(_admitting("forget")),
                    hsm.target("/IdentificationRegistry/serving/busy"),
                ),
                hsm.transition(
                    hsm.on(LinkEvent),
                    hsm.effect(_admitting("link")),
                    hsm.target("/IdentificationRegistry/serving/busy"),
                ),
                hsm.transition(hsm.after(_remaining), hsm.target("/IdentificationRegistry/serving/pruning")),
            ),
            hsm.state(
                "busy",
                hsm.defer(*_REQUESTS),
                hsm.activity(_serve),
                hsm.transition(
                    hsm.on(_DoneEvent),
                    hsm.guard(_is_current),
                    hsm.effect(_reply_done),
                    hsm.target("/IdentificationRegistry/serving/after_busy"),
                ),
                hsm.transition(
                    hsm.on(_FailedEvent),
                    hsm.guard(_is_current),
                    hsm.effect(_reply_failed),
                    hsm.target("/IdentificationRegistry/serving/after_busy"),
                ),
                hsm.transition(
                    hsm.after(_timeout_delay),
                    hsm.effect(_reply_timeout),
                    hsm.target("/IdentificationRegistry/serving/after_busy"),
                ),
            ),
            hsm.choice(
                "after_busy",
                hsm.transition(hsm.guard(_is_prune_due), hsm.target("/IdentificationRegistry/serving/pruning")),
                hsm.transition(hsm.target("/IdentificationRegistry/serving/idle")),
            ),
            hsm.state(
                "pruning",
                hsm.defer(*_REQUESTS),
                hsm.activity(_prune),
                hsm.exit(_rearm),
                hsm.transition(
                    hsm.on(_PrunedEvent, _PruneFailedEvent),
                    hsm.guard(_is_this_prune),
                    hsm.effect(_count_pruned),
                    hsm.target("/IdentificationRegistry/serving/idle"),
                ),
                hsm.transition(
                    hsm.after(_prune_timeout_delay),
                    hsm.effect(_count_prune_timeout),
                    hsm.target("/IdentificationRegistry/serving/idle"),
                ),
            ),
        ),
    )


__all__ = [
    "Decision",
    "EnrollData",
    "EnrollEvent",
    "EnrolledEvent",
    "FailedEvent",
    "FailureData",
    "FailureKind",
    "ForgetData",
    "ForgetEvent",
    "ForgottenData",
    "ForgottenEvent",
    "Id",
    "IdentifiedData",
    "IdentifiedEvent",
    "Identifier",
    "IdentifyData",
    "IdentifyEvent",
    "Kind",
    "LinkData",
    "LinkEvent",
    "LinkedData",
    "LinkedEvent",
    "Match",
    "MatchPolicy",
    "Mint",
    "Neighbor",
    "Print",
    "Recognition",
    "Ref",
    "Refine",
    "Registry",
    "StoreError",
    "decide",
    "participant",
    "ref",
]
