"""Seeing: the eye. Holds what is in view, hands on measured change, and looks when asked.

An eye does not decide what a sight is for. It holds the newest view of each source it has, asks
its one injected change measure how much a view changed since the last view of that source it
handed on, and hands on a view that changed past a threshold. Everything model- or device-specific
is a plain collaborator injected at construction (``vision.change.ChangeMeasure``): Seeing never
names a camera, a model or a provider, and it never substitutes another measure when its own one
fails.

Rates are bounded in topology, whatever the source's frame rate:

* at most one measurement per ``measure_every`` (frames in between only replace the held view);
* at most one sight per ``settle_after``, plus deliberate looks;
* a ``look`` is served at the next opportunity, ahead of any measurement, and coalesces with any
  other look waiting for the same opportunity.

Every sight, change or look, is followed by a full ``settle_after``.

What a view shows is worked out by optional perception stages, each a plain injected capacity
behind its own typed wrapper the eye owns: objects (``vision.detection.ObjectDetector``), faces
(``vision.faces.FaceDetector``), when a face ``identification.Registry`` is composed who each
printed face is, and a plain description (``vision.description.Describer``). Stages run only on
views handed on, all views of a sight in parallel, each call bounded by ``stage_timeout``. A stage that fails or times out leaves only its own field ``None``.
A stage that timed out is stuck in its applying behavior; the eye recovers it by detaching and
re-attaching it (typed events, each bounded by ``restart_timeout``), which cancels the stuck work.
A stage whose recovery fails is skipped until the eye itself is detached and reattached.

Identified faces are stamped as ``face:<id>`` source participants. Biometric prints exist only in
the face stage's output, the identify request and the interpretation's locals: never in a sight,
a span, a metric or a log.

Telemetry (scope ``bot.abilities.seeing``; every attribute value is a closed set from code):

* ``bot.seeing.view.count`` — ``media_type`` (3) × ``bot.outcome`` {held, evicted, expired, dropped};
  a dropped stimulus has no held view and carries ``media_type`` all the same: 12 series.
* ``bot.seeing.measure.count`` — ``bot.outcome`` {salient, below_threshold, failed, timeout, first}: 5 series.
* ``bot.seeing.measure.duration`` — seconds per measurement, no attributes: 1 series.
* ``bot.seeing.change`` — the measured change (0..1) of every compared view, no attributes: 1 series.
* ``bot.seeing.sight.count`` — ``cause`` {change, look}: 2 series.
* ``bot.seeing.look.view.count`` — views handed on per look, no attributes: 1 series.
* ``bot.seeing.stage.count`` — ``stage`` {objects, faces, identification, description} ×
  ``bot.outcome`` {ok, failed, timeout}: at most 12 series.
* ``bot.seeing.object.count`` — objects found per view the object stage served, no attributes: 1 series.
* ``bot.seeing.face.count`` — faces handed on, by ``recognized`` {true, false, none}: 3 series.
* ``bot.seeing.stage.restart.count`` — stuck-stage recoveries, ``stage`` {objects, faces,
  description} × ``bot.outcome`` {ok, failed}: 6 series.
* Spans ``bot.seeing.measure``, ``bot.seeing.interpret`` and ``bot.seeing.handoff`` carry int and
  bool attributes, plus the closed ``bot.seeing.stage`` of an interpretation's first failure. A
  measure failure is ``measure_failed``, a measurement or stage call past its modeled deadline is
  ``timeout``, a stage that answered with a failure is ``stage_failed``, a face registry that
  answered for another kind is ``invalid_response``, and an identification failure carries the
  registry's own kind (FAILURE_KINDS.md).

Never used as attributes, logs or captured payloads: source ids, participant refs, face ids,
labels, boxes, descriptions, prints, signatures, change values (outside the histogram), image sizes.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import time
import typing
import uuid

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import event
from mosfet import media
from mosfet import telemetry
from mosfet.abilities import cognition
from mosfet.environment import VisualData, VisualEvent
from mosfet.protocols import attachment
from mosfet.telemetry import span

from .. import ability
from .. import identification
from .. import vision

MAX_VIEWS: typing.Final = 4
"""How many sources' current views Seeing holds; held image bytes stay within 4 × ``media.MAX_IMAGE_BYTES``."""

Cause = typing.Literal["change", "look"]

_MAX_PARTICIPANTS: typing.Final = 32
_SCOPE: typing.Final = "bot.abilities.seeing"
_COMPONENT: typing.Final = "seeing"
_MEASURE_FAILED: typing.Final = "measure_failed"
_STAGE_FAILED: typing.Final = "stage_failed"
_TIMEOUT: typing.Final = "timeout"
_INVALID_RESPONSE: typing.Final = "invalid_response"
_DEFAULT_MEASURE_EVERY: typing.Final = datetime.timedelta(milliseconds=500)
_DEFAULT_MEASURE_TIMEOUT: typing.Final = datetime.timedelta(seconds=2)
_DEFAULT_SETTLE_AFTER: typing.Final = datetime.timedelta(seconds=5)
_DEFAULT_MAX_VIEW_AGE: typing.Final = datetime.timedelta(seconds=60)
_DEFAULT_STAGE_TIMEOUT: typing.Final = datetime.timedelta(seconds=30)
_DEFAULT_RESTART_TIMEOUT: typing.Final = datetime.timedelta(seconds=10)

_Stage = typing.Literal["objects", "faces", "description"]
"""A perception stage the eye recovers when it is stuck (the face registry has its own deadline)."""
# A recovery detaches then re-attaches each stuck stage, each step bounded by restart_timeout, so
# the recovery as a whole is backstopped at that many restart_timeouts.
_RECOVERY_STEPS: typing.Final = 2 * len(typing.get_args(_Stage))
_StageName = typing.Literal["objects", "faces", "identification", "description"]
"""Every stage call the eye makes, as a closed metric and span value."""

_LOGGER = logging.getLogger(__name__)

_METER = metrics.get_meter(_SCOPE)
_VIEWS = _METER.create_counter(
    "bot.seeing.view.count",
    unit="{view}",
    description="Visual stimuli by what happened to their view: held, evicted, expired or dropped.",
)
_MEASUREMENTS = _METER.create_counter(
    "bot.seeing.measure.count",
    unit="{measurement}",
    description="Change measurements by outcome: salient, below_threshold, failed, timeout or first.",
)
_MEASURE_DURATION = _METER.create_histogram(
    "bot.seeing.measure.duration",
    unit="s",
    description="Time the injected change measure took per measured view.",
)
_CHANGE = _METER.create_histogram(
    "bot.seeing.change",
    unit="1",
    description="Measured change (0..1) of every view compared against the last view of its source handed on.",
)
_SIGHTS = _METER.create_counter(
    "bot.seeing.sight.count",
    unit="{sight}",
    description="Sights handed on, by cause: change or look.",
)
_LOOK_VIEWS = _METER.create_histogram(
    "bot.seeing.look.view.count",
    unit="{view}",
    description="Views handed on per look.",
)
_STAGE_CALLS = _METER.create_counter(
    "bot.seeing.stage.count",
    unit="{call}",
    description=(
        "Perception stage calls by stage (objects, faces, identification, description) and outcome "
        "(ok, failed, timeout)."
    ),
)
_OBJECTS = _METER.create_histogram(
    "bot.seeing.object.count",
    unit="{object}",
    description="Objects found per view the object stage served.",
)
_FACES = _METER.create_counter(
    "bot.seeing.face.count",
    unit="{face}",
    description="Faces handed on, by whether identification recognized them (true, false, none).",
)
_RESTARTS = _METER.create_counter(
    "bot.seeing.stage.restart.count",
    unit="{recovery}",
    description="Stuck-stage recoveries (detach then re-attach) by stage and outcome (ok, failed).",
)


class Face(pydantic.BaseModel):
    """One face in a view: where it is and, when a face registry is composed, who it is."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A face in view. face_id names who it is when identification ran: recognized=true is someone "
                "seen before, recognized=false someone new whose id was just made. Null fields mean no "
                "identification ran for this face."
            ),
            "examples": [
                {
                    "box": {"x": 0.42, "y": 0.18, "width": 0.12, "height": 0.2},
                    "confidence": 0.93,
                    "face_id": "3f2a9c",
                    "recognized": True,
                    "similarity": 0.82,
                },
                {
                    "box": {"x": 0.1, "y": 0.2, "width": 0.1, "height": 0.15},
                    "confidence": None,
                    "face_id": None,
                    "recognized": None,
                    "similarity": None,
                },
            ],
        },
    )

    box: vision.Box = pydantic.Field(description="Where the face is, in normalized image coordinates.")
    confidence: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How sure the face detector was, 0..1; null when it does not rate faces.",
        examples=[0.93, None],
    )
    face_id: identification.Id | None = pydantic.Field(
        default=None,
        description=(
            "Who this face is, as the face registry's id; the same person seen again has the same id. Null "
            "when no identification ran or it could not tell."
        ),
        examples=["3f2a9c", None],
    )
    recognized: bool | None = pydantic.Field(
        default=None,
        description=(
            "True: this face matched someone seen before. False: nobody known; a new id was made for them now. "
            "Null: no identification ran."
        ),
        examples=[True, False, None],
    )
    similarity: float | None = pydantic.Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="How close the face was to the matched (or nearest) known face, -1..1; null when none was compared.",
        examples=[0.82, None],
    )


class View(pydantic.BaseModel):
    """One source's view, as the eye hands it on."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "What one source showed: the image itself, how much it changed since the last view of that "
                "source the eye handed on, and how old it was when handed on."
            ),
            "examples": [
                {
                    "image": {"content": "/9j/4AAQ", "media_type": "image/jpeg", "width_px": 1280, "height_px": 720},
                    "kind": "camera",
                    "participants": [],
                    "change": 0.42,
                    "age_ms": 120,
                    "objects": [
                        {"label": "cup", "confidence": 0.87, "box": {"x": 0.4, "y": 0.5, "width": 0.1, "height": 0.15}}
                    ],
                    "faces": [],
                    "description": "A cup on a desk beside an open laptop.",
                }
            ],
        },
    )

    image: media.Image = pydantic.Field(
        description="The view as one encoded image, exactly as its source produced it.",
    )
    kind: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Low-cardinality provenance hint copied from the stimulus (for example camera, screen).",
        examples=["camera", "screen"],
    )
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=_MAX_PARTICIPANTS,
        description="Who the source's device stamped on the stimulus, copied unchanged. Empty when it knew no one.",
        examples=[[{"ref": "person-42", "role": "source"}]],
    )
    change: float | None = pydantic.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "How much this view changed since the last view of the same source that was handed on, 0 (same) to "
            "1 (entirely different), as the injected change measure rates it. Null for a source's first view "
            "and for a look."
        ),
        examples=[0.42, None],
    )
    age_ms: int = pydantic.Field(
        ge=0,
        description="How long ago the view arrived, in milliseconds, when it was handed on.",
        examples=[0, 120],
    )
    objects: tuple[vision.DetectedObject, ...] | None = pydantic.Field(
        default=None,
        max_length=vision.detection.MAX_OBJECTS,
        description=(
            "Objects found in the view, most confident first. Empty when the object stage found nothing; "
            "null when no object stage is composed or it could not answer."
        ),
    )
    faces: tuple[Face, ...] | None = pydantic.Field(
        default=None,
        max_length=vision.faces.MAX_FACES,
        description=(
            "Faces in the view, largest first. Empty when the face stage found none; null when no face stage "
            "is composed or it could not answer."
        ),
    )
    description: str | None = pydantic.Field(
        default=None,
        min_length=1,
        max_length=vision.description.MAX_DESCRIPTION_CHARS,
        description=(
            "A few plain sentences about what the view shows, as the describer put it; null when no describer "
            "is composed or it could not answer."
        ),
        examples=["A cup on a desk beside an open laptop.", None],
    )


class SightData(pydantic.BaseModel):
    """What the eye saw: one changed view, or every fresh view when it was asked to look."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A sight. cause=change hands on exactly one view that changed past the eye's threshold; "
                "cause=look hands on every fresh view, newest first, and holds no view when nothing is in view."
            ),
            "examples": [
                {
                    "cause": "change",
                    "views": [
                        {
                            "image": {"content": "/9j/4AAQ", "media_type": "image/jpeg"},
                            "kind": "camera",
                            "participants": [],
                            "change": None,
                            "age_ms": 0,
                        }
                    ],
                    "participants": [],
                },
                {"cause": "look", "views": [], "participants": []},
            ],
        },
    )

    cause: Cause = pydantic.Field(
        description="Why this sight was handed on: a view changed (change), or you asked to look (look).",
        examples=["change", "look"],
    )
    views: tuple[View, ...] = pydantic.Field(
        max_length=MAX_VIEWS,
        description="The views seen, newest first. Exactly one for a change; none to four for a look.",
    )
    participants: tuple[mosfet.Participant, ...] = pydantic.Field(
        default=(),
        max_length=_MAX_PARTICIPANTS,
        description=(
            "Who is in this sight: the views' stamped participants, then face:<id> for every identified face, "
            "first occurrence first."
        ),
        examples=[[{"ref": "person-42", "role": "source"}, {"ref": "face:3f2a9c", "role": "source"}]],
    )

    @pydantic.model_validator(mode="after")
    def _one_view_per_change(self) -> typing.Self:
        if self.cause == "change" and len(self.views) != 1:
            raise ValueError("a change sight carries exactly one view")
        return self


SightEvent = hsm.Event[SightData](
    name="bot.ability.seeing.sight",
    schema=SightData,
)


class LookData(pydantic.BaseModel):
    """Look at what is in view now."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Look at what is in view now. What you see arrives as your next observation "
                "(bot.ability.seeing.sight); if nothing is in view you see nothing."
            ),
            "examples": [{}],
        },
    )


LookEvent = hsm.Event[LookData](
    name="bot.ability.seeing.look",
    kind=event.EventKind,
    schema=LookData,
)


class _MeasuredData(pydantic.BaseModel):
    """A measured view: the stimulus it came from, when it arrived, and the measurement."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    source: str = pydantic.Field(min_length=1, description="Transducer the view came off.")
    visual: VisualData = pydantic.Field(description="The measured stimulus.")
    received_at: datetime.datetime = pydantic.Field(description="When the view arrived, on Seeing's clock.")
    measurement: vision.change.Measurement = pydantic.Field(description="What the change measure answered.")


class _MeasureFailedData(pydantic.BaseModel):
    """A view could not be measured; nothing is handed on and nothing else measures it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    kind: typing.Literal["measure_failed"] = pydantic.Field(description="Failure kind (FAILURE_KINDS.md).")
    message: str = pydantic.Field(description="Operator-facing reason; never a metric or span value.")


class _InterpretData(pydantic.BaseModel):
    """Views on their way to being handed on, and where the sight comes from."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    cause: Cause = pydantic.Field(description="Why these views are handed on.")
    source: str = pydantic.Field(
        min_length=1,
        description="Sight envelope source: the transducer for a change, Seeing itself for a look.",
    )
    views: tuple[View, ...] = pydantic.Field(max_length=MAX_VIEWS, description="The views, newest first.")


class _StageFailureData(pydantic.BaseModel):
    """One stage call that did not answer with a result."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    stage: _StageName = pydantic.Field(description="The stage that was called.")
    kind: str = pydantic.Field(min_length=1, max_length=64, description="Failure kind (FAILURE_KINDS.md).")


class _InterpretedData(_InterpretData):
    """Interpreted views, and how each stage fared while interpreting them."""

    timed_out: frozenset[_Stage] = pydantic.Field(
        default=frozenset(),
        description="Stages a call timed out on: each is stuck in its applying behavior until recovered.",
    )
    served: frozenset[_Stage] = pydantic.Field(
        default=frozenset(),
        description="Stages that answered at least one call with a result.",
    )
    failures: tuple[_StageFailureData, ...] = pydantic.Field(
        default=(),
        description="Every stage call that did not answer with a result, in the order they ended.",
    )


class _RecoveredData(pydantic.BaseModel):
    """How the recovery of stuck stages went."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    recovered: frozenset[_Stage] = pydantic.Field(description="Stages detached and re-attached: they serve again.")
    failed: frozenset[_Stage] = pydantic.Field(
        description="Stages whose detach or re-attach failed or did not answer: skipped until the eye is reattached.",
    )


_MeasuredEvent = hsm.Event[_MeasuredData](
    name="bot.ability.seeing.measure.completed",
    kind=hsm.CompletionEventKind,
    schema=_MeasuredData,
)
_MeasureFailedEvent = hsm.Event[_MeasureFailedData](
    name="bot.ability.seeing.measure.failed",
    kind=hsm.ErrorEventKind,
    schema=_MeasureFailedData,
)
# Completion kinds are served ahead of queued frames and looks, so a request is consumed by the
# state it was addressed to; anything that queued before it is handled by perceiving's own
# internal transitions.
_InterpretRequestedEvent = hsm.Event[_InterpretData](
    name="bot.ability.seeing.interpret.requested",
    kind=hsm.CompletionEventKind,
    schema=_InterpretData,
)
_InterpretedEvent = hsm.Event[_InterpretedData](
    name="bot.ability.seeing.interpret.completed",
    kind=hsm.CompletionEventKind,
    schema=_InterpretedData,
)
_RecoveredEvent = hsm.Event[_RecoveredData](
    name="bot.ability.seeing.recover.completed",
    kind=hsm.CompletionEventKind,
    schema=_RecoveredData,
)


@dataclasses.dataclass(slots=True)
class _Interpretation:
    """One interpretation's running tally across its parallel stage calls (activity locals only)."""

    timed_out: set[_Stage] = dataclasses.field(default_factory=set[_Stage])
    served: set[_Stage] = dataclasses.field(default_factory=set[_Stage])
    failures: list[_StageFailureData] = dataclasses.field(default_factory=list[_StageFailureData])
    skipped: int = 0

    def fail(self, stage: _StageName, kind: str, outcome: typing.Literal["failed", "timeout"]) -> None:
        self.failures.append(_StageFailureData(stage=stage, kind=kind))
        _STAGE_CALLS.add(1, {"stage": stage, "bot.outcome": outcome})


@dataclasses.dataclass(slots=True)
class _View:
    """Seeing's durable record of one source's newest view (machine bookkeeping, HSM-STATE-001)."""

    visual: VisualData
    received_at: datetime.datetime
    measured: bool
    # When this source was last measured: the round-robin key, kept across the source's frames.
    measured_at: datetime.datetime | None
    # Trace context of the stimulus, for the spans and the sight it may become (telemetry only).
    metadata: dict[str, object]


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _age(now: datetime.datetime, then: datetime.datetime) -> datetime.timedelta:
    return max(datetime.timedelta(0), now - then)


def _age_ms(now: datetime.datetime, then: datetime.datetime) -> int:
    return int(_age(now, then) / datetime.timedelta(milliseconds=1))


def _merged_participants(views: collections.abc.Sequence[View]) -> tuple[mosfet.Participant, ...]:
    """The views' stamped participants, then ``face:<id>`` for every identified face.

    First occurrence wins, bounded like the field.
    """

    stamped = (participant for view in views for participant in view.participants)
    identified = (
        identification.participant("face", face.face_id)
        for view in views
        for face in view.faces or ()
        if face.face_id is not None
    )
    return tuple(dict.fromkeys((*stamped, *identified)))[:_MAX_PARTICIPANTS]


class Seeing(ability.Ability[VisualData, cognition.InputData]):
    """Sensory ability that may turn ``environment.visual`` stimuli into ``cognition.InputEvent``.

    Public input is ``environment.visual``; the success terminal is cognitive input whose stimulus
    is a ``bot.ability.seeing.sight``. A view is handed on when the injected change measure rates
    it at or past ``change_threshold`` against the last view of its source that was handed on, or
    on a source's first view. ``bot.ability.seeing.look`` is the only model-callable event: it hands
    on every view fresher than ``max_view_age`` (none when nothing is in view).

    Every handed-on sight is remembered with its image bytes, within cognition's memory image
    capacity. Camera clients should therefore send JPEG with the long edge at or under 1280 px.
    Retaining or pruning the image bytes of old sights belongs to the memory lane, not to the eye.

    A measure that fails or does not answer within ``measure_timeout`` hands nothing on, is
    counted, and is never replaced by another measure; a look still sees the view.

    Composed stages fill each handed-on view's ``objects``, ``faces`` and ``description``; a composed face
    ``identification.Registry`` names who each printed face is and stamps ``face:<id>``. Each stage
    child serves one request at a time, so size ``stage_timeout`` to cover ``MAX_VIEWS`` times a
    stage's slowest healthy call: a queued request that times out sends a healthy stage through
    recovery.
    """

    input_event: typing.ClassVar[hsm.Event[VisualData]] = VisualEvent
    output_event: typing.ClassVar[hsm.Event[cognition.InputData]] = cognition.InputEvent
    effort: typing.ClassVar[ability.Effort] = ability.Effort.M
    _composite_attachment_lifecycle: typing.ClassVar[bool] = True
    _attachment_group: attachment.Group | None
    _change: vision.change.ChangeMeasure
    _change_threshold: float
    _measure_every: datetime.timedelta
    _measure_timeout: datetime.timedelta
    _settle_after: datetime.timedelta
    _max_view_age: datetime.timedelta
    _clock: collections.abc.Callable[[], datetime.datetime]
    _object_detection: vision.detection.ObjectDetection | None
    _face_detection: vision.faces.FaceDetection | None
    _registry: identification.Registry | None
    _description: vision.description.Description | None
    _stage_timeout: datetime.timedelta
    _restart_timeout: datetime.timedelta
    # Durable machine data; nothing here bridges one step to the next.
    _views: dict[str, _View]
    _seen: dict[str, vision.change.Signature]
    _look_pending: bool
    # Stages whose recovery failed: skipped until the eye is detached and reattached.
    _unavailable: set[_Stage]
    # Recoveries per stage since its last served call (D-10 counting).
    _consecutive_recoveries: dict[_Stage, int]

    def __init__(
        self,
        *,
        change: vision.change.ChangeMeasure,
        object_detector: vision.detection.ObjectDetector | None = None,
        face_detector: vision.faces.FaceDetector | None = None,
        faces: identification.Registry | None = None,
        describer: vision.description.Describer | None = None,
        change_threshold: float = 0.1,
        measure_every: datetime.timedelta = _DEFAULT_MEASURE_EVERY,
        measure_timeout: datetime.timedelta = _DEFAULT_MEASURE_TIMEOUT,
        settle_after: datetime.timedelta = _DEFAULT_SETTLE_AFTER,
        max_view_age: datetime.timedelta = _DEFAULT_MAX_VIEW_AGE,
        stage_timeout: datetime.timedelta = _DEFAULT_STAGE_TIMEOUT,
        restart_timeout: datetime.timedelta = _DEFAULT_RESTART_TIMEOUT,
        clock: collections.abc.Callable[[], datetime.datetime] | None = None,
    ) -> None:
        """See through ``change``, the one measure of how much a view changed; there is no fallback.

        ``object_detector``, ``face_detector`` and ``describer`` are optional perception stages,
        each wrapped in the typed stage this eye owns. ``faces`` is an unstarted face ``identification.Registry``
        (constructed with ``kind="face"``) that this eye owns from here on: its composite group
        starts it and attaches it for the eye's attached lifetime. The composition root may keep it
        to send it ``identification.ForgetEvent``. ``clock`` is the timezone-aware clock view ages are
        read from (UTC wall clock by default). The actors ``change.attachments`` names are attached
        for Seeing's attached lifetime.

        Raises:
            ValueError: If ``change_threshold`` is outside (0, 1], a duration is not positive,
                ``measure_every`` is longer than ``settle_after``, or ``faces`` is given without a
                ``face_detector`` to print the faces it would identify.
        """

        if not 0.0 < change_threshold <= 1.0:
            raise ValueError("Seeing change_threshold must be in (0, 1].")
        durations = {
            "measure_every": measure_every,
            "measure_timeout": measure_timeout,
            "settle_after": settle_after,
            "max_view_age": max_view_age,
            "stage_timeout": stage_timeout,
            "restart_timeout": restart_timeout,
        }
        for name, duration in durations.items():
            if duration <= datetime.timedelta(0):
                raise ValueError(f"Seeing {name} must be positive.")
        if measure_every > settle_after:
            raise ValueError("Seeing measure_every cannot be longer than settle_after.")
        if faces is not None and face_detector is None:
            raise ValueError("Seeing faces needs a face_detector: a registry identifies the prints it detects.")
        super().__init__()
        self._change = change
        self._change_threshold = change_threshold
        self._measure_every = measure_every
        self._measure_timeout = measure_timeout
        self._settle_after = settle_after
        self._max_view_age = max_view_age
        self._stage_timeout = stage_timeout
        self._restart_timeout = restart_timeout
        self._clock = _utc_now if clock is None else clock
        self._object_detection = (
            None if object_detector is None else vision.detection.ObjectDetection(classifier=object_detector)
        )
        self._face_detection = None if face_detector is None else vision.faces.FaceDetection(classifier=face_detector)
        self._registry = faces
        self._description = None if describer is None else vision.description.Description(decoder=describer)
        self._views = {}
        self._seen = {}
        self._look_pending = False
        self._unavailable = set()
        self._consecutive_recoveries = {}
        stages = tuple(
            child
            for child in (self._object_detection, self._face_detection, faces, self._description)
            if child is not None
        )
        self._attachment_group = attachment.Group(*stages, *change.attachments)

    # Durable bookkeeping.

    @staticmethod
    def _is_fresh(instance: "Seeing", view: _View, now: datetime.datetime) -> bool:
        return _age(now, view.received_at) <= instance._max_view_age

    @staticmethod
    def _forget_source(instance: "Seeing", source: str) -> None:
        del instance._views[source]
        _ = instance._seen.pop(source, None)

    @staticmethod
    def _is_visual(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, VisualData)

    @staticmethod
    def _is_look(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, LookData)

    @staticmethod
    def _hold(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Keep this view as its source's newest (latest wins); expire stale views; evict past ``MAX_VIEWS``."""

        del ctx
        visual = event.data
        assert isinstance(visual, VisualData)
        media_type = visual.image.media_type
        now = instance._clock()
        for source, view in tuple(instance._views.items()):
            if not Seeing._is_fresh(instance, view, now):
                Seeing._forget_source(instance, source)
                _VIEWS.add(1, {"media_type": view.visual.image.media_type, "bot.outcome": "expired"})
        if not event.source:
            # A view is keyed by its transducer; a stimulus that names none cannot be held.
            _VIEWS.add(1, {"media_type": media_type, "bot.outcome": "dropped"})
            return
        previous = instance._views.pop(event.source, None)
        instance._views[event.source] = _View(
            visual=visual,
            received_at=now,
            measured=False,
            measured_at=None if previous is None else previous.measured_at,
            metadata=dict(event.metadata),
        )
        _VIEWS.add(1, {"media_type": media_type, "bot.outcome": "held"})
        while len(instance._views) > MAX_VIEWS:
            # Dict order is receipt order: the first entry is the least recently received.
            evicted = next(iter(instance._views))
            view = instance._views[evicted]
            Seeing._forget_source(instance, evicted)
            _VIEWS.add(1, {"media_type": view.visual.image.media_type, "bot.outcome": "evicted"})

    @staticmethod
    def _request_look(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Remember that a look is wanted; repeated looks before it is served are one look."""

        del ctx, event
        instance._look_pending = True

    @staticmethod
    def _clear_state(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """A detached eye holds nothing: a reattached Seeing starts exactly as a fresh one."""

        del ctx, event
        instance._views.clear()
        instance._seen.clear()
        instance._look_pending = False
        instance._unavailable.clear()
        instance._consecutive_recoveries.clear()

    # The next choice.

    @staticmethod
    def _is_look_pending(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._look_pending

    @staticmethod
    def _has_unmeasured_view(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        now = instance._clock()
        return any(not view.measured and Seeing._is_fresh(instance, view, now) for view in instance._views.values())

    @staticmethod
    def _take_look(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Hand every fresh view on, newest first, as they are now; what was handed on before is untouched."""

        del event
        instance._look_pending = False
        now = instance._clock()
        fresh = sorted(
            (view for view in instance._views.values() if Seeing._is_fresh(instance, view, now)),
            key=lambda view: view.received_at,
            reverse=True,
        )
        views = tuple(
            View(
                image=view.visual.image,
                kind=view.visual.kind,
                participants=view.visual.participants,
                change=None,
                age_ms=_age_ms(now, view.received_at),
            )
            for view in fresh
        )
        _LOOK_VIEWS.record(len(views))
        _ = hsm.dispatch(
            ctx,
            instance,
            telemetry.inject_context(
                dataclasses.replace(
                    _InterpretRequestedEvent.with_data_and_id(
                        _InterpretData(cause="look", source=hsm.id(instance), views=views),
                        uuid.uuid4().hex,
                    ),
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                )
            ),
        )

    # Measuring.

    @staticmethod
    async def _measure(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Measure the least recently measured source's newest view against its last handed-on signature."""

        del event
        # The pick happens before the first await, so it sees exactly the views the next choice saw:
        # unmeasured fresh views, least recently measured source first (never measured first), ties
        # broken by the oldest arrival. That is a round robin across sources.
        now = instance._clock()
        candidates = [
            (source, view)
            for source, view in instance._views.items()
            if not view.measured and Seeing._is_fresh(instance, view, now)
        ]
        if not candidates:
            # The view aged out between the choice and this pick.
            _ = hsm.dispatch(
                ctx,
                instance,
                _MeasureFailedEvent.with_data_and_id(
                    _MeasureFailedData(kind=_MEASURE_FAILED, message="no fresh view was left to measure"),
                    uuid.uuid4().hex,
                ),
            )
            return
        source, view = min(
            candidates,
            key=lambda candidate: (
                candidate[1].measured_at is not None,
                candidate[1].measured_at or now,
                candidate[1].received_at,
            ),
        )
        view.measured = True
        view.measured_at = now
        previous = instance._seen.get(source)
        stimulus = dataclasses.replace(VisualEvent.with_data(view.visual), metadata=dict(view.metadata))
        with span.operation(
            "bot.seeing.measure",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="measure",
            context=telemetry.event_context(stimulus),
        ) as active:
            active.set_attribute("bot.seeing.change.first", previous is None)
            started = time.monotonic()  # telemetry only; hsm.after owns the measurement deadline
            try:
                measurement = await instance._change.measure(previous, view.visual.image, ctx=ctx)
            except Exception as error:
                # Boundary: whatever the injected measure raised, the view could not be measured.
                # It is not handed on and no other measure is asked (W3-7).
                _MEASURE_DURATION.record(time.monotonic() - started)
                span.record_failure(active, _MEASURE_FAILED)
                completion = _MeasureFailedEvent.with_data(
                    _MeasureFailedData(kind=_MEASURE_FAILED, message=f"{type(error).__name__}: {error}")
                )
            else:
                _MEASURE_DURATION.record(time.monotonic() - started)
                completion = _MeasuredEvent.with_data(
                    _MeasuredData(
                        source=source,
                        visual=view.visual,
                        received_at=view.received_at,
                        measurement=measurement,
                    )
                )
            _ = hsm.dispatch(
                ctx,
                instance,
                telemetry.inject_context(
                    dataclasses.replace(
                        completion,
                        id=uuid.uuid4().hex,
                        source=source,
                        target=hsm.id(instance),
                        metadata=dict(view.metadata),
                    )
                ),
            )

    @staticmethod
    def _is_salient(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _MeasuredData):
            return False
        change = data.measurement.change
        return change is None or change >= instance._change_threshold

    @staticmethod
    def _is_below_threshold(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _MeasuredData):
            return False
        change = data.measurement.change
        return change is not None and change < instance._change_threshold

    @staticmethod
    def _is_measure_failure(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _MeasureFailedData)

    @staticmethod
    def _take_change(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Hand the changed view on; its signature becomes what later views of its source are measured against."""

        measured = event.data
        assert isinstance(measured, _MeasuredData)
        change = measured.measurement.change
        if change is None:
            _MEASUREMENTS.add(1, {"bot.outcome": "first"})
        else:
            _MEASUREMENTS.add(1, {"bot.outcome": "salient"})
            _CHANGE.record(change)
        if measured.source in instance._views:
            instance._seen[measured.source] = measured.measurement.signature
        view = View(
            image=measured.visual.image,
            kind=measured.visual.kind,
            participants=measured.visual.participants,
            change=change,
            age_ms=_age_ms(instance._clock(), measured.received_at),
        )
        _ = hsm.dispatch(
            ctx,
            instance,
            telemetry.inject_context(
                dataclasses.replace(
                    _InterpretRequestedEvent.with_data_and_id(
                        _InterpretData(cause="change", source=measured.source, views=(view,)),
                        uuid.uuid4().hex,
                    ),
                    source=hsm.id(instance),
                    target=hsm.id(instance),
                    metadata=dict(event.metadata),
                )
            ),
        )

    @staticmethod
    def _count_below_threshold(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Not handed on, and the last handed-on signature stays: drift keeps accumulating against it."""

        del ctx, instance
        measured = event.data
        assert isinstance(measured, _MeasuredData)
        _MEASUREMENTS.add(1, {"bot.outcome": "below_threshold"})
        if measured.measurement.change is not None:
            _CHANGE.record(measured.measurement.change)

    @staticmethod
    def _count_measure_failure(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _MEASUREMENTS.add(1, {"bot.outcome": "failed"})

    @staticmethod
    def _count_measure_timeout(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _MEASUREMENTS.add(1, {"bot.outcome": "timeout"})

    @staticmethod
    def _measure_timeout_delay(
        ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._measure_timeout

    @staticmethod
    def _measure_every_delay(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._measure_every

    @staticmethod
    def _settle_after_delay(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._settle_after

    # Interpreting and handing on.

    @staticmethod
    def _carries_views(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _InterpretData)

    @staticmethod
    def _is_interpreted(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _InterpretedData)

    @staticmethod
    async def _ask_stage[T](
        instance: "Seeing",
        stage: _Stage,
        child: ability.Ability[typing.Any, typing.Any],
        request: object,
        expected: type[T],
        tally: _Interpretation,
        metadata: dict[str, typing.Any],
    ) -> T | None:
        """One stage call bounded by ``stage_timeout``: its result, or ``None`` (tallied) when it gave none."""

        try:
            terminal = await ability.run_terminal_operation(
                instance.context(),
                child=child,
                request=dataclasses.replace(
                    child.input_event.with_data_and_id(request, uuid.uuid4().hex), metadata=dict(metadata)
                ),
                terminals=(child.output_event, child.failed_event),
                timeout=instance._stage_timeout,
            )
        except TimeoutError:
            # The child is still applying the request: it is stuck until recovered.
            tally.timed_out.add(stage)
            tally.fail(stage, _TIMEOUT, "timeout")
            return None
        result = terminal.data
        if not isinstance(result, expected):
            tally.fail(stage, _STAGE_FAILED, "failed")
            return None
        tally.served.add(stage)
        _STAGE_CALLS.add(1, {"stage": stage, "bot.outcome": "ok"})
        return result

    @staticmethod
    async def _identify(
        instance: "Seeing",
        registry: identification.Registry,
        prints: tuple[identification.Print, ...],
        tally: _Interpretation,
        metadata: dict[str, typing.Any],
    ) -> tuple[identification.Recognition | None, ...] | None:
        """Who each print is, in order; ``None`` (tallied) when the face registry gave no usable answer."""

        try:
            terminal = await ability.run_terminal_operation(
                instance.context(),
                child=registry,
                request=dataclasses.replace(
                    identification.IdentifyEvent.with_data_and_id(
                        identification.IdentifyData(prints=prints), uuid.uuid4().hex
                    ),
                    metadata=dict(metadata),
                ),
                terminals=(identification.IdentifiedEvent, identification.FailedEvent),
                timeout=instance._stage_timeout,
            )
        except TimeoutError:
            tally.fail("identification", _TIMEOUT, "timeout")
            return None
        reply = terminal.data
        if isinstance(reply, identification.FailureData):
            tally.fail("identification", reply.kind.value, "failed")
            return None
        if not isinstance(reply, identification.IdentifiedData) or reply.kind != "face":
            # A registry of another kind never files faces under its ids.
            tally.fail("identification", _INVALID_RESPONSE, "failed")
            return None
        if len(reply.recognitions) != len(prints):
            tally.fail("identification", _INVALID_RESPONSE, "failed")
            return None
        _STAGE_CALLS.add(1, {"stage": "identification", "bot.outcome": "ok"})
        return reply.recognitions

    @staticmethod
    async def _interpret_view(
        instance: "Seeing", view: View, tally: _Interpretation, metadata: dict[str, typing.Any]
    ) -> View:
        """What one view shows: objects and faces in parallel, then who the printed faces are, then a description."""

        async def nothing() -> None:
            return None

        def ask[T](
            stage: _Stage,
            child: ability.Ability[typing.Any, typing.Any] | None,
            request: object,
            expected: type[T],
        ) -> collections.abc.Awaitable[T | None]:
            if child is None:
                return nothing()
            if stage in instance._unavailable:
                tally.skipped += 1
                return nothing()
            return Seeing._ask_stage(instance, stage, child, request, expected, tally, metadata)

        found, detected = await asyncio.gather(
            ask(
                "objects",
                instance._object_detection,
                vision.detection.InputData(image=view.image),
                vision.detection.OutputData,
            ),
            ask("faces", instance._face_detection, vision.faces.InputData(image=view.image), vision.faces.OutputData),
        )
        objects = None if found is None else found.objects
        if objects is not None:
            _OBJECTS.record(len(objects))
        faces: tuple[Face, ...] | None = None
        if detected is not None:
            prints = tuple(face.print for face in detected.faces if face.print is not None)
            recognitions: tuple[identification.Recognition | None, ...] = ()
            if prints and instance._registry is not None:
                recognitions = await Seeing._identify(instance, instance._registry, prints, tally, metadata) or ()
            # Recognitions answer the printed faces in order; the prints themselves stay here.
            answers = iter(recognitions)
            seen: list[Face] = []
            for face in detected.faces:
                recognition = next(answers, None) if face.print is not None else None
                _FACES.add(1, {"recognized": "none" if recognition is None else str(recognition.recognized).lower()})
                seen.append(
                    Face(
                        box=face.box,
                        confidence=face.confidence,
                        face_id=None if recognition is None else recognition.id,
                        recognized=None if recognition is None else recognition.recognized,
                        similarity=None if recognition is None else recognition.similarity,
                    )
                )
            faces = tuple(seen)
        description = await ask("description", instance._description, view.image, str)
        return View(
            image=view.image,
            kind=view.kind,
            participants=view.participants,
            change=view.change,
            age_ms=view.age_ms,
            objects=objects,
            faces=faces,
            description=description,
        )

    @staticmethod
    async def _interpret(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Work out what the requested views show; the request that entered this state is the only input.

        Views are interpreted in parallel. A stage that is not composed, failed, timed out or is
        unavailable leaves only its own field ``None``.
        """

        request = event.data
        assert isinstance(request, _InterpretData)
        with span.operation(
            "bot.seeing.interpret",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="interpret",
            context=telemetry.event_context(event),
        ) as active:
            active.set_attribute("bot.seeing.view.count", len(request.views))
            active.set_attribute(
                "bot.seeing.stage.restart.consecutive", max(instance._consecutive_recoveries.values(), default=0)
            )
            tally = _Interpretation()
            views = await asyncio.gather(
                *(Seeing._interpret_view(instance, view, tally, dict(event.metadata)) for view in request.views)
            )
            active.set_attribute("bot.seeing.stage.skipped.count", tally.skipped)
            if tally.failures:
                first = tally.failures[0]
                active.set_attribute("bot.seeing.stage", first.stage)
                span.record_failure(active, first.kind)
            _ = hsm.dispatch(
                ctx,
                instance,
                telemetry.inject_context(
                    dataclasses.replace(
                        _InterpretedEvent.with_data_and_id(
                            _InterpretedData(
                                cause=request.cause,
                                source=request.source,
                                views=tuple(views),
                                timed_out=frozenset(tally.timed_out),
                                served=frozenset(tally.served),
                                failures=tuple(tally.failures),
                            ),
                            event.id or uuid.uuid4().hex,
                        ),
                        source=hsm.id(instance),
                        target=hsm.id(instance),
                        metadata=dict(event.metadata),
                    )
                ),
            )

    @staticmethod
    def _perceive(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Hand the sight on through this ability's terminal, under fresh ids.

        The sight and its handoff never reuse an id that came in: a look selected on a turn carries
        that turn's operation id, and reusing it would make the sight's turn a duplicate of it.
        """

        interpreted = event.data
        assert isinstance(interpreted, _InterpretedData)
        for stage in interpreted.served:
            _ = instance._consecutive_recoveries.pop(stage, None)
        with span.operation(
            "bot.seeing.handoff",
            scope=_SCOPE,
            component=_COMPONENT,
            stage="cognition_handoff",
            context=telemetry.event_context(event),
        ) as active:
            participants = _merged_participants(interpreted.views)
            active.set_attribute("bot.seeing.view.count", len(interpreted.views))
            active.set_attribute("bot.participant.count", len(participants))
            sight = telemetry.inject_context(
                dataclasses.replace(
                    SightEvent.with_data_and_id(
                        SightData(cause=interpreted.cause, views=interpreted.views, participants=participants),
                        uuid.uuid4().hex,
                    ),
                    source=interpreted.source,
                    metadata=dict(event.metadata),
                )
            )
            handoff = dataclasses.replace(
                cognition.InputEvent.with_data_and_id(
                    cognition.InputData(stimulus=sight, participants=participants),
                    uuid.uuid4().hex,
                ),
                source=hsm.id(instance),
                target=None,
                metadata=dict(sight.metadata),
            )
            _SIGHTS.add(1, {"cause": interpreted.cause})
            _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(handoff))

    # Recovering stuck stages.

    @staticmethod
    def _has_stuck_stage(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, _InterpretedData) and bool(data.timed_out - instance._unavailable)

    @staticmethod
    def _stage_child(instance: "Seeing", stage: _Stage) -> ability.Ability[typing.Any, typing.Any]:
        children: dict[_Stage, ability.Ability[typing.Any, typing.Any] | None] = {
            "objects": instance._object_detection,
            "faces": instance._face_detection,
            "description": instance._description,
        }
        child = children[stage]
        assert child is not None
        return child

    @staticmethod
    async def _restart(instance: "Seeing", stage: _Stage) -> bool:
        """Detach the stuck stage (leaving ``attached`` cancels its stuck work), then attach it again.

        Both are typed requests to the child where it lives, each bounded by ``restart_timeout``.
        """

        child = Seeing._stage_child(instance, stage)
        try:
            detached = await ability.run_terminal_operation(
                instance.context(),
                child=child,
                request=lambda operation: attachment.DetachEvent.with_data_and_id(
                    attachment.DetachData(actor=instance, reply_to=operation, timeout=instance._restart_timeout),
                    uuid.uuid4().hex,
                ),
                terminals=(attachment.DetachedEvent, attachment.DetachFailedEvent),
                timeout=instance._restart_timeout,
            )
            if not isinstance(detached.data, attachment.DetachedData):
                return False
            attached = await ability.run_terminal_operation(
                instance.context(),
                child=child,
                request=lambda operation: attachment.AttachEvent.with_data_and_id(
                    attachment.AttachData(actor=instance, reply_to=operation, timeout=instance._restart_timeout),
                    uuid.uuid4().hex,
                ),
                terminals=(attachment.AttachCompleteEvent, attachment.AttachFailedEvent),
                timeout=instance._restart_timeout,
            )
        except TimeoutError:
            return False
        return isinstance(attached.data, attachment.AttachCompleteData)

    @staticmethod
    async def _recover_stages(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """Recover, one after another, the stages the interpretation that entered this state timed out on."""

        interpreted = event.data
        assert isinstance(interpreted, _InterpretedData)
        stuck = frozenset(interpreted.timed_out - instance._unavailable)
        recovered: set[_Stage] = set()
        try:
            for stage in sorted(stuck):
                if await Seeing._restart(instance, stage):
                    recovered.add(stage)
        except asyncio.CancelledError:
            # Cut short (the backstop, or the eye leaving perceiving): what did not come back failed.
            _ = hsm.dispatch(
                instance.context(),
                instance,
                _RecoveredEvent.with_data_and_id(
                    _RecoveredData(recovered=frozenset(recovered), failed=stuck - recovered), uuid.uuid4().hex
                ),
            )
            raise
        _ = hsm.dispatch(
            ctx,
            instance,
            _RecoveredEvent.with_data_and_id(
                _RecoveredData(recovered=frozenset(recovered), failed=stuck - recovered), uuid.uuid4().hex
            ),
        )

    @staticmethod
    def _is_recovered(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _RecoveredData)

    @staticmethod
    def _record_recoveries(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        """A recovered stage serves again; a failed one is skipped until the eye is reattached."""

        del ctx
        data = event.data
        assert isinstance(data, _RecoveredData)
        for stage in sorted(data.recovered | data.failed):
            failed = stage in data.failed
            instance._consecutive_recoveries[stage] = instance._consecutive_recoveries.get(stage, 0) + 1
            _RESTARTS.add(1, {"stage": stage, "bot.outcome": "failed" if failed else "ok"})
            if failed:
                instance._unavailable.add(stage)
                _LOGGER.warning(
                    "Seeing could not recover its %s stage; it is skipped until Seeing is detached and reattached.",
                    stage,
                )

    @staticmethod
    def _count_recovery_timeout(ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        _LOGGER.warning(
            "Seeing stage recovery did not finish within its backstop; stages it had not recovered are skipped."
        )

    @staticmethod
    def _recovery_backstop_delay(
        ctx: hsm.Context, instance: "Seeing", event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        # Constant: every stage detached and re-attached, each step within restart_timeout.
        del ctx, event
        return _RECOVERY_STEPS * instance._restart_timeout

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Seeing",
        hsm.initial(hsm.target("/Seeing/initializing")),
        hsm.state(
            "initializing",
            hsm.defer(input_event, LookEvent, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._attach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_attach_complete),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Seeing/perceiving"),
            ),
        ),
        hsm.state(
            "perceiving",
            hsm.initial(hsm.target("/Seeing/perceiving/watching")),
            # Wherever perception is, the newest frame replaces the held view and a look is
            # remembered. Internal: neither interrupts what the eye is doing.
            hsm.transition(hsm.on(input_event), hsm.guard(_is_visual), hsm.effect(_hold)),
            hsm.transition(hsm.on(LookEvent), hsm.guard(_is_look), hsm.effect(_request_look)),
            # A recovery cut short reports after its state was left; what it reports still holds.
            hsm.transition(hsm.on(_RecoveredEvent), hsm.guard(_is_recovered), hsm.effect(_record_recoveries)),
            hsm.state(
                "watching",
                hsm.transition(
                    hsm.on(input_event),
                    hsm.guard(_is_visual),
                    hsm.effect(_hold),
                    hsm.target("/Seeing/perceiving/next"),
                ),
                hsm.transition(
                    hsm.on(LookEvent),
                    hsm.guard(_is_look),
                    hsm.effect(_request_look),
                    hsm.target("/Seeing/perceiving/next"),
                ),
            ),
            hsm.state(
                "measuring",
                hsm.activity(_measure),
                hsm.transition(
                    hsm.on(_MeasuredEvent),
                    hsm.guard(_is_salient),
                    hsm.effect(_take_change),
                    hsm.target("/Seeing/perceiving/interpreting"),
                ),
                hsm.transition(
                    hsm.on(_MeasuredEvent),
                    hsm.guard(_is_below_threshold),
                    hsm.effect(_count_below_threshold),
                    hsm.target("/Seeing/perceiving/pacing"),
                ),
                hsm.transition(
                    hsm.on(_MeasureFailedEvent),
                    hsm.guard(_is_measure_failure),
                    hsm.effect(_count_measure_failure),
                    hsm.target("/Seeing/perceiving/pacing"),
                ),
                hsm.transition(
                    hsm.after(_measure_timeout_delay),
                    hsm.effect(_count_measure_timeout),
                    hsm.target("/Seeing/perceiving/pacing"),
                ),
            ),
            # At most one measurement per measure_every, whatever the source's frame rate.
            hsm.state(
                "pacing",
                hsm.transition(hsm.after(_measure_every_delay), hsm.target("/Seeing/perceiving/next")),
                hsm.transition(
                    hsm.on(LookEvent),
                    hsm.guard(_is_look),
                    hsm.effect(_request_look),
                    hsm.target("/Seeing/perceiving/next"),
                ),
            ),
            hsm.state(
                "interpreting",
                hsm.initial(hsm.target("/Seeing/perceiving/interpreting/requested")),
                hsm.transition(
                    hsm.on(_InterpretedEvent),
                    hsm.guard(_is_interpreted),
                    hsm.effect(_perceive),
                    hsm.target("/Seeing/perceiving/recovering"),
                ),
                hsm.state(
                    "requested",
                    hsm.transition(
                        hsm.on(_InterpretRequestedEvent),
                        hsm.guard(_carries_views),
                        hsm.target("/Seeing/perceiving/interpreting/running"),
                    ),
                ),
                hsm.state("running", hsm.activity(_interpret)),
            ),
            # A stage that timed out is stuck applying the request it never answered.
            hsm.choice(
                "recovering",
                hsm.transition(hsm.guard(_has_stuck_stage), hsm.target("/Seeing/perceiving/recovering_stages")),
                hsm.transition(hsm.target("/Seeing/perceiving/settling")),
            ),
            hsm.state(
                "recovering_stages",
                hsm.activity(_recover_stages),
                hsm.transition(
                    hsm.on(_RecoveredEvent),
                    hsm.guard(_is_recovered),
                    hsm.effect(_record_recoveries),
                    hsm.target("/Seeing/perceiving/settling"),
                ),
                hsm.transition(
                    hsm.after(_recovery_backstop_delay),
                    hsm.effect(_count_recovery_timeout),
                    hsm.target("/Seeing/perceiving/settling"),
                ),
            ),
            # Every sight is followed by a full settle_after: at most one per settle_after, plus looks.
            hsm.state(
                "settling",
                hsm.transition(hsm.after(_settle_after_delay), hsm.target("/Seeing/perceiving/next")),
            ),
            hsm.choice(
                "next",
                # A pending look goes first, ahead of any measurement, so motion cannot starve it.
                hsm.transition(
                    hsm.guard(_is_look_pending),
                    hsm.effect(_take_look),
                    hsm.target("/Seeing/perceiving/interpreting"),
                ),
                hsm.transition(hsm.guard(_has_unmeasured_view), hsm.target("/Seeing/perceiving/measuring")),
                hsm.transition(hsm.target("/Seeing/perceiving/watching")),
            ),
        ),
        hsm.state(
            "detaching",
            hsm.entry(_clear_state),
            hsm.defer(input_event, LookEvent, attachment.AttachEvent, attachment.DetachEvent),
            hsm.activity(ability.Ability._detach_composite_group),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_rollback_failure),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Seeing/degraded"),
            ),
            hsm.transition(
                hsm.on(ability.Ability._composite_attachment_terminal_event),
                hsm.guard(ability.Ability._is_composite_detach_failed),
                hsm.effect(ability.Ability._deliver_composite_attachment_terminal),
                hsm.target("/Seeing/perceiving"),
            ),
        ),
        hsm.state("degraded"),
    )


__all__ = [
    "MAX_VIEWS",
    "Cause",
    "Face",
    "LookData",
    "LookEvent",
    "Seeing",
    "SightData",
    "SightEvent",
    "View",
]
