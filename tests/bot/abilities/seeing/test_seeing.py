"""HSM runtime tests for ``seeing.Seeing``: measured change, pacing, refractory, look, bounds, stages."""

import asyncio
import collections.abc
import dataclasses
import datetime
import hashlib
import math
import pathlib
import typing

import hsm
import mosfet
import pydantic
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mosfet import abilities, environment, media
from mosfet import event as event_contract
from mosfet.abilities import cognition, identification, processing, seeing, vision
from mosfet.abilities.vision import change
from mosfet.protocols import attachment
from mosfet.telemetry import span

from tests import metric_reader

_WAIT = 3.0
_T0 = datetime.datetime(2026, 10, 15, 12, 0, tzinfo=datetime.UTC)
_PERSON = mosfet.Participant(ref="person-42", role=mosfet.ParticipantRole.SOURCE)
_OTHER = mosfet.Participant(ref="person-7", role=mosfet.ParticipantRole.SOURCE)
_MS = datetime.timedelta(milliseconds=1)
_MEASURE_EVERY = datetime.timedelta(milliseconds=20)
_MEASURE_TIMEOUT = datetime.timedelta(seconds=2)
_SETTLE_AFTER = datetime.timedelta(milliseconds=40)
_MAX_VIEW_AGE = datetime.timedelta(seconds=60)
_STAGE_TIMEOUT = datetime.timedelta(seconds=2)
_RESTART_TIMEOUT = datetime.timedelta(seconds=2)
_BOX = vision.Box(x=0.1, y=0.1, width=0.2, height=0.2)


def _jpeg(label: str) -> media.Image:
    return media.Image(content=b"\xff\xd8\xff" + label.encode(), media_type="image/jpeg")


def _label(image: media.Image) -> str:
    return image.content[3:].decode()


def _signature(image: media.Image) -> change.Signature:
    return change.Signature(measure="fake", digest=hashlib.sha256(image.content).hexdigest())


@dataclasses.dataclass
class Call:
    previous: change.Signature | None
    image: media.Image
    cancelled: bool = False


class FakeChange(change.ChangeMeasure):
    """Scripted change measure: ``score`` rates a compared view; the first view of a source is ``None``."""

    calls: list[Call]
    score: typing.Callable[[media.Image], float]
    fail: bool
    hang: bool

    def __init__(
        self,
        score: typing.Callable[[media.Image], float] = lambda image: 1.0,
        *,
        fail: bool = False,
        hang: bool = False,
        attachments: tuple[hsm.Instance, ...] = (),
    ) -> None:
        self.score = score
        self.fail = fail
        self.hang = hang
        self.attachments: tuple[hsm.Instance, ...] = attachments
        self.calls = []

    @typing.override
    async def measure(
        self, previous: change.Signature | None, image: media.Image, *, ctx: hsm.Context
    ) -> change.Measurement:
        del ctx
        call = Call(previous=previous, image=image)
        self.calls.append(call)
        if self.hang:
            try:
                _ = await asyncio.Event().wait()
            except asyncio.CancelledError:
                call.cancelled = True
                raise
        if self.fail:
            raise change.Error("scripted failure")
        return change.Measurement(change=None if previous is None else self.score(image), signature=_signature(image))


@dataclasses.dataclass
class StageCall:
    image: media.Image
    cancelled: bool = False


class Script:
    """How a fake stage answers: hang on its first call, fail, or wait for a gate."""

    calls: list[StageCall]
    hang_first: bool
    fail: bool
    gate: asyncio.Event | None

    def __init__(self, *, hang_first: bool = False, fail: bool = False, gate: asyncio.Event | None = None) -> None:
        self.hang_first = hang_first
        self.fail = fail
        self.gate = gate
        self.calls = []

    async def step(self, image: media.Image) -> None:
        call = StageCall(image=image)
        self.calls.append(call)
        if self.gate is not None:
            _ = await self.gate.wait()
        if self.hang_first and len(self.calls) == 1:
            try:
                _ = await asyncio.Event().wait()
            except asyncio.CancelledError:
                call.cancelled = True
                raise
        if self.fail:
            raise RuntimeError("scripted stage failure")


class FakeObjects(vision.detection.ObjectDetector):
    found: tuple[vision.DetectedObject, ...]
    script: Script

    def __init__(self, *found: vision.DetectedObject, script: Script | None = None) -> None:
        self.found = found
        self.script = Script() if script is None else script

    @typing.override
    async def classify(self, input: vision.detection.InputData) -> vision.detection.OutputData:
        await self.script.step(input.image)
        return vision.detection.OutputData(objects=self.found)


class FakeFaces(vision.faces.FaceDetector):
    found: tuple[vision.faces.DetectedFace, ...]
    script: Script

    def __init__(self, *found: vision.faces.DetectedFace, script: Script | None = None) -> None:
        self.found = found
        self.script = Script() if script is None else script

    @typing.override
    async def classify(self, input: vision.faces.InputData) -> vision.faces.OutputData:
        await self.script.step(input.image)
        return vision.faces.OutputData(faces=self.found)


class FakeIdentifier(identification.Identifier):
    """Answers every identify with ``recognitions`` (one per print by default)."""

    identified: list[tuple[identification.Print, ...]]
    recognitions: tuple[identification.Recognition | None, ...]

    def __init__(self, *recognitions: identification.Recognition | None) -> None:
        self.recognitions = recognitions
        self.identified = []

    @typing.override
    async def identify(
        self, kind: identification.Kind, prints: tuple[identification.Print, ...]
    ) -> tuple[identification.Recognition | None, ...]:
        del kind
        self.identified.append(prints)
        return self.recognitions

    @typing.override
    async def enroll(
        self, kind: identification.Kind, id: identification.Id, print: identification.Print, own: bool
    ) -> identification.Recognition:
        del kind, print
        return identification.Recognition(id=id, recognized=False, similarity=None, own=own)

    @typing.override
    async def forget(self, kind: identification.Kind, id: identification.Id) -> bool:
        del kind, id
        return False

    @typing.override
    async def prune(self, kind: identification.Kind, *, unseen_before: datetime.datetime, max_ids: int) -> int:
        del kind, unseen_before, max_ids
        return 0


def _registry(identifier: identification.Identifier, kind: identification.Kind = "face") -> identification.Registry:
    return identification.Registry(
        identifier=identifier, kind=kind, max_unlinked_age=datetime.timedelta(days=30), max_ids=100
    )


def _found(label: str, confidence: float | None) -> vision.DetectedObject:
    return vision.DetectedObject(label=label, confidence=confidence, box=_BOX)


def _detected_face(size: float, print: identification.Print | None = None) -> vision.faces.DetectedFace:
    return vision.faces.DetectedFace(box=vision.Box(x=0.0, y=0.0, width=size, height=size), confidence=0.9, print=print)


def _print(*vector: float) -> identification.Print:
    return identification.Print(vector=vector, model="face-model")


class RecordingDetection(vision.detection.ObjectDetection):
    """Object stage that records the lifecycle requests it receives; it may refuse or swallow one attach."""

    lifecycle: list[hsm.Event[typing.Any]]
    refuse_attach: int | None
    swallow_attach: int | None

    def __init__(
        self,
        *,
        classifier: vision.detection.ObjectDetector,
        refuse_attach: int | None = None,
        swallow_attach: int | None = None,
    ) -> None:
        super().__init__(classifier=classifier)
        self.lifecycle = []
        self.refuse_attach = refuse_attach
        self.swallow_attach = swallow_attach

    def attaches(self) -> list[hsm.Event[typing.Any]]:
        return [event for event in self.lifecycle if isinstance(event.data, attachment.AttachData)]

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event) -> collections.abc.Awaitable[bool]:
        data = event.data
        if isinstance(data, (attachment.AttachData, attachment.DetachData)):
            self.lifecycle.append(event)
        if isinstance(data, attachment.AttachData):
            count = len(self.attaches())
            if count == self.swallow_attach:
                return asyncio.get_running_loop().create_future()
            if count == self.refuse_attach:
                reply_to = data.actor if data.reply_to is None else data.reply_to
                return hsm.dispatch(
                    ctx,
                    reply_to,
                    dataclasses.replace(
                        attachment.AttachFailedEvent.with_data(
                            attachment.FailedData(
                                actor=data.actor, kind=attachment.FailureKind.INITIALIZATION, message="refused"
                            )
                        ),
                        id=event.id,
                        source=hsm.id(self),
                        target=hsm.id(reply_to),
                    ),
                )
        return super().dispatch(ctx, event)


def _record_detections(
    monkeypatch: pytest.MonkeyPatch, *, refuse_attach: int | None = None, swallow_attach: int | None = None
) -> list[RecordingDetection]:
    """Have Seeing build ``RecordingDetection`` object stages; returns the ones it built."""

    built: list[RecordingDetection] = []

    def build(*, classifier: vision.detection.ObjectDetector) -> RecordingDetection:
        child = RecordingDetection(classifier=classifier, refuse_attach=refuse_attach, swallow_attach=swallow_attach)
        built.append(child)
        return child

    monkeypatch.setattr(vision.detection, "ObjectDetection", build)
    return built


class ManualClock:
    now: datetime.datetime

    def __init__(self) -> None:
        self.now = _T0

    def __call__(self) -> datetime.datetime:
        return self.now


class Owner(hsm.Instance):
    """Owner that records every event delivered to it with its arrival time."""

    events: list[tuple[float, hsm.Event[typing.Any]]]
    changed: asyncio.Event

    @staticmethod
    def _record(ctx: hsm.Context, instance: "Owner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.events.append((asyncio.get_running_loop().time(), event))
        instance.changed.set()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SeeingTestOwner",
        hsm.initial(hsm.target("recording")),
        hsm.state("recording", hsm.transition(hsm.on(hsm.AnyEvent), hsm.effect(_record))),
    )

    def __init__(self) -> None:
        super().__init__()
        self.events = []
        self.changed = asyncio.Event()

    def named(self, *events: hsm.Event[typing.Any]) -> list[hsm.Event[typing.Any]]:
        names = {event.name for event in events}
        return [event for _, event in self.events if event.name in names]

    def handoffs(self) -> list[tuple[float, hsm.Event[typing.Any]]]:
        return [(at, event) for at, event in self.events if event.name == cognition.InputEvent.name]

    async def until(self, predicate: typing.Callable[[], bool], timeout: float = _WAIT) -> None:
        async with asyncio.timeout(timeout):
            while not predicate():
                self.changed.clear()
                _ = await self.changed.wait()


class Probe(hsm.Instance, attachment.Attachment):
    """A root-started provider actor that answers attach and detach, optionally holding the attach."""

    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str
    attach_calls: list[hsm.Event[attachment.AttachData]]
    detach_calls: list[hsm.Event[attachment.DetachData]]
    attach_release: asyncio.Event

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "SeeingTestProbe",
        hsm.initial(hsm.target("ready")),
        hsm.state("ready"),
    )

    def __init__(self, *, hold_attach: bool = False) -> None:
        super().__init__()
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self.attach_calls = []
        self.detach_calls = []
        self.attach_release = asyncio.Event()
        if not hold_attach:
            self.attach_release.set()

    async def _answer(
        self,
        ctx: hsm.Context,
        event: hsm.Event[typing.Any],
        data: attachment.AttachData | attachment.DetachData,
        result: hsm.Event[typing.Any],
    ) -> None:
        target = data.actor if data.reply_to is None else data.reply_to
        _ = await hsm.dispatch(
            ctx,
            target,
            dataclasses.replace(result, id=event.id, source=hsm.id(self), target=hsm.id(target)),
        )

    @typing.override
    async def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> bool:
        self.attach_calls.append(event)
        _ = await self.attach_release.wait()
        data = event.data
        assert isinstance(data, attachment.AttachData)
        complete = attachment.AttachCompleteData(actor=data.actor, created=True, reply_to=data.reply_to)
        await self._answer(ctx, event, data, attachment.AttachCompleteEvent.with_data(complete))
        return True

    @typing.override
    async def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> bool:
        self.detach_calls.append(event)
        data = event.data
        assert isinstance(data, attachment.DetachData)
        detached = attachment.DetachedData(actor=data.actor, removed=True)
        await self._answer(ctx, event, data, attachment.DetachedEvent.with_data(detached))
        return True


@dataclasses.dataclass
class Harness:
    ctx: environment.Environment
    eye: seeing.Seeing
    owner: Owner
    measure: FakeChange

    async def attach(self, request_id: str = "attach") -> None:
        _ = await self.eye.attach(
            self.ctx,
            attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=self.owner), request_id),
        )

    async def attached(self, request_id: str = "attach") -> None:
        await self.attach(request_id)
        await self.owner.until(
            lambda: any(event.id == request_id for event in self.owner.named(attachment.AttachCompleteEvent))
        )

    async def detached(self, request_id: str = "detach") -> None:
        _ = await self.eye.detach(
            self.ctx,
            attachment.DetachEvent.with_data_and_id(attachment.DetachData(actor=self.owner), request_id),
        )
        await self.owner.until(
            lambda: any(event.id == request_id for event in self.owner.named(attachment.DetachedEvent))
        )

    async def frame(
        self,
        source: str,
        image: media.Image,
        *,
        event_id: str | None = None,
        kind: str | None = "camera",
        participants: tuple[mosfet.Participant, ...] = (),
    ) -> None:
        visual = environment.VisualEvent.with_data_and_id(
            environment.VisualData(image=image, kind=kind, participants=participants),
            event_id or f"visual-{_label(image)}",
        )
        _ = await hsm.dispatch(self.ctx, self.eye, dataclasses.replace(visual, source=source, target=hsm.id(self.eye)))

    async def look(self, event_id: str = "turn-1") -> None:
        look = seeing.LookEvent.with_data_and_id(seeing.LookData(), event_id)
        _ = await hsm.dispatch(
            self.ctx, self.eye, dataclasses.replace(look, source="cognition", target=hsm.id(self.eye))
        )

    def sights(self, cause: seeing.Cause | None = None) -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        found: list[tuple[float, hsm.Event[typing.Any], seeing.SightData]] = []
        for at, handoff in self.owner.handoffs():
            data = handoff.data
            assert isinstance(data, cognition.InputData)
            sight = data.stimulus
            assert isinstance(sight, hsm.Event) and sight.name == seeing.SightEvent.name
            assert isinstance(sight.data, seeing.SightData)
            if cause is None or sight.data.cause == cause:
                found.append((at, handoff, sight.data))
        return found

    async def until_calls(self, count: int) -> None:
        async with asyncio.timeout(_WAIT):
            while len(self.measure.calls) < count:
                await asyncio.sleep(0.001)


async def _harness(
    measure: FakeChange | None = None,
    *,
    started: tuple[hsm.Instance, ...] = (),
    attach: bool = True,
    change_threshold: float = 0.1,
    measure_every: datetime.timedelta = _MEASURE_EVERY,
    measure_timeout: datetime.timedelta = _MEASURE_TIMEOUT,
    settle_after: datetime.timedelta = _SETTLE_AFTER,
    max_view_age: datetime.timedelta = _MAX_VIEW_AGE,
    clock: ManualClock | None = None,
    object_detector: vision.detection.ObjectDetector | None = None,
    face_detector: vision.faces.FaceDetector | None = None,
    faces: identification.Registry | None = None,
    stage_timeout: datetime.timedelta = _STAGE_TIMEOUT,
    restart_timeout: datetime.timedelta = _RESTART_TIMEOUT,
) -> Harness:
    ctx = environment.Environment()
    owner = Owner()
    _ = await mosfet.started(ctx, owner, Owner.model)
    for actor in started:
        _ = await mosfet.started(ctx, actor, Probe.model)
    fake = FakeChange() if measure is None else measure
    eye = seeing.Seeing(
        change=fake,
        change_threshold=change_threshold,
        measure_every=measure_every,
        measure_timeout=measure_timeout,
        settle_after=settle_after,
        max_view_age=max_view_age,
        clock=clock,
        object_detector=object_detector,
        face_detector=face_detector,
        faces=faces,
        stage_timeout=stage_timeout,
        restart_timeout=restart_timeout,
    )
    harness = Harness(ctx=ctx, eye=eye, owner=owner, measure=fake)
    if attach:
        await harness.attached()
    return harness


def _sight_event(handoff: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    data = handoff.data
    assert isinstance(data, cognition.InputData)
    sight = data.stimulus
    assert isinstance(sight, hsm.Event)
    return sight


# Contracts


def test_seeing_exports_its_contract() -> None:
    assert seeing.SightEvent.name == "bot.ability.seeing.sight"
    assert seeing.LookEvent.name == "bot.ability.seeing.look"
    assert hsm.kind.Is(seeing.LookEvent.kind, event_contract.EventKind)
    assert not hsm.kind.Is(seeing.SightEvent.kind, event_contract.EventKind)
    assert seeing.Seeing.input_event is environment.VisualEvent
    assert seeing.Seeing.output_event is cognition.InputEvent
    assert seeing.Seeing.failed_event is abilities.FailedEvent
    assert seeing.Seeing.effort is abilities.ability.Effort.M
    assert seeing.MAX_VIEWS == 4
    assert set(seeing.__all__) >= {
        "Seeing",
        "SightData",
        "SightEvent",
        "View",
        "LookData",
        "LookEvent",
        "Cause",
        "Face",
        "MAX_VIEWS",
    }
    assert seeing.LookData.model_fields == {}
    description = seeing.LookData.model_json_schema().get("description")
    assert isinstance(description, str) and "bot.ability.seeing.sight" in description


def test_sight_and_view_are_bounded_frozen_values() -> None:
    view = seeing.View(image=_jpeg("a"), age_ms=0)

    assert view.change is None and view.kind is None and view.participants == ()
    assert seeing.SightData(cause="look", views=()).views == ()
    assert seeing.SightData(cause="change", views=(view,)).views == (view,)
    with pytest.raises(pydantic.ValidationError, match="exactly one view"):
        _ = seeing.SightData(cause="change", views=())
    with pytest.raises(pydantic.ValidationError, match="exactly one view"):
        _ = seeing.SightData(cause="change", views=(view, view))
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.SightData(cause="look", views=(view,) * 5)
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.View(image=_jpeg("a"), age_ms=-1)
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.View(image=_jpeg("a"), age_ms=0, change=1.5)
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.View.model_validate({"image": _jpeg("a"), "age_ms": 0, "labels": []})
    assert view.objects is None and view.faces is None
    found = vision.DetectedObject(label="cup", confidence=0.5, box=_BOX)
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.View(image=_jpeg("a"), age_ms=0, objects=(found,) * (vision.detection.MAX_OBJECTS + 1))
    face = seeing.Face(box=_BOX)
    assert (face.face_id, face.recognized, face.similarity, face.confidence) == (None, None, None, None)
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.View(image=_jpeg("a"), age_ms=0, faces=(face,) * (vision.faces.MAX_FACES + 1))
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.Face(box=_BOX, face_id="Not An Id")
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.Face.model_validate({"box": _BOX, "print": None})
    with pytest.raises(pydantic.ValidationError):
        _ = seeing.LookData.model_validate({"anything": 1})
    for model in (seeing.View, seeing.SightData, seeing.LookData, seeing.Face):
        assert model.model_config.get("frozen") is True


def test_construction_requires_a_change_measure_and_valid_settings() -> None:
    measure = FakeChange()
    with pytest.raises(TypeError):
        _ = typing.cast(typing.Callable[[], object], seeing.Seeing)()
    for threshold in (0.0, -0.1, 1.01):
        with pytest.raises(ValueError, match="change_threshold"):
            _ = seeing.Seeing(change=measure, change_threshold=threshold)
    zero = datetime.timedelta(0)
    invalid: dict[str, typing.Callable[[], seeing.Seeing]] = {
        "measure_every": lambda: seeing.Seeing(change=measure, measure_every=zero),
        "measure_timeout": lambda: seeing.Seeing(change=measure, measure_timeout=zero),
        "settle_after": lambda: seeing.Seeing(change=measure, settle_after=zero),
        "max_view_age": lambda: seeing.Seeing(change=measure, max_view_age=zero),
    }
    for name, build in invalid.items():
        with pytest.raises(ValueError, match=name):
            _ = build()
    with pytest.raises(ValueError, match="measure_every"):
        _ = seeing.Seeing(
            change=measure, measure_every=datetime.timedelta(seconds=2), settle_after=datetime.timedelta(seconds=1)
        )
    with pytest.raises(ValueError, match="stage_timeout"):
        _ = seeing.Seeing(change=measure, stage_timeout=zero)
    with pytest.raises(ValueError, match="restart_timeout"):
        _ = seeing.Seeing(change=measure, restart_timeout=zero)
    with pytest.raises(ValueError, match="face_detector"):
        _ = seeing.Seeing(change=measure, faces=_registry(FakeIdentifier()))
    assert isinstance(seeing.Seeing(change=measure, change_threshold=1.0), abilities.Ability)


# Change path


def test_a_sources_first_view_is_handed_on_as_a_change_sight_under_fresh_ids() -> None:
    async def run() -> tuple[Harness, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        harness = await _harness(clock=ManualClock())
        await harness.frame("camera-1", _jpeg("a"), event_id="visual-1", participants=(_PERSON,))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-1", _jpeg("b"), event_id="visual-2", participants=(_PERSON, _OTHER))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness, harness.sights()

    harness, sights = asyncio.run(run())

    (_, first_handoff, first), (_, second_handoff, second) = sights
    assert first.cause == "change"
    (view,) = first.views
    assert view.image == _jpeg("a")
    assert view.change is None
    assert view.age_ms == 0
    assert view.kind == "camera"
    assert view.participants == (_PERSON,)
    assert first.participants == (_PERSON,)
    assert second.views[0].change == 1.0
    assert second.participants == (_PERSON, _OTHER)
    for handoff, sight_data in ((first_handoff, first), (second_handoff, second)):
        data = handoff.data
        assert isinstance(data, cognition.InputData)
        assert data.participants == sight_data.participants
        sight = _sight_event(handoff)
        assert sight.source == "camera-1"
        assert handoff.source == hsm.id(harness.eye)
    ids = [first_handoff.id, second_handoff.id, _sight_event(first_handoff).id, _sight_event(second_handoff).id]
    assert all(ids) and len(set(ids)) == 4
    assert not {"visual-1", "visual-2"} & set(ids)


def test_a_change_below_the_threshold_is_not_handed_on_and_drift_accumulates() -> None:
    scores = {"b": 0.05, "c": 0.5}

    async def run() -> tuple[FakeChange, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        harness = await _harness(FakeChange(lambda image: scores[_label(image)]))
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.until_calls(2)
        await asyncio.sleep(0.1)
        assert len(harness.sights()) == 1
        await harness.frame("camera-1", _jpeg("c"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.measure, harness.sights()

    measure, sights = asyncio.run(run())

    assert [_label(call.image) for call in measure.calls] == ["a", "b", "c"]
    assert measure.calls[0].previous is None
    # Not handed on: the last handed-on signature stays, so c is measured against a, not b.
    assert measure.calls[1].previous == measure.calls[2].previous == _signature(_jpeg("a"))
    assert [sight.views[0].image for _, _, sight in sights] == [_jpeg("a"), _jpeg("c")]
    assert sights[1][2].views[0].change == 0.5


def test_a_change_at_the_threshold_is_handed_on() -> None:
    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(FakeChange(lambda image: 0.25), change_threshold=0.25)
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.sights()

    assert asyncio.run(run())[1][2].views[0].change == 0.25


def test_measurements_are_paced_whatever_the_frame_rate() -> None:
    every = datetime.timedelta(milliseconds=100)

    async def run() -> tuple[FakeChange, float]:
        harness = await _harness(FakeChange(lambda image: 0.0), measure_every=every, settle_after=every)
        loop = asyncio.get_running_loop()
        started = loop.time()
        for index in range(50):
            await harness.frame("camera-1", _jpeg(f"{index:02d}"))
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        return harness.measure, loop.time() - started

    measure, elapsed = asyncio.run(run())

    measured = [int(_label(call.image)) for call in measure.calls]
    assert 2 <= len(measured) <= math.ceil(elapsed / every.total_seconds()) + 1
    # The newest frame is measured each time: frames in between replaced the held view, none queued.
    assert measured == sorted(set(measured))
    assert len(measured) < 50


def test_sources_are_measured_round_robin() -> None:
    async def run() -> tuple[FakeChange, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        every = datetime.timedelta(milliseconds=40)
        harness = await _harness(measure_every=every, settle_after=every)
        stop = asyncio.Event()

        async def feed() -> None:
            index = 0
            while not stop.is_set():
                await harness.frame("camera-a", _jpeg(f"a{index}"))
                await asyncio.sleep(0.002)
                await harness.frame("camera-b", _jpeg(f"b{index}"))
                await asyncio.sleep(0.018)
                index += 1

        feeder = asyncio.create_task(feed())
        try:
            await harness.until_calls(10)
        finally:
            stop.set()
            await feeder
        return harness.measure, harness.sights()

    measure, sights = asyncio.run(run())

    sources = [_label(call.image)[0] for call in measure.calls[:10]]
    # Never the same source twice in a row while the other has a fresh unmeasured view.
    assert all(left != right for left, right in zip(sources, sources[1:], strict=False))
    assert {_sight_event(handoff).source for _, handoff, _ in sights[:2]} == {"camera-a", "camera-b"}


def test_a_stimulus_without_a_source_is_dropped_and_counted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness()
        await harness.frame("", _jpeg("a"))
        await asyncio.sleep(0.05)
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.seeing.view.count")
        sights = asyncio.run(run())
        after = reader.points("bot.seeing.view.count")

    ((_, _, sight),) = sights
    assert sight.cause == "look" and sight.views == ()
    assert _counts(before, after) == {("image/jpeg", "dropped"): 1}


# No fallback


def test_a_failed_measure_hands_nothing_on_and_a_look_still_sees(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[FakeChange, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        harness = await _harness(FakeChange(fail=True))
        await harness.frame("camera-1", _jpeg("a"))
        await harness.until_calls(1)
        await asyncio.sleep(0.1)
        assert harness.sights() == []
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.measure, harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.seeing.measure.count")
        measure, sights = asyncio.run(run())
        after = reader.points("bot.seeing.measure.count")

    assert len(measure.calls) == 1
    ((_, _, sight),) = sights
    assert sight.cause == "look"
    assert [view.image for view in sight.views] == [_jpeg("a")]
    assert _outcomes(before, after) == {"failed": 1}


def test_a_measure_that_never_answers_ends_by_its_timeout_and_is_cancelled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> tuple[FakeChange, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        harness = await _harness(FakeChange(hang=True), measure_timeout=datetime.timedelta(milliseconds=50))
        await harness.frame("camera-1", _jpeg("a"))
        await harness.until_calls(1)
        async with asyncio.timeout(_WAIT):
            while not harness.measure.calls[0].cancelled:
                await asyncio.sleep(0.005)
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.measure, harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.seeing.measure.count")
        measure, sights = asyncio.run(run())
        after = reader.points("bot.seeing.measure.count")

    assert len(measure.calls) == 1 and measure.calls[0].cancelled
    ((_, _, sight),) = sights
    assert sight.cause == "look" and [view.image for view in sight.views] == [_jpeg("a")]
    assert _outcomes(before, after) == {"timeout": 1}


def test_measurement_outcomes_are_counted(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scores = {"b": 0.05, "c": 0.5}

    async def run() -> None:
        harness = await _harness(FakeChange(lambda image: scores[_label(image)]))
        for label, calls in (("a", 1), ("b", 2), ("c", 3)):
            await harness.frame("camera-1", _jpeg(label))
            await harness.until_calls(calls)
            await asyncio.sleep(0.08)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.seeing.measure.count")
        sights_before = reader.points("bot.seeing.sight.count")
        asyncio.run(run())
        after = reader.points("bot.seeing.measure.count")
        sights_after = reader.points("bot.seeing.sight.count")

    assert _outcomes(before, after) == {"first": 1, "below_threshold": 1, "salient": 1}
    assert _changes(sights_before, sights_after, "cause") == {"change": 2}


# Refractory


def test_salient_frames_within_settle_after_give_one_sight_now_and_one_after_it() -> None:
    settle = datetime.timedelta(milliseconds=300)

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(settle_after=settle)
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.frame("camera-1", _jpeg("c"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        await asyncio.sleep(0.4)
        return harness.sights()

    sights = asyncio.run(run())

    assert len(sights) == 2
    (first_at, _, _), (second_at, _, second) = sights
    assert second_at - first_at >= settle.total_seconds() - 0.02
    assert second.views[0].image == _jpeg("c")


# Look


def test_a_look_hands_on_every_fresh_view_newest_first_under_fresh_ids() -> None:
    async def run() -> tuple[Harness, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        clock = ManualClock()
        harness = await _harness(FakeChange(fail=True), clock=clock)
        await harness.frame("camera-a", _jpeg("a"), participants=(_PERSON,))
        await harness.until_calls(1)
        clock.now = _T0 + 100 * _MS
        await harness.frame("camera-b", _jpeg("b"), participants=(_OTHER, _PERSON), kind="screen")
        await harness.until_calls(2)
        await asyncio.sleep(0.05)
        clock.now = _T0 + 250 * _MS
        await harness.look("turn-1")
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness, harness.sights()

    harness, sights = asyncio.run(run())

    ((_, handoff, sight),) = sights
    assert sight.cause == "look"
    assert [(view.image, view.age_ms, view.change, view.kind) for view in sight.views] == [
        (_jpeg("b"), 150, None, "screen"),
        (_jpeg("a"), 250, None, "camera"),
    ]
    assert sight.participants == (_OTHER, _PERSON)
    event = _sight_event(handoff)
    # A selected look carries the selecting turn's operation id; the sight never reuses it.
    assert "turn-1" not in {handoff.id, event.id}
    assert handoff.id and event.id and handoff.id != event.id
    assert event.source == hsm.id(harness.eye)


def test_a_look_leaves_the_last_handed_on_signature_alone() -> None:
    async def run() -> FakeChange:
        harness = await _harness(FakeChange(lambda image: 0.0))
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.until_calls(2)
        await asyncio.sleep(0.05)
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights("look")) == 1)
        await asyncio.sleep(0.1)
        await harness.frame("camera-1", _jpeg("c"))
        await harness.until_calls(3)
        return harness.measure

    measure = asyncio.run(run())

    assert measure.calls[2].previous == _signature(_jpeg("a"))


def test_a_look_with_nothing_or_only_stale_views_in_view_sees_nothing() -> None:
    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        clock = ManualClock()
        harness = await _harness(FakeChange(fail=True), clock=clock, max_view_age=datetime.timedelta(seconds=1))
        await harness.look("empty")
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-1", _jpeg("a"))
        await harness.until_calls(1)
        await asyncio.sleep(0.05)
        clock.now = _T0 + datetime.timedelta(seconds=2)
        await harness.look("stale")
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.sights()

    assert [sight.views for _, _, sight in asyncio.run(run())] == [(), ()]


def test_a_look_while_settling_is_served_when_it_ends_then_settles_in_full() -> None:
    settle = datetime.timedelta(milliseconds=250)

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(settle_after=settle)
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 3)
        return harness.sights()

    (first_at, _, first), (look_at, _, look), (next_at, _, after) = asyncio.run(run())

    assert (first.cause, look.cause, after.cause) == ("change", "look", "change")
    assert look_at - first_at >= settle.total_seconds() - 0.02
    assert next_at - look_at >= settle.total_seconds() - 0.02


def test_a_look_while_pacing_is_served_at_once() -> None:
    async def run() -> float:
        long = datetime.timedelta(seconds=2)
        harness = await _harness(FakeChange(fail=True), measure_every=long, settle_after=long)
        await harness.frame("camera-1", _jpeg("a"))
        await harness.until_calls(1)
        await asyncio.sleep(0.05)
        loop = asyncio.get_running_loop()
        asked = loop.time()
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()[0][0] - asked

    assert asyncio.run(run()) < 0.5


def test_a_look_is_served_under_continuous_change() -> None:
    settle = datetime.timedelta(milliseconds=60)

    async def run() -> tuple[float, float]:
        harness = await _harness(settle_after=settle)
        stop = asyncio.Event()

        async def feed() -> None:
            index = 0
            while not stop.is_set():
                await harness.frame("camera-1", _jpeg(f"{index}"))
                await asyncio.sleep(0.01)
                index += 1

        feeder = asyncio.create_task(feed())
        try:
            await harness.owner.until(lambda: len(harness.sights()) >= 2)
            loop = asyncio.get_running_loop()
            asked = loop.time()
            await harness.look()
            await harness.owner.until(lambda: len(harness.sights("look")) == 1)
            answered = harness.sights("look")[0][0]
        finally:
            stop.set()
            await feeder
        return asked, answered

    asked, answered = asyncio.run(run())

    assert answered - asked < settle.total_seconds() + 0.5


def test_looks_while_settling_coalesce_into_one_look() -> None:
    settle = datetime.timedelta(milliseconds=200)

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(FakeChange(lambda image: 0.0), settle_after=settle)
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        for index in range(3):
            await harness.look(f"turn-{index}")
        await asyncio.sleep(0.6)
        return harness.sights("look")

    assert len(asyncio.run(run())) == 1


def test_looks_count_the_views_they_hand_on(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        harness = await _harness(FakeChange(fail=True))
        await harness.frame("camera-a", _jpeg("a"))
        await harness.frame("camera-b", _jpeg("b"))
        await harness.until_calls(2)
        await asyncio.sleep(0.05)
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 1)

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.seeing.look.view.count")
        sights_before = reader.points("bot.seeing.sight.count")
        asyncio.run(run())
        after = reader.points("bot.seeing.look.view.count")
        sights_after = reader.points("bot.seeing.sight.count")

    assert sum(point.count for point in after) - sum(point.count for point in before) == 1
    assert sum(point.sum for point in after) - sum(point.sum for point in before) == 2
    assert _changes(sights_before, sights_after, "cause") == {"look": 1}


# Bounds


def test_a_fifth_source_evicts_the_least_recently_received_view(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(FakeChange(fail=True))
        for index in range(5):
            await harness.frame(f"camera-{index}", _jpeg(f"{index}"))
        await asyncio.sleep(0.05)
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.seeing.view.count")
        sights = asyncio.run(run())
        after = reader.points("bot.seeing.view.count")

    ((_, _, sight),) = sights
    assert sorted(_label(view.image) for view in sight.views) == ["1", "2", "3", "4"]
    assert _counts(before, after) == {("image/jpeg", "held"): 5, ("image/jpeg", "evicted"): 1}


def test_a_stale_view_is_expired_when_the_next_frame_arrives(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        clock = ManualClock()
        harness = await _harness(FakeChange(fail=True), clock=clock, max_view_age=datetime.timedelta(seconds=1))
        await harness.frame("camera-a", _jpeg("a"))
        await harness.until_calls(1)
        clock.now = _T0 + datetime.timedelta(seconds=2)
        await harness.frame("camera-b", _jpeg("b"))
        await harness.until_calls(2)
        await asyncio.sleep(0.05)
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.seeing.view.count")
        sights = asyncio.run(run())
        after = reader.points("bot.seeing.view.count")

    ((_, _, sight),) = sights
    assert [view.image for view in sight.views] == [_jpeg("b")]
    assert _counts(before, after) == {("image/jpeg", "held"): 2, ("image/jpeg", "expired"): 1}


def test_detaching_clears_what_the_eye_holds() -> None:
    async def run() -> tuple[list[tuple[float, hsm.Event[typing.Any], seeing.SightData]], FakeChange]:
        harness = await _harness(FakeChange(lambda image: 0.0), settle_after=datetime.timedelta(milliseconds=300))
        await harness.frame("camera-0", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.look("pending")
        await harness.detached()
        await harness.attached("reattach")
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        await harness.look("after")
        await harness.owner.until(lambda: len(harness.sights()) == 3)
        await harness.frame("camera-0", _jpeg("a2"))
        await harness.until_calls(3)
        return harness.sights(), harness.measure

    sights, measure = asyncio.run(run())

    # No look survives the detach, no view is still held, and every source starts over.
    assert [sight.cause for _, _, sight in sights[:3]] == ["change", "change", "look"]
    assert [view.image for view in sights[2][2].views] == [_jpeg("b")]
    assert [_label(call.image) for call in measure.calls] == ["a", "b", "a2"]
    assert measure.calls[2].previous is None


# Offering and attachments


def test_look_is_offered_only_while_perceiving() -> None:
    async def run() -> tuple[set[str], set[str], set[str]]:
        probe = Probe(hold_attach=True)
        harness = await _harness(FakeChange(attachments=(probe,)), started=(probe,), attach=False)
        await harness.attach()
        async with asyncio.timeout(_WAIT):
            while not probe.attach_calls:
                await asyncio.sleep(0.001)
        initializing = {event.name for event in processing.enabled_call_events(harness.eye)}
        probe.attach_release.set()
        await harness.owner.until(lambda: bool(harness.owner.named(attachment.AttachCompleteEvent)))
        perceiving = {event.name for event in processing.enabled_call_events(harness.eye)}
        await harness.detached()
        detached = {event.name for event in processing.enabled_call_events(harness.eye)}
        return initializing, perceiving, detached

    initializing, perceiving, detached = asyncio.run(run())

    assert seeing.LookEvent.name not in initializing
    assert perceiving == {seeing.LookEvent.name}
    assert seeing.LookEvent.name not in detached


def test_the_change_measures_attachments_are_attached_for_the_eyes_attached_lifetime() -> None:
    async def run() -> tuple[int, int, int]:
        probe = Probe()
        harness = await _harness(FakeChange(attachments=(probe,)), started=(probe,))
        attached = (len(probe.attach_calls), len(probe.detach_calls))
        await harness.detached()
        return attached[0], attached[1], len(probe.detach_calls)

    assert asyncio.run(run()) == (1, 0, 1)


# Stages


def test_stages_fill_each_views_objects_and_faces_in_contract_order(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = FakeObjects(_found("cup", 0.5), _found("person", 0.9))
    faces = FakeFaces(_detected_face(0.1), _detected_face(0.4))

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(object_detector=objects, face_detector=faces)
        await harness.frame("camera-1", _jpeg("a"), participants=(_PERSON,))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        objects.found, faces.found = (), ()
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in _STAGE_METRICS}
        sights = asyncio.run(run())
        after = {name: reader.points(name) for name in _STAGE_METRICS}

    (_, _, first), (_, _, second) = sights
    (view,) = first.views
    assert view.objects is not None and [found.label for found in view.objects] == ["person", "cup"]
    assert view.faces == (
        seeing.Face(box=vision.Box(x=0.0, y=0.0, width=0.4, height=0.4), confidence=0.9),
        seeing.Face(box=vision.Box(x=0.0, y=0.0, width=0.1, height=0.1), confidence=0.9),
    )
    # No registry composed: no face is identified and no face participant is stamped.
    assert first.participants == (_PERSON,)
    assert (second.views[0].objects, second.views[0].faces) == ((), ())
    assert [call.image for call in objects.script.calls] == [_jpeg("a"), _jpeg("b")]
    assert _delta(before, after, "bot.seeing.stage.count") == {
        (("bot.outcome", "ok"), ("stage", "faces")): 2,
        (("bot.outcome", "ok"), ("stage", "objects")): 2,
    }
    assert _delta(before, after, "bot.seeing.face.count") == {(("recognized", "none"),): 2}
    histogram = [point for point in after["bot.seeing.object.count"] if point.count]
    assert (
        sum(point.count for point in histogram) - sum(point.count for point in before["bot.seeing.object.count"]) == 2
    )


def test_stages_run_only_on_views_handed_on_and_on_every_view_a_look_hands_on() -> None:
    objects = FakeObjects()

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(FakeChange(lambda image: 0.0), object_detector=objects)
        await harness.frame("camera-a", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-b", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        await harness.frame("camera-a", _jpeg("a2"))
        await harness.until_calls(3)
        await asyncio.sleep(0.1)
        assert len(objects.script.calls) == 2
        await harness.look()
        await harness.owner.until(lambda: len(harness.sights("look")) == 1)
        return harness.sights("look")

    ((_, _, look),) = asyncio.run(run())

    assert [call.image for call in objects.script.calls[2:]] == [_jpeg("a2"), _jpeg("b")]
    assert [view.objects for view in look.views] == [(), ()]


def test_identified_faces_are_stamped_as_face_participants() -> None:
    identifier = FakeIdentifier(
        identification.Recognition(id="a", recognized=True, similarity=0.9, own=False),
        identification.Recognition(id="b", recognized=False, similarity=None, own=False),
    )
    first, second = _print(1.0, 0.0), _print(0.0, 1.0)
    faces = FakeFaces(_detected_face(0.2, second), _detected_face(0.4, first))

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(face_detector=faces, faces=_registry(identifier))
        await harness.frame("camera-1", _jpeg("a"), participants=(_PERSON,))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()

    ((_, handoff, sight),) = asyncio.run(run())

    assert identifier.identified == [(first, second)]
    (view,) = sight.views
    assert view.faces is not None
    assert [(face.face_id, face.recognized, face.similarity) for face in view.faces] == [
        ("a", True, 0.9),
        ("b", False, None),
    ]
    assert sight.participants == (
        _PERSON,
        mosfet.Participant(ref="face:a", role=mosfet.ParticipantRole.SOURCE),
        mosfet.Participant(ref="face:b", role=mosfet.ParticipantRole.SOURCE),
    )
    data = handoff.data
    assert isinstance(data, cognition.InputData)
    assert data.participants == sight.participants


def test_a_face_without_a_print_or_a_recognition_has_no_id_and_no_participant() -> None:
    identifier = FakeIdentifier(None, identification.Recognition(id="c", recognized=True, similarity=0.7, own=False))
    large, medium = _print(1.0, 0.0), _print(0.0, 1.0)
    faces = FakeFaces(_detected_face(0.5, large), _detected_face(0.1), _detected_face(0.3, medium))

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(face_detector=faces, faces=_registry(identifier))
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()

    ((_, _, sight),) = asyncio.run(run())

    # Only printed faces are identified, in face order.
    assert identifier.identified == [(large, medium)]
    assert sight.views[0].faces is not None
    assert [(face.face_id, face.recognized, face.similarity) for face in sight.views[0].faces] == [
        (None, None, None),
        ("c", True, 0.7),
        (None, None, None),
    ]
    assert sight.participants == (mosfet.Participant(ref="face:c", role=mosfet.ParticipantRole.SOURCE),)


def test_no_print_vector_or_signature_reaches_a_sight() -> None:
    vector = (0.123456, 0.654321)
    face_print = _print(*vector)
    identifier = FakeIdentifier(identification.Recognition(id="a", recognized=False, similarity=None, own=False))

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(
            object_detector=FakeObjects(_found("cup", 0.5)),
            face_detector=FakeFaces(_detected_face(0.3, face_print)),
            faces=_registry(identifier),
        )
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.sights()

    sights = asyncio.run(run())

    numbers = {round(value, 4) for value in (*vector, *face_print.vector)}

    def walk(value: object) -> collections.abc.Iterator[tuple[str, object]]:
        if isinstance(value, dict):
            for key, item in typing.cast(dict[str, object], value).items():
                yield key, item
                yield from walk(item)
        elif isinstance(value, (list, tuple)):
            for item in typing.cast(collections.abc.Sequence[object], value):
                yield "", item
                yield from walk(item)

    for _, handoff, sight in sights:
        dumped = list(walk(sight.model_dump()))
        assert not {"print", "vector", "signature", "digest", "values"} & {key for key, _ in dumped}
        floats = {round(item, 4) for _, item in dumped if isinstance(item, float)}
        assert not numbers & floats
        assert "Signature" not in repr(handoff.data)


def test_a_stage_failure_or_timeout_leaves_only_its_own_field_empty(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        failing = await _harness(
            object_detector=FakeObjects(script=Script(fail=True)), face_detector=FakeFaces(_detected_face(0.2))
        )
        await failing.frame("camera-1", _jpeg("a"))
        await failing.owner.until(lambda: len(failing.sights()) == 1)
        stuck = FakeFaces(script=Script(hang_first=True))
        hanging = await _harness(
            object_detector=FakeObjects(_found("cup", 0.5)),
            face_detector=stuck,
            stage_timeout=datetime.timedelta(milliseconds=100),
        )
        await hanging.frame("camera-1", _jpeg("a"))
        await hanging.owner.until(lambda: len(hanging.sights()) == 1)
        # The next sight comes after the stuck stage's recovery: nothing is left in flight at the end.
        await hanging.frame("camera-1", _jpeg("b"))
        await hanging.owner.until(lambda: len(hanging.sights()) == 2)
        assert stuck.script.calls[0].cancelled
        return failing.sights() + hanging.sights()[:1]

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in _STAGE_METRICS}
        (_, _, failed), (_, _, timed_out) = asyncio.run(run())
        after = {name: reader.points(name) for name in _STAGE_METRICS}

    assert failed.views[0].objects is None
    assert failed.views[0].faces == (seeing.Face(box=vision.Box(x=0.0, y=0.0, width=0.2, height=0.2), confidence=0.9),)
    assert timed_out.views[0].faces is None
    assert timed_out.views[0].objects is not None and [found.label for found in timed_out.views[0].objects] == ["cup"]
    assert _delta(before, after, "bot.seeing.stage.count") == {
        (("bot.outcome", "failed"), ("stage", "objects")): 1,
        (("bot.outcome", "ok"), ("stage", "faces")): 2,
        (("bot.outcome", "ok"), ("stage", "objects")): 2,
        (("bot.outcome", "timeout"), ("stage", "faces")): 1,
    }


def test_a_stuck_stage_is_detached_and_reattached_and_serves_again(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _record_detections(monkeypatch)
    objects = FakeObjects(_found("cup", 0.5), script=Script(hang_first=True))

    async def run() -> tuple[Harness, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        harness = await _harness(object_detector=objects, stage_timeout=datetime.timedelta(milliseconds=100))
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        async with asyncio.timeout(_WAIT):
            while len(built[0].attaches()) < 2:
                await asyncio.sleep(0.005)
        await asyncio.sleep(0.1)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness, harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in _STAGE_METRICS}
        harness, sights = asyncio.run(run())
        after = {name: reader.points(name) for name in _STAGE_METRICS}

    (_, _, stuck), (_, _, served) = sights
    assert stuck.views[0].objects is None
    assert served.views[0].objects is not None and served.views[0].objects[0].label == "cup"
    # The stuck call's context was cancelled when the detach left attached.
    assert objects.script.calls[0].cancelled
    (child,) = built
    group_attach, detach, reattach = child.lifecycle
    assert isinstance(group_attach.data, attachment.AttachData)
    assert isinstance(detach.data, attachment.DetachData) and detach.data.actor is harness.eye
    assert isinstance(reattach.data, attachment.AttachData) and reattach.data.actor is harness.eye
    # Each recovery request names its own one-shot operation as its reply address.
    assert detach.data.reply_to is not None and reattach.data.reply_to is not None
    assert detach.data.reply_to is not reattach.data.reply_to
    assert _delta(before, after, "bot.seeing.stage.restart.count") == {(("bot.outcome", "ok"), ("stage", "objects")): 1}


def test_a_stage_whose_recovery_fails_is_skipped_until_the_eye_is_reattached(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Attach 1 is the group's; attach 2 is the recovery's and is refused; attach 3 is the reattach.
    built = _record_detections(monkeypatch, refuse_attach=2)
    objects = FakeObjects(_found("cup", 0.5), script=Script(hang_first=True))
    settle = datetime.timedelta(milliseconds=300)

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(
            object_detector=objects, stage_timeout=datetime.timedelta(milliseconds=100), settle_after=settle
        )
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        async with asyncio.timeout(_WAIT):
            while len(built[0].attaches()) < 2:
                await asyncio.sleep(0.005)
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        assert len(objects.script.calls) == 1
        # A look left pending and an unavailable stage do not survive a detach.
        await harness.look("pending")
        await harness.detached()
        await harness.attached("reattach")
        await harness.frame("camera-1", _jpeg("c"))
        await harness.owner.until(lambda: len(harness.sights()) == 3)
        await asyncio.sleep(settle.total_seconds() + 0.1)
        return harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in _STAGE_METRICS}
        sights = asyncio.run(run())
        after = {name: reader.points(name) for name in _STAGE_METRICS}

    assert [sight.cause for _, _, sight in sights] == ["change", "change", "change"]
    assert [sight.views[0].objects for _, _, sight in sights[:2]] == [None, None]
    assert sights[2][2].views[0].objects is not None and sights[2][2].views[0].objects[0].label == "cup"
    assert [call.image for call in objects.script.calls] == [_jpeg("a"), _jpeg("c")]
    # The reattach was not refused as a conflict: the stage's attachments stayed consistent.
    assert len(built[0].attaches()) == 3
    assert _delta(before, after, "bot.seeing.stage.restart.count") == {
        (("bot.outcome", "failed"), ("stage", "objects")): 1
    }


def test_a_recovery_that_never_finishes_is_cut_by_its_backstop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _record_detections(monkeypatch, swallow_attach=2)
    objects = FakeObjects(_found("cup", 0.5), script=Script(hang_first=True))
    restart = datetime.timedelta(milliseconds=50)

    async def run() -> tuple[float, list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]]:
        harness = await _harness(
            object_detector=objects, stage_timeout=datetime.timedelta(milliseconds=100), restart_timeout=restart
        )
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        async with asyncio.timeout(_WAIT):
            while len(built[0].attaches()) < 2:
                await asyncio.sleep(0.002)
        swallowed = asyncio.get_running_loop().time()
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.sights()[1][0] - swallowed, harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in _STAGE_METRICS}
        waited, sights = asyncio.run(run())
        after = {name: reader.points(name) for name in _STAGE_METRICS}

    # Two stages could be stuck: a detach and an attach each, every step within restart_timeout.
    assert waited >= 4 * restart.total_seconds() - 0.1
    assert sights[1][2].views[0].objects is None
    assert len(objects.script.calls) == 1
    assert _delta(before, after, "bot.seeing.stage.restart.count") == {
        (("bot.outcome", "failed"), ("stage", "objects")): 1
    }


def test_a_look_during_interpretation_is_served_after_it() -> None:
    gate = asyncio.Event()
    objects = FakeObjects(_found("cup", 0.5), script=Script(gate=gate))

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(object_detector=objects)
        await harness.frame("camera-1", _jpeg("a"))
        async with asyncio.timeout(_WAIT):
            while not objects.script.calls:
                await asyncio.sleep(0.002)
        await harness.look()
        await asyncio.sleep(0.05)
        gate.set()
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.sights()

    (_, _, change_sight), (_, _, look) = asyncio.run(run())

    assert change_sight.cause == "change"
    assert look.cause == "look"
    assert [view.image for view in look.views] == [_jpeg("a")]
    assert [view.objects for view in look.views] == [(_found("cup", 0.5),)]
    assert len(objects.script.calls) == 2


def test_a_face_registry_of_another_kind_is_an_invalid_response(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(span, "tracer_provider", lambda: provider)
    identifier = FakeIdentifier(identification.Recognition(id="v", recognized=True, similarity=0.9, own=False))

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(
            face_detector=FakeFaces(_detected_face(0.3, _print(1.0, 0.0))), faces=_registry(identifier, "voice")
        )
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = {name: reader.points(name) for name in _STAGE_METRICS}
        ((_, _, sight),) = asyncio.run(run())
        after = {name: reader.points(name) for name in _STAGE_METRICS}

    assert sight.views[0].faces is not None
    assert [(face.face_id, face.recognized) for face in sight.views[0].faces] == [(None, None)]
    assert sight.participants == ()
    assert _delta(before, after, "bot.seeing.stage.count") == {
        (("bot.outcome", "ok"), ("stage", "faces")): 1,
        (("bot.outcome", "failed"), ("stage", "identification")): 1,
    }
    (interpret,) = [recorded for recorded in exporter.get_finished_spans() if recorded.name == "bot.seeing.interpret"]
    attributes = dict(interpret.attributes or {})
    assert attributes["bot.failure.kind"] == "invalid_response"
    assert attributes["bot.seeing.stage"] == "identification"
    assert all(isinstance(value, (bool, int, str)) for value in attributes.values())


def test_the_composition_root_can_forget_a_face_the_eye_enrolled() -> None:
    face_print = _print(0.6, 0.8)
    registry = identification.Registry(
        identifier=identification.Matcher(
            store=identification.InProcessStore(max_ids=10),
            policy=identification.MatchPolicy(threshold=0.5, margin=0.05, refine_threshold=0.65),
        ),
        kind="face",
        max_unlinked_age=datetime.timedelta(days=30),
        max_ids=10,
    )

    async def run() -> tuple[list[tuple[float, hsm.Event[typing.Any], seeing.SightData]], hsm.Event[typing.Any]]:
        harness = await _harness(face_detector=FakeFaces(_detected_face(0.3, face_print)), faces=registry)
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        faces = harness.sights()[0][2].views[0].faces
        assert faces is not None and faces[0].face_id is not None
        forgotten = await abilities.run_terminal_operation(
            harness.ctx,
            child=registry,
            request=identification.ForgetEvent.with_data_and_id(
                identification.ForgetData(id=faces[0].face_id), "forget-1"
            ),
            terminals=(identification.ForgottenEvent, identification.FailedEvent),
            timeout=datetime.timedelta(seconds=2),
        )
        await harness.frame("camera-1", _jpeg("b"))
        await harness.owner.until(lambda: len(harness.sights()) == 2)
        return harness.sights(), forgotten

    sights, forgotten = asyncio.run(run())

    first, second = (sight.views[0].faces for _, _, sight in sights)
    assert first is not None and second is not None
    assert forgotten.data == identification.ForgottenData(id=first[0].face_id or "", forgotten=True)
    assert (first[0].recognized, second[0].recognized) == (False, False)
    assert second[0].face_id is not None and second[0].face_id != first[0].face_id


class FakeBackend:
    """One model that serves two roles: it detects objects and faces from the same call."""

    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    async def look(self, role: str) -> None:
        self.calls.append(role)


class FakeObjectAdapter(vision.detection.ObjectDetector):
    backend: FakeBackend

    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend

    @typing.override
    async def classify(self, input: vision.detection.InputData) -> vision.detection.OutputData:
        del input
        await self.backend.look("objects")
        return vision.detection.OutputData(objects=(_found("lamp", 0.8),))


class FakeFaceAdapter(vision.faces.FaceDetector):
    backend: FakeBackend

    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend

    @typing.override
    async def classify(self, input: vision.faces.InputData) -> vision.faces.OutputData:
        del input
        await self.backend.look("faces")
        return vision.faces.OutputData(faces=(_detected_face(0.2),))


def test_one_backend_serves_both_stages_through_per_role_adapters() -> None:
    backend = FakeBackend()

    async def run() -> list[tuple[float, hsm.Event[typing.Any], seeing.SightData]]:
        harness = await _harness(object_detector=FakeObjectAdapter(backend), face_detector=FakeFaceAdapter(backend))
        await harness.frame("camera-1", _jpeg("a"))
        await harness.owner.until(lambda: len(harness.sights()) == 1)
        return harness.sights()

    ((_, _, sight),) = asyncio.run(run())

    assert sorted(backend.calls) == ["faces", "objects"]
    (view,) = sight.views
    assert view.objects is not None and view.objects[0].label == "lamp"
    assert view.faces is not None and len(view.faces) == 1


# Metric helpers


def _counts(
    before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...]
) -> dict[tuple[str, str], float]:
    def key(point: metric_reader.Point) -> tuple[str, str]:
        assert set(point.attributes) == {"media_type", "bot.outcome"}
        return str(point.attributes["media_type"]), str(point.attributes["bot.outcome"])

    previous = {key(point): point.value for point in before}
    changes = {key(point): point.value - previous.get(key(point), 0.0) for point in after}
    return {name: value for name, value in changes.items() if value}


def _changes(
    before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...], attribute: str
) -> dict[str, float]:
    previous = {str(point.attributes.get(attribute)): point.value for point in before}
    changes = {
        str(point.attributes.get(attribute)): point.value - previous.get(str(point.attributes.get(attribute)), 0.0)
        for point in after
    }
    assert all(set(point.attributes) == {attribute} for point in after)
    return {name: value for name, value in changes.items() if value}


_STAGE_METRICS = (
    "bot.seeing.stage.count",
    "bot.seeing.object.count",
    "bot.seeing.face.count",
    "bot.seeing.stage.restart.count",
)


def _delta(
    before: dict[str, tuple[metric_reader.Point, ...]],
    after: dict[str, tuple[metric_reader.Point, ...]],
    name: str,
) -> dict[tuple[tuple[str, object], ...], float]:
    def key(point: metric_reader.Point) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(point.attributes.items()))

    previous = {key(point): point.value for point in before[name]}
    changes = {key(point): point.value - previous.get(key(point), 0.0) for point in after[name]}
    return {attributes: value for attributes, value in changes.items() if value}


def _outcomes(before: tuple[metric_reader.Point, ...], after: tuple[metric_reader.Point, ...]) -> dict[str, float]:
    return _changes(before, after, "bot.outcome")
