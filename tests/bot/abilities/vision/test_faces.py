"""``vision.faces``: detected faces with optional prints and the ``FaceDetection`` wrapper."""

import asyncio
import collections.abc
import datetime
import json
import pathlib
import typing

import hsm
import mosfet
import mosfet.telemetry
import pydantic
import pytest
from opentelemetry import _logs, trace

from mosfet import abilities, environment, media
from mosfet import event as event_contract
from mosfet.abilities import identification, vision
from mosfet.abilities.vision import faces
from mosfet.protocols import attachment
from mosfet.telemetry.configure import logger_provider, tracer_provider

from tests.bot.abilities.support import start_abilities_for_test
from tests.type_helpers import invalid_value, model_view

_JPEG = media.Image(content=b"\xff\xd8\xff" + b"faces", media_type="image/jpeg")
_VECTOR = (0.1234567, 0.7654321, 0.4567891)
_PRINT = identification.Print(vector=_VECTOR, model="face-model")
_ENV = (
    "BOT_OTEL_DISABLED",
    "BOT_OTEL_LOG_FILE",
    "BOT_OTEL_SPAN_FILE",
    "BOT_OTEL_CAPTURE",
    "BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "BOT_OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


def _face(width: float, height: float, print: identification.Print | None = None) -> faces.DetectedFace:
    return faces.DetectedFace(box=vision.Box(x=0.0, y=0.0, width=width, height=height), confidence=0.9, print=print)


class FixedFaceDetector(faces.FaceDetector):
    found: tuple[faces.DetectedFace, ...]

    def __init__(self, found: tuple[faces.DetectedFace, ...]) -> None:
        self.found = found

    @typing.override
    async def classify(self, input: faces.InputData) -> faces.OutputData:
        del input
        return faces.OutputData(faces=self.found)


class FailingFaceDetector(faces.FaceDetector):
    @typing.override
    async def classify(self, input: faces.InputData) -> faces.OutputData:
        del input
        raise RuntimeError("face detector unavailable")


class WrongFaceDetector(faces.FaceDetector):
    @typing.override
    async def classify(self, input: faces.InputData) -> faces.OutputData:
        del input
        return invalid_value(faces.OutputData, ())


async def _detect(ctx: hsm.Context, detector: faces.FaceDetector) -> hsm.Event[typing.Any]:
    child = faces.FaceDetection(classifier=detector)
    await start_abilities_for_test(ctx, child)
    return await abilities.run_terminal_operation(
        ctx,
        child=child,
        request=child.input_event.with_data_and_id(faces.InputData(image=_JPEG), "faces-1"),
        terminals=(child.output_event, child.failed_event),
        timeout=datetime.timedelta(seconds=2),
    )


def test_vision_exports_the_faces_module_with_its_own_maximum() -> None:
    assert vision.faces is faces
    assert "faces" in vision.__all__
    assert faces.MAX_FACES == 16
    assert not hasattr(vision, "MAX_FACES")


def test_detected_faces_are_bounded_frozen_values_that_may_carry_a_print() -> None:
    face = _face(0.2, 0.3, _PRINT)

    assert face.print == _PRINT
    assert _face(0.2, 0.3).print is None
    for model in (faces.InputData, faces.DetectedFace, faces.OutputData):
        assert model.model_config.get("frozen") is True
    box = vision.Box(x=0.0, y=0.0, width=0.1, height=0.1)
    for invalid in (
        {"box": box, "confidence": 1.5, "print": None},
        {"box": box, "confidence": 0.5, "print": None, "id": "face-1"},
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = faces.DetectedFace.model_validate(invalid)
    with pytest.raises(pydantic.ValidationError):
        _ = faces.OutputData(faces=(face,) * (faces.MAX_FACES + 1))
    assert faces.OutputData(faces=()).faces == ()


def test_detected_faces_are_ordered_by_descending_box_area() -> None:
    small, large, medium = _face(0.1, 0.1), _face(0.5, 0.5), _face(0.3, 0.2)

    assert faces.OutputData(faces=(small, large, medium)).faces == (large, medium, small)


def test_face_detection_answers_with_the_detectors_faces() -> None:
    async def run() -> hsm.Event[typing.Any]:
        return await _detect(hsm.Context(), FixedFaceDetector((_face(0.1, 0.1), _face(0.4, 0.4, _PRINT))))

    terminal = asyncio.run(run())

    assert terminal.name == "bot.ability.vision.faces.output"
    assert terminal.id == "faces-1"
    assert isinstance(terminal.data, faces.OutputData)
    assert [face.print for face in terminal.data.faces] == [_PRINT, None]


@pytest.mark.parametrize(
    ("detector", "message"),
    [(FailingFaceDetector(), "face detector unavailable"), (WrongFaceDetector(), "output schema")],
    ids=["failure", "invalid-output"],
)
def test_face_detection_answers_failures_with_its_failed_event(detector: faces.FaceDetector, message: str) -> None:
    async def run() -> hsm.Event[typing.Any]:
        return await _detect(hsm.Context(), detector)

    terminal = asyncio.run(run())

    assert terminal.name == abilities.FailedEvent.name
    assert isinstance(terminal.data, abilities.FailureData)
    assert message in terminal.data.message


def test_face_detection_events_are_named_in_the_vision_domain() -> None:
    assert faces.FaceDetection.input_event.name == "bot.ability.vision.faces.input"
    assert faces.FaceDetection.output_event.name == "bot.ability.vision.faces.output"
    assert faces.FaceDetection.input_data_type is faces.InputData
    assert faces.FaceDetection.output_data_type is faces.OutputData
    model = model_view(faces.FaceDetection.model)
    behavior = "/FaceDetectionLifecycle/attached/behavior"
    assert "bot.ability.vision.faces.input" in model.transition_map[f"{behavior}/idle"]
    assert "bot.ability.vision.faces.apply.completed" in model.transition_map[f"{behavior}/applying"]
    assert "bot.ability.vision.faces.apply.failed" in model.transition_map[f"{behavior}/applying"]
    assert not hsm.kind.Is(faces.FaceDetection.input_event.kind, event_contract.EventKind)


# Full capture never writes a face print.


def _install_nothing(provider: object) -> None:
    del provider


@pytest.fixture
def full_capture(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> collections.abc.Iterator[None]:
    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", _install_nothing)
    monkeypatch.setattr(trace, "set_tracer_provider", _install_nothing)
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True
    yield
    mosfet.telemetry.reset()


class _Owner(hsm.Instance):
    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "FacesCaptureOwner",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
        hsm.observe(mosfet.telemetry.observer),
    )


def _bodies() -> list[tuple[str, str]]:
    """Every full-capture record as (event name, serialized body)."""

    for provider in (logger_provider(), tracer_provider()):
        if provider is not None:
            _ = provider.force_flush()
    path = pathlib.Path("otel-logs.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    found: list[tuple[str, str]] = []
    for line in lines:
        if not line.strip():
            continue
        record = typing.cast(dict[str, object], json.loads(line))
        attributes = typing.cast(dict[str, object], record["attributes"])
        found.append((str(attributes.get("hsm.event.name")), json.dumps(record["body"])))
    return found


def test_full_capture_of_face_detection_and_identification_never_writes_a_print(full_capture: None) -> None:
    del full_capture

    async def run() -> tuple[hsm.Event[typing.Any], hsm.Event[typing.Any]]:
        scope = environment.Environment()
        detected = await _detect(scope, FixedFaceDetector((_face(0.4, 0.4, _PRINT),)))
        assert isinstance(detected.data, faces.OutputData)
        owner = _Owner()
        _ = await mosfet.started(scope, owner, owner.model)
        registry = identification.Registry(
            identifier=identification.Matcher(
                store=identification.InProcessStore(max_ids=8),
                policy=identification.MatchPolicy(threshold=0.5, margin=0.05, refine_threshold=0.65),
            ),
            kind="face",
            max_unlinked_age=datetime.timedelta(days=30),
            max_ids=8,
        )
        _ = await registry.attach(
            scope, attachment.AttachEvent.with_data_and_id(attachment.AttachData(actor=owner), "attach")
        )
        prints = tuple(face.print for face in detected.data.faces if face.print is not None)
        identified = await abilities.run_terminal_operation(
            scope,
            child=registry,
            request=identification.IdentifyEvent.with_data_and_id(identification.IdentifyData(prints=prints), "id-1"),
            terminals=(identification.IdentifiedEvent, identification.FailedEvent),
            timeout=datetime.timedelta(seconds=2),
        )
        return detected, identified

    detected, identified = asyncio.run(run())

    assert detected.name == faces.FaceDetection.output_event.name
    assert identified.name == identification.IdentifiedEvent.name
    bodies = _bodies()
    assert any("<biometric:Print>" in body for name, body in bodies if name == faces.FaceDetection.output_event.name)
    assert any("<biometric:Print>" in body for name, body in bodies if name == identification.IdentifyEvent.name)
    numbers = {f"{value:.6f}"[:7] for value in (*_VECTOR, *_PRINT.vector)}
    assert not [name for name, body in bodies for number in numbers if number in body]
