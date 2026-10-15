"""``vision.detection``: detected objects and the ``ObjectDetection`` wrapper over an injected detector."""

import asyncio
import collections.abc
import datetime
import json
import pathlib
import typing

import hsm
import pydantic
import pytest

from mosfet import abilities, media
from mosfet import event as event_contract
from mosfet.abilities import vision
from mosfet.abilities.language import text
from mosfet.abilities.vision import detection

from tests import metric_reader
from tests.bot.abilities.support import start_abilities_for_test
from tests.type_helpers import invalid_value, model_view

_JPEG = media.Image(content=b"\xff\xd8\xff" + b"room", media_type="image/jpeg")
_PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"room", media_type="image/png")
_BOX = vision.Box(x=0.1, y=0.2, width=0.3, height=0.4)
_SEES_JPEG = media.ImageCapacity(max_images=1, max_total_bytes=1024, media_types=frozenset({"image/jpeg"}))


def _object(label: str, confidence: float | None) -> vision.DetectedObject:
    return vision.DetectedObject(label=label, confidence=confidence, box=_BOX)


class FixedDetector(detection.ObjectDetector):
    inputs: list[detection.InputData]
    objects: tuple[vision.DetectedObject, ...]

    def __init__(self, objects: tuple[vision.DetectedObject, ...]) -> None:
        self.objects = objects
        self.inputs = []

    @typing.override
    async def classify(self, input: detection.InputData) -> detection.OutputData:
        self.inputs.append(input)
        return detection.OutputData(objects=self.objects)


class FailingDetector(detection.ObjectDetector):
    @typing.override
    async def classify(self, input: detection.InputData) -> detection.OutputData:
        del input
        raise RuntimeError("detector unavailable")


class WrongDetector(detection.ObjectDetector):
    @typing.override
    async def classify(self, input: detection.InputData) -> detection.OutputData:
        del input
        return invalid_value(detection.OutputData, {"objects": []})


def _detect(detector: detection.ObjectDetector) -> hsm.Event[typing.Any]:
    async def run() -> hsm.Event[typing.Any]:
        child = detection.ObjectDetection(classifier=detector)
        ctx = hsm.Context()
        await start_abilities_for_test(ctx, child)
        return await abilities.run_terminal_operation(
            ctx,
            child=child,
            request=child.input_event.with_data_and_id(detection.InputData(image=_JPEG), "detect-1"),
            terminals=(child.output_event, child.failed_event),
            timeout=datetime.timedelta(seconds=2),
        )

    return asyncio.run(run())


def test_vision_exports_detection_and_detected_object() -> None:
    assert vision.detection is detection
    assert vision.DetectedObject is detection.DetectedObject
    assert {"detection", "DetectedObject"} <= set(vision.__all__)
    assert detection.MAX_OBJECTS == 32
    # The maximum has one home.
    assert not hasattr(vision, "MAX_OBJECTS")
    assert {"GenerativeDetector", "DETECT_INSTRUCTIONS", "Error"} <= set(detection.__all__)
    assert issubclass(detection.Error, RuntimeError)


def test_detected_objects_are_bounded_frozen_values() -> None:
    found = _object("cup", 0.9)

    assert found.label == "cup" and found.confidence == 0.9 and found.box == _BOX
    assert _object("cup", None).confidence is None
    for model in (detection.InputData, detection.DetectedObject, detection.OutputData):
        assert model.model_config.get("frozen") is True
    for invalid in (
        {"label": "", "confidence": 0.5, "box": _BOX},
        {"label": "x" * 101, "confidence": 0.5, "box": _BOX},
        {"label": "cup", "confidence": 1.5, "box": _BOX},
        {"label": "cup", "confidence": -0.1, "box": _BOX},
        {"label": "cup", "confidence": 0.5, "box": _BOX, "id": "track-1"},
    ):
        with pytest.raises(pydantic.ValidationError):
            _ = detection.DetectedObject.model_validate(invalid)
    with pytest.raises(pydantic.ValidationError):
        _ = detection.OutputData(objects=(found,) * (detection.MAX_OBJECTS + 1))
    assert len(detection.OutputData(objects=(found,) * detection.MAX_OBJECTS).objects) == detection.MAX_OBJECTS
    assert detection.OutputData(objects=()).objects == ()


def test_detected_objects_are_ordered_by_descending_confidence_with_unrated_last() -> None:
    output = detection.OutputData(
        objects=(_object("a", None), _object("b", 0.4), _object("c", 0.9), _object("d", None), _object("e", 0.4))
    )

    assert [found.label for found in output.objects] == ["c", "b", "e", "a", "d"]


def test_object_detection_answers_with_the_detectors_objects() -> None:
    detector = FixedDetector((_object("cup", 0.6), _object("person", 0.95)))

    terminal = _detect(detector)

    assert terminal.name == "bot.ability.vision.detection.output"
    assert terminal.id == "detect-1"
    assert isinstance(terminal.data, detection.OutputData)
    assert [found.label for found in terminal.data.objects] == ["person", "cup"]
    assert detector.inputs == [detection.InputData(image=_JPEG)]


def test_object_detection_answers_a_detector_failure_with_its_failed_event() -> None:
    terminal = _detect(FailingDetector())

    assert terminal.name == abilities.FailedEvent.name
    assert isinstance(terminal.data, abilities.FailureData)
    assert terminal.data.message == "detector unavailable"


def test_object_detection_answers_output_outside_its_schema_with_a_failure() -> None:
    terminal = _detect(WrongDetector())

    assert terminal.name == abilities.FailedEvent.name
    assert isinstance(terminal.data, abilities.FailureData)
    assert "output schema" in terminal.data.message


def test_object_detection_events_are_named_in_the_vision_domain() -> None:
    assert detection.ObjectDetection.input_event.name == "bot.ability.vision.detection.input"
    assert detection.ObjectDetection.output_event.name == "bot.ability.vision.detection.output"
    assert detection.ObjectDetection.input_data_type is detection.InputData
    assert detection.ObjectDetection.output_data_type is detection.OutputData
    model = model_view(detection.ObjectDetection.model)
    behavior = "/ObjectDetectionLifecycle/attached/behavior"
    assert "bot.ability.vision.detection.input" in model.transition_map[f"{behavior}/idle"]
    assert "bot.ability.vision.detection.apply.completed" in model.transition_map[f"{behavior}/applying"]
    assert "bot.ability.vision.detection.apply.failed" in model.transition_map[f"{behavior}/applying"]
    assert not hsm.kind.Is(detection.ObjectDetection.input_event.kind, event_contract.EventKind)


# GenerativeDetector


class FakeGenerator(text.TextGenerator):
    inputs: list[text.InputData]
    image_capacity: media.ImageCapacity
    reply: str

    def __init__(self, reply: str, *, capacity: media.ImageCapacity = _SEES_JPEG) -> None:
        self.reply = reply
        self.image_capacity = capacity
        self.inputs = []

    @typing.override
    async def generate(
        self, input: text.InputData, *, deltas: collections.abc.Callable[[text.Delta], None] | None = None
    ) -> text.OutputData:
        self.inputs.append(input)
        return text.OutputData(content=self.reply)


def _item(label: object, confidence: object = 0.5, **box: object) -> dict[str, object]:
    return {"label": label, "confidence": confidence, "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, **box}}


def _generated(
    reply: str,
    *,
    instructions: str = detection.DETECT_INSTRUCTIONS,
    max_objects: int = detection.MAX_OBJECTS,
    max_reply_bytes: int = 65_536,
) -> tuple[detection.OutputData, FakeGenerator]:
    generator = FakeGenerator(reply)
    detector = detection.GenerativeDetector(
        generator=generator, instructions=instructions, max_objects=max_objects, max_reply_bytes=max_reply_bytes
    )
    return asyncio.run(detector.classify(detection.InputData(image=_JPEG))), generator


def _invalid(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, reply: str) -> tuple[detection.OutputData, float]:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = sum(point.value for point in reader.points("bot.vision.detection.invalid"))
        output, _ = _generated(reply)
        after = reader.points("bot.vision.detection.invalid")
    assert all(point.attributes == {} for point in after)
    return output, sum(point.value for point in after) - before


def test_generative_detector_asks_once_with_the_instructions_then_the_image() -> None:
    output, generator = _generated(json.dumps([_item("cup", 0.6), _item("person", 0.9)]))

    assert [(found.label, found.confidence) for found in output.objects] == [("person", 0.9), ("cup", 0.6)]
    assert output.objects[0].box == _BOX
    (asked,) = generator.inputs
    (message,) = asked.messages
    assert message.role == text.TextRole.USER
    assert message.content == (text.TextPart(text=detection.DETECT_INSTRUCTIONS), text.ImagePart(image=_JPEG))
    assert asked.tools == ()
    assert "JSON array" in detection.DETECT_INSTRUCTIONS


def test_generative_detector_reads_an_empty_array_as_nothing_found() -> None:
    output, _ = _generated("[]")

    assert output.objects == ()


def test_generative_detector_reads_an_array_inside_a_code_fence() -> None:
    output, _ = _generated("```json\n" + json.dumps([_item("cup")]) + "\n```")

    assert [found.label for found in output.objects] == ["cup"]


def test_generative_detector_instructions_are_an_overridable_default() -> None:
    _, generator = _generated("[]", instructions="Find every cup.")

    content = generator.inputs[0].messages[0].content
    assert isinstance(content, tuple) and content[0] == text.TextPart(text="Find every cup.")


def test_generative_detector_skips_and_counts_invalid_items(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = json.dumps(
        [
            _item("cup"),
            _item(""),
            _item("lamp", confidence=1.5),
            _item("door", width=0.95),
            "a chair",
            {"label": "desk"},
            _item("person", confidence=None),
        ]
    )

    output, skipped = _invalid(tmp_path, monkeypatch, reply)

    assert [found.label for found in output.objects] == ["cup", "person"]
    assert skipped == 5


def test_generative_detector_truncates_after_validation_in_reply_order() -> None:
    reply = json.dumps([_item("bad", confidence=2.0), _item("a", 0.1), _item("b", 0.2), _item("c", 0.9)])

    output, _ = _generated(reply, max_objects=2)

    # The invalid first item does not use up a place; the first two valid ones are kept, then ordered.
    assert [found.label for found in output.objects] == ["b", "a"]


@pytest.mark.parametrize(
    "reply",
    ['{"label": "cup"}', "a cup and a person", "", '"cup"'],
    ids=["object", "prose", "empty", "string"],
)
def test_generative_detector_refuses_a_reply_that_is_not_an_array(reply: str) -> None:
    with pytest.raises(detection.Error, match="JSON array"):
        _ = _generated(reply)


def test_generative_detector_refuses_an_oversized_reply_before_parsing() -> None:
    reply = json.dumps([_item("cup")] * 4)

    with pytest.raises(detection.Error, match="max_reply_bytes"):
        _ = _generated(reply, max_reply_bytes=len(reply.encode()) - 1)
    output, _ = _generated(reply, max_reply_bytes=len(reply.encode()))
    assert len(output.objects) == 4


def test_generative_detector_refuses_an_image_its_model_cannot_take_before_asking() -> None:
    for generator, image in (
        (FakeGenerator("[]", capacity=media.ImageCapacity()), _JPEG),
        (FakeGenerator("[]"), _PNG),
    ):
        detector = detection.GenerativeDetector(generator=generator)
        with pytest.raises(detection.Error, match="cannot take"):
            _ = asyncio.run(detector.classify(detection.InputData(image=image)))
        assert generator.inputs == []


def test_generative_detector_settings_are_checked() -> None:
    generator = FakeGenerator("[]")
    invalid: tuple[tuple[typing.Callable[[], detection.GenerativeDetector], str], ...] = (
        (lambda: detection.GenerativeDetector(generator=generator, instructions=" "), "instructions"),
        (lambda: detection.GenerativeDetector(generator=generator, max_objects=0), "max_objects"),
        (
            lambda: detection.GenerativeDetector(generator=generator, max_objects=detection.MAX_OBJECTS + 1),
            "max_objects",
        ),
        (lambda: detection.GenerativeDetector(generator=generator, max_reply_bytes=0), "max_reply_bytes"),
    )
    for build, name in invalid:
        with pytest.raises(ValueError, match=name):
            _ = build()
    assert isinstance(detection.GenerativeDetector(generator=generator), detection.ObjectDetector)


def test_generative_detector_failures_reach_the_wrapper_as_its_failed_event() -> None:
    terminal = _detect(detection.GenerativeDetector(generator=FakeGenerator("no objects here")))

    assert terminal.name == abilities.FailedEvent.name
    assert isinstance(terminal.data, abilities.FailureData)
    assert "JSON array" in terminal.data.message
    # The reply itself never becomes the failure message.
    assert "no objects here" not in terminal.data.message
