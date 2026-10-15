"""``opencv.ObjectDetector``: YOLOX-S rows decoded into ``vision.detection`` output, over a fake ``cv2``."""

import asyncio
import math
import pathlib
import threading

import numpy
import pytest

import mosfet.media as media
from mosfet.abilities import vision
from mosfet.providers import opencv

from tests import opencv_fakes

_YOLOX = pathlib.Path("/models/yolox.onnx")
_CELLS = opencv.INPUT_SIZE // 8


def _put(
    output: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.float32]],
    *,
    center: tuple[float, float],
    size: tuple[float, float],
    objectness: float,
    label: str,
    cell: tuple[int, int],
) -> int:
    """Write one stride-8 row whose decoded box (in letterboxed pixels) is ``center``/``size``; return its row."""

    column, row = cell
    index = row * _CELLS + column
    output[0, index, 0] = center[0] / 8 - column
    output[0, index, 1] = center[1] / 8 - row
    output[0, index, 2] = math.log(size[0] / 8)
    output[0, index, 3] = math.log(size[1] / 8)
    output[0, index, 4] = objectness
    output[0, index, 5 + opencv.COCO_LABELS.index(label)] = 1.0
    return index


def _scene(height: int, width: int) -> tuple[opencv.ObjectDetector, opencv_fakes.FakeCv2, media.Image]:
    fake = opencv_fakes.FakeCv2()
    content = fake.frame(opencv_fakes.JPEG_SIGNATURE + b"room", numpy.zeros((height, width, 3), dtype=numpy.uint8))
    detector = opencv.ObjectDetector(models=opencv.Models(yolox=_YOLOX, cv2_loader=opencv_fakes.Loader(fake)))
    return detector, fake, media.Image(content=content, media_type="image/jpeg")


def _classify(detector: opencv.ObjectDetector, image: media.Image) -> vision.detection.OutputData:
    return asyncio.run(detector.classify(vision.detection.InputData(image=image)))


def test_coco_labels_are_the_eighty_classes_in_order() -> None:
    assert len(opencv.COCO_LABELS) == 80
    assert (opencv.COCO_LABELS[0], opencv.COCO_LABELS[41], opencv.COCO_LABELS[79]) == ("person", "cup", "toothbrush")


def test_rows_decode_into_labelled_normalized_boxes() -> None:
    detector, fake, image = _scene(320, 640)
    _ = _put(fake.object_output, center=(100, 60), size=(40, 20), objectness=0.9, label="cup", cell=(12, 7))

    (found,) = _classify(detector, image).objects

    assert found.label == "cup"
    assert found.confidence == pytest.approx(0.9)
    assert found.box.x == pytest.approx(80 / 640) and found.box.y == pytest.approx(50 / 320)
    assert found.box.width == pytest.approx(40 / 640) and found.box.height == pytest.approx(20 / 320)


def test_the_image_is_letterboxed_and_boxes_are_mapped_back() -> None:
    detector, fake, image = _scene(1280, 640)
    _ = _put(fake.object_output, center=(100, 60), size=(40, 20), objectness=0.9, label="person", cell=(12, 7))

    (found,) = _classify(detector, image).objects

    # The image was scaled by 0.5 into the letterbox, so the box doubles back.
    assert found.box.x == pytest.approx(160 / 640) and found.box.y == pytest.approx(100 / 1280)
    assert found.box.width == pytest.approx(80 / 640) and found.box.height == pytest.approx(40 / 1280)
    assert [call.args for call in fake.named("cvtColor")] == [(fake.COLOR_BGR2RGB,)]
    assert [call.args for call in fake.named("resize")] == [((320, 640), fake.INTER_LINEAR)]
    (set_input,) = fake.named("net.setInput")
    blob = set_input.args[0]
    assert isinstance(blob, numpy.ndarray) and blob.shape == (1, 3, 640, 640) and blob.dtype == numpy.float32
    assert float(blob[0, 0, 0, 0]) == 0.0 and float(blob[0, 0, 639, 639]) == 114.0


def test_objects_below_the_threshold_are_dropped_and_overlaps_are_suppressed() -> None:
    detector, fake, image = _scene(640, 640)
    _ = _put(fake.object_output, center=(100, 60), size=(40, 20), objectness=0.9, label="cup", cell=(12, 7))
    _ = _put(fake.object_output, center=(300, 300), size=(40, 40), objectness=0.2, label="dog", cell=(37, 37))
    _ = _put(fake.object_output, center=(500, 400), size=(40, 40), objectness=0.8, label="person", cell=(62, 50))
    fake.suppressed = {1}

    found = _classify(detector, image).objects

    assert [item.label for item in found] == ["cup"]
    ((boxes, scores, classes, score_threshold, nms_threshold),) = [
        call.args for call in fake.named("dnn.NMSBoxesBatched")
    ]
    # Only rows at or above the score threshold reach suppression, as top-left x, y, width, height.
    assert boxes == [pytest.approx([80.0, 50.0, 40.0, 20.0]), pytest.approx([480.0, 380.0, 40.0, 40.0])]
    assert scores == [pytest.approx(0.9), pytest.approx(0.8)]
    assert classes == [41, 0]
    assert (score_threshold, nms_threshold) == (0.35, 0.5)


def test_the_most_confident_objects_are_kept() -> None:
    detector, fake, image = _scene(640, 640)
    for index in range(vision.detection.MAX_OBJECTS + 8):
        _ = _put(
            fake.object_output,
            center=(index * 8 + 4.0, 4.0),
            size=(8, 8),
            objectness=0.4 + index * 0.01,
            label="cup",
            cell=(index, 0),
        )

    found = _classify(detector, image).objects

    assert len(found) == vision.detection.MAX_OBJECTS
    assert found[0].confidence == pytest.approx(0.4 + (vision.detection.MAX_OBJECTS + 7) * 0.01)
    assert found[-1].confidence == pytest.approx(0.4 + 8 * 0.01)


def test_nothing_found_is_an_empty_answer_and_runs_off_the_event_loop() -> None:
    detector, fake, image = _scene(320, 640)

    assert _classify(detector, image).objects == ()
    assert threading.get_ident() not in {call.thread for call in fake.calls}


def test_an_unexpected_output_shape_is_an_opencv_error() -> None:
    detector, fake, image = _scene(320, 640)
    fake.object_output = numpy.zeros((1, 100, 85), dtype=numpy.float32)

    with pytest.raises(opencv.Error, match="unexpected number of rows"):
        _ = _classify(detector, image)


def test_construction_needs_a_yolox_model_and_valid_thresholds() -> None:
    loader = opencv_fakes.Loader(opencv_fakes.FakeCv2())
    with pytest.raises(ValueError, match="YOLOX"):
        _ = opencv.ObjectDetector(models=opencv.Models(cv2_loader=loader))
    models = opencv.Models(yolox=_YOLOX, cv2_loader=loader)
    for threshold in (0.0, 1.0):
        with pytest.raises(ValueError, match="score_threshold"):
            _ = opencv.ObjectDetector(models=models, score_threshold=threshold)
        with pytest.raises(ValueError, match="nms_threshold"):
            _ = opencv.ObjectDetector(models=models, nms_threshold=threshold)
    assert isinstance(opencv.ObjectDetector(models=models), vision.detection.ObjectDetector)


def test_one_models_backend_serves_both_role_adapters() -> None:
    fake = opencv_fakes.FakeCv2()
    loader = opencv_fakes.Loader(fake)
    models = opencv.Models(yunet=pathlib.Path("/models/yunet.onnx"), yolox=_YOLOX, cv2_loader=loader)
    content = fake.frame(opencv_fakes.JPEG_SIGNATURE + b"room", numpy.zeros((320, 640, 3), dtype=numpy.uint8))
    image = media.Image(content=content, media_type="image/jpeg")

    async def both() -> None:
        _ = await asyncio.gather(
            opencv.ObjectDetector(models=models).classify(vision.detection.InputData(image=image)),
            opencv.FaceDetector(models=models).classify(vision.faces.InputData(image=image)),
        )

    asyncio.run(both())

    assert loader.loads == 1
    assert len(fake.named("dnn.readNet")) == 1 and len(fake.named("FaceDetectorYN.create")) == 1
