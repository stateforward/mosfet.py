"""``opencv.FaceDetector``: YuNet boxes and SFace prints as ``vision.faces`` output, over a fake ``cv2``."""

import asyncio
import math
import pathlib
import threading

import numpy
import pytest

import mosfet.media as media
from mosfet.abilities import identification, vision
from mosfet.providers import opencv

from tests import opencv_fakes

_YUNET = pathlib.Path("/models/yunet.onnx")
_SFACE = pathlib.Path("/models/sface.onnx")


def _row(x: float, y: float, width: float, height: float, score: float) -> list[float]:
    return [x, y, width, height, *([0.0] * 10), score]


def _scene(
    rows: list[list[float]], *, sface: bool = True
) -> tuple[opencv.FaceDetector, opencv_fakes.FakeCv2, media.Image]:
    fake = opencv_fakes.FakeCv2()
    fake.face_rows = numpy.array(rows, dtype=numpy.float32).reshape(-1, 15) if rows else None
    content = fake.frame(opencv_fakes.JPEG_SIGNATURE + b"room", numpy.zeros((200, 100, 3), dtype=numpy.uint8))
    models = opencv.Models(yunet=_YUNET, sface=_SFACE if sface else None, cv2_loader=opencv_fakes.Loader(fake))
    return opencv.FaceDetector(models=models), fake, media.Image(content=content, media_type="image/jpeg")


def _classify(detector: opencv.FaceDetector, image: media.Image) -> vision.faces.OutputData:
    return asyncio.run(detector.classify(vision.faces.InputData(image=image)))


def test_faces_are_normalized_boxes_with_sface_prints() -> None:
    detector, fake, image = _scene([_row(10, 20, 30, 40, 0.95)])

    (face,) = _classify(detector, image).faces

    assert face.box == vision.Box(x=0.1, y=0.1, width=0.3, height=0.2)
    assert face.confidence == pytest.approx(0.95)
    assert face.print is not None
    assert face.print.model == opencv.PRINT_MODEL == "opencv-sface-2021dec"
    # The fake's feature is (x + 1, y + 1, 1); the shared Print stores it L2-normalized.
    norm = math.sqrt(11**2 + 21**2 + 1)
    assert face.print.vector == pytest.approx((11 / norm, 21 / norm, 1 / norm))
    assert [call.args[0] for call in fake.named("imdecode")] == [fake.IMREAD_COLOR]
    assert [call.args[1] for call in fake.named("sface.alignCrop")] == [(10.0, 20.0, 30.0, 40.0)]


def test_without_an_sface_model_faces_carry_no_print() -> None:
    detector, fake, image = _scene([_row(10, 20, 30, 40, 0.95)], sface=False)

    (face,) = _classify(detector, image).faces

    assert face.print is None
    assert fake.named("FaceRecognizerSF.create") == []


def test_faces_are_clipped_to_the_image_and_those_outside_it_are_dropped() -> None:
    detector, _, image = _scene([_row(-10, -20, 30, 40, 0.9), _row(150, 20, 30, 40, 0.9)])

    (face,) = _classify(detector, image).faces

    assert face.box == vision.Box(x=0.0, y=0.0, width=0.2, height=0.1)


def test_the_largest_faces_are_kept_largest_first() -> None:
    rows = [_row(0, 0, 1 + index, 1 + index, 0.9) for index in range(vision.faces.MAX_FACES + 4)]
    detector, fake, image = _scene(rows)

    faces = _classify(detector, image).faces

    assert len(faces) == vision.faces.MAX_FACES
    widths = [round(face.box.width * 100) for face in faces]
    assert widths == list(range(vision.faces.MAX_FACES + 4, 4, -1))
    # Only kept faces are printed.
    assert len(fake.named("sface.feature")) == vision.faces.MAX_FACES


def test_no_face_found_is_an_empty_answer() -> None:
    detector, _, image = _scene([])

    assert _classify(detector, image).faces == ()


def test_detection_runs_off_the_event_loop_thread() -> None:
    detector, fake, image = _scene([_row(10, 20, 30, 40, 0.95)])

    _ = _classify(detector, image)

    assert {call.thread for call in fake.calls} and threading.get_ident() not in {call.thread for call in fake.calls}


def test_an_undecodable_image_and_a_raw_failure_are_opencv_errors() -> None:
    detector, fake, image = _scene([])
    unknown = media.Image(content=opencv_fakes.JPEG_SIGNATURE + b"unknown", media_type="image/jpeg")

    with pytest.raises(opencv.Error, match="could not decode"):
        _ = _classify(detector, unknown)
    # A malformed detector table fails inside the provider; only the exception type leaves it.
    fake.face_rows = numpy.zeros((1, 3), dtype=numpy.float32)
    with pytest.raises(opencv.Error, match="^OpenCV faces failed: IndexError$"):
        _ = _classify(detector, image)


def test_construction_needs_a_yunet_model_and_a_valid_threshold() -> None:
    loader = opencv_fakes.Loader(opencv_fakes.FakeCv2())
    with pytest.raises(ValueError, match="YuNet"):
        _ = opencv.FaceDetector(models=opencv.Models(sface=_SFACE, cv2_loader=loader))
    for threshold in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="score_threshold"):
            _ = opencv.FaceDetector(models=opencv.Models(yunet=_YUNET, cv2_loader=loader), score_threshold=threshold)
    assert isinstance(
        opencv.FaceDetector(models=opencv.Models(yunet=_YUNET, cv2_loader=loader)), vision.faces.FaceDetector
    )


def test_prints_are_the_shared_identification_print() -> None:
    detector, _, image = _scene([_row(10, 20, 30, 40, 0.95)])

    (face,) = _classify(detector, image).faces

    assert isinstance(face.print, identification.Print)
