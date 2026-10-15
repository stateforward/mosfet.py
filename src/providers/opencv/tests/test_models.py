"""``opencv.Models``: one lazily loaded backend for the face and object adapters."""

import concurrent.futures
import pathlib

import numpy
import pytest

from mosfet.providers import opencv

from tests import opencv_fakes

_YUNET = pathlib.Path("/models/face_detection_yunet_2023mar.onnx")
_SFACE = pathlib.Path("/models/face_recognition_sface_2021dec.onnx")
_YOLOX = pathlib.Path("/models/object_detection_yolox_2022nov.onnx")


def _models(fake: opencv_fakes.FakeCv2, **paths: pathlib.Path) -> tuple[opencv.Models, opencv_fakes.Loader]:
    loader = opencv_fakes.Loader(fake)
    return (
        opencv.Models(yunet=paths.get("yunet"), sface=paths.get("sface"), yolox=paths.get("yolox"), cv2_loader=loader),
        loader,
    )


def test_nothing_is_loaded_until_a_network_is_used() -> None:
    fake = opencv_fakes.FakeCv2()

    models, loader = _models(fake, yunet=_YUNET, sface=_SFACE, yolox=_YOLOX)

    assert loader.loads == 0 and fake.calls == []
    assert (models.yunet, models.sface, models.yolox) == (_YUNET, _SFACE, _YOLOX)


def test_each_network_loads_once_from_its_configured_file() -> None:
    fake = opencv_fakes.FakeCv2()
    models, loader = _models(fake, yunet=_YUNET, sface=_SFACE, yolox=_YOLOX)
    image = numpy.zeros((48, 64, 3), dtype=numpy.uint8)
    face = numpy.array([1, 2, 3, 4, *([0] * 10), 0.9], dtype=numpy.float32)
    blob = numpy.zeros((1, 3, 640, 640), dtype=numpy.float32)

    for _ in range(3):
        _ = models.detect_faces(image, score_threshold=0.8)
        _ = models.print_face(image, face)
        _ = models.detect_objects(blob)

    assert loader.loads == 1
    ((yunet,),) = [call.args[:1] for call in fake.named("FaceDetectorYN.create")]
    assert yunet == str(_YUNET)
    assert fake.named("FaceDetectorYN.create")[0].args[1:] == ("", (64, 48), 0.8, 0.3, 5000)
    assert [call.args for call in fake.named("FaceRecognizerSF.create")] == [(str(_SFACE), "")]
    assert [call.args for call in fake.named("dnn.readNet")] == [(str(_YOLOX),)]
    # Every image sets the detector's size and threshold; the network is not rebuilt.
    assert [call.args for call in fake.named("yunet.setInputSize")] == [((64, 48),)] * 3
    assert [call.args for call in fake.named("yunet.setScoreThreshold")] == [(0.8,)] * 3


def test_concurrent_first_calls_load_a_network_once() -> None:
    fake = opencv_fakes.FakeCv2()
    fake.load_delay = 0.05
    models, loader = _models(fake, yunet=_YUNET)
    image = numpy.zeros((8, 8, 3), dtype=numpy.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: models.detect_faces(image, score_threshold=0.9), range(8)))

    assert len(results) == 8
    assert loader.loads == 1
    assert len(fake.named("FaceDetectorYN.create")) == 1


def test_detect_faces_answers_an_empty_table_when_nothing_is_found() -> None:
    fake = opencv_fakes.FakeCv2()
    models, _ = _models(fake, yunet=_YUNET)

    rows = models.detect_faces(numpy.zeros((8, 8, 3), dtype=numpy.uint8), score_threshold=0.9)

    assert rows.shape == (0, 15)


def test_detect_objects_answers_the_networks_first_output() -> None:
    fake = opencv_fakes.FakeCv2()
    models, _ = _models(fake, yolox=_YOLOX)
    blob = numpy.ones((1, 3, 640, 640), dtype=numpy.float32)

    output = models.detect_objects(blob)

    assert output is fake.object_output
    (set_input,) = fake.named("net.setInput")
    assert set_input.args[0] is blob
    assert [call.args for call in fake.named("net.forward")] == [(("output",),)]


def test_a_role_whose_model_is_not_configured_is_an_error() -> None:
    models, _ = _models(opencv_fakes.FakeCv2())
    image = numpy.zeros((8, 8, 3), dtype=numpy.uint8)

    with pytest.raises(opencv.Error, match="YuNet"):
        _ = models.detect_faces(image, score_threshold=0.9)
    with pytest.raises(opencv.Error, match="SFace"):
        _ = models.print_face(image, numpy.zeros(15, dtype=numpy.float32))
    with pytest.raises(opencv.Error, match="YOLOX"):
        _ = models.detect_objects(numpy.zeros((1, 3, 640, 640), dtype=numpy.float32))
