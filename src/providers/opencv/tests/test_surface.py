"""``opencv.surface``: lazy ``cv2``, decoding, box clipping, error wrapping and call telemetry."""

import pathlib
import subprocess
import sys

import numpy
import pytest

from mosfet.abilities import vision
from mosfet.abilities.vision import change
from mosfet.providers import opencv
from mosfet.providers.opencv import surface

from tests import metric_reader, opencv_fakes


def test_importing_the_provider_does_not_load_cv2() -> None:
    probe = "import sys, mosfet.providers.opencv; print('cv2' in sys.modules)"

    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60)

    assert loaded.stdout.strip() == "False"


def test_the_package_exports_its_contract() -> None:
    assert opencv.surface is surface
    assert opencv.Error is surface.Error and issubclass(opencv.Error, RuntimeError)
    assert opencv.Cv2Loader is surface.Cv2Loader
    assert set(opencv.__all__) >= {
        "Downscaled",
        "Error",
        "FaceDetector",
        "FrameDiff",
        "Models",
        "ObjectDetector",
        "surface",
    }


def test_decode_answers_the_decoded_pixels_or_an_error() -> None:
    fake = opencv_fakes.FakeCv2()
    pixels = numpy.zeros((2, 3), dtype=numpy.uint8)
    content = fake.frame(opencv_fakes.JPEG_SIGNATURE + b"a", pixels)

    assert surface.decode(fake, content, fake.IMREAD_COLOR) is pixels
    assert [call.args for call in fake.named("imdecode")] == [(fake.IMREAD_COLOR,)]
    with pytest.raises(opencv.Error, match="could not decode"):
        _ = surface.decode(fake, b"unknown", fake.IMREAD_COLOR)


def test_clipped_box_normalizes_pixels_and_clips_to_the_image() -> None:
    assert surface.clipped_box(10, 20, 30, 40, 100, 200) == vision.Box(x=0.1, y=0.1, width=0.3, height=0.2)
    assert surface.clipped_box(-10, -20, 30, 40, 100, 200) == vision.Box(x=0.0, y=0.0, width=0.2, height=0.1)
    assert surface.clipped_box(90, 190, 30, 40, 100, 200) == vision.Box(x=0.9, y=0.95, width=0.1, height=0.05)
    assert surface.clipped_box(100, 0, 10, 10, 100, 200) is None
    assert surface.clipped_box(-20, 0, 10, 10, 100, 200) is None


def test_measured_counts_calls_and_wraps_raw_failures(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        before = reader.points("bot.opencv.call.count")
        with surface.measured("faces", opencv.Error):
            pass
        with pytest.raises(opencv.Error, match="OpenCV objects failed: ValueError") as raw:
            with surface.measured("objects", opencv.Error):
                raise ValueError("/models/secret/path.onnx")
        with pytest.raises(change.Error, match="OpenCV frame_diff failed: OpenCV could not decode the image"):
            with surface.measured("frame_diff", change.Error):
                raise opencv.Error("OpenCV could not decode the image")
        with pytest.raises(change.Error, match="^own$"):
            with surface.measured("downscale", change.Error):
                raise change.Error("own")
        after = reader.points("bot.opencv.call.count")
        durations = reader.points("bot.opencv.call.duration")

    # The raw message (here a path) never reaches the provider's error; it stays chained as the cause.
    assert "secret" not in str(raw.value) and isinstance(raw.value.__cause__, ValueError)
    previous = {tuple(sorted(point.attributes.items())): point.value for point in before}
    counted = {
        key: point.value - previous.get(key, 0.0)
        for point in after
        if (key := tuple(sorted(point.attributes.items()))) and point.value - previous.get(key, 0.0)
    }
    assert counted == {
        (("bot.outcome", "ok"), ("operation", "faces")): 1,
        (("bot.outcome", "failed"), ("operation", "objects")): 1,
        (("bot.outcome", "failed"), ("operation", "frame_diff")): 1,
        (("bot.outcome", "failed"), ("operation", "downscale")): 1,
    }
    assert {str(point.attributes["operation"]) for point in durations} >= {
        "faces",
        "objects",
        "frame_diff",
        "downscale",
    }
    assert all(set(point.attributes) == {"operation"} for point in durations)
