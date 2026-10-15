"""Live smoke of ``opencv`` with the real ``cv2`` and the OpenCV Zoo model files.

Opt in with ``MOSFET_LIVE_OPENCV_MODELS`` naming a directory that holds
``face_detection_yunet_2023mar.onnx``, ``face_recognition_sface_2021dec.onnx`` and
``object_detection_yolox_2022nov.onnx`` (see the README for where to fetch them). Optionally set
``MOSFET_LIVE_OPENCV_IMAGE`` to a JPEG or PNG photo to run the detectors on. Per-frame times are
printed as a report, never asserted::

    MOSFET_LIVE_OPENCV_MODELS=~/models/opencv \\
      uv run --package mosfet-provider-opencv --group dev python -m pytest \\
      src/providers/opencv/tests/test_live.py -m live -v -s
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import time

import hsm
import numpy
import pytest

import mosfet.media as media
from mosfet.abilities import vision
from mosfet.providers import opencv

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("MOSFET_LIVE_OPENCV_MODELS"),
        reason="set MOSFET_LIVE_OPENCV_MODELS to a directory with the OpenCV Zoo model files",
    ),
]

_YUNET = "face_detection_yunet_2023mar.onnx"
_SFACE = "face_recognition_sface_2021dec.onnx"
_YOLOX = "object_detection_yolox_2022nov.onnx"
_FRAMES = 20


def _models_dir() -> pathlib.Path:
    return pathlib.Path(os.environ["MOSFET_LIVE_OPENCV_MODELS"]).expanduser()


def _still(shift: int) -> media.Image:
    """A 1280x720 camera-like JPEG: a gradient with a bright square that moves with ``shift``."""

    cv2 = opencv.surface.import_cv2()
    across = numpy.linspace(0, 255, 1280, dtype=numpy.float32)
    frame = numpy.repeat(numpy.tile(across, (720, 1))[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
    frame[300:420, 200 + shift : 320 + shift] = 255
    encoded, buffer = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, 85))
    assert encoded
    return media.Image(content=buffer.tobytes(), media_type="image/jpeg", width_px=1280, height_px=720)


def _photo() -> media.Image:
    configured = os.environ.get("MOSFET_LIVE_OPENCV_IMAGE")
    if not configured:
        return _still(0)
    path = pathlib.Path(configured).expanduser()
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return media.Image(content=path.read_bytes(), media_type=media_type)


def test_frame_diff_per_frame_cost_report() -> None:
    measure = opencv.FrameDiff()
    frames = [_still(index * 8) for index in range(_FRAMES)]

    async def run() -> list[float]:
        timings: list[float] = []
        previous = None
        for frame in frames:
            started = time.perf_counter()
            measured = await measure.measure(previous, frame, ctx=hsm.Context())
            timings.append((time.perf_counter() - started) * 1000)
            previous = measured.signature
            assert measured.change is None or 0.0 <= measured.change <= 1.0
        return timings

    timings = asyncio.run(run())
    print(f"opencv.FrameDiff on a 1280x720 JPEG: median {numpy.median(timings):.2f} ms, max {max(timings):.2f} ms")


def test_downscaled_per_frame_cost_report() -> None:
    measure = opencv.Downscaled(inner=vision.change.ByteIdentity())
    frame = _still(0)

    async def run() -> list[float]:
        timings: list[float] = []
        for _ in range(_FRAMES):
            started = time.perf_counter()
            _ = await measure.measure(None, frame, ctx=hsm.Context())
            timings.append((time.perf_counter() - started) * 1000)
        return timings

    timings = asyncio.run(run())
    print(f"opencv.Downscaled(max_edge=448) on a 1280x720 JPEG: median {numpy.median(timings):.2f} ms")


def test_faces_and_objects_through_the_zoo_models() -> None:
    directory = _models_dir()
    models = opencv.Models(yunet=directory / _YUNET, sface=directory / _SFACE, yolox=directory / _YOLOX)
    image = _photo()

    async def run() -> tuple[vision.faces.OutputData, vision.detection.OutputData, float, float]:
        started = time.perf_counter()
        faces = await opencv.FaceDetector(models=models).classify(vision.faces.InputData(image=image))
        faces_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        objects = await opencv.ObjectDetector(models=models).classify(vision.detection.InputData(image=image))
        return faces, objects, faces_ms, (time.perf_counter() - started) * 1000

    faces, objects, faces_ms, objects_ms = asyncio.run(run())

    assert all(face.print is not None and face.print.model == opencv.PRINT_MODEL for face in faces.faces)
    print(
        f"opencv faces: {len(faces.faces)} in {faces_ms:.1f} ms (first call loads the models); "
        f"objects: {[found.label for found in objects.objects]} in {objects_ms:.1f} ms"
    )
