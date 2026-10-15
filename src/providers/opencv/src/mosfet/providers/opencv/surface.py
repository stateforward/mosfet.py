"""The slice of OpenCV this provider calls: its typed surface, lazy loader, error, helpers and telemetry.

``cv2`` is imported only when a model or measure first needs it, never at import time, so the
package imports without loading the native library, and tests inject a fake that implements
``Cv2`` (and the network protocols it returns).

Telemetry (scope ``bot.providers.opencv``; every attribute value is a closed set from code):

* ``bot.opencv.call.count`` — ``operation`` {frame_diff, downscale, faces, objects} ×
  ``bot.outcome`` {ok, failed}: 8 series.
* ``bot.opencv.call.duration`` — seconds per call, by ``operation``: 4 series.

Never used as attributes or logs: image sizes, labels, boxes, prints, model paths.
"""

from __future__ import annotations

import collections.abc
import contextlib
import importlib
import time
import typing

import numpy
import numpy.typing as npt
from opentelemetry import metrics

from mosfet.abilities import vision

Pixels = npt.NDArray[numpy.uint8]
"""An 8-bit image: height × width, or height × width × 3 (BGR)."""

Floats = npt.NDArray[numpy.float32]
"""A float32 array, such as a network output or a face feature."""

Operation = typing.Literal["frame_diff", "downscale", "faces", "objects"]


class Error(RuntimeError):
    """An OpenCV model or measure could not answer: a missing model file, an undecodable image, or a failed call."""


class FaceDetectorYN(typing.Protocol):
    """OpenCV's YuNet face detector, as this provider calls it."""

    def setInputSize(self, input_size: tuple[int, int], /) -> None: ...

    def setScoreThreshold(self, score_threshold: float, /) -> None: ...

    def detect(self, image: Pixels, /) -> tuple[int, Floats | None]: ...


class FaceDetectorYNFactory(typing.Protocol):
    def create(
        self,
        model: str,
        config: str,
        input_size: tuple[int, int],
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
        /,
    ) -> FaceDetectorYN: ...


class FaceRecognizerSF(typing.Protocol):
    """OpenCV's SFace face recognizer, as this provider calls it."""

    def alignCrop(self, image: Pixels, face: Floats, /) -> Pixels: ...

    def feature(self, aligned: Pixels, /) -> Floats: ...


class FaceRecognizerSFFactory(typing.Protocol):
    def create(self, model: str, config: str, /) -> FaceRecognizerSF: ...


class Net(typing.Protocol):
    """An OpenCV DNN network, as this provider calls it."""

    def setInput(self, blob: Floats, /) -> None: ...

    def forward(self, names: collections.abc.Sequence[str], /) -> collections.abc.Sequence[Floats]: ...

    def getUnconnectedOutLayersNames(self) -> collections.abc.Sequence[str]: ...


class Dnn(typing.Protocol):
    def readNet(self, model: str, /) -> Net: ...

    def NMSBoxesBatched(
        self,
        bboxes: collections.abc.Sequence[collections.abc.Sequence[float]],
        scores: collections.abc.Sequence[float],
        class_ids: collections.abc.Sequence[int],
        score_threshold: float,
        nms_threshold: float,
        /,
    ) -> collections.abc.Sequence[int]: ...


class Cv2(typing.Protocol):
    """The ``cv2`` module surface this provider uses."""

    IMREAD_COLOR: int
    IMREAD_REDUCED_GRAYSCALE_8: int
    IMWRITE_JPEG_QUALITY: int
    INTER_AREA: int
    INTER_LINEAR: int
    COLOR_BGR2RGB: int
    FaceDetectorYN: FaceDetectorYNFactory
    FaceRecognizerSF: FaceRecognizerSFFactory
    dnn: Dnn

    def imdecode(self, buf: Pixels, flags: int, /) -> Pixels | None: ...

    def imencode(self, ext: str, image: Pixels, params: collections.abc.Sequence[int], /) -> tuple[bool, Pixels]: ...

    def resize(self, image: Pixels, dsize: tuple[int, int], /, *, interpolation: int) -> Pixels: ...

    def GaussianBlur(self, image: Pixels, ksize: tuple[int, int], sigmaX: float, /) -> Pixels: ...

    def cvtColor(self, image: Pixels, code: int, /) -> Pixels: ...


Cv2Loader = collections.abc.Callable[[], Cv2]
"""Returns the ``cv2`` surface; injected so tests never load the native library."""


def import_cv2() -> Cv2:
    """Import ``cv2`` (opencv-python); the module satisfies ``Cv2`` structurally."""

    # cv2 ships no protocol of its own: the module object is the surface this provider declares.
    return typing.cast(Cv2, typing.cast(object, importlib.import_module("cv2")))


def decode(cv2: Cv2, content: bytes, flags: int) -> Pixels:
    """Decode encoded image bytes with ``flags``.

    Raises:
        Error: If OpenCV cannot decode them.
    """

    decoded = cv2.imdecode(numpy.frombuffer(content, dtype=numpy.uint8), flags)
    if decoded is None or decoded.size == 0:
        raise Error("OpenCV could not decode the image")
    return decoded


_METER = metrics.get_meter("bot.providers.opencv")
_CALLS = _METER.create_counter(
    "bot.opencv.call.count",
    unit="{call}",
    description="OpenCV provider calls by operation (frame_diff, downscale, faces, objects) and outcome (ok, failed).",
)
_DURATION = _METER.create_histogram(
    "bot.opencv.call.duration",
    unit="s",
    description="Time one OpenCV provider call took, by operation; the per-frame cost of a measure.",
)


@contextlib.contextmanager
def measured(operation: Operation, error: type[Exception]) -> collections.abc.Generator[None]:
    """Count and time one call; any failure is counted and raised as ``error``.

    An exception that is already an ``error`` passes through. Anything else (a raw ``cv2.error``,
    a numpy error) is replaced by ``error`` naming only the operation and the exception type, so
    provider messages (which may carry model paths) never leave the provider; the original stays
    chained as its cause.
    """

    started = time.monotonic()
    try:
        yield
    except error:
        _CALLS.add(1, {"operation": operation, "bot.outcome": "failed"})
        raise
    except Exception as raised:
        _CALLS.add(1, {"operation": operation, "bot.outcome": "failed"})
        # This provider's own messages carry no paths; anything else is named only by its type.
        reason = str(raised) if isinstance(raised, Error) else type(raised).__name__
        raise error(f"OpenCV {operation} failed: {reason}") from raised
    finally:
        _DURATION.record(time.monotonic() - started, {"operation": operation})
    _CALLS.add(1, {"operation": operation, "bot.outcome": "ok"})


def clipped_box(
    x: float, y: float, width: float, height: float, image_width: int, image_height: int
) -> vision.Box | None:
    """A pixel rectangle clipped to the image and normalized, or ``None`` when nothing of it is inside."""

    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(image_width), x + width), min(float(image_height), y + height)
    if right <= left or bottom <= top:
        return None
    return vision.Box(
        x=left / image_width,
        y=top / image_height,
        width=(right - left) / image_width,
        height=(bottom - top) / image_height,
    )
