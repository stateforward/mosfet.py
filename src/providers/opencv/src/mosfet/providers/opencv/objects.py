"""COCO-80 objects through OpenCV DNN and YOLOX-S: one role adapter over ``opencv.Models``."""

from __future__ import annotations

import asyncio
import typing

import numpy

from mosfet.abilities import vision

from .surface import Error, Floats, clipped_box, decode, measured
from .models import Models

COCO_LABELS: typing.Final = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)  # fmt: skip
"""The 80 COCO class names, in the order YOLOX's class scores come."""

INPUT_SIZE: typing.Final = 640
"""YOLOX-S takes one letterboxed ``640 × 640`` RGB image."""

_STRIDES: typing.Final = (8, 16, 32)
_PAD: typing.Final = 114.0
_DEFAULT_SCORE_THRESHOLD: typing.Final = 0.35
_DEFAULT_NMS_THRESHOLD: typing.Final = 0.5


def _anchors() -> tuple[Floats, Floats]:
    """Each YOLOX output row's grid cell and stride, in the network's output order."""

    grids: list[Floats] = []
    strides: list[Floats] = []
    for stride in _STRIDES:
        cells = INPUT_SIZE // stride
        across, down = numpy.meshgrid(numpy.arange(cells), numpy.arange(cells))
        grids.append(numpy.stack((across, down), 2).reshape(-1, 2).astype(numpy.float32))
        strides.append(numpy.full((cells * cells, 1), stride, dtype=numpy.float32))
    return numpy.concatenate(grids), numpy.concatenate(strides)


_GRIDS, _GRID_STRIDES = _anchors()


class ObjectDetector(vision.detection.ObjectDetector):
    """Finds COCO-80 objects with YOLOX-S through OpenCV DNN.

    Runs in a worker thread. The image is letterboxed into ``INPUT_SIZE`` (padding 114, as the
    OpenCV Zoo demo does), the network's rows are decoded against their grid cells, objects below
    ``score_threshold`` are dropped, overlaps per class are suppressed at ``nms_threshold``, and
    boxes are mapped back, clipped and normalized. The most confident ``vision.detection.MAX_OBJECTS``
    objects are kept. Defaults are the OpenCV Zoo's.
    """

    _models: Models
    _score_threshold: float
    _nms_threshold: float

    def __init__(
        self,
        *,
        models: Models,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = _DEFAULT_NMS_THRESHOLD,
    ) -> None:
        """Raises ``ValueError`` if ``models`` has no YOLOX model or a threshold is outside (0, 1)."""

        if models.yolox is None:
            raise ValueError("opencv.ObjectDetector needs Models with a YOLOX model (yolox=).")
        if not 0.0 < score_threshold < 1.0:
            raise ValueError("opencv.ObjectDetector score_threshold must be in (0, 1).")
        if not 0.0 < nms_threshold < 1.0:
            raise ValueError("opencv.ObjectDetector nms_threshold must be in (0, 1).")
        self._models = models
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold

    @typing.override
    async def classify(self, input: vision.detection.InputData) -> vision.detection.OutputData:
        """The objects in ``input.image``, most confident first.

        Raises:
            opencv.Error: If the image cannot be decoded, the network's output has an unexpected
                shape, or a network call fails; a raw OpenCV error is its cause.
        """

        return await asyncio.to_thread(self._classify_blocking, input.image.content)

    def _classify_blocking(self, content: bytes) -> vision.detection.OutputData:
        with measured("objects", Error):
            cv2 = self._models.cv2()
            image = cv2.cvtColor(decode(cv2, content, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            ratio = min(INPUT_SIZE / height, INPUT_SIZE / width)
            scaled_height, scaled_width = max(1, int(height * ratio)), max(1, int(width * ratio))
            letterboxed = numpy.full((INPUT_SIZE, INPUT_SIZE, 3), _PAD, dtype=numpy.float32)
            letterboxed[:scaled_height, :scaled_width] = cv2.resize(
                image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
            )
            blob = numpy.ascontiguousarray(numpy.transpose(letterboxed, (2, 0, 1))[numpy.newaxis])
            rows = numpy.asarray(self._models.detect_objects(blob), dtype=numpy.float32).reshape(
                -1, 5 + len(COCO_LABELS)
            )
            if rows.shape[0] != _GRIDS.shape[0]:
                raise Error("the YOLOX network answered with an unexpected number of rows")
            centers = (rows[:, :2] + _GRIDS) * _GRID_STRIDES
            sizes = numpy.exp(rows[:, 2:4]) * _GRID_STRIDES
            scores = rows[:, 4:5] * rows[:, 5:]
            classes = numpy.argmax(scores, axis=1)
            best = scores[numpy.arange(scores.shape[0]), classes]
            candidates = numpy.flatnonzero(best >= self._score_threshold)
            corners = centers[candidates] - sizes[candidates] / 2
            kept = cv2.dnn.NMSBoxesBatched(
                [[float(x), float(y), float(w), float(h)] for (x, y), (w, h) in zip(corners, sizes[candidates])],
                [float(score) for score in best[candidates]],
                [int(index) for index in classes[candidates]],
                self._score_threshold,
                self._nms_threshold,
            )
            ranked = sorted((int(index) for index in kept), key=lambda index: -float(best[candidates[index]]))
            found: list[vision.DetectedObject] = []
            for index in ranked:
                if len(found) == vision.detection.MAX_OBJECTS:
                    break
                row = candidates[index]
                (x, y), (w, h) = corners[index] / ratio, sizes[row] / ratio
                box = clipped_box(float(x), float(y), float(w), float(h), width, height)
                if box is None:
                    continue
                found.append(
                    vision.DetectedObject(
                        label=COCO_LABELS[int(classes[row])],
                        confidence=min(1.0, max(0.0, float(best[row]))),
                        box=box,
                    )
                )
            return vision.detection.OutputData(objects=tuple(found))


__all__ = ["COCO_LABELS", "INPUT_SIZE", "ObjectDetector"]
