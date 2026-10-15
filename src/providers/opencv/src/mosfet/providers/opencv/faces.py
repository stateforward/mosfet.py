"""Faces and face prints through OpenCV YuNet and SFace: one role adapter over ``opencv.Models``."""

from __future__ import annotations

import asyncio
import typing

from mosfet.abilities import identification, vision

from .surface import Error, clipped_box, decode, measured
from .models import Models

PRINT_MODEL: typing.Final = "opencv-sface-2021dec"
"""The model every print this detector takes is tagged with; stores keep one vector space per model."""

_DEFAULT_SCORE_THRESHOLD: typing.Final = 0.9


class FaceDetector(vision.faces.FaceDetector):
    """Finds faces with YuNet and, when an SFace model is configured, prints each one.

    Runs in a worker thread. A face is clipped to the image; one with nothing inside it is
    dropped. The largest ``vision.faces.MAX_FACES`` faces are kept. ``score_threshold`` is the
    YuNet confidence below which a face is not reported (the OpenCV Zoo default is 0.9).
    """

    _models: Models
    _score_threshold: float

    def __init__(self, *, models: Models, score_threshold: float = _DEFAULT_SCORE_THRESHOLD) -> None:
        """Raises ``ValueError`` if ``models`` has no YuNet model or ``score_threshold`` is outside (0, 1)."""

        if models.yunet is None:
            raise ValueError("opencv.FaceDetector needs Models with a YuNet model (yunet=).")
        if not 0.0 < score_threshold < 1.0:
            raise ValueError("opencv.FaceDetector score_threshold must be in (0, 1).")
        self._models = models
        self._score_threshold = score_threshold

    @typing.override
    async def classify(self, input: vision.faces.InputData) -> vision.faces.OutputData:
        """The faces in ``input.image``, largest first, each with its SFace print when configured.

        Raises:
            opencv.Error: If the image cannot be decoded or a network call fails; a raw OpenCV
                error is its cause.
        """

        return await asyncio.to_thread(self._classify_blocking, input.image.content)

    def _classify_blocking(self, content: bytes) -> vision.faces.OutputData:
        with measured("faces", Error):
            cv2 = self._models.cv2()
            image = decode(cv2, content, cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            rows = self._models.detect_faces(image, score_threshold=self._score_threshold)
            # Largest first, so the kept faces are the largest whatever the detector's order.
            ordered = sorted(rows, key=lambda row: -float(row[2]) * float(row[3]))
            faces: list[vision.faces.DetectedFace] = []
            for row in ordered:
                if len(faces) == vision.faces.MAX_FACES:
                    break
                box = clipped_box(float(row[0]), float(row[1]), float(row[2]), float(row[3]), width, height)
                if box is None:
                    continue
                face_print = None
                if self._models.sface is not None:
                    feature = self._models.print_face(image, row)
                    face_print = identification.Print(
                        vector=tuple(float(value) for value in feature.ravel()), model=PRINT_MODEL
                    )
                faces.append(
                    vision.faces.DetectedFace(box=box, confidence=min(1.0, max(0.0, float(row[14]))), print=face_print)
                )
            return vision.faces.OutputData(faces=tuple(faces))


__all__ = ["PRINT_MODEL", "FaceDetector"]
