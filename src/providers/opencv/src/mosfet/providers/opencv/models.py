"""One provider-local backend for the OpenCV Zoo networks the face and object adapters share."""

from __future__ import annotations

import pathlib
import threading
import typing

import numpy

from .surface import Cv2, Cv2Loader, Error, FaceDetectorYN, FaceRecognizerSF, Floats, Net, Pixels, import_cv2

_FACE_NMS_THRESHOLD: typing.Final = 0.3
_FACE_TOP_K: typing.Final = 5000
_FACE_ROW: typing.Final = 15


class Models:
    """The OpenCV Zoo networks behind ``opencv.FaceDetector`` and ``opencv.ObjectDetector``.

    Not an actor: a plain collaborator the adapters call directly. Each network is loaded lazily,
    once, from its configured file, and every call into a network holds that network's lock, so
    one ``Models`` may serve several adapters (and several eyes) at once. Nothing is downloaded:
    model files come from configuration.

    * ``yunet`` — YuNet face detection (``face_detection_yunet_2023mar.onnx``, MIT).
    * ``sface`` — SFace 128-d face prints (``face_recognition_sface_2021dec.onnx``, Apache-2.0).
    * ``yolox`` — YOLOX-S COCO-80 objects (``object_detection_yolox_2022nov.onnx``, Apache-2.0).

    The configured paths are plain public attributes, read by the adapters to refuse a role whose
    model is missing at construction.
    """

    yunet: pathlib.Path | None
    sface: pathlib.Path | None
    yolox: pathlib.Path | None
    _cv2_loader: Cv2Loader
    _cv2_module: Cv2 | None
    _face_detector: FaceDetectorYN | None
    _face_recognizer: FaceRecognizerSF | None
    _object_net: Net | None
    _load_lock: threading.Lock
    _face_lock: threading.Lock
    _print_lock: threading.Lock
    _object_lock: threading.Lock

    def __init__(
        self,
        *,
        yunet: pathlib.Path | None = None,
        sface: pathlib.Path | None = None,
        yolox: pathlib.Path | None = None,
        cv2_loader: Cv2Loader = import_cv2,
    ) -> None:
        """Serve the networks whose files are given; ``cv2_loader`` supplies the ``cv2`` surface."""

        self.yunet = yunet
        self.sface = sface
        self.yolox = yolox
        self._cv2_loader = cv2_loader
        self._cv2_module = None
        self._face_detector = None
        self._face_recognizer = None
        self._object_net = None
        self._load_lock = threading.Lock()
        self._face_lock = threading.Lock()
        self._print_lock = threading.Lock()
        self._object_lock = threading.Lock()

    def cv2(self) -> Cv2:
        """The ``cv2`` surface, loaded once."""

        with self._load_lock:
            if self._cv2_module is None:
                self._cv2_module = self._cv2_loader()
            return self._cv2_module

    def detect_faces(self, image: Pixels, *, score_threshold: float) -> Floats:
        """YuNet rows for a BGR image: ``x, y, w, h`` in pixels, five landmarks, then the score.

        Blocking; call it off the event loop. Returns an empty ``(0, 15)`` array when no face is found.

        Raises:
            Error: If no YuNet model is configured.
        """

        cv2 = self.cv2()
        height, width = image.shape[:2]
        with self._face_lock:
            if self._face_detector is None:
                if self.yunet is None:
                    raise Error("no YuNet face detection model is configured")
                self._face_detector = cv2.FaceDetectorYN.create(
                    str(self.yunet), "", (width, height), score_threshold, _FACE_NMS_THRESHOLD, _FACE_TOP_K
                )
            self._face_detector.setInputSize((width, height))
            self._face_detector.setScoreThreshold(score_threshold)
            _, faces = self._face_detector.detect(image)
        if faces is None:
            return numpy.zeros((0, _FACE_ROW), dtype=numpy.float32)
        return faces

    def print_face(self, image: Pixels, face: Floats) -> Floats:
        """The SFace feature of one YuNet face row in a BGR image: aligned, cropped, then embedded.

        Blocking; call it off the event loop.

        Raises:
            Error: If no SFace model is configured.
        """

        cv2 = self.cv2()
        with self._print_lock:
            if self._face_recognizer is None:
                if self.sface is None:
                    raise Error("no SFace face recognition model is configured")
                self._face_recognizer = cv2.FaceRecognizerSF.create(str(self.sface), "")
            aligned = self._face_recognizer.alignCrop(image, face)
            return self._face_recognizer.feature(aligned)

    def detect_objects(self, blob: Floats) -> Floats:
        """The YOLOX network's raw output for one ``1 × 3 × 640 × 640`` input blob.

        Blocking; call it off the event loop.

        Raises:
            Error: If no YOLOX model is configured.
        """

        cv2 = self.cv2()
        with self._object_lock:
            if self._object_net is None:
                if self.yolox is None:
                    raise Error("no YOLOX object detection model is configured")
                self._object_net = cv2.dnn.readNet(str(self.yolox))
            self._object_net.setInput(blob)
            outputs = self._object_net.forward(self._object_net.getUnconnectedOutLayersNames())
        return outputs[0]


__all__ = ["Models"]
