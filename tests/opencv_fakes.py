"""An in-process fake of the ``cv2`` surface ``mosfet.providers.opencv`` uses: no native library, no model.

Decoding looks frames up by their encoded bytes; every call is recorded with the thread it ran on.
"""

import collections.abc
import dataclasses
import threading
import time
import typing

import numpy
import numpy.typing as npt

from mosfet.providers.opencv import surface

Pixels = npt.NDArray[numpy.uint8]
Floats = npt.NDArray[numpy.float32]

JPEG_SIGNATURE: typing.Final = b"\xff\xd8\xff"


@dataclasses.dataclass
class Call:
    name: str
    args: tuple[object, ...]
    thread: int = dataclasses.field(default_factory=threading.get_ident)


class FakeYuNet(surface.FaceDetectorYN):
    rows: Floats | None
    cv2: "FakeCv2"

    def __init__(self, cv2: "FakeCv2", rows: Floats | None) -> None:
        self.cv2 = cv2
        self.rows = rows

    def setInputSize(self, input_size: tuple[int, int], /) -> None:
        self.cv2.record("yunet.setInputSize", input_size)

    def setScoreThreshold(self, score_threshold: float, /) -> None:
        self.cv2.record("yunet.setScoreThreshold", score_threshold)

    def detect(self, image: Pixels, /) -> tuple[int, Floats | None]:
        self.cv2.record("yunet.detect", image.shape)
        return 1, self.rows


class FakeYuNetFactory(surface.FaceDetectorYNFactory):
    cv2: "FakeCv2"

    def __init__(self, cv2: "FakeCv2") -> None:
        self.cv2 = cv2

    def create(
        self,
        model: str,
        config: str,
        input_size: tuple[int, int],
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
        /,
    ) -> FakeYuNet:
        self.cv2.record("FaceDetectorYN.create", model, config, input_size, score_threshold, nms_threshold, top_k)
        time.sleep(self.cv2.load_delay)
        return FakeYuNet(self.cv2, self.cv2.face_rows)


class FakeSFace(surface.FaceRecognizerSF):
    cv2: "FakeCv2"

    def __init__(self, cv2: "FakeCv2") -> None:
        self.cv2 = cv2

    def alignCrop(self, image: Pixels, face: Floats, /) -> Pixels:
        self.cv2.record("sface.alignCrop", image.shape, tuple(float(value) for value in face[:4]))
        # The "aligned crop" carries the face's corner so the feature can tell faces apart.
        return numpy.array([[int(face[0]) % 256, int(face[1]) % 256]], dtype=numpy.uint8)

    def feature(self, aligned: Pixels, /) -> Floats:
        self.cv2.record("sface.feature", aligned.shape)
        x, y = (float(value) for value in aligned.ravel()[:2])
        return numpy.array([[x + 1.0, y + 1.0, 1.0]], dtype=numpy.float32)


class FakeSFaceFactory(surface.FaceRecognizerSFFactory):
    cv2: "FakeCv2"

    def __init__(self, cv2: "FakeCv2") -> None:
        self.cv2 = cv2

    def create(self, model: str, config: str, /) -> FakeSFace:
        self.cv2.record("FaceRecognizerSF.create", model, config)
        return FakeSFace(self.cv2)


class FakeNet(surface.Net):
    cv2: "FakeCv2"

    def __init__(self, cv2: "FakeCv2") -> None:
        self.cv2 = cv2

    def setInput(self, blob: Floats, /) -> None:
        self.cv2.record("net.setInput", blob)

    def forward(self, names: collections.abc.Sequence[str], /) -> collections.abc.Sequence[Floats]:
        self.cv2.record("net.forward", tuple(names))
        return (self.cv2.object_output,)

    def getUnconnectedOutLayersNames(self) -> collections.abc.Sequence[str]:
        return ("output",)


class FakeDnn(surface.Dnn):
    cv2: "FakeCv2"

    def __init__(self, cv2: "FakeCv2") -> None:
        self.cv2 = cv2

    def readNet(self, model: str, /) -> FakeNet:
        self.cv2.record("dnn.readNet", model)
        return FakeNet(self.cv2)

    def NMSBoxesBatched(
        self,
        bboxes: collections.abc.Sequence[collections.abc.Sequence[float]],
        scores: collections.abc.Sequence[float],
        class_ids: collections.abc.Sequence[int],
        score_threshold: float,
        nms_threshold: float,
        /,
    ) -> collections.abc.Sequence[int]:
        self.cv2.record(
            "dnn.NMSBoxesBatched", list(bboxes), list(scores), list(class_ids), score_threshold, nms_threshold
        )
        return [index for index in range(len(bboxes)) if index not in self.cv2.suppressed]


class FakeCv2:
    """The ``cv2`` surface as plain Python: frames by encoded bytes, scripted network outputs."""

    IMREAD_COLOR: int = 1
    IMREAD_REDUCED_GRAYSCALE_8: int = 64
    IMWRITE_JPEG_QUALITY: int = 1
    INTER_AREA: int = 3
    INTER_LINEAR: int = 1
    COLOR_BGR2RGB: int = 4

    frames: dict[bytes, Pixels]
    calls: list[Call]
    face_rows: Floats | None
    object_output: Floats
    suppressed: set[int]
    load_delay: float
    FaceDetectorYN: surface.FaceDetectorYNFactory
    FaceRecognizerSF: surface.FaceRecognizerSFFactory
    dnn: surface.Dnn
    _lock: threading.Lock

    def __init__(self) -> None:
        self.frames = {}
        self.calls = []
        self.face_rows = None
        self.object_output = numpy.zeros((1, 8400, 85), dtype=numpy.float32)
        self.suppressed = set()
        self.load_delay = 0.0
        self.FaceDetectorYN = FakeYuNetFactory(self)
        self.FaceRecognizerSF = FakeSFaceFactory(self)
        self.dnn = FakeDnn(self)
        self._lock = threading.Lock()

    def record(self, name: str, *args: object) -> None:
        with self._lock:
            self.calls.append(Call(name=name, args=args))

    def named(self, name: str) -> list[Call]:
        return [call for call in self.calls if call.name == name]

    def frame(self, content: bytes, pixels: Pixels) -> bytes:
        """Register what decoding ``content`` gives; return ``content``."""

        self.frames[content] = pixels
        return content

    def imdecode(self, buf: Pixels, flags: int, /) -> Pixels | None:
        self.record("imdecode", flags)
        pixels = self.frames.get(buf.tobytes())
        if pixels is not None and flags == self.IMREAD_REDUCED_GRAYSCALE_8 and pixels.ndim == 3:
            return pixels.mean(axis=2).astype(numpy.uint8)
        return pixels

    def imencode(self, ext: str, image: Pixels, params: collections.abc.Sequence[int], /) -> tuple[bool, Pixels]:
        self.record("imencode", ext, image.shape, tuple(params))
        height, width = image.shape[:2]
        encoded = JPEG_SIGNATURE + f"{width}x{height}".encode()
        self.frames[encoded] = image
        return True, numpy.frombuffer(encoded, dtype=numpy.uint8)

    def resize(self, image: Pixels, dsize: tuple[int, int], /, *, interpolation: int) -> Pixels:
        self.record("resize", dsize, interpolation)
        width, height = dsize
        rows = numpy.arange(height) * image.shape[0] // height
        columns = numpy.arange(width) * image.shape[1] // width
        return image[rows][:, columns]

    def GaussianBlur(self, image: Pixels, ksize: tuple[int, int], sigmaX: float, /) -> Pixels:
        self.record("GaussianBlur", ksize, sigmaX)
        return image

    def cvtColor(self, image: Pixels, code: int, /) -> Pixels:
        self.record("cvtColor", code)
        return image[..., ::-1]


class Loader:
    """A ``Cv2Loader`` that counts how often it was asked."""

    cv2: FakeCv2
    loads: int

    def __init__(self, cv2: FakeCv2) -> None:
        self.cv2 = cv2
        self.loads = 0

    def __call__(self) -> FakeCv2:
        self.loads += 1
        return self.cv2
