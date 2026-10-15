"""OpenCV vision provider: YuNet faces, SFace face prints, YOLOX-S objects and cheap change measures.

Per-role adapters over one ``Models`` backend serve the core vision stages, and the change measures
serve ``seeing.Seeing``'s required ``vision.change.ChangeMeasure``:

* ``FaceDetector`` — a ``vision.faces.FaceDetector`` (YuNet boxes, SFace ``identification.Print``).
* ``ObjectDetector`` — a ``vision.detection.ObjectDetector`` (YOLOX-S, COCO-80 labels).
* ``FrameDiff`` — a cheap ``vision.change.ChangeMeasure`` on a reduced grayscale decode.
* ``Downscaled`` — hands an expensive measure a small JPEG.

``cv2`` is loaded lazily and injected through ``Cv2Loader`` (its typed slice is ``surface``); model
files come from configuration.
"""

from . import surface
from .surface import Cv2, Cv2Loader, Error
from .change import Downscaled, FrameDiff
from .faces import PRINT_MODEL, FaceDetector
from .models import Models
from .objects import COCO_LABELS, INPUT_SIZE, ObjectDetector

__version__ = "0.1.0"

__all__ = [
    "COCO_LABELS",
    "INPUT_SIZE",
    "PRINT_MODEL",
    "Cv2",
    "Cv2Loader",
    "Downscaled",
    "Error",
    "FaceDetector",
    "FrameDiff",
    "Models",
    "ObjectDetector",
    "__version__",
    "surface",
]
