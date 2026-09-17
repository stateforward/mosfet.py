"""Voice hearing subpackage.

Import ``InputData`` / ``OutputData`` from the defining leaf module
(``detection``, ``diarization``, ``identification``) — this package does not
re-export colliding type names.
"""

from . import detection, diarization, identification, segment
from .detection import (
    ApplyData,
    EndData,
    EndEvent,
    OutputEvent,
    StartData,
    StartEvent,
    VoiceDetection,
    VoiceDetectionSegment,
    VoiceActivityClassifier,
)
from .diarization import (
    VoiceDiarization,
    VoiceDiarizationSegment,
    VoiceDiarizer,
)
from .identification import (
    VoiceIdentification,
    VoiceEmbedding,
)
from .segment import VoiceSegment

__all__ = [
    "ApplyData",
    "EndData",
    "EndEvent",
    "OutputEvent",
    "StartData",
    "StartEvent",
    "VoiceDiarization",
    "VoiceDiarizationSegment",
    "VoiceDetection",
    "VoiceDetectionSegment",
    "VoiceActivityClassifier",
    "VoiceDiarizer",
    "VoiceIdentification",
    "VoiceEmbedding",
    "VoiceSegment",
    "detection",
    "diarization",
    "identification",
    "segment",
]
