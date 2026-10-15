"""Voice hearing subpackage.

Import ``InputData`` / ``OutputData`` from the defining leaf module
(``affect``, ``detection``, ``diarization``, ``identification``) — this package does not
re-export colliding type names.
"""

from . import affect, detection, diarization, identification, segment
from .affect import Affect, AffectClassifier
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
    HeardVoice,
    VoiceIdentification,
    VoiceEmbedding,
)
from .segment import VoiceSegment

__all__ = [
    "Affect",
    "AffectClassifier",
    "ApplyData",
    "EndData",
    "EndEvent",
    "HeardVoice",
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
    "affect",
    "detection",
    "diarization",
    "identification",
    "segment",
]
