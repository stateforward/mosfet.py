"""Voice hearing subpackage.

Import ``InputData`` / ``OutputData`` from the defining leaf module
(``detection``, ``diarization``, ``identification``) — this package does not
re-export colliding type names.
"""

from . import detection, diarization, identification
from .detection import VoiceDetection, VoiceDetector
from .diarization import (
    VoiceDiarization,
    VoiceDiarizationSegment,
    VoiceDiarizer,
)
from .identification import (
    VoiceIdentification,
    VoiceIdentificationSegment,
    VoiceIdentifier,
    VoiceSignature,
)

__all__ = [
    "VoiceDiarization",
    "VoiceDiarizationSegment",
    "VoiceDetection",
    "VoiceDetector",
    "VoiceDiarizer",
    "VoiceIdentification",
    "VoiceIdentificationSegment",
    "VoiceIdentifier",
    "VoiceSignature",
    "detection",
    "diarization",
    "identification",
]
