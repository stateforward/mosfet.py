"""Clients for a speech server's own wire protocols: streaming transcription, diarization and voice embeddings.

The OpenAI-compatible speech endpoints (``/v1/audio/speech``, ``/v1/audio/transcriptions``) are served by
``mosfet.providers.openai_compat``; this package covers the rest of the protocol family.
"""

from . import request, resample, transcription
from .diarizer import Diarizer
from .request import RequestError
from .transcription import Transcription
from .voice_embedder import MIN_SEGMENT_SECONDS, VoiceEmbedder

__version__ = "0.1.0"

__all__ = [
    "Diarizer",
    "MIN_SEGMENT_SECONDS",
    "RequestError",
    "Transcription",
    "VoiceEmbedder",
    "__version__",
    "request",
    "resample",
    "transcription",
]
