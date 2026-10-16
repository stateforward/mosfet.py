"""Clients for a speech server's own wire protocols: streaming transcription, diarization, voice embeddings, affect,
and the voices it holds and designs.

The OpenAI-compatible speech endpoints (``/v1/audio/speech``, ``/v1/audio/transcriptions``) are served by
``mosfet.providers.openai_compat``; this package covers the rest of the protocol family.
"""

from . import request, resample, transcription, voices
from .affect_classifier import AffectClassifier
from .diarizer import Diarizer
from .request import RequestError
from .transcription import Transcription
from .voice_embedder import MIN_SEGMENT_SECONDS, VoiceEmbedder
from .voices import Catalog, Designer

__version__ = "0.1.0"

__all__ = [
    "AffectClassifier",
    "Catalog",
    "Designer",
    "Diarizer",
    "MIN_SEGMENT_SECONDS",
    "RequestError",
    "Transcription",
    "VoiceEmbedder",
    "__version__",
    "request",
    "resample",
    "transcription",
    "voices",
]
