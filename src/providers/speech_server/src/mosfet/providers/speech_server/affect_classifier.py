"""``AffectClassifier``: core voice affect served by a speech server's ``POST /v1/emotion`` (emotion2vec+).

The segment is posted raw as s16le mono. The server takes 16, 24 or 48 kHz as it is, so mono audio at one of those
rates passes through unchanged (other channel counts are mixed to mono at the same rate); any other rate is resampled
to 16 kHz (:mod:`resample`). The query names the rate sent. The answer ``{label, scores, valence, confidence, ...}``
becomes ``voice.Affect``; the fields the server adds beyond those (``snr_db``, ``model``, timings) are dropped. The
server refuses audio under :data:`~mosfet.providers.speech_server.MIN_SEGMENT_SECONDS`, so a shorter segment is
refused as ``invalid_input`` before any request is sent.
"""

from __future__ import annotations

import asyncio
import dataclasses
import typing

import httpx
import pydantic

from mosfet.abilities.hearing import voice

from . import request, resample
from .voice_embedder import MIN_SEGMENT_SECONDS

_RATES_HZ: typing.Final = frozenset({16_000, 24_000, 48_000})
_FALLBACK_RATE_HZ: typing.Final = 16_000
_FIELDS: typing.Final = ("label", "scores", "valence", "confidence")


@dataclasses.dataclass(frozen=True, kw_only=True)
class AffectClassifier(voice.AffectClassifier):
    """Hears the affect of a voice segment through a speech server's ``/v1/emotion``.

    ``client`` is an injected ``httpx.AsyncClient``; without one, each request opens and closes its own.
    ``timeout_seconds`` bounds each request.
    """

    base_url: str
    client: httpx.AsyncClient | None = None
    timeout_seconds: float = 3.0

    @typing.override
    async def classify(self, input: voice.VoiceSegment) -> voice.Affect:
        """The affect the server hears in ``input``.

        Raises:
            request.RequestError: The segment is shorter than ``MIN_SEGMENT_SECONDS`` or was refused
                (``invalid_input``), the server was unreachable or failing (``unavailable``) or slow (``timeout``),
                or it answered without a valid affect (``invalid_response``).
        """

        if input.duration_seconds < MIN_SEGMENT_SECONDS:
            raise request.RequestError(
                "invalid_input", f"affect classification needs a segment of at least {MIN_SEGMENT_SECONDS} s."
            )
        rate = input.sample_rate_hz if input.sample_rate_hz in _RATES_HZ else _FALLBACK_RATE_HZ
        mono = await asyncio.to_thread(
            resample.to_mono, input.audio, sample_rate_hz=input.sample_rate_hz, channels=input.channels, rate_hz=rate
        )
        answer = await request.post(
            base_url=self.base_url,
            path="/v1/emotion",
            operation="emotion",
            content=mono,
            params={"sample_rate": rate},
            client=self.client,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            return voice.Affect.model_validate({name: answer[name] for name in _FIELDS if name in answer})
        except pydantic.ValidationError as error:
            message = f"speech-server emotion answer is invalid ({error.error_count()} errors)."
            raise request.RequestError("invalid_response", message) from error


__all__ = ["AffectClassifier"]
