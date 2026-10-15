"""``VoiceEmbedder``: core voice embeddings served by a speech server's ``POST /v1/embed`` (WeSpeaker).

Each voice segment is resampled to 16 kHz mono (:mod:`resample`) and posted raw, one request per segment in input
order. The answer ``{embedding, model, ...}`` becomes ``voice.VoiceEmbedding(embedding, model)``: the server's model
name is always set, which identification needs to keep one vector space per model. The server refuses audio under
:data:`MIN_SEGMENT_SECONDS`, so a shorter segment is never sent: it has no voiceprint (``None`` in its place), and
core leaves it unidentified (W3-13). An input of only short segments is not an error; every entry is ``None``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import typing

import httpx
import pydantic

from mosfet.abilities import classifying
from mosfet.abilities.hearing import voice

from . import request, resample

MIN_SEGMENT_SECONDS: typing.Final = 0.3
"""The shortest audio the embedding and emotion servers accept."""

_MAX_DIMENSIONS: typing.Final = 4096


class _Embedding(pydantic.BaseModel):
    """The server's embedding answer; fields it adds beyond these are ignored."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, allow_inf_nan=False)

    embedding: tuple[float, ...] = pydantic.Field(min_length=1, max_length=_MAX_DIMENSIONS)
    model: str = pydantic.Field(min_length=1, max_length=128)


@dataclasses.dataclass(frozen=True, kw_only=True)
class VoiceEmbedder(classifying.Classifier[voice.identification.InputData, voice.identification.OutputData]):
    """Embeds voice segments through a speech server's ``/v1/embed``.

    ``client`` is an injected ``httpx.AsyncClient``; without one, each request opens and closes its own.
    ``timeout_seconds`` bounds each request.
    """

    base_url: str
    client: httpx.AsyncClient | None = None
    timeout_seconds: float = 3.0

    @typing.override
    async def classify(self, input: voice.identification.InputData) -> voice.identification.OutputData:
        """Embed every segment of ``input``, in order; a segment under ``MIN_SEGMENT_SECONDS`` is ``None``, unsent.

        Raises:
            request.RequestError: A segment was refused (``invalid_input``), the server was unreachable or failing
                (``unavailable``) or slow (``timeout``), or it answered without a valid embedding
                (``invalid_response``).
        """

        embeddings: list[voice.VoiceEmbedding | None] = []
        for segment in input.segments:
            if segment.duration_seconds < MIN_SEGMENT_SECONDS:
                embeddings.append(None)
                continue
            mono = await asyncio.to_thread(
                resample.to_mono, segment.audio, sample_rate_hz=segment.sample_rate_hz, channels=segment.channels
            )
            answer = await request.post(
                base_url=self.base_url,
                path="/v1/embed",
                operation="embed",
                content=mono,
                client=self.client,
                timeout_seconds=self.timeout_seconds,
            )
            try:
                served = _Embedding.model_validate(answer)
            except pydantic.ValidationError as error:
                message = f"speech-server embedding answer is invalid ({error.error_count()} errors)."
                raise request.RequestError("invalid_response", message) from error
            embeddings.append(voice.VoiceEmbedding(embedding=served.embedding, model=served.model))
        return voice.identification.OutputData(embeddings=tuple(embeddings))


__all__ = ["MIN_SEGMENT_SECONDS", "VoiceEmbedder"]
