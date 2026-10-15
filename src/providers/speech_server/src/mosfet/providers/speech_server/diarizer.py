"""``Diarizer``: core voice diarization served by a speech server's ``POST /v1/diarize`` (Sortformer).

The input is resampled to 16 kHz mono (:mod:`resample`) and posted raw with ``sample_rate=16000``. Each server
segment ``{start, end, speaker}`` becomes a ``voice.VoiceDiarizationSegment`` that clips the *original* audio at its
own rate and channels, snapped to whole frames and clipped to the audio's end; the server's local speaker index is
dropped, as the pyannote diarizer drops its labels. Segments are ordered by start time. An answer with no usable
segment is ``invalid_response``, because the core result needs at least one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import typing

import httpx
import pydantic

from mosfet.abilities.hearing import voice

from . import request, resample

_RATE_HZ: typing.Final = 16_000
_MAX_SEGMENTS: typing.Final = 1024


class _Segment(pydantic.BaseModel):
    """One server segment, validated at the wire."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, allow_inf_nan=False)

    start: float = pydantic.Field(ge=0.0)
    end: float
    speaker: int = pydantic.Field(ge=0)

    @pydantic.model_validator(mode="after")
    def _positive_span(self) -> typing.Self:
        if self.end <= self.start:
            raise ValueError("a segment must end after it starts.")
        return self


class _Diarization(pydantic.BaseModel):
    """The server's diarization answer; fields it adds beyond segments are ignored."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    segments: tuple[_Segment, ...] = pydantic.Field(max_length=_MAX_SEGMENTS)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Diarizer(voice.VoiceDiarizer):
    """Diarizes raw PCM through a speech server's ``/v1/diarize``.

    ``client`` is an injected ``httpx.AsyncClient``; without one, each request opens and closes its own.
    ``timeout_seconds`` bounds each request.
    """

    base_url: str
    client: httpx.AsyncClient | None = None
    timeout_seconds: float = 3.0

    @typing.override
    async def classify(self, input: voice.diarization.InputData) -> voice.diarization.OutputData:
        """Diarize ``input`` into ordered clips of its own audio.

        Raises:
            request.RequestError: The audio is not whole frames or was refused (``invalid_input``), the server was
                unreachable or failing (``unavailable``) or slow (``timeout``), or it answered without a usable
                segment (``invalid_response``).
        """

        try:
            mono = await asyncio.to_thread(
                resample.to_mono, input.audio, sample_rate_hz=input.sample_rate_hz, channels=input.channels
            )
        except ValueError as error:
            raise request.RequestError("invalid_input", "diarization input is not whole s16le frames.") from error
        answer = await request.post(
            base_url=self.base_url,
            path="/v1/diarize",
            operation="diarize",
            content=mono,
            params={"sample_rate": _RATE_HZ},
            client=self.client,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            served = _Diarization.model_validate(answer)
        except pydantic.ValidationError as error:
            message = f"speech-server diarization answer is invalid ({error.error_count()} errors)."
            raise request.RequestError("invalid_response", message) from error
        rate = input.sample_rate_hz
        frame = 2 * input.channels
        frames = len(input.audio) // frame
        segments: list[voice.VoiceDiarizationSegment] = []
        for served_segment in sorted(served.segments, key=lambda segment: segment.start):
            start = min(math.floor(served_segment.start * rate + 0.5), frames)
            end = min(math.floor(served_segment.end * rate + 0.5), frames)
            if end <= start:
                continue
            segments.append(
                voice.VoiceDiarizationSegment(
                    audio=input.audio[start * frame : end * frame],
                    media_type=input.media_type,
                    sample_rate_hz=rate,
                    channels=input.channels,
                    start_seconds=start / rate,
                    end_seconds=end / rate,
                )
            )
        if not segments:
            raise request.RequestError("invalid_response", "speech-server diarization found no speaker segment.")
        return voice.diarization.OutputData(segments=tuple(segments))


__all__ = ["Diarizer"]
