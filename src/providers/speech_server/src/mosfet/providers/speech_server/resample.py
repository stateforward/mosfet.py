"""Convert s16le PCM to the mono rate the speech servers take, with the standard library only.

The servers (``/v1/diarize``, ``/v1/embed``) take raw s16le mono at 16 kHz. :func:`to_mono` averages the channels
into one and resamples by linear interpolation between neighboring samples. There is no anti-alias filter: this is
the cheap conversion the speech lane chose for perception models, not a mastering resampler. It is CPU work, so
callers on an event loop run it in a worker thread.
"""

from __future__ import annotations

import array
import sys

_SAMPLE_BYTES = 2
"""s16le: two bytes per sample per channel."""


def to_mono(audio: bytes, *, sample_rate_hz: int, channels: int, rate_hz: int = 16_000) -> bytes:
    """Return ``audio`` (interleaved s16le at ``sample_rate_hz``) as s16le mono at ``rate_hz``.

    Mono audio already at ``rate_hz`` is returned unchanged. The output has ``frames * rate_hz // sample_rate_hz``
    samples; the last input sample is held where interpolation would read past the end.

    Raises:
        ValueError: A rate or channel count is not positive, or ``audio`` is not whole frames.
    """

    if sample_rate_hz < 1 or channels < 1 or rate_hz < 1:
        raise ValueError("sample rates and channel count must be positive.")
    if len(audio) % (_SAMPLE_BYTES * channels):
        raise ValueError("audio must be whole s16le frames for its channel count.")
    if channels == 1 and sample_rate_hz == rate_hz:
        return audio
    samples = array.array("h")
    samples.frombytes(audio)
    if sys.byteorder == "big":
        samples.byteswap()
    if channels == 1:
        mono: list[int] = samples.tolist()
    else:
        lanes = (samples[channel::channels] for channel in range(channels))
        mono = [round(sum(frame) / channels) for frame in zip(*lanes, strict=True)]
    converted: list[int] = []
    if sample_rate_hz == rate_hz or not mono:
        converted = mono
    else:
        count = len(mono)
        last = count - 1
        for index in range(count * rate_hz // sample_rate_hz):
            position, remainder = divmod(index * sample_rate_hz, rate_hz)
            left = mono[position]
            right = mono[position + 1] if position < last else left
            converted.append(round(left + (right - left) * remainder / rate_hz))
    output = array.array("h", converted)
    if sys.byteorder == "big":
        output.byteswap()
    return output.tobytes()


__all__ = ["to_mono"]
