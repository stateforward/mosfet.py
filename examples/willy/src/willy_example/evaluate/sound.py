"""The evaluator's audio: everything at the room's rate (16 kHz mono s16le), mixed as the room heard it, and WAV.

Willy's voice reaches the room's speaker at its speech engine's rate (24 kHz from the lab); the person's lines are
said at the microphone's. :func:`resampled` brings either to the room's rate, :func:`reply` is Willy's voice in an
exchange one chunk after another (what is transcribed), :func:`mixed` is an exchange on one timeline, both voices
where they played (what is kept), and :func:`wav` wraps it. Pure functions: nothing here touches a file.
"""

import array
import io
import sys
import wave

from willy_example import room
from willy_example.evaluate import turns

RATE_HZ = room.MICROPHONE_RATE_HZ
"""The rate everything the evaluator keeps and transcribes is at: the room's microphone's."""
_PEAK = 32_767
_FLOOR = -32_768


def _samples(pcm: bytes) -> array.array[int]:
    values = array.array("h")
    values.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == "big":
        values.byteswap()
    return values


def _bytes(values: array.array[int]) -> bytes:
    if sys.byteorder == "big":
        values.byteswap()
    return values.tobytes()


def resampled(pcm: bytes, *, sample_rate_hz: int, channels: int = 1) -> bytes:
    """s16le ``pcm`` at ``sample_rate_hz`` with ``channels`` as mono at :data:`RATE_HZ`, by linear interpolation."""

    source = _samples(pcm)
    if channels > 1:
        frames = len(source) // channels
        source = array.array(
            "h", (round(sum(source[frame * channels : (frame + 1) * channels]) / channels) for frame in range(frames))
        )
    if not source:
        return b""
    if sample_rate_hz == RATE_HZ:
        return _bytes(source)
    step = sample_rate_hz / RATE_HZ
    last = len(source) - 1
    out = array.array("h")
    for index in range(len(source) * RATE_HZ // sample_rate_hz):
        at = index * step
        before = min(int(at), last)
        after = min(before + 1, last)
        out.append(round(source[before] + (source[after] - source[before]) * (at - before)))
    return _bytes(out)


def reply(chunks: tuple[turns.ChunkData, ...]) -> bytes:
    """Willy's voice in an exchange, its chunks one after another at :data:`RATE_HZ`."""

    return b"".join(
        resampled(chunk.pcm, sample_rate_hz=chunk.sample_rate_hz, channels=chunk.channels) for chunk in chunks
    )


def mixed(exchange: turns.ExchangeData, *, line: bytes, cut_in: bytes | None) -> bytes:
    """``exchange`` as the room heard it, at :data:`RATE_HZ`: the person's ``line`` (and ``cut_in``) and Willy's
    voice, each where it played, summed and clipped to 16 bits."""

    placed: list[tuple[float, bytes]] = [(exchange.said_from, line)]
    if cut_in is not None and exchange.cut_in_from is not None:
        placed.append((exchange.cut_in_from, cut_in))
    placed.extend(
        (chunk.start, resampled(chunk.pcm, sample_rate_hz=chunk.sample_rate_hz, channels=chunk.channels))
        for chunk in exchange.chunks
    )
    origin = min(at for at, _ in placed)
    length = max(round((at - origin) * RATE_HZ) + len(pcm) // 2 for at, pcm in placed)
    timeline = [0] * length
    for at, pcm in placed:
        offset = round((at - origin) * RATE_HZ)
        for index, value in enumerate(_samples(pcm)):
            timeline[offset + index] += value
    return _bytes(array.array("h", (min(_PEAK, max(_FLOOR, value)) for value in timeline)))


def wav(pcm: bytes) -> bytes:
    """Mono s16le ``pcm`` at :data:`RATE_HZ` as a WAV file."""

    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE_HZ)
        stream.writeframes(pcm)
    return output.getvalue()


__all__ = ["RATE_HZ", "mixed", "reply", "resampled", "wav"]
