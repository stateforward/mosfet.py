"""PCM / WAV helpers for Moonshine float-sample APIs."""

from __future__ import annotations

import array
import io
import struct
import typing
import wave


def audio_bytes_to_float_pcm(
    audio: bytes,
    *,
    default_sample_rate_hz: int = 16_000,
) -> tuple[list[float], int]:
    """Decode WAV or raw int16 LE PCM into mono float samples in ``[-1, 1]``."""

    if not audio:
        raise ValueError("Moonshine audio input is empty.")
    if _is_wav_container(audio):
        return _wav_to_float_pcm(audio)
    if len(audio) % 2 != 0:
        raise ValueError("Raw PCM audio must be little-endian int16 mono (even byte length).")
    return _int16_le_to_float(audio), default_sample_rate_hz


def float_pcm_to_wav(samples: list[float] | tuple[float, ...], sample_rate_hz: int) -> bytes:
    """Encode mono float samples as 16-bit PCM WAV."""

    if sample_rate_hz < 1:
        raise ValueError("sample_rate_hz must be positive.")
    pcm = array.array("h", (_float_to_int16(sample) for sample in samples))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def float_pcm_to_int16_le(samples: list[float] | tuple[float, ...]) -> bytes:
    """Encode mono float samples as raw little-endian int16 PCM."""

    return array.array("h", (_float_to_int16(sample) for sample in samples)).tobytes()


def _is_wav_container(audio: bytes) -> bool:
    return len(audio) >= 12 and audio[0:4] == b"RIFF" and audio[8:12] == b"WAVE"


def _wav_to_float_pcm(audio: bytes) -> tuple[list[float], int]:
    with wave.open(io.BytesIO(audio), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)
    if channels < 1:
        raise ValueError("WAV audio must have at least one channel.")
    if sample_width != 2:
        raise ValueError("Moonshine WAV decode currently supports 16-bit PCM only.")
    if sample_rate < 1:
        raise ValueError("WAV sample rate must be positive.")
    if len(frames) % (sample_width * channels) != 0:
        raise ValueError("WAV payload is truncated.")
    if channels == 1:
        return _int16_le_to_float(frames), sample_rate
    # Downmix multi-channel to mono by averaging samples per frame.
    frame_count = len(frames) // (sample_width * channels)
    mono: list[float] = []
    for index in range(frame_count):
        offset = index * sample_width * channels
        total = 0.0
        for channel in range(channels):
            start = offset + channel * sample_width
            sample = typing.cast(int, struct.unpack_from("<h", frames, start)[0])
            total += float(sample) / 32768.0
        mono.append(total / float(channels))
    return mono, sample_rate


def _int16_le_to_float(pcm: bytes) -> list[float]:
    samples = array.array("h")
    samples.frombytes(pcm)
    if samples.itemsize != 2:
        raise ValueError("Unexpected array item size for int16 PCM.")
    return [sample / 32768.0 for sample in samples]


def _float_to_int16(sample: float) -> int:
    if sample >= 1.0:
        return 32767
    if sample <= -1.0:
        return -32768
    return int(sample * 32767.0)


__all__ = [
    "audio_bytes_to_float_pcm",
    "float_pcm_to_int16_le",
    "float_pcm_to_wav",
]
