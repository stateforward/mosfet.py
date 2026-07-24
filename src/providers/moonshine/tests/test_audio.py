from __future__ import annotations

import array
import io
import wave

import pytest

from bot.providers.moonshine._audio import (
    audio_bytes_to_float_pcm,
    float_pcm_to_int16_le,
    float_pcm_to_wav,
)


def _wav_bytes(*, samples: list[int], sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    pcm = array.array("h", samples)
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_audio_bytes_to_float_pcm_reads_wav() -> None:
    wav = _wav_bytes(samples=[0, 16384, -16384], sample_rate=16_000)

    samples, rate = audio_bytes_to_float_pcm(wav)

    assert rate == 16_000
    assert samples[0] == 0.0
    assert samples[1] == pytest.approx(0.5, abs=1e-3)
    assert samples[2] == pytest.approx(-0.5, abs=1e-3)


def test_audio_bytes_to_float_pcm_reads_raw_int16() -> None:
    raw = array.array("h", [0, 32767]).tobytes()

    samples, rate = audio_bytes_to_float_pcm(raw, default_sample_rate_hz=48_000)

    assert rate == 48_000
    assert samples[0] == 0.0
    assert samples[1] == pytest.approx(0.99997, abs=1e-3)


def test_float_pcm_roundtrip_wav() -> None:
    original = [0.0, 0.25, -0.25]
    wav = float_pcm_to_wav(original, 24_000)

    samples, rate = audio_bytes_to_float_pcm(wav)

    assert rate == 24_000
    assert samples[0] == pytest.approx(0.0, abs=1e-3)
    assert samples[1] == pytest.approx(0.25, abs=1e-3)
    assert samples[2] == pytest.approx(-0.25, abs=1e-3)


def test_float_pcm_to_int16_le_length() -> None:
    pcm = float_pcm_to_int16_le([0.0, 1.0, -1.0])
    assert len(pcm) == 6
