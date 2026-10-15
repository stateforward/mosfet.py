"""Unit tests for ``speech_server.resample``: s16le PCM to the servers' 16 kHz mono."""

from __future__ import annotations

import array
import sys

import pytest

from mosfet.providers.speech_server import resample


def _pcm(*samples: int) -> bytes:
    values = array.array("h", samples)
    if sys.byteorder == "big":
        values.byteswap()
    return values.tobytes()


def _samples(audio: bytes) -> list[int]:
    values = array.array("h")
    values.frombytes(audio)
    if sys.byteorder == "big":
        values.byteswap()
    return values.tolist()


def test_sixteen_kilohertz_mono_passes_unchanged() -> None:
    audio = _pcm(1, -2, 3, -4)

    assert resample.to_mono(audio, sample_rate_hz=16_000, channels=1) == audio


def test_forty_eight_kilohertz_stereo_becomes_a_third_as_many_mono_frames() -> None:
    frames = 4_800  # 100 ms
    audio = _pcm(*(value for index in range(frames) for value in (index % 100, -(index % 100))))

    converted = resample.to_mono(audio, sample_rate_hz=48_000, channels=2)

    assert len(converted) == frames // 3 * 2
    assert set(_samples(converted)) == {0}


def test_channels_are_averaged_into_one() -> None:
    audio = _pcm(100, 300, -100, -301, 32_767, 32_767)

    assert _samples(resample.to_mono(audio, sample_rate_hz=16_000, channels=2)) == [200, -200, 32_767]


def test_downsampling_interpolates_linearly_between_neighbors() -> None:
    audio = _pcm(0, 30, 60, 90, 120, 150)

    assert _samples(resample.to_mono(audio, sample_rate_hz=24_000, channels=1)) == [0, 45, 90, 135]


def test_upsampling_interpolates_and_holds_the_last_sample() -> None:
    audio = _pcm(0, 100)

    assert _samples(resample.to_mono(audio, sample_rate_hz=8_000, channels=1)) == [0, 50, 100, 100]


def test_another_target_rate_is_honored() -> None:
    audio = _pcm(*range(48))

    assert len(resample.to_mono(audio, sample_rate_hz=48_000, channels=1, rate_hz=24_000)) == 24 * 2


def test_empty_audio_stays_empty() -> None:
    assert resample.to_mono(b"", sample_rate_hz=48_000, channels=2) == b""


@pytest.mark.parametrize(
    ("audio", "sample_rate_hz", "channels", "rate_hz"),
    [
        (b"\x01\x00\x02", 16_000, 1, 16_000),
        (b"\x01\x00", 16_000, 2, 16_000),
        (b"", 0, 1, 16_000),
        (b"", 16_000, 0, 16_000),
        (b"", 16_000, 1, 0),
    ],
)
def test_partial_frames_and_non_positive_formats_are_refused(
    audio: bytes, sample_rate_hz: int, channels: int, rate_hz: int
) -> None:
    with pytest.raises(ValueError):
        _ = resample.to_mono(audio, sample_rate_hz=sample_rate_hz, channels=channels, rate_hz=rate_hz)
