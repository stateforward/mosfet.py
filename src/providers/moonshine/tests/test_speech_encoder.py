from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing

import pytest

from mosfet.providers.moonshine import SpeechEncoder, SpeechEncodingError
from mosfet.providers.moonshine._audio import audio_bytes_to_float_pcm


@dataclasses.dataclass
class FakeTextToSpeech:
    samples: list[float]
    sample_rate: int = 24_000
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        volume: float | None = None,
        options: typing.Mapping[str, object] | None = None,
    ) -> tuple[list[float], int]:
        self.calls.append(
            {
                "text": text,
                "speed": speed,
                "volume": volume,
                "options": options,
            }
        )
        return list(self.samples), self.sample_rate


class FailingTextToSpeech:
    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        volume: float | None = None,
        options: typing.Mapping[str, object] | None = None,
    ) -> tuple[list[float], int]:
        del text, speed, volume, options
        raise RuntimeError("tts unavailable")


async def await_encode(output: collections.abc.Awaitable[bytes] | bytes) -> bytes:
    if isinstance(output, bytes):
        return output
    return await typing.cast(collections.abc.Awaitable[bytes], output)


def test_speech_encoder_uses_injected_tts_and_returns_wav() -> None:
    tts = FakeTextToSpeech(samples=[0.0, 0.5, -0.5], sample_rate=24_000)
    encoder = SpeechEncoder(tts=tts, speed=1.1, audio_format="wav")

    wav = asyncio.run(await_encode(encoder.encode(b"Hello Moonshine")))

    assert wav[0:4] == b"RIFF"
    samples, rate = audio_bytes_to_float_pcm(wav)
    assert rate == 24_000
    assert samples[1] == pytest.approx(0.5, abs=1e-2)
    assert tts.calls == [
        {
            "text": "Hello Moonshine",
            "speed": 1.1,
            "volume": None,
            "options": None,
        }
    ]


def test_speech_encoder_pcm_format() -> None:
    tts = FakeTextToSpeech(samples=[0.0, 1.0], sample_rate=16_000)
    encoder = SpeechEncoder(tts=tts, audio_format="pcm")

    pcm = asyncio.run(await_encode(encoder.encode(b"hi")))

    assert len(pcm) == 4
    assert tts.calls[0]["text"] == "hi"


def test_speech_encoder_uses_injected_loader() -> None:
    loaded: list[FakeTextToSpeech] = []

    def load_tts(**kwargs: object) -> FakeTextToSpeech:
        assert kwargs["language"] == "en-us"
        assert kwargs["voice"] == "kokoro_af_heart"
        tts = FakeTextToSpeech(samples=[0.0], sample_rate=22_050)
        loaded.append(tts)
        return tts

    encoder = SpeechEncoder(language="en-us", voice="kokoro_af_heart", load_tts=load_tts)
    wav = asyncio.run(await_encode(encoder.encode(b"ok")))

    assert wav[0:4] == b"RIFF"
    assert len(loaded) == 1


def test_speech_encoder_wraps_runtime_errors() -> None:
    encoder = SpeechEncoder(tts=FailingTextToSpeech())

    with pytest.raises(SpeechEncodingError, match="Moonshine speech encoding failed"):
        _ = asyncio.run(await_encode(encoder.encode(b"fail")))


def test_speech_encoder_rejects_blank_text() -> None:
    encoder = SpeechEncoder(tts=FakeTextToSpeech(samples=[0.0]))

    with pytest.raises(SpeechEncodingError, match="non-empty text"):
        _ = asyncio.run(await_encode(encoder.encode(b"   ")))
