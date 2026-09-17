from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing

import pytest

from mosfet.abilities.hearing import speech
from mosfet.providers.moonshine import SpeechDecoder, SpeechDecodingError


@dataclasses.dataclass(frozen=True)
class FakeLine:
    text: str


@dataclasses.dataclass(frozen=True)
class FakeTranscript:
    lines: tuple[FakeLine, ...]


@dataclasses.dataclass
class FakeTranscriber:
    result: FakeTranscript
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def transcribe_without_streaming(
        self,
        audio_data: list[float],
        sample_rate: int = 16000,
        flags: int = 0,
    ) -> FakeTranscript:
        self.calls.append(
            {
                "audio_data": list(audio_data),
                "sample_rate": sample_rate,
                "flags": flags,
            }
        )
        return self.result


class FailingTranscriber:
    def transcribe_without_streaming(
        self,
        audio_data: list[float],
        sample_rate: int = 16000,
        flags: int = 0,
    ) -> FakeTranscript:
        del audio_data, sample_rate, flags
        raise RuntimeError("moonshine unavailable")


async def await_speech_decoding(output: collections.abc.Awaitable[bytes] | bytes) -> bytes:
    if isinstance(output, bytes):
        return output
    return await typing.cast(collections.abc.Awaitable[bytes], output)


def test_speech_decoder_uses_injected_transcriber() -> None:
    transcriber = FakeTranscriber(result=FakeTranscript(lines=(FakeLine(text="hello moonshine"),)))
    decoder = SpeechDecoder(transcriber=transcriber, sample_rate_hz=16_000)

    # Three int16 samples (silence-ish) as raw PCM.
    pcm = (0).to_bytes(2, "little", signed=True) * 3
    output = asyncio.run(await_speech_decoding(decoder.decode(pcm)))

    assert output == b"hello moonshine"
    assert len(transcriber.calls) == 1
    assert transcriber.calls[0]["sample_rate"] == 16_000
    assert len(typing.cast(list[object], transcriber.calls[0]["audio_data"])) == 3
    assert isinstance(decoder, speech.SpeechDecoder)


def test_speech_decoder_uses_injected_loader() -> None:
    transcribers: list[FakeTranscriber] = []

    def load_transcriber(**kwargs: object) -> FakeTranscriber:
        assert kwargs["language"] == "en"
        transcriber = FakeTranscriber(result=FakeTranscript(lines=(FakeLine(text="loaded"),)))
        transcribers.append(transcriber)
        return transcriber

    decoder = SpeechDecoder(language="en", load_transcriber=load_transcriber)
    pcm = (0).to_bytes(2, "little", signed=True) * 2
    output = asyncio.run(await_speech_decoding(decoder.decode(pcm)))

    assert output == b"loaded"
    assert len(transcribers) == 1


def test_speech_decoder_wraps_runtime_errors() -> None:
    decoder = SpeechDecoder(transcriber=FailingTranscriber())
    pcm = (0).to_bytes(2, "little", signed=True) * 2

    with pytest.raises(SpeechDecodingError, match="Moonshine speech decoding failed"):
        _ = asyncio.run(await_speech_decoding(decoder.decode(pcm)))


def test_speech_decoder_rejects_empty_transcript() -> None:
    transcriber = FakeTranscriber(result=FakeTranscript(lines=(FakeLine(text="  "),)))
    decoder = SpeechDecoder(transcriber=transcriber)
    pcm = (0).to_bytes(2, "little", signed=True) * 2

    with pytest.raises(SpeechDecodingError, match="empty text"):
        _ = asyncio.run(await_speech_decoding(decoder.decode(pcm)))
