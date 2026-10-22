from __future__ import annotations

import array
import asyncio
import collections.abc
import dataclasses
import time

import pytest

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech
from mosfet.providers.moonshine import SpeechEncoder
from tests.bot.abilities.vocal.speech import contract


@dataclasses.dataclass
class FakeTextToSpeech:
    samples: list[float]
    sample_rate: int = 24_000
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list[dict[str, object]])

    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        volume: float | None = None,
        options: collections.abc.Mapping[str, object] | None = None,
    ) -> tuple[list[float], int]:
        self.calls.append({"text": text, "speed": speed, "volume": volume, "options": options})
        return list(self.samples), self.sample_rate


class FailingTextToSpeech:
    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        volume: float | None = None,
        options: collections.abc.Mapping[str, object] | None = None,
    ) -> tuple[list[float], int]:
        del text, speed, volume, options
        raise RuntimeError("tts unavailable")


async def _all(stream: collections.abc.AsyncIterator[speech.Audio]) -> list[bytes]:
    return [chunk.pcm async for chunk in stream]


def _text(text: str) -> speech.Text:
    return contract.text(contract.words(text))


def test_encoder_yields_one_pcm_chunk_at_its_declared_rate() -> None:
    tts = FakeTextToSpeech(samples=[0.0, 0.5, -0.5], sample_rate=24_000)
    encoder = SpeechEncoder(tts=tts, speed=1.1)

    chunks = asyncio.run(contract.assert_speech_encoder(encoder, contract.words("Hello ", "Moonshine")))

    assert encoder.format == speech.Format(sample_rate_hz=24_000, channels=1)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello Moonshine"
    samples = array.array("h")
    samples.frombytes(chunks[0].pcm)
    assert samples.tolist()[0] == 0
    assert abs(samples.tolist()[1] - 0.5 * 32767) <= 2
    # One call per non-blank stream (the whole text, then the one closed after its chunk); blank text makes none.
    assert tts.calls == [{"text": "Hello Moonshine", "speed": 1.1, "volume": None, "options": None}] * 2


def test_encoder_loads_the_model_with_the_constructor_voice() -> None:
    loaded: list[dict[str, object]] = []

    def load_tts(**kwargs: object) -> FakeTextToSpeech:
        loaded.append(kwargs)
        return FakeTextToSpeech(samples=[0.0], sample_rate=22_050)

    encoder = SpeechEncoder(language="en-us", voice="kokoro_af_heart", sample_rate_hz=22_050, load_tts=load_tts)

    assert asyncio.run(_all(encoder.encode(_text("ok")))) == [b"\x00\x00"]
    assert [entry["voice"] for entry in loaded] == ["kokoro_af_heart"]
    assert {entry["language"] for entry in loaded} == {"en-us"}


def test_encoder_loads_its_model_once_across_calls() -> None:
    loaded: list[dict[str, object]] = []

    def load_tts(**kwargs: object) -> FakeTextToSpeech:
        loaded.append(kwargs)
        time.sleep(0.02)
        return FakeTextToSpeech(samples=[0.0], sample_rate=22_050)

    encoder = SpeechEncoder(sample_rate_hz=22_050, load_tts=load_tts)

    async def encode_repeatedly() -> None:
        _ = await _all(encoder.encode(_text("one")))
        _ = await _all(encoder.encode(_text("two")))
        _ = await asyncio.gather(*(_all(encoder.encode(_text("again"))) for _ in range(4)))

    asyncio.run(encode_repeatedly())

    assert len(loaded) == 1


def test_audio_at_another_rate_is_refused_rather_than_played_at_the_wrong_speed() -> None:
    encoder = SpeechEncoder(tts=FakeTextToSpeech(samples=[0.0], sample_rate=16_000))

    with pytest.raises(encoding.Error, match="sample_rate_hz=16000") as refused:
        _ = asyncio.run(_all(encoder.encode(_text("hi"))))
    assert refused.value.kind == "invalid_response"


def test_encoder_failures_are_encoding_errors() -> None:
    with pytest.raises(encoding.Error, match="Moonshine speech encoding failed") as failed:
        _ = asyncio.run(_all(SpeechEncoder(tts=FailingTextToSpeech()).encode(_text("fail"))))
    assert failed.value.kind == "encode_failed"

    with pytest.raises(encoding.Error) as empty:
        _ = asyncio.run(_all(SpeechEncoder(tts=FakeTextToSpeech(samples=[])).encode(_text("hi"))))
    assert empty.value.kind == "invalid_response"

    with pytest.raises(encoding.Error, match="non-blank") as blank:
        _ = asyncio.run(_all(SpeechEncoder(tts=FakeTextToSpeech(samples=[0.0])).encode(_text("   "))))
    assert blank.value.kind == "invalid_input"


def test_every_part_is_one_run_said_with_words_only() -> None:
    tts = FakeTextToSpeech(samples=[0.0], sample_rate=24_000)
    encoder = SpeechEncoder(tts=tts)
    pieces = [
        speech.Piece(part=0, words="Oh, ", expression=speech.Expression(sound="laugh", pace=1.5)),
        speech.Piece(part=1, words="hi.", expression=speech.Expression(emotion="calm")),
    ]

    chunks = asyncio.run(contract.assert_speech_encoder(encoder, pieces))

    assert encoder.expressiveness == speech.Expressiveness()
    assert [chunk.text for chunk in chunks] == ["Oh, hi."]


def test_a_pause_splits_the_text_and_is_said_as_silence_in_its_place() -> None:
    tts = FakeTextToSpeech(samples=[0.5], sample_rate=24_000)

    _ = asyncio.run(contract.assert_pauses(SpeechEncoder(tts=tts)))

    assert [call["text"] for call in tts.calls] == ["Hi.", " there."]
