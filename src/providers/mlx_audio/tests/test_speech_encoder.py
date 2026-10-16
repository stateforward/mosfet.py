from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import typing

import pytest

from mosfet.abilities import encoding
from mosfet.abilities.vocal import speech
from mosfet.providers.mlx_audio import SpeechEncoder
from mosfet.providers.mlx_audio._mlx import SpeechGenerationResult
from tests.bot.abilities.vocal.speech import contract


@dataclasses.dataclass
class FakeSpeechGenerationResult:
    audio: object
    sample_rate: int


@dataclasses.dataclass
class FakeSpeechEncodingModel:
    results: tuple[SpeechGenerationResult, ...]
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list[dict[str, object]])
    generated: int = 0
    closed: bool = False

    def generate(self, text: str, **kwargs: object) -> collections.abc.Iterable[SpeechGenerationResult]:
        self.calls.append({"text": text, **kwargs})
        return self._stream()

    def _stream(self) -> collections.abc.Generator[SpeechGenerationResult]:
        try:
            for result in self.results:
                self.generated += 1
                yield result
        finally:
            self.closed = True


class FailingSpeechEncodingModel:
    def generate(self, text: str, **kwargs: object) -> collections.abc.Iterable[SpeechGenerationResult]:
        del text, kwargs
        raise RuntimeError("mlx unavailable")


def as_pcm(audio: object) -> bytes:
    assert isinstance(audio, tuple)
    return bytes(len(typing.cast(tuple[float, ...], audio)) * 2)


def chunk(samples: int, *, sample_rate: int = 24_000) -> FakeSpeechGenerationResult:
    return FakeSpeechGenerationResult(audio=(0.0,) * samples, sample_rate=sample_rate)


async def _all(stream: collections.abc.AsyncIterator[speech.Audio]) -> list[speech.Audio]:
    return [item async for item in stream]


def _text(*pieces: str) -> speech.Text:
    return contract.text(contract.words(*pieces))


def _parts(*parts: tuple[str, speech.Expression | None]) -> speech.Text:
    return contract.text(
        [
            speech.Piece(part=index, words=words, expression=expression)
            for index, (words, expression) in enumerate(parts)
        ]
    )


def test_encoder_reads_the_text_to_its_end_and_streams_each_generated_chunk_aligned_to_the_whole_text() -> None:
    model = FakeSpeechEncodingModel(results=(chunk(2), chunk(3)))
    encoder = SpeechEncoder(model=model, voice="Chelsie", speed=1.2, lang_code="English", convert=as_pcm)

    chunks = asyncio.run(contract.assert_speech_encoder(encoder, contract.words("hel", "lo")))

    assert encoder.format == speech.Format(sample_rate_hz=24_000, channels=1)
    assert chunks == [speech.Audio(pcm=bytes(4), text=""), speech.Audio(pcm=bytes(6), text="hello")]
    # One generation per non-blank stream (the whole text, then the one closed after its chunk); blank text has none.
    assert (
        model.calls
        == [
            {
                "text": "hello",
                "voice": "Chelsie",
                "speed": 1.2,
                "lang_code": "English",
                "verbose": False,
                "stream": True,
            }
        ]
        * 2
    )


def test_encoder_loads_its_model_once_and_speaks_in_the_constructor_voice() -> None:
    loaded: list[str] = []
    model = FakeSpeechEncodingModel(results=(chunk(1),))

    def load_model(model_id: str) -> FakeSpeechEncodingModel:
        loaded.append(model_id)
        return model

    default = SpeechEncoder(model_id="local/qwen3-tts", load_model=load_model, convert=as_pcm)
    ryan = SpeechEncoder(model=model, voice="Ryan", convert=as_pcm)

    _ = asyncio.run(_all(default.encode(_text("one"))))
    _ = asyncio.run(_all(default.encode(_text("two"))))
    _ = asyncio.run(_all(ryan.encode(_text("three"))))

    assert loaded == ["local/qwen3-tts"]
    assert "voice" not in model.calls[0]
    assert model.calls[2]["voice"] == "Ryan"


def test_closing_the_stream_between_chunks_stops_the_generation() -> None:
    model = FakeSpeechEncodingModel(results=(chunk(1), chunk(1), chunk(1)))

    async def run() -> None:
        stream = SpeechEncoder(model=model, convert=as_pcm).encode(_text("hello"))
        _ = await anext(stream)
        assert isinstance(stream, collections.abc.AsyncGenerator)
        await stream.aclose()

    asyncio.run(run())

    # The first chunk is held until the second arrives, so the last one can carry the text.
    assert model.closed
    assert model.generated == 2


def test_a_chunk_at_another_rate_is_refused() -> None:
    model = FakeSpeechEncodingModel(results=(chunk(1, sample_rate=16_000),))

    with pytest.raises(encoding.Error, match="sample_rate_hz=16000") as refused:
        _ = asyncio.run(_all(SpeechEncoder(model=model, convert=as_pcm).encode(_text("hi"))))

    assert refused.value.kind == "invalid_response"


def test_provider_failures_are_encoding_errors() -> None:
    with pytest.raises(encoding.Error) as failed:
        _ = asyncio.run(_all(SpeechEncoder(model=FailingSpeechEncodingModel()).encode(_text("hello"))))

    assert failed.value.kind == "encode_failed"
    assert isinstance(failed.value.__cause__, RuntimeError)


def test_blank_text_and_no_audio_are_refused() -> None:
    with pytest.raises(encoding.Error) as blank:
        _ = asyncio.run(_all(SpeechEncoder(model=FakeSpeechEncodingModel(results=())).encode(_text(" "))))
    with pytest.raises(encoding.Error) as silent:
        _ = asyncio.run(_all(SpeechEncoder(model=FakeSpeechEncodingModel(results=())).encode(_text("hi"))))

    assert (blank.value.kind, silent.value.kind) == ("invalid_input", "invalid_response")


_QWEN = speech.Expressiveness(
    cues=frozenset({"instructions", "emotion", "intensity", "emphasis", "pace"}), pace=speech.Range(min=0.8, max=1.5)
)
"""A checkpoint that takes ``instruct`` and ``speed``, as composition declares it."""


def test_a_declared_pace_is_the_runs_speed_clamped_and_natural_language_cues_are_its_instruct() -> None:
    model = FakeSpeechEncodingModel(results=(chunk(1),))
    encoder = SpeechEncoder(model=model, expressiveness=_QWEN, speed=1.1, convert=as_pcm)
    secret = speech.Expression(instructions="Like a secret.", emotion="amused", intensity=0.5, emphasis=True, pace=2.0)

    chunks = asyncio.run(_all(encoder.encode(_parts(("Psst. ", secret), ("Hello.", None)))))

    assert [(call["text"], call["speed"], call.get("instruct")) for call in model.calls] == [
        ("Psst. ", 1.5, "Like a secret. Emotion: amused. Intensity: 0.5 out of 1. Stress these words."),
        ("Hello.", 1.1, None),
    ]
    assert [item.text for item in chunks] == ["Psst. ", "Hello."]


def test_parts_with_one_direction_merge_into_one_generation_and_sounds_are_ignored() -> None:
    model = FakeSpeechEncodingModel(results=(chunk(1), chunk(2)))
    encoder = SpeechEncoder(model=model, expressiveness=_QWEN, convert=as_pcm)

    chunks = asyncio.run(
        _all(
            encoder.encode(
                _parts(
                    ("Oh, ", speech.Expression(sound="laugh", emotion="amused")),
                    ("you got me.", speech.Expression(emotion="amused", pitch=4.0)),
                )
            )
        )
    )

    assert [(call["text"], call["instruct"]) for call in model.calls] == [("Oh, you got me.", "Emotion: amused.")]
    assert chunks == [speech.Audio(pcm=bytes(2), text=""), speech.Audio(pcm=bytes(4), text="Oh, you got me.")]


def test_undeclared_cues_never_change_a_generation() -> None:
    model = FakeSpeechEncodingModel(results=(chunk(1),))
    encoder = SpeechEncoder(model=model, convert=as_pcm)

    _ = asyncio.run(_all(encoder.encode(_parts(("A ", speech.Expression(pace=1.4, emotion="sad")), ("b.", None)))))

    assert model.calls == [{"text": "A b.", "speed": 1.0, "lang_code": "auto", "verbose": False, "stream": True}]


def test_a_cue_a_generation_cannot_carry_is_refused_at_construction() -> None:
    for cue in ("style", "sound", "pitch", "volume"):
        declared = speech.Expressiveness.model_validate(
            {"cues": {cue}}
            | ({"styles": ("whisper",)} if cue == "style" else {})
            | ({"sounds": {"laugh"}} if cue == "sound" else {})
            | ({cue: {"min": 0.5, "max": 1.5}} if cue in ("pitch", "volume") else {})
        )
        with pytest.raises(ValueError, match=cue):
            _ = SpeechEncoder(model=FakeSpeechEncodingModel(results=()), expressiveness=declared)
