from __future__ import annotations

from bot.abilities.vocal import speech

import asyncio
import collections.abc
import dataclasses

import hsm
import pytest

from bot import abilities

from bot.providers.mlx_audio import SpeechEncoder, SpeechEncodingError
from bot.providers.mlx_audio._mlx import SpeechGenerationResult
from tests.hsm_instance_state import start_ability_tree

@dataclasses.dataclass
class FakeSpeechGenerationResult:
    audio: object
    sample_rate: int

@dataclasses.dataclass
class FakeSpeechEncodingModel:
    results: tuple[SpeechGenerationResult, ...]
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def generate(self, text: str, **kwargs: object) -> collections.abc.Iterable[SpeechGenerationResult]:
        self.calls.append({"text": text, **kwargs})
        return self.results

class FailingSpeechEncodingModel:
    def generate(self, text: str, **kwargs: object) -> collections.abc.Iterable[SpeechGenerationResult]:
        del text, kwargs
        raise RuntimeError("mlx unavailable")

async def await_speech_encoding(output: collections.abc.Awaitable[bytes]) -> bytes:
    return await output

def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model

async def await_speech_encoding_ability(
    ability: speech.SpeechEncoding,
    input: bytes,
    completed: collections.abc.Callable[[], bool],
) -> None:
    await start_ability_tree(None, ability)
    _ = await ability.apply(input)
    for _ in range(100):
        if completed():
            return
        await asyncio.sleep(0)
    raise AssertionError("speech encoding ability did not run encoder")

def test_speech_encoder_uses_injected_model() -> None:
    writer_calls: list[tuple[object, int, str]] = []
    model = FakeSpeechEncodingModel(results=(FakeSpeechGenerationResult(audio=("samples",), sample_rate=24000),))

    def write_audio(audio: object, sample_rate: int, audio_format: str) -> bytes:
        writer_calls.append((audio, sample_rate, audio_format))
        return b"encoded-wav"

    encoder = SpeechEncoder(
        model=model,
        voice="Chelsie",
        speed=1.2,
        lang_code="English",
        audio_format="wav",
        write_audio=write_audio,
    )

    output = asyncio.run(await_speech_encoding(encoder.encode("hello".encode("utf-8"))))

    assert output == b"encoded-wav"
    assert model.calls == [
        {
            "text": "hello",
            "voice": "Chelsie",
            "speed": 1.2,
            "lang_code": "English",
            "verbose": False,
            "stream": False,
        }
    ]
    assert writer_calls == [(("samples",), 24000, "wav")]
    assert isinstance(encoder, abilities.Encoder)

def test_speech_encoder_uses_injected_loader() -> None:
    models: list[FakeSpeechEncodingModel] = []

    def load_model(model_id: str) -> FakeSpeechEncodingModel:
        assert model_id == "local/qwen3-tts"
        model = FakeSpeechEncodingModel(results=(FakeSpeechGenerationResult(audio=(0.1, 0.2), sample_rate=24000),))
        models.append(model)
        return model

    encoder = SpeechEncoder(
        model_id="local/qwen3-tts",
        load_model=load_model,
        write_audio=lambda audio, sample_rate, audio_format: f"{audio}:{sample_rate}:{audio_format}".encode("utf-8"),
    )

    output = asyncio.run(await_speech_encoding(encoder.encode(b"hello")))

    assert output == b"(0.1, 0.2):24000:wav"
    assert len(models) == 1

def test_speech_encoder_can_drive_speech_encoding_ability() -> None:
    writer_calls: list[tuple[object, int, str]] = []
    model = FakeSpeechEncodingModel(results=(FakeSpeechGenerationResult(audio=(0.1, 0.2), sample_rate=24000),))

    def write_audio(audio: object, sample_rate: int, audio_format: str) -> bytes:
        writer_calls.append((audio, sample_rate, audio_format))
        return b"encoded speech"

    encoder = SpeechEncoder(
        model=model,
        write_audio=write_audio,
    )
    ability = speech.SpeechEncoding(encoder=encoder)

    asyncio.run(await_speech_encoding_ability(ability, b"hello", lambda: bool(writer_calls)))

    assert model.calls == [
        {
            "text": "hello",
            "speed": 1.0,
            "lang_code": "auto",
            "verbose": False,
            "stream": False,
        }
    ]
    assert writer_calls == [((0.1, 0.2), 24000, "wav")]

def test_speech_encoder_is_awaitable() -> None:
    encoder = SpeechEncoder(
        model=FakeSpeechEncodingModel(results=(FakeSpeechGenerationResult(audio=(0.1,), sample_rate=24000),)),
        write_audio=lambda audio, sample_rate, audio_format: b"encoded speech",
    )

    output = encoder.encode(b"hello")

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == b"encoded speech"

def test_speech_encoder_wraps_provider_errors() -> None:
    encoder = SpeechEncoder(model=FailingSpeechEncodingModel())

    with pytest.raises(SpeechEncodingError) as error:
        _ = asyncio.run(await_speech_encoding(encoder.encode(b"hello")))

    assert isinstance(error.value.__cause__, RuntimeError)
