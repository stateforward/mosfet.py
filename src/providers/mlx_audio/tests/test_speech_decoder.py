from __future__ import annotations

from bot.abilities.hearing import speech

import asyncio
import collections.abc
import dataclasses
import pathlib
import sys
import types
import typing

import hsm
import pytest

from bot.providers.mlx_audio import SpeechDecoder, SpeechDecodingError
from bot.providers.mlx_audio._mlx import SpeechDecodingModel
from tests.hsm_instance_state import start_ability_tree

@dataclasses.dataclass(frozen=True)
class FakeTranscription:
    text: str

@dataclasses.dataclass
class FakeSpeechDecodingModel:
    result: object
    expected_audio: bytes = b"audio"
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def generate(self, audio: str, **kwargs: object) -> object:
        audio_path = pathlib.Path(audio)
        assert audio_path.read_bytes() == self.expected_audio
        self.calls.append({"audio": audio_path, **kwargs})
        return self.result

class FailingSpeechDecodingModel:
    def generate(self, audio: str, **kwargs: object) -> object:
        del audio, kwargs
        raise RuntimeError("mlx unavailable")

@dataclasses.dataclass
class StrictSpeechDecodingModel:
    calls: list[dict[str, object]] = dataclasses.field(default_factory=list)

    def generate(self, audio: str, *, verbose: bool) -> object:
        audio_path = pathlib.Path(audio)
        assert audio_path.read_bytes() == b"audio"
        self.calls.append({"audio": audio_path, "verbose": verbose})
        return {"text": "strict transcript"}

async def await_speech_decoding(output: collections.abc.Awaitable[bytes]) -> bytes:
    return await output

def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model

async def await_speech_decoding_ability(
    ability: speech.SpeechDecoding,
    input: bytes,
    completed: collections.abc.Callable[[], bool],
) -> None:
    await start_ability_tree(None, ability)
    _ = await ability.apply(input)
    for _ in range(100):
        if completed():
            return
        await asyncio.sleep(0)
    raise AssertionError("speech decoding ability did not run decoder")

def test_speech_decoder_uses_injected_model() -> None:
    model = FakeSpeechDecodingModel(result=FakeTranscription(text="hello there"))
    decoder = SpeechDecoder(
        model=model,
        model_id="local/whisper",
        language="en",
        verbose=True,
        generate_kwargs={"max_tokens": 32},
    )

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"hello there"
    assert len(model.calls) == 1
    audio_path = model.calls[0]["audio"]
    assert isinstance(audio_path, pathlib.Path)
    assert not audio_path.exists()
    assert model.calls[0] == {
        "audio": model.calls[0]["audio"],
        "language": "en",
        "verbose": True,
        "max_tokens": 32,
    }
    assert isinstance(decoder, speech.SpeechDecoder)

def test_speech_decoder_uses_injected_loader() -> None:
    models: list[FakeSpeechDecodingModel] = []

    def load_model(model_id: str) -> FakeSpeechDecodingModel:
        assert model_id == "local/whisper"
        model = FakeSpeechDecodingModel(result={"text": "loaded transcript"})
        models.append(model)
        return model

    decoder = SpeechDecoder(model_id="local/whisper", load_model=load_model)

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"loaded transcript"
    assert len(models) == 1

def test_speech_decoder_default_constructor_uses_mlx_audio_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_model_ids: list[str] = []
    model = FakeSpeechDecodingModel(result={"text": "default transcript"})
    mlx_audio_module = types.ModuleType("mlx_audio")
    stt_module = types.ModuleType("mlx_audio.stt")
    stt_utils_module = types.ModuleType("mlx_audio.stt.utils")

    def load_model(model_id: str) -> FakeSpeechDecodingModel:
        loaded_model_ids.append(model_id)
        return model

    setattr(stt_utils_module, "load_model", load_model)
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", stt_utils_module)
    decoder = SpeechDecoder()

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"default transcript"
    assert loaded_model_ids == ["mlx-community/whisper-large-v3-turbo"]
    assert len(model.calls) == 1

def test_speech_decoder_filters_unsupported_generate_kwargs() -> None:
    model = StrictSpeechDecodingModel()
    strict_model = typing.cast(SpeechDecodingModel, typing.cast(object, model))
    decoder = SpeechDecoder(
        model=strict_model,
        language="en",
        verbose=True,
        generate_kwargs={"max_tokens": 32},
    )

    output = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert output == b"strict transcript"
    assert len(model.calls) == 1
    audio_path = model.calls[0]["audio"]
    assert isinstance(audio_path, pathlib.Path)
    assert not audio_path.exists()
    assert model.calls[0] == {"audio": model.calls[0]["audio"], "verbose": True}

def test_speech_decoder_can_drive_speech_decoding_ability() -> None:
    model = FakeSpeechDecodingModel(result=FakeTranscription(text="ability transcript"))
    decoder = SpeechDecoder(model=model)
    ability = speech.SpeechDecoding(decoder=decoder)

    asyncio.run(await_speech_decoding_ability(ability, b"audio", lambda: bool(model.calls)))

    assert len(model.calls) == 1

def test_speech_decoder_is_awaitable() -> None:
    decoder = SpeechDecoder(model=FakeSpeechDecodingModel(result="plain transcript"))

    output = decoder.decode(b"audio")

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == b"plain transcript"

def test_speech_decoder_wraps_provider_errors() -> None:
    decoder = SpeechDecoder(model=FailingSpeechDecodingModel())

    with pytest.raises(SpeechDecodingError) as error:
        _ = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert isinstance(error.value.__cause__, RuntimeError)

def test_speech_decoder_wraps_loader_errors() -> None:
    def load_model(model_id: str) -> FakeSpeechDecodingModel:
        del model_id
        raise RuntimeError("model unavailable")

    decoder = SpeechDecoder(load_model=load_model)

    with pytest.raises(SpeechDecodingError) as error:
        _ = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert isinstance(error.value.__cause__, RuntimeError)

def test_speech_decoder_rejects_missing_transcription_text() -> None:
    decoder = SpeechDecoder(model=FakeSpeechDecodingModel(result={"segments": []}))

    with pytest.raises(SpeechDecodingError) as error:
        _ = asyncio.run(await_speech_decoding(decoder.decode(b"audio")))

    assert isinstance(error.value.__cause__, TypeError)
