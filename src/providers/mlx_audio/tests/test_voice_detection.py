from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import collections.abc
import dataclasses
import pathlib

import pytest

from bot.providers.mlx_audio import VoiceDetectionError, VoiceDetector

@dataclasses.dataclass
class FakeVoiceDetectionModel:
    timestamps: tuple[object, ...]
    expected_audio: bytes = b"audio"
    calls: list[pathlib.Path] = dataclasses.field(default_factory=list)

    def get_speech_timestamps(self, audio: str, *, return_seconds: bool) -> tuple[object, ...]:
        audio_path = pathlib.Path(audio)
        assert audio_path.read_bytes() == self.expected_audio
        assert return_seconds is True
        self.calls.append(audio_path)
        return self.timestamps

class FailingVoiceDetectionModel:
    def get_speech_timestamps(self, audio: str, *, return_seconds: bool) -> tuple[object, ...]:
        del audio, return_seconds
        raise RuntimeError("mlx unavailable")

async def await_voice_detection(output: collections.abc.Awaitable[voice.detection.OutputData]) -> voice.detection.OutputData:
    return await output

def test_voice_detector_uses_injected_model() -> None:
    model = FakeVoiceDetectionModel(timestamps=({"start": 0.0, "end": 0.7, "probability": 0.93},))
    detector = VoiceDetector(model=model)

    output = asyncio.run(await_voice_detection(detector.classify(b"audio")))

    assert output == voice.detection.OutputData(is_voice=True, confidence=0.93)
    assert len(model.calls) == 1
    assert not model.calls[0].exists()
    assert isinstance(detector, voice.VoiceDetector)

def test_voice_detector_uses_injected_loader() -> None:
    models: list[FakeVoiceDetectionModel] = []

    def load_model(model_id: str) -> FakeVoiceDetectionModel:
        assert model_id == "local/silero"
        model = FakeVoiceDetectionModel(timestamps=())
        models.append(model)
        return model

    detector = VoiceDetector(model_id="local/silero", load_model=load_model)

    output = asyncio.run(await_voice_detection(detector.classify(b"audio")))

    assert output == voice.detection.OutputData(is_voice=False, confidence=None)
    assert len(models) == 1

def test_voice_detector_is_awaitable() -> None:
    detector = VoiceDetector(model=FakeVoiceDetectionModel(timestamps=()))

    output = detector.classify(b"audio")

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output) == voice.detection.OutputData(is_voice=False, confidence=None)

def test_voice_detector_wraps_provider_errors() -> None:
    detector = VoiceDetector(model=FailingVoiceDetectionModel())

    with pytest.raises(VoiceDetectionError) as error:
        _ = asyncio.run(await_voice_detection(detector.classify(b"audio")))

    assert isinstance(error.value.__cause__, RuntimeError)
