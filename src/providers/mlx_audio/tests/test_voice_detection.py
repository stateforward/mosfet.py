from __future__ import annotations

from bot.abilities.hearing import voice
from bot.devices import phone

import asyncio
import collections.abc
import dataclasses
import pathlib

import pytest

from bot.providers.mlx_audio import VoiceDetectionError, VoiceDetector

SPEECH_WAV = (pathlib.Path(__file__).parent / "assets" / "speech.wav").read_bytes()
"""Real human speech, same encoding as the ring clip (see assets/SOURCES.md)."""

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
    # Bare start/end is the whole record MLX Audio emits, so a positive carries no confidence.
    model = FakeVoiceDetectionModel(timestamps=({"start": 0.0, "end": 0.7},))
    detector = VoiceDetector(model=model)

    output = asyncio.run(await_voice_detection(detector.classify(b"audio")))

    assert output == voice.detection.OutputData(is_voice=True, confidence=None)
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

@pytest.mark.live
def test_real_detector_hears_no_voice_in_the_ring_but_hears_speech() -> None:
    """Pin what a real Silero VAD perceives in the shipped ring, against a paired positive control.

    The ring must come back as *not* voice. Voice routes hearing to speech-to-text; the ring has to
    stay on the sound-classification route for the phone to be recognised as ringing at all. Nothing
    else in the suite exercises the real model, so a model or weights change that started hearing
    voice in a ringtone would silently cost the phone its ring perception.

    The speech assertion is what makes the ring assertion mean anything, and it is not optional.
    ``is_voice`` is ``bool(timestamps)``, so "no timestamps" is equally what this detector returns
    when the weights fail to load, when the model returns nothing, or when it degrades to
    always-False — the ring assertion passes trivially in every one of those cases. Putting the
    *same* detector instance against real speech proves it loaded, ran, and is able to answer True,
    which is what turns "no timestamps" into "correctly heard no voice." Neither half is worth
    keeping without the other.
    """

    detector = VoiceDetector()

    ring = asyncio.run(await_voice_detection(detector.classify(phone.RING_SOUND_WAV)))
    speech = asyncio.run(await_voice_detection(detector.classify(SPEECH_WAV)))

    assert ring.is_voice is False
    assert speech.is_voice is True
