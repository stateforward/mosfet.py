from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import collections.abc
import dataclasses
import pathlib

import pytest

from bot.providers.mlx_audio import VoiceDiarizationError, VoiceDiarizer


@dataclasses.dataclass(frozen=True)
class FakeDiarizationSegment:
    speaker: int
    start: float
    end: float
    confidence: float


@dataclasses.dataclass(frozen=True)
class FakeDiarizationResult:
    segments: tuple[object, ...]


@dataclasses.dataclass
class FakeVoiceDiarizationModel:
    result: FakeDiarizationResult
    expected_audio: bytes = b"audio"
    calls: list[tuple[pathlib.Path, float, bool]] = dataclasses.field(default_factory=list)

    def generate(self, audio: str, *, threshold: float, verbose: bool) -> FakeDiarizationResult:
        audio_path = pathlib.Path(audio)
        assert audio_path.read_bytes() == self.expected_audio
        self.calls.append((audio_path, threshold, verbose))
        return self.result


class FailingVoiceDiarizationModel:
    def generate(self, audio: str, *, threshold: float, verbose: bool) -> FakeDiarizationResult:
        del audio, threshold, verbose
        raise RuntimeError("mlx unavailable")


async def await_voice_diarization(
    output: collections.abc.Awaitable[voice.diarization.OutputData],
) -> voice.diarization.OutputData:
    return await output


def test_voice_diarizer_uses_injected_model() -> None:
    model = FakeVoiceDiarizationModel(
        result=FakeDiarizationResult(
            segments=(
                FakeDiarizationSegment(speaker=1, start=0.0, end=0.7, confidence=0.93),
                {
                    "speaker_label": "guest",
                    "start_seconds": 0.8,
                    "end_seconds": 1.4,
                    "score": 0.86,
                },
            )
        )
    )
    diarizer = VoiceDiarizer(model=model, threshold=0.6, verbose=True)

    output = asyncio.run(await_voice_diarization(diarizer.classify(b"audio")))

    assert output == voice.diarization.OutputData(
        segments=(
            voice.VoiceDiarizationSegment(
                speaker_label="speaker_1",
                start_seconds=0.0,
                end_seconds=0.7,
                confidence=0.93,
            ),
            voice.VoiceDiarizationSegment(
                speaker_label="guest",
                start_seconds=0.8,
                end_seconds=1.4,
                confidence=0.86,
            ),
        )
    )
    assert len(model.calls) == 1
    audio_path, threshold, verbose = model.calls[0]
    assert not audio_path.exists()
    assert threshold == 0.6
    assert verbose is True
    assert isinstance(diarizer, voice.VoiceDiarizer)


def test_voice_diarizer_uses_injected_loader() -> None:
    models: list[FakeVoiceDiarizationModel] = []

    def load_model(model_id: str) -> FakeVoiceDiarizationModel:
        assert model_id == "local/sortformer"
        model = FakeVoiceDiarizationModel(
            result=FakeDiarizationResult(
                segments=(FakeDiarizationSegment(speaker=0, start=0.0, end=0.3, confidence=0.81),)
            )
        )
        models.append(model)
        return model

    diarizer = VoiceDiarizer(model_id="local/sortformer", load_model=load_model)

    output = asyncio.run(await_voice_diarization(diarizer.classify(b"audio")))

    assert output.segments == (
        voice.VoiceDiarizationSegment(speaker_label="speaker_0", start_seconds=0.0, end_seconds=0.3, confidence=0.81),
    )
    assert len(models) == 1


def test_voice_diarizer_is_awaitable() -> None:
    diarizer = VoiceDiarizer(
        model=FakeVoiceDiarizationModel(
            result=FakeDiarizationResult(
                segments=(FakeDiarizationSegment(speaker=0, start=0.0, end=0.3, confidence=0.81),)
            )
        )
    )

    output = diarizer.classify(b"audio")

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output).segments[0].speaker_label == "speaker_0"


def test_voice_diarizer_wraps_provider_errors() -> None:
    diarizer = VoiceDiarizer(model=FailingVoiceDiarizationModel())

    with pytest.raises(VoiceDiarizationError) as error:
        _ = asyncio.run(await_voice_diarization(diarizer.classify(b"audio")))

    assert isinstance(error.value.__cause__, RuntimeError)
