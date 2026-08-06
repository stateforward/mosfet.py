from __future__ import annotations

from bot.abilities.hearing import voice
from bot import abilities
import bot.providers.pyannote as pyannote
from bot.providers.pyannote import (
    Classifier,
    VoiceIdentificationError,
    load_speaker_embedding_inference,
)

import asyncio
import collections.abc
import dataclasses
import importlib
import os
import pathlib
import types
import wave

import pytest


@dataclasses.dataclass(frozen=True)
class FakeEmbedding:
    values: tuple[tuple[float, ...], ...]

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self.values]


@dataclasses.dataclass(frozen=True)
class FakeSlidingWindowFeature:
    data: tuple[tuple[float, ...], ...]


@dataclasses.dataclass
class FakeInference:
    embeddings: list[object]
    expected_audio: list[bytes] = dataclasses.field(default_factory=list)
    calls: list[pathlib.Path] = dataclasses.field(default_factory=list)

    def __call__(self, audio: pathlib.Path) -> object:
        with wave.open(str(audio), "rb") as staged_audio:
            assert staged_audio.getframerate() == 16000
            assert staged_audio.getnchannels() == 1
            assert staged_audio.getsampwidth() == 2
            if self.expected_audio:
                assert staged_audio.readframes(staged_audio.getnframes()) == self.expected_audio.pop(0)
        self.calls.append(audio)
        return self.embeddings.pop(0)


class FailingInference:
    def __init__(self) -> None:
        self.error: RuntimeError = RuntimeError("pyannote unavailable")

    def __call__(self, audio: pathlib.Path) -> object:
        del audio
        raise self.error


async def _await_output(
    output: collections.abc.Awaitable[voice.identification.OutputData],
) -> voice.identification.OutputData:
    return await output


def test_classifier_converts_tolist_embedding_to_core_output() -> None:
    input_data = _input()
    inference = FakeInference(
        embeddings=[FakeEmbedding(values=((0.1, 0.2, 0.3),))],
        expected_audio=[input_data.segments[0].audio],
    )
    classifier = Classifier(inference=inference, model_id="local/embedding")

    output = asyncio.run(_await_output(classifier.classify(input_data)))

    assert output.embeddings == (
        voice.VoiceEmbedding(embedding=(0.1, 0.2, 0.3), model="local/embedding", confidence=None),
    )
    assert len(inference.calls) == 1
    assert not inference.calls[0].exists()
    assert isinstance(classifier, abilities.Classifier)
    assert not hasattr(pyannote, "VoiceIdentifier")


def test_classifier_unwraps_sliding_window_feature_data() -> None:
    inference = FakeInference(
        embeddings=[FakeSlidingWindowFeature(data=((0.1, 0.2, 0.3),))],
    )
    classifier = Classifier(inference=inference, model_id="local/embedding")

    output = asyncio.run(_await_output(classifier.classify(_input())))

    assert output.embeddings[0].embedding == (0.1, 0.2, 0.3)


def test_classifier_uses_injected_loader_and_preserves_input_order() -> None:
    inferences: list[FakeInference] = []

    def load_inference(model_id: str) -> FakeInference:
        assert model_id == "local/embedding"
        inference = FakeInference(embeddings=[FakeEmbedding(values=((0.1,),)), FakeEmbedding(values=((0.2,),))])
        inferences.append(inference)
        return inference

    classifier = Classifier(model_id="local/embedding", load_inference=load_inference)

    output = asyncio.run(_await_output(classifier.classify(_input(segment_count=2))))

    assert [embedding.embedding for embedding in output.embeddings] == [(0.1,), (0.2,)]
    assert len(inferences) == 1


def test_classifier_is_awaitable() -> None:
    classifier = Classifier(inference=FakeInference(embeddings=[FakeEmbedding(values=((0.1,),))]))

    output = classifier.classify(_input())

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output).embeddings[0].embedding == (0.1,)


def test_classifier_failure_names_underlying_inference_error() -> None:
    inference = FailingInference()
    classifier = Classifier(inference=inference)

    with pytest.raises(VoiceIdentificationError) as error:
        _ = asyncio.run(_await_output(classifier.classify(_input())))

    assert error.value.__cause__ is inference.error
    message = str(error.value)
    assert "RuntimeError" in message
    assert "pyannote unavailable" in message


def test_classifier_rejects_non_finite_embedding() -> None:
    classifier = Classifier(inference=FakeInference(embeddings=[FakeEmbedding(values=((float("nan"),),))]))

    with pytest.raises(VoiceIdentificationError) as error:
        _ = asyncio.run(_await_output(classifier.classify(_input())))

    assert isinstance(error.value.__cause__, ValueError)


def test_default_loader_matches_pyannote_model_and_inference_api(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_model = object()
    calls: list[tuple[object, ...]] = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id: str) -> object:
            del cls
            calls.append((model_id, loaded_model))
            return loaded_model

    class FakeInferenceLoader:
        def __init__(self, model: object, *, window: str) -> None:
            calls.append(("Inference", model, window))

    module = types.SimpleNamespace(Model=FakeModel, Inference=FakeInferenceLoader)

    def import_module(name: str) -> types.SimpleNamespace:
        assert name == "pyannote.audio"
        return module

    monkeypatch.setattr(importlib, "import_module", import_module)

    inference = load_speaker_embedding_inference("pyannote/wespeaker-voxceleb-resnet34-LM")

    assert isinstance(inference, FakeInferenceLoader)
    assert calls == [
        ("pyannote/wespeaker-voxceleb-resnet34-LM", loaded_model),
        ("Inference", loaded_model, "whole"),
    ]


@pytest.mark.parametrize("inherited_backend", [None, "module://matplotlib_inline.backend_inline"])
def test_default_loader_selects_headless_matplotlib_backend(
    monkeypatch: pytest.MonkeyPatch,
    inherited_backend: str | None,
) -> None:
    if inherited_backend is None:
        monkeypatch.delenv("MPLBACKEND", raising=False)
    else:
        monkeypatch.setenv("MPLBACKEND", inherited_backend)
    calls: list[tuple[object, ...]] = []
    loaded_model = object()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id: str) -> object:
            del cls
            calls.append((model_id, loaded_model))
            return loaded_model

    class FakeInferenceLoader:
        def __init__(self, model: object, *, window: str) -> None:
            calls.append(("Inference", model, window))

    module = types.SimpleNamespace(Model=FakeModel, Inference=FakeInferenceLoader)

    def import_module(name: str) -> types.SimpleNamespace:
        assert name == "pyannote.audio"
        assert os.environ["MPLBACKEND"] == "Agg"
        return module

    monkeypatch.setattr(importlib, "import_module", import_module)

    _ = load_speaker_embedding_inference("pyannote/wespeaker-voxceleb-resnet34-LM")


def test_classifier_defaults_to_pyannote_wespeaker_model() -> None:
    assert Classifier().model_id == "pyannote/wespeaker-voxceleb-resnet34-LM"


def _input(*, segment_count: int = 1) -> voice.identification.InputData:
    return voice.identification.InputData(
        segments=tuple(
            voice.VoiceSegment(
                audio=b"\x00\x00" * 12_000,
                media_type="audio/pcm",
                sample_rate_hz=16000,
                channels=1,
                start_seconds=float(index),
                end_seconds=float(index) + 0.75,
            )
            for index in range(segment_count)
        )
    )
