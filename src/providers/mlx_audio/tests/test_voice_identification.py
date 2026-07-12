from __future__ import annotations

from bot.abilities.hearing import voice

import asyncio
import base64
import collections.abc
import dataclasses
import json
import typing

import pytest

from bot.providers.mlx_audio import VoiceIdentificationError, VoiceIdentifier

@dataclasses.dataclass(frozen=True)
class DecodedAudio:
    samples: tuple[float, ...]

@dataclasses.dataclass(frozen=True)
class FakeEmbedding:
    values: tuple[tuple[float, ...], ...]

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self.values]

@dataclasses.dataclass
class FakeVoiceIdentificationModel:
    embeddings: list[FakeEmbedding]
    calls: list[tuple[object, int]] = dataclasses.field(default_factory=list)

    def extract_speaker_embedding(self, audio: object, *, sr: int) -> FakeEmbedding:
        self.calls.append((audio, sr))
        return self.embeddings.pop(0)

class FailingVoiceIdentificationModel:
    def extract_speaker_embedding(self, audio: object, *, sr: int) -> FakeEmbedding:
        del audio, sr
        raise RuntimeError("mlx unavailable")

async def await_voice_identification(
    output: collections.abc.Awaitable[voice.identification.OutputData],
) -> voice.identification.OutputData:
    return await output

def test_voice_identifier_uses_injected_model() -> None:
    decoded_audio = DecodedAudio(samples=(0.1, 0.2, 0.3))
    decoded_inputs: list[tuple[bytes, int]] = []
    model = FakeVoiceIdentificationModel(embeddings=[FakeEmbedding(values=((0.1, 0.2),))])

    def decode_audio(audio: bytes, sample_rate: int) -> DecodedAudio:
        decoded_inputs.append((audio, sample_rate))
        return decoded_audio

    identifier = VoiceIdentifier(model=model, model_id="local/qwen3-speaker", decode_audio=decode_audio)

    output = asyncio.run(await_voice_identification(identifier.classify(_input())))

    assert len(output.signatures) == 1
    assert output.signatures[0].speaker_label == "speaker_1"
    assert output.signatures[0].confidence is None
    assert _signature_payload(output.signatures[0].signature) == {
        "embeddings": [[[0.1, 0.2]]],
        "format": "mlx-audio-speaker-embedding-v1",
        "model_id": "local/qwen3-speaker",
    }
    assert decoded_inputs == [(b"speaker audio 0", 24000)]
    assert model.calls == [(decoded_audio, 24000)]
    assert isinstance(identifier, voice.VoiceIdentifier)

def test_voice_identifier_uses_injected_loader() -> None:
    models: list[FakeVoiceIdentificationModel] = []

    def load_model(model_id: str) -> FakeVoiceIdentificationModel:
        assert model_id == "local/speaker-model"
        model = FakeVoiceIdentificationModel(embeddings=[FakeEmbedding(values=((0.3, 0.4),))])
        models.append(model)
        return model

    identifier = VoiceIdentifier(
        model_id="local/speaker-model",
        load_model=load_model,
        decode_audio=lambda audio, sample_rate: DecodedAudio(samples=(float(len(audio)), float(sample_rate))),
    )

    output = asyncio.run(await_voice_identification(identifier.classify(_input())))

    assert _signature_payload(output.signatures[0].signature)["embeddings"] == [[[0.3, 0.4]]]
    assert len(models) == 1

def test_voice_identifier_aggregates_repeated_speaker_segments() -> None:
    model = FakeVoiceIdentificationModel(
        embeddings=[
            FakeEmbedding(values=((0.1, 0.2),)),
            FakeEmbedding(values=((0.3, 0.4),)),
        ]
    )
    identifier = VoiceIdentifier(
        model=model,
        model_id="local/qwen3-speaker",
        decode_audio=lambda audio, sample_rate: DecodedAudio(samples=(float(len(audio)), float(sample_rate))),
    )

    output = asyncio.run(await_voice_identification(identifier.classify(_input(segment_count=2))))

    assert len(output.signatures) == 1
    assert output.signatures[0].speaker_label == "speaker_1"
    assert _signature_payload(output.signatures[0].signature)["embeddings"] == [
        [[0.1, 0.2]],
        [[0.3, 0.4]],
    ]

def test_voice_identifier_is_awaitable() -> None:
    identifier = VoiceIdentifier(
        model=FakeVoiceIdentificationModel(embeddings=[FakeEmbedding(values=((0.1,),))]),
        decode_audio=lambda audio, sample_rate: DecodedAudio(samples=(float(len(audio)), float(sample_rate))),
    )

    output = identifier.classify(_input())

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output).signatures[0].speaker_label == "speaker_1"

def test_voice_identifier_wraps_provider_errors() -> None:
    identifier = VoiceIdentifier(
        model=FailingVoiceIdentificationModel(),
        decode_audio=lambda audio, sample_rate: DecodedAudio(samples=(float(len(audio)), float(sample_rate))),
    )

    with pytest.raises(VoiceIdentificationError) as error:
        _ = asyncio.run(await_voice_identification(identifier.classify(_input())))

    assert isinstance(error.value.__cause__, RuntimeError)

def _input(*, segment_count: int = 1) -> voice.identification.InputData:
    return voice.identification.InputData(
        segments=tuple(
            voice.VoiceIdentificationSegment(
                diarization=voice.VoiceDiarizationSegment(
                    speaker_label="speaker_1",
                    start_seconds=float(index),
                    end_seconds=float(index) + 0.75,
                    confidence=0.87,
                ),
                audio=f"speaker audio {index}".encode("utf-8"),
            )
            for index in range(segment_count)
        )
    )

def _signature_payload(signature: str) -> dict[str, object]:
    prefix = "mlx-audio:speaker-embedding:"
    assert signature.startswith(prefix)
    payload = base64.urlsafe_b64decode(signature.removeprefix(prefix).encode("ascii"))
    decoded = typing.cast(object, json.loads(payload.decode("utf-8")))
    assert isinstance(decoded, dict)
    return typing.cast(dict[str, object], decoded)
