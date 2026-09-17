from __future__ import annotations

from mosfet.abilities.hearing import voice

import asyncio
import collections.abc
import dataclasses
import io
import pathlib
import wave

import pytest

from mosfet.providers.mlx_audio import VoiceDiarizationError, VoiceDiarizer


@dataclasses.dataclass(frozen=True)
class FakeDiarizationSegment:
    start_seconds: float
    end_seconds: float
    confidence: float


@dataclasses.dataclass(frozen=True)
class FakeDiarizationResult:
    segments: tuple[object, ...]


@dataclasses.dataclass
class FakeVoiceDiarizationModel:
    result: FakeDiarizationResult
    expected_audio: bytes
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
    input_data = _input()
    model = FakeVoiceDiarizationModel(
        result=FakeDiarizationResult(
            segments=(
                FakeDiarizationSegment(start_seconds=0.0, end_seconds=0.7, confidence=0.93),
                {
                    "start_seconds": 0.8,
                    "end_seconds": 1.4,
                    "score": 0.86,
                },
            )
        ),
        expected_audio=_wav_audio(input_data.audio, sample_rate_hz=input_data.sample_rate_hz),
    )
    clipped_audio: list[tuple[bytes, float, float]] = []

    def clip_audio(input: voice.diarization.InputData, start_seconds: float, end_seconds: float) -> bytes:
        clipped_audio.append((input.audio, start_seconds, end_seconds))
        frame_count = round((end_seconds - start_seconds) * input.sample_rate_hz)
        return b"\x00" * (2 * input.channels * max(1, frame_count))

    diarizer = VoiceDiarizer(model=model, threshold=0.6, verbose=True, clip_audio=clip_audio)

    output = asyncio.run(await_voice_diarization(diarizer.classify(input_data)))

    assert output == voice.diarization.OutputData(
        segments=(
            voice.VoiceDiarizationSegment(
                audio=b"\x00\x00" * 7,
                media_type="audio/pcm",
                sample_rate_hz=10,
                channels=1,
                start_seconds=0.0,
                end_seconds=0.7,
                confidence=0.93,
            ),
            voice.VoiceDiarizationSegment(
                audio=b"\x00\x00" * 6,
                media_type="audio/pcm",
                sample_rate_hz=10,
                channels=1,
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
    assert clipped_audio == [(input_data.audio, 0.0, 0.7), (input_data.audio, 0.8, 1.4)]
    assert isinstance(diarizer, voice.VoiceDiarizer)


def test_voice_diarizer_uses_injected_loader() -> None:
    input_data = _input()
    models: list[FakeVoiceDiarizationModel] = []

    def load_model(model_id: str) -> FakeVoiceDiarizationModel:
        assert model_id == "local/sortformer"
        model = FakeVoiceDiarizationModel(
            result=FakeDiarizationResult(
                segments=(FakeDiarizationSegment(start_seconds=0.0, end_seconds=0.3, confidence=0.81),)
            ),
            expected_audio=_wav_audio(input_data.audio, sample_rate_hz=input_data.sample_rate_hz),
        )
        models.append(model)
        return model

    diarizer = VoiceDiarizer(model_id="local/sortformer", load_model=load_model, clip_audio=_clip_stub)

    output = asyncio.run(await_voice_diarization(diarizer.classify(input_data)))

    assert output.segments[0].audio
    assert output.segments[0].start_seconds == 0.0
    assert output.segments[0].end_seconds == 0.3
    assert output.segments[0].confidence == 0.81
    assert len(models) == 1


def test_voice_diarizer_is_awaitable() -> None:
    input_data = _input()
    diarizer = VoiceDiarizer(
        model=FakeVoiceDiarizationModel(
            result=FakeDiarizationResult(
                segments=(FakeDiarizationSegment(start_seconds=0.0, end_seconds=0.3, confidence=0.81),)
            ),
            expected_audio=_wav_audio(input_data.audio, sample_rate_hz=input_data.sample_rate_hz),
        ),
        clip_audio=_clip_stub,
    )

    output = diarizer.classify(input_data)

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output).segments[0].audio


def test_voice_diarizer_wraps_provider_errors() -> None:
    diarizer = VoiceDiarizer(model=FailingVoiceDiarizationModel(), clip_audio=_clip_stub)

    with pytest.raises(VoiceDiarizationError) as error:
        _ = asyncio.run(await_voice_diarization(diarizer.classify(_input())))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_voice_diarizer_stages_raw_pcm_as_wav_and_clips_on_frame_boundaries() -> None:
    input_data = _input(audio=b"\x01\x00\x02\x00\x03\x00\x04\x00\x05\x00", sample_rate_hz=10)
    diarizer = VoiceDiarizer(
        model=FakeVoiceDiarizationModel(
            result=FakeDiarizationResult(
                segments=(FakeDiarizationSegment(start_seconds=0.11, end_seconds=0.39, confidence=0.81),)
            ),
            expected_audio=_wav_audio(input_data.audio, sample_rate_hz=input_data.sample_rate_hz),
        )
    )

    output = asyncio.run(await_voice_diarization(diarizer.classify(input_data)))

    assert output.segments[0] == voice.VoiceDiarizationSegment(
        audio=b"\x02\x00\x03\x00\x04\x00",
        media_type="audio/pcm",
        sample_rate_hz=10,
        channels=1,
        start_seconds=0.11,
        end_seconds=0.39,
        confidence=0.81,
    )


def test_voice_diarizer_rejects_out_of_range_span() -> None:
    input_data = _input(audio=b"\x00\x00" * 10, sample_rate_hz=10)
    diarizer = VoiceDiarizer(
        model=FakeVoiceDiarizationModel(
            result=FakeDiarizationResult(
                segments=(FakeDiarizationSegment(start_seconds=0.5, end_seconds=1.1, confidence=0.81),)
            ),
            expected_audio=_wav_audio(input_data.audio, sample_rate_hz=input_data.sample_rate_hz),
        )
    )

    with pytest.raises(VoiceDiarizationError) as error:
        _ = asyncio.run(await_voice_diarization(diarizer.classify(input_data)))

    assert isinstance(error.value.__cause__, ValueError)
    assert "outside the source audio duration" in str(error.value.__cause__)


def _input(
    *,
    audio: bytes = b"\x00\x00" * 20,
    sample_rate_hz: int = 10,
) -> voice.diarization.InputData:
    return voice.diarization.InputData(
        audio=audio,
        media_type="audio/pcm",
        sample_rate_hz=sample_rate_hz,
        channels=1,
    )


def _wav_audio(pcm_audio: bytes, *, sample_rate_hz: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm_audio)
    return output.getvalue()


def _clip_stub(input: voice.diarization.InputData, start_seconds: float, end_seconds: float) -> bytes:
    frame_count = round((end_seconds - start_seconds) * input.sample_rate_hz)
    return b"\x00" * (2 * input.channels * max(1, frame_count))
