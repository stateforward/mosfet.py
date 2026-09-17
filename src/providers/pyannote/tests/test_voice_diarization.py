from __future__ import annotations

from mosfet.abilities.hearing import voice
from mosfet.abilities.hearing.voice import diarization
from mosfet.providers.pyannote import VoiceDiarizationError, VoiceDiarizer

import asyncio
import collections.abc
import dataclasses
import pathlib
import wave

import pytest


@dataclasses.dataclass(frozen=True)
class FakeTurn:
    start: float
    end: float
    score: float | None = None


@dataclasses.dataclass(frozen=True)
class FakeAnnotation:
    entries: tuple[tuple[FakeTurn, int, str], ...]

    def itertracks(self, *, yield_label: bool) -> tuple[tuple[FakeTurn, int, str], ...]:
        assert yield_label is True
        return self.entries


@dataclasses.dataclass(frozen=True)
class FakeResult:
    speaker_diarization: FakeAnnotation


@dataclasses.dataclass
class FakePipeline:
    result: FakeResult
    expected_audio: bytes
    expected_sample_rate_hz: int = 10
    expected_channels: int = 1
    calls: list[pathlib.Path] = dataclasses.field(default_factory=list)

    def __call__(self, audio: pathlib.Path) -> FakeResult:
        with wave.open(str(audio), "rb") as staged_audio:
            assert staged_audio.getframerate() == self.expected_sample_rate_hz
            assert staged_audio.getnchannels() == self.expected_channels
            assert staged_audio.getsampwidth() == 2
            assert staged_audio.readframes(staged_audio.getnframes()) == self.expected_audio
        self.calls.append(audio)
        return self.result


class FailingPipeline:
    def __call__(self, audio: pathlib.Path) -> object:
        del audio
        raise RuntimeError("pyannote unavailable")


async def _await_output(
    output: collections.abc.Awaitable[diarization.OutputData],
) -> diarization.OutputData:
    return await output


def test_diarizer_stages_pcm_as_wav_and_preserves_segment_metadata() -> None:
    raw_audio = _pcm_audio(frame_count=20)
    pipeline = FakePipeline(
        result=FakeResult(
            speaker_diarization=FakeAnnotation(
                entries=(
                    (FakeTurn(start=1.0, end=1.5, score=0.82), 0, "SPEAKER_42"),
                    (FakeTurn(start=0.0, end=0.7, score=0.91), 0, "SPEAKER_07"),
                )
            )
        ),
        expected_audio=raw_audio,
    )
    input_data = _input(audio=raw_audio)
    clipped_audio: list[tuple[voice.diarization.InputData, float, float]] = []

    def clip_audio(
        audio: voice.diarization.InputData,
        start_seconds: float,
        end_seconds: float,
    ) -> bytes:
        clipped_audio.append((audio, start_seconds, end_seconds))
        frame_count = round((end_seconds - start_seconds) * audio.sample_rate_hz)
        return b"\x00" * (2 * audio.channels * max(1, frame_count))

    diarizer = VoiceDiarizer(pipeline=pipeline, clip_audio=clip_audio)

    output = asyncio.run(_await_output(diarizer.classify(input_data)))

    assert output == voice.diarization.OutputData(
        segments=(
            voice.VoiceDiarizationSegment(
                audio=b"\x00\x00" * 7,
                media_type="audio/pcm",
                sample_rate_hz=10,
                channels=1,
                start_seconds=0.0,
                end_seconds=0.7,
                confidence=0.91,
            ),
            voice.VoiceDiarizationSegment(
                audio=b"\x00\x00" * 5,
                media_type="audio/pcm",
                sample_rate_hz=10,
                channels=1,
                start_seconds=1.0,
                end_seconds=1.5,
                confidence=0.82,
            ),
        )
    )
    dumped = repr(output)
    assert "SPEAKER_07" not in dumped
    assert "SPEAKER_42" not in dumped
    assert clipped_audio == [(input_data, 0.0, 0.7), (input_data, 1.0, 1.5)]
    assert not pipeline.calls[0].exists()
    assert isinstance(diarizer, voice.VoiceDiarizer)


def test_diarizer_uses_injected_loader() -> None:
    pipelines: list[FakePipeline] = []

    def load_pipeline(model_id: str) -> FakePipeline:
        assert model_id == "local/community-1"
        pipeline = FakePipeline(
            result=FakeResult(
                speaker_diarization=FakeAnnotation(entries=((FakeTurn(start=0.0, end=0.3), 0, "remote"),))
            ),
            expected_audio=_pcm_audio(frame_count=20),
        )
        pipelines.append(pipeline)
        return pipeline

    diarizer = VoiceDiarizer(model_id="local/community-1", load_pipeline=load_pipeline, clip_audio=_clip_stub)

    output = asyncio.run(_await_output(diarizer.classify(_input(audio=_pcm_audio(frame_count=20)))))

    assert output.segments[0].audio == b"\x00\x00" * 3
    assert output.segments[0].start_seconds == 0.0
    assert output.segments[0].end_seconds == 0.3
    assert len(pipelines) == 1


def test_diarizer_is_awaitable() -> None:
    diarizer = VoiceDiarizer(
        pipeline=FakePipeline(
            result=FakeResult(
                speaker_diarization=FakeAnnotation(entries=((FakeTurn(start=0.0, end=0.3), 0, "speaker"),))
            ),
            expected_audio=_pcm_audio(frame_count=20),
        ),
        clip_audio=_clip_stub,
    )

    output = diarizer.classify(_input(audio=_pcm_audio(frame_count=20)))

    assert isinstance(output, collections.abc.Coroutine)
    assert asyncio.run(output).segments[0].audio == b"\x00\x00" * 3


def test_diarizer_wraps_provider_errors() -> None:
    diarizer = VoiceDiarizer(pipeline=FailingPipeline())

    with pytest.raises(VoiceDiarizationError) as error:
        _ = asyncio.run(_await_output(diarizer.classify(_input(audio=_pcm_audio(frame_count=20)))))

    assert isinstance(error.value.__cause__, RuntimeError)


def test_diarizer_rejects_out_of_range_span() -> None:
    diarizer = VoiceDiarizer(
        pipeline=FakePipeline(
            result=FakeResult(
                speaker_diarization=FakeAnnotation(entries=((FakeTurn(start=0.0, end=0.31, score=0.81), 0, "speaker"),))
            ),
            expected_audio=_pcm_audio(frame_count=3),
        )
    )

    with pytest.raises(VoiceDiarizationError) as error:
        _ = asyncio.run(_await_output(diarizer.classify(_input(audio=_pcm_audio(frame_count=3)))))

    assert isinstance(error.value.__cause__, ValueError)


def _input(*, audio: bytes) -> voice.diarization.InputData:
    return voice.diarization.InputData(
        audio=audio,
        media_type="audio/pcm",
        sample_rate_hz=10,
        channels=1,
    )


def _pcm_audio(*, frame_count: int) -> bytes:
    return b"\x01\x00" * frame_count


def _clip_stub(
    audio: voice.diarization.InputData,
    start_seconds: float,
    end_seconds: float,
) -> bytes:
    frame_count = round((end_seconds - start_seconds) * audio.sample_rate_hz)
    return b"\x00" * (2 * audio.channels * max(1, frame_count))
