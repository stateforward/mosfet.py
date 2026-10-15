"""Unit tests for ``speech_server.Diarizer``: the ``/v1/diarize`` wire mapped to core diarization segments."""

from __future__ import annotations

import array
import asyncio
import collections.abc
import sys

import hsm
import httpx
import pytest

from mosfet.abilities.hearing import voice
from mosfet.providers import speech_server

Handler = collections.abc.Callable[[httpx.Request], httpx.Response]


def _pcm(frames: int, *, channels: int = 1) -> bytes:
    values = array.array("h", (index % 1000 for index in range(frames * channels)))
    if sys.byteorder == "big":
        values.byteswap()
    return values.tobytes()


def _input(*, seconds: float = 2.0, rate: int = 16_000, channels: int = 1) -> voice.diarization.InputData:
    return voice.diarization.InputData(
        audio=_pcm(round(seconds * rate), channels=channels),
        media_type="audio/pcm",
        sample_rate_hz=rate,
        channels=channels,
    )


def _answer(requests: list[httpx.Request], body: object) -> Handler:
    def handle(sent: httpx.Request) -> httpx.Response:
        requests.append(sent)
        return httpx.Response(200, json=body)

    return handle


def _classify(handle: Handler, data: voice.diarization.InputData) -> voice.diarization.OutputData:
    async def run() -> voice.diarization.OutputData:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            diarizer = speech_server.Diarizer(base_url="http://diarize.test", client=client)
            return await diarizer.classify(data)

    return asyncio.run(run())


def _failure(handle: Handler, data: voice.diarization.InputData) -> speech_server.RequestError:
    with pytest.raises(speech_server.RequestError) as caught:
        _ = _classify(handle, data)
    return caught.value


_TWO_SPEAKERS = {
    "segments": [
        {"start": 1.0, "end": 1.75, "speaker": 1},
        {"start": 0.25, "end": 1.0, "speaker": 0},
    ],
    "speakers": 2,
    "seconds": 2.0,
    "ms": 41.0,
}


def test_diarizer_is_a_core_voice_diarizer() -> None:
    diarizer = speech_server.Diarizer(base_url="http://diarize.test")

    assert isinstance(diarizer, voice.VoiceDiarizer)
    assert not isinstance(diarizer, hsm.Instance)


def test_sixteen_kilohertz_mono_is_posted_raw_with_its_rate() -> None:
    requests: list[httpx.Request] = []
    data = _input()

    _ = _classify(_answer(requests, _TWO_SPEAKERS), data)

    [sent] = requests
    assert sent.method == "POST"
    assert str(sent.url) == "http://diarize.test/v1/diarize?sample_rate=16000"
    assert sent.content == data.audio


def test_other_audio_is_resampled_to_sixteen_kilohertz_mono_before_posting() -> None:
    requests: list[httpx.Request] = []

    _ = _classify(_answer(requests, _TWO_SPEAKERS), _input(rate=48_000, channels=2))

    assert len(requests[0].content) == 2 * 16_000 * 2


def test_server_segments_become_ordered_clips_of_the_original_audio_without_speaker_labels() -> None:
    data = _input(rate=48_000, channels=2)

    output = _classify(_answer([], _TWO_SPEAKERS), data)

    first, second = output.segments
    frame = 2 * 2
    assert (first.start_seconds, first.end_seconds) == (0.25, 1.0)
    assert (second.start_seconds, second.end_seconds) == (1.0, 1.75)
    assert first.audio == data.audio[12_000 * frame : 48_000 * frame]
    assert second.audio == data.audio[48_000 * frame : 84_000 * frame]
    assert all((segment.sample_rate_hz, segment.channels) == (48_000, 2) for segment in output.segments)
    assert all(segment.confidence is None for segment in output.segments)
    assert "speaker" not in output.model_dump_json()


def test_a_segment_past_the_audio_is_clipped_to_its_end() -> None:
    body = {"segments": [{"start": 1.5, "end": 2.4, "speaker": 0}, {"start": 2.1, "end": 2.5, "speaker": 1}]}

    [segment] = _classify(_answer([], body), _input()).segments

    assert (segment.start_seconds, segment.end_seconds) == (1.5, 2.0)


@pytest.mark.parametrize(
    "body",
    [
        {"segments": []},
        {"segments": [{"start": 0.5, "end": 0.5, "speaker": 0}]},
        {"segments": [{"start": -1.0, "end": 0.5, "speaker": 0}]},
        {"segments": [{"start": "soon", "end": 0.5, "speaker": 0}]},
        {"speakers": 1},
    ],
)
def test_an_answer_without_usable_segments_is_an_invalid_response(body: object) -> None:
    assert _failure(_answer([], body), _input()).kind == "invalid_response"


@pytest.mark.parametrize(("status", "kind"), [(400, "invalid_input"), (422, "invalid_input"), (503, "unavailable")])
def test_server_refusals_map_to_kinds(status: int, kind: str) -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(status, json={"detail": "no"})

    assert _failure(handle, _input()).kind == kind


def test_audio_that_is_not_whole_frames_is_refused_without_a_request() -> None:
    requests: list[httpx.Request] = []
    data = voice.diarization.InputData(audio=b"\x01\x00\x02", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)

    assert _failure(_answer(requests, _TWO_SPEAKERS), data).kind == "invalid_input"
    assert requests == []
