"""Unit tests for ``speech_server.VoiceEmbedder``: one ``/v1/embed`` request per voice segment."""

from __future__ import annotations

import array
import asyncio
import collections.abc
import json
import sys

import hsm
import httpx
import pytest

from mosfet.abilities import classifying
from mosfet.abilities.hearing import voice
from mosfet.providers import speech_server

Handler = collections.abc.Callable[[httpx.Request], httpx.Response]
_VECTOR = [0.6, 0.8]


def _segment(*, start: float, end: float, rate: int = 16_000, channels: int = 1) -> voice.VoiceSegment:
    frames = round((end - start) * rate)
    values = array.array("h", (index % 500 for index in range(frames * channels)))
    if sys.byteorder == "big":
        values.byteswap()
    return voice.VoiceSegment(
        audio=values.tobytes(),
        media_type="audio/pcm",
        sample_rate_hz=rate,
        channels=channels,
        start_seconds=start,
        end_seconds=end,
    )


def _input(*segments: voice.VoiceSegment) -> voice.identification.InputData:
    return voice.identification.InputData(segments=segments)


def _answer(requests: list[httpx.Request], *vectors: list[float]) -> Handler:
    def handle(sent: httpx.Request) -> httpx.Response:
        index = len(requests)
        requests.append(sent)
        vector = vectors[index] if vectors else _VECTOR
        return httpx.Response(200, json={"embedding": vector, "model": "wespeaker-resnet34", "seconds": 1.0, "ms": 9})

    return handle


def _classify(handle: Handler, data: voice.identification.InputData) -> voice.identification.OutputData:
    async def run() -> voice.identification.OutputData:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            embedder = speech_server.VoiceEmbedder(base_url="http://embed.test", client=client)
            return await embedder.classify(data)

    return asyncio.run(run())


def _failure(handle: Handler, data: voice.identification.InputData) -> speech_server.RequestError:
    with pytest.raises(speech_server.RequestError) as caught:
        _ = _classify(handle, data)
    return caught.value


def test_voice_embedder_is_a_core_voice_classifier() -> None:
    embedder = speech_server.VoiceEmbedder(base_url="http://embed.test")

    assert isinstance(embedder, classifying.Classifier)
    assert not isinstance(embedder, hsm.Instance)


def test_each_segment_is_posted_and_embedded_in_input_order() -> None:
    requests: list[httpx.Request] = []
    first, second = _segment(start=0.0, end=0.5), _segment(start=0.5, end=1.5)

    output = _classify(_answer(requests, [1.0, 0.0], [0.0, 1.0]), _input(first, second))

    assert [str(sent.url) for sent in requests] == ["http://embed.test/v1/embed"] * 2
    assert [sent.content for sent in requests] == [first.audio, second.audio]
    assert output == voice.identification.OutputData(
        embeddings=(
            voice.VoiceEmbedding(embedding=(1.0, 0.0), model="wespeaker-resnet34"),
            voice.VoiceEmbedding(embedding=(0.0, 1.0), model="wespeaker-resnet34"),
        )
    )


def test_segments_are_resampled_to_sixteen_kilohertz_mono() -> None:
    requests: list[httpx.Request] = []

    _ = _classify(_answer(requests), _input(_segment(start=0.0, end=0.5, rate=48_000, channels=2)))

    assert len(requests[0].content) == 8_000 * 2


def test_a_segment_shorter_than_the_server_floor_is_refused_without_a_request() -> None:
    requests: list[httpx.Request] = []
    data = _input(_segment(start=0.0, end=0.5), _segment(start=0.5, end=0.75))

    failure = _failure(_answer(requests), data)

    assert failure.kind == "invalid_input"
    assert requests == []


@pytest.mark.parametrize(
    "body",
    [
        {"embedding": [], "model": "m"},
        {"embedding": [0.1, "x"], "model": "m"},
        {"embedding": [0.1, 0.2]},
        {"embedding": [0.1, 0.2], "model": ""},
        {"model": "m"},
    ],
)
def test_an_answer_that_is_not_an_embedding_is_an_invalid_response(body: object) -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(200, content=json.dumps(body).replace('"x"', "NaN"))

    failure = _failure(handle, _input(_segment(start=0.0, end=0.5)))

    assert failure.kind == "invalid_response"
    assert "0.1" not in str(failure)


@pytest.mark.parametrize(("status", "kind"), [(400, "invalid_input"), (422, "invalid_input"), (503, "unavailable")])
def test_server_refusals_map_to_kinds(status: int, kind: str) -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(status, json={"detail": "no"})

    assert _failure(handle, _input(_segment(start=0.0, end=0.5))).kind == kind
