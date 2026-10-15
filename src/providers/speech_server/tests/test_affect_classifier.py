"""Unit tests for ``speech_server.AffectClassifier``: one ``/v1/emotion`` request per voice segment."""

from __future__ import annotations

import array
import asyncio
import collections.abc
import json
import pathlib
import sys
import typing

import hsm
import httpx
import pytest
from opentelemetry import _logs, trace

import mosfet.telemetry
from mosfet.abilities.hearing import voice
from mosfet.providers import speech_server

Handler = collections.abc.Callable[[httpx.Request], httpx.Response]
_ANSWER = {
    "label": "happy",
    "scores": {"happy": 0.7, "neutral": 0.2, "sad": 0.1},
    "valence": 0.6,
    "confidence": 0.7,
    "snr_db": 21.5,
    "model": "emotion2vec/emotion2vec_plus_large",
    "seconds": 0.5,
    "latency_ms": 14.2,
}


def _segment(*, seconds: float = 0.5, rate: int = 16_000, channels: int = 1) -> voice.VoiceSegment:
    frames = round(seconds * rate)
    values = array.array("h", (index % 500 for index in range(frames * channels)))
    if sys.byteorder == "big":
        values.byteswap()
    return voice.VoiceSegment(
        audio=values.tobytes(),
        media_type="audio/pcm",
        sample_rate_hz=rate,
        channels=channels,
        start_seconds=1.0,
        end_seconds=1.0 + seconds,
    )


def _answer(requests: list[httpx.Request], body: object = _ANSWER) -> Handler:
    def handle(sent: httpx.Request) -> httpx.Response:
        requests.append(sent)
        return httpx.Response(200, content=json.dumps(body))

    return handle


def _classify(handle: Handler, segment: voice.VoiceSegment) -> voice.Affect:
    async def run() -> voice.Affect:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            classifier = speech_server.AffectClassifier(base_url="http://emotion.test/", client=client)
            return await classifier.classify(segment)

    return asyncio.run(run())


def _failure(handle: Handler, segment: voice.VoiceSegment) -> speech_server.RequestError:
    with pytest.raises(speech_server.RequestError) as caught:
        _ = _classify(handle, segment)
    return caught.value


def test_affect_classifier_is_a_core_affect_classifier() -> None:
    classifier = speech_server.AffectClassifier(base_url="http://emotion.test")

    assert isinstance(classifier, voice.AffectClassifier)
    assert not isinstance(classifier, hsm.Instance)


def test_a_segment_is_posted_raw_and_its_answer_is_its_affect() -> None:
    requests: list[httpx.Request] = []
    segment = _segment()

    affect = _classify(_answer(requests), segment)

    (sent,) = requests
    assert sent.method == "POST"
    assert str(sent.url) == "http://emotion.test/v1/emotion?sample_rate=16000"
    assert sent.headers["content-type"] == "application/octet-stream"
    assert sent.content == segment.audio
    assert affect == voice.Affect(
        label="happy", scores={"happy": 0.7, "neutral": 0.2, "sad": 0.1}, valence=0.6, confidence=0.7
    )


@pytest.mark.parametrize("rate", [24_000, 48_000])
def test_mono_at_a_rate_the_server_takes_passes_through(rate: int) -> None:
    requests: list[httpx.Request] = []
    segment = _segment(rate=rate)

    _ = _classify(_answer(requests), segment)

    (sent,) = requests
    assert sent.url.params["sample_rate"] == str(rate)
    assert sent.content == segment.audio


def test_stereo_is_mixed_to_mono_at_its_own_rate() -> None:
    requests: list[httpx.Request] = []

    _ = _classify(_answer(requests), _segment(rate=48_000, channels=2))

    (sent,) = requests
    assert sent.url.params["sample_rate"] == "48000"
    assert len(sent.content) == 24_000 * 2


def test_another_rate_is_resampled_to_sixteen_kilohertz() -> None:
    requests: list[httpx.Request] = []

    _ = _classify(_answer(requests), _segment(rate=44_100))

    (sent,) = requests
    assert sent.url.params["sample_rate"] == "16000"
    assert len(sent.content) == 8_000 * 2


def test_a_segment_shorter_than_the_server_floor_is_refused_without_a_request() -> None:
    requests: list[httpx.Request] = []

    failure = _failure(_answer(requests), _segment(seconds=0.25))

    assert failure.kind == "invalid_input"
    assert requests == []


@pytest.mark.parametrize(
    "body",
    [
        {"scores": {"happy": 1.0}},
        {"label": ""},
        {"label": "happy", "scores": {"happy": 2.0}},
        {"label": "happy", "valence": 3.0},
        {"label": "happy", "confidence": "high"},
        ["happy"],
    ],
    ids=("no-label", "empty-label", "score-out-of-range", "valence-out-of-range", "confidence-not-a-number", "list"),
)
def test_an_answer_that_is_not_an_affect_is_an_invalid_response(body: object) -> None:
    failure = _failure(_answer([], body), _segment())

    assert failure.kind == "invalid_response"
    assert "happy" not in str(failure)


@pytest.mark.parametrize(("status", "kind"), [(400, "invalid_input"), (422, "invalid_input"), (503, "unavailable")])
def test_server_refusals_map_to_kinds(status: int, kind: str) -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(status, json={"detail": "no"})

    assert _failure(handle, _segment()).kind == kind


def _install_nothing(provider: object) -> None:
    del provider


def test_each_request_is_timed_as_the_emotion_operation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("BOT_OTEL_DISABLED", "BOT_OTEL_LOG_FILE", "BOT_OTEL_SPAN_FILE", "BOT_OTEL_METRIC_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", _install_nothing)
    monkeypatch.setattr(trace, "set_tracer_provider", _install_nothing)
    mosfet.telemetry.reset()
    assert mosfet.telemetry.configure(metric_file="metrics.jsonl") is True
    try:
        _ = _classify(_answer([]), _segment())
        provider = mosfet.telemetry.meter_provider()
        assert provider is not None
        _ = provider.force_flush()
    finally:
        mosfet.telemetry.reset()

    points: list[dict[str, typing.Any]] = []
    for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        payload = typing.cast(dict[str, typing.Any], json.loads(line))
        if payload["name"] == "bot.provider.speech_server.request.duration":
            points = typing.cast(list[dict[str, typing.Any]], payload["data"]["data_points"])
    assert {"bot.outcome": "ok", "operation": "emotion"} in [point["attributes"] for point in points]
