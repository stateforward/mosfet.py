"""Unit tests for ``speech_server.request``: one raw-PCM POST, its JSON answer, failure kinds and the duration metric."""

from __future__ import annotations

import asyncio
import collections.abc
import json
import pathlib
import typing

import httpx
import pytest
from opentelemetry import _logs, trace

import mosfet.telemetry
from mosfet.providers import speech_server
from mosfet.providers.speech_server import request

_DURATION = "bot.provider.speech_server.request.duration"

Handler = collections.abc.Callable[[httpx.Request], httpx.Response]


def _post(handle: Handler, *, base_url: str = "http://speech.test", content: bytes = b"\x01\x00") -> dict[str, object]:
    async def run() -> dict[str, object]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await request.post(
                base_url=base_url,
                path="/v1/embed",
                operation="embed",
                content=content,
                params={"sample_rate": 16_000},
                client=client,
                timeout_seconds=3.0,
            )

    return asyncio.run(run())


def _failure(handle: Handler) -> speech_server.RequestError:
    with pytest.raises(speech_server.RequestError) as caught:
        _ = _post(handle)
    return caught.value


def test_raw_pcm_is_posted_and_the_json_object_returned() -> None:
    requests: list[httpx.Request] = []

    def handle(sent: httpx.Request) -> httpx.Response:
        requests.append(sent)
        return httpx.Response(200, json={"ok": True})

    assert _post(handle, base_url="http://speech.test/", content=b"\x01\x00\x02\x00") == {"ok": True}
    [sent] = requests
    assert sent.method == "POST"
    assert str(sent.url) == "http://speech.test/v1/embed?sample_rate=16000"
    assert sent.headers["content-type"] == "application/octet-stream"
    assert sent.content == b"\x01\x00\x02\x00"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "invalid_input"),
        (413, "invalid_input"),
        (422, "invalid_input"),
        (500, "unavailable"),
        (503, "unavailable"),
    ],
)
def test_http_status_failures_map_to_kinds(status: int, kind: str) -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(status, json={"detail": "audio too short"})

    failure = _failure(handle)

    assert failure.kind == kind
    assert failure.failure_kind == kind
    assert "audio too short" not in str(failure)


def test_an_unreachable_server_is_unavailable() -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=sent)

    assert _failure(handle).kind == "unavailable"


def test_a_server_that_does_not_answer_in_time_is_a_timeout() -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=sent)

    assert _failure(handle).kind == "timeout"


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]"])
def test_an_answer_that_is_not_a_json_object_is_an_invalid_response(body: bytes) -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    assert _failure(handle).kind == "invalid_response"


def test_an_oversized_answer_is_an_invalid_response() -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(200, content=b'{"pad": "' + b"x" * request.MAX_RESPONSE_BYTES + b'"}')

    assert _failure(handle).kind == "invalid_response"


def test_a_redirect_is_an_invalid_response() -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(307, headers={"location": "http://elsewhere.test/"})

    assert _failure(handle).kind == "invalid_response"


def _histogram_counts(path: pathlib.Path) -> dict[tuple[tuple[str, object], ...], int]:
    provider = mosfet.telemetry.meter_provider()
    assert provider is not None
    _ = provider.force_flush()
    latest: list[dict[str, typing.Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        payload = typing.cast(dict[str, typing.Any], json.loads(line))
        if payload["name"] == _DURATION:
            latest = typing.cast(list[dict[str, typing.Any]], payload["data"]["data_points"])
    return {
        tuple(sorted(typing.cast(dict[str, object], point["attributes"]).items())): typing.cast(int, point["count"])
        for point in latest
    }


def _install_nothing(provider: object) -> None:
    del provider


def test_duration_metric_carries_operation_and_outcome_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("BOT_OTEL_DISABLED", "BOT_OTEL_LOG_FILE", "BOT_OTEL_SPAN_FILE", "BOT_OTEL_METRIC_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", _install_nothing)
    monkeypatch.setattr(trace, "set_tracer_provider", _install_nothing)
    mosfet.telemetry.reset()
    assert mosfet.telemetry.configure(metric_file="metrics.jsonl") is True

    def ok(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(200, json={})

    def unavailable(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(503, json={})

    try:
        path = tmp_path / "metrics.jsonl"
        before = _histogram_counts(path)
        _ = _post(ok)
        _ = _failure(unavailable)
        after = _histogram_counts(path)
    finally:
        mosfet.telemetry.reset()

    succeeded = (("bot.outcome", "ok"), ("operation", "embed"))
    failed = (("bot.outcome", "failed"), ("operation", "embed"))
    assert after[succeeded] - before.get(succeeded, 0) == 1
    assert after[failed] - before.get(failed, 0) == 1
    assert {key for point in after for key, _ in point} == {"bot.outcome", "operation"}
