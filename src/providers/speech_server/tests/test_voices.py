"""Unit tests for ``speech_server.Catalog`` and ``speech_server.Designer``: the lab's ``/v1/voices`` protocol."""

from __future__ import annotations

import asyncio
import collections.abc
import json
import pathlib
import typing

import hsm
import httpx
import pytest
from opentelemetry import _logs, trace

import mosfet.telemetry
from mosfet.abilities.vocal import voice
from mosfet.providers import speech_server

Handler = collections.abc.Callable[[httpx.Request], httpx.Response]
_DESCRIPTION = "A bright, quick young voice with a light rasp and an easy laugh in it."
_LISTING = {
    "default": "warm",
    "voices": [
        {"id": "warm", "description": "A warm, unhurried adult voice."},
        {"id": "buddy", "description": ""},
        {"id": "v-0123456789ab", "description": _DESCRIPTION, "custom": True},
    ],
    "emotive_tags": ["laugh", "giggle", "chuckle", "gasp", "sigh", "whisper"],
}
_PREVIEW = voice.Preview.model_validate({"url": "http://lab.test:8002/v1/voices/v-0123456789ab/preview"})
_MADE = {
    "voices": [
        {"id": "v-0123456789ab", "description": _DESCRIPTION, "preview_url": "/v1/voices/v-0123456789ab/preview"}
    ]
}


def _answer(requests: list[httpx.Request], body: object, *, status: int = 200) -> Handler:
    def handle(sent: httpx.Request) -> httpx.Response:
        requests.append(sent)
        return httpx.Response(status, content=json.dumps(body))

    return handle


def _status(status: int) -> Handler:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(status, json={"detail": "the engine said something private"})

    return handle


def _raise(error: type[httpx.TransportError]) -> Handler:
    def handle(sent: httpx.Request) -> httpx.Response:
        raise error("the socket said something private", request=sent)

    return handle


def _list(handle: Handler, *, languages: tuple[str, ...] = ("en",)) -> tuple[voice.Entry, ...]:
    async def run() -> tuple[voice.Entry, ...]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            catalog = speech_server.Catalog("http://lab.test:8002/", source="lab", languages=languages, client=client)
            return await catalog.voices(languages=("fr",), descriptors=voice.Descriptors(gender="feminine"))

    return asyncio.run(run())


def _design(handle: Handler, description: str = _DESCRIPTION) -> voice.Designed:
    async def run() -> voice.Designed:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            designer = speech_server.Designer("http://lab.test:8002", source="lab", client=client)
            return await designer.create(voice.DesignRequest(description=description, languages=("en",)))

    return asyncio.run(run())


def _list_failure(handle: Handler) -> voice.Error:
    with pytest.raises(voice.Error) as caught:
        _ = _list(handle)
    assert "private" not in str(caught.value)
    return caught.value


def _design_failure(handle: Handler, description: str = _DESCRIPTION) -> voice.Error:
    with pytest.raises(voice.Error) as caught:
        _ = _design(handle, description)
    assert "private" not in str(caught.value)
    return caught.value


def test_catalog_and_designer_are_core_voice_collaborators_of_one_source() -> None:
    catalog = speech_server.Catalog("http://lab.test", source="lab")
    designer = speech_server.Designer("http://lab.test", source="lab")

    assert isinstance(catalog, voice.Catalog)
    assert isinstance(designer, voice.Designer)
    assert not isinstance(catalog, hsm.Instance)
    assert not isinstance(designer, hsm.Instance)
    assert (catalog.source, designer.source) == ("lab", "lab")
    assert designer.max_description == 300


@pytest.mark.parametrize(
    "build",
    [
        lambda: speech_server.Catalog("http://lab.test", source="Lab TTS"),
        lambda: speech_server.Catalog("http://lab.test", source="lab", languages=("English",)),
        lambda: speech_server.Designer("http://lab.test", source=""),
    ],
    ids=("catalog-source", "catalog-language", "designer-source"),
)
def test_construction_refuses_a_source_or_language_core_cannot_carry(
    build: collections.abc.Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        _ = build()


def test_the_listing_maps_built_in_and_designed_voices_to_entries() -> None:
    requests: list[httpx.Request] = []

    entries = _list(_answer(requests, _LISTING))

    (sent,) = requests
    assert sent.method == "GET"
    assert str(sent.url) == "http://lab.test:8002/v1/voices"
    assert entries == (
        voice.Entry(
            ref=voice.Ref(source="lab", key="warm"),
            name="warm",
            description="A warm, unhurried adult voice.",
            languages=("en",),
        ),
        voice.Entry(ref=voice.Ref(source="lab", key="buddy"), name="buddy", languages=("en",)),
        voice.Entry(
            ref=voice.Ref(source="lab", key="v-0123456789ab"),
            name="v-0123456789ab",
            description=_DESCRIPTION,
            languages=("en",),
            preview=_PREVIEW,
            labels={"custom": "true"},
        ),
    )


def test_the_listing_filters_are_hints_the_server_is_not_sent() -> None:
    requests: list[httpx.Request] = []

    _ = _list(_answer(requests, _LISTING))

    (sent,) = requests
    assert sent.url.params == httpx.QueryParams()


def test_the_listing_languages_are_the_ones_composition_declared() -> None:
    entries = _list(_answer([], _LISTING), languages=())

    assert {entry.languages for entry in entries} == {()}


@pytest.mark.parametrize(
    "body",
    [
        {"default": "warm"},
        {"voices": [{"description": "no id"}]},
        {"voices": [{"id": ""}]},
        {"voices": [{"id": "warm", "description": "x" * 1001}]},
        {"voices": "warm"},
        ["warm"],
    ],
    ids=("no-voices", "no-id", "empty-id", "description-too-long", "voices-not-a-list", "list"),
)
def test_a_listing_that_is_not_a_voice_list_is_an_invalid_response(body: object) -> None:
    assert _list_failure(_answer([], body)).kind == "invalid_response"


def test_a_listing_that_is_not_json_is_an_invalid_response() -> None:
    def handle(sent: httpx.Request) -> httpx.Response:
        del sent
        return httpx.Response(200, content=b"private, not json")

    assert _list_failure(handle).kind == "invalid_response"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (302, "invalid_response"),
        (400, "invalid_input"),
        (404, "unsupported_input"),
        (422, "invalid_input"),
        (500, "unavailable"),
        (503, "unavailable"),
        (507, "overloaded"),
    ],
)
def test_listing_statuses_map_to_kinds(status: int, kind: str) -> None:
    assert _list_failure(_status(status)).kind == kind


@pytest.mark.parametrize(
    ("error", "kind"), [(httpx.ConnectError, "unavailable"), (httpx.ReadTimeout, "timeout")], ids=("connect", "timeout")
)
def test_listing_transport_failures_map_to_kinds(error: type[httpx.TransportError], kind: str) -> None:
    assert _list_failure(_raise(error)).kind == kind


def test_a_design_asks_for_one_voice_from_the_description_and_keeps_it() -> None:
    requests: list[httpx.Request] = []

    designed = _design(_answer(requests, _MADE, status=201))

    (sent,) = requests
    assert sent.method == "POST"
    assert str(sent.url) == "http://lab.test:8002/v1/voices"
    assert sent.headers["content-type"] == "application/json"
    assert json.loads(sent.content) == {"description": _DESCRIPTION, "count": 1}
    assert designed == voice.Designed(
        ref=voice.Ref(source="lab", key="v-0123456789ab"),
        description=_DESCRIPTION,
        preview=_PREVIEW,
    )


def test_a_description_longer_than_the_lab_takes_is_refused_before_any_request() -> None:
    requests: list[httpx.Request] = []

    failure = _design_failure(_answer(requests, _MADE, status=201), "x" * 301)

    assert failure.kind == "invalid_input"
    assert requests == []


@pytest.mark.parametrize(
    "body",
    [
        {"voices": []},
        {"voices": [_MADE["voices"][0], _MADE["voices"][0]]},
        {"voices": [{"description": _DESCRIPTION}]},
        {"voice": "v-0123456789ab"},
    ],
    ids=("none", "two", "no-id", "no-voices"),
)
def test_a_design_answer_that_is_not_one_voice_is_an_invalid_response(body: object) -> None:
    assert _design_failure(_answer([], body, status=201)).kind == "invalid_response"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "design_failed"),
        (404, "unsupported_input"),
        (422, "invalid_input"),
        (500, "design_failed"),
        (503, "unavailable"),
        (507, "overloaded"),
    ],
)
def test_design_statuses_map_to_kinds(status: int, kind: str) -> None:
    assert _design_failure(_status(status)).kind == kind


@pytest.mark.parametrize(
    ("error", "kind"), [(httpx.ConnectError, "unavailable"), (httpx.ReadTimeout, "timeout")], ids=("connect", "timeout")
)
def test_design_transport_failures_map_to_kinds(error: type[httpx.TransportError], kind: str) -> None:
    assert _design_failure(_raise(error)).kind == kind


def _install_nothing(provider: object) -> None:
    del provider


def test_each_request_is_timed_by_operation_and_outcome(
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
        _ = _list(_answer([], _LISTING))
        _ = _design(_answer([], _MADE, status=201))
        _ = _design_failure(_status(503))
        provider = mosfet.telemetry.meter_provider()
        assert provider is not None
        _ = provider.force_flush()
    finally:
        mosfet.telemetry.reset()

    points: list[dict[str, typing.Any]] = []
    for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        payload = typing.cast(dict[str, typing.Any], json.loads(line))
        if payload["name"] == "bot.provider.speech_server.voices.duration":
            points = typing.cast(list[dict[str, typing.Any]], payload["data"]["data_points"])
    observed = {(point["attributes"]["operation"], point["attributes"]["bot.outcome"]) for point in points}
    assert {("list", "ok"), ("design", "ok"), ("design", "failed")} <= observed
    assert {key for point in points for key in point["attributes"]} == {"operation", "bot.outcome"}
