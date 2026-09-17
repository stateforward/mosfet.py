from __future__ import annotations

import asyncio
import importlib
import typing

import hsm
import pytest

import mosfet
from mosfet.environment import Environment
from mosfet.start import live_payload

_DEFINE = importlib.import_module("mosfet.define")


def _demo() -> hsm.Model:
    return mosfet.define(
        "Demo",
        mosfet.initial(mosfet.target("idle")),
        mosfet.state("idle", mosfet.transition(mosfet.on("go"), mosfet.target("../run"))),
        mosfet.state("run"),
    )


class Demo(hsm.Instance):
    pass


def test_started_snapshot_is_initial_real_state() -> None:
    model = _demo()

    async def run() -> str:
        demo = Demo()
        _ = await mosfet.started(None, demo, model)
        return demo.take_snapshot().State

    assert asyncio.run(run()) == "/Demo/idle"


def test_start_snapshot_is_initial_real_state() -> None:
    model = _demo()

    async def run() -> str:
        demo = hsm.new(Demo(), model)
        _ = await mosfet.start(None, demo)
        return demo.take_snapshot().State

    assert asyncio.run(run()) == "/Demo/idle"


def test_live_payload_includes_runtime_owner() -> None:
    owner_model = _demo()
    child_model = mosfet.define(
        "Child",
        mosfet.initial(mosfet.target("idle")),
        mosfet.state("idle"),
    )

    async def run() -> str:
        owner = await mosfet.started(None, Demo(), owner_model)
        child = await mosfet.started(None, Demo(), child_model, owner=owner)
        payload = live_payload(child, owner=owner)
        owner_name = payload.get("owner")
        assert owner_name is not None
        return owner_name

    assert asyncio.run(run()) == "/Demo"


def test_live_payload_can_explicitly_clear_runtime_owner() -> None:
    payload = live_payload(Demo(), clear_owner=True)

    assert "owner" in payload
    assert payload["owner"] is None


def test_register_refreshes_owner_without_restarting_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[dict[str, object], str]] = []

    def _record(payload: dict[str, object], url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)
    owner_one_model = mosfet.define("OwnerOne", mosfet.initial(mosfet.target("ready")), mosfet.state("ready"))
    owner_two_model = mosfet.define("OwnerTwo", mosfet.initial(mosfet.target("ready")), mosfet.state("ready"))
    child_model = mosfet.define("Child", mosfet.initial(mosfet.target("ready")), mosfet.state("ready"))

    async def run() -> tuple[str, str]:
        owner_one = await mosfet.started(None, Demo(), owner_one_model)
        child = await mosfet.started(None, Demo(), child_model, owner=owner_one)
        owner_two = await mosfet.started(None, Demo(), owner_two_model)
        calls.clear()
        before = hsm.id(child)
        _ = mosfet.register(child, child_model, owner=owner_two)
        return before, hsm.id(child)

    before, after = asyncio.run(run())
    live_payloads = [payload for payload, url in calls if url.endswith("/v1/models/live")]
    assert before == after
    assert len(live_payloads) == 1
    live_payload = live_payloads[0]
    owner = live_payload.get("owner")
    assert isinstance(owner, str)
    assert owner == "/OwnerTwo"


def test_started_without_otlp_does_not_publish_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    calls: list[object] = []

    def _fail_post(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _fail_post)

    async def run() -> None:
        _ = await mosfet.started(None, Demo(), _demo())

    asyncio.run(run())
    assert calls == []


def test_started_with_endpoint_posts_live(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _demo()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        _ = await mosfet.started(None, Demo(), model)

    asyncio.run(run())
    assert (
        {"name": "/Demo", "component": "Demo", "state": "/Demo/idle", "live": True},
        "http://127.0.0.1:5173/v1/models/live",
    ) in calls


def test_started_in_environment_publishes_explicit_null_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _demo()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        _ = await mosfet.started(Environment(), Demo(), model)

    asyncio.run(run())
    assert any(
        typing.cast(dict[str, object], payload).get("owner") is None
        for payload, url in calls
        if url.endswith("/v1/models") and isinstance(payload, dict)
    )


def test_started_in_derived_environment_publishes_explicit_null_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _demo()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        environment = Environment()
        _ = await mosfet.started(environment.with_value("derived", True), Demo(), model)

    asyncio.run(run())
    assert any(
        typing.cast(dict[str, object], payload).get("owner") is None
        for payload, url in calls
        if url.endswith("/v1/models") and isinstance(payload, dict)
    )
    assert any(
        typing.cast(dict[str, object], payload).get("owner") is None
        for payload, url in calls
        if url.endswith("/v1/models/live") and isinstance(payload, dict)
    )


def test_started_in_private_context_omits_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _demo()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)

    async def run() -> None:
        _ = await mosfet.started(hsm.Context(), Demo(), model)

    asyncio.run(run())
    for payload, url in calls:
        if url.endswith("/v1/models") or url.endswith("/v1/models/live"):
            assert isinstance(payload, dict)
            assert "owner" not in payload


def test_started_publish_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _demo()
    monkeypatch.setenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    def _boom(payload: object, url: str) -> None:
        del payload, url
        raise TimeoutError("collector down")

    monkeypatch.setattr(_DEFINE, "post_model", _boom)

    async def run() -> str:
        demo = Demo()
        _ = await mosfet.started(None, demo, model)
        return demo.take_snapshot().State

    assert asyncio.run(run()) == "/Demo/idle"
