from __future__ import annotations

import asyncio
import importlib

import hsm
import pytest

import bot
from bot.define import topology

_DEFINE = importlib.import_module("bot.define")


def _demo() -> hsm.Model:
    return bot.define(
        "Demo",
        bot.initial(bot.target("idle")),
        bot.state("idle", bot.transition(bot.on("go"), bot.target("../run"))),
        bot.state("run"),
    )


def test_define_returns_working_model() -> None:
    model = _demo()
    go = hsm.Event[None](name="go")

    class Demo(hsm.Instance):
        pass

    async def run() -> str:
        demo = Demo()
        _ = await bot.started(None, demo, model)
        await hsm.dispatch(None, demo, go)
        return demo.state() or ""

    assert asyncio.run(run()) == "/Demo/run"


def test_topology_includes_idle_run_and_go() -> None:
    payload = topology(_demo())
    assert payload["name"] == "/Demo"
    assert payload["initial"] == "/Demo/.initial"
    states = {state["qualified_name"]: state for state in payload["states"]}
    assert "/Demo/idle" in states
    assert "/Demo/run" in states
    assert states["/Demo/idle"]["parent"] == "/Demo"
    assert states["/Demo/run"]["parent"] == "/Demo"
    assert any(
        transition["source"] == "/Demo/idle" and transition["target"] == "/Demo/run" and "go" in transition["events"]
        for transition in payload["transitions"]
    )


def test_topology_includes_runtime_owner_when_supplied() -> None:
    owner_model = bot.define("Owner", bot.initial(bot.target("ready")), bot.state("ready"))
    child_model = bot.define("Child", bot.initial(bot.target("ready")), bot.state("ready"))

    async def run() -> str:
        owner = await bot.started(None, hsm.Instance(), owner_model)
        payload = topology(child_model, owner=owner)
        owner_name = payload.get("owner")
        assert owner_name is not None
        return owner_name

    assert asyncio.run(run()) == "/Owner"


def test_topology_can_explicitly_clear_runtime_owner() -> None:
    payload = topology(_demo(), clear_owner=True)

    assert "owner" in payload
    assert payload["owner"] is None


def test_define_without_otlp_does_not_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_MODEL_PUBLISH", raising=False)
    calls: list[object] = []

    def _fail_post(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _fail_post)
    _ = _demo()
    assert calls == []


def test_define_with_endpoint_publishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_MODEL_PUBLISH", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)
    model = _demo()
    assert len(calls) == 1
    payload, url = calls[0]
    published = topology(model)
    assert url == "http://127.0.0.1:5173/v1/models"
    assert payload == published
    assert any(
        transition["source"] == "/Demo/idle" and "go" in transition["events"] for transition in published["transitions"]
    )


def test_define_with_endpoint_but_without_opt_in_does_not_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOT_MODEL_PUBLISH", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    calls: list[tuple[object, str]] = []

    def _record(payload: object, url: str) -> None:
        calls.append((payload, url))

    monkeypatch.setattr(_DEFINE, "post_model", _record)
    _ = _demo()
    assert calls == []


def test_define_publish_failure_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_MODEL_PUBLISH", "1")
    monkeypatch.setenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    def _boom(payload: object, url: str) -> None:
        del payload, url
        raise TimeoutError("collector down")

    monkeypatch.setattr(_DEFINE, "post_model", _boom)
    model = _demo()
    assert model.qualified_name == "/Demo"
