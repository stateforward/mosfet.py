from __future__ import annotations

import asyncio
import collections.abc
import threading
import time

import hsm
import pytest

from bot.telemetry import control


@pytest.fixture(autouse=True)
def _reset_control(monkeypatch: pytest.MonkeyPatch) -> collections.abc.Iterator[None]:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    control.reset()
    yield
    control.reset()


def test_listen_without_endpoint_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = hsm.Context()
    opened: list[str] = []

    def _forbidden(subscription: object) -> collections.abc.Iterator[tuple[str, str]]:
        del subscription
        opened.append("iter")
        return iter(())

    monkeypatch.setattr(control, "_iter_commands", _forbidden)
    control.listen(environment)
    control.unlisten(environment)
    assert opened == []


def test_grpc_target_strips_legacy_http_path() -> None:
    assert control.grpc_target("http://127.0.0.1:4317/v1/traces") == "127.0.0.1:4317"
    assert control.grpc_target("http://127.0.0.1:4317") == "127.0.0.1:4317"


def test_listen_dispatches_via_dispatch_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    environment = hsm.Context()
    delivered: list[tuple[object, str, object]] = []
    ready = threading.Event()

    def fake_iter(subscription: object) -> collections.abc.Iterator[tuple[str, str]]:
        del subscription
        yield ("phone.ring", '{"source":"dashboard"}')

    def fake_dispatch_all(target: object, event: hsm.Event[object]) -> None:
        delivered.append((target, event.name, event.data))
        ready.set()

    monkeypatch.setattr(control, "_iter_commands", fake_iter)
    monkeypatch.setattr(hsm, "dispatch_all", fake_dispatch_all)

    async def run() -> None:
        control.listen(environment)
        deadline = time.monotonic() + 2.0
        while not ready.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        control.unlisten(environment)

    asyncio.run(run())
    assert delivered == [(environment, "phone.ring", {"source": "dashboard"})]


def test_listen_is_idempotent_per_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317")
    environment = hsm.Context()
    starts = 0

    def fake_iter(subscription: object) -> collections.abc.Iterator[tuple[str, str]]:
        nonlocal starts
        starts += 1
        del subscription
        return iter(())

    monkeypatch.setattr(control, "_iter_commands", fake_iter)
    control.listen(environment)
    control.listen(environment)
    time.sleep(0.05)
    control.unlisten(environment)
    control.unlisten(environment)
    assert starts == 1
