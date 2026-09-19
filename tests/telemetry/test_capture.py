import asyncio
import collections.abc
import json
import pathlib
import typing

import hsm
import mosfet
import mosfet.telemetry
import pydantic
import pytest
from opentelemetry import _logs

from mosfet.telemetry import capture
from mosfet.telemetry.configure import logger_provider, tracer_provider

_ENV = (
    "BOT_OTEL_DISABLED",
    "BOT_OTEL_LOG_FILE",
    "BOT_OTEL_SPAN_FILE",
    "BOT_OTEL_CAPTURE",
    "BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "BOT_OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


@pytest.fixture(autouse=True)
def _telemetry(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> collections.abc.Iterator[None]:
    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", lambda _provider: None)
    yield
    mosfet.telemetry.reset()


def _records() -> list[dict[str, typing.Any]]:
    provider = logger_provider()
    if provider is not None:
        _ = provider.force_flush()
    traces = tracer_provider()
    if traces is not None:
        _ = traces.force_flush()
    path = pathlib.Path("otel-logs.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _GreetData(pydantic.BaseModel):
    text: str
    api_key: str


_GREET = hsm.Event[_GreetData](name="demo.greet", schema=_GreetData)


class _Demo(hsm.Instance):
    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "CaptureDemo",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.transition(hsm.on(_GREET), hsm.target("../greeted"))),
        hsm.state("greeted"),
        hsm.observe(mosfet.telemetry.observer),
    )


def _run_demo() -> None:
    async def run() -> None:
        demo = _Demo()
        _ = await mosfet.started(None, demo, demo.model)
        event = _GREET.with_data_and_id(_GreetData(text="Hey", api_key="sk-live-abcdefghijkl"), "turn-1")
        _ = await hsm.dispatch(None, demo, event)

    asyncio.run(run())


def test_jsonable_redacts_credential_keys_and_token_values() -> None:
    tree = capture.jsonable(
        {
            "headers": {"Authorization": "Bearer abc.def", "x": "Bearer abc.def"},
            "api_key": "k",
            "input_tokens": 3,
            "note": "use sk-abcdefghijklmnop please",
        }
    )
    assert tree == {
        "headers": {"Authorization": "<redacted>", "x": "<redacted>"},
        "api_key": "<redacted>",
        "input_tokens": 3,
        "note": "use <redacted> please",
    }


def test_full_capture_is_off_by_default_and_hsm_events_stay_payload_free() -> None:
    assert mosfet.telemetry.configure() is True
    assert not capture.full_enabled()
    _run_demo()
    assert all(record["attributes"].get("hsm.event.name") != "demo.greet" for record in _records())
    assert "Hey" not in pathlib.Path("otel-spans.jsonl").read_text(encoding="utf-8")


def test_full_capture_records_event_payload_ids_and_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True
    _run_demo()
    greets = [
        record
        for record in _records()
        if record["attributes"].get("hsm.event.name") == "demo.greet" and record["body"]["occurrence"] == "event"
    ]
    assert len(greets) == 1
    body = greets[0]["body"]
    assert body["event"]["id"] == "turn-1"
    assert body["event"]["data"] == {"text": "Hey", "api_key": "<redacted>"}
    assert body["transition"] == {"from": "/CaptureDemo/idle", "to": "/CaptureDemo/greeted"}
    assert greets[0]["trace_id"]
    assert "sk-live" not in pathlib.Path("otel-logs.jsonl").read_text(encoding="utf-8")
    spans = pathlib.Path("otel-spans.jsonl").read_text(encoding="utf-8")
    assert "Hey" not in spans


def test_generator_response_recorded_only_under_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    assert mosfet.telemetry.configure() is True
    mosfet.telemetry.record_generator_response(provider="p", model="m", response={"text": "hi"}, latency_s=0.5)
    assert _records() == []
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    mosfet.telemetry.record_generator_response(
        provider="p", model="m", response={"text": "hi"}, latency_s=0.5, error=ValueError("bad")
    )
    (record,) = _records()
    assert record["attributes"] == {
        "component": "text.generator",
        "provider": "p",
        "stage": "response",
        "outcome": "failed",
        "model": "m",
    }
    assert record["body"] == {"response": {"text": "hi"}, "latency_ms": 500.0, "error": "ValueError: bad"}


def test_record_dispatch_logs_acceptance_under_full_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True

    async def run() -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        delivery = capture.record_dispatch(
            future, component="phone.holders", event=hsm.Event(name="bot.input", id="e1"), source="a", target="b"
        )
        future.set_result(False)
        result = await delivery
        await asyncio.sleep(0)
        return result

    assert asyncio.run(run()) is False
    (record,) = _records()
    assert record["body"] == {"event": "bot.input", "id": "e1", "source": "a", "target": "b", "accepted": False}
