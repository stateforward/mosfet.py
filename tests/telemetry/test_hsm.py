import asyncio
import datetime
import logging
import typing

import hsm
import bot
import pydantic
import pytest

import bot.telemetry.hsm as telemetry


class _RecordingCounter:
    calls: list[tuple[int, dict[str, object]]]

    def __init__(self) -> None:
        self.calls = []

    def add(self, amount: int, attributes: dict[str, object]) -> None:
        self.calls.append((amount, attributes))


class _FailingCounter:
    def add(self, amount: int, attributes: dict[str, object]) -> None:
        del amount, attributes
        raise RuntimeError("metrics backend unavailable")


class _RecordingSpan:
    name: str
    attributes: dict[str, object]
    statuses: list[object]

    def __init__(self, name: str, attributes: dict[str, object]) -> None:
        self.name = name
        self.attributes = attributes
        self.statuses = []

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, exc, traceback
        return False

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        self.statuses.append(status)


class _RecordingTracer:
    spans: list[_RecordingSpan]

    def __init__(self) -> None:
        self.spans = []

    def start_as_current_span(
        self,
        name: str,
        *,
        context: object = None,
        attributes: dict[str, object] | None = None,
    ) -> _RecordingSpan:
        del context
        span = _RecordingSpan(name, dict(attributes or {}))
        self.spans.append(span)
        return span


class _DemoInstance(hsm.Instance):
    pass


def _observation_for(event: hsm.Event[object], *, source: str = "/Demo/.initial") -> hsm.Event[dict[str, object]]:
    return hsm.Event[dict[str, object]](
        name="hsm/observation",
        source=source,
        data={
            "event": event,
            "occurrence": "event",
            "time": datetime.datetime.now(datetime.UTC),
        },
    )


def test_observation_attributes_exclude_high_cardinality_event_values() -> None:
    event = hsm.Event[dict[str, str]](
        name="bot.input",
        kind=280,
        id="request-123",
        source="operator-42",
        target="bot-99",
        data={"message": "private"},
        metadata={"traceparent": "00-trace", "operator.id": "private"},
    )

    attributes = telemetry.observation_attributes(
        _observation_for(event, source="/Bot/active/observing/to_processing"),
        instance=_DemoInstance(),
        outcome="observed",
    )

    assert attributes == {
        "hsm.observation.occurrence": "event",
        "hsm.observation.source": "/Bot/active/observing/to_processing",
        "hsm.event.name": "bot.input",
        "hsm.event.kind": 280,
        "hsm.event.has_data": True,
        "hsm.event.has_metadata": True,
        "hsm.machine.name": "",
        "hsm.machine.state": "",
        "bot.component.name": "_DemoInstance",
        "bot.outcome": "observed",
    }
    assert "hsm.event.id" not in attributes
    assert "hsm.event.source" not in attributes
    assert "hsm.event.target" not in attributes
    assert "hsm.event.data" not in attributes
    assert "operator.id" not in attributes


def test_observer_records_metric_and_short_span(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _RecordingCounter()
    tracer = _RecordingTracer()
    event = hsm.Event[None](name="bot.ability.input", metadata={"traceparent": "00-parent"})
    observation = _observation_for(event, source="/TextGeneration/idle/to_applying")

    monkeypatch.setattr(telemetry, "_OBSERVATIONS", counter)
    monkeypatch.setattr(telemetry, "_TRACER", tracer)

    telemetry.observer(hsm.Context(), _DemoInstance(), observation)

    assert counter.calls == [
        (
            1,
            {
                "hsm.observation.occurrence": "event",
                "hsm.observation.source": "/TextGeneration/idle/to_applying",
                "hsm.event.name": "bot.ability.input",
                "hsm.event.kind": 280,
                "hsm.event.has_data": False,
                "hsm.event.has_metadata": True,
                "hsm.machine.name": "",
                "hsm.machine.state": "",
                "bot.component.name": "_DemoInstance",
                "bot.outcome": "observed",
            },
        )
    ]
    assert tracer.spans[0].name == "bot.hsm.observe"
    assert tracer.spans[0].attributes["bot.outcome"] == "observed"
    assert tracer.spans[0].statuses == []


def test_hsm_observe_records_transition_without_dispatch_telemetry_api(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _RecordingCounter()
    tracer = _RecordingTracer()
    go_event = hsm.Event[None](name="demo.go")

    class Demo(_DemoInstance):
        model: typing.ClassVar[hsm.Model] = bot.define(
            "Demo",
            hsm.initial(hsm.target("idle")),
            hsm.state(
                "idle",
                hsm.transition(
                    hsm.on(go_event),
                    hsm.target("../done"),
                ),
            ),
            hsm.state("done"),
            hsm.observe(telemetry.observer),
        )

    monkeypatch.setattr(telemetry, "_OBSERVATIONS", counter)
    monkeypatch.setattr(telemetry, "_TRACER", tracer)

    async def run() -> None:
        demo = Demo()
        _ = await bot.started(None, demo, demo.model)
        await hsm.dispatch(None, demo, go_event)

    asyncio.run(run())

    event_names = [attributes["hsm.event.name"] for _, attributes in counter.calls]
    assert "hsm/initial" in event_names
    assert "demo.go" in event_names
    assert all(span.name == "bot.hsm.observe" for span in tracer.spans)


def test_hsm_observe_metric_failure_does_not_block_domain_transition(monkeypatch: pytest.MonkeyPatch) -> None:
    failure_counter = _RecordingCounter()
    tracer = _RecordingTracer()
    go_event = hsm.Event[None](name="demo.go")
    effects: list[str] = []

    def record_domain_effect(ctx: hsm.Context, instance: hsm.Instance, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        effects.append(event.name)

    class Demo(_DemoInstance):
        model: typing.ClassVar[hsm.Model] = bot.define(
            "Demo",
            hsm.initial(hsm.target("idle")),
            hsm.state(
                "idle",
                hsm.transition(
                    hsm.on(go_event),
                    hsm.effect(record_domain_effect),
                    hsm.target("../done"),
                ),
            ),
            hsm.state("done"),
            hsm.observe(telemetry.observer),
        )

    monkeypatch.setattr(telemetry, "_OBSERVATIONS", _FailingCounter())
    monkeypatch.setattr(telemetry, "_OBSERVATION_FAILURES", failure_counter)
    monkeypatch.setattr(telemetry, "_TRACER", tracer)

    async def run() -> Demo:
        demo = Demo()
        _ = await bot.started(None, demo, demo.model)
        await hsm.dispatch(None, demo, go_event)
        return demo

    demo = asyncio.run(run())

    assert effects == ["demo.go"]
    assert demo.take_snapshot().State == "/Demo/done"
    failed_event_names = [attributes["hsm.event.name"] for _, attributes in failure_counter.calls]
    assert "hsm/initial" in failed_event_names
    assert "demo.go" in failed_event_names
    assert all(attributes["bot.outcome"] == "failed" for _, attributes in failure_counter.calls)


def test_observer_logs_the_reason_a_failure_carries(caplog: pytest.LogCaptureFixture) -> None:
    """A failure that knows why it failed must say so, not report ``has_data=True``.

    Every ability failure in the system was invisible this way: the message reached typed event
    data and nothing printed it, so a mute robot and a working one produced identical logs.
    """

    class FailureData(pydantic.BaseModel):
        message: str

    failed = hsm.Event[FailureData](
        name="bot.ability.test.apply.failed",
        kind=hsm.ErrorEventKind,
        schema=FailureData,
    ).with_data(FailureData(message="Device is not started in this environment."))

    with caplog.at_level(logging.ERROR, logger="bot.telemetry.hsm"):
        telemetry.observer(hsm.Context(), _DemoInstance(), _observation_for(failed))

    failures = [record for record in caplog.records if "hsm failure" in record.getMessage()]
    assert len(failures) == 1
    assert "Device is not started in this environment." in failures[0].getMessage()
    assert "bot.ability.test.apply.failed" in failures[0].getMessage()


def test_observer_stays_quiet_for_ordinary_events(caplog: pytest.LogCaptureFixture) -> None:
    """Only failures are raised to ERROR; ordinary traffic keeps its DEBUG line."""

    ordinary = hsm.Event[None](name="bot.ability.test.applied")

    with caplog.at_level(logging.ERROR, logger="bot.telemetry.hsm"):
        telemetry.observer(hsm.Context(), _DemoInstance(), _observation_for(ordinary))

    assert [record for record in caplog.records if "hsm failure" in record.getMessage()] == []


def test_observer_keeps_an_unstamped_event_in_the_caller_trace() -> None:
    """An in-process observation nests under the span that is running, not a new root trace.

    Almost no event in this codebase is stamped with `inject_context` — stamping is for crossing a
    process or transport. If an unstamped event reparented to nothing, every observation along a
    stimulus path would land in its own trace and the path could not be read end to end.
    """

    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    tracer = provider.get_tracer("bot.telemetry.test")
    event = hsm.Event[None](name="bot.test.in_process")
    assert event.metadata == {}

    with tracer.start_as_current_span("caller") as caller:
        with tracer.start_as_current_span(
            "observed",
            context=telemetry.event_context(event),
        ) as observed:
            assert observed.get_span_context().trace_id == caller.get_span_context().trace_id
