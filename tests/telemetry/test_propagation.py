"""Trace context rides every event and every behavior an event drives.

- `Traced.dispatch` stamps the dispatcher's context on the event, so an event queued behind a busy
  machine is processed in its own trace, not in the trace that started that processing run.
- `EventContextBinding` (the validator and finalizer of every ``mosfet.define`` model) runs every
  behavior and guard in its event's context, sync and async alike, accepts class-body
  ``staticmethod`` activities without relying on an observer wrapper, and follows the model
  through ``hsm.redefine``: what a redefine adds is bound and observed exactly once.
- A stage that dispatches to its own machine and is cancelled by the transition it caused reads
  ``handed_off``; a stage cancelled by the machine stopping reads ``cancelled``.
- No model is observed twice: observation is wired on the machine a runtime starts, never also on
  a submodel embedded in it.
"""

from __future__ import annotations

import asyncio
import collections.abc
import importlib
import json
import pathlib
import pkgutil
import typing

import hsm
import mosfet
import mosfet.abilities
import mosfet.device
import mosfet.telemetry
import pytest
from opentelemetry import _logs, trace
from opentelemetry import context as otel_context

from mosfet.telemetry import span
from mosfet.telemetry.configure import span_file, tracer_provider

_ENV = (
    "BOT_OTEL_DISABLED",
    "BOT_OTEL_LOG_FILE",
    "BOT_OTEL_SPAN_FILE",
    "BOT_OTEL_CAPTURE",
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
    assert mosfet.telemetry.configure() is True
    yield
    mosfet.telemetry.reset()


def _spans() -> list[dict[str, typing.Any]]:
    provider = tracer_provider()
    if provider is not None:
        _ = provider.force_flush()
    path = span_file()
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _trace_id() -> int:
    return trace.get_current_span().get_span_context().trace_id


_FIRST = hsm.Event(name="test.first")
_SECOND = hsm.Event(name="test.second")
_GO = hsm.Event(name="test.go")
_DONE = hsm.Event(name="test.done")


class _Recorder(mosfet.telemetry.Traced):
    """Records the trace each behavior ran in; the first effect queues a second event from another trace."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, int] = {}
        self.foreign: otel_context.Context = otel_context.Context()

    @staticmethod
    def _on_first(ctx: hsm.Context, instance: "_Recorder", event: hsm.Event[typing.Any]) -> None:
        instance.seen["first"] = _trace_id()
        # The machine is busy processing this event, so the next one is queued, not run here.
        token = otel_context.attach(instance.foreign)
        try:
            _ = hsm.dispatch(ctx, instance, _SECOND)
        finally:
            otel_context.detach(token)

    @staticmethod
    def _on_second(ctx: hsm.Context, instance: "_Recorder", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance.seen["second"] = _trace_id()

    @staticmethod
    def _allow_second(ctx: hsm.Context, instance: "_Recorder", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        instance.seen["guard"] = _trace_id()
        return True

    @staticmethod
    async def _work(ctx: hsm.Context, instance: "_Recorder", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance.seen["activity"] = _trace_id()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "PropagationRecorder",
        hsm.initial(hsm.target("idle")),
        hsm.state(
            "idle",
            hsm.transition(hsm.on(_FIRST), hsm.effect(_on_first), hsm.target("../busy")),
        ),
        hsm.state(
            "busy",
            hsm.activity(_work),
            hsm.transition(hsm.on(_SECOND), hsm.guard(_allow_second), hsm.effect(_on_second), hsm.target("../idle")),
        ),
    )


def test_queued_event_runs_in_its_dispatcher_trace_and_behaviors_follow_their_event() -> None:
    tracer = span.tracer("tests.telemetry.propagation")

    async def run() -> tuple[_Recorder, int, int]:
        recorder = _Recorder()
        _ = await mosfet.started(None, recorder, recorder.model)
        with tracer.start_as_current_span("foreign") as foreign:
            recorder.foreign = otel_context.get_current()
            foreign_trace = foreign.get_span_context().trace_id
        with tracer.start_as_current_span("turn") as turn:
            turn_trace = turn.get_span_context().trace_id
            _ = await hsm.dispatch(None, recorder, _FIRST)
        await asyncio.sleep(0.01)
        await hsm.stop(recorder)
        return recorder, turn_trace, foreign_trace

    recorder, turn_trace, foreign_trace = asyncio.run(run())

    assert turn_trace != foreign_trace
    assert recorder.seen["first"] == turn_trace
    assert recorder.seen["activity"] == turn_trace
    assert recorder.seen["guard"] == foreign_trace
    assert recorder.seen["second"] == foreign_trace


def test_stamp_keeps_a_foreign_trace_and_refreshes_its_own() -> None:
    tracer = span.tracer("tests.telemetry.propagation")
    with tracer.start_as_current_span("origin"):
        foreign = mosfet.telemetry.inject_context(hsm.Event(name="test.remote"))
    with tracer.start_as_current_span("turn") as turn:
        kept = mosfet.telemetry.stamp_context(foreign)
        with tracer.start_as_current_span("stage") as stage:
            own = mosfet.telemetry.stamp_context(mosfet.telemetry.inject_context(hsm.Event(name="test.own")))
            inner = mosfet.telemetry.stamp_context(own)
    assert kept.metadata == foreign.metadata
    carried = trace.get_current_span(mosfet.telemetry.event_context(inner)).get_span_context()
    assert carried.trace_id == turn.get_span_context().trace_id
    assert carried.span_id == stage.get_span_context().span_id
    assert mosfet.telemetry.stamp_context(hsm.Event(name="test.idle")).metadata == {}


class _Stage(mosfet.telemetry.Traced):
    """An activity that opens a stage span and either hands off to its own machine or waits."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = asyncio.Event()

    @staticmethod
    async def _run(ctx: hsm.Context, instance: "_Stage", event: hsm.Event[typing.Any]) -> None:
        with span.operation("test.stage", scope="tests", component="test.stage", stage="work"):
            if event.name == _GO.name:
                await hsm.dispatch(ctx, instance, _DONE)
            instance.waiting.set()
            await asyncio.Event().wait()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "HandoffStage",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle", hsm.transition(hsm.on(_GO), hsm.target("../working"))),
        hsm.state("parked", hsm.activity(_run)),
        hsm.state("working", hsm.activity(_run), hsm.transition(hsm.on(_DONE), hsm.target("../idle"))),
        hsm.transition(hsm.on(_FIRST), hsm.target("/HandoffStage/parked")),
    )


def test_self_dispatch_cancellation_is_handed_off_and_stop_is_cancelled() -> None:
    async def run() -> None:
        stage = _Stage()
        _ = await mosfet.started(None, stage, stage.model)
        _ = await hsm.dispatch(None, stage, _GO)
        await asyncio.sleep(0.01)
        _ = await hsm.dispatch(None, stage, _FIRST)
        await stage.waiting.wait()
        await hsm.stop(stage)

    asyncio.run(run())

    outcomes = [
        (record["attributes"]["bot.outcome"], record["attributes"].get("bot.handoff.event"))
        for record in _spans()
        if record["name"] == "test.stage"
    ]
    assert outcomes == [("handed_off", _DONE.name), ("cancelled", None)]


def _observation_layers(operation: object) -> int:
    layers = 0
    current: object | None = operation
    while current is not None:
        # The code object, not ``__qualname__``: a ``functools.wraps`` wrapper copies the latter.
        code = getattr(current, "__code__", None)
        if code is not None and "wrap_behavior" in code.co_qualname:
            layers += 1
        closure = typing.cast("tuple[typing.Any, ...]", getattr(current, "__closure__", None) or ())
        names = getattr(code, "co_freevars", ())
        cells: dict[str, object] = {
            name: typing.cast(object, cell.cell_contents)
            for name, cell in zip(typing.cast(tuple[str, ...], names), closure, strict=False)
        }
        current = getattr(current, "__wrapped__", None) or cells.get("original") or cells.get("operation")
        if current is operation:
            break
    return layers


def _models() -> list[hsm.Model]:
    for module in pkgutil.walk_packages(mosfet.abilities.__path__, "mosfet.abilities."):
        _ = importlib.import_module(module.name)
    seen: list[hsm.Model] = []
    pending: list[type] = [mosfet.abilities.Ability, mosfet.Bot, mosfet.device.Device]
    visited: set[type] = set()
    while pending:
        cls = pending.pop()
        if cls in visited:
            continue
        visited.add(cls)
        pending.extend(cls.__subclasses__())
        model = cls.__dict__.get("model")
        if isinstance(model, hsm.Model):
            seen.append(model)
    return seen


def test_no_behavior_is_observed_more_than_once() -> None:
    doubled = sorted(
        {
            f"{model.qualified_name}:{name}"
            for model in _models()
            for name, member in model.members.items()
            if isinstance(member, hsm.BehaviorElement)
            and _observation_layers(typing.cast(hsm.BehaviorElement[typing.Any], member).operation) > 1
        }
    )
    assert doubled == []


_EXTEND = hsm.Event(name="test.extend")


class _Extended(mosfet.telemetry.Traced):
    """A ``mosfet.define`` model, observed, then extended with ``hsm.redefine``."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, int] = {}
        self.observed: list[tuple[str, str, int]] = []

    @staticmethod
    def _observe(ctx: hsm.Context, instance: "_Extended", observation: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.observed.append((observation.source, mosfet.telemetry.observed_occurrence(observation), _trace_id()))

    @staticmethod
    def _on_extend(ctx: hsm.Context, instance: "_Extended", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance.seen["effect"] = _trace_id()

    @staticmethod
    def _enter_extended(ctx: hsm.Context, instance: "_Extended", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance.seen["entry"] = _trace_id()

    @staticmethod
    async def _run_extended(ctx: hsm.Context, instance: "_Extended", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance.seen["activity"] = _trace_id()

    base: typing.ClassVar[hsm.Model] = mosfet.define(
        "Extended",
        hsm.initial(hsm.target("idle")),
        hsm.state("idle"),
        hsm.observe(_observe),
    )
    model: typing.ClassVar[hsm.Model] = hsm.redefine(
        base,
        hsm.transition(hsm.on(_EXTEND), hsm.source("idle"), hsm.effect(_on_extend), hsm.target("extended")),
        hsm.state("extended", hsm.entry(_enter_extended), hsm.activity(_run_extended)),
    )


def test_redefine_added_behaviors_run_in_their_event_trace_and_are_observed_once() -> None:
    tracer = span.tracer("tests.telemetry.propagation")

    async def run() -> tuple[_Extended, int]:
        extended = _Extended()
        _ = await mosfet.started(None, extended, extended.model)
        extended.observed.clear()
        with tracer.start_as_current_span("turn") as turn:
            turn_trace = turn.get_span_context().trace_id
            _ = await hsm.dispatch(None, extended, _EXTEND)
        await asyncio.sleep(0.01)
        await hsm.stop(extended)
        return extended, turn_trace

    extended, turn_trace = asyncio.run(run())

    assert extended.seen == {"effect": turn_trace, "entry": turn_trace, "activity": turn_trace}
    added = ("_on_extend", "_enter_extended", "_run_extended")
    behaviors = sorted(
        (source.rsplit("/", 1)[-1], trace_id)
        for source, occurrence, trace_id in extended.observed
        if occurrence == "behavior" and source.rsplit("/", 1)[-1] in added
    )
    assert behaviors == sorted((name, turn_trace) for name in added)
    transitions = [
        (source, trace_id)
        for source, occurrence, trace_id in extended.observed
        if occurrence == "event" and source.startswith("/Extended/transition")
    ]
    assert len(transitions) == 1
    assert transitions[0][1] == turn_trace
    doubled = [
        name
        for name, member in extended.model.members.items()
        if isinstance(member, hsm.BehaviorElement)
        and _observation_layers(typing.cast(hsm.BehaviorElement[typing.Any], member).operation) > 1
    ]
    assert doubled == []
