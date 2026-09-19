"""One bot turn is one trace, end to end.

A phone text drives the whole stack with fake model processors: phone ingress -> holder
(``bot.input``) -> cognition tiers (autonomy, intuition, reasoning, reflection) ->
``phone.send_text_message`` -> firmware -> messaging service -> the service's verdict. Every span
and every log record emitted while that turn runs must carry the same trace id, every span's
parent must be a span of that trace (one root: the phone ingress), no observed event may be
recorded twice, and a behavior status write straight to memory must leave its own record.
"""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import dataclasses
import json
import pathlib
import typing

import hsm
import mosfet
import mosfet.telemetry
import pytest
from mosfet.abilities import cognition, memory, processing
from mosfet.bot import Bot
from mosfet.devices import phone
from mosfet.environment import Environment
from opentelemetry import _logs

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
_VISITOR = "visitor"


@pytest.fixture(autouse=True)
def _telemetry(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> collections.abc.Iterator[None]:
    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_logs, "set_logger_provider", lambda _provider: None)
    monkeypatch.setenv("BOT_OTEL_CAPTURE", "full")
    assert mosfet.telemetry.configure() is True
    yield
    mosfet.telemetry.reset()


def _flush() -> None:
    logs = logger_provider()
    if logs is not None:
        _ = logs.force_flush()
    spans = tracer_provider()
    if spans is not None:
        _ = spans.force_flush()


def _lines(path: str) -> list[dict[str, typing.Any]]:
    file = pathlib.Path(path)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]


class _Carrier:
    """The phone's messaging service: reports every send to the visitor as sent."""

    def __init__(self) -> None:
        self.target: hsm.Instance | None = None
        self.sent: asyncio.Queue[str] = asyncio.Queue()

    async def attach(self, environment: Environment, target: hsm.Instance) -> None:
        del environment
        self.target = target

    async def detach(self, environment: Environment, target: hsm.Instance) -> None:
        del environment
        if self.target is target:
            self.target = None

    def publish(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        if event.name != phone.ServiceTextMessageSendRequestedEvent.name or self.target is None:
            return
        request = event.data
        assert isinstance(request, phone.SendTextMessageData)
        self.sent.put_nowait(request.text)
        verdict = phone.ServiceTextMessageSentEvent.with_data(
            phone.TextMessageSentData(to=request.to, text=request.text)
        )
        _ = asyncio.ensure_future(self.target.dispatch(ctx, dataclasses.replace(verdict, id=event.id)))


class _Unhandled(processing.Processor):
    """Intuition that recognizes nothing, so the turn cascades to reasoning."""

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return typing.cast(processing.Events, typing.cast(object, processing.Result[processing.Events].unhandled()))


class _Reply(processing.Processor):
    """Reasoning that texts the visitor back, logging its model call the way a provider does."""

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        mosfet.telemetry.record_generator_request(provider="fake", model="fake-model", messages=[])
        _ = await asyncio.to_thread(lambda: None)
        mosfet.telemetry.record_generator_response(provider="fake", model="fake-model", response={}, latency_s=0.0)
        return (
            processing.SelectedEvent(
                event=phone.SendTextMessageEvent.name,
                data={"to": _VISITOR, "text": "hello back"},
                reason="answer the visitor",
            ),
        )


class _NoChange(processing.Processor):
    """Reflection that selects no behavior to author."""

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        del input
        return ()


class _PhoneBot(Bot):
    def __init__(self, *, handset: phone.Phone, cognition_ability: cognition.Cognition) -> None:
        super().__init__({"phone": handset}, cognition=cognition_ability)


async def _idle(settle: float = 0.3, limit: float = 10.0) -> None:
    """Wait until no telemetry record has been written for ``settle`` seconds."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    last = -1
    while loop.time() < deadline:
        _flush()
        count = len(_lines("otel-logs.jsonl")) + len(_lines("otel-spans.jsonl"))
        if count == last:
            return
        last = count
        await asyncio.sleep(settle)


def _run_turn() -> tuple[str, list[dict[str, typing.Any]], list[dict[str, typing.Any]]]:
    async def run() -> tuple[str, int, int]:
        store = memory.ShortTermMemory()
        ability = cognition.Cognition(
            autonomy=cognition.Autonomy(memory=store),
            intuition=cognition.Intuition(processor=_Unhandled()),
            reasoning=cognition.Reasoning(processor=_Reply(), memory=store),
            reflection=cognition.Reflection(processor=_NoChange(), memory=store),
        )
        carrier = _Carrier()
        handset = phone.Phone(service=carrier)
        body = _PhoneBot(handset=handset, cognition_ability=ability)
        environment = Environment()
        await body.attach(environment)
        await _idle()
        logs_before = len(_lines("otel-logs.jsonl"))
        spans_before = len(_lines("otel-spans.jsonl"))
        message_id = "turn-message-1"
        await handset.dispatch(
            handset.context(),
            dataclasses.replace(
                phone.SmsTextEvent.with_data(phone.SmsTextData(id=message_id, sender=_VISITOR, text="Hey")),
                id=message_id,
            ),
        )
        reply = await asyncio.wait_for(carrier.sent.get(), timeout=10)
        await _idle()
        return reply, logs_before, spans_before

    reply, logs_before, spans_before = asyncio.run(run())
    _flush()
    return reply, _lines("otel-logs.jsonl")[logs_before:], _lines("otel-spans.jsonl")[spans_before:]


def test_phone_text_turn_is_one_trace_with_intact_parents_and_no_duplicate_events() -> None:
    reply, logs, spans = _run_turn()

    assert reply == "hello back"
    assert spans and logs
    trace_ids = collections.Counter(record["trace_id"] for record in [*spans, *logs])
    assert len(trace_ids) == 1, trace_ids

    by_id = {span["span_id"]: span for span in spans}
    roots = [span for span in spans if span["parent_span_id"] is None]
    assert [(root["name"], root["attributes"].get("bot.component.name")) for root in roots] == [
        ("bot.device.ingress", "phone.ingress")
    ]
    orphans = [span["name"] for span in spans if span["parent_span_id"] and span["parent_span_id"] not in by_id]
    assert orphans == []
    for record in logs:
        assert record["span_id"] in by_id, record

    names = {span["name"] for span in spans}
    assert {"bot.body.cognition_dispatch", "bot.reasoning.stage", "bot.reflection.stage"} <= names
    # A stage cancelled by its own hand-off reads handed_off; nothing in a clean turn is cancelled.
    outcomes = collections.Counter(span["attributes"].get("bot.outcome") for span in spans)
    assert outcomes["cancelled"] == 0 and outcomes["failed"] == 0, outcomes
    assert outcomes["handed_off"] >= 1

    events = [record for record in logs if isinstance(record["body"], dict) and "occurrence" in record["body"]]
    identities = collections.Counter(
        json.dumps([record["body"]["instance"], record["body"]["element"], record["body"]["event"]], sort_keys=True)
        for record in events
    )
    duplicates = {identity: count for identity, count in identities.items() if count > 1}
    assert duplicates == {}

    generator = [
        record["attributes"]["stage"] for record in logs if record["attributes"].get("component") == "text.generator"
    ]
    assert generator == ["request", "response"]

    observed = {record["body"]["event"]["name"] for record in events}
    assert {
        "phone.notification",
        "bot.input",
        phone.SendTextMessageEvent.name,
        phone.ServiceTextMessageSentEvent.name,
    } <= observed


def test_behavior_inventory_writes_record_status_change_in_the_active_trace() -> None:
    from mosfet.abilities.cognition import inventory
    from mosfet.behavior import storage as behavior_storage
    from mosfet.behavior.instance import Instance

    async def run() -> tuple[str, Instance]:
        store = memory.ShortTermMemory()
        assert store.model is not None
        _ = await mosfet.started(None, store, store.model)
        draft = behavior_storage.mark_draft(Instance(name="GreetBack", source="", triggers=("bot.input",)))
        tracer = mosfet.telemetry.span.tracer("tests.telemetry")
        with tracer.start_as_current_span("turn") as turn:
            inventory.store_behavior(store, draft, cause="draft")
            accepted = behavior_storage.mark_active(draft.model_copy(update={"source": "behavior = 1"}))
            inventory.store_behavior(store, accepted, cause="accept")
            trace_id = format(turn.get_span_context().trace_id, "032x")
        await hsm.stop(store)
        return trace_id, accepted

    trace_id, accepted = asyncio.run(run())
    _flush()
    writes = [record for record in _lines("otel-logs.jsonl") if record["attributes"].get("stage") == "behavior_status"]
    assert [(record["body"]["from"], record["body"]["to"], record["body"]["cause"]) for record in writes] == [
        ("absent", "DRAFT", "draft"),
        ("DRAFT", "ACTIVE", "accept"),
    ]
    assert {record["body"]["behavior"] for record in writes} == {"GreetBack"}
    assert writes[0]["body"]["revision"] == "empty"
    assert writes[1]["body"]["previous_revision"] == "empty"
    assert writes[1]["body"]["revision"] == inventory.revision(accepted)
    assert {record["trace_id"] for record in writes} == {trace_id}
