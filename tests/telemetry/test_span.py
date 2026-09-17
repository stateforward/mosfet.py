from __future__ import annotations

import asyncio
import collections.abc
import json
import pathlib
import queue as queue_module
import threading
import typing

import mosfet.telemetry
import hsm
import pytest
from opentelemetry import _logs, trace

from mosfet.telemetry import span
from mosfet.telemetry.configure import span_file, tracer_provider


def _flush() -> None:
    provider = tracer_provider()
    if provider is not None:
        _ = provider.force_flush()


def _spans() -> list[dict[str, object]]:
    _flush()
    path = span_file()
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _attributes(record: dict[str, object]) -> dict[str, object]:
    attributes = record["attributes"]
    assert isinstance(attributes, dict)
    return typing.cast(dict[str, object], attributes)


@pytest.fixture(autouse=True)
def _reset_telemetry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> collections.abc.Iterator[None]:
    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    monkeypatch.delenv("BOT_OTEL_DISABLED", raising=False)
    monkeypatch.delenv("BOT_OTEL_LOG_FILE", raising=False)
    monkeypatch.delenv("BOT_OTEL_SPAN_FILE", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("BOT_OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    # Silence set-once warnings; emission uses the module-retained providers.
    def _noop_set_logger_provider(_provider: object) -> None:
        return None

    def _noop_set_tracer_provider(_provider: object) -> None:
        return None

    monkeypatch.setattr(_logs, "set_logger_provider", _noop_set_logger_provider)
    monkeypatch.setattr(trace, "set_tracer_provider", _noop_set_tracer_provider)
    yield
    mosfet.telemetry.reset()


def test_configure_exports_spans_to_local_jsonl() -> None:
    assert mosfet.telemetry.configure() is True
    assert span_file() == pathlib.Path("otel-spans.jsonl").resolve()

    with span.operation(
        "bot.test.operation",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="probe",
    ) as active:
        active.set_attribute("bot.frames.count", 3)

    exported = _spans()
    assert len(exported) == 1
    record = exported[0]
    assert record["name"] == "bot.test.operation"
    assert record["status"] == "UNSET"
    attributes = record["attributes"]
    assert attributes == {
        "bot.component.name": "telemetry.test",
        "bot.stage": "probe",
        "bot.frames.count": 3,
        "bot.outcome": "ok",
    }
    assert isinstance(record["trace_id"], str)
    assert len(str(record["trace_id"])) == 32


def test_span_file_honours_override() -> None:
    assert mosfet.telemetry.configure(span_file="traces/spans.jsonl") is True
    assert span_file() == (pathlib.Path("traces") / "spans.jsonl").resolve()


def test_disabled_configure_exports_no_spans() -> None:
    assert mosfet.telemetry.configure(enabled=False) is False
    assert span_file() is None
    with span.operation(
        "bot.test.operation",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="probe",
    ):
        pass
    assert not pathlib.Path("otel-spans.jsonl").exists()


def test_emission_is_noop_until_configure() -> None:
    """Library models never install a provider; observation stays opt-in."""

    with span.operation(
        "bot.test.operation",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="probe",
    ):
        pass
    assert not pathlib.Path("otel-spans.jsonl").exists()


class _KindedError(Exception):
    failure_kind = "connect_failed"


def test_operation_records_normalized_failure_kind() -> None:
    assert mosfet.telemetry.configure() is True
    with pytest.raises(_KindedError):
        with span.operation(
            "bot.test.operation",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="connect",
        ):
            raise _KindedError

    record = _spans()[0]
    assert record["status"] == "ERROR"
    assert record["attributes"] == {
        "bot.component.name": "telemetry.test",
        "bot.stage": "connect",
        "bot.outcome": "failed",
        "bot.failure.kind": "connect_failed",
    }


def test_operation_falls_back_to_snake_cased_exception_type() -> None:
    assert mosfet.telemetry.configure() is True
    with pytest.raises(ValueError):
        with span.operation(
            "bot.test.operation",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="decode",
        ):
            raise ValueError("payload text that must not reach a span attribute")

    attributes = _spans()[0]["attributes"]
    assert isinstance(attributes, dict)
    assert attributes["bot.failure.kind"] == "value_error"
    assert "payload text" not in json.dumps(_spans()[0])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("connect_failed", "connect_failed"),
        ("Connect Failed", "connect_failed"),
        ("RoomAudioError", "room_audio_error"),
        ("  ", "unknown"),
        ("!!!", "unknown"),
        ("x" * 200, "x" * 64),
    ],
)
def test_normalized_kind(raw: str, expected: str) -> None:
    assert span.normalized_kind(raw) == expected


def test_failure_kind_prefers_error_kind_over_fallback() -> None:
    assert span.failure_kind(_KindedError(), "publish_failed") == "connect_failed"
    assert span.failure_kind(ValueError("boom"), "publish_failed") == "publish_failed"


def _trace_ids(records: list[dict[str, object]]) -> dict[str, str]:
    return {str(record["name"]): str(record["trace_id"]) for record in records}


def test_bind_is_safe_through_to_thread() -> None:
    """``asyncio.to_thread`` already copies the context; binding must not break that."""

    assert mosfet.telemetry.configure() is True

    def work() -> None:
        with span.operation(
            "bot.test.child",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="worker",
        ):
            pass

    async def parent() -> None:
        with span.operation(
            "bot.test.parent",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="parent",
        ):
            await asyncio.to_thread(span.bind(work))

    asyncio.run(parent())

    traces = _trace_ids(_spans())
    assert traces["bot.test.child"] == traces["bot.test.parent"]


def test_unbound_thread_hop_detaches_into_a_new_trace() -> None:
    """The failure this mechanism exists to prevent, pinned so it stays visible."""

    assert mosfet.telemetry.configure() is True

    def work() -> None:
        with span.operation(
            "bot.test.child",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="worker",
        ):
            pass

    with span.operation(
        "bot.test.parent",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="parent",
    ):
        worker = threading.Thread(target=work)
        worker.start()
        worker.join()

    traces = _trace_ids(_spans())
    assert traces["bot.test.child"] != traces["bot.test.parent"]


def test_long_lived_worker_thread_needs_binding_per_item() -> None:
    """The mlx-style hop: one worker started at bring-up, items handed over later."""

    assert mosfet.telemetry.configure() is True
    queue: queue_module.Queue[collections.abc.Callable[[], None] | None] = queue_module.Queue()

    def consume() -> None:
        while True:
            item = queue.get()
            try:
                if item is None:
                    return
                item()
            finally:
                queue.task_done()

    def work() -> None:
        with span.operation(
            "bot.test.child",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="worker",
        ):
            pass

    # Started outside any span, exactly like a detector worker at bring-up.
    worker = threading.Thread(target=consume)
    worker.start()
    try:
        with span.operation(
            "bot.test.parent",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="parent",
        ):
            queue.put(span.bind(work))
            queue.join()
    finally:
        queue.put(None)
        worker.join()

    traces = _trace_ids(_spans())
    assert traces["bot.test.child"] == traces["bot.test.parent"]


def test_bind_reattaches_context_in_a_dedicated_worker_thread() -> None:
    assert mosfet.telemetry.configure() is True

    def work() -> None:
        with span.operation(
            "bot.test.child",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="worker",
        ):
            pass

    with span.operation(
        "bot.test.parent",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="parent",
    ):
        worker = threading.Thread(target=span.bind(work))
        worker.start()
        worker.join()

    traces = _trace_ids(_spans())
    assert traces["bot.test.child"] == traces["bot.test.parent"]


def test_bind_restores_the_worker_thread_context() -> None:
    """Attach/detach must be balanced so a pooled thread is not left contaminated."""

    assert mosfet.telemetry.configure() is True
    observed: list[bool] = []

    def work() -> None:
        with span.operation(
            "bot.test.child",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="worker",
        ):
            pass

    def after() -> None:
        observed.append(trace.get_current_span().get_span_context().is_valid)

    with span.operation(
        "bot.test.parent",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="parent",
    ):
        bound = span.bind(work)

    def run() -> None:
        bound()
        after()

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()

    assert observed == [False]


def test_inject_context_round_trips_across_a_process_boundary() -> None:
    assert mosfet.telemetry.configure() is True
    event = hsm.Event[None](name="bot.test.crossing")

    with span.operation(
        "bot.test.emit",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="emit",
    ) as emitting:
        carried = mosfet.telemetry.inject_context(event)
        emitted_trace_id = emitting.get_span_context().trace_id

    assert "traceparent" in carried.metadata
    assert event.metadata == {}

    # Receiving process: no active context, only the serialized carrier.
    received = mosfet.telemetry.event_context(carried)
    with span.operation(
        "bot.test.receive",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="receive",
        context=received,
    ) as receiving:
        assert receiving.get_span_context().trace_id == emitted_trace_id


def test_inject_context_without_an_active_span_is_a_noop() -> None:
    assert mosfet.telemetry.configure() is True
    event = hsm.Event[None](name="bot.test.crossing")
    assert mosfet.telemetry.inject_context(event) is event


def test_operation_keeps_a_recorded_failure_on_clean_exit() -> None:
    """A failure that is reported rather than raised must not close as ``ok``.

    A guard that declines, a chunk that is dropped, a stage that answers with a typed failure
    event all leave the block normally. Defaulting the outcome to ``ok`` there would make every
    non-exception failure look like a success — which is most of them in this codebase.
    """

    assert mosfet.telemetry.configure() is True
    with span.operation(
        "bot.test.operation",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="guard",
    ) as active:
        span.record_failure(active, "decode_failed")

    record = _spans()[0]
    assert record["status"] == "ERROR"
    assert _attributes(record)["bot.outcome"] == "failed"
    assert _attributes(record)["bot.failure.kind"] == "decode_failed"


def test_record_current_failure_marks_the_enclosing_operation() -> None:
    assert mosfet.telemetry.configure() is True
    with span.operation(
        "bot.test.operation",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="decode",
    ):
        span.record_current_failure("SpeechDecodeFailed")

    record = _spans()[0]
    assert _attributes(record)["bot.outcome"] == "failed"
    assert _attributes(record)["bot.failure.kind"] == "speech_decode_failed"


def test_operation_still_closes_ok_without_a_recorded_failure() -> None:
    assert mosfet.telemetry.configure() is True
    with span.operation(
        "bot.test.operation",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="decode",
    ):
        pass

    assert _attributes(_spans()[0])["bot.outcome"] == "ok"


def test_unstamped_event_stays_in_the_ambient_trace() -> None:
    """An in-process event carries no traceparent and must not start a new root trace.

    ``propagate.extract`` on an empty carrier yields an empty context, not the ambient one, so
    observing an unstamped event through it would detach the event into its own trace at exactly
    the point the caller's span is the answer.
    """

    assert mosfet.telemetry.configure() is True
    event = hsm.Event[None](name="bot.test.in_process")
    assert event.metadata == {}

    with span.operation(
        "bot.test.parent",
        scope="bot.telemetry.test",
        component="telemetry.test",
        stage="parent",
    ) as parent:
        with span.operation(
            "bot.test.observed",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="observed",
            context=mosfet.telemetry.event_context(event),
        ) as observed:
            assert observed.get_span_context().trace_id == parent.get_span_context().trace_id


def test_operation_records_cancellation_as_its_own_outcome() -> None:
    assert mosfet.telemetry.configure() is True
    with pytest.raises(asyncio.CancelledError):
        with span.operation(
            "bot.test.operation",
            scope="bot.telemetry.test",
            component="telemetry.test",
            stage="decode",
        ):
            raise asyncio.CancelledError

    record = _spans()[0]
    assert _attributes(record)["bot.outcome"] == "failed"
    assert _attributes(record)["bot.failure.kind"] == "cancelled"
