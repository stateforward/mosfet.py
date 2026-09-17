from mosfet import behavior
from mosfet.behavior import source as behavior_source
from mosfet.telemetry.configure import span_file, tracer_provider

import collections.abc
import concurrent.futures
import json
import pathlib
import subprocess
import sys
import time
import typing

import mosfet.telemetry
import pytest
from opentelemetry import _logs, trace

from tests.bot.behavior.support import greeting_behavior_source

SOURCE_WORKER_TIMEOUT_SECONDS = 5.0
PARALLEL_SOURCE_WORKERS = 4


def test_parse_source_uses_real_starlark_hsm_model_logic() -> None:
    source = """
def behavior_event(suffix):
    return hsm.event(
        name = "bot.behavior.answer_greeting." + suffix,
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

input_event = behavior_event("input")
output_event = behavior_event("output")

def emit_uppercase(event):
    hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})

behavior = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.effect("emit_uppercase"),
        ),
    ),
)
"""

    spec = behavior.parse_source(source)

    assert spec.model["kind"] == "define"
    assert spec.model["name"] == "AnswerGreeting"
    assert spec.input_event.name == "bot.behavior.answer_greeting.input"


def test_normalize_source_preserves_strings_while_rewriting_hsm_namespace() -> None:
    source = 'message = "hsm.define is documentation text"\nbehavior = hsm.define("M")\n'

    normalized = behavior.source.normalize_source(source)

    assert '"hsm.define is documentation text"' in normalized
    assert "hsm.define" not in normalized.splitlines()[1]
    assert "define" in normalized.splitlines()[1]


def test_parse_source_rejects_old_ability_step_orchestration() -> None:
    source = """
input_event = hsm.event(name = "bot.behavior.answer_greeting.input")
output_event = hsm.event(name = "bot.behavior.answer_greeting.output")
behavior = behavior_program(
    kind = "behavior_program",
    name = "AnswerGreeting",
    input_event = input_event,
    output_event = output_event,
    steps = [ability_step(name = "uppercase_reply", ability = "uppercase")],
)
"""

    with pytest.raises(behavior.SourceError, match="ability_step"):
        _ = behavior.parse_source(source)


def test_parse_source_rejects_unapproved_starlark_calls() -> None:
    with pytest.raises(behavior.SourceError, match="open"):
        _ = behavior.parse_source('behavior = open("unsafe.star")')


def test_parse_source_rejects_unsupported_schema_keywords() -> None:
    source = greeting_behavior_source().replace(
        '"properties": {"text": {"type": "string"}},',
        '"properties": {"text": {"type": "string", "format": "email"}},',
        1,
    )

    with pytest.raises(behavior.SourceError, match="unsupported JSON schema keyword"):
        _ = behavior.parse_source(source)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            'examples = [{"text": "hello"}],',
            'examples = [{"text": "hello"}],\n    source = "unexpected",',
        ),
    ],
)
def test_parse_source_rejects_extra_starlark_builder_fields(old: str, new: str) -> None:
    source = greeting_behavior_source().replace(old, new, 1)

    with pytest.raises(behavior.SourceError, match="Extra inputs|unexpected keyword"):
        _ = behavior.parse_source(source)


def test_parse_source_rejects_extra_hsm_builder_keywords() -> None:
    source = """
input_event = hsm.event(name = "bot.behavior.answer_greeting.input")
output_event = hsm.event(name = "bot.behavior.answer_greeting.output")
behavior = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state("idle", extra = "unexpected"),
)
"""

    with pytest.raises(behavior.SourceError, match="unexpected keyword"):
        _ = behavior.parse_source(source)


def test_parse_source_rejects_unknown_targets() -> None:
    source = greeting_behavior_source().replace(
        'hsm.target("/AnswerGreeting/idle")',
        'hsm.target("/AnswerGreeting/missing")',
        1,
    )

    with pytest.raises(behavior.SourceError, match="unknown states"):
        _ = behavior.parse_source(source)


def test_parse_source_rejects_duplicate_model_event_names() -> None:
    source = (
        greeting_behavior_source()
        .replace(
            "behavior = hsm.define(",
            """
duplicate_output = hsm.event(name = "bot.behavior.answer_greeting.output")

behavior = hsm.define(""",
        )
        .replace(
            "hsm.on(input_event),",
            "hsm.on(input_event, duplicate_output),",
            1,
        )
    )

    with pytest.raises(behavior.SourceError, match="declared more than once"):
        _ = behavior.parse_source(source)


def test_parse_source_rejects_string_event_references() -> None:
    source = greeting_behavior_source().replace(
        "hsm.on(input_event)",
        'hsm.on("bot.behavior.answer_greeting.input")',
        1,
    )

    with pytest.raises(behavior.SourceError, match="event spec"):
        _ = behavior.parse_source(source)


def test_behavior_program_spec_rejects_final_state_contents() -> None:
    event = {"name": "bot.behavior.answer_greeting.input", "schema": {"type": "object"}}

    with pytest.raises(ValueError, match="unexpected hsm element fields"):
        _ = behavior.Source.model_validate(
            {
                "kind": "behavior_program",
                "name": "AnswerGreeting",
                "input_event": event,
                "output_event": {"name": "bot.behavior.answer_greeting.output", "schema": {"type": "object"}},
                "model": {
                    "kind": "define",
                    "name": "AnswerGreeting",
                    "elements": (
                        {"kind": "initial", "elements": ({"kind": "target", "path": "/AnswerGreeting/done"},)},
                        {
                            "kind": "final",
                            "name": "done",
                            "elements": ({"kind": "transition", "elements": ()},),
                        },
                    ),
                },
            }
        )


def test_parse_source_rejects_raw_function_callbacks() -> None:
    source = greeting_behavior_source().replace('hsm.effect("emit_uppercase")', "hsm.effect(emit_uppercase)", 1)

    with pytest.raises(behavior.SourceError, match="convert|callback"):
        _ = behavior.parse_source(source)


def test_parse_source_rejects_top_level_recursive_evaluation() -> None:
    source = """
def recurse(value):
    return recurse(value)

recurse(0)
input_event = hsm.event(name = "bot.behavior.hostile.input")
output_event = hsm.event(name = "bot.behavior.hostile.output")
behavior = hsm.define(
    "Hostile",
    hsm.initial(hsm.target("/Hostile/idle")),
    hsm.state("idle"),
)
"""

    with pytest.raises(behavior.SourceError, match="recurs|budget"):
        _ = behavior.parse_source(source)


def test_parse_source_rejects_oversized_source_before_evaluation() -> None:
    with pytest.raises(behavior.SourceError, match="byte budget"):
        _ = behavior.parse_source("#" + ("x" * (64 * 1024)))


def _deadline_probe_source() -> behavior.Source:
    return behavior.Source.model_validate(
        {
            "kind": "behavior_program",
            "name": "DeadlineProbe",
            "input_event": {"name": "bot.behavior.deadline_probe.input", "schema": {"type": "object"}},
            "output_event": {"name": "bot.behavior.deadline_probe.output", "schema": {"type": "object"}},
            "model": {
                "kind": "define",
                "name": "DeadlineProbe",
                "elements": (
                    {
                        "kind": "initial",
                        "elements": ({"kind": "target", "path": "/DeadlineProbe/idle"},),
                    },
                    {"kind": "state", "name": "idle", "elements": ()},
                ),
            },
        }
    )


def _install_fake_isolated_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    delayed_boundary: str | None = None,
    delay_seconds: float = 0.0,
    budget_seconds: float | None = None,
    ready_seconds: float | None = None,
    result_ready: bool = True,
) -> None:
    response = json.dumps(
        {"ok": True, "spec": _deadline_probe_source().model_dump(mode="json", by_alias=True)},
        separators=(",", ":"),
    ).encode("utf-8")

    class FakeConnection:
        def send_bytes(self, buffer: bytes) -> None:
            del buffer
            if delayed_boundary == "send":
                time.sleep(delay_seconds)

        def wait_ready(self, *, timeout: float) -> None:
            if delayed_boundary != "ready":
                return
            if delay_seconds <= timeout:
                time.sleep(delay_seconds)
                return
            time.sleep(timeout)
            error = behavior.SourceError(f"isolated Starlark worker did not become ready within {timeout:g} seconds.")
            error.failure_kind = "timeout"
            raise error

        def poll(self, timeout: float = 0.0) -> bool:
            del timeout
            return result_ready

        def recv_bytes(self, maxlength: int | None = None) -> bytes:
            del maxlength
            return response

        def close(self) -> None:
            return

    class FakeProcess:
        alive: bool = True

        def start(self) -> None:
            if delayed_boundary == "start":
                time.sleep(delay_seconds)

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            return

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            del timeout
            if delayed_boundary == "join":
                time.sleep(delay_seconds)
            self.alive = False

    class FakeContext:
        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            del duplex
            return FakeConnection(), FakeConnection()

        def Process(
            self,
            *,
            target: collections.abc.Callable[..., object],
            args: tuple[object, ...],
        ) -> FakeProcess:
            del target, args
            return FakeProcess()

    if budget_seconds is not None:
        monkeypatch.setattr(behavior_source, "SOURCE_EVALUATION_SECONDS", budget_seconds)
    if ready_seconds is not None:
        monkeypatch.setattr(behavior_source, "SOURCE_WORKER_READY_SECONDS", ready_seconds)
    monkeypatch.setattr(behavior_source, "_evaluation_context", lambda: FakeContext())


@pytest.mark.parametrize("delayed_boundary", ["start", "ready", "join"])
def test_parse_source_evaluation_budget_excludes_worker_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    delayed_boundary: str,
) -> None:
    delay_seconds = 0.12
    budget_seconds = 0.03
    _install_fake_isolated_worker(
        monkeypatch,
        delayed_boundary=delayed_boundary,
        delay_seconds=delay_seconds,
        budget_seconds=budget_seconds,
    )

    spec = behavior.parse_source(greeting_behavior_source())

    assert spec.name == "DeadlineProbe"


def test_parse_source_evaluation_budget_covers_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delay_seconds = 0.12
    budget_seconds = 0.03
    _install_fake_isolated_worker(
        monkeypatch,
        delayed_boundary="send",
        delay_seconds=delay_seconds,
        budget_seconds=budget_seconds,
    )

    started = time.monotonic()
    with pytest.raises(behavior.SourceError, match="budget") as raised:
        _ = behavior.parse_source(greeting_behavior_source())
    elapsed = time.monotonic() - started
    time.sleep(delay_seconds + budget_seconds)

    assert elapsed >= budget_seconds
    assert raised.value.failure_kind == "timeout"


def test_parse_source_ready_budget_covers_worker_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delay_seconds = 0.12
    ready_seconds = 0.03
    _install_fake_isolated_worker(
        monkeypatch,
        delayed_boundary="ready",
        delay_seconds=delay_seconds,
        ready_seconds=ready_seconds,
    )

    started = time.monotonic()
    with pytest.raises(behavior.SourceError, match="ready") as raised:
        _ = behavior.parse_source(greeting_behavior_source())
    elapsed = time.monotonic() - started
    time.sleep(delay_seconds + ready_seconds)

    assert elapsed < delay_seconds
    assert raised.value.failure_kind == "timeout"


def test_parse_source_timeout_emits_evaluate_span(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    delay_seconds = 0.12
    budget_seconds = 0.03
    _install_fake_isolated_worker(
        monkeypatch,
        delayed_boundary="send",
        delay_seconds=delay_seconds,
        budget_seconds=budget_seconds,
        result_ready=False,
    )

    monkeypatch.chdir(tmp_path)
    mosfet.telemetry.reset()
    monkeypatch.setattr(_logs, "set_logger_provider", lambda _provider: None)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda _provider: None)
    assert mosfet.telemetry.configure() is True
    try:
        with pytest.raises(behavior.SourceError, match="budget"):
            _ = behavior.parse_source(greeting_behavior_source())
        provider = tracer_provider()
        if provider is not None:
            _ = provider.force_flush()
        path = span_file()
        assert path is not None and path.is_file()
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        evaluate = next(record for record in records if record["name"] == "bot.behavior.source.evaluate")
        attributes = typing.cast(dict[str, object], evaluate["attributes"])
        assert attributes["bot.component.name"] == "behavior.source"
        assert attributes["bot.stage"] == "evaluate"
        assert attributes["bot.outcome"] == "failed"
        assert attributes["bot.failure.kind"] == "timeout"
    finally:
        mosfet.telemetry.reset()


def test_source_worker_module_evaluates_in_a_clean_interpreter() -> None:
    completed = subprocess.run(
        (sys.executable, "-m", "mosfet.behavior.source"),
        input=greeting_behavior_source().encode("utf-8"),
        capture_output=True,
        check=False,
        timeout=SOURCE_WORKER_TIMEOUT_SECONDS,
    )

    ready, separator, rest = completed.stdout.partition(b"\n")
    payload = json.loads(rest.decode("utf-8"))
    assert completed.returncode == 0
    assert separator == b"\n"
    assert ready + separator == behavior_source.SOURCE_WORKER_READY
    assert payload["ok"] is True
    assert payload["spec"]["name"] == "AnswerGreeting"


def test_parse_source_isolated_workers_complete_under_parallel_spawn() -> None:
    source = greeting_behavior_source()

    def parse_one(index: int) -> str:
        del index
        return behavior.parse_source(source).name

    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_SOURCE_WORKERS) as pool:
        names = list(pool.map(parse_one, range(PARALLEL_SOURCE_WORKERS)))

    assert names == ["AnswerGreeting"] * PARALLEL_SOURCE_WORKERS
