from mosfet import behavior
from mosfet.behavior import runtime

import ast
import asyncio
import collections.abc
import dataclasses
import json
import inspect
import pathlib
import threading
import time
import typing
import weakref

import hsm
import mosfet
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context

from tests.bot.behavior.support import (
    greeting_behavior_source,
    strict_greeting_behavior_source,
)
from tests.hsm_instance_state import start_ability_tree


def _behavior_idle_state(model_name: str, *, guarded: bool = True) -> str:
    suffix = "/guard_ready" if guarded else ""
    return f"/{model_name}BehaviorLifecycle/attached/behavior/idle{suffix}"


_PEER_REQUEST = hsm.Event[object](
    name="bot.behavior.test.peer.request",
    schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)
_PEER_REPLY = hsm.Event[object](
    name="bot.behavior.test.peer.reply",
    schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)


def _reply_to_behavior_source(
    ctx: hsm.Context,
    instance: hsm.Instance,
    event: hsm.Event[object],
) -> None:
    """Device/ability style: reply to the source of the inbound request."""

    reply = dataclasses.replace(
        _PEER_REPLY.with_data(event.data),
        id=event.id or None,
        source=hsm.id(instance),
        target=event.source,
        metadata=dict(event.metadata),
    )
    _ = hsm.dispatch_to(ctx, reply, event.source)


class _PeerMachine(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "PeerMachine",
        hsm.initial(hsm.target("/PeerMachine/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(_PEER_REQUEST),
                hsm.effect(_reply_to_behavior_source),
            ),
        ),
    )


def initial_effect_failure_recovery_behavior_source() -> str:
    return """
input_event = hsm.event(
    name = "bot.behavior.initial_effect_recovery.input",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
output_event = hsm.event(
    name = "bot.behavior.initial_effect_recovery.output",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
reset_event = hsm.event(
    name = "bot.behavior.initial_effect_recovery.reset",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

def emit_reset(event):
    hsm.dispatch(reset_event, {"text": "reset"})

def emit_output(event):
    hsm.dispatch(output_event, event["data"])

def emit_invalid_output(event):
    hsm.dispatch(output_event, "not an object payload")

behavior = hsm.define(
    "InitialEffectRecovery",
    hsm.initial(
        hsm.effect("emit_reset"),
        hsm.target("/InitialEffectRecovery/idle"),
    ),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(reset_event),
            hsm.effect("emit_output"),
        ),
        hsm.transition(
            hsm.on(input_event),
            hsm.target("/InitialEffectRecovery/failing"),
        ),
    ),
    hsm.state(
        "failing",
        hsm.entry("emit_invalid_output"),
    ),
)
"""


def test_compiled_behavior_returns_starlark_effect_output_through_dispatch_waiter() -> None:
    async def run() -> tuple[dict[str, object], str]:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)

        output = await dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"})

        return typing.cast(dict[str, object], output), compiled.state()

    output, state = asyncio.run(run())

    assert output == {"text": "HELLO"}
    assert state == _behavior_idle_state("AnswerGreeting")


def test_compiled_behavior_attachment_waits_for_callback_warmup_before_starting_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, str]:
        warm_started = asyncio.Event()
        release_warmup = asyncio.Event()

        async def controlled_warm(callback_runtime: runtime.CallbackRuntime) -> None:
            del callback_runtime
            warm_started.set()
            await release_warmup.wait()

        monkeypatch.setattr(runtime.CallbackRuntime, "warm", controlled_warm)
        compiled = behavior.build(greeting_behavior_source())
        evaluator = object.__getattribute__(compiled, "_guard_evaluator")
        start = asyncio.create_task(start_ability_tree(None, compiled))

        await asyncio.wait_for(warm_started.wait(), timeout=1.0)
        assert evaluator.state() == ""

        release_warmup.set()
        await asyncio.wait_for(start, timeout=1.0)
        return compiled.state(), evaluator.state()

    state, evaluator_state = asyncio.run(run())

    assert state == _behavior_idle_state("AnswerGreeting")
    assert evaluator_state == "/BehaviorGuardEvaluator/idle"


def test_compiled_behavior_dispatches_output_when_used_as_modeled_event(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> tuple[list[object], list[str], str]:
        compiled = behavior.build(greeting_behavior_source())
        original_dispatch = compiled.dispatch
        outputs: list[object] = []
        sources: list[str] = []

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                outputs.append(event.data)
                sources.append(event.source)
            return original_dispatch(ctx, event)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        await start_ability_tree(None, compiled)

        _ = await compiled.apply({"text": "hello"})

        return outputs, sources, compiled.state()

    outputs, sources, state = asyncio.run(run())

    assert outputs == [{"text": "HELLO"}]
    assert len(sources) == 1 and sources[0]  # live behavior id stamped as source
    assert state == _behavior_idle_state("AnswerGreeting")


def test_compiled_behavior_output_event_source_is_behavior_id() -> None:
    async def run() -> tuple[str, str]:
        compiled = behavior.build(greeting_behavior_source())
        recorded: list[str] = []
        original = compiled.dispatch

        def capture(ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                recorded.append(event.source)
            return original(ctx, event)

        compiled.dispatch = capture  # type: ignore[method-assign]
        await start_ability_tree(None, compiled)
        _ = await dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"})
        return recorded[0], hsm.id(compiled)

    source, behavior_id = asyncio.run(run())
    assert source == behavior_id


def test_compiled_behavior_runs_starlark_activity_without_async_starlark() -> None:
    async def run() -> tuple[dict[str, object], str]:
        compiled = behavior.build(
            """
input_event = hsm.event(
    name = "bot.behavior.answer_greeting.input",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)
output_event = hsm.event(
    name = "bot.behavior.answer_greeting.output",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

def emit_uppercase(event):
    hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})

behavior = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.target("/AnswerGreeting/responding"),
        ),
    ),
    hsm.state(
        "responding",
        hsm.activity("emit_uppercase"),
        hsm.transition(
            hsm.on(output_event),
            hsm.target("/AnswerGreeting/idle"),
        ),
    ),
)
"""
        )
        await start_ability_tree(None, compiled)

        output = await dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"})

        return typing.cast(dict[str, object], output), compiled.state()

    output, state = asyncio.run(run())

    assert output == {"text": "HELLO"}
    assert state == _behavior_idle_state("AnswerGreeting", guarded=False)


def test_compiled_behavior_generated_failure_recovery_preserves_root_initial_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, list[object], str]:
        compiled = behavior.build(initial_effect_failure_recovery_behavior_source())
        original_dispatch = compiled.dispatch
        outputs: list[object] = []
        reset_observed = asyncio.Event()

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                outputs.append(event.data)
                if event.data == {"text": "reset"}:
                    reset_observed.set()
            return original_dispatch(ctx, event)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        await start_ability_tree(None, compiled)
        outputs.clear()
        reset_observed.clear()

        with pytest.raises(RuntimeError, match="event schema") as error:
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"text": "go"}), timeout=1.0)
        await asyncio.wait_for(reset_observed.wait(), timeout=1.0)
        return str(error.value), outputs, compiled.state()

    message, outputs, state = asyncio.run(run())

    assert "bot.behavior.initial_effect_recovery.output payload does not match its event schema." in message
    assert outputs == [{"text": "reset"}]
    assert state == _behavior_idle_state("InitialEffectRecovery", guarded=False)


def test_compiled_behavior_rejects_invalid_input_before_starlark_runs() -> None:
    async def run() -> str:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="input schema") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"wrong": "hello"}), timeout=1.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "AnswerGreeting input does not match its input schema." in message


def test_compiled_behavior_enforces_additional_input_properties_before_starlark_runs() -> None:
    async def run() -> str:
        compiled = behavior.build(strict_greeting_behavior_source())
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="input schema") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello", "unexpected": True}),
                timeout=1.0,
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "AnswerGreeting input does not match its input schema." in message


def test_compiled_behavior_rejects_invalid_output_before_public_dispatch() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace(
            '{"text": event["data"]["text"].upper()}',
            '"not an object payload"',
        )
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="event schema") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "bot.behavior.answer_greeting.output payload does not match its event schema." in message


def test_compiled_behavior_rejects_additional_output_properties_before_public_dispatch() -> None:
    async def run() -> str:
        source = strict_greeting_behavior_source().replace(
            '{"text": event["data"]["text"].upper()}',
            '{"text": event["data"]["text"].upper(), "unexpected": True}',
        )
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="event schema") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "bot.behavior.answer_greeting.output payload does not match its event schema." in message


def test_compiled_behavior_rejects_guard_dispatch() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace(
            'return "text" in event["data"]',
            'hsm.dispatch(output_event, {"text": "NO"})\n    return True',
        )
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="guards cannot dispatch") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "guards cannot dispatch events." in message


def test_compiled_behavior_rejects_non_boolean_guard_result() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace('return "text" in event["data"]', 'return "yes"')
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="must return a bool") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "behavior guard 'has_text' must return a bool." in message


def test_compiled_behavior_rejected_guard_settles_without_leaving_source_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[object], str]:
        source = greeting_behavior_source().replace('return "text" in event["data"]', "return False", 1)
        compiled = behavior.build(source)
        outputs: list[object] = []
        original_dispatch = compiled.dispatch

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                outputs.append(event.data)
            return original_dispatch(ctx, event)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        await start_ability_tree(None, compiled)
        await compiled.apply({"text": "hello"})
        return outputs, compiled.state()

    outputs, state = asyncio.run(run())

    assert outputs == []
    assert state == _behavior_idle_state("AnswerGreeting")


def test_compiled_behavior_slow_guard_does_not_stall_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)
        guard_started = threading.Event()
        release_guard = threading.Event()
        original_process = vars(runtime)["_run_callback_process"]

        def delayed_guard(request: dict[str, object], *, deadline: float) -> object:
            if request.get("callback") == "has_text":
                guard_started.set()
                if not release_guard.wait(timeout=1.0):
                    raise AssertionError("guard release was not delivered")
            return original_process(request, deadline=deadline)

        monkeypatch.setattr(runtime, "_run_callback_process", delayed_guard)

        async def invoke() -> None:
            await compiled.apply({"text": "hello"})

        apply_task = asyncio.create_task(invoke())
        assert await asyncio.to_thread(guard_started.wait, 1.0)
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
        release_guard.set()
        await asyncio.wait_for(apply_task, timeout=2.0)

    asyncio.run(run())


def test_direct_hsm_dispatch_remains_delivery_only_during_guard_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[bool, dict[str, object]]:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)
        release = asyncio.Event()
        output: asyncio.Future[dict[str, object]] = asyncio.get_running_loop().create_future()
        original_dispatch = compiled.dispatch
        original_evaluate = runtime.CallbackRuntime.evaluate_guard

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name and not output.done():
                output.set_result(typing.cast(dict[str, object], event.data))
            return original_dispatch(ctx, event)

        async def delayed_guard(
            self: runtime.CallbackRuntime,
            *,
            callback: str,
            event: hsm.Event[object],
            behavior_id: str,
        ) -> bool:
            _ = await release.wait()
            return await original_evaluate(self, callback=callback, event=event, behavior_id=behavior_id)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        monkeypatch.setattr(runtime.CallbackRuntime, "evaluate_guard", delayed_guard)
        await asyncio.wait_for(
            hsm.dispatch(
                hsm.Context(),
                compiled,
                compiled.input_event.with_data_and_id({"text": "hello"}, "direct-operation"),
            ),
            timeout=0.1,
        )
        returned_before_outcome = not output.done()
        _ = release.set()
        return returned_before_outcome, await asyncio.wait_for(output, timeout=2.0)

    returned_before_outcome, output = asyncio.run(run())

    assert returned_before_outcome
    assert output == {"text": "HELLO"}


def test_compiled_behavior_unmatched_valid_input_settles_apply() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace("hsm.on(input_event)", "hsm.on(output_event)", 1)
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)
        await asyncio.wait_for(compiled.apply({"text": "hello"}), timeout=1.0)
        return compiled.state()

    assert asyncio.run(run()) == _behavior_idle_state("AnswerGreeting")


@pytest.mark.parametrize("delayed_boundary", ["start", "send", "join"])
def test_callback_deadline_includes_process_setup_ipc_and_teardown(
    monkeypatch: pytest.MonkeyPatch,
    delayed_boundary: str,
) -> None:
    delay_seconds = 0.12
    budget_seconds = 0.03
    response = json.dumps(
        {
            "ok": True,
            "result": {"kind": "bool", "value": True},
            "dispatches": [],
            "declared_events": [],
        },
        separators=(",", ":"),
    ).encode("utf-8")

    class FakeConnection:
        def send_bytes(self, buffer: bytes) -> None:
            del buffer
            if delayed_boundary == "send":
                _ = threading.Event().wait(timeout=delay_seconds)

        def poll(self, timeout: float = 0.0) -> bool:
            del timeout
            return True

        def recv_bytes(self, maxlength: int | None = None) -> bytes:
            del maxlength
            return response

        def close(self) -> None:
            return

    class FakeProcess:
        alive: bool = True

        def start(self) -> None:
            if delayed_boundary == "start":
                _ = threading.Event().wait(timeout=delay_seconds)

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            return

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            del timeout
            if delayed_boundary == "join":
                _ = threading.Event().wait(timeout=delay_seconds)
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

    monkeypatch.setattr(runtime, "CALLBACK_EVALUATION_SECONDS", budget_seconds)
    monkeypatch.setattr(
        runtime,
        "_evaluation_context",
        lambda: FakeContext(),
    )
    request: dict[str, object] = {
        "program": greeting_behavior_source(),
        "callback": "has_text",
        "event": {"name": "test", "data": {"text": "hello"}},
        "behavior_id": "behavior",
        "dispatch_allowed": False,
    }

    started = time.monotonic()
    with pytest.raises(runtime.CallbackError, match="budget"):
        worker = typing.cast(
            collections.abc.Callable[[dict[str, object]], object],
            vars(runtime)["_run_callback_worker"],
        )
        _ = worker(request)
    elapsed = time.monotonic() - started
    assert elapsed < delay_seconds


def test_compiled_behavior_guard_timeout_settles_as_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> str:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)

        async def fail_guard(
            self: runtime.CallbackRuntime,
            *,
            callback: str,
            event: hsm.Event[object],
            behavior_id: str,
        ) -> bool:
            del self, callback, event, behavior_id
            raise runtime.CallbackError("Starlark guard evaluation exceeded its budget.")

        monkeypatch.setattr(runtime.CallbackRuntime, "evaluate_guard", fail_guard)
        await compiled.apply({"text": "hello"})
        return compiled.state()

    assert asyncio.run(run()) == _behavior_idle_state("AnswerGreeting")


def test_compiled_behavior_cancelled_apply_does_not_consume_next_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> dict[str, object]:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)
        release = asyncio.Event()
        guard_started = asyncio.Event()
        original_evaluate = runtime.CallbackRuntime.evaluate_guard

        async def delayed_guard(
            self: runtime.CallbackRuntime,
            *,
            callback: str,
            event: hsm.Event[object],
            behavior_id: str,
        ) -> bool:
            guard_started.set()
            _ = await release.wait()
            return await original_evaluate(self, callback=callback, event=event, behavior_id=behavior_id)

        monkeypatch.setattr(runtime.CallbackRuntime, "evaluate_guard", delayed_guard)

        async def invoke() -> None:
            await compiled.apply({"text": "first"})

        cancelled = asyncio.create_task(invoke())
        await asyncio.wait_for(guard_started.wait(), timeout=1.0)
        _ = cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        _ = release.set()
        output = await dispatch_ability_for_test(compiled, hsm.Context(), {"text": "second"})
        return typing.cast(dict[str, object], output)

    assert asyncio.run(run()) == {"text": "SECOND"}


def test_cancelled_apply_does_not_cancel_effects_after_dispatch_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> dict[str, object]:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)
        effect_started = threading.Event()
        release_effect = threading.Event()
        output: asyncio.Future[dict[str, object]] = asyncio.get_running_loop().create_future()
        original_worker = vars(runtime)["_run_callback_worker"]
        original_dispatch = compiled.dispatch

        def controlled_worker(request: dict[str, object]) -> object:
            if request.get("callback") == "emit_uppercase":
                effect_started.set()
                if not release_effect.wait(timeout=1.0):
                    raise AssertionError("effect release was not delivered")
            return original_worker(request)

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name and not output.done():
                output.set_result(typing.cast(dict[str, object], event.data))
            return original_dispatch(ctx, event)

        monkeypatch.setattr(runtime, "_run_callback_worker", controlled_worker)
        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)

        async def invoke() -> None:
            await compiled.apply({"text": "committed"})

        apply_task = asyncio.create_task(invoke())
        assert await asyncio.to_thread(effect_started.wait, 1.0)

        apply_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await apply_task
        release_effect.set()
        return await asyncio.wait_for(output, timeout=2.0)

    assert asyncio.run(run()) == {"text": "COMMITTED"}


def test_behavior_apply_uses_typed_operation_topology_not_ability_waiter_registries() -> None:
    behavior_module = __import__("mosfet.behavior.behavior", fromlist=["Behavior"])
    apply_source = inspect.getsource(vars(behavior_module)["Behavior"].apply)

    assert "terminal_waiter" not in apply_source
    assert "_terminal_waiters" not in pathlib.Path(behavior_module.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize(("second_admitted", "expected"), [(True, "SECOND"), (False, "FALLBACK")])
def test_compiled_behavior_preserves_guard_candidate_order_and_unguarded_fallback(
    second_admitted: bool,
    expected: str,
) -> None:
    source = (
        greeting_behavior_source()
        .replace(
            'def has_text(event):\n    return "text" in event["data"]',
            """def reject_first(event):
    return False

def reject_second(event):
    return SECOND_ADMITTED

def emit_first(event):
    hsm.dispatch(output_event, {"text": "FIRST"})

def emit_second(event):
    hsm.dispatch(output_event, {"text": "SECOND"})

def emit_fallback(event):
    hsm.dispatch(output_event, {"text": "FALLBACK"})""",
            1,
        )
        .replace(
            """hsm.transition(
            hsm.on(input_event),
            hsm.guard("has_text"),
            hsm.effect("emit_uppercase"),
        ),""",
            """hsm.transition(
            hsm.on(input_event),
            hsm.guard("reject_first"),
            hsm.effect("emit_first"),
        ),
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("reject_second"),
            hsm.effect("emit_second"),
        ),
        hsm.transition(
            hsm.on(input_event),
            hsm.effect("emit_fallback"),
        ),""",
            1,
        )
        .replace("SECOND_ADMITTED", "True" if second_admitted else "False")
    )

    async def run() -> dict[str, object]:
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)
        return typing.cast(
            dict[str, object],
            await dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}),
        )

    assert asyncio.run(run()) == {"text": expected}


def test_compiled_behavior_ignores_stale_and_duplicate_guard_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[list[object], str]:
        compiled = behavior.build(greeting_behavior_source())
        outputs: list[object] = []
        original_dispatch = compiled.dispatch

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                outputs.append(event.data)
            return original_dispatch(ctx, event)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        await start_ability_tree(None, compiled)
        stale = dataclasses.replace(
            hsm.Event[object](name="bot.behavior.guard.accepted").with_data({"transition_token": "stale"}),
            id="stale-operation",
            source="stale-evaluator",
            target=hsm.id(compiled),
        )
        await hsm.dispatch(hsm.Context(), compiled, stale)
        await hsm.dispatch(hsm.Context(), compiled, stale)
        return outputs, compiled.state()

    outputs, state = asyncio.run(run())

    assert outputs == []
    assert state == _behavior_idle_state("AnswerGreeting")


def test_compiled_behavior_guard_rejection_preserves_entry_exit_and_activity_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        greeting_behavior_source()
        .replace(
            'return "text" in event["data"]',
            "return False",
            1,
        )
        .replace(
            "def emit_uppercase(event):",
            """def mark_entry(event):
    hsm.dispatch(output_event, {"text": "ENTRY"})

def mark_exit(event):
    hsm.dispatch(output_event, {"text": "EXIT"})

def mark_activity(event):
    hsm.dispatch(output_event, {"text": "ACTIVITY"})

def emit_uppercase(event):""",
            1,
        )
        .replace(
            'hsm.state(\n        "idle",',
            'hsm.state(\n        "idle",\n        hsm.entry("mark_entry"),\n        hsm.exit("mark_exit"),\n        hsm.activity("mark_activity"),',
            1,
        )
    )

    async def run() -> list[object]:
        compiled = behavior.build(source)
        outputs: list[object] = []
        activity_observed = asyncio.Event()
        original_dispatch = compiled.dispatch

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                outputs.append(event.data)
                if event.data == {"text": "ACTIVITY"}:
                    activity_observed.set()
            return original_dispatch(ctx, event)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        await start_ability_tree(None, compiled)
        await asyncio.wait_for(activity_observed.wait(), timeout=1.0)
        outputs.clear()
        await compiled.apply({"text": "hello"})
        return outputs

    assert asyncio.run(run()) == []


def test_compiled_behavior_concurrent_apply_correlations_settle_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> list[object]:
        compiled = behavior.build(greeting_behavior_source())
        outputs: list[object] = []
        original_dispatch = compiled.dispatch

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                outputs.append(event.data)
            return original_dispatch(ctx, event)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        await start_ability_tree(None, compiled)
        _ = await asyncio.gather(
            compiled.apply({"text": "first"}),
            compiled.apply({"text": "second"}),
        )
        return outputs

    assert asyncio.run(run()) == [{"text": "FIRST"}, {"text": "SECOND"}]


def test_guard_callbacks_and_telemetry_are_not_reachable_from_hsm_guards() -> None:
    runtime_path = pathlib.Path(runtime.__file__)
    runtime_source = runtime_path.read_text(encoding="utf-8")
    tree = ast.parse(runtime_source)
    callback_runtime = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CallbackRuntime"
    )

    assert not any(isinstance(node, ast.FunctionDef) and node.name == "guard" for node in callback_runtime.body)
    assert "bot.behavior.callback" not in runtime_source


def test_starlark_effect_evaluation_is_an_explicit_async_boundary() -> None:
    assert inspect.iscoroutinefunction(runtime.CallbackRuntime.effect)


def test_slow_starlark_effect_keeps_event_loop_heartbeat_live(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> dict[str, object]:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)
        effect_started = threading.Event()
        release_effect = threading.Event()
        original_worker = vars(runtime)["_run_callback_worker"]

        def controlled_worker(request: dict[str, object]) -> object:
            if request.get("callback") == "emit_uppercase":
                effect_started.set()
                if not release_effect.wait(timeout=1.0):
                    raise AssertionError("effect release was not delivered")
            return original_worker(request)

        monkeypatch.setattr(runtime, "_run_callback_worker", controlled_worker)

        async def invoke() -> None:
            await compiled.apply({"text": "hello"})

        apply_task = asyncio.create_task(invoke())
        assert await asyncio.to_thread(effect_started.wait, 1.0)

        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
        release_effect.set()
        await asyncio.wait_for(apply_task, timeout=2.0)
        return {"text": "HELLO"}

    assert asyncio.run(run()) == {"text": "HELLO"}


def test_guard_operational_events_publish_complete_pydantic_contracts() -> None:
    behavior_module = __import__("mosfet.behavior.behavior", fromlist=["_GuardEvaluationRequestedData"])
    request_schema = vars(behavior_module)["_GuardEvaluationRequestedData"].model_json_schema()
    result_schema = vars(behavior_module)["_GuardEvaluationResultData"].model_json_schema()

    assert set(request_schema["required"]) == {"callback", "candidate_id", "original_event"}
    assert set(result_schema["required"]) >= {"request"}


def test_guard_operational_result_rejects_malformed_payload() -> None:
    behavior_module = __import__("mosfet.behavior.behavior", fromlist=["_GuardEvaluationResultData"])
    result_data = vars(behavior_module)["_GuardEvaluationResultData"]

    with pytest.raises(ValueError):
        _ = result_data.model_validate({"request": {"callback": "has_text"}})


def test_guard_lowering_uses_explicit_candidate_states_without_token_guards() -> None:
    behavior_module = __import__("mosfet.behavior.behavior", fromlist=["Behavior"])
    compiled = behavior.build(greeting_behavior_source())
    model = typing.cast(hsm.Model, compiled.submodel)

    assert "guard_outcome_matches" not in vars(behavior_module)["Behavior"].__dict__
    assert any("guard_candidate" in name for name in model.members)


def test_compiled_behavior_rejects_undeclared_event_dispatch() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace(
            'hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})',
            'hsm.dispatch("not.a.declared.event", {"text": "x"})',
        )
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="undeclared event") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "behavior callback cannot dispatch undeclared event 'not.a.declared.event'." in message


def test_compiled_behavior_rejects_undeclared_event_spec_dispatch() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace(
            'hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})',
            'hsm.dispatch({"name": "bot.behavior.answer_greeting.rogue", "schema": {"type": "object"}}, {"attack": True})',
            1,
        )
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="undeclared event") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=2.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "behavior callback cannot dispatch undeclared event 'bot.behavior.answer_greeting.rogue'." in message


def test_compiled_behavior_bounds_recursive_callback_evaluation() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace(
            'def emit_uppercase(event):\n    hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})',
            "def emit_uppercase(event):\n    return emit_uppercase(event)",
            1,
        )
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="recurs|budget") as error:
            _ = await asyncio.wait_for(
                dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=2.0
            )
        return str(error.value)

    message = asyncio.run(run())

    assert "recurs" in message.lower() or "budget" in message.lower()


def test_compiled_behavior_dispatches_with_source_id_so_peer_can_reply_for_processing() -> None:
    """Behaviors stamp source=hsm.id(behavior); peers reply to that id; behavior processes the reply."""

    behavior_source = f"""
input_event = hsm.event(
    name = "bot.behavior.answer_with_peer.input",
    schema = {{
        "type": "object",
        "properties": {{"text": {{"type": "string"}}, "peer": {{"type": "string"}}}},
        "required": ["text", "peer"],
    }},
)
output_event = hsm.event(
    name = "bot.behavior.answer_with_peer.output",
    schema = {{
        "type": "object",
        "properties": {{"text": {{"type": "string"}}}},
        "required": ["text"],
    }},
)
request_event = hsm.event(
    name = "{_PEER_REQUEST.name}",
    schema = {{
        "type": "object",
        "properties": {{"text": {{"type": "string"}}}},
        "required": ["text"],
    }},
)
reply_event = hsm.event(
    name = "{_PEER_REPLY.name}",
    schema = {{
        "type": "object",
        "properties": {{"text": {{"type": "string"}}}},
        "required": ["text"],
    }},
)

def request_peer(event):
    hsm.dispatch(request_event, {{"text": event["data"]["text"]}}, event["data"]["peer"])

def emit_reply(event):
    hsm.dispatch(output_event, {{"text": event["data"]["text"].upper()}})

behavior = hsm.define(
    "AnswerWithPeer",
    hsm.initial(hsm.target("/AnswerWithPeer/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.target("/AnswerWithPeer/waiting"),
        ),
    ),
    hsm.state(
        "waiting",
        hsm.activity("request_peer"),
        hsm.transition(
            hsm.on(reply_event),
            hsm.effect("emit_reply"),
            hsm.target("/AnswerWithPeer/idle"),
        ),
    ),
)
"""

    async def run() -> tuple[dict[str, object], str, str, str]:
        ctx = shared_hsm_context()
        peer = _PeerMachine()
        _ = await mosfet.started(ctx, peer, typing.cast(hsm.Model, peer.model))
        peer_id = hsm.id(peer)

        compiled = behavior.build(behavior_source)
        await start_ability_tree(ctx, compiled)
        # Ensure the live behavior is addressable for reply routing.
        instances = ctx.value(hsm.Keys.Instances)
        assert isinstance(instances, weakref.WeakValueDictionary)
        instances[hsm.id(compiled)] = compiled

        output = await dispatch_ability_for_test(
            compiled,
            ctx,
            {"text": "hello", "peer": peer_id},
        )
        return typing.cast(dict[str, object], output), compiled.state(), hsm.id(compiled), peer_id

    output, state, behavior_id, peer_id = asyncio.run(run())

    assert output == {"text": "HELLO"}
    assert state == _behavior_idle_state("AnswerWithPeer", guarded=False)
    assert behavior_id
    assert peer_id
    assert behavior_id != peer_id
