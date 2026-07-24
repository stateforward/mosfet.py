from bot import behavior

import asyncio
import collections.abc
import dataclasses
import typing
import weakref

import hsm
import pytest

from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context

from tests.bot.behavior.support import (
    greeting_behavior_source,
    strict_greeting_behavior_source,
)
from tests.hsm_instance_state import start_ability_tree


def _behavior_idle_state(model_name: str) -> str:
    return f"/{model_name}BehaviorLifecycle/attached/behavior/idle"


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
    model: typing.ClassVar[hsm.Model | None] = hsm.define(
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
    assert state == _behavior_idle_state("AnswerGreeting")


def test_compiled_behavior_generated_failure_recovery_preserves_root_initial_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> tuple[str, list[object], str]:
        compiled = behavior.build(initial_effect_failure_recovery_behavior_source())
        original_dispatch = compiled.dispatch
        outputs: list[object] = []

        def recording_dispatch(
            ctx: hsm.Context,
            event: hsm.Event[typing.Any],
        ) -> collections.abc.Awaitable[None]:
            if event.name == compiled.output_event.name:
                outputs.append(event.data)
            return original_dispatch(ctx, event)

        monkeypatch.setattr(compiled, "dispatch", recording_dispatch)
        await start_ability_tree(None, compiled)
        outputs.clear()

        with pytest.raises(RuntimeError, match="event schema") as error:
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"text": "go"}), timeout=1.0)
        return str(error.value), outputs, compiled.state()

    message, outputs, state = asyncio.run(run())

    assert "bot.behavior.initial_effect_recovery.output payload does not match its event schema." in message
    assert outputs == [{"text": "reset"}]
    assert state == _behavior_idle_state("InitialEffectRecovery")


def test_compiled_behavior_rejects_invalid_input_before_starlark_runs() -> None:
    async def run() -> str:
        compiled = behavior.build(greeting_behavior_source())
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="input schema") as error:
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"wrong": "hello"}), timeout=1.0)
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
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0)
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
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0)
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
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0)
        return str(error.value)

    message = asyncio.run(run())

    assert "guards cannot dispatch events." in message


def test_compiled_behavior_rejects_non_boolean_guard_result() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace('return "text" in event["data"]', 'return "yes"')
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="must return a bool") as error:
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0)
        return str(error.value)

    message = asyncio.run(run())

    assert "behavior guard 'has_text' must return a bool." in message


def test_compiled_behavior_rejects_undeclared_event_dispatch() -> None:
    async def run() -> str:
        source = greeting_behavior_source().replace(
            'hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})',
            'hsm.dispatch("not.a.declared.event", {"text": "x"})',
        )
        compiled = behavior.build(source)
        await start_ability_tree(None, compiled)

        with pytest.raises(RuntimeError, match="undeclared event") as error:
            _ = await asyncio.wait_for(dispatch_ability_for_test(compiled, hsm.Context(), {"text": "hello"}), timeout=1.0)
        return str(error.value)

    message = asyncio.run(run())

    assert "behavior callback cannot dispatch undeclared event 'not.a.declared.event'." in message


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
        _ = await hsm.started(ctx, peer, typing.cast(hsm.Model, peer.model))
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
    assert state == _behavior_idle_state("AnswerWithPeer")
    assert behavior_id
    assert peer_id
    assert behavior_id != peer_id
