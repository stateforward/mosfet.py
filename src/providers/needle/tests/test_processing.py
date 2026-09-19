"""Processor contract tests with a stubbed Needle engine (no native library, no weights)."""

from __future__ import annotations

import asyncio
import collections.abc
import typing

import hsm
import pytest

from mosfet.abilities import cognition, processing
from mosfet.abilities.cognition import intuition
from mosfet.devices import phone
from mosfet.providers import needle


class _StubEngine:
    """Records each bound turn and answers with a fixed Needle envelope."""

    def __init__(self, response: dict[str, typing.Any]) -> None:
        self.response: collections.abc.Mapping[str, object] = response
        self.turns: list[dict[str, object]] = []

    def complete(
        self,
        *,
        system: str,
        tools: collections.abc.Sequence[collections.abc.Mapping[str, object]],
        text: str,
    ) -> collections.abc.Mapping[str, object]:
        self.turns.append({"system": system, "tools": [dict(tool) for tool in tools], "text": text})
        return self.response


class _FailingEngine:
    def complete(
        self,
        *,
        system: str,
        tools: collections.abc.Sequence[collections.abc.Mapping[str, object]],
        text: str,
    ) -> collections.abc.Mapping[str, object]:
        del system, tools, text
        raise needle.EngineError("needle_init failed")


def _notification() -> hsm.Event[phone.NotificationData]:
    return hsm.Event[phone.NotificationData](
        name="phone.notification",
        schema=phone.NotificationData,
        data=phone.NotificationData.model_validate(
            {
                "id": "message-1",
                "name": "phone.sms.text",
                "data": {"id": "message-1", "sender": "+15555550101", "text": "Are we still on for 10:15?"},
            }
        ),
    )


def _offered() -> tuple[hsm.Event[typing.Any], ...]:
    return (phone.SendTextMessageEvent, cognition.IgnoreEvent)


def _run(engine: needle.Engine, *, actor_events: dict[str, tuple[str, ...]] | None = None) -> object:
    turn = processing.InputData(
        input=_notification(),
        schemas=_offered(),
        actors={},
        actor_events=actor_events or {},
        authority=None,
        instructions="You are the bot's reflexes.",
    )
    return asyncio.run(needle.Processor(engine=engine).process(turn))


def test_offered_events_become_identifier_named_tools_with_payload_schemas() -> None:
    engine = _StubEngine({"function_calls": [], "confidence": 0.0})
    _ = _run(engine)

    (turn,) = engine.turns
    tools = typing.cast(list[dict[str, dict[str, object]]], turn["tools"])
    assert [tool["name"] for tool in tools] == ["phone_send_text_message", "bot_ability_cognition_ignore"]
    send = tools[0]
    assert set(typing.cast(list[str], send["parameters"]["required"])) == {"to", "text"}
    assert send["description"]
    assert turn["system"] == "You are the bot's reflexes."
    assert "Are we still on for 10:15?" in typing.cast(str, turn["text"])


def test_call_maps_back_to_canonical_event_with_authored_payload_and_confidence() -> None:
    engine = _StubEngine(
        {
            "function_calls": [
                {"name": "phone_send_text_message", "arguments": {"to": "+15555550101", "text": "Yes, see you then."}}
            ],
            "confidence": 0.82,
        }
    )
    selections = _run(engine, actor_events={"phone.send_text_message": ("phone",)})

    assert selections == (
        processing.SelectedEvent(
            event="phone.send_text_message",
            target="phone",
            data={"to": "+15555550101", "text": "Yes, see you then."},
            reason=None,
            confidence=82,
        ),
    )


def test_no_calls_is_explicit_unhandled_not_handled_empty() -> None:
    engine = _StubEngine(
        {
            "function_calls": [],
            "suppressed_calls": [{"name": "phone_send_text_message", "arguments": {}}],
            "confidence": 0.05,
        }
    )
    result = _run(engine)

    assert isinstance(result, intuition.OutputData)
    assert result.result is None


def test_ungrounded_call_is_withheld() -> None:
    engine = _StubEngine(
        {
            "function_calls": [
                {"name": "phone_send_text_message", "arguments": {"to": "+15555550199", "text": "hi"}},
                {"name": "bot_ability_cognition_ignore", "arguments": {}},
            ],
            "confidence": 0.6,
            "validation": {"ungrounded": ["phone_send_text_message.to"]},
        }
    )
    selections = typing.cast(tuple[processing.SelectedEvent, ...], _run(engine))

    assert [item.event for item in selections] == ["bot.ability.cognition.ignore"]
    assert selections[0].data is None


def test_unoffered_tool_is_a_typed_error() -> None:
    engine = _StubEngine({"function_calls": [{"name": "phone_hang_up", "arguments": {}}], "confidence": 0.9})
    with pytest.raises(needle.ProcessingError, match="unoffered"):
        _ = _run(engine)


def test_engine_failure_surfaces_as_processing_error() -> None:
    with pytest.raises(needle.ProcessingError, match="needle_init failed"):
        _ = _run(_FailingEngine())


def test_nothing_offered_skips_the_engine() -> None:
    engine = _StubEngine({"function_calls": [], "confidence": 0.0})
    turn = processing.InputData(input=_notification(), schemas=(), actors={}, authority=None)
    assert asyncio.run(needle.Processor(engine=engine).process(turn)) == ()
    assert engine.turns == []
