"""Behavior events: the routine tick contract (``bot.behavior.tick``)."""

import datetime

import pydantic
import pytest

from mosfet import behavior

_SCHEDULED = datetime.datetime(2026, 10, 5, 12, 0, tzinfo=datetime.UTC)


def test_tick_event_carries_tick_data() -> None:
    tick = behavior.TickData(
        name="MorningBriefing",
        trigger="at",
        scheduled_at=_SCHEDULED,
        fired_at=_SCHEDULED + datetime.timedelta(milliseconds=12),
        output=({"event": "bot.speaking.say", "data": {"text": "Good morning."}},),
    )
    event = behavior.TickEvent.with_data(tick)
    assert event.name == "bot.behavior.tick"
    assert event.data == tick
    assert tick.model_dump(mode="json")["scheduled_at"] == "2026-10-05T12:00:00Z"


def test_tick_data_requires_timezone_aware_times() -> None:
    with pytest.raises(pydantic.ValidationError, match="timezone"):
        _ = behavior.TickData(
            name="MorningBriefing",
            trigger="every",
            scheduled_at=_SCHEDULED.replace(tzinfo=None),
            fired_at=_SCHEDULED,
        )


def test_tick_data_rejects_unknown_trigger_and_unnamed_selection() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = behavior.TickData.model_validate(
            {"name": "X", "trigger": "cron", "scheduled_at": _SCHEDULED, "fired_at": _SCHEDULED}
        )
    with pytest.raises(pydantic.ValidationError, match="must name a non-empty event"):
        _ = behavior.TickData(
            name="X", trigger="every", scheduled_at=_SCHEDULED, fired_at=_SCHEDULED, output=({"data": {}},)
        )


def test_tick_schema_documents_every_field_for_model_use() -> None:
    schema = behavior.TickData.model_json_schema()
    assert schema["examples"]
    properties = schema["properties"]
    assert set(properties) == {"name", "trigger", "scheduled_at", "fired_at", "output"}
    assert all(properties[name].get("description") for name in properties)
