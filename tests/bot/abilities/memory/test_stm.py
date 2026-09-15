"""StmEventMemory: bounded rolling register of admitted environment stimuli."""

from bot.abilities import memory
from bot.abilities.memory import stm

import dataclasses
import datetime
import pathlib
import typing
import uuid

import hsm
import pydantic
import pytest

_TIMELINE_START = datetime.datetime(2026, 7, 7, 12, 0, 0, tzinfo=datetime.timezone.utc)


class _ControlledClock:
    """Injectable virtual clock: the test sets ``now`` explicitly before each call site."""

    def __init__(self) -> None:
        self.now = _TIMELINE_START

    def __call__(self) -> datetime.datetime:
        return self.now


def _advance(clock: _ControlledClock, seconds: float) -> None:
    clock.now = clock.now + datetime.timedelta(seconds=seconds)


def _sound_event(
    *,
    name: str = "environment.sound",
    data: object | None = None,
    source: str = "device-a",
) -> hsm.Event[typing.Any]:
    event = hsm.Event[object](name=name, schema=object).with_data(
        {"kind": "knock", "source": source} if data is None else data
    )
    return event


def _notable_event(mark: str, *, name: str = "environment.sound") -> hsm.Event[typing.Any]:
    return _sound_event(name=name, data={"kind": "knock", "source": mark})


class _SensoryPayload(pydantic.BaseModel):
    media_type: str | None = None
    sample_rate_hz: int | None = None
    audio: bytes | None = None


def test_register_bounds_capacity_to_newest_entries() -> None:
    clock = _ControlledClock()
    register = memory.StmEventMemory(capacity=2, clock=clock)

    for mark in "one", "two", "three", "four":
        _ = register.record(_notable_event(mark))
        _advance(clock, 1.0)

    recent = register.recent(limit=16)
    assert [entry.payload["source"] for entry in recent] == ["four", "three"]


def test_register_recent_orders_newest_first_and_filters_by_name() -> None:
    clock = _ControlledClock()
    register = memory.StmEventMemory(clock=clock)

    _ = register.record(_notable_event("ring-1"))
    _advance(clock, 1.0)
    _ = register.record(_notable_event("knock-1", name="environment.sound"))
    _advance(clock, 1.0)
    _ = register.record(_notable_event("ring-2", name="environment.visual"))
    _advance(clock, 1.0)
    _ = register.record(_notable_event("ring-3"))

    all_recent = register.recent(limit=16)
    assert [entry.payload["source"] for entry in all_recent] == ["ring-3", "ring-2", "knock-1", "ring-1"]

    sounds = register.recent("environment.sound", limit=16)
    assert [entry.payload["source"] for entry in sounds] == ["ring-3", "knock-1", "ring-1"]
    assert sounds[0].stimulus_name == "environment.sound"


def test_register_recency_window_excludes_stale_observations() -> None:
    clock = _ControlledClock()
    register = memory.StmEventMemory(recency_window=datetime.timedelta(minutes=10), clock=clock)

    _ = register.record(_notable_event("first"))
    assert [entry.payload["source"] for entry in register.recent(limit=16)] == ["first"]
    # Second record six minutes later; both observations are within the ten-minute window.
    _advance(clock, 6 * 60)
    _ = register.record(_notable_event("second"))
    assert [entry.payload["source"] for entry in register.recent(limit=16)] == ["second", "first"]
    # Third record twelve minutes after the first: "first" is now stale and drops out.
    _advance(clock, 6 * 60)
    _ = register.record(_notable_event("third"))
    assert [entry.payload["source"] for entry in register.recent(limit=16)] == ["third", "second"]


def test_register_projection_keeps_bounded_scalars_only() -> None:
    register = memory.StmEventMemory()

    long_id = "d" * (stm.MAX_PROJECTION_TEXT_LENGTH + 1)
    sound = _sound_event(
        data=_SensoryPayload(
            media_type="audio/pcm",
            sample_rate_hz=48_000,
            audio=b"payload-bytes",
        )
    )
    recorded = register.record(sound)

    assert recorded.payload == {"media_type": "audio/pcm", "sample_rate_hz": 48_000}

    long_event = _sound_event(data={"kind": "knock", "long_id": long_id})
    dropped = register.record(long_event)
    assert dropped.payload == {"kind": "knock"}

    scalar_event = _sound_event(data="just-a-word")
    scalar_record = register.record(scalar_event)
    assert scalar_record.payload == {"value": "just-a-word"}


def test_register_records_envelope_identity() -> None:
    register = memory.StmEventMemory()
    event = _notable_event("device-b")
    enveloped = dataclasses.replace(event, id="turn-1", source="device-b", target="bot")

    recorded = register.record(enveloped)
    assert recorded.stimulus_name == "environment.sound"
    assert recorded.event_id == "turn-1"
    assert recorded.source == "device-b"
    assert recorded.target == "bot"

    recent = register.recent("environment.sound", limit=1)
    assert recent[0].stimulus_name == "environment.sound"
    assert recent[0].payload == {"kind": "knock", "source": "device-b"}
    assert recent[0].event_id == "turn-1"
    assert recent[0].source == "device-b"
    assert recent[0].target == "bot"


def test_register_survives_restart_on_file_backed_store(tmp_path: pathlib.Path) -> None:
    database = str(tmp_path / "register.db")
    dom_id = uuid.uuid4().hex

    writer = memory.StmEventMemory(database=database)
    _ = writer.record(_notable_event(dom_id))

    reopened = memory.StmEventMemory(database=database)
    recent = reopened.recent(limit=16)
    assert [entry.payload["source"] for entry in recent] == [dom_id]


def test_register_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        memory.StmEventMemory(capacity=0)
    with pytest.raises(ValueError):
        memory.StmEventMemory(recency_window=datetime.timedelta(0))
    register = memory.StmEventMemory()
    with pytest.raises(ValueError):
        _ = register.recent(limit=0)
    with pytest.raises(ValueError):
        _ = register.recent(limit=1000)


def test_register_record_data_rejects_unbounded_text() -> None:
    with pytest.raises(ValueError):
        memory.RecordData(stimulus_name="environment.sound", payload={"note": "z" * 64})
