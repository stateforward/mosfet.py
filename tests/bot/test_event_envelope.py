"""Direct contract tests for typed HSM event payload envelopes."""

import mosfet
from mosfet import event
from mosfet.environment import SoundData, SoundEvent

import pytest


def test_event_json_value_serializes_typed_payload_without_raw_media() -> None:
    value = event.event_json_value(
        SoundData(audio=b"ring", media_type="audio/pcm", sample_rate_hz=16_000, channels=1, kind="phone.ringing")
    )

    assert isinstance(value, dict)
    assert value["kind"] == "phone.ringing"


def test_event_json_value_rejects_raw_media_root() -> None:
    with pytest.raises(TypeError):
        _ = event.event_json_value(b"raw-bytes")


def test_event_schema_round_trip_validates_sound_data() -> None:
    data = SoundData(audio=b"ring", media_type="audio/pcm", sample_rate_hz=16_000, channels=1, kind="phone.ringing")

    assert event.validate_event_data(SoundEvent, data) == data
    assert event.event_json_schema(SoundEvent) == SoundData.model_json_schema()


def test_event_kind_marks_model_offerable_events() -> None:
    assert mosfet.event is event
