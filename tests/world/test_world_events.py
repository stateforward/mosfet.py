from bot.world import SoundData, SoundEvent, VisualData, VisualEvent

import pytest

from tests.type_helpers import object_dict


def test_sound_event_uses_concrete_pydantic_schema() -> None:
    schema = object_dict(SoundEvent.schema)

    assert SoundEvent.name == "world.sound"
    assert schema == SoundData.model_json_schema()
    assert schema["properties"]["audio"]["format"] in {"binary", "base64", "base64url"}
    assert "description" in schema


def test_visual_event_uses_concrete_pydantic_schema() -> None:
    schema = object_dict(VisualEvent.schema)

    assert VisualEvent.name == "world.visual"
    assert schema == VisualData.model_json_schema()
    assert schema["properties"]["image"]["format"] in {"binary", "base64", "base64url"}
    assert "description" in schema


def test_sound_data_rejects_empty_audio() -> None:
    with pytest.raises(Exception):
        _ = SoundData(audio=b"")


def test_visual_data_rejects_empty_image() -> None:
    with pytest.raises(Exception):
        _ = VisualData(image=b"")
