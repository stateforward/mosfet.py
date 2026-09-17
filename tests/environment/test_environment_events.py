from mosfet.environment import SoundData, SoundEvent, VisualData, VisualEvent

import pytest

from tests.type_helpers import object_dict


def test_sound_event_uses_concrete_pydantic_schema() -> None:
    schema = object_dict(SoundEvent.schema)
    properties = object_dict(schema["properties"])
    audio = object_dict(properties["audio"])

    assert SoundEvent.name == "environment.sound"
    assert schema == SoundData.model_json_schema()
    assert audio["format"] in {"binary", "base64", "base64url"}
    assert "description" in schema


def test_visual_event_uses_concrete_pydantic_schema() -> None:
    schema = object_dict(VisualEvent.schema)
    properties = object_dict(schema["properties"])
    image = object_dict(properties["image"])

    assert VisualEvent.name == "environment.visual"
    assert schema == VisualData.model_json_schema()
    assert image["format"] in {"binary", "base64", "base64url"}
    assert "description" in schema


def test_sound_data_rejects_empty_audio() -> None:
    with pytest.raises(Exception):
        _ = SoundData(audio=b"")


def test_visual_data_rejects_empty_image() -> None:
    with pytest.raises(Exception):
        _ = VisualData(image=b"")
