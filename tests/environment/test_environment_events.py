from mosfet import media
from mosfet.environment import SoundData, SoundEvent, VisualData, VisualEvent

import pydantic
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


_PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"frame-body", media_type="image/png")


def test_visual_event_uses_concrete_pydantic_schema() -> None:
    schema = object_dict(VisualEvent.schema)
    properties = object_dict(schema["properties"])
    image = object_dict(properties["image"])
    image_schema = object_dict(object_dict(schema["$defs"])["Image"])
    image_properties = object_dict(image_schema["properties"])

    assert VisualEvent.name == "environment.visual"
    assert schema == VisualData.model_json_schema()
    assert image["$ref"] == "#/$defs/Image"
    assert set(properties) == {"image", "kind"}
    assert object_dict(image_properties["content"])["format"] in {"binary", "base64", "base64url"}
    assert "description" in schema


def test_visual_data_carries_a_typed_image() -> None:
    visual = VisualData(image=_PNG, kind="camera")

    assert visual.image == _PNG
    assert visual.image.media_type == "image/png"
    assert VisualData.model_validate_json(visual.model_dump_json()) == visual


def test_sound_data_rejects_empty_audio() -> None:
    with pytest.raises(Exception):
        _ = SoundData(audio=b"")


def test_visual_data_rejects_raw_image_bytes() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = VisualData.model_validate({"image": _PNG.content})
