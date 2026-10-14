import mosfet
from mosfet import media
from mosfet.devices import phone
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
    assert set(properties) == {"image", "kind", "participants"}
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


def _participant(index: int) -> mosfet.Participant:
    return mosfet.Participant(ref=f"person-{index}", role=mosfet.ParticipantRole.SOURCE)


@pytest.mark.parametrize("payload", [SoundData(audio=b"pcm"), VisualData(image=_PNG)], ids=["sound", "visual"])
def test_stimuli_carry_no_participants_unless_their_device_stamps_them(payload: SoundData | VisualData) -> None:
    assert payload.participants == ()


@pytest.mark.parametrize("data_type", [SoundData, VisualData], ids=["sound", "visual"])
def test_stimulus_participants_are_bounded_at_32(data_type: type[SoundData] | type[VisualData]) -> None:
    fields: dict[str, object] = {"audio": b"pcm"} if data_type is SoundData else {"image": _PNG}

    stamped = data_type.model_validate({**fields, "participants": [_participant(index) for index in range(32)]})

    assert len(stamped.participants) == 32
    with pytest.raises(pydantic.ValidationError):
        _ = data_type.model_validate({**fields, "participants": [_participant(index) for index in range(33)]})


@pytest.mark.parametrize("data_type", [SoundData, VisualData], ids=["sound", "visual"])
def test_stimulus_participants_schema_says_the_emitting_device_stamps_them(
    data_type: type[SoundData] | type[VisualData],
) -> None:
    properties = object_dict(object_dict(data_type.model_json_schema())["properties"])
    participants = object_dict(properties["participants"])

    assert "stamped by the device that emitted it" in str(participants["description"])
    assert participants["maxItems"] == 32
    assert participants["examples"]


def test_stimulus_participants_round_trip_through_json() -> None:
    sound = SoundData(audio=b"pcm", participants=(_participant(1),))

    assert SoundData.model_validate_json(sound.model_dump_json()) == sound


def test_phone_sound_inherits_participants_from_the_environment_stimulus() -> None:
    ring = phone.SoundData(audio=b"ring", kind="phone.ringing", caller="Front desk")

    assert "participants" in phone.SoundData.model_fields
    assert ring.participants == ()
