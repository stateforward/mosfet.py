from mosfet.devices import computer

import json
import typing

import hsm
import pydantic
import pytest

from mosfet import event, media
from mosfet.abilities import processing
from mosfet.environment import SoundData

PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, media_type="image/png")


def _schema_text(model: type[pydantic.BaseModel]) -> str:
    return json.dumps(model.model_json_schema())


def test_one_json_message_on_a_computer_stream_is_bounded_at_64_kib() -> None:
    assert computer.MAX_MESSAGE_BYTES == 64 * 1024


def test_layout_boxes_are_normalized() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = computer.BoxData(id="b", x=1.2, y=0.0, width=0.1, height=0.1)
    with pytest.raises(pydantic.ValidationError):
        _ = computer.BoxData(id="b", x=0.0, y=-0.1, width=0.1, height=0.1)


def test_content_and_show_reject_duplicate_element_ids() -> None:
    twice = ({"kind": "button", "id": "b", "label": "One"}, {"kind": "button", "id": "b", "label": "Two"})

    with pytest.raises(pydantic.ValidationError, match="unique"):
        _ = computer.ShowData.model_validate({"elements": twice})
    with pytest.raises(pydantic.ValidationError, match="unique"):
        _ = computer.ContentData.model_validate({"elements": twice})


def _kinds(schema: object) -> dict[str, list[object]]:
    """Each object schema under ``schema`` whose ``kind`` is a constant, by that constant: what it requires."""

    found: dict[str, list[object]] = {}
    pending = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            pending.extend(typing.cast(list[object], node))
        if not isinstance(node, dict):
            continue
        mapping = typing.cast(dict[str, object], node)
        pending.extend(mapping.values())
        properties = mapping.get("properties")
        kind = typing.cast(dict[str, object], properties).get("kind") if isinstance(properties, dict) else None
        if isinstance(kind, dict) and isinstance(const := typing.cast(dict[str, object], kind).get("const"), str):
            found[const] = list(typing.cast(list[object], mapping.get("required", [])))
    return found


def test_the_model_facing_show_schema_requires_each_elements_kind_as_its_validator_does() -> None:
    # Live (builder check round 3, H2): 20 of 26 shows left out `kind`, which the schema called optional, and
    # every one was rejected for its missing discriminator, so the person saw a blank screen for 18 minutes.
    required = _kinds(processing.dispatch_tool((computer.ShowEvent,)))

    assert sorted(required) == ["button", "field", "text"]
    assert all("kind" in names for names in required.values()), required
    with pytest.raises(pydantic.ValidationError, match="union_tag_not_found"):
        _ = computer.ShowData.model_validate({"elements": [{"id": "a", "text": "b"}]})


@pytest.mark.parametrize("model", [computer.ContentData, computer.LayoutData, computer.StillData])
def test_each_link_plane_payload_schema_requires_its_kind(model: type[pydantic.BaseModel]) -> None:
    assert "kind" in model.model_json_schema()["required"]


def test_a_constructed_element_or_payload_keeps_its_kind_by_default() -> None:
    assert computer.TextElement(id="a", text="b").kind == "text"
    assert computer.FieldElement(id="a").kind == "field"
    assert computer.ButtonElement(id="a", label="b").kind == "button"
    assert computer.ContentData().kind == "display.content"
    assert computer.LayoutData().kind == "display.layout"
    assert computer.StillData(image=PNG).kind == "camera.still"


def test_still_needs_a_typed_image_not_raw_bytes() -> None:
    assert computer.StillData(image=PNG).image is PNG
    with pytest.raises(pydantic.ValidationError):
        _ = computer.StillData.model_validate({"image": b"\x89PNG\r\n\x1a\n"})


def test_show_is_the_one_model_callable_event_and_carries_no_media() -> None:
    assert computer.ShowEvent.name == "computer.show"
    assert hsm.kind.Is(computer.ShowEvent.kind, event.EventKind)
    schema = _schema_text(computer.ShowData)
    assert "image" not in schema
    assert "binary" not in schema and "base64" not in schema
    for observation in (
        computer.TextEnteredEvent,
        computer.ActivatedEvent,
        computer.LinkUpEvent,
        computer.LinkDownEvent,
        computer.ContentEvent,
        computer.LayoutEvent,
        computer.StillEvent,
        computer.HeardEvent,
    ):
        assert not hsm.kind.Is(observation.kind, event.EventKind), observation.name


def test_microphone_sound_is_a_sound_subtype_with_its_own_provenance() -> None:
    sound = computer.MicrophoneSoundData(audio=b"pcm")

    assert isinstance(sound, SoundData)
    assert sound.kind == "computer.microphone"
    assert set(computer.MicrophoneSoundData.model_fields) == set(SoundData.model_fields)


def test_heard_keeps_the_sound_it_was_given_as_is() -> None:
    sound = computer.MicrophoneSoundData(audio=b"pcm")

    assert computer.HeardData(sound=sound).sound is sound


@pytest.mark.parametrize("person_id", ["", "p" * 201])
def test_observations_bound_the_person_id(person_id: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = computer.ActivatedData(device_id="studio-laptop", person_id=person_id, element_id="b")


def test_text_entered_must_carry_text() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = computer.TextEnteredData(device_id="studio-laptop", field_id="f", text="")
    entered = computer.TextEnteredData(device_id="studio-laptop", field_id="f", text="hello")
    assert entered.person_id is None


def test_text_entered_with_nothing_focused_names_no_field() -> None:
    assert computer.TextEnteredData(device_id="studio-laptop", text="hello").field_id is None


@pytest.mark.parametrize("device_id", ["desk", "my_desk"])
def test_observations_carry_the_id_the_bot_knows_the_computer_by(device_id: str) -> None:
    """DSL-30: ``device_id`` is the computer's ``mosfet.device.DeviceId``, not its peer name."""

    assert computer.LinkUpData(device_id=device_id, transport="websocket").device_id == device_id
    with pytest.raises(pydantic.ValidationError):
        _ = computer.LinkUpData(device_id="Desk", transport="websocket")


def test_link_reasons_are_closed() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = computer.LinkDownData.model_validate({"device_id": "studio-laptop", "reason": "lost"})
