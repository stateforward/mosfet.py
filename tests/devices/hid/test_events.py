from mosfet.devices import hid

import typing

import hsm
import pydantic
import pytest

from mosfet import event
from tests.type_helpers import object_dict


def test_hid_events_use_pydantic_schemas() -> None:
    assert hid.KeyEvent.name == "devices.hid.key"
    assert object_dict(hid.KeyEvent.schema) == hid.KeyData.model_json_schema()
    assert hid.PointerEvent.name == "devices.hid.pointer"
    assert object_dict(hid.PointerEvent.schema) == hid.PointerData.model_json_schema()


def test_hid_events_are_not_model_callable() -> None:
    """A key or a pointer movement is something the human did; a model is never offered it as a tool."""

    assert not hsm.kind.Is(hid.KeyEvent.kind, event.EventKind)
    assert not hsm.kind.Is(hid.PointerEvent.kind, event.EventKind)


def test_key_data_describes_one_w3c_key_transition() -> None:
    data = hid.KeyData(action="down", key="A", code="KeyA", modifiers=frozenset({"shift"}), text="A")

    assert data.kind == "hid.key"
    assert data.action == "down"
    assert data.key == "A"
    assert data.code == "KeyA"
    assert data.modifiers == frozenset({"shift"})
    assert data.text == "A"
    assert hid.KeyData(action="up", key="Enter", code="Enter").modifiers == frozenset()
    assert hid.KeyData(action="up", key="Enter", code="Enter").text is None
    assert hid.KeyData(action="down", key="Unidentified", code="").code == ""


def test_key_data_round_trips_its_discriminated_json() -> None:
    data = hid.KeyData(action="down", key="a", code="KeyA", modifiers=frozenset({"control", "alt"}), text=None)

    assert hid.KeyData.model_validate_json(data.model_dump_json()) == data


@pytest.mark.parametrize(
    "fields",
    [
        {"key": ""},
        {"key": "k" * 33},
        {"code": "c" * 33},
        {"text": "t" * 257},
        {"action": "press"},
        {"modifiers": frozenset({"hyper"})},
        {"kind": "hid.pointer"},
    ],
)
def test_key_data_enforces_its_limits(fields: dict[str, object]) -> None:
    valid: dict[str, object] = {"action": "down", "key": "a", "code": "KeyA"}

    with pytest.raises(pydantic.ValidationError):
        _ = hid.KeyData.model_validate({**valid, **fields})


def test_key_data_accepts_its_bounds() -> None:
    data = hid.KeyData(action="down", key="k" * 32, code="c" * 32, text="t" * 256)

    assert len(data.key) == 32
    assert len(data.code) == 32
    assert data.text is not None and len(data.text) == 256


def test_pointer_data_describes_one_normalized_pointer_transition() -> None:
    data = hid.PointerData(action="down", pointer_type="touch", pointer_id=3, x=0.25, y=0.75, pressure=0.5)

    assert data.kind == "hid.pointer"
    assert data.action == "down"
    assert data.pointer_type == "touch"
    assert data.pointer_id == 3
    assert (data.x, data.y) == (0.25, 0.75)
    assert data.pressure == 0.5
    assert hid.PointerData(action="move", pointer_type="mouse", pointer_id=0, x=0.0, y=1.0).pressure is None
    assert hid.PointerData.model_validate_json(data.model_dump_json()) == data


@pytest.mark.parametrize(
    "fields",
    [
        {"x": -0.01},
        {"x": 1.01},
        {"y": -0.01},
        {"y": 1.01},
        {"pointer_id": -1},
        {"pointer_id": 32},
        {"pressure": -0.01},
        {"pressure": 1.01},
        {"action": "click"},
        {"pointer_type": "trackball"},
        {"kind": "hid.key"},
    ],
)
def test_pointer_data_enforces_its_limits(fields: dict[str, object]) -> None:
    valid: dict[str, object] = {"action": "move", "pointer_type": "mouse", "pointer_id": 0, "x": 0.5, "y": 0.5}

    with pytest.raises(pydantic.ValidationError):
        _ = hid.PointerData.model_validate({**valid, **fields})


def test_pointer_data_accepts_its_bounds() -> None:
    low = hid.PointerData(action="up", pointer_type="pen", pointer_id=0, x=0.0, y=0.0, pressure=0.0)
    high = hid.PointerData(action="cancel", pointer_type="pen", pointer_id=31, x=1.0, y=1.0, pressure=1.0)

    assert (low.pointer_id, low.x, low.y, low.pressure) == (0, 0.0, 0.0, 0.0)
    assert (high.pointer_id, high.x, high.y, high.pressure) == (31, 1.0, 1.0, 1.0)


def test_hid_data_is_immutable() -> None:
    key = hid.KeyData(action="down", key="a", code="KeyA")
    pointer = hid.PointerData(action="move", pointer_type="mouse", pointer_id=0, x=0.5, y=0.5)

    with pytest.raises(pydantic.ValidationError):
        typing.cast(typing.Any, key).key = "b"
    with pytest.raises(pydantic.ValidationError):
        typing.cast(typing.Any, pointer).x = 0.1
