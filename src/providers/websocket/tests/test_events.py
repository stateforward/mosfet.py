from __future__ import annotations

import json
import typing

import hsm
import pydantic
import pytest

import mosfet.event
from mosfet.protocols import transform, transport
from mosfet.providers import websocket


class _Actor(hsm.Instance):
    pass


class _NoteData(pydantic.BaseModel):
    text: str


_NoteEvent = hsm.Event[_NoteData](name="test.websocket.note", schema=_NoteData)


def properties(model: type[pydantic.BaseModel]) -> set[str]:
    return set(typing.cast(dict[str, object], model.model_json_schema().get("properties", {})))


PUBLIC = (
    (websocket.AuthenticatedEvent, "providers.websocket.authenticated"),
    (websocket.RefusedEvent, "providers.websocket.refused"),
    (websocket.BoundEvent, "providers.websocket.bound"),
    (websocket.StreamOpenEvent, "providers.websocket.stream.open"),
)


@pytest.mark.parametrize(("public", "name"), PUBLIC)
def test_transport_events_are_named_in_their_domain_and_are_not_tools(public: hsm.Event[object], name: str) -> None:
    assert public.name == name
    assert public.kind is not mosfet.event.EventKind


def test_the_package_names_no_device() -> None:
    exported = set(websocket.__all__)

    assert not {name for name in exported if "Computer" in name or "Register" in name or "Close" in name}


def test_authenticated_names_a_valid_peer_and_carries_the_connection_at_runtime_only() -> None:
    connection = _Actor()
    data = websocket.AuthenticatedData(peer_id="studio-laptop", connection=connection)

    assert data.connection is connection
    assert properties(websocket.AuthenticatedData) == {"peer_id"}
    with pytest.raises(pydantic.ValidationError):
        _ = websocket.AuthenticatedData(peer_id="not a peer/", connection=connection)


def test_bound_carries_the_endpoint_and_its_channel_table_at_runtime_only() -> None:
    endpoint, port = _Actor(), _Actor()
    channel = transport.ChannelData(
        name="keyboard",
        owner=port,
        codec=transform.JsonCodec(_NoteEvent, max_message_bytes=1024),
        kind="data",
    )

    data = websocket.BoundData(endpoint=endpoint, channels=(channel,))

    assert (data.endpoint, data.channels) == (endpoint, (channel,))
    assert data.model_dump() == {}
    assert properties(websocket.BoundData) == set()


def test_refusal_is_always_unknown_peer() -> None:
    assert websocket.RefusedData().code == 4404
    with pytest.raises(pydantic.ValidationError):
        _ = websocket.RefusedData.model_validate({"code": 4401})


def test_stream_open_is_a_typed_channel_and_version_whose_schema_clients_take() -> None:
    opened = websocket.StreamOpenData.model_validate_json('{"channel":"microphone","v":1}')
    schema = websocket.StreamOpenData.model_json_schema()

    assert (opened.channel, opened.v) == ("microphone", 1)
    assert json.loads(opened.model_dump_json()) == {"channel": "microphone", "v": 1}
    assert schema["required"] == ["channel", "v"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["v"]["const"] == 1
    assert schema["properties"]["channel"]["pattern"] == r"^[a-z][a-z0-9._-]{0,63}$"
    assert {"channel": "microphone", "v": 1} in schema["examples"]


@pytest.mark.parametrize(
    "wire",
    [
        '{"name":"microphone","v":1}',
        '{"channel":"microphone","v":2}',
        '{"channel":"Microphone","v":1}',
        '{"channel":"microphone"}',
        '{"channel":"microphone","v":1,"device":"kiosk-1"}',
    ],
)
def test_stream_open_refuses_anything_but_its_schema(wire: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = websocket.StreamOpenData.model_validate_json(wire)
