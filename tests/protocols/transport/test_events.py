"""Tests for the transport peer contract between a device link and a transport."""

import datetime

import typing

import hsm
import pydantic
import pytest

import mosfet.event
from mosfet.protocols import attachment, transform, transport
from tests.type_helpers import object_dict


class _ColumnsData(pydantic.BaseModel):
    columns: int


_ColumnsEvent = hsm.Event[_ColumnsData](name="test.transport.columns", schema=_ColumnsData)


def _properties(model: type[pydantic.BaseModel]) -> set[str]:
    return set(object_dict(typing.cast(object, model.model_json_schema()["properties"])))


def _channel(name: str, owner: hsm.Instance | None = None) -> transport.ChannelData:
    return transport.ChannelData(
        name=name,
        owner=owner or hsm.Instance(),
        codec=transform.JsonCodec(_ColumnsEvent, max_message_bytes=1024),
        kind="data",
    )


@pytest.mark.parametrize(
    ("public", "name"),
    [
        (transport.PeerConnectedEvent, "protocol.transport.peer.connected"),
        (transport.PeerDisconnectedEvent, "protocol.transport.peer.disconnected"),
        (transport.AcceptPeerEvent, "protocol.transport.peer.accept"),
        (transport.ClosePeerEvent, "protocol.transport.peer.close"),
    ],
)
def test_peer_events_are_named_in_their_domain_and_are_not_tools(public: hsm.Event[object], name: str) -> None:
    assert public.name == name
    assert public.kind is not mosfet.event.EventKind


def test_peer_attach_is_an_attach_request_that_carries_its_channel_table_at_runtime_only() -> None:
    link = hsm.Instance()
    keyboard_port = hsm.Instance()
    keyboard = _channel("keyboard", keyboard_port)

    data = transport.PeerAttachData(
        actor=link,
        reply_to=link,
        peer_id="studio-laptop",
        channels=(keyboard, _channel("display")),
        timeout=datetime.timedelta(seconds=10),
    )
    request = attachment.AttachEvent.with_data_and_id(data, "register")

    assert isinstance(request.data, attachment.AttachData)
    assert data.channels[0] is keyboard
    assert data.channels[0].owner is keyboard_port
    assert data.channels[0].codec.name == "json"
    assert [channel.name for channel in data.channels] == ["keyboard", "display"]
    assert "channels" not in data.model_dump()
    assert _properties(transport.PeerAttachData) == {"actor", "timeout", "peer_id"}


def test_peer_attach_defaults_to_no_channels() -> None:
    assert transport.PeerAttachData(actor=hsm.Instance(), peer_id="kiosk-1").channels == ()


def test_peer_attach_refuses_two_channels_with_one_name() -> None:
    with pytest.raises(pydantic.ValidationError, match="keyboard"):
        _ = transport.PeerAttachData(
            actor=hsm.Instance(), peer_id="kiosk-1", channels=(_channel("keyboard"), _channel("keyboard"))
        )


def test_channel_names_its_kind_and_keeps_owner_and_codec_off_the_wire() -> None:
    owner = hsm.Instance()
    codec = transform.JsonCodec(_ColumnsEvent, max_message_bytes=1024)

    channel = transport.ChannelData(name="microphone", owner=owner, codec=codec, kind="audio")

    assert channel.owner is owner and channel.codec is codec
    assert channel.model_dump() == {"name": "microphone", "kind": "audio"}
    assert _properties(transport.ChannelData) == {"name", "kind"}
    with pytest.raises(pydantic.ValidationError):
        _ = transport.ChannelData.model_validate(
            {"name": "microphone", "owner": owner, "codec": codec, "kind": "video"}
        )
    with pytest.raises(pydantic.ValidationError):
        _ = transport.ChannelData(name="Microphone", owner=owner, codec=codec, kind="audio")


def test_peer_connected_is_stamped_by_the_transport_and_carries_its_connection_at_runtime_only() -> None:
    connection = hsm.Instance()

    data = transport.PeerConnectedData(
        peer_id="studio-laptop", connection_id="c0ffee", transport="websocket", connection=connection
    )

    assert data.connection is connection
    assert data.model_dump() == {"peer_id": "studio-laptop", "connection_id": "c0ffee", "transport": "websocket"}
    assert _properties(transport.PeerConnectedData) == {"peer_id", "connection_id", "transport"}


@pytest.mark.parametrize("reason", ["left", "transport_lost", "closed"])
def test_peer_disconnected_says_why(reason: str) -> None:
    data = transport.PeerDisconnectedData.model_validate(
        {"peer_id": "kiosk-1", "connection_id": "1", "transport": "livekit", "reason": reason}
    )

    assert data.reason == reason


@pytest.mark.parametrize("reason", ["superseded", "released"])
def test_close_peer_lets_one_connection_go_for_a_closed_reason(reason: str) -> None:
    data = transport.ClosePeerData.model_validate({"peer_id": "kiosk-1", "connection_id": "1", "reason": reason})

    assert data.model_dump() == {"peer_id": "kiosk-1", "connection_id": "1", "reason": reason}


def test_accept_peer_names_only_the_peer_and_its_connection() -> None:
    data = transport.AcceptPeerData(peer_id="kiosk-1", connection_id="attempt-3")

    assert data.model_dump() == {"peer_id": "kiosk-1", "connection_id": "attempt-3"}
    assert _properties(transport.AcceptPeerData) == {"peer_id", "connection_id"}


def test_reasons_are_closed_sets() -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = transport.PeerDisconnectedData.model_validate(
            {"peer_id": "kiosk-1", "connection_id": "1", "transport": "livekit", "reason": "superseded"}
        )
    with pytest.raises(pydantic.ValidationError):
        _ = transport.ClosePeerData.model_validate({"peer_id": "kiosk-1", "connection_id": "1", "reason": "left"})


def test_peer_payloads_are_distinct_types() -> None:
    accept: object = transport.AcceptPeerData(peer_id="kiosk-1", connection_id="1")

    assert not isinstance(accept, transport.ClosePeerData)
    assert not isinstance(accept, transport.PeerConnectedData)


@pytest.mark.parametrize("peer_id", ["studio-laptop", "Kiosk.1", "a", "0_peer", "a" * 128])
def test_peer_ids_are_path_safe_names(peer_id: str) -> None:
    assert transport.AcceptPeerData(peer_id=peer_id, connection_id="1").peer_id == peer_id


@pytest.mark.parametrize("peer_id", ["", "-peer", ".peer", "peer/1", "peer 1", "a" * 129, "pé"])
def test_peer_ids_outside_the_pattern_are_refused(peer_id: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = transport.AcceptPeerData(peer_id=peer_id, connection_id="1")


@pytest.mark.parametrize("connection_id", ["", "x" * 65])
def test_connection_ids_are_one_to_sixty_four_characters(connection_id: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = transport.AcceptPeerData(peer_id="kiosk-1", connection_id=connection_id)
    assert transport.AcceptPeerData(peer_id="kiosk-1", connection_id="x" * 64).connection_id == "x" * 64


@pytest.mark.parametrize("name", ["websocket", "livekit", "a", "tcp_v2", "a" * 32])
def test_transport_names_are_low_cardinality_labels(name: str) -> None:
    data = transport.PeerDisconnectedData(peer_id="kiosk-1", connection_id="1", transport=name, reason="left")

    assert data.transport == name


@pytest.mark.parametrize("name", ["", "WebSocket", "1ws", "web-socket", "a" * 33])
def test_transport_names_outside_the_pattern_are_refused(name: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        _ = transport.PeerDisconnectedData(peer_id="kiosk-1", connection_id="1", transport=name, reason="left")
