"""Peer events between a device link (the endpoint) and a transport that carries one remote peer.

A transport authenticates a peer before any stream exists, then tells the endpoint the peer is reachable.
The endpoint decides which connection is current with ``AcceptPeerEvent`` and lets one go with
``ClosePeerEvent``; only after acceptance does the transport hand each stream to the owner the endpoint
registered for that channel. Identity is the transport's stamp and the stream's ownership, never a payload
read from the wire.
"""

import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from mosfet.protocols import attachment, transform

PeerId = typing.Annotated[
    str,
    pydantic.Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        description=(
            "How a transport names one remote peer: the WebSocket path segment or the LiveKit participant identity. "
            "URL-path-safe."
        ),
        examples=["studio-laptop", "kiosk-1"],
    ),
]
ChannelKind = typing.Literal["data", "audio"]
TransportName = typing.Annotated[
    str,
    pydantic.Field(
        pattern=r"^[a-z][a-z0-9_]{0,31}$",
        description="Low-cardinality label of the transport that stamped the event.",
        examples=["websocket", "livekit"],
    ),
]
_ConnectionId = typing.Annotated[
    str,
    pydantic.Field(
        min_length=1,
        max_length=64,
        description="Transport-stamped identifier of one connection or attempt, unique within that transport.",
        examples=["9f2c4e1ab07d4c55a1f0e3b2c6d7e8f9", "3"],
    ),
]


class ChannelData(pydantic.BaseModel):
    """One row of an endpoint's channel table: which owner takes the stream named ``name``, through which codec."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    name: transform.ChannelName
    owner: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description="Runtime-only port that owns this channel's stream and receives every event decoded from it.",
    )
    codec: SkipJsonSchema[pydantic.SkipValidation[transform.Codec]] = pydantic.Field(
        exclude=True,
        repr=False,
        description="Runtime-only codec the stream's host applies between messages and the owner's events.",
    )
    kind: ChannelKind = pydantic.Field(
        description="Whether the channel carries data messages or PCM audio.",
        examples=["data", "audio"],
    )


class PeerAttachData(attachment.AttachData):
    """Register ``actor`` as the endpoint for one peer, with the channel table its streams are handed to.

    It is the payload of ``attachment.AttachEvent`` sent to a transport. The transport answers
    ``attachment.AttachCompleteEvent``, or ``attachment.AttachFailedEvent`` with ``conflict`` when another
    endpoint already holds ``peer_id``. ``attachment.DetachEvent`` unregisters.
    """

    peer_id: PeerId = pydantic.Field(description="Remote peer the endpoint registers for.")
    channels: SkipJsonSchema[tuple[ChannelData, ...]] = pydantic.Field(
        default=(),
        exclude=True,
        repr=False,
        description="Runtime-only channel table: each channel's name, owning port, codec and kind.",
    )

    @pydantic.model_validator(mode="after")
    def _one_row_per_channel(self) -> typing.Self:
        names = [channel.name for channel in self.channels]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Channel table names {', '.join(repeated)} more than once.")
        return self


class _PeerData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    peer_id: PeerId = pydantic.Field(description="Authenticated remote peer this event concerns.")
    connection_id: _ConnectionId


class PeerConnectedData(_PeerData):
    """An authenticated peer is reachable on one connection."""

    transport: TransportName
    connection: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=(
            "Runtime-only address that takes this connection's AcceptPeerEvent and ClosePeerEvent. It is never "
            "read and never serialized."
        ),
    )


class PeerDisconnectedData(_PeerData):
    """The connection is gone; sent exactly once per ``PeerConnectedEvent``."""

    transport: TransportName
    reason: typing.Literal["left", "transport_lost", "closed"] = pydantic.Field(
        description=(
            "The peer left cleanly, the transport was lost, or the transport closed it at the endpoint's request."
        ),
        examples=["left", "transport_lost", "closed"],
    )


class AcceptPeerData(_PeerData):
    """The endpoint takes this connection as current: hand its streams to the registered channel owners."""


class ClosePeerData(_PeerData):
    """The endpoint lets this connection go."""

    reason: typing.Literal["superseded", "released"] = pydantic.Field(
        description="A newer connection superseded this one, or the endpoint released the peer.",
        examples=["superseded", "released"],
    )


PeerConnectedEvent = hsm.Event[PeerConnectedData](name="protocol.transport.peer.connected", schema=PeerConnectedData)
PeerDisconnectedEvent = hsm.Event[PeerDisconnectedData](
    name="protocol.transport.peer.disconnected",
    schema=PeerDisconnectedData,
)
AcceptPeerEvent = hsm.Event[AcceptPeerData](name="protocol.transport.peer.accept", schema=AcceptPeerData)
ClosePeerEvent = hsm.Event[ClosePeerData](name="protocol.transport.peer.close", schema=ClosePeerData)

__all__ = [
    "AcceptPeerData",
    "AcceptPeerEvent",
    "ChannelData",
    "ChannelKind",
    "ClosePeerData",
    "ClosePeerEvent",
    "PeerAttachData",
    "PeerConnectedData",
    "PeerConnectedEvent",
    "PeerDisconnectedData",
    "PeerDisconnectedEvent",
    "PeerId",
    "TransportName",
]
