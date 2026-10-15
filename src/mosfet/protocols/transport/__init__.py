"""Transport peers: how a device link registers for one remote peer and learns which connection is current.

Transports carry bytes only. A yamux stream (or a LiveKit topic) is a channel; the endpoint's channel table
names the port that owns each channel and the :class:`mosfet.protocols.transform.Codec` its host applies.
"""

from .events import (
    AcceptPeerData,
    AcceptPeerEvent,
    ChannelData,
    ChannelKind,
    ClosePeerData,
    ClosePeerEvent,
    PeerAttachData,
    PeerConnectedData,
    PeerConnectedEvent,
    PeerDisconnectedData,
    PeerDisconnectedEvent,
    PeerId,
    TransportName,
)

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
