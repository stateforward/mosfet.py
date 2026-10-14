"""Carry remote peers' channels over yamux-over-WebSocket: a shared, device-agnostic transport ``Server``.

from mosfet.devices import computer
from mosfet.providers import websocket

tokens = websocket.HmacTokens(secret)
server = websocket.Server(host="0.0.0.0", port=8765, verifier=tokens)  # in Assembly.actors
computer.Computer(device_id="studio-laptop", transport=server)
"""

from .connection import Connection, Socket
from .events import (
    AuthenticatedData,
    AuthenticatedEvent,
    BoundData,
    BoundEvent,
    RefusedData,
    RefusedEvent,
    StreamOpenData,
    StreamOpenEvent,
)
from .server import Handler, Listener, Serve, Server
from .tokens import HmacTokens, TokenVerifier

__version__ = "0.1.0"

__all__ = [
    "AuthenticatedData",
    "AuthenticatedEvent",
    "BoundData",
    "BoundEvent",
    "Connection",
    "Handler",
    "HmacTokens",
    "Listener",
    "RefusedData",
    "RefusedEvent",
    "Serve",
    "Server",
    "Socket",
    "StreamOpenData",
    "StreamOpenEvent",
    "TokenVerifier",
    "__version__",
]
