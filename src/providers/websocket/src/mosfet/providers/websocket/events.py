"""The WebSocket transport's own events: routing a connection to its endpoint, and naming a stream.

``Server`` keeps the route table (peer id → the endpoint registered for it and that endpoint's channel
table) and binds an authenticated ``Connection`` to its endpoint, or refuses it. Routing data — the
endpoint, the connection, the channel table — travels in event data; no machine reads another's state.

``StreamOpenData`` is the first message of every stream the remote machine opens: it names the channel
the stream carries. Its JSON Schema (``StreamOpenData.model_json_schema()``) is the contract for clients.

None of these is a tool: they are runtime-only wiring between provider machines, never offered to a model.
"""

import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from mosfet.protocols import transform, transport

_RUNTIME_ONLY = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)


class AuthenticatedData(pydantic.BaseModel):
    """A connection proved it speaks for ``peer_id`` and waits to be routed."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _RUNTIME_ONLY

    peer_id: transport.PeerId
    connection: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description="The authenticated connection. Runtime-only.",
    )


class RefusedData(pydantic.BaseModel):
    """No endpoint is registered for this peer: the connection closes with ``4404``."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"code": 4404}]},
    )

    code: typing.Literal[4404] = pydantic.Field(
        default=4404,
        description="WebSocket close code: no endpoint takes this peer.",
        examples=[4404],
    )


class BoundData(pydantic.BaseModel):
    """The endpoint an authenticated connection now serves, and the channel table its streams are handed to."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = _RUNTIME_ONLY

    endpoint: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=(
            "The endpoint that registered for this peer: it is told protocol.transport.peer.connected and "
            "peer.disconnected, and accepts or closes the connection. Runtime-only."
        ),
    )
    channels: SkipJsonSchema[tuple[transport.ChannelData, ...]] = pydantic.Field(
        default=(),
        exclude=True,
        repr=False,
        description="The endpoint's channel table: each channel's name, owning port, codec and kind. Runtime-only.",
    )


class StreamOpenData(pydantic.BaseModel):
    """The first message on every stream the remote machine opens: which channel the stream carries.

    It travels as compact UTF-8 JSON, length-framed like every stream message, for example
    ``{"channel": "microphone", "v": 1}``. A channel the endpoint did not register, or a message that is
    not this schema, resets the stream. Everything after it on the stream is the channel's own codec.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{"channel": "microphone", "v": 1}, {"channel": "keyboard", "v": 1}]},
    )

    channel: transform.ChannelName = pydantic.Field(
        description="The channel this stream carries, one of the names the endpoint registered.",
        examples=["keyboard", "pointer", "display", "camera", "microphone", "speaker"],
    )
    v: typing.Literal[1] = pydantic.Field(
        description="Version of this stream-opening message.",
        examples=[1],
    )


AuthenticatedEvent = hsm.Event[AuthenticatedData](name="providers.websocket.authenticated", schema=AuthenticatedData)
RefusedEvent = hsm.Event[RefusedData](name="providers.websocket.refused", schema=RefusedData)
BoundEvent = hsm.Event[BoundData](name="providers.websocket.bound", schema=BoundData)
StreamOpenEvent = hsm.Event[StreamOpenData](name="providers.websocket.stream.open", schema=StreamOpenData)

__all__ = [
    "AuthenticatedData",
    "AuthenticatedEvent",
    "BoundData",
    "BoundEvent",
    "RefusedData",
    "RefusedEvent",
    "StreamOpenData",
    "StreamOpenEvent",
]
