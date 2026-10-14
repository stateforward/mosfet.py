"""One accepted WebSocket: authenticate the remote peer, then carry its yamux session for its endpoint.

The first frame must be text, ``{"token": "..."}`` (browsers cannot set ``Authorization`` on a
WebSocket). A token the verifier refuses closes the socket ``4401``; no token within ``auth_timeout``
closes it ``4408``. A verified connection tells the server, which either binds it to the endpoint
registered for its peer (with that endpoint's channel table) or refuses it (``4404``).

Bound, the connection owns one server-role yamux session and is its socket pump: every binary message is
decoded into yamux frames for the session, and every frame the session transmits is sent as one binary
message, through a bounded outbox. It tells the endpoint ``protocol.transport.peer.connected`` and, once
for that, ``peer.disconnected``. The remote peer opens every stream. Until the endpoint accepts the
connection (``protocol.transport.peer.accept``) each new stream is held with no codec, so its bytes stay
unacknowledged. Accepted, the connection takes each stream itself to read its first message, a
``StreamOpenData`` naming its channel, and hands it to the channel's owner with that channel's codec. A
channel the endpoint did not register, or a stream past ``max_channels``, is reset.

A yamux data frame larger than ``max_message_bytes`` closes the socket ``4413``; a text message, a frame
that is not yamux, or the session failing closes it ``1002``; frames the socket cannot take fast enough,
past four times ``max_message_bytes`` queued, close it ``1011``. The endpoint closes a connection with
``4409`` (superseded) or ``1000`` (released).
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import logging
import struct
import typing
import uuid

import hsm
import pydantic
import websockets.exceptions

from mosfet.define import define
from mosfet.protocols import transform, transport, yamux
from mosfet.start import started
from mosfet.telemetry import observer
from mosfet.telemetry.hsm import Traced

from . import telemetry
from .events import (
    AuthenticatedData,
    AuthenticatedEvent,
    BoundData,
    BoundEvent,
    RefusedEvent,
    StreamOpenData,
    StreamOpenEvent,
)
from .tokens import TokenVerifier

_LOG = logging.getLogger(__name__)
_COMPONENT = "websocket.connection"
_TRANSPORT = "websocket"
_PEER_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(transport.PeerId)
# The token frame is tiny; anything larger is not a token.
_MAX_TOKEN_MESSAGE = 1024
# A stream-opening message is a channel name and a version; anything larger is not one.
_MAX_STREAM_OPEN_BYTES = 1024
# Frames queued for the socket, as a multiple of the largest message, before the socket counts as stuck.
_OUTBOX_MESSAGES = 4
_HEADER = struct.Struct("!BBHII")

_UNAUTHENTICATED = 4401
_UNKNOWN_PEER = 4404
_AUTH_TIMEOUT = 4408
_SUPERSEDED = 4409
_OVERSIZE = 4413
_NORMAL = 1000
_PROTOCOL_ERROR = 1002
_BACKLOG = 1011


class Socket(typing.Protocol):
    """An accepted WebSocket, as a connection uses it (``websockets``' ``ServerConnection`` is one).

    ``recv`` and ``send`` raise ``websockets.exceptions.ConnectionClosed`` once the socket is closed.
    """

    def recv(self) -> collections.abc.Awaitable[str | bytes]:
        """The next whole message: text as ``str``, binary as ``bytes``."""
        ...

    def send(self, message: bytes) -> collections.abc.Awaitable[None]:
        """Send one binary message."""
        ...

    def close(self, code: int = 1000, reason: str = "") -> collections.abc.Awaitable[None]:
        """Close the socket with ``code`` and wait for the closing handshake."""
        ...

    def wait_closed(self) -> collections.abc.Awaitable[None]:
        """Resolve once the socket is closed, by either side."""
        ...


class _TokenMessage(pydantic.BaseModel):
    """The first frame a remote peer sends: its credential."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    token: pydantic.SecretStr = pydantic.Field(min_length=1, max_length=512)


class _StreamOpenCodec:
    """Reads a stream's first message: ``StreamOpenData`` as JSON, decoded to ``StreamOpenEvent``.

    The connection applies it only to streams it holds itself; it encodes nothing.
    """

    name: str
    max_message_bytes: int
    encodes_any: bool

    def __init__(self) -> None:
        self.name = "stream_open"
        self.max_message_bytes = _MAX_STREAM_OPEN_BYTES
        self.encodes_any = False

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """Return the ``StreamOpenEvent`` ``message`` carries.

        Raises:
            transform.DecodeError: ``oversize`` past 1 KiB, ``malformed`` when it is not JSON,
                ``schema_invalid`` when it is not a ``StreamOpenData``.
        """

        if len(message) > self.max_message_bytes:
            raise transform.DecodeError("oversize", f"Stream-opening message exceeds {self.max_message_bytes} bytes.")
        try:
            opened = StreamOpenData.model_validate_json(message)
        except pydantic.ValidationError as error:
            if any(detail["type"] == "json_invalid" for detail in error.errors()):
                raise transform.DecodeError("malformed", "Stream-opening message is not JSON.") from error
            raise transform.DecodeError("schema_invalid", "Stream-opening message is not a StreamOpenData.") from error
        return StreamOpenEvent.with_data(opened)

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """Nothing: the connection never writes on a stream it reads the name of."""

        del event
        return False

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """Refuse every event.

        Raises:
            transform.EncodeError: always ``schema_invalid``.
        """

        raise transform.EncodeError("schema_invalid", f"Event {event.name!r} is never written by this codec.")


_STREAM_OPEN = _StreamOpenCodec()


class _SignalData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


class _SocketClosedData(pydantic.BaseModel):
    """The remote peer closed the socket."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


class _BrokenData(pydantic.BaseModel):
    """What the remote peer sent cannot be carried: a data frame over the limit, or not yamux at all."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    reason: typing.Literal["oversize", "protocol_error"]


# Private: each is dispatched by this class's own activities to itself only.
_VerifiedEvent = hsm.Event[_SignalData](name="providers.websocket.connection.verified", schema=_SignalData)
_RejectedEvent = hsm.Event[_SignalData](name="providers.websocket.connection.rejected", schema=_SignalData)
_SocketClosedEvent = hsm.Event[_SocketClosedData](
    name="providers.websocket.connection.socket_closed", schema=_SocketClosedData
)
_BrokenEvent = hsm.Event[_BrokenData](
    name="providers.websocket.connection.broken", kind=hsm.ErrorEventKind, schema=_BrokenData
)
_ClosedEvent = hsm.Event[_SignalData](
    name="providers.websocket.connection.closed", kind=hsm.CompletionEventKind, schema=_SignalData
)


def _declared_data_length(buffer: bytes) -> int:
    """The payload length a data frame header announces (0 for any other frame type)."""

    _, frame_type, _, _, length = typing.cast(tuple[int, int, int, int, int], _HEADER.unpack_from(buffer))
    return length if frame_type == yamux.FrameType.DATA else 0


class Connection(Traced):
    """One remote peer's WebSocket, from its token to its close, and the yamux session it carries.

    The server constructs, starts and stops one per accepted socket. The connection builds its own
    server-role session, starts it once bound (under its own lifetime), and stops it when it stops.
    """

    _socket: Socket
    _peer_id: str
    _server: hsm.Instance
    _verifier: TokenVerifier
    _auth_timeout: datetime.timedelta
    _max_message_bytes: int
    _max_channels: int
    _session: yamux.Server
    _connection_id: str
    # Whether the bound read activity started the session; written only there.
    _session_started: bool
    # The endpoint and its channel table; written only by _bind.
    _endpoint: hsm.Instance | None
    _channels: dict[str, transport.ChannelData]
    # Streams accepted on this connection: how many, those held until the endpoint accepts, and those
    # whose first message is being read, by stream id. Bounded by max_channels.
    _stream_count: int
    _held: list[hsm.Instance]
    _naming: dict[str, hsm.Instance]
    # Frames for the socket, bounded in bytes; written by _enqueue and drained by _run_write / _run_close.
    _outbox: asyncio.Queue[bytes]
    _outbox_bytes: int
    # How the socket is closed; written only by the transitions into `closing`.
    _close_code: int

    def __init__(
        self,
        *,
        socket: Socket,
        peer_id: str,
        server: hsm.Instance,
        verifier: TokenVerifier,
        auth_timeout: datetime.timedelta,
        max_message_bytes: int,
        max_channels: int,
    ) -> None:
        """Build an unstarted connection for the socket a remote peer opened as ``peer_id``.

        Raises:
            pydantic.ValidationError: If ``peer_id`` is not a ``transport.PeerId``.
            ValueError: If ``auth_timeout``, ``max_message_bytes`` or ``max_channels`` is not positive.
        """

        super().__init__()
        if auth_timeout <= datetime.timedelta():
            raise ValueError("auth_timeout must be a positive duration.")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive.")
        if max_channels <= 0:
            raise ValueError("max_channels must be positive.")
        self._socket = socket
        self._peer_id = _PEER_ID.validate_python(peer_id)
        self._server = server
        self._verifier = verifier
        self._auth_timeout = auth_timeout
        self._max_message_bytes = max_message_bytes
        self._max_channels = max_channels
        self._session = yamux.Server(connection=self)
        self._connection_id = uuid.uuid4().hex
        self._session_started = False
        self._endpoint = None
        self._channels = {}
        self._stream_count = 0
        self._held = []
        self._naming = {}
        self._outbox = asyncio.Queue()
        self._outbox_bytes = 0
        self._close_code = _PROTOCOL_ERROR

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop the connection, then its session: every stream's owner is told the stream closed."""

        await super().stop(ctx)
        if self._session_started:
            self._session_started = False
            await hsm.stop(self._session)

    # -- guards -----------------------------------------------------------------------------------

    @staticmethod
    def _is_this_connection(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> bool:
        """Correlate the endpoint's accept or close with this connection."""

        del ctx
        data = event.data
        return (
            isinstance(data, transport.AcceptPeerData | transport.ClosePeerData)
            and data.connection_id == instance._connection_id
            and data.peer_id == instance._peer_id
        )

    @staticmethod
    def _has_room_for_stream(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._stream_count < instance._max_channels

    @staticmethod
    def _is_naming(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> bool:
        """The message is the first one of a stream this connection is reading the name of."""

        del ctx
        return isinstance(event.data, StreamOpenData) and event.source in instance._naming

    @staticmethod
    def _names_registered_channel(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> bool:
        data = event.data
        return (
            Connection._is_naming(ctx, instance, event)
            and isinstance(data, StreamOpenData)
            and data.channel in instance._channels
        )

    @staticmethod
    def _outbox_has_room(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, yamux.TransmitData)
            and instance._outbox_bytes + len(data.frame) <= instance._max_message_bytes * _OUTBOX_MESSAGES
        )

    @staticmethod
    def _is_drained(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, yamux.EndedData) and data.reason == "drained"

    @staticmethod
    def _is_oversize(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _BrokenData) and data.reason == "oversize"

    # -- timers -----------------------------------------------------------------------------------

    @staticmethod
    def _auth_timeout_value(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._auth_timeout

    # -- authenticating ---------------------------------------------------------------------------

    @staticmethod
    async def _run_authenticate(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """Read the first frame and verify its token for this peer; bounded by ``auth_timeout``."""

        del event
        try:
            message = await instance._socket.recv()
        except websockets.exceptions.ConnectionClosed:
            _ = hsm.dispatch(ctx, instance, _SocketClosedEvent.with_data(_SocketClosedData()))
            return
        verified = False
        if isinstance(message, str) and len(message) <= _MAX_TOKEN_MESSAGE:
            try:
                token = _TokenMessage.model_validate_json(message).token
            except pydantic.ValidationError:
                token = None
            if token is not None:
                try:
                    verified = instance._verifier.verify(instance._peer_id, token)
                except Exception as error:
                    # A verifier that cannot decide admits nobody.
                    _LOG.warning("websocket transport token verifier failed kind=%s", type(error).__name__)
        signal = _VerifiedEvent if verified else _RejectedEvent
        _ = hsm.dispatch(ctx, instance, signal.with_data(_SignalData()))

    @staticmethod
    def _announce(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """Tell the server this connection speaks for its peer and waits to be routed."""

        server = instance._server
        _ = hsm.dispatch(
            ctx,
            server,
            dataclasses.replace(
                AuthenticatedEvent.with_data(AuthenticatedData(peer_id=instance._peer_id, connection=instance)),
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(server),
                metadata=dict(event.metadata),
            ),
        )

    # -- the close code, one effect per reason ----------------------------------------------------

    @staticmethod
    def _close_unauthenticated(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._close_code = _UNAUTHENTICATED
        telemetry.reject("unauthenticated")
        telemetry.connect("failed", component=_COMPONENT)

    @staticmethod
    def _close_refused(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._close_code = _UNKNOWN_PEER
        telemetry.connect("failed", component=_COMPONENT)

    @staticmethod
    def _close_auth_timed_out(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._close_code = _AUTH_TIMEOUT
        telemetry.connect("timed_out", component=_COMPONENT)

    @staticmethod
    def _close_oversize(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._close_code = _OVERSIZE
        telemetry.reject("oversize")

    @staticmethod
    def _close_protocol_error(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._close_code = _PROTOCOL_ERROR
        telemetry.reject("protocol_error")

    @staticmethod
    def _close_drained(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._close_code = _NORMAL

    @staticmethod
    def _close_backlog(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._close_code = _BACKLOG
        telemetry.send_dropped("backlog")

    @staticmethod
    def _close_requested(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transport.ClosePeerData)
        instance._close_code = _SUPERSEDED if data.reason == "superseded" else _NORMAL

    # -- bound ------------------------------------------------------------------------------------

    @staticmethod
    def _bind(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, BoundData)
        instance._endpoint = data.endpoint
        instance._channels = {channel.name: channel for channel in data.channels}

    @staticmethod
    def _tell_endpoint(
        ctx: hsm.Context, instance: Connection, trigger: hsm.Event[typing.Any], told: hsm.Event[typing.Any]
    ) -> None:
        endpoint = instance._endpoint
        assert endpoint is not None
        _ = hsm.dispatch(
            ctx,
            endpoint,
            dataclasses.replace(told, id=uuid.uuid4().hex, source=hsm.id(instance), metadata=dict(trigger.metadata)),
        )

    @staticmethod
    def _connected(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """The peer is reachable: count it and offer this connection to the endpoint."""

        telemetry.connect("connected", component=_COMPONENT)
        telemetry.peer_active(1)
        Connection._tell_endpoint(
            ctx,
            instance,
            event,
            transport.PeerConnectedEvent.with_data(
                transport.PeerConnectedData(
                    peer_id=instance._peer_id,
                    connection_id=instance._connection_id,
                    transport=_TRANSPORT,
                    connection=instance,
                )
            ),
        )

    @staticmethod
    def _disconnected(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """The connection is gone, for whichever reason the trigger's payload type says; told once."""

        data = event.data
        reason: typing.Literal["left", "transport_lost", "closed"] = "closed"
        if isinstance(data, _SocketClosedData) or (isinstance(data, yamux.EndedData) and data.reason == "drained"):
            reason = "left"
        elif isinstance(data, _BrokenData | yamux.EndedData | yamux.TransmitData):
            reason = "transport_lost"
        telemetry.peer_active(-1)
        Connection._tell_endpoint(
            ctx,
            instance,
            event,
            transport.PeerDisconnectedEvent.with_data(
                transport.PeerDisconnectedData(
                    peer_id=instance._peer_id,
                    connection_id=instance._connection_id,
                    transport=_TRANSPORT,
                    reason=reason,
                )
            ),
        )
        instance._held.clear()
        instance._naming.clear()

    @staticmethod
    async def _run_read(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """Start the session, then decode each binary message into yamux frames for it, in order.

        Frames may span messages; the buffer holds at most one incomplete frame, bounded by
        ``max_message_bytes``. Each frame is awaited into the session, so a slow session slows the socket.
        """

        del event
        session = instance._session
        # The session lives as long as the connection (HSM-CONTEXT-001), not this activity.
        _ = await started(instance.context(), session, yamux.Server.model)
        instance._session_started = True
        socket = instance._socket
        limit = instance._max_message_bytes
        buffer = b""
        while True:
            try:
                message = await socket.recv()
            except websockets.exceptions.ConnectionClosed:
                _ = hsm.dispatch(ctx, instance, _SocketClosedEvent.with_data(_SocketClosedData()))
                return
            if not isinstance(message, bytes):
                _ = hsm.dispatch(ctx, instance, _BrokenEvent.with_data(_BrokenData(reason="protocol_error")))
                return
            telemetry.transferred(len(message), "in")
            buffer += message
            while len(buffer) >= yamux.HEADER_LENGTH:
                if _declared_data_length(buffer) > limit:
                    _ = hsm.dispatch(ctx, instance, _BrokenEvent.with_data(_BrokenData(reason="oversize")))
                    return
                try:
                    frame, buffer = yamux.decode_frame(buffer, max_data_length=limit)
                except yamux.FrameIncomplete:
                    break
                except yamux.FrameError:
                    _ = hsm.dispatch(ctx, instance, _BrokenEvent.with_data(_BrokenData(reason="protocol_error")))
                    return
                _ = await hsm.dispatch(ctx, session, yamux.event_from_frame(frame))

    @staticmethod
    def _enqueue(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """Queue one frame the session transmits; the write activity sends it."""

        del ctx
        data = event.data
        assert isinstance(data, yamux.TransmitData)
        instance._outbox_bytes += len(data.frame)
        instance._outbox.put_nowait(data.frame)

    @staticmethod
    async def _run_write(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """Send every queued frame as one binary message, in order."""

        del event
        outbox = instance._outbox
        while True:
            frame = await outbox.get()
            instance._outbox_bytes -= len(frame)
            try:
                await instance._socket.send(frame)
            except websockets.exceptions.ConnectionClosed:
                _ = hsm.dispatch(ctx, instance, _SocketClosedEvent.with_data(_SocketClosedData()))
                return
            telemetry.transferred(len(frame), "out")

    @staticmethod
    def _drop_stale(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        telemetry.send_dropped("stale_connection")

    # -- streams ----------------------------------------------------------------------------------

    @staticmethod
    def _tell_stream(
        ctx: hsm.Context,
        instance: Connection,
        trigger: hsm.Event[typing.Any],
        stream: hsm.Instance,
        told: hsm.Event[typing.Any],
    ) -> None:
        _ = hsm.dispatch(
            ctx,
            stream,
            dataclasses.replace(told, id=uuid.uuid4().hex, source=hsm.id(instance), metadata=dict(trigger.metadata)),
        )

    @staticmethod
    def _read_name(
        ctx: hsm.Context, instance: Connection, trigger: hsm.Event[typing.Any], stream: hsm.Instance
    ) -> None:
        """Take ``stream`` itself, with no channel, to read its first message."""

        instance._naming[hsm.id(stream)] = stream
        Connection._tell_stream(
            ctx,
            instance,
            trigger,
            stream,
            yamux.HandOverEvent.with_data(yamux.HandOverData(owner=instance, codec=_STREAM_OPEN)),
        )

    @staticmethod
    def _hold(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """Keep a stream opened before the endpoint accepted: no codec, its bytes unacknowledged."""

        del ctx
        data = event.data
        assert isinstance(data, yamux.StreamAcceptedData)
        instance._stream_count += 1
        instance._held.append(data.stream)

    @staticmethod
    def _name(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, yamux.StreamAcceptedData)
        instance._stream_count += 1
        Connection._read_name(ctx, instance, event, data.stream)

    @staticmethod
    def _name_held(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """The endpoint accepted: read the name of every stream held until now."""

        held, instance._held = instance._held, []
        for stream in held:
            Connection._read_name(ctx, instance, event, stream)

    @staticmethod
    def _refuse_extra_stream(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, yamux.StreamAcceptedData)
        telemetry.reject("too_many_channels")
        Connection._tell_stream(ctx, instance, event, data.stream, yamux.ResetEvent.with_data(yamux.ResetData()))

    @staticmethod
    def _hand_to_port(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """The stream named a registered channel: its owner takes it, with the channel's codec."""

        data = event.data
        assert isinstance(data, StreamOpenData)
        stream = instance._naming.pop(event.source)
        channel = instance._channels[data.channel]
        Connection._tell_stream(
            ctx,
            instance,
            event,
            stream,
            yamux.HandOverEvent.with_data(
                yamux.HandOverData(owner=channel.owner, codec=channel.codec, channel=channel.name)
            ),
        )

    @staticmethod
    def _refuse_channel(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """The stream named a channel the endpoint did not register: reset it."""

        stream = instance._naming.pop(event.source)
        telemetry.reject("refused_channel")
        Connection._tell_stream(ctx, instance, event, stream, yamux.ResetEvent.with_data(yamux.ResetData()))

    # -- closing ----------------------------------------------------------------------------------

    @staticmethod
    async def _run_close(ctx: hsm.Context, instance: Connection, event: hsm.Event[typing.Any]) -> None:
        """Send what is already queued (a session's last frame), unless the socket is stuck, then close."""

        del event
        socket = instance._socket
        if instance._close_code != _BACKLOG:
            while not instance._outbox.empty():
                frame = instance._outbox.get_nowait()
                instance._outbox_bytes -= len(frame)
                try:
                    await socket.send(frame)
                except websockets.exceptions.ConnectionClosed:
                    break
        await socket.close(instance._close_code)
        _ = hsm.dispatch(ctx, instance, _ClosedEvent.with_data(_SignalData()))

    model: typing.ClassVar[hsm.Model] = define(
        "Connection",
        hsm.initial(hsm.target("/Connection/authenticating")),
        hsm.state(
            "authenticating",
            hsm.activity(_run_authenticate),
            hsm.transition(hsm.on(_VerifiedEvent), hsm.effect(_announce), hsm.target("/Connection/awaiting_route")),
            hsm.transition(
                hsm.on(_RejectedEvent), hsm.effect(_close_unauthenticated), hsm.target("/Connection/closing")
            ),
            hsm.transition(hsm.on(_SocketClosedEvent), hsm.target("/Connection/closed")),
            hsm.transition(
                hsm.after(_auth_timeout_value), hsm.effect(_close_auth_timed_out), hsm.target("/Connection/closing")
            ),
        ),
        hsm.state(
            "awaiting_route",
            hsm.transition(hsm.on(BoundEvent), hsm.effect(_bind), hsm.target("/Connection/bound")),
            hsm.transition(hsm.on(RefusedEvent), hsm.effect(_close_refused), hsm.target("/Connection/closing")),
            hsm.transition(
                hsm.after(_auth_timeout_value), hsm.effect(_close_auth_timed_out), hsm.target("/Connection/closing")
            ),
        ),
        hsm.state(
            "bound",
            hsm.entry(_connected),
            hsm.exit(_disconnected),
            hsm.activity(_run_read, _run_write),
            hsm.initial(hsm.target("/Connection/bound/offered")),
            hsm.transition(hsm.on(yamux.TransmitEvent), hsm.guard(_outbox_has_room), hsm.effect(_enqueue)),
            hsm.transition(hsm.on(yamux.TransmitEvent), hsm.effect(_close_backlog), hsm.target("/Connection/closing")),
            hsm.transition(
                hsm.on(yamux.EndedEvent),
                hsm.guard(_is_drained),
                hsm.effect(_close_drained),
                hsm.target("/Connection/closing"),
            ),
            hsm.transition(
                hsm.on(yamux.EndedEvent), hsm.effect(_close_protocol_error), hsm.target("/Connection/closing")
            ),
            hsm.transition(
                hsm.on(transport.ClosePeerEvent),
                hsm.guard(_is_this_connection),
                hsm.effect(_close_requested),
                hsm.target("/Connection/closing"),
            ),
            hsm.transition(hsm.on(_SocketClosedEvent), hsm.target("/Connection/closed")),
            hsm.transition(
                hsm.on(_BrokenEvent),
                hsm.guard(_is_oversize),
                hsm.effect(_close_oversize),
                hsm.target("/Connection/closing"),
            ),
            hsm.transition(hsm.on(_BrokenEvent), hsm.effect(_close_protocol_error), hsm.target("/Connection/closing")),
            hsm.state(
                "offered",
                hsm.transition(hsm.on(yamux.StreamAcceptedEvent), hsm.guard(_has_room_for_stream), hsm.effect(_hold)),
                hsm.transition(hsm.on(yamux.StreamAcceptedEvent), hsm.effect(_refuse_extra_stream)),
                hsm.transition(
                    hsm.on(transport.AcceptPeerEvent),
                    hsm.guard(_is_this_connection),
                    hsm.effect(_name_held),
                    hsm.target("/Connection/bound/accepted"),
                ),
            ),
            hsm.state(
                "accepted",
                hsm.transition(hsm.on(yamux.StreamAcceptedEvent), hsm.guard(_has_room_for_stream), hsm.effect(_name)),
                hsm.transition(hsm.on(yamux.StreamAcceptedEvent), hsm.effect(_refuse_extra_stream)),
                hsm.transition(
                    hsm.on(StreamOpenEvent), hsm.guard(_names_registered_channel), hsm.effect(_hand_to_port)
                ),
                hsm.transition(hsm.on(StreamOpenEvent), hsm.guard(_is_naming), hsm.effect(_refuse_channel)),
            ),
        ),
        hsm.state(
            "closing",
            hsm.activity(_run_close),
            hsm.transition(hsm.on(yamux.TransmitEvent), hsm.effect(_drop_stale)),
            hsm.transition(hsm.on(_ClosedEvent), hsm.target("/Connection/closed")),
            # A close handshake the remote peer never finishes is bounded too.
            hsm.transition(hsm.after(_auth_timeout_value), hsm.target("/Connection/closed")),
        ),
        hsm.final("closed"),
        hsm.observe(observer),
    )


__all__ = ["Connection", "Socket"]
