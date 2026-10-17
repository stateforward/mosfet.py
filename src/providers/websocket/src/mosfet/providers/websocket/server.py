"""The WebSocket server remote peers connect to: a device-agnostic transport.

One ``Server`` per process, a shared actor (``runtime.Assembly.actors``): it listens, accepts each
socket on ``/v1/<peer_id>`` as a ``Connection`` (which owns its own server-role yamux session), and keeps
the route table of endpoints by peer. An endpoint (a device link) registers for one peer with
``attachment.attach`` carrying a ``transport.PeerAttachData`` — the peer id and the channel table its
streams are handed to — and unregisters with ``attachment.detach``. An authenticated connection for a
registered peer is bound to that endpoint (``providers.websocket.bound``); one for any other peer is
refused (``4404``). "Registered" is decided only by the server's own route table.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import logging
import re
import ssl
import typing
import uuid

import hsm
import mosfet.lifecycle
import pydantic
import websockets.asyncio.server
from pydantic.json_schema import SkipJsonSchema

from mosfet.define import define
from mosfet.media import MAX_IMAGE_BYTES
from mosfet.protocols import attachment, transport, yamux
from mosfet.start import started
from mosfet.telemetry import observer
from mosfet.telemetry.hsm import Traced

from . import telemetry
from .connection import Connection, Socket
from .events import AuthenticatedData, AuthenticatedEvent, BoundData, BoundEvent, RefusedData, RefusedEvent
from .tokens import TokenVerifier

_LOG = logging.getLogger(__name__)
_PATH = re.compile(r"^/v1/(?P<peer_id>[^/]+)$")
_PEER_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(transport.PeerId)
_UNKNOWN_PEER = 4404

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765
_DEFAULT_START_TIMEOUT = datetime.timedelta(seconds=10)
_DEFAULT_AUTH_TIMEOUT = datetime.timedelta(seconds=5)
_DEFAULT_MAX_MESSAGE_BYTES = MAX_IMAGE_BYTES + 64 * 1024
_DEFAULT_MAX_CHANNELS = 8

type Handler = collections.abc.Callable[[Socket, str], collections.abc.Awaitable[None]]
"""Serves one accepted socket, given the request path, until the socket closes."""


class Listener(typing.Protocol):
    """A listening WebSocket server, as ``Server`` holds it (``websockets``' ``Server`` is one)."""

    def close(self) -> None:
        """Stop accepting and close every open socket."""
        ...

    def wait_closed(self) -> collections.abc.Awaitable[None]:
        """Resolve once every socket's handler has returned."""
        ...


type Serve = collections.abc.Callable[[Handler], collections.abc.Awaitable[Listener]]
"""Start listening, handing each accepted socket and its request path to the handler."""


class _ListeningData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    listener: SkipJsonSchema[pydantic.SkipValidation[Listener]] = pydantic.Field(
        exclude=True, repr=False, description="The listening server. Runtime-only."
    )


class _ListenFailedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    kind: str = pydantic.Field(description="The exception type that stopped the server from listening.")


class _AcceptedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    connection: SkipJsonSchema[pydantic.SkipValidation[Connection]] = pydantic.Field(
        exclude=True, repr=False, description="The accepted socket's connection. Runtime-only."
    )


# Private: each is dispatched by this class's own activities and socket handler to itself only.
_ListeningEvent = hsm.Event[_ListeningData](
    name="providers.websocket.server.listening", kind=hsm.CompletionEventKind, schema=_ListeningData
)
_ListenFailedEvent = hsm.Event[_ListenFailedData](
    name="providers.websocket.server.listen_failed", kind=hsm.ErrorEventKind, schema=_ListenFailedData
)
_AcceptedEvent = hsm.Event[_AcceptedData](name="providers.websocket.server.accepted", schema=_AcceptedData)
_ReleasedEvent = hsm.Event[_AcceptedData](name="providers.websocket.server.released", schema=_AcceptedData)


class Server(Traced, attachment.Attachment):
    """Listens for remote peers and binds each authenticated connection to its peer's endpoint.

    It belongs in ``runtime.Assembly.actors``: started before the bots, stopped after them. Stopping it
    closes the listener and every open socket, and stops every connection it started.

    An endpoint registers one peer with ``attachment.attach`` carrying a ``transport.PeerAttachData``,
    answered ``attach.complete`` (``created``, or not when it already holds that peer) or ``attach.failed``
    (``conflict``) when another endpoint holds the peer, the request is not a ``PeerAttachData``, or the
    server failed to listen. Registrations before the server listens wait for it. ``attachment.detach``
    withdraws the endpoint's route (``removed``, or not when it held none).
    """

    _attachment_limit: typing.ClassVar[int | None] = None
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str

    _verifier: TokenVerifier
    _start_timeout: datetime.timedelta
    _auth_timeout: datetime.timedelta
    _max_message_bytes: int
    _max_channels: int
    _serve: Serve
    # The listener, once listening; written only by _keep_listener.
    _listener: Listener | None
    # Peer id → the endpoint registered for it and its channel table; written only by _route / _unroute.
    _routes: dict[str, tuple[hsm.Instance, tuple[transport.ChannelData, ...]]]
    # Every open socket's connection; written only by _record / _forget.
    _connections: set[Connection]

    def __init__(
        self,
        *,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        verifier: TokenVerifier,
        ssl: ssl.SSLContext | None = None,
        start_timeout: datetime.timedelta = _DEFAULT_START_TIMEOUT,
        auth_timeout: datetime.timedelta = _DEFAULT_AUTH_TIMEOUT,
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
        max_channels: int = _DEFAULT_MAX_CHANNELS,
        serve: Serve | None = None,
    ) -> None:
        """Build an unstarted server for ``host``:``port`` (``ssl`` for ``wss``).

        It listens on loopback, ``127.0.0.1:8765``, unless told otherwise; ``host="0.0.0.0"`` lets other
        machines reach it. ``verifier`` has no default: it holds the peers' credential.
        ``auth_timeout`` bounds a connection's token, its routing, and its closing handshake.
        ``max_message_bytes`` bounds one WebSocket message and one yamux data frame. ``max_channels``
        bounds the streams one connection opens. ``serve`` starts the listener (default: ``websockets``
        on ``host``:``port``).

        Raises:
            ValueError: If ``port`` is outside 0..65535, a timeout is not positive, or
                ``max_message_bytes`` or ``max_channels`` is not positive.
        """

        super().__init__()
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535.")
        for name, value in (("start_timeout", start_timeout), ("auth_timeout", auth_timeout)):
            if value <= datetime.timedelta():
                raise ValueError(f"{name} must be a positive duration.")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive.")
        if max_channels <= 0:
            raise ValueError("max_channels must be positive.")
        tls = ssl

        async def default_serve(handler: Handler) -> Listener:
            async def accept(socket: websockets.asyncio.server.ServerConnection) -> None:
                request = socket.request
                await handler(socket, request.path if request is not None else "")

            return await websockets.asyncio.server.serve(
                accept,
                host,
                port,
                ssl=tls,
                compression=None,
                max_size=max_message_bytes + yamux.HEADER_LENGTH,
                close_timeout=auth_timeout.total_seconds(),
            )

        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self._verifier = verifier
        self._start_timeout = start_timeout
        self._auth_timeout = auth_timeout
        self._max_message_bytes = max_message_bytes
        self._max_channels = max_channels
        self._serve = serve if serve is not None else default_serve
        self._listener = None
        self._routes = {}
        self._connections = set()

    @typing.override
    def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> collections.abc.Awaitable[bool]:
        """Deliver one ``attachment.attach`` request; resolves to whether it was delivered.

        The answer goes to the request's ``reply_to`` (else its ``actor``), never a return value.
        """

        return self.dispatch(ctx, event)

    @typing.override
    def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> collections.abc.Awaitable[bool]:
        """Deliver one ``attachment.detach`` request; resolves to whether it was delivered."""

        return self.dispatch(ctx, event)

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Close the listener and every socket, then stop; no connection outlives the server.

        Closing the listener ends each socket's handler, which stops its connection while the server
        still runs. Any the handlers have not stopped yet are stopped after it, so none is left started
        under the server's cancelled context.
        """

        listener = self._listener
        if listener is not None:
            listener.close()
            await listener.wait_closed()
        await super().stop(ctx)
        # Teardown only: whatever a handler has not stopped yet goes with the server.
        for connection in tuple(self._connections):
            if mosfet.lifecycle.is_started(connection):
                await hsm.stop(connection)

    # -- guards -----------------------------------------------------------------------------------

    @staticmethod
    def _routes_free_peer(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> bool:
        """A registration for a peer no endpoint holds, from an endpoint holding none."""

        data = event.data
        return (
            isinstance(data, transport.PeerAttachData)
            and data.peer_id not in instance._routes
            and attachment.Attachment._can_attach(ctx, instance, event)
        )

    @staticmethod
    def _routes_same_endpoint(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> bool:
        """A registration repeated by the endpoint that already holds the peer."""

        del ctx
        data = event.data
        if not isinstance(data, transport.PeerAttachData):
            return False
        route = instance._routes.get(data.peer_id)
        return route is not None and route[0] is data.actor

    @staticmethod
    def _is_routed(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, AuthenticatedData)
            and data.peer_id in instance._routes
            and isinstance(data.connection, Connection)
            and data.connection in instance._connections
        )

    # -- timers -----------------------------------------------------------------------------------

    @staticmethod
    def _start_timeout_value(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._start_timeout

    # -- starting ---------------------------------------------------------------------------------

    @staticmethod
    def _handler(instance: Server) -> Handler:
        """Serve one accepted socket: a connection for as long as the socket is open."""

        async def handle(socket: Socket, path: str) -> None:
            match = _PATH.match(path)
            try:
                peer_id = _PEER_ID.validate_python(match["peer_id"] if match is not None else "")
            except pydantic.ValidationError:
                telemetry.reject("unknown_peer")
                await socket.close(_UNKNOWN_PEER)
                return
            connection = Connection(
                socket=socket,
                peer_id=peer_id,
                server=instance,
                verifier=instance._verifier,
                auth_timeout=instance._auth_timeout,
                max_message_bytes=instance._max_message_bytes,
                max_channels=instance._max_channels,
            )
            accepted = _AcceptedData(connection=connection)
            # Recorded before the connection starts, so its authentication always finds it recorded.
            _ = await hsm.dispatch(instance.context(), instance, _AcceptedEvent.with_data(accepted))
            try:
                # It lives as long as the socket: started under the server's own context
                # (HSM-CONTEXT-001), stopped here when the socket closes.
                _ = await started(instance.context(), connection, Connection.model)
                await socket.wait_closed()
            finally:
                if mosfet.lifecycle.is_started(connection):
                    await hsm.stop(connection)
                # Teardown only: a server already stopped has forgotten every socket.
                if mosfet.lifecycle.is_started(instance):
                    _ = await hsm.dispatch(instance.context(), instance, _ReleasedEvent.with_data(accepted))

        return handle

    @staticmethod
    async def _run_start(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        """Start listening; bounded by ``start_timeout``."""

        del event
        try:
            listener = await instance._serve(Server._handler(instance))
        except Exception as error:
            _ = hsm.dispatch(ctx, instance, _ListenFailedEvent.with_data(_ListenFailedData(kind=type(error).__name__)))
            return
        _ = hsm.dispatch(ctx, instance, _ListeningEvent.with_data(_ListeningData(listener=listener)))

    @staticmethod
    def _keep_listener(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _ListeningData)
        instance._listener = data.listener

    @staticmethod
    def _log_listen_failed(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        data = event.data
        assert isinstance(data, _ListenFailedData)
        _LOG.error("websocket transport server could not listen kind=%s", data.kind)

    @staticmethod
    def _log_start_timed_out(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _LOG.error(
            "websocket transport server did not start listening within %ss", instance._start_timeout.total_seconds()
        )

    # -- listening --------------------------------------------------------------------------------

    @staticmethod
    def _record(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _AcceptedData)
        instance._connections.add(data.connection)

    @staticmethod
    def _forget(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _AcceptedData)
        instance._connections.discard(data.connection)

    @staticmethod
    def _route(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transport.PeerAttachData)
        instance._routes[data.peer_id] = (data.actor, data.channels)

    @staticmethod
    def _unroute(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        """Withdraw every route the detaching endpoint holds."""

        del ctx
        data = event.data
        assert isinstance(data, attachment.DetachData)
        for peer_id in [peer_id for peer_id, (endpoint, _) in instance._routes.items() if endpoint is data.actor]:
            del instance._routes[peer_id]

    @staticmethod
    def _bind(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        """Bind the connection to its peer's endpoint, handing it the endpoint's channel table."""

        data = event.data
        assert isinstance(data, AuthenticatedData) and isinstance(data.connection, Connection)
        endpoint, channels = instance._routes[data.peer_id]
        connection = data.connection
        _ = hsm.dispatch(
            ctx,
            connection,
            dataclasses.replace(
                BoundEvent.with_data(BoundData(endpoint=endpoint, channels=channels)),
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(connection),
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _refuse(ctx: hsm.Context, instance: Server, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, AuthenticatedData)
        connection = data.connection
        telemetry.reject("unknown_peer")
        _ = hsm.dispatch(
            ctx,
            connection,
            dataclasses.replace(
                RefusedEvent.with_data(RefusedData()),
                id=uuid.uuid4().hex,
                source=hsm.id(instance),
                target=hsm.id(connection),
                metadata=dict(event.metadata),
            ),
        )

    model: typing.ClassVar[hsm.Model] = define(
        "Server",
        hsm.initial(hsm.target("/Server/starting")),
        hsm.state(
            "starting",
            hsm.activity(_run_start),
            # Endpoints may register, and the first sockets arrive, before the listener is recorded.
            hsm.defer(
                attachment.AttachEvent, attachment.DetachEvent, AuthenticatedEvent, _AcceptedEvent, _ReleasedEvent
            ),
            hsm.transition(hsm.on(_ListeningEvent), hsm.effect(_keep_listener), hsm.target("/Server/listening")),
            hsm.transition(hsm.on(_ListenFailedEvent), hsm.effect(_log_listen_failed), hsm.target("/Server/failed")),
            hsm.transition(
                hsm.after(_start_timeout_value), hsm.effect(_log_start_timed_out), hsm.target("/Server/failed")
            ),
        ),
        hsm.state(
            "listening",
            hsm.transition(hsm.on(_AcceptedEvent), hsm.effect(_record)),
            hsm.transition(hsm.on(_ReleasedEvent), hsm.effect(_forget)),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(_routes_free_peer),
                hsm.effect(
                    attachment.Attachment._attach, _route, attachment.Attachment._dispatch_attach_complete_created
                ),
            ),
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(_routes_same_endpoint),
                hsm.effect(attachment.Attachment._dispatch_attach_complete_existing),
            ),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_attached),
                hsm.effect(
                    _unroute, attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed
                ),
            ),
            hsm.transition(hsm.on(AuthenticatedEvent), hsm.guard(_is_routed), hsm.effect(_bind)),
            hsm.transition(hsm.on(AuthenticatedEvent), hsm.effect(_refuse)),
        ),
        # The server never listened: every registration is answered failed.
        hsm.state("failed"),
        hsm.transition(
            hsm.on(attachment.AttachEvent),
            hsm.guard(attachment.Attachment._is_attach_request),
            hsm.effect(attachment.Attachment._dispatch_attach_failed),
        ),
        hsm.transition(
            hsm.on(attachment.DetachEvent),
            hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
        ),
        hsm.observe(observer),
    )
    """Starting, then listening or failed; registrations and connections are routed only while listening."""


__all__ = ["Handler", "Listener", "Serve", "Server"]
