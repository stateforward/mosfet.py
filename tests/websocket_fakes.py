"""Shared fakes for the WebSocket transport tests: a socket, a listener, and a browser.

``FakeSocket`` is the runtime's end of one WebSocket, in process. ``FakeServe`` stands in for
``websockets.serve``: the test hands it sockets as if remote machines connected. ``Browser`` plays the
remote machine on the far end of a ``FakeSocket``: it sends the token frame, then is the connection of a
client-role ``yamux.Client`` over that socket and the owner of every stream it opens. Its streams carry
raw messages (``RawCodec``), so a test writes the stream-opening message and the channel's messages as
bytes, and reads back what the runtime sends as bytes.
"""

from __future__ import annotations

import asyncio
import json
import typing

import hsm
import mosfet
import mosfet.lifecycle
import pydantic
import websockets.exceptions

from mosfet.protocols import transform, yamux
from mosfet.providers import websocket


class RawData(pydantic.BaseModel):
    """One whole stream message, as bytes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    message: bytes


RawEvent = hsm.Event[RawData](name="test.websocket.raw", schema=RawData)


class RawCodec:
    """The browser's codec: every message is one ``RawEvent`` carrying its bytes, both ways."""

    name: str = "raw"
    max_message_bytes: int = 1 << 20
    encodes_any: bool = True

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        return RawEvent.with_data(RawData(message=message))

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        return event.name == RawEvent.name

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        data = event.data
        assert isinstance(data, RawData)
        return data.message


def stream_open(channel: str) -> bytes:
    """The stream-opening message naming ``channel``."""

    return websocket.StreamOpenData(channel=channel, v=1).model_dump_json().encode()


def _closed() -> websockets.exceptions.ConnectionClosedOK:
    return websockets.exceptions.ConnectionClosedOK(None, None)


class FakeSocket:
    """The runtime's end of one WebSocket. The test (or a ``Browser``) plays the other end."""

    close_codes: list[int]
    sent: asyncio.Queue[bytes]
    _inbound: asyncio.Queue[str | bytes | None]
    _closed_event: asyncio.Event
    _send_gate: asyncio.Event | None

    def __init__(self, *, send_gate: asyncio.Event | None = None) -> None:
        self.close_codes = []
        self.sent = asyncio.Queue()
        self._inbound = asyncio.Queue()
        self._closed_event = asyncio.Event()
        self._send_gate = send_gate

    # -- the runtime's side -------------------------------------------------------------------

    async def recv(self) -> str | bytes:
        if self._closed_event.is_set():
            raise _closed()
        message = await self._inbound.get()
        if message is None:
            self._closed_event.set()
            raise _closed()
        return message

    async def send(self, message: bytes) -> None:
        if self._send_gate is not None:
            _ = await self._send_gate.wait()
        if self._closed_event.is_set():
            raise _closed()
        self.sent.put_nowait(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del reason
        self.close_codes.append(code)
        self._closed_event.set()
        self._inbound.put_nowait(None)

    async def wait_closed(self) -> None:
        _ = await self._closed_event.wait()

    # -- the remote machine's side ------------------------------------------------------------

    def deliver(self, message: str | bytes) -> None:
        """The remote machine sends one whole message."""

        self._inbound.put_nowait(message)

    def hang_up(self) -> None:
        """The remote machine closes the socket."""

        self._closed_event.set()
        self._inbound.put_nowait(None)

    def is_closed(self) -> bool:
        return self._closed_event.is_set()


class FakeListener:
    """What ``FakeServe`` returns: closing it closes every socket it accepted (``1001``)."""

    closed: bool
    _sockets: list[FakeSocket]
    _handlers: set[asyncio.Task[None]]

    def __init__(self) -> None:
        self.closed = False
        self._sockets = []
        self._handlers = set()

    def track(self, socket: FakeSocket, handler: asyncio.Task[None]) -> None:
        self._sockets.append(socket)
        self._handlers.add(handler)
        handler.add_done_callback(self._handlers.discard)

    def close(self) -> None:
        self.closed = True
        for socket in self._sockets:
            if not socket.is_closed():
                socket.close_codes.append(1001)
                socket.hang_up()

    async def wait_closed(self) -> None:
        if self._handlers:
            _ = await asyncio.wait(set(self._handlers))


class FakeServe:
    """Stands in for ``websockets.serve``; ``connect`` plays a remote machine opening a socket."""

    listener: FakeListener
    handler: websocket.Handler | None
    calls: int
    _gate: asyncio.Event | None
    _error: BaseException | None

    def __init__(self, *, gate: asyncio.Event | None = None, error: BaseException | None = None) -> None:
        self.listener = FakeListener()
        self.handler = None
        self.calls = 0
        self._gate = gate
        self._error = error

    async def __call__(self, handler: websocket.Handler) -> FakeListener:
        self.calls += 1
        if self._gate is not None:
            _ = await self._gate.wait()
        if self._error is not None:
            raise self._error
        self.handler = handler
        return self.listener

    def connect(self, path: str) -> FakeSocket:
        """Accept one socket on ``path``; its handler runs as ``websockets`` would run it."""

        handler = self.handler
        assert handler is not None, "the server is not listening"
        socket = FakeSocket()

        async def serve_socket() -> None:
            await handler(socket, path)

        task = asyncio.get_running_loop().create_task(serve_socket())
        self.listener.track(socket, task)
        return socket


class Browser(hsm.Instance):
    """A remote machine on the far end of ``socket``: a token, then a client-role yamux session.

    It is the session's connection (its frames go out on the socket) and the owner of every stream it
    opens, so what the runtime sends on a stream comes back to it as ``RawEvent``s.
    """

    socket: FakeSocket
    session: yamux.Client
    streams: dict[str, hsm.Instance]
    received: dict[str, list[bytes]]
    ended: dict[str, list[str]]
    session_ended: list[str]
    _channels: dict[str, str]
    _pump: asyncio.Task[None] | None

    def __init__(self, socket: FakeSocket) -> None:
        super().__init__()
        self.socket = socket
        self.session = yamux.Client(connection=self)
        self.streams = {}
        self.received = {}
        self.ended = {}
        self.session_ended = []
        self._channels = {}
        self._pump = None

    @staticmethod
    def _carry(ctx: hsm.Context, instance: Browser, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, yamux.TransmitData)
        instance.socket.deliver(data.frame)

    @staticmethod
    def _owned(ctx: hsm.Context, instance: Browser, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transform.OwnedData)
        instance.streams[data.channel] = data.stream
        instance._channels[hsm.id(data.stream)] = data.channel

    @staticmethod
    def _receive(ctx: hsm.Context, instance: Browser, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, RawData)
        instance.received.setdefault(instance._channels[event.source], []).append(data.message)

    @staticmethod
    def _stream_ended(ctx: hsm.Context, instance: Browser, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, transform.EndedData)
        instance.ended.setdefault(data.channel, []).append(data.reason)

    @staticmethod
    def _session_ended(ctx: hsm.Context, instance: Browser, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, yamux.EndedData)
        instance.session_ended.append(data.reason)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Browser",
        hsm.initial(hsm.target("browsing")),
        hsm.state(
            "browsing",
            hsm.transition(hsm.on(yamux.TransmitEvent), hsm.effect(_carry)),
            hsm.transition(hsm.on(transform.OwnedEvent), hsm.effect(_owned)),
            hsm.transition(hsm.on(RawEvent), hsm.effect(_receive)),
            hsm.transition(hsm.on(transform.EndedEvent), hsm.effect(_stream_ended)),
            hsm.transition(hsm.on(yamux.EndedEvent), hsm.effect(_session_ended)),
        ),
    )

    async def connect(self, token: str) -> None:
        """Send the token frame, then start the browser and its yamux session over the socket."""

        self.socket.deliver(json.dumps({"token": token}))
        _ = await mosfet.started(None, self, Browser.model)
        _ = await mosfet.started(None, self.session, yamux.Client.model)
        self._pump = asyncio.get_running_loop().create_task(self._pump_in())

    async def _pump_in(self) -> None:
        buffer = b""
        while True:
            buffer += await self.socket.sent.get()
            while True:
                try:
                    frame, buffer = yamux.decode_frame(buffer)
                except yamux.FrameIncomplete:
                    break
                _ = await hsm.dispatch(self.session.context(), self.session, yamux.event_from_frame(frame))

    async def open(self, channel: str, *, first: bytes | None = None) -> hsm.Instance:
        """Open a stream (named ``channel`` on this side) and send its first message: by default, its name."""

        _ = await hsm.dispatch(
            self.session.context(),
            self.session,
            yamux.OpenStreamEvent.with_data(yamux.OpenStreamData(owner=self, codec=RawCodec(), channel=channel)),
        )
        await _wait_for(lambda: channel in self.streams)
        stream = self.streams[channel]
        await self.send(channel, stream_open(channel) if first is None else first)
        return stream

    async def send(self, channel: str, message: bytes) -> None:
        """Send one message on the stream opened as ``channel``."""

        stream = self.streams[channel]
        _ = await hsm.dispatch(stream.context(), stream, RawEvent.with_data(RawData(message=message)))

    async def close(self) -> None:
        """Stop the pump, the session and the browser; the socket is left as it is."""

        pump, self._pump = self._pump, None
        if pump is not None:
            _ = pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        for machine in (self.session, self):
            if mosfet.lifecycle.is_started(machine):
                await hsm.stop(machine)


async def _wait_for(condition: typing.Callable[[], bool], *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() >= deadline:
            raise TimeoutError("Timed out waiting for condition.")
        await asyncio.sleep(0)
