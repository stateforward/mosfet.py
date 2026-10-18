"""The far side of the builder's computer, in a terminal: the person's screen, keyboard and pointer.

    builder-terminal [--url ws://127.0.0.1:8765] [--peer builder-desk] [--secret-file PATH]

The terminal is a remote machine as the websocket transport's version 1 wire describes one. It connects to
``<url>/v1/<peer>``, sends its token (``websocket.HmacTokens(secret).issue(peer)``), and is the client role
of one yamux session, on which it opens three channels: ``keyboard``, ``pointer`` and ``display``. Each
channel's first message is a ``websocket.StreamOpenData``; after it, the channel carries the core JSON
events (``hid.KeyEvent``, ``hid.PointerEvent``, ``computer.ContentEvent`` and ``computer.LayoutEvent``).
The terminal defines no wire type of its own. The camera, microphone and speaker are not opened.

What the screen shows is what the computer shows: every ``computer.display.content`` is drawn top to bottom,
and the terminal answers each with a ``computer.display.layout`` of one row per element, so the computer's
app can tell what a click landed on. A line the person types is typed key by key, then Enter. A line
longer than :data:`MAX_LINE` characters is not typed at all, and the terminal says so: its keys would not
all fit the keyboard channel at once. ``/click <element id>`` presses and releases the pointer at the
centre of that element's row. ``/quit``, or the end of input, leaves. The computer's app decides what every
key and click does; the terminal decides nothing. A line typed while nothing on the screen is focused goes to
the computer like any other, and reaches whoever holds it with no field.

The secret is ``computer_token_secret``, resolved through ``runtime.Secrets``: the Docker secret
``/run/secrets/computer_token_secret``, then the file ``--secret-file`` names (or ``COMPUTER_TOKEN_SECRET_FILE``),
then ``COMPUTER_TOKEN_SECRET``. Exit codes: ``0`` when the person leaves or the computer closes the session
normally (``1000``, or ``1001`` when the bot's process goes away), ``1`` when the connection fails, times out,
is refused (``4401``, ``4404``, ``4408``, ``4409``) or ends abnormally, ``64`` for bad flags, ``78`` for a
missing or unreadable secret. The screen goes to stdout; the terminal's own notices and diagnostics go to
stderr.

Telemetry (meter scope ``builder_bot_example.terminal``), every attribute a closed set from code:
``bot.example.terminal.connect.count{outcome}``, ``bot.example.terminal.sent.count{channel}``,
``bot.example.terminal.render.count``, ``bot.example.terminal.closed.count{reason}`` and one
``bot.example.terminal.session`` span per session. Typed text, element ids, the peer, the token and the url
never are attributes.
"""

import argparse
import asyncio
import codecs
import collections.abc
import datetime
import json
import logging
import os
import pathlib
import sys
import threading
import typing
import urllib.parse

import hsm
import mosfet
import pydantic
import websockets.asyncio.client
import websockets.exceptions
from mosfet import runtime, telemetry
from mosfet.devices import computer, hid
from mosfet.protocols import transform, transport, yamux
from mosfet.providers import websocket
from mosfet.telemetry import span
from mosfet.telemetry.hsm import Traced
from opentelemetry import metrics
from pydantic.json_schema import SkipJsonSchema

SCOPE: typing.Final = "builder_bot_example.terminal"
SCREEN_RULE: typing.Final = "─" * 40 + "\n"
"""The line drawn above every screen the terminal shows."""

type Reason = typing.Literal["quit", "normal", "refused", "abnormal", "timeout"]
type ConnectOutcome = typing.Literal["linked", "refused", "timeout", "failed"]
type Channel = typing.Literal["keyboard", "pointer", "display"]

_LOG = logging.getLogger(__name__)
_METER = metrics.get_meter(SCOPE)
_CONNECTS = _METER.create_counter(
    "bot.example.terminal.connect.count",
    unit="{session}",
    description="Terminal sessions by how connecting ended: linked, refused, timeout, or failed.",
)
_SENT = _METER.create_counter(
    "bot.example.terminal.sent.count",
    unit="{event}",
    description="Events the terminal sent its computer, by channel: keyboard, pointer, or display.",
)
_RENDERS = _METER.create_counter(
    "bot.example.terminal.render.count",
    unit="{screen}",
    description="Screens the terminal drew from the computer's display content.",
)
_CLOSED = _METER.create_counter(
    "bot.example.terminal.closed.count",
    unit="{session}",
    description="Terminal sessions by why they closed: quit, normal, refused, abnormal, or timeout.",
)

_DEFAULT_URL = "ws://127.0.0.1:8765"
_DEFAULT_PEER = "builder-desk"
_SECRET = "computer_token_secret"
_SECRETS_DIRECTORY = pathlib.Path("/run/secrets")
_DEFAULT_TIMEOUT = datetime.timedelta(seconds=10)
_NORMAL_CODES = frozenset({1000, 1001})
_REFUSED_CODES = frozenset({4401, 4404, 4408, 4409})
_ABNORMAL_CLOSURE = 1006
_CHANNELS: tuple[Channel, ...] = ("keyboard", "pointer", "display")
# The most one key event takes on the keyboard channel: a 4-byte length, then at most 124 bytes of JSON (a
# control character, escaped as six bytes in both `key` and `text`). Enter takes as much.
_KEY_BYTES = 128
MAX_LINE: typing.Final = 2 * yamux.INITIAL_STREAM_WINDOW // _KEY_BYTES - 1
"""The longest line the terminal types, in characters: 4095.

A line is typed all at once, so its keys and its Enter must fit what one yamux stream holds before the
computer takes any of them: the computer's receive window (at least ``yamux.INITIAL_STREAM_WINDOW``) and
as much again waiting on this side. Past that the stream drops keys. A longer line is not typed.
"""
_READ_CHUNK = 4096
_PEER_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(transport.PeerId)

_USAGE = 64
_CONFIG = 78
_INTERRUPTED = 130
_BOLD, _DIM, _PLAIN = "\x1b[1m", "\x1b[2m", "\x1b[0m"
_CLEAR = "\x1b[H\x1b[2J"


class Socket(typing.Protocol):
    """The terminal's end of its WebSocket, as it uses it (``websockets``' ``ClientConnection`` is one).

    ``recv`` and ``send`` raise ``websockets.exceptions.ConnectionClosed`` once the socket is closed.
    """

    def recv(self) -> collections.abc.Awaitable[str | bytes]:
        """The next whole message: text as ``str``, binary as ``bytes``."""
        ...

    def send(self, message: str | bytes) -> collections.abc.Awaitable[None]:
        """Send one message: ``str`` as text, ``bytes`` as binary."""
        ...

    def close(self, code: int = 1000, reason: str = "") -> collections.abc.Awaitable[None]:
        """Close the socket with ``code`` and wait for the closing handshake."""
        ...


type Connect = collections.abc.Callable[[str], collections.abc.Awaitable[Socket]]
"""Open the WebSocket at a ``ws://`` URI. The terminal bounds it with its own ``connect_timeout``."""

type Lines = collections.abc.Callable[[], collections.abc.Awaitable[str | None]]
"""The next line the person typed, without its line ending, or ``None`` once their input has ended."""


async def connect_websocket(uri: str) -> Socket:
    """Open ``uri`` with ``websockets``; every bound and keepalive is the terminal's, so the library sets none."""

    return await websockets.asyncio.client.connect(uri, open_timeout=None, ping_interval=None)


class ClosedData(pydantic.BaseModel):
    """How one terminal session ended, told once to whoever started the terminal."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"reason": "quit", "code": None, "exit_code": 0},
                {"reason": "refused", "code": 4401, "exit_code": 1},
            ]
        },
    )

    reason: Reason = pydantic.Field(
        description=(
            "Why: the person left (`quit`), the computer closed the session normally (`normal`), the computer "
            "refused this terminal (`refused`), the connection failed or broke (`abnormal`), or connecting or "
            "the computer's first screen took too long (`timeout`)."
        ),
        examples=["quit", "refused"],
    )
    code: int | None = pydantic.Field(
        description="The WebSocket close code the computer sent, when the computer closed the socket.",
        examples=[None, 1000, 4401],
    )
    exit_code: typing.Literal[0, 1] = pydantic.Field(
        description="The process exit code: 0 for `quit` and `normal`, 1 otherwise.",
        examples=[0, 1],
    )


def render(content: computer.ContentData) -> str:
    """The screen for ``content``, top to bottom: one block per element.

    A heading is bold, a caption dim, code verbatim and a paragraph plain. A field shows its id and value
    (its placeholder in parentheses while empty) and ``[focused]`` when keys type into it. A button shows
    ``[label]`` and its id, which is what ``/click`` takes.
    """

    blocks: list[str] = []
    for element in content.elements:
        match element:
            case computer.TextElement(role="heading"):
                blocks.append(f"{_BOLD}{element.text}{_PLAIN}")
            case computer.TextElement(role="caption"):
                blocks.append(f"{_DIM}{element.text}{_PLAIN}")
            case computer.TextElement():
                blocks.append(element.text)
            case computer.FieldElement():
                shown = element.value or (f"({element.placeholder})" if element.placeholder else "")
                blocks.append(f"{element.id}: {shown}" + (" [focused]" if element.focused else ""))
            case computer.ButtonElement():
                blocks.append(f"[{element.label}] ({element.id})")
    return "".join(f"{block}\n" for block in blocks)


def layout(content: computer.ContentData) -> computer.LayoutData:
    """Where :func:`render` drew each element: one full-width row each, top to bottom, sharing the height."""

    count = len(content.elements)
    return computer.LayoutData(
        boxes=tuple(
            computer.BoxData(id=element.id, x=0.0, y=index / count, width=1.0, height=1 / count)
            for index, element in enumerate(content.elements)
        )
    )


class _ChannelCodec:
    """One channel's JSON codec on this side, which also writes the stream's first message.

    The first message is ``websocket.StreamOpenData`` as compact JSON; every other message is the channel's
    core JSON codec, both ways.
    """

    name: str
    max_message_bytes: int
    encodes_any: bool
    _channel: transform.JsonCodec

    def __init__(self, channel: transform.JsonCodec) -> None:
        self.name = channel.name
        self.max_message_bytes = channel.max_message_bytes
        self.encodes_any = True
        self._channel = channel

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """The channel's event ``message`` carries.

        Raises:
            transform.DecodeError: as the channel's JSON codec raises it.
        """

        return self._channel.decode(message)

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """The stream-opening message, or an event the channel carries: a pure lookup by name."""

        return event.name == websocket.StreamOpenEvent.name or self._channel.encodes(event)

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """The message for ``event``.

        Raises:
            transform.EncodeError: as the channel's JSON codec raises it.
        """

        if isinstance(event.data, websocket.StreamOpenData):
            return event.data.model_dump_json().encode()
        return self._channel.encode(event)


_CODECS: dict[Channel, _ChannelCodec] = {
    "keyboard": _ChannelCodec(transform.JsonCodec(hid.KeyEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES)),
    "pointer": _ChannelCodec(transform.JsonCodec(hid.PointerEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES)),
    "display": _ChannelCodec(
        transform.JsonCodec(computer.ContentEvent, computer.LayoutEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES)
    ),
}


class _SignalData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


class _ConnectedData(pydantic.BaseModel):
    """The socket is open, the token sent and the yamux session started."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    socket: SkipJsonSchema[pydantic.SkipValidation[Socket]] = pydantic.Field(
        exclude=True, repr=False, description="The open socket. Runtime-only."
    )


class _SocketClosedData(pydantic.BaseModel):
    """The socket closed; ``code`` is the close code the computer sent (``1006`` when it sent none)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    code: int


class _LineData(pydantic.BaseModel):
    """One line the person typed, without its line ending."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    text: str


# Private: each is dispatched by this class's own activities to itself only.
_ConnectedEvent = hsm.Event[_ConnectedData](name="builder_bot_example.terminal.connected", schema=_ConnectedData)
_ConnectFailedEvent = hsm.Event[_SignalData](
    name="builder_bot_example.terminal.connect_failed", kind=hsm.ErrorEventKind, schema=_SignalData
)
_SocketClosedEvent = hsm.Event[_SocketClosedData](
    name="builder_bot_example.terminal.socket_closed", schema=_SocketClosedData
)
_BrokenEvent = hsm.Event[_SignalData](
    name="builder_bot_example.terminal.broken", kind=hsm.ErrorEventKind, schema=_SignalData
)
_LineEvent = hsm.Event[_LineData](name="builder_bot_example.terminal.line", schema=_LineData)
_EndOfInputEvent = hsm.Event[_SignalData](name="builder_bot_example.terminal.end_of_input", schema=_SignalData)
_ReleasedEvent = hsm.Event[_SignalData](
    name="builder_bot_example.terminal.released", kind=hsm.CompletionEventKind, schema=_SignalData
)


def _peer_uri(url: str, peer: str) -> str:
    """The version 1 path for ``peer`` at ``url``.

    Raises:
        ValueError: ``url`` is not ``ws://host[:port]`` with no path, or ``peer`` is not a transport peer id.
    """

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "ws" or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("url must be ws://host:port, with no path.")
    try:
        peer_id = _PEER_ID.validate_python(peer)
    except pydantic.ValidationError as error:
        raise ValueError("peer must be a transport peer id, such as builder-desk.") from error
    return f"{url.rstrip('/')}/v1/{peer_id}"


def _click_target(text: str) -> str | None:
    """The element id of a ``/click <id>`` line (empty when it names none), or ``None`` for any other line."""

    command, _, rest = text.strip().partition(" ")
    return rest.strip() if command == "/click" else None


class Terminal(Traced):
    """One person's session at the far side of a computer, from connecting to closed.

    ``connecting`` opens the socket, sends the token and starts the client-role yamux session, within
    ``connect_timeout``. ``linked`` pumps the socket both ways for the session. In ``opening`` it opens the
    three channels and waits, within ``open_timeout``, for the computer's first screen: the computer shows its
    display on every newly linked screen, so that screen is what says the computer took this terminal.
    ``ready`` reads the person's lines. ``closing`` stops the session and closes the socket, then the session
    is ``closed`` and ``on_closed`` is told how, once.

    What the screen shows is the last content the computer sent; it is the only thing the terminal keeps from
    the session besides the socket and the channels' streams.
    """

    _uri: str
    _token: pydantic.SecretStr
    _lines: Lines
    _screen: typing.TextIO
    _notices: typing.TextIO
    _on_closed: collections.abc.Callable[[ClosedData], None]
    _connect: Connect
    _connect_timeout: datetime.timedelta
    _open_timeout: datetime.timedelta
    _session: yamux.Client
    # Frames for the socket, in order; written by _enqueue and drained by _run_write.
    _outbox: asyncio.Queue[bytes]
    # The open socket; written only by the transition into `linked`.
    _socket: Socket | None
    # Each opened channel's stream: an address, dispatched to and never read. Written only by _own.
    _streams: dict[str, hsm.Instance]
    # What the screen shows: the last content the computer sent. Written only by _show.
    _content: computer.ContentData
    # How the session ends; written only by the transitions into `closing` and `closed`.
    _reason: Reason
    _code: int | None

    def __init__(
        self,
        *,
        url: str,
        peer: str,
        token: pydantic.SecretStr,
        lines: Lines,
        screen: typing.TextIO,
        notices: typing.TextIO,
        on_closed: collections.abc.Callable[[ClosedData], None],
        connect: Connect = connect_websocket,
        connect_timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
        open_timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
    ) -> None:
        """Build an unstarted terminal for the computer ``peer`` served at ``url`` (``ws://host:port``).

        Raises:
            ValueError: ``url`` is not a ``ws://`` URL with a host and no path, ``peer`` is not a transport
                peer id, or a timeout is not positive.
        """

        super().__init__()
        if connect_timeout <= datetime.timedelta() or open_timeout <= datetime.timedelta():
            raise ValueError("connect_timeout and open_timeout must be positive.")
        self._uri = _peer_uri(url, peer)
        self._token = token
        self._lines = lines
        self._screen = screen
        self._notices = notices
        self._on_closed = on_closed
        self._connect = connect
        self._connect_timeout = connect_timeout
        self._open_timeout = open_timeout
        self._session = yamux.Client(connection=self)
        self._outbox = asyncio.Queue()
        self._socket = None
        self._streams = {}
        self._content = computer.ContentData()
        self._reason = "abnormal"
        self._code = None

    # -- guards and timers ------------------------------------------------------------------------

    @staticmethod
    def _is_quit(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _LineData) and data.text.strip() == "/quit"

    @staticmethod
    def _is_click(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _LineData) and _click_target(data.text) is not None

    @staticmethod
    def _clicks_shown_element(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        if not isinstance(data, _LineData):
            return False
        target = _click_target(data.text)
        return target is not None and any(element.id == target for element in instance._content.elements)

    @staticmethod
    def _is_too_long(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _LineData) and len(data.text) > MAX_LINE

    @staticmethod
    def _connect_timeout_of(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._connect_timeout

    @staticmethod
    def _open_timeout_of(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._open_timeout

    # -- connecting -------------------------------------------------------------------------------

    @staticmethod
    async def _run_connect(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Open the socket, send the token, then start the yamux session under the terminal's lifetime."""

        del event
        try:
            socket = await instance._connect(instance._uri)
            await socket.send(json.dumps({"token": instance._token.get_secret_value()}))
        except (OSError, websockets.exceptions.WebSocketException) as error:
            _LOG.warning("builder terminal connect failed kind=%s", type(error).__name__)
            _ = hsm.dispatch(ctx, instance, _ConnectFailedEvent.with_data(_SignalData()))
            return
        # The session lives as long as the terminal (HSM-CONTEXT-001), not this activity.
        _ = await mosfet.started(instance.context(), instance._session, yamux.Client.model)
        _ = hsm.dispatch(ctx, instance, _ConnectedEvent.with_data(_ConnectedData(socket=socket)))

    @staticmethod
    def _link(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _ConnectedData)
        instance._socket = data.socket

    # -- outcomes ---------------------------------------------------------------------------------

    @staticmethod
    def _count_connect(outcome: ConnectOutcome) -> None:
        _CONNECTS.add(1, {"outcome": outcome})

    @staticmethod
    def _unreachable(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Terminal._count_connect("failed")
        instance._reason = "abnormal"

    @staticmethod
    def _timed_out(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        Terminal._count_connect("timeout")
        instance._reason = "timeout"

    @staticmethod
    def _count_linked(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        Terminal._count_connect("linked")

    @staticmethod
    def _count_unlinked(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """The socket closed before the computer's first screen: it refused or let go of this terminal, or broke."""

        del ctx, instance
        data = event.data
        assert isinstance(data, _SocketClosedData)
        Terminal._count_connect("refused" if data.code in _REFUSED_CODES | _NORMAL_CODES else "failed")

    @staticmethod
    def _count_failed(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance, event
        Terminal._count_connect("failed")

    @staticmethod
    def _close_for_socket(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _SocketClosedData)
        instance._code = data.code
        instance._reason = (
            "normal" if data.code in _NORMAL_CODES else "refused" if data.code in _REFUSED_CODES else "abnormal"
        )

    @staticmethod
    def _close_for_session(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """The yamux session ended: drained is the computer finishing normally; anything else broke it."""

        del ctx
        data = event.data
        assert isinstance(data, yamux.EndedData)
        instance._reason = "normal" if data.reason == "drained" else "abnormal"

    @staticmethod
    def _close_abnormal(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._reason = "abnormal"

    @staticmethod
    def _quit(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._reason = "quit"

    @staticmethod
    def _finish(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Count how the session ended, leave its span, and tell whoever started the terminal, once."""

        del ctx, event
        reason = instance._reason
        _CLOSED.add(1, {"reason": reason})
        with span.operation(
            "bot.example.terminal.session", scope=SCOPE, component="terminal", stage="session"
        ) as active:
            if reason not in ("quit", "normal"):
                span.record_failure(active, reason)
        instance._on_closed(
            ClosedData(reason=reason, code=instance._code, exit_code=0 if reason in ("quit", "normal") else 1)
        )

    # -- linked: the socket and the session -------------------------------------------------------

    @staticmethod
    async def _run_read(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Decode each binary message into yamux frames for the session, in order; frames may span messages."""

        del event
        socket = instance._socket
        assert socket is not None
        session = instance._session
        buffer = b""
        while True:
            try:
                message = await socket.recv()
            except websockets.exceptions.ConnectionClosed as closed:
                code = _ABNORMAL_CLOSURE if closed.rcvd is None else closed.rcvd.code
                _ = hsm.dispatch(ctx, instance, _SocketClosedEvent.with_data(_SocketClosedData(code=code)))
                return
            if not isinstance(message, bytes):
                _ = hsm.dispatch(ctx, instance, _BrokenEvent.with_data(_SignalData()))
                return
            buffer += message
            while True:
                try:
                    frame, buffer = yamux.decode_frame(buffer)
                except yamux.FrameIncomplete:
                    break
                except yamux.FrameError:
                    _ = hsm.dispatch(ctx, instance, _BrokenEvent.with_data(_SignalData()))
                    return
                _ = await hsm.dispatch(ctx, session, yamux.event_from_frame(frame))

    @staticmethod
    async def _run_write(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Send every frame the session transmits as one binary message, in order."""

        del event
        socket = instance._socket
        assert socket is not None
        while True:
            frame = await instance._outbox.get()
            try:
                await socket.send(frame)
            except websockets.exceptions.ConnectionClosed as closed:
                code = _ABNORMAL_CLOSURE if closed.rcvd is None else closed.rcvd.code
                _ = hsm.dispatch(ctx, instance, _SocketClosedEvent.with_data(_SocketClosedData(code=code)))
                return

    @staticmethod
    def _enqueue(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, yamux.TransmitData)
        instance._outbox.put_nowait(data.frame)

    @staticmethod
    def _open(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Open the keyboard, pointer and display channels; this terminal owns each stream it opens."""

        del event
        for channel in _CHANNELS:
            opened = yamux.OpenStreamData(owner=instance, codec=_CODECS[channel], channel=channel)
            _ = hsm.dispatch(ctx, instance._session, yamux.OpenStreamEvent.with_data(opened))

    @staticmethod
    def _own(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Keep the opened stream's address and send its first message, naming its channel."""

        data = event.data
        assert isinstance(data, transform.OwnedData)
        instance._streams[data.channel] = data.stream
        opening = websocket.StreamOpenEvent.with_data(websocket.StreamOpenData(channel=data.channel, v=1))
        _ = hsm.dispatch(ctx, data.stream, opening)

    @staticmethod
    def _send(ctx: hsm.Context, instance: "Terminal", channel: Channel, event: hsm.Event[typing.Any]) -> None:
        _ = hsm.dispatch(ctx, instance._streams[channel], event)
        _SENT.add(1, {"channel": channel})

    # -- the screen, the keyboard and the pointer -------------------------------------------------

    @staticmethod
    def _show(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Draw what the computer shows, then tell it where each element was drawn."""

        data = event.data
        assert isinstance(data, computer.ContentData)
        instance._content = data
        screen = instance._screen
        _ = screen.write((_CLEAR if screen.isatty() else "") + SCREEN_RULE + render(data))
        screen.flush()
        _RENDERS.add(1)
        Terminal._send(ctx, instance, "display", computer.LayoutEvent.with_data(layout(data)))

    @staticmethod
    def _type(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Type the line key by key, then press Enter."""

        data = event.data
        assert isinstance(data, _LineData)
        for character in data.text:
            key = hid.KeyData(action="down", key=character, code="", text=character)
            Terminal._send(ctx, instance, "keyboard", hid.KeyEvent.with_data(key))
        enter = hid.KeyData(action="down", key="Enter", code="Enter", text=None)
        Terminal._send(ctx, instance, "keyboard", hid.KeyEvent.with_data(enter))

    @staticmethod
    def _too_long(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _ = instance._notices.write(f"line too long: at most {MAX_LINE} characters, so nothing was typed\n")
        instance._notices.flush()

    @staticmethod
    def _click(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Press and release the pointer at the centre of the element's row."""

        data = event.data
        assert isinstance(data, _LineData)
        target = _click_target(data.text)
        elements = instance._content.elements
        (row,) = [index for index, element in enumerate(elements) if element.id == target]
        y = (row + 0.5) / len(elements)
        for action in ("down", "up"):
            pressed = hid.PointerData(action=action, pointer_type="mouse", pointer_id=0, x=0.5, y=y)
            Terminal._send(ctx, instance, "pointer", hid.PointerEvent.with_data(pressed))

    @staticmethod
    def _no_such_element(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _LineData)
        _ = instance._notices.write(f"no such element: {_click_target(data.text)}\n")
        instance._notices.flush()

    @staticmethod
    async def _run_input(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """One event per line the person types, and one when their input ends."""

        del event
        while True:
            line = await instance._lines()
            if line is None:
                _ = hsm.dispatch(ctx, instance, _EndOfInputEvent.with_data(_SignalData()))
                return
            _ = hsm.dispatch(ctx, instance, _LineEvent.with_data(_LineData(text=line)))

    # -- closing ----------------------------------------------------------------------------------

    @staticmethod
    async def _run_close(ctx: hsm.Context, instance: "Terminal", event: hsm.Event[typing.Any]) -> None:
        """Stop the session (every stream with it), then close the socket; bounded by ``connect_timeout``."""

        del event
        await hsm.stop(instance._session)
        socket = instance._socket
        if socket is not None:
            try:
                await socket.close(1000)
            except (OSError, websockets.exceptions.WebSocketException) as error:
                _LOG.warning("builder terminal socket close failed kind=%s", type(error).__name__)
        _ = hsm.dispatch(ctx, instance, _ReleasedEvent.with_data(_SignalData()))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Terminal",
        hsm.initial(hsm.target("/Terminal/connecting")),
        hsm.state(
            "connecting",
            hsm.activity(_run_connect),
            hsm.transition(hsm.on(_ConnectedEvent), hsm.effect(_link), hsm.target("/Terminal/linked")),
            hsm.transition(
                hsm.on(_ConnectFailedEvent), hsm.effect(_unreachable, _finish), hsm.target("/Terminal/closed")
            ),
            hsm.transition(
                hsm.after(_connect_timeout_of), hsm.effect(_timed_out, _finish), hsm.target("/Terminal/closed")
            ),
        ),
        hsm.state(
            "linked",
            hsm.activity(_run_read, _run_write),
            hsm.initial(hsm.target("/Terminal/linked/opening")),
            hsm.transition(hsm.on(yamux.TransmitEvent), hsm.effect(_enqueue)),
            hsm.transition(hsm.on(transform.OwnedEvent), hsm.effect(_own)),
            hsm.transition(hsm.on(_SocketClosedEvent), hsm.effect(_close_for_socket), hsm.target("/Terminal/closing")),
            hsm.transition(hsm.on(yamux.EndedEvent), hsm.effect(_close_for_session), hsm.target("/Terminal/closing")),
            # A channel the computer reset, a socket that is not yamux, or a session that opens nothing more.
            hsm.transition(
                hsm.on(transform.EndedEvent, _BrokenEvent, yamux.OpenStreamFailedEvent),
                hsm.effect(_close_abnormal),
                hsm.target("/Terminal/closing"),
            ),
            hsm.state(
                "opening",
                hsm.entry(_open),
                hsm.transition(hsm.on(computer.ContentEvent), hsm.effect(_show), hsm.target("/Terminal/linked/ready")),
                hsm.transition(hsm.after(_open_timeout_of), hsm.effect(_timed_out), hsm.target("/Terminal/closing")),
                hsm.transition(
                    hsm.on(_SocketClosedEvent),
                    hsm.effect(_count_unlinked, _close_for_socket),
                    hsm.target("/Terminal/closing"),
                ),
                hsm.transition(
                    hsm.on(yamux.EndedEvent),
                    hsm.effect(_count_failed, _close_for_session),
                    hsm.target("/Terminal/closing"),
                ),
                hsm.transition(
                    hsm.on(transform.EndedEvent, _BrokenEvent, yamux.OpenStreamFailedEvent),
                    hsm.effect(_count_failed, _close_abnormal),
                    hsm.target("/Terminal/closing"),
                ),
            ),
            hsm.state(
                "ready",
                hsm.entry(_count_linked),
                hsm.activity(_run_input),
                hsm.transition(hsm.on(computer.ContentEvent), hsm.effect(_show)),
                hsm.transition(
                    hsm.on(_LineEvent), hsm.guard(_is_quit), hsm.effect(_quit), hsm.target("/Terminal/closing")
                ),
                hsm.transition(hsm.on(_LineEvent), hsm.guard(_clicks_shown_element), hsm.effect(_click)),
                hsm.transition(hsm.on(_LineEvent), hsm.guard(_is_click), hsm.effect(_no_such_element)),
                hsm.transition(hsm.on(_LineEvent), hsm.guard(_is_too_long), hsm.effect(_too_long)),
                hsm.transition(hsm.on(_LineEvent), hsm.effect(_type)),
                hsm.transition(hsm.on(_EndOfInputEvent), hsm.effect(_quit), hsm.target("/Terminal/closing")),
            ),
        ),
        hsm.state(
            "closing",
            hsm.activity(_run_close),
            hsm.transition(hsm.on(_ReleasedEvent), hsm.effect(_finish), hsm.target("/Terminal/closed")),
            hsm.transition(hsm.after(_connect_timeout_of), hsm.effect(_finish), hsm.target("/Terminal/closed")),
        ),
        hsm.final("closed"),
    )


class StdinLines:
    """The lines a person types into ``stream`` (the process's stdin), read by one daemon thread.

    The thread reads the stream's file descriptor (``os.read``), never the buffered stream itself, so it
    holds no interpreter lock a shutdown could wait on. It starts on the first call, posting each line to the
    running loop. A line longer than :data:`MAX_LINE` characters arrives cut one character past it, still too
    long to type, and the rest of it is dropped up to its line ending. A stream with no file descriptor, the
    end of input, or a read error end the lines: that call and every later one return ``None``.
    """

    _stream: typing.TextIO
    _queue: asyncio.Queue[str | None]
    _thread: threading.Thread | None
    _ended: bool

    def __init__(self, stream: typing.TextIO) -> None:
        self._stream = stream
        self._queue = asyncio.Queue()
        self._thread = None
        self._ended = False

    async def __call__(self) -> str | None:
        """The next line the person typed, without its line ending, or ``None`` once input has ended."""

        if self._thread is None:
            try:
                fd = self._stream.fileno()
            except (OSError, ValueError):
                fd = -1
            self._thread = threading.Thread(
                target=self._read, args=(fd, asyncio.get_running_loop()), name="builder-terminal-stdin", daemon=True
            )
            self._thread.start()
        if self._ended:
            return None
        line = await self._queue.get()
        self._ended = line is None
        return line

    def _read(self, fd: int, loop: asyncio.AbstractEventLoop) -> None:
        def post(line: str | None) -> bool:
            try:
                _ = loop.call_soon_threadsafe(self._queue.put_nowait, line)
            except RuntimeError:
                # The loop is closed: nobody is reading any more.
                return False
            return True

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        # Inside a line already posted cut: what is left of it, up to its line ending, goes nowhere.
        dropping = False
        while True:
            try:
                chunk = os.read(fd, _READ_CHUNK) if fd >= 0 else b""
            except OSError:
                chunk = b""
            text = decoder.decode(chunk, final=not chunk)
            if dropping:
                _, ending, text = text.partition("\n")
                dropping = not ending
            *lines, pending = (pending + text).split("\n")
            lines = [line[: MAX_LINE + 1] for line in lines]
            if len(pending) > MAX_LINE:
                lines.append(pending[: MAX_LINE + 1])
                pending = ""
                dropping = True
            if not chunk and pending:
                lines.append(pending)
            for line in lines:
                if not post(line.removesuffix("\r")):
                    return
            if not chunk:
                _ = post(None)
                return


class _UsageError(Exception):
    """A bad flag: the command line is not one this program takes."""


class _Parser(argparse.ArgumentParser):
    @typing.override
    def error(self, message: str) -> typing.NoReturn:
        raise _UsageError(message)


class _Arguments(argparse.Namespace):
    url: str = _DEFAULT_URL
    peer: str = _DEFAULT_PEER
    secret_file: pathlib.Path | None = None


async def _session(*, url: str, peer: str, token: pydantic.SecretStr) -> ClosedData:
    """Run one terminal session on this process's terminal until it closes; return how it closed."""

    closed: asyncio.Future[ClosedData] = asyncio.get_running_loop().create_future()
    machine = Terminal(
        url=url,
        peer=peer,
        token=token,
        lines=StdinLines(sys.stdin),
        screen=sys.stdout,
        notices=sys.stderr,
        on_closed=closed.set_result,
    )
    _ = await mosfet.started(None, machine, Terminal.model)
    try:
        return await closed
    finally:
        await hsm.stop(machine)


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    """Run ``builder-terminal`` with ``argv`` (default: ``sys.argv[1:]``) and return its exit code."""

    parser = _Parser(
        prog="builder-terminal",
        description=(
            "Sit at the far side of the builder bot's computer: its screen here, your keys and clicks there. "
            + "Type a line and press Enter to type it into the focused field, or to the computer when nothing is "
            + "focused; /click <element id> to click; "
            + "/quit to leave. Exit codes: 0 left or closed normally, 1 failed or refused, 64 bad flags, "
            + "78 missing secret."
        ),
    )
    _ = parser.add_argument("--url", default=_DEFAULT_URL, help=f"the bot's computer server; default: {_DEFAULT_URL}")
    _ = parser.add_argument("--peer", default=_DEFAULT_PEER, help=f"the computer's peer id; default: {_DEFAULT_PEER}")
    _ = parser.add_argument(
        "--secret-file",
        type=pathlib.Path,
        help="file holding computer_token_secret; else COMPUTER_TOKEN_SECRET_FILE, else COMPUTER_TOKEN_SECRET",
    )
    try:
        arguments = parser.parse_args(argv, namespace=_Arguments())
        _ = _peer_uri(arguments.url, arguments.peer)
    except (_UsageError, ValueError) as error:
        print(f"builder-terminal: {error}", file=sys.stderr)
        return _USAGE
    except SystemExit as requested:
        # argparse ends --help this way, after printing the requested help to stdout.
        return requested.code if isinstance(requested.code, int) else 0

    # The process environment is read once, here.
    environ = dict(os.environ)
    if arguments.secret_file is not None:
        environ[f"{_SECRET.upper()}_FILE"] = str(arguments.secret_file)
    try:
        secret = runtime.Secrets(directory=_SECRETS_DIRECTORY, environ=environ).get(_SECRET)
        token = websocket.HmacTokens(secret).issue(arguments.peer)
    except runtime.MissingSecretError as error:
        print(f"builder-terminal: {error}", file=sys.stderr)
        return _CONFIG
    except OSError as error:
        print(f"builder-terminal: cannot read secret {_SECRET!r}: {error.strerror}", file=sys.stderr)
        return _CONFIG
    except ValueError:
        print(f"builder-terminal: secret {_SECRET!r} is empty", file=sys.stderr)
        return _CONFIG

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        _ = telemetry.configure(service_name="builder-terminal")
    except (OSError, ValueError) as error:
        print(f"builder-terminal: cannot configure telemetry: {error}", file=sys.stderr)
        return _CONFIG
    try:
        closed = asyncio.run(_session(url=arguments.url, peer=arguments.peer, token=token))
    except KeyboardInterrupt:
        return _INTERRUPTED
    finally:
        telemetry.reset()
    if closed.reason not in ("quit", "normal"):
        detail = "" if closed.code is None else f" ({closed.code})"
        print(f"builder-terminal: session ended: {closed.reason}{detail}", file=sys.stderr)
    return closed.exit_code


__all__ = [
    "SCOPE",
    "SCREEN_RULE",
    "ClosedData",
    "Connect",
    "ConnectOutcome",
    "MAX_LINE",
    "Lines",
    "Reason",
    "Socket",
    "StdinLines",
    "Terminal",
    "connect_websocket",
    "layout",
    "main",
    "render",
]
