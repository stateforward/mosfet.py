"""The far side of one room's computer: the room's microphone, speaker and camera, carried to the companion.

    python -m companion_example.room --url ws://127.0.0.1:8765 --peer kitchen [--secret-file PATH] [--camera]

``Room`` is the far side's machine and takes its devices injected, so another far side (the browser bridge,
``companion_example.ui``) reuses it with devices of its own; this module's ``main`` runs it on the host's sound card.

The room is a remote machine as the websocket transport's version 1 wire describes one. It connects to
``<url>/v1/<peer>``, sends its token (``websocket.HmacTokens(secret).issue(peer)``), and is the client role of
one yamux session, on which it opens the ``microphone`` and ``speaker`` channels, ``camera`` with ``--camera``,
and ``keyboard`` when it is given one. Each channel's first message is a ``websocket.StreamOpenData``; after it,
the microphone carries 16 kHz mono s16le PCM in and the speaker whatever the computer plays out (24 kHz from
the companion's voice), both as ``audio.PcmCodec`` messages, the camera one JPEG still at a time as
``computer.StillCodec`` reads them, and the keyboard each line typed as ``hid.KeyEvent`` key presses, then Enter,
as the computer's JSON keyboard codec reads them. With nothing focused on the computer's screen, the computer
tells whoever holds it ``computer.text.entered`` with no field. The room defines no wire type of its own.

It decides nothing: it captures and plays. Everything the microphone hears goes to the computer, which puts it
into the house at the room's place; everything the computer plays comes out of the speaker. The sound card is
``sounddevice`` (PortAudio), and the camera the first OpenCV camera. Echo cancellation is the operating system's
or the device's: capture through a device or a voice-processing input that cancels what the speaker plays, or
the companion hears itself late (it knows its own voice, so it is never taken for someone else).

The secret is ``computer_token_secret``, resolved through ``runtime.Secrets``: the Docker secret
``/run/secrets/computer_token_secret``, then the file ``--secret-file`` names (or ``COMPUTER_TOKEN_SECRET_FILE``),
then ``COMPUTER_TOKEN_SECRET``. Exit codes: ``0`` when the room is left (SIGINT or SIGTERM) or the computer closes
the session normally (``1000``, or ``1001`` when the companion's process goes away), ``1`` when the connection fails,
times out, is refused (``4401``, ``4404``, ``4408``, ``4409``) or ends abnormally, ``64`` for bad flags, ``78``
for a missing or unreadable secret. The room writes nothing to stdout; its diagnostics go to stderr.

Telemetry (meter scope ``companion_example.room``), every attribute a closed set from code:
``bot.example.room.connect.count{outcome}`` (``connected``, ``timeout``, ``failed``),
``bot.example.room.frames.count{channel}`` (``microphone``, ``camera`` and ``keyboard`` sent, ``speaker`` played),
``bot.example.room.dropped.count{channel}`` (captured while the link could not take it),
``bot.example.room.closed.count{reason}`` and one ``bot.example.room.session`` span per session. The peer, the
url, the token, every line typed and every sound and image never are attributes.
"""

import argparse
import asyncio
import collections
import collections.abc
import datetime
import json
import logging
import os
import pathlib
import signal
import sys
import threading
import time
import typing
import urllib.parse

import hsm
import mosfet
import pydantic
import websockets.asyncio.client
import websockets.exceptions
from mosfet import media, runtime, telemetry
from mosfet.devices import audio, computer, hid
from mosfet.protocols import transform, transport, yamux
from mosfet.providers import websocket
from mosfet.telemetry import span
from mosfet.telemetry.hsm import Traced
from opentelemetry import context as otel_context
from opentelemetry import metrics
from pydantic.json_schema import SkipJsonSchema

SCOPE: typing.Final = "companion_example.room"
MICROPHONE_RATE_HZ: typing.Final = 16_000
"""The rate the microphone captures at, mono s16le: what the speech server hears."""
MICROPHONE_BLOCK_SECONDS: typing.Final = 0.2
"""How much sound each microphone message carries."""
CAMERA_EVERY_SECONDS: typing.Final = 1.0
"""How often the camera sends a still; the companion's eye looks at most twice a second whatever the rate."""
CAMERA_LONG_EDGE: typing.Final = 1280
"""The longest edge of a still, in pixels, so every view the companion remembers stays small."""
MAX_PENDING: typing.Final = 64
"""Most captures held while the link cannot take them; past it, the newest is dropped and counted.

A typed line is one capture, however many keys it takes."""
# The most one key event takes on the keyboard channel: a 4-byte length, then at most 124 bytes of JSON (a
# control character, escaped as six bytes in both `key` and `text`). Enter takes as much.
_KEY_BYTES = 128
MAX_LINE: typing.Final = 2 * yamux.INITIAL_STREAM_WINDOW // _KEY_BYTES - 1
"""The longest line the room types, in characters: 4095.

A line is typed all at once, so its keys and its Enter must fit what one yamux stream holds before the computer
takes any of them: the computer's receive window (at least ``yamux.INITIAL_STREAM_WINDOW``) and as much again
waiting on this side. A longer line is not typed; it is dropped and counted.
"""

type Reason = typing.Literal["left", "normal", "refused", "abnormal", "timeout"]
type ConnectOutcome = typing.Literal["connected", "timeout", "failed"]
type Channel = typing.Literal["microphone", "speaker", "camera", "keyboard"]

_LOG = logging.getLogger(__name__)
_METER = metrics.get_meter(SCOPE)
_CONNECTS = _METER.create_counter(
    "bot.example.room.connect.count",
    unit="{session}",
    description="Room sessions by how connecting ended: connected, timeout, or failed.",
)
_FRAMES = _METER.create_counter(
    "bot.example.room.frames.count",
    unit="{message}",
    description="Messages the room carried, by channel: microphone and camera sent, speaker played.",
)
_DROPPED = _METER.create_counter(
    "bot.example.room.dropped.count",
    unit="{message}",
    description="Captured messages dropped because the link could not take them, by channel.",
)
_CLOSED = _METER.create_counter(
    "bot.example.room.closed.count",
    unit="{session}",
    description="Room sessions by why they closed: left, normal, refused, abnormal, or timeout.",
)

_DEFAULT_URL = "ws://127.0.0.1:8765"
_DEFAULT_PEER = "living-room"
_SECRET = "computer_token_secret"
_SECRETS_DIRECTORY = pathlib.Path("/run/secrets")
_DEFAULT_TIMEOUT = datetime.timedelta(seconds=10)
_NORMAL_CODES = frozenset({1000, 1001})
_REFUSED_CODES = frozenset({4401, 4404, 4408, 4409})
_ABNORMAL_CLOSURE = 1006
_PEER_ID: pydantic.TypeAdapter[str] = pydantic.TypeAdapter(transport.PeerId)
_MEDIA_TYPE = "audio/pcm"

_USAGE = 64
_CONFIG = 78


class Socket(typing.Protocol):
    """The room's end of its WebSocket, as it uses it (``websockets``' ``ClientConnection`` is one).

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
"""Open the WebSocket at a ``ws://`` URI. The room bounds it with its own ``connect_timeout``."""


class Microphone(typing.Protocol):
    """The room's microphone: it captures 16 kHz mono s16le and hands each block to ``heard``, from any thread."""

    def start(self, heard: collections.abc.Callable[[bytes], None], /) -> None:
        """Start capturing; each block is handed to ``heard`` as it is captured."""
        ...

    def stop(self) -> None:
        """Stop capturing; nothing is handed on after it returns."""
        ...


class Speaker(typing.Protocol):
    """The room's speaker: it plays PCM as it is given, without blocking the caller."""

    def play(self, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
        """Queue ``pcm`` (s16le at ``sample_rate_hz`` with ``channels``) to play after what is already queued."""
        ...

    def stop(self) -> None:
        """Stop playing and let go of what is queued."""
        ...


class Keyboard(typing.Protocol):
    """The room's keyboard: it hands each line typed on it to ``typed``, without its line ending, from any thread."""

    def start(self, typed: collections.abc.Callable[[str], None], /) -> None:
        """Start taking lines; each is handed to ``typed`` as it is entered."""
        ...

    def stop(self) -> None:
        """Stop taking lines; nothing is handed on after it returns."""
        ...


class Camera(typing.Protocol):
    """The room's camera: it hands a still to ``seen`` every so often, from any thread."""

    def start(self, seen: collections.abc.Callable[[media.Image], None], /) -> None:
        """Start capturing stills."""
        ...

    def stop(self) -> None:
        """Stop capturing; nothing is handed on after it returns."""
        ...


async def connect_websocket(uri: str) -> Socket:
    """Open ``uri`` with ``websockets``; every bound and keepalive is the room's, so the library sets none."""

    return await websockets.asyncio.client.connect(uri, open_timeout=None, ping_interval=None)


class ClosedData(pydantic.BaseModel):
    """How one room session ended, told once to whoever started the room."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"reason": "left", "code": None, "exit_code": 0},
                {"reason": "refused", "code": 4401, "exit_code": 1},
            ]
        },
    )

    reason: Reason = pydantic.Field(
        description=(
            "Why: the room was left (`left`), the computer closed the session normally (`normal`), the computer "
            "refused this room (`refused`), the connection failed or broke (`abnormal`), or connecting or opening "
            "the channels took too long (`timeout`)."
        ),
        examples=["left", "refused"],
    )
    code: int | None = pydantic.Field(
        description="The WebSocket close code the computer sent, when the computer closed the socket.",
        examples=[None, 1000, 4401],
    )
    exit_code: typing.Literal[0, 1] = pydantic.Field(
        description="The process exit code: 0 for `left` and `normal`, 1 otherwise.", examples=[0, 1]
    )


class LeaveData(pydantic.BaseModel):
    """Leave the room: stop capturing and playing, and close the session normally."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")


LeaveEvent = hsm.Event[LeaveData](name="companion_example.room.leave", schema=LeaveData)
"""Leave the room; the one event anyone but the room itself sends it (its process does, on SIGINT or SIGTERM)."""


class _OpeningCodec:
    """One channel's codec on this side, which also writes the stream's first message as compact JSON."""

    name: str
    max_message_bytes: int
    encodes_any: bool
    _channel: transform.Codec

    def __init__(self, channel: transform.Codec) -> None:
        self.name = channel.name
        self.max_message_bytes = channel.max_message_bytes
        self.encodes_any = True
        self._channel = channel

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """The channel's event ``message`` carries.

        Raises:
            transform.DecodeError: as the channel's codec raises it.
        """

        return self._channel.decode(message)

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """The stream-opening message, or an event the channel carries: a pure lookup by name."""

        return event.name == websocket.StreamOpenEvent.name or self._channel.encodes(event)

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """The message for ``event``.

        Raises:
            transform.EncodeError: as the channel's codec raises it.
        """

        if isinstance(event.data, websocket.StreamOpenData):
            return event.data.model_dump_json().encode()
        return self._channel.encode(event)


class _StillSender:
    """The camera channel from this side: each ``computer.camera.still`` as ``computer.StillCodec`` reads one.

    A message is one byte giving the media type's length, the ASCII media type, then the image file.
    """

    name: str
    max_message_bytes: int
    encodes_any: bool
    _reader: computer.StillCodec

    def __init__(self) -> None:
        self._reader = computer.StillCodec()
        self.name = self._reader.name
        self.max_message_bytes = self._reader.max_message_bytes
        self.encodes_any = True

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """The still ``message`` carries; the computer sends nothing on this channel.

        Raises:
            transform.DecodeError: as ``computer.StillCodec`` raises it.
        """

        return self._reader.decode(message)

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """A still: a pure lookup by name."""

        return event.name == computer.StillEvent.name

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """The message for one still.

        Raises:
            transform.EncodeError: ``schema_invalid`` for anything but a still.
        """

        data = event.data
        if not isinstance(data, computer.StillData):
            raise transform.EncodeError("schema_invalid", f"the camera channel sends stills, not {event.name!r}.")
        media_type = data.image.media_type.encode("ascii")
        return bytes((len(media_type),)) + media_type + data.image.content


_CODECS: dict[Channel, transform.Codec] = {
    "microphone": _OpeningCodec(audio.PcmCodec()),
    "speaker": _OpeningCodec(audio.PcmCodec()),
    "camera": _OpeningCodec(_StillSender()),
    "keyboard": _OpeningCodec(transform.JsonCodec(hid.KeyEvent, max_message_bytes=computer.MAX_MESSAGE_BYTES)),
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


# Private: each is dispatched by this class's own activities to itself only.
_ConnectedEvent = hsm.Event[_ConnectedData](name="companion_example.room.connected", schema=_ConnectedData)
_ConnectFailedEvent = hsm.Event[_SignalData](
    name="companion_example.room.connect_failed", kind=hsm.ErrorEventKind, schema=_SignalData
)
_SocketClosedEvent = hsm.Event[_SocketClosedData](name="companion_example.room.socket_closed", schema=_SocketClosedData)
_BrokenEvent = hsm.Event[_SignalData](name="companion_example.room.broken", kind=hsm.ErrorEventKind, schema=_SignalData)
_ReleasedEvent = hsm.Event[_SignalData](
    name="companion_example.room.released", kind=hsm.CompletionEventKind, schema=_SignalData
)


def peer_uri(url: str, peer: str) -> str:
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
        raise ValueError("peer must be a transport peer id, such as kitchen.") from error
    return f"{url.rstrip('/')}/v1/{peer_id}"


class Room(Traced):
    """One room's far side, from connecting to closed.

    ``connecting`` opens the socket, sends the token and starts the client-role yamux session, within
    ``connect_timeout``. ``linked`` pumps the socket both ways for the session. In ``opening`` it opens the room's
    channels, within ``open_timeout``. ``streaming`` captures and plays: the microphone's blocks (and the
    camera's stills) go to the computer as they are captured, and what the computer plays comes out of the
    speaker. ``closing`` stops the session and closes the socket, then the session is ``closed`` and
    ``on_closed`` is told how, once. ``LeaveEvent`` leaves at any point.

    Captured sound, stills and typed lines cross from the devices' threads into the room's loop through one bounded
    queue, the only thing the room holds of what it carries; past ``MAX_PENDING`` the newest is dropped and counted.
    """

    _uri: str
    _token: pydantic.SecretStr
    _microphone: Microphone
    _speaker: Speaker
    _camera: Camera | None
    _keyboard: Keyboard | None
    _channels: tuple[Channel, ...]
    _on_closed: collections.abc.Callable[[ClosedData], None]
    _connect: Connect
    _connect_timeout: datetime.timedelta
    _open_timeout: datetime.timedelta
    # The wall clock, in nanoseconds since the epoch, that the session span starts on.
    _clock: collections.abc.Callable[[], int]
    _session: yamux.Client
    # Frames for the socket, in order; written by _enqueue and drained by _run_write.
    _outbox: asyncio.Queue[bytes]
    # What the devices captured, for the link, one capture per item (a typed line is all its keys); written from
    # their threads and drained by _run_capture.
    _captured: asyncio.Queue[tuple[Channel, tuple[hsm.Event[typing.Any], ...]]]
    # The open socket; written only by the transition into `linked`.
    _socket: Socket | None
    # Each opened channel's stream: an address, dispatched to and never read. Written only by _own.
    _streams: dict[str, hsm.Instance]
    # How the session ends; written only by the transitions into `closing` and `closed`.
    _reason: Reason
    _code: int | None
    # When the session began (ns since the epoch) and the trace it began in, for its span; written only on
    # entering `connecting`, read by _finish.
    _began: tuple[int, otel_context.Context] | None

    def __init__(
        self,
        *,
        url: str,
        peer: str,
        token: pydantic.SecretStr,
        microphone: Microphone,
        speaker: Speaker,
        on_closed: collections.abc.Callable[[ClosedData], None],
        camera: Camera | None = None,
        keyboard: Keyboard | None = None,
        connect: Connect = connect_websocket,
        connect_timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
        open_timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
        clock: collections.abc.Callable[[], int] = time.time_ns,
    ) -> None:
        """Build an unstarted room for the computer ``peer`` served at ``url`` (``ws://host:port``).

        ``clock`` reads the wall clock in nanoseconds since the epoch; the session's span starts at its reading.

        Raises:
            ValueError: ``url`` is not a ``ws://`` URL with a host and no path, ``peer`` is not a transport
                peer id, or a timeout is not positive.
        """

        super().__init__()
        if connect_timeout <= datetime.timedelta() or open_timeout <= datetime.timedelta():
            raise ValueError("connect_timeout and open_timeout must be positive.")
        self._uri = peer_uri(url, peer)
        self._token = token
        self._microphone = microphone
        self._speaker = speaker
        self._camera = camera
        self._keyboard = keyboard
        self._channels = (
            "microphone",
            "speaker",
            *(("camera",) if camera is not None else ()),
            *(("keyboard",) if keyboard is not None else ()),
        )
        self._on_closed = on_closed
        self._connect = connect
        self._connect_timeout = connect_timeout
        self._open_timeout = open_timeout
        self._clock = clock
        self._session = yamux.Client(connection=self)
        self._outbox = asyncio.Queue()
        self._captured = asyncio.Queue(maxsize=MAX_PENDING)
        self._socket = None
        self._streams = {}
        self._reason = "abnormal"
        self._code = None
        self._began = None

    # -- guards and timers ------------------------------------------------------------------------

    @staticmethod
    def _opens_the_last(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return len(instance._streams) + 1 == len(instance._channels)

    @staticmethod
    def _is_sound(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, audio.InputData) and data.sample_rate_hz is not None and data.channels is not None

    @staticmethod
    def _connect_timeout_of(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._connect_timeout

    @staticmethod
    def _open_timeout_of(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._open_timeout

    # -- connecting -------------------------------------------------------------------------------

    @staticmethod
    async def _run_connect(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Open the socket, send the token, then start the yamux session under the room's lifetime."""

        del event
        try:
            socket = await instance._connect(instance._uri)
            await socket.send(json.dumps({"token": instance._token.get_secret_value()}))
        except (OSError, websockets.exceptions.WebSocketException) as error:
            _LOG.warning("companion room connect failed kind=%s", type(error).__name__)
            _ = hsm.dispatch(ctx, instance, _ConnectFailedEvent.with_data(_SignalData()))
            return
        # The session lives as long as the room (HSM-CONTEXT-001), not this activity.
        _ = await mosfet.started(instance.context(), instance._session, yamux.Client.model)
        _ = hsm.dispatch(ctx, instance, _ConnectedEvent.with_data(_ConnectedData(socket=socket)))

    @staticmethod
    def _link(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _ConnectedData)
        instance._socket = data.socket
        _CONNECTS.add(1, {"outcome": "connected"})

    # -- outcomes ---------------------------------------------------------------------------------

    @staticmethod
    def _unreachable(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _CONNECTS.add(1, {"outcome": "failed"})
        instance._reason = "abnormal"

    @staticmethod
    def _connect_timed_out(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        _CONNECTS.add(1, {"outcome": "timeout"})
        instance._reason = "timeout"

    @staticmethod
    def _open_timed_out(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._reason = "timeout"

    @staticmethod
    def _close_for_socket(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _SocketClosedData)
        instance._code = data.code
        instance._reason = (
            "normal" if data.code in _NORMAL_CODES else "refused" if data.code in _REFUSED_CODES else "abnormal"
        )

    @staticmethod
    def _close_for_session(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """The yamux session ended: drained is the computer finishing normally; anything else broke it."""

        del ctx
        data = event.data
        assert isinstance(data, yamux.EndedData)
        instance._reason = "normal" if data.reason == "drained" else "abnormal"

    @staticmethod
    def _close_abnormal(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._reason = "abnormal"

    @staticmethod
    def _leave(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._reason = "left"

    @staticmethod
    def _begin(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Note when, and in which trace, the session began: its span starts there."""

        del ctx
        instance._began = (instance._clock(), telemetry.event_context(event))

    @staticmethod
    def _finish(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Count how the session ended, leave its span, and tell whoever started the room, once.

        The span runs from when the session began to now. It is opened here, with that start, rather than held open
        through the session, so a room stopped before it closes leaves no span half open.
        """

        del ctx, event
        reason = instance._reason
        _CLOSED.add(1, {"reason": reason})
        began = instance._began
        assert began is not None
        started_at, began_in = began
        session = span.tracer(SCOPE).start_span(
            "bot.example.room.session",
            context=began_in,
            start_time=started_at,
            attributes={"bot.component.name": "room", "bot.stage": "session"},
        )
        if reason in ("left", "normal"):
            session.set_attribute("bot.outcome", span.Ok)
        else:
            span.record_failure(session, reason)
        session.end()
        instance._on_closed(
            ClosedData(reason=reason, code=instance._code, exit_code=0 if reason in ("left", "normal") else 1)
        )

    # -- linked: the socket and the session -------------------------------------------------------

    @staticmethod
    async def _run_read(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
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
    async def _run_write(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
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
    def _enqueue(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, yamux.TransmitData)
        instance._outbox.put_nowait(data.frame)

    @staticmethod
    def _open(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Open the room's channels; this room owns each stream it opens."""

        del event
        for channel in instance._channels:
            opened = yamux.OpenStreamData(owner=instance, codec=_CODECS[channel], channel=channel)
            _ = hsm.dispatch(ctx, instance._session, yamux.OpenStreamEvent.with_data(opened))

    @staticmethod
    def _own(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Keep the opened stream's address and send its first message, naming its channel."""

        data = event.data
        assert isinstance(data, transform.OwnedData)
        instance._streams[data.channel] = data.stream
        opening = websocket.StreamOpenEvent.with_data(websocket.StreamOpenData(channel=data.channel, v=1))
        _ = hsm.dispatch(ctx, data.stream, opening)

    # -- streaming: the microphone, the camera and the speaker ------------------------------------

    @staticmethod
    async def _run_capture(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Start the room's devices and send what they capture, as it is captured, until the state is left."""

        del event
        loop = asyncio.get_running_loop()
        captured = instance._captured

        def hand_on(channel: Channel, items: tuple[hsm.Event[typing.Any], ...]) -> None:
            if captured.full():
                _DROPPED.add(1, {"channel": channel})
                return
            captured.put_nowait((channel, items))

        def post(channel: Channel, *items: hsm.Event[typing.Any]) -> None:
            try:
                _ = loop.call_soon_threadsafe(hand_on, channel, items)
            except RuntimeError:
                # The loop is closed: nobody is sending any more.
                return

        def heard(pcm: bytes) -> None:
            sound = audio.OutputData(audio=pcm, media_type=_MEDIA_TYPE, sample_rate_hz=MICROPHONE_RATE_HZ, channels=1)
            post("microphone", audio.OutputEvent.with_data(sound))

        def seen(image: media.Image) -> None:
            post("camera", computer.StillEvent.with_data(computer.StillData(image=image)))

        def typed(line: str) -> None:
            if len(line) > MAX_LINE:
                _DROPPED.add(1, {"channel": "keyboard"})
                return
            keys = [hid.KeyData(action="down", key=character, code="", text=character) for character in line]
            keys.append(hid.KeyData(action="down", key="Enter", code="Enter", text=None))
            post("keyboard", *(hid.KeyEvent.with_data(key) for key in keys))

        devices: list[Microphone | Camera | Keyboard] = [instance._microphone]
        try:
            instance._microphone.start(heard)
            if instance._camera is not None:
                devices.append(instance._camera)
                instance._camera.start(seen)
            if instance._keyboard is not None:
                devices.append(instance._keyboard)
                instance._keyboard.start(typed)
        except Exception as error:
            # The device boundary: a device that cannot open ends the session, named by kind only.
            _LOG.warning("companion room device failed kind=%s", type(error).__name__)
            for device in devices:
                device.stop()
            _ = hsm.dispatch(ctx, instance, _BrokenEvent.with_data(_SignalData()))
            return
        try:
            while True:
                channel, items = await captured.get()
                for item in items:
                    _ = hsm.dispatch(ctx, instance._streams[channel], item)
                _FRAMES.add(len(items), {"channel": channel})
        finally:
            for device in devices:
                device.stop()

    @staticmethod
    def _play(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Play what the computer sent the speaker."""

        del ctx
        data = event.data
        assert isinstance(data, audio.InputData)
        assert data.sample_rate_hz is not None and data.channels is not None
        instance._speaker.play(data.audio, sample_rate_hz=data.sample_rate_hz, channels=data.channels)
        _FRAMES.add(1, {"channel": "speaker"})

    @staticmethod
    def _silence(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._speaker.stop()

    # -- closing ----------------------------------------------------------------------------------

    @staticmethod
    async def _run_close(ctx: hsm.Context, instance: "Room", event: hsm.Event[typing.Any]) -> None:
        """Stop the session (every stream with it), then close the socket; bounded by ``connect_timeout``."""

        del event
        await hsm.stop(instance._session)
        socket = instance._socket
        if socket is not None:
            try:
                await socket.close(1000)
            except (OSError, websockets.exceptions.WebSocketException) as error:
                _LOG.warning("companion room socket close failed kind=%s", type(error).__name__)
        _ = hsm.dispatch(ctx, instance, _ReleasedEvent.with_data(_SignalData()))

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Room",
        hsm.initial(hsm.target("/Room/connecting")),
        hsm.state(
            "connecting",
            hsm.entry(_begin),
            hsm.activity(_run_connect),
            hsm.transition(hsm.on(_ConnectedEvent), hsm.effect(_link), hsm.target("/Room/linked")),
            hsm.transition(hsm.on(_ConnectFailedEvent), hsm.effect(_unreachable, _finish), hsm.target("/Room/closed")),
            hsm.transition(
                hsm.after(_connect_timeout_of), hsm.effect(_connect_timed_out, _finish), hsm.target("/Room/closed")
            ),
            hsm.transition(hsm.on(LeaveEvent), hsm.effect(_leave, _finish), hsm.target("/Room/closed")),
        ),
        hsm.state(
            "linked",
            hsm.activity(_run_read, _run_write),
            hsm.initial(hsm.target("/Room/linked/opening")),
            hsm.transition(hsm.on(yamux.TransmitEvent), hsm.effect(_enqueue)),
            hsm.transition(hsm.on(transform.OwnedEvent), hsm.effect(_own)),
            hsm.transition(hsm.on(LeaveEvent), hsm.effect(_leave), hsm.target("/Room/closing")),
            hsm.transition(hsm.on(_SocketClosedEvent), hsm.effect(_close_for_socket), hsm.target("/Room/closing")),
            hsm.transition(hsm.on(yamux.EndedEvent), hsm.effect(_close_for_session), hsm.target("/Room/closing")),
            # A channel the computer reset, a socket that is not yamux, or a session that opens nothing more.
            hsm.transition(
                hsm.on(transform.EndedEvent, _BrokenEvent, yamux.OpenStreamFailedEvent),
                hsm.effect(_close_abnormal),
                hsm.target("/Room/closing"),
            ),
            hsm.state(
                "opening",
                hsm.entry(_open),
                hsm.transition(
                    hsm.on(transform.OwnedEvent),
                    hsm.guard(_opens_the_last),
                    hsm.effect(_own),
                    hsm.target("/Room/linked/streaming"),
                ),
                hsm.transition(hsm.after(_open_timeout_of), hsm.effect(_open_timed_out), hsm.target("/Room/closing")),
            ),
            hsm.state(
                "streaming",
                hsm.activity(_run_capture),
                hsm.exit(_silence),
                hsm.transition(hsm.on(audio.InputEvent), hsm.guard(_is_sound), hsm.effect(_play)),
            ),
        ),
        hsm.state(
            "closing",
            hsm.activity(_run_close),
            hsm.transition(hsm.on(_ReleasedEvent), hsm.effect(_finish), hsm.target("/Room/closed")),
            hsm.transition(hsm.after(_connect_timeout_of), hsm.effect(_finish), hsm.target("/Room/closed")),
        ),
        hsm.final("closed"),
    )


class _Stream(typing.Protocol):
    """One ``sounddevice`` raw stream, as the room uses it."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class _Filled(typing.Protocol):
    """The buffer PortAudio hands an output callback to fill: exactly ``frames`` frames of bytes."""

    def __setitem__(self, key: slice, value: bytes, /) -> None: ...


type _InputCallback = collections.abc.Callable[[collections.abc.Buffer, int, object, object], None]
type _OutputCallback = collections.abc.Callable[[_Filled, int, object, object], None]


class _SoundDevice(typing.Protocol):
    """The slice of ``sounddevice`` the room uses; the library ships no types of its own."""

    def RawInputStream(
        self, *, samplerate: float, blocksize: int, channels: int, dtype: str, callback: _InputCallback
    ) -> _Stream: ...

    def RawOutputStream(
        self, *, samplerate: float, channels: int, dtype: str, callback: _OutputCallback
    ) -> _Stream: ...


def _sounddevice() -> _SoundDevice:
    """Import ``sounddevice``, which loads PortAudio: only when a sound card is first opened."""

    import importlib

    # sounddevice ships no types: the module object is the surface _SoundDevice declares.
    return typing.cast(_SoundDevice, typing.cast(object, importlib.import_module("sounddevice")))


class SoundDeviceMicrophone:
    """The host's default input through PortAudio (``sounddevice``), at 16 kHz mono s16le.

    Each block of ``MICROPHONE_BLOCK_SECONDS`` is handed on from PortAudio's own thread as it is captured.
    """

    _load: collections.abc.Callable[[], _SoundDevice]
    _stream: _Stream | None

    def __init__(self, *, load: collections.abc.Callable[[], _SoundDevice] = _sounddevice) -> None:
        self._load = load
        self._stream = None

    def start(self, heard: collections.abc.Callable[[bytes], None], /) -> None:
        """Open and start the default input.

        Raises:
            Exception: whatever ``sounddevice`` raises when PortAudio or the input cannot be opened.
        """

        def captured(data: collections.abc.Buffer, frames: int, at: object, status: object) -> None:
            del frames, at, status
            heard(bytes(data))

        stream = self._load().RawInputStream(
            samplerate=MICROPHONE_RATE_HZ,
            blocksize=round(MICROPHONE_BLOCK_SECONDS * MICROPHONE_RATE_HZ),
            channels=1,
            dtype="int16",
            callback=captured,
        )
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        """Stop and close the input, if it is open."""

        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()


class SoundDeviceSpeaker:
    """The host's default output through PortAudio (``sounddevice``), opened at the rate of what it plays.

    What it is given waits in a buffer PortAudio's thread plays from, so ``play`` never blocks; at most
    ``max_queued_seconds`` of sound waits, and the oldest goes first. Sound at another rate or channel count
    reopens the output at that rate.
    """

    _load: collections.abc.Callable[[], _SoundDevice]
    _max_queued_seconds: float
    _lock: threading.Lock
    _queued: bytearray
    _stream: _Stream | None
    _format: tuple[int, int] | None

    def __init__(
        self, *, load: collections.abc.Callable[[], _SoundDevice] = _sounddevice, max_queued_seconds: float = 30.0
    ) -> None:
        self._load = load
        self._max_queued_seconds = max_queued_seconds
        self._lock = threading.Lock()
        self._queued = bytearray()
        self._stream = None
        self._format = None

    def play(self, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
        """Queue ``pcm`` to play.

        Raises:
            Exception: whatever ``sounddevice`` raises when PortAudio or the output cannot be opened.
        """

        if self._format != (sample_rate_hz, channels):
            self.stop()
            self._open(sample_rate_hz, channels)
        limit = round(self._max_queued_seconds * sample_rate_hz) * 2 * channels
        with self._lock:
            self._queued.extend(pcm)
            if len(self._queued) > limit:
                del self._queued[: len(self._queued) - limit]

    def _open(self, sample_rate_hz: int, channels: int) -> None:
        def fill(data: _Filled, frames: int, at: object, status: object) -> None:
            del at, status
            wanted = frames * 2 * channels
            with self._lock:
                chunk = bytes(self._queued[:wanted])
                del self._queued[:wanted]
            data[:] = chunk + bytes(wanted - len(chunk))

        stream = self._load().RawOutputStream(
            samplerate=sample_rate_hz, channels=channels, dtype="int16", callback=fill
        )
        stream.start()
        self._stream = stream
        self._format = (sample_rate_hz, channels)

    def stop(self) -> None:
        """Stop and close the output, and let go of what is queued."""

        stream, self._stream, self._format = self._stream, None, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            self._queued.clear()


class OpenCvCamera:
    """The host's camera ``index`` through OpenCV, a JPEG still every ``every`` seconds, long edge at most 1280 px.

    Stills are captured on a thread of the camera's own, which stops when the camera does.
    """

    _index: int
    _every: float
    _stopping: threading.Event
    _thread: threading.Thread | None

    def __init__(self, *, index: int = 0, every: float = CAMERA_EVERY_SECONDS) -> None:
        self._index = index
        self._every = every
        self._stopping = threading.Event()
        self._thread = None

    def start(self, seen: collections.abc.Callable[[media.Image], None], /) -> None:
        """Open the camera and start capturing on its own thread."""

        self._stopping.clear()
        self._thread = threading.Thread(target=self._capture, args=(seen,), name="companion-room-camera", daemon=True)
        self._thread.start()

    def _capture(self, seen: collections.abc.Callable[[media.Image], None]) -> None:
        import cv2

        capture = cv2.VideoCapture(self._index)
        try:
            while not self._stopping.is_set():
                read, frame = capture.read()
                if read:
                    height, width = typing.cast(tuple[int, ...], frame.shape)[:2]
                    scale = min(1.0, CAMERA_LONG_EDGE / max(height, width))
                    if scale < 1.0:
                        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                    encoded, jpeg = cv2.imencode(".jpg", frame)
                    if encoded:
                        seen(media.Image(content=jpeg.tobytes(), media_type="image/jpeg"))
                _ = self._stopping.wait(self._every)
        finally:
            capture.release()

    def stop(self) -> None:
        """Stop capturing and wait, briefly, for the camera to be let go."""

        self._stopping.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2 * self._every)


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
    camera: bool = False


async def _session(*, url: str, peer: str, token: pydantic.SecretStr, camera: bool) -> ClosedData:
    """Run one room session on this host's sound card (and camera) until it closes; return how it closed."""

    loop = asyncio.get_running_loop()
    closed: asyncio.Future[ClosedData] = loop.create_future()
    room = Room(
        url=url,
        peer=peer,
        token=token,
        microphone=SoundDeviceMicrophone(),
        speaker=SoundDeviceSpeaker(),
        camera=OpenCvCamera() if camera else None,
        on_closed=closed.set_result,
    )
    _ = await mosfet.started(None, room, Room.model)
    for stop in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(stop, lambda: room.dispatch(room.context(), LeaveEvent.with_data(LeaveData())))
    try:
        return await closed
    finally:
        for stop in (signal.SIGINT, signal.SIGTERM):
            _ = loop.remove_signal_handler(stop)
        await hsm.stop(room)


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    """Run ``python -m companion_example.room`` with ``argv`` (default: ``sys.argv[1:]``) and return its exit code."""

    parser = _Parser(
        prog="python -m companion_example.room",
        description=(
            "Be one room's far side of the companion's computer there: this host's microphone and speaker (and camera) "
            + "in the room. SIGINT or SIGTERM leaves. Exit codes: 0 left or closed normally, 1 failed or refused, "
            + "64 bad flags, 78 missing secret."
        ),
    )
    _ = parser.add_argument(
        "--url", default=_DEFAULT_URL, help=f"the companion's computer server; default: {_DEFAULT_URL}"
    )
    _ = parser.add_argument("--peer", default=_DEFAULT_PEER, help=f"the room's name; default: {_DEFAULT_PEER}")
    _ = parser.add_argument(
        "--secret-file",
        type=pathlib.Path,
        help="file holding computer_token_secret; else COMPUTER_TOKEN_SECRET_FILE, else COMPUTER_TOKEN_SECRET",
    )
    _ = parser.add_argument("--camera", action="store_true", help="also send this host's camera")
    try:
        arguments = parser.parse_args(argv, namespace=_Arguments())
        _ = peer_uri(arguments.url, arguments.peer)
    except (_UsageError, ValueError) as error:
        print(f"companion-room: {error}", file=sys.stderr)
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
        print(f"companion-room: {error}", file=sys.stderr)
        return _CONFIG
    except OSError as error:
        print(f"companion-room: cannot read secret {_SECRET!r}: {error.strerror}", file=sys.stderr)
        return _CONFIG
    except ValueError:
        print(f"companion-room: secret {_SECRET!r} is empty", file=sys.stderr)
        return _CONFIG

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        _ = telemetry.configure(service_name="companion-room")
    except (OSError, ValueError) as error:
        print(f"companion-room: cannot configure telemetry: {error}", file=sys.stderr)
        return _CONFIG
    try:
        closed = asyncio.run(_session(url=arguments.url, peer=arguments.peer, token=token, camera=arguments.camera))
    finally:
        telemetry.reset()
    if closed.reason not in ("left", "normal"):
        detail = "" if closed.code is None else f" ({closed.code})"
        print(f"companion-room: session ended: {closed.reason}{detail}", file=sys.stderr)
    return closed.exit_code


__all__ = [
    "CAMERA_EVERY_SECONDS",
    "CAMERA_LONG_EDGE",
    "MAX_LINE",
    "MAX_PENDING",
    "MICROPHONE_BLOCK_SECONDS",
    "MICROPHONE_RATE_HZ",
    "SCOPE",
    "Camera",
    "Channel",
    "ClosedData",
    "Connect",
    "ConnectOutcome",
    "Keyboard",
    "LeaveData",
    "LeaveEvent",
    "Microphone",
    "OpenCvCamera",
    "Reason",
    "Room",
    "Socket",
    "SoundDeviceMicrophone",
    "SoundDeviceSpeaker",
    "Speaker",
    "connect_websocket",
    "main",
    "peer_uri",
]

if __name__ == "__main__":
    raise SystemExit(main())
