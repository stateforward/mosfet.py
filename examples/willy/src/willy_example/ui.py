"""A room's far side in a browser: talk to Willy, hear it, and type to it, from one page.

    python -m willy_example.ui --url ws://127.0.0.1:8765 --peer living-room \\
        --secret-file examples/willy/secrets/computer_token_secret [--port 8780] [--camera]

The bridge is a room's far side (``room.Room``) whose devices are a page in the browser. It serves that page,
one self-contained HTML file, at ``http://127.0.0.1:<port>/`` (loopback only: there is no host to choose) and
the page's WebSocket at ``/ws``, which only a page served from this bridge may open (its ``Origin`` must be this
bridge's own). Opening the page in a second tab takes the bridge over from the first.

The room connects as ``--peer`` and opens the computer's ``microphone``, ``speaker`` and ``keyboard`` channels
(and ``camera`` with ``--camera``), exactly as ``willy_example.room`` does, so Willy's computer cannot tell the two
apart. Between the page and the bridge:

* the page's microphone, captured and downsampled to 16 kHz mono s16le in the browser, comes as binary messages
  tagged :data:`MICROPHONE` and goes to the room's microphone;
* a line typed on the page comes as ``{"type": "line", "text": ...}`` (:class:`LineData`) and goes to the room's
  keyboard, which types it on the computer key by key, then Enter: with nothing focused there, whoever holds the
  computer is told ``computer.text.entered`` with no field;
* with ``--camera``, a JPEG still from the page's camera comes, at the room's camera cadence, as a binary message
  tagged :data:`CAMERA` and goes to the room's camera;
* whatever the computer plays on the room's speaker goes to the page as a binary message: its rate (``uint32``
  little-endian, 24 kHz for Willy's voice) and channel count (``uint8``), then the s16le PCM, which the page
  plays in order, without gaps; when the room stops its speaker, the page is told :class:`SilenceData`;
* the page is told the room's state as :class:`StateData` (``connecting``, ``linked``, ``unlinked`` and, once,
  ``closed`` with why), and each message the bridge could not take as :class:`DroppedData`.

The bridge and the page decide nothing: they capture, carry and play. Echo cancellation is the browser's (the
page asks for an echo-cancelled microphone, which cancels what the page itself plays).

The secret is ``computer_token_secret``, resolved as ``willy_example.room`` resolves it. Exit codes mirror the
room's: ``0`` when the bridge is left (SIGINT or SIGTERM) or Willy's computer closes the session normally, ``1``
when the connection fails, times out, is refused or ends abnormally, or the page cannot be served on its port,
``64`` for bad flags, ``78`` for a missing or unreadable secret. The bridge writes nothing to stdout; the page's
address and its diagnostics go to stderr.

Telemetry (meter scope ``willy_example.ui``), every attribute a closed set from code:
``bot.example.ui.page.count{outcome}`` (``opened``, ``replaced``, ``refused``),
``bot.example.ui.messages.count{kind}`` (``microphone``, ``camera`` and ``line`` taken from the page, ``speaker``
sent to it) and ``bot.example.ui.dropped.count{kind,why}`` (``kind`` one of those or ``message``; ``why`` one of
``invalid``, ``not_linked``, ``busy``, ``no_page``). The room's own telemetry is ``willy_example.room``'s. The
peer, the url, the token, every line typed and every sound and image never are attributes.
"""

import argparse
import asyncio
import collections.abc
import contextlib
import http
import importlib.resources
import json
import logging
import os
import pathlib
import signal
import struct
import sys
import typing
import urllib.parse

import hsm
import mosfet
import pydantic
import websockets.asyncio.server
import websockets.exceptions
import websockets.http11
from mosfet import media, runtime, telemetry
from mosfet.providers import websocket
from opentelemetry import metrics

from willy_example import room

SCOPE: typing.Final = "willy_example.ui"
HOST: typing.Final = "127.0.0.1"
"""The only address the page is served on."""
DEFAULT_PORT: typing.Final = 8780
PAGE_PATH: typing.Final = "/"
SOCKET_PATH: typing.Final = "/ws"
MICROPHONE: typing.Final = 0x01
"""The tag byte of a page message carrying microphone PCM: 16 kHz mono s16le."""
CAMERA: typing.Final = 0x02
"""The tag byte of a page message carrying one JPEG still."""
MAX_MICROPHONE_BYTES: typing.Final = 2 * room.MICROPHONE_RATE_HZ
"""The most sound one microphone message carries: one second."""
MAX_MESSAGE_BYTES: typing.Final = 1 + media.MAX_IMAGE_BYTES
"""The largest message the page may send: a tagged still."""
MAX_QUEUED: typing.Final = 256
"""Most messages waiting for the page; past it, the newest is dropped and counted."""

type State = typing.Literal["connecting", "linked", "unlinked", "closed"]
type Kind = typing.Literal["microphone", "camera", "line", "speaker", "message"]
type Why = typing.Literal["invalid", "not_linked", "busy", "no_page"]

_LOG = logging.getLogger(__name__)
_METER = metrics.get_meter(SCOPE)
_PAGES = _METER.create_counter(
    "bot.example.ui.page.count",
    unit="{page}",
    description="Pages that opened the bridge's socket, took it over from another page, or were refused.",
)
_MESSAGES = _METER.create_counter(
    "bot.example.ui.messages.count",
    unit="{message}",
    description="Messages carried, by kind: microphone, camera and line from the page, speaker to it.",
)
_DROPPED = _METER.create_counter(
    "bot.example.ui.dropped.count",
    unit="{message}",
    description="Messages the bridge could not carry, by kind and why.",
)

_DEFAULT_URL = "ws://127.0.0.1:8765"
_DEFAULT_PEER = "living-room"
_SECRET = "computer_token_secret"
_SECRETS_DIRECTORY = pathlib.Path("/run/secrets")
_SOUND_HEADER = struct.Struct("<IB")
_REPLACED = 4000
_DRAIN_SECONDS = 1.0
_SETTINGS_SLOT = "/*SETTINGS*/null"

_USAGE = 64
_CONFIG = 78


class LineData(pydantic.BaseModel):
    """A line the person typed on the page and entered, for the room's keyboard to type."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, extra="forbid", json_schema_extra={"examples": [{"type": "line", "text": "Hello, Willy"}]}
    )

    type: typing.Literal["line"] = pydantic.Field(description="What the message is: a typed line.")
    text: str = pydantic.Field(
        min_length=1,
        max_length=room.MAX_LINE,
        pattern=r"^[^\x00-\x1f\x7f]*$",
        description="The line, without its line ending or any other control character.",
        examples=["Hello, Willy"],
    )


class StateData(pydantic.BaseModel):
    """Where the room is, told to the page whenever it changes and when the page opens."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {"type": "state", "state": "linked", "reason": None, "code": None},
                {"type": "state", "state": "closed", "reason": "refused", "code": 4401},
            ]
        },
    )

    type: typing.Literal["state"] = pydantic.Field(default="state", description="What the message is.")
    state: State = pydantic.Field(
        description=(
            "`connecting` until the room's channels are open, `linked` while they carry the page's devices, "
            "`unlinked` once they stop, and `closed` when the session has ended."
        ),
        examples=["linked"],
    )
    reason: room.Reason | None = pydantic.Field(
        default=None, description="Why the session ended, once `closed`.", examples=[None, "refused"]
    )
    code: int | None = pydantic.Field(
        default=None, description="The close code Willy's computer sent, when it sent one.", examples=[None, 4401]
    )


class DroppedData(pydantic.BaseModel):
    """A message from the page the bridge did not carry, and why."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, json_schema_extra={"examples": [{"type": "dropped", "what": "line", "why": "not_linked"}]}
    )

    type: typing.Literal["dropped"] = pydantic.Field(default="dropped", description="What the message is.")
    what: Kind = pydantic.Field(description="What was dropped.", examples=["line"])
    why: Why = pydantic.Field(
        description=(
            "`invalid` when the message is not one the page sends, `not_linked` when the room's device for it is not "
            "running, `busy` when too much waits for the link."
        ),
        examples=["not_linked"],
    )


class SilenceData(pydantic.BaseModel):
    """The room stopped its speaker: the page lets go of what it has not played yet."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    type: typing.Literal["silence"] = pydantic.Field(default="silence", description="What the message is.")


class _Settings(pydantic.BaseModel):
    """What the page is told when it is served."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    camera: bool
    microphone_tag: int = MICROPHONE
    camera_tag: int = CAMERA
    microphone_rate_hz: int = room.MICROPHONE_RATE_HZ
    microphone_block_samples: int = round(room.MICROPHONE_BLOCK_SECONDS * room.MICROPHONE_RATE_HZ)
    camera_every_seconds: float = room.CAMERA_EVERY_SECONDS
    camera_long_edge: int = room.CAMERA_LONG_EDGE
    max_line: int = room.MAX_LINE


def _dropped(kind: Kind, why: Why) -> None:
    _DROPPED.add(1, {"kind": kind, "why": why})


class _Inlet[T]:
    """One of the room's capture devices, fed by the page: the room starts and stops it, the page hands it on."""

    _changed: collections.abc.Callable[[bool], None] | None
    _handler: collections.abc.Callable[[T], None] | None

    def __init__(self, changed: collections.abc.Callable[[bool], None] | None = None) -> None:
        self._changed = changed
        self._handler = None

    def start(self, handler: collections.abc.Callable[[T], None]) -> None:
        """The room is linked: hand what the page sends to ``handler``."""

        self._handler = handler
        if self._changed is not None:
            self._changed(True)

    def stop(self) -> None:
        """The room let go of this device: nothing more is handed on."""

        self._handler = None
        if self._changed is not None:
            self._changed(False)

    def hand(self, item: T) -> bool:
        """Hand ``item`` to the room, if it is running this device; whether it was."""

        handler = self._handler
        if handler is None:
            return False
        handler(item)
        return True


class _Speaker:
    """The room's speaker: what the computer plays goes to the page, its rate and channel count first."""

    _browser: "Browser"

    def __init__(self, browser: "Browser") -> None:
        self._browser = browser

    def play(self, pcm: bytes, *, sample_rate_hz: int, channels: int) -> None:
        """Send ``pcm`` to the page, to play after what it is already playing."""

        if self._browser.send("speaker", _SOUND_HEADER.pack(sample_rate_hz, channels) + pcm):
            _MESSAGES.add(1, {"kind": "speaker"})

    def stop(self) -> None:
        """Tell the page to let go of what it has not played."""

        _ = self._browser.send("speaker", SilenceData().model_dump_json())


class Browser:
    """The page in the browser, as a room's devices: ``microphone``, ``speaker``, ``keyboard`` and ``camera``.

    It is the websocket handler for the page's socket (:meth:`page`): what the page sends is handed to the
    room's capture devices while the room runs them, and what the room's speaker plays is sent to the page. One
    page at a time: a page that opens the socket takes it over, and the one before it is closed. What waits for
    the page is bounded by :data:`MAX_QUEUED`.
    """

    microphone: _Inlet[bytes]
    keyboard: _Inlet[str]
    camera: _Inlet[media.Image]
    speaker: _Speaker
    # The page's socket and what waits to be sent on it; written only by page().
    _page: tuple[websockets.asyncio.server.ServerConnection, asyncio.Queue[str | bytes]] | None
    # What the page is told on opening; written only by the microphone's changes and closed().
    _state: StateData

    def __init__(self) -> None:
        self.microphone = _Inlet(self._linked)
        self.keyboard = _Inlet()
        self.camera = _Inlet()
        self.speaker = _Speaker(self)
        self._page = None
        self._state = StateData(state="connecting")

    def _linked(self, linked: bool) -> None:
        if self._state.state != "closed":
            self._tell(StateData(state="linked" if linked else "unlinked"))

    def closed(self, data: room.ClosedData) -> None:
        """The room's session ended: tell the page how."""

        self._tell(StateData(state="closed", reason=data.reason, code=data.code))

    def _tell(self, state: StateData) -> None:
        self._state = state
        _ = self.send("message", state.model_dump_json())

    def send(self, kind: Kind, message: str | bytes) -> bool:
        """Queue ``message`` for the page, if one is open and has room for it; whether it was queued."""

        current = self._page
        if current is None:
            if kind == "speaker":
                _dropped(kind, "no_page")
            return False
        try:
            current[1].put_nowait(message)
        except asyncio.QueueFull:
            _dropped(kind, "busy")
            return False
        return True

    async def drained(self) -> None:
        """Wait until the open page has been sent everything queued for it."""

        current = self._page
        if current is not None:
            await current[1].join()

    async def page(self, connection: websockets.asyncio.server.ServerConnection) -> None:
        """Serve one page for as long as its socket is open."""

        outbox: asyncio.Queue[str | bytes] = asyncio.Queue(maxsize=MAX_QUEUED)
        previous, self._page = self._page, (connection, outbox)
        outbox.put_nowait(self._state.model_dump_json())
        _PAGES.add(1, {"outcome": "opened"})
        if previous is not None:
            _PAGES.add(1, {"outcome": "replaced"})
            await previous[0].close(_REPLACED, "another page took over")
        writer = asyncio.create_task(self._write(connection, outbox), name="willy-ui-page-writer")
        try:
            async for message in connection:
                self._take(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            _ = writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await writer
            if self._page is not None and self._page[0] is connection:
                self._page = None

    @staticmethod
    async def _write(
        connection: websockets.asyncio.server.ServerConnection, outbox: asyncio.Queue[str | bytes]
    ) -> None:
        while True:
            message = await outbox.get()
            try:
                await connection.send(message)
            except websockets.exceptions.ConnectionClosed:
                return
            finally:
                outbox.task_done()

    def _take(self, message: str | bytes) -> None:
        """Hand one message from the page to the room's device for it, or tell the page it was dropped."""

        kind: Kind
        if isinstance(message, str):
            try:
                line = LineData.model_validate_json(message)
            except pydantic.ValidationError as error:
                text_only = all(problem["loc"][:1] == ("text",) for problem in error.errors())
                self._drop("line" if text_only else "message", "invalid")
                return
            kind, handed = "line", self.keyboard.hand(line.text)
        elif message[:1] == bytes((MICROPHONE,)):
            pcm = message[1:]
            if not pcm or len(pcm) % 2 or len(pcm) > MAX_MICROPHONE_BYTES:
                self._drop("microphone", "invalid")
                return
            kind, handed = "microphone", self.microphone.hand(pcm)
        elif message[:1] == bytes((CAMERA,)):
            try:
                still = media.Image(content=message[1:], media_type="image/jpeg")
            except pydantic.ValidationError:
                self._drop("camera", "invalid")
                return
            kind, handed = "camera", self.camera.hand(still)
        else:
            self._drop("message", "invalid")
            return
        if not handed:
            self._drop(kind, "not_linked")
            return
        _MESSAGES.add(1, {"kind": kind})

    def _drop(self, kind: Kind, why: Why) -> None:
        _dropped(kind, why)
        _ = self.send("message", DroppedData(what=kind, why=why).model_dump_json())


def page(*, camera: bool) -> str:
    """The page, with its settings written into it; read from this package, so it is file I/O."""

    template = importlib.resources.files("willy_example").joinpath("ui.html").read_text(encoding="utf-8")
    return template.replace(_SETTINGS_SLOT, json.dumps(_Settings(camera=camera).model_dump()), 1)


@contextlib.asynccontextmanager
async def serve(
    browser: Browser, *, port: int, page: str
) -> collections.abc.AsyncIterator[websockets.asyncio.server.Server]:
    """Serve ``page`` at ``/`` and ``browser``'s socket at ``/ws`` on loopback ``port`` (``0``: any free one).

    Only a page served from this bridge may open the socket: its ``Origin`` must be ``http://127.0.0.1:<port>``
    or ``http://localhost:<port>``.

    Raises:
        OSError: the port cannot be bound.
    """

    def answer(
        connection: websockets.asyncio.server.ServerConnection, request: websockets.http11.Request
    ) -> websockets.http11.Response | None:
        path = urllib.parse.urlsplit(request.path).path
        if path == PAGE_PATH:
            response = connection.respond(http.HTTPStatus.OK, page)
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            response.headers["Cache-Control"] = "no-store"
            return response
        if path != SOCKET_PATH:
            return connection.respond(http.HTTPStatus.NOT_FOUND, "Not Found\n")
        bound = typing.cast(tuple[str, int], connection.local_address)[1]
        if request.headers.get("Origin") not in (f"http://{HOST}:{bound}", f"http://localhost:{bound}"):
            _PAGES.add(1, {"outcome": "refused"})
            return connection.respond(http.HTTPStatus.FORBIDDEN, "Forbidden\n")
        return None

    async with websockets.asyncio.server.serve(
        browser.page, HOST, port, process_request=answer, max_size=MAX_MESSAGE_BYTES
    ) as server:
        yield server


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
    port: int = DEFAULT_PORT
    camera: bool = False


def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"{text!r} is not a port, 0 to 65535")
    return port


async def _session(
    *, url: str, peer: str, token: pydantic.SecretStr, camera: bool, port: int, page: str
) -> room.ClosedData:
    """Serve the page and run one room session on it until the session closes; return how it closed.

    Raises:
        OSError: the page cannot be served on ``port``.
    """

    loop = asyncio.get_running_loop()
    closed: asyncio.Future[room.ClosedData] = loop.create_future()
    browser = Browser()

    def on_closed(data: room.ClosedData) -> None:
        browser.closed(data)
        closed.set_result(data)

    async with serve(browser, port=port, page=page) as server:
        bound = typing.cast(tuple[str, int], next(iter(server.sockets)).getsockname())[1]
        print(f"willy-ui: open http://{HOST}:{bound}/ in a browser", file=sys.stderr, flush=True)
        machine = room.Room(
            url=url,
            peer=peer,
            token=token,
            microphone=browser.microphone,
            speaker=browser.speaker,
            keyboard=browser.keyboard,
            camera=browser.camera if camera else None,
            on_closed=on_closed,
        )
        _ = await mosfet.started(None, machine, room.Room.model)
        for stop in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                stop, lambda: machine.dispatch(machine.context(), room.LeaveEvent.with_data(room.LeaveData()))
            )
        try:
            ended = await closed
            # Let the page hear how the session ended before its socket goes.
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(_DRAIN_SECONDS):
                    await browser.drained()
            return ended
        finally:
            for stop in (signal.SIGINT, signal.SIGTERM):
                _ = loop.remove_signal_handler(stop)
            await hsm.stop(machine)


def main(argv: collections.abc.Sequence[str] | None = None) -> int:
    """Run ``python -m willy_example.ui`` with ``argv`` (default: ``sys.argv[1:]``) and return its exit code."""

    parser = _Parser(
        prog="python -m willy_example.ui",
        description=(
            "Be one room's far side of Willy's computer in a browser: talk, listen and type to Willy from a page "
            + f"served on {HOST}. SIGINT or SIGTERM leaves. Exit codes: 0 left or closed normally, 1 failed or "
            + "refused, 64 bad flags, 78 missing secret."
        ),
    )
    _ = parser.add_argument("--url", default=_DEFAULT_URL, help=f"Willy's computer server; default: {_DEFAULT_URL}")
    _ = parser.add_argument("--peer", default=_DEFAULT_PEER, help=f"the room's name; default: {_DEFAULT_PEER}")
    _ = parser.add_argument(
        "--secret-file",
        type=pathlib.Path,
        help="file holding computer_token_secret; else COMPUTER_TOKEN_SECRET_FILE, else COMPUTER_TOKEN_SECRET",
    )
    _ = parser.add_argument(
        "--port", type=_port, default=DEFAULT_PORT, help=f"the page's port on {HOST}; default: {DEFAULT_PORT}"
    )
    _ = parser.add_argument("--camera", action="store_true", help="also send the browser's camera")
    try:
        arguments = parser.parse_args(argv, namespace=_Arguments())
        _ = room.peer_uri(arguments.url, arguments.peer)
    except (_UsageError, ValueError) as error:
        print(f"willy-ui: {error}", file=sys.stderr)
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
        print(f"willy-ui: {error}", file=sys.stderr)
        return _CONFIG
    except OSError as error:
        print(f"willy-ui: cannot read secret {_SECRET!r}: {error.strerror}", file=sys.stderr)
        return _CONFIG
    except ValueError:
        print(f"willy-ui: secret {_SECRET!r} is empty", file=sys.stderr)
        return _CONFIG

    served = page(camera=arguments.camera)
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    try:
        _ = telemetry.configure(service_name="willy-ui")
    except (OSError, ValueError) as error:
        print(f"willy-ui: cannot configure telemetry: {error}", file=sys.stderr)
        return _CONFIG
    try:
        closed = asyncio.run(
            _session(
                url=arguments.url,
                peer=arguments.peer,
                token=token,
                camera=arguments.camera,
                port=arguments.port,
                page=served,
            )
        )
    except OSError as error:
        print(f"willy-ui: cannot serve the page on {HOST}:{arguments.port}: {error.strerror}", file=sys.stderr)
        return 1
    finally:
        telemetry.reset()
    if closed.reason not in ("left", "normal"):
        detail = "" if closed.code is None else f" ({closed.code})"
        print(f"willy-ui: session ended: {closed.reason}{detail}", file=sys.stderr)
    return closed.exit_code


__all__ = [
    "CAMERA",
    "DEFAULT_PORT",
    "HOST",
    "MAX_MESSAGE_BYTES",
    "MAX_MICROPHONE_BYTES",
    "MAX_QUEUED",
    "MICROPHONE",
    "PAGE_PATH",
    "SCOPE",
    "SOCKET_PATH",
    "Browser",
    "DroppedData",
    "Kind",
    "LineData",
    "SilenceData",
    "State",
    "StateData",
    "Why",
    "main",
    "page",
    "serve",
]

if __name__ == "__main__":
    raise SystemExit(main())
