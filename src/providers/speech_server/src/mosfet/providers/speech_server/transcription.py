"""A streaming speech-to-text session over a speech server's WebSocket protocol (``hearing.speech.stream``).

One :class:`Transcription` is one session for one sound source. Its controller attaches to it
(``attachment.attach``); while attached it keeps itself connected to
``<url>/v1/audio/stream?sample_rate=16000&language=<language>&chunk=<chunk_seconds>&turns=1`` and
reconnects after any loss with a growing backoff (``backoff``: first, then doubled up to last). Detaching
says ``{"type":"end"}`` and closes the socket.

Wire (client → server): audio resampled to s16le mono 16 kHz (``resample.to_mono``, in a worker thread)
in ``frame``-sized binary frames; ``{"type":"agent","speaking":bool}`` whenever the bot starts or stops
producing (``stream.ProducingEvent``, sent only on change); ``{"type":"end"}`` on detach. Audio that
arrives before the server says ``ready`` is held, at most :data:`MAX_UNREADY_SECONDS` of it (the oldest
let go), and sent once it is.

Time: the server's clock counts the audio it has been sent on one connection and starts at zero on each.
The session reports seconds of the audio it has received instead (the ``hearing.speech.stream`` contract):
each connection's clock is placed at the session time of the first audio sent on it, and audio let go
(past the hold, past the send queue, or while detached) still advances session time though the server never
hears it.

Wire (server → client), each frame mapped to one ``hearing.speech.stream`` output to the controller:
``ready``; ``speech_start``/``speech_end`` (voice started/ended); ``partial``; ``final`` (transcript, its
span from the word times when given); ``turn_end`` (turn ended); ``interrupt`` (overlap: the server says
someone has talked over the agent for at least :data:`INTERRUPT_SECONDS`). ``eot``, ``resumed`` and
``rejoin`` are not part of the contract and are dropped and counted. An ``error`` frame, a lost socket or
an unreachable server is ``unavailable``; close 1013 is ``overloaded`` and 1003 ``invalid_input``; a frame
outside the protocol is ``invalid_response``; a server that does not connect or say ``ready`` within
``connect_timeout`` is ``timeout``. Each is a ``stream.FailedEvent`` to the controller, then the backoff.

Telemetry (low cardinality; never URLs, text or audio):

* ``bot.hearing.speech.stream.session.active`` (up/down counter): connected sessions;
* ``bot.hearing.speech.stream.event.count{kind}``: outputs by kind, ``ignored`` for dropped frames;
* ``bot.hearing.speech.stream.failed.count{kind}``: failures by kind;
* ``bot.hearing.speech.stream.dropped.count{reason}``: audio not sent (``unready`` past the hold,
  ``overflow`` past the send queue, ``detached`` with no controller);
* span ``bot.hearing.speech.stream.connect``; INFO logs on open and close.
"""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import dataclasses
import datetime
import json
import logging
import typing
import urllib.parse

import hsm
import pydantic
import websockets.asyncio.client
import websockets.exceptions
from opentelemetry import metrics
from pydantic.json_schema import SkipJsonSchema

from mosfet.abilities.hearing.speech import stream
from mosfet.define import define
from mosfet.protocols import attachment
from mosfet.telemetry import span
from mosfet.telemetry.hsm import Traced

from . import resample

_LOG = logging.getLogger(__name__)
_METER = metrics.get_meter("mosfet.providers.speech_server")
_ACTIVE = _METER.create_up_down_counter(
    "bot.hearing.speech.stream.session.active",
    unit="{session}",
    description="Streaming speech-to-text sessions connected now.",
)
_EVENTS = _METER.create_counter(
    "bot.hearing.speech.stream.event.count",
    unit="{event}",
    description="Streaming session outputs by kind; ignored counts server frames outside the contract.",
)
_FAILED = _METER.create_counter(
    "bot.hearing.speech.stream.failed.count",
    unit="{failure}",
    description="Streaming session failures by kind.",
)
_DROPPED = _METER.create_counter(
    "bot.hearing.speech.stream.dropped.count",
    unit="{chunk}",
    description="Audio chunks not sent, by reason: unready (past the hold), overflow (send queue), detached.",
)

RATE_HZ: typing.Final = 16_000
"""The audio the server takes: s16le mono at this rate."""
MAX_UNREADY_SECONDS: typing.Final = 2.0
"""How much audio is held while the server is not ready; older audio is let go."""
INTERRUPT_SECONDS: typing.Final = 0.8
"""How long the server's ``interrupt`` says another voice has talked over the agent, at least."""
_MAX_QUEUE = 512
_MAX_CLOCK_POINTS = 64
_DEFAULT_CONNECT_TIMEOUT = datetime.timedelta(seconds=5)
_DEFAULT_BACKOFF = (datetime.timedelta(seconds=1), datetime.timedelta(seconds=30))
_DEFAULT_FRAME = datetime.timedelta(milliseconds=50)
_MAX_MESSAGE_BYTES = 1 << 20
_CLOSE_OVERLOADED = 1013
_CLOSE_INVALID = 1003

type _Outbound = str | stream.AudioData
type _EventKind = typing.Literal[
    "ready", "voice_started", "voice_ended", "partial", "transcript", "turn_ended", "overlap", "failed", "ignored"
]


class Closed(Exception):
    """The socket closed; ``code`` is the close code the server sent, when it sent one."""

    code: int | None

    def __init__(self, code: int | None) -> None:
        super().__init__(f"socket closed ({code})")
        self.code = code


class Socket(typing.Protocol):
    """A connected WebSocket, as a session uses it."""

    async def send(self, message: str | bytes) -> None:
        """Send one text or binary message."""
        ...

    async def recv(self) -> str | bytes:
        """Receive one message; raises :class:`Closed` once the socket is closed."""
        ...

    async def close(self) -> None:
        """Close the socket."""
        ...


type Connect = collections.abc.Callable[[str], collections.abc.Awaitable[Socket]]
"""Open a WebSocket to the given URL."""


class _WebSocket:
    """A ``websockets`` client connection as a :class:`Socket`."""

    def __init__(self, connection: websockets.asyncio.client.ClientConnection) -> None:
        self._connection: websockets.asyncio.client.ClientConnection = connection

    async def send(self, message: str | bytes) -> None:
        try:
            await self._connection.send(message)
        except websockets.exceptions.ConnectionClosed as error:
            raise Closed(None if error.rcvd is None else error.rcvd.code) from error

    async def recv(self) -> str | bytes:
        try:
            return await self._connection.recv()
        except websockets.exceptions.ConnectionClosed as error:
            raise Closed(None if error.rcvd is None else error.rcvd.code) from error

    async def close(self) -> None:
        await self._connection.close()


async def _connect(url: str) -> Socket:
    connection = await websockets.asyncio.client.connect(
        url, compression=None, max_size=_MAX_MESSAGE_BYTES, open_timeout=None
    )
    return _WebSocket(connection)


def _seconds(data: stream.AudioData) -> float:
    """How long ``data`` plays, counting whole frames only (as the controller counts what it sends)."""

    return len(data.audio) // (2 * data.channels) / data.sample_rate_hz


class _ProtocolError(Exception):
    """A server frame outside the protocol."""


class _Decoder:
    """Server frames to contract outputs, remembering when the current voice started (for overlaps)."""

    def __init__(self) -> None:
        self.voice_started_at: float | None = None

    @staticmethod
    def _number(frame: dict[str, object], key: str) -> float:
        value = frame.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise _ProtocolError(f"{key} is not a time")
        return float(value)

    @staticmethod
    def _optional(frame: dict[str, object], key: str) -> float | None:
        value = frame.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _ProtocolError(f"{key} is not a number")
        return min(1.0, max(0.0, float(value)))

    @staticmethod
    def _text(frame: dict[str, object], key: str, *, required: bool) -> str | None:
        value = frame.get(key)
        if value is None and not required:
            return None
        if not isinstance(value, str):
            raise _ProtocolError(f"{key} is not text")
        return value

    def decode(self, message: str | bytes) -> hsm.Event[typing.Any] | None:
        """The contract output a server frame says, or ``None`` for a frame outside the contract.

        Raises:
            _ProtocolError: The frame is not a JSON object of the protocol.
            Closed: The frame is the server's ``error``; the session treats it as a lost server.
        """

        if isinstance(message, bytes):
            raise _ProtocolError("binary frame from the server")
        try:
            frame = typing.cast(object, json.loads(message))
        except ValueError as error:
            raise _ProtocolError("frame is not JSON") from error
        if not isinstance(frame, dict):
            raise _ProtocolError("frame is not an object")
        fields = {str(key): value for key, value in typing.cast(dict[object, object], frame).items()}
        kind = fields.get("type")
        try:
            match kind:
                case "ready":
                    return stream.ReadyEvent.with_data(stream.ReadyData())
                case "speech_start":
                    at = self._number(fields, "t")
                    self.voice_started_at = at
                    return stream.VoiceStartedEvent.with_data(stream.VoiceStartedData(at_seconds=at))
                case "speech_end":
                    return stream.VoiceEndedEvent.with_data(stream.VoiceEndedData(at_seconds=self._number(fields, "t")))
                case "partial":
                    return stream.PartialEvent.with_data(
                        stream.PartialData(text=self._text(fields, "text", required=True) or "")
                    )
                case "final":
                    return stream.TranscriptEvent.with_data(self._transcript(fields))
                case "turn_end":
                    return stream.TurnEndedEvent.with_data(
                        stream.TurnEndedData(
                            at_seconds=self._number(fields, "t"),
                            probability=self._optional(fields, "p"),
                            text=self._text(fields, "text", required=False),
                        )
                    )
                case "interrupt":
                    at = self._number(fields, "t")
                    started = self.voice_started_at
                    voiced = at - started if started is not None and started <= at else INTERRUPT_SECONDS
                    return stream.OverlapEvent.with_data(stream.OverlapData(at_seconds=at, voiced_seconds=voiced))
                case "eot" | "resumed" | "rejoin":
                    return None
                case "error":
                    raise Closed(None)
                case _:
                    raise _ProtocolError("unknown frame type")
        except pydantic.ValidationError as error:
            raise _ProtocolError("frame fields out of range") from error

    def _transcript(self, fields: dict[str, object]) -> stream.TranscriptData:
        text = self._text(fields, "text", required=True) or ""
        words = fields.get("words")
        times: list[float] = []
        if isinstance(words, list):
            for word in typing.cast(list[object], words):
                if isinstance(word, dict):
                    at = typing.cast(dict[str, object], word).get("t")
                    if isinstance(at, int | float) and not isinstance(at, bool) and at >= 0:
                        times.append(float(at))
        start = times[0] if times else (self.voice_started_at or 0.0)
        end = max(times[-1] if times else start, start)
        return stream.TranscriptData(
            text=text, confidence=self._optional(fields, "confidence"), start_seconds=start, end_seconds=end
        )


class _ConnectedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    socket: SkipJsonSchema[pydantic.SkipValidation[Socket]] = pydantic.Field(
        exclude=True, repr=False, description="The connected socket. Runtime-only."
    )


class _ReceivedData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True)

    output: SkipJsonSchema[pydantic.SkipValidation[hsm.Event[typing.Any]]] = pydantic.Field(
        exclude=True, repr=False, description="The contract output a server frame said."
    )


class _LostData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    kind: stream.FailureKind


# Private: each is dispatched by this class's own activities to itself only.
_ConnectedEvent = hsm.Event[_ConnectedData](
    name="providers.speech_server.transcription.connected", kind=hsm.CompletionEventKind, schema=_ConnectedData
)
_ReceivedEvent = hsm.Event[_ReceivedData](name="providers.speech_server.transcription.received", schema=_ReceivedData)
_LostEvent = hsm.Event[_LostData](
    name="providers.speech_server.transcription.lost", kind=hsm.ErrorEventKind, schema=_LostData
)


def _lost_kind(error: BaseException) -> stream.FailureKind:
    if isinstance(error, _ProtocolError):
        return "invalid_response"
    if isinstance(error, Closed) and error.code == _CLOSE_OVERLOADED:
        return "overloaded"
    if isinstance(error, Closed) and error.code == _CLOSE_INVALID:
        return "invalid_input"
    return "unavailable"


class Transcription(Traced, attachment.Attachment):
    """One streaming session to a speech server; its one controller attaches to it and hears what it hears."""

    _attachment_limit: typing.ClassVar[int | None] = 1
    _attachments: list[hsm.Instance]
    _attachment_timeout: datetime.timedelta
    _attachment_request_id: str

    _url: str
    _connect: Connect
    _connect_timeout: datetime.timedelta
    _first_backoff: datetime.timedelta
    _last_backoff: datetime.timedelta
    _frame_bytes: int
    # Durable across connections: whether the bot is producing (latest word from the controller), the
    # next backoff, and the audio held while the server is not ready (bounded by MAX_UNREADY_SECONDS).
    _producing: bool
    _retry_delay: datetime.timedelta
    _unready: collections.deque[stream.AudioData]
    _unready_seconds: float
    # Session time: the audio received so far, and the current connection's clock as (server seconds,
    # session seconds) points: its origin, then one more at each server time where audio sent on it was let
    # go (bounded by _MAX_CLOCK_POINTS: past it the oldest point after the origin folds into it, see _queue),
    # with the audio queued on it so far in server seconds.
    _received_seconds: float
    _clock: collections.deque[tuple[float, float]]
    _sent_seconds: float
    # The current connection: its socket, its send queue, and what the server was last told about the
    # bot's mouth. Written only by _keep_socket / _forget_socket and the ready transitions.
    _socket: Socket | None
    _outbound: asyncio.Queue[_Outbound] | None
    _agent_told: bool

    def __init__(
        self,
        *,
        url: str,
        language: str = "en",
        chunk_seconds: float = 0.56,
        connect: Connect | None = None,
        connect_timeout: datetime.timedelta = _DEFAULT_CONNECT_TIMEOUT,
        backoff: tuple[datetime.timedelta, datetime.timedelta] = _DEFAULT_BACKOFF,
        frame: datetime.timedelta = _DEFAULT_FRAME,
    ) -> None:
        """Build an unstarted session for the speech server at ``url`` (``ws://`` or ``wss://``, no path).

        ``chunk_seconds`` is the server's decoding chunk; ``frame`` the size of each audio message sent.

        Raises:
            ValueError: If ``url`` is not a WebSocket URL, ``language`` is empty, or a duration is not
                positive, or the backoff's first step is longer than its last.
        """

        super().__init__()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("ws", "wss") or not parsed.netloc:
            raise ValueError("url must be a ws:// or wss:// server URL.")
        if not language:
            raise ValueError("language must not be empty.")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")
        first, last = backoff
        for name, value in (("connect_timeout", connect_timeout), ("frame", frame), ("backoff", first)):
            if value <= datetime.timedelta(0):
                raise ValueError(f"{name} must be positive.")
        if last < first:
            raise ValueError("backoff must not shrink: (first, last).")
        query = urllib.parse.urlencode(
            {"sample_rate": RATE_HZ, "language": language, "chunk": chunk_seconds, "turns": 1}
        )
        self._url = f"{url.rstrip('/')}/v1/audio/stream?{query}"
        self._connect = connect if connect is not None else _connect
        self._connect_timeout = connect_timeout
        self._first_backoff = first
        self._last_backoff = last
        self._frame_bytes = 2 * max(1, round(frame.total_seconds() * RATE_HZ))
        self._attachments = []
        self._attachment_timeout = datetime.timedelta(seconds=30)
        self._attachment_request_id = ""
        self._producing = False
        self._retry_delay = first
        self._unready = collections.deque()
        self._unready_seconds = 0.0
        self._received_seconds = 0.0
        self._clock = collections.deque([(0.0, 0.0)])
        self._sent_seconds = 0.0
        self._socket = None
        self._outbound = None
        self._agent_told = False

    @typing.override
    def attach(self, ctx: hsm.Context, event: hsm.Event[attachment.AttachData]) -> collections.abc.Awaitable[bool]:
        """Deliver one ``attachment.attach`` request; the answer goes to its ``reply_to`` (else its actor)."""

        return self.dispatch(ctx, event)

    @typing.override
    def detach(self, ctx: hsm.Context, event: hsm.Event[attachment.DetachData]) -> collections.abc.Awaitable[bool]:
        """Deliver one ``attachment.detach`` request; the answer goes to its ``reply_to`` (else its actor)."""

        return self.dispatch(ctx, event)

    # -- to the controller ------------------------------------------------------------------------

    @staticmethod
    def _tell(ctx: hsm.Context, instance: Transcription, output: hsm.Event[typing.Any], kind: _EventKind) -> None:
        _EVENTS.add(1, {"kind": kind})
        for controller in instance._attachments:
            _ = hsm.dispatch(
                ctx,
                controller,
                dataclasses.replace(
                    output, source=hsm.id(instance), target=attachment.Attachment._actor_id(controller)
                ),
            )

    @staticmethod
    def _fail(ctx: hsm.Context, instance: Transcription, kind: stream.FailureKind) -> None:
        _FAILED.add(1, {"kind": kind})
        _LOG.info("speech-server stream session lost kind=%s retry_in=%ss", kind, instance._retry_delay.total_seconds())
        Transcription._tell(ctx, instance, stream.FailedEvent.with_data(stream.FailedData(kind=kind)), "failed")

    @staticmethod
    def _report_lost(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _LostData)
        Transcription._fail(ctx, instance, data.kind)

    @staticmethod
    def _report_timeout(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        del event
        Transcription._fail(ctx, instance, "timeout")

    @staticmethod
    def _relay(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        """A server frame's output, to the controller."""

        data = event.data
        assert isinstance(data, _ReceivedData)
        output = data.output
        kind = typing.cast(_EventKind, output.name.rsplit(".", 1)[-1])
        said = typing.cast(object, output.data)
        match said:
            case stream.VoiceStartedData() | stream.VoiceEndedData() | stream.TurnEndedData() | stream.OverlapData():
                at = Transcription._session_time(instance, said.at_seconds)
                output = dataclasses.replace(output, data=said.model_copy(update={"at_seconds": at}))
            case stream.TranscriptData():
                start = Transcription._session_time(instance, said.start_seconds)
                end = Transcription._session_time(instance, said.end_seconds)
                output = dataclasses.replace(
                    output, data=said.model_copy(update={"start_seconds": start, "end_seconds": max(start, end)})
                )
            case _:
                pass
        Transcription._tell(ctx, instance, output, kind)

    @staticmethod
    def _session_time(instance: Transcription, at: float) -> float:
        """A time on the current connection's clock, as seconds of the audio this session has received."""

        server, session = instance._clock[0]
        for point in instance._clock:
            if point[0] > at:
                break
            server, session = point
        return session + at - server

    @staticmethod
    def _start_clock(instance: Transcription) -> None:
        """A new connection's clock starts at the oldest audio still held: the first it will be sent."""

        instance._clock = collections.deque([(0.0, instance._received_seconds - instance._unready_seconds)])
        instance._sent_seconds = 0.0

    # -- audio --------------------------------------------------------------------------------------

    @staticmethod
    def _hold(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        """Hold audio while the server is not ready, letting the oldest go past the hold."""

        del ctx
        data = event.data
        assert isinstance(data, stream.AudioData)
        instance._received_seconds += _seconds(data)
        instance._unready.append(data)
        instance._unready_seconds += _seconds(data)
        while instance._unready_seconds > MAX_UNREADY_SECONDS and instance._unready:
            oldest = instance._unready.popleft()
            instance._unready_seconds -= _seconds(oldest)
            _DROPPED.add(1, {"reason": "unready"})

    @staticmethod
    def _queue(instance: Transcription, message: _Outbound) -> None:
        outbound = instance._outbound
        assert outbound is not None, "only a connected session queues messages"
        try:
            outbound.put_nowait(message)
        except asyncio.QueueFull:
            _DROPPED.add(1, {"reason": "overflow"})
            if not isinstance(message, str):
                # The server never hears it, so its clock falls behind session time by that much from here.
                clock, at = instance._clock, instance._sent_seconds
                point = (at, Transcription._session_time(instance, at) + _seconds(message))
                if clock[-1][0] == at:
                    # Nothing was sent since the last let go: a run of them is one point.
                    clock[-1] = point
                    return
                clock.append(point)
                if len(clock) > _MAX_CLOCK_POINTS:
                    # The origin stays; the oldest point after it folds into it. Server times before that
                    # point stay exact, and those from it to the next point map earlier by at most the audio
                    # let go at the points folded so far.
                    del clock[1]
            return
        if not isinstance(message, str):
            instance._sent_seconds += _seconds(message)

    @staticmethod
    def _send_audio(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.AudioData)
        instance._received_seconds += _seconds(data)
        Transcription._queue(instance, data)

    @staticmethod
    def _drop_detached(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.AudioData)
        instance._received_seconds += _seconds(data)
        _DROPPED.add(1, {"reason": "detached"})

    # -- the bot's mouth ---------------------------------------------------------------------------

    @staticmethod
    def _agent(producing: bool) -> str:
        return json.dumps({"type": "agent", "speaking": producing})

    @staticmethod
    def _remember_producing(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, stream.ProducingData)
        instance._producing = data.producing

    @staticmethod
    def _tell_producing(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        """Tell the server when the bot started or stopped producing; only on change."""

        Transcription._remember_producing(ctx, instance, event)
        if instance._producing != instance._agent_told:
            instance._agent_told = instance._producing
            Transcription._queue(instance, Transcription._agent(instance._producing))

    # -- connecting ---------------------------------------------------------------------------------

    @staticmethod
    async def _run_connect(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        """Connect; bounded by ``connect_timeout`` in topology."""

        del event
        with span.operation(
            "bot.hearing.speech.stream.connect",
            scope="mosfet.providers.speech_server",
            component="speech_server.transcription",
            stage="connect",
        ) as active:
            try:
                socket = await instance._connect(instance._url)
            except Exception as error:
                kind = _lost_kind(error)
                span.record_failure(active, kind)
                _ = hsm.dispatch(ctx, instance, _LostEvent.with_data(_LostData(kind=kind)))
                return
        _ = hsm.dispatch(ctx, instance, _ConnectedEvent.with_data(_ConnectedData(socket=socket)))

    @staticmethod
    def _keep_socket(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _ConnectedData)
        instance._socket = data.socket
        instance._outbound = asyncio.Queue(maxsize=_MAX_QUEUE)
        instance._agent_told = False
        Transcription._start_clock(instance)
        _ACTIVE.add(1)
        _LOG.info("speech-server stream session open")

    @staticmethod
    def _forget_socket(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._socket = None
        instance._outbound = None
        _ACTIVE.add(-1)
        _LOG.info("speech-server stream session closed")

    @staticmethod
    def _become_ready(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        """The server is ready: say so, send what was held, and tell it about the bot's mouth."""

        Transcription._start_clock(instance)
        Transcription._relay(ctx, instance, event)
        instance._retry_delay = instance._first_backoff
        if instance._producing:
            instance._agent_told = True
            Transcription._queue(instance, Transcription._agent(True))
        while instance._unready:
            Transcription._queue(instance, instance._unready.popleft())
        instance._unready_seconds = 0.0

    @staticmethod
    def _grow_backoff(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._retry_delay = min(instance._retry_delay * 2, instance._last_backoff)

    @staticmethod
    def _connect_timeout_value(
        ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]
    ) -> datetime.timedelta:
        del ctx, event
        return instance._connect_timeout

    @staticmethod
    def _backoff_value(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._retry_delay

    @staticmethod
    def _is_ready_frame(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, _ReceivedData) and isinstance(data.output.data, stream.ReadyData)

    # -- streaming ----------------------------------------------------------------------------------

    @staticmethod
    async def _send(instance: Transcription, socket: Socket, outbound: asyncio.Queue[_Outbound]) -> None:
        """Send queued messages in order; audio is resampled and cut into whole frames."""

        carry = b""
        while True:
            message = await outbound.get()
            if isinstance(message, str):
                await socket.send(message)
                continue
            audio = await asyncio.to_thread(
                resample.to_mono, message.audio, sample_rate_hz=message.sample_rate_hz, channels=message.channels
            )
            carry += audio
            size = instance._frame_bytes
            while len(carry) >= size:
                frame, carry = carry[:size], carry[size:]
                await socket.send(frame)

    @staticmethod
    async def _receive(ctx: hsm.Context, instance: Transcription, socket: Socket) -> None:
        """Map each server frame to its contract output, dispatched to this session in order."""

        decoder = _Decoder()
        while True:
            output = decoder.decode(await socket.recv())
            if output is None:
                _EVENTS.add(1, {"kind": "ignored"})
                continue
            _ = hsm.dispatch(ctx, instance, _ReceivedEvent.with_data(_ReceivedData(output=output)))

    @staticmethod
    async def _run_stream(ctx: hsm.Context, instance: Transcription, event: hsm.Event[typing.Any]) -> None:
        """Send and receive until the socket is lost; leaving this state says ``end`` and closes it."""

        del event
        socket, outbound = instance._socket, instance._outbound
        assert socket is not None and outbound is not None
        try:
            async with asyncio.TaskGroup() as group:
                _ = group.create_task(Transcription._send(instance, socket, outbound))
                _ = group.create_task(Transcription._receive(ctx, instance, socket))
        except* Exception as failure:
            # The boundary: whatever ended the socket is classified, reported, and recovered from.
            _ = hsm.dispatch(ctx, instance, _LostEvent.with_data(_LostData(kind=_lost_kind(failure.exceptions[0]))))
        finally:
            try:
                await socket.send(json.dumps({"type": "end"}))
            except (Closed, OSError):
                pass
            try:
                await socket.close()
            except (Closed, OSError):
                pass

    model: typing.ClassVar[hsm.Model] = define(
        "Transcription",
        hsm.initial(hsm.target("/Transcription/detached")),
        hsm.state(
            "detached",
            hsm.transition(
                hsm.on(attachment.AttachEvent),
                hsm.guard(attachment.Attachment._can_attach),
                hsm.effect(attachment.Attachment._attach, attachment.Attachment._dispatch_attach_complete_created),
                hsm.target("/Transcription/attached"),
            ),
            hsm.transition(hsm.on(stream.AudioEvent), hsm.effect(_drop_detached)),
        ),
        hsm.state(
            "attached",
            hsm.initial(hsm.target("/Transcription/attached/connecting")),
            hsm.transition(
                hsm.on(attachment.DetachEvent),
                hsm.guard(attachment.Attachment._is_attached),
                hsm.effect(attachment.Attachment._detach, attachment.Attachment._dispatch_detach_complete_removed),
                hsm.target("/Transcription/detached"),
            ),
            hsm.transition(hsm.on(stream.AudioEvent), hsm.effect(_hold)),
            hsm.transition(hsm.on(stream.ProducingEvent), hsm.effect(_remember_producing)),
            hsm.state(
                "connecting",
                hsm.activity(_run_connect),
                hsm.transition(
                    hsm.on(_ConnectedEvent), hsm.effect(_keep_socket), hsm.target("/Transcription/attached/streaming")
                ),
                hsm.transition(
                    hsm.on(_LostEvent), hsm.effect(_report_lost), hsm.target("/Transcription/attached/backing_off")
                ),
                hsm.transition(
                    hsm.after(_connect_timeout_value),
                    hsm.effect(_report_timeout),
                    hsm.target("/Transcription/attached/backing_off"),
                ),
            ),
            hsm.state(
                "streaming",
                hsm.initial(hsm.target("/Transcription/attached/streaming/waiting_ready")),
                hsm.exit(_forget_socket),
                hsm.activity(_run_stream),
                hsm.transition(
                    hsm.on(_LostEvent), hsm.effect(_report_lost), hsm.target("/Transcription/attached/backing_off")
                ),
                hsm.transition(hsm.on(_ReceivedEvent), hsm.effect(_relay)),
                hsm.state(
                    "waiting_ready",
                    hsm.transition(
                        hsm.on(_ReceivedEvent),
                        hsm.guard(_is_ready_frame),
                        hsm.effect(_become_ready),
                        hsm.target("/Transcription/attached/streaming/ready"),
                    ),
                    hsm.transition(
                        hsm.after(_connect_timeout_value),
                        hsm.effect(_report_timeout),
                        hsm.target("/Transcription/attached/backing_off"),
                    ),
                ),
                hsm.state(
                    "ready",
                    hsm.transition(hsm.on(stream.AudioEvent), hsm.effect(_send_audio)),
                    hsm.transition(hsm.on(stream.ProducingEvent), hsm.effect(_tell_producing)),
                ),
            ),
            hsm.state(
                "backing_off",
                hsm.transition(
                    hsm.after(_backoff_value),
                    hsm.effect(_grow_backoff),
                    hsm.target("/Transcription/attached/connecting"),
                ),
            ),
        ),
        # Anything else asking to attach while one controller holds the session is refused.
        hsm.transition(
            hsm.on(attachment.AttachEvent),
            hsm.guard(attachment.Attachment._is_attach_request),
            hsm.effect(attachment.Attachment._dispatch_attach_failed),
        ),
        hsm.transition(
            hsm.on(attachment.DetachEvent),
            hsm.guard(attachment.Attachment._is_detach_request),
            hsm.effect(attachment.Attachment._dispatch_detach_complete_absent),
        ),
    )
    """Detached, or attached: connecting, streaming (waiting for ready, then ready), backing off."""


__all__ = ["INTERRUPT_SECONDS", "MAX_UNREADY_SECONDS", "RATE_HZ", "Closed", "Connect", "Socket", "Transcription"]
