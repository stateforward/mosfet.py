import collections.abc
import asyncio
import dataclasses
import datetime
import types
import typing

import hsm
import pydantic

from bot.protocols.yamux.events import (
    CloseStreamEvent,
    GoAwayEvent,
    OpenStreamEvent,
    PingEvent,
    ReceiveDataFrameEvent,
    ReceiveGoAwayFrameEvent,
    ReceivePingFrameEvent,
    ReceiveWindowUpdateFrameEvent,
    ResetStreamEvent,
    SendDataEvent,
    SessionDrainedEvent,
    CloseStreamData,
    GoAwayData,
    OpenStreamData,
    PingData,
    ReceiveDataFrameData,
    ReceiveGoAwayFrameData,
    ReceivePingFrameData,
    ReceiveWindowUpdateFrameData,
    ResetStreamData,
    SendData,
    SessionDrainedData,
    StreamIdData,
    FailureKind,
    SessionStage,
)
from bot.protocols.yamux.frame import (
    INITIAL_STREAM_WINDOW,
    Flag,
    FrameData,
    FrameType,
    GoAwayCode,
)
from bot.protocols.yamux.stream import Stream, StreamSnapshot, StreamState

Role = typing.Literal["client", "server"]
DEFAULT_PING_TIMEOUT = datetime.timedelta(seconds=30)
_MAX_UINT32 = (1 << 32) - 1
TSessionReturn = typing.TypeVar("TSessionReturn")


@dataclasses.dataclass(frozen=True, slots=True)
class SessionFailure:
    """Normalized recoverable or terminal session failure."""

    stage: SessionStage
    failure_kind: FailureKind
    stream_id: int | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SessionSnapshot(hsm.Snapshot):
    """Stable observation of a Yamux session without exposing mutable HSM-owned members."""

    initial_stream_window: int
    next_stream_id: int
    accepts_new_streams: bool
    outstanding_ping: int | None
    closed_goaway_code: GoAwayCode | None
    ping_timeout: datetime.timedelta
    streams: collections.abc.Mapping[int, StreamSnapshot]
    failed_operations: tuple[SessionFailure, ...]
    has_active_streams: bool


class FrameStream:
    """Async stream of outbound Yamux frames emitted by a session."""

    _frames: asyncio.Queue[FrameData]

    def __init__(self, frames: asyncio.Queue[FrameData]) -> None:
        self._frames = frames

    async def read(self) -> FrameData:
        """Read the next outbound frame."""

        return await self._frames.get()

    def empty(self) -> bool:
        """Return true when no outbound frame is waiting to be read."""

        return self._frames.empty()


class _FrameSink:
    _frames: asyncio.Queue[FrameData]

    def __init__(self, frames: asyncio.Queue[FrameData]) -> None:
        self._frames = frames

    def write(self, frame: FrameData) -> None:
        self._frames.put_nowait(frame)


class ReadStream:
    """Read-only Python stream facade for application-owned inbound bytes."""

    __slots__: typing.ClassVar[tuple[str, ...]] = (
        "__at_eof",
        "__read",
        "__readexactly",
        "__readline",
        "__take_snapshot",
    )

    __at_eof: collections.abc.Callable[[], bool]
    __read: collections.abc.Callable[[int], collections.abc.Awaitable[bytes]]
    __readexactly: collections.abc.Callable[[int], collections.abc.Awaitable[bytes]]
    __readline: collections.abc.Callable[[], collections.abc.Awaitable[bytes]]
    __take_snapshot: collections.abc.Callable[[], StreamSnapshot]

    def __init__(
        self,
        *,
        take_snapshot: collections.abc.Callable[[], StreamSnapshot],
        at_eof: collections.abc.Callable[[], bool],
        read: collections.abc.Callable[[int], collections.abc.Awaitable[bytes]],
        readexactly: collections.abc.Callable[[int], collections.abc.Awaitable[bytes]],
        readline: collections.abc.Callable[[], collections.abc.Awaitable[bytes]],
    ) -> None:
        self.__take_snapshot = take_snapshot
        self.__at_eof = at_eof
        self.__read = read
        self.__readexactly = readexactly
        self.__readline = readline

    def take_snapshot(self) -> StreamSnapshot:
        """Return a stable observation of the underlying Yamux stream."""

        return self.__take_snapshot()

    def at_eof(self) -> bool:
        """Return true when the inbound byte stream reached EOF and its buffer is empty."""

        return self.__at_eof()

    async def read(self, n: int = -1) -> bytes:
        """Read bytes from the inbound Yamux stream."""

        return await self.__read(n)

    async def readexactly(self, n: int) -> bytes:
        """Read exactly n bytes from the inbound Yamux stream."""

        return await self.__readexactly(n)

    async def readline(self) -> bytes:
        """Read one line from the inbound Yamux stream."""

        return await self.__readline()


class _WindowConsumedData(pydantic.BaseModel):
    """Private event payload emitted when application code reads stream bytes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1, "byte_count": 4096}]},
    )

    stream_id: typing.Annotated[
        int,
        pydantic.Field(
            ge=1,
            le=(1 << 32) - 1,
            description="Yamux stream whose Python reader consumed bytes.",
            examples=[1],
        ),
    ]
    byte_count: typing.Annotated[
        int,
        pydantic.Field(
            ge=1,
            le=(1 << 32) - 1,
            description="Number of stream payload bytes consumed by application code.",
            examples=[4096],
        ),
    ]


_WindowConsumedEvent = hsm.Event[_WindowConsumedData](
    name="protocol.yamux.session.window_consumed",
    schema=_WindowConsumedData,
)


def event_from_frame(frame: FrameData) -> hsm.Event[typing.Any]:
    """Project a byte-validated Yamux frame into the typed event consumed by Session."""

    if frame.frame_type is FrameType.DATA:
        return ReceiveDataFrameEvent.with_data(
            ReceiveDataFrameData(stream_id=frame.stream_id, flags=int(frame.flags), payload=frame.payload)
        )
    if frame.frame_type is FrameType.WINDOW_UPDATE:
        return ReceiveWindowUpdateFrameEvent.with_data(
            ReceiveWindowUpdateFrameData(stream_id=frame.stream_id, flags=int(frame.flags), delta=frame.length)
        )
    if frame.frame_type is FrameType.PING:
        ping_flags = typing.cast(typing.Literal[1, 2], int(frame.flags))
        return ReceivePingFrameEvent.with_data(ReceivePingFrameData(opaque=frame.length, flags=ping_flags))
    return ReceiveGoAwayFrameEvent.with_data(
        ReceiveGoAwayFrameData(code=typing.cast(typing.Literal[0, 1, 2], frame.length))
    )


def _wrap_session_method(
    fn: typing.Callable[["Session", hsm.Context, hsm.Event[typing.Any]], TSessionReturn],
) -> typing.Callable[[hsm.Context, "Session", hsm.Event[typing.Any]], TSessionReturn]:
    method_name = fn.__name__

    def wrapped(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> TSessionReturn:
        method = typing.cast(
            typing.Callable[[hsm.Context, hsm.Event[typing.Any]], TSessionReturn],
            getattr(instance, method_name),
        )
        return method(ctx, event)

    wrapped.__name__ = method_name
    return wrapped


def _session_ping_timeout(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> datetime.timedelta:
    del ctx, event
    return _session_ping_timeout_value(instance)


def _send_data_event_data(event: hsm.Event[typing.Any]) -> SendData | None:
    data = event.data
    return data if isinstance(data, SendData) else None


def _stream_id_event_data(event: hsm.Event[typing.Any]) -> StreamIdData | None:
    data = event.data
    return data if isinstance(data, StreamIdData) else None


def _received_data_event_data(event: hsm.Event[typing.Any]) -> ReceiveDataFrameData | None:
    data = event.data
    return data if isinstance(data, ReceiveDataFrameData) else None


def _received_window_event_data(event: hsm.Event[typing.Any]) -> ReceiveWindowUpdateFrameData | None:
    data = event.data
    return data if isinstance(data, ReceiveWindowUpdateFrameData) else None


def _window_consumed_event_data(event: hsm.Event[typing.Any]) -> _WindowConsumedData | None:
    data = event.data
    return data if isinstance(data, _WindowConsumedData) else None


def _flags(value: int) -> Flag:
    return Flag(value)


def _stream_is_writable(stream: Stream) -> bool:
    return _stream_state(stream) in {
        StreamState.LOCAL_OPENING,
        StreamState.OPEN,
        StreamState.REMOTE_CLOSED,
    }


def _stream_is_readable(stream: Stream) -> bool:
    return _stream_state(stream) in {
        StreamState.LOCAL_OPENING,
        StreamState.OPEN,
        StreamState.LOCAL_CLOSED,
    }


def _stream_is_terminal(stream: Stream) -> bool:
    return _stream_state(stream) in {StreamState.CLOSED, StreamState.RESET}


def _stream_accepts_ack(stream: Stream) -> bool:
    return _stream_awaiting_ack(stream)


def _data_frame_is_ack_only_for_stream(stream: Stream, data: ReceiveDataFrameData) -> bool:
    flags = _flags(data.flags)
    return (
        _stream_awaiting_ack(stream)
        and bool(flags & Flag.ACK)
        and not bool(flags & (Flag.SYN | Flag.FIN | Flag.RST))
        and len(data.payload) == 0
    )


def _window_frame_is_ack_only_for_stream(stream: Stream, data: ReceiveWindowUpdateFrameData) -> bool:
    flags = _flags(data.flags)
    return (
        _stream_awaiting_ack(stream)
        and bool(flags & Flag.ACK)
        and not bool(flags & (Flag.SYN | Flag.FIN | Flag.RST))
        and data.delta == 0
    )


def _peer_stream_id_is_valid(role: Role, stream_id: int) -> bool:
    if role == "client":
        return stream_id % 2 == 0
    return stream_id % 2 == 1


def _session_role(instance: "Session") -> Role:
    role = typing.cast(Role, object.__getattribute__(instance, "_role"))
    assert role in ("client", "server")
    return role


def _session_initial_stream_window(instance: "Session") -> int:
    return typing.cast(int, object.__getattribute__(instance, "_initial_stream_window"))


def _session_next_stream_id(instance: "Session") -> int:
    return typing.cast(int, object.__getattribute__(instance, "_next_stream_id"))


def _session_accepts_new_streams(instance: "Session") -> bool:
    return typing.cast(bool, object.__getattribute__(instance, "_accept_new_streams"))


def _session_outstanding_ping(instance: "Session") -> int | None:
    return typing.cast(int | None, object.__getattribute__(instance, "_outstanding_ping"))


def _session_ping_timeout_value(instance: "Session") -> datetime.timedelta:
    return typing.cast(datetime.timedelta, object.__getattribute__(instance, "_ping_timeout"))


def _session_streams(instance: "Session") -> dict[int, Stream]:
    return typing.cast(dict[int, Stream], object.__getattribute__(instance, "_streams"))


def _session_stream(instance: "Session", stream_id: int) -> Stream | None:
    return _session_streams(instance).get(stream_id)


def _stream_state(stream: Stream) -> StreamState:
    return typing.cast(StreamState, object.__getattribute__(stream, "_state"))


def _stream_awaiting_ack(stream: Stream) -> bool:
    return typing.cast(bool, object.__getattribute__(stream, "_awaiting_ack"))


def _stream_send_window(stream: Stream) -> int:
    return typing.cast(int, object.__getattribute__(stream, "_send_window"))


def _stream_receive_window(stream: Stream) -> int:
    return typing.cast(int, object.__getattribute__(stream, "_receive_window"))


def _has_stream_id_capacity(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx, event
    return _session_accepts_new_streams(instance) and _session_next_stream_id(instance) <= _MAX_UINT32


def _cannot_open_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    return isinstance(event.data, OpenStreamData) and (
        not _session_accepts_new_streams(instance) or _session_next_stream_id(instance) > _MAX_UINT32
    )


def _has_active_streams(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx, event
    return any(not _stream_is_terminal(stream) for stream in _session_streams(instance).values())


def _can_send_data(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and _stream_is_writable(stream) and len(data.payload) <= _stream_send_window(stream)


def _send_data_unknown_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    return data is not None and _session_stream(instance, data.stream_id) is None


def _send_data_inactive_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and not _stream_is_writable(stream)


def _send_data_exhausts_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and _stream_is_writable(stream) and len(data.payload) > _stream_send_window(stream)


def _can_close_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and _stream_is_writable(stream)


def _stream_command_unknown(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    return data is not None and _session_stream(instance, data.stream_id) is None


def _stream_command_inactive(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and not _stream_is_writable(stream)


def _can_reset_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and not _stream_is_terminal(stream)


def _received_data_has_duplicate_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and bool(_flags(data.flags) & Flag.SYN)
        and _session_stream(instance, data.stream_id) is not None
    )


def _received_data_has_bad_peer_stream_id(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and _session_stream(instance, data.stream_id) is None
        and bool(_flags(data.flags) & Flag.SYN)
        and not _peer_stream_id_is_valid(_session_role(instance), data.stream_id)
    )


def _received_data_has_bad_ack(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None or not bool(_flags(data.flags) & Flag.ACK):
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is None or not _stream_accepts_ack(stream)


def _received_data_unknown_without_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and _session_stream(instance, data.stream_id) is None
        and not bool(_flags(data.flags) & Flag.SYN)
    )


def _received_data_hits_terminal_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and _stream_is_terminal(stream) and not _data_frame_is_ack_only_for_stream(stream, data)


def _received_data_hits_inactive_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None or bool(_flags(data.flags) & Flag.RST):
        return False
    stream = _session_stream(instance, data.stream_id)
    return (
        stream is not None and not _stream_is_readable(stream) and not _data_frame_is_ack_only_for_stream(stream, data)
    )


def _received_data_exhausts_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None or bool(_flags(data.flags) & Flag.RST):
        return False
    stream = _session_stream(instance, data.stream_id)
    if stream is None:
        return bool(_flags(data.flags) & Flag.SYN) and len(data.payload) > _session_initial_stream_window(instance)
    return _stream_is_readable(stream) and len(data.payload) > _stream_receive_window(stream)


def _can_receive_existing_data(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    if stream is None:
        return False
    if bool(_flags(data.flags) & Flag.RST):
        return not _stream_is_terminal(stream)
    if _data_frame_is_ack_only_for_stream(stream, data):
        return True
    if _stream_is_terminal(stream):
        return False
    return _stream_is_readable(stream) and len(data.payload) <= _stream_receive_window(stream)


def _can_receive_new_data(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and _session_accepts_new_streams(instance)
        and _session_stream(instance, data.stream_id) is None
        and bool(_flags(data.flags) & Flag.SYN)
        and _peer_stream_id_is_valid(_session_role(instance), data.stream_id)
        and len(data.payload) <= _session_initial_stream_window(instance)
    )


def _received_data_new_stream_while_draining(
    ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]
) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and not _session_accepts_new_streams(instance)
        and _session_stream(instance, data.stream_id) is None
    )


def _received_window_has_duplicate_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and bool(_flags(data.flags) & Flag.SYN)
        and _session_stream(instance, data.stream_id) is not None
    )


def _received_window_has_bad_peer_stream_id(
    ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]
) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and _session_stream(instance, data.stream_id) is None
        and bool(_flags(data.flags) & Flag.SYN)
        and not _peer_stream_id_is_valid(_session_role(instance), data.stream_id)
    )


def _received_window_has_bad_ack(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None or not bool(_flags(data.flags) & Flag.ACK):
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is None or not _stream_accepts_ack(stream)


def _received_window_unknown_without_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and _session_stream(instance, data.stream_id) is None
        and not bool(_flags(data.flags) & Flag.SYN)
    )


def _received_window_hits_terminal_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and _stream_is_terminal(stream) and not _window_frame_is_ack_only_for_stream(stream, data)


def _received_window_has_bad_fin(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None or not bool(_flags(data.flags) & Flag.FIN) or bool(_flags(data.flags) & Flag.RST):
        return False
    stream = _session_stream(instance, data.stream_id)
    return (
        stream is not None
        and not _stream_is_readable(stream)
        and not _window_frame_is_ack_only_for_stream(stream, data)
    )


def _received_window_overflows(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None or bool(_flags(data.flags) & Flag.RST):
        return False
    stream = _session_stream(instance, data.stream_id)
    if stream is None:
        return bool(_flags(data.flags) & Flag.SYN) and INITIAL_STREAM_WINDOW + data.delta > _MAX_UINT32
    return not _stream_is_terminal(stream) and _stream_send_window(stream) + data.delta > _MAX_UINT32


def _can_receive_existing_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    data = _received_window_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return (
        stream is not None
        and (not _stream_is_terminal(stream) or _window_frame_is_ack_only_for_stream(stream, data))
        and (bool(_flags(data.flags) & Flag.RST) or not _received_window_overflows(ctx, instance, event))
    )


def _can_receive_new_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and _session_accepts_new_streams(instance)
        and _session_stream(instance, data.stream_id) is None
        and bool(_flags(data.flags) & Flag.SYN)
        and _peer_stream_id_is_valid(_session_role(instance), data.stream_id)
        and INITIAL_STREAM_WINDOW + data.delta <= _MAX_UINT32
    )


def _received_window_new_stream_while_draining(
    ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]
) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and not _session_accepts_new_streams(instance)
        and _session_stream(instance, data.stream_id) is None
    )


def _can_grant_receive_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _window_consumed_event_data(event)
    if data is None:
        return False
    stream = _session_stream(instance, data.stream_id)
    return stream is not None and _stream_is_readable(stream)


def _is_inbound_ping(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx, instance
    data = event.data
    return isinstance(data, ReceivePingFrameData) and data.flags == int(Flag.SYN)


def _is_outstanding_ping_ack(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = event.data
    return (
        isinstance(data, ReceivePingFrameData)
        and data.flags == int(Flag.ACK)
        and _session_outstanding_ping(instance) == data.opaque
    )


class Session(hsm.Instance):
    """HSM-owned Yamux session coordinating normal Python streams."""

    _accept_new_streams: bool
    _closed_goaway_code: GoAwayCode | None
    _failed_operations: list[SessionFailure]
    _frame_sink: _FrameSink
    _initial_stream_window: int
    _next_stream_id: int
    _outbound: FrameStream
    _outstanding_ping: int | None
    _ping_timeout: datetime.timedelta
    _role: Role
    _streams: dict[int, Stream]

    def __init__(
        self,
        *,
        role: str,
        initial_stream_window: int = INITIAL_STREAM_WINDOW,
        ping_timeout: datetime.timedelta = DEFAULT_PING_TIMEOUT,
    ) -> None:
        super().__init__()
        if role not in ("client", "server"):
            raise ValueError("role must be 'client' or 'server'.")
        _require_uint32("initial_stream_window", initial_stream_window, minimum=INITIAL_STREAM_WINDOW)
        if ping_timeout <= datetime.timedelta():
            raise ValueError("ping_timeout must be positive.")
        self._role = role
        self._initial_stream_window = initial_stream_window
        self._next_stream_id = 1 if role == "client" else 2
        self._streams = {}
        outbound_frames: asyncio.Queue[FrameData] = asyncio.Queue()
        self._frame_sink = _FrameSink(outbound_frames)
        self._outbound = FrameStream(outbound_frames)
        self._failed_operations = []
        self._accept_new_streams = True
        self._closed_goaway_code = None
        self._outstanding_ping = None
        self._ping_timeout = ping_timeout

    @typing.override
    def take_snapshot(self) -> SessionSnapshot:
        snapshot = super().take_snapshot()
        streams = {stream_id: stream.take_snapshot() for stream_id, stream in self._streams.items()}
        return SessionSnapshot(
            ID=snapshot.ID,
            QualifiedName=snapshot.QualifiedName,
            State=snapshot.State,
            Attributes=snapshot.Attributes,
            QueueLen=snapshot.QueueLen,
            Transitions=snapshot.Transitions,
            initial_stream_window=self._initial_stream_window,
            next_stream_id=self._next_stream_id,
            accepts_new_streams=self._accept_new_streams,
            outstanding_ping=self._outstanding_ping,
            closed_goaway_code=self._closed_goaway_code,
            ping_timeout=self._ping_timeout,
            streams=typing.cast(collections.abc.Mapping[int, StreamSnapshot], types.MappingProxyType(streams)),
            failed_operations=tuple(self._failed_operations),
            has_active_streams=self._active_streams_exist(),
        )

    def outbound(self) -> FrameStream:
        """Return the async stream of outbound Yamux frames."""

        return self._outbound

    def stream(self, stream_id: int) -> ReadStream | None:
        """Return a read-only Python stream for application reads."""

        stream = self._streams.get(stream_id)
        if stream is None:
            return None
        return ReadStream(
            take_snapshot=stream.take_snapshot,
            at_eof=stream.at_eof,
            read=stream.read,
            readexactly=stream.readexactly,
            readline=stream.readline,
        )

    async def receive_frame(self, ctx: hsm.Context, frame: FrameData) -> None:
        """Dispatch a decoded Yamux frame into the session HSM."""

        await self.dispatch(ctx, event_from_frame(frame))

    async def _dispatch_window_consumed(self, stream_id: int, byte_count: int) -> None:
        await self.dispatch(
            self.context(),
            _WindowConsumedEvent.with_data(_WindowConsumedData(stream_id=stream_id, byte_count=byte_count)),
        )

    def _write_input(self, frame: FrameData) -> None:
        self._frame_sink.write(frame)

    def _new_stream(
        self,
        ctx: hsm.Context,
        stream_id: int,
        *,
        locally_initiated: bool,
        initial_send_window: int = INITIAL_STREAM_WINDOW,
        initial_receive_window: int | None = None,
    ) -> Stream:
        stream = Stream(
            stream_id=stream_id,
            initial_send_window=initial_send_window,
            initial_receive_window=self._initial_stream_window
            if initial_receive_window is None
            else initial_receive_window,
            locally_initiated=locally_initiated,
            on_window_consumed=self._dispatch_window_consumed,
        )
        self._start_stream(ctx, stream)
        self._streams[stream_id] = stream
        return stream

    def _start_stream(self, ctx: hsm.Context, stream: Stream) -> None:
        # Streams outlive the opening transition/activity; parent under session lifetime (HSM-CONTEXT-001).
        del ctx
        startup = typing.cast(
            collections.abc.Coroutine[typing.Any, typing.Any, Stream],
            hsm.started(self.context(), stream, stream.model),
        )
        task = asyncio.Task(startup, loop=asyncio.get_running_loop(), eager_start=True)
        if not task.done():
            _ = task.cancel()
            raise RuntimeError("Yamux stream HSM startup did not complete synchronously.")
        error = task.exception()
        if error is not None:
            raise error

    def _open_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, OpenStreamData)
        stream_id = self._next_stream_id
        self._next_stream_id += 2
        _ = self._new_stream(
            ctx,
            stream_id,
            locally_initiated=True,
            initial_receive_window=data.initial_window_size,
        )
        delta = max(0, data.initial_window_size - INITIAL_STREAM_WINDOW)
        self._write_input(FrameData.window_update(stream_id=stream_id, delta=delta, flags=Flag.SYN))

    def _fail_open_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        self._record_failure("open_stream", "session_draining")

    def _send_data(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, SendData)
        stream = self._streams[data.stream_id]
        flags = Flag.FIN if data.end_stream else Flag(0)
        stream.send(data.payload, end_stream=data.end_stream)
        self._write_input(FrameData.data(stream_id=data.stream_id, payload=data.payload, flags=flags))
        self._dispatch_drained_if_ready(ctx)

    def _fail_send_unknown_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, SendData)
        self._record_failure("send_data", "unknown_stream", stream_id=data.stream_id)

    def _fail_send_inactive_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, SendData)
        self._record_failure("send_data", "stream_reset", stream_id=data.stream_id)

    def _fail_send_window_exhausted(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, SendData)
        self._record_failure("send_data", "flow_control_exhausted", stream_id=data.stream_id)

    def _close_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, CloseStreamData)
        stream = self._streams[data.stream_id]
        stream.local_fin()
        self._write_input(FrameData.data(stream_id=data.stream_id, flags=Flag.FIN))
        self._dispatch_drained_if_ready(ctx)

    def _fail_close_unknown_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, CloseStreamData)
        self._record_failure("close_stream", "unknown_stream", stream_id=data.stream_id)

    def _fail_close_inactive_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, CloseStreamData)
        self._record_failure("close_stream", "stream_reset", stream_id=data.stream_id)

    def _reset_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ResetStreamData)
        stream = self._streams[data.stream_id]
        stream.reset()
        self._write_input(FrameData.data(stream_id=data.stream_id, flags=Flag.RST))
        self._dispatch_drained_if_ready(ctx)

    def _fail_reset_unknown_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ResetStreamData)
        self._record_failure("reset_stream", "unknown_stream", stream_id=data.stream_id)

    def _fail_reset_inactive_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ResetStreamData)
        self._record_failure("reset_stream", "stream_reset", stream_id=data.stream_id)

    def _receive_new_data(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        stream = self._new_stream(ctx, data.stream_id, locally_initiated=False)
        delta = max(0, self._initial_stream_window - INITIAL_STREAM_WINDOW)
        self._write_input(FrameData.window_update(stream_id=data.stream_id, delta=delta, flags=Flag.ACK))
        self._apply_data_frame(ctx, stream, data)

    def _receive_existing_data(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        stream = self._streams[data.stream_id]
        self._apply_data_frame(ctx, stream, data)

    def _apply_data_frame(self, ctx: hsm.Context, stream: Stream, data: ReceiveDataFrameData) -> None:
        flags = _flags(data.flags)
        if flags & Flag.RST:
            stream.reset()
            self._dispatch_drained_if_ready(ctx)
            return
        if flags & Flag.ACK:
            stream.acknowledge()
        if data.payload or bool(flags & Flag.FIN):
            stream.receive(data.payload, end_stream=bool(flags & Flag.FIN))
        self._dispatch_drained_if_ready(ctx)

    def _fail_receive_data_protocol(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        self._fail_protocol("receive_data", stream_id=data.stream_id)

    def _fail_receive_data_terminal_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        self._reset_after_stream_failure("receive_data", "stream_reset", data.stream_id)

    def _fail_receive_data_receive_window(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        self._reset_after_stream_failure("receive_data", "receive_window_exhausted", data.stream_id)

    def _fail_receive_data_draining(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        self._reset_after_stream_failure("receive_data", "session_draining", data.stream_id)

    def _receive_new_window(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        stream = self._new_stream(
            ctx,
            data.stream_id,
            locally_initiated=False,
            initial_send_window=INITIAL_STREAM_WINDOW + data.delta,
        )
        delta = max(0, self._initial_stream_window - INITIAL_STREAM_WINDOW)
        self._write_input(FrameData.window_update(stream_id=data.stream_id, delta=delta, flags=Flag.ACK))
        self._apply_window_frame(ctx, stream, data, already_applied_delta=True)

    def _receive_existing_window(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        stream = self._streams[data.stream_id]
        self._apply_window_frame(ctx, stream, data, already_applied_delta=False)

    def _apply_window_frame(
        self,
        ctx: hsm.Context,
        stream: Stream,
        data: ReceiveWindowUpdateFrameData,
        *,
        already_applied_delta: bool,
    ) -> None:
        flags = _flags(data.flags)
        if flags & Flag.RST:
            stream.reset()
            self._dispatch_drained_if_ready(ctx)
            return
        if flags & Flag.ACK:
            stream.acknowledge()
        if data.delta and not already_applied_delta:
            stream.update_send_window(data.delta)
        if flags & Flag.FIN:
            stream.remote_fin()
        self._dispatch_drained_if_ready(ctx)

    def _fail_receive_window_protocol(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        self._fail_protocol("receive_window_update", stream_id=data.stream_id)

    def _fail_receive_window_terminal_stream(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        self._reset_after_stream_failure("receive_window_update", "stream_reset", data.stream_id)

    def _fail_receive_window_overflow(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        self._fail_protocol("receive_window_update", stream_id=data.stream_id)

    def _fail_receive_window_draining(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        self._reset_after_stream_failure("receive_window_update", "session_draining", data.stream_id)

    def _send_ping(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, PingData)
        self._outstanding_ping = data.opaque
        self._write_input(FrameData.ping(opaque=data.opaque, ack=False))

    def _receive_ping(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceivePingFrameData)
        flags = Flag(data.flags)
        if flags is Flag.SYN:
            self._write_input(FrameData.ping(opaque=data.opaque, ack=True))
            return
        if self._outstanding_ping == data.opaque:
            self._outstanding_ping = None

    def _grant_receive_window(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, _WindowConsumedData)
        stream = self._streams[data.stream_id]
        stream.grant_receive_window(data.byte_count)
        self._write_input(FrameData.window_update(stream_id=data.stream_id, delta=data.byte_count))

    def _send_goaway(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, GoAwayData)
        code = GoAwayCode(data.code)
        self._accept_new_streams = False
        self._closed_goaway_code = code
        self._outstanding_ping = None
        self._write_input(FrameData.go_away(code=code))

    def _receive_goaway(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveGoAwayFrameData)
        self._accept_new_streams = False
        self._closed_goaway_code = GoAwayCode(data.code)
        self._outstanding_ping = None

    def _ping_timed_out(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        if self._outstanding_ping is not None:
            self._record_failure("ping", "ping_timeout")
            self._outstanding_ping = None

    def _reset_after_stream_failure(self, stage: SessionStage, failure_kind: FailureKind, stream_id: int) -> None:
        stream = self._streams.get(stream_id)
        if stream is not None and not _stream_is_terminal(stream):
            stream.reset()
        self._write_input(FrameData.data(stream_id=stream_id, flags=Flag.RST))
        self._record_failure(stage, failure_kind, stream_id=stream_id)

    def _fail_protocol(self, stage: SessionStage, *, stream_id: int | None = None) -> None:
        self._accept_new_streams = False
        self._closed_goaway_code = GoAwayCode.PROTOCOL_ERROR
        self._outstanding_ping = None
        if stream_id is not None:
            stream = self._streams.get(stream_id)
            if stream is not None and not _stream_is_terminal(stream):
                stream.reset()
            self._write_input(FrameData.data(stream_id=stream_id, flags=Flag.RST))
        self._record_failure(stage, "protocol_error", stream_id=stream_id)
        self._write_input(FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR))

    def _record_failure(
        self,
        stage: SessionStage,
        failure_kind: FailureKind,
        *,
        stream_id: int | None = None,
    ) -> None:
        self._failed_operations.append(SessionFailure(stage=stage, failure_kind=failure_kind, stream_id=stream_id))

    def _dispatch_drained_if_ready(self, ctx: hsm.Context) -> None:
        if self._accept_new_streams or self._active_streams_exist():
            return
        code = (
            None
            if self._closed_goaway_code is None
            else typing.cast(typing.Literal[0, 1, 2], int(self._closed_goaway_code))
        )
        _ = self.dispatch(ctx, SessionDrainedEvent.with_data(SessionDrainedData(code=code)))

    def _active_streams_exist(self) -> bool:
        return any(not _stream_is_terminal(stream) for stream in self._streams.values())

    model: typing.ClassVar[hsm.Model] = hsm.define(
        "Session",
        hsm.initial(hsm.target("/Session/connected/open")),
        hsm.state(
            "connected",
            hsm.initial(hsm.target("/Session/connected/open")),
            hsm.transition(
                hsm.on(OpenStreamEvent),
                hsm.guard(_has_stream_id_capacity),
                hsm.effect(_wrap_session_method(_open_stream)),
            ),
            hsm.transition(
                hsm.on(OpenStreamEvent),
                hsm.guard(_cannot_open_stream),
                hsm.effect(_wrap_session_method(_fail_open_stream)),
            ),
            hsm.transition(
                hsm.on(SendDataEvent), hsm.guard(_can_send_data), hsm.effect(_wrap_session_method(_send_data))
            ),
            hsm.transition(
                hsm.on(SendDataEvent),
                hsm.guard(_send_data_unknown_stream),
                hsm.effect(_wrap_session_method(_fail_send_unknown_stream)),
            ),
            hsm.transition(
                hsm.on(SendDataEvent),
                hsm.guard(_send_data_inactive_stream),
                hsm.effect(_wrap_session_method(_fail_send_inactive_stream)),
            ),
            hsm.transition(
                hsm.on(SendDataEvent),
                hsm.guard(_send_data_exhausts_window),
                hsm.effect(_wrap_session_method(_fail_send_window_exhausted)),
            ),
            hsm.transition(
                hsm.on(CloseStreamEvent),
                hsm.guard(_can_close_stream),
                hsm.effect(_wrap_session_method(_close_stream)),
            ),
            hsm.transition(
                hsm.on(CloseStreamEvent),
                hsm.guard(_stream_command_unknown),
                hsm.effect(_wrap_session_method(_fail_close_unknown_stream)),
            ),
            hsm.transition(
                hsm.on(CloseStreamEvent),
                hsm.guard(_stream_command_inactive),
                hsm.effect(_wrap_session_method(_fail_close_inactive_stream)),
            ),
            hsm.transition(
                hsm.on(ResetStreamEvent),
                hsm.guard(_can_reset_stream),
                hsm.effect(_wrap_session_method(_reset_stream)),
            ),
            hsm.transition(
                hsm.on(ResetStreamEvent),
                hsm.guard(_stream_command_unknown),
                hsm.effect(_wrap_session_method(_fail_reset_unknown_stream)),
            ),
            hsm.transition(
                hsm.on(ResetStreamEvent),
                hsm.guard(_stream_command_inactive),
                hsm.effect(_wrap_session_method(_fail_reset_inactive_stream)),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_has_duplicate_syn),
                hsm.effect(_wrap_session_method(_fail_receive_data_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_new_stream_while_draining),
                hsm.effect(_wrap_session_method(_fail_receive_data_draining)),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_has_bad_peer_stream_id),
                hsm.effect(_wrap_session_method(_fail_receive_data_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_has_bad_ack),
                hsm.effect(_wrap_session_method(_fail_receive_data_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_unknown_without_syn),
                hsm.effect(_wrap_session_method(_fail_receive_data_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_hits_terminal_stream),
                hsm.effect(_wrap_session_method(_fail_receive_data_terminal_stream)),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_hits_inactive_stream),
                hsm.effect(_wrap_session_method(_fail_receive_data_terminal_stream)),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_received_data_exhausts_window),
                hsm.effect(_wrap_session_method(_fail_receive_data_receive_window)),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_can_receive_existing_data),
                hsm.effect(_wrap_session_method(_receive_existing_data)),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_can_receive_new_data),
                hsm.effect(_wrap_session_method(_receive_new_data)),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_has_duplicate_syn),
                hsm.effect(_wrap_session_method(_fail_receive_window_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_new_stream_while_draining),
                hsm.effect(_wrap_session_method(_fail_receive_window_draining)),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_has_bad_peer_stream_id),
                hsm.effect(_wrap_session_method(_fail_receive_window_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_has_bad_ack),
                hsm.effect(_wrap_session_method(_fail_receive_window_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_unknown_without_syn),
                hsm.effect(_wrap_session_method(_fail_receive_window_protocol)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_hits_terminal_stream),
                hsm.effect(_wrap_session_method(_fail_receive_window_terminal_stream)),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_has_bad_fin),
                hsm.effect(_wrap_session_method(_fail_receive_window_terminal_stream)),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_received_window_overflows),
                hsm.effect(_wrap_session_method(_fail_receive_window_overflow)),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_can_receive_existing_window),
                hsm.effect(_wrap_session_method(_receive_existing_window)),
            ),
            hsm.transition(
                hsm.on(ReceiveWindowUpdateFrameEvent),
                hsm.guard(_can_receive_new_window),
                hsm.effect(_wrap_session_method(_receive_new_window)),
            ),
            hsm.transition(
                hsm.on(ReceivePingFrameEvent),
                hsm.guard(_is_inbound_ping),
                hsm.effect(_wrap_session_method(_receive_ping)),
            ),
            hsm.transition(
                hsm.on(_WindowConsumedEvent),
                hsm.guard(_can_grant_receive_window),
                hsm.effect(_wrap_session_method(_grant_receive_window)),
            ),
            hsm.transition(
                hsm.on(GoAwayEvent),
                hsm.effect(_wrap_session_method(_send_goaway)),
                hsm.target("/Session/connected/routing_disconnect"),
            ),
            hsm.transition(
                hsm.on(ReceiveGoAwayFrameEvent),
                hsm.effect(_wrap_session_method(_receive_goaway)),
                hsm.target("/Session/connected/routing_disconnect"),
            ),
            hsm.choice(
                "routing_disconnect",
                hsm.transition(hsm.guard(_has_active_streams), hsm.target("/Session/connected/draining")),
                hsm.transition(hsm.target("/Session/disconnected")),
            ),
            hsm.state(
                "open",
                hsm.transition(
                    hsm.on(PingEvent),
                    hsm.effect(_wrap_session_method(_send_ping)),
                    hsm.target("/Session/connected/pinging"),
                ),
            ),
            hsm.state(
                "pinging",
                hsm.transition(
                    hsm.on(ReceivePingFrameEvent),
                    hsm.guard(_is_outstanding_ping_ack),
                    hsm.effect(_wrap_session_method(_receive_ping)),
                    hsm.target("/Session/connected/open"),
                ),
                hsm.transition(
                    hsm.after(_session_ping_timeout),
                    hsm.effect(_wrap_session_method(_ping_timed_out)),
                    hsm.target("/Session/errored"),
                ),
            ),
            hsm.state(
                "draining",
                hsm.transition(hsm.on(SessionDrainedEvent), hsm.target("/Session/disconnected")),
            ),
        ),
        hsm.state("errored"),
        hsm.state("disconnected"),
    )


def _require_uint32(name: str, value: int, *, minimum: int = 0) -> None:
    if not minimum <= value <= _MAX_UINT32:
        raise ValueError(f"{name} must be between {minimum} and {_MAX_UINT32}.")
