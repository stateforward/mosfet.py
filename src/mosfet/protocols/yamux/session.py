import collections.abc
import asyncio
import dataclasses
import datetime
import types
import typing

import hsm
import mosfet
import pydantic

from mosfet.protocols.yamux.events import (
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
from mosfet.protocols.yamux.frame import (
    INITIAL_STREAM_WINDOW,
    Flag,
    FrameData,
    FrameType,
    GoAwayCode,
)
from mosfet.protocols.yamux.stream import Stream, StreamSnapshot, StreamState
from mosfet.telemetry.hsm import Traced

Role = typing.Literal["client", "server"]
DEFAULT_PING_TIMEOUT = datetime.timedelta(seconds=30)
_MAX_UINT32 = (1 << 32) - 1
TSessionReturn = typing.TypeVar("TSessionReturn")
_RETAINED_TERMINAL_STREAMS = 64


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


@dataclasses.dataclass(slots=True)
class _StreamAdmission:
    """Session-owned admission view for one stream (HSM-OWNERSHIP-001).

    Stream machines own their private counters and lifecycle for Stream HSM guards.
    Session never peeks those fields; effects update this ledger when they drive Stream APIs.
    """

    lifecycle: StreamState
    awaiting_ack: bool
    send_window: int
    receive_window: int


def _admission_is_writable(admission: _StreamAdmission) -> bool:
    return admission.lifecycle in {
        StreamState.LOCAL_OPENING,
        StreamState.OPEN,
        StreamState.REMOTE_CLOSED,
    }


def _admission_is_readable(admission: _StreamAdmission) -> bool:
    return admission.lifecycle in {
        StreamState.LOCAL_OPENING,
        StreamState.OPEN,
        StreamState.LOCAL_CLOSED,
    }


def _admission_is_terminal(admission: _StreamAdmission) -> bool:
    return admission.lifecycle in {StreamState.CLOSED, StreamState.RESET}


def _admission_accepts_ack(admission: _StreamAdmission) -> bool:
    return admission.awaiting_ack


def _data_frame_is_ack_only_for_admission(admission: _StreamAdmission, data: ReceiveDataFrameData) -> bool:
    flags = _flags(data.flags)
    return (
        admission.awaiting_ack
        and bool(flags & Flag.ACK)
        and not bool(flags & (Flag.SYN | Flag.FIN | Flag.RST))
        and len(data.payload) == 0
    )


def _window_frame_is_ack_only_for_admission(admission: _StreamAdmission, data: ReceiveWindowUpdateFrameData) -> bool:
    flags = _flags(data.flags)
    return (
        admission.awaiting_ack
        and bool(flags & Flag.ACK)
        and not bool(flags & (Flag.SYN | Flag.FIN | Flag.RST))
        and data.delta == 0
    )


def _peer_stream_id_is_valid(role: Role, stream_id: int) -> bool:
    if role == "client":
        return stream_id % 2 == 0
    return stream_id % 2 == 1


def _session_role(instance: "Session") -> Role:
    return Session.role_of(instance)


def _session_initial_stream_window(instance: "Session") -> int:
    return Session.initial_window_of(instance)


def _session_next_stream_id(instance: "Session") -> int:
    return Session.next_stream_id_of(instance)


def _session_accepts_new_streams(instance: "Session") -> bool:
    return Session.accepts_new_streams_of(instance)


def _session_outstanding_ping(instance: "Session") -> int | None:
    return Session.outstanding_ping_of(instance)


def _session_ping_timeout_value(instance: "Session") -> datetime.timedelta:
    return Session.ping_timeout_of(instance)


def _session_admissions(instance: "Session") -> dict[int, _StreamAdmission]:
    return Session.admissions_of(instance)


def _session_admission(instance: "Session", stream_id: int) -> _StreamAdmission | None:
    return Session.admission_of(instance, stream_id)


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
    return any(not _admission_is_terminal(admission) for admission in _session_admissions(instance).values())


def _can_send_data(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is not None and _admission_is_writable(admission) and len(data.payload) <= admission.send_window


def _send_data_unknown_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    return data is not None and _session_admission(instance, data.stream_id) is None


def _send_data_inactive_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is not None and not _admission_is_writable(admission)


def _send_data_exhausts_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _send_data_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is not None and _admission_is_writable(admission) and len(data.payload) > admission.send_window


def _can_close_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is not None and _admission_is_writable(admission)


def _stream_command_unknown(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    return data is not None and _session_admission(instance, data.stream_id) is None


def _stream_command_inactive(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is not None and not _admission_is_writable(admission)


def _can_reset_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _stream_id_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is not None and not _admission_is_terminal(admission)


def _received_data_has_duplicate_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and bool(_flags(data.flags) & Flag.SYN)
        and _session_admission(instance, data.stream_id) is not None
    )


def _received_data_has_bad_peer_stream_id(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and _session_admission(instance, data.stream_id) is None
        and bool(_flags(data.flags) & Flag.SYN)
        and not _peer_stream_id_is_valid(_session_role(instance), data.stream_id)
    )


def _received_data_has_bad_ack(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None or not bool(_flags(data.flags) & Flag.ACK):
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is None or not _admission_accepts_ack(admission)


def _received_data_unknown_without_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and _session_admission(instance, data.stream_id) is None
        and not bool(_flags(data.flags) & Flag.SYN)
    )


def _received_data_hits_terminal_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return (
        admission is not None
        and _admission_is_terminal(admission)
        and not _data_frame_is_ack_only_for_admission(admission, data)
    )


def _received_data_hits_inactive_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None or bool(_flags(data.flags) & Flag.RST):
        return False
    admission = _session_admission(instance, data.stream_id)
    return (
        admission is not None
        and not _admission_is_readable(admission)
        and not _data_frame_is_ack_only_for_admission(admission, data)
    )


def _received_data_exhausts_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None or bool(_flags(data.flags) & Flag.RST):
        return False
    admission = _session_admission(instance, data.stream_id)
    if admission is None:
        return bool(_flags(data.flags) & Flag.SYN) and len(data.payload) > _session_initial_stream_window(instance)
    return _admission_is_readable(admission) and len(data.payload) > admission.receive_window


def _can_receive_existing_data(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    if admission is None:
        return False
    if bool(_flags(data.flags) & Flag.RST):
        return not _admission_is_terminal(admission)
    if _data_frame_is_ack_only_for_admission(admission, data):
        return True
    if _admission_is_terminal(admission):
        return False
    return _admission_is_readable(admission) and len(data.payload) <= admission.receive_window


def _can_receive_new_data(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_data_event_data(event)
    return (
        data is not None
        and _session_accepts_new_streams(instance)
        and _session_admission(instance, data.stream_id) is None
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
        and _session_admission(instance, data.stream_id) is None
    )


def _received_window_has_duplicate_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and bool(_flags(data.flags) & Flag.SYN)
        and _session_admission(instance, data.stream_id) is not None
    )


def _received_window_has_bad_peer_stream_id(
    ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]
) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and _session_admission(instance, data.stream_id) is None
        and bool(_flags(data.flags) & Flag.SYN)
        and not _peer_stream_id_is_valid(_session_role(instance), data.stream_id)
    )


def _received_window_has_bad_ack(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None or not bool(_flags(data.flags) & Flag.ACK):
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is None or not _admission_accepts_ack(admission)


def _received_window_unknown_without_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and _session_admission(instance, data.stream_id) is None
        and not bool(_flags(data.flags) & Flag.SYN)
    )


def _received_window_hits_terminal_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return (
        admission is not None
        and _admission_is_terminal(admission)
        and not _window_frame_is_ack_only_for_admission(admission, data)
    )


def _received_window_has_bad_fin(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None or not bool(_flags(data.flags) & Flag.FIN) or bool(_flags(data.flags) & Flag.RST):
        return False
    admission = _session_admission(instance, data.stream_id)
    return (
        admission is not None
        and not _admission_is_readable(admission)
        and not _window_frame_is_ack_only_for_admission(admission, data)
    )


def _received_window_overflows(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    if data is None or bool(_flags(data.flags) & Flag.RST):
        return False
    admission = _session_admission(instance, data.stream_id)
    if admission is None:
        return bool(_flags(data.flags) & Flag.SYN) and INITIAL_STREAM_WINDOW + data.delta > _MAX_UINT32
    return not _admission_is_terminal(admission) and admission.send_window + data.delta > _MAX_UINT32


def _can_receive_existing_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    data = _received_window_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return (
        admission is not None
        and (not _admission_is_terminal(admission) or _window_frame_is_ack_only_for_admission(admission, data))
        and (bool(_flags(data.flags) & Flag.RST) or not _received_window_overflows(ctx, instance, event))
    )


def _can_receive_new_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _received_window_event_data(event)
    return (
        data is not None
        and _session_accepts_new_streams(instance)
        and _session_admission(instance, data.stream_id) is None
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
        and _session_admission(instance, data.stream_id) is None
    )


def _can_grant_receive_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
    del ctx
    data = _window_consumed_event_data(event)
    if data is None:
        return False
    admission = _session_admission(instance, data.stream_id)
    return admission is not None and _admission_is_readable(admission)


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


class Session(Traced):
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
    _stream_admission: dict[int, _StreamAdmission]

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
        self._stream_admission = {}
        outbound_frames: asyncio.Queue[FrameData] = asyncio.Queue()
        self._frame_sink = _FrameSink(outbound_frames)
        self._outbound = FrameStream(outbound_frames)
        self._failed_operations = []
        self._accept_new_streams = True
        self._closed_goaway_code = None
        self._outstanding_ping = None
        self._ping_timeout = ping_timeout

    @staticmethod
    def role_of(instance: "Session") -> Role:
        """Owning-class read of the session role."""

        role = instance._role
        assert role in ("client", "server")
        return role

    @staticmethod
    def initial_window_of(instance: "Session") -> int:
        return instance._initial_stream_window

    @staticmethod
    def next_stream_id_of(instance: "Session") -> int:
        return instance._next_stream_id

    @staticmethod
    def accepts_new_streams_of(instance: "Session") -> bool:
        return instance._accept_new_streams

    @staticmethod
    def outstanding_ping_of(instance: "Session") -> int | None:
        return instance._outstanding_ping

    @staticmethod
    def ping_timeout_of(instance: "Session") -> datetime.timedelta:
        return instance._ping_timeout

    @staticmethod
    def admissions_of(instance: "Session") -> dict[int, _StreamAdmission]:
        return instance._stream_admission

    @staticmethod
    def admission_of(instance: "Session", stream_id: int) -> _StreamAdmission | None:
        return instance._stream_admission.get(stream_id)

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
        receive_window = self._initial_stream_window if initial_receive_window is None else initial_receive_window
        stream = Stream(
            stream_id=stream_id,
            initial_send_window=initial_send_window,
            initial_receive_window=receive_window,
            locally_initiated=locally_initiated,
            on_window_consumed=self._dispatch_window_consumed,
        )
        self._start_stream(ctx, stream)
        self._streams[stream_id] = stream
        self._stream_admission[stream_id] = _StreamAdmission(
            lifecycle=StreamState.LOCAL_OPENING if locally_initiated else StreamState.OPEN,
            awaiting_ack=locally_initiated,
            send_window=initial_send_window,
            receive_window=receive_window,
        )
        return stream

    def _mark_admission_ack(self, stream_id: int) -> None:
        admission = self._stream_admission[stream_id]
        admission.awaiting_ack = False
        if admission.lifecycle is StreamState.LOCAL_OPENING:
            admission.lifecycle = StreamState.OPEN

    def _mark_admission_local_fin(self, stream_id: int) -> None:
        admission = self._stream_admission[stream_id]
        if admission.lifecycle is StreamState.REMOTE_CLOSED:
            admission.lifecycle = StreamState.CLOSED
        elif admission.lifecycle in {StreamState.LOCAL_OPENING, StreamState.OPEN}:
            admission.lifecycle = StreamState.LOCAL_CLOSED

    def _mark_admission_remote_fin(self, stream_id: int) -> None:
        admission = self._stream_admission[stream_id]
        if admission.lifecycle is StreamState.LOCAL_CLOSED:
            admission.lifecycle = StreamState.CLOSED
        elif admission.lifecycle in {StreamState.LOCAL_OPENING, StreamState.OPEN}:
            admission.lifecycle = StreamState.REMOTE_CLOSED

    def _mark_admission_reset(self, stream_id: int) -> None:
        admission = self._stream_admission[stream_id]
        admission.lifecycle = StreamState.RESET
        admission.awaiting_ack = False

    def _record_admission_sent(self, stream_id: int, payload: bytes, *, end_stream: bool) -> None:
        admission = self._stream_admission[stream_id]
        admission.send_window -= len(payload)
        if end_stream:
            self._mark_admission_local_fin(stream_id)

    def _record_admission_received(self, stream_id: int, payload: bytes, *, end_stream: bool) -> None:
        admission = self._stream_admission[stream_id]
        admission.receive_window -= len(payload)
        if end_stream:
            self._mark_admission_remote_fin(stream_id)

    def _record_admission_send_window(self, stream_id: int, delta: int) -> None:
        self._stream_admission[stream_id].send_window += delta

    def _record_admission_receive_window(self, stream_id: int, delta: int) -> None:
        self._stream_admission[stream_id].receive_window += delta

    def _drive_stream_send(self, stream_id: int, payload: bytes, *, end_stream: bool) -> None:
        """Drive Stream.send and update Session admission in one path (no dual-write drift)."""

        self._streams[stream_id].send(payload, end_stream=end_stream)
        self._record_admission_sent(stream_id, payload, end_stream=end_stream)

    def _drive_stream_receive(self, stream_id: int, payload: bytes, *, end_stream: bool) -> None:
        self._streams[stream_id].receive(payload, end_stream=end_stream)
        self._record_admission_received(stream_id, payload, end_stream=end_stream)

    def _drive_stream_acknowledge(self, stream_id: int) -> None:
        self._streams[stream_id].acknowledge()
        self._mark_admission_ack(stream_id)

    def _drive_stream_local_fin(self, stream_id: int) -> None:
        self._streams[stream_id].local_fin()
        self._mark_admission_local_fin(stream_id)

    def _drive_stream_remote_fin(self, stream_id: int) -> None:
        self._streams[stream_id].remote_fin()
        self._mark_admission_remote_fin(stream_id)

    def _drive_stream_reset(self, stream_id: int) -> None:
        self._streams[stream_id].reset()
        self._mark_admission_reset(stream_id)

    def _drive_stream_update_send_window(self, stream_id: int, delta: int) -> None:
        self._streams[stream_id].update_send_window(delta)
        self._record_admission_send_window(stream_id, delta)

    def _drive_stream_grant_receive_window(self, stream_id: int, delta: int) -> None:
        self._streams[stream_id].grant_receive_window(delta)
        self._record_admission_receive_window(stream_id, delta)

    def _start_stream(self, ctx: hsm.Context, stream: Stream) -> None:
        # Streams outlive the opening transition/activity; parent under session lifetime (HSM-CONTEXT-001).
        del ctx
        startup = typing.cast(
            collections.abc.Coroutine[typing.Any, typing.Any, Stream],
            mosfet.started(self.context(), stream, stream.model),
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
        flags = Flag.FIN if data.end_stream else Flag(0)
        self._drive_stream_send(data.stream_id, data.payload, end_stream=data.end_stream)
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
        self._drive_stream_local_fin(data.stream_id)
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
        self._drive_stream_reset(data.stream_id)
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
        _ = self._new_stream(ctx, data.stream_id, locally_initiated=False)
        delta = max(0, self._initial_stream_window - INITIAL_STREAM_WINDOW)
        self._write_input(FrameData.window_update(stream_id=data.stream_id, delta=delta, flags=Flag.ACK))
        self._apply_data_frame(ctx, data)

    def _receive_existing_data(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        self._apply_data_frame(ctx, data)

    def _apply_data_frame(self, ctx: hsm.Context, data: ReceiveDataFrameData) -> None:
        flags = _flags(data.flags)
        stream_id = data.stream_id
        if flags & Flag.RST:
            self._drive_stream_reset(stream_id)
            self._dispatch_drained_if_ready(ctx)
            return
        if flags & Flag.ACK:
            self._drive_stream_acknowledge(stream_id)
        if data.payload or bool(flags & Flag.FIN):
            self._drive_stream_receive(stream_id, data.payload, end_stream=bool(flags & Flag.FIN))
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
        _ = self._new_stream(
            ctx,
            data.stream_id,
            locally_initiated=False,
            initial_send_window=INITIAL_STREAM_WINDOW + data.delta,
        )
        delta = max(0, self._initial_stream_window - INITIAL_STREAM_WINDOW)
        self._write_input(FrameData.window_update(stream_id=data.stream_id, delta=delta, flags=Flag.ACK))
        self._apply_window_frame(ctx, data, already_applied_delta=True)

    def _receive_existing_window(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        self._apply_window_frame(ctx, data, already_applied_delta=False)

    def _apply_window_frame(
        self,
        ctx: hsm.Context,
        data: ReceiveWindowUpdateFrameData,
        *,
        already_applied_delta: bool,
    ) -> None:
        flags = _flags(data.flags)
        stream_id = data.stream_id
        if flags & Flag.RST:
            self._drive_stream_reset(stream_id)
            self._dispatch_drained_if_ready(ctx)
            return
        if flags & Flag.ACK:
            self._drive_stream_acknowledge(stream_id)
        if data.delta and not already_applied_delta:
            self._drive_stream_update_send_window(stream_id, data.delta)
        elif data.delta and already_applied_delta:
            # New-stream path baked the delta into initial send window construction.
            pass
        if flags & Flag.FIN:
            self._drive_stream_remote_fin(stream_id)
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
        self._drive_stream_grant_receive_window(data.stream_id, data.byte_count)
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
        admission = self._stream_admission.get(stream_id)
        if admission is not None and not _admission_is_terminal(admission):
            self._drive_stream_reset(stream_id)
        self._write_input(FrameData.data(stream_id=stream_id, flags=Flag.RST))
        self._record_failure(stage, failure_kind, stream_id=stream_id)
        self._retire_terminal_streams()

    def _fail_protocol(self, stage: SessionStage, *, stream_id: int | None = None) -> None:
        self._accept_new_streams = False
        self._closed_goaway_code = GoAwayCode.PROTOCOL_ERROR
        self._outstanding_ping = None
        if stream_id is not None:
            admission = self._stream_admission.get(stream_id)
            if admission is not None and not _admission_is_terminal(admission):
                self._drive_stream_reset(stream_id)
            self._write_input(FrameData.data(stream_id=stream_id, flags=Flag.RST))
            self._retire_terminal_streams()
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
        self._retire_terminal_streams()
        if self._accept_new_streams or self._active_streams_exist():
            return
        code = (
            None
            if self._closed_goaway_code is None
            else typing.cast(typing.Literal[0, 1, 2], int(self._closed_goaway_code))
        )
        _ = self.dispatch(ctx, SessionDrainedEvent.with_data(SessionDrainedData(code=code)))

    def _active_streams_exist(self) -> bool:
        return any(not _admission_is_terminal(admission) for admission in self._stream_admission.values())

    def _retire_terminal_streams(self) -> None:
        """Reclaim finished streams instead of retaining every terminal id forever.

        A closed or reset stream stays meaningful until its final ACK lands and the
        application has drained the inbound buffer. We keep a bounded window of the most
        recent finished streams and release only older entries that no longer owe an ACK
        and whose buffer is already at EOF. This bounds memory for long-lived Yamux
        sessions while preserving late-ACK handling and the terminal lifecycle visible in
        snapshots. Late frames for a retired id fall through to the unknown/inactive
        handlers as a bounded protocol failure.
        """

        safe = [
            stream_id
            for stream_id, admission in self._stream_admission.items()
            if _admission_is_terminal(admission) and not admission.awaiting_ack and self._streams[stream_id].at_eof()
        ]
        overflow = len(safe) - _RETAINED_TERMINAL_STREAMS
        if overflow <= 0:
            return
        for stream_id in safe[:overflow]:
            _ = self._streams.pop(stream_id, None)
            _ = self._stream_admission.pop(stream_id, None)

    model: typing.ClassVar[hsm.Model] = mosfet.define(
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
