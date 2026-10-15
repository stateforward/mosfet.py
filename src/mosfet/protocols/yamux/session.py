"""A yamux session: stream ids, demultiplexing, ping, GoAway and the drain, reached only by message.

The session's ``connection`` owns the socket. It dispatches each decoded frame to the session
(``event_from_frame``) and takes from it ``TransmitEvent`` (frames to send, in order),
``StreamAcceptedEvent`` (a stream the peer opened, owned by the connection until handed over) and
``EndedEvent`` (close the socket). Per-stream state lives wholly in each :class:`Stream`; the session checks
only what is session-wide (id parity, duplicate SYN, unknown streams, draining) and re-addresses each
stream's frames to it. Yamux carries bytes: the session takes no codec.
"""

import asyncio
import collections
import collections.abc
import dataclasses
import datetime
import types
import typing
import uuid

import hsm
import mosfet

from mosfet.protocols.yamux.events import (
    EndedData,
    EndedEvent,
    FailureKind,
    FrameOutData,
    FrameOutEvent,
    GoAwayData,
    GoAwayEvent,
    OpenStreamData,
    OpenStreamEvent,
    OpenStreamFailedData,
    OpenStreamFailedEvent,
    PingData,
    PingEvent,
    ReceiveDataFrameData,
    ReceiveDataFrameEvent,
    ReceiveGoAwayFrameData,
    ReceiveGoAwayFrameEvent,
    ReceivePingFrameData,
    ReceivePingFrameEvent,
    ReceiveWindowUpdateFrameData,
    ReceiveWindowUpdateFrameEvent,
    SessionDrainedData,
    SessionDrainedEvent,
    SessionStage,
    StreamAcceptedData,
    StreamAcceptedEvent,
    StreamEndedEvent,
    StreamIdData,
    TransmitData,
    TransmitEvent,
    ViolationData,
    ViolationEvent,
)
from mosfet.protocols.yamux.frame import INITIAL_STREAM_WINDOW, Flag, FrameData, FrameType, GoAwayCode
from mosfet.protocols.yamux.stream import Stream, StreamSnapshot
from mosfet.telemetry.hsm import Traced

Role = typing.Literal["client", "server"]
DEFAULT_PING_TIMEOUT = datetime.timedelta(seconds=30)
_MAX_UINT32 = (1 << 32) - 1
# How many retired stream ids the session remembers, so a late frame for one is ignored rather than refused.
_RECENTLY_RETIRED = 64


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


def _tell(
    ctx: hsm.Context, recipient: hsm.Instance, event: hsm.Event[typing.Any], trigger: hsm.Event[typing.Any]
) -> None:
    """Dispatch ``event`` to ``recipient`` with a fresh id, stamped from this session, carrying the trigger's telemetry."""

    _ = hsm.dispatch(
        ctx,
        recipient,
        dataclasses.replace(event, id=uuid.uuid4().hex, source="", target="", metadata=dict(trigger.metadata)),
    )


def _frame_data(event: hsm.Event[typing.Any]) -> ReceiveDataFrameData | ReceiveWindowUpdateFrameData | None:
    data = event.data
    return data if isinstance(data, ReceiveDataFrameData | ReceiveWindowUpdateFrameData) else None


def _syn(data: ReceiveDataFrameData | ReceiveWindowUpdateFrameData) -> bool:
    return bool(Flag(data.flags) & Flag.SYN)


def _stage(data: ReceiveDataFrameData | ReceiveWindowUpdateFrameData) -> SessionStage:
    return "receive_window_update" if isinstance(data, ReceiveWindowUpdateFrameData) else "receive_data"


class Session(Traced):
    """HSM-owned Yamux session; every stream is a :class:`Stream` it starts and reaches by message.

    ``connection`` is the actor that owns the socket: an address, dispatched to and never read.
    """

    _accept_new_streams: bool
    _closed_goaway_code: GoAwayCode | None
    _connection: hsm.Instance
    _failed_operations: list[SessionFailure]
    _initial_stream_window: int
    _next_stream_id: int
    _outstanding_ping: int | None
    _ping_timeout: datetime.timedelta
    _retired: collections.deque[int]
    _role: Role
    _streams: dict[int, Stream]

    def __init__(
        self,
        *,
        role: str,
        connection: hsm.Instance,
        initial_stream_window: int = INITIAL_STREAM_WINDOW,
        ping_timeout: datetime.timedelta = DEFAULT_PING_TIMEOUT,
    ) -> None:
        super().__init__()
        if role not in ("client", "server"):
            raise ValueError("role must be 'client' or 'server'.")
        _require_uint32("initial_stream_window", initial_stream_window, minimum=INITIAL_STREAM_WINDOW)
        if ping_timeout <= datetime.timedelta():
            raise ValueError("ping_timeout must be positive.")
        self._role = "client" if role == "client" else "server"
        self._connection = connection
        self._initial_stream_window = initial_stream_window
        self._next_stream_id = 1 if role == "client" else 2
        self._streams = {}
        self._retired = collections.deque(maxlen=_RECENTLY_RETIRED)
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
            has_active_streams=bool(self._streams),
        )

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop every live stream (each tells its owner the stream closed), then the session."""

        for stream in tuple(self._streams.values()):
            await hsm.stop(stream)
        await hsm.Instance.stop(self, ctx)

    # Guards.

    @staticmethod
    def _can_open_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._accept_new_streams and instance._next_stream_id <= _MAX_UINT32

    @staticmethod
    def _has_active_streams(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return bool(instance._streams)

    @staticmethod
    def _has_duplicate_syn(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _frame_data(event)
        return data is not None and _syn(data) and data.stream_id in instance._streams

    @staticmethod
    def _targets_live_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _frame_data(event)
        return data is not None and data.stream_id in instance._streams

    @staticmethod
    def _targets_retired_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _frame_data(event)
        return data is not None and data.stream_id in instance._retired

    @staticmethod
    def _opens_while_draining(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        """A frame for an unknown stream while the session no longer opens any. Guarded after the live and
        retired checks."""

        del ctx
        return _frame_data(event) is not None and not instance._accept_new_streams

    @staticmethod
    def _breaks_session_rules(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        """An unknown stream's frame without SYN, with ACK, with the peer's wrong parity, or a SYN window that
        would overflow. Guarded after the live and retired checks."""

        del ctx
        data = _frame_data(event)
        if data is None:
            return False
        flags = Flag(data.flags)
        peer_parity = 0 if instance._role == "client" else 1
        return (
            not flags & Flag.SYN
            or bool(flags & Flag.ACK)
            or data.stream_id % 2 != peer_parity
            or (isinstance(data, ReceiveWindowUpdateFrameData) and INITIAL_STREAM_WINDOW + data.delta > _MAX_UINT32)
        )

    @staticmethod
    def _opens_past_initial_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, ReceiveDataFrameData) and len(data.payload) > instance._initial_stream_window

    @staticmethod
    def _is_protocol_violation(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, ViolationData) and data.kind == "protocol_error"

    @staticmethod
    def _is_inbound_ping(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        data = event.data
        return isinstance(data, ReceivePingFrameData) and data.flags == int(Flag.SYN)

    @staticmethod
    def _is_outstanding_ping_ack(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, ReceivePingFrameData)
            and data.flags == int(Flag.ACK)
            and instance._outstanding_ping == data.opaque
        )

    @staticmethod
    def _ping_timeout_of(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> datetime.timedelta:
        del ctx, event
        return instance._ping_timeout

    # Effects.

    @staticmethod
    def _open_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        """Write SYN, then start the stream, which tells its owner it owns it."""

        data = event.data
        assert isinstance(data, OpenStreamData)
        stream_id = instance._next_stream_id
        instance._next_stream_id += 2
        delta = data.initial_window_size - INITIAL_STREAM_WINDOW
        Session._transmit(
            ctx, instance, FrameData.window_update(stream_id=stream_id, delta=delta, flags=Flag.SYN), event
        )
        Session._start_stream(
            instance,
            stream_id,
            Stream(
                stream_id=stream_id,
                session=instance,
                owner=data.owner,
                codec=data.codec,
                channel=data.channel,
                locally_initiated=True,
                initial_receive_window=data.initial_window_size,
            ),
        )

    @staticmethod
    def _refuse_open(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        """Answer the owner that asked, keeping its request id."""

        data = event.data
        assert isinstance(data, OpenStreamData)
        kind: typing.Literal["session_draining", "stream_rejected"] = (
            "session_draining" if not instance._accept_new_streams else "stream_rejected"
        )
        Session._record_failure(instance, "open_stream", kind)
        _ = hsm.dispatch(
            ctx,
            data.owner,
            dataclasses.replace(
                OpenStreamFailedEvent.with_data(OpenStreamFailedData(kind=kind)),
                id=event.id,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    def _accept_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        """The peer opened a stream: ACK it, give it to the connection without a codec, then hand it the frame."""

        data = _frame_data(event)
        assert data is not None
        stream = Stream(
            stream_id=data.stream_id,
            session=instance,
            owner=instance._connection,
            codec=None,
            channel=None,
            locally_initiated=False,
            initial_receive_window=instance._initial_stream_window,
        )
        delta = instance._initial_stream_window - INITIAL_STREAM_WINDOW
        Session._transmit(
            ctx, instance, FrameData.window_update(stream_id=data.stream_id, delta=delta, flags=Flag.ACK), event
        )
        Session._start_stream(instance, data.stream_id, stream)
        _tell(
            ctx,
            instance._connection,
            StreamAcceptedEvent.with_data(StreamAcceptedData(stream_id=data.stream_id, stream=stream)),
            event,
        )
        Session._forward(ctx, instance, event)

    @staticmethod
    def _forward_to_stream(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        Session._forward(ctx, instance, event)

    @staticmethod
    def _ignore_retired(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = _frame_data(event)
        assert data is not None
        Session._record_failure(instance, _stage(data), "unknown_stream", stream_id=data.stream_id)

    @staticmethod
    def _refuse_while_draining(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        Session._refuse_stream(ctx, instance, event, "session_draining")

    @staticmethod
    def _refuse_past_initial_window(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        Session._refuse_stream(ctx, instance, event, "receive_window_exhausted")

    @staticmethod
    def _fail_frame_protocol(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        """RST the offending stream, then GoAway(protocol error), then tell the connection the session ended."""

        data = _frame_data(event)
        assert data is not None
        Session._transmit(ctx, instance, FrameData.data(stream_id=data.stream_id, flags=Flag.RST), event)
        Session._fail_protocol(ctx, instance, event, _stage(data), data.stream_id)

    @staticmethod
    def _fail_stream_protocol(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        """The stream already wrote its RST; GoAway(protocol error) follows it."""

        data = event.data
        assert isinstance(data, ViolationData)
        Session._fail_protocol(ctx, instance, event, data.stage, data.stream_id)

    @staticmethod
    def _record_violation(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ViolationData)
        Session._record_failure(instance, data.stage, data.kind, stream_id=data.stream_id)

    @staticmethod
    def _transmit_frame_out(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, FrameOutData)
        _tell(ctx, instance._connection, TransmitEvent.with_data(TransmitData(frame=data.frame)), event)

    @staticmethod
    def _retire(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        """Forget a finished stream, remembering its id for late frames, and finish a drain it was holding."""

        data = event.data
        assert isinstance(data, StreamIdData)
        if instance._streams.pop(data.stream_id, None) is None:
            return
        instance._retired.append(data.stream_id)
        if instance._accept_new_streams or instance._streams:
            return
        code = (
            None
            if instance._closed_goaway_code is None
            else typing.cast(typing.Literal[0, 1, 2], int(instance._closed_goaway_code))
        )
        _ = hsm.dispatch(ctx, instance, SessionDrainedEvent.with_data(SessionDrainedData(code=code)))

    @staticmethod
    def _send_ping(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, PingData)
        instance._outstanding_ping = data.opaque
        Session._transmit(ctx, instance, FrameData.ping(opaque=data.opaque, ack=False), event)

    @staticmethod
    def _receive_ping(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceivePingFrameData)
        if Flag(data.flags) is Flag.SYN:
            Session._transmit(ctx, instance, FrameData.ping(opaque=data.opaque, ack=True), event)
            return
        if instance._outstanding_ping == data.opaque:
            instance._outstanding_ping = None

    @staticmethod
    def _send_goaway(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, GoAwayData)
        code = GoAwayCode(data.code)
        instance._accept_new_streams = False
        instance._closed_goaway_code = code
        instance._outstanding_ping = None
        Session._transmit(ctx, instance, FrameData.go_away(code=code), event)

    @staticmethod
    def _receive_goaway(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, ReceiveGoAwayFrameData)
        instance._accept_new_streams = False
        instance._closed_goaway_code = GoAwayCode(data.code)
        instance._outstanding_ping = None

    @staticmethod
    def _ping_timed_out(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        if instance._outstanding_ping is not None:
            Session._record_failure(instance, "ping", "ping_timeout")
            instance._outstanding_ping = None
        instance._accept_new_streams = False
        _tell(ctx, instance._connection, EndedEvent.with_data(EndedData(reason="ping_timeout")), event)

    @staticmethod
    def _end_drained(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        _tell(ctx, instance._connection, EndedEvent.with_data(EndedData(reason="drained")), event)

    # Helpers called from the effects above.

    @staticmethod
    def _transmit(ctx: hsm.Context, instance: "Session", frame: FrameData, trigger: hsm.Event[typing.Any]) -> None:
        _tell(ctx, instance._connection, TransmitEvent.with_data(TransmitData(frame=frame.to_bytes())), trigger)

    @staticmethod
    def _forward(ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any]) -> None:
        """Re-address a stream's frame, from this session, to the stream; its id and telemetry are kept."""

        data = _frame_data(event)
        assert data is not None
        _ = hsm.dispatch(ctx, instance._streams[data.stream_id], dataclasses.replace(event, source="", target=""))

    @staticmethod
    def _refuse_stream(
        ctx: hsm.Context, instance: "Session", event: hsm.Event[typing.Any], failure_kind: FailureKind
    ) -> None:
        data = _frame_data(event)
        assert data is not None
        Session._transmit(ctx, instance, FrameData.data(stream_id=data.stream_id, flags=Flag.RST), event)
        Session._record_failure(instance, _stage(data), failure_kind, stream_id=data.stream_id)

    @staticmethod
    def _fail_protocol(
        ctx: hsm.Context,
        instance: "Session",
        event: hsm.Event[typing.Any],
        stage: SessionStage,
        stream_id: int | None,
    ) -> None:
        instance._accept_new_streams = False
        instance._closed_goaway_code = GoAwayCode.PROTOCOL_ERROR
        instance._outstanding_ping = None
        Session._record_failure(instance, stage, "protocol_error", stream_id=stream_id)
        Session._transmit(ctx, instance, FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR), event)
        _tell(ctx, instance._connection, EndedEvent.with_data(EndedData(reason="protocol_error")), event)

    @staticmethod
    def _record_failure(
        instance: "Session", stage: SessionStage, failure_kind: FailureKind, *, stream_id: int | None = None
    ) -> None:
        instance._failed_operations.append(SessionFailure(stage=stage, failure_kind=failure_kind, stream_id=stream_id))

    @staticmethod
    def _start_stream(instance: "Session", stream_id: int, stream: Stream) -> None:
        """Start ``stream`` now, under the session's lifetime (HSM-CONTEXT-001), and keep it as live."""

        startup = typing.cast(
            collections.abc.Coroutine[typing.Any, typing.Any, Stream],
            mosfet.started(instance.context(), stream, stream.model),
        )
        task = asyncio.Task(startup, loop=asyncio.get_running_loop(), eager_start=True)
        if not task.done():
            _ = task.cancel()
            raise RuntimeError("Yamux stream HSM startup did not complete synchronously.")
        error = task.exception()
        if error is not None:
            raise error
        instance._streams[stream_id] = stream

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Session",
        hsm.initial(hsm.target("/Session/connected/open")),
        hsm.transition(hsm.on(OpenStreamEvent), hsm.effect(_refuse_open)),
        hsm.transition(hsm.on(StreamEndedEvent), hsm.effect(_retire)),
        hsm.transition(hsm.on(ViolationEvent), hsm.effect(_record_violation)),
        hsm.state(
            "connected",
            hsm.initial(hsm.target("/Session/connected/open")),
            hsm.transition(hsm.on(OpenStreamEvent), hsm.guard(_can_open_stream), hsm.effect(_open_stream)),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                hsm.guard(_has_duplicate_syn),
                hsm.effect(_fail_frame_protocol),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                hsm.guard(_targets_live_stream),
                hsm.effect(_forward_to_stream),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                hsm.guard(_targets_retired_stream),
                hsm.effect(_ignore_retired),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                hsm.guard(_opens_while_draining),
                hsm.effect(_refuse_while_draining),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                hsm.guard(_breaks_session_rules),
                hsm.effect(_fail_frame_protocol),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(
                hsm.on(ReceiveDataFrameEvent),
                hsm.guard(_opens_past_initial_window),
                hsm.effect(_refuse_past_initial_window),
            ),
            hsm.transition(hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent), hsm.effect(_accept_stream)),
            hsm.transition(
                hsm.on(ViolationEvent),
                hsm.guard(_is_protocol_violation),
                hsm.effect(_fail_stream_protocol),
                hsm.target("/Session/errored"),
            ),
            hsm.transition(hsm.on(FrameOutEvent), hsm.effect(_transmit_frame_out)),
            hsm.transition(hsm.on(ReceivePingFrameEvent), hsm.guard(_is_inbound_ping), hsm.effect(_receive_ping)),
            hsm.transition(
                hsm.on(GoAwayEvent),
                hsm.effect(_send_goaway),
                hsm.target("/Session/connected/routing_disconnect"),
            ),
            hsm.transition(
                hsm.on(ReceiveGoAwayFrameEvent),
                hsm.effect(_receive_goaway),
                hsm.target("/Session/connected/routing_disconnect"),
            ),
            hsm.choice(
                "routing_disconnect",
                hsm.transition(hsm.guard(_has_active_streams), hsm.target("/Session/connected/draining")),
                hsm.transition(hsm.target("/Session/disconnected")),
            ),
            hsm.state(
                "open",
                hsm.transition(hsm.on(PingEvent), hsm.effect(_send_ping), hsm.target("/Session/connected/pinging")),
            ),
            hsm.state(
                "pinging",
                hsm.transition(
                    hsm.on(ReceivePingFrameEvent),
                    hsm.guard(_is_outstanding_ping_ack),
                    hsm.effect(_receive_ping),
                    hsm.target("/Session/connected/open"),
                ),
                hsm.transition(
                    hsm.after(_ping_timeout_of),
                    hsm.effect(_ping_timed_out),
                    hsm.target("/Session/errored"),
                ),
            ),
            hsm.state(
                "draining",
                hsm.transition(hsm.on(SessionDrainedEvent), hsm.target("/Session/disconnected")),
            ),
        ),
        hsm.state("errored"),
        hsm.state("disconnected", hsm.entry(_end_drained)),
    )


def _require_uint32(name: str, value: int, *, minimum: int = 0) -> None:
    if not minimum <= value <= _MAX_UINT32:
        raise ValueError(f"{name} must be between {minimum} and {_MAX_UINT32}.")
