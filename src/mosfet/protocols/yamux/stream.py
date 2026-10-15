"""One yamux stream: the channel itself, owned by one actor, decoding to it and encoding for it.

Whoever owns a stream is the dispatch target of what comes out of it; ownership is the binding. The stream
is the transform host, the way a ``TransformStream`` sits in a pipe: inbound bytes are split into
length-framed messages (a 4-byte big-endian length, then the message), each decoded by the owner's codec
into one typed event dispatched to the owner, and an event the owner dispatches to the stream is encoded,
framed and written as yamux data. Window credit for a message goes back to the peer only after the owner
has processed its event, so a slow owner slows only its own stream. The stream sizes its receive window to
its codec, so the session and the socket under it never need to know the encoding.

The stream reaches its session only by message: ``FrameOutEvent`` for every frame it writes,
``ViolationEvent`` when the peer breaks the stream's rules, and ``StreamEndedEvent`` once it is over.
"""

import asyncio
import collections
import collections.abc
import dataclasses
import enum
import struct
import typing
import uuid

import hsm
import mosfet
import pydantic

from mosfet.protocols import transform
from mosfet.protocols.yamux.events import (
    CloseEvent,
    FailureKind,
    FrameOutData,
    FrameOutEvent,
    HandOverData,
    HandOverEvent,
    ReceiveDataFrameData,
    ReceiveDataFrameEvent,
    ReceiveWindowUpdateFrameData,
    ReceiveWindowUpdateFrameEvent,
    ResetEvent,
    SessionStage,
    StreamEndedEvent,
    StreamIdData,
    ViolationData,
    ViolationEvent,
)
from mosfet.protocols.yamux.frame import DEFAULT_MAX_DATA_LENGTH, INITIAL_STREAM_WINDOW, Flag, FrameData
from mosfet.telemetry.hsm import Traced

_MAX_UINT32 = (1 << 32) - 1
_LENGTH = struct.Struct("!I")


class StreamState(enum.StrEnum):
    """Yamux stream lifecycle."""

    LOCAL_OPENING = "local_opening"
    OPEN = "open"
    LOCAL_CLOSED = "local_closed"
    REMOTE_CLOSED = "remote_closed"
    CLOSED = "closed"
    RESET = "reset"


_READABLE = frozenset({StreamState.LOCAL_OPENING, StreamState.OPEN, StreamState.LOCAL_CLOSED})
_WRITABLE = frozenset({StreamState.LOCAL_OPENING, StreamState.OPEN, StreamState.REMOTE_CLOSED})
_TERMINAL = frozenset({StreamState.CLOSED, StreamState.RESET})


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StreamSnapshot(hsm.Snapshot):
    """Stable observation of one Yamux stream without exposing the mutable stream instance."""

    stream_id: int
    lifecycle: StreamState
    awaiting_ack: bool
    send_window: int
    receive_window: int
    sent_bytes: int
    received_bytes: int


class _DeliveredData(pydantic.BaseModel):
    """The owner finished with the message in flight; ``delivered`` is false when it could not take it."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    delivered: bool = pydantic.Field(description="Whether the owner's machine queued the decoded event.")


class _UndecodableData(pydantic.BaseModel):
    """The next message cannot become an event."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    reason: transform.DecodeReason


_DeliveredEvent = hsm.Event[_DeliveredData](name="protocol.yamux.stream.delivered", schema=_DeliveredData)
_UndecodableEvent = hsm.Event[_UndecodableData](name="protocol.yamux.stream.undecodable", schema=_UndecodableData)


def _tell(ctx: hsm.Context, recipient: hsm.Instance, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
    """Dispatch ``event`` to ``recipient`` with a fresh id, stamped from the current machine to ``recipient``."""

    return hsm.dispatch(ctx, recipient, dataclasses.replace(event, id=uuid.uuid4().hex, source="", target=""))


class Stream(Traced):
    """One yamux stream, owned by one actor and hosting that owner's codec.

    ``owner`` receives every event decoded from the stream and is the only actor that may hand the stream
    on (``HandOverEvent``). A stream the peer opened starts owned by the session's connection with no
    codec: its bytes stay unacknowledged, bounded by the receive window, until it is handed over. A duplex
    owner (one whose codec encodes something) named by a channel is told ``protocol.transform.owned`` when
    it gets the stream and ``protocol.transform.ended`` when it no longer has it.

    Each message is at most ``codec.max_message_bytes``. Since a message is credited only once it is
    delivered, its owner's codec sets how large the receive window must be: when the stream gets a codec
    whose message, plus the 4-byte length, does not fit the window granted so far, it grants the peer the
    difference (a window update), so every message the codec allows can complete. Events the owner sends
    past the send window wait in order, up to ``max_pending_bytes``; past that they are dropped and counted.
    """

    _stream_id: int
    _session: hsm.Instance
    _owner: hsm.Instance
    _codec: transform.Codec | None
    _channel: str | None
    _locally_initiated: bool
    _state: StreamState
    _awaiting_ack: bool
    _send_window: int
    _receive_window: int
    # The whole receive window granted to the peer so far: grown to fit the codec, never shrunk.
    _receive_capacity: int
    _sent_bytes: int
    _received_bytes: int
    # Received bytes not yet delivered; bounded by the receive window, since uncredited bytes cannot exceed it.
    _inbound: bytearray
    # Framed size of the message the owner is processing (None: none). Zero blocks delivery for a reset.
    _in_flight: int | None
    _delivery: asyncio.Future[bool] | None
    _pending: collections.deque[bytes]
    _pending_bytes: int
    _max_pending_bytes: int
    _fin_pending: bool
    _owner_ended: bool
    _session_ended: bool

    def __init__(
        self,
        *,
        stream_id: int,
        session: hsm.Instance,
        owner: hsm.Instance,
        codec: transform.Codec | None,
        channel: str | None,
        locally_initiated: bool,
        initial_send_window: int = INITIAL_STREAM_WINDOW,
        initial_receive_window: int = INITIAL_STREAM_WINDOW,
        max_pending_bytes: int = INITIAL_STREAM_WINDOW,
    ) -> None:
        super().__init__()
        _require_uint32("stream_id", stream_id, minimum=1)
        _require_uint32("initial_send_window", initial_send_window)
        _require_uint32("initial_receive_window", initial_receive_window)
        _require_uint32("max_pending_bytes", max_pending_bytes)
        self._stream_id = stream_id
        self._session = session
        self._owner = owner
        self._codec = codec
        self._channel = channel
        self._locally_initiated = locally_initiated
        self._state = StreamState.LOCAL_OPENING if locally_initiated else StreamState.OPEN
        self._awaiting_ack = locally_initiated
        self._send_window = initial_send_window
        self._receive_window = initial_receive_window
        self._receive_capacity = initial_receive_window
        self._sent_bytes = 0
        self._received_bytes = 0
        self._inbound = bytearray()
        self._in_flight = None
        self._delivery = None
        self._pending = collections.deque()
        self._pending_bytes = 0
        self._max_pending_bytes = max_pending_bytes
        self._fin_pending = False
        self._owner_ended = False
        self._session_ended = False

    @typing.override
    def take_snapshot(self) -> StreamSnapshot:
        """Return a stable observation of stream state and flow-control counters."""

        snapshot = super().take_snapshot()
        return StreamSnapshot(
            ID=snapshot.ID,
            QualifiedName=snapshot.QualifiedName,
            State=snapshot.State,
            Attributes=snapshot.Attributes,
            QueueLen=snapshot.QueueLen,
            Transitions=snapshot.Transitions,
            stream_id=self._stream_id,
            lifecycle=self._state,
            awaiting_ack=self._awaiting_ack,
            send_window=self._send_window,
            receive_window=self._receive_window,
            sent_bytes=self._sent_bytes,
            received_bytes=self._received_bytes,
        )

    # Guards. Each reads only this stream's own fields and the frame it is given.

    @staticmethod
    def _is_ack_only(instance: "Stream", flags: Flag, length: int) -> bool:
        """The empty ACK a locally opened stream still awaits."""

        return (
            instance._awaiting_ack
            and bool(flags & Flag.ACK)
            and not bool(flags & (Flag.SYN | Flag.FIN | Flag.RST))
            and length == 0
        )

    @staticmethod
    def _is_locally_initiated(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._locally_initiated

    @staticmethod
    def _is_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state is StreamState.CLOSED

    @staticmethod
    def _is_open(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state is StreamState.OPEN

    @staticmethod
    def _is_local_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state is StreamState.LOCAL_CLOSED

    @staticmethod
    def _is_remote_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state is StreamState.REMOTE_CLOSED

    @staticmethod
    def _has_bad_ack(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        flags = _frame_flags(event)
        return flags is not None and bool(flags & Flag.ACK) and not instance._awaiting_ack

    @staticmethod
    def _is_rst(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        flags = _frame_flags(event)
        return flags is not None and bool(flags & Flag.RST)

    @staticmethod
    def _data_hits_unreadable(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, ReceiveDataFrameData)
            and instance._state not in _READABLE
            and not Stream._is_ack_only(instance, Flag(data.flags), len(data.payload))
        )

    @staticmethod
    def _data_exhausts_window(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, ReceiveDataFrameData) and len(data.payload) > instance._receive_window

    @staticmethod
    def _window_has_bad_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return (
            isinstance(data, ReceiveWindowUpdateFrameData)
            and bool(Flag(data.flags) & Flag.FIN)
            and instance._state not in _READABLE
            and not Stream._is_ack_only(instance, Flag(data.flags), data.delta)
        )

    @staticmethod
    def _window_overflows(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = event.data
        return isinstance(data, ReceiveWindowUpdateFrameData) and instance._send_window + data.delta > _MAX_UINT32

    @staticmethod
    def _is_late_frame(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        """A frame for a closed stream that is neither its awaited ACK nor a window update for pending bytes."""

        del ctx
        data = event.data
        if isinstance(data, ReceiveDataFrameData):
            return not Stream._is_ack_only(instance, Flag(data.flags), len(data.payload))
        return isinstance(data, ReceiveWindowUpdateFrameData) and bool(Flag(data.flags) & Flag.FIN)

    @staticmethod
    def _is_writable(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state in _WRITABLE

    @staticmethod
    def _encodes(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        """Pure name lookup in the owner's codec; yamux's own events are never registered there."""

        del ctx
        codec = instance._codec
        return instance._state in _WRITABLE and codec is not None and codec.encodes(event)

    @staticmethod
    def _from_owner(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        """Correlate a hand-over with this stream's own owner: only the current owner gives it away."""

        del ctx
        if not isinstance(event.data, HandOverData) or instance._state is StreamState.RESET:
            return False
        try:
            owner_id = hsm.id(instance._owner)
        except RuntimeError:
            return False
        return event.source == owner_id

    @staticmethod
    def _is_not_reset(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state is not StreamState.RESET

    # Effects.

    @staticmethod
    def _enter_active(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        """Fit the window to the first owner's codec and tell that owner it owns the stream."""

        del event
        Stream._fit_window(ctx, instance)
        Stream._tell_owner(ctx, instance, None)

    @staticmethod
    def _accept_data(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveDataFrameData)
        flags = Flag(data.flags)
        if flags & Flag.ACK:
            Stream._acknowledge(instance)
        if data.payload:
            instance._inbound += data.payload
            instance._receive_window -= len(data.payload)
            instance._received_bytes += len(data.payload)
        if flags & Flag.FIN:
            Stream._remote_fin(instance)
        Stream._deliver_next(ctx, instance)

    @staticmethod
    def _accept_window(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ReceiveWindowUpdateFrameData)
        flags = Flag(data.flags)
        if flags & Flag.ACK:
            Stream._acknowledge(instance)
        instance._send_window += data.delta
        Stream._flush(ctx, instance)
        if flags & Flag.FIN:
            Stream._remote_fin(instance)
            Stream._deliver_next(ctx, instance)

    @staticmethod
    def _accept_closing(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        """An awaited ACK or a window update for bytes still pending, once both sides closed."""

        data = event.data
        assert isinstance(data, ReceiveDataFrameData | ReceiveWindowUpdateFrameData)
        if Flag(data.flags) & Flag.ACK:
            Stream._acknowledge(instance)
        if isinstance(data, ReceiveWindowUpdateFrameData):
            instance._send_window += data.delta
            Stream._flush(ctx, instance)
        Stream._end_if_done(ctx, instance)

    @staticmethod
    def _violate_protocol(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        Stream._violate(ctx, instance, event, "protocol_error")

    @staticmethod
    def _violate_stream(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        Stream._violate(ctx, instance, event, "stream_reset")

    @staticmethod
    def _violate_window(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        Stream._violate(ctx, instance, event, "receive_window_exhausted")

    @staticmethod
    def _write_rst(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del event
        Stream._write(ctx, instance, FrameData.data(stream_id=instance._stream_id, flags=Flag.RST))

    @staticmethod
    def _refuse_undecodable(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _UndecodableData)
        codec = instance._codec
        assert codec is not None
        transform.telemetry.decode_failed(codec, data.reason)
        Stream._write(ctx, instance, FrameData.data(stream_id=instance._stream_id, flags=Flag.RST))

    @staticmethod
    def _send(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        """Encode what the owner sent, frame it, and write what the send window allows; queue the rest."""

        codec = instance._codec
        assert codec is not None
        try:
            message = codec.encode(event)
        except transform.EncodeError:
            transform.telemetry.dropped(codec, "encode_failed")
            return
        framed = _LENGTH.pack(len(message)) + message
        if instance._pending_bytes + len(framed) > instance._max_pending_bytes:
            transform.telemetry.dropped(codec, "flow_control")
            return
        transform.telemetry.transferred(codec, "out", len(message))
        instance._pending.append(framed)
        instance._pending_bytes += len(framed)
        Stream._flush(ctx, instance)

    @staticmethod
    def _close(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        """Half-close: FIN goes after every message already sent, once the window lets them out."""

        del event
        instance._fin_pending = True
        Stream._flush(ctx, instance)
        instance._state = (
            StreamState.CLOSED if instance._state is StreamState.REMOTE_CLOSED else StreamState.LOCAL_CLOSED
        )

    @staticmethod
    def _hand_over(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, HandOverData)
        Stream._tell_owner(ctx, instance, "handed_over")
        instance._owner = data.owner
        instance._codec = data.codec
        instance._channel = data.channel
        instance._owner_ended = False
        Stream._fit_window(ctx, instance)
        Stream._tell_owner(ctx, instance, None)
        Stream._deliver_next(ctx, instance)

    @staticmethod
    def _delivered(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        """Credit the delivered message back to the peer, then deliver the next one."""

        data = event.data
        assert isinstance(data, _DeliveredData)
        in_flight = instance._in_flight
        if not in_flight:
            return
        codec = instance._codec
        if not data.delivered and codec is not None:
            transform.telemetry.dropped(codec, "unowned")
        instance._in_flight = None
        instance._delivery = None
        if instance._state in _READABLE:
            instance._receive_window += in_flight
            Stream._write(ctx, instance, FrameData.window_update(stream_id=instance._stream_id, delta=in_flight))
        Stream._deliver_next(ctx, instance)
        Stream._end_if_done(ctx, instance)

    @staticmethod
    def _mark_reset(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        """Drop every buffered byte either way, block delivery, and say so to the owner and the session."""

        del event
        instance._state = StreamState.RESET
        instance._awaiting_ack = False
        instance._inbound.clear()
        instance._pending.clear()
        instance._pending_bytes = 0
        instance._fin_pending = False
        instance._in_flight = 0
        Stream._tell_owner(ctx, instance, "reset")
        Stream._end_if_done(ctx, instance)

    @staticmethod
    def _finish(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del event
        Stream._end_if_done(ctx, instance)

    @staticmethod
    def _end_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        """Leaving ``owned`` happens only on stop: tell an owner not yet told that the stream closed."""

        del event
        Stream._tell_owner(ctx, instance, "closed")

    # Helpers called from the effects above.

    @staticmethod
    def _tell_owner(
        ctx: hsm.Context,
        instance: "Stream",
        reason: typing.Literal["finished", "reset", "closed", "handed_over"] | None,
    ) -> None:
        """Tell a duplex owner named by a channel that it owns the stream (no reason) or no longer does (once)."""

        codec = instance._codec
        channel = instance._channel
        if codec is None or not codec.encodes_any or channel is None or instance._owner_ended:
            return
        if reason is None:
            told: hsm.Event[typing.Any] = transform.OwnedEvent.with_data(
                transform.OwnedData(stream=instance, channel=channel)
            )
        else:
            instance._owner_ended = True
            told = transform.EndedEvent.with_data(transform.EndedData(stream=instance, channel=channel, reason=reason))
        _ = _tell(ctx, instance._owner, told)

    @staticmethod
    def _fit_window(ctx: hsm.Context, instance: "Stream") -> None:
        """Grow the receive window, while the peer may still send, until one whole message of the codec fits."""

        codec = instance._codec
        if codec is None or instance._state not in _READABLE:
            return
        needed = min(_LENGTH.size + codec.max_message_bytes, _MAX_UINT32)
        delta = needed - instance._receive_capacity
        if delta <= 0:
            return
        instance._receive_capacity = needed
        instance._receive_window += delta
        Stream._write(ctx, instance, FrameData.window_update(stream_id=instance._stream_id, delta=delta))

    @staticmethod
    def _acknowledge(instance: "Stream") -> None:
        instance._awaiting_ack = False
        if instance._state is StreamState.LOCAL_OPENING:
            instance._state = StreamState.OPEN

    @staticmethod
    def _remote_fin(instance: "Stream") -> None:
        instance._state = (
            StreamState.CLOSED if instance._state is StreamState.LOCAL_CLOSED else StreamState.REMOTE_CLOSED
        )

    @staticmethod
    def _write(ctx: hsm.Context, instance: "Stream", frame: FrameData) -> None:
        _ = _tell(
            ctx,
            instance._session,
            FrameOutEvent.with_data(FrameOutData(stream_id=instance._stream_id, frame=frame.to_bytes())),
        )

    @staticmethod
    def _flush(ctx: hsm.Context, instance: "Stream") -> None:
        """Write pending bytes as far as the send window allows, then a pending FIN once nothing is left."""

        while instance._pending and instance._send_window > 0:
            chunk = instance._pending[0]
            size = min(len(chunk), instance._send_window, DEFAULT_MAX_DATA_LENGTH)
            if size == len(chunk):
                _ = instance._pending.popleft()
            else:
                instance._pending[0] = chunk[size:]
            instance._pending_bytes -= size
            instance._send_window -= size
            instance._sent_bytes += size
            Stream._write(ctx, instance, FrameData.data(stream_id=instance._stream_id, payload=chunk[:size]))
        if instance._fin_pending and not instance._pending:
            instance._fin_pending = False
            Stream._write(ctx, instance, FrameData.data(stream_id=instance._stream_id, flags=Flag.FIN))

    @staticmethod
    def _deliver_next(ctx: hsm.Context, instance: "Stream") -> None:
        """Decode the next whole message to the owner, unless one is in flight or there is no codec yet."""

        codec = instance._codec
        if instance._in_flight is not None or codec is None or len(instance._inbound) < _LENGTH.size:
            Stream._drop_truncated(instance)
            return
        length = int.from_bytes(instance._inbound[: _LENGTH.size], "big")
        if length > codec.max_message_bytes:
            Stream._undecodable(ctx, instance, "oversize")
            return
        framed = _LENGTH.size + length
        if len(instance._inbound) < framed:
            Stream._drop_truncated(instance)
            return
        message = bytes(instance._inbound[_LENGTH.size : framed])
        del instance._inbound[:framed]
        try:
            decoded = codec.decode(message)
        except transform.DecodeError as error:
            Stream._undecodable(ctx, instance, error.reason)
            return
        transform.telemetry.transferred(codec, "in", len(message))
        instance._in_flight = framed
        delivery = asyncio.ensure_future(_tell(ctx, instance._owner, decoded))
        instance._delivery = delivery
        delivery.add_done_callback(lambda done: Stream._report_delivery(instance, done))

    @staticmethod
    def _report_delivery(instance: "Stream", done: asyncio.Future[bool]) -> None:
        """Delivery callback: one typed event back to this stream, so the credit is a transition of its own."""

        delivered = not done.cancelled() and done.exception() is None and done.result()
        _ = hsm.dispatch(
            instance.context(),
            instance,
            _DeliveredEvent.with_data(_DeliveredData(delivered=delivered)),
        )

    @staticmethod
    def _undecodable(ctx: hsm.Context, instance: "Stream", reason: transform.DecodeReason) -> None:
        instance._in_flight = 0
        instance._inbound.clear()
        _ = _tell(ctx, instance, _UndecodableEvent.with_data(_UndecodableData(reason=reason)))

    @staticmethod
    def _drop_truncated(instance: "Stream") -> None:
        """Once the peer finished, bytes that never became a whole message are dropped and counted."""

        codec = instance._codec
        if (
            instance._in_flight is None
            and instance._inbound
            and codec is not None
            and instance._state in (StreamState.REMOTE_CLOSED, StreamState.CLOSED)
        ):
            instance._inbound.clear()
            transform.telemetry.decode_failed(codec, "malformed")

    @staticmethod
    def _end_if_done(ctx: hsm.Context, instance: "Stream") -> None:
        """Once both sides are done and everything is delivered and written, tell the owner and the session."""

        if instance._state not in _TERMINAL or instance._session_ended:
            return
        drained = instance._state is StreamState.RESET or (
            instance._in_flight is None
            and not instance._inbound
            and not instance._pending
            and not instance._fin_pending
        )
        if not drained:
            return
        Stream._tell_owner(ctx, instance, "finished")
        if instance._awaiting_ack:
            return
        instance._session_ended = True
        _ = _tell(ctx, instance._session, StreamEndedEvent.with_data(StreamIdData(stream_id=instance._stream_id)))

    @staticmethod
    def _violate(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any], kind: FailureKind) -> None:
        Stream._write(ctx, instance, FrameData.data(stream_id=instance._stream_id, flags=Flag.RST))
        _ = _tell(
            ctx,
            instance._session,
            ViolationEvent.with_data(ViolationData(stream_id=instance._stream_id, stage=_stage(event), kind=kind)),
        )

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Stream",
        hsm.initial(hsm.target("/Stream/owned")),
        hsm.state(
            "owned",
            hsm.exit(_end_closed),
            hsm.initial(hsm.target("/Stream/owned/routing_initial")),
            hsm.choice(
                "routing_initial",
                hsm.transition(hsm.guard(_is_locally_initiated), hsm.target("/Stream/owned/active/local_opening")),
                hsm.transition(hsm.target("/Stream/owned/active/open")),
            ),
            hsm.transition(hsm.on(_DeliveredEvent), hsm.effect(_delivered)),
            hsm.transition(hsm.on(HandOverEvent), hsm.guard(_from_owner), hsm.effect(_hand_over)),
            hsm.transition(
                hsm.on(ResetEvent),
                hsm.guard(_is_not_reset),
                hsm.effect(_write_rst),
                hsm.target("/Stream/owned/reset"),
            ),
            hsm.transition(
                hsm.on(_UndecodableEvent),
                hsm.guard(_is_not_reset),
                hsm.effect(_refuse_undecodable),
                hsm.target("/Stream/owned/reset"),
            ),
            hsm.state(
                "active",
                hsm.entry(_enter_active),
                hsm.initial(hsm.target("/Stream/owned/active/local_opening")),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_has_bad_ack),
                    hsm.effect(_violate_protocol),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_is_rst),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent),
                    hsm.guard(_data_hits_unreadable),
                    hsm.effect(_violate_stream),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent),
                    hsm.guard(_data_exhausts_window),
                    hsm.effect(_violate_window),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent),
                    hsm.effect(_accept_data),
                    hsm.target("/Stream/owned/active/settle"),
                ),
                hsm.transition(
                    hsm.on(ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_window_has_bad_fin),
                    hsm.effect(_violate_stream),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_window_overflows),
                    hsm.effect(_violate_protocol),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveWindowUpdateFrameEvent),
                    hsm.effect(_accept_window),
                    hsm.target("/Stream/owned/active/settle"),
                ),
                hsm.transition(
                    hsm.on(CloseEvent),
                    hsm.guard(_is_writable),
                    hsm.effect(_close),
                    hsm.target("/Stream/owned/active/settle"),
                ),
                hsm.transition(hsm.on(hsm.AnyEvent), hsm.guard(_encodes), hsm.effect(_send)),
                hsm.choice(
                    "settle",
                    hsm.transition(hsm.guard(_is_closed), hsm.target("/Stream/owned/closed")),
                    hsm.transition(hsm.guard(_is_remote_closed), hsm.target("/Stream/owned/active/remote_closed")),
                    hsm.transition(hsm.guard(_is_local_closed), hsm.target("/Stream/owned/active/local_closed")),
                    hsm.transition(hsm.guard(_is_open), hsm.target("/Stream/owned/active/open")),
                    hsm.transition(hsm.target("/Stream/owned/active/local_opening")),
                ),
                hsm.state("local_opening"),
                hsm.state("open"),
                hsm.state("local_closed"),
                hsm.state("remote_closed"),
            ),
            hsm.state(
                "closed",
                hsm.entry(_finish),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_has_bad_ack),
                    hsm.effect(_violate_protocol),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_is_rst),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_is_late_frame),
                    hsm.effect(_violate_stream),
                ),
                hsm.transition(
                    hsm.on(ReceiveWindowUpdateFrameEvent),
                    hsm.guard(_window_overflows),
                    hsm.effect(_violate_protocol),
                    hsm.target("/Stream/owned/reset"),
                ),
                hsm.transition(
                    hsm.on(ReceiveDataFrameEvent, ReceiveWindowUpdateFrameEvent),
                    hsm.effect(_accept_closing),
                ),
            ),
            hsm.state("reset", hsm.entry(_mark_reset)),
        ),
    )


def _frame_flags(event: hsm.Event[typing.Any]) -> Flag | None:
    data = event.data
    if isinstance(data, ReceiveDataFrameData | ReceiveWindowUpdateFrameData):
        return Flag(data.flags)
    return None


def _stage(event: hsm.Event[typing.Any]) -> SessionStage:
    return "receive_window_update" if isinstance(event.data, ReceiveWindowUpdateFrameData) else "receive_data"


def _require_uint32(name: str, value: int, *, minimum: int = 0) -> None:
    if not minimum <= value <= _MAX_UINT32:
        raise ValueError(f"{name} must be between {minimum} and {_MAX_UINT32}.")
