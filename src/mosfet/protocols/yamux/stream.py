import asyncio
import collections.abc
import dataclasses
import enum
import typing

import hsm
import mosfet
import pydantic

from mosfet.protocols.yamux.frame import INITIAL_STREAM_WINDOW

_MAX_UINT32 = (1 << 32) - 1
WindowConsumedCallback = collections.abc.Callable[[int, int], collections.abc.Awaitable[None] | None]


@typing.runtime_checkable
class _CompletedDispatch(typing.Protocol):
    def done(self) -> bool: ...

    def exception(self) -> BaseException | None: ...


class StreamState(enum.StrEnum):
    """Yamux stream lifecycle as ordinary Python stream state."""

    LOCAL_OPENING = "local_opening"
    OPEN = "open"
    LOCAL_CLOSED = "local_closed"
    REMOTE_CLOSED = "remote_closed"
    CLOSED = "closed"
    RESET = "reset"


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


class StreamSendData(pydantic.BaseModel):
    """Private stream event payload for bytes emitted by local application code."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"payload": "hello", "end_stream": False}]},
    )

    payload: bytes = pydantic.Field(
        default=b"",
        description="Raw Yamux stream payload bytes to account against the peer-granted send window.",
        examples=[b"hello"],
    )
    end_stream: bool = pydantic.Field(
        default=False,
        description="Whether this send also half-closes local writes with FIN.",
        examples=[False],
    )


class StreamReceiveData(pydantic.BaseModel):
    """Private stream event payload for bytes delivered by the remote peer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"payload": "hello", "end_stream": True}]},
    )

    payload: bytes = pydantic.Field(
        default=b"",
        description="Raw Yamux stream payload bytes to feed to the Python reader.",
        examples=[b"hello"],
    )
    end_stream: bool = pydantic.Field(
        default=False,
        description="Whether this receive also half-closes remote writes with FIN.",
        examples=[True],
    )


class StreamWindowData(pydantic.BaseModel):
    """Private stream event payload for Yamux flow-control window changes."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"delta": 4096}]},
    )

    delta: typing.Annotated[
        int,
        pydantic.Field(
            ge=0,
            le=_MAX_UINT32,
            description="Unsigned 32-bit byte-count delta added to a stream flow-control window.",
            examples=[4096],
        ),
    ]


@dataclasses.dataclass(frozen=True, slots=True)
class _StreamOperationFailure:
    message: str


class _StreamSignalData(pydantic.BaseModel):
    """Typed empty signal for Yamux stream lifecycle transitions.

    Acknowledge, local/remote FIN, and reset carry no payload; this type gives those
    signals a modeled schema instead of an untyped ``Event[None]``. Delivery still
    matches on the event name, so the schema change is wire-compatible.
    """

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": "Empty lifecycle signal for one Yamux stream; carries no payload.",
            "examples": [{}],
        },
    )


_AcknowledgeEvent = hsm.Event[_StreamSignalData](
    name="protocol.yamux.stream.acknowledge",
    schema=_StreamSignalData,
)
_SendEvent = hsm.Event[StreamSendData](
    name="protocol.yamux.stream.send",
    schema=StreamSendData,
)
_ReceiveEvent = hsm.Event[StreamReceiveData](
    name="protocol.yamux.stream.receive",
    schema=StreamReceiveData,
)
_UpdateSendWindowEvent = hsm.Event[StreamWindowData](
    name="protocol.yamux.stream.send_window.update",
    schema=StreamWindowData,
)
_GrantReceiveWindowEvent = hsm.Event[StreamWindowData](
    name="protocol.yamux.stream.receive_window.grant",
    schema=StreamWindowData,
)
_LocalFinEvent = hsm.Event[_StreamSignalData](
    name="protocol.yamux.stream.local_fin",
    schema=_StreamSignalData,
)
_RemoteFinEvent = hsm.Event[_StreamSignalData](
    name="protocol.yamux.stream.remote_fin",
    schema=_StreamSignalData,
)
_ResetEvent = hsm.Event[_StreamSignalData](
    name="protocol.yamux.stream.reset",
    schema=_StreamSignalData,
)


def _send_data(event: hsm.Event[typing.Any]) -> StreamSendData | None:
    data = event.data
    return data if isinstance(data, StreamSendData) else None


def _receive_data(event: hsm.Event[typing.Any]) -> StreamReceiveData | None:
    data = event.data
    return data if isinstance(data, StreamReceiveData) else None


def _window_data(event: hsm.Event[typing.Any]) -> StreamWindowData | None:
    data = event.data
    return data if isinstance(data, StreamWindowData) else None


@dataclasses.dataclass(kw_only=True)
class Stream(hsm.Instance):
    """Python stream abstraction for one Yamux stream.

    The Yamux lifecycle is modeled as an HSM class contract. The object still owns the embedded Python reader,
    half-close observation, and flow-control counters used by the session HSM.
    """

    @staticmethod
    def _is_remote_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state is StreamState.REMOTE_CLOSED

    @staticmethod
    def _is_local_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._state is StreamState.LOCAL_CLOSED

    @staticmethod
    def _is_locally_initiated(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._locally_initiated

    @staticmethod
    def _can_send_without_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _send_data(event)
        return (
            data is not None
            and not data.end_stream
            and instance._can_send()
            and len(data.payload) <= instance._send_window
        )

    @staticmethod
    def _can_send_with_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _send_data(event)
        return (
            data is not None and data.end_stream and instance._can_send() and len(data.payload) <= instance._send_window
        )

    @staticmethod
    def _can_receive_without_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _receive_data(event)
        return (
            data is not None
            and not data.end_stream
            and instance._can_receive()
            and len(data.payload) <= instance._receive_window
        )

    @staticmethod
    def _can_receive_with_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _receive_data(event)
        return (
            data is not None
            and data.end_stream
            and instance._can_receive()
            and len(data.payload) <= instance._receive_window
        )

    @staticmethod
    def _can_update_send_window(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _window_data(event)
        return (
            data is not None
            and not instance._is_terminal()
            and instance._send_window + data.delta <= instance._max_window_size
        )

    @staticmethod
    def _can_grant_receive_window(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _window_data(event)
        return (
            data is not None
            and not instance._is_terminal()
            and instance._receive_window + data.delta <= instance._max_window_size
        )

    @staticmethod
    def _can_local_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._can_send()

    @staticmethod
    def _can_remote_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._can_receive()

    @staticmethod
    def _can_reset(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return not instance._is_terminal()

    @staticmethod
    def _can_acknowledge(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._awaiting_ack

    @staticmethod
    def _cannot_acknowledge(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return not instance._awaiting_ack

    @staticmethod
    def _cannot_send(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _send_data(event)
        return data is not None and (not instance._can_send() or len(data.payload) > instance._send_window)

    @staticmethod
    def _cannot_receive(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _receive_data(event)
        return data is not None and (not instance._can_receive() or len(data.payload) > instance._receive_window)

    @staticmethod
    def _cannot_update_send_window(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _window_data(event)
        return data is not None and (
            instance._is_terminal() or instance._send_window + data.delta > instance._max_window_size
        )

    @staticmethod
    def _cannot_grant_receive_window(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        data = _window_data(event)
        return data is not None and (
            instance._is_terminal() or instance._receive_window + data.delta > instance._max_window_size
        )

    @staticmethod
    def _cannot_local_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return not instance._can_send()

    @staticmethod
    def _cannot_remote_fin(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return not instance._can_receive()

    @staticmethod
    def _cannot_reset(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> bool:
        del ctx, event
        return instance._is_terminal()

    @staticmethod
    def _reject_send(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure(
            "Yamux stream is not writable or send window is exhausted."
        )

    @staticmethod
    def _reject_receive(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure(
            "Yamux stream is not readable or receive window is exhausted."
        )

    @staticmethod
    def _reject_send_window_update(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure("Yamux stream send window cannot be updated.")

    @staticmethod
    def _reject_receive_window_grant(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure("Yamux stream receive window cannot be granted.")

    @staticmethod
    def _reject_local_fin(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure("Yamux stream local writes are already closed.")

    @staticmethod
    def _reject_remote_fin(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure("Yamux stream remote writes are already closed.")

    @staticmethod
    def _reject_reset(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure("Yamux stream is terminal.")

    @staticmethod
    def _reject_acknowledge(ctx: hsm.Context, instance: "Stream", _event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance._operation_failure = _StreamOperationFailure("Yamux stream is not awaiting acknowledgement.")

    @staticmethod
    def _record_sent(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, StreamSendData)
        instance._record_sent_payload(data.payload)

    @staticmethod
    def _record_received(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, StreamReceiveData)
        instance._record_received_payload(data.payload)

    @staticmethod
    def _apply_send_window_update(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, StreamWindowData)
        instance._increase_send_window(data.delta)

    @staticmethod
    def _apply_receive_window_grant(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx
        data = event.data
        assert isinstance(data, StreamWindowData)
        instance._increase_receive_window(data.delta)

    @staticmethod
    def _mark_acknowledged(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._awaiting_ack = False

    @staticmethod
    def _mark_open(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._state = StreamState.OPEN

    @staticmethod
    def _mark_local_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._state = StreamState.LOCAL_CLOSED

    @staticmethod
    def _mark_remote_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._state = StreamState.REMOTE_CLOSED
        instance._feed_eof()

    @staticmethod
    def _mark_closed(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._state = StreamState.CLOSED
        instance._feed_eof()

    @staticmethod
    def _mark_reset(ctx: hsm.Context, instance: "Stream", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._state = StreamState.RESET
        instance._awaiting_ack = False
        instance._feed_eof()

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "Stream",
        hsm.initial(hsm.target("/Stream/routing_initial")),
        hsm.transition(
            hsm.on(_AcknowledgeEvent),
            hsm.guard(_cannot_acknowledge),
            hsm.effect(_reject_acknowledge),
        ),
        hsm.transition(hsm.on(_SendEvent), hsm.guard(_cannot_send), hsm.effect(_reject_send)),
        hsm.transition(hsm.on(_ReceiveEvent), hsm.guard(_cannot_receive), hsm.effect(_reject_receive)),
        hsm.transition(
            hsm.on(_UpdateSendWindowEvent),
            hsm.guard(_cannot_update_send_window),
            hsm.effect(_reject_send_window_update),
        ),
        hsm.transition(
            hsm.on(_GrantReceiveWindowEvent),
            hsm.guard(_cannot_grant_receive_window),
            hsm.effect(_reject_receive_window_grant),
        ),
        hsm.transition(hsm.on(_LocalFinEvent), hsm.guard(_cannot_local_fin), hsm.effect(_reject_local_fin)),
        hsm.transition(hsm.on(_RemoteFinEvent), hsm.guard(_cannot_remote_fin), hsm.effect(_reject_remote_fin)),
        hsm.transition(hsm.on(_ResetEvent), hsm.guard(_cannot_reset), hsm.effect(_reject_reset)),
        hsm.choice(
            "routing_initial",
            hsm.transition(hsm.guard(_is_locally_initiated), hsm.target("/Stream/active/local_opening")),
            hsm.transition(hsm.effect(_mark_open), hsm.target("/Stream/active/open")),
        ),
        hsm.state(
            "active",
            hsm.initial(hsm.target("/Stream/active/local_opening")),
            hsm.transition(hsm.on(_SendEvent), hsm.guard(_can_send_without_fin), hsm.effect(_record_sent)),
            hsm.transition(
                hsm.on(_SendEvent),
                hsm.guard(_can_send_with_fin),
                hsm.effect(_record_sent),
                hsm.target("/Stream/active/routing_local_fin"),
            ),
            hsm.transition(hsm.on(_ReceiveEvent), hsm.guard(_can_receive_without_fin), hsm.effect(_record_received)),
            hsm.transition(
                hsm.on(_ReceiveEvent),
                hsm.guard(_can_receive_with_fin),
                hsm.effect(_record_received),
                hsm.target("/Stream/active/routing_remote_fin"),
            ),
            hsm.transition(
                hsm.on(_UpdateSendWindowEvent),
                hsm.guard(_can_update_send_window),
                hsm.effect(_apply_send_window_update),
            ),
            hsm.transition(
                hsm.on(_GrantReceiveWindowEvent),
                hsm.guard(_can_grant_receive_window),
                hsm.effect(_apply_receive_window_grant),
            ),
            hsm.transition(
                hsm.on(_LocalFinEvent),
                hsm.guard(_can_local_fin),
                hsm.target("/Stream/active/routing_local_fin"),
            ),
            hsm.transition(
                hsm.on(_RemoteFinEvent),
                hsm.guard(_can_remote_fin),
                hsm.target("/Stream/active/routing_remote_fin"),
            ),
            hsm.transition(
                hsm.on(_ResetEvent), hsm.guard(_can_reset), hsm.effect(_mark_reset), hsm.target("/Stream/reset")
            ),
            hsm.choice(
                "routing_local_fin",
                hsm.transition(hsm.guard(_is_remote_closed), hsm.effect(_mark_closed), hsm.target("/Stream/closed")),
                hsm.transition(hsm.effect(_mark_local_closed), hsm.target("/Stream/active/local_closed")),
            ),
            hsm.choice(
                "routing_remote_fin",
                hsm.transition(hsm.guard(_is_local_closed), hsm.effect(_mark_closed), hsm.target("/Stream/closed")),
                hsm.transition(hsm.effect(_mark_remote_closed), hsm.target("/Stream/active/remote_closed")),
            ),
            hsm.state(
                "local_opening",
                hsm.transition(
                    hsm.on(_AcknowledgeEvent),
                    hsm.guard(_can_acknowledge),
                    hsm.effect(_mark_acknowledged),
                    hsm.effect(_mark_open),
                    hsm.target("/Stream/active/open"),
                ),
            ),
            hsm.state("open"),
            hsm.state(
                "local_closed",
                hsm.transition(
                    hsm.on(_AcknowledgeEvent),
                    hsm.guard(_can_acknowledge),
                    hsm.effect(_mark_acknowledged),
                ),
            ),
            hsm.state(
                "remote_closed",
                hsm.transition(
                    hsm.on(_AcknowledgeEvent),
                    hsm.guard(_can_acknowledge),
                    hsm.effect(_mark_acknowledged),
                ),
            ),
        ),
        hsm.state(
            "closed",
            hsm.transition(
                hsm.on(_AcknowledgeEvent),
                hsm.guard(_can_acknowledge),
                hsm.effect(_mark_acknowledged),
            ),
        ),
        hsm.state("reset"),
    )
    stream_id: dataclasses.InitVar[int]
    locally_initiated: dataclasses.InitVar[bool]
    initial_send_window: dataclasses.InitVar[int] = INITIAL_STREAM_WINDOW
    initial_receive_window: dataclasses.InitVar[int] = INITIAL_STREAM_WINDOW
    max_window_size: dataclasses.InitVar[int] = _MAX_UINT32
    on_window_consumed: dataclasses.InitVar[WindowConsumedCallback | None] = None
    _awaiting_ack: bool = dataclasses.field(init=False, repr=False)
    _locally_initiated: bool = dataclasses.field(init=False, repr=False)
    _max_window_size: int = dataclasses.field(init=False, repr=False)
    _on_window_consumed: WindowConsumedCallback | None = dataclasses.field(init=False, repr=False)
    _reader: asyncio.StreamReader | None = dataclasses.field(init=False, repr=False)
    _received_bytes: int = dataclasses.field(init=False, repr=False)
    _receive_window: int = dataclasses.field(init=False, repr=False)
    _send_window: int = dataclasses.field(init=False, repr=False)
    _sent_bytes: int = dataclasses.field(init=False, repr=False)
    _state: StreamState = dataclasses.field(init=False, repr=False)
    _stream_id: int = dataclasses.field(init=False, repr=False)
    _operation_failure: _StreamOperationFailure | None = dataclasses.field(init=False, repr=False)

    def __post_init__(
        self,
        stream_id: int,
        locally_initiated: bool,
        initial_send_window: int = INITIAL_STREAM_WINDOW,
        initial_receive_window: int = INITIAL_STREAM_WINDOW,
        max_window_size: int = _MAX_UINT32,
        on_window_consumed: WindowConsumedCallback | None = None,
    ) -> None:
        _require_uint32("stream_id", stream_id, minimum=1)
        _require_uint32("initial_send_window", initial_send_window)
        _require_uint32("initial_receive_window", initial_receive_window)
        _require_uint32("max_window_size", max_window_size)
        if initial_send_window > max_window_size:
            raise ValueError("initial_send_window must not exceed max_window_size.")
        if initial_receive_window > max_window_size:
            raise ValueError("initial_receive_window must not exceed max_window_size.")
        self._stream_id = stream_id
        self._send_window = initial_send_window
        self._receive_window = initial_receive_window
        self._sent_bytes = 0
        self._received_bytes = 0
        self._max_window_size = max_window_size
        self._awaiting_ack = locally_initiated
        self._locally_initiated = locally_initiated
        self._on_window_consumed = on_window_consumed
        self._reader = None
        self._state = StreamState.LOCAL_OPENING if locally_initiated else StreamState.OPEN
        self._operation_failure = None

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

    def _can_send(self) -> bool:
        return self._state in {
            StreamState.LOCAL_OPENING,
            StreamState.OPEN,
            StreamState.REMOTE_CLOSED,
        }

    def _can_receive(self) -> bool:
        return self._state in {
            StreamState.LOCAL_OPENING,
            StreamState.OPEN,
            StreamState.LOCAL_CLOSED,
        }

    def _is_terminal(self) -> bool:
        return self._state in {StreamState.CLOSED, StreamState.RESET}

    def at_eof(self) -> bool:
        """Return true when the inbound byte stream reached EOF and its buffer is empty."""

        reader = self._reader
        if reader is None:
            return self._state in {StreamState.REMOTE_CLOSED, StreamState.CLOSED, StreamState.RESET}
        return reader.at_eof()

    async def read(self, n: int = -1) -> bytes:
        """Read bytes from the inbound Yamux stream."""

        data = await self._ensure_reader().read(n)
        await self._record_window_consumed(len(data))
        return data

    async def readexactly(self, n: int) -> bytes:
        """Read exactly n bytes from the inbound Yamux stream."""

        try:
            data = await self._ensure_reader().readexactly(n)
        except asyncio.IncompleteReadError as error:
            await self._record_window_consumed(len(error.partial))
            raise
        await self._record_window_consumed(len(data))
        return data

    async def readline(self) -> bytes:
        """Read one line from the inbound Yamux stream."""

        data = await self._ensure_reader().readline()
        await self._record_window_consumed(len(data))
        return data

    def acknowledge(self) -> None:
        """Mark a locally initiated stream as accepted by the peer."""

        self._dispatch_stream_event(_AcknowledgeEvent.with_data(_StreamSignalData()))

    def send(self, payload: bytes, *, end_stream: bool = False) -> None:
        """Record sent data bytes and optionally local FIN."""

        self._dispatch_stream_event(_SendEvent.with_data(StreamSendData(payload=payload, end_stream=end_stream)))

    def receive(self, payload: bytes, *, end_stream: bool = False) -> None:
        """Feed received data bytes and optionally remote FIN."""

        self._dispatch_stream_event(_ReceiveEvent.with_data(StreamReceiveData(payload=payload, end_stream=end_stream)))

    def update_send_window(self, delta: int) -> None:
        """Increase the peer-granted send window."""

        _require_uint32("delta", delta)
        self._dispatch_stream_event(_UpdateSendWindowEvent.with_data(StreamWindowData(delta=delta)))

    def grant_receive_window(self, delta: int) -> None:
        """Increase the local receive window after application bytes are consumed."""

        _require_uint32("delta", delta)
        self._dispatch_stream_event(_GrantReceiveWindowEvent.with_data(StreamWindowData(delta=delta)))

    def local_fin(self) -> None:
        """Half-close local writes."""

        self._dispatch_stream_event(_LocalFinEvent.with_data(_StreamSignalData()))

    def remote_fin(self) -> None:
        """Half-close remote writes and publish EOF to the Python reader."""

        self._dispatch_stream_event(_RemoteFinEvent.with_data(_StreamSignalData()))

    def reset(self) -> None:
        """Hard-close the stream."""

        self._dispatch_stream_event(_ResetEvent.with_data(_StreamSignalData()))

    def _dispatch_stream_event(self, event: hsm.Event[typing.Any]) -> None:
        completion = self.dispatch(self.context(), event)
        if not isinstance(completion, _CompletedDispatch):
            raise RuntimeError("Yamux stream HSM dispatch did not expose completion.")
        if not completion.done():
            raise RuntimeError("Yamux stream HSM dispatch did not complete synchronously.")
        error = completion.exception()
        if error is not None:
            raise error
        failure = self._operation_failure
        self._operation_failure = None
        if isinstance(failure, _StreamOperationFailure):
            raise RuntimeError(failure.message)

    def _ensure_reader(self) -> asyncio.StreamReader:
        if self._reader is None:
            self._reader = asyncio.StreamReader(loop=asyncio.get_running_loop())
            if self._state in {StreamState.REMOTE_CLOSED, StreamState.CLOSED, StreamState.RESET}:
                self._reader.feed_eof()
        return self._reader

    def _feed_eof(self) -> None:
        if self._reader is not None and not self._reader.at_eof():
            self._reader.feed_eof()

    def _record_sent_payload(self, payload: bytes) -> None:
        self._send_window -= len(payload)
        self._sent_bytes += len(payload)

    def _record_received_payload(self, payload: bytes) -> None:
        if payload:
            self._ensure_reader().feed_data(payload)
        self._receive_window -= len(payload)
        self._received_bytes += len(payload)

    def _increase_send_window(self, delta: int) -> None:
        self._send_window += delta

    def _increase_receive_window(self, delta: int) -> None:
        self._receive_window += delta

    async def _record_window_consumed(self, byte_count: int) -> None:
        if byte_count <= 0 or self._on_window_consumed is None:
            return
        result = self._on_window_consumed(self._stream_id, byte_count)
        if result is not None:
            await result


def _require_uint32(name: str, value: int, *, minimum: int = 0) -> None:
    if not minimum <= value <= _MAX_UINT32:
        raise ValueError(f"{name} must be between {minimum} and {_MAX_UINT32}.")
