import typing

import hsm
import pydantic

from bot.protocols.yamux.frame import INITIAL_STREAM_WINDOW


StreamId = typing.Annotated[
    int,
    pydantic.Field(
        ge=1,
        le=(1 << 32) - 1,
        description="Nonzero Yamux stream identifier. Stream ID 0 is reserved for session frames.",
        examples=[1, 2],
    ),
]
Uint32 = typing.Annotated[
    int,
    pydantic.Field(
        ge=0,
        le=(1 << 32) - 1,
        description="Unsigned 32-bit integer encoded in a Yamux length or stream identifier field.",
        examples=[0, 262144],
    ),
]
PositiveUint32 = typing.Annotated[
    int,
    pydantic.Field(
        ge=1,
        le=(1 << 32) - 1,
        description="Positive unsigned 32-bit integer used for Yamux byte counts and window deltas.",
        examples=[1, 4096],
    ),
]
InitialWindowSize = typing.Annotated[
    int,
    pydantic.Field(
        ge=INITIAL_STREAM_WINDOW,
        le=(1 << 32) - 1,
        description="Initial Yamux receive window. Yamux peers assume at least the required 256 KiB window.",
        examples=[INITIAL_STREAM_WINDOW],
    ),
]
FrameFlags = typing.Annotated[
    int,
    pydantic.Field(
        ge=0,
        le=0xF,
        description="Yamux flag bitset using SYN=1, ACK=2, FIN=4, and RST=8.",
        examples=[1, 2, 4, 8],
    ),
]
FailureKind = typing.Literal[
    "protocol_error",
    "stream_reset",
    "stream_rejected",
    "flow_control_exhausted",
    "receive_window_exhausted",
    "flow_control_overflow",
    "session_draining",
    "ping_timeout",
    "unknown_stream",
]
SessionStage = typing.Literal[
    "open_stream",
    "send_data",
    "close_stream",
    "reset_stream",
    "receive_data",
    "receive_window_update",
    "ping",
    "go_away",
]


class _FrozenModel(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


class StreamIdData(_FrozenModel):
    """Payload scoped to one Yamux stream."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1}]},
    )

    stream_id: StreamId


class OpenStreamData(_FrozenModel):
    """Command asking a Yamux session to open a locally initiated stream."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"initial_window_size": INITIAL_STREAM_WINDOW}]},
    )

    initial_window_size: InitialWindowSize = pydantic.Field(
        default=INITIAL_STREAM_WINDOW,
        description=(
            "Receive window this side is prepared to grant to the peer. The default is Yamux's required initial "
            "256 KiB stream window."
        ),
        examples=[INITIAL_STREAM_WINDOW],
    )


class SendData(StreamIdData):
    """Command asking a Yamux session to send bytes on a stream."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1, "payload": "aGVsbG8=", "end_stream": False}]},
    )

    payload: bytes = pydantic.Field(
        description="Data bytes to frame on the selected Yamux stream.",
        examples=[b"hello"],
    )
    end_stream: bool = pydantic.Field(
        default=False,
        description="When true, send FIN with this data frame to half-close local writes after the payload.",
    )


class CloseStreamData(StreamIdData):
    """Command asking a Yamux session to send FIN for one stream."""


class ResetStreamData(StreamIdData):
    """Command asking a Yamux session to hard-reset one stream."""


class PingData(_FrozenModel):
    """Command asking a Yamux session to send or track a ping."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"opaque": 42}]},
    )

    opaque: Uint32 = pydantic.Field(
        description="Opaque 32-bit value carried in the Yamux ping length field and echoed by the ACK.",
        examples=[42],
    )


class GoAwayData(_FrozenModel):
    """Command or observation that a Yamux session is terminating."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"code": 0}]},
    )

    code: typing.Literal[0, 1, 2] = pydantic.Field(
        description="Yamux GoAway code: 0 normal termination, 1 protocol error, or 2 internal error.",
        examples=[0, 1, 2],
    )


class ReceiveDataFrameData(StreamIdData):
    """Incoming Yamux data frame payload after byte-level validation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1, "flags": 1, "payload": "aGVsbG8="}]},
    )

    flags: FrameFlags
    payload: bytes = pydantic.Field(
        description="Data bytes following the Yamux header. Empty data frames are valid when flags carry meaning.",
        examples=[b"hello"],
    )


class ReceiveWindowUpdateFrameData(StreamIdData):
    """Incoming Yamux window update frame after byte-level validation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1, "flags": 2, "delta": 32768}]},
    )

    flags: FrameFlags
    delta: Uint32 = pydantic.Field(
        description="Receive-window delta carried in the Yamux length field.",
        examples=[32768],
    )


class ReceivePingFrameData(PingData):
    """Incoming Yamux ping frame after byte-level validation."""

    flags: typing.Literal[1, 2] = pydantic.Field(
        description="Yamux ping direction flag: SYN=1 for outbound ping, ACK=2 for response.",
        examples=[1, 2],
    )


class ReceiveGoAwayFrameData(GoAwayData):
    """Incoming Yamux GoAway frame after byte-level validation."""


class SessionDrainedData(_FrozenModel):
    """Private completion event payload emitted after a draining session has no active streams."""

    code: typing.Literal[0, 1, 2] | None = pydantic.Field(
        default=None,
        description="GoAway code associated with the drain, when known.",
        examples=[0],
    )


OpenStreamEvent = hsm.Event[OpenStreamData](
    name="protocol.yamux.open_stream",
    schema=OpenStreamData,
)
SendDataEvent = hsm.Event[SendData](
    name="protocol.yamux.send_data",
    schema=SendData,
)
CloseStreamEvent = hsm.Event[CloseStreamData](
    name="protocol.yamux.close_stream",
    schema=CloseStreamData,
)
ResetStreamEvent = hsm.Event[ResetStreamData](
    name="protocol.yamux.reset_stream",
    schema=ResetStreamData,
)
PingEvent = hsm.Event[PingData](
    name="protocol.yamux.ping",
    schema=PingData,
)
GoAwayEvent = hsm.Event[GoAwayData](
    name="protocol.yamux.go_away",
    schema=GoAwayData,
)
ReceiveDataFrameEvent = hsm.Event[ReceiveDataFrameData](
    name="protocol.yamux.frame.data.received",
    schema=ReceiveDataFrameData,
)
ReceiveWindowUpdateFrameEvent = hsm.Event[ReceiveWindowUpdateFrameData](
    name="protocol.yamux.frame.window_update.received",
    schema=ReceiveWindowUpdateFrameData,
)
ReceivePingFrameEvent = hsm.Event[ReceivePingFrameData](
    name="protocol.yamux.frame.ping.received",
    schema=ReceivePingFrameData,
)
ReceiveGoAwayFrameEvent = hsm.Event[ReceiveGoAwayFrameData](
    name="protocol.yamux.frame.go_away.received",
    schema=ReceiveGoAwayFrameData,
)
SessionDrainedEvent = hsm.Event[SessionDrainedData](
    name="protocol.yamux.session.drained",
    kind=hsm.CompletionEventKind,
    schema=SessionDrainedData,
)
