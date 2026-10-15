"""Typed events of a yamux session, its streams, and the connection that carries the session's bytes.

The session and its streams talk only by message. The connection (the socket's owner) feeds the session
``ReceiveXFrameEvent`` inputs and takes ``TransmitEvent`` frames, ``StreamAcceptedEvent`` and ``EndedEvent``
from it. A stream's owner sends it the events its codec encodes, ``CloseEvent``, ``ResetEvent`` and
``HandOverEvent``. Yamux carries bytes only: no event here names an encoding.
"""

import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from mosfet.protocols import transform
from mosfet.protocols.yamux.frame import INITIAL_STREAM_WINDOW


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

_OWNER_DESCRIPTION = (
    "Runtime-only address of the actor that owns the stream: every event decoded from the stream is dispatched "
    "to it. It is never read and never serialized."
)
_CODEC_DESCRIPTION = (
    "Runtime-only codec the stream applies between its length-framed messages and its owner's events. It is never "
    "serialized."
)
_FRAME_DESCRIPTION = "One whole Yamux frame in wire encoding: the 12-byte big-endian header and its payload."


class _FrozenModel(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)


class _RuntimeModel(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )


class StreamIdData(_FrozenModel):
    """Payload scoped to one Yamux stream."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1}]},
    )

    stream_id: StreamId


class OpenStreamData(_RuntimeModel):
    """Ask a Yamux session to open a stream from this side, owned by ``owner`` and named ``channel``.

    The session answers the owner: the new stream tells it ``protocol.transform.owned`` when the codec encodes
    anything, and a refused open is ``protocol.yamux.open_stream.failed`` carrying this request's ``id``.
    """

    initial_window_size: InitialWindowSize = pydantic.Field(
        default=INITIAL_STREAM_WINDOW,
        description=(
            "Receive window this side is prepared to grant to the peer. The default is Yamux's required initial "
            "256 KiB stream window."
        ),
        examples=[INITIAL_STREAM_WINDOW],
    )
    owner: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=_OWNER_DESCRIPTION,
    )
    codec: SkipJsonSchema[pydantic.SkipValidation[transform.Codec]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=_CODEC_DESCRIPTION,
    )
    channel: transform.ChannelName = pydantic.Field(
        description="Channel the stream carries; the owner is told it in protocol.transform.owned.",
    )


class OpenStreamFailedData(_FrozenModel):
    """The session refused an open; nothing was sent to the peer."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"kind": "session_draining"}]},
    )

    kind: typing.Literal["session_draining", "stream_rejected"] = pydantic.Field(
        description=(
            "Why: the session no longer opens streams (it is going away, errored or ended), or its stream ids are "
            "used up."
        ),
        examples=["session_draining", "stream_rejected"],
    )


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


class TransmitData(_FrozenModel):
    """A frame the session writes: the connection sends ``frame`` on its socket, in the order received."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"frame": "AAEAAQAAAAEAAAAA"}]},
    )

    frame: bytes = pydantic.Field(description=_FRAME_DESCRIPTION, examples=[b"\x00\x01\x00\x01\x00\x00\x00\x01"])


class StreamAcceptedData(_RuntimeModel):
    """The peer opened ``stream``; the connection owns it, with no codec, until it hands it over."""

    stream_id: StreamId
    stream: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=(
            "Runtime-only address of the new stream actor. Its bytes stay unacknowledged until the connection "
            "sends it protocol.yamux.stream.hand_over."
        ),
    )


class EndedData(_FrozenModel):
    """The session is over: the connection closes its socket after the frames it already received."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"reason": "drained"}]},
    )

    reason: typing.Literal["protocol_error", "ping_timeout", "drained"] = pydantic.Field(
        description=(
            "The peer broke the protocol (a GoAway with the protocol-error code was the last frame), a ping went "
            "unanswered, or a GoAway was exchanged and every stream finished."
        ),
        examples=["protocol_error", "ping_timeout", "drained"],
    )


class CloseData(_FrozenModel):
    """Half-close the stream: FIN after every message already sent; the owner sends nothing more."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{}]},
    )


class ResetData(_FrozenModel):
    """Hard-close the stream at once: RST, and every buffered byte either way is dropped."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"examples": [{}]},
    )


class HandOverData(_RuntimeModel):
    """The stream's current owner gives it to ``owner``, which decodes with ``codec`` from the next message on."""

    owner: SkipJsonSchema[pydantic.SkipValidation[hsm.Instance]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=_OWNER_DESCRIPTION,
    )
    codec: SkipJsonSchema[pydantic.SkipValidation[transform.Codec]] = pydantic.Field(
        exclude=True,
        repr=False,
        description=_CODEC_DESCRIPTION,
    )
    channel: transform.ChannelName | None = pydantic.Field(
        default=None,
        description=(
            "Channel the new owner is told it owns, or none while the stream is not yet named (a connection reading "
            "the stream's first message itself)."
        ),
    )


class FrameOutData(StreamIdData):
    """A frame one stream writes; its session transmits it while the session is connected."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1, "frame": "AAAAAAAAAAEAAAAA"}]},
    )

    frame: bytes = pydantic.Field(description=_FRAME_DESCRIPTION)


class ViolationData(StreamIdData):
    """The peer broke one stream's rules; the stream already reset itself and wrote RST."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={"examples": [{"stream_id": 1, "stage": "receive_data", "kind": "protocol_error"}]},
    )

    stage: SessionStage = pydantic.Field(description="Which input broke the rule.")
    kind: FailureKind = pydantic.Field(
        description="What was broken. A protocol_error ends the session with GoAway; other kinds end only the stream.",
    )


OpenStreamEvent = hsm.Event[OpenStreamData](name="protocol.yamux.open_stream", schema=OpenStreamData)
OpenStreamFailedEvent = hsm.Event[OpenStreamFailedData](
    name="protocol.yamux.open_stream.failed",
    schema=OpenStreamFailedData,
)
PingEvent = hsm.Event[PingData](name="protocol.yamux.ping", schema=PingData)
GoAwayEvent = hsm.Event[GoAwayData](name="protocol.yamux.go_away", schema=GoAwayData)
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
TransmitEvent = hsm.Event[TransmitData](name="protocol.yamux.transmit", schema=TransmitData)
StreamAcceptedEvent = hsm.Event[StreamAcceptedData](
    name="protocol.yamux.stream.accepted",
    schema=StreamAcceptedData,
)
EndedEvent = hsm.Event[EndedData](name="protocol.yamux.session.ended", schema=EndedData)
CloseEvent = hsm.Event[CloseData](name="protocol.yamux.stream.close", schema=CloseData)
ResetEvent = hsm.Event[ResetData](name="protocol.yamux.stream.reset", schema=ResetData)
HandOverEvent = hsm.Event[HandOverData](name="protocol.yamux.stream.hand_over", schema=HandOverData)
FrameOutEvent = hsm.Event[FrameOutData](name="protocol.yamux.frame.out", schema=FrameOutData)
StreamEndedEvent = hsm.Event[StreamIdData](name="protocol.yamux.stream.ended", schema=StreamIdData)
ViolationEvent = hsm.Event[ViolationData](name="protocol.yamux.stream.violation", schema=ViolationData)
