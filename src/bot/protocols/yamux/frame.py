import dataclasses
import enum
import struct
import typing


VERSION = 0
HEADER_LENGTH = 12
INITIAL_STREAM_WINDOW = 256 * 1024
DEFAULT_MAX_DATA_LENGTH = 16 * 1024 * 1024
_MAX_UINT32 = (1 << 32) - 1
_SUPPORTED_FLAGS = 0x1 | 0x2 | 0x4 | 0x8


class FrameError(ValueError):
    """Base error for invalid Yamux frame data."""


class ProtocolError(FrameError):
    """FrameData violates the Yamux protocol contract."""


class FrameIncomplete(FrameError):
    """More bytes are required before a complete frame can be decoded."""


class FrameType(enum.IntEnum):
    """Yamux frame type values encoded in the 8-bit type field."""

    DATA = 0x0
    WINDOW_UPDATE = 0x1
    PING = 0x2
    GO_AWAY = 0x3


class Flag(enum.IntFlag):
    """Yamux flag bits encoded in the 16-bit flags field."""

    SYN = 0x1
    ACK = 0x2
    FIN = 0x4
    RST = 0x8


class GoAwayCode(enum.IntEnum):
    """Yamux GoAway error codes encoded in the length field."""

    NORMAL = 0x0
    PROTOCOL_ERROR = 0x1
    INTERNAL_ERROR = 0x2


@dataclasses.dataclass(frozen=True, slots=True)
class FrameData:
    """One Yamux frame with a 12-byte big-endian header and optional data payload."""

    frame_type: FrameType
    flags: Flag
    stream_id: int
    length: int
    payload: bytes = b""
    version: int = VERSION

    def __post_init__(self) -> None:
        # Declared contract requires enums; coerce helpers stay on the decode path
        # (decode_frame coerces wire ints before constructing). Rejecting ints here
        # keeps the frozen dataclass free of post-init mutation workarounds.
        if not isinstance(self.frame_type, FrameType):
            raise ProtocolError(f"Unsupported Yamux frame type {self.frame_type!r}.")
        if not isinstance(self.flags, Flag):
            raise ProtocolError(f"Unsupported Yamux flag bits {self.flags!r}.")
        frame_type = self.frame_type
        flags = self.flags
        _validate_uint8("version", self.version)
        _validate_uint32("stream_id", self.stream_id)
        _validate_uint32("length", self.length)
        if self.version != VERSION:
            raise ProtocolError(f"Unsupported Yamux version {self.version}.")
        if self.stream_id == 0 and frame_type in {FrameType.DATA, FrameType.WINDOW_UPDATE}:
            raise ProtocolError("Data and window update frames must address a nonzero stream ID.")
        if self.stream_id != 0 and frame_type in {FrameType.PING, FrameType.GO_AWAY}:
            raise ProtocolError("Ping and GoAway frames must use stream ID 0.")
        if frame_type is FrameType.DATA and self.length != len(self.payload):
            raise ProtocolError("Data frame length must match payload byte length.")
        if frame_type is not FrameType.DATA and self.payload:
            raise ProtocolError("Only data frames may carry bytes after the header.")
        if frame_type is FrameType.PING and flags not in {Flag.SYN, Flag.ACK}:
            raise ProtocolError("Ping frames must use exactly SYN for outbound or ACK for response.")
        if frame_type is FrameType.GO_AWAY:
            if flags:
                raise ProtocolError("GoAway frames must not set stream flags.")
            try:
                _ = GoAwayCode(self.length)
            except ValueError as error:
                raise ProtocolError(f"Unsupported GoAway code {self.length}.") from error

    @classmethod
    def data(
        cls,
        *,
        stream_id: int,
        payload: bytes = b"",
        flags: Flag | None = None,
    ) -> typing.Self:
        """Build a Yamux data frame whose length is the payload byte count."""

        return cls(
            frame_type=FrameType.DATA,
            flags=flags if flags is not None else Flag(0),
            stream_id=stream_id,
            length=len(payload),
            payload=payload,
        )

    @classmethod
    def window_update(
        cls,
        *,
        stream_id: int,
        delta: int,
        flags: Flag | None = None,
    ) -> typing.Self:
        """Build a Yamux window update frame whose length is the window delta."""

        return cls(
            frame_type=FrameType.WINDOW_UPDATE,
            flags=flags if flags is not None else Flag(0),
            stream_id=stream_id,
            length=delta,
        )

    @classmethod
    def ping(cls, *, opaque: int, ack: bool = False) -> typing.Self:
        """Build a Yamux ping frame using SYN for outbound and ACK for response."""

        return cls(
            frame_type=FrameType.PING,
            flags=Flag.ACK if ack else Flag.SYN,
            stream_id=0,
            length=opaque,
        )

    @classmethod
    def go_away(cls, *, code: GoAwayCode) -> typing.Self:
        """Build a Yamux GoAway frame with a spec-defined termination code."""

        return cls(frame_type=FrameType.GO_AWAY, flags=Flag(0), stream_id=0, length=int(code))

    @property
    def is_syn(self) -> bool:
        return bool(self.flags & Flag.SYN)

    @property
    def is_ack(self) -> bool:
        return bool(self.flags & Flag.ACK)

    @property
    def is_fin(self) -> bool:
        return bool(self.flags & Flag.FIN)

    @property
    def is_rst(self) -> bool:
        return bool(self.flags & Flag.RST)

    def to_bytes(self) -> bytes:
        """EncodeData this frame as a 12-byte big-endian Yamux header plus payload."""

        header = struct.pack(
            "!BBHII",
            self.version,
            int(self.frame_type),
            int(self.flags),
            self.stream_id,
            self.length,
        )
        return header + self.payload


def decode_frame(buffer: bytes, *, max_data_length: int = DEFAULT_MAX_DATA_LENGTH) -> tuple[FrameData, bytes]:
    """DecodeData the first complete Yamux frame from a byte buffer and return the frame plus unused bytes."""

    if max_data_length < 0:
        raise ValueError("max_data_length must be non-negative.")
    if len(buffer) < HEADER_LENGTH:
        raise FrameIncomplete("Yamux frame header is incomplete.")
    version, frame_type_value, flags_value, stream_id, length = typing.cast(
        tuple[int, int, int, int, int],
        struct.unpack("!BBHII", buffer[:HEADER_LENGTH]),
    )
    frame_type = _coerce_frame_type(frame_type_value)
    flags = _coerce_flags(flags_value)
    if frame_type is FrameType.DATA:
        if length > max_data_length:
            raise ProtocolError(f"Data frame length {length} exceeds configured maximum {max_data_length}.")
        end = HEADER_LENGTH + length
        if len(buffer) < end:
            raise FrameIncomplete("Yamux data frame payload is incomplete.")
        payload = buffer[HEADER_LENGTH:end]
        rest = buffer[end:]
    else:
        payload = b""
        rest = buffer[HEADER_LENGTH:]
    return (
        FrameData(
            version=version,
            frame_type=frame_type,
            flags=flags,
            stream_id=stream_id,
            length=length,
            payload=payload,
        ),
        rest,
    )


def _coerce_frame_type(value: int | FrameType) -> FrameType:
    try:
        return FrameType(value)
    except ValueError as error:
        raise ProtocolError(f"Unsupported Yamux frame type {value}.") from error


def _coerce_flags(value: int | Flag) -> Flag:
    flags = Flag(value)
    if int(flags) & ~_SUPPORTED_FLAGS:
        raise ProtocolError(f"Unsupported Yamux flag bits 0x{int(flags) & ~_SUPPORTED_FLAGS:x}.")
    return flags


def _validate_uint8(name: str, value: int) -> None:
    if not 0 <= value <= 0xFF:
        raise ProtocolError(f"{name} must fit in uint8.")


def _validate_uint32(name: str, value: int) -> None:
    if not 0 <= value <= _MAX_UINT32:
        raise ProtocolError(f"{name} must fit in uint32.")
