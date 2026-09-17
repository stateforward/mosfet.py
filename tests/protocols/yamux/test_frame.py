import struct

from mosfet.protocols.yamux.frame import (
    HEADER_LENGTH,
    VERSION,
    Flag,
    FrameData,
    FrameIncomplete,
    FrameType,
    GoAwayCode,
    ProtocolError,
    decode_frame,
)


def test_yamux_frame_encodes_data_header_big_endian() -> None:
    frame = FrameData.data(stream_id=1, payload=b"hello", flags=Flag.SYN)

    encoded = frame.to_bytes()

    assert encoded == b"\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00\x00\x05hello"
    decoded, rest = decode_frame(encoded + b"next")
    assert decoded == frame
    assert rest == b"next"
    assert HEADER_LENGTH == 12
    assert VERSION == 0


def test_yamux_frame_encodes_window_ping_and_goaway_length_meanings() -> None:
    assert FrameData.window_update(stream_id=3, delta=32768, flags=Flag.ACK).to_bytes() == (
        b"\x00\x01\x00\x02\x00\x00\x00\x03\x00\x00\x80\x00"
    )
    assert FrameData.ping(opaque=42).to_bytes() == b"\x00\x02\x00\x01\x00\x00\x00\x00\x00\x00\x00\x2a"
    assert FrameData.ping(opaque=42, ack=True).to_bytes() == b"\x00\x02\x00\x02\x00\x00\x00\x00\x00\x00\x00\x2a"
    assert FrameData.go_away(code=GoAwayCode.PROTOCOL_ERROR).to_bytes() == (
        b"\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01"
    )


def test_yamux_frame_rejects_incomplete_and_oversized_data() -> None:
    try:
        _ = decode_frame(b"\x00")
    except FrameIncomplete:
        pass
    else:
        raise AssertionError("Short Yamux headers must be incomplete.")

    encoded = struct.pack("!BBHII", 0, FrameType.DATA, 0, 1, 5) + b"abc"
    try:
        _ = decode_frame(encoded)
    except FrameIncomplete:
        pass
    else:
        raise AssertionError("Data frames must include the declared payload bytes.")

    encoded = struct.pack("!BBHII", 0, FrameType.DATA, 0, 1, 6)
    try:
        _ = decode_frame(encoded, max_data_length=5)
    except ProtocolError:
        pass
    else:
        raise AssertionError("Data frame length must respect configured maximums.")


def test_yamux_frame_rejects_protocol_violations() -> None:
    invalid_headers = [
        struct.pack("!BBHII", 1, FrameType.DATA, 0, 1, 0),
        struct.pack("!BBHII", 0, 99, 0, 1, 0),
        struct.pack("!BBHII", 0, FrameType.DATA, 0x10, 1, 0),
        struct.pack("!BBHII", 0, FrameType.DATA, 0, 0, 0),
        struct.pack("!BBHII", 0, FrameType.PING, Flag.SYN, 1, 0),
        struct.pack("!BBHII", 0, FrameType.GO_AWAY, 0, 0, 99),
    ]

    for header in invalid_headers:
        try:
            _ = decode_frame(header)
        except ProtocolError:
            pass
        else:
            raise AssertionError(f"Invalid Yamux header was accepted: {header!r}")


def test_yamux_frame_rejects_invalid_ping_and_goaway_flags() -> None:
    for flags in (Flag(0), Flag.SYN | Flag.ACK, Flag.FIN):
        try:
            _ = FrameData(frame_type=FrameType.PING, flags=flags, stream_id=0, length=1)
        except ProtocolError:
            pass
        else:
            raise AssertionError(f"Invalid ping flags were accepted: {flags!r}")

    try:
        _ = FrameData(frame_type=FrameType.GO_AWAY, flags=Flag.SYN, stream_id=0, length=0)
    except ProtocolError:
        pass
    else:
        raise AssertionError("GoAway frames must reject flags.")
