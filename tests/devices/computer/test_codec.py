from mosfet.devices import computer

import pytest

from mosfet import media
from mosfet.protocols import transform

PNG = media.Image(content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, media_type="image/png")


def _message(media_type: bytes, content: bytes) -> bytes:
    return bytes([len(media_type)]) + media_type + content


def _decode_reason(message: bytes) -> str:
    with pytest.raises(transform.DecodeError) as raised:
        _ = computer.StillCodec().decode(message)
    return raised.value.reason


def test_still_codec_is_an_inbound_only_binary_codec() -> None:
    codec: transform.Codec = computer.StillCodec()

    assert codec.name == "still"
    assert codec.encodes_any is False
    assert codec.max_message_bytes == 1 + len("image/jpeg") + media.MAX_IMAGE_BYTES
    assert not codec.encodes(computer.StillEvent.with_data(computer.StillData(image=PNG)))
    with pytest.raises(transform.EncodeError) as raised:
        _ = codec.encode(computer.StillEvent.with_data(computer.StillData(image=PNG)))
    assert raised.value.reason == "schema_invalid"


def test_still_codec_decodes_one_still_into_a_validated_image() -> None:
    event = computer.StillCodec().decode(_message(b"image/png", PNG.content))

    assert event.name == computer.StillEvent.name
    assert event.data == computer.StillData(image=PNG)
    assert (event.id, event.source, event.target) == ("", "", "")


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (b"", "malformed"),
        (b"\x00" + PNG.content, "malformed"),
        (b"\x20image/png", "malformed"),
        (_message("imäge/png".encode(), PNG.content), "malformed"),
        (_message(b"image/gif", b"GIF89a"), "media_type"),
        (_message(b"image/jpeg", PNG.content), "media_type"),
        (_message(b"image/png", b""), "schema_invalid"),
        (_message(b"image/png", PNG.content + b"\x00" * media.MAX_IMAGE_BYTES), "oversize"),
    ],
)
def test_still_codec_refuses_what_is_not_one_accepted_image(message: bytes, reason: str) -> None:
    assert _decode_reason(message) == reason
