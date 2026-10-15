from mosfet.devices import audio

import struct
import typing

import hsm
import pytest

from mosfet.protocols import transform

PCM = b"\x01\x00\xff\x7f"


def _decode_reason(codec: transform.Codec, message: bytes) -> str:
    with pytest.raises(transform.DecodeError) as raised:
        _ = codec.decode(message)
    return raised.value.reason


def _encode_reason(codec: transform.Codec, event: hsm.Event[typing.Any]) -> str:
    with pytest.raises(transform.EncodeError) as raised:
        _ = codec.encode(event)
    return raised.value.reason


def _output(**fields: object) -> hsm.Event[typing.Any]:
    values: dict[str, object] = {"audio": PCM, "media_type": "audio/pcm", "sample_rate_hz": 48_000, "channels": 2}
    values.update(fields)
    return audio.OutputEvent.with_data(audio.OutputData.model_validate(values))


def test_pcm_codec_is_a_binary_codec_that_encodes_speaker_output_only() -> None:
    codec: transform.Codec = audio.PcmCodec()

    assert (codec.name, codec.max_message_bytes, codec.encodes_any) == ("pcm", 64 * 1024, True)
    assert codec.encodes(_output())
    assert not codec.encodes(audio.InputEvent)


def test_pcm_codec_decodes_the_rate_and_channels_header_into_microphone_input() -> None:
    event = audio.PcmCodec().decode(struct.pack("!IH", 16_000, 1) + PCM)

    assert event.name == audio.InputEvent.name
    assert event.data == audio.InputData(audio=PCM, media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    assert (event.id, event.source, event.target) == ("", "", "")


def test_pcm_codec_encodes_output_as_header_then_pcm() -> None:
    assert audio.PcmCodec().encode(_output()) == struct.pack("!IH", 48_000, 2) + PCM


def test_pcm_codec_never_puts_a_playout_mark_on_the_wire() -> None:
    assert audio.PcmCodec().encode(_output(mark="u1.1")) == audio.PcmCodec().encode(_output())


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (b"\x00\x00", "malformed"),
        (struct.pack("!IH", 16_000, 1), "malformed"),
        (struct.pack("!IH", 16_000, 1) + b"\x01\x02\x03", "malformed"),
        (struct.pack("!IH", 7_999, 1) + PCM, "malformed"),
        (struct.pack("!IH", 192_001, 1) + PCM, "malformed"),
        (struct.pack("!IH", 16_000, 0) + PCM, "malformed"),
        (struct.pack("!IH", 16_000, 9) + PCM, "malformed"),
        (struct.pack("!IH", 16_000, 1) + b"\x00" * 64, "oversize"),
    ],
)
def test_pcm_codec_refuses_what_is_not_one_bounded_pcm_chunk(message: bytes, reason: str) -> None:
    assert _decode_reason(audio.PcmCodec(max_message_bytes=32), message) == reason


@pytest.mark.parametrize(
    "event",
    [
        audio.InputEvent.with_data(audio.InputData(audio=PCM)),
        _output(media_type="audio/opus"),
        _output(sample_rate_hz=None),
        _output(channels=None),
        _output(channels=9),
        _output(sample_rate_hz=4_000),
        _output(audio=b"\x01\x02\x03"),
    ],
)
def test_pcm_codec_refuses_to_encode_what_is_not_pcm_speaker_output(event: hsm.Event[typing.Any]) -> None:
    assert _encode_reason(audio.PcmCodec(), event) == "schema_invalid"


def test_pcm_codec_refuses_to_encode_past_its_bound_and_to_be_built_too_small() -> None:
    assert _encode_reason(audio.PcmCodec(max_message_bytes=8), _output(audio=PCM * 2)) == "oversize"
    with pytest.raises(ValueError, match="header"):
        _ = audio.PcmCodec(max_message_bytes=7)
