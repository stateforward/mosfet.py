"""Tests for the plain transform codecs: one stream message to one typed event and back."""

import datetime
import json
import typing

import hsm
import pydantic
import pytest

from mosfet.protocols import transform


class _KeyData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )

    key: str = pydantic.Field(min_length=1)
    pressed_at: datetime.datetime
    chord: tuple[str, ...] = ()
    owner: pydantic.SkipValidation[hsm.Instance] | None = pydantic.Field(default=None, exclude=True)


class _LayoutData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    columns: int = pydantic.Field(ge=1)


_KeyEvent = hsm.Event[_KeyData](name="test.transform.key", schema=_KeyData)
_LayoutEvent = hsm.Event[_LayoutData](name="test.transform.layout", kind=hsm.CompletionEventKind, schema=_LayoutData)
_UnregisteredEvent = hsm.Event[_LayoutData](name="test.transform.unregistered", schema=_LayoutData)
_PRESSED_AT = datetime.datetime(2026, 10, 14, 12, 30, tzinfo=datetime.UTC)


def _codec(max_message_bytes: int = 4096) -> transform.JsonCodec:
    return transform.JsonCodec(_KeyEvent, _LayoutEvent, max_message_bytes=max_message_bytes)


def _decode_reason(codec: transform.Codec, message: bytes) -> str:
    with pytest.raises(transform.DecodeError) as raised:
        _ = codec.decode(message)
    return raised.value.reason


def _encode_reason(codec: transform.Codec, event: hsm.Event[typing.Any]) -> str:
    with pytest.raises(transform.EncodeError) as raised:
        _ = codec.encode(event)
    return raised.value.reason


def test_json_codec_is_a_codec_named_json_that_encodes_its_events() -> None:
    codec: transform.Codec = _codec(max_message_bytes=128)

    assert codec.name == "json"
    assert codec.max_message_bytes == 128
    assert codec.encodes_any is True
    assert codec.encodes(_KeyEvent.with_data(_KeyData(key="a", pressed_at=_PRESSED_AT)))
    assert codec.encodes(_LayoutEvent)
    assert not codec.encodes(_UnregisteredEvent.with_data(_LayoutData(columns=1)))


def test_json_codec_round_trips_an_event_in_json_mode_and_leaves_stamping_to_the_host() -> None:
    codec = _codec()
    owner = hsm.Instance()
    sent = _KeyEvent.with_data_and_id(
        _KeyData(key="a", pressed_at=_PRESSED_AT, chord=("ctrl", "shift"), owner=owner), "local-id"
    )

    message = codec.encode(sent)
    received = codec.decode(message)

    assert json.loads(message) == {
        "name": "test.transform.key",
        "data": {"key": "a", "pressed_at": "2026-10-14T12:30:00Z", "chord": ["ctrl", "shift"]},
    }
    assert received.name == _KeyEvent.name
    assert received.kind == _KeyEvent.kind
    assert received.schema is _KeyData
    assert received.data == _KeyData(key="a", pressed_at=_PRESSED_AT, chord=("ctrl", "shift"))
    assert (received.id, received.source, received.target, received.metadata) == ("", "", "", {})


def test_json_codec_keeps_the_registered_event_kind() -> None:
    codec = _codec()

    received = codec.decode(codec.encode(_LayoutEvent.with_data(_LayoutData(columns=3))))

    assert received.kind == hsm.CompletionEventKind
    assert received.data == _LayoutData(columns=3)


def test_json_codec_writes_compact_utf8() -> None:
    message = _codec().encode(_KeyEvent.with_data(_KeyData(key="é", pressed_at=_PRESSED_AT)))

    assert b" " not in message
    assert "é".encode() in message


@pytest.mark.parametrize(
    "message",
    [
        b"\xff\xfe",
        '{"name": "test.transform.layout", "data": {"columns": 1}}'.encode("utf-16"),
        b"not json",
        b"[]",
        b'"text"',
        b'{"name": "test.transform.layout"}',
        b'{"data": {"columns": 1}}',
        b'{"name": 7, "data": {"columns": 1}}',
        b'{"name": "test.transform.layout", "data": {"columns": 1}, "extra": true}',
        b'{"name": "test.transform.layout", "data": {"columns": NaN}}',
        b'{"name": "test.transform.layout", "data": ' + b"[" * 1000 + b"]" * 1000 + b"}",
        b'{"name": "test.transform.layout", "data": ' + b"[" * 30000 + b"]" * 30000 + b"}",
    ],
    ids=[
        "not-utf8",
        "utf16",
        "not-json",
        "array",
        "string",
        "no-data",
        "no-name",
        "name-not-text",
        "extra-key",
        "nan-constant",
        "nesting-past-the-schema-parser",
        "nesting-past-the-envelope-parser",
    ],
)
def test_json_codec_refuses_a_message_that_is_not_an_event_envelope_as_malformed(message: bytes) -> None:
    assert _decode_reason(_codec(max_message_bytes=64 * 1024), message) == "malformed"


def test_json_codec_refuses_an_unregistered_name_as_unknown_event() -> None:
    message = b'{"name": "test.transform.unregistered", "data": {"columns": 1}}'

    assert _decode_reason(_codec(), message) == "unknown_event"


@pytest.mark.parametrize(
    "data",
    [
        {"columns": 0},
        {"columns": "three"},
        None,
    ],
    ids=["out-of-range", "wrong-type", "null"],
)
def test_json_codec_refuses_data_its_schema_rejects_as_schema_invalid(data: object) -> None:
    message = json.dumps({"name": "test.transform.layout", "data": data}).encode()

    assert _decode_reason(_codec(), message) == "schema_invalid"


def test_json_codec_validates_data_with_json_semantics() -> None:
    codec = _codec()
    message = json.dumps(
        {"name": "test.transform.key", "data": {"key": "a", "pressed_at": "2026-10-14T12:30:00Z"}}
    ).encode()

    received = codec.decode(message)

    assert received.data == _KeyData(key="a", pressed_at=_PRESSED_AT)


def test_json_codec_refuses_an_oversize_message_before_reading_it() -> None:
    codec = _codec(max_message_bytes=32)
    valid_but_long = json.dumps({"name": "test.transform.layout", "data": {"columns": 1}}).encode()

    assert len(valid_but_long) > 32
    assert _decode_reason(codec, valid_but_long) == "oversize"
    assert _decode_reason(codec, b"x" * 33) == "oversize"


def test_json_codec_refuses_to_encode_what_it_did_not_register_or_cannot_fit() -> None:
    codec = _codec(max_message_bytes=48)

    assert _encode_reason(codec, _UnregisteredEvent.with_data(_LayoutData(columns=1))) == "schema_invalid"
    assert _encode_reason(codec, _LayoutEvent) == "schema_invalid"
    assert _encode_reason(codec, _LayoutEvent.with_data(_KeyData(key="a", pressed_at=_PRESSED_AT))) == "schema_invalid"
    assert _encode_reason(codec, _KeyEvent.with_data(_KeyData(key="a" * 64, pressed_at=_PRESSED_AT))) == "oversize"


def test_codec_errors_are_value_errors_that_name_a_closed_reason() -> None:
    decode = transform.DecodeError("malformed", "Message is not UTF-8 JSON.")
    encode = transform.EncodeError("oversize", "Encoded message exceeds 64 bytes.")

    assert isinstance(decode, ValueError) and decode.reason == "malformed"
    assert isinstance(encode, ValueError) and encode.reason == "oversize"
    assert str(decode) == "Message is not UTF-8 JSON."


def test_json_codec_construction_rejects_ambiguous_or_unusable_registrations() -> None:
    same_name = hsm.Event[_LayoutData](name=_LayoutEvent.name, schema=_LayoutData)
    schemaless = hsm.Event[None](name="test.transform.bare")
    dict_schema = hsm.Event[dict[str, object]](name="test.transform.dict", schema={"type": "object"})

    with pytest.raises(ValueError, match="test.transform.layout"):
        _ = transform.JsonCodec(_LayoutEvent, same_name, max_message_bytes=64)
    with pytest.raises(TypeError, match="test.transform.bare"):
        _ = transform.JsonCodec(schemaless, max_message_bytes=64)
    with pytest.raises(TypeError, match="test.transform.dict"):
        _ = transform.JsonCodec(dict_schema, max_message_bytes=64)
    with pytest.raises(ValueError, match="at least one event"):
        _ = transform.JsonCodec(max_message_bytes=64)
    with pytest.raises(ValueError, match="max_message_bytes"):
        _ = transform.JsonCodec(_LayoutEvent, max_message_bytes=0)


def _refuse_to_serialize(value: int) -> int:
    raise ValueError(f"{value} cannot be written.")


class _UnwritableData(pydantic.BaseModel):
    value: typing.Annotated[int, pydantic.PlainSerializer(_refuse_to_serialize)]


def test_json_codec_refuses_to_encode_data_that_cannot_be_written_as_json() -> None:
    unwritable = hsm.Event[_UnwritableData](name="test.transform.unwritable", schema=_UnwritableData)
    codec = transform.JsonCodec(unwritable, max_message_bytes=64)

    assert _encode_reason(codec, unwritable.with_data(_UnwritableData(value=1))) == "schema_invalid"
