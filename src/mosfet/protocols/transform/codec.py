"""Plain codecs that turn one stream message into one typed event and one event back into a message.

A codec is a value object, not a state machine: the actor that already owns a stream's bytes applies it
inline, the way a transform stream sits in a pipe. Framing is the host's; a codec sees whole messages.
"""

import json
import typing

import hsm
import pydantic

type DecodeReason = typing.Literal["malformed", "unknown_event", "schema_invalid", "oversize", "media_type"]
type EncodeReason = typing.Literal["schema_invalid", "oversize"]


class DecodeError(ValueError):
    """One message could not become an event. ``reason`` is a closed label safe for telemetry."""

    reason: DecodeReason

    def __init__(self, reason: DecodeReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class EncodeError(ValueError):
    """One event could not become a message. ``reason`` is a closed label safe for telemetry."""

    reason: EncodeReason

    def __init__(self, reason: EncodeReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class Codec(typing.Protocol):
    """Pure conversion between one stream message and one typed event.

    ``name`` is a low-cardinality label fixed by the codec's code (such as ``"json"``), never by the wire.
    ``max_message_bytes`` bounds one message in either direction. ``encodes_any`` is constant: whether this
    codec encodes any event at all, so a host tells only owners that can send which stream to send to.
    """

    name: str
    max_message_bytes: int
    encodes_any: bool

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """Return the event ``message`` carries, unstamped; raise :class:`DecodeError` when it carries none."""
        ...

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """Whether this codec encodes events named like ``event``: a pure lookup by name."""
        ...

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """Return the message carrying ``event``; raise :class:`EncodeError` when it cannot be carried."""
        ...


def _refuse_constant(constant: str) -> typing.NoReturn:
    raise ValueError(f"JSON constant {constant} is not allowed.")


class JsonCodec:
    """A JSON codec for control and display streams.

    One message is UTF-8 JSON ``{"name": <event name>, "data": <payload>}``. Decoding maps a registered name
    to its event's Pydantic schema and validates ``data`` with JSON semantics; this is typed decoding, not
    policy, so an unregistered name is ``unknown_event`` and invalid data is ``schema_invalid``. The decoded
    event carries no ``id``, ``source`` or ``target``: the host that owns the stream stamps those.

    Raises:
        ValueError: no events, two events with one name, or a non-positive ``max_message_bytes``.
        TypeError: an event whose schema is not a Pydantic model class.
    """

    name: str
    max_message_bytes: int
    encodes_any: bool
    _events: dict[str, tuple[hsm.Event[typing.Any], type[pydantic.BaseModel]]]

    def __init__(self, *events: hsm.Event[typing.Any], max_message_bytes: int) -> None:
        if max_message_bytes < 1:
            raise ValueError(f"JsonCodec max_message_bytes must be positive, got {max_message_bytes}.")
        if not events:
            raise ValueError("JsonCodec needs at least one event to decode and encode.")
        registered: dict[str, tuple[hsm.Event[typing.Any], type[pydantic.BaseModel]]] = {}
        for event in events:
            schema = event.schema
            if not (isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)):
                raise TypeError(f"JsonCodec event {event.name!r} needs a Pydantic model class as its schema.")
            if event.name in registered:
                raise ValueError(f"JsonCodec registers event {event.name!r} more than once.")
            registered[event.name] = (event, schema)
        self.name = "json"
        self.max_message_bytes = max_message_bytes
        self.encodes_any = True
        self._events = registered

    def decode(self, message: bytes) -> hsm.Event[typing.Any]:
        """Return the registered event ``message`` names, carrying its validated data.

        Raises:
            DecodeError: ``oversize`` past ``max_message_bytes`` (checked before parsing), ``malformed`` when
                the message is not a UTF-8 JSON object with exactly ``name`` (text) and ``data``,
                ``unknown_event`` for an unregistered name, ``schema_invalid`` when the schema rejects ``data``.
        """

        if len(message) > self.max_message_bytes:
            raise DecodeError("oversize", f"Message exceeds {self.max_message_bytes} bytes.")
        try:
            envelope = typing.cast(object, json.loads(message.decode("utf-8"), parse_constant=_refuse_constant))
        except (ValueError, RecursionError) as error:
            raise DecodeError("malformed", "Message is not UTF-8 JSON.") from error
        if not isinstance(envelope, dict):
            raise DecodeError("malformed", "Message is not a JSON object.")
        fields = typing.cast(dict[str, object], envelope)
        name = fields.get("name")
        if fields.keys() != {"name", "data"} or not isinstance(name, str):
            raise DecodeError("malformed", 'Message must hold exactly "name" (text) and "data".')
        registered = self._events.get(name)
        if registered is None:
            raise DecodeError("unknown_event", "Message names an event this codec does not decode.")
        event, schema = registered
        try:
            data = schema.model_validate_json(json.dumps(fields["data"]))
        except RecursionError as error:
            raise DecodeError("malformed", "Message data nests too deeply.") from error
        except pydantic.ValidationError as error:
            if any(detail["type"] == "json_invalid" for detail in error.errors()):
                raise DecodeError("malformed", "Message data nests too deeply.") from error
            raise DecodeError("schema_invalid", f"Message data is not a valid {event.name!r} payload.") from error
        return event.with_data(data)

    def encodes(self, event: hsm.Event[typing.Any]) -> bool:
        """Whether ``event``'s name is registered."""

        return event.name in self._events

    def encode(self, event: hsm.Event[typing.Any]) -> bytes:
        """Return compact UTF-8 JSON ``{"name", "data"}`` for ``event``; runtime-only fields stay off the wire.

        Raises:
            EncodeError: ``schema_invalid`` for an unregistered name or data that is not the registered schema,
                ``oversize`` when the message would exceed ``max_message_bytes``.
        """

        registered = self._events.get(event.name)
        if registered is None:
            raise EncodeError("schema_invalid", f"Event {event.name!r} is not registered with this codec.")
        _, schema = registered
        if not isinstance(event.data, schema):
            raise EncodeError("schema_invalid", f"Event {event.name!r} data is not a {schema.__name__}.")
        try:
            message = json.dumps(
                {"name": event.name, "data": event.data.model_dump(mode="json")},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except ValueError as error:  # pydantic_core.PydanticSerializationError is a ValueError.
            raise EncodeError("schema_invalid", f"Event {event.name!r} data cannot be written as JSON.") from error
        if len(message) > self.max_message_bytes:
            raise EncodeError("oversize", f"Encoded message exceeds {self.max_message_bytes} bytes.")
        return message


__all__ = ["Codec", "DecodeError", "DecodeReason", "EncodeError", "EncodeReason", "JsonCodec"]
