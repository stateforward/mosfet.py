"""Utilities for typed HSM event payload contracts."""

import base64
import collections.abc
import dataclasses
import enum
import re
import typing
import xml.etree.ElementTree as ElementTree

import hsm
import pydantic

# Marks an event as offerable to a model as a tool. Derived from ``hsm.EventKind``, so
# ``hsm.kind.Is(EventKind, hsm.EventKind)`` still holds and HSM treats it as any other event.
#
# This is a stateforward.bot concern, not an HSM one: the owning domain is ``abilities.processing``,
# which builds each turn's tool menu from live topology and refuses selections the target never
# offered. It is defined here rather than there because ``bot.events``, ``bot.behavior.events``, and
# ``abilities.speaking`` all stamp events and all cycle on importing ``processing``; this module is a
# leaf. Import it as ``processing.EventKind`` wherever that does not cycle.
#
# Do NOT reuse ``hsm.CallEventKind`` for this. That kind means "an invocation of a declared
# ``hsm.Operation``" and is what HSM mints for one; conflating the two left no way to express a real
# call event that is not a tool, or a tool that is not an operation invocation.
EventKind = hsm.kind.Make(hsm.EventKind)

JsonSchema: typing.TypeAlias = dict[str, object]
_LOCAL_DEFS_REF_PREFIX = "#/$defs/"
_SUPPORTED_JSON_SCHEMA_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
_SUPPORTED_JSON_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "default",
        "description",
        "enum",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "multipleOf",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "not",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
    }
)
_STRING_SCHEMA_KEYWORDS = frozenset({"maxLength", "minLength", "pattern"})
_NUMBER_SCHEMA_KEYWORDS = frozenset({"exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum"})
_ARRAY_SCHEMA_KEYWORDS = frozenset({"items", "maxItems", "minItems"})
_OBJECT_SCHEMA_KEYWORDS = frozenset(
    {"additionalProperties", "maxProperties", "minProperties", "properties", "required"}
)


def event_schema_json_schema(schema: object | None) -> JsonSchema:
    """Return the JSON schema described by an event schema contract."""

    if schema is None:
        return {}
    if isinstance(schema, dict):
        return dict(typing.cast(dict[str, object], schema))
    if isinstance(schema, bool):
        return {}
    json_schema = typing.cast(dict[str, object], _event_schema_adapter(schema).json_schema())
    return _inline_root_ref(dict(json_schema))


def event_json_schema(event: hsm.Event[typing.Any]) -> JsonSchema:
    """Return the JSON schema for an event's payload contract."""

    return event_schema_json_schema(event.schema)


def validate_event_data(event: hsm.Event[typing.Any], data: object) -> object:
    """Validate and coerce data through an event's typed payload contract."""

    return validate_event_schema_data(event.schema, data)


def validate_event_schema_data(schema: object | None, data: object) -> object:
    """Validate and coerce data through a typed event schema contract."""

    if schema is None or isinstance(schema, dict | bool):
        return data
    return _event_schema_adapter(schema).validate_python(data)


def project_json_value(value: object) -> object:
    """Project a value into a JSON-compatible tree for Starlark / selection envelopes.

    Contract (agnostic for every event payload):

    - ``bytes`` / ``bytearray`` become **URL-safe base64 ASCII text** — the same wire form
      pydantic uses for ``ser_json_bytes="base64"`` (``-``/``_``, not ``+``/``/``).
    - Never ``str(bytes)`` (``"b'\\x00…'"``), which cannot rehydrate.
    - Pydantic models use ``model_dump(mode="json")`` so their own byte fields match.
    - Typed rehydration is the receiving event schema's job (``bytes_from_base64`` /
      ``val_json_bytes`` / model validators), not this projector.
    """

    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return project_json_value(dataclasses.asdict(value))
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): project_json_value(item) for key, item in mapping.items()}
    if isinstance(value, tuple | list):
        sequence = typing.cast(collections.abc.Sequence[object], value)
        return [project_json_value(item) for item in sequence]
    if isinstance(value, memoryview):
        return base64.urlsafe_b64encode(value.tobytes()).decode("ascii")
    if isinstance(value, (bytes, bytearray)):
        return base64.urlsafe_b64encode(bytes(value)).decode("ascii")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def model_facing_json_value(value: object) -> object:
    """Project a value for a model prompt, with raw media replaced by a descriptor.

    ``project_json_value`` is the *wire* projection: it keeps media as base64 so a peer can
    rehydrate it. A model cannot rehydrate anything. Handing it base64 spends the context window
    on a waveform it can only describe back ("your audio appears to be garbled"), so this
    projection is the one every model-facing surface uses instead.

    Contract:

    - ``bytes`` / ``bytearray`` / ``memoryview`` become ``{"media": "bytes", "bytes": <length>}``.
      The descriptor is non-reversible on purpose; it says how much media there was, never what
      it contained.
    - Everything the model can actually reason about survives: sibling scalars such as
      ``content_type``, ``sample_rate_hz``, ``channels``, and voice spans are ordinary fields and
      stay, so "1.2s of undecoded audio/pcm" is still legible without the samples.
    - Pydantic models are dumped in ``python`` mode and re-projected here, so nested media is
      described rather than base64-encoded by the model's own ``ser_json_bytes`` config.
    """

    if isinstance(value, bytes | bytearray):
        return {"media": "bytes", "bytes": len(value)}
    if isinstance(value, memoryview):
        return {"media": "bytes", "bytes": value.nbytes}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, enum.Enum):
        return model_facing_json_value(value.value)
    if isinstance(value, hsm.Event):
        event = typing.cast(hsm.Event[object], value)
        return {
            "name": event.name,
            "data": model_facing_json_value(event.data),
            "kind": model_facing_json_value(event.kind),
            "id": event.id or "",
            "source": event.source or "",
            "target": event.target or "",
            "metadata": model_facing_json_value(dict(event.metadata)) if event.metadata else {},
        }
    if isinstance(value, pydantic.BaseModel):
        return model_facing_json_value(value.model_dump(mode="python"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return model_facing_json_value(dataclasses.asdict(value))
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): model_facing_json_value(item) for key, item in mapping.items()}
    if isinstance(value, collections.abc.Set):
        members = typing.cast(collections.abc.Set[object], value)
        # Sets have no inherent order; sort the projected members so one stimulus always
        # renders one prompt.
        return sorted((model_facing_json_value(item) for item in members), key=repr)
    if isinstance(value, collections.abc.Sequence):
        sequence = typing.cast(collections.abc.Sequence[object], value)
        return [model_facing_json_value(item) for item in sequence]
    return str(value)


# Source-tree layer packages. These name where a domain *lives*, never the domain itself, so a
# tag derived from a payload type's module skips them (``bot.abilities.listening…`` is the
# ``listening`` domain, not the ``abilities`` one). Kept here rather than as a per-payload tag
# table: nothing registers a name, the module path already carries it.
_LAYER_PACKAGES: typing.Final = frozenset({"abilities", "devices", "providers"})
_MEDIA_DESCRIPTOR_PREFIX: typing.Final = "bytes:"
_TAG_NAMESPACE_PREFIX: typing.Final = "urn:stateforward.bot:"
# Envelope attributes live under their own prefix so they can never collide with a payload field.
_STIMULUS_PREFIX: typing.Final = "stimulus"


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _payload_tag(payload_type: type) -> str:
    """Derive one XML tag from a payload type: ``<domain>:<local>``.

    Nothing is registered — both halves come from what the type already carries:

    - **domain** is the first module segment after the ``bot`` root and any layer package, so
      ``bot.environment.events`` → ``environment`` and ``bot.abilities.listening.interpretation``
      → ``listening``.
    - **local** is the class name minus its ``Data`` suffix, snake-cased, minus a leading
      repetition of the domain (``PhoneSoundData`` in ``bot.devices.phone`` → ``sound``).

    The result matches the owning event name where one exists (``SoundData`` → ``environment:sound``
    for ``environment.sound``) without reading the event: ancestry levels and contained payloads
    have no event of their own, so the type is the only source that works everywhere.
    """

    segments = [segment for segment in (getattr(payload_type, "__module__", "") or "").split(".") if segment]
    if segments and segments[0] == "bot":
        segments = segments[1:]
        while segments and segments[0] in _LAYER_PACKAGES:
            segments = segments[1:]
    else:
        # Not a domain payload at all (``bytes``, ``str``, a test-local model): there is no domain
        # to qualify it with, so the local name stands alone rather than inventing one.
        segments = []
    domain = segments[0] if segments else ""
    local = _snake_case(payload_type.__name__.removesuffix("Data") or payload_type.__name__)
    if domain and local.startswith(f"{domain}_"):
        local = local.removeprefix(f"{domain}_")
    return f"{domain}:{local}" if domain else local


def _payload_ancestry(payload_type: type[pydantic.BaseModel]) -> tuple[type[pydantic.BaseModel], ...]:
    """Model levels of ``payload_type``, most general first (``BaseModel`` itself excluded)."""

    return tuple(
        level
        for level in reversed(payload_type.__mro__)
        if issubclass(level, pydantic.BaseModel) and level is not pydantic.BaseModel
    )


def _model_element(model: pydantic.BaseModel) -> ElementTree.Element:
    """Render one payload as its inheritance chain: ancestors outside, concrete type inside.

    Each level carries only the fields it declares, so nothing is repeated between levels and a
    subclass never re-states what it inherited. Containment nests the same way (a field holding a
    payload becomes a child element), which is why a deeper hierarchy or a payload that carries
    another payload needs no change here.
    """

    fields = type(model).model_fields
    root: ElementTree.Element | None = None
    current: ElementTree.Element | None = None
    rendered: set[str] = set()
    for level in _payload_ancestry(type(model)):
        element = ElementTree.Element(_payload_tag(level))
        if current is None:
            root = element
        else:
            current.append(element)
        current = element
        declared = getattr(level, "__annotations__", {})
        for name in fields:
            if name in rendered or name not in declared:
                continue
            rendered.add(name)
            _fill_field(element, name, getattr(model, name, None))
    if root is None:
        root = ElementTree.Element(_payload_tag(type(model)))
    return root


def _fill_field(element: ElementTree.Element, name: str, value: object) -> None:
    """Place one field on ``element``: scalars as attributes, structure as child elements."""

    if value is None:
        return
    if isinstance(value, bytes | bytearray):
        # Media descriptor, never the media: how much there was, never what it contained.
        element.set(name, f"{_MEDIA_DESCRIPTOR_PREFIX}{len(value)}")
        return
    if isinstance(value, memoryview):
        element.set(name, f"{_MEDIA_DESCRIPTOR_PREFIX}{value.nbytes}")
        return
    if isinstance(value, enum.Enum):
        member: object = value.value
        _fill_field(element, name, member)
        return
    if isinstance(value, bool):
        element.set(name, "true" if value else "false")
        return
    if isinstance(value, str | int | float):
        element.set(name, str(value))
        return
    if isinstance(value, pydantic.BaseModel):
        child = ElementTree.SubElement(element, name)
        child.append(_model_element(value))
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _fill_field(element, name, dataclasses.asdict(value))
        return
    if isinstance(value, collections.abc.Mapping):
        child = ElementTree.SubElement(element, name)
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        for key, item in mapping.items():
            _fill_field(child, str(key), item)
        return
    if isinstance(value, collections.abc.Set | collections.abc.Sequence):
        items = tuple(typing.cast(collections.abc.Collection[object], value))
        child = ElementTree.SubElement(element, name)
        child.set("count", str(len(items)))
        for item in items:
            if isinstance(item, pydantic.BaseModel):
                child.append(_model_element(item))
            elif isinstance(item, str):
                ElementTree.SubElement(child, "item").text = item
        # Numeric members (voice embeddings, raw sample runs) stay a count for the same reason
        # media does: a model reasons about how many there were, never about the values.
        return
    element.set(name, str(value))


def model_facing_xml(value: object) -> str:
    """Project a stimulus as one XML element, with raw media replaced by a descriptor.

    This is the model-facing sibling of ``model_facing_json_value`` and carries the same absolute
    guarantee: ``bytes`` never reach a prompt in any encoding. Where the JSON projection writes
    ``{"media": "bytes", "bytes": 115200}``, this writes ``content="bytes:115200"`` — equally
    non-reversible, and equally legible as "there was 115200 bytes of it".

    The stimulus *is* the root element — there is no envelope around it. A payload renders as its
    inheritance chain (see ``_model_element``), so the root is the most general model level and the
    concrete type nests inside it; a payload with no model of its own (raw media, plain text) is
    tagged by its type.

    An ``hsm.Event`` stimulus carries its envelope as ``stimulus:``-prefixed attributes on that same
    root: ``event`` plus whichever of ``id`` / ``source`` / ``target`` it was stamped with. The
    prefix is what keeps the envelope from colliding with a payload field that happens to be called
    ``source`` or ``id``. ``metadata`` is telemetry propagation and never domain content, so it is
    not projected.

    Escaping is ``ElementTree``'s, so transcripts, caller IDs, and any other remote- or
    model-authored text cannot close an element or inject markup.
    """

    event: hsm.Event[object] | None = value if isinstance(value, hsm.Event) else None
    payload: object = event.data if event is not None else value
    if isinstance(payload, pydantic.BaseModel):
        root = _model_element(payload)
    else:
        root = ElementTree.Element(_payload_tag(type(payload)))
        if isinstance(payload, str):
            root.text = payload
        elif payload is not None:
            _fill_field(root, "content", payload)
    if event is not None:
        root.set(f"{_STIMULUS_PREFIX}:event", event.name)
        for attribute, carried in (("id", event.id), ("source", event.source), ("target", event.target)):
            if carried:
                root.set(f"{_STIMULUS_PREFIX}:{attribute}", carried)
    # Every prefix a tag or attribute used is declared on the root, so the result is
    # namespace-well-formed XML rather than names that merely contain a colon. The URI is derived
    # from the prefix; nothing registers one.
    prefixes = {element.tag.split(":", 1)[0] for element in root.iter() if ":" in element.tag}
    prefixes.update(name.split(":", 1)[0] for name in root.keys() if ":" in name)
    for prefix in sorted(prefixes):
        root.set(f"xmlns:{prefix}", f"{_TAG_NAMESPACE_PREFIX}{prefix}")
    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode")


def bytes_from_base64(value: object) -> bytes:
    """Rehydrate media from a JSON hop: ``bytes`` as-is, or base64 text only.

    Uses **URL-safe** decoding so values match pydantic ``ser_json_bytes="base64"``
    (``-``/``_``). Python's urlsafe decoder also accepts the standard ``+``/``/`` alphabet.
    """

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if not isinstance(value, str):
        raise ValueError(
            f"audio content must be bytes or base64 text, got {type(value).__name__}"
        )
    text = value.strip()
    if not text:
        return b""
    pad = (-len(text)) % 4
    if pad:
        text = text + ("=" * pad)
    try:
        return base64.urlsafe_b64decode(text)
    except Exception as error:
        raise ValueError("audio content must be bytes or base64 text") from error


def validate_supported_json_schema(schema: JsonSchema, *, path: str = "$") -> None:
    """Reject schemas whose keywords cannot be enforced by the local matcher."""

    for keyword, value in schema.items():
        if keyword not in _SUPPORTED_JSON_SCHEMA_KEYWORDS:
            raise ValueError(f"unsupported JSON schema keyword {keyword!r} at {path}")
        _validate_json_schema_keyword(keyword, value, path=path)


def matches_json_schema(data: object, schema: collections.abc.Mapping[str, object]) -> bool:
    """Return whether data matches the supported stateforward.bot JSON schema subset."""

    try:
        validate_supported_json_schema(dict(schema))
    except ValueError:
        return False
    return _matches_json_schema(data, schema, root=schema)


def _event_schema_adapter(schema: object) -> pydantic.TypeAdapter[object]:
    if isinstance(schema, pydantic.TypeAdapter):
        return typing.cast(pydantic.TypeAdapter[object], schema)
    return pydantic.TypeAdapter(schema)


def _validate_json_schema_keyword(keyword: str, value: object, *, path: str) -> None:
    if keyword == "type":
        _validate_json_schema_type(value, path=path)
        return
    if keyword in {"title", "description", "default"}:
        return
    if keyword == "examples":
        if not isinstance(value, list):
            raise ValueError(f"JSON schema examples at {path} must be a list")
        return
    if keyword == "$ref":
        if not isinstance(value, str) or not value.startswith("#/"):
            raise ValueError(f"JSON schema $ref at {path} must be a local reference string")
        return
    if keyword == "$defs":
        if not isinstance(value, dict):
            raise ValueError(f"JSON schema $defs at {path} must be an object")
        definitions = typing.cast(dict[object, object], value)
        for definition_name, definition_schema in definitions.items():
            if not isinstance(definition_name, str):
                raise ValueError(f"JSON schema $defs names at {path} must be strings")
            if not isinstance(definition_schema, dict):
                raise ValueError(f"JSON schema $defs {definition_name!r} at {path} must be an object")
            validate_supported_json_schema(
                typing.cast(JsonSchema, definition_schema),
                path=f"{path}.$defs.{definition_name}",
            )
        return
    if keyword in {"anyOf", "oneOf", "allOf"}:
        _validate_schema_options(value, keyword=keyword, path=path)
        return
    if keyword == "not":
        if not isinstance(value, dict):
            raise ValueError(f"JSON schema not at {path} must be an object")
        validate_supported_json_schema(typing.cast(JsonSchema, value), path=f"{path}.not")
        return
    if keyword == "properties":
        if not isinstance(value, dict):
            raise ValueError(f"JSON schema properties at {path} must be an object")
        properties = typing.cast(dict[object, object], value)
        for property_name, property_schema in properties.items():
            if not isinstance(property_name, str):
                raise ValueError(f"JSON schema property names at {path} must be strings")
            if not isinstance(property_schema, dict):
                raise ValueError(f"JSON schema property {property_name!r} at {path} must be an object")
            validate_supported_json_schema(
                typing.cast(JsonSchema, property_schema),
                path=f"{path}.properties.{property_name}",
            )
        return
    if keyword == "required":
        required = typing.cast(list[object], value) if isinstance(value, list) else None
        if required is None or not all(isinstance(item, str) for item in required):
            raise ValueError(f"JSON schema required at {path} must be a list of strings")
        return
    if keyword == "additionalProperties":
        if isinstance(value, bool):
            return
        if isinstance(value, dict):
            validate_supported_json_schema(typing.cast(JsonSchema, value), path=f"{path}.additionalProperties")
            return
        raise ValueError(f"JSON schema additionalProperties at {path} must be a boolean or object")
    if keyword == "items":
        if not isinstance(value, dict):
            raise ValueError(f"JSON schema items at {path} must be an object")
        validate_supported_json_schema(typing.cast(JsonSchema, value), path=f"{path}.items")
        return
    if keyword == "enum":
        if not isinstance(value, list):
            raise ValueError(f"JSON schema enum at {path} must be a list")
        return
    if keyword in _STRING_SCHEMA_KEYWORDS:
        _validate_string_schema_keyword(keyword, value, path=path)
        return
    if keyword in _NUMBER_SCHEMA_KEYWORDS:
        _validate_number_schema_keyword(keyword, value, path=path)
        return
    if keyword == "multipleOf":
        _validate_positive_number(value, keyword=keyword, path=path)
        return
    if keyword in {"maxItems", "minItems", "maxProperties", "minProperties"}:
        _validate_non_negative_integer(value, keyword=keyword, path=path)


def _validate_json_schema_type(value: object, *, path: str) -> None:
    if isinstance(value, str):
        if value not in _SUPPORTED_JSON_SCHEMA_TYPES:
            raise ValueError(f"unsupported JSON schema type {value!r} at {path}")
        return
    values = typing.cast(list[object], value) if isinstance(value, list) else None
    if values is not None and values and all(isinstance(item, str) for item in values):
        schema_types = [typing.cast(str, item) for item in values]
        unsupported = tuple(sorted(set(schema_types).difference(_SUPPORTED_JSON_SCHEMA_TYPES)))
        if unsupported:
            raise ValueError(f"unsupported JSON schema types at {path}: {', '.join(unsupported)}")
        return
    raise ValueError(f"JSON schema type at {path} must be a string or non-empty list of strings")


def _validate_string_schema_keyword(keyword: str, value: object, *, path: str) -> None:
    if keyword == "pattern":
        if not isinstance(value, str):
            raise ValueError(f"JSON schema pattern at {path} must be a string")
        try:
            _ = re.compile(value)
        except re.error as error:
            raise ValueError(f"JSON schema pattern at {path} is invalid: {error}") from error
        return
    _validate_non_negative_integer(value, keyword=keyword, path=path)


def _validate_number_schema_keyword(keyword: str, value: object, *, path: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"JSON schema {keyword} at {path} must be a number")


def _validate_positive_number(value: object, *, keyword: str, path: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"JSON schema {keyword} at {path} must be a positive number")


def _validate_non_negative_integer(value: object, *, keyword: str, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"JSON schema {keyword} at {path} must be a non-negative integer")


def _validate_schema_options(value: object, *, keyword: str, path: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"JSON schema {keyword} at {path} must be a non-empty list")
    for index, option in enumerate(typing.cast(list[object], value)):
        if not isinstance(option, dict):
            raise ValueError(f"JSON schema {keyword} option at {path}[{index}] must be an object")
        validate_supported_json_schema(typing.cast(JsonSchema, option), path=f"{path}.{keyword}[{index}]")


def _matches_json_schema(
    data: object,
    schema: collections.abc.Mapping[str, object],
    *,
    root: collections.abc.Mapping[str, object],
) -> bool:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        resolved = _resolve_local_ref(root, reference)
        if resolved is None:
            return False
        return _matches_json_schema(data, resolved, root=root)

    if not schema:
        return True
    not_schema = schema.get("not")
    if isinstance(not_schema, collections.abc.Mapping) and _matches_json_schema(
        data,
        typing.cast(collections.abc.Mapping[str, object], not_schema),
        root=root,
    ):
        return False
    any_of = _schema_options(schema.get("anyOf"))
    if any_of is not None and not any(_matches_json_schema(data, option, root=root) for option in any_of):
        return False
    one_of = _schema_options(schema.get("oneOf"))
    if one_of is not None:
        match_count = sum(1 for option in one_of if _matches_json_schema(data, option, root=root))
        if match_count != 1:
            return False
    all_of = _schema_options(schema.get("allOf"))
    if all_of is not None and not all(_matches_json_schema(data, option, root=root) for option in all_of):
        return False
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_json_schema_type(data, expected_type):
        return False
    enum = schema.get("enum")
    if isinstance(enum, list) and data not in enum:
        return False
    const = schema.get("const")
    if "const" in schema and data != const:
        return False
    if _string_schema_expected(schema):
        if not isinstance(data, str) or not _matches_string_schema(data, schema):
            return False
    if _number_schema_expected(schema):
        if not isinstance(data, int | float) or isinstance(data, bool) or not _matches_number_schema(data, schema):
            return False
    if _object_schema_expected(schema):
        if not isinstance(data, dict):
            return False
        object_data = typing.cast(dict[object, object], data)
        if not _matches_object_schema(object_data, schema, root=root):
            return False
    if _array_schema_expected(schema):
        if not isinstance(data, list | tuple):
            return False
        sequence = typing.cast(collections.abc.Sequence[object], data)
        if not _matches_array_schema(tuple(sequence), schema, root=root):
            return False
    return True


def _resolve_local_ref(
    root: collections.abc.Mapping[str, object],
    reference: str,
) -> collections.abc.Mapping[str, object] | None:
    if not reference.startswith("#/"):
        return None
    current: object = root
    for part in reference[2:].split("/"):
        if not isinstance(current, collections.abc.Mapping):
            return None
        mapping = typing.cast(collections.abc.Mapping[object, object], current)
        current = mapping.get(part.replace("~1", "/").replace("~0", "~"))
    if isinstance(current, collections.abc.Mapping):
        return typing.cast(collections.abc.Mapping[str, object], current)
    return None


def _schema_options(value: object) -> tuple[collections.abc.Mapping[str, object], ...] | None:
    if not isinstance(value, list):
        return None
    options: list[collections.abc.Mapping[str, object]] = []
    for item in typing.cast(list[object], value):
        if not isinstance(item, collections.abc.Mapping):
            return ()
        options.append(typing.cast(collections.abc.Mapping[str, object], item))
    return tuple(options)


def _matches_json_schema_type(data: object, expected_type: object) -> bool:
    if isinstance(expected_type, list):
        expected_types = typing.cast(list[object], expected_type)
        return any(_matches_json_schema_type(data, item) for item in expected_types)
    if not isinstance(expected_type, str):
        return True
    if expected_type == "object":
        return isinstance(data, dict)
    if expected_type == "array":
        return isinstance(data, list | tuple)
    if expected_type == "string":
        return isinstance(data, str)
    if expected_type == "integer":
        return isinstance(data, int) and not isinstance(data, bool)
    if expected_type == "number":
        return isinstance(data, int | float) and not isinstance(data, bool)
    if expected_type == "boolean":
        return isinstance(data, bool)
    if expected_type == "null":
        return data is None
    return True


def _object_schema_expected(schema: collections.abc.Mapping[str, object]) -> bool:
    return any(keyword in schema for keyword in _OBJECT_SCHEMA_KEYWORDS)


def _array_schema_expected(schema: collections.abc.Mapping[str, object]) -> bool:
    return any(keyword in schema for keyword in _ARRAY_SCHEMA_KEYWORDS)


def _string_schema_expected(schema: collections.abc.Mapping[str, object]) -> bool:
    return any(keyword in schema for keyword in _STRING_SCHEMA_KEYWORDS)


def _number_schema_expected(schema: collections.abc.Mapping[str, object]) -> bool:
    return any(keyword in schema for keyword in _NUMBER_SCHEMA_KEYWORDS) or "multipleOf" in schema


def _matches_object_schema(
    data: dict[object, object],
    schema: collections.abc.Mapping[str, object],
    *,
    root: collections.abc.Mapping[str, object],
) -> bool:
    if any(not isinstance(key, str) for key in data):
        return False
    min_properties = schema.get("minProperties")
    if isinstance(min_properties, int) and len(data) < min_properties:
        return False
    max_properties = schema.get("maxProperties")
    if isinstance(max_properties, int) and len(data) > max_properties:
        return False
    required = schema.get("required")
    if isinstance(required, list):
        required_keys = typing.cast(list[object], required)
        for key in required_keys:
            if isinstance(key, str) and key not in data:
                return False
    properties = schema.get("properties")
    if isinstance(properties, dict):
        property_schemas = typing.cast(dict[object, object], properties)
        for key, property_schema in property_schemas.items():
            if isinstance(key, str) and key in data and isinstance(property_schema, dict):
                schema_object = typing.cast(JsonSchema, property_schema)
                if not _matches_json_schema(data[key], schema_object, root=root):
                    return False
    additional_properties = schema.get("additionalProperties")
    property_schemas = typing.cast(dict[object, object], properties) if isinstance(properties, dict) else {}
    allowed = {key for key in property_schemas if isinstance(key, str)}
    additional_keys = tuple(key for key in data if isinstance(key, str) and key not in allowed)
    if additional_properties is False:
        if additional_keys:
            return False
    elif isinstance(additional_properties, dict):
        additional_schema = typing.cast(JsonSchema, additional_properties)
        for key in additional_keys:
            if not _matches_json_schema(data[key], additional_schema, root=root):
                return False
    return True


def _matches_array_schema(
    data: tuple[object, ...],
    schema: collections.abc.Mapping[str, object],
    *,
    root: collections.abc.Mapping[str, object],
) -> bool:
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(data) < min_items:
        return False
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(data) > max_items:
        return False
    item_schema = schema.get("items")
    if isinstance(item_schema, dict):
        item_schema_object = typing.cast(JsonSchema, item_schema)
        return all(_matches_json_schema(item, item_schema_object, root=root) for item in data)
    return True


def _matches_string_schema(data: str, schema: collections.abc.Mapping[str, object]) -> bool:
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(data) < min_length:
        return False
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(data) > max_length:
        return False
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, data) is None:
        return False
    return True


def _matches_number_schema(data: int | float, schema: collections.abc.Mapping[str, object]) -> bool:
    minimum = schema.get("minimum")
    if isinstance(minimum, int | float) and data < minimum:
        return False
    maximum = schema.get("maximum")
    if isinstance(maximum, int | float) and data > maximum:
        return False
    exclusive_minimum = schema.get("exclusiveMinimum")
    if isinstance(exclusive_minimum, int | float) and data <= exclusive_minimum:
        return False
    exclusive_maximum = schema.get("exclusiveMaximum")
    if isinstance(exclusive_maximum, int | float) and data >= exclusive_maximum:
        return False
    multiple_of = schema.get("multipleOf")
    if isinstance(multiple_of, int | float) and data % multiple_of != 0:
        return False
    return True


def _inline_root_ref(schema: JsonSchema) -> JsonSchema:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith(_LOCAL_DEFS_REF_PREFIX):
        return schema
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return schema
    defs_by_name = typing.cast(dict[str, object], defs)
    definition = defs_by_name.get(ref.removeprefix(_LOCAL_DEFS_REF_PREFIX))
    if not isinstance(definition, dict):
        return schema
    inlined = dict(typing.cast(dict[str, object], definition))
    for key, value in schema.items():
        if key not in {"$defs", "$ref"}:
            inlined[key] = value
    if _has_local_defs_ref(inlined):
        inlined["$defs"] = defs
    return inlined


def _has_local_defs_ref(value: object) -> bool:
    if isinstance(value, dict):
        mapping = typing.cast(dict[object, object], value)
        ref = mapping.get("$ref")
        if isinstance(ref, str) and ref.startswith(_LOCAL_DEFS_REF_PREFIX):
            return True
        return any(_has_local_defs_ref(item) for key, item in mapping.items() if key != "$defs")
    if isinstance(value, list | tuple):
        sequence = typing.cast(collections.abc.Sequence[object], value)
        return any(_has_local_defs_ref(item) for item in sequence)
    return False


def _local_defs_table(schema: collections.abc.Mapping[str, object]) -> dict[str, object]:
    """Return the document-local ``$defs`` table for ``schema`` when it is a document root."""

    raw = schema.get("$defs")
    if not isinstance(raw, dict):
        return {}
    table: dict[str, object] = {}
    for name, definition in typing.cast(dict[object, object], raw).items():
        if isinstance(name, str):
            table[name] = definition
    return table


def _close_local_defs_refs(
    value: object,
    *,
    defs: collections.abc.Mapping[str, object],
    resolving: frozenset[str],
) -> object:
    """Deep-copy ``value``, inlining ``#/$defs/…`` against ``defs`` (cycle- and miss-safe)."""

    if isinstance(value, dict):
        mapping = typing.cast(dict[str, object], dict(typing.cast(dict[object, object], value)))
        ref = mapping.get("$ref")
        if isinstance(ref, str) and ref.startswith(_LOCAL_DEFS_REF_PREFIX):
            name = ref.removeprefix(_LOCAL_DEFS_REF_PREFIX)
            if not name:
                raise ValueError(f"JSON schema $ref {ref!r} is not a named local definition")
            if name in resolving:
                raise ValueError(f"JSON schema $ref cycle involving {name!r}")
            definition = defs.get(name)
            if not isinstance(definition, dict):
                raise ValueError(f"JSON schema $ref {ref!r} does not resolve in $defs")
            definition_schema = typing.cast(dict[str, object], dict(typing.cast(dict[object, object], definition)))
            closed_definition = _close_local_defs_refs(
                definition_schema,
                defs=defs,
                resolving=resolving | {name},
            )
            if not isinstance(closed_definition, dict):
                raise ValueError(f"JSON schema $defs {name!r} must resolve to an object schema")
            merged: dict[str, object] = dict(typing.cast(dict[str, object], closed_definition))
            for key, child in mapping.items():
                if key in {"$ref", "$defs"}:
                    continue
                merged[key] = _close_local_defs_refs(child, defs=defs, resolving=resolving)
            return merged
        closed: dict[str, object] = {}
        for key, child in mapping.items():
            if key == "$defs":
                # Embeddable fragments must not keep a local $defs table: #/$defs refs are
                # document-root relative and break when this object is nested under another schema.
                continue
            closed[key] = _close_local_defs_refs(child, defs=defs, resolving=resolving)
        return closed
    if isinstance(value, list | tuple):
        sequence = typing.cast(collections.abc.Sequence[object], value)
        return [_close_local_defs_refs(item, defs=defs, resolving=resolving) for item in sequence]
    return value


def embeddable_json_schema(schema: collections.abc.Mapping[str, object]) -> JsonSchema:
    """Return a ref-closed copy of ``schema`` safe to nest inside a larger JSON Schema document.

    Pydantic (and ``event_schema_json_schema``) may emit document-root ``#/$defs/Name`` refs when a
    payload is its own schema document. Those refs are valid only while that payload remains the
    document root. Tool composition (e.g. ``dispatch`` ``anyOf`` branches) nests payloads under a
    larger parameters document, so document-root refs must be closed before embedding.

    This projection:
    - resolves every local ``#/$defs/…`` against this schema's own ``$defs``
    - merges sibling keywords onto inlined definitions (Pydantic often pairs ``$ref`` with
      ``description`` / ``examples``)
    - strips ``$defs`` from the result
    - does not invent domain fields; it only eliminates nesting-unsafe structure

    Validation of live event data still uses typed Pydantic models, not this closed shape.
    """

    root = dict(schema)
    defs = _local_defs_table(root)
    closed = _close_local_defs_refs(root, defs=defs, resolving=frozenset())
    if not isinstance(closed, dict):
        raise ValueError("embeddable_json_schema requires an object schema root")
    result = typing.cast(JsonSchema, closed)
    if _has_local_defs_ref(result):
        raise ValueError("embeddable_json_schema left unresolved local $defs references")
    return result


def json_schema_is_embeddable(schema: collections.abc.Mapping[str, object]) -> bool:
    """Return whether ``schema`` has no local ``#/$defs/…`` refs (safe to nest as a fragment)."""

    return not _has_local_defs_ref(schema)


__all__ = [
    "JsonSchema",
    "bytes_from_base64",
    "embeddable_json_schema",
    "event_json_schema",
    "event_schema_json_schema",
    "json_schema_is_embeddable",
    "matches_json_schema",
    "model_facing_json_value",
    "model_facing_xml",
    "project_json_value",
    "validate_event_data",
    "validate_event_schema_data",
    "validate_supported_json_schema",
]
