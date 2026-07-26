from __future__ import annotations

from bot.abilities import processing
from bot.abilities.language import text

import collections.abc
import copy
import json
import re
import typing

import pydantic


from .client import ContentClient, jsonable
from .text_generator import TextGenerator as ProviderTextGenerator


class ProcessingError(RuntimeError):
    """Raised when an Gemini processing response cannot be validated."""


class _StructuredOutputValidationError(ValueError):
    """Raised when model output violates the structured-output schema sent to the provider."""


_IGNORED_STRICT_SCHEMA_KEYS = frozenset({"default", "examples", "title"})
_UNSUPPORTED_STRICT_SCHEMA_KEYS = (
    "prefixItems",
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
)
_ALLOWED_STRICT_SCHEMA_KEYS = frozenset(
    {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "definitions",
        "description",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "type",
    }
)
_STRICT_SCHEMA_SHAPE_KEYS = frozenset({"type", "$ref", "anyOf", "enum", "const", "properties"})


def _mapping_dict(value: object) -> dict[str, object]:
    if not isinstance(value, collections.abc.Mapping):
        return {}
    mapping = typing.cast(collections.abc.Mapping[object, object], value)
    return {str(key): item for key, item in mapping.items()}


def _is_schema_mapping(value: object) -> typing.TypeGuard[collections.abc.Mapping[str, object]]:
    return isinstance(value, collections.abc.Mapping)


def _schema_sequence(value: object) -> list[object]:
    if not isinstance(value, collections.abc.Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return list(value)


def _decode_ref_part(part: str) -> str:
    return part.replace("~1", "/").replace("~0", "~")


def _resolve_schema_ref(
    ref: object,
    root_schema: collections.abc.Mapping[str, object],
) -> collections.abc.Mapping[str, object] | None:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    current: object = root_schema
    for part in ref[2:].split("/"):
        if not _is_schema_mapping(current):
            return None
        current = current.get(_decode_ref_part(part))
    if not _is_schema_mapping(current):
        return None
    return current


def _copy_schema(schema: collections.abc.Mapping[str, object]) -> dict[str, object]:
    return copy.deepcopy(dict(schema))


def _require_const_properties(schema: collections.abc.Mapping[str, object]) -> dict[str, object]:
    output = _copy_schema(schema)
    _mark_const_properties_required(output)
    return output


def _mark_const_properties_required(schema: object) -> None:
    if not isinstance(schema, collections.abc.MutableMapping):
        return
    schema_mapping = typing.cast(collections.abc.MutableMapping[str, object], schema)
    raw_properties = schema_mapping.get("properties")
    if isinstance(raw_properties, collections.abc.Mapping):
        properties = typing.cast(collections.abc.Mapping[str, object], raw_properties)
        required = [item for item in _schema_sequence(schema_mapping.get("required")) if isinstance(item, str)]
        required_names = set(required)
        for property_name, property_schema in properties.items():
            if isinstance(property_schema, collections.abc.Mapping):
                property_schema_map = typing.cast(collections.abc.Mapping[str, object], property_schema)
                if "const" in property_schema_map and property_name not in required_names:
                    required.append(property_name)
                    required_names.add(property_name)
                _mark_const_properties_required(property_schema_map)
        if required:
            schema_mapping["required"] = required
    for key in ("$defs", "definitions"):
        raw_definitions = schema_mapping.get(key)
        if isinstance(raw_definitions, collections.abc.Mapping):
            definitions = typing.cast(collections.abc.Mapping[str, object], raw_definitions)
            for definition in definitions.values():
                _mark_const_properties_required(definition)
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in _schema_sequence(schema_mapping.get(key)):
            _mark_const_properties_required(branch)


def _normalize_gemini_schema(
    schema: collections.abc.Mapping[str, object],
    *,
    root_schema: collections.abc.Mapping[str, object] | None = None,
) -> dict[str, object]:
    root_schema = root_schema or schema
    for unsupported_key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if unsupported_key in schema:
            raise ValueError(f"Gemini structured outputs do not support JSON Schema keyword: {unsupported_key}.")
    for key in schema:
        if key not in _IGNORED_STRICT_SCHEMA_KEYS and key not in _ALLOWED_STRICT_SCHEMA_KEYS:
            raise ValueError(f"Gemini structured outputs do not support JSON Schema keyword: {key}.")
    if "$ref" in schema and len(schema) > 1:
        referenced_schema = _resolve_schema_ref(schema.get("$ref"), root_schema)
        if referenced_schema is not None:
            merged_schema = dict(referenced_schema)
            merged_schema.update({key: value for key, value in schema.items() if key != "$ref"})
            return _normalize_gemini_schema(merged_schema, root_schema=root_schema)

    normalized: dict[str, object] = {}
    for key, value in schema.items():
        if key in _IGNORED_STRICT_SCHEMA_KEYS:
            continue
        if key in {"properties", "$defs", "definitions"}:
            properties: dict[str, object] = {}
            for property_name, property_schema in _mapping_dict(value).items():
                if isinstance(property_schema, collections.abc.Mapping):
                    properties[property_name] = _normalize_gemini_schema(
                        typing.cast(collections.abc.Mapping[str, object], property_schema),
                        root_schema=root_schema,
                    )
                else:
                    properties[property_name] = property_schema
            normalized[key] = properties
            continue
        if key == "oneOf":
            normalized["anyOf"] = [
                _normalize_gemini_schema(
                    typing.cast(collections.abc.Mapping[str, object], item),
                    root_schema=root_schema,
                )
                if isinstance(item, collections.abc.Mapping)
                else item
                for item in _schema_sequence(value)
            ]
            continue
        if key == "anyOf":
            normalized[key] = [
                _normalize_gemini_schema(
                    typing.cast(collections.abc.Mapping[str, object], item),
                    root_schema=root_schema,
                )
                if isinstance(item, collections.abc.Mapping)
                else item
                for item in _schema_sequence(value)
            ]
            continue
        if key == "additionalProperties" and value is not False:
            raise ValueError("Gemini structured outputs do not support dynamic object keys.")
        if key == "items" and isinstance(value, collections.abc.Mapping):
            normalized[key] = _normalize_gemini_schema(
                typing.cast(collections.abc.Mapping[str, object], value),
                root_schema=root_schema,
            )
            continue
        normalized[key] = value

    properties = _mapping_dict(normalized.get("properties"))
    if normalized.get("type") == "object" or properties:
        normalized["additionalProperties"] = False
        normalized["required"] = list(properties)
    if not any(key in normalized for key in _STRICT_SCHEMA_SHAPE_KEYS):
        message = (
            "Gemini structured outputs require every schema branch to declare a type, reference, enum, const, "
            + "or anyOf."
        )
        raise ValueError(message)
    return normalized


def _schema_requires_root_wrapper(schema: collections.abc.Mapping[str, object]) -> bool:
    return schema.get("type") != "object" or "anyOf" in schema or "oneOf" in schema


def _wrap_output_schema(schema: collections.abc.Mapping[str, object]) -> dict[str, object]:
    result_schema = dict(schema)
    root_schema: dict[str, object] = {
        "type": "object",
        "properties": {"result": result_schema},
        "required": ["result"],
        "additionalProperties": False,
    }
    for definitions_key in ("$defs", "definitions"):
        definitions = result_schema.pop(definitions_key, None)
        if definitions is not None:
            root_schema[definitions_key] = definitions
    return root_schema


def _gemini_structured_output_schema(
    schema: collections.abc.Mapping[str, object],
    *,
    strict: bool,
) -> tuple[dict[str, object], bool]:
    raw_schema = _normalize_gemini_schema(schema) if strict else _copy_schema(schema)
    output_schema = _require_const_properties(raw_schema)
    if not _schema_requires_root_wrapper(output_schema):
        return output_schema, False
    return _wrap_output_schema(output_schema), True


def _schema_name_for(output_type: object) -> str:
    raw_name = getattr(output_type, "__name__", "processing_output")
    if not isinstance(raw_name, str):
        raw_name = "processing_output"
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name).strip("_").lower()
    return name or "processing_output"


def _event_payload_schema(
    event: processing.Event[typing.Any],
    *,
    patch: processing.SchemaPatch | None = None,
) -> dict[str, object]:
    schema = _copy_schema(processing.model_facing_event_json_schema(event, patch=patch))
    if not schema:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    if schema.get("type") == "object" or "properties" in schema:
        return schema
    raise ValueError("Gemini event tools require object event data schemas.")


def _validate_event_payload(
    event: processing.Event[typing.Any],
    data: collections.abc.Mapping[str, object],
    *,
    patch: processing.SchemaPatch | None = None,
) -> None:
    schema = _event_payload_schema(event, patch=patch)
    _validate_structured_json(dict(data), schema, root_schema=schema)


def _events_from_dispatch_tool_calls(
    tool_calls: tuple[text.generation.TextToolCall, ...],
    offered: collections.abc.Sequence[processing.Event[typing.Any]],
    *,
    patch: processing.SchemaPatch | None = None,
) -> processing.Events:
    if not tool_calls:
        return ()
    # Prefer the dedicated dispatch tool; ignore unexpected names.
    dispatch_calls = [call for call in tool_calls if call.name == processing.DISPATCH_TOOL_NAME]
    if not dispatch_calls:
        names = ", ".join(sorted({call.name for call in tool_calls}))
        raise _StructuredOutputValidationError(
            f"processing expected tool {processing.DISPATCH_TOOL_NAME!r}, got: {names}"
        )
    # One dispatch call is the contract; if the model emits several, merge in order.
    merged: list[processing.SelectedEvent] = []
    by_name = {event.name: event for event in offered}
    for call in dispatch_calls:
        try:
            selections = processing.events_from_dispatch_args(
                call.args,
                patch=patch,
                offered=offered,
            )
        except ValueError as error:
            raise _StructuredOutputValidationError(str(error)) from error
        for selection in selections:
            event = by_name[selection.event]
            raw = selection.data if selection.data is not None else {}
            # Validate pre-unpatch shape: re-merge meta into data for schema check when present.
            payload = dict(raw)
            if selection.meta:
                payload.update(selection.meta)
            elif selection.confidence is not None and "confidence" not in payload:
                payload["confidence"] = selection.confidence
            _validate_event_payload(event, payload, patch=patch)
            merged.append(selection)
    return tuple(merged)


_MISSING = object()


def _field(value: object, name: str, default: object = _MISSING) -> object:
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return mapping.get(name, default)
    return getattr(value, name, default)


def _looks_like_event_selection(output: object) -> bool:
    """Return whether a value looks like one cognition event selection (has string `event`)."""

    return isinstance(_field(output, "event", None), str)


def _validate_selection_payloads(
    selections: processing.Events,
    offered: collections.abc.Sequence[processing.Event[typing.Any]],
    *,
    patch: processing.SchemaPatch | None = None,
) -> None:
    by_name = {event.name: event for event in offered}
    for selection in selections:
        event = by_name.get(selection.event)
        if event is None:
            raise _StructuredOutputValidationError(
                f"processing response selected an unavailable event: {selection.event}"
            )
        payload = dict(selection.data or {})
        if selection.meta:
            payload.update(selection.meta)
        elif selection.confidence is not None:
            payload.setdefault("confidence", selection.confidence)
        _validate_event_payload(event, payload, patch=patch)


def _events_from_content(
    content: str,
    offered: collections.abc.Sequence[processing.Event[typing.Any]],
    *,
    patch: processing.SchemaPatch | None = None,
) -> processing.Events:
    if not content.strip():
        return ()
    try:
        parsed: object = json.loads(content)
    except json.JSONDecodeError as error:
        raise _StructuredOutputValidationError("processing response was not JSON") from error
    if isinstance(parsed, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[str, object], parsed)
        if "events" in mapping:
            parsed = mapping["events"]
        elif "result" in mapping:
            parsed = mapping["result"]
    try:
        if isinstance(parsed, collections.abc.Mapping) and "events" in parsed:
            selections = processing.events_from_dispatch_args(
                typing.cast(collections.abc.Mapping[str, object], parsed),
                patch=patch,
                offered=offered,
            )
        else:
            selections = processing.coerce_event_selections(parsed, patch=patch)
            if selections is None:
                raise ValueError("processing response was not an events array")
            if offered:
                allowed = {event.name for event in offered}
                for item in selections:
                    if item.event not in allowed:
                        raise ValueError(f"dispatch selected unavailable event: {item.event}.")
    except ValueError as error:
        raise _StructuredOutputValidationError(str(error)) from error
    if offered:
        _validate_selection_payloads(selections, offered, patch=patch)
    return selections


def _schema_type_matches(value: object, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "object":
        return isinstance(value, collections.abc.Mapping)
    if schema_type == "array":
        return isinstance(value, collections.abc.Sequence) and not isinstance(value, str | bytes | bytearray)
    return False


def _schema_types(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(item for item in _schema_sequence(value) if isinstance(item, str))


def _validate_structured_json(
    value: object,
    schema: collections.abc.Mapping[str, object],
    *,
    root_schema: collections.abc.Mapping[str, object],
) -> None:
    ref_schema = _resolve_schema_ref(schema.get("$ref"), root_schema)
    if ref_schema is not None:
        _validate_structured_json(value, ref_schema, root_schema=root_schema)
        return

    any_of = schema.get("anyOf")
    if any_of is not None:
        for branch in _schema_sequence(any_of):
            if not isinstance(branch, collections.abc.Mapping):
                continue
            try:
                _validate_structured_json(
                    value,
                    typing.cast(collections.abc.Mapping[str, object], branch),
                    root_schema=root_schema,
                )
            except _StructuredOutputValidationError:
                continue
            return
        raise _StructuredOutputValidationError("value did not match any allowed schema branch")

    if "const" in schema and value != schema.get("const"):
        raise _StructuredOutputValidationError("value did not match const")
    enum_values = schema.get("enum")
    if enum_values is not None and value not in _schema_sequence(enum_values):
        raise _StructuredOutputValidationError("value did not match enum")

    schema_types = _schema_types(schema.get("type"))
    if schema_types and not any(_schema_type_matches(value, schema_type) for schema_type in schema_types):
        raise _StructuredOutputValidationError("value did not match schema type")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise _StructuredOutputValidationError("string value shorter than minLength")
        max_length = schema.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            raise _StructuredOutputValidationError("string value longer than maxLength")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise _StructuredOutputValidationError("string value did not match pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if isinstance(minimum, int | float) and value < minimum:
            raise _StructuredOutputValidationError("numeric value below minimum")
        maximum = schema.get("maximum")
        if isinstance(maximum, int | float) and value > maximum:
            raise _StructuredOutputValidationError("numeric value above maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, int | float) and value <= exclusive_minimum:
            raise _StructuredOutputValidationError("numeric value below exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(exclusive_maximum, int | float) and value >= exclusive_maximum:
            raise _StructuredOutputValidationError("numeric value above exclusiveMaximum")

    if "object" in schema_types:
        _validate_structured_object(value, schema, root_schema=root_schema)
    if "array" in schema_types:
        _validate_structured_array(value, schema, root_schema=root_schema)


def _validate_structured_object(
    value: object,
    schema: collections.abc.Mapping[str, object],
    *,
    root_schema: collections.abc.Mapping[str, object],
) -> None:
    if not isinstance(value, collections.abc.Mapping):
        raise _StructuredOutputValidationError("object value must be a mapping")
    value_mapping = typing.cast(collections.abc.Mapping[object, object], value)
    value_keys = {str(key) for key in value_mapping}
    properties = _mapping_dict(schema.get("properties"))
    required = {item for item in _schema_sequence(schema.get("required")) if isinstance(item, str)}

    missing = required - value_keys
    if missing:
        raise _StructuredOutputValidationError("object value is missing required keys")
    if schema.get("additionalProperties") is False:
        extra = value_keys - set(properties)
        if extra:
            raise _StructuredOutputValidationError("object value included additional keys")

    string_keyed_value = {str(key): item for key, item in value_mapping.items()}
    for property_name, property_schema in properties.items():
        if property_name not in string_keyed_value or not isinstance(property_schema, collections.abc.Mapping):
            continue
        _validate_structured_json(
            string_keyed_value[property_name],
            typing.cast(collections.abc.Mapping[str, object], property_schema),
            root_schema=root_schema,
        )


def _validate_structured_array(
    value: object,
    schema: collections.abc.Mapping[str, object],
    *,
    root_schema: collections.abc.Mapping[str, object],
) -> None:
    if not isinstance(value, collections.abc.Sequence) or isinstance(value, str | bytes | bytearray):
        raise _StructuredOutputValidationError("array value must be a sequence")
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        raise _StructuredOutputValidationError("array value shorter than minItems")
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        raise _StructuredOutputValidationError("array value longer than maxItems")
    item_schema = schema.get("items")
    if not isinstance(item_schema, collections.abc.Mapping):
        return
    item_schema_mapping = typing.cast(collections.abc.Mapping[str, object], item_schema)
    for item in value:
        _validate_structured_json(item, item_schema_mapping, root_schema=root_schema)


def _same_typed_completion_value(value: object, validated: object) -> bool:
    if validated is value:
        return True
    if isinstance(value, collections.abc.Mapping) and isinstance(validated, collections.abc.Mapping):
        value_mapping = typing.cast(collections.abc.Mapping[object, object], value)
        validated_mapping = typing.cast(collections.abc.Mapping[object, object], validated)
        if value_mapping.keys() != validated_mapping.keys():
            return False
        return all(_same_typed_completion_value(item, validated_mapping[key]) for key, item in value_mapping.items())
    if isinstance(value, list) and isinstance(validated, list):
        value_items = typing.cast(list[object], value)
        validated_items = typing.cast(list[object], validated)
        if len(value_items) != len(validated_items):
            return False
        return all(_same_typed_completion_value(item, validated_items[index]) for index, item in enumerate(value_items))
    if isinstance(value, tuple) and isinstance(validated, tuple):
        value_tuple = typing.cast(tuple[object, ...], value)
        validated_tuple = typing.cast(tuple[object, ...], validated)
        if len(value_tuple) != len(validated_tuple):
            return False
        return all(_same_typed_completion_value(item, validated_tuple[index]) for index, item in enumerate(value_tuple))
    if value is None or validated is None:
        return value is None and validated is None
    if isinstance(value, bool) or isinstance(validated, bool):
        return isinstance(value, bool) and isinstance(validated, bool) and value == validated
    if isinstance(value, int) and isinstance(validated, int):
        return value == validated
    if isinstance(value, float) and isinstance(validated, float):
        return value == validated
    if isinstance(value, str) and isinstance(validated, str):
        return value == validated
    return False


class Processor(processing.Processor):
    """Gemini transport: ``process`` maps a Processing input (+ stamped instructions) → events."""

    _generator: text.generation.TextGenerator

    def __init__(
        self,
        *,
        generator: text.generation.TextGenerator | None = None,
        client: ContentClient | None = None,
        provider: str | None = "gemini",
        config: collections.abc.Mapping[str, object] | None = None,
    ) -> None:
        if generator is not None and client is not None:
            raise ValueError("Provide either generator or client, not both.")
        if generator is None:
            if client is None:
                raise ValueError("Processing requires either a generator or a client.")
            generator = ProviderTextGenerator(
                client=client,
                provider=provider,
                config=config or {},
            )
        self._generator = generator

    @staticmethod
    def _user_content(input: processing.InputData) -> str:
        """Project a processing input for the model without dumping HSM schema adapters."""

        projected: list[dict[str, object]] = []
        for event in input.schemas:
            schema = processing.model_facing_event_json_schema(event, patch=input.patch)
            item: dict[str, object] = {"event": event.name, "schema": schema}
            description = schema.get("description")
            if isinstance(description, str) and description:
                item["description"] = description
            projected.append(item)
        payload = {"input": jsonable(input.input), "schemas": projected}
        return json.dumps(payload, separators=(",", ":"))

    @typing.override
    async def process(self, input: processing.InputData) -> processing.Events:
        instructions = (input.instructions or "").strip()
        if not instructions:
            raise ProcessingError("Gemini processing requires non-blank instructions stamped by Processing.")
        operation_tools: tuple[dict[str, object], ...] = ()
        if input.schemas:
            operation_tools = (processing.dispatch_tool(input.schemas, patch=input.patch),)
        generation_input = text.generation.InputData(
            messages=(
                text.generation.TextMessage(role=text.generation.TextRole.SYSTEM, content=instructions),
                text.generation.TextMessage(
                    role=text.generation.TextRole.USER,
                    content=self._user_content(input),
                ),
            ),
            tools=operation_tools,
            tool_selection=(
                # Required so multi-select is always a filled events array (possibly empty).
                text.generation.ToolSelectionPolicy.REQUIRED
                if operation_tools
                else text.generation.ToolSelectionPolicy.NONE
            ),
        )
        output = await self._generator.generate(generation_input)
        try:
            if output.tool_calls:
                return _events_from_dispatch_tool_calls(
                    output.tool_calls,
                    input.schemas,
                    patch=input.patch,
                )
            return _events_from_content(output.content, input.schemas, patch=input.patch)
        except (pydantic.ValidationError, _StructuredOutputValidationError) as error:
            message = f"Gemini processing response did not produce a valid events array: {error}"
            raise ProcessingError(message) from error


__all__ = ["Processor", "ProcessingError"]
