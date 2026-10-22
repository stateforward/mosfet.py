from __future__ import annotations

import mosfet.media
from mosfet.abilities import processing
from mosfet.abilities.language import text

import collections.abc
import copy
import json
import re
import typing

import pydantic


from .client import ContentClient
from .text_generator import TextGenerator as ProviderTextGenerator


class ProcessingError(RuntimeError):
    """Raised when an Gemini processing response cannot be validated."""


class _StructuredOutputValidationError(ValueError):
    """Raised when model output violates the structured-output schema sent to the provider."""


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
    targets_by_event: collections.abc.Mapping[str, collections.abc.Sequence[str]] | None = None,
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
                targets_by_event=targets_by_event,
            )
        except ValueError as error:
            raise _StructuredOutputValidationError(str(error)) from error
        for selection in selections:
            event = by_name[selection.event]
            raw = selection.data if selection.data is not None else {}
            # Validate pre-unpatch shape: re-merge meta into data for schema check when present.
            payload = dict(typing.cast(collections.abc.Mapping[str, object], raw))
            if selection.meta:
                payload.update(selection.meta)
            elif selection.confidence is not None and "confidence" not in payload:
                payload["confidence"] = selection.confidence
            _validate_event_payload(event, payload, patch=patch)
            merged.append(selection)
    return tuple(merged)


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
    targets_by_event: collections.abc.Mapping[str, collections.abc.Sequence[str]] | None = None,
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
                targets_by_event=targets_by_event,
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
            if targets_by_event is not None:
                selections = processing.fill_unique_selection_targets(selections, targets_by_event)
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


class Processor(processing.Processor):
    """Gemini transport: ``process`` maps a Processing input (+ stamped instructions, a system message when not blank) → events.

    ``image_capacity`` is stamped once from the generator, so the projection attaches exactly the
    images that generator accepts.
    """

    _generator: text.generation.TextGenerator
    image_capacity: mosfet.media.ImageCapacity

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
        self.image_capacity = generator.image_capacity

    @typing.override
    async def process(
        self, input: processing.InputData, *, drafts: processing.DraftSink | None = None
    ) -> processing.Events:
        # Blank instructions (inception's Processing stamps none) send no system message: the user message alone.
        instructions = (input.instructions or "").strip()
        operation_tools: tuple[dict[str, object], ...] = ()
        if input.schemas:
            operation_tools = (
                processing.dispatch_tool(
                    input.schemas,
                    patch=input.patch,
                    targets_by_event=input.actor_events,
                ),
            )
        generation_input = text.generation.InputData(
            messages=(
                *(
                    (text.generation.TextMessage(role=text.generation.TextRole.SYSTEM, content=instructions),)
                    if instructions
                    else ()
                ),
                text.generation.TextMessage(
                    role=text.generation.TextRole.USER,
                    content=input.model_facing_content(self.image_capacity),
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
                    targets_by_event=input.actor_events,
                )
            return _events_from_content(
                output.content,
                input.schemas,
                patch=input.patch,
                targets_by_event=input.actor_events,
            )
        except (pydantic.ValidationError, _StructuredOutputValidationError) as error:
            message = f"Gemini processing response did not produce a valid events array: {error}"
            raise ProcessingError(message) from error


__all__ = ["Processor", "ProcessingError"]
