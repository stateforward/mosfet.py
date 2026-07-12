"""Deliberative processing: select events, then dispatch them.

Topology::

    idle → applying → (choice) → idle
                     ↘ dispatching → idle

- **applying** — ``Processor.process``; completion carries input + events
- **choice** — no events → terminal empty product → idle; else → dispatching
- **dispatching** — fire selected events (concurrent); terminal product → idle
"""

from __future__ import annotations

from . import ability

import abc
import collections.abc
import dataclasses
import re
import typing

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot.event_schema import event_json_schema, event_schema_json_schema, validate_event_data
from bot.telemetry import observer

# Processing inputs offer live HSM events (not a parallel offer DTO).
Event = hsm.Event

_FOCUS_DEVICE_EVENT = "bot.focus_device"
_CLEAR_FOCUS_EVENT = "bot.clear_focus"

# Single model-facing tool: multi-select is an events array, not N parallel tools.
DISPATCH_TOOL_NAME: typing.Final[str] = "dispatch"

# Optional model-facing overlay: a BaseModel type whose fields are create_model-patched onto
# every offered event for this InputData only. ``None`` → pure domain schemas (no force patch).
SchemaPatch: typing.TypeAlias = type[pydantic.BaseModel]

CONFIDENCE_MIN: typing.Final[int] = 0
CONFIDENCE_MAX: typing.Final[int] = 100


@dataclasses.dataclass(frozen=True)
class SelectedEvent:
    """One selected event to dispatch.

    ``confidence`` / ``meta`` hold model-facing patch values lifted by ``unpatch_event_data``
    (not domain event data). ``meta`` is the full patch map; ``confidence`` is a convenience
    when the active patch includes that field (0–100 integer scale).
    """

    event: str
    target: str | None = None
    data: dict[str, object] | None = None
    reason: str | None = None
    confidence: int | None = None
    meta: dict[str, object] | None = None


Events: typing.TypeAlias = tuple[SelectedEvent, ...]


def _safe_model_name(event_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", event_name).strip("_")
    return (cleaned or "event") + "ModelFacing"


def _payload_base_model(event: Event[typing.Any]) -> type[pydantic.BaseModel] | None:
    schema = getattr(event, "schema", None)
    if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel):
        return schema
    return None


def _is_schema_patch(patch: object) -> typing.TypeGuard[SchemaPatch]:
    return isinstance(patch, type) and issubclass(patch, pydantic.BaseModel)


def patch_field_names(patch: SchemaPatch | None) -> frozenset[str]:
    """Field names overlaid by ``patch`` (empty when patch is None)."""

    if patch is None or not _is_schema_patch(patch):
        return frozenset()
    return frozenset(patch.model_fields)


def _patch_create_model_fields(patch: SchemaPatch) -> dict[str, tuple[object, pydantic.fields.FieldInfo]]:
    fields: dict[str, tuple[object, pydantic.fields.FieldInfo]] = {}
    for name, field_info in patch.model_fields.items():
        annotation: object = field_info.annotation if field_info.annotation is not None else object
        fields[name] = (annotation, field_info)
    return fields


def _example_value_for_patch_field(field_info: pydantic.fields.FieldInfo) -> object:
    examples = field_info.examples
    if isinstance(examples, list) and examples:
        return examples[0]
    if field_info.default is not None and field_info.default is not pydantic.fields.PydanticUndefined:
        return field_info.default
    annotation = field_info.annotation
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union:
        non_none = [item for item in args if item is not type(None)]
        if non_none:
            annotation = non_none[0]
    if annotation is int:
        return 86
    if annotation is float:
        return 0.86
    if annotation is bool:
        return True
    if annotation is str:
        return "example"
    return None


def _enrich_model_facing_schema(
    schema: dict[str, object],
    *,
    patch: SchemaPatch | None,
) -> dict[str, object]:
    """Merge patch field descriptions/examples into the projected event schema for the model."""

    if patch is None or not _is_schema_patch(patch) or not patch.model_fields:
        return schema

    notes: list[str] = []
    for name, field_info in patch.model_fields.items():
        description = field_info.description
        if isinstance(description, str) and description.strip():
            notes.append(f"Always set {name}: {description.strip()}")
        else:
            notes.append(f"Always set model-facing field {name!r} on this payload.")
    note = " ".join(notes)
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        lower = description.lower()
        if not any(name in lower for name in patch.model_fields):
            schema["description"] = f"{description.rstrip()} {note}"
    else:
        schema["description"] = note

    examples = schema.get("examples")
    if isinstance(examples, list) and examples:
        enriched: list[object] = []
        defaults = {
            name: _example_value_for_patch_field(field_info)
            for name, field_info in patch.model_fields.items()
        }
        for example in examples:
            if isinstance(example, dict):
                item = dict(typing.cast(dict[str, object], example))
                for name, value in defaults.items():
                    if name not in item and value is not None:
                        item[name] = value
                enriched.append(item)
            else:
                enriched.append(example)
        # Second object example using alternate field examples when available.
        first = enriched[0]
        if isinstance(first, dict):
            alt = dict(typing.cast(dict[str, object], first))
            for name, field_info in patch.model_fields.items():
                examples_list = field_info.examples
                if isinstance(examples_list, list) and len(examples_list) > 1:
                    alt[name] = examples_list[1]
            if alt not in enriched:
                enriched.append(alt)
        schema["examples"] = enriched
    return schema


def patched_event_data_model(
    event: Event[typing.Any],
    *,
    patch: SchemaPatch | None = None,
) -> type[pydantic.BaseModel]:
    """Return a temporary Pydantic model: domain payload + optional patch fields.

    Built with ``create_model`` so providers can project tool/parameters schemas. Never mutates
    the live HSM event schema. When ``patch`` is None, returns the domain model when possible.
    """

    name = _safe_model_name(event.name)
    base = _payload_base_model(event)
    if patch is None or not _is_schema_patch(patch) or not patch.model_fields:
        if base is not None:
            return base
        return typing.cast(
            type[pydantic.BaseModel],
            pydantic.create_model(name, __config__=pydantic.ConfigDict(extra="allow")),
        )

    patch_fields = _patch_create_model_fields(patch)
    if base is not None:
        for key in tuple(patch_fields):
            if key in base.model_fields:
                del patch_fields[key]
        if not patch_fields:
            return base
        return typing.cast(
            type[pydantic.BaseModel],
            pydantic.create_model(name, __base__=base, **patch_fields),
        )
    return typing.cast(
        type[pydantic.BaseModel],
        pydantic.create_model(
            name,
            __config__=pydantic.ConfigDict(extra="allow"),
            **patch_fields,
        ),
    )


def model_facing_event_json_schema(
    event: Event[typing.Any],
    *,
    patch: SchemaPatch | None = None,
) -> dict[str, object]:
    """JSON schema for one offered event as shown to the model (domain + optional patch)."""

    if patch is None:
        return event_json_schema(event)

    base = _payload_base_model(event)
    if base is not None or getattr(event, "schema", None) is None:
        return _enrich_model_facing_schema(
            event_schema_json_schema(patched_event_data_model(event, patch=patch)),
            patch=patch,
        )
    # Typed non-BaseModel payloads: merge patch field JSON into the projected schema.
    schema = dict(event_json_schema(event))
    if not schema:
        return _enrich_model_facing_schema(
            event_schema_json_schema(patched_event_data_model(event, patch=patch)),
            patch=patch,
        )
    if schema.get("type") not in (None, "object") and "properties" not in schema:
        return schema
    properties_raw = schema.get("properties")
    properties: dict[str, object]
    if isinstance(properties_raw, dict):
        properties = typing.cast(dict[str, object], properties_raw)
    else:
        properties = {}
        schema["properties"] = properties
        schema.setdefault("type", "object")
    patch_schema = event_schema_json_schema(patch)
    patch_properties = patch_schema.get("properties")
    names = patch_field_names(patch)
    if isinstance(patch_properties, dict):
        for key, value in typing.cast(dict[str, object], patch_properties).items():
            if key in names:
                properties[key] = value
    return _enrich_model_facing_schema(schema, patch=patch)


def normalize_confidence(value: object) -> int | None:
    """Coerce a model-facing confidence value to an int in ``[0, 100]``.

    Whole numbers are preferred. Floats in ``(0, 1]`` are treated as a 0–1 fraction and
    scaled to 0–100 (legacy / accidental fractional reports). Integer ``1`` stays ``1``.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and 0.0 < value <= 1.0:
        scaled = int(round(value * float(CONFIDENCE_MAX)))
        return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, scaled))
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, int(round(float(value)))))


def unpatch_event_data(
    data: object,
    *,
    patch: SchemaPatch | None = None,
) -> tuple[object, dict[str, object] | None]:
    """Strip model-facing patch fields from event data; return (domain data, patch values).

    When ``patch`` is None, data is returned unchanged (no forced strip). Non-mapping payloads
    (e.g. deliberative ``InputData`` frames) pass through with empty patch values.
    """

    if data is None:
        return None, None
    if not isinstance(data, collections.abc.Mapping):
        return data, None
    mapping = typing.cast(collections.abc.Mapping[str, object], data)
    names = patch_field_names(patch)
    if not names:
        return dict(mapping), None
    meta: dict[str, object] = {}
    cleaned: dict[str, object] = {}
    for key, value in mapping.items():
        if key in names:
            meta[key] = value
        else:
            cleaned[key] = value
    return (cleaned if cleaned else None), (meta if meta else None)


def selection_confidence(selections: Events) -> int | None:
    """Aggregate selection confidence (min of reported scores; conservative escalate)."""

    values = tuple(item.confidence for item in selections if item.confidence is not None)
    if not values:
        return None
    return min(values)


def dispatch_tool(
    events: collections.abc.Sequence[Event[typing.Any]],
    *,
    patch: SchemaPatch | None = None,
) -> dict[str, object]:
    """Build the single model-facing ``dispatch`` function tool.

    Multi-select is one tool call with ``events: [...]`` using canonical HSM event names
    (no per-event tool name mangling). Offered payload shapes stay in the user message /
    schema projection; item ``data`` is validated after the call.
    """

    names = tuple(dict.fromkeys(event.name for event in events if event.name))
    offered = ", ".join(names) if names else "(none)"
    patch_note = ""
    if patch is not None and _is_schema_patch(patch) and patch.model_fields:
        field_bits: list[str] = []
        for field_name, field_info in patch.model_fields.items():
            description = field_info.description
            if isinstance(description, str) and description.strip():
                field_bits.append(f"{field_name}: {description.strip()}")
            else:
                field_bits.append(field_name)
        patch_note = (
            " Each selected event's data must also include these model-facing fields: "
            + "; ".join(field_bits)
            + "."
        )
    description = (
        "Dispatch zero or more modeled events for this turn. "
        "Call this function once. Put every event that should run in the events array "
        "(multi-select is normal—for example speaking.input together with reasoning.input "
        "when the user is waiting on speech while deliberation continues). "
        f"Allowed event names: {offered}. "
        "Use the exact canonical event string. data must match that event's payload schema "
        "from the offered schemas in the user message."
        f"{patch_note} "
        "Return events: [] when no offered event should run."
    )
    event_property: dict[str, object] = {
        "type": "string",
        "description": (
            "Canonical HSM event name to dispatch. Must be one of the offered schema names."
        ),
        "examples": list(names[:3]) if names else ["phone.answer_call"],
    }
    if names:
        event_property["enum"] = list(names)
    parameters: dict[str, object] = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "description": (
                    "Ordered list of events to dispatch this turn. Include multiple items when "
                    "several actions should run together. Empty array selects none."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "event": event_property,
                        "data": {
                            "type": "object",
                            "description": (
                                "JSON object payload for the selected event. Match the schema for "
                                "that event name (and any patched fields such as confidence). "
                                "Omit or use {} when the event has no payload fields."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                "Optional actor name that should receive the event when more than "
                                "one actor can accept it."
                            ),
                            "examples": ["speaking", "phone", "bot"],
                        },
                        "reason": {
                            "type": "string",
                            "description": "Optional short reason this event was selected.",
                            "examples": ["User greeted; reply now while deliberating keypad request."],
                        },
                    },
                    "required": ["event"],
                    "additionalProperties": False,
                },
                "examples": [
                    [],
                    [
                        {
                            "event": names[0] if names else "bot.ability.speaking.input",
                            "data": {"text": "One moment.", "confidence": 86}
                            if patch is not None
                            else {"text": "One moment."},
                            "reason": "Acknowledge while handling a hard request.",
                        }
                    ],
                ],
            }
        },
        "required": ["events"],
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": DISPATCH_TOOL_NAME,
            "description": description,
            "parameters": parameters,
        },
    }


def events_from_dispatch_args(
    args: collections.abc.Mapping[str, object],
    *,
    patch: SchemaPatch | None = None,
    offered: collections.abc.Sequence[Event[typing.Any]] | None = None,
) -> Events:
    """Parse a ``dispatch`` tool's args into validated ``Events``."""

    raw_events = args.get("events")
    if raw_events is None:
        raise ValueError("dispatch tool args must include an events array.")
    selections = coerce_event_selections(raw_events, patch=patch)
    if selections is None:
        raise ValueError("dispatch tool events must be an array of event selections.")
    if offered is not None:
        allowed = {event.name for event in offered}
        for item in selections:
            if item.event not in allowed:
                raise ValueError(f"dispatch selected unavailable event: {item.event}.")
    return selections


def _confidence_from_meta(meta: dict[str, object] | None) -> int | None:
    if not meta or "confidence" not in meta:
        return None
    return normalize_confidence(meta.get("confidence"))


@dataclasses.dataclass(frozen=True)
class Result(typing.Generic[ability.TOutput]):
    """Explicit decline so a later ability/operation may run (composition only)."""

    is_handled: bool
    output: ability.TOutput | None = None

    @classmethod
    def handled(cls, output: ability.TOutput) -> "Result[ability.TOutput]":
        return cls(is_handled=True, output=output)

    @classmethod
    def unhandled(cls) -> "Result[ability.TOutput]":
        return cls(is_handled=False)


class InputData(pydantic.BaseModel):
    """Processing input: stimulus plus selectable event schemas."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "description": (
                "Processing input: stimulus payload plus HSM event schemas available for selection. "
                "Processor output is always an array of selected events."
            ),
            "examples": [{"input": "incoming phone speech", "schemas": ["phone.answer_call"]}],
        },
    )

    input: object = pydantic.Field(
        description="Incoming stimulus event or payload.",
        examples=["incoming phone speech"],
    )
    schemas: SkipJsonSchema[tuple[Event[typing.Any], ...]] = pydantic.Field(
        default=(),
        exclude=True,
        repr=False,
        description="Selectable HSM call events (JSON schemas projected for the model).",
    )
    actors: SkipJsonSchema[collections.abc.Mapping[str, hsm.Instance]] = pydantic.Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description="Named instances used only to dispatch selected events (not model-facing).",
    )
    instructions: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description=(
            "Optional system policy stamped by Processing for model-facing processors. "
            "Not part of the stimulus; providers use this as the system prompt when present."
        ),
        examples=["Return an immediate typed result only when the input is already unambiguous."],
    )
    patch: SkipJsonSchema[SchemaPatch | None] = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Optional model-facing BaseModel type whose fields are overlaid on every offered "
            "event schema for this input only (create_model patch). None means pure domain "
            "schemas. Different processing abilities stamp different patches (e.g. intuition "
            "adds confidence; reasoning may leave this None)."
        ),
    )

    @pydantic.model_serializer(mode="wrap")
    def _serialize_model(self, serializer: typing.Callable[[typing.Any], dict[str, object]]) -> dict[str, object]:
        del serializer
        projected: list[dict[str, object]] = []
        for event in self.schemas:
            schema = model_facing_event_json_schema(event, patch=self.patch)
            item: dict[str, object] = {"event": event.name, "schema": schema}
            description = schema.get("description")
            if isinstance(description, str) and description:
                item["description"] = description
            projected.append(item)
        # Stimulus + tools only; instructions are system policy for the provider, not user content.
        return {"input": self.input, "schemas": projected}


_MISSING = object()


def coerce_event_selections(
    output: object,
    *,
    patch: SchemaPatch | None = None,
) -> Events | None:
    """Normalize process product to ``Events``; return None when not event-shaped.

    When ``patch`` is set, strip those fields from event data into SelectedEvent.meta / confidence.
    """

    if output is None:
        return None
    if isinstance(output, Result):
        if not output.is_handled or output.output is None:
            return None
        return coerce_event_selections(output.output, patch=patch)
    # Ability envelopes (intuition/reasoning OutputData) carry events on ``result``.
    result = getattr(output, "result", _MISSING)
    if result is not _MISSING and result is not output:
        return coerce_event_selections(result, patch=patch)
    if isinstance(output, SelectedEvent):
        one = _one_selection(output, patch=patch)
        return (one,) if one is not None else None
    if isinstance(output, collections.abc.Sequence) and not isinstance(output, str | bytes | bytearray):
        if len(output) == 0:
            return ()
        items: list[SelectedEvent] = []
        for item in output:
            one = _one_selection(item, patch=patch)
            if one is None:
                return None
            items.append(one)
        return tuple(items)
    one = _one_selection(output, patch=patch)
    return (one,) if one is not None else None


def _as_event_data_dict(data: object) -> dict[str, object] | None:
    if data is None:
        return None
    if isinstance(data, collections.abc.Mapping):
        return dict(typing.cast(collections.abc.Mapping[str, object], data))
    return None


def _one_selection(
    value: object,
    *,
    patch: SchemaPatch | None = None,
) -> SelectedEvent | None:
    if isinstance(value, SelectedEvent):
        domain, meta = unpatch_event_data(value.data, patch=patch)
        if isinstance(domain, collections.abc.Mapping):
            domain_data = _as_event_data_dict(domain)
        elif domain is None and value.data is not None and not isinstance(value.data, collections.abc.Mapping):
            domain_data = value.data
        else:
            domain_data = None if domain is None else value.data
        resolved_meta = value.meta if value.meta is not None else meta
        resolved_confidence = (
            value.confidence if value.confidence is not None else _confidence_from_meta(resolved_meta)
        )
        if (
            domain_data == value.data
            and resolved_confidence == value.confidence
            and resolved_meta == value.meta
        ):
            return value
        return dataclasses.replace(
            value,
            data=domain_data,
            confidence=resolved_confidence,
            meta=resolved_meta,
        )
    if isinstance(value, collections.abc.Mapping):
        event = value.get("event")
        if not isinstance(event, str) or not event:
            return None
        target = value.get("target")
        data = value.get("data")
        reason = value.get("reason")
        data_map = _as_event_data_dict(data)
        domain, meta = unpatch_event_data(data_map, patch=patch)
        domain_data = _as_event_data_dict(domain)
        if meta is None and patch is None:
            # Top-level confidence without an active patch still lifts for convenience.
            top_conf = normalize_confidence(value.get("confidence"))
            if top_conf is not None:
                meta = {"confidence": top_conf}
        confidence = _confidence_from_meta(meta)
        if confidence is None:
            confidence = normalize_confidence(value.get("confidence"))
        return SelectedEvent(
            event=event,
            target=target if isinstance(target, str) else None,
            data=domain_data,
            reason=reason if isinstance(reason, str) else None,
            confidence=confidence,
            meta=meta,
        )
    event = getattr(value, "event", None)
    if not isinstance(event, str) or not event:
        return None
    target = getattr(value, "target", None)
    data = getattr(value, "data", None)
    reason = getattr(value, "reason", None)
    existing_confidence = getattr(value, "confidence", None)
    existing_meta = getattr(value, "meta", None)
    if data is None:
        data_map = None
    elif isinstance(data, collections.abc.Mapping):
        data_map = _as_event_data_dict(data)
    elif hasattr(data, "model_dump"):
        dumped = data.model_dump(mode="json")
        data_map = dict(dumped) if isinstance(dumped, dict) else None
    else:
        data_map = None
    domain, meta = unpatch_event_data(data_map, patch=patch)
    domain_data = _as_event_data_dict(domain)
    if isinstance(existing_meta, dict) and meta is None:
        meta = dict(typing.cast(dict[str, object], existing_meta))
    confidence = _confidence_from_meta(meta)
    if confidence is None:
        confidence = normalize_confidence(existing_confidence)
    return SelectedEvent(
        event=event,
        target=target if isinstance(target, str) else None,
        data=domain_data,
        reason=reason if isinstance(reason, str) else None,
        confidence=confidence,
        meta=meta,
    )


def _instance_event_map(instance: hsm.Instance) -> dict[str, Event[typing.Any]]:
    mapped: dict[str, Event[typing.Any]] = {}
    for model in (getattr(instance, "model", None), getattr(instance, "firmware_model", None)):
        raw = getattr(model, "events", None)
        if not isinstance(raw, collections.abc.Mapping):
            continue
        for name, event in raw.items():
            if isinstance(event, hsm.Event):
                mapped[str(name)] = event
    return mapped


def enabled_call_events(instance: hsm.Instance) -> tuple[Event[typing.Any], ...]:
    """Enabled CallEventKind events on ``instance`` from its current transition snapshot."""

    event_map = _instance_event_map(instance)
    offered: list[Event[typing.Any]] = []
    seen: set[str] = set()
    snapshot = instance.take_snapshot()
    for transition in snapshot.Transitions:
        for event_name in transition.events:
            if event_name in seen:
                continue
            event = event_map.get(event_name)
            if event is None or event.kind != hsm.CallEventKind:
                continue
            seen.add(event_name)
            offered.append(event)
    return tuple(offered)


def _enabled_call_event_names(instance: hsm.Instance) -> set[str]:
    return {event.name for event in enabled_call_events(instance)}


def _unavailable_message(*, event: str, target: str | None = None) -> str:
    if target is None:
        return f"Processing selected unavailable event: {event}."
    return f"Processing selected unavailable event for target {target}: {event}."


def _resolve_target(input: InputData, selection: SelectedEvent) -> hsm.Instance:
    if selection.target is not None:
        instance = input.actors.get(selection.target)
        if instance is None:
            raise RuntimeError(_unavailable_message(event=selection.event, target=selection.target))
        return instance
    matches = [
        name
        for name, instance in input.actors.items()
        if selection.event in _enabled_call_event_names(instance)
    ]
    if len(matches) == 1:
        return input.actors[matches[0]]
    if len(input.actors) == 1:
        return next(iter(input.actors.values()))
    raise RuntimeError(_unavailable_message(event=selection.event))


def dispatch_selected_events(
    ctx: hsm.Context,
    input: InputData,
    selections: Events,
    *,
    metadata: collections.abc.Mapping[str, object] | None = None,
) -> None:
    """Validate selections and dispatch them fire-and-forget (all issued before return)."""

    if not selections:
        return
    by_name = {event.name: event for event in input.schemas}
    event_metadata = dict(metadata or {})

    for selection in selections:
        if selection.event not in by_name:
            raise RuntimeError(_unavailable_message(event=selection.event, target=selection.target))
        # Domain validate never sees model-facing patches; strip using this input's patch type.
        domain_data, _meta = unpatch_event_data(selection.data, patch=input.patch)
        if domain_data is None:
            raw: object = {}
        elif isinstance(domain_data, collections.abc.Mapping):
            raw = dict(typing.cast(collections.abc.Mapping[str, object], domain_data))
        else:
            # Live deliberative frames (processing.InputData) and other non-mapping payloads.
            raw = domain_data

        # Body focus events resolve to the bot actor, not via enabled CallEvent discovery.
        # Handle them before _resolve_target so extra actors (e.g. reasoning) do not break focus.
        if selection.event == _FOCUS_DEVICE_EVENT:
            import bot as bot_mod
            from bot.device import Device

            if selection.target is not None and selection.target != "bot":
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
            data = bot_mod.FocusDeviceEventData.model_validate(raw)
            # Only real devices — sibling abilities on the actor map (e.g. reasoning) are not focus targets.
            candidates = tuple(
                name for name, instance in input.actors.items() if name != "bot" and isinstance(instance, Device)
            )
            meta_candidates = event_metadata.get("bot.focus_candidates")
            if isinstance(meta_candidates, collections.abc.Sequence) and not isinstance(
                meta_candidates, str | bytes | bytearray
            ):
                restricted = tuple(item for item in meta_candidates if isinstance(item, str) and item)
                if restricted:
                    candidates = restricted
            if not candidates:
                continue
            if data.device not in candidates:
                raise RuntimeError("Processing selected focus_device outside available device candidates.")
            bot_target = input.actors.get("bot")
            if bot_target is None:
                raise RuntimeError(_unavailable_message(event=selection.event, target="bot"))
            _ = bot_target.dispatch(
                ctx,
                dataclasses.replace(bot_mod.FocusDeviceEvent.with_data(data), metadata=event_metadata),
            )
            continue

        if selection.event == _CLEAR_FOCUS_EVENT:
            import bot as bot_mod
            if selection.target is not None and selection.target != "bot":
                raise RuntimeError("Processing selected clear_focus outside available device candidates.")
            data = bot_mod.ClearFocusEventData.model_validate(raw)
            bot_target = input.actors.get("bot")
            if bot_target is None:
                raise RuntimeError(_unavailable_message(event=selection.event, target="bot"))
            _ = bot_target.dispatch(
                ctx,
                dataclasses.replace(bot_mod.ClearFocusEvent.with_data(data), metadata=event_metadata),
            )
            continue

        target = _resolve_target(input, selection)
        live = _instance_event_map(target).get(selection.event)
        if live is None or live.kind != hsm.CallEventKind:
            raise RuntimeError(_unavailable_message(event=selection.event, target=selection.target))
        if selection.event not in _enabled_call_event_names(target):
            raise RuntimeError(_unavailable_message(event=selection.event, target=selection.target))

        try:
            validated = validate_event_data(live, raw)
        except Exception as error:
            raise RuntimeError(f"Processing selected invalid event data for event: {selection.event}.") from error
        dispatch_event = live if validated is None else live.with_data(validated)
        dispatch_event = dataclasses.replace(dispatch_event, metadata=event_metadata)
        _ = target.dispatch(ctx, dispatch_event)


# --- private completion payloads / events ---


@dataclasses.dataclass(frozen=True)
class _AppliedData:
    """Applying product: input needed for dispatch + events (HSM-COMPLETION-001)."""

    input: InputData
    events: Events
    unhandled: bool = False


_AppliedEvent = hsm.Event[object](
    name="bot.ability.processing.applied",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)
_DispatchedEvent = hsm.Event[object](
    name="bot.ability.processing.dispatched",
    kind=hsm.CompletionEventKind,
    schema=pydantic.TypeAdapter(object),
)
_FailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.processing.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


class Processor(abc.ABC):
    """Like Encoder / Decoder: map input to selected events."""

    @abc.abstractmethod
    def process(self, input: InputData) -> collections.abc.Awaitable[Events]:
        """Return selected events (empty = none)."""
        ...


class Processing(ability.Ability[InputData, Events]):
    """Leaf ability: apply an injected ``Processor``, then dispatch selected events if any.

    Model-facing system policy lives here (``instructions``), not on the provider Processor.
    On apply, non-blank instructions are stamped onto the input before ``processor.process``.
    """

    processor: Processor
    instructions: typing.ClassVar[str] = ""
    _instructions: str
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = object
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = ability.ability_input_event(
        name="bot.ability.processing.input",
        data_type=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = ability.ability_output_event(
        name="bot.ability.processing.output",
        data_type=object,
        description="Selected events array produced by processing.",
        examples=[[{"event": "phone.answer_call"}]],
    )

    # Aliases for older callers (Intuition, tests) that still name these "apply completed".
    _apply_completed_event: typing.ClassVar[hsm.Event[object]] = _DispatchedEvent
    _apply_failed_event: typing.ClassVar[hsm.Event[ability.FailureData]] = _FailedEvent

    @staticmethod
    def _has_input(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _has_applied(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _AppliedData)

    @staticmethod
    def _applied_has_events(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _AppliedData) and bool(event.data.events) and not event.data.unhandled

    @staticmethod
    def _applied_is_unhandled(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _AppliedData) and event.data.unhandled

    @staticmethod
    def _has_failure(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, ability.FailureData)

    @staticmethod
    def _emit_output(
        ctx: hsm.Context,
        instance: "Processing",
        event: hsm.Event[typing.Any],
        output: object,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))

    @staticmethod
    def _emit_failure(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, ability.FailureData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))

    @staticmethod
    def _complete_empty(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        """No events selected: terminal empty product, skip dispatching."""

        Processing._emit_output(ctx, instance, event, ())

    @staticmethod
    def _complete_unhandled(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        """Processor declined the input (composition)."""

        Processing._emit_output(ctx, instance, event, Result[Events].unhandled())

    @staticmethod
    def _complete_dispatched(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        Processing._emit_output(ctx, instance, event, event.data)

    @staticmethod
    def _input_for_processor(instance: "Processing", input: InputData) -> InputData:
        """Stamp Processing instructions onto the input for model-facing processors."""

        instructions = instance._instructions.strip()
        if not instructions:
            return input
        if input.instructions == instructions:
            return input
        return input.model_copy(update={"instructions": instructions})

    @staticmethod
    async def _apply_activity(
        ctx: hsm.Context,
        instance: "Processing",
        event: hsm.Event[InputData],
    ) -> None:
        """applying: run process (via ``_apply`` so subclasses may override)."""

        input = Processing._input_for_processor(instance, typing.cast(InputData, event.data))
        try:
            raw = await instance.processor.process(input)
            if isinstance(raw, Result) and not raw.is_handled:
                applied = _AppliedData(input=input, events=(), unhandled=True)
            else:
                payload = raw.output if isinstance(raw, Result) else raw
                coerced = coerce_event_selections(payload, patch=input.patch)
                if coerced is None:
                    raise TypeError("Processor must return an array of events.")
                applied = _AppliedData(input=input, events=coerced, unhandled=False)
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _FailedEvent.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _AppliedEvent.with_data(applied),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _dispatch_activity(
        ctx: hsm.Context,
        instance: "Processing",
        event: hsm.Event[typing.Any],
    ) -> None:
        """dispatching: fire all selected events concurrently when actors exist."""

        data = event.data
        assert isinstance(data, _AppliedData)
        try:
            # No actors: product only (e.g. reflection write phases).
            if data.input.actors:
                dispatch_selected_events(
                    ctx,
                    data.input,
                    data.events,
                    metadata=dict(event.metadata),
                )
        except Exception as error:
            _ = hsm.dispatch(
                ctx,
                instance,
                dataclasses.replace(
                    _FailedEvent.with_data(ability.FailureData(message=str(error))),
                    id=event.id or None,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = hsm.dispatch(
            ctx,
            instance,
            dataclasses.replace(
                _DispatchedEvent.with_data(data.events),
                id=event.id or None,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = hsm.define(
        "Processing",
        hsm.initial(hsm.target("/Processing/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(input_event),
                hsm.guard(_has_input),
                hsm.target("/Processing/applying"),
            ),
        ),
        hsm.state(
            "applying",
            hsm.defer(input_event),
            hsm.activity(_apply_activity),
            hsm.transition(
                hsm.on(_AppliedEvent),
                hsm.guard(_has_applied),
                hsm.target("/Processing/routing"),
            ),
            hsm.transition(
                hsm.on(_FailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_emit_failure),
                hsm.target("/Processing/idle"),
            ),
        ),
        hsm.choice(
            "routing",
            hsm.transition(
                hsm.guard(_applied_is_unhandled),
                hsm.effect(_complete_unhandled),
                hsm.target("/Processing/idle"),
            ),
            hsm.transition(
                hsm.guard(_applied_has_events),
                hsm.target("/Processing/dispatching"),
            ),
            # Default: empty events → terminal () without dispatching.
            hsm.transition(
                hsm.effect(_complete_empty),
                hsm.target("/Processing/idle"),
            ),
        ),
        hsm.state(
            "dispatching",
            hsm.defer(input_event),
            hsm.activity(_dispatch_activity),
            hsm.transition(
                hsm.on(_DispatchedEvent),
                hsm.effect(_complete_dispatched),
                hsm.target("/Processing/idle"),
            ),
            hsm.transition(
                hsm.on(_FailedEvent),
                hsm.guard(_has_failure),
                hsm.effect(_emit_failure),
                hsm.target("/Processing/idle"),
            ),
        ),
        hsm.observe(observer),
    )

    def __init__(self, *, processor: Processor, instructions: str | None = None) -> None:
        super().__init__()
        resolved = type(self).instructions if instructions is None else instructions
        if instructions is not None and not instructions.strip():
            raise ValueError("instructions must not be blank when provided.")
        self.processor = processor
        self._instructions = resolved.strip() if resolved else ""


InputEvent = Processing.input_event
OutputEvent = Processing.output_event

__all__ = [
    "CONFIDENCE_MAX",
    "CONFIDENCE_MIN",
    "DISPATCH_TOOL_NAME",
    "Event",
    "Events",
    "InputEvent",
    "OutputEvent",
    "InputData",
    "Processor",
    "Processing",
    "Result",
    "SchemaPatch",
    "SelectedEvent",
    "coerce_event_selections",
    "dispatch_selected_events",
    "dispatch_tool",
    "enabled_call_events",
    "events_from_dispatch_args",
    "model_facing_event_json_schema",
    "normalize_confidence",
    "patch_field_names",
    "patched_event_data_model",
    "selection_confidence",
    "unpatch_event_data",
]
