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
import asyncio
import collections.abc
import dataclasses
import enum
import json
import re
import typing
import uuid
import weakref

import bot
from bot import lifecycle
import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema

from bot.event import (
    EventKind,
    embeddable_json_schema,
    event_json_value,
    event_json_schema,
    event_schema_json_schema,
    validate_event_data,
)
from .cognition.event import model_facing_xml
from bot.telemetry import observer

# Processing inputs offer live HSM events (not a parallel offer DTO).
Event = hsm.Event


class DispatchTrust(enum.StrEnum):
    """Trust policy for selected event data crossing the dispatch boundary."""

    MODEL = "model"
    TRUSTED_BEHAVIOR = "trusted_behavior"


_MAX_REJECTION_MESSAGE_LENGTH: typing.Final[int] = 2_048
_MAX_VALIDATION_ERRORS: typing.Final[int] = 8
_MAX_DIAGNOSTIC_FRAGMENT_LENGTH: typing.Final[int] = 256
_MAX_MODEL_PROMPT_BYTES: typing.Final[int] = 262_144
_MAX_MODEL_PROMPT_SCALAR_CHARACTERS: typing.Final[int] = 65_536
_URL_PATTERN = re.compile(r'(?i)(?:\b(?:https?|ftp|mailto):|(?<!\w)www\.)[^\s<>"\']+')
_INPUT_VALUE_PATTERN = re.compile(r"(?i)\binput_value\s*=\s*.*?(?=,\s*input_type\s*=|$)")
_SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(?:password|passwd|secret|token|api[_ -]?key|authorization|cookie|credential|private[_ -]?key)\b"
    + r"\s*[:=]\s*[\"']?[^,\s\"']+"
)
_SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?i)(?:password|passwd|secret|token|api[_ -]?key|authorization|cookie|credential|private[_ -]?key)"
)


def _safe_diagnostic_text(value: object, *, max_length: int) -> str:
    """Normalize untrusted diagnostic text without carrying payload values or URLs."""

    text = " ".join(str(value).split())
    text = _INPUT_VALUE_PATTERN.sub("[redacted]", text)
    text = _SENSITIVE_ASSIGNMENT_PATTERN.sub("[redacted]", text)
    text = _URL_PATTERN.sub("[redacted-url]", text)
    text = text.replace("input_value", "[redacted]")
    if len(text) > max_length:
        return f"{text[: max_length - 1]}…"
    return text


def _validation_details(error: pydantic.ValidationError) -> tuple[str, ...]:
    details: list[str] = []
    error_items = error.errors()
    for item in error_items[:_MAX_VALIDATION_ERRORS]:
        location = item.get("loc", ())
        if isinstance(location, (str, bytes)):
            location_parts = (location,)
        else:
            location_parts = typing.cast(collections.abc.Iterable[object], location)
        path = ".".join(_safe_diagnostic_text(part, max_length=64) for part in location_parts) or "<root>"
        error_type = _safe_diagnostic_text(item.get("type", "validation_error"), max_length=64)
        if _SENSITIVE_FIELD_PATTERN.search(path):
            message = "[redacted validation message]"
        else:
            message = _safe_diagnostic_text(
                item.get("msg", "validation error"),
                max_length=_MAX_DIAGNOSTIC_FRAGMENT_LENGTH,
            )
        details.append(f"{path}: {error_type}: {message}")
    if len(error_items) > _MAX_VALIDATION_ERRORS:
        details.append("[additional validation errors omitted]")
    return tuple(details)


class SelectionRejectionError(RuntimeError):
    """A model-selected event was rejected before delivery to its recipient.

    The complete message is intended for model repair feedback. ``normalized`` deliberately
    excludes dynamic validation details such as Pydantic's ``input_value`` so consecutive
    equivalent rejections can be compared without losing the original diagnostic text.
    """

    normalized: str

    def __init__(self, message: str, *, normalized: str | None = None) -> None:
        safe_message = _safe_diagnostic_text(message, max_length=_MAX_REJECTION_MESSAGE_LENGTH)
        super().__init__(safe_message)
        self.normalized = _safe_diagnostic_text(
            safe_message if normalized is None else normalized,
            max_length=_MAX_REJECTION_MESSAGE_LENGTH,
        )

    @classmethod
    def from_validation(
        cls,
        *,
        event_name: str,
        prefix: str,
        error: Exception,
    ) -> "SelectionRejectionError":
        if isinstance(error, pydantic.ValidationError):
            details = _validation_details(error)
            diagnostic = "; ".join(details) or "<root>: validation_error: validation error"
            normalized_details = tuple(detail.replace(": ", ":", 2) for detail in details)
            normalized_diagnostic = "|".join((event_name, *normalized_details))
        else:
            diagnostic = (
                f"{type(error).__name__}: {_safe_diagnostic_text(error, max_length=_MAX_DIAGNOSTIC_FRAGMENT_LENGTH)}"
            )
            normalized_diagnostic = f"{event_name}|{diagnostic}"
        message = f"{prefix}: {_safe_diagnostic_text(diagnostic, max_length=_MAX_REJECTION_MESSAGE_LENGTH)}"
        normalized = f"{prefix}: {normalized_diagnostic}"
        return cls(message, normalized=normalized)


# Single model-facing tool: multi-select is an events array, not N parallel tools.
DISPATCH_TOOL_NAME: typing.Final[str] = "dispatch"

# Optional model-facing overlay: a BaseModel type whose fields are create_model-patched onto
# every offered event for this InputData only. ``None`` → pure domain schemas (no force patch).
SchemaPatch: typing.TypeAlias = type[pydantic.BaseModel]

CONFIDENCE_MIN: typing.Final[int] = 0
CONFIDENCE_MAX: typing.Final[int] = 100


def is_deliberative_handoff_schema(schema: object | None) -> bool:
    """True when a model-facing schema marks deliberate (System-2) handoff.

    Schemas opt in with ``__deliberative_handoff__ = True`` (for example reasoning
    ``CallData``). Intuition and other stages detect handoff via this marker so they
    never hard-import the reasoning module.
    """

    return getattr(schema, "__deliberative_handoff__", False) is True


@dataclasses.dataclass(frozen=True)
class SelectedEvent:
    """One selected event to dispatch.

    ``confidence`` / ``meta`` hold model-facing patch values lifted by ``unpatch_event_data``
    (not domain event data). ``meta`` is the full patch map; ``confidence`` is a convenience
    when the active patch includes that field (0–100 integer scale). Selection rationale is
    ``reason`` on this envelope — not on the event payload.
    """

    event: str
    target: str | None = None
    # Domain payload: JSON dict from model/Starlark selections, or a typed event data model
    # (e.g. TurnData with audio bytes) when rebuilt from a live stimulus.
    data: object | None = None
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


def _patch_create_model_fields(patch: SchemaPatch) -> dict[str, typing.Any]:
    """Build create_model field kwargs from a patch model (fresh Field, not borrowed FieldInfo)."""

    fields: dict[str, typing.Any] = {}
    for name, field_info in patch.model_fields.items():
        annotation: object = field_info.annotation if field_info.annotation is not None else object
        kwargs: dict[str, typing.Any] = {}
        if field_info.description is not None:
            kwargs["description"] = field_info.description
        if field_info.examples is not None:
            kwargs["examples"] = field_info.examples
        if field_info.is_required():
            # Required patch fields stay required on the projected model.
            pass
        elif field_info.default is not pydantic.fields.PydanticUndefined:
            kwargs["default"] = field_info.default
        elif field_info.default_factory is not None:
            kwargs["default_factory"] = field_info.default_factory
        metadata = list(field_info.metadata)
        for item in metadata:
            ge = getattr(item, "ge", None)
            le = getattr(item, "le", None)
            if ge is not None:
                kwargs["ge"] = ge
            if le is not None:
                kwargs["le"] = le
        fields[name] = (annotation, pydantic.Field(**kwargs))
    return fields


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
    """JSON schema for one offered event payload as shown to the model.

    Required fields, descriptions, and examples come only from the event's Pydantic/schema
    contract (and an optional faculty ``patch`` BaseModel via create_model). Processing does
    not invent or rewrite domain schema text here.
    """

    if patch is None:
        return event_json_schema(event)

    base = _payload_base_model(event)
    if base is not None or getattr(event, "schema", None) is None:
        # Domain BaseModel (+ patch fields) projected as-is from Pydantic.
        return event_schema_json_schema(patched_event_data_model(event, patch=patch))
    # Typed non-BaseModel payloads: merge patch field JSON into the projected domain schema.
    schema = dict(event_json_schema(event))
    if not schema:
        return event_schema_json_schema(patched_event_data_model(event, patch=patch))
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
    return schema


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


def _object_schema_required_names(schema: collections.abc.Mapping[str, object]) -> list[str]:
    required = schema.get("required")
    if isinstance(required, list):
        return [item for item in required if isinstance(item, str)]
    return []


def collect_offered_events(
    actors: collections.abc.Mapping[str, hsm.Instance],
) -> tuple[tuple[Event[typing.Any], ...], collections.abc.Mapping[str, tuple[str, ...]]]:
    """Collect enabled call events and the actor keys that enable each event name.

    Walks ``actors.items()`` so provenance is live topology keys, never hard-coded product
    names. One canonical ``Event`` object is kept per name (first enabler). Actor keys per
    name are sorted for stable tool enums.
    """

    events_by_name: dict[str, Event[typing.Any]] = {}
    targets_by_name: dict[str, list[str]] = {}
    for actor_key, instance in actors.items():
        for event in enabled_call_events(instance):
            if not event.name:
                continue
            if event.name not in events_by_name:
                events_by_name[event.name] = event
            targets_by_name.setdefault(event.name, []).append(actor_key)
    actor_events = {name: tuple(sorted(set(keys))) for name, keys in targets_by_name.items()}
    return tuple(events_by_name.values()), actor_events


def _normalized_targets(targets: collections.abc.Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({key for key in targets if key}))


def _selection_item_branch(
    event: Event[typing.Any],
    *,
    patch: SchemaPatch | None = None,
    targets: collections.abc.Sequence[str] = (),
) -> dict[str, object]:
    """One anyOf branch: const event name + projected event payload schema as ``data``.

    Payload required/description/examples come only from the event (and optional patch) models.
    Nested model ``$defs``/``$ref`` from Pydantic are closed via ``embeddable_json_schema`` so
    document-root ``#/$defs/…`` refs remain valid after this branch is nested under ``dispatch``.

    ``target`` is stamped from live topology keys that enable this event this turn:
    single enabler → JSON Schema ``const`` (required); multiple → ``enum`` of those keys
    (required). Free-form target strings are never offered. Domain event payloads stay free of
    routing target.
    """

    data_schema = model_facing_event_json_schema(event, patch=patch)
    if not data_schema:
        data_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    elif data_schema.get("type") not in (None, "object") and "properties" not in data_schema:
        data_schema = {
            "type": "object",
            "properties": {"value": embeddable_json_schema(data_schema)},
            "required": ["value"],
            "additionalProperties": False,
        }
    else:
        data_schema = embeddable_json_schema(data_schema)
        data_schema.setdefault("type", "object")
        data_schema.setdefault("additionalProperties", False)

    # Branch description is the event schema's own description when present.
    event_description = data_schema.get("description")
    if not isinstance(event_description, str) or not event_description.strip():
        event_description = event.name

    data_required = _object_schema_required_names(data_schema)
    item_required = ["event", "data"] if data_required else ["event"]

    properties: dict[str, object] = {
        "event": {
            "type": "string",
            "const": event.name,
        },
        "data": data_schema,
        "reason": {
            "type": "string",
            "description": "Optional short reason this event was selected.",
        },
    }
    legal_targets = _normalized_targets(targets)
    if len(legal_targets) == 1:
        properties["target"] = {
            "type": "string",
            "const": legal_targets[0],
            "description": (
                "Actor that receives this event. Fixed for this turn because only one actor "
                "enables it in the live topology."
            ),
        }
        item_required.append("target")
    elif len(legal_targets) > 1:
        properties["target"] = {
            "type": "string",
            "enum": list(legal_targets),
            "description": (
                "Actor that should receive the event. Required because multiple actors enable "
                "this event this turn; choose exactly one offered actor key."
            ),
        }
        item_required.append("target")

    branch: dict[str, object] = {
        "type": "object",
        "description": event_description,
        "properties": properties,
        "required": item_required,
        "additionalProperties": False,
    }
    data_examples = data_schema.get("examples")
    if isinstance(data_examples, list) and data_examples:
        first = data_examples[0]
        if isinstance(first, dict):
            example: dict[str, object] = {"event": event.name, "data": first}
            if len(legal_targets) == 1:
                example["target"] = legal_targets[0]
            branch["examples"] = [example]
    return branch


def dispatch_tool(
    events: collections.abc.Sequence[Event[typing.Any]],
    *,
    patch: SchemaPatch | None = None,
    targets_by_event: collections.abc.Mapping[str, collections.abc.Sequence[str]] | None = None,
) -> dict[str, object]:
    """Build the single model-facing ``dispatch`` function tool.

    Structural only: one tool, ``events`` array, anyOf item per offered event. Each branch's
    ``data`` is the embeddable (ref-closed) projection of the event payload schema
    (Pydantic/event contract + optional patch). Composition never nests document-root
    ``#/$defs/…`` refs under the tool parameters document.

    ``targets_by_event`` maps event name → actor keys that enable it this turn (from live
    ``enabled_call_events``). Branches stamp ``target`` as ``const`` or ``enum`` from that map.
    Events with an empty target list in the map are omitted (not offerable without a receiver).
    """

    target_map = targets_by_event or {}
    unique: list[Event[typing.Any]] = []
    seen: set[str] = set()
    for event in events:
        if not event.name or event.name in seen:
            continue
        # When the offer map lists this event with no enablers, drop it from tools.
        if event.name in target_map and not _normalized_targets(target_map[event.name]):
            continue
        seen.add(event.name)
        unique.append(event)

    description = (
        "Dispatch all events necessary for the input. Call once. "
        "Multi-select by listing every offered event this turn requires in the events array "
        "(for example a device action and speech together when both are needed). "
        "Each item must match one offered event branch; payload fields and requirements are "
        "defined on that event's data schema. An empty events list leaves the turn unhandled for "
        "the host cascade (e.g. deliberative reasoning); use an explicit ignore/pass event when "
        "the stage should handle the turn with no environment actions."
    )

    def branch_for(event: Event[typing.Any]) -> dict[str, object]:
        targets = target_map.get(event.name, ())
        return _selection_item_branch(event, patch=patch, targets=targets)

    if unique:
        item_schema: dict[str, object] = {
            "anyOf": [branch_for(event) for event in unique],
        }
    else:
        item_schema = {
            "type": "object",
            "properties": {
                "event": {"type": "string"},
                "data": {"type": "object", "additionalProperties": False},
            },
            "required": ["event"],
            "additionalProperties": False,
        }

    array_examples: list[object] = [[]]
    if unique:
        first_branch = branch_for(unique[0])
        branch_examples = first_branch.get("examples")
        if isinstance(branch_examples, list) and branch_examples:
            array_examples.append(branch_examples)

    parameters: dict[str, object] = {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "description": (
                    "All events necessary for the input this turn. Each item is one offered "
                    "event branch (const name + that event's data schema). Include every action "
                    "required together; empty array selects none."
                ),
                "items": item_schema,
                "examples": array_examples,
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


def fill_unique_selection_targets(
    selections: Events,
    targets_by_event: collections.abc.Mapping[str, collections.abc.Sequence[str]],
) -> Events:
    """Fill ``SelectedEvent.target`` when omitted and exactly one actor enables the event."""

    filled: list[SelectedEvent] = []
    for item in selections:
        if item.target is not None:
            filled.append(item)
            continue
        legal = _normalized_targets(targets_by_event.get(item.event, ()))
        if len(legal) == 1:
            filled.append(dataclasses.replace(item, target=legal[0]))
        else:
            filled.append(item)
    return tuple(filled)


def events_from_dispatch_args(
    args: collections.abc.Mapping[str, object],
    *,
    patch: SchemaPatch | None = None,
    offered: collections.abc.Sequence[Event[typing.Any]] | None = None,
    targets_by_event: collections.abc.Mapping[str, collections.abc.Sequence[str]] | None = None,
) -> Events:
    """Parse a ``dispatch`` tool's args into validated ``Events``.

    When ``targets_by_event`` is provided and a selection omits ``target`` while exactly one
    actor enables that event, the unique actor key is filled so dispatch is explicit.
    Multi-enabler omissions stay unresolved until ``_resolve_target`` fails closed.
    """

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
    if targets_by_event is not None:
        selections = fill_unique_selection_targets(selections, targets_by_event)
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


class OutputData(pydantic.BaseModel):
    """Typed terminal product from one processing operation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        json_schema_extra={
            "description": (
                "Terminal processing result. handled is false only when the processor explicitly declined the input; "
                "events contains the validated selections produced by the operation. Actor-backed selections are "
                "included only after every recipient dispatch completes successfully."
            ),
            "examples": [
                {
                    "handled": True,
                    "events": [{"event": "bot.focus_device", "target": "bot"}],
                },
                {"handled": False, "events": []},
            ],
        },
    )

    handled: bool = pydantic.Field(
        default=True,
        description="Whether the processor handled the input rather than explicitly declining it.",
    )
    events: Events = pydantic.Field(
        default=(),
        description=(
            "Validated event selections. When actors are supplied, every recipient dispatch completed successfully."
        ),
    )


class CancelData(pydantic.BaseModel):
    """Request cancellation of one active processing operation."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(
        min_length=1,
        description="Parent operation identifier whose active processing work must be cancelled.",
        examples=["turn-123"],
    )
    token: str = pydantic.Field(
        min_length=1,
        description="Opaque cancellation capability created by the owning operation actor.",
        examples=["8d72b83f17654f1788c012ab132b4afd"],
    )
    parent_operation_id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Optional owning operation identifier echoed by the cancellation completion.",
    )


class CancelledData(pydantic.BaseModel):
    """Confirmation that one processing operation no longer owns active work."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(
        min_length=1,
        description="Parent operation identifier whose processing work reached the idle boundary.",
        examples=["turn-123"],
    )
    token: str = pydantic.Field(
        min_length=1,
        description="Exact cancellation capability accepted by this processing actor.",
        examples=["8d72b83f17654f1788c012ab132b4afd"],
    )
    parent_operation_id: str | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Owning operation identifier echoed from the cancellation request, when supplied.",
    )


CancelEvent = hsm.Event[CancelData](
    name="bot.ability.processing.cancel",
    schema=CancelData,
)
CancelledEvent = hsm.Event[CancelledData](
    name="bot.ability.processing.cancelled",
    kind=hsm.CompletionEventKind,
    schema=CancelledData,
)


class Operation(hsm.Instance):
    """Operation-scoped actor used as the live capability for one request ID."""


class _OperationData(pydantic.BaseModel):
    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True)

    operation_id: str = pydantic.Field(min_length=1)


_OperationFinishedEvent = hsm.Event[_OperationData](
    name="bot.ability.processing.operation.finished",
    kind=hsm.CompletionEventKind,
    schema=_OperationData,
)


# Owner-scoped operation store for live capabilities (HSM-CONTEXT-001: liveness is store
# membership, never ``state()`` probing). Keyed by ``(hsm.id(owner), operation_id)`` so one
# machine's capabilities are never addressable through another machine's scope. Weak values
# mirror the environment Instances registry lifetime: a capability disappears when its actor
# is garbage collected, and ``finish_operation`` removes it explicitly before retirement.
# Confined to the event loop; entries vanish with their actors, so tests need no reset hook.
_operations: weakref.WeakValueDictionary[tuple[str, str], Operation] = weakref.WeakValueDictionary()


def cancellation_operation_id(
    parent_operation_id: str,
    child_operation_id: str,
    token: str,
    child_id: str,
) -> str:
    """Build the typed live-capability identity for one exact child cancellation."""

    return f"cancel:{parent_operation_id}:{child_operation_id}:{token}:{child_id}"


async def start_operation(owner: hsm.Instance, operation_id: str) -> Operation:
    """Start and register the live capability for an operation."""

    async def hold(
        ctx: hsm.Context,
        instance: hsm.Instance,
        event: hsm.Event[typing.Any],
    ) -> None:
        del ctx, event
        await asyncio.Future[None]()
        _ = instance

    operation = Operation()
    await bot.started(
        owner.context(),
        operation,
        bot.define(
            "ProcessingOperation",
            hsm.initial(hsm.target("active")),
            hsm.state(
                "active",
                hsm.activity(hold),
                hsm.transition(
                    hsm.on(_OperationFinishedEvent),
                    hsm.target("/ProcessingOperation/done"),
                ),
            ),
            hsm.final("done"),
        ),
    )
    instances = owner.context().value(hsm.Keys.Instances)
    if isinstance(instances, collections.abc.MutableMapping):
        # Keep private operation actors out of the scope addressing map.
        _ = instances.pop(hsm.id(operation), None)
    _operations[(hsm.id(owner), operation_id)] = operation
    return operation


def active_operation(owner: hsm.Instance, operation_id: str) -> Operation | None:
    """Resolve the exact live operation capability owned by a machine.

    Liveness is store membership. ``finish_operation`` removes the entry before the
    capability is retired (HSM-CONTEXT-001: do not probe ``state()``).
    """

    return _operations.get((hsm.id(owner), operation_id))


def active_operation_id(owner: hsm.Instance) -> str | None:
    """Return the sole active operation ID owned by an operation-scoped actor."""

    owner_id = hsm.id(owner)
    active = [operation_id for candidate_owner_id, operation_id in _operations if candidate_owner_id == owner_id]
    return active[0] if len(active) == 1 else None


def matches_operation(owner: hsm.Instance, operation_id: str, actor_id: str) -> bool:
    """Return whether typed correlation names the exact live operation actor."""

    operation = active_operation(owner, operation_id)
    return operation is not None and hsm.id(operation) == actor_id


def finish_operation(ctx: hsm.Context, owner: hsm.Instance, operation_id: str) -> None:
    """Retire an operation capability so delayed terminals and cancels cannot match it."""

    operation = _operations.pop((hsm.id(owner), operation_id), None)
    if operation is not None and lifecycle.is_started(operation):
        _ = hsm.dispatch(
            ctx,
            operation,
            dataclasses.replace(
                _OperationFinishedEvent.with_data(_OperationData(operation_id=operation_id)),
                id=operation_id,
                source=hsm.id(owner),
                target=hsm.id(operation),
            ),
        )


def finish_operations(ctx: hsm.Context, owner: hsm.Instance) -> None:
    """Retire every live operation capability owned by a machine abandoning its work."""

    owner_id = hsm.id(owner)
    operation_ids = tuple(
        operation_id for candidate_owner_id, operation_id in _operations if candidate_owner_id == owner_id
    )
    for operation_id in operation_ids:
        finish_operation(ctx, owner, operation_id)


def matches_private_terminal(
    owner: hsm.Instance,
    event: Event[typing.Any],
    operation: tuple[str, str] | None,
) -> bool:
    """Whether a machine-private event is one ``owner`` sent itself for a live operation.

    ``operation`` is the ``(operation_id, generation)`` the caller decoded from the typed
    payload; ``None`` means the payload carries no correlation and never matches. Topology has
    already selected the transition — this is post-delivery correlation only (HSM-DELIVERY-001).
    """

    if operation is None:
        return False
    operation_id, generation = operation
    return (
        matches_operation(owner, operation_id, generation)
        and event.id == operation_id
        and event.source == hsm.id(owner)
        and event.target == hsm.id(owner)
    )


def matches_child_terminal(
    owner: hsm.Instance,
    child: hsm.Instance,
    event: Event[typing.Any],
    *,
    request_id: str,
    operation_id: str,
    generation: str,
) -> bool:
    """Whether ``event`` is the exact terminal ``child`` owes ``owner`` for one live operation.

    Callers narrow the typed payload first (output vs failure); this correlates the already-delivered
    terminal with the live operation actor and the request identity the owner used to address
    ``child``. Topology has already selected the transition — post-delivery correlation only
    (HSM-DELIVERY-001).
    """

    return (
        matches_operation(owner, operation_id, generation)
        and event.target == hsm.id(owner)
        and event.source == hsm.id(child)
        and event.id == request_id
    )


def dispatch_terminal_output(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, typing.Any],
    *,
    operation_id: str | None,
    metadata: dict[str, object],
    output: object,
) -> None:
    """Emit ``instance``'s owner-facing output terminal and retire the operation capability."""

    terminal = dataclasses.replace(
        instance.output_event.with_data(output),
        id=operation_id,
        metadata=dict(metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
    if operation_id is not None:
        finish_operation(ctx, instance, operation_id)


def dispatch_terminal_failure(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, typing.Any],
    *,
    operation_id: str | None,
    metadata: dict[str, object],
    failure: ability.FailureData,
) -> None:
    """Emit ``instance``'s owner-facing failure terminal and retire the operation capability."""

    terminal = dataclasses.replace(
        instance.failed_event.with_data(failure),
        id=operation_id,
        metadata=dict(metadata),
        source=hsm.id(instance),
    )
    _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
    if operation_id is not None:
        finish_operation(ctx, instance, operation_id)


def request_reboot(
    ctx: hsm.Context,
    instance: ability.Ability[typing.Any, typing.Any],
    event: Event[typing.Any],
    *,
    reason: "bot.RebootReason",
) -> None:
    """Ask this ability for a clean robot lifecycle restart.

    ``reason`` is the calling ability's own domain-scoped failure category; this helper owns only
    the envelope. Ability lifecycle forwards the carried reboot to whichever owner is attached.
    """

    request_id = event.id or uuid.uuid4().hex
    _ = hsm.dispatch(
        ctx,
        instance,
        dataclasses.replace(
            ability.RebootRequestEvent.with_data(
                dataclasses.replace(
                    bot.RebootEvent.with_data(bot.RebootEventData(reason=reason)),
                    id=request_id,
                )
            ),
            id=request_id,
            metadata=dict(event.metadata),
        ),
    )


def dispatch_child_cancel(
    ctx: hsm.Context,
    owner: hsm.Instance,
    child: hsm.Instance,
    event: Event[typing.Any],
    *,
    request_id: str,
    parent_operation_id: str,
    token: str,
) -> None:
    """Ask ``child`` to cancel the work ``owner`` addressed to it as ``request_id``."""

    _ = hsm.dispatch(
        ctx,
        child,
        dataclasses.replace(
            CancelEvent.with_data(
                CancelData(
                    operation_id=request_id,
                    token=token,
                    parent_operation_id=parent_operation_id,
                )
            ),
            id=request_id,
            source=hsm.id(owner),
            target=hsm.id(child),
            metadata=dict(event.metadata),
        ),
    )


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
            "examples": [{"input": "incoming speech", "schemas": ["bot.ability.cognition.ignore"]}],
        },
    )

    input: object = pydantic.Field(
        description="Incoming stimulus event or payload.",
        examples=["incoming speech"],
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
    actor_events: SkipJsonSchema[collections.abc.Mapping[str, tuple[str, ...]]] = pydantic.Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description=(
            "Event name → sorted actor keys that enable that event this turn (live "
            "enabled_call_events snapshot). Used to stamp dispatch-tool target const/enum and "
            "to fill unique omitted targets at parse. Dispatch delivery still validates against "
            "declared call events on the actor model."
        ),
    )
    authority: SkipJsonSchema[hsm.Instance | None] = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
        description="Runtime actor that authorizes selected-event dispatch for this processing turn.",
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

    @pydantic.field_serializer("input", when_used="json")
    def _serialize_input(self, value: object) -> object:
        """Serialize the stimulus through the canonical event/Data JSON projection."""

        return event_json_value(value)

    def model_facing_payload(self) -> str:
        """Project this turn exactly as a model may see it: the stimulus, as one XML element.

        The terminal stimulus is the root element — nothing wraps it. Typed causal parents are
        retained in runtime and canonical event data but omitted from this prompt projection. Its
        envelope (event name, and the
        ``id`` / ``source`` / ``target`` it was stamped with) rides on that same root under the
        ``stimulus:`` prefix.

        This is an explicit presentation projection for processors that render a prompt. The
        canonical Pydantic serialization of this model remains ``model_dump``; this helper is
        intentionally separate from that contract. The cognition renderer uses the shared
        canonical event/Data tree and escapes every value it renders. Instructions are system
        policy for the provider, not part of this content.

        ``schemas`` are deliberately absent. The offered events of a turn reach the model through
        the provider's own tool API — ``dispatch_tool`` projects exactly these schemas as the
        ``dispatch`` function — so repeating them in the prompt body was a second copy of the tool
        menu, competing with the authoritative one for the same context window. The stimulus is
        what this content is for; the tool menu is what the tool channel is for.
        """

        rendered = model_facing_xml(self.input)
        instructions = self.instructions or ""
        if len(instructions) > _MAX_MODEL_PROMPT_SCALAR_CHARACTERS:
            raise ValueError("prompt projection scalar length budget exceeded")
        tool = json.dumps(
            dispatch_tool(self.schemas),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        encoded_bytes = sum(len(part.encode("utf-8")) for part in (instructions, rendered, tool))
        if encoded_bytes > _MAX_MODEL_PROMPT_BYTES:
            raise ValueError("prompt projection cumulative encoded byte budget exceeded")
        return rendered


class CompletionData(pydantic.BaseModel):
    """Typed processing terminal carrying both the request and its product."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "input": {"input": "incoming stimulus"},
                    "output": {"handled": True, "events": []},
                }
            ]
        },
    )

    input: InputData = pydantic.Field(description="Exact typed processing request that produced this terminal.")
    output: OutputData = pydantic.Field(description="Validated processing result for the correlated request.")


class FailureData(ability.FailureData):
    """Typed processing failure carrying the request that failed."""

    input: InputData = pydantic.Field(description="Exact typed processing request that failed.")


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
    if isinstance(output, OutputData):
        return output.events if output.handled else None
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
        resolved_confidence = value.confidence if value.confidence is not None else _confidence_from_meta(resolved_meta)
        if domain_data == value.data and resolved_confidence == value.confidence and resolved_meta == value.meta:
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
                lifted: dict[str, object] = {"confidence": top_conf}
                meta = lifted
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
    """Enabled model-offerable events on ``instance`` from its current transition snapshot."""

    event_map = _instance_event_map(instance)
    offered: list[Event[typing.Any]] = []
    seen: set[str] = set()
    snapshot = instance.take_snapshot()
    for transition in snapshot.Transitions:
        for event_name in transition.events:
            if event_name in seen:
                continue
            event = event_map.get(event_name)
            if event is None or event.kind != EventKind:
                continue
            seen.add(event_name)
            offered.append(event)
    return tuple(offered)


def _declared_call_event_names(
    instance: hsm.Instance,
    *,
    dispatch_trust: DispatchTrust = DispatchTrust.MODEL,
) -> set[str]:
    """Call events declared by the instance's model (explicit transitions).

    Delivery validation reads the model, not a point-in-time transition snapshot: the
    observer can be mid-transition when sampled, so snapshot-derived readiness is stale on
    arrival (HSM-CONTEXT-001). Whether the event applies right now is the target's own
    topology and guards.
    """

    return {
        name
        for name, event in _instance_event_map(instance).items()
        if isinstance(event, hsm.Event) and _is_dispatchable_event(event, dispatch_trust=dispatch_trust)
    }


def _unavailable_message(*, event: str, target: str | None = None) -> str:
    if target is None:
        return f"Processing selected unavailable event: {event}."
    return f"Processing selected unavailable event for target {target}: {event}."


def _producer_stamped_fields(value: object) -> tuple[str, ...]:
    if not isinstance(value, pydantic.BaseModel):
        return ()
    names = getattr(type(value), "__producer_stamped_fields__", ())
    if not isinstance(names, frozenset):
        return ()
    field_names = typing.cast(frozenset[str], names)
    return tuple(name for name in field_names if getattr(value, name, None) is not None)


def _reject_untrusted_provenance(value: object, *, trust: DispatchTrust) -> None:
    if trust is DispatchTrust.TRUSTED_BEHAVIOR:
        return
    fields = _producer_stamped_fields(value)
    if fields:
        names = ", ".join(sorted(fields))
        raise SelectionRejectionError(f"Processing model selection includes producer-stamped fields: {names}.")


def _is_dispatchable_event(event: Event[typing.Any], *, dispatch_trust: DispatchTrust) -> bool:
    return event.kind == EventKind or (dispatch_trust is DispatchTrust.TRUSTED_BEHAVIOR and event.kind == hsm.EventKind)


def _resolve_target(
    input: InputData,
    selection: SelectedEvent,
    *,
    dispatch_trust: DispatchTrust,
) -> hsm.Instance:
    if selection.target is not None:
        instance = input.actors.get(selection.target)
        if instance is None or selection.event not in _declared_call_event_names(
            instance,
            dispatch_trust=dispatch_trust,
        ):
            raise SelectionRejectionError(_unavailable_message(event=selection.event, target=selection.target))
        return instance
    matches = [
        name
        for name, instance in input.actors.items()
        if selection.event
        in _declared_call_event_names(
            instance,
            dispatch_trust=dispatch_trust,
        )
    ]
    if len(matches) == 1:
        return input.actors[matches[0]]
    raise SelectionRejectionError(_unavailable_message(event=selection.event))


async def dispatch_selected_events(
    ctx: hsm.Context,
    input: InputData,
    selections: Events,
    *,
    operation_id: str,
    source: hsm.Instance,
    metadata: collections.abc.Mapping[str, object] | None = None,
    dispatch_trust: DispatchTrust = DispatchTrust.MODEL,
) -> None:
    """Validate selections, dispatch actor messages concurrently, and await every recipient."""

    if not selections:
        return
    # Prefer offered-map unique fill so illegal multi-target omissions fail the same way as
    # parse-time omissions; dispatch still re-validates with declared call events.
    if input.actor_events:
        selections = fill_unique_selection_targets(selections, input.actor_events)
    by_name = {event.name: event for event in input.schemas}
    if dispatch_trust is DispatchTrust.TRUSTED_BEHAVIOR:
        for instance in input.actors.values():
            for event in _instance_event_map(instance).values():
                if _is_dispatchable_event(event, dispatch_trust=dispatch_trust):
                    by_name.setdefault(event.name, event)
    event_metadata = dict(metadata or {})
    prepared: list[tuple[hsm.Instance, hsm.Event[typing.Any]]] = []

    for selection in selections:
        offered = by_name.get(selection.event)
        if offered is None:
            raise SelectionRejectionError(_unavailable_message(event=selection.event, target=selection.target))
        # Domain validate never sees model-facing patches; strip using this input's patch type.
        domain_data, _meta = unpatch_event_data(selection.data, patch=input.patch)
        if domain_data is None:
            raw: object = {}
        elif isinstance(domain_data, pydantic.BaseModel):
            # Already typed event data (or a selection that carried a model instance).
            raw = domain_data
        elif isinstance(domain_data, collections.abc.Mapping):
            # JSON / Starlark hops carry the canonical JSON-compatible event/Data tree.
            raw = event_json_value(dict(typing.cast(collections.abc.Mapping[str, object], domain_data)))
        else:
            # Live deliberative frames (processing.InputData) and other non-mapping payloads.
            raw = domain_data

        try:
            offered_validated = validate_event_data(offered, raw)
        except Exception as error:
            raise SelectionRejectionError.from_validation(
                event_name=selection.event,
                prefix=f"Processing selected invalid event data for event: {selection.event}",
                error=error,
            ) from error
        _reject_untrusted_provenance(
            raw if offered_validated is None else offered_validated,
            trust=dispatch_trust,
        )

        target = _resolve_target(input, selection, dispatch_trust=dispatch_trust)
        declared = _instance_event_map(target).get(selection.event)
        if declared is None or not _is_dispatchable_event(declared, dispatch_trust=dispatch_trust):
            raise SelectionRejectionError(_unavailable_message(event=selection.event, target=selection.target))
        try:
            validated = validate_event_data(declared, raw)
        except Exception as error:
            raise SelectionRejectionError.from_validation(
                event_name=selection.event,
                prefix=f"Processing selected invalid event data for event: {selection.event}",
                error=error,
            ) from error
        _reject_untrusted_provenance(raw if validated is None else validated, trust=dispatch_trust)
        dispatch_event = declared if validated is None else declared.with_data(validated)
        prepared.append(
            (
                target,
                dataclasses.replace(
                    dispatch_event,
                    id=operation_id,
                    source=hsm.id(input.authority or source),
                    target=hsm.id(target),
                    metadata=event_metadata,
                ),
            )
        )

    async def dispatch_target(target: hsm.Instance, events: list[hsm.Event[typing.Any]]) -> None:
        for event in events:
            await target.dispatch(ctx, event)

    by_target: dict[int, tuple[hsm.Instance, list[hsm.Event[typing.Any]]]] = {}
    for target, event in prepared:
        target_group = by_target.get(id(target))
        if target_group is None:
            target_group = (target, list[hsm.Event[typing.Any]]())
            by_target[id(target)] = target_group
        target_group[1].append(event)

    try:
        async with asyncio.TaskGroup() as tasks:
            for target, events in by_target.values():
                _ = tasks.create_task(dispatch_target(target, events))
    except ExceptionGroup as errors:
        first = errors.exceptions[0]
        if isinstance(first, Exception):
            raise first
        raise


# --- private completion payloads / events ---


class _AppliedEventData(pydantic.BaseModel):
    """Applying product: input needed for dispatch + events (HSM-COMPLETION-001)."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    input: InputData = pydantic.Field(description="Validated processing input for this operation.")
    output: OutputData = pydantic.Field(description="Typed processor result awaiting actor dispatch.")


_AppliedEvent = hsm.Event[_AppliedEventData](
    name="bot.ability.processing.applied",
    kind=hsm.CompletionEventKind,
    schema=_AppliedEventData,
)
_DispatchedEvent = hsm.Event[_AppliedEventData](
    name="bot.ability.processing.dispatched",
    kind=hsm.CompletionEventKind,
    schema=_AppliedEventData,
)
_FailedEvent = hsm.Event[FailureData](
    name="bot.ability.processing.failed",
    kind=hsm.ErrorEventKind,
    schema=FailureData,
)


class Processor(abc.ABC):
    """Like Encoder / Decoder: map input to selected events."""

    @abc.abstractmethod
    def process(self, input: InputData) -> collections.abc.Awaitable[Events]:
        """Return selected events (empty = none)."""
        ...


class Processing(ability.Ability[InputData, CompletionData]):
    """Leaf ability: apply an injected ``Processor``, then dispatch selected events if any.

    Model-facing system policy lives here (``instructions``), not on the provider Processor.
    On apply, non-blank instructions are stamped onto the input before ``processor.process``.
    """

    processor: Processor
    instructions: typing.ClassVar[str] = ""
    _instructions: str
    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = InputData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = CompletionData
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = hsm.Event[InputData](
        name="bot.ability.processing.input",
        schema=InputData,
    )
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = hsm.Event[CompletionData](
        name="bot.ability.processing.output",
        schema=CompletionData,
    )
    failed_event: typing.ClassVar[hsm.Event[typing.Any]] = hsm.Event[FailureData](
        name=ability.FailedEvent.name,
        kind=hsm.ErrorEventKind,
        schema=FailureData,
    )
    cancel_event: typing.ClassVar[hsm.Event[typing.Any] | None] = CancelEvent
    cancelled_event: typing.ClassVar[hsm.Event[typing.Any] | None] = CancelledEvent

    @staticmethod
    def _has_input(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, InputData)

    @staticmethod
    def _has_applied(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _AppliedEventData)

    @staticmethod
    def _applied_has_events(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return (
            isinstance(event.data, _AppliedEventData) and event.data.output.handled and bool(event.data.output.events)
        )

    @staticmethod
    def _applied_is_unhandled(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, _AppliedEventData) and not event.data.output.handled

    @staticmethod
    def _has_failure(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, FailureData)

    @staticmethod
    def _is_cancel_request(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        if (
            not isinstance(event.data, CancelData)
            or event.id != event.data.operation_id
            or event.target != hsm.id(instance)
        ):
            return False
        return (
            bool(instance._attachments)
            and event.source == hsm.id(instance._attachments[0])
            and active_operation(instance, event.data.operation_id) is not None
        )

    @staticmethod
    def _emit_cancelled(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, CancelData)
        owner = instance._attachments[0]
        _ = hsm.dispatch(
            ctx,
            owner,
            dataclasses.replace(
                CancelledEvent.with_data(
                    CancelledData(
                        operation_id=data.operation_id,
                        token=data.token,
                        parent_operation_id=data.parent_operation_id,
                    )
                ),
                id=event.id,
                source=hsm.id(instance),
                target=hsm.id(owner),
                metadata=dict(event.metadata),
            ),
        )
        finish_operation(ctx, instance, data.operation_id)

    @staticmethod
    def _emit_output(
        ctx: hsm.Context,
        instance: "Processing",
        event: hsm.Event[typing.Any],
        output: CompletionData,
    ) -> None:
        terminal = dataclasses.replace(
            instance.output_event.with_data(output),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalOutputEvent.with_data(terminal))
        finish_operation(ctx, instance, event.id)

    @staticmethod
    def _emit_failure(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, FailureData)
        terminal = dataclasses.replace(
            instance.failed_event.with_data(data),
            id=event.id or None,
            metadata=dict(event.metadata),
            source=hsm.id(instance),
            target=event.source if event.target == hsm.id(instance) else "",
        )
        _ = hsm.dispatch(ctx, instance, ability.TerminalErrorEvent.with_data(terminal))
        finish_operation(ctx, instance, event.id)

    @staticmethod
    def _complete_empty(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        """No events selected: terminal empty product, skip dispatching."""

        data = event.data
        assert isinstance(data, _AppliedEventData)
        Processing._emit_output(
            ctx,
            instance,
            event,
            CompletionData(input=data.input, output=OutputData()),
        )

    @staticmethod
    def _complete_unhandled(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        """Processor declined the input (composition)."""

        data = event.data
        assert isinstance(data, _AppliedEventData)
        Processing._emit_output(
            ctx,
            instance,
            event,
            CompletionData(input=data.input, output=OutputData(handled=False)),
        )

    @staticmethod
    def _complete_dispatched(ctx: hsm.Context, instance: "Processing", event: hsm.Event[typing.Any]) -> None:
        data = event.data
        assert isinstance(data, _AppliedEventData)
        Processing._emit_output(
            ctx,
            instance,
            event,
            CompletionData(input=data.input, output=data.output),
        )

    @staticmethod
    def _input_for_processor(instance: "Processing", input: InputData) -> InputData:
        """Stamp Processing instructions onto the input for model-facing processors.

        The static policy leads; per-turn live context the input already carries (for example
        the device-state block composed at ``build_processing_input``) follows it, so a turn
        never loses either half to the other.
        """

        instructions = instance._instructions.strip()
        if not instructions:
            return input
        if input.instructions == instructions:
            return input
        if input.instructions:
            return input.model_copy(update={"instructions": f"{instructions}\n\n{input.instructions}"})
        return input.model_copy(update={"instructions": instructions})

    @staticmethod
    async def _apply_activity(
        ctx: hsm.Context,
        instance: "Processing",
        event: hsm.Event[InputData],
    ) -> None:
        """applying: run process (via ``_apply`` so subclasses may override)."""

        input = Processing._input_for_processor(instance, typing.cast(InputData, event.data))
        if active_operation(instance, event.id) is None:
            await start_operation(instance, event.id)
        try:
            raw = await instance.processor.process(input)
            if isinstance(raw, Result) and not raw.is_handled:
                output = OutputData(handled=False)
            else:
                payload = raw.output if isinstance(raw, Result) else raw
                coerced = coerce_event_selections(payload, patch=input.patch)
                if coerced is None:
                    raise TypeError("Processor must return an array of events.")
                output = OutputData(events=coerced)
            applied = _AppliedEventData(input=input, output=output)
        except Exception as error:
            _ = instance.dispatch(
                ctx,
                dataclasses.replace(
                    _FailedEvent.with_data(FailureData(message=str(error), input=input)),
                    id=event.id or None,
                    source=event.source,
                    target=event.target,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                _AppliedEvent.with_data(applied),
                id=event.id or None,
                source=event.source,
                target=event.target,
                metadata=dict(event.metadata),
            ),
        )

    @staticmethod
    async def _dispatch_activity(
        ctx: hsm.Context,
        instance: "Processing",
        event: hsm.Event[typing.Any],
    ) -> None:
        """dispatching: validate and concurrently deliver every selected event."""

        data = event.data
        assert isinstance(data, _AppliedEventData)
        try:
            # Actorless products are consumed by the owning composed ability (for example Reflection).
            if data.input.actors:
                await dispatch_selected_events(
                    ctx,
                    data.input,
                    data.output.events,
                    operation_id=event.id or hsm.id(instance),
                    source=instance,
                    metadata=dict(event.metadata),
                )
        except Exception as error:
            _ = instance.dispatch(
                ctx,
                dataclasses.replace(
                    _FailedEvent.with_data(FailureData(message=str(error), input=data.input)),
                    id=event.id or None,
                    source=event.source,
                    target=event.target,
                    metadata=dict(event.metadata),
                ),
            )
            return
        _ = instance.dispatch(
            ctx,
            dataclasses.replace(
                _DispatchedEvent.with_data(data),
                id=event.id or None,
                source=event.source,
                target=event.target,
                metadata=dict(event.metadata),
            ),
        )

    submodel: typing.ClassVar[hsm.Model | None] = bot.define(
        "Processing",
        hsm.initial(hsm.target("/Processing/idle")),
        hsm.state(
            "idle",
            hsm.transition(
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
            ),
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
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
                hsm.target("/Processing/idle"),
            ),
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
                hsm.on(CancelEvent),
                hsm.guard(_is_cancel_request),
                hsm.effect(_emit_cancelled),
                hsm.target("/Processing/idle"),
            ),
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
    "CompletionData",
    "FailureData",
    "DISPATCH_TOOL_NAME",
    "DispatchTrust",
    "EventKind",
    "Event",
    "Events",
    "InputEvent",
    "OutputEvent",
    "OutputData",
    "Operation",
    "InputData",
    "Processor",
    "Processing",
    "Result",
    "SelectionRejectionError",
    "SchemaPatch",
    "SelectedEvent",
    "active_operation",
    "active_operation_id",
    "cancellation_operation_id",
    "coerce_event_selections",
    "collect_offered_events",
    "dispatch_child_cancel",
    "dispatch_selected_events",
    "dispatch_terminal_failure",
    "dispatch_terminal_output",
    "dispatch_tool",
    "enabled_call_events",
    "events_from_dispatch_args",
    "fill_unique_selection_targets",
    "finish_operation",
    "finish_operations",
    "is_deliberative_handoff_schema",
    "model_facing_event_json_schema",
    "matches_child_terminal",
    "matches_operation",
    "matches_private_terminal",
    "normalize_confidence",
    "patch_field_names",
    "patched_event_data_model",
    "request_reboot",
    "selection_confidence",
    "start_operation",
    "unpatch_event_data",
]
