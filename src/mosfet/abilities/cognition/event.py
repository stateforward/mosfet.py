"""Cognition-owned projections of typed events for model input."""

import collections.abc
import dataclasses
import enum
import html
import typing
import xml.etree.ElementTree as ElementTree

import hsm
import pydantic
from mosfet import event


# Source-tree layer packages. These name where a domain *lives*, never the domain itself, so a
# tag derived from a payload type's module skips them (``mosfet.abilities.listening…`` is the
# ``listening`` domain, not the ``abilities`` one). Kept here rather than as a per-payload tag
# table: nothing registers a name, the module path already carries it.
_LAYER_PACKAGES: typing.Final = frozenset({"abilities", "devices", "providers"})
# Root of this distribution's import package, derived rather than spelled, so a package
# rename can never silently drop the domain qualifier from every model-facing tag.
_ROOT_PACKAGE: typing.Final = __name__.split(".", 1)[0]
# Envelope attributes live under their own prefix so they can never collide with a payload field.
_STIMULUS_PREFIX: typing.Final = "stimulus"
_MAX_PROMPT_PROJECTION_BYTES: typing.Final = 262_144
_MAX_PROMPT_PROJECTION_NODES: typing.Final = 4_096
_MAX_PROMPT_PROJECTION_ITEMS: typing.Final = 4_096
_MAX_PROMPT_PROJECTION_SCALAR_CHARACTERS: typing.Final = 65_536


@dataclasses.dataclass
class _ProjectionBudget:
    encoded_bytes: int = 0
    nodes: int = 0
    collection_items: int = 0

    def consume_node(self, name: str) -> None:
        self.nodes += 1
        if self.nodes > _MAX_PROMPT_PROJECTION_NODES:
            raise ValueError("prompt projection node budget exceeded")
        self.consume_markup(name)

    def consume_collection(self, count: int) -> None:
        self.collection_items += count
        if self.collection_items > _MAX_PROMPT_PROJECTION_ITEMS:
            raise ValueError("prompt projection collection item budget exceeded")

    def consume_scalar(self, value: str) -> None:
        if len(value) > _MAX_PROMPT_PROJECTION_SCALAR_CHARACTERS:
            raise ValueError("prompt projection scalar length budget exceeded")
        self.consume_markup(html.escape(value, quote=True))

    def consume_markup(self, value: str) -> None:
        self.encoded_bytes += len(value.encode("utf-8"))
        if self.encoded_bytes > _MAX_PROMPT_PROJECTION_BYTES:
            raise ValueError("prompt projection encoded byte budget exceeded")

    def verify_rendered(self, value: str) -> None:
        if len(value.encode("utf-8")) > _MAX_PROMPT_PROJECTION_BYTES:
            raise ValueError("prompt projection encoded byte budget exceeded")


def _element(name: str, budget: _ProjectionBudget) -> ElementTree.Element:
    budget.consume_node(name)
    return ElementTree.Element(name)


def _subelement(element: ElementTree.Element, name: str, budget: _ProjectionBudget) -> ElementTree.Element:
    budget.consume_node(name)
    return ElementTree.SubElement(element, name)


def _set_attribute(element: ElementTree.Element, name: str, value: str, budget: _ProjectionBudget) -> None:
    budget.consume_markup(name)
    budget.consume_scalar(value)
    element.set(name, value)


def _excluded_field_names(level: object) -> frozenset[str]:
    """Read a model-level exclusion marker as a bounded string set."""

    raw = getattr(level, "__model_facing_excluded_fields__", ())
    if not isinstance(raw, collections.abc.Collection):
        return frozenset()
    return frozenset(name for name in typing.cast(collections.abc.Collection[object], raw) if isinstance(name, str))


def _snake_case(name: str) -> str:
    # Shared helper lives on conversation; a top-level import would cycle
    # cognition -> communication -> cognition through the package inits.
    from mosfet.abilities.communication.conversation.conversation import snake_case

    return snake_case(name)


def _payload_tag(payload_type: type) -> str:
    """Derive one XML tag from a payload type: ``<domain>:<local>``."""

    segments = [segment for segment in (getattr(payload_type, "__module__", "") or "").split(".") if segment]
    if segments and segments[0] == _ROOT_PACKAGE:
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


def _set_event_attributes(element: ElementTree.Element, event: object, budget: _ProjectionBudget) -> None:
    """Stamp modeled event identity on the payload element, never on telemetry metadata."""

    if getattr(type(event), "__model_facing_event_data__", False):
        event_name = getattr(event, "event", None)
        if isinstance(event_name, str) and event_name:
            _set_attribute(element, f"{_STIMULUS_PREFIX}:event", event_name, budget)
        for attribute in ("id", "source", "target"):
            carried = getattr(event, attribute, None)
        if carried:
            _set_attribute(element, f"{_STIMULUS_PREFIX}:{attribute}", typing.cast(str, carried), budget)
        return
    if isinstance(event, hsm.Event):
        _set_attribute(element, f"{_STIMULUS_PREFIX}:event", event.name, budget)
        for attribute, carried in (("id", event.id), ("source", event.source), ("target", event.target)):
            if carried:
                _set_attribute(element, f"{_STIMULUS_PREFIX}:{attribute}", carried, budget)


def _model_element(
    model: pydantic.BaseModel,
    *,
    budget: _ProjectionBudget,
    child: ElementTree.Element | None = None,
    envelope: object | None = None,
    serialized: object | None = None,
) -> ElementTree.Element:
    """Render one terminal payload and its model-facing fields.

    Typed causal parents remain part of the runtime and canonical event data, but they are not
    rendered into the terminal-event prompt projection. Payload inheritance is still represented
    structurally, one element per model level.
    """

    if getattr(type(model), "__model_facing_event_data__", False):
        serialized_model = (
            typing.cast(collections.abc.Mapping[str, object], serialized)
            if isinstance(serialized, collections.abc.Mapping)
            else typing.cast(collections.abc.Mapping[str, object], event.event_json_value(model))
        )
        payload = typing.cast(object, getattr(model, "data", None))
        if isinstance(payload, pydantic.BaseModel):
            return _model_element(
                payload,
                budget=budget,
                child=child,
                envelope=model,
                serialized=serialized_model.get("data"),
            )
        root = _element(_payload_tag(type(payload)), budget)
        if payload is not None:
            _fill_field(root, "content", payload, serialized_model.get("data"), budget)
        if child is not None:
            root.append(child)
        _set_event_attributes(root, model, budget)
        return root

    fields = type(model).model_fields
    serialized_model = (
        typing.cast(dict[str, object], serialized)
        if isinstance(serialized, collections.abc.Mapping)
        else typing.cast(dict[str, object], event.event_json_value(model))
    )
    excluded_fields = frozenset(
        name for level in _payload_ancestry(type(model)) for name in _excluded_field_names(level)
    )
    root: ElementTree.Element | None = None
    current: ElementTree.Element | None = None
    rendered: set[str] = set()
    for level in _payload_ancestry(type(model)):
        element = _element(_payload_tag(level), budget)
        if current is None:
            root = element
        else:
            current.append(element)
        current = element
        declared = getattr(level, "__annotations__", {})
        for name in fields:
            if name in rendered or name not in declared or name in excluded_fields or name not in serialized_model:
                continue
            rendered.add(name)
            value = getattr(model, name, None)
            if (
                name == "parent"
                and isinstance(value, pydantic.BaseModel)
                and getattr(type(value), "__model_facing_event_data__", False)
            ):
                continue
            _fill_field(element, name, value, serialized_model[name], budget)
    if root is None:
        root = _element(_payload_tag(type(model)), budget)
    if envelope is not None:
        _set_event_attributes(root, envelope, budget)
    if child is not None and current is not None:
        current.append(child)
    return root


def _fill_field(
    element: ElementTree.Element,
    name: str,
    value: object,
    serialized: object,
    budget: _ProjectionBudget,
) -> None:
    """Place one canonical JSON field on ``element`` as a presentation detail."""

    if serialized is None:
        return
    if isinstance(value, bytes | bytearray | memoryview):
        raise TypeError("raw media must be excluded by its owning Data model before XML rendering")
    if isinstance(value, hsm.Event):
        child = _subelement(element, name, budget)
        payload = value.data
        serialized_event = typing.cast(collections.abc.Mapping[str, object], serialized)
        if isinstance(payload, pydantic.BaseModel):
            child.append(
                _model_element(payload, budget=budget, envelope=value, serialized=serialized_event.get("data"))
            )
        else:
            if payload is not None:
                _fill_field(child, "content", payload, serialized_event.get("data"), budget)
            _set_event_attributes(child, value, budget)
        return
    if isinstance(value, pydantic.BaseModel):
        child = _subelement(element, name, budget)
        child.append(_model_element(value, budget=budget, serialized=serialized))
        return
    if isinstance(serialized, bool):
        _set_attribute(element, name, "true" if serialized else "false", budget)
        return
    if isinstance(serialized, str | int | float):
        _set_attribute(element, name, str(serialized), budget)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _fill_field(element, name, dataclasses.asdict(value), serialized, budget)
        return
    if isinstance(value, collections.abc.Mapping):
        child = _subelement(element, name, budget)
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        serialized_mapping = typing.cast(collections.abc.Mapping[str, object], serialized)
        budget.consume_collection(len(serialized_mapping))
        for key, item in mapping.items():
            if not isinstance(key, str) or key not in serialized_mapping:
                continue
            entry = _subelement(child, "entry", budget)
            _set_attribute(entry, "key", key, budget)
            _fill_field(entry, "value", item, serialized_mapping[key], budget)
        return
    if isinstance(value, collections.abc.Set | collections.abc.Sequence):
        items = tuple(typing.cast(collections.abc.Collection[object], value))
        serialized_items = tuple(typing.cast(collections.abc.Collection[object], serialized))
        budget.consume_collection(len(serialized_items))
        child = _subelement(element, name, budget)
        _set_attribute(child, "count", str(len(serialized_items)), budget)
        for index, item in enumerate(items):
            serialized_item = serialized_items[index]
            if isinstance(item, pydantic.BaseModel):
                child.append(_model_element(item, budget=budget, serialized=serialized_item))
            elif isinstance(serialized_item, bool | str | int | float):
                text = str(serialized_item).lower() if isinstance(serialized_item, bool) else str(serialized_item)
                budget.consume_scalar(text)
                _subelement(child, "item", budget).text = text
        return
    if isinstance(value, enum.Enum):
        _set_attribute(element, name, str(serialized), budget)
    else:
        raise TypeError(f"canonical event field {name!r} is not XML-presentable")


def model_facing_xml(value: object) -> str:
    """Render the terminal event/Data payload as pseudo-XML for model prompts.

    Typed causal parents are intentionally omitted from this presentation projection. They remain
    intact in the event/Data objects and canonical JSON serialization used for routing and
    telemetry; ordinary non-causal fields named ``parent`` render normally.
    """

    budget = _ProjectionBudget()
    if isinstance(value, pydantic.BaseModel) and getattr(type(value), "__model_facing_input__", False):
        value = getattr(value, "stimulus", None)
    envelope: hsm.Event[object] | None = value if isinstance(value, hsm.Event) else None
    payload: object = envelope.data if envelope is not None else value
    if isinstance(payload, pydantic.BaseModel):
        root = _model_element(payload, budget=budget, envelope=envelope)
    else:
        root = _element(_payload_tag(type(payload)), budget)
        if isinstance(payload, str):
            budget.consume_scalar(payload)
            root.text = payload
        elif payload is not None:
            _fill_field(root, "content", payload, event.event_json_value(payload), budget)
        if envelope is not None:
            _set_event_attributes(root, envelope, budget)
    ElementTree.indent(root, space="  ")
    rendered = ElementTree.tostring(root, encoding="unicode")
    budget.verify_rendered(rendered)
    return rendered


__all__ = ["model_facing_xml"]
