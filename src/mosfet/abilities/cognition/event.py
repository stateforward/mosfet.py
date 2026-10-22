"""Cognition-owned projections of typed events for model input.

``model_facing_content`` renders a typed stimulus as pseudo-XML text and attaches the images it
carries as image content parts. Each ``media.Image`` found anywhere in the payload renders as a
``<media:image …/>`` placeholder and never as bytes; the images the model's injected
``media.ImageCapacity`` admits follow their placeholder as an ``ImagePart``.

A payload may declare recalled context it carries as budgeted sequence fields
(``__model_facing_budgeted_fields__``). Those render last, trimmed item by item to the bytes the
rest of the stimulus leaves, so recalled context never makes a prompt fail.

``memory_content`` renders what memory keeps of a stimulus through the same projection. A payload
that declares ``__memory_fields__`` contributes only those fields, typed causal parents included
(the prompt projection skips them); memory never fails a turn, so an unrenderable or oversized
stimulus is kept as its bare element marked ``omitted``.

Metrics (scope ``bot.abilities.processing``; the image metrics count model prompts only, never
``memory_content``):

- ``bot.processing.prompt.image.count`` with ``outcome`` (``attached``, ``omitted_capacity``,
  ``omitted_media_type``) and ``media_type`` (the three ``media.ImageMediaType`` values), once per
  image placeholder;
- ``bot.processing.prompt.image.size`` (bytes, attached images only) with ``media_type``;
- ``bot.processing.prompt.recalled.omitted`` (items a budgeted field left out) with
  ``bot.processing.recalled.field``, the budgeted field's code-defined name (a closed set of
  identifiers declared by payload classes, never a value).
"""

import collections.abc
import dataclasses
import enum
import html
import typing
import xml.etree.ElementTree as ElementTree

import hsm
import pydantic
from opentelemetry import metrics

from mosfet import event, media
from mosfet.abilities.language import text


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
_METRIC_SCOPE: typing.Final = "bot.abilities.processing"
_INDENT: typing.Final = "  "
# Most bytes ``_attach_images`` adds to one placeholder for its verdict: a ``ref`` (numbered at
# most once per node) or an ``omitted`` reason.
_IMAGE_VERDICT_BYTES: typing.Final = max(
    len(f' ref="image-{_MAX_PROMPT_PROJECTION_NODES}"'),
    *(len(f' omitted="{omission}"') for omission in typing.cast(tuple[str, ...], typing.get_args(media.ImageOmission))),
)

_IMAGE_COUNT = metrics.get_meter(_METRIC_SCOPE).create_counter(
    "bot.processing.prompt.image.count",
    unit="{image}",
    description="Images met while projecting a stimulus for a model, by outcome and media type.",
)
_IMAGE_SIZE = metrics.get_meter(_METRIC_SCOPE).create_histogram(
    "bot.processing.prompt.image.size",
    unit="By",
    description="Encoded size of each image attached to a model prompt, by media type.",
)
_RECALLED_OMITTED = metrics.get_meter(_METRIC_SCOPE).create_counter(
    "bot.processing.prompt.recalled.omitted",
    unit="{item}",
    description="Recalled items a budgeted field left out of a model prompt to fit the budget, by field.",
)


@dataclasses.dataclass(frozen=True)
class _BudgetedField:
    """A budgeted sequence field whose items render after the rest of the payload."""

    element: ElementTree.Element
    name: str
    keep: typing.Literal["head", "tail"]
    items: tuple[object, ...]
    serialized: tuple[object, ...]


@dataclasses.dataclass(frozen=True)
class _Mark:
    encoded_bytes: int
    nodes: int
    collection_items: int
    images: int


@dataclasses.dataclass
class _ProjectionBudget:
    # Bytes the rendered text may take; the running ``encoded_bytes`` tally is an early guard
    # against the projection's own cap.
    limit_bytes: int = _MAX_PROMPT_PROJECTION_BYTES
    encoded_bytes: int = 0
    nodes: int = 0
    collection_items: int = 0
    # Placeholders for ``media.Image`` values met while rendering, each with the image it stands
    # for; capacity is applied once the whole payload is rendered.
    images: list[tuple[ElementTree.Element, media.Image]] = dataclasses.field(
        default_factory=list[tuple[ElementTree.Element, media.Image]]
    )
    # Budgeted fields met while rendering; their items render once the rest of the payload has.
    # Items of a budgeted field render whole: a budgeted field inside one is not deferred again.
    deferred: list[_BudgetedField] = dataclasses.field(default_factory=list[_BudgetedField])
    defer: bool = True
    # Rendering what memory keeps: ``__memory_fields__`` select fields, causal parents included.
    memory: bool = False

    def mark(self) -> _Mark:
        return _Mark(self.encoded_bytes, self.nodes, self.collection_items, len(self.images))

    def rewind(self, mark: _Mark) -> None:
        self.encoded_bytes = mark.encoded_bytes
        self.nodes = mark.nodes
        self.collection_items = mark.collection_items
        del self.images[mark.images :]

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
        if len(value.encode("utf-8")) > self.limit_bytes:
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


def _set_image_attributes(
    element: ElementTree.Element,
    image: media.Image | media.ReleasedImage,
    budget: _ProjectionBudget,
    *,
    ref: str | None = None,
    omitted: media.ImageOmission | None = None,
) -> None:
    """Describe an image on its placeholder: ``ref`` leads, any envelope attributes trail."""

    carried = dict(element.attrib)
    element.attrib.clear()
    if ref is not None:
        _set_attribute(element, "ref", ref, budget)
    _set_attribute(element, "media_type", image.media_type, budget)
    if image.width_px is not None:
        _set_attribute(element, "width_px", str(image.width_px), budget)
    if image.height_px is not None:
        _set_attribute(element, "height_px", str(image.height_px), budget)
    if omitted is not None:
        _set_attribute(element, "omitted", omitted, budget)
    element.attrib.update(carried)


def _image_placeholder(image: media.Image | media.ReleasedImage, budget: _ProjectionBudget) -> ElementTree.Element:
    """Render an image as an empty placeholder element; its bytes are never rendered."""

    placeholder = _element(_payload_tag(media.Image), budget)
    if isinstance(image, media.ReleasedImage):
        _set_image_attributes(placeholder, image, budget, omitted=image.reason)
    else:
        budget.images.append((placeholder, image))
    return placeholder


def _attachment_bytes(image: media.Image) -> int:
    """Most bytes ``_attach_images`` adds to the placeholder of ``image``."""

    described = (("media_type", image.media_type), ("width_px", image.width_px), ("height_px", image.height_px))
    return _IMAGE_VERDICT_BYTES + sum(len(f' {name}="{value}"') for name, value in described if value is not None)


def _serialized_bytes(element: ElementTree.Element, *, level: int) -> int:
    """Encoded size of ``element`` rendered at indentation ``level`` (its tail excluded)."""

    ElementTree.indent(element, space=_INDENT, level=level)
    return len(ElementTree.tostring(element, encoding="unicode").encode("utf-8"))


def _field_names(level: object, marker: str) -> frozenset[str] | None:
    """Read a model-level field-name marker as a bounded string set; None when it is not declared."""

    raw = getattr(level, marker, None)
    if not isinstance(raw, collections.abc.Collection) or isinstance(raw, str):
        return None
    return frozenset(name for name in typing.cast(collections.abc.Collection[object], raw) if isinstance(name, str))


def _excluded_field_names(level: object) -> frozenset[str]:
    """Read a model-level exclusion marker as a bounded string set."""

    return _field_names(level, "__model_facing_excluded_fields__") or frozenset()


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
    structurally, one element per model level. A ``media.Image`` or ``media.ReleasedImage``
    renders as an image placeholder.
    """

    if isinstance(model, media.Image | media.ReleasedImage):
        placeholder = _image_placeholder(model, budget)
        if envelope is not None:
            _set_event_attributes(placeholder, envelope, budget)
        return placeholder
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
    kept = _field_names(type(model), "__memory_fields__") if budget.memory else None
    budgeted = typing.cast(
        collections.abc.Mapping[object, object], getattr(type(model), "__model_facing_budgeted_fields__", {})
    )
    if any(keep not in ("head", "tail") for keep in budgeted.values()):
        raise TypeError("__model_facing_budgeted_fields__ maps field names to 'head' or 'tail'")
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
            if kept is not None and name not in kept:
                continue
            rendered.add(name)
            value = getattr(model, name, None)
            if (
                kept is None
                and name == "parent"
                and isinstance(value, pydantic.BaseModel)
                and getattr(type(value), "__model_facing_event_data__", False)
            ):
                continue
            _fill_field(
                element,
                name,
                value,
                serialized_model[name],
                budget,
                keep=typing.cast(typing.Literal["head", "tail"] | None, budgeted.get(name)),
            )
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
    *,
    keep: typing.Literal["head", "tail"] | None = None,
) -> None:
    """Place one canonical JSON field on ``element`` as a presentation detail.

    A sequence field with ``keep`` set is budgeted: its element is placed now and its items are
    deferred to ``_render_budgeted``.
    """

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
        excluded = _excluded_field_names(type(value))
        fields = {key: item for key, item in dataclasses.asdict(value).items() if key not in excluded}
        _fill_field(element, name, fields, serialized, budget)
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
        if keep is not None and budget.defer and items:
            container = _subelement(element, name, budget)
            # Upper bounds for the final count and omission, settled once the items render.
            _set_attribute(container, "count", str(len(items)), budget)
            _set_attribute(container, "omitted", str(len(items)), budget)
            budget.deferred.append(
                _BudgetedField(element=container, name=name, keep=keep, items=items, serialized=serialized_items)
            )
            return
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
            elif isinstance(serialized_item, collections.abc.Mapping | list | tuple):
                # A JSON object or list among the items (a selection's untyped data) renders as one item.
                _fill_field(child, "item", item, serialized_items[index], budget)
        return
    if isinstance(value, enum.Enum):
        _set_attribute(element, name, str(serialized), budget)
    else:
        raise TypeError(f"canonical event field {name!r} is not XML-presentable")


def _render_budgeted(root: ElementTree.Element, budget: _ProjectionBudget) -> None:
    """Render deferred budgeted items, item by item, into the bytes the rest of ``root`` leaves.

    Raises ``ValueError`` when ``root`` without any budgeted item already exceeds the budget: the
    stimulus alone overflows. An item that would exceed any budget dimension is not rendered;
    its field records how many it left out as ``omitted="<n>"``. ``head`` fields offer their items
    first to last (ranked), ``tail`` fields last to first (chronological); kept items keep their
    order. Sizes are measured as rendered and indented, and every image placeholder reserves the
    attribute bytes ``_attach_images`` may add, so the rendered text fits the budget exactly.
    """

    if not budget.deferred:
        return
    budget.defer = False
    depth = {id(root): 0}
    for parent in root.iter():
        for child in parent:
            depth[id(child)] = depth[id(parent)] + 1
    used = _serialized_bytes(root, level=0) + sum(_attachment_bytes(image) for _, image in budget.images)
    if used > budget.limit_bytes:
        raise ValueError("prompt projection encoded byte budget exceeded")
    for field in budget.deferred:
        level = depth[id(field.element)] + 1
        # Each item rides on its own indented line; the first one also turns the empty
        # ``<field … />`` into ``<field …>`` + indentation + ``</field>``.
        line = len("\n") + len(_INDENT) * level
        opening = len(">") - len(" />") + len(f"</{field.element.tag}>".encode("utf-8")) + line
        order = range(len(field.items)) if field.keep == "head" else range(len(field.items) - 1, -1, -1)
        kept: dict[int, ElementTree.Element] = {}
        skipped = 0
        for index in order:
            item, serialized = field.items[index], field.serialized[index]
            mark = budget.mark()
            try:
                budget.consume_collection(1)
                if isinstance(item, pydantic.BaseModel):
                    element = _model_element(item, budget=budget, serialized=serialized)
                elif isinstance(serialized, bool | str | int | float):
                    rendered = str(serialized).lower() if isinstance(serialized, bool) else str(serialized)
                    budget.consume_scalar(rendered)
                    element = _element("item", budget)
                    element.text = rendered
                else:
                    budget.rewind(mark)
                    skipped += 1
                    continue
            except ValueError:
                budget.rewind(mark)
                continue
            attachments = sum(_attachment_bytes(image) for _, image in budget.images[mark.images :])
            cost = _serialized_bytes(element, level=level) + line + attachments + (0 if kept else opening)
            tally = budget.encoded_bytes + sum(_attachment_bytes(image) for _, image in budget.images)
            if used + cost > budget.limit_bytes or tally > _MAX_PROMPT_PROJECTION_BYTES:
                budget.rewind(mark)
                continue
            used += cost
            kept[index] = element
        for index in sorted(kept):
            field.element.append(kept[index])
        omitted = len(field.items) - len(kept) - skipped
        field.element.set("count", str(len(kept)))
        if omitted:
            field.element.set("omitted", str(omitted))
            _RECALLED_OMITTED.add(omitted, attributes={"bot.processing.recalled.field": field.name})
        else:
            del field.element.attrib["omitted"]


def _attach_images(
    root: ElementTree.Element, capacity: media.ImageCapacity, budget: _ProjectionBudget
) -> list[media.Image]:
    """Apply ``capacity`` to the placeholders in document order; return the admitted images in order."""

    order = {id(element): index for index, element in enumerate(root.iter())}
    placed = sorted(budget.images, key=lambda entry: order[id(entry[0])])
    admitted: list[media.Image] = []
    for (placeholder, image), omission in zip(placed, capacity.admit([image for _, image in placed]), strict=True):
        if omission is None:
            admitted.append(image)
            _set_image_attributes(placeholder, image, budget, ref=f"image-{len(admitted)}")
            if not budget.memory:
                _IMAGE_COUNT.add(1, attributes={"outcome": "attached", "media_type": image.media_type})
                _IMAGE_SIZE.record(len(image.content), attributes={"media_type": image.media_type})
        else:
            _set_image_attributes(placeholder, image, budget, omitted=omission)
            if not budget.memory:
                _IMAGE_COUNT.add(1, attributes={"outcome": f"omitted_{omission}", "media_type": image.media_type})
    return admitted


def model_facing_content(
    value: object, *, capacity: media.ImageCapacity, budget_bytes: int = _MAX_PROMPT_PROJECTION_BYTES
) -> tuple[text.ContentPart, ...]:
    """Render the terminal event/Data payload for a model: pseudo-XML text with images attached.

    Typed causal parents are intentionally omitted from this presentation projection. They remain
    intact in the event/Data objects and canonical JSON serialization used for routing and
    telemetry; ordinary non-causal fields named ``parent`` render normally.

    Every ``media.Image`` in the payload renders as ``<media:image media_type="…" …/>`` and never
    as bytes. ``capacity.admit`` decides, newest in document order first, which images attach:
    an admitted placeholder gains ``ref="image-N"`` (numbered in document order among admitted
    images) and its ``ImagePart`` follows it directly; an omitted one gains
    ``omitted="capacity"`` or ``omitted="media_type"``. A ``media.ReleasedImage`` renders as a
    placeholder with ``omitted="<reason>"`` and is never offered to ``capacity``. The result
    alternates text and image parts, starting with text; with no admitted image it is exactly
    one ``TextPart``.

    ``budget_bytes`` bounds the encoded bytes of the rendered text (never above the projection's
    own cap); a caller passes what its instructions and tool menu leave. A payload class may
    declare ``__model_facing_budgeted_fields__``, a mapping of tuple field name to ``"head"``
    (ranked: keep the first items) or ``"tail"`` (chronological: keep the last items). Those
    fields hold recalled context: they render after the rest of the payload, item by item, and
    an item that would exceed any budget dimension is left out and counted in the field's
    ``omitted="<n>"`` attribute, so a budgeted field never makes the projection fail. Items of a
    budgeted field render whole (a budgeted field inside one is not trimmed again).

    Raises ``TypeError`` for raw bytes outside a ``media.Image`` or a malformed
    ``__model_facing_budgeted_fields__``, and ``ValueError`` when the payload without its
    budgeted items exceeds the projection budgets.
    """

    return _project(value, capacity=capacity, budget_bytes=budget_bytes, memory=False)


def memory_content(
    value: object, *, capacity: media.ImageCapacity, budget_bytes: int = _MAX_PROMPT_PROJECTION_BYTES
) -> tuple[text.ContentPart, ...]:
    """Render what memory keeps of a stimulus: pseudo-XML text with the images ``capacity`` admits.

    The same projection as ``model_facing_content``, with two differences. A payload class may
    declare ``__memory_fields__``, a set of field names: only those fields render, and a typed
    causal parent among them renders too (the prompt projection always skips it), so memory keeps
    what caused an emission rather than what it re-sends. And it never raises: a stimulus that
    exceeds ``budget_bytes`` or any projection budget, or that cannot be rendered, is kept as its
    bare element with its envelope attributes and ``omitted="size"`` or ``omitted="content"``.
    """

    try:
        return _project(value, capacity=capacity, budget_bytes=budget_bytes, memory=True)
    except ValueError:
        omitted = "size"
    except TypeError:
        omitted = "content"
    if isinstance(value, pydantic.BaseModel) and getattr(type(value), "__model_facing_input__", False):
        value = getattr(value, "stimulus", None)
    envelope: hsm.Event[object] | None = value if isinstance(value, hsm.Event) else None
    payload: object = envelope.data if envelope is not None else value
    budget = _ProjectionBudget()
    root = _element(_payload_tag(type(payload)), budget)
    if envelope is not None:
        _set_event_attributes(root, envelope, budget)
    root.set("omitted", omitted)
    return (text.TextPart(text=ElementTree.tostring(root, encoding="unicode")),)


def _project(
    value: object, *, capacity: media.ImageCapacity, budget_bytes: int, memory: bool
) -> tuple[text.ContentPart, ...]:
    budget = _ProjectionBudget(limit_bytes=min(budget_bytes, _MAX_PROMPT_PROJECTION_BYTES), memory=memory)
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
    _render_budgeted(root, budget)
    admitted = _attach_images(root, capacity, budget)
    ElementTree.indent(root, space=_INDENT)
    rendered = ElementTree.tostring(root, encoding="unicode")
    budget.verify_rendered(rendered)
    # Every rendered value is escaped, so ``<`` opens only real elements: a payload cannot forge
    # the start of a placeholder, and an empty placeholder's own markup ends at its first ``/>``.
    tag = _payload_tag(media.Image)
    parts: list[text.ContentPart] = []
    cursor = 0
    for ref, image in enumerate(admitted, start=1):
        start = rendered.index(f'<{tag} ref="image-{ref}"', cursor)
        end = rendered.index("/>", start) + len("/>")
        parts.append(text.TextPart(text=rendered[cursor:end]))
        parts.append(text.ImagePart(image=image))
        cursor = end
    if cursor < len(rendered):
        parts.append(text.TextPart(text=rendered[cursor:]))
    return tuple(parts)


__all__ = ["memory_content", "model_facing_content"]
