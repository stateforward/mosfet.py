"""InputData event payload for the cognition (`bot.ability.cognition.input`).

Matches the usual ability/cognition naming pattern: ``*InputData`` data for ``*.input`` events.

Bot forwards live robot state — not a prebuilt decision input:

- ``stimulus``: the event that had no body transition
- ``abilities``: ability instances on the bot
- ``actors``: named HSM instances (devices, input/output abilities, …) the cognition may snapshot/dispatch to
- ``focus``: name of the instance the bot is looking at, if any
"""

from __future__ import annotations

import bot
from .. import ability
from .. import processing

import collections.abc
import json
import typing
from xml.sax import saxutils

import hsm
import pydantic
from pydantic.json_schema import SkipJsonSchema


class InputData(pydantic.BaseModel):
    """Payload of ``bot.ability.cognition.input``: stimulus plus live body references for one turn."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "InputData payload for the cognition. Bot supplies the stimulus, abilities, named HSM instances, and "
                "focus name; the cognition builds tools from snapshots. Selected events are dispatched by Processing."
            ),
            "examples": [
                {
                    "focus": "device-a",
                    "focus_candidates": ["device-a"],
                }
            ],
        },
    )

    stimulus: SkipJsonSchema[bot.BotInputData] = pydantic.Field(
        exclude=True,
        repr=False,
        description="Event or bot input payload that triggered the cognition.",
    )
    abilities: SkipJsonSchema[tuple[ability.Ability[typing.Any, typing.Any], ...]] = pydantic.Field(
        default=(),
        exclude=True,
        repr=False,
        description="Ability instances currently attached on the bot body.",
    )
    actors: SkipJsonSchema[collections.abc.Mapping[str, hsm.Instance]] = pydantic.Field(
        default_factory=dict,
        exclude=True,
        repr=False,
        description=(
            "Named HSM instances available for snapshot and dispatch (devices plus bot input/output "
            "and other attached abilities, e.g. speaking)."
        ),
    )
    focus: str | None = pydantic.Field(
        default=None,
        description="Instance name the bot is looking at, if any.",
        examples=["device-a"],
    )
    focus_candidates: tuple[str, ...] = pydantic.Field(
        default=(),
        description="Instance names that may become focus for this turn (body attention policy).",
        examples=[["device-a"], ["device-a", "device-b"]],
    )


def is_input(value: object) -> typing.TypeGuard[InputData]:
    return isinstance(value, InputData)


@typing.runtime_checkable
class ModelRepr(typing.Protocol):
    """Attribute value that owns its own model-facing serialization.

    The extension point for snapshot attributes whose value is not a plain scalar: implement
    ``__model_repr__`` and the value's fragment is inserted as-is (unescaped XML content —
    the value owns its well-formedness). The builder holds no presentation policy — it asks
    the value. Keep the fragment compact, legible, and free of raw objects, credentials, and
    high-cardinality diagnostics: it goes straight into a model prompt.
    """

    def __model_repr__(self) -> str:
        """Return this value's model-facing form (the XML content of its element)."""
        ...


def _xml_tag(name: str) -> str:
    """XML-safe tag name from an attribute key token (leaf names are already close)."""

    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in name)
    if not safe or safe[0].isdigit() or safe[0] in ".-":
        safe = f"_{safe}"
    return safe


def _singular_item_tag(leaf: str) -> str:
    """Item tag for a mapping's entries: the leaf's final word, de-pluralized.

    The one generic mapping rule: ``owned_devices`` holds devices, so its entries render as
    ``device`` elements — the name comes from the attribute's own key, never from a special
    case.
    """

    token = leaf.rpartition("_")[2] or leaf
    if len(token) > 1 and token.endswith("s"):
        token = token[:-1]
    return _xml_tag(token)


def _element(
    tag: str,
    attributes: collections.abc.Mapping[str, str],
    children: list[str],
    depth: int,
) -> str:
    """One indented XML element; every attribute value escaped via saxutils.quoteattr."""

    rendered_attributes = "".join(f" {name}={saxutils.quoteattr(value)}" for name, value in attributes.items())
    indent = "  " * depth
    if not children:
        return f"{indent}<{tag}{rendered_attributes}/>"
    inner = "\n".join(children)
    return f"{indent}<{tag}{rendered_attributes}>\n{inner}\n{indent}</{tag}>"


def _text_element(tag: str, text: str, depth: int) -> str:
    return f"{'  ' * depth}<{tag}>{text}</{tag}>"


def _render_attribute(key: str, value: object, depth: int) -> str | None:
    """Render one snapshot attribute as an XML element, or None when it has no honest form.

    A :class:`ModelRepr` value's fragment is inserted as-is; a scalar becomes element text
    (escaped); a Mapping nests generically — one child element per entry, tagged by the
    leaf's singular final word, with ``ref``/``id`` attributes — which is how
    ``owned_devices`` yields ``<owned_devices><device ref=".." id=".."/></owned_devices>``
    with no special case. ``None`` and opaque objects render nothing: an empty attribute
    says nothing the state does not already say, and raw objects never reach the model.
    """

    leaf = _xml_tag(key.rpartition("/")[2])
    if isinstance(value, ModelRepr):
        return _text_element(leaf, value.__model_repr__(), depth)
    if value is None:
        return None
    if isinstance(value, str):
        return _text_element(leaf, saxutils.escape(value), depth)
    if isinstance(value, bool):
        return _text_element(leaf, json.dumps(value), depth)
    if isinstance(value, int | float):
        return _text_element(leaf, json.dumps(value), depth)
    if isinstance(value, collections.abc.Mapping):
        item_tag = _singular_item_tag(key.rpartition("/")[2])
        children: list[str] = []
        for entry_key, entry_value in typing.cast(collections.abc.Mapping[object, object], value).items():
            if not isinstance(entry_key, str):
                continue
            if isinstance(entry_value, str):
                text = entry_value
            elif isinstance(entry_value, bool | int | float):
                text = json.dumps(entry_value)
            else:
                continue
            children.append(_element(item_tag, {"ref": entry_key, "id": text}, [], depth + 1))
        return _element(leaf, {}, children, depth)
    return None


def _render_actor_element(
    tag: str,
    attributes: collections.abc.Mapping[str, str],
    snapshot: hsm.Snapshot,
    depth: int,
) -> str:
    """One actor element: tag, given attributes plus state, snapshot attributes as children."""

    children: list[str] = []
    if snapshot.Attributes:
        snapshot_attributes = typing.cast(collections.abc.Mapping[str, object], snapshot.Attributes)
        for key, value in snapshot_attributes.items():
            child = _render_attribute(key, value, depth + 1)
            if child is not None:
                children.append(child)
    return _element(tag, {**attributes, "state": snapshot.State}, children, depth)


def _owned_device_references(actor: hsm.Instance) -> dict[str, str] | None:
    """Read one bot actor's declared ``owned_devices`` off its own snapshot, if it declares any.

    Matched by leaf name so a redefined body model (any root name) still resolves. The
    declaration binds each configured reference to the device's live runtime id; a value that
    is not exactly that mapping is not an ownership declaration anyone can honor.
    """

    snapshot = actor.take_snapshot()
    if not snapshot.Attributes:
        return None
    attributes = typing.cast(collections.abc.Mapping[str, object], snapshot.Attributes)
    for key, value in attributes.items():
        if key.rpartition("/")[2] != "owned_devices":
            continue
        if not isinstance(value, dict):
            return None
        owned: dict[str, str] = {}
        for reference, runtime_id in typing.cast(dict[object, object], value).items():
            if not isinstance(reference, str) or not isinstance(runtime_id, str):
                return None
            owned[reference] = runtime_id
        return owned
    return None


def _device_state_instructions(actors: collections.abc.Mapping[str, hsm.Instance]) -> str | None:
    """Compose the live-state XML block for one cognition turn, read off actor snapshots.

    The builder owns only the envelope: a ``live_state`` root holding one ``bot`` element
    (its state as an XML attribute, its own snapshot attributes as child elements — the bot
    describing itself, ``owned_devices`` among them) and one ``device`` element per reference
    the bot declares (``ref``, ``id``, ``state`` as XML attributes, the device's attributes
    as children). The ``id`` carries the live runtime id a stimulus envelope names as its
    source, so the model can match the two with no translation layer. Attribute values own
    their model-facing form (:class:`ModelRepr` is the extension point); every other value
    follows the generic scalar/mapping rules or is dropped. No header text, no presentation
    policy beyond the envelope. Snapshots are the sanctioned observation path (HSM-OBS-001):
    cognition reads them to build deliberative input, never to coordinate with the device.
    """

    # Lazy like Device._report: bot.device imports bot.abilities at module level, so a
    # top-level import here would close a cycle back through this package.
    from bot.device import Device

    owner = actors.get("bot")
    if owner is None:
        return None
    owned = _owned_device_references(owner)
    if owned is None:
        return None
    elements = [_render_actor_element("bot", {}, owner.take_snapshot(), 1)]
    for reference, runtime_id in owned.items():
        actor = actors.get(reference)
        if not isinstance(actor, Device):
            continue
        elements.append(_render_actor_element("device", {"ref": reference, "id": runtime_id}, actor.take_snapshot(), 1))
    return _element("live_state", {}, elements, 0)


def build_processing_input(
    cognition_input: InputData,
    *,
    extra_actors: collections.abc.Mapping[str, hsm.Instance] | None = None,
    authority: hsm.Instance | None = None,
) -> processing.InputData:
    """Build the deliberative input and callable schemas for one cognition turn.

    Offered tools are deduced from each actor's live HSM transition snapshot
    (``enabled_call_events``). Body attention (``bot.focus_device`` /
    ``bot.clear_focus``) appears when the bot actor's active topology enables
    those ``processing.EventKind`` transitions — never via a parallel hard-coded schema
    allowlist. The cognition host itself is included when ``authority`` is the
    live Cognition instance so host model-offerable events such as
    ``bot.ability.cognition.ignore`` appear only when that topology enables them.

    The per-turn instructions open a live device-state block (same snapshots), which the
    provider sends as the system message; any static Processing policy is stamped ahead of
    it at apply time.
    """

    actors: dict[str, hsm.Instance] = dict(cognition_input.actors)
    if extra_actors:
        actors.update(extra_actors)
    if authority is not None and not any(actor is authority for actor in actors.values()):
        # Host call surface (ignore, …) comes from Cognition's snapshot, not a schema allowlist.
        actors = {**actors, "cognition": authority}
    schemas: list[processing.Event[typing.Any]] = []
    seen: set[str] = set()
    for instance in actors.values():
        for event in processing.enabled_call_events(instance):
            if event.name not in seen:
                seen.add(event.name)
                schemas.append(event)
    return processing.InputData(
        input=cognition_input.stimulus,
        schemas=tuple(schemas),
        actors=actors,
        authority=authority,
        instructions=_device_state_instructions(actors),
    )


__all__ = [
    "InputData",
    "ModelRepr",
    "build_processing_input",
    "is_input",
]
