"""Private rendering helpers for the model-facing world block :class:`Environment` composes.

``Environment.model_snapshot`` (``environment.py``) is the one public entry point into this
module. Everything here turns what :meth:`hsm.Instance.take_snapshot` already declared into XML a
model can read. This module holds no state of its own and makes no coordination decisions — only
presentation choices about content the system already knows.
"""

from __future__ import annotations

import collections.abc
import json
import typing
from xml.sax import saxutils

import hsm

from mosfet import lifecycle


@typing.runtime_checkable
class ModelRepr(typing.Protocol):
    """Attribute value that owns its own model-facing serialization.

    The extension point for snapshot attributes whose value is not a plain scalar: implement
    ``__model_repr__`` and the value's fragment is inserted as-is (unescaped XML content —
    the value owns its well-formedness). The renderer holds no presentation policy of its own
    for these values — it asks the value. Keep the fragment compact, legible, and free of raw
    objects, credentials, and high-cardinality diagnostics: it goes straight into a model prompt.
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
    (escaped); a Mapping nests recursively — one child element per entry, each rendered by
    this same rule keyed by its own entry name — which is how a device's folded peripheral
    observation (``display`` → ``{"caller_id": "alice"}``) yields
    ``<display><caller_id>alice</caller_id></display>`` with no special case. ``None`` and
    opaque objects render nothing: an empty attribute says nothing the state does not already
    say, and raw objects never reach the model.

    ``owned_devices`` does not go through this rule: its values are live runtime ids, not
    attribute content, so resolving them into full nested device elements is
    :func:`_render_owned_devices`'s job, called instead of this one where that key appears.
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
        children: list[str] = []
        for entry_key, entry_value in typing.cast(collections.abc.Mapping[object, object], value).items():
            if not isinstance(entry_key, str):
                continue
            child = _render_attribute(entry_key, entry_value, depth + 1)
            if child is not None:
                children.append(child)
        return _element(leaf, {}, children, depth)
    return None


def _render_actor_element(
    tag: str,
    attributes: collections.abc.Mapping[str, str],
    actor_snapshot: hsm.Snapshot,
    depth: int,
) -> str:
    """One actor element: tag, given attributes plus state, snapshot attributes as children."""

    children: list[str] = []
    if actor_snapshot.Attributes:
        snapshot_attributes = typing.cast(collections.abc.Mapping[str, object], actor_snapshot.Attributes)
        for key, value in snapshot_attributes.items():
            child = _render_attribute(key, value, depth + 1)
            if child is not None:
                children.append(child)
    return _element(tag, {**attributes, "state": actor_snapshot.State}, children, depth)


def _render_owned_devices(
    value: object,
    resolve_device: collections.abc.Callable[[str], hsm.Instance | None],
    depth: int,
) -> str | None:
    """Render a perspective's ``owned_devices`` declaration as live, fully-described device elements.

    The one attribute this module treats specially, because its values are not attribute
    content: they are runtime ids naming live actors elsewhere in scope. ``resolve_device`` looks
    up a live actor by runtime id in the owning environment's own addressing scope — never a
    caller-supplied actor map, which would let cognition decide what the world contains. A
    reference the perspective names but the environment cannot currently give an honest snapshot
    for (undeclared, detached, not started in this scope, or stopped — still addressable, but with
    no live state to show) is dropped rather than invented — the perspective's ownership claim
    outlives any one turn's live actor set. A value that is not exactly a ``str`` → ``str`` mapping
    is not an ownership declaration anyone can honor and renders nothing.
    """

    # Lazy: mosfet.device imports mosfet.environment at module level, so a top-level import here
    # would close a cycle back through this package.
    from mosfet.device import Device

    if not isinstance(value, collections.abc.Mapping):
        return None
    children: list[str] = []
    for reference, runtime_id in typing.cast(collections.abc.Mapping[object, object], value).items():
        if not isinstance(reference, str) or not isinstance(runtime_id, str):
            continue
        actor = resolve_device(runtime_id)
        if not isinstance(actor, Device):
            continue
        actor_snapshot = lifecycle.snapshot_if_started(actor)
        if actor_snapshot is None:
            continue
        children.append(
            _render_actor_element("device", {"ref": reference, "id": runtime_id}, actor_snapshot, depth + 1)
        )
    return _element("owned_devices", {}, children, depth)


def render_self(
    perspective: hsm.Instance,
    resolve_device: collections.abc.Callable[[str], hsm.Instance | None],
    depth: int,
) -> str | None:
    """Render ``perspective``'s own live snapshot as ``<self>``, or ``None`` when not honestly observable.

    ``perspective``'s state becomes the ``state`` attribute, and its own snapshot attributes
    become children. ``owned_devices`` is the one attribute rendered by
    :func:`_render_owned_devices` instead of the generic :func:`_render_attribute` rule, because
    its values name live actors rather than holding attribute content directly; every other
    attribute — including a nested peripheral observation mapping — follows the generic
    scalar/mapping rendering. ``None`` when ``perspective`` has no running HSM to snapshot: an
    unstarted or stopped actor has no honest state to show.
    """

    perspective_snapshot = lifecycle.snapshot_if_started(perspective)
    if perspective_snapshot is None:
        return None
    children: list[str] = []
    if perspective_snapshot.Attributes:
        attributes = typing.cast(collections.abc.Mapping[str, object], perspective_snapshot.Attributes)
        for key, value in attributes.items():
            leaf = key.rpartition("/")[2]
            child = (
                _render_owned_devices(value, resolve_device, depth + 1)
                if leaf == "owned_devices"
                else _render_attribute(key, value, depth + 1)
            )
            if child is not None:
                children.append(child)
    return _element("self", {"state": perspective_snapshot.State}, children, depth)


def render_environment(environment_id: str, self_element: str) -> str:
    """Compose the environment root — the world, not any one actor — around one ``<self>`` block.

    The root holds no state of its own beyond its own identity (``id``): the world is not an
    actor and has no behavioral state to show. No header text, no presentation policy beyond the
    envelope, and no sibling top-level device elements — a device is only ever described nested
    under the perspective that owns it.
    """

    return _element("environment", {"id": environment_id}, [self_element], 0)


__all__ = ["ModelRepr"]
