"""Opt-in full-content capture for local debugging.

Default telemetry is low-cardinality and payload-free. A runtime that needs to reconstruct one
turn end to end opts in with ``BOT_OTEL_CAPTURE=full``; then every observed HSM event (name,
kind, id, source, target, serialized data, and the transition that consumed it), every
generator/processor request *and* response, and every dispatch outcome that telemetry is told
about lands in the log file as a structured record, correlated by trace id and event id.

``BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD=1`` keeps its narrower meaning: generator/processor
request and response payloads only.

Privacy: full capture writes user text, transcripts, prompts, model output, and behavior source
to the local log file. It is a development switch (``EXCEPTIONS.md``, PY-LOG-002). Credentials
are never written: mapping keys that name a credential are replaced with ``<redacted>`` and
string values shaped like bearer tokens or API keys are masked, at every depth. Image content is
never written either: a ``mosfet.media.Image`` anywhere in a payload, including nested inside a
Pydantic model, becomes ``<image:<media_type>:<N> bytes>``; raw bytes become ``<bytes:N>``; and an
RFC 2397 base64 data URL becomes ``<data-url:<media_type>:<N> chars>``.

Log bodies carry the payload; attributes stay low-cardinality (component, stage, event name).
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import enum
import os
import re
import typing

from opentelemetry import context as otel_context
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context as OtelContext
from opentelemetry.util.types import AnyValue
import pydantic

from mosfet import media
from mosfet.telemetry.configure import is_enabled, logger_provider

_CAPTURE_ENV = "BOT_OTEL_CAPTURE"
_CAPTURE_FULL = "full"
_CAPTURE_PAYLOAD_ENV = "BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
# Bound recursive walks (nested mappings/sequences/dataclasses).
_MAX_DEPTH = 32
_TRUNCATED = "<truncated:max-depth>"
_REDACTED = "<redacted>"
_SECRET_KEY = re.compile(r"(authorization|api[_-]?key|apikey|secret|password|passwd|(^|_)token$|bearer|cookie)", re.I)
_SECRET_VALUE = re.compile(
    r"(\bBearer\s+\S+|\bsk-[A-Za-z0-9_\-]{8,}|\bxai-[A-Za-z0-9_\-]{8,}|\bAIza[0-9A-Za-z_\-]{20,})"
)
_DATA_URL = re.compile(r"data:([A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+);base64,([A-Za-z0-9+/_-]+=*)")


def full_enabled() -> bool:
    """Whether the runtime opted in to full-content capture (``BOT_OTEL_CAPTURE=full``)."""

    return os.environ.get(_CAPTURE_ENV, "").strip().lower() == _CAPTURE_FULL


def payload_enabled() -> bool:
    """Whether generator/processor request and response payloads are captured."""

    return full_enabled() or os.environ.get(_CAPTURE_PAYLOAD_ENV, "").strip().lower() in _TRUE_VALUES


def _beside_source(source: object, dumped: object, *, depth: int) -> AnyValue:
    """Project ``dumped`` (the JSON dump of ``source``) while walking ``source`` beside it.

    The dump alone has already turned an ``Image``'s bytes into base64 text, so finding an image
    nested at any depth inside a model means walking the model's own values next to its dump.
    Leaves keep the dump's JSON form; only an ``Image`` source is replaced by its size marker.
    """

    if depth > _MAX_DEPTH:
        return _TRUNCATED
    if isinstance(source, media.Image):
        return f"<image:{source.media_type}:{len(source.content)} bytes>"
    if isinstance(source, pydantic.RootModel):
        return _beside_source(typing.cast(object, source.root), dumped, depth=depth)
    if isinstance(dumped, dict):
        dumped_mapping = typing.cast(dict[str, object], dumped)
        sources = list(dumped_mapping.values())
        if isinstance(source, pydantic.BaseModel) or (
            dataclasses.is_dataclass(source) and not isinstance(source, type)
        ):
            sources = [typing.cast(object, getattr(source, key, item)) for key, item in dumped_mapping.items()]
        elif isinstance(source, collections.abc.Mapping):
            source_values = list(typing.cast(collections.abc.Mapping[object, object], source).values())
            if len(source_values) == len(dumped_mapping):
                sources = source_values
        return {
            key: _REDACTED if _SECRET_KEY.search(key) else _beside_source(item_source, item, depth=depth + 1)
            for (key, item), item_source in zip(dumped_mapping.items(), sources, strict=True)
        }
    if (
        isinstance(dumped, list | tuple)
        and isinstance(source, collections.abc.Sequence | collections.abc.Set)
        and not isinstance(source, str | bytes | bytearray)
    ):
        dumped_items = typing.cast(collections.abc.Sequence[object], dumped)
        source_items = list(typing.cast(collections.abc.Iterable[object], source))
        if len(source_items) == len(dumped_items):
            return [
                _beside_source(item_source, item, depth=depth + 1)
                for item_source, item in zip(source_items, dumped_items, strict=True)
            ]
    return jsonable(typing.cast(object, dumped), _depth=depth)


def jsonable(value: object, *, _depth: int = 0) -> AnyValue:
    """Convert a value to an OTEL ``AnyValue`` JSON tree with credentials and image content redacted.

    Recursion is bounded by ``_MAX_DEPTH``; past it the subtree becomes ``<truncated:max-depth>``.
    Mapping keys that name a credential and string values shaped like one are masked. A
    ``media.Image`` at any depth becomes ``<image:<media_type>:<N> bytes>``, raw bytes become
    ``<bytes:N>``, and a base64 data URL inside a string becomes ``<data-url:<media_type>:<N> chars>``.
    """

    if _depth > _MAX_DEPTH:
        return _TRUNCATED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        without_media = _DATA_URL.sub(lambda match: f"<data-url:{match[1]}:{len(match[2])} chars>", value)
        return _SECRET_VALUE.sub(_REDACTED, without_media)
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, enum.Enum):
        member_value = typing.cast(object, value.value)
        return jsonable(member_value, _depth=_depth + 1)
    if isinstance(value, pydantic.BaseModel):
        try:
            dumped = typing.cast(object, value.model_dump(mode="json"))
        except Exception:
            dumped = typing.cast(object, value.model_dump(mode="python"))
        return _beside_source(value, dumped, depth=_depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {field.name: getattr(value, field.name, None) for field in dataclasses.fields(value)}
        return jsonable(fields, _depth=_depth + 1)
    if isinstance(value, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {
            str(key): _REDACTED if _SECRET_KEY.search(str(key)) else jsonable(item, _depth=_depth + 1)
            for key, item in mapping.items()
        }
    if isinstance(value, collections.abc.Iterable) and not isinstance(value, (str, bytes, bytearray)):
        if isinstance(value, collections.abc.Sequence | collections.abc.Set):
            return [jsonable(item, _depth=_depth + 1) for item in typing.cast(collections.abc.Iterable[object], value)]
    return _SECRET_VALUE.sub(_REDACTED, str(value))


def emit(
    logger_name: str,
    *,
    body: collections.abc.Mapping[str, object],
    attributes: collections.abc.Mapping[str, str | bool | int | float],
    severity: SeverityNumber = SeverityNumber.INFO,
    context: OtelContext | None = None,
) -> None:
    """Emit one structured capture record in the active trace context (no-op when unconfigured)."""

    if not is_enabled():
        return
    provider = logger_provider()
    if provider is None:
        return
    provider.get_logger(logger_name).emit(
        context=context,
        body=jsonable(body),
        attributes=dict(attributes),
        severity_number=severity,
        severity_text=severity.name,
    )


def record_dispatch(
    delivery: collections.abc.Awaitable[bool],
    *,
    component: str,
    event: object,
    source: str,
    target: str,
) -> collections.abc.Awaitable[bool]:
    """Under full capture, log whether ``target`` accepted ``event`` once delivery resolves.

    Returns ``delivery`` unchanged; the result is observed through a done-callback so the
    caller's dispatch semantics (fire-and-forget or awaited) are untouched.
    """

    if not full_enabled() or not isinstance(delivery, asyncio.Future):
        return delivery
    name = getattr(event, "name", "")
    event_id = getattr(event, "id", "")
    captured = otel_context.get_current()

    def _done(future: asyncio.Future[bool]) -> None:
        if future.cancelled():
            outcome: object = "cancelled"
        elif future.exception() is not None:
            outcome = f"error: {type(future.exception()).__name__}: {future.exception()}"
        else:
            outcome = future.result()
        emit(
            "bot.telemetry.dispatch",
            body={"event": name, "id": event_id, "source": source, "target": target, "accepted": outcome},
            attributes={"component": component, "stage": "dispatch", "hsm.event.name": str(name)},
            context=captured,
        )

    delivery.add_done_callback(_done)
    return delivery


__all__ = ["emit", "full_enabled", "jsonable", "payload_enabled", "record_dispatch"]
