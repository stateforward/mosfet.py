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
string values shaped like bearer tokens or API keys are masked, at every depth.

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


def full_enabled() -> bool:
    """Whether the runtime opted in to full-content capture (``BOT_OTEL_CAPTURE=full``)."""

    return os.environ.get(_CAPTURE_ENV, "").strip().lower() == _CAPTURE_FULL


def payload_enabled() -> bool:
    """Whether generator/processor request and response payloads are captured."""

    return full_enabled() or os.environ.get(_CAPTURE_PAYLOAD_ENV, "").strip().lower() in _TRUE_VALUES


def _pydantic_model_dump(value: object) -> object | None:
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover
        return None
    if isinstance(value, BaseModel):
        try:
            return typing.cast(object, value.model_dump(mode="json"))
        except Exception:
            return typing.cast(object, value.model_dump(mode="python"))
    return None


def jsonable(value: object, *, _depth: int = 0) -> AnyValue:
    """Convert a value to an OTEL ``AnyValue`` JSON tree with credentials redacted.

    Recursion is bounded by ``_MAX_DEPTH``; past it the subtree becomes ``<truncated:max-depth>``.
    Mapping keys that name a credential and string values shaped like one are masked.
    """

    if _depth > _MAX_DEPTH:
        return _TRUNCATED
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _SECRET_VALUE.sub(_REDACTED, value)
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, enum.Enum):
        member_value = typing.cast(object, value.value)
        return jsonable(member_value, _depth=_depth + 1)
    dumped = _pydantic_model_dump(value)
    if dumped is not None:
        return jsonable(dumped, _depth=_depth + 1)
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
