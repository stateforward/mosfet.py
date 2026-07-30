"""OpenTelemetry log emission for text-generator requests."""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import typing

from opentelemetry._logs import SeverityNumber
from opentelemetry.util.types import AnyValue

from bot.telemetry.configure import is_enabled, logger_provider

_LOGGER_NAME = "bot.telemetry.generator"
_COMPONENT = "text.generator"
_STAGE = "request"
# Bound recursive walks in ``_jsonable`` (nested mappings/sequences/dataclasses).
_MAX_JSONABLE_DEPTH = 32
_JSONABLE_TRUNCATED = "<truncated:max-depth>"


def _pydantic_model_dump(value: object) -> AnyValue | None:
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover
        return None
    if isinstance(value, BaseModel):
        # PY-TYPE-003: model_dump(mode="json") yields a JSON tree (scalars /
        # list / dict) that matches OTEL AnyValue; pydantic types it wider.
        return typing.cast(AnyValue, value.model_dump(mode="json"))
    return None


def _jsonable(value: object, *, _depth: int = 0) -> AnyValue:
    """Convert a value to an OTEL ``AnyValue``-friendly JSON tree.

    Recursion into mappings, sequences, and dataclasses is bounded by
    ``_MAX_JSONABLE_DEPTH`` (32). Past that bound, returns
    ``_JSONABLE_TRUNCATED`` instead of descending further.
    """

    if _depth > _MAX_JSONABLE_DEPTH:
        return _JSONABLE_TRUNCATED
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        # PY-TYPE-003: Enum.value for str/int/float/bool (and nested JSON)
        # members is AnyValue-compatible; checker types .value as object/Any.
        return typing.cast(AnyValue, value.value)
    dumped = _pydantic_model_dump(value)
    if dumped is not None:
        return dumped
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value), _depth=_depth + 1)
    if isinstance(value, collections.abc.Mapping):
        # PY-TYPE-003: isinstance proves Mapping; cast only names key/value as
        # object so items() iteration type-checks under basedpyright.
        mapping = typing.cast(collections.abc.Mapping[object, object], value)
        return {str(key): _jsonable(item, _depth=_depth + 1) for key, item in mapping.items()}
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item, _depth=_depth + 1) for item in value]
    return str(value)


def record_generator_request(
    *,
    provider: str,
    model: str | None,
    messages: object,
    tools: object = (),
) -> None:
    """Emit an OTEL log for a text-generator request.

    Payload (messages/tools) is placed in the log record body only. Attributes
    stay low-cardinality: component, provider, optional model, and stage.

    Emission is a no-op until an application boundary calls ``configure()``.
    This library does not auto-install a LoggerProvider. When ``configure()``
    has been invoked, opt-out still applies via env / ``enabled=False``.
    Also a no-op when telemetry is disabled, no exporting provider is active,
    or ``provider`` is blank.
    """

    if not provider or not provider.strip():
        return
    if not is_enabled():
        return
    provider_instance = logger_provider()
    if provider_instance is None:
        return

    attributes: dict[str, str] = {
        "component": _COMPONENT,
        "provider": provider,
        "stage": _STAGE,
    }
    if model is not None:
        attributes["model"] = model

    body: dict[str, AnyValue] = {
        "messages": _jsonable(messages),
        "tools": _jsonable(tools),
    }
    logger = provider_instance.get_logger(_LOGGER_NAME)
    logger.emit(
        # PY-TYPE-003: body values are already AnyValue; emit()'s body param
        # is typed as AnyValue (not Mapping), so cast the finished dict.
        body=typing.cast(AnyValue, body),
        attributes=attributes,
        severity_number=SeverityNumber.INFO,
        severity_text="INFO",
    )


__all__ = ["record_generator_request"]
