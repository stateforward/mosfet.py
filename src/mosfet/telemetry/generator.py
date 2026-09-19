"""OpenTelemetry log emission for text-generator requests."""

from __future__ import annotations

import collections.abc
import typing

from opentelemetry._logs import SeverityNumber
from opentelemetry.util.types import AnyValue

from mosfet.telemetry import capture
from mosfet.telemetry.configure import is_enabled, logger_provider

_LOGGER_NAME = "bot.telemetry.generator"
_COMPONENT = "text.generator"
_STAGE = "request"
_USAGE_STAGE = "usage"
_RESPONSE_STAGE = "response"


def _size(value: object) -> int | None:
    if isinstance(value, collections.abc.Sized):
        return len(value)
    return None


def record_generator_request(
    *,
    provider: str,
    model: str | None,
    messages: object,
    tools: object = (),
) -> None:
    """Emit an OTEL log for a text-generator request.

    Payload (messages/tools) is placed in the log record body only when the runtime
    explicitly opts in via ``BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD`` or ``BOT_OTEL_CAPTURE=full``
    (credentials redacted, see ``mosfet.telemetry.capture``). Without that opt-in the
    record carries low-cardinality request metadata (counts), never prompt/tool content.
    Attributes stay low-cardinality: component, provider, optional model, and stage.

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

    body: dict[str, AnyValue]
    if capture.payload_enabled():
        body = {
            "messages": capture.jsonable(messages),
            "tools": capture.jsonable(tools),
        }
    else:
        body = {
            "capture": "disabled",
            "message_count": _size(messages) if _size(messages) is not None else 0,
            "tool_count": _size(tools) if _size(tools) is not None else 0,
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


def record_generator_usage(
    *,
    provider: str,
    model: str | None,
    usage: collections.abc.Mapping[str, int],
) -> None:
    """Emit an OTEL log for text-generator token usage.

    ``usage`` holds token counts only (for example ``input_tokens``, ``output_tokens``,
    ``reasoning_tokens``), so it is recorded without the payload opt-in. Attributes and
    no-op conditions match ``record_generator_request``; stage is ``usage``.
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
        "stage": _USAGE_STAGE,
    }
    if model is not None:
        attributes["model"] = model
    body: dict[str, AnyValue] = {str(key): int(value) for key, value in usage.items()}
    logger = provider_instance.get_logger(_LOGGER_NAME)
    logger.emit(
        # PY-TYPE-003: body values are ints (AnyValue); emit() types body as AnyValue.
        body=typing.cast(AnyValue, body),
        attributes=attributes,
        severity_number=SeverityNumber.INFO,
        severity_text="INFO",
    )


def record_generator_response(
    *,
    provider: str,
    model: str | None,
    response: object,
    latency_s: float,
    error: BaseException | None = None,
) -> None:
    """Emit an OTEL log for what a text generator or processor answered.

    Recorded only under payload capture (``BOT_OTEL_CAPTURE_GENERATOR_PAYLOAD`` or
    ``BOT_OTEL_CAPTURE=full``); default telemetry is unchanged. The body carries the raw
    provider response (tool calls and arguments, text, reasoning summary, finish/incomplete
    status) with credentials redacted, the wall-clock latency, and the error text when the call
    or its mapping failed. Attributes match ``record_generator_request`` with stage ``response``
    and a low-cardinality ``outcome``.
    """

    if not provider or not provider.strip() or not capture.payload_enabled():
        return
    attributes: dict[str, str | bool | int | float] = {
        "component": _COMPONENT,
        "provider": provider,
        "stage": _RESPONSE_STAGE,
        "outcome": "failed" if error is not None else "ok",
    }
    if model is not None:
        attributes["model"] = model
    capture.emit(
        _LOGGER_NAME,
        body={
            "response": response,
            "latency_ms": round(latency_s * 1000.0, 1),
            "error": None if error is None else f"{type(error).__name__}: {error}",
        },
        attributes=attributes,
        severity=SeverityNumber.ERROR if error is not None else SeverityNumber.INFO,
    )


__all__ = ["record_generator_request", "record_generator_response", "record_generator_usage"]
