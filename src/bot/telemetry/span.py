"""Span helpers so instrumentation is uniform across bot code and providers.

Spans are the second-choice signal (``METRICS > SPANS > Logs``): reach for a
span when the question is *what happened to this one thing as it moved*, which
metrics cannot answer. Attributes stay low-cardinality and payload-free —
component, stage, outcome, normalized failure kind. Never audio bytes,
transcripts, prompts, credentials, or per-event/instance identifiers.

Failure-kind vocabulary and how kinds relate to typed failure events:
see ``FAILURE_KINDS.md`` next to this module.

Emission is a no-op until an application boundary calls
``bot.telemetry.configure()``; library models never install a provider
themselves.
"""

from __future__ import annotations

import collections.abc
import contextlib
import functools
import re

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from bot.telemetry.configure import tracer_provider

_COMPONENT_ATTR = "bot.component.name"
_STAGE_ATTR = "bot.stage"
_OUTCOME_ATTR = "bot.outcome"
_FAILURE_KIND_ATTR = "bot.failure.kind"

_MAX_KIND_LENGTH = 64
_UNKNOWN_KIND = "unknown"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

Ok = "ok"
Failed = "failed"

AttributeValue = str | bool | int | float
Attributes = collections.abc.Mapping[str, AttributeValue]


def tracer(scope: str) -> Tracer:
    """Return a tracer from the configured exporting provider, else the API default.

    Falling back to ``trace.get_tracer`` keeps observation opt-in: with no
    provider installed the returned tracer is a no-op.
    """

    provider = tracer_provider()
    if provider is not None:
        return provider.get_tracer(scope)
    return trace.get_tracer(scope)


def failure_kind(error: BaseException, fallback: str) -> str:
    """Normalize an exception into a stable, low-cardinality failure kind.

    Prefers a ``failure_kind`` string carried by the error (the same attribute
    provider errors already expose and typed failure event data already
    carries), else ``fallback``. Never derives the kind from the error message.
    """

    candidate = getattr(error, "failure_kind", None)
    if isinstance(candidate, str) and candidate.strip():
        return normalized_kind(candidate)
    return normalized_kind(fallback)


def normalized_kind(kind: str) -> str:
    """Coerce a failure kind to ``lower_snake_case`` within the length bound."""

    split = _CAMEL_BOUNDARY.sub("_", kind.strip())
    cleaned = "".join(
        character if character.isascii() and character.isalnum() else "_" for character in split.lower()
    )
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    if not collapsed:
        return _UNKNOWN_KIND
    return collapsed[:_MAX_KIND_LENGTH]


def record_failure(span: Span, kind: str) -> None:
    """Mark ``span`` failed with a normalized failure kind and ERROR status."""

    stable = normalized_kind(kind)
    span.set_attribute(_OUTCOME_ATTR, Failed)
    span.set_attribute(_FAILURE_KIND_ATTR, stable)
    span.set_status(Status(StatusCode.ERROR, stable))


@contextlib.contextmanager
def operation(
    name: str,
    *,
    scope: str,
    component: str,
    stage: str,
    attributes: Attributes | None = None,
    context: Context | None = None,
) -> collections.abc.Generator[Span]:
    """Run one traced operation, defaulting its outcome to ``ok``.

    ``name``, ``component``, and ``stage`` must be constants at the call site —
    they are span/metric dimensions, not per-item labels. An exception escaping
    the block is recorded as ``bot.failure.kind`` derived by ``failure_kind``
    from the exception's own stable kind (fallback: the exception type name in
    snake case) and re-raised.

    Example::

        with span.operation(
            "bot.hearing.decode",
            scope="bot.abilities.hearing",
            component="hearing.speech.decoding",
            stage="decode",
        ) as active:
            result = await decode(frame)
            active.set_attribute("bot.frames.count", result.frames)
    """

    span_attributes: dict[str, AttributeValue] = {
        _COMPONENT_ATTR: component,
        _STAGE_ATTR: stage,
    }
    if attributes is not None:
        span_attributes.update(attributes)
    # record_exception would put the raw exception message and stacktrace on the
    # span; the normalized failure kind is the low-cardinality report, and the
    # message belongs in a log line at most (FAILURE_KINDS.md).
    with tracer(scope).start_as_current_span(
        name,
        context=context,
        attributes=span_attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as active:
        try:
            yield active
        except Exception as error:
            record_failure(active, failure_kind(error, type(error).__name__))
            raise
        else:
            active.set_attribute(_OUTCOME_ATTR, Ok)


def bind[**P, T](func: collections.abc.Callable[P, T]) -> collections.abc.Callable[P, T]:
    """Capture the current OTEL context now and reattach it inside ``func``.

    A span started on the far side of a thread hop starts a new trace root
    unless the caller's context travels with the work. ``asyncio.to_thread``
    copies the calling context for you; a ``threading.Thread`` target and any
    long-lived worker thread do not — a worker started once at bring-up runs
    forever in whatever context existed then, which is not the context of the
    item it is currently processing.

    Bind at the point the work is handed over, not at thread start::

        threading.Thread(target=span.bind(self._run), daemon=True).start()
        self._queue.put(span.bind(functools.partial(self._decode, frame)))

    Attach/detach is balanced, so a pooled or long-lived thread is left in the
    context it had before the call.
    """

    captured = otel_context.get_current()

    @functools.wraps(func)
    def run(*args: P.args, **kwargs: P.kwargs) -> T:
        token = otel_context.attach(captured)
        try:
            return func(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return run


__all__ = [
    "Attributes",
    "AttributeValue",
    "Failed",
    "Ok",
    "bind",
    "failure_kind",
    "normalized_kind",
    "operation",
    "record_failure",
    "tracer",
]
