"""OpenTelemetry observation helpers for stateforward.mosfet HSM models.

Observation is opt-in at the application boundary: wire ``hsm.observe(observer)``
into a model and call ``mosfet.telemetry.configure()`` from the owning runtime.
Until ``configure()`` installs the JSONL exporters, emission bottoms out in the
OpenTelemetry API no-op implementations, so an unwired process pays only the
attribute-building cost of each observation and records nothing.

Reusable library/protocol models must not force observation; their callers
decide (see ``AGENTS.md`` observability). The ``hsm.observe(observer)`` lines
on models in this repo are that wiring, applied by each model's owning
boundary — not a mandate on downstream consumers.

``hsm.event.kind`` is the numeric ``hsm`` event-kind discriminant, kept as-is
so dashboards can filter on the stable value. Mapping (from ``hsm``):

| ``hsm.event.kind`` | Meaning |
| --- | --- |
| 280 | ``EventKind`` — an ordinary domain event (the default) |
| 71705 | ``CompletionEventKind`` — a terminal/completion signal |
| 71707 | ``TimeEventKind`` — a modeled ``hsm.after(...)`` deadline |
| 71709 | ``CallEventKind`` — an enabled call event (a model-callable tool) |
| 18356506 | ``ErrorEventKind`` — a typed failure event |
"""

import collections.abc
import dataclasses
import datetime
import logging
import sys
import threading
import time
import typing

import hsm
from opentelemetry import context as otel_context
from opentelemetry import metrics, propagate, trace
from opentelemetry.context import Context as OtelContext
from opentelemetry.trace import Span, Status, StatusCode


_INSTRUMENTATION_SCOPE = "bot.telemetry.hsm"
_OBSERVATION_SPAN_NAME = "bot.hsm.observe"
_LOG = logging.getLogger(_INSTRUMENTATION_SCOPE)

_METER = metrics.get_meter(_INSTRUMENTATION_SCOPE)
_TRACER = trace.get_tracer(_INSTRUMENTATION_SCOPE)

_OBSERVATIONS = _METER.create_counter(
    "bot.hsm.observation.count",
    unit="{observation}",
    description="Count of stateforward.mosfet HSM transitions and behavior callbacks observed through hsm.Observe.",
)
_OBSERVATION_FAILURES = _METER.create_counter(
    "bot.hsm.observation.failure.count",
    unit="{failure}",
    description="Count of stateforward.mosfet HSM telemetry observations that failed to record.",
)

_ATTRIBUTE_VALUE = str | bool | int | float
_Attributes = dict[str, _ATTRIBUTE_VALUE]
_OBSERVATION_EVENT_NAME = "hsm/observation"

# Telemetry-about-telemetry: when the observation pipeline itself is down
# (metrics backend and tracer both failing), the failure must still reach an
# operator instead of vanishing inside ``except: pass``. Reports are
# rate-limited to one log + stderr line per minute; anything suppressed in
# between is counted and reported with the next line. Never raises.
_PIPELINE_FAILURE_LOG_INTERVAL_S = 60.0
_pipeline_failure_lock = threading.Lock()
_pipeline_failure_total = 0
_pipeline_failure_suppressed = 0
_pipeline_failure_last_log_monotonic = 0.0


def _note_pipeline_failure(stage: str) -> None:
    """Record that recording one observation failed at ``stage`` (never raises)."""

    global _pipeline_failure_total, _pipeline_failure_suppressed, _pipeline_failure_last_log_monotonic
    try:
        now = time.monotonic()
        with _pipeline_failure_lock:
            _pipeline_failure_total += 1
            _pipeline_failure_suppressed += 1
            if now - _pipeline_failure_last_log_monotonic < _PIPELINE_FAILURE_LOG_INTERVAL_S:
                return
            _pipeline_failure_last_log_monotonic = now
            total = _pipeline_failure_total
            suppressed = _pipeline_failure_suppressed
            _pipeline_failure_suppressed = 0
        _LOG.warning(
            "hsm telemetry pipeline degraded stage=%s suppressed=%d total=%d",
            stage,
            suppressed,
            total,
        )
        # Fallback counter when logging itself is unwired: a batch-processor
        # outage must be operator-visible even with no log handler installed.
        print(
            f"hsm telemetry pipeline degraded stage={stage} suppressed={suppressed} total={total}",
            file=sys.stderr,
        )
    except Exception:
        pass


def pipeline_failure_total() -> int:
    """Return how many observation recordings have failed in this process."""

    with _pipeline_failure_lock:
        return _pipeline_failure_total


class ObservationData(typing.TypedDict):
    """Runtime payload shape emitted by `hsm.Observe`."""

    event: hsm.Event[typing.Any]
    occurrence: str
    time: datetime.datetime


def _metric_attributes(attributes: dict[str, object]) -> _Attributes:
    return {key: value for key, value in attributes.items() if isinstance(value, str | bool | int | float)}


def _carrier_from_metadata(metadata: collections.abc.Mapping[str, object]) -> dict[str, str]:
    return {key: value for key, value in metadata.items() if isinstance(value, str)}


def event_context(event: hsm.Event[object]) -> OtelContext:
    """Return the trace context an observed event belongs to.

    An event stamped by `inject_context` carries its origin's context in metadata and reparents
    there — that is the point of stamping, and it is how a bot-to-bot or transport hop stays one
    trace. An event minted in-process carries nothing, and `propagate.extract` on an empty carrier
    yields an *empty* context, not the ambient one: starting a span in it detaches the event into
    a brand-new root trace at exactly the moment the caller's span is the answer. So an unstamped
    event stays where it already is.
    """

    carrier = _carrier_from_metadata(event.metadata)
    if not carrier:
        return otel_context.get_current()
    return propagate.extract(carrier=carrier)


def inject_context[TEventData](event: hsm.Event[TEventData]) -> hsm.Event[TEventData]:
    """Return ``event`` with the active trace context stamped into its metadata.

    The extract counterpart is `event_context`. In-process the active span rides
    contextvars, so this is only needed where an event crosses a process or
    transport boundary (bot-to-bot over a room, a serialized queue): stamp on the
    way out, extract on the way in.

    `hsm.Event.metadata` carries telemetry propagation only — never domain data,
    identity, or progression decisions. Returns a new event so a shared metadata
    dict is never mutated; a no-op (same event) when no context is active.
    """

    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    if not carrier:
        return event
    return dataclasses.replace(event, metadata={**event.metadata, **carrier})


def _empty_event() -> hsm.Event[typing.Any]:
    return hsm.Event(name="hsm.unknown")


def observed_event(observation: hsm.Event[typing.Any]) -> hsm.Event[typing.Any]:
    """Return the original event carried by an HSM observation event."""

    data = observation.data
    if not isinstance(data, dict):
        return _empty_event()
    values = typing.cast(dict[str, object], data)
    event = values.get("event")
    if isinstance(event, hsm.Event):
        return event
    return _empty_event()


def observed_occurrence(observation: hsm.Event[typing.Any]) -> str:
    """Return the finite observation occurrence emitted by `hsm.Observe`."""

    data = observation.data
    if not isinstance(data, dict):
        return "unknown"
    values = typing.cast(dict[str, object], data)
    occurrence = values.get("occurrence")
    if isinstance(occurrence, str):
        return occurrence
    return "unknown"


def observation_attributes(
    observation: hsm.Event[typing.Any],
    *,
    instance: hsm.Instance | None,
    outcome: str,
    exception_type: str | None = None,
) -> dict[str, object]:
    """Build low-cardinality attributes for an HSM observation.

    Event IDs, instance IDs, source/target IDs, payload values, metadata values, and raw error messages are
    deliberately excluded.
    """

    event = observed_event(observation)
    snapshot = instance.take_snapshot() if instance is not None else None
    attributes: dict[str, object] = {
        "hsm.observation.occurrence": observed_occurrence(observation),
        "hsm.observation.source": observation.source,
        "hsm.event.name": event.name,
        "hsm.event.kind": event.kind,
        "hsm.event.has_data": event.data is not None,
        "hsm.event.has_metadata": bool(event.metadata),
        "hsm.machine.name": snapshot.QualifiedName if snapshot is not None else "",
        "hsm.machine.state": snapshot.State if snapshot is not None else "",
        "bot.component.name": type(instance).__name__ if instance is not None else "unknown",
        "bot.outcome": outcome,
    }
    if exception_type is not None:
        attributes["exception.type"] = exception_type
    return attributes


def _set_span_outcome(span: Span, outcome: str, exception_type: str | None = None) -> None:
    span.set_attribute("bot.outcome", outcome)
    if exception_type is not None:
        span.set_attribute("exception.type", exception_type)
        span.set_status(Status(StatusCode.ERROR, exception_type))


def _record_observation_failure(
    observation: hsm.Event[typing.Any],
    *,
    instance: hsm.Instance,
    span: Span,
    error: Exception,
) -> None:
    exception_type = type(error).__name__
    failure_attributes = observation_attributes(
        observation,
        instance=instance,
        outcome="failed",
        exception_type=exception_type,
    )
    try:
        _OBSERVATION_FAILURES.add(1, attributes=_metric_attributes(failure_attributes))
    except Exception:
        # The failure counter is the last resort before silence: if it is down
        # too, the outage is reported through the rate-limited fallback.
        _note_pipeline_failure("failure_counter")
    try:
        _set_span_outcome(span, "failed", exception_type)
    except Exception:
        _note_pipeline_failure("failure_span")


def _failure_message(event: hsm.Event[typing.Any]) -> str:
    """The human-readable reason carried by a failure event, if it carries one."""

    message = getattr(event.data, "message", None)
    return message if isinstance(message, str) and message else ""


def observer(ctx: hsm.Context, instance: hsm.Instance, observation: hsm.Event[typing.Any]) -> None:
    """Record one HSM observation as metrics and a short trace span.

    When the ``mosfet.telemetry.hsm`` logger is at DEBUG, also emit a structured log line
    with low-cardinality machine/event fields (no payloads, IDs, or audio bytes).

    A failure that carries a reason is logged at ERROR *with* that reason. Metric and span
    attributes stay low-cardinality and payload-free, but a machine reporting why it failed and
    nothing printing it is how a mute robot looks identical to a working one in a log.
    """

    del ctx
    if observation.name != _OBSERVATION_EVENT_NAME:
        return
    event = observed_event(observation)
    attributes = observation_attributes(observation, instance=instance, outcome="observed")
    if event.kind == hsm.ErrorEventKind:
        reason = _failure_message(event)
        if reason:
            _LOG.error(
                "hsm failure component=%s state=%s event=%s reason=%s",
                attributes.get("bot.component.name"),
                attributes.get("hsm.machine.state"),
                attributes.get("hsm.event.name"),
                reason,
            )
    if _LOG.isEnabledFor(logging.DEBUG):
        _LOG.debug(
            "hsm observe component=%s state=%s occurrence=%s event=%s kind=%s has_data=%s",
            attributes.get("bot.component.name"),
            attributes.get("hsm.machine.state"),
            attributes.get("hsm.observation.occurrence"),
            attributes.get("hsm.event.name"),
            attributes.get("hsm.event.kind"),
            attributes.get("hsm.event.has_data"),
        )
    try:
        with _TRACER.start_as_current_span(
            _OBSERVATION_SPAN_NAME,
            context=event_context(event),
            attributes=_metric_attributes(attributes),
        ) as span:
            try:
                _OBSERVATIONS.add(1, attributes=_metric_attributes(attributes))
            except Exception as error:
                _record_observation_failure(observation, instance=instance, span=span, error=error)
                return
            _set_span_outcome(span, "observed")
    except Exception:
        # Never break the HSM transition being observed, but never vanish
        # either: a dead tracer must page the operator once a minute, not
        # swallow every observation for the life of the process.
        _note_pipeline_failure("observe_span")


__all__ = [
    "ObservationData",
    "event_context",
    "inject_context",
    "observer",
    "observation_attributes",
    "observed_event",
    "observed_occurrence",
    "pipeline_failure_total",
    "propagate",
]
