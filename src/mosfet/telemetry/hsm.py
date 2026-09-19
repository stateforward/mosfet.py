"""OpenTelemetry observation helpers for stateforward.mosfet HSM models.

Observation is opt-in at the application boundary: wire ``hsm.observe(observer)``
into a model and call ``mosfet.telemetry.configure()`` from the owning runtime.
Until ``configure()`` installs the JSONL exporters, emission bottoms out in the
OpenTelemetry API no-op implementations, so an unwired process pays only the
attribute-building cost of each observation and records nothing.

Reusable library/protocol models must not force observation; their callers
decide (see ``AGENTS.md`` observability). The ``hsm.observe(observer)`` lines
on models in this repo are that wiring, applied by each model's owning
boundary — not a mandate on downstream consumers. Wire it once, on the machine that is started
(an ability's lifecycle model, a bot, a device), never also on a submodel embedded in it: ``hsm``
wraps every member the observation sees, so an observed submodel inside an observed lifecycle
records each of its events twice.

Propagation — one stimulus is one trace:

- A device ingress (``bot.device.ingress``) roots the trace of a stimulus that arrives carrying
  none, or continues the one it carries.
- Every dispatch into a `Traced` machine stamps the dispatcher's context into
  ``hsm.Event.metadata`` (`stamp_context`); ``hsm`` runs a machine's queue in whichever task
  started its processing run, so without the stamp an event queued behind a busy machine would
  join that run's trace instead of its own.
- `EventContextBinding`, installed by ``mosfet.define`` as the model's validator and finalizer, runs every behavior and guard with its
  event's context attached, so spans, log records, tasks, thread hops (``asyncio.to_thread`` and
  ``span.bind``), provider calls, and further dispatches all join the event's trace with the
  right parent.
- Work a turn starts in the background — reflection, behavior revision, inventory writes — is
  caused by the turn's own events and runs while the turn's context is attached, so it is a child
  in the same trace (parent/child, not a span link). A span link would be the right shape only
  for work that is scheduled by one trace and later run on behalf of several; nothing here does.

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
import contextvars
import dataclasses
import datetime
import functools
import inspect
import logging
import sys
import threading
import time
import typing

import hsm
from opentelemetry import context as otel_context
from opentelemetry import metrics, propagate, trace
from opentelemetry.context import Context as OtelContext
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from mosfet.telemetry import capture
from mosfet.telemetry.span import expect_handoff, handoff_scope, tracer

_INSTRUMENTATION_SCOPE = "bot.telemetry.hsm"
_OBSERVATION_SPAN_NAME = "bot.hsm.observe"
_LOG = logging.getLogger(_INSTRUMENTATION_SCOPE)

_METER = metrics.get_meter(_INSTRUMENTATION_SCOPE)


def _tracer() -> Tracer:
    """The configured exporting provider's tracer, resolved per observation.

    OpenTelemetry lets the global tracer provider be installed once per process, so a tracer taken
    from it at import keeps writing to whichever provider was installed first even after
    ``configure()`` / ``reset()`` replaced it; observation spans would then leave the trace file
    their own log records land in.
    """

    return tracer(_INSTRUMENTATION_SCOPE)


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
_CAPTURE_LOGGER_NAME = "bot.telemetry.hsm.event"

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


def _carried_trace_id(event: hsm.Event[object]) -> int | None:
    carrier = _carrier_from_metadata(event.metadata)
    if not carrier:
        return None
    carried = trace.get_current_span(propagate.extract(carrier=carrier)).get_span_context()
    return carried.trace_id if carried.is_valid else None


def stamp_context[TEventData](event: hsm.Event[TEventData]) -> hsm.Event[TEventData]:
    """Return ``event`` carrying the trace context it is dispatched in.

    Called on every dispatch into a traced machine (`Traced.dispatch`), so an event queued behind
    a busy machine is still processed in its own dispatcher's trace rather than in whichever
    trace happened to start that machine's processing run.

    The active context wins when the event carries nothing, or carries a context of the same trace
    (the active span is then the more precise parent: a descendant of what was carried). An event
    that already carries a *different* trace keeps it: that is a foreign origin (another bot, a
    transport hop, a replayed stimulus), and overwriting it would cut the chain it came from.
    """

    active = trace.get_current_span().get_span_context()
    if not active.is_valid:
        return event
    carried = _carried_trace_id(event)
    if carried is not None and carried != active.trace_id:
        return event
    return inject_context(event)


class Traced(hsm.Instance):
    """An ``hsm.Instance`` whose every dispatch carries the dispatcher's trace context.

    ``hsm`` processes a machine's queue in the task that started the current processing run, so
    ``contextvars`` alone lose the dispatcher of any event queued while the machine was busy. The
    context therefore rides the event (`stamp_context`), and `EventContextBinding` puts it back
    around every behavior that event runs (`EventContextBinding`).
    """

    @typing.override
    def dispatch(self, ctx: hsm.Context, event: hsm.Event[typing.Any]) -> collections.abc.Awaitable[bool]:
        return deliver(self, ctx, event)


def deliver(
    target: hsm.Instance,
    ctx: hsm.Context,
    event: hsm.Event[typing.Any],
) -> collections.abc.Awaitable[bool]:
    """Queue ``event`` on ``target``'s own machine, stamped with the active trace context.

    The stamping counterpart of ``hsm.Instance.dispatch`` for overrides that route past their own
    ``dispatch`` (a device shell, an attachment group reply) — every queue entry is stamped.
    """

    if _BEHAVIOR_INSTANCE.get() is target:
        # A behavior dispatching to its own machine may be cancelled by the transition it causes.
        expect_handoff(event.name, moved=_alive(target))
    return hsm.Instance.dispatch(target, ctx, stamp_context(event))


def _alive(machine: hsm.Instance) -> collections.abc.Callable[[], bool]:
    """Whether a cancellation of ``machine``'s behavior is a hand-off rather than a stop.

    Stopping cancels the machine's own context before exiting its states; a transition leaves it
    running.
    """

    return lambda: not machine.context().is_done()


# The machine whose behavior is running in this context (set by `EventContextBinding`).
_BEHAVIOR_INSTANCE: contextvars.ContextVar[hsm.Instance | None] = contextvars.ContextVar(
    "mosfet_telemetry_behavior_instance", default=None
)


_EVENT_CONTEXT_BOUND = "__mosfet_event_context__"

type _Operation = collections.abc.Callable[[hsm.Context, typing.Any, hsm.Event[typing.Any]], typing.Any]


def _bound_operation(operation: _Operation) -> _Operation:
    if getattr(operation, _EVENT_CONTEXT_BOUND, False):
        return operation
    if isinstance(operation, staticmethod):
        # A class-body reference (``hsm.activity(_run)`` under ``@staticmethod``) is the descriptor,
        # not the function: unwrapped, an ``async def`` reads as the coroutine function it is.
        operation = typing.cast(_Operation, typing.cast(typing.Any, operation).__func__)
    if inspect.iscoroutinefunction(operation):

        @functools.wraps(operation)
        async def run_async(ctx: hsm.Context, instance: typing.Any, event: hsm.Event[typing.Any]) -> typing.Any:
            token = otel_context.attach(event_context(event))
            owner = _BEHAVIOR_INSTANCE.set(instance)
            try:
                with handoff_scope():
                    return await operation(ctx, instance, event)
            finally:
                _BEHAVIOR_INSTANCE.reset(owner)
                otel_context.detach(token)

        setattr(run_async, _EVENT_CONTEXT_BOUND, True)
        return run_async

    @functools.wraps(operation)
    def run(ctx: hsm.Context, instance: typing.Any, event: hsm.Event[typing.Any]) -> typing.Any:
        # A returned awaitable is a task or future already created here, so it copied this context.
        token = otel_context.attach(event_context(event))
        owner = _BEHAVIOR_INSTANCE.set(instance)
        try:
            with handoff_scope():
                return operation(ctx, instance, event)
        finally:
            _BEHAVIOR_INSTANCE.reset(owner)
            otel_context.detach(token)

    setattr(run, _EVENT_CONTEXT_BOUND, True)
    return run


@dataclasses.dataclass(kw_only=True)
class EventContextBinding(hsm.DefaultModelValidator):
    """Model validator and finalizer: run every behavior and guard in its triggering event's trace context.

    ``mosfet.define`` installs one instance as both the model's ``hsm.validator`` and its
    ``hsm.finalizer``. Both are model elements, so they follow the model through ``hsm.redefine``:
    every member, including members a redefine adds, is bound whenever the model is built, not
    only the members present when the model was first defined.

    Binding runs before ``hsm.DefaultModelFinalizer`` (so a class-body ``staticmethod`` activity
    reads as the coroutine function it wraps) and again after it, so the ``hsm.observe``
    wrappers that finalizer applies also run in the event's context. The model is validated
    once, finished: ``hsm`` applies observations when it finalizes, and a guarded transition
    with no target and no effect (one that consumes a stale or late event) is valid only once
    its observation effect is in place.

    Entry, exit, effect, activity, operation, and guard callables are wrapped so the event's
    carried context (`event_context`) is the active one while they run: spans they start,
    records they log, tasks they create, and events they dispatch all join the event's trace
    under the right parent. Wrapping is idempotent (a submodel embedded in several models, or a
    member seen by both passes, is bound once), sync stays sync and async stays async, as ``hsm``
    validates both.
    """

    @staticmethod
    def _bind(model: hsm.Model) -> None:
        for member in model.members.values():
            if isinstance(member, hsm.BehaviorElement):
                behavior = typing.cast(hsm.BehaviorElement[typing.Any], member)
                behavior.operation = _bound_operation(typing.cast(_Operation, behavior.operation))
            elif isinstance(member, hsm.ConstraintElement):
                constraint = typing.cast(hsm.ConstraintElement[typing.Any], member)
                if callable(constraint.expression):
                    constraint.expression = _bound_operation(typing.cast(_Operation, constraint.expression))

    @typing.override
    def validate(self, model: hsm.Model) -> None:
        # Validation waits for the finished model (see `finalize`).
        self._bind(model)

    def finalize(self, model: hsm.Model) -> hsm.Model:
        finalized = hsm.DefaultModelFinalizer().finalize(model)
        self._bind(finalized)
        super().validate(finalized)
        return finalized


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


def _transition_endpoints(instance: hsm.Instance, qualified_name: str) -> tuple[str, str] | None:
    for transition in instance.take_snapshot().Transitions:
        if transition.qualified_name == qualified_name:
            return transition.source, transition.target
    return None


def _capture_observation(
    observation: hsm.Event[typing.Any],
    *,
    instance: hsm.Instance,
    attributes: dict[str, object],
) -> None:
    """Under ``BOT_OTEL_CAPTURE=full``, log the complete observed event in the active trace.

    Transition occurrences carry the event's serialized data, correlation ids, and the
    consuming transition's ``from``/``to`` states; behavior occurrences (entry/exit/activity)
    name the behavior and the event id without repeating the payload.
    """

    event = observed_event(observation)
    occurrence = observed_occurrence(observation)
    body: dict[str, object] = {
        "occurrence": occurrence,
        "component": attributes.get("bot.component.name"),
        "machine": attributes.get("hsm.machine.name"),
        "instance": hsm.id(instance),
        "state": attributes.get("hsm.machine.state"),
        "element": observation.source,
        "event": {
            "name": event.name,
            "kind": event.kind,
            "id": event.id,
            "source": event.source,
            "target": event.target,
        },
    }
    if occurrence == "event":
        typing.cast(dict[str, object], body["event"])["data"] = event.data
        endpoints = _transition_endpoints(instance, observation.source)
        if endpoints is not None:
            body["transition"] = {"from": endpoints[0], "to": endpoints[1]}
    capture.emit(
        _CAPTURE_LOGGER_NAME,
        body=body,
        attributes={
            "component": str(attributes.get("bot.component.name", "")),
            "stage": occurrence,
            "hsm.event.name": event.name,
        },
    )


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
        with _tracer().start_as_current_span(
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
            if capture.full_enabled():
                try:
                    _capture_observation(observation, instance=instance, attributes=attributes)
                except Exception:
                    _note_pipeline_failure("capture")
    except Exception:
        # Never break the HSM transition being observed, but never vanish
        # either: a dead tracer must page the operator once a minute, not
        # swallow every observation for the life of the process.
        _note_pipeline_failure("observe_span")


__all__ = [
    "ObservationData",
    "Traced",
    "EventContextBinding",
    "deliver",
    "stamp_context",
    "event_context",
    "inject_context",
    "observer",
    "observation_attributes",
    "observed_event",
    "observed_occurrence",
    "pipeline_failure_total",
    "propagate",
]
