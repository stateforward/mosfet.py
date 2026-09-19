"""Span helpers so instrumentation is uniform across bot code and providers.

Spans are the second-choice signal (``METRICS > SPANS > Logs``): reach for a
span when the question is *what happened to this one thing as it moved*, which
metrics cannot answer. Attributes stay low-cardinality and payload-free —
component, stage, outcome, normalized failure kind. Never audio bytes,
transcripts, prompts, credentials, or per-event/instance identifiers.

Failure-kind vocabulary and how kinds relate to typed failure events:
see ``FAILURE_KINDS.md`` next to this module.

Emission is a no-op until an application boundary calls
``mosfet.telemetry.configure()``; library models never install a provider
themselves.
"""

from __future__ import annotations

import asyncio
import collections.abc
import contextlib
import contextvars
import dataclasses
import functools
import re
import typing

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from mosfet.telemetry.configure import tracer_provider

_COMPONENT_ATTR = "bot.component.name"
_STAGE_ATTR = "bot.stage"
_OUTCOME_ATTR = "bot.outcome"
_FAILURE_KIND_ATTR = "bot.failure.kind"

_MAX_KIND_LENGTH = 64
_UNKNOWN_KIND = "unknown"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

Ok = "ok"
Failed = "failed"
Cancelled = "cancelled"
HandedOff = "handed_off"
_HANDOFF_ATTR = "bot.handoff.event"

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
    cleaned = "".join(character if character.isascii() and character.isalnum() else "_" for character in split.lower())
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


@dataclasses.dataclass(eq=False)
class _Stage:
    """One open ``operation()`` and the hand-off it is waiting to be cancelled by, if any."""

    span: Span
    handoff: str | None = None
    moved: collections.abc.Callable[[], bool] | None = None


# The operations open in the current behavior (see `handoff_scope`), innermost last. A list, not a
# tuple, on purpose: a task a behavior creates copies this context and must register into the same
# scope, so a hand-off dispatched from it reaches the stages that are awaiting it.
_OPEN_STAGES: contextvars.ContextVar[list[_Stage] | None] = contextvars.ContextVar(
    "mosfet_telemetry_open_stages", default=None
)


@contextlib.contextmanager
def handoff_scope() -> collections.abc.Generator[None]:
    """Start a fresh set of open stages for one HSM behavior invocation.

    `expect_handoff` marks only the stages opened inside the behavior that dispatches, never those
    of another machine's behavior further up the await chain.
    """

    token = _OPEN_STAGES.set([])
    try:
        yield
    finally:
        _OPEN_STAGES.reset(token)


def expect_handoff(event_name: str, *, moved: collections.abc.Callable[[], bool]) -> None:
    """Mark every stage open in this behavior as handing off by dispatching ``event_name``.

    An activity that dispatches an event to its own machine may be cancelled by the transition
    that event takes. If one of these stages is later cancelled while ``moved()`` holds, the
    cancellation is that hand-off and the span reads ``bot.outcome=handed_off`` with
    ``bot.handoff.event=<event_name>``; otherwise (the machine stopping, a caller giving up) it is
    a real cancellation and reads ``cancelled``. ``event_name`` must be a modeled event name.
    """

    for stage in _OPEN_STAGES.get() or ():
        stage.handoff = event_name
        stage.moved = moved


def record_current_failure(kind: str) -> None:
    """Mark the innermost active span failed with a normalized failure kind.

    For code that reports a failure by dispatching a typed failure event instead
    of raising: the enclosing ``operation()`` would otherwise close ``ok`` and the
    trace would show a stage that finished when it did not. Call it from the one
    place that converts a stage error into its typed failure event, with the same
    kind that goes on the event.
    """

    record_failure(trace.get_current_span(), kind)


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
        open_stage = _Stage(active)
        stages = _OPEN_STAGES.get()
        if stages is None:
            stages = []
            _ = _OPEN_STAGES.set(stages)
        stages.append(open_stage)
        try:
            yield active
        except Exception as error:
            record_failure(active, failure_kind(error, type(error).__name__))
            raise
        except asyncio.CancelledError:
            # Cancellation is not a fault, but a stage that leaves no outcome at all is
            # indistinguishable from one still running. A stage whose own dispatch moved its
            # machine on (``expect_handoff``) was cancelled by that hand-off: record it as such so
            # it never reads like an error. Any other cancellation is recorded as ``cancelled``
            # without an ERROR status (FAILURE_KINDS.md: cancelled, not broken).
            if open_stage.handoff is not None and (open_stage.moved is None or open_stage.moved()):
                active.set_attribute(_HANDOFF_ATTR, open_stage.handoff)
                active.set_attribute(_OUTCOME_ATTR, HandedOff)
            else:
                active.set_attribute(_OUTCOME_ATTR, Cancelled)
            raise
        except BaseException:
            record_failure(active, "cancelled")
            raise
        else:
            # Not every failure raises. A guard that declines, a chunk that is dropped, a stage
            # that reports by dispatching a typed failure event all leave the block cleanly after
            # calling record_failure; overwriting the outcome here would report them as successes.
            recorded = typing.cast("dict[str, object] | None", getattr(active, "attributes", None))
            if recorded is not None and recorded.get(_OUTCOME_ATTR) == Failed:
                return
            active.set_attribute(_OUTCOME_ATTR, Ok)
        finally:
            if open_stage in stages:
                stages.remove(open_stage)


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
    "Cancelled",
    "Failed",
    "HandedOff",
    "Ok",
    "bind",
    "expect_handoff",
    "failure_kind",
    "handoff_scope",
    "normalized_kind",
    "operation",
    "record_current_failure",
    "record_failure",
    "tracer",
]
