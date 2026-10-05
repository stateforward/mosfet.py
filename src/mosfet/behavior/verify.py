"""Dry-run apply of a behavior against a sample input for install-time diagnostics.

A turn behavior is applied to the live turn's input. A persistent routine has no input to
apply: it is started on a timer clock that fires its first schedule wait at once, and the
tick it emits is checked instead.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import datetime
import typing
import uuid

import concurrent.futures

import hsm
import mosfet

from mosfet.abilities import processing
from mosfet.abilities.ability import FailureData
from mosfet.protocols import attachment

from . import compiler
from . import diagnostic
from . import events
from . import instance
from . import schedule
from mosfet.telemetry import span
from mosfet.telemetry.hsm import Traced


# Fixed wall time a routine's dry-run tick reads unless the caller injects a clock.
_VERIFY_NOW = datetime.datetime(2026, 1, 5, 8, 0, tzinfo=datetime.UTC)


def _verify_clock() -> datetime.datetime:
    return _VERIFY_NOW


def _error(*, code: str, message: str, stage: diagnostic.Stage) -> diagnostic.Diagnostic:
    return diagnostic.diagnostic(
        code=code,
        message=message.strip() or code,
        stage=stage,
        help=diagnostic.help_for(code),
    )


class _TerminalOwner(Traced):
    """Minimal owner that completes when the behavior ability forwards a terminal.

    Topology selects the terminal via explicit ``hsm.on(output_event)`` /
    ``hsm.on(failed_event)`` transitions with typed-payload guards (no ``event.name``
    routing). Single-shot: the first correlated terminal settles ``result``.
    """

    result: asyncio.Future[object]

    def __init__(self) -> None:
        super().__init__()
        self.result = asyncio.get_running_loop().create_future()

    @staticmethod
    def _is_output_data(ctx: hsm.Context, instance: "_TerminalOwner", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        # Topology already selected the behavior's output event; any payload the
        # behavior produced (selections object or list) is the terminal data.
        return event.data is not None

    @staticmethod
    def _is_failure_data(ctx: hsm.Context, instance: "_TerminalOwner", event: hsm.Event[typing.Any]) -> bool:
        del ctx, instance
        return isinstance(event.data, FailureData)

    @staticmethod
    def _settle_output(ctx: hsm.Context, instance: "_TerminalOwner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if not instance.result.done():
            instance.result.set_result(event.data)

    @staticmethod
    def _settle_failed(ctx: hsm.Context, instance: "_TerminalOwner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        if not instance.result.done():
            failure = event.data
            assert isinstance(failure, FailureData)
            message = failure.message if failure.message else str(failure)
            instance.result.set_exception(RuntimeError(message))

    @classmethod
    def model_for(cls, output_event: hsm.Event[typing.Any], failed_event: hsm.Event[typing.Any]) -> hsm.Model:
        """Define the one-shot waiting model for this behavior's terminal events."""

        return mosfet.define(
            "BehaviorVerifyOwner",
            hsm.initial(hsm.target("/BehaviorVerifyOwner/waiting")),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(output_event),
                    hsm.guard(cls._is_output_data),
                    hsm.effect(cls._settle_output),
                    hsm.target("/BehaviorVerifyOwner/done"),
                ),
                hsm.transition(
                    hsm.on(failed_event),
                    hsm.guard(cls._is_failure_data),
                    hsm.effect(cls._settle_failed),
                    hsm.target("/BehaviorVerifyOwner/done"),
                ),
            ),
            hsm.final("done"),
        )


class _FirstScheduleWaitFires(hsm.Clock):
    """Timer clock for a routine dry run: once armed, one schedule wait fires at once; no other does.

    The routine's lifecycle waits on this clock too (its attach timeout), so it stays unarmed
    until the routine reports attach complete; by then the attach wait is cancelled and every
    open wait belongs to a schedule of the attached routine.
    """

    _armed: bool
    _fired: bool
    _waits: list[asyncio.Future[datetime.datetime]]
    _wall_clock: schedule.WallClock

    def __init__(self, *, clock: schedule.WallClock) -> None:
        self._armed = False
        self._fired = False
        self._waits = []
        self._wall_clock = clock
        super().__init__(after=self._wait)

    def _fire(self, waiter: asyncio.Future[datetime.datetime]) -> None:
        self._fired = True
        waiter.set_result(self._wall_clock())

    def _wait(self, duration: datetime.timedelta) -> asyncio.Future[datetime.datetime]:
        del duration
        waiter: asyncio.Future[datetime.datetime] = asyncio.get_running_loop().create_future()
        if self._armed and not self._fired:
            self._fire(waiter)
        else:
            self._waits.append(waiter)
        return waiter

    def arm(self) -> None:
        """Fire the oldest open schedule wait now, or the next one the routine opens."""

        self._armed = True
        for waiter in self._waits:
            if not self._fired and not waiter.done():
                self._fire(waiter)


class _TickOwner(_TerminalOwner):
    """Dry-run owner of a routine: arms the timer clock on attach, then settles on the first tick."""

    _timers: _FirstScheduleWaitFires

    def __init__(self, *, timers: _FirstScheduleWaitFires) -> None:
        super().__init__()
        self._timers = timers

    @staticmethod
    def _arm(ctx: hsm.Context, instance: "_TickOwner", event: hsm.Event[typing.Any]) -> None:
        del ctx, event
        instance._timers.arm()

    @classmethod
    def tick_model(cls, failed_event: hsm.Event[typing.Any]) -> hsm.Model:
        """Wait for attach complete, then for the routine's first tick or failure."""

        return mosfet.define(
            "BehaviorVerifyTickOwner",
            hsm.initial(hsm.target("/BehaviorVerifyTickOwner/attaching")),
            hsm.state(
                "attaching",
                hsm.transition(
                    hsm.on(attachment.AttachCompleteEvent),
                    hsm.effect(cls._arm),
                    hsm.target("/BehaviorVerifyTickOwner/waiting"),
                ),
                hsm.transition(
                    hsm.on(failed_event),
                    hsm.guard(cls._is_failure_data),
                    hsm.effect(cls._settle_failed),
                    hsm.target("/BehaviorVerifyTickOwner/done"),
                ),
            ),
            hsm.state(
                "waiting",
                hsm.transition(
                    hsm.on(events.TickEvent),
                    hsm.guard(cls._is_output_data),
                    hsm.effect(cls._settle_output),
                    hsm.target("/BehaviorVerifyTickOwner/done"),
                ),
                hsm.transition(
                    hsm.on(failed_event),
                    hsm.guard(cls._is_failure_data),
                    hsm.effect(cls._settle_failed),
                    hsm.target("/BehaviorVerifyTickOwner/done"),
                ),
            ),
            hsm.final("done"),
        )


async def _apply_once(
    program: str,
    *,
    input_data: object,
    metadata: collections.abc.Mapping[str, object],
    timeout: float,
    operation_id: str | None = None,
    event_id: str | None = None,
    source: str | None = None,
    target: str | None = None,
) -> tuple[object, str, str]:
    """Dry-run one behavior apply. Returns ``(output, operation_id, owner_id)``.

    When ``event_id`` / ``source`` / ``target`` are provided they stamp the behavior
    event fields (same shape Autonomy uses for the live turn event). Otherwise
    ``source`` defaults to the verify owner so replies targeting ``event['source']``
    can route back.
    """

    behavior = compiler.build(program)
    owner = _TerminalOwner()
    ctx = hsm.Context()
    _ = await mosfet.started(ctx, owner, _TerminalOwner.model_for(behavior.output_event, behavior.failed_event))
    owner_id = hsm.id(owner)
    attached = False
    try:
        _ = await behavior.attach(
            ctx,
            dataclasses.replace(
                attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
                source=owner_id,
            ),
        )
        attached = True
        resolved_operation_id = operation_id if operation_id else uuid.uuid4().hex
        input_event = dataclasses.replace(
            behavior.input_event.with_data(input_data),
            id=event_id or resolved_operation_id,
            metadata=dict(metadata),
            source=source or owner_id,
            target=target or hsm.id(behavior),
        )
        _ = await hsm.dispatch(ctx, behavior, input_event)
        output = await asyncio.wait_for(owner.result, timeout=timeout)
        return output, resolved_operation_id, owner_id
    finally:
        if attached:
            try:
                await behavior.detach(
                    ctx,
                    dataclasses.replace(
                        attachment.DetachEvent.with_data(attachment.DetachData(actor=owner, reply_to=owner)),
                        source=owner_id,
                        target=hsm.id(behavior),
                    ),
                )
            except Exception:
                # Tear down is best-effort for a dry-run diagnostic; the behavior outcome
                # is authoritative, but the owner must still be released below.
                pass
        _ = await hsm.stop(owner, ctx)


async def _tick_once(
    program: str,
    *,
    clock: schedule.WallClock,
    timeout: float,
) -> events.TickData:
    """Dry-run one routine tick: start it on a timer that fires once, wait for its tick terminal."""

    behavior = compiler.build(program, clock=clock)
    timers = _FirstScheduleWaitFires(clock=clock)
    owner = _TickOwner(timers=timers)
    ctx = hsm.Context()
    _ = await mosfet.started(ctx, owner, _TickOwner.tick_model(behavior.failed_event))
    owner_id = hsm.id(owner)
    attached = False
    try:
        model = behavior.model
        assert model is not None
        _ = await mosfet.started(ctx, behavior, model, hsm.Config(clock=timers))
        _ = await behavior.attach(
            ctx,
            dataclasses.replace(
                attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
                source=owner_id,
            ),
        )
        attached = True
        output = await asyncio.wait_for(owner.result, timeout=timeout)
        if not isinstance(output, events.TickData):
            raise RuntimeError("persistent routine emitted output that is not a bot.behavior.tick.")
        return output
    finally:
        if attached:
            try:
                _ = await behavior.detach(
                    ctx,
                    dataclasses.replace(
                        attachment.DetachEvent.with_data(attachment.DetachData(actor=owner, reply_to=owner)),
                        source=owner_id,
                        target=hsm.id(behavior),
                    ),
                )
            except Exception:
                # Tear down is best-effort for a dry-run diagnostic; the tick outcome is authoritative.
                pass
        await behavior.stop(ctx)
        _ = await hsm.stop(owner, ctx)


def _live_binding_values(
    input_data: object,
    *,
    event_id: str = "",
    source: str = "",
    target: str = "",
) -> dict[str, object]:
    """Collect scalar live values for selection binding equality.

    Live bindings come from input_data fields and optional event
    id/source/target. Telemetry metadata is never a coordination ID source.
    """

    values: dict[str, object] = {}
    if isinstance(input_data, collections.abc.Mapping):
        mapping = typing.cast(collections.abc.Mapping[object, object], input_data)
        for key, item in mapping.items():
            if isinstance(key, str) and isinstance(item, str | int | float | bool):
                values[key] = item
    if event_id:
        _ = values.setdefault("id", event_id)
    if source:
        _ = values.setdefault("source", source)
    if target:
        _ = values.setdefault("target", target)
    return values


def _selection_binding_errors(
    output: object,
    *,
    live_values: collections.abc.Mapping[str, object],
) -> tuple[str, ...]:
    selections = processing.coerce_event_selections(output)
    if selections is None:
        return ("behavior output must be a cognition event selection object (or list of them).",)
    if not selections:
        return ("behavior output must select at least one event for the observed pattern.",)
    errors: list[str] = []
    for selection in selections:
        data = selection.data
        if not isinstance(data, collections.abc.Mapping):
            continue
        mapping = typing.cast(collections.abc.Mapping[object, object], data)
        for key, value in mapping.items():
            if not isinstance(key, str):
                continue
            live = live_values.get(key)
            if live is None:
                continue
            if value != live:
                message = (
                    f"selection data[{key!r}]={value!r} does not match live event value "
                    + f"{live!r}; read identifiers from event['data'] and "
                    + "event['id']/event['source']/event['target'] at runtime."
                )
                errors.append(message)
    return tuple(errors)


def _run_coroutine(coro: collections.abc.Awaitable[object]) -> object:
    """Run ``coro`` whether or not a loop is already running (HSM effects are sync)."""

    try:
        _ = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(typing.cast(collections.abc.Coroutine[typing.Any, typing.Any, object], coro))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        # The worker thread starts in an empty context: carry the caller's trace into it.
        return pool.submit(
            span.bind(asyncio.run),
            typing.cast(collections.abc.Coroutine[typing.Any, typing.Any, object], coro),
        ).result()


def verify_apply(
    program: str,
    *,
    name: str | None = None,
    triggers: tuple[str, ...] | None = None,
    description: str | None = None,
    examples: tuple[str, ...] | None = None,
    input_data: object,
    metadata: collections.abc.Mapping[str, object] | None = None,
    timeout: float = 2.0,
    event_id: str | None = None,
    source: str | None = None,
    target: str | None = None,
    min_every: datetime.timedelta = instance.ROUTINE_MIN_EVERY,
    clock: schedule.WallClock = _verify_clock,
) -> diagnostic.Checked[instance.Instance]:
    """Parse/build, then dry-run apply; return structured diagnostics on failure.

    Optional ``event_id`` / ``source`` / ``target`` stamp the behavior input event the
    same way Autonomy does for a live turn event (not metadata). A persistent routine is
    instead started on a timer that fires its first schedule at once (``clock`` is the wall
    clock its tick reads; ``min_every`` the shortest interval it may use), and the
    selections its tick emits are checked; ``input_data`` does not apply to it.
    """

    checked = instance.check(
        program,
        name=name,
        triggers=triggers,
        description=description,
        examples=examples,
        require_build=True,
        min_every=min_every,
    )
    if not checked.ok or checked.value is None:
        return checked

    if checked.value.lifetime == instance.LIFETIME_PERSISTENT:
        return _verify_tick(program, checked=checked, clock=clock, timeout=timeout)

    # metadata is telemetry pass-through into apply_once only — never selection binding.
    meta = dict(metadata or {})
    operation_id = uuid.uuid4().hex
    try:
        output, resolved_operation_id, owner_id = typing.cast(
            tuple[object, str, str],
            _run_coroutine(
                _apply_once(
                    program,
                    input_data=input_data,
                    metadata=meta,
                    timeout=timeout,
                    operation_id=operation_id,
                    event_id=event_id,
                    source=source,
                    target=target,
                )
            ),
        )
    except Exception as error:
        report = diagnostic.report_of(
            _error(
                code=diagnostic.E0008_APPLY,
                message=str(error),
                stage=diagnostic.Stage.APPLY,
            )
        )
        return diagnostic.Checked[instance.Instance](value=None, report=report)

    live_event_id = event_id or resolved_operation_id
    live_values = _live_binding_values(
        input_data,
        event_id=live_event_id,
        source=source or owner_id,
        target=target or "",
    )
    binding_errors = _selection_binding_errors(output, live_values=live_values)
    if binding_errors:
        report = diagnostic.report_of(
            *(
                _error(
                    code=diagnostic.E0008_APPLY,
                    message=message,
                    stage=diagnostic.Stage.APPLY,
                )
                for message in binding_errors
            )
        )
        return diagnostic.Checked[instance.Instance](value=None, report=report)

    return checked


def _verify_tick(
    program: str,
    *,
    checked: diagnostic.Checked[instance.Instance],
    clock: schedule.WallClock,
    timeout: float,
) -> diagnostic.Checked[instance.Instance]:
    try:
        tick = typing.cast(
            events.TickData,
            _run_coroutine(_tick_once(program, clock=clock, timeout=timeout)),
        )
    except Exception as error:
        message = str(error) or "persistent routine did not emit a tick for its first schedule."
        report = diagnostic.report_of(
            _error(code=diagnostic.E0008_APPLY, message=message, stage=diagnostic.Stage.APPLY)
        )
        return diagnostic.Checked[instance.Instance](value=None, report=report)
    live_values = _live_binding_values(tick.model_dump(mode="json", exclude={"output"}))
    binding_errors = _selection_binding_errors(tick.output, live_values=live_values)
    if binding_errors:
        report = diagnostic.report_of(
            *(
                _error(code=diagnostic.E0008_APPLY, message=message, stage=diagnostic.Stage.APPLY)
                for message in binding_errors
            )
        )
        return diagnostic.Checked[instance.Instance](value=None, report=report)
    return checked


__all__ = ["verify_apply"]
