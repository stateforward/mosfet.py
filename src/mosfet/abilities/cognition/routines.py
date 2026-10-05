"""Routines: the persistent routines cognition keeps running across turns and restarts.

A routine is a behavior whose inventory ``lifetime`` is ``persistent`` (see
``mosfet.behavior``). Routines loads every ACTIVE persistent row from the injected Memory
when it attaches, builds each one with the injected wall clock, and starts it under its own
lifetime context with itself as the attachment owner, so the routine's schedules run until
it is stopped. Nothing is replayed after a restart: intervals restart from start and
``at()`` schedules wait for their next occurrence.

Cognition dispatches ``ReconcileEvent`` after Reflection settles a turn (Reflection is the
inventory writer): Routines re-reads the persistent rows and stops a routine that is gone,
broken, or no longer ACTIVE, restarts one whose source changed, and starts one that is new.
Each tick a running routine emits (``bot.behavior.tick``) is forwarded to Cognition, which
hands it to the body as an observation that begins a turn; the bot decides what to do.

Telemetry: ``bot.behavior.routine.tick.count`` {``bot.behavior.routine.trigger`` every|at,
``bot.behavior.routine.outcome`` forwarded|stale}, up-down ``bot.behavior.routine.active``,
and spans ``bot.behavior.routine.reconcile`` / ``.start`` / ``.stop``. Routine names are
never telemetry attributes.
"""

from __future__ import annotations

from .. import ability
from .. import memory

import asyncio
import dataclasses
import logging
import typing

import hsm
import mosfet
import pydantic
from opentelemetry import metrics

from mosfet import behavior
from mosfet.behavior import schedule
from mosfet.behavior import storage as behavior_storage
from mosfet.behavior.instance import LIFETIME_PERSISTENT
from mosfet.behavior.instance import Instance
from mosfet.protocols import attachment
from mosfet.telemetry import span

from . import inventory

_LOG = logging.getLogger(__name__)
_SCOPE = "bot.abilities.cognition.routines"

_TICKS = metrics.get_meter(_SCOPE).create_counter(
    "bot.behavior.routine.tick.count",
    unit="{tick}",
    description="Persistent routine ticks Routines received, by schedule kind and whether it forwarded them.",
)
_ACTIVE = metrics.get_meter(_SCOPE).create_up_down_counter(
    "bot.behavior.routine.active",
    unit="{routine}",
    description="Persistent routines currently running.",
)


class ReconcileData(pydantic.BaseModel):
    """Payload for ``bot.ability.routines.reconcile``: bring running routines in line with inventory."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Ask Routines to re-read the ACTIVE persistent routines in behavior inventory and reconcile "
                "the running set: stop removed or broken routines, restart changed ones, start new ones. "
                "Cognition sends it after Reflection, the inventory writer, settles a turn. Carries no data: "
                "the inventory itself is the record."
            ),
            "examples": [{}],
        },
    )


ReconcileEvent = hsm.Event[ReconcileData](
    name="bot.ability.routines.reconcile",
    schema=ReconcileData,
)


class _ReconciledData(pydantic.BaseModel):
    """Private completion: the running set now matches the persistent inventory that loaded."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    running: int = pydantic.Field(ge=0, description="Routines running after reconciling.")


_ReconciledEvent = hsm.Event[_ReconciledData](
    name="bot.ability.routines.reconciled",
    kind=hsm.CompletionEventKind,
    schema=_ReconciledData,
)
_ReconcileFailedEvent = hsm.Event[ability.FailureData](
    name="bot.ability.routines.reconcile.failed",
    kind=hsm.ErrorEventKind,
    schema=ability.FailureData,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Running:
    """One started routine: its source revision and the live behavior running it."""

    revision: str
    routine: behavior.Behavior
    routine_id: str


class Routines(ability.Ability[ReconcileData, behavior.TickData]):
    """Keeps ACTIVE persistent routines running and forwards their ticks to its owner."""

    input_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = ReconcileData
    output_data_type: typing.ClassVar[type[object] | tuple[type[object], ...] | None] = behavior.TickData
    input_event: typing.ClassVar[hsm.Event[typing.Any]] = ReconcileEvent
    output_event: typing.ClassVar[hsm.Event[typing.Any]] = behavior.TickEvent

    _memory: memory.Memory
    _clock: schedule.WallClock
    _timers: hsm.Clock | None
    _running: dict[str, _Running]
    _stopping: list[behavior.Behavior]

    @staticmethod
    async def _stop_routine(ctx: hsm.Context, routine: behavior.Behavior) -> None:
        with span.operation(
            "bot.behavior.routine.stop",
            scope=_SCOPE,
            component="cognition.routines",
            stage="stop",
        ):
            await routine.stop(ctx)

    @staticmethod
    async def _start_routine(instance: "Routines", row: Instance) -> _Running:
        with span.operation(
            "bot.behavior.routine.start",
            scope=_SCOPE,
            component="cognition.routines",
            stage="start",
        ):
            # Parsing runs an isolated Starlark worker process: keep it off the event loop.
            routine = await asyncio.to_thread(behavior.build, row.source, clock=instance._clock)
            model = routine.model
            assert model is not None
            lifetime = instance.context()
            # Parent under Routines' own lifetime context, not this reconcile activity's
            # (HSM-CONTEXT-001): the routine outlives the activity that starts it.
            _ = await mosfet.started(lifetime, routine, model, hsm.Config(clock=instance._timers))
            try:
                _ = await routine.attach(
                    lifetime,
                    dataclasses.replace(
                        attachment.AttachEvent.with_data(attachment.AttachData(actor=instance)),
                        source=hsm.id(instance),
                    ),
                )
            except BaseException:
                await routine.stop(lifetime)
                raise
            _ACTIVE.add(1)
            return _Running(revision=inventory.revision(row), routine=routine, routine_id=hsm.id(routine))

    @staticmethod
    async def _reconcile_activity(ctx: hsm.Context, instance: "Routines", event: hsm.Event[typing.Any]) -> None:
        del event
        with span.operation(
            "bot.behavior.routine.reconcile",
            scope=_SCOPE,
            component="cognition.routines",
            stage="reconcile",
        ) as active:
            lifetime = instance.context()
            while instance._stopping:
                await Routines._stop_routine(lifetime, instance._stopping.pop())
            try:
                output = instance._memory.execute(
                    memory.InputData(
                        statements=memory.compile_statements(
                            *behavior_storage.select_active_behaviors_clauses(lifetime=LIFETIME_PERSISTENT)
                        )
                    )
                )
                desired = {
                    row.name: row
                    for row in behavior_storage.instances_from_behavior_results(
                        tuple(item.as_mapping() for item in output.results[0].rows),
                        tuple(item.as_mapping() for item in output.results[1].rows),
                    )
                }
            except Exception as error:
                _ = hsm.dispatch(
                    ctx,
                    instance,
                    _ReconcileFailedEvent.with_data(
                        ability.FailureData(message=f"Routine inventory load failed: {error}")
                    ),
                )
                return
            stale = tuple(
                name
                for name, running in instance._running.items()
                if name not in desired or inventory.revision(desired[name]) != running.revision
            )
            for name in stale:
                running = instance._running.pop(name)
                _ACTIVE.add(-1)
                await Routines._stop_routine(lifetime, running.routine)
            failed = 0
            for name, row in desired.items():
                if name in instance._running:
                    continue
                try:
                    instance._running[name] = await Routines._start_routine(instance, row)
                except Exception:
                    failed += 1
                    _LOG.exception("persistent routine %s failed to start; it stays stopped until changed", name)
            active.set_attribute("bot.behavior.routine.stopped.count", len(stale))
            active.set_attribute("bot.behavior.routine.start_failed.count", failed)
            active.set_attribute("bot.behavior.routine.running.count", len(instance._running))
            _ = hsm.dispatch(ctx, instance, _ReconciledEvent.with_data(_ReconciledData(running=len(instance._running))))

    @staticmethod
    def _log_reconcile_failure(ctx: hsm.Context, instance: "Routines", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        failure = event.data
        assert isinstance(failure, ability.FailureData)
        _LOG.error("persistent routines kept their running set: %s", failure.message)

    @staticmethod
    def _is_running_tick(ctx: hsm.Context, instance: "Routines", event: hsm.Event[typing.Any]) -> bool:
        del ctx
        tick = event.data
        if not isinstance(tick, behavior.TickData):
            return False
        running = instance._running.get(tick.name)
        return running is not None and event.source == running.routine_id

    @staticmethod
    def _is_stale_tick(ctx: hsm.Context, instance: "Routines", event: hsm.Event[typing.Any]) -> bool:
        return isinstance(event.data, behavior.TickData) and not Routines._is_running_tick(ctx, instance, event)

    @staticmethod
    def _forward_tick(ctx: hsm.Context, instance: "Routines", event: hsm.Event[typing.Any]) -> None:
        tick = event.data
        assert isinstance(tick, behavior.TickData)
        _TICKS.add(
            1, attributes={"bot.behavior.routine.trigger": tick.trigger, "bot.behavior.routine.outcome": "forwarded"}
        )
        # The owner (Cognition) gets the tick as this ability's output terminal.
        _ = hsm.dispatch(
            ctx,
            instance,
            ability.TerminalOutputEvent.with_data(
                dataclasses.replace(
                    instance.output_event.with_data(tick), id=event.id, target="", metadata=dict(event.metadata)
                )
            ),
        )

    @staticmethod
    def _drop_stale_tick(ctx: hsm.Context, instance: "Routines", event: hsm.Event[typing.Any]) -> None:
        del ctx, instance
        tick = event.data
        assert isinstance(tick, behavior.TickData)
        _TICKS.add(
            1, attributes={"bot.behavior.routine.trigger": tick.trigger, "bot.behavior.routine.outcome": "stale"}
        )

    @staticmethod
    def _detach_routines(ctx: hsm.Context, instance: "Routines", event: hsm.Event[typing.Any]) -> None:
        # Shared exit fires on every exit; only detaching Routines detaches its routines.
        if not isinstance(event.data, attachment.DetachData):
            return
        for running in instance._running.values():
            # Detaching ends the routine's attached lifecycle, cancelling its schedule timers now;
            # the machine itself stops on the next reconcile or when Routines stops.
            _ = running.routine.detach(
                ctx,
                dataclasses.replace(
                    attachment.DetachEvent.with_data(attachment.DetachData(actor=instance, reply_to=instance)),
                    source=hsm.id(instance),
                    target=running.routine_id,
                ),
            )
            _ACTIVE.add(-1)
            instance._stopping.append(running.routine)
        instance._running.clear()

    submodel: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "Routines",
        hsm.initial(hsm.target("/Routines/active")),
        hsm.state(
            "active",
            hsm.initial(hsm.target("/Routines/active/reconciling")),
            hsm.exit(_detach_routines),
            hsm.state(
                "reconciling",
                hsm.activity(_reconcile_activity),
                # Ticks and further reconcile requests wait for the running set to settle.
                hsm.defer(ReconcileEvent, behavior.TickEvent),
                hsm.transition(
                    hsm.on(_ReconciledEvent),
                    hsm.target("/Routines/active/running"),
                ),
                hsm.transition(
                    hsm.on(_ReconcileFailedEvent),
                    hsm.effect(_log_reconcile_failure),
                    hsm.target("/Routines/active/running"),
                ),
            ),
            hsm.state(
                "running",
                hsm.transition(
                    hsm.on(ReconcileEvent),
                    hsm.target("/Routines/active/reconciling"),
                ),
                hsm.transition(
                    hsm.on(behavior.TickEvent),
                    hsm.guard(_is_running_tick),
                    hsm.effect(_forward_tick),
                ),
                hsm.transition(
                    hsm.on(behavior.TickEvent),
                    hsm.guard(_is_stale_tick),
                    hsm.effect(_drop_stale_tick),
                ),
            ),
        ),
    )

    def __init__(
        self,
        *,
        memory: memory.Memory,
        clock: schedule.WallClock,
        timers: hsm.Clock | None = None,
    ) -> None:
        """Construct Routines over an injected behavior inventory and clocks.

        ``memory`` holds the behavior inventory; ``clock`` is the timezone-aware wall clock every
        routine's schedules read; ``timers`` is the HSM timer clock routines wait on (``None``
        uses the HSM default, real time).
        """

        super().__init__()
        self._memory = memory
        self._clock = clock
        self._timers = timers
        self._running = {}
        self._stopping = []

    @typing.override
    async def stop(self, ctx: hsm.Context) -> None:
        """Stop every routine this ability started, then the ability itself."""

        routines = [running.routine for running in self._running.values()]
        if routines:
            _ACTIVE.add(-len(routines))
        routines.extend(self._stopping)
        self._running.clear()
        self._stopping.clear()
        for routine in routines:
            await Routines._stop_routine(ctx, routine)
        await super().stop(ctx)


__all__ = ["ReconcileData", "ReconcileEvent", "Routines"]
