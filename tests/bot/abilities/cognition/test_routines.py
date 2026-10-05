"""Routines: persistent routines rehydrate on attach, reconcile with inventory, and forward ticks."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import pathlib
import time
import typing

import hsm
import mosfet

from mosfet import behavior
from mosfet.abilities import cognition
from mosfet.abilities import memory
from mosfet.behavior import storage as behavior_storage
from mosfet.protocols import attachment

from tests.bot.behavior.support import ManualTimers, routine_source

_EVERY = "hsm.every(seconds = 300)"
_START = datetime.datetime(2026, 10, 5, 12, 0, tzinfo=datetime.UTC)


class _Owner(hsm.Instance):
    """Cognition stand-in: records the ticks Routines hands its owner."""

    ticks: list[hsm.Event[typing.Any]]

    def __init__(self) -> None:
        super().__init__()
        self.ticks = []

    @staticmethod
    def record(ctx: hsm.Context, instance: "_Owner", event: hsm.Event[typing.Any]) -> None:
        del ctx
        instance.ticks.append(event)

    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "RoutinesOwner",
        hsm.initial(hsm.target("/RoutinesOwner/idle")),
        hsm.state("idle", hsm.transition(hsm.on(behavior.TickEvent), hsm.effect(record))),
    )


def _install(store: memory.Memory, program: str) -> behavior.Instance:
    checked = behavior.check(program)
    assert checked.ok and checked.value is not None, checked.report.render()
    _ = store.execute(
        memory.InputData(
            statements=memory.compile_statements(*behavior_storage.replace_behavior_clauses(checked.value))
        )
    )
    return checked.value


async def _until(predicate: typing.Callable[[], bool], *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.01)


async def _attach(store: memory.Memory, timers: ManualTimers) -> tuple[cognition.Routines, _Owner, hsm.Context]:
    routines = cognition.Routines(memory=store, clock=lambda: _START, timers=timers)
    owner = _Owner()
    ctx = hsm.Context()
    _ = await mosfet.started(ctx, owner, typing.cast(hsm.Model, owner.model))
    _ = await routines.attach(
        ctx,
        dataclasses.replace(
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
            source=hsm.id(owner),
        ),
    )
    await _until(lambda: routines.state().endswith("/active/running"))
    return routines, owner, ctx


async def _reconcile(routines: cognition.Routines, ctx: hsm.Context) -> None:
    _ = await hsm.dispatch(
        ctx, routines, cognition.routines.ReconcileEvent.with_data(cognition.routines.ReconcileData())
    )
    await _until(lambda: routines.state().endswith("/active/running"))


def _open_waits(timers: ManualTimers) -> list[datetime.timedelta]:
    return [duration for duration, waiter in timers.pending if not waiter.done()]


async def _tick(owner: _Owner, timers: ManualTimers) -> behavior.TickData:
    seen = len(owner.ticks)
    _ = await timers.fire(_START)
    await _until(lambda: len(owner.ticks) > seen)
    tick = owner.ticks[-1].data
    assert isinstance(tick, behavior.TickData)
    return tick


def test_routines_rehydrate_from_inventory_across_a_restart(tmp_path: pathlib.Path) -> None:
    database = str(tmp_path / "bot.db")

    async def lifetime(*, install: bool) -> tuple[behavior.TickData, hsm.Event[typing.Any], str]:
        store = memory.Memory(database=database)
        if install:
            _ = _install(store, routine_source(schedule=_EVERY))
        timers = ManualTimers()
        routines, owner, ctx = await _attach(store, timers)
        try:
            tick = await _tick(owner, timers)
            return tick, owner.ticks[-1], hsm.id(routines)
        finally:
            await routines.stop(ctx)
            await hsm.stop(owner, ctx)

    first, first_event, first_routines = asyncio.run(lifetime(install=True))
    # A new process: nothing but the database survives; the routine comes back on attach.
    second, second_event, second_routines = asyncio.run(lifetime(install=False))

    for tick, event, routines_id in ((first, first_event, first_routines), (second, second_event, second_routines)):
        assert tick.name == "MorningBriefing"
        assert tick.trigger == "every"
        assert tick.output[0]["event"] == "bot.speaking.say"
        # Routines hands its owner the tick as its own output terminal.
        assert event.name == behavior.TickEvent.name
        assert event.source == routines_id


def test_routines_never_run_turn_behaviors_or_drafts(tmp_path: pathlib.Path) -> None:
    from tests.bot.behavior.support import greeting_behavior_source

    async def run() -> list[datetime.timedelta]:
        store = memory.Memory(database=str(tmp_path / "bot.db"))
        _ = _install(store, greeting_behavior_source())
        checked = behavior.check(routine_source(schedule=_EVERY))
        assert checked.value is not None
        draft = behavior_storage.mark_draft(checked.value)
        _ = store.execute(
            memory.InputData(statements=memory.compile_statements(*behavior_storage.replace_behavior_clauses(draft)))
        )
        timers = ManualTimers()
        routines, owner, ctx = await _attach(store, timers)
        try:
            await asyncio.sleep(0.05)
            return _open_waits(timers)
        finally:
            await routines.stop(ctx)
            await hsm.stop(owner, ctx)

    assert asyncio.run(run()) == []


def test_routines_stop_a_broken_routine_on_reconcile() -> None:
    async def run() -> tuple[list[datetime.timedelta], list[datetime.timedelta], int]:
        store = memory.Memory()
        installed = _install(store, routine_source(schedule=_EVERY))
        timers = ManualTimers()
        routines, owner, ctx = await _attach(store, timers)
        try:
            _ = await _tick(owner, timers)
            before = _open_waits(timers)
            _ = store.execute(
                memory.InputData(
                    statements=memory.compile_statements(
                        *behavior_storage.replace_behavior_clauses(
                            behavior_storage.mark_broken(installed, reason="annoying")
                        )
                    )
                )
            )
            await _reconcile(routines, ctx)
            return before, _open_waits(timers), len(owner.ticks)
        finally:
            await routines.stop(ctx)
            await hsm.stop(owner, ctx)

    before, after, ticks = asyncio.run(run())

    assert before == [datetime.timedelta(seconds=300)]
    assert after == []
    assert ticks == 1


def test_routines_restart_a_changed_routine_and_start_a_new_one_on_reconcile() -> None:
    async def run() -> tuple[str, list[str], list[datetime.timedelta]]:
        store = memory.Memory()
        _ = _install(store, routine_source(schedule=_EVERY))
        timers = ManualTimers()
        routines, owner, ctx = await _attach(store, timers)
        try:
            first = (await _tick(owner, timers)).output[0]
            _ = _install(store, routine_source(schedule=_EVERY, text="Rise and shine."))
            _ = _install(store, routine_source(name="EveningWindDown", schedule="hsm.every(seconds = 600)"))
            await _reconcile(routines, ctx)
            waits = sorted(_open_waits(timers))
            texts: list[str] = []
            for _ in waits:
                tick = await _tick(owner, timers)
                texts.append(f"{tick.name}:{typing.cast(dict[str, object], tick.output[0]['data'])['text']}")
            return str(typing.cast(dict[str, object], first["data"])["text"]), sorted(texts), waits
        finally:
            await routines.stop(ctx)
            await hsm.stop(owner, ctx)

    first, texts, waits = asyncio.run(run())

    assert first == "Good morning."
    assert waits == [datetime.timedelta(seconds=300), datetime.timedelta(seconds=600)]
    assert texts == ["EveningWindDown:Good morning.", "MorningBriefing:Rise and shine."]


def test_routines_stop_ticking_when_detached() -> None:
    async def run() -> tuple[list[datetime.timedelta], list[datetime.timedelta]]:
        store = memory.Memory()
        _ = _install(store, routine_source(schedule=_EVERY))
        timers = ManualTimers()
        routines, owner, ctx = await _attach(store, timers)
        try:
            before = _open_waits(timers)
            _ = await routines.detach(
                ctx,
                dataclasses.replace(
                    attachment.DetachEvent.with_data(attachment.DetachData(actor=owner, reply_to=owner)),
                    source=hsm.id(owner),
                    target=hsm.id(routines),
                ),
            )
            await _until(lambda: not _open_waits(timers))
            return before, _open_waits(timers)
        finally:
            await routines.stop(ctx)
            await hsm.stop(owner, ctx)

    before, after = asyncio.run(run())

    assert before == [datetime.timedelta(seconds=300)]
    assert after == []


def test_routines_reconcile_event_is_a_documented_contract() -> None:
    schema = cognition.routines.ReconcileData.model_json_schema()
    assert schema["description"]
    assert schema["examples"] == [{}]
    assert cognition.routines.ReconcileEvent.name == "bot.ability.routines.reconcile"
    assert cognition.Routines.output_event is behavior.TickEvent
