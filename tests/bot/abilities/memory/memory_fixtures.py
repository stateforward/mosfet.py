"""Shared fixtures for memory tests: ability trees and register helpers."""

from mosfet import abilities
from mosfet.abilities import memory
from mosfet.protocols import attachment

import asyncio
import collections.abc
import datetime
import typing
import uuid

import hsm
import mosfet
from sqlalchemy.engine import Engine

from tests.hsm_instance_state import ability_terminal_owner, remember_ability_terminal_owner

_TResult = typing.TypeVar("_TResult")


def _record_memory_fixture_terminal_mirror_event(
    ctx: hsm.Context,
    instance: "MemoryFixtureAbilityTerminalMirror",
    event: hsm.Event[typing.Any],
) -> None:
    del ctx
    instance.record(event)


class MemoryFixtureAbilityTerminalMirror(hsm.Instance):
    model: typing.ClassVar[hsm.Model | None] = mosfet.define(
        "MemoryFixtureAbilityTerminalMirror",
        hsm.initial(hsm.target("/MemoryFixtureAbilityTerminalMirror/recording")),
        hsm.state(
            "recording",
            hsm.transition(
                hsm.on(hsm.AnyEvent),
                hsm.effect(_record_memory_fixture_terminal_mirror_event),
            ),
        ),
    )
    ability: abilities.Ability[typing.Any, typing.Any]
    results: dict[str, asyncio.Future[object]]

    def __init__(self, ability: abilities.Ability[typing.Any, typing.Any]) -> None:
        super().__init__()
        self.ability = ability
        self.results = {}

    def result_for(self, operation_id: str) -> asyncio.Future[object]:
        result = self.results.get(operation_id)
        if result is None:
            result = asyncio.get_running_loop().create_future()
            self.results[operation_id] = result
        return result

    def record(self, event: hsm.Event[typing.Any]) -> None:
        operation_id = event.id if event.id else None
        result = self.results.get(operation_id) if operation_id is not None else None
        if event.name == self.ability.output_event.name:
            values = getattr(self.ability, "outputs", None)
            if isinstance(values, list):
                values.append(event.data)
            if result is not None and not result.done():
                result.set_result(event.data)
        if event.name == self.ability.failed_event.name:
            values = getattr(self.ability, "failures", None)
            if isinstance(values, list):
                values.append(event.data)
            if result is not None and not result.done():
                failure = event.data
                message = failure.message if isinstance(failure, abilities.FailureData) else str(failure)
                result.set_exception(RuntimeError(message))


async def start_ability_tree(*abilities: abilities.Ability[typing.Any, typing.Any]) -> None:
    ctx = hsm.Context()
    for ability in abilities:
        owner = MemoryFixtureAbilityTerminalMirror(ability)
        assert owner.model is not None
        _ = await mosfet.started(ctx, owner, owner.model)
        _ = await ability.attach(
            ctx,
            attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)),
        )
        remember_ability_terminal_owner(ability, owner)


async def stop_ability_tree(*abilities: abilities.Ability[typing.Any, typing.Any]) -> None:
    for ability in abilities:
        owner = ability_terminal_owner(ability)
        assert owner is not None
        _ = await ability.detach(
            ability.context(),
            attachment.DetachEvent.with_data(attachment.DetachData(actor=owner)),
        )


async def wait_until(condition: collections.abc.Callable[[], bool], *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met before timeout")


def require_model(ability: abilities.Ability[typing.Any, typing.Any] | hsm.Model | None) -> hsm.Model:
    if isinstance(ability, abilities.Ability):
        assert ability.model is not None
        return ability.model
    assert ability is not None
    return ability


async def register_recent(
    ctx: hsm.Context,
    register: memory.StmMemory,
    stimulus_name: str | None = None,
    *,
    limit: int,
) -> tuple[memory.ObservedEvent, ...]:
    """Read the short-term register the way a peer does: a RecentEvent and its correlated reply."""

    terminal = await abilities.run_terminal_operation(
        ctx,
        child=register,
        request=memory.stm.RecentEvent.with_data_and_id(
            memory.stm.RecentData(stimulus_name=stimulus_name, limit=limit), uuid.uuid4().hex
        ),
        terminals=(memory.stm.RecentOutputEvent, register.failed_event),
        timeout=datetime.timedelta(seconds=5),
    )
    assert isinstance(terminal.data, memory.stm.RecentOutputData), terminal.data
    return terminal.data.entries


async def register_record(ctx: hsm.Context, register: memory.StmMemory, event: hsm.Event[typing.Any]) -> None:
    """Record one stimulus by message; the read that follows is answered only after it lands."""

    _ = await hsm.dispatch(ctx, register, memory.stm.RecordEvent.with_data(memory.stm.event_record(event)))
    _ = await register_recent(ctx, register, limit=1)


def execute_on(engine: Engine, data: memory.InputData) -> memory.OutputData:
    """Run one statement transaction straight on a test database engine.

    For seeding and inspecting a test database only. Production code never calls into a Memory:
    it sends ``memory.transact``. A test that shares the Memory's engine reads and writes rows here.
    """

    results: list[memory.StatementResult] = []
    with engine.begin() as connection:
        for statement in data.statements:
            cursor = connection.exec_driver_sql(statement.sql, statement.parameters)
            if cursor.returns_rows:
                columns = tuple(cursor.keys())
                rows = tuple(
                    memory.Row(
                        columns=columns, values=tuple(typing.cast(tuple[memory.ParameterValue, ...], tuple(row)))
                    )
                    for row in cursor.fetchall()
                )
                results.append(memory.StatementResult(rowcount=len(rows), rows=rows))
            else:
                results.append(memory.StatementResult(rowcount=max(cursor.rowcount, 0)))
    return memory.OutputData(results=tuple(results))
