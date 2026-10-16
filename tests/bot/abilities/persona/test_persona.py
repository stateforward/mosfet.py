"""Persona: owner-given traits, and a description of itself the bot writes and keeps.

Every test pins a *capability* and never a choice. A revision happens only because the test
dispatched the revise event itself, standing in for reflection; nothing asserts that a bot ever
revises. A bot that keeps its first (empty) description for life passes this whole file.
"""

from __future__ import annotations

import asyncio
import collections
import collections.abc
import datetime
import pathlib
import sqlite3
import typing
from typing import override

import hsm
import pydantic
import pytest

from mosfet import event
from mosfet.abilities import memory
from mosfet.abilities import persona
from mosfet.abilities import processing

from tests import metric_reader
from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test

_KNOWING = "/knowing"
_TRAITS = persona.Traits(traits=("curious", "patient with children"))


class WriteRefusingMemory(memory.Memory):
    """Memory that can be read but not written, as a full or read-only store would be."""

    @override
    def _execute(self, data: memory.InputData) -> memory.OutputData:
        if any(statement.sql.lstrip().upper().startswith("INSERT") for statement in data.statements):
            raise RuntimeError("memory is read-only")
        return super()._execute(data)


class RecordingPersona(persona.Persona):
    """Persona that mirrors its terminals for assertions (see tests.bot.abilities.support)."""

    outputs: list[object]
    failures: list[object]

    def __init__(self, *, traits: persona.Traits, memory: memory.Memory | None = None) -> None:
        super().__init__(traits=traits, memory=memory)
        self.outputs = []
        self.failures = []


def open_store(store_type: type[memory.Memory] = memory.Memory) -> tuple[memory.Memory, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return store_type(engine=memory.store.open_sqlite_engine(connection=connection)), connection


async def wait_until(condition: collections.abc.Callable[[], bool]) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.001)


async def settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


async def start_persona(
    ctx: hsm.Context,
    *,
    traits: persona.Traits = _TRAITS,
    store: memory.Memory | None = None,
    attach_store: bool = True,
    timeout: datetime.timedelta | None = None,
) -> RecordingPersona:
    if store is not None and attach_store and not store.state():
        # Stands in for the owner that attaches the bot's memory; persona is only its peer.
        await start_abilities_for_test(ctx, store)
    ability = RecordingPersona(traits=traits, memory=store)
    await start_abilities_for_test(ctx, ability, timeout=timeout)
    await wait_until(lambda: ability.state().endswith(_KNOWING) and bool(ability.outputs))
    return ability


async def revise(ability: RecordingPersona, ctx: hsm.Context, text: str, *, reason: str = "I noticed it.") -> None:
    """Stand in for reflection selecting a revision. The bot is never made to do this."""

    told = len(ability.outputs) + len(ability.failures)
    _ = await hsm.dispatch(ctx, ability, persona.ReviseEvent.with_data(persona.ReviseData(text=text, reason=reason)))
    await wait_until(lambda: len(ability.outputs) + len(ability.failures) > told)
    await wait_until(lambda: ability.state().endswith(_KNOWING))


def outputs(ability: RecordingPersona) -> list[persona.OutputData]:
    told: list[persona.OutputData] = []
    for output in ability.outputs:
        assert isinstance(output, persona.OutputData)
        told.append(output)
    return told


def test_a_bot_wakes_knowing_its_traits_and_with_no_description() -> None:
    """Traits come from the owner; a description exists only once the bot writes one."""

    async def run() -> tuple[list[persona.OutputData], list[object]]:
        store, connection = open_store()
        try:
            ability = await start_persona(shared_hsm_context(), store=store)
            await settle()
            return outputs(ability), list(ability.failures)
        finally:
            connection.close()

    told, failures = asyncio.run(run())

    assert told == [persona.OutputData(traits=_TRAITS.traits, description=None)]
    assert failures == []


def test_a_revision_is_written_down_with_its_traits_and_told() -> None:
    async def run() -> tuple[list[persona.OutputData], persona.Description | None]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_persona(ctx, store=store)
            await revise(ability, ctx, "I like explaining things slowly.", reason="A long talk with a child.")
            recalled = await memory.transact(ctx, store, persona.description_select_input())
            return outputs(ability), persona.latest_description(recalled)
        finally:
            connection.close()

    told, stored = asyncio.run(run())

    expected = persona.Description(
        text="I like explaining things slowly.", reason="A long talk with a child.", traits=_TRAITS.traits
    )
    assert stored == expected
    assert told[-1] == persona.OutputData(traits=_TRAITS.traits, description=expected)


def test_a_bot_wakes_remembering_the_description_it_wrote() -> None:
    """The second ability is a different instance over the same memory and was told nothing."""

    async def run() -> list[persona.OutputData]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            first = await start_persona(ctx, store=store)
            await revise(first, ctx, "I am patient.")
            await hsm.stop(first, ctx)
            second = await start_persona(ctx, store=store)
            return outputs(second)
        finally:
            connection.close()

    told = asyncio.run(run())

    assert len(told) == 1
    description = told[0].description
    assert description is not None
    assert description.text == "I am patient."


def test_the_persona_is_kept_when_the_owner_changes_the_traits() -> None:
    """New traits are the bot's nature from now on; what it wrote under the old ones still stands."""

    async def run() -> list[persona.OutputData]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            first = await start_persona(ctx, store=store)
            await revise(first, ctx, "I am patient.")
            await hsm.stop(first, ctx)
            second = await start_persona(ctx, store=store, traits=persona.Traits(traits=("bold",)))
            return outputs(second)
        finally:
            connection.close()

    told = asyncio.run(run())

    assert told[0].traits == ("bold",)
    description = told[0].description
    assert description is not None
    assert description.text == "I am patient."
    assert description.traits == _TRAITS.traits


def test_a_bot_with_no_memory_keeps_its_description_only_while_it_runs() -> None:
    async def run() -> tuple[list[persona.OutputData], list[persona.OutputData]]:
        ctx = shared_hsm_context()
        first = await start_persona(ctx)
        await revise(first, ctx, "I am patient.")
        second = await start_persona(ctx)
        return outputs(first), outputs(second)

    first_told, second_told = asyncio.run(run())

    first_description = first_told[-1].description
    assert first_description is not None
    assert first_description.text == "I am patient."
    assert second_told == [persona.OutputData(traits=_TRAITS.traits, description=None)]


def test_a_description_that_cannot_be_written_down_is_not_taken() -> None:
    """Fail closed and say so: the bot keeps the description it could remember."""

    async def run() -> tuple[list[persona.OutputData], list[object], str]:
        store, connection = open_store(WriteRefusingMemory)
        try:
            ctx = shared_hsm_context()
            ability = await start_persona(ctx, store=store)
            await revise(ability, ctx, "I am patient.")
            return outputs(ability), list(ability.failures), ability.state()
        finally:
            connection.close()

    told, failures, state = asyncio.run(run())

    assert told == [persona.OutputData(traits=_TRAITS.traits, description=None)]
    assert len(failures) == 1
    assert state.endswith(_KNOWING)


def test_revisions_that_arrive_while_one_is_being_written_wait_their_turn() -> None:
    async def run() -> tuple[list[persona.OutputData], persona.Description | None]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = await start_persona(ctx, store=store)
            for text in ("First.", "Second."):
                _ = await hsm.dispatch(
                    ctx, ability, persona.ReviseEvent.with_data(persona.ReviseData(text=text, reason="Because."))
                )
            await wait_until(lambda: len(ability.outputs) == 3)
            recalled = await memory.transact(ctx, store, persona.description_select_input())
            return outputs(ability), persona.latest_description(recalled)
        finally:
            connection.close()

    told, stored = asyncio.run(run())

    assert [output.description.text if output.description else None for output in told] == [None, "First.", "Second."]
    assert stored is not None
    assert stored.text == "Second."


def test_revising_is_a_model_callable_tool_offered_from_live_topology() -> None:
    async def run() -> tuple[str, ...]:
        store, connection = open_store()
        try:
            ability = await start_persona(shared_hsm_context(), store=store)
            return tuple(offered.name for offered in processing.enabled_call_events(ability))
        finally:
            connection.close()

    offered = asyncio.run(run())

    assert persona.ReviseEvent.kind == event.EventKind
    assert persona.ReviseEvent.name == "bot.ability.persona.revise"
    assert persona.ReviseEvent.name in offered


def test_a_memory_that_never_answers_leaves_the_bot_with_its_traits_once_the_budget_runs_out() -> None:
    async def run() -> tuple[str, list[persona.OutputData], list[object]]:
        store, connection = open_store()
        try:
            ctx = shared_hsm_context()
            ability = RecordingPersona(traits=_TRAITS, memory=store)
            await start_abilities_for_test(ctx, ability, timeout=datetime.timedelta(milliseconds=300))
            await asyncio.sleep(0.15)
            while_waiting = ability.state()
            await wait_until(lambda: ability.state().endswith(_KNOWING) and bool(ability.outputs))
            return while_waiting, outputs(ability), list(ability.failures)
        finally:
            connection.close()

    while_waiting, told, failures = asyncio.run(run())

    assert "/recalling/" in while_waiting, "a memory that is not answering yet is asked again, not read as empty"
    assert len(failures) == 1
    assert told == [persona.OutputData(traits=_TRAITS.traits, description=None)]


def test_revise_names_its_bounds_and_offers_no_example_to_copy() -> None:
    schema = persona.ReviseData.model_json_schema()
    description = typing.cast(str, schema["description"])

    assert set(typing.cast(dict[str, object], schema["properties"])) == {"text", "reason"}
    assert "examples" not in schema["properties"]["text"]
    assert "examples" not in schema
    assert "traits" in description
    assert "your voice was given to you when you were made and does not change; how you deliver a reply does" in (
        description.lower()
    )


@pytest.mark.parametrize(
    ("field", "value"), [("text", ""), ("text", "  "), ("text", "x" * 2001), ("reason", ""), ("reason", "x" * 501)]
)
def test_a_revision_is_bounded(field: str, value: str) -> None:
    payload = {"text": "I am.", "reason": "Because."}
    payload[field] = value

    with pytest.raises(pydantic.ValidationError):
        _ = persona.ReviseData.model_validate(payload)


def test_the_persona_renders_owner_traits_and_a_self_written_description() -> None:
    written = persona.Description(text="I like <slow> & careful talk.", reason="Because.", traits=("curious",))

    with_description = persona.OutputData(traits=("curious", "kind"), description=written)
    without = persona.OutputData(traits=("curious",), description=None)

    assert with_description.__model_repr__() == (
        '<traits given_by="owner"><trait>curious</trait><trait>kind</trait></traits>'
        '<description written_by="you">I like &lt;slow&gt; &amp; careful talk.</description>'
    )
    assert without.__model_repr__() == (
        '<traits given_by="owner"><trait>curious</trait></traits><description written_by="you"/>'
    )


def _counts(reader: metric_reader.Reader, name: str) -> collections.Counter[str]:
    return collections.Counter(
        {str(point.attributes.get("bot.outcome")): int(point.value) for point in reader.points(name)}
    )


def test_recall_and_revision_are_counted_by_outcome(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run(reader: metric_reader.Reader) -> tuple[collections.Counter[str], collections.Counter[str]]:
        recall_before = _counts(reader, "bot.persona.recall.count")
        revision_before = _counts(reader, "bot.persona.revision.count")
        store, connection = open_store()
        refusing, refusing_connection = open_store(WriteRefusingMemory)
        try:
            ctx = shared_hsm_context()
            ability = await start_persona(ctx, store=store)
            await revise(ability, ctx, "I am patient.")
            again = await start_persona(ctx, store=store)
            await revise(again, ctx, "I am patient and bold.")
            failing = await start_persona(ctx, store=refusing)
            await revise(failing, ctx, "Never written.")
        finally:
            connection.close()
            refusing_connection.close()
        return (
            _counts(reader, "bot.persona.recall.count") - recall_before,
            _counts(reader, "bot.persona.revision.count") - revision_before,
        )

    with metric_reader.recording(tmp_path, monkeypatch) as reader:
        recalls, revisions = asyncio.run(run(reader))

    assert recalls == collections.Counter({"empty": 2, "recalled": 1})
    assert revisions == collections.Counter({"revised": 2, "failed": 1})
