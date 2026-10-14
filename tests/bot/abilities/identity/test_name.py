"""Storing the name a bot adopted, on the memory schema every other durable fact uses."""

from __future__ import annotations

import asyncio
import datetime
import sqlite3
import typing

import hsm
import mosfet
from sqlalchemy.engine import Engine

from mosfet import abilities, environment
from mosfet.abilities import identity
from mosfet.abilities import memory
from mosfet.protocols import attachment

from tests.bot.abilities.memory.memory_fixtures import execute_on


def open_store() -> tuple[Engine, sqlite3.Connection]:
    """A migrated memory database: name rows are plain relational rows, read and written as SQL."""

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return memory.store.open_sqlite_engine(connection=connection), connection


def test_a_store_with_no_name_row_reads_as_nameless() -> None:
    """Having no name is a normal state of memory, not a missing row to be repaired."""

    store, connection = open_store()
    try:
        recalled = execute_on(store, identity.name_select_input())
    finally:
        connection.close()

    assert identity.names_from_output(recalled) == ()
    assert identity.adopted_name(recalled) is None


def test_an_adopted_name_is_a_fact_about_the_bot_itself() -> None:
    """A name is what the bot *is*, so it is stored as a fact whose subject is the bot.

    It is deliberately not a standing directive: a directive is something to do and stays
    outstanding until released, while a name is never completed or acted on.
    """

    store, connection = open_store()
    try:
        _ = execute_on(store, identity.name_insert_input(identity.Name(text="Bob")))
        recalled = execute_on(store, identity.name_select_input())
        rows = [row.as_mapping() for row in recalled.results[0].rows]
    finally:
        connection.close()

    assert identity.adopted_name(recalled) == identity.Name(text="Bob")
    assert len(rows) == 1
    row = rows[0]
    assert row["query_tags"] == identity.NAME_QUERY
    assert row["subject_ref"] == identity.NAME_SUBJECT
    assert row["kind"] == "fact"
    assert row["retention"] == "retain"
    assert row["sensitivity"] == "standard"
    assert row["scope"] == "long_term"
    assert row["content_format"] == "application/json"


def test_a_name_is_always_a_fact() -> None:
    """A name is a fact about the bot, never an instruction; the schema admits nothing else."""

    schema = identity.Name.model_json_schema()

    assert identity.Name(text="Bob").kind == "fact"
    assert schema["properties"]["kind"]["const"] == "fact"
    assert "$defs" not in schema


def test_a_name_row_written_before_the_kind_was_narrowed_still_decodes() -> None:
    """Rows already on disk hold ``kind`` ``fact`` in both the column and the JSON content."""

    store, connection = open_store()
    try:
        _ = execute_on(store, identity.name_insert_input(identity.Name(text="Bob")))
        recalled = execute_on(store, identity.name_select_input())
    finally:
        connection.close()

    row = recalled.results[0].rows[0].as_mapping()
    assert row["kind"] == "fact"
    assert identity.Name.model_validate_json('{"text": "Bob", "kind": "fact"}') == identity.Name(text="Bob")
    assert identity.adopted_name(recalled) == identity.Name(text="Bob")


def test_the_most_recently_adopted_name_is_the_one_the_bot_answers_to() -> None:
    """Adoption is append-only: a renamed bot still remembers having been called something else.

    ``created_at`` is written explicitly precisely so two adoptions made in the same second still
    have an order — the column default has one-second resolution.
    """

    first = datetime.datetime(2026, 7, 26, 12, 0, 0, 100_000, tzinfo=datetime.UTC)
    second = datetime.datetime(2026, 7, 26, 12, 0, 0, 200_000, tzinfo=datetime.UTC)

    store, connection = open_store()
    try:
        _ = execute_on(store, identity.name_insert_input(identity.Name(text="Bob"), created_at=first))
        _ = execute_on(store, identity.name_insert_input(identity.Name(text="Ada"), created_at=second))
        recalled = execute_on(store, identity.name_select_input())
    finally:
        connection.close()

    assert [name.text for name in identity.names_from_output(recalled)] == ["Bob", "Ada"]
    adopted = identity.adopted_name(recalled)
    assert adopted is not None and adopted.text == "Ada"


def test_a_name_is_never_remembered_as_an_episode() -> None:
    """A name row and remembered episodes share the database and are never read as each other.

    A bot that recalled its own name as something that happened to it would surface it as an
    association on every turn.
    """

    async def run() -> tuple[tuple[memory.Episode, ...], identity.Name | None]:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        engine = memory.store.open_sqlite_engine(connection=connection)
        _ = execute_on(engine, identity.name_insert_input(identity.Name(text="Bob")))
        store = memory.Memory(engine=engine)
        ctx = environment.Environment()
        owner = _Owner()
        _ = await mosfet.started(ctx, owner, _Owner.model)
        _ = await store.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))
        reply = await abilities.run_terminal_operation(
            ctx,
            child=store,
            request=memory.RecentEvent.with_data_and_id(memory.RecentData(limit=8), "recent"),
            terminals=(memory.RecentOutputEvent, memory.FailedEvent),
            timeout=datetime.timedelta(seconds=3),
        )
        assert isinstance(reply.data, memory.EpisodesData), reply.data
        named = identity.adopted_name(execute_on(engine, identity.name_select_input()))
        await hsm.stop(store, hsm.Context())
        connection.close()
        return reply.data.episodes, named

    episodes, named = asyncio.run(run())

    assert episodes == ()
    assert named == identity.Name(text="Bob")


class _Owner(hsm.Instance):
    """Attaches the memory the test reads from; records nothing."""

    model: typing.ClassVar[hsm.Model] = mosfet.define(
        "NameTestMemoryOwner", hsm.initial(hsm.target("ready")), hsm.state("ready")
    )


def test_a_bot_renamed_more_often_than_the_recall_window_answers_to_its_latest_name() -> None:
    """Recall reaches the newest adoptions, so a long rename history never hides the current name."""

    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    store, connection = open_store()
    try:
        for index in range(60):
            _ = execute_on(
                store,
                identity.name_insert_input(
                    identity.Name(text=f"Name {index:02d}"),
                    created_at=base + datetime.timedelta(minutes=index),
                ),
            )
        recalled = execute_on(store, identity.name_select_input(limit=50))
    finally:
        connection.close()

    assert [name.text for name in identity.names_from_output(recalled)] == [
        f"Name {index:02d}" for index in range(10, 60)
    ]
    assert identity.adopted_name(recalled) == identity.Name(text="Name 59")
