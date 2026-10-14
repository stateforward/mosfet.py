"""SQL-transaction MemoryStore tests (ability boundary only)."""

from mosfet.abilities import memory

import asyncio
import datetime
import pathlib
import typing

import hsm
import mosfet
import pytest
from sqlalchemy.engine import Engine

from tests.bot.abilities.memory.memory_fixtures import execute_on
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test


def _insert_content(
    *,
    content: str,
    scope: str = "memory",
    context_ref: str | None = None,
    subject_ref: str | None = None,
    kind: str | None = "task",
    query_tags: str | None = None,
    content_format: str = "text/plain",
    memory_id: str | None = None,
) -> memory.Statement:
    import uuid
    from sqlalchemy import insert

    table = memory.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=scope,
        context_ref=context_ref,
        subject_ref=subject_ref,
        kind=kind,
        sensitivity="standard",
        retention="retain",
        content=content,
        content_format=content_format,
        query_tags=query_tags,
    )
    return memory.compile_statement(clause)


def _select_by_query_tags(*, query_tags: str, context_ref: str | None = None, limit: int = 50) -> memory.Statement:
    from sqlalchemy import or_, select

    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == query_tags)
    if context_ref is not None:
        clause = clause.where(or_(table.c.context_ref.is_(None), table.c.context_ref == context_ref))
    clause = clause.order_by(table.c.created_at).limit(limit)
    return memory.compile_statement(clause)


def test_memory_store_ability_apply_is_transaction() -> None:
    async def run() -> memory.OutputData:
        store = memory.MemoryStore()
        await start_abilities_for_test(shared_hsm_context(), store)
        return await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    _insert_content(content="via-ability", query_tags="t"),
                    memory.Statement(
                        sql=f"SELECT content FROM {memory.MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("t",),
                    ),
                )
            ),
        )

    output = asyncio.run(run())
    assert output.contents(statement_index=1) == ("via-ability",)


def test_memory_store_ability_rolls_back_on_failure() -> None:
    async def run() -> tuple[str, ...]:
        store = memory.Memory()
        await start_abilities_for_test(shared_hsm_context(), store)
        with pytest.raises(RuntimeError):
            _ = await dispatch_ability_for_test(
                store,
                None,
                memory.InputData(
                    statements=(
                        _insert_content(content="should-not-remain", query_tags="x"),
                        memory.Statement(sql="INSERT INTO not_a_table (a) VALUES (1)", parameters=()),
                    )
                ),
            )
        check = await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    memory.Statement(
                        sql=f"SELECT content FROM {memory.MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("x",),
                    ),
                )
            ),
        )
        return check.contents()

    assert asyncio.run(run()) == ()


def test_input_requires_at_least_one_statement() -> None:
    with pytest.raises(Exception):
        _ = memory.InputData(statements=())


def test_memory_store_accepts_injected_engine() -> None:
    """The ability runs on a caller-provided engine; the SQLite default stays in the factory."""

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()

    async def run() -> memory.OutputData:
        injected = memory.MemoryStore(engine=engine)
        await start_abilities_for_test(shared_hsm_context(), injected)
        return await dispatch_ability_for_test(
            injected,
            None,
            memory.InputData(statements=(_insert_content(content="injected-engine", query_tags="task"),)),
        )

    output = asyncio.run(run())
    assert len(output.results) == 1
    selected = execute_on(engine, memory.InputData(statements=(_select_by_query_tags(query_tags="task"),)))
    assert "injected-engine" in selected.contents()
    with pytest.raises(ValueError, match="either engine or database/connection"):
        _ = memory.MemoryStore(engine=engine, database="/tmp/memory-store-injected.db")


def _schema_revision_state(engine: Engine) -> tuple[str | None, str | None, list[object]]:
    """Current DB revision, head script revision, and autogenerate diffs vs ``memory.metadata``."""

    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    from mosfet.abilities.memory import store as store_module

    head = ScriptDirectory.from_config(store_module.migration_config()).get_current_head()
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        diffs = typing.cast("list[object]", compare_metadata(context, memory.metadata))
        return context.get_current_revision(), head, diffs


def test_open_sqlite_engine_upgrades_fresh_database_to_head_without_drift() -> None:
    """Bring-up runs every revision; the migrated schema matches ``memory.metadata`` exactly."""

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()
    current, head, diffs = _schema_revision_state(engine)
    engine.dispose()
    assert head is not None
    assert current == head
    assert diffs == []


def test_open_sqlite_engine_rejects_an_unversioned_database_that_holds_memory_tables(tmp_path: pathlib.Path) -> None:
    """Memory tables without a version table were not made by these migrations and are not guessed at."""

    import contextlib
    import sqlite3

    from mosfet.abilities.memory import store as store_module

    path = tmp_path / "unversioned.db"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)")
    with pytest.raises(store_module.MigrationError, match="Unversioned memory database"):
        _ = store_module.open_sqlite_engine(database=str(path))


def test_the_one_baseline_revision_creates_and_drops_every_memory_table() -> None:
    """Revision 0001 is the whole schema (relational, behavior, register, episodic graph); no data moves."""

    from alembic import command
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine, inspect

    from mosfet.abilities.memory import store as store_module

    config = store_module.migration_config()
    assert [script.revision for script in ScriptDirectory.from_config(config).walk_revisions()] == ["0001"]
    engine = create_engine("sqlite+pysqlite://")
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
        created = set(inspect(connection).get_table_names())
        command.downgrade(config, "base")
        remaining = set(inspect(connection).get_table_names())
    engine.dispose()
    assert set(memory.metadata.tables) <= created
    assert remaining == {store_module.VERSION_TABLE}
    lifetime = {column.name for column in memory.metadata.tables["bot_behavior"].columns}
    assert "lifetime" in lifetime


def test_recent_select_returns_same_instant_rows_in_insertion_order() -> None:
    """Recall reaches the latest rows; within the window they read in the order they happened.

    ``created_at`` decides recency even when ``memory_id`` sorts the other way, and rows that share
    one ``created_at`` read in insertion order, never in ``memory_id`` (uuid) order, both when the
    window is cut and when it is re-read oldest first.
    """

    import uuid

    from sqlalchemy import insert

    table = memory.memory_table
    same_instant = "2026-01-01 00:00:03.000000"
    rows = (
        ("oldest", "2026-01-01 00:00:01.000000", uuid.UUID(int=0xF).hex),
        ("older", "2026-01-01 00:00:02.000000", uuid.UUID(int=0xE).hex),
        ("first", same_instant, uuid.UUID(int=0xC).hex),
        ("second", same_instant, uuid.UUID(int=0xA).hex),
        ("third", same_instant, uuid.UUID(int=0xB).hex),
        ("other-tag", "2026-01-01 00:00:04.000000", uuid.UUID(int=0x1).hex),
    )

    async def run() -> tuple[memory.OutputData, memory.OutputData]:
        store = memory.Memory()
        await start_abilities_for_test(shared_hsm_context(), store)
        for content, created_at, memory_id in rows:
            _ = await dispatch_ability_for_test(
                store,
                None,
                memory.InputData(
                    statements=memory.compile_statements(
                        insert(table).values(
                            memory_id=memory_id,
                            content=content,
                            query_tags="other" if content == "other-tag" else "t",
                            created_at=created_at,
                        )
                    )
                ),
            )

        async def recall(limit: int) -> memory.OutputData:
            return await dispatch_ability_for_test(
                store,
                None,
                memory.InputData(
                    statements=memory.compile_statements(memory.recent_select(table.c.query_tags == "t", limit=limit))
                ),
            )

        return await recall(4), await recall(2)

    window, cut = asyncio.run(run())

    assert window.contents() == ("older", "first", "second", "third")
    assert cut.contents() == ("second", "third")
    assert set(window.results[0].rows[0].as_mapping()) == {column.name for column in table.c}


def test_transact_reaches_the_store_by_message_and_returns_its_committed_reply() -> None:
    async def run() -> memory.OutputData:
        store = memory.Memory()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        return await memory.transact(
            ctx,
            store,
            memory.InputData(
                statements=(
                    _insert_content(content="by-message", query_tags="m"),
                    _select_by_query_tags(query_tags="m"),
                )
            ),
        )

    assert asyncio.run(run()).contents(statement_index=1) == ("by-message",)


def test_transact_raises_request_error_when_the_store_rolls_back() -> None:
    async def run() -> tuple[str, ...]:
        store = memory.Memory()
        ctx = shared_hsm_context()
        await start_abilities_for_test(ctx, store)
        with pytest.raises(memory.RequestError, match="not_a_table"):
            _ = await memory.transact(
                ctx,
                store,
                memory.InputData(
                    statements=(
                        _insert_content(content="rolled-back", query_tags="r"),
                        memory.Statement(sql="INSERT INTO not_a_table (a) VALUES (1)", parameters=()),
                    )
                ),
            )
        check = await memory.transact(ctx, store, memory.InputData(statements=(_select_by_query_tags(query_tags="r"),)))
        return check.contents()

    assert asyncio.run(run()) == ()


def test_transact_raises_request_error_for_a_store_that_is_not_started() -> None:
    async def run() -> None:
        with pytest.raises(memory.RequestError, match="cannot be reached"):
            _ = await memory.transact(
                shared_hsm_context(),
                memory.Memory(),
                memory.InputData(statements=(_select_by_query_tags(query_tags="none"),)),
            )

    asyncio.run(run())


def test_transact_raises_request_error_when_the_store_never_answers() -> None:
    async def run() -> None:
        store = memory.MemoryStore()
        ctx = shared_hsm_context()
        # Started but never attached: the request reaches a machine that cannot serve it yet.
        _ = await mosfet.started(ctx, store, typing.cast(hsm.Model, store.model))
        with pytest.raises(memory.RequestError, match="did not answer"):
            _ = await memory.transact(
                ctx,
                store,
                memory.InputData(statements=(_select_by_query_tags(query_tags="none"),)),
                timeout=datetime.timedelta(milliseconds=50),
            )

    asyncio.run(run())


def test_in_memory_engine_is_shared_across_threads() -> None:
    """A row written on a worker thread is visible from the caller thread: one ``:memory:`` database."""

    import concurrent.futures

    from sqlalchemy import insert, select

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()
    table = memory.memory_table

    def write() -> None:
        with engine.begin() as connection:
            _ = connection.execute(insert(table).values(memory_id="threaded", content="written on a worker"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(write).result()
    with engine.connect() as connection:
        contents = connection.execute(select(table.c.content)).scalars().all()
    engine.dispose()
    assert contents == ["written on a worker"]


def test_open_sqlite_engine_runs_the_connect_hook_before_migrating() -> None:
    """``on_connect`` sees every new DBAPI connection before the schema is brought up."""

    import sqlite3

    from mosfet.abilities.memory import store as store_module

    seen: list[list[str]] = []

    def on_connect(connection: sqlite3.Connection) -> None:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        seen.append(tables)

    engine = store_module.open_sqlite_engine(on_connect=on_connect)
    engine.dispose()
    assert seen == [[]]
