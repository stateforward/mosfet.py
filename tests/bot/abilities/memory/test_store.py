"""SQL-transaction MemoryStore tests (ability boundary only)."""

from mosfet.abilities import memory

import asyncio
import pathlib
import typing

import pytest
from sqlalchemy.engine import Engine

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
    injected = memory.MemoryStore(engine=engine)
    output = injected.execute(
        memory.InputData(statements=(_insert_content(content="injected-engine", query_tags="task"),))
    )
    assert len(output.results) == 1
    selected = injected.execute(memory.InputData(statements=(_select_by_query_tags(query_tags="task"),)))
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


def _create_all_era_database(path: pathlib.Path) -> None:
    """A database as ``metadata.create_all`` left it before migrations: baseline tables, no version table."""

    from alembic import command
    from sqlalchemy import create_engine, text

    from mosfet.abilities.memory import store as store_module

    config = store_module.migration_config()
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, store_module.BASELINE_REVISION)
        _ = connection.execute(text(f"DROP TABLE {store_module.VERSION_TABLE}"))
    engine.dispose()


def test_open_sqlite_engine_upgrades_fresh_database_to_head_without_drift() -> None:
    """Bring-up runs every revision; the migrated schema matches ``memory.metadata`` exactly."""

    from mosfet.abilities.memory import store as store_module

    engine = store_module.open_sqlite_engine()
    current, head, diffs = _schema_revision_state(engine)
    engine.dispose()
    assert head is not None
    assert current == head
    assert diffs == []


def test_open_sqlite_engine_stamps_and_upgrades_create_all_era_database(tmp_path: pathlib.Path) -> None:
    """A pre-migration database keeps its rows, is stamped at the baseline, and reaches head."""

    from sqlalchemy import create_engine, inspect

    from mosfet.abilities.memory import store as store_module

    path = tmp_path / "legacy.db"
    _create_all_era_database(path)
    legacy = create_engine(f"sqlite+pysqlite:///{path}")
    assert store_module.VERSION_TABLE not in inspect(legacy).get_table_names()
    _ = memory.MemoryStore(engine=legacy).execute(
        memory.InputData(statements=(_insert_content(content="pre-cutover", query_tags="legacy"),))
    )
    legacy.dispose()

    engine = store_module.open_sqlite_engine(database=str(path))
    current, head, diffs = _schema_revision_state(engine)
    assert current == head
    assert diffs == []
    selected = memory.MemoryStore(engine=engine).execute(
        memory.InputData(statements=(_select_by_query_tags(query_tags="legacy"),))
    )
    engine.dispose()
    assert selected.contents() == ("pre-cutover",)


def test_open_sqlite_engine_rejects_unversioned_partial_database(tmp_path: pathlib.Path) -> None:
    """Some baseline tables without a version table match no revision and are not guessed at."""

    import contextlib
    import sqlite3

    from mosfet.abilities.memory import store as store_module

    path = tmp_path / "partial.db"
    with contextlib.closing(sqlite3.connect(path)) as connection, connection:
        _ = connection.execute(f"CREATE TABLE {memory.MEMORY_TABLE} (memory_id TEXT PRIMARY KEY)")
    with pytest.raises(store_module.MigrationError, match="lacks baseline tables"):
        _ = store_module.open_sqlite_engine(database=str(path))
