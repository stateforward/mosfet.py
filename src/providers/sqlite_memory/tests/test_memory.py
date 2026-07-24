"""SQLite memory provider: SQL-transaction ability."""

from __future__ import annotations

from bot.abilities import memory

import asyncio
import pathlib

from bot.providers.sqlite_memory import (
    MEMORY_TABLE,
    MemoryStore,
    SqliteMemory,
)
from tests.bot.abilities.support import dispatch_ability_for_test, start_abilities_for_test




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

def test_sqlite_memory_is_memory_ability(tmp_path: pathlib.Path) -> None:
    store = SqliteMemory(database_path=tmp_path / "memory.sqlite3")
    assert isinstance(store, memory.Memory)
    assert store.input_event.name == "bot.ability.memory.sqlite.input"
    assert MemoryStore is SqliteMemory


def test_sqlite_memory_transaction_persists_to_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "memory.sqlite3"

    async def write() -> memory.OutputData:
        store = SqliteMemory(database_path=path)
        await start_abilities_for_test(None, store)
        return await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    _insert_content(
                        content="persisted note",
                        context_ref="task-1",
                        query_tags="note",
                        scope="sqlite",
                    ),
                )
            ),
        )

    async def read() -> memory.OutputData:
        store = SqliteMemory(database_path=path)
        await start_abilities_for_test(None, store)
        return await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    memory.Statement(
                        sql=f"SELECT content FROM {MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("note",),
                    ),
                )
            ),
        )

    written = asyncio.run(write())
    assert written.results[0].rowcount == 1
    read_output = asyncio.run(read())
    assert read_output.contents() == ("persisted note",)


def test_sqlite_memory_rolls_back_failed_transaction(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "memory.sqlite3"

    async def run() -> None:
        store = SqliteMemory(database_path=path)
        await start_abilities_for_test(None, store)
        try:
            _ = await dispatch_ability_for_test(
                store,
                None,
                memory.InputData(
                    statements=(
                        _insert_content(content="gone", query_tags="x"),
                        memory.Statement(sql="INSERT INTO missing_table(a) VALUES (1)", parameters=()),
                    )
                ),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected failed transaction")
        ok = await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    memory.Statement(
                        sql=f"SELECT content FROM {MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("x",),
                    ),
                )
            ),
        )
        assert ok.contents() == ()

    asyncio.run(run())
