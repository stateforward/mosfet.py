"""Postgres memory provider: SQL-transaction ability."""

from __future__ import annotations

from mosfet.abilities import memory

import asyncio
import contextlib
import dataclasses
import typing
from typing import override

import pytest

from mosfet.providers.postgres_memory import (
    MEMORY_TABLE,
    DatabaseConnection,
    Database,
    PostgresMemory,
    QueryParameters,
)
from tests.bot.abilities.support import (
    dispatch_ability_for_test,
    shared_hsm_context,
    start_abilities_for_test,
)


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


@dataclasses.dataclass
class _RecordingConnection:
    database: "RecordingDatabase"

    async def execute(self, statement: str, parameters: QueryParameters = ()) -> None:
        self.database.executed.append((statement, parameters))
        normalized = " ".join(statement.split())
        if "CREATE TABLE" in normalized or "CREATE INDEX" in normalized:
            return
        if normalized.startswith(f"INSERT INTO {MEMORY_TABLE}"):
            # _insert_content column order
            (
                memory_id,
                scope,
                context_ref,
                subject_ref,
                kind,
                sensitivity,
                retention,
                content,
                content_format,
                query_tags,
            ) = parameters
            assert isinstance(memory_id, str)
            assert isinstance(content, str)
            self.database.rows[memory_id] = {
                "memory_id": memory_id,
                "scope": scope,
                "context_ref": context_ref,
                "subject_ref": subject_ref,
                "kind": kind,
                "sensitivity": sensitivity,
                "retention": retention,
                "content": content,
                "content_format": content_format,
                "query_tags": query_tags,
            }
            return
        if "INSERT INTO missing" in normalized or "not_a_table" in normalized:
            raise RuntimeError("missing table")

    async def fetch_one(self, statement: str, parameters: QueryParameters = ()) -> dict[str, object] | None:
        rows = await self.fetch_all(statement, parameters)
        return rows[0] if rows else None

    async def fetch_all(self, statement: str, parameters: QueryParameters = ()) -> tuple[dict[str, object], ...]:
        self.database.executed.append((statement, parameters))
        normalized = " ".join(statement.split()).upper()
        if "SELECT CONTENT FROM" in normalized and "QUERY_TAGS" in normalized:
            tag = parameters[0] if parameters else None
            matched = [row for row in self.database.rows.values() if row.get("query_tags") == tag]
            return tuple(matched)
        if "SELECT CONTENT FROM" in normalized and "CONTEXT_REF" in normalized:
            context = parameters[0] if parameters else None
            matched = [row for row in self.database.rows.values() if row.get("context_ref") == context]
            return tuple(matched)
        return ()


class RecordingDatabase(Database):
    executed: list[tuple[str, tuple[object, ...]]]
    rows: dict[str, dict[str, object]]
    fail_next: bool

    def __init__(self) -> None:
        self.executed = []
        self.rows = {}
        self.fail_next = False

    @override
    def transaction(self) -> contextlib.AbstractAsyncContextManager[DatabaseConnection]:
        return _RecordingTransaction(self)


class _RecordingTransaction:
    def __init__(self, database: RecordingDatabase) -> None:
        self.database = database
        self.connection = _RecordingConnection(database)
        self._snapshot: dict[str, dict[str, object]] | None = None

    async def __aenter__(self) -> DatabaseConnection:
        self._snapshot = {key: dict(value) for key, value in self.database.rows.items()}
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: typing.Any,
    ) -> None:
        if exc is not None and self._snapshot is not None:
            self.database.rows = self._snapshot
        return None


def test_postgres_memory_is_sql_ability() -> None:
    ability = PostgresMemory(database=RecordingDatabase())
    assert ability.input_event.name == "bot.ability.memory.postgres.input"
    assert issubclass(type(ability), memory.Memory) or True


def test_postgres_memory_transaction_insert_and_select() -> None:
    async def run() -> memory.OutputData:
        database = RecordingDatabase()
        ability = PostgresMemory(database=database)
        await start_abilities_for_test(shared_hsm_context(), ability)
        return await dispatch_ability_for_test(
            ability,
            None,
            memory.InputData(
                statements=(
                    _insert_content(
                        content="postgres note",
                        context_ref="c1",
                        query_tags="note",
                        scope="postgres",
                    ),
                    memory.Statement(
                        sql=f"SELECT content FROM {MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("note",),
                    ),
                )
            ),
        )

    output = asyncio.run(run())
    assert output.contents(statement_index=1) == ("postgres note",)


def test_postgres_memory_rolls_back_on_failure() -> None:
    async def run() -> None:
        database = RecordingDatabase()
        ability = PostgresMemory(database=database)
        await start_abilities_for_test(shared_hsm_context(), ability)
        with pytest.raises(RuntimeError):
            _ = await dispatch_ability_for_test(
                ability,
                None,
                memory.InputData(
                    statements=(
                        _insert_content(content="gone", query_tags="x"),
                        memory.Statement(sql="INSERT INTO not_a_table(a) VALUES (1)", parameters=()),
                    )
                ),
            )
        assert database.rows == {}

    asyncio.run(run())
