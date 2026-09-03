"""SQL-transaction MemoryStore tests (ability boundary only)."""

from bot.abilities import memory

import asyncio

import pytest

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

    from bot.abilities.memory import store as store_module

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
