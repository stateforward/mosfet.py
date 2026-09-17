from mosfet.abilities import memory

import asyncio

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


def test_short_term_memory_event_names_and_scope() -> None:
    assert memory.ShortTermMemory.default_scope == "short_term"
    assert memory.ShortTermMemory.input_event.name == "bot.ability.memory.short_term.input"
    assert memory.ShortTermMemory.output_event.name == "bot.ability.memory.short_term.output"


def test_short_term_memory_sql_apply() -> None:
    async def run() -> memory.OutputData:
        short_term = memory.ShortTermMemory()
        await start_abilities_for_test(shared_hsm_context(), short_term)
        return await dispatch_ability_for_test(
            short_term,
            None,
            memory.InputData(
                statements=(
                    _insert_content(
                        content="short",
                        scope="short_term",
                        query_tags="t",
                    ),
                    memory.Statement(
                        sql=f"SELECT content FROM {memory.MEMORY_TABLE} WHERE query_tags = ?",
                        parameters=("t",),
                    ),
                )
            ),
        )

    output = asyncio.run(run())
    assert output.contents(statement_index=1) == ("short",)
