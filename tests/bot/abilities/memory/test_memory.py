"""Core Memory SQL-transaction and generation contracts."""

from bot.abilities import memory

import asyncio
import inspect

import pytest

from tests.bot.abilities.memory.memory_fixtures import (
    PreferenceMemoryGenerator,
    GeneratedMemoryEncoder,
    preference_memory_generation,
    require_model,
    start_ability_tree,
)
from tests.bot.abilities.support import dispatch_ability_for_test, shared_hsm_context, start_abilities_for_test
from tests.type_helpers import object_dict


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


def test_memory_is_sql_transaction_ability_without_encoder() -> None:
    assert "encoder" not in inspect.signature(memory.Memory).parameters
    assert "decoder" not in inspect.signature(memory.Memory).parameters
    store = memory.Memory()
    assert store.input_event.name == "bot.ability.memory.input"
    assert issubclass(memory.ShortTermMemory, memory.Memory)
    assert issubclass(memory.LongTermMemory, memory.Memory)


def test_memory_apply_select_and_insert_transaction() -> None:
    async def run() -> memory.OutputData:
        store = memory.Memory()
        await start_abilities_for_test(shared_hsm_context(), store)
        return await dispatch_ability_for_test(
            store,
            None,
            memory.InputData(
                statements=(
                    _insert_content(
                        content="Gabe prefers terse handoff notes.",
                        context_ref="active-task",
                        subject_ref="operator",
                        kind="preference",
                        query_tags="handoff",
                    ),
                    memory.Statement(
                        sql=(f"SELECT content FROM {memory.MEMORY_TABLE} WHERE context_ref = ? ORDER BY created_at"),
                        parameters=("active-task",),
                    ),
                )
            ),
        )

    output = asyncio.run(run())
    assert output.contents(statement_index=1) == ("Gabe prefers terse handoff notes.",)


def test_memory_generation_still_exists_as_sibling_ability() -> None:
    generation = preference_memory_generation()
    assert generation.input_event.name == "bot.ability.memory.generation.input"
    assert generation.output_event.name == "bot.ability.memory.generation.output"
    assert "SourceData" in memory.SourceData.__name__
    input_schema = object_dict(generation.input_event.schema)
    assert input_schema == memory.SourceData.model_json_schema()


def test_memory_generation_produces_candidate_for_insert_params() -> None:
    async def run() -> memory.CandidateData:
        generation = memory.MemoryGeneration(
            generator=PreferenceMemoryGenerator(),
            encoder=GeneratedMemoryEncoder(),
        )
        await start_ability_tree(generation)
        return await dispatch_ability_for_test(
            generation,
            None,
            memory.SourceData(
                decoded="Gabe prefers terse handoff notes.",
                subject_ref="operator",
            ),
        )

    candidate = asyncio.run(run())
    assert candidate.memory.content == "The operator prefers terse handoff notes."
    assert candidate.encoded is not None


def test_memory_input_statements_min_length() -> None:
    with pytest.raises(Exception):
        _ = memory.InputData(statements=())


def test_memory_model_is_idle_applying() -> None:
    model = require_model(memory.Memory.model)
    assert model is not None
