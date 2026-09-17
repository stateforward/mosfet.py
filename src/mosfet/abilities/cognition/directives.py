"""Standing directives: things the bot was told, recalled the way prior turns are.

A directive is an instruction memory row (``kind=instruction``, ``query_tags=standing_directive``)
on the existing ``bot_memory`` schema — no new table, no migration.

Nothing here names a device, a deadline, a priority, or a completion flag. "Call Bob" is not
about the phone until cognition decides the phone is how you reach Bob, and a bot knows it
already called because it *remembers calling* — episodes carry that. A completion flag would
make topology track goal state.
"""

from __future__ import annotations

from .. import memory

import typing
import uuid

import pydantic
from sqlalchemy import insert
from sqlalchemy import or_
from sqlalchemy import select

# Stable query_tags value for standing directives in bot_memory.
DIRECTIVE_QUERY = "standing_directive"


class Directive(pydantic.BaseModel):
    """One standing instruction the bot was told and has not been released from."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "Recallable standing directive (JSON in memory, query_tags=standing_directive). "
                "What the bot was told; it stays recalled until it is removed from memory."
            ),
            "examples": [{"text": "Call Bob when you get a chance.", "kind": "instruction"}],
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description="The instruction exactly as the bot was told it, in plain language.",
        examples=["Call Bob when you get a chance."],
    )
    kind: memory.MemoryClassificationKind = pydantic.Field(
        default=memory.MemoryClassificationKind.INSTRUCTION,
        description="Memory classification written on the directive row.",
        examples=[memory.MemoryClassificationKind.INSTRUCTION],
    )


def directives_from_output(output: memory.OutputData, *, statement_index: int = 0) -> tuple[Directive, ...]:
    """Decode standing directives from a committed memory SELECT result."""

    directives: list[Directive] = []
    for content in output.contents(statement_index=statement_index):
        try:
            directives.append(Directive.model_validate_json(content))
        except Exception:
            continue
    return tuple(directives)


def directive_select_input(
    *,
    context_ref: str | None = None,
    limit: int = 50,
) -> memory.InputData:
    """Memory apply input that SELECTs the standing directives the bot was given."""

    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == DIRECTIVE_QUERY)
    if context_ref is not None:
        clause = clause.where(or_(table.c.context_ref.is_(None), table.c.context_ref == context_ref))
    clause = clause.order_by(table.c.created_at).limit(limit)
    return memory.InputData(statements=memory.compile_statements(clause))


def directive_insert_input(
    directive: Directive,
    *,
    context_ref: str | None,
    scope: str = "short_term",
    memory_id: str | None = None,
) -> memory.InputData:
    """Memory apply input that INSERTs one standing directive the bot was told."""

    table = memory.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=scope,
        context_ref=context_ref,
        subject_ref=None,
        kind=str(directive.kind),
        sensitivity="standard",
        retention="retain",
        content=directive.model_dump_json(),
        content_format="application/json",
        query_tags=DIRECTIVE_QUERY,
    )
    return memory.InputData(statements=memory.compile_statements(clause))


__all__ = [
    "DIRECTIVE_QUERY",
    "Directive",
    "directive_insert_input",
    "directive_select_input",
    "directives_from_output",
]
