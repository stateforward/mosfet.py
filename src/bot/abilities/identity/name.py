"""The name a bot answers to, stored the way every other durable fact is.

A name is a **fact about a subject**, and the subject is the bot itself: ``kind=fact``,
``subject_ref=self``, ``query_tags=self_name`` on the existing ``bot_memory`` schema — no new
table, no migration. It is deliberately *not* a standing directive: a directive is something the
bot was told to *do* and stays outstanding until released (see ``cognition.directives``), while a
name is something the bot *is*. Nothing about a name is ever completed, released, or acted on.

Adoption is append-only. A bot that was renamed still remembers having been called something
else, so the current name is simply the most recently adopted row.
"""

from __future__ import annotations

from .. import memory

import datetime
import typing
import uuid

import pydantic
from sqlalchemy import insert
from sqlalchemy import select

# Stable query_tags value for adopted names in bot_memory.
NAME_QUERY = "self_name"
# Stable subject_ref for a memory that is about the bot itself.
NAME_SUBJECT = "self"


class Name(pydantic.BaseModel):
    """One name the bot adopted, exactly as it was given."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "description": (
                "A name the bot answers to (JSON in memory, query_tags=self_name). A bot starts with "
                "none; this row exists only because the bot adopted one."
            ),
            "examples": [{"text": "Bob", "kind": "fact"}],
        },
    )

    text: str = pydantic.Field(
        min_length=1,
        description="The name as the bot adopted it, in plain language.",
        examples=["Bob"],
    )
    kind: memory.MemoryClassificationKind = pydantic.Field(
        default=memory.MemoryClassificationKind.FACT,
        description="Memory classification written on the name row.",
        examples=[memory.MemoryClassificationKind.FACT],
    )


def names_from_output(output: memory.OutputData, *, statement_index: int = 0) -> tuple[Name, ...]:
    """Decode adopted names from a committed memory SELECT result, oldest first."""

    names: list[Name] = []
    for content in output.contents(statement_index=statement_index):
        try:
            names.append(Name.model_validate_json(content))
        except Exception:
            continue
    return tuple(names)


def adopted_name(output: memory.OutputData, *, statement_index: int = 0) -> Name | None:
    """Return the name the bot currently answers to, or None when it has never adopted one."""

    names = names_from_output(output, statement_index=statement_index)
    return names[-1] if names else None


def name_select_input(*, limit: int = 50) -> memory.InputData:
    """Memory apply input that SELECTs the names this bot has adopted, oldest first."""

    table = memory.memory_table
    clause = (
        select(table)
        .where(table.c.query_tags == NAME_QUERY)
        .where(table.c.subject_ref == NAME_SUBJECT)
        .order_by(table.c.created_at)
        .limit(limit)
    )
    return memory.InputData(statements=memory.compile_statements(clause))


def name_insert_input(
    name: Name,
    *,
    scope: str = "long_term",
    memory_id: str | None = None,
    created_at: datetime.datetime | None = None,
) -> memory.InputData:
    """Memory apply input that INSERTs one adopted name.

    ``created_at`` is written explicitly rather than left to the column default: the default is
    ``CURRENT_TIMESTAMP``, whose one-second resolution cannot order two adoptions made in the
    same second, and "which name is current" is exactly that ordering.
    """

    stamp = created_at if created_at is not None else datetime.datetime.now(datetime.UTC)
    table = memory.memory_table
    clause = insert(table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope=scope,
        context_ref=None,
        subject_ref=NAME_SUBJECT,
        kind=str(name.kind),
        sensitivity="standard",
        retention="retain",
        content=name.model_dump_json(),
        content_format="application/json",
        query_tags=NAME_QUERY,
        created_at=stamp.isoformat(),
    )
    return memory.InputData(statements=memory.compile_statements(clause))


__all__ = [
    "NAME_QUERY",
    "NAME_SUBJECT",
    "Name",
    "adopted_name",
    "name_insert_input",
    "name_select_input",
    "names_from_output",
]
