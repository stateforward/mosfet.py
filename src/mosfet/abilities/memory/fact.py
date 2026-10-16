"""Facts a bot holds about itself, stored the way every other durable fact is.

A self-fact is a **fact about a subject**, and the subject is the bot itself: ``kind=fact``,
``subject_ref=self`` on the existing ``bot_memory`` schema, told apart by ``query_tags`` — no new
table, no migration. It records something the bot *is*, never something it was told to *do*, so
nothing about it is ever completed, released, or acted on.

Self-facts are append-only. The current value of one is simply its most recent row, so every row
is stamped explicitly at microsecond resolution: the column default is ``CURRENT_TIMESTAMP``,
whose one-second resolution cannot order two facts written in the same second.
"""

from __future__ import annotations

from . import graph, schema, store

import datetime
import uuid

import pydantic
from sqlalchemy import insert

# Stable subject_ref for a memory that is about the bot itself.
SELF_SUBJECT = "self"


def self_fact_select_input(query_tags: str, *, limit: int = 50) -> store.InputData:
    """Memory apply input that SELECTs the ``limit`` most recent self-facts tagged ``query_tags``, oldest first."""

    table = schema.memory_table
    clause = store.recent_select(table.c.query_tags == query_tags, table.c.subject_ref == SELF_SUBJECT, limit=limit)
    return store.InputData(statements=store.compile_statements(clause))


def self_fact_insert_input(
    query_tags: str,
    content: pydantic.BaseModel,
    *,
    created_at: datetime.datetime | None = None,
    memory_id: str | None = None,
) -> store.InputData:
    """Memory apply input that INSERTs one self-fact tagged ``query_tags`` with ``content`` as JSON.

    The row is a retained, long-term, standard-sensitivity ``fact`` about ``SELF_SUBJECT``.
    ``created_at`` (now, when omitted) is written in UTC at microsecond resolution.
    """

    stamp = created_at if created_at is not None else datetime.datetime.now(datetime.UTC)
    clause = insert(schema.memory_table).values(
        memory_id=memory_id or uuid.uuid4().hex,
        scope="long_term",
        context_ref=None,
        subject_ref=SELF_SUBJECT,
        kind="fact",
        sensitivity="standard",
        retention="retain",
        content=content.model_dump_json(),
        content_format="application/json",
        query_tags=query_tags,
        created_at=graph.utc_text(stamp),
    )
    return store.InputData(statements=store.compile_statements(clause))


__all__ = [
    "SELF_SUBJECT",
    "self_fact_insert_input",
    "self_fact_select_input",
]
