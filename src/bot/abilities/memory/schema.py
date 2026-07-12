"""Canonical relational schema for bot memory (SQLAlchemy Core).

Domain code builds Core ``select`` / ``insert`` / ``update`` / ``delete`` against
``memory_table``. The store and providers compile those clauses to dialect SQL.
"""

from __future__ import annotations

from sqlalchemy import Column
from sqlalchemy import Index
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import func

# Logical table name (dialect-neutral identifier).
MEMORY_TABLE = "bot_memory"

metadata = MetaData()

memory_table = Table(
    MEMORY_TABLE,
    metadata,
    Column("memory_id", Text, primary_key=True),
    Column("scope", Text, nullable=False, server_default="memory"),
    Column("context_ref", Text, nullable=True),
    Column("subject_ref", Text, nullable=True),
    Column("kind", Text, nullable=True),
    Column("sensitivity", Text, nullable=True),
    Column("retention", Text, nullable=True),
    Column("content", Text, nullable=False),
    Column("content_format", Text, nullable=False, server_default="text/plain"),
    Column("query_tags", Text, nullable=True),
    Column("metadata_json", Text, nullable=False, server_default="{}"),
    # Stored as text for dialect neutrality; providers may map to timestamptz.
    Column("created_at", Text, nullable=False, server_default=func.now()),
    Index("bot_memory_context_idx", "context_ref", "subject_ref"),
    Index("bot_memory_query_idx", "query_tags"),
    Index("bot_memory_scope_idx", "scope", "created_at"),
)

# Register habit tables on the same MetaData so create_all materializes them.
from bot.habit import storage as _habit_storage  # noqa: E402, F401

__all__ = [
    "MEMORY_TABLE",
    "memory_table",
    "metadata",
]
