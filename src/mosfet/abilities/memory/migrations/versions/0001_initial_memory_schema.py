"""Initial memory schema: the full create_all-era schema.

Revision ID: 0001
Revises:
Create Date: 2026-10-05 09:29:19.541641

Creates exactly the tables and indexes ``schema.metadata.create_all`` created before
migrations existed (same SQLite DDL, byte for byte).

One-time cutover: a database ``create_all`` created has all of these tables but no
``alembic_version`` table. ``mosfet.abilities.memory.store.migrate`` detects that shape,
stamps it at this revision without running it, then upgrades to head like any other
database. An unversioned database holding only some of these tables matches no revision
and raises ``store.MigrationError`` instead of being guessed at. This baseline is frozen:
schema changes go in new revisions, never here.
"""

from __future__ import annotations

import collections.abc

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | collections.abc.Sequence[str] | None = None
branch_labels: str | collections.abc.Sequence[str] | None = None
depends_on: str | collections.abc.Sequence[str] | None = None


def upgrade() -> None:
    _ = op.create_table(
        "bot_memory",
        sa.Column("memory_id", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), server_default="memory", nullable=False),
        sa.Column("context_ref", sa.Text(), nullable=True),
        sa.Column("subject_ref", sa.Text(), nullable=True),
        sa.Column("kind", sa.Text(), nullable=True),
        sa.Column("sensitivity", sa.Text(), nullable=True),
        sa.Column("retention", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_format", sa.Text(), server_default="text/plain", nullable=False),
        sa.Column("query_tags", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.Text(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index("bot_memory_context_idx", "bot_memory", ["context_ref", "subject_ref"])
    op.create_index("bot_memory_query_idx", "bot_memory", ["query_tags"])
    op.create_index("bot_memory_scope_idx", "bot_memory", ["scope", "created_at"])

    _ = op.create_table(
        "bot_behavior",
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("examples_json", sa.Text(), server_default="[]", nullable=False),
        sa.Column("status", sa.Text(), server_default="ACTIVE", nullable=False),
        sa.Column("status_reason", sa.Text(), server_default="", nullable=False),
        sa.Column("status_updated_at", sa.Text(), server_default="", nullable=False),
        sa.Column("used_count", sa.Text(), server_default="0", nullable=False),
        sa.Column("last_used_at", sa.Text(), server_default="", nullable=False),
        sa.Column("failed_count", sa.Text(), server_default="0", nullable=False),
        sa.Column("last_failed_at", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.Text(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.Text(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("name"),
    )

    _ = op.create_table(
        "bot_behavior_trigger",
        sa.Column("behavior_name", sa.Text(), nullable=False),
        sa.Column("trigger", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["behavior_name"], ["bot_behavior.name"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("behavior_name", "trigger"),
    )
    op.create_index("bot_behavior_trigger_event_idx", "bot_behavior_trigger", ["trigger"])

    _ = op.create_table(
        "bot_stm_memory",
        sa.Column("stm_id", sa.Text(), nullable=False),
        sa.Column("stimulus_name", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), server_default="{}", nullable=False),
        sa.Column("event_id", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("target", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_accessed_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("stm_id"),
    )
    op.create_index("bot_stm_memory_recent_idx", "bot_stm_memory", ["stimulus_name", "created_at"])
    op.create_index("bot_stm_memory_created_idx", "bot_stm_memory", ["created_at"])


def downgrade() -> None:
    op.drop_index("bot_stm_memory_created_idx", table_name="bot_stm_memory")
    op.drop_index("bot_stm_memory_recent_idx", table_name="bot_stm_memory")
    op.drop_table("bot_stm_memory")
    op.drop_index("bot_behavior_trigger_event_idx", table_name="bot_behavior_trigger")
    op.drop_table("bot_behavior_trigger")
    op.drop_table("bot_behavior")
    op.drop_index("bot_memory_scope_idx", table_name="bot_memory")
    op.drop_index("bot_memory_query_idx", table_name="bot_memory")
    op.drop_index("bot_memory_context_idx", table_name="bot_memory")
    op.drop_table("bot_memory")
