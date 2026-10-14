"""Memory schema baseline: every table the memory store owns.

Revision ID: 0001
Revises:
Create Date: 2026-10-14 16:30:00.000000

The one baseline revision: relational memory rows, behavior inventory (with each behavior's
``lifetime``), the short-term register, and the episodic graph (nodes, parts, edges,
canonical vectors, index attempts). Provider-created vector index tables are not managed
here. Schema changes after this baseline go in new revisions.
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
        sa.Column("lifetime", sa.Text(), server_default="turn", nullable=False),
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

    _ = op.create_table(
        "bot_memory_node",
        sa.Column("seq", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("node_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("key", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), server_default="", nullable=False),
        sa.Column("content_json", sa.Text(), server_default="{}", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("recalled_at", sa.Text(), nullable=True),
        sa.Column("recall_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("reviewed_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("seq"),
        sa.UniqueConstraint("node_id", name="bot_memory_node_id_key"),
        sa.UniqueConstraint("kind", "key", name="bot_memory_node_kind_key"),
        sqlite_autoincrement=True,
    )
    op.create_index("bot_memory_node_recent_idx", "bot_memory_node", ["kind", "created_at", "seq"])
    op.create_index("bot_memory_node_review_idx", "bot_memory_node", ["kind", "reviewed_at", "seq"])

    _ = op.create_table(
        "bot_memory_part",
        sa.Column("node_seq", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("media_type", sa.Text(), nullable=True),
        sa.Column("width_px", sa.Integer(), nullable=True),
        sa.Column("height_px", sa.Integer(), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.PrimaryKeyConstraint("node_seq", "ordinal", name="bot_memory_part_pkey"),
    )

    _ = op.create_table(
        "bot_memory_edge",
        sa.Column("source_seq", sa.Integer(), nullable=False),
        sa.Column("target_seq", sa.Integer(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("weight", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("origin", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("source_seq", "target_seq", "relation", name="bot_memory_edge_pkey"),
    )
    op.create_index("bot_memory_edge_target_idx", "bot_memory_edge", ["target_seq"])

    _ = op.create_table(
        "bot_memory_vector",
        sa.Column("node_seq", sa.Integer(), nullable=False),
        sa.Column("space_model", sa.Text(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("node_seq", "space_model", "dimensions", name="bot_memory_vector_pkey"),
    )

    _ = op.create_table(
        "bot_memory_index_attempt",
        sa.Column("node_seq", sa.Integer(), nullable=False),
        sa.Column("space_model", sa.Text(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("failure_kind", sa.Text(), nullable=False),
        sa.Column("permanent", sa.Integer(), nullable=False),
        sa.Column("pass_id", sa.Text(), nullable=True),
        sa.Column("attempted_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("node_seq", "space_model", "dimensions", name="bot_memory_index_attempt_pkey"),
    )
    op.create_index(
        "bot_memory_index_attempt_space_idx",
        "bot_memory_index_attempt",
        ["space_model", "dimensions", "permanent"],
    )


def downgrade() -> None:
    op.drop_index("bot_memory_index_attempt_space_idx", table_name="bot_memory_index_attempt")
    op.drop_table("bot_memory_index_attempt")
    op.drop_table("bot_memory_vector")
    op.drop_index("bot_memory_edge_target_idx", table_name="bot_memory_edge")
    op.drop_table("bot_memory_edge")
    op.drop_table("bot_memory_part")
    op.drop_index("bot_memory_node_review_idx", table_name="bot_memory_node")
    op.drop_index("bot_memory_node_recent_idx", table_name="bot_memory_node")
    op.drop_table("bot_memory_node")
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
