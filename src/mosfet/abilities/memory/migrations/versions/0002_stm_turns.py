"""Short-term memory of turns: what the bot heard and said.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-20 12:00:00.000000

Adds ``bot_stm_turn``: one row per message a Conversation committed, in words, beside the
short-term register of admitted stimuli. No data moves.
"""

from __future__ import annotations

import collections.abc

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | collections.abc.Sequence[str] | None = "0001"
branch_labels: str | collections.abc.Sequence[str] | None = None
depends_on: str | collections.abc.Sequence[str] | None = None


def upgrade() -> None:
    _ = op.create_table(
        "bot_stm_turn",
        sa.Column("seq", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("turn_id", sa.Text(), nullable=False),
        sa.Column("at", sa.Text(), nullable=False),
        sa.Column("direction", sa.Text(), nullable=False),
        sa.Column("speakers_json", sa.Text(), server_default="[]", nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("cut", sa.Integer(), server_default="0", nullable=False),
        sa.Column("conversation", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("seq"),
        sqlite_autoincrement=True,
    )
    op.create_index("bot_stm_turn_at_idx", "bot_stm_turn", ["at"])


def downgrade() -> None:
    op.drop_index("bot_stm_turn_at_idx", table_name="bot_stm_turn")
    op.drop_table("bot_stm_turn")
