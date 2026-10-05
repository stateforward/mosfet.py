"""Behavior lifetime: turn behaviors versus persistent routines.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05 10:00:46.173412

Adds ``bot_behavior.lifetime``. ``turn`` rows are behaviors Autonomy runs for one matching
turn; ``persistent`` rows are routines the Routines ability keeps running across turns and
restarts. Every behavior stored before this revision was a turn behavior, so existing rows
take the ``turn`` server default.
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
    with op.batch_alter_table("bot_behavior", schema=None) as batch_op:
        batch_op.add_column(sa.Column("lifetime", sa.Text(), server_default="turn", nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("bot_behavior", schema=None) as batch_op:
        batch_op.drop_column("lifetime")
