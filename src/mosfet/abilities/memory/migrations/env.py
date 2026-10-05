"""Alembic environment for bot memory storage.

Alembic script directory (package data, not an importable package). Online only: the
caller (``mosfet.abilities.memory.store.migrate``) opens the transaction and hands its
connection in through ``config.attributes["connection"]`` on ``store.migration_config()``.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy.engine import Connection

from mosfet.abilities.memory import schema

if context.is_offline_mode():
    raise RuntimeError("Memory migrations run online only; pass a connection via config.attributes['connection'].")

_connection = context.config.attributes.get("connection")
if not isinstance(_connection, Connection):
    raise RuntimeError("Memory migrations require config.attributes['connection'] to be a SQLAlchemy Connection.")

context.configure(
    connection=_connection,
    target_metadata=schema.metadata,
    # SQLite cannot ALTER most column/constraint changes in place; batch mode copies the table.
    render_as_batch=_connection.dialect.name == "sqlite",
)

with context.begin_transaction():
    context.run_migrations()
