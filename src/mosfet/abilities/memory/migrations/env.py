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


def _include_name(name: str | None, type_: str, parent_names: object) -> bool:
    """Leave provider-created vector tables (and their vec0 shadow tables) alone.

    Both prefixes are the providers': memory's index (``schema.VECTOR_INDEX_PREFIX``,
    ``bot_memory_vec_``) and the identity prints (``schema.IDENTITY_VECTOR_PREFIX``,
    ``bot_identity_vec_``).
    """

    del parent_names
    return not (
        type_ == "table"
        and name is not None
        and name.startswith((schema.VECTOR_INDEX_PREFIX, schema.IDENTITY_VECTOR_PREFIX))
    )


context.configure(
    connection=_connection,
    target_metadata=schema.metadata,
    include_name=_include_name,
    # SQLite cannot ALTER most column/constraint changes in place; batch mode copies the table.
    render_as_batch=_connection.dialect.name == "sqlite",
)

with context.begin_transaction():
    context.run_migrations()
