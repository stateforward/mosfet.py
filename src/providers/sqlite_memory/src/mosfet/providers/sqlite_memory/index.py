"""``VecIndex``: approximate nearest search for ``memory.Memory`` on sqlite-vec ``vec0`` tables.

One ``vec0`` virtual table per embedding space, named only from the space's dimensions and a hash
of its model name (``bot_memory_vec_<dims>_<sha256(model)[:8]>``), never from caller text. Every
call runs inside Memory's open transaction, so index rows commit or roll back with the canonical
vectors in ``bot_memory_vector``; the index can always be dropped and rebuilt from them.
"""

from __future__ import annotations

import collections.abc
import hashlib

from mosfet.abilities import embedding, memory
from sqlalchemy import text
from sqlalchemy.engine import Connection


def table_name(space: embedding.Space) -> str:
    """The deterministic, identifier-safe vec0 table name of ``space``."""

    digest = hashlib.sha256(space.model.encode("utf-8")).hexdigest()[:8]
    return f"{memory.schema.VECTOR_INDEX_PREFIX}{space.dimensions}_{digest}"


class VecIndex(memory.Index):
    """Cosine nearest search over one ``vec0`` table per space, filtered by node kind."""

    def ensure(self, connection: Connection, space: embedding.Space) -> None:
        _ = connection.exec_driver_sql(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table_name(space)} USING vec0("
            f"node_seq INTEGER PRIMARY KEY, embedding float[{space.dimensions}] distance_metric=cosine, kind TEXT)"
        )

    def add(
        self,
        connection: Connection,
        space: embedding.Space,
        seq: int,
        values: collections.abc.Sequence[float],
    ) -> None:
        kind = connection.execute(
            text(f"SELECT kind FROM {memory.schema.NODE_TABLE} WHERE seq = :seq"), {"seq": seq}
        ).scalar_one()
        table = table_name(space)
        _ = connection.exec_driver_sql(f"DELETE FROM {table} WHERE node_seq = ?", (seq,))
        _ = connection.exec_driver_sql(
            f"INSERT INTO {table} (node_seq, embedding, kind) VALUES (?, ?, ?)",
            (seq, memory.index.pack(values), kind),
        )

    def remove(self, connection: Connection, seqs: collections.abc.Sequence[int]) -> None:
        if not seqs:
            return
        tables = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ? AND sql LIKE '%USING vec0%'",
            (f"{memory.schema.VECTOR_INDEX_PREFIX}%",),
        ).scalars()
        marks = ", ".join("?" for _ in seqs)
        for table in list(tables):
            _ = connection.exec_driver_sql(f"DELETE FROM {table} WHERE node_seq IN ({marks})", tuple(seqs))

    def nearest(
        self,
        connection: Connection,
        space: embedding.Space,
        values: collections.abc.Sequence[float],
        *,
        k: int,
        kinds: collections.abc.Sequence[memory.NodeKind],
    ) -> tuple[memory.Neighbor, ...]:
        if k <= 0 or not kinds:
            return ()
        table = table_name(space)
        exists = connection.exec_driver_sql(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).first()
        if exists is None:
            return ()
        marks = ", ".join("?" for _ in kinds)
        rows = connection.exec_driver_sql(
            f"SELECT node_seq, distance FROM {table} WHERE embedding MATCH ? AND k = ? AND kind IN ({marks}) "
            "ORDER BY distance",
            (memory.index.pack(values), k, *(kind.value for kind in kinds)),
        ).all()
        # vec0 allows only ``ORDER BY distance``; equal distances break ties by insertion order here.
        ordered = sorted(((float(distance), int(seq)) for seq, distance in rows))
        return tuple(memory.Neighbor(seq=seq, distance=distance) for distance, seq in ordered)


__all__ = ["VecIndex", "table_name"]
