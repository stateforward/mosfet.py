"""Nearest-neighbor search over Memory's canonical vectors.

Canonical vectors always live in ``bot_memory_vector``, written by Memory itself. An ``Index``
is only an accelerator over them: it can be dropped and rebuilt (``ensure`` plus a backfill)
at any time. Every method is blocking and runs on Memory's database worker, inside Memory's
open transaction, on that transaction's connection, so an index write commits or rolls back
with the rows it indexes.

``ExactIndex`` is the core implementation: it keeps nothing of its own and scans the canonical
vectors of one space. Its cost grows with the number of nodes, so it suits tests and small bots;
a provider supplies an approximate index (``sqlite_memory.VecIndex``) for larger memories.
"""

from __future__ import annotations

import abc
import collections.abc
import heapq
import math
import struct
import typing

from sqlalchemy import select
from sqlalchemy.engine import Connection

from ..embedding import payload as embedding_payload
from . import payload, schema


class Neighbor(typing.NamedTuple):
    """One nearest node: its insertion sequence and its cosine distance to the query."""

    seq: int
    distance: float


def pack(values: collections.abc.Sequence[float]) -> bytes:
    """Encode a vector as little-endian float32, the canonical column format."""

    return struct.pack(f"<{len(values)}f", *values)


def unpack(blob: bytes) -> tuple[float, ...]:
    """Decode a little-endian float32 vector."""

    return struct.unpack(f"<{len(blob) // 4}f", blob)


class Index(abc.ABC):
    """A non-actor accelerator for nearest-vector search in one embedding space.

    Implementations hold no transaction of their own: every call receives Memory's open
    connection. They must be safe to call repeatedly (``ensure`` is idempotent per process).
    """

    @abc.abstractmethod
    def ensure(self, connection: Connection, space: embedding_payload.Space) -> None:
        """Make the index for ``space`` exist (idempotent)."""

    @abc.abstractmethod
    def add(
        self,
        connection: Connection,
        space: embedding_payload.Space,
        seq: int,
        values: collections.abc.Sequence[float],
    ) -> None:
        """Index node ``seq`` under ``values`` in ``space``, replacing any earlier entry.

        The node row already exists in the same transaction, so an index that filters by kind
        reads it from ``bot_memory_node``.
        """

    @abc.abstractmethod
    def remove(self, connection: Connection, seqs: collections.abc.Sequence[int]) -> None:
        """Remove nodes from every space this index holds."""

    @abc.abstractmethod
    def nearest(
        self,
        connection: Connection,
        space: embedding_payload.Space,
        values: collections.abc.Sequence[float],
        *,
        k: int,
        kinds: collections.abc.Sequence[payload.NodeKind],
    ) -> tuple[Neighbor, ...]:
        """The ``k`` nodes of ``kinds`` nearest to ``values`` in ``space``, nearest first."""


class ExactIndex(Index):
    """Exact search by scanning the canonical vectors of one space; holds no state of its own."""

    @typing.override
    def ensure(self, connection: Connection, space: embedding_payload.Space) -> None:
        del connection, space

    @typing.override
    def add(
        self,
        connection: Connection,
        space: embedding_payload.Space,
        seq: int,
        values: collections.abc.Sequence[float],
    ) -> None:
        del connection, space, seq, values

    @typing.override
    def remove(self, connection: Connection, seqs: collections.abc.Sequence[int]) -> None:
        del connection, seqs

    @typing.override
    def nearest(
        self,
        connection: Connection,
        space: embedding_payload.Space,
        values: collections.abc.Sequence[float],
        *,
        k: int,
        kinds: collections.abc.Sequence[payload.NodeKind],
    ) -> tuple[Neighbor, ...]:
        if k <= 0 or not kinds:
            return ()
        vectors = schema.vector_table
        nodes = schema.node_table
        rows = connection.execute(
            select(vectors.c.node_seq, vectors.c.embedding)
            .join(nodes, nodes.c.seq == vectors.c.node_seq)
            .where(
                vectors.c.space_model == space.model,
                vectors.c.dimensions == space.dimensions,
                nodes.c.kind.in_([kind.value for kind in kinds]),
            )
        ).all()
        # Vectors are unit length (embedding contract), so cosine similarity is the dot product.
        scored = (
            (math.fsum(a * b for a, b in zip(values, unpack(typing.cast(bytes, blob)), strict=True)), int(seq))
            for seq, blob in rows
        )
        best = heapq.nlargest(k, scored, key=lambda item: (item[0], -item[1]))
        return tuple(Neighbor(seq=seq, distance=1.0 - similarity) for similarity, seq in best)


__all__ = ["ExactIndex", "Index", "Neighbor", "pack", "unpack"]
