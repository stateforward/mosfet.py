"""``IdentityStore``: ``identification.Store`` on the memory database, with prints in ``vec0`` tables.

Identities are rows of core ``memory.schema.identity_table`` (``bot_identity``, in the 0001
baseline). Each print lives in one ``vec0`` table per (model, dims), named only from the dims and
a hash of the model (``bot_identity_vec_<dims>_<sha256(model)[:8]>``), never from caller text, and
created on first use. The row's ``seq`` is the print's rowid; a ``kind`` metadata column keeps
voices and faces apart even in one model's table, and distances are cosine (similarity is
``1 - distance``). The prefix is disjoint from memory's ``bot_memory_vec_``, so Memory's index
never touches identity prints and migrations leave both alone.

Every call is one ``BEGIN IMMEDIATE`` transaction on its own pooled connection, under one lock
per store: the write lock is taken up front, Memory's writers wait on ``busy_timeout``, and two
registries sharing the store (or a call a registry abandoned on its timeout) never overlap.
"""

from __future__ import annotations

import collections.abc
import datetime
import hashlib
import sqlite3
import threading
import typing
import uuid

import sqlalchemy
import sqlalchemy.exc
from mosfet.abilities import identification, memory
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import StaticPool

_IDENTITIES = memory.schema.identity_table


def table_name(model: str, dimensions: int) -> str:
    """The deterministic, identifier-safe ``vec0`` table name of one print model's space."""

    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{memory.schema.IDENTITY_VECTOR_PREFIX}{dimensions}_{digest}"


def _tables(connection: Connection, model: str) -> dict[str, int]:
    """The existing ``vec0`` print tables of ``model``, with their dims (one, unless hashes collide)."""

    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    names = connection.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB ? AND sql LIKE '%USING vec0%'",
        (f"{memory.schema.IDENTITY_VECTOR_PREFIX}[0-9]*_{digest}",),
    ).scalars()
    prefix = len(memory.schema.IDENTITY_VECTOR_PREFIX)
    return {str(name): int(str(name)[prefix:].split("_", 1)[0]) for name in names}


def _space(connection: Connection, print: identification.Print) -> str:
    """The print table of ``print``'s model, created on first use; a model keeps one dims.

    Raises:
        identification.StoreError: ``INVALID_INPUT`` when the model's space has other dims.
    """

    dimensions = len(print.vector)
    for recorded in _tables(connection, print.model).values():
        if recorded != dimensions:
            raise identification.StoreError(
                identification.FailureKind.INVALID_INPUT,
                f"a print has {dimensions} dimensions; its model's space has {recorded}",
            )
    table = table_name(print.model, dimensions)
    _ = connection.exec_driver_sql(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
        f"seq INTEGER PRIMARY KEY, embedding float[{dimensions}] distance_metric=cosine, kind TEXT)"
    )
    return table


def _stored(connection: Connection, table: str, seq: int) -> tuple[float, ...]:
    blob = connection.exec_driver_sql(f"SELECT embedding FROM {table} WHERE seq = ?", (seq,)).scalar_one()
    return memory.index.unpack(typing.cast(bytes, blob))


def _delete(connection: Connection, rows: collections.abc.Sequence[sqlalchemy.Row[tuple[int, str]]]) -> int:
    """Delete identities (``seq``, ``model``) and their prints; how many."""

    for seq, model in rows:
        for table in _tables(connection, model):
            _ = connection.exec_driver_sql(f"DELETE FROM {table} WHERE seq = ?", (seq,))
    if rows:
        _ = connection.execute(sqlalchemy.delete(_IDENTITIES).where(_IDENTITIES.c.seq.in_([seq for seq, _ in rows])))
    return len(rows)


class IdentityStore(identification.Store):
    """Identities and their prints in the memory database, one ``BEGIN IMMEDIATE`` transaction per call.

    It has no hard cap: retention (``prune``) bounds it. It never prunes ``own`` or ``linked`` ids,
    and ``forget`` deletes only the identity row and its print; memory nodes that carry the
    identity's ref stay.
    """

    _engine: Engine
    _lock: threading.Lock

    def __init__(self, *, engine: Engine) -> None:
        """Keep identities in ``engine``'s file database, built by ``sqlite_memory.open_engine``.

        The engine may be the one Memory uses: each call takes its own pooled connection.

        Raises:
            ValueError: If ``engine`` is one shared connection (an in-memory database).
        """

        if isinstance(engine.pool, StaticPool):
            raise ValueError("IdentityStore needs a file database; use identification.InProcessStore in memory")
        self._engine = engine
        self._lock = threading.Lock()

    def _transaction[T](self, work: collections.abc.Callable[[Connection], T]) -> T:
        """Run ``work`` in one ``BEGIN IMMEDIATE`` transaction under the store's lock (database I/O).

        Raises:
            identification.StoreError: ``work``'s own, or ``STORE_FAILED`` for a database failure. A
                database error renders its bound parameters (packed prints among them), so the
                ``STORE_FAILED`` error names only the failure's type and carries no cause or context.
        """

        try:
            with self._lock, self._engine.connect() as connection:
                # pysqlite's legacy transaction mode emits no BEGIN of its own once one is open, so
                # this statement is the transaction; IMMEDIATE takes the write lock up front.
                _ = connection.exec_driver_sql("BEGIN IMMEDIATE")
                try:
                    result = work(connection)
                    _ = connection.exec_driver_sql("COMMIT")
                except BaseException:
                    # SQLite may already have rolled back (a full disk); a ROLLBACK would then mask the error.
                    driver = typing.cast(sqlite3.Connection, connection.connection.driver_connection)
                    if driver.in_transaction:
                        _ = connection.exec_driver_sql("ROLLBACK")
                    raise
                return result
        except (sqlalchemy.exc.SQLAlchemyError, sqlite3.Error) as error:
            failure = type(error).__name__
        # Raised outside the handler, so the parameter-bearing error is neither cause nor context.
        raise identification.StoreError(
            identification.FailureKind.STORE_FAILED, f"the identity store failed ({failure})"
        )

    @typing.override
    def identify(
        self,
        kind: identification.Kind,
        prints: tuple[identification.Print, ...],
        policy: identification.MatchPolicy,
        *,
        at: datetime.datetime,
    ) -> tuple[identification.Recognition, ...]:
        stamp = memory.graph.utc_text(at)

        def work(connection: Connection) -> tuple[identification.Recognition, ...]:
            recognitions: list[identification.Recognition] = []
            for print in prints:
                table = _space(connection, print)
                # ``decide`` reads only the best neighbor and the runner-up.
                nearest = connection.exec_driver_sql(
                    f"SELECT seq, distance FROM {table} WHERE embedding MATCH ? AND k = 2 AND kind = ? "
                    "ORDER BY distance",
                    (memory.index.pack(print.vector), kind),
                ).all()
                distances = {int(seq): float(distance) for seq, distance in nearest}
                rows = connection.execute(
                    sqlalchemy.select(
                        _IDENTITIES.c.seq, _IDENTITIES.c.id, _IDENTITIES.c.own, _IDENTITIES.c.seen_count
                    ).where(_IDENTITIES.c.seq.in_(list(distances)), _IDENTITIES.c.model == print.model)
                ).all()
                by_id = {str(id): (int(seq), int(seen_count)) for seq, id, _, seen_count in rows}
                neighbors = [
                    identification.Neighbor(
                        id=str(id), similarity=max(-1.0, min(1.0, 1.0 - distances[int(seq)])), own=bool(own)
                    )
                    for seq, id, own, _ in rows
                ]
                decision = identification.decide(neighbors, policy)
                if isinstance(decision, identification.Mint):
                    minted = uuid.uuid4().hex
                    result = connection.execute(
                        sqlalchemy.insert(_IDENTITIES).values(
                            kind=kind,
                            id=minted,
                            model=print.model,
                            own=0,
                            linked=0,
                            created_at=stamp,
                            seen_at=stamp,
                            seen_count=1,
                        )
                    )
                    seq = int(typing.cast(tuple[int, ...], result.inserted_primary_key)[0])
                    _ = connection.exec_driver_sql(
                        f"INSERT INTO {table} (seq, embedding, kind) VALUES (?, ?, ?)",
                        (seq, memory.index.pack(print.vector), kind),
                    )
                    recognitions.append(
                        identification.Recognition(
                            id=minted, recognized=False, similarity=decision.similarity, own=False
                        )
                    )
                    continue
                seq, seen_count = by_id[decision.id]
                values: dict[str, object] = {"seen_at": stamp}
                if isinstance(decision, identification.Refine):
                    weight = min(seen_count, policy.refine_weight_cap)
                    refined = self._refined(_stored(connection, table, seq), weight, print.vector)
                    _ = connection.exec_driver_sql(
                        f"UPDATE {table} SET embedding = ? WHERE seq = ?", (memory.index.pack(refined), seq)
                    )
                    values["seen_count"] = min(seen_count + 1, policy.refine_weight_cap)
                _ = connection.execute(sqlalchemy.update(_IDENTITIES).where(_IDENTITIES.c.seq == seq).values(**values))
                recognitions.append(
                    identification.Recognition(
                        id=decision.id, recognized=True, similarity=decision.similarity, own=decision.own
                    )
                )
            return tuple(recognitions)

        return self._transaction(work)

    @typing.override
    def enroll(
        self,
        kind: identification.Kind,
        id: identification.Id | None,
        print: identification.Print,
        own: bool,
        policy: identification.MatchPolicy,
        *,
        at: datetime.datetime,
    ) -> identification.Recognition:
        if id is None and not own:
            raise identification.StoreError(
                identification.FailureKind.INVALID_INPUT, "an enrollment without an id must be the bot's own"
            )
        stamp = memory.graph.utc_text(at)

        def work(connection: Connection) -> identification.Recognition:
            table = _space(connection, print)
            enrolled = id
            minted_similarity: float | None = None
            if enrolled is None:
                # Own ids are few, so each is compared exactly; a k-nearest query could miss them all
                # behind closer ids that are not own.
                own_rows = connection.execute(
                    sqlalchemy.select(_IDENTITIES.c.seq, _IDENTITIES.c.id).where(
                        _IDENTITIES.c.kind == kind, _IDENTITIES.c.model == print.model, _IDENTITIES.c.own == 1
                    )
                ).all()
                decision = identification.decide(
                    [
                        identification.Neighbor(
                            id=str(own_id),
                            similarity=self._similarity(_stored(connection, table, int(seq)), print.vector),
                            own=True,
                        )
                        for seq, own_id in own_rows
                    ],
                    policy,
                )
                if isinstance(decision, identification.Mint):
                    enrolled, minted_similarity = uuid.uuid4().hex, decision.similarity
                else:
                    enrolled = decision.id
            row = connection.execute(
                sqlalchemy.select(
                    _IDENTITIES.c.seq, _IDENTITIES.c.model, _IDENTITIES.c.own, _IDENTITIES.c.seen_count
                ).where(_IDENTITIES.c.kind == kind, _IDENTITIES.c.id == enrolled)
            ).first()
            if row is None:
                result = connection.execute(
                    sqlalchemy.insert(_IDENTITIES).values(
                        kind=kind,
                        id=enrolled,
                        model=print.model,
                        own=int(own),
                        linked=0,
                        created_at=stamp,
                        seen_at=stamp,
                        seen_count=1,
                    )
                )
                seq = int(typing.cast(tuple[int, ...], result.inserted_primary_key)[0])
                _ = connection.exec_driver_sql(
                    f"INSERT INTO {table} (seq, embedding, kind) VALUES (?, ?, ?)",
                    (seq, memory.index.pack(print.vector), kind),
                )
                return identification.Recognition(id=enrolled, recognized=False, similarity=minted_similarity, own=own)
            seq, model, stored_own, seen_count = int(row[0]), str(row[1]), bool(row[2]), int(row[3])
            if model != print.model:
                raise identification.StoreError(
                    identification.FailureKind.INVALID_INPUT, "the print's model differs from the id's model"
                )
            stored = _stored(connection, table, seq)
            similarity = self._similarity(stored, print.vector)
            refined = self._refined(stored, min(seen_count, policy.refine_weight_cap), print.vector)
            _ = connection.exec_driver_sql(
                f"UPDATE {table} SET embedding = ? WHERE seq = ?", (memory.index.pack(refined), seq)
            )
            sticky = stored_own or own
            _ = connection.execute(
                sqlalchemy.update(_IDENTITIES)
                .where(_IDENTITIES.c.seq == seq)
                .values(own=int(sticky), seen_at=stamp, seen_count=min(seen_count + 1, policy.refine_weight_cap))
            )
            return identification.Recognition(id=enrolled, recognized=True, similarity=similarity, own=sticky)

        return self._transaction(work)

    @typing.override
    def forget(self, kind: identification.Kind, id: identification.Id) -> bool:
        def work(connection: Connection) -> bool:
            rows = connection.execute(
                sqlalchemy.select(_IDENTITIES.c.seq, _IDENTITIES.c.model).where(
                    _IDENTITIES.c.kind == kind, _IDENTITIES.c.id == id
                )
            ).all()
            return _delete(connection, rows) > 0

        return self._transaction(work)

    @typing.override
    def prune(self, kind: identification.Kind, *, unseen_before: datetime.datetime, max_ids: int) -> int:
        cutoff = memory.graph.utc_text(unseen_before)
        evictable = sqlalchemy.select(_IDENTITIES.c.seq, _IDENTITIES.c.model).where(
            _IDENTITIES.c.kind == kind, _IDENTITIES.c.own == 0, _IDENTITIES.c.linked == 0
        )

        def work(connection: Connection) -> int:
            removed = _delete(connection, connection.execute(evictable.where(_IDENTITIES.c.seen_at < cutoff)).all())
            remaining = connection.execute(
                sqlalchemy.select(sqlalchemy.func.count()).select_from(_IDENTITIES).where(_IDENTITIES.c.kind == kind)
            ).scalar_one()
            excess = int(remaining) - max_ids
            if excess > 0:
                oldest = evictable.order_by(_IDENTITIES.c.seen_at, _IDENTITIES.c.seq).limit(excess)
                removed += _delete(connection, connection.execute(oldest).all())
            return removed

        return self._transaction(work)


__all__ = ["IdentityStore", "table_name"]
