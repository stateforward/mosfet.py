"""``memory.ExactIndex``: exact nearest search over the canonical vectors of one space."""

import math
import typing

import sqlalchemy
from sqlalchemy.engine import Engine

from mosfet.abilities import embedding, memory

_SPACE = embedding.Space(model="test/embedder", dimensions=2)
_OTHER = embedding.Space(model="test/other", dimensions=2)


def _unit(angle: float) -> tuple[float, float]:
    return (math.cos(angle), math.sin(angle))


def _seed(engine: Engine, rows: list[tuple[str, embedding.Space, tuple[float, float]]]) -> list[int]:
    seqs: list[int] = []
    with engine.begin() as connection:
        for kind, space, values in rows:
            result = connection.execute(
                memory.graph.node_insert_clause(
                    node_id=f"node-{len(seqs)}",
                    kind=memory.NodeKind(kind),
                    text=kind,
                    content={},
                    at="2026-10-14T00:00:00.000000+00:00",
                )
            )
            (seq,) = typing.cast(tuple[int], result.inserted_primary_key)
            _ = connection.execute(memory.graph.vector_upsert_clause(seq=seq, space=space, values=values))
            seqs.append(seq)
    return seqs


def test_exact_index_returns_the_nearest_of_the_asked_kinds_in_one_space() -> None:
    engine = memory.store.open_sqlite_engine()
    near, far, concept, other_space = _seed(
        engine,
        [
            ("episode", _SPACE, _unit(0.0)),
            ("episode", _SPACE, _unit(1.5)),
            ("concept", _SPACE, _unit(0.1)),
            ("episode", _OTHER, _unit(0.0)),
        ],
    )
    index = memory.ExactIndex()
    with engine.begin() as connection:
        index.ensure(connection, _SPACE)
        both = index.nearest(
            connection, _SPACE, _unit(0.0), k=2, kinds=(memory.NodeKind.EPISODE, memory.NodeKind.CONCEPT)
        )
        episodes = index.nearest(connection, _SPACE, _unit(0.0), k=5, kinds=(memory.NodeKind.EPISODE,))
        none = index.nearest(connection, _SPACE, _unit(0.0), k=0, kinds=(memory.NodeKind.EPISODE,))
    assert [neighbor.seq for neighbor in both] == [near, concept]
    assert [neighbor.seq for neighbor in episodes] == [near, far]
    assert other_space not in [neighbor.seq for neighbor in episodes]
    assert math.isclose(episodes[0].distance, 0.0, abs_tol=1e-6)
    assert none == ()


def test_exact_index_on_an_empty_space_finds_nothing() -> None:
    engine = memory.store.open_sqlite_engine()
    with engine.begin() as connection:
        assert memory.ExactIndex().nearest(connection, _SPACE, _unit(0.0), k=3, kinds=(memory.NodeKind.EPISODE,)) == ()
        assert connection.execute(sqlalchemy.text("SELECT count(*) FROM bot_memory_vector")).scalar() == 0


def test_vectors_pack_as_little_endian_float32() -> None:
    packed = memory.index.pack((0.5, -1.0))
    assert packed == b"\x00\x00\x00?\x00\x00\x80\xbf"
    assert memory.index.unpack(packed) == (0.5, -1.0)
