"""``VecIndex`` agrees with core ``ExactIndex`` and keeps its rows inside Memory's transaction."""

from __future__ import annotations

import datetime
import math
import pathlib
import typing

import pytest
import sqlalchemy.event
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mosfet.abilities import embedding, identification, memory
from mosfet.providers import sqlite_memory
from mosfet.providers.sqlite_memory import index as provider_index

_SPACE = embedding.Space(model="test/embedder", dimensions=2)


def _unit(angle: float) -> tuple[float, float]:
    return (math.cos(angle), math.sin(angle))


def _seed(engine: Engine, vec: sqlite_memory.VecIndex, rows: list[tuple[str, float]]) -> list[int]:
    seqs: list[int] = []
    with engine.begin() as connection:
        vec.ensure(connection, _SPACE)
        for position, (kind, angle) in enumerate(rows):
            result = connection.execute(
                memory.graph.node_insert_clause(
                    node_id=f"node-{position}",
                    kind=memory.NodeKind(kind),
                    text=kind,
                    content={},
                    at="2026-10-14T00:00:00.000000+00:00",
                )
            )
            (seq,) = typing.cast(tuple[int], result.inserted_primary_key)
            _ = connection.execute(memory.graph.vector_upsert_clause(seq=seq, space=_SPACE, values=_unit(angle)))
            vec.add(connection, _SPACE, seq, _unit(angle))
            seqs.append(seq)
    return seqs


def test_vec_index_agrees_with_the_exact_index() -> None:
    engine = sqlite_memory.open_engine()
    vec = sqlite_memory.VecIndex()
    _ = _seed(engine, vec, [("episode", 0.0), ("episode", 0.7), ("concept", 0.2), ("episode", 1.5), ("person", 0.0)])
    kinds = (memory.NodeKind.EPISODE, memory.NodeKind.CONCEPT)
    with engine.begin() as connection:
        approximate = vec.nearest(connection, _SPACE, _unit(0.1), k=3, kinds=kinds)
        exact = memory.ExactIndex().nearest(connection, _SPACE, _unit(0.1), k=3, kinds=kinds)
    engine.dispose()
    assert [neighbor.seq for neighbor in approximate] == [neighbor.seq for neighbor in exact]
    for left, right in zip(approximate, exact, strict=True):
        assert math.isclose(left.distance, right.distance, abs_tol=1e-5)


def test_vec_index_filters_by_kind_and_unknown_spaces_find_nothing() -> None:
    engine = sqlite_memory.open_engine()
    vec = sqlite_memory.VecIndex()
    episode, person = _seed(engine, vec, [("episode", 0.0), ("person", 0.0)])
    with engine.begin() as connection:
        found = vec.nearest(connection, _SPACE, _unit(0.0), k=5, kinds=(memory.NodeKind.EPISODE,))
        elsewhere = vec.nearest(
            connection,
            embedding.Space(model="never/used", dimensions=2),
            _unit(0.0),
            k=5,
            kinds=(memory.NodeKind.EPISODE,),
        )
    engine.dispose()
    assert [neighbor.seq for neighbor in found] == [episode]
    assert person not in [neighbor.seq for neighbor in found]
    assert elsewhere == ()


def test_a_removal_in_a_rolled_back_transaction_keeps_the_row() -> None:
    engine = sqlite_memory.open_engine()
    vec = sqlite_memory.VecIndex()
    (seq,) = _seed(engine, vec, [("episode", 0.0)])
    with pytest.raises(RuntimeError, match="roll back"):
        with engine.begin() as connection:
            vec.remove(connection, [seq])
            raise RuntimeError("roll back")
    with engine.begin() as connection:
        found = vec.nearest(connection, _SPACE, _unit(0.0), k=1, kinds=(memory.NodeKind.EPISODE,))
        vec.remove(connection, [seq])
    with engine.begin() as connection:
        gone = vec.nearest(connection, _SPACE, _unit(0.0), k=1, kinds=(memory.NodeKind.EPISODE,))
        count = connection.execute(text(f"SELECT count(*) FROM {provider_index.table_name(_SPACE)}")).scalar()
    engine.dispose()
    assert [neighbor.seq for neighbor in found] == [seq]
    assert gone == ()
    assert count == 0


def test_removal_touches_only_memory_vec0_tables(tmp_path: pathlib.Path) -> None:
    """``remove`` deletes from ``bot_memory_vec_<dims>_<hash>`` vec0 tables only: never identity tables,
    never a vec0 shadow table, and never another vec0 table that merely shares the memory prefix."""

    engine = sqlite_memory.open_engine(database_path=tmp_path / "memory.sqlite3")
    vec = sqlite_memory.VecIndex()
    (seq,) = _seed(engine, vec, [("episode", 0.0)])
    identities = sqlite_memory.IdentityStore(engine=engine)
    _ = identities.identify(
        "voice",
        (identification.Print(vector=_unit(0.0), model="test/embedder"),),
        identification.MatchPolicy(threshold=0.5, margin=0.05, refine_threshold=0.65),
        at=datetime.datetime(2026, 10, 15, tzinfo=datetime.UTC),
        mint=(True,),
    )
    with engine.begin() as connection:
        _ = connection.exec_driver_sql(
            "CREATE VIRTUAL TABLE bot_memory_vec_scratch USING vec0("
            "node_seq INTEGER PRIMARY KEY, embedding float[2] distance_metric=cosine, kind TEXT)"
        )
        _ = connection.exec_driver_sql(
            "INSERT INTO bot_memory_vec_scratch (node_seq, embedding, kind) VALUES (?, ?, 'episode')",
            (seq, memory.index.pack(_unit(0.0))),
        )
        statements: list[str] = []
        sqlalchemy.event.listen(connection, "before_cursor_execute", lambda *args: statements.append(str(args[2])))
        vec.remove(connection, [seq])
        scratch = connection.execute(text("SELECT count(*) FROM bot_memory_vec_scratch")).scalar()
        prints = connection.execute(text("SELECT count(*) FROM bot_identity")).scalar()
    engine.dispose()
    deletes = [statement for statement in statements if statement.startswith("DELETE")]
    assert deletes == [f"DELETE FROM {provider_index.table_name(_SPACE)} WHERE node_seq IN (?)"]
    assert scratch == 1
    assert prints == 1


def test_table_names_come_only_from_the_space_shape_and_a_hash() -> None:
    hostile = embedding.Space(model="x'); DROP TABLE bot_memory_node; --", dimensions=768)
    name = provider_index.table_name(hostile)
    assert name == provider_index.table_name(hostile)
    assert name.startswith("bot_memory_vec_768_")
    assert name.replace("_", "").isalnum()


def test_memory_remembers_and_searches_through_the_vec_index() -> None:
    import asyncio
    import datetime
    import uuid

    import hsm

    from mosfet.abilities import ability
    from mosfet.abilities.language import text as language
    from mosfet.define import define
    from mosfet.environment import Environment
    from mosfet.protocols import attachment
    from mosfet.start import started

    class Embedder(embedding.Embedder):
        def __init__(self) -> None:
            self.capacity = embedding.Capacity(space=_SPACE, max_items=8, max_input_tokens=2048)

        async def embed(self, input: embedding.InputData, *, ctx: hsm.Context) -> embedding.OutputData:
            del ctx
            vectors = []
            for item in input.items:
                words = " ".join(part.text for part in item.parts if isinstance(part, language.TextPart))
                vectors.append(embedding.Vector(values=_unit(0.0 if "noon" in words else 1.4)))
            return embedding.OutputData(space=_SPACE, role=input.role, vectors=tuple(vectors))

    class Owner(hsm.Instance):
        model = define("VecIndexTestOwner", hsm.initial(hsm.target("idle")), hsm.state("idle"))

    async def run() -> list[str]:
        ctx = Environment()
        owner = Owner()
        _ = await started(ctx, owner, Owner.model)
        store = memory.Memory(engine=sqlite_memory.open_engine(), index=sqlite_memory.VecIndex(), embedder=Embedder())
        _ = await store.attach(ctx, attachment.AttachEvent.with_data(attachment.AttachData(actor=owner)))

        async def ask(request: hsm.Event[object], data: object, reply: hsm.Event[object]) -> hsm.Event[object]:
            return await ability.run_terminal_operation(
                ctx,
                child=store,
                request=request.with_data_and_id(data, uuid.uuid4().hex),
                terminals=(reply, memory.FailedEvent),
                timeout=datetime.timedelta(seconds=5),
            )

        for words in ("lunch at noon", "weather report"):
            episode = memory.EpisodeData(
                turn_id=words, parts=(language.TextPart(text=words),), outcome=memory.Outcome.COMPLETED
            )
            _ = await ask(memory.RememberEvent, episode, memory.RememberedEvent)
        found = await ask(
            memory.SearchEvent,
            memory.SearchData(cue=(language.TextPart(text="noon"),), nearest=1, recent=0, linked=0),
            memory.FoundEvent,
        )
        assert isinstance(found.data, memory.FoundData)
        return [candidate.node.text for candidate in found.data.candidates]

    assert asyncio.run(run()) == ["lunch at noon"]
