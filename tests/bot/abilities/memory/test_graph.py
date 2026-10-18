"""The episodic graph's Core builders compile to parameterized SQL over schema identifiers only."""

import pytest
import sqlalchemy
import sqlalchemy.exc
from sqlalchemy.dialects import sqlite
from sqlalchemy.sql import ClauseElement

from mosfet.abilities import embedding, memory

_SPACE = embedding.Space(model="evil'); DROP TABLE bot_memory_node; --", dimensions=2)
_CALLER_TEXT = "'); DROP TABLE bot_memory_node; --"


def _sql(clause: ClauseElement) -> tuple[str, dict[str, object]]:
    compiled = clause.compile(dialect=sqlite.dialect())
    params: dict[str, object] = {str(key): value for key, value in (compiled.params or {}).items()}
    return str(compiled), params


def test_builders_bind_caller_text_as_parameters_never_identifiers() -> None:
    clauses: list[ClauseElement] = [
        memory.graph.node_insert_clause(
            node_id=_CALLER_TEXT, kind=memory.NodeKind.CONCEPT, text=_CALLER_TEXT, content={"x": _CALLER_TEXT}, at="t"
        ),
        memory.graph.person_insert_clause(person_id=_CALLER_TEXT, at="t", name=_CALLER_TEXT),
        memory.graph.person_ref_upsert_clause(ref=_CALLER_TEXT, person=1, origin="model", at="t"),
        memory.graph.persons_select_clause([_CALLER_TEXT]),
        memory.graph.edge_upsert_clause(source=1, target=2, relation=_CALLER_TEXT, weight=1.0, origin="model", at="t"),
        memory.graph.vector_upsert_clause(seq=1, space=_SPACE, values=(1.0, 0.0)),
        memory.graph.backfill_clause(space=_SPACE, pass_id=_CALLER_TEXT, limit=4),
    ]
    for clause in clauses:
        statement, params = _sql(clause)
        assert "DROP TABLE" not in statement
        assert any(_CALLER_TEXT in str(value) or _SPACE.model == value for value in params.values())


def test_forgetting_deletes_every_dependent_row() -> None:
    tables = [_sql(clause)[0].split()[2] for clause in memory.graph.forget_clauses([1, 2])]
    assert tables == [
        "bot_memory_edge",
        "bot_memory_part",
        "bot_memory_vector",
        "bot_memory_index_attempt",
        "bot_memory_node",
    ]


def test_stored_times_are_fixed_width_utc() -> None:
    import datetime

    early = memory.graph.utc_text(datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC))
    late = memory.graph.utc_text(datetime.datetime(2026, 1, 1, 0, 0, 0, 1, tzinfo=datetime.UTC))
    assert len(early) == len(late)
    assert early < late
    assert memory.graph.parse_utc(late) == datetime.datetime(2026, 1, 1, 0, 0, 0, 1, tzinfo=datetime.UTC)


def test_a_ref_belongs_to_exactly_one_person() -> None:
    """The ref is the link table's primary key: a second person can never take it by insert."""

    engine = memory.store.open_sqlite_engine()
    try:
        with engine.begin() as connection:
            _ = connection.execute(
                sqlalchemy.insert(memory.schema.person_ref_table).values(
                    ref="voice:a", person_seq=1, origin="stamped", linked_at="t"
                )
            )
        with pytest.raises(sqlalchemy.exc.IntegrityError), engine.begin() as connection:
            _ = connection.execute(
                sqlalchemy.insert(memory.schema.person_ref_table).values(
                    ref="voice:a", person_seq=2, origin="model", linked_at="t"
                )
            )
    finally:
        engine.dispose()


def test_a_search_resolves_who_is_present_in_its_own_transaction() -> None:
    """A ref never seen before is somebody already: a provisional person, unknown until named or declared."""

    import datetime

    import mosfet

    engine = memory.store.open_sqlite_engine()
    try:
        stranger = mosfet.Participant(ref="voice:new", role=mosfet.ParticipantRole.SOURCE)
        mouth = mosfet.Participant(ref="voice:mouth", role=mosfet.ParticipantRole.TARGET, own=True)
        found, created, conflicts = memory.graph.search(
            engine,
            memory.ExactIndex(),
            memory.SearchData(participants=(stranger, mouth)),
            space=None,
            cue=None,
            at=datetime.datetime(2026, 10, 18, tzinfo=datetime.UTC),
        )
        again, created_again, _ = memory.graph.search(
            engine,
            memory.ExactIndex(),
            memory.SearchData(participants=(stranger,)),
            space=None,
            cue=None,
            at=datetime.datetime(2026, 10, 18, tzinfo=datetime.UTC),
        )
    finally:
        engine.dispose()

    assert [(present.ref, present.part, present.known) for present in found.people] == [
        ("voice:new", mosfet.ParticipantRole.SOURCE, False),
        ("voice:mouth", mosfet.ParticipantRole.TARGET, True),
    ]
    assert found.people[1].person == memory.SELF
    assert found.you == memory.Person(id=memory.SELF, refs=("voice:mouth",))
    assert (created, conflicts) == (2, 0)
    assert again.people[0].person == found.people[0].person and created_again == 0
