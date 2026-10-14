"""The episodic graph's Core builders compile to parameterized SQL over schema identifiers only."""

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
        memory.graph.person_upsert_clause(ref=_CALLER_TEXT, at="t"),
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
