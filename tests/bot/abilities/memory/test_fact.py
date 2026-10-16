"""Facts a bot holds about itself: one row shape on ``bot_memory``, shared by every self-fact."""

from __future__ import annotations

import datetime
import sqlite3
import typing

import pydantic
from sqlalchemy import insert
from sqlalchemy.engine import Engine

from mosfet.abilities import memory

from tests.bot.abilities.memory.memory_fixtures import execute_on


class _Fact(pydantic.BaseModel):
    """A stand-in self-fact: any typed content the bot keeps about itself."""

    model_config: typing.ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(frozen=True, extra="forbid")

    text: str


def open_store() -> tuple[Engine, sqlite3.Connection]:
    """A migrated memory database read and written as SQL."""

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return memory.store.open_sqlite_engine(connection=connection), connection


def test_the_subject_of_a_self_fact_is_the_bot_itself() -> None:
    """Every fact the bot keeps about itself names the same subject."""

    assert memory.SELF_SUBJECT == "self"


def test_a_self_fact_is_a_retained_long_term_fact_about_the_bot() -> None:
    """A self-fact records what the bot is, so every self-fact row has the same shape."""

    stamp = datetime.datetime(2026, 10, 16, 12, 0, 0, 123_456, tzinfo=datetime.UTC)
    store, connection = open_store()
    try:
        _ = execute_on(
            store,
            memory.self_fact_insert_input("self_test", _Fact(text="kept"), created_at=stamp, memory_id="m-1"),
        )
        recalled = execute_on(store, memory.self_fact_select_input("self_test"))
    finally:
        connection.close()

    rows = [row.as_mapping() for row in recalled.results[0].rows]
    assert len(rows) == 1
    row = rows[0]
    assert row["memory_id"] == "m-1"
    assert row["subject_ref"] == memory.SELF_SUBJECT
    assert row["query_tags"] == "self_test"
    assert row["context_ref"] is None
    assert row["kind"] == "fact"
    assert row["scope"] == "long_term"
    assert row["sensitivity"] == "standard"
    assert row["retention"] == "retain"
    assert row["content_format"] == "application/json"
    assert row["created_at"] == "2026-10-16T12:00:00.123456+00:00"
    assert _Fact.model_validate_json(recalled.contents()[0]) == _Fact(text="kept")


def test_a_self_fact_is_stamped_at_microsecond_resolution_in_utc() -> None:
    """The column default has one-second resolution, so the stamp is always written explicitly.

    A stamp in another offset is written in UTC with fixed width, so text order stays
    chronological order; one with no explicit stamp is stamped now.
    """

    offset = datetime.timezone(datetime.timedelta(hours=2))
    store, connection = open_store()
    try:
        _ = execute_on(
            store,
            memory.self_fact_insert_input(
                "self_test",
                _Fact(text="zoned"),
                created_at=datetime.datetime(2026, 10, 16, 14, 0, 0, tzinfo=offset),
            ),
        )
        before = datetime.datetime.now(datetime.UTC)
        _ = execute_on(store, memory.self_fact_insert_input("self_test", _Fact(text="now")))
        after = datetime.datetime.now(datetime.UTC)
        recalled = execute_on(store, memory.self_fact_select_input("self_test"))
    finally:
        connection.close()

    zoned, stamped = (row.as_mapping()["created_at"] for row in recalled.results[0].rows)
    fixed_width = "2026-10-16T12:00:00.000000+00:00"
    assert zoned == fixed_width
    assert isinstance(stamped, str)
    assert len(stamped) == len(fixed_width)
    assert before <= datetime.datetime.fromisoformat(stamped) <= after


def test_self_facts_recall_only_their_own_query_and_subject() -> None:
    """Another kind of self-fact, or a fact about someone else, is never read as this one."""

    store, connection = open_store()
    try:
        _ = execute_on(store, memory.self_fact_insert_input("self_test", _Fact(text="mine")))
        _ = execute_on(store, memory.self_fact_insert_input("self_other", _Fact(text="other kind")))
        _ = execute_on(
            store,
            memory.InputData(
                statements=memory.compile_statements(
                    insert(memory.memory_table).values(
                        memory_id="someone",
                        subject_ref="person:ada",
                        kind="fact",
                        content='{"text": "about ada"}',
                        query_tags="self_test",
                    )
                )
            ),
        )
        recalled = execute_on(store, memory.self_fact_select_input("self_test"))
    finally:
        connection.close()

    assert [_Fact.model_validate_json(content).text for content in recalled.contents()] == ["mine"]


def test_self_fact_recall_reaches_the_most_recent_rows_oldest_first() -> None:
    """Self-facts are append-only; recall reads the latest ``limit`` of them in the order they happened."""

    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    store, connection = open_store()
    try:
        for index in range(6):
            _ = execute_on(
                store,
                memory.self_fact_insert_input(
                    "self_test",
                    _Fact(text=f"fact {index}"),
                    created_at=base + datetime.timedelta(microseconds=index),
                ),
            )
        recalled = execute_on(store, memory.self_fact_select_input("self_test", limit=3))
    finally:
        connection.close()

    assert [_Fact.model_validate_json(content).text for content in recalled.contents()] == [
        "fact 3",
        "fact 4",
        "fact 5",
    ]
