"""Storing the name a bot adopted, on the memory schema every other durable fact uses."""

from __future__ import annotations

import datetime
import sqlite3

from mosfet.abilities import identity
from mosfet.abilities import memory


def open_store() -> tuple[memory.Memory, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return memory.Memory(connection=connection), connection


def test_a_store_with_no_name_row_reads_as_nameless() -> None:
    """Having no name is a normal state of memory, not a missing row to be repaired."""

    store, connection = open_store()
    try:
        recalled = store.execute(identity.name_select_input())
    finally:
        connection.close()

    assert identity.names_from_output(recalled) == ()
    assert identity.adopted_name(recalled) is None


def test_an_adopted_name_is_a_fact_about_the_bot_itself() -> None:
    """A name is what the bot *is*, so it is stored as a fact whose subject is the bot.

    It is deliberately not a standing directive: a directive is something to do and stays
    outstanding until released, while a name is never completed or acted on.
    """

    store, connection = open_store()
    try:
        _ = store.execute(identity.name_insert_input(identity.Name(text="Bob")))
        recalled = store.execute(identity.name_select_input())
        rows = [row.as_mapping() for row in recalled.results[0].rows]
    finally:
        connection.close()

    assert identity.adopted_name(recalled) == identity.Name(text="Bob")
    assert len(rows) == 1
    row = rows[0]
    assert row["query_tags"] == identity.NAME_QUERY
    assert row["subject_ref"] == identity.NAME_SUBJECT
    assert row["kind"] == str(memory.MemoryClassificationKind.FACT)
    assert row["retention"] == "retain"
    assert row["sensitivity"] == "standard"
    assert row["scope"] == "long_term"
    assert row["content_format"] == "application/json"


def test_the_most_recently_adopted_name_is_the_one_the_bot_answers_to() -> None:
    """Adoption is append-only: a renamed bot still remembers having been called something else.

    ``created_at`` is written explicitly precisely so two adoptions made in the same second still
    have an order — the column default has one-second resolution.
    """

    first = datetime.datetime(2026, 7, 26, 12, 0, 0, 100_000, tzinfo=datetime.UTC)
    second = datetime.datetime(2026, 7, 26, 12, 0, 0, 200_000, tzinfo=datetime.UTC)

    store, connection = open_store()
    try:
        _ = store.execute(identity.name_insert_input(identity.Name(text="Bob"), created_at=first))
        _ = store.execute(identity.name_insert_input(identity.Name(text="Ada"), created_at=second))
        recalled = store.execute(identity.name_select_input())
    finally:
        connection.close()

    assert [name.text for name in identity.names_from_output(recalled)] == ["Bob", "Ada"]
    adopted = identity.adopted_name(recalled)
    assert adopted is not None and adopted.text == "Ada"


def test_a_name_is_not_recalled_as_a_standing_directive() -> None:
    """The two live on the same table and must never be read as each other.

    A bot that reads its own name as an instruction would treat being called Bob as something
    to carry out.
    """

    from mosfet.abilities.cognition import directives

    store, connection = open_store()
    try:
        _ = store.execute(identity.name_insert_input(identity.Name(text="Bob")))
        recalled_directives = store.execute(directives.directive_select_input())
        recalled_names = store.execute(identity.name_select_input())
    finally:
        connection.close()

    assert directives.directives_from_output(recalled_directives) == ()
    assert identity.adopted_name(recalled_names) == identity.Name(text="Bob")
