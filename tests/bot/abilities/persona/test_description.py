"""Storing the description a bot writes of itself, on the memory schema every other self-fact uses."""

from __future__ import annotations

import datetime
import sqlite3
import typing

import pydantic
import pytest
from sqlalchemy.engine import Engine

from mosfet.abilities import memory
from mosfet.abilities import persona

from tests.bot.abilities.memory.memory_fixtures import execute_on


def open_store() -> tuple[Engine, sqlite3.Connection]:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    return memory.store.open_sqlite_engine(connection=connection), connection


def written(text: str, *, traits: tuple[str, ...] = ("curious",)) -> persona.Description:
    return persona.Description(text=text, reason="I noticed it about myself.", traits=traits)


def test_a_store_with_no_description_reads_as_none() -> None:
    """A bot that never wrote about itself has no description, and that is a normal state."""

    store, connection = open_store()
    try:
        recalled = execute_on(store, persona.description_select_input())
    finally:
        connection.close()

    assert persona.latest_description(recalled) is None


def test_a_description_is_a_self_fact_tagged_self_persona() -> None:
    store, connection = open_store()
    try:
        _ = execute_on(store, persona.description_insert_input(written("I like slow explanations.")))
        recalled = execute_on(store, persona.description_select_input())
        rows = [row.as_mapping() for row in recalled.results[0].rows]
    finally:
        connection.close()

    assert persona.latest_description(recalled) == written("I like slow explanations.")
    assert len(rows) == 1
    row = rows[0]
    assert row["query_tags"] == persona.PERSONA_QUERY == "self_persona"
    assert row["subject_ref"] == memory.SELF_SUBJECT
    assert row["kind"] == "fact"
    assert row["retention"] == "retain"
    assert row["content_format"] == "application/json"


def test_revisions_are_append_only_and_the_latest_is_current() -> None:
    """Two revisions in the same second are still ordered: the self-fact stamp keeps microseconds."""

    stamp = datetime.datetime(2026, 10, 16, 12, 0, 0, tzinfo=datetime.UTC)
    store, connection = open_store()
    try:
        _ = execute_on(store, persona.description_insert_input(written("First."), created_at=stamp))
        _ = execute_on(
            store,
            persona.description_insert_input(
                written("Second.", traits=("curious", "bold")),
                created_at=stamp + datetime.timedelta(microseconds=1),
            ),
        )
        recalled = execute_on(store, persona.description_select_input())
        history = [persona.Description.model_validate_json(content) for content in recalled.contents()]
    finally:
        connection.close()

    assert [description.text for description in history] == ["First.", "Second."]
    latest = persona.latest_description(recalled)
    assert latest is not None
    assert latest.text == "Second."
    assert latest.traits == ("curious", "bold"), "each revision records the traits it was written under"


def test_a_row_that_does_not_decode_is_skipped_not_raised() -> None:
    store, connection = open_store()
    try:
        _ = execute_on(store, persona.description_insert_input(written("Kept.")))
        _ = execute_on(
            store, memory.self_fact_insert_input(persona.PERSONA_QUERY, persona.Traits(traits=("not a description",)))
        )
        recalled = execute_on(store, persona.description_select_input())
    finally:
        connection.close()

    latest = persona.latest_description(recalled)
    assert latest is not None
    assert latest.text == "Kept."


def test_a_description_is_always_a_fact_with_canonical_fields_only() -> None:
    schema = persona.Description.model_json_schema()

    assert written("x").kind == "fact"
    assert schema["properties"]["kind"]["const"] == "fact"
    assert set(typing.cast(dict[str, object], schema["properties"])) == {"text", "reason", "traits", "kind"}
    assert "examples" not in schema["properties"]["text"], "an example description becomes the bot's description"


@pytest.mark.parametrize(
    ("field", "value"),
    [("text", ""), ("text", " "), ("text", "x" * 2001), ("reason", ""), ("reason", "x" * 501), ("traits", ())],
)
def test_a_description_is_bounded(field: str, value: object) -> None:
    payload: dict[str, object] = {"text": "I am.", "reason": "Because.", "traits": ("curious",)}
    payload[field] = value

    with pytest.raises(pydantic.ValidationError):
        _ = persona.Description.model_validate(payload)
