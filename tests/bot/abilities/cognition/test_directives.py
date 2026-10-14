from mosfet.abilities import memory
from mosfet.abilities.cognition import directives
from mosfet.abilities.cognition import episodes

from sqlalchemy import select
from sqlalchemy.engine import Engine

from tests.bot.abilities.memory.memory_fixtures import execute_on


def _rows(store: Engine, query_tag: str) -> tuple[memory.Row, ...]:
    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == query_tag)
    output = execute_on(store, memory.InputData(statements=memory.compile_statements(clause)))
    return output.results[0].rows


def test_a_directive_round_trips_as_an_instruction_row_on_the_existing_schema() -> None:
    store = memory.store.open_sqlite_engine()

    _ = execute_on(
        store,
        directives.directive_insert_input(
            directives.Directive(text="Call Bob when you get a chance."),
            context_ref="phone",
        ),
    )
    recalled = directives.directives_from_output(
        execute_on(store, directives.directive_select_input(context_ref="phone"))
    )

    assert [directive.text for directive in recalled] == ["Call Bob when you get a chance."]
    assert recalled[0].kind is memory.MemoryClassificationKind.INSTRUCTION

    row = _rows(store, directives.DIRECTIVE_QUERY)[0].as_mapping()
    assert row["kind"] == "instruction"
    assert row["query_tags"] == "standing_directive"
    assert row["content_format"] == "application/json"
    assert row["context_ref"] == "phone"


def test_a_directive_carries_no_device_deadline_priority_or_done_flag() -> None:
    """What is absent is the design.

    "Call Bob" is not about the phone until the bot decides the phone is how you reach Bob,
    and nothing marks a directive done — a real agent knows it already called because it
    remembers calling, and episodes carry that. A completion flag here would make topology
    track goal state.
    """

    assert set(directives.Directive.model_fields) == {"text", "kind"}


def test_directive_recall_is_scoped_to_directives_and_survives_undecodable_rows() -> None:
    store = memory.store.open_sqlite_engine()
    table = memory.memory_table

    _ = execute_on(
        store,
        episodes.episode_insert_input(
            episodes.CognitiveEpisode(focus="phone", stimulus_name="bot.input", output=()),
            context_ref="phone",
        ),
    )
    _ = execute_on(
        store, directives.directive_insert_input(directives.Directive(text="Water the plants."), context_ref=None)
    )
    _ = execute_on(
        store,
        memory.InputData(
            statements=memory.compile_statements(
                table.insert().values(
                    memory_id="not-a-directive",
                    scope="short_term",
                    content="{not json}",
                    content_format="application/json",
                    query_tags=directives.DIRECTIVE_QUERY,
                )
            )
        ),
    )

    recalled = directives.directives_from_output(execute_on(store, directives.directive_select_input()))

    assert [directive.text for directive in recalled] == ["Water the plants."]


def test_a_bot_with_nothing_in_memory_recalls_no_directives() -> None:
    store = memory.store.open_sqlite_engine()

    recalled = directives.directives_from_output(
        execute_on(store, directives.directive_select_input(context_ref="phone"))
    )

    assert recalled == ()


def test_directive_recall_returns_the_most_recent_window_oldest_first() -> None:
    """A bot told many things recalls what it was told lately, in the order it was told.

    Rows written within the same second tie on ``created_at`` and fall back to ``memory_id``, which
    these rows carry in insertion order.
    """

    store = memory.store.open_sqlite_engine()
    for index in range(60):
        _ = execute_on(
            store,
            directives.directive_insert_input(
                directives.Directive(text=f"Directive {index:02d}."),
                context_ref="phone",
                memory_id=f"directive-{index:02d}",
            ),
        )

    recalled = directives.directives_from_output(
        execute_on(store, directives.directive_select_input(context_ref="phone"))
    )

    assert [directive.text for directive in recalled] == [f"Directive {index:02d}." for index in range(10, 60)]
