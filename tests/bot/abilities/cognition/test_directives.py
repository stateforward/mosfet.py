from bot.abilities import memory
from bot.abilities.cognition import directives
from bot.abilities.cognition import episodes

from sqlalchemy import select


def _rows(store: memory.Memory, query_tag: str) -> tuple[memory.Row, ...]:
    table = memory.memory_table
    clause = select(table).where(table.c.query_tags == query_tag)
    output = store.execute(memory.InputData(statements=memory.compile_statements(clause)))
    return output.results[0].rows


def test_a_directive_round_trips_as_an_instruction_row_on_the_existing_schema() -> None:
    store = memory.Memory()

    _ = store.execute(
        directives.directive_insert_input(
            directives.Directive(text="Call Bob when you get a chance."),
            context_ref="phone",
        )
    )
    recalled = directives.directives_from_output(store.execute(directives.directive_select_input(context_ref="phone")))

    assert [directive.text for directive in recalled] == ["Call Bob when you get a chance."]
    assert recalled[0].kind is memory.MemoryClassificationKind.INSTRUCTION

    record = memory.MemoryRecord.from_row(_rows(store, directives.DIRECTIVE_QUERY)[0])
    assert record.kind == memory.MemoryClassificationKind.INSTRUCTION
    assert record.query_tags == "standing_directive"
    assert record.content_format == "application/json"
    assert record.context_ref == "phone"


def test_a_directive_carries_no_device_deadline_priority_or_done_flag() -> None:
    """What is absent is the design.

    "Call Bob" is not about the phone until judgment decides the phone is how you reach Bob,
    and nothing marks a directive done — a real agent knows it already called because it
    remembers calling, and episodes carry that. A completion flag here would make topology
    track goal state.
    """

    assert set(directives.Directive.model_fields) == {"text", "kind"}


def test_directive_recall_is_scoped_to_directives_and_survives_undecodable_rows() -> None:
    store = memory.Memory()
    table = memory.memory_table

    _ = store.execute(
        episodes.episode_insert_input(
            episodes.CognitiveEpisode(focus="phone", stimulus_name="bot.idle", output=()),
            context_ref="phone",
        )
    )
    _ = store.execute(
        directives.directive_insert_input(directives.Directive(text="Water the plants."), context_ref=None)
    )
    _ = store.execute(
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
        )
    )

    recalled = directives.directives_from_output(store.execute(directives.directive_select_input()))

    assert [directive.text for directive in recalled] == ["Water the plants."]


def test_a_bot_with_nothing_in_memory_recalls_no_directives() -> None:
    store = memory.Memory()

    recalled = directives.directives_from_output(store.execute(directives.directive_select_input(context_ref="phone")))

    assert recalled == ()
