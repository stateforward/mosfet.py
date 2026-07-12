from bot import habit


def test_check_rejects_empty_source() -> None:
    result = habit.check("   ")
    assert not result.ok
    assert result.value is None
    assert result.report.errors
    assert result.report.errors[0].code == habit.diagnostic.E0001_EMPTY
    assert "E0001" in result.report.render()


def test_check_rejects_invalid_topology_with_build_code() -> None:
    # initial with a transition (invalid) — should surface as build diagnostic.
    program = """
input_event = hsm.event(name="bot.habit.bad.input", schema={"type": "object"})
output_event = hsm.event(name="bot.habit.bad.output", schema={"type": "object"})
def noop(event):
    hsm.dispatch(output_event, {})
habit = hsm.define(
    "BadTopo",
    hsm.initial(
        hsm.transition(hsm.on(input_event), hsm.effect("noop")),
    ),
)
"""
    result = habit.check(program)
    assert not result.ok
    assert any(item.code == habit.diagnostic.E0007_BUILD for item in result.report.errors)
    rendered = result.report.render()
    assert "error[E0007]" in rendered
    assert "help:" in rendered


def test_check_accepts_valid_greeting_source() -> None:
    from tests.bot.habit.support import greeting_habit_source

    result = habit.check(greeting_habit_source())
    assert result.ok
    assert result.value is not None
    assert result.value.name == "AnswerGreeting"
    assert result.report.ok


def test_start_raises_source_error_with_report() -> None:
    import pytest

    with pytest.raises(habit.SourceError) as error:
        _ = habit.start("   ")
    assert error.value.report is not None
    assert not error.value.report.ok
