from bot import habit

import pytest

from tests.bot.habit.support import greeting_habit_source


def test_habit_defines_autonomous_starlark_backed_behavior() -> None:
    source = greeting_habit_source()
    compiled = habit.Instance(
        name="AnswerGreeting",
        source=source,
        triggers=("conversation.greeting.recognized",),
        description="Respond to a familiar greeting without deliberation.",
        examples=("answer hello with a matching greeting",),
    )

    assert compiled.name == "AnswerGreeting"
    assert compiled.triggers == ("conversation.greeting.recognized",)
    assert compiled.description == "Respond to a familiar greeting without deliberation."
    assert compiled.examples == ("answer hello with a matching greeting",)
    assert compiled.source == source
    assert compiled.status == "ACTIVE"
    assert compiled.used_count == 0
    assert compiled.last_used_at is None
    assert compiled.failed_count == 0
    assert compiled.last_failed_at is None
    assert compiled.status_reason is None
    assert compiled.status_updated_at is None
    assert habit.Instance.__doc__ is not None
    assert "hsm.Instance" in habit.Instance.__doc__


def test_start_builds_executable_installable_habit() -> None:
    compiled = habit.start(greeting_habit_source())

    assert compiled.name == "AnswerGreeting"
    assert "hsm.define" in compiled.source
    assert compiled.triggers == ("conversation.greeting.recognized",)
    compiled = habit.build(compiled.source)
    assert compiled.input_event.name == "bot.habit.answer_greeting.input"


def test_start_rejects_name_mismatch() -> None:
    with pytest.raises(ValueError, match="must match starlark model name"):
        _ = habit.start(greeting_habit_source(), name="WrongName")


def test_start_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _ = habit.start("   ")
