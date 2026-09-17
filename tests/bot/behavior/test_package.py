from mosfet import behavior

import pytest

from tests.bot.behavior.support import greeting_behavior_source


def test_behavior_defines_autonomous_starlark_backed_behavior() -> None:
    source = greeting_behavior_source()
    compiled = behavior.Instance(
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
    assert behavior.Instance.__doc__ is not None
    assert "hsm.Instance" in behavior.Instance.__doc__


def test_start_builds_executable_installable_behavior() -> None:
    compiled = behavior.start(greeting_behavior_source())

    assert compiled.name == "AnswerGreeting"
    assert "hsm.define" in compiled.source
    assert compiled.triggers == ("conversation.greeting.recognized",)
    compiled = behavior.build(compiled.source)
    assert compiled.input_event.name == "bot.behavior.answer_greeting.input"


def test_start_rejects_name_mismatch() -> None:
    with pytest.raises(ValueError, match="must match starlark model name"):
        _ = behavior.start(greeting_behavior_source(), name="WrongName")


def test_start_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _ = behavior.start("   ")
