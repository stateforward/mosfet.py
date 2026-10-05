from mosfet import behavior

import pytest

from tests.bot.behavior.support import greeting_behavior_source
from tests.hsm_instance_state import behavior_spec
from tests.type_helpers import object_dict


def test_compile_preserves_starlark_event_contracts() -> None:
    compiled = behavior.build(greeting_behavior_source())
    spec = behavior_spec(compiled)

    assert isinstance(compiled, behavior.Behavior)
    assert compiled.__class__.__module__ == "mosfet.behavior"
    assert spec.name == "AnswerGreeting"
    assert spec.triggers == ("conversation.greeting.recognized",)
    assert compiled.input_event.name == "bot.behavior.answer_greeting.input"
    assert compiled.output_event.name == "bot.behavior.answer_greeting.output"
    input_schema = object_dict(compiled.input_event.schema)
    output_schema = object_dict(compiled.output_event.schema)
    assert input_schema["description"] == "Greeting text recognized by cognition or conversation."
    assert output_schema["examples"] == [{"text": "HELLO"}]


def test_compile_rejects_source_event_name_collisions() -> None:
    source = greeting_behavior_source().replace(
        'name = "bot.behavior.answer_greeting.output"',
        'name = "bot.behavior.answer_greeting.input"',
    )

    with pytest.raises(ValueError, match="event names"):
        _ = behavior.build(source)


def test_compile_rejects_generated_event_name_collisions() -> None:
    source = greeting_behavior_source().replace(
        'name = "bot.behavior.answer_greeting.output"',
        'name = "bot.behavior.answer_greeting.failed"',
    )

    with pytest.raises(ValueError, match="generated behavior event names"):
        _ = behavior.build(source)


def test_compile_rejects_internal_generated_event_name_collisions() -> None:
    source = (
        greeting_behavior_source()
        .replace(
            "behavior = hsm.define(",
            """
internal_failure = hsm.event(name = "bot.behavior.answer_greeting.failed")

behavior = hsm.define(""",
        )
        .replace(
            "hsm.on(input_event),",
            "hsm.on(input_event, internal_failure),",
            1,
        )
    )

    with pytest.raises(ValueError, match="generated behavior event names"):
        _ = behavior.build(source)


def test_compile_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _ = behavior.build("   ")


def test_compile_rejects_routine_events_that_collide_with_generated_tick_events() -> None:
    from tests.bot.behavior.support import routine_source

    source = routine_source(schedule="hsm.every(seconds = 300)").replace(
        'name = "bot.behavior.morning_briefing.output"',
        'name = "bot.behavior.morning_briefing.tick.0"',
    )

    with pytest.raises(ValueError, match="generated behavior event names"):
        _ = behavior.build(source)


def test_compile_routine_generates_one_tick_event_per_schedule() -> None:
    from tests.bot.behavior.support import routine_source

    source = routine_source(schedule="hsm.every(seconds = 300)").replace(
        'hsm.transition(hsm.every(seconds = 300), hsm.effect("emit")),',
        'hsm.transition(hsm.every(seconds = 300), hsm.effect("emit")),\n'
        + '        hsm.transition(hsm.at(time = "08:00", tz = "UTC"), hsm.effect("emit")),',
    )
    compiled = behavior.build(source)

    assert behavior.behavior.generated_event_names(behavior_spec(compiled)) == (
        "bot.behavior.morning_briefing.failed",
        "bot.behavior.morning_briefing.tick.0",
        "bot.behavior.morning_briefing.tick.1",
    )
