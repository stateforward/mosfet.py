from bot import habit

import pytest

from tests.bot.habit.support import greeting_habit_source

def test_parse_source_uses_real_starlark_hsm_model_logic() -> None:
    source = """
def habit_event(suffix):
    return hsm.event(
        name = "bot.habit.answer_greeting." + suffix,
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

input_event = habit_event("input")
output_event = habit_event("output")

def emit_uppercase(event):
    hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})

habit = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.effect("emit_uppercase"),
        ),
    ),
)
"""

    spec = habit.parse_source(source)

    assert spec.model["kind"] == "define"
    assert spec.model["name"] == "AnswerGreeting"
    assert spec.input_event.name == "bot.habit.answer_greeting.input"

def test_normalize_source_preserves_strings_while_rewriting_hsm_namespace() -> None:
    source = 'message = "hsm.define is documentation text"\nhabit = hsm.define("M")\n'

    normalized = habit.source.normalize_source(source)

    assert '"hsm.define is documentation text"' in normalized
    assert "hsm.define" not in normalized.splitlines()[1]
    assert "define" in normalized.splitlines()[1]

def test_parse_source_rejects_old_ability_step_orchestration() -> None:
    source = """
input_event = hsm.event(name = "bot.habit.answer_greeting.input")
output_event = hsm.event(name = "bot.habit.answer_greeting.output")
habit = habit_behavior(
    kind = "habit_behavior",
    name = "AnswerGreeting",
    input_event = input_event,
    output_event = output_event,
    steps = [ability_step(name = "uppercase_reply", ability = "uppercase")],
)
"""

    with pytest.raises(habit.SourceError, match="ability_step"):
        _ = habit.parse_source(source)

def test_parse_source_rejects_unapproved_starlark_calls() -> None:
    with pytest.raises(habit.SourceError, match="open"):
        _ = habit.parse_source('habit = open("unsafe.star")')

def test_parse_source_rejects_unsupported_schema_keywords() -> None:
    source = greeting_habit_source().replace(
        '"properties": {"text": {"type": "string"}},',
        '"properties": {"text": {"type": "string", "format": "email"}},',
        1,
    )

    with pytest.raises(habit.SourceError, match="unsupported JSON schema keyword"):
        _ = habit.parse_source(source)

@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            'examples = [{"text": "hello"}],',
            'examples = [{"text": "hello"}],\n    source = "unexpected",',
        ),
    ],
)
def test_parse_source_rejects_extra_starlark_builder_fields(old: str, new: str) -> None:
    source = greeting_habit_source().replace(old, new, 1)

    with pytest.raises(habit.SourceError, match="Extra inputs|unexpected keyword"):
        _ = habit.parse_source(source)

def test_parse_source_rejects_extra_hsm_builder_keywords() -> None:
    source = """
input_event = hsm.event(name = "bot.habit.answer_greeting.input")
output_event = hsm.event(name = "bot.habit.answer_greeting.output")
habit = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state("idle", extra = "unexpected"),
)
"""

    with pytest.raises(habit.SourceError, match="unexpected keyword"):
        _ = habit.parse_source(source)

def test_parse_source_rejects_unknown_targets() -> None:
    source = greeting_habit_source().replace(
        'hsm.target("/AnswerGreeting/idle")',
        'hsm.target("/AnswerGreeting/missing")',
        1,
    )

    with pytest.raises(habit.SourceError, match="unknown states"):
        _ = habit.parse_source(source)

def test_parse_source_rejects_duplicate_model_event_names() -> None:
    source = (
        greeting_habit_source()
        .replace(
            "habit = hsm.define(",
            """
duplicate_output = hsm.event(name = "bot.habit.answer_greeting.output")

habit = hsm.define(""",
        )
        .replace(
            "hsm.on(input_event),",
            "hsm.on(input_event, duplicate_output),",
            1,
        )
    )

    with pytest.raises(habit.SourceError, match="declared more than once"):
        _ = habit.parse_source(source)

def test_parse_source_rejects_string_event_references() -> None:
    source = greeting_habit_source().replace(
        "hsm.on(input_event)",
        'hsm.on("bot.habit.answer_greeting.input")',
        1,
    )

    with pytest.raises(habit.SourceError, match="event spec"):
        _ = habit.parse_source(source)

def test_habit_behavior_spec_rejects_final_state_contents() -> None:
    event = {"name": "bot.habit.answer_greeting.input", "schema": {"type": "object"}}

    with pytest.raises(ValueError, match="unexpected hsm element fields"):
        _ = habit.Source.model_validate(
            {
                "kind": "habit_behavior",
                "name": "AnswerGreeting",
                "input_event": event,
                "output_event": {"name": "bot.habit.answer_greeting.output", "schema": {"type": "object"}},
                "model": {
                    "kind": "define",
                    "name": "AnswerGreeting",
                    "elements": (
                        {"kind": "initial", "elements": ({"kind": "target", "path": "/AnswerGreeting/done"},)},
                        {
                            "kind": "final",
                            "name": "done",
                            "elements": ({"kind": "transition", "elements": ()},),
                        },
                    ),
                },
            }
        )

def test_parse_source_rejects_raw_function_callbacks() -> None:
    source = greeting_habit_source().replace('hsm.effect("emit_uppercase")', "hsm.effect(emit_uppercase)", 1)

    with pytest.raises(habit.SourceError, match="convert|callback"):
        _ = habit.parse_source(source)
