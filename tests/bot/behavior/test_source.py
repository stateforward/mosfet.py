from bot import behavior

import pytest

from tests.bot.behavior.support import greeting_behavior_source

def test_parse_source_uses_real_starlark_hsm_model_logic() -> None:
    source = """
def behavior_event(suffix):
    return hsm.event(
        name = "bot.behavior.answer_greeting." + suffix,
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

input_event = behavior_event("input")
output_event = behavior_event("output")

def emit_uppercase(event):
    hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})

behavior = hsm.define(
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

    spec = behavior.parse_source(source)

    assert spec.model["kind"] == "define"
    assert spec.model["name"] == "AnswerGreeting"
    assert spec.input_event.name == "bot.behavior.answer_greeting.input"

def test_normalize_source_preserves_strings_while_rewriting_hsm_namespace() -> None:
    source = 'message = "hsm.define is documentation text"\nbehavior = hsm.define("M")\n'

    normalized = behavior.source.normalize_source(source)

    assert '"hsm.define is documentation text"' in normalized
    assert "hsm.define" not in normalized.splitlines()[1]
    assert "define" in normalized.splitlines()[1]

def test_parse_source_rejects_old_ability_step_orchestration() -> None:
    source = """
input_event = hsm.event(name = "bot.behavior.answer_greeting.input")
output_event = hsm.event(name = "bot.behavior.answer_greeting.output")
behavior = behavior_program(
    kind = "behavior_program",
    name = "AnswerGreeting",
    input_event = input_event,
    output_event = output_event,
    steps = [ability_step(name = "uppercase_reply", ability = "uppercase")],
)
"""

    with pytest.raises(behavior.SourceError, match="ability_step"):
        _ = behavior.parse_source(source)

def test_parse_source_rejects_unapproved_starlark_calls() -> None:
    with pytest.raises(behavior.SourceError, match="open"):
        _ = behavior.parse_source('behavior = open("unsafe.star")')

def test_parse_source_rejects_unsupported_schema_keywords() -> None:
    source = greeting_behavior_source().replace(
        '"properties": {"text": {"type": "string"}},',
        '"properties": {"text": {"type": "string", "format": "email"}},',
        1,
    )

    with pytest.raises(behavior.SourceError, match="unsupported JSON schema keyword"):
        _ = behavior.parse_source(source)

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
    source = greeting_behavior_source().replace(old, new, 1)

    with pytest.raises(behavior.SourceError, match="Extra inputs|unexpected keyword"):
        _ = behavior.parse_source(source)

def test_parse_source_rejects_extra_hsm_builder_keywords() -> None:
    source = """
input_event = hsm.event(name = "bot.behavior.answer_greeting.input")
output_event = hsm.event(name = "bot.behavior.answer_greeting.output")
behavior = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state("idle", extra = "unexpected"),
)
"""

    with pytest.raises(behavior.SourceError, match="unexpected keyword"):
        _ = behavior.parse_source(source)

def test_parse_source_rejects_unknown_targets() -> None:
    source = greeting_behavior_source().replace(
        'hsm.target("/AnswerGreeting/idle")',
        'hsm.target("/AnswerGreeting/missing")',
        1,
    )

    with pytest.raises(behavior.SourceError, match="unknown states"):
        _ = behavior.parse_source(source)

def test_parse_source_rejects_duplicate_model_event_names() -> None:
    source = (
        greeting_behavior_source()
        .replace(
            "behavior = hsm.define(",
            """
duplicate_output = hsm.event(name = "bot.behavior.answer_greeting.output")

behavior = hsm.define(""",
        )
        .replace(
            "hsm.on(input_event),",
            "hsm.on(input_event, duplicate_output),",
            1,
        )
    )

    with pytest.raises(behavior.SourceError, match="declared more than once"):
        _ = behavior.parse_source(source)

def test_parse_source_rejects_string_event_references() -> None:
    source = greeting_behavior_source().replace(
        "hsm.on(input_event)",
        'hsm.on("bot.behavior.answer_greeting.input")',
        1,
    )

    with pytest.raises(behavior.SourceError, match="event spec"):
        _ = behavior.parse_source(source)

def test_behavior_program_spec_rejects_final_state_contents() -> None:
    event = {"name": "bot.behavior.answer_greeting.input", "schema": {"type": "object"}}

    with pytest.raises(ValueError, match="unexpected hsm element fields"):
        _ = behavior.Source.model_validate(
            {
                "kind": "behavior_program",
                "name": "AnswerGreeting",
                "input_event": event,
                "output_event": {"name": "bot.behavior.answer_greeting.output", "schema": {"type": "object"}},
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
    source = greeting_behavior_source().replace('hsm.effect("emit_uppercase")', "hsm.effect(emit_uppercase)", 1)

    with pytest.raises(behavior.SourceError, match="convert|callback"):
        _ = behavior.parse_source(source)
