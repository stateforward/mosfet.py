import hsm


def require_model(model: hsm.Model | None) -> hsm.Model:
    assert model is not None
    return model


def greeting_behavior_source() -> str:
    return """
input_event = hsm.event(
    name = "bot.behavior.answer_greeting.input",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    description = "Greeting text recognized by cognition or conversation.",
    examples = [{"text": "hello"}],
)

output_event = hsm.event(
    name = "bot.behavior.answer_greeting.output",
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    description = "Behavior response that can be dispatched without deliberation.",
    examples = [{"text": "HELLO"}],
)

triggers = ["conversation.greeting.recognized"]
description = "Answer a recurring greeting without asking reasoning to choose every step."

def has_text(event):
    return "text" in event["data"]

def emit_uppercase(event):
    hsm.dispatch(output_event, {"text": event["data"]["text"].upper()})

behavior = hsm.define(
    "AnswerGreeting",
    hsm.initial(hsm.target("/AnswerGreeting/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("has_text"),
            hsm.effect("emit_uppercase"),
        ),
    ),
)
""".strip()


def strict_greeting_behavior_source() -> str:
    return greeting_behavior_source().replace(
        '"required": ["text"],',
        '"required": ["text"],\n        "additionalProperties": False,',
    )
