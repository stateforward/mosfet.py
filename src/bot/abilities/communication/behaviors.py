"""Cognition behaviors shipped with Communication.

Install-time inventory rows only. Listening does not import Communication;
Communication does not call Listening. Autonomy matches the speech product
trigger and selects Communication.input; Communication routes to the active
Conversation (lookup/swap later).
"""

from __future__ import annotations

from bot.abilities import listening
from bot.abilities import memory
from bot.behavior import instance as behavior_instance
from bot.behavior import storage as behavior_storage

from . import communication as communication_module

SPEECH_HEARD_NAME = "SpeechHeard"
SPEECH_HEARD_TRIGGERS: tuple[str, ...] = (listening.SpeechEvent.name,)

_SPEECH_EVENT = listening.SpeechEvent.name
_COMMUNICATION_INPUT = communication_module.InputEvent.name

SPEECH_HEARD_SOURCE = f"""
input_event = hsm.event(
    name = "bot.behavior.speech_heard.input",
    schema = {{
        "type": "object",
        "properties": {{
            "audio": {{"type": "string"}},
            "source_ids": {{"type": "array"}},
            "content_type": {{"type": "string"}},
            "sample_rate_hz": {{"type": "integer"}},
            "channels": {{"type": "integer"}},
            "media_type": {{"type": "string"}},
            "focus": {{"type": "string"}},
            "focus_candidates": {{"type": "array"}},
        }},
    }},
)
output_event = hsm.event(
    name = "bot.behavior.speech_heard.output",
    schema = {{
        "type": "object",
        "properties": {{
            "event": {{"type": "string"}},
            "target": {{"type": "string"}},
            "data": {{"type": "object"}},
            "reason": {{"type": "string"}},
        }},
        "required": ["event"],
    }},
)
triggers = ["{_SPEECH_EVENT}"]
description = "Communication: admit labeled Listening speech; Communication routes to the active Conversation."

def has_source_ids(event):
    data = event["data"] or {{}}
    source_ids = data.get("source_ids") or []
    return len(source_ids) > 0

def admit_speech(event):
    data = event["data"] or {{}}
    # Select Communication.input; processing resolves the communication actor.
    # Communication routes to _active_conversation (future: lookup then route).
    media_type = data.get("media_type") or "audio/pcm"
    hsm.dispatch(output_event, {{
        "event": "{_COMMUNICATION_INPUT}",
        "data": {{
            "source_ids": data.get("source_ids") or [],
            "target_ids": [],
            "content": data.get("audio"),
            "content_type": media_type,
            "sample_rate_hz": data.get("sample_rate_hz"),
            "channels": data.get("channels"),
        }},
        "reason": "seeded speech admit via communication",
    }})

behavior = hsm.define(
    "{SPEECH_HEARD_NAME}",
    hsm.initial(hsm.target("/{SPEECH_HEARD_NAME}/idle")),
    hsm.state(
        "idle",
        hsm.transition(
            hsm.on(input_event),
            hsm.guard("has_source_ids"),
            hsm.effect("admit_speech"),
        ),
    ),
)
""".strip()


def speech_heard_instance() -> behavior_instance.Instance:
    """Build the ACTIVE inventory instance for SpeechEvent → Communication.input."""

    return behavior_instance.start(
        SPEECH_HEARD_SOURCE,
        name=SPEECH_HEARD_NAME,
        triggers=SPEECH_HEARD_TRIGGERS,
        description=(
            "Communication: admit labeled Listening speech; Communication routes to the active Conversation."
        ),
    )


def install_seed_behaviors(store: memory.Memory) -> tuple[behavior_instance.Instance, ...]:
    """Insert Communication seed behaviors into ``store`` for Autonomy to load on attach."""

    installed = speech_heard_instance()
    _ = store.execute(
        memory.InputData(
            statements=memory.compile_statements(*behavior_storage.insert_behavior_clauses(installed))
        )
    )
    return (installed,)


__all__ = [
    "SPEECH_HEARD_NAME",
    "SPEECH_HEARD_SOURCE",
    "SPEECH_HEARD_TRIGGERS",
    "install_seed_behaviors",
    "speech_heard_instance",
]
