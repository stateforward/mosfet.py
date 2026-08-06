import bot
from bot.abilities import processing
from bot.devices import audio

import importlib

from tests.type_helpers import object_dict

def test_bot_body_events_import_from_agent_domain_without_ability_barrel_cycle() -> None:
    _ = importlib.import_module("bot.bot")

    assert bot.ProcessingCompletedEvent.name == "bot.processing.completed"

def test_bot_completion_events_use_pydantic_schemas() -> None:
    activating_done_schema = object_dict(bot.ActivatingDoneEvent.schema)
    activating_failed_schema = object_dict(bot.ActivatingFailedEvent.schema)

    assert bot.ActivatingDoneEvent.name == "bot.activated"
    assert activating_done_schema == bot.ActivatingDoneEventData.model_json_schema()
    assert activating_done_schema["description"]
    assert activating_done_schema["examples"] == [{}]

    assert bot.ActivatingFailedEvent.name == "bot.activating.failed"
    assert activating_failed_schema == bot.ActivatingFailedEventData.model_json_schema()
    assert activating_failed_schema["description"]
    assert activating_failed_schema["examples"] == [{}]

def test_observed_bot_event_serializes_binary_payload_as_type_only() -> None:
    payload = processing.InputData(
        input=audio.OutputEvent.with_data(
            audio.AudioOutputData(audio=b"playback-audio", media_type="audio/pcm", sample_rate_hz=48_000, channels=1)
        )
    )

    # Custom wrap serializer projects the stimulus only (not full Event schema adapters).
    dumped = payload.model_dump(mode="python")

    assert isinstance(dumped, str)
    assert f'event="{audio.OutputEvent.name}"' in dumped
    # Audio bytes must not appear as raw content in any projection of the processing input.
    assert "playback-audio" not in dumped
    assert 'audio="bytes:14"' in dumped
