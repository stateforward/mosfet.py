import bot
from bot import event_schema
from bot.abilities import cognition, listening, processing
from bot.abilities.communication import communication, conversation
from bot.devices import audio
from bot.devices.phone import events as phone_events
from bot.environment import SoundData, SoundEvent
from bot.abilities.hearing import voice

import dataclasses
import importlib
import xml.etree.ElementTree as ElementTree

import pydantic

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
    # The stimulus is the root element; its envelope rides on it, nothing wraps it.
    assert dumped.startswith("<audio:frame ")
    assert f'stimulus:event="{audio.OutputEvent.name}"' in dumped
    # Audio bytes must not appear as raw content in any projection of the processing input.
    assert "playback-audio" not in dumped
    assert 'audio="bytes:14"' in dumped


def test_speech_model_facing_xml_keeps_environment_sound_parent() -> None:
    """A speech product must retain its typed environment.sound causal parent."""

    sound = SoundData(audio=b"source-audio", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    speech = listening.SpeechData(
        content=b"decoded-audio",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        parent=bot.StimulusData.from_event(SoundEvent.with_data_and_id(sound, "sound-1")),
    )

    xml = processing.InputData(input=listening.SpeechEvent.with_data_and_id(speech, "speech-1")).model_facing_payload()

    assert xml.index("<environment:sound") < xml.index("<listening:speech")
    assert 'stimulus:event="environment.sound"' in xml
    assert 'audio="bytes:12"' in xml
    assert "source-audio" not in xml


def test_response_model_facing_xml_nests_causal_event_ancestry() -> None:
    """Each emitted event keeps its own envelope while parents render outside children."""

    sound = SoundData(audio=b"source-audio", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    sound_event = dataclasses.replace(
        SoundEvent.with_data_and_id(sound, "sound-1"), source="microphone", target="listening-1"
    )
    speech = listening.SpeechData(
        content="hello",
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"caller"}),
        parent=bot.StimulusData.from_event(sound_event),
    )
    speech_event = dataclasses.replace(
        listening.SpeechEvent.with_data_and_id(speech, "speech-1"), source="listening-1", target="cognition-1"
    )
    input_data = conversation.ConversationInputData(
        source_ids=frozenset({"caller"}),
        target_ids=frozenset(),
        content="hello",
        content_type="text/plain",
        parent=bot.StimulusData.from_event(speech_event),
    )
    communication_event = dataclasses.replace(
        communication.InputEvent.with_data_and_id(input_data, "communication-1"),
        source="communication-1",
        target="conversation-1",
    )
    response = conversation.Response(
        source_ids=input_data.source_ids,
        target_ids=input_data.target_ids,
        content="hi",
        content_type="text/plain",
        session_ref="session-1",
        parent=bot.StimulusData.from_event(communication_event),
    )

    xml = processing.InputData(
        input=conversation.OutputEvent.with_data_and_id(response, "response-1")
    ).model_facing_payload()
    root = ElementTree.fromstring(xml)
    event_key = "{urn:stateforward.bot:stimulus}event"
    id_key = "{urn:stateforward.bot:stimulus}id"
    source_key = "{urn:stateforward.bot:stimulus}source"
    target_key = "{urn:stateforward.bot:stimulus}target"

    assert root.tag == "{urn:stateforward.bot:environment}sound"
    assert root.attrib[event_key] == SoundEvent.name
    assert root.attrib[id_key] == "sound-1"
    assert root.attrib[source_key] == "microphone"
    assert root.attrib[target_key] == "listening-1"
    speech_element = next(element for element in root.iter() if element.tag == "{urn:stateforward.bot:listening}speech")
    communication_element = next(
        element for element in root.iter() if element.tag == "{urn:stateforward.bot:communication}conversation_input"
    )
    response_element = next(
        element for element in root.iter() if element.tag == "{urn:stateforward.bot:communication}response"
    )
    assert speech_element.attrib[event_key] == listening.SpeechEvent.name
    assert speech_element.attrib[id_key] == "speech-1"
    assert speech_element.attrib[source_key] == "listening-1"
    assert speech_element.attrib[target_key] == "cognition-1"
    assert communication_element.attrib[event_key] == communication.InputEvent.name
    assert communication_element.attrib[id_key] == "communication-1"
    assert communication_element.attrib[source_key] == "communication-1"
    assert communication_element.attrib[target_key] == "conversation-1"
    assert response_element.attrib[event_key] == conversation.OutputEvent.name
    assert response_element.attrib[id_key] == "response-1"
    assert xml.count("<communication:conversation_input") == 1
    assert "source-audio" not in xml


def test_nested_cognition_event_renders_typed_stimulus_without_hsm_metadata() -> None:
    """Nested HSM events use their typed payload, not dataclasses.asdict event envelopes."""

    sound = SoundData(audio=b"raw-audio", media_type="audio/pcm", sample_rate_hz=16_000, channels=1)
    speech = listening.SpeechData(
        content="hello",
        content_type="text/plain",
        voice_detection=voice.detection.ApplyData(segments=()),
        sample_rate_hz=16_000,
        channels=1,
        media_type="audio/pcm",
        source_ids=frozenset({"caller"}),
        parent=bot.StimulusData.from_event(SoundEvent.with_data_and_id(sound, "sound-1")),
    )
    cognition_input = cognition.InputData(
        stimulus=listening.SpeechEvent.with_data_and_id(speech, "speech-1"),
        abilities=(),
        actors={},
        focus=None,
        focus_candidates=(),
    )

    xml = processing.InputData(input=cognition_input).model_facing_payload()

    assert xml.index("<environment:sound") < xml.index("<listening:speech")
    assert 'stimulus:event="environment.sound"' in xml
    assert 'stimulus:event="bot.ability.listening.speech.output"' in xml
    assert "raw-audio" not in xml
    assert "metadata" not in xml
    assert 'name="bot.ability.listening.speech.output"' not in xml


def test_model_facing_xml_renders_noncausal_parent_fields_normally() -> None:
    """Only typed StimulusData parents change causal nesting semantics."""

    class Payload(pydantic.BaseModel):
        parent: str

    assert 'parent="ordinary"' in event_schema.model_facing_xml(Payload(parent="ordinary"))


def test_inherited_phone_sound_stamps_event_envelope_on_outer_root() -> None:
    """An inherited payload keeps the environment event envelope on its outer root element."""

    sound = phone_events.PhoneSoundData(
        audio=b"ring-audio",
        media_type="audio/pcm",
        sample_rate_hz=16_000,
        channels=1,
        kind="phone.ringing",
        caller="Front desk",
    )
    event = dataclasses.replace(
        SoundEvent.with_data_and_id(sound, "sound-1"),
        source="phone-1",
        target="listening-1",
    )

    root = ElementTree.fromstring(event_schema.model_facing_xml(event))
    event_key = "{urn:stateforward.bot:stimulus}event"
    id_key = "{urn:stateforward.bot:stimulus}id"
    source_key = "{urn:stateforward.bot:stimulus}source"
    target_key = "{urn:stateforward.bot:stimulus}target"

    assert root.tag == "{urn:stateforward.bot:environment}sound"
    assert root.attrib[event_key] == SoundEvent.name
    assert root.attrib[id_key] == "sound-1"
    assert root.attrib[source_key] == "phone-1"
    assert root.attrib[target_key] == "listening-1"
    phone_element = next(element for element in root.iter() if element.tag == "{urn:stateforward.bot:phone}sound")
    assert event_key not in phone_element.attrib
